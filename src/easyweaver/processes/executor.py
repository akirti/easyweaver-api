"""DAG-aware batched process executor.

Replaces the original parallel-gather approach with dependency-ordered
execution waves, adaptive batch sizing, and optional progress callbacks.
"""

import asyncio
import copy
import json
import re
import time
import uuid
from typing import Any

import polars as pl
import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.connectors.registry import get_connector
from easyweaver.core.exceptions import ProcessExecutionError
from easyweaver.processes.batch_adapter import adapt_batch_size
from easyweaver.processes.dag import build_dag, detect_cycles, get_ready_datasets
from easyweaver.processes.schemas import ProcessConfig, ProcessQueryConfig
from easyweaver.queries.executor import (
    _coerce_join_keys,
    apply_sort,
    execute_single_source,
    select_columns,
)
from easyweaver.queries.operations.derived import apply_derived_columns
from easyweaver.queries.operations.distinct import apply_distinct
from easyweaver.queries.operations.filter import apply_filters
from easyweaver.queries.operations.group_by import apply_group_by
from easyweaver.queries.operations.transform import apply_transforms
from easyweaver.queries.schemas import QuerySourceConfig
from easyweaver.sources.service import get_source

logger = structlog.get_logger()

_PARAM_RE = re.compile(r"\{(\w+)\}")


# ---------------------------------------------------------------------------
# Parameter helpers (unchanged)
# ---------------------------------------------------------------------------

def coerce_param_values(param_values: dict, param_defs: dict) -> dict:
    """Coerce param values to their declared types from param definitions.

    Also fills in default values for any params not provided.
    """
    coerced = dict(param_values)

    # Fill in defaults for missing params
    for name, defn in param_defs.items():
        if name not in coerced:
            default = defn.get("default") if isinstance(defn, dict) else getattr(defn, "default", None)
            if default is not None:
                coerced[name] = default

    # Coerce types
    for name, value in coerced.items():
        defn = param_defs.get(name)
        if not defn:
            continue
        ptype = defn.get("type", "string") if isinstance(defn, dict) else getattr(defn, "type", "string")
        if ptype == "number" and not isinstance(value, (int, float)):
            try:
                coerced[name] = int(value) if "." not in str(value) else float(value)
            except (ValueError, TypeError):
                pass
        elif ptype == "boolean" and not isinstance(value, bool):
            if isinstance(value, str):
                coerced[name] = value.lower() in ("true", "1", "yes")
        elif ptype == "select":
            # Leave as-is: string value selected from options; no coercion needed
            coerced[name] = value
        elif ptype == "multi_select":
            # Ensure value is a list
            if isinstance(value, list):
                coerced[name] = value  # already a list
            elif isinstance(value, str):
                coerced[name] = [v.strip() for v in value.split(",") if v.strip()]
            else:
                coerced[name] = [value]
        elif ptype == "boolean_yesno":
            if isinstance(value, bool):
                coerced[name] = value  # already a bool; no coercion needed
            elif isinstance(value, str):
                coerced[name] = value.lower() in ("yes",)
            else:
                coerced[name] = bool(value)
        elif ptype == "boolean_truefalse":
            if isinstance(value, bool):
                coerced[name] = value  # already a bool; no coercion needed
            elif isinstance(value, str):
                coerced[name] = value.lower() in ("true",)
            else:
                coerced[name] = bool(value)
    return coerced


def resolve_params(config_dict: dict, param_values: dict) -> dict:
    """Deep-clone config and replace {param_name} placeholders with actual values."""
    config_dict = copy.deepcopy(config_dict)
    return _walk_replace(config_dict, param_values)


def _walk_replace(obj, param_values: dict):
    if isinstance(obj, dict):
        return {k: _walk_replace(v, param_values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_replace(item, param_values) for item in obj]
    if isinstance(obj, str):
        # If the entire string is a single param reference, return the raw value
        match = _PARAM_RE.fullmatch(obj)
        if match:
            name = match.group(1)
            if name in param_values:
                return param_values[name]
            return obj
        # Otherwise do string interpolation for partial matches
        def replacer(m):
            name = m.group(1)
            return str(param_values[name]) if name in param_values else m.group(0)

        return _PARAM_RE.sub(replacer, obj)
    return obj


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

async def _emit(callback, event_type: str, **data) -> None:
    """Fire a progress callback if one is provided."""
    if callback is not None:
        await callback(event_type, **data)


# ---------------------------------------------------------------------------
# Binding resolution helpers (process-level, not query-builder-level)
# ---------------------------------------------------------------------------

def _resolve_distinct_bindings_from_df(
    source_df: pl.DataFrame,
    mappings: list,
) -> list[dict]:
    """Build IN-filter dicts from a completed source DataFrame."""
    filters = []
    for mapping in mappings:
        src_col = mapping.source_column if hasattr(mapping, "source_column") else mapping["source_column"]
        tgt_col = mapping.target_column if hasattr(mapping, "target_column") else mapping["target_column"]
        values = (
            source_df.select(pl.col(src_col))
            .unique()
            .drop_nulls()
            .to_series()
            .to_list()
        )
        values = [v.item() if hasattr(v, "item") else v for v in values]
        if values:
            filters.append({"column": tgt_col, "operator": "in", "value": values})
    return filters


def _resolve_row_pair_filter_from_df(
    source_df: pl.DataFrame,
    mappings: list,
) -> pl.DataFrame | None:
    """Build a semi-join DataFrame from a completed source DataFrame."""
    src_cols = []
    tgt_cols = []
    for m in mappings:
        src_cols.append(m.source_column if hasattr(m, "source_column") else m["source_column"])
        tgt_cols.append(m.target_column if hasattr(m, "target_column") else m["target_column"])
    if not src_cols:
        return None
    rename_map = dict(zip(src_cols, tgt_cols))
    return source_df.select(src_cols).unique().rename(rename_map)


# ---------------------------------------------------------------------------
# Single query execution
# ---------------------------------------------------------------------------

async def _execute_query(
    db: AsyncIOMotorDatabase | None,
    query_config: ProcessQueryConfig,
    key: str,
    extra_filters: list[dict] | None = None,
) -> pl.DataFrame:
    """Execute a single process query.

    Self-sufficient path: if embedded credentials exist, decrypt and connect directly.
    Legacy fallback: look up the source from MongoDB via db.
    """
    all_filters = list(extra_filters or [])

    if query_config.encrypted_credentials and query_config.source_type:
        # Self-sufficient path -- use embedded credentials
        from easyweaver.core.security import decrypt_credentials

        creds = json.loads(decrypt_credentials(query_config.encrypted_credentials))
        connector = get_connector(query_config.source_type, creds)

        async with connector:
            rows = await connector.execute_query(
                table=query_config.table,
                columns=query_config.columns,
                filters=all_filters,
                filter_logic=query_config.filter_logic,
            )
        df = pl.DataFrame(rows) if rows else pl.DataFrame()
    else:
        # Legacy path -- look up source from MongoDB
        if db is None:
            raise ProcessExecutionError(
                f"Query '{key}' has no embedded credentials and no database connection available"
            )

        source = await get_source(db, query_config.source_id)
        source_config = QuerySourceConfig(
            source_id=source.id,
            table=query_config.table,
            columns=query_config.columns,
            filters=all_filters,
            filter_logic=query_config.filter_logic,
        )
        df = await execute_single_source(source, source_config)

    # Apply per-query filters post-execution if any
    if query_config.filters:
        filter_dicts = [f.model_dump() for f in query_config.filters]
        df = apply_filters(df, filter_dicts, query_config.filter_logic)

    logger.info("process_query_executed", key=key, rows=len(df))
    return df


async def _execute_query_batched(
    db: AsyncIOMotorDatabase | None,
    query_config: ProcessQueryConfig,
    key: str,
    extra_filters: list[dict] | None = None,
    progress_callback=None,
    control: dict | None = None,
    redis_store=None,
    run_id: str | None = None,
) -> pl.DataFrame:
    """Execute a query using batched fetching with adaptive batch sizing.

    Falls back to single-shot ``_execute_query`` if the connector does not
    support batching or has no embedded credentials.
    """
    # Determine if we can do batched fetch
    if not (query_config.encrypted_credentials and query_config.source_type):
        # Legacy path - no batching support
        df = await _execute_query(db, query_config, key, extra_filters)
        await _emit(progress_callback, "fetch_complete", dataset=key, total_rows=len(df))
        return df

    from easyweaver.core.security import decrypt_credentials

    creds = json.loads(decrypt_credentials(query_config.encrypted_credentials))
    connector = get_connector(query_config.source_type, creds)

    if not connector.supports_batching:
        # File/REST connectors - single shot
        async with connector:
            rows = await connector.execute_query(
                table=query_config.table,
                columns=query_config.columns,
                filters=list(extra_filters or []),
                filter_logic=query_config.filter_logic,
            )
        df = pl.DataFrame(rows) if rows else pl.DataFrame()
        if query_config.filters:
            filter_dicts = [f.model_dump() for f in query_config.filters]
            df = apply_filters(df, filter_dicts, query_config.filter_logic)
        await _emit(progress_callback, "fetch_complete", dataset=key, total_rows=len(df))
        logger.info("process_query_executed", key=key, rows=len(df))
        return df

    # Batched fetch path
    ctrl = control or {}
    target_seconds = ctrl.get("target_batch_seconds", 10.0)
    adaptive_enabled = ctrl.get("adaptive_enabled", True)

    # Get initial batch size
    batch_size = 10_000
    if db is not None:
        try:
            from easyweaver.processes.batch_adapter import get_initial_batch_size
            batch_size = await get_initial_batch_size(
                query_config.source_id,
                query_config.source_type,
                query_config.table,
                db,
            )
        except Exception:
            pass

    all_rows: list[dict] = []
    batch_number = 0
    offset = 0
    last_key: Any = None
    batch_sizes: list[int] = []
    batch_times: list[float] = []
    all_filters_combined = list(extra_filters or [])

    async with connector:
        while True:
            # Check control flags
            if ctrl.get("cancelled", False):
                raise asyncio.CancelledError("Process cancelled by user")

            # Pause loop
            while ctrl.get("paused", False):
                await asyncio.sleep(0.5)
                if ctrl.get("cancelled", False):
                    raise asyncio.CancelledError("Process cancelled by user")

            # Override batch size if set
            effective_batch_size = ctrl.get("batch_size_override") or batch_size

            batch_number += 1
            t0 = time.monotonic()

            rows, has_more, last_key = await connector.execute_query_batched(
                table=query_config.table,
                columns=query_config.columns,
                filters=all_filters_combined,
                filter_logic=query_config.filter_logic,
                batch_size=effective_batch_size,
                offset=offset,
                last_key=last_key,
            )

            batch_time = time.monotonic() - t0
            batch_sizes.append(effective_batch_size)
            batch_times.append(batch_time)

            all_rows.extend(rows)
            offset += len(rows)

            await _emit(
                progress_callback,
                "fetch_progress",
                dataset=key,
                rows_fetched=len(all_rows),
                batch_number=batch_number,
                batch_size=effective_batch_size,
                batch_time_ms=round(batch_time * 1000),
                status="fetching",
            )

            if not has_more or not rows:
                break

            # Adaptive batch sizing
            if adaptive_enabled and not ctrl.get("batch_size_override"):
                from easyweaver.settings import settings
                max_remaining = settings.max_result_rows - len(all_rows)
                old_batch_size = batch_size
                batch_size = adapt_batch_size(
                    effective_batch_size,
                    batch_time,
                    target_seconds,
                    max_remaining,
                )
                if batch_size != old_batch_size:
                    await _emit(
                        progress_callback,
                        "batch_adjusted",
                        dataset=key,
                        old_batch_size=old_batch_size,
                        new_batch_size=batch_size,
                        reason="adaptive",
                    )

    # Save optimal batch size for future runs
    if db is not None and batch_sizes:
        try:
            from easyweaver.processes.batch_adapter import save_optimal_batch_size
            await save_optimal_batch_size(
                query_config.source_id,
                query_config.source_type,
                query_config.table,
                batch_sizes,
                batch_times,
                target_seconds,
                db,
            )
        except Exception:
            pass

    df = pl.DataFrame(all_rows) if all_rows else pl.DataFrame()

    # Apply per-query filters
    if query_config.filters:
        filter_dicts = [f.model_dump() for f in query_config.filters]
        df = apply_filters(df, filter_dicts, query_config.filter_logic)

    # Store intermediate result in Redis if requested
    if redis_store and run_id:
        cache_key = f"ew:batch:{run_id}:{key}"
        try:
            from easyweaver.settings import settings as _settings
            await redis_store.store_result(cache_key, df, ttl=_settings.batch_intermediate_ttl_seconds)
        except Exception as exc:
            logger.warning("intermediate_cache_failed", key=key, error=str(exc))

    await _emit(progress_callback, "fetch_complete", dataset=key, total_rows=len(df))
    logger.info("process_query_executed", key=key, rows=len(df))
    return df


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

async def execute_process(
    config: ProcessConfig,
    param_values: dict,
    db: AsyncIOMotorDatabase | None = None,
    progress_callback=None,
    control: dict | None = None,
    redis_store=None,
    run_id: str | None = None,
) -> pl.DataFrame:
    """Execute a saved process configuration and return the final DataFrame.

    Uses DAG-based dependency ordering with eager scheduling so that
    independent datasets are fetched concurrently, while dependent
    datasets wait for their sources to complete before applying binding
    filters.

    Args:
        config: The process configuration to execute.
        param_values: Parameter values for placeholder substitution.
        db: Optional MongoDB connection for legacy source lookups and
            batch size history.
        progress_callback: Optional ``async def cb(event_type, **data)``
            called at key execution milestones.
        control: Optional mutable dict for pause/cancel/override.
            Keys: ``paused``, ``cancelled``, ``batch_size_override``,
            ``target_batch_seconds``, ``adaptive_enabled``.
        redis_store: Optional Redis result store for intermediate caching.
        run_id: Unique run identifier for intermediate cache keys.
    """
    # Resolve parameters in the config
    resolved_dict = resolve_params(config.model_dump(), param_values)
    resolved_config = ProcessConfig.model_validate(resolved_dict)

    if run_id is None:
        run_id = uuid.uuid4().hex

    # ── Phase 0: Build DAG & validate ─────────────────────────────────
    dag = build_dag(resolved_config.queries)
    detect_cycles(dag)

    # Build a flat lookup: key -> (schema_name, query_name, query_config)
    query_lookup: dict[str, ProcessQueryConfig] = {}
    for schema_name, queries in resolved_config.queries.items():
        for query_name, query_config in queries.items():
            key = f"{schema_name}.{query_name}"
            query_lookup[key] = query_config

    if not query_lookup:
        raise ProcessExecutionError("Process has no queries to execute")

    results: dict[str, pl.DataFrame] = {}
    total_datasets = len(query_lookup)

    # ── Phase 1: Fetching (DAG-ordered, eager scheduling) ─────────────
    await _emit(
        progress_callback,
        "phase",
        phase="fetching",
        phase_index=1,
        total_phases=3,
    )

    completed: set[str] = set()
    in_flight: dict[str, asyncio.Task] = {}

    def _start_ready():
        """Launch fetch tasks for all datasets whose deps are met."""
        ready = get_ready_datasets(dag, completed)
        for dataset_key in ready:
            if dataset_key in in_flight or dataset_key in completed:
                continue
            query_config = query_lookup[dataset_key]

            # Resolve bindings from already-completed sources
            extra_filters: list[dict] = []
            row_pair_df: pl.DataFrame | None = None

            for binding in query_config.bindings:
                source_key = binding.source_dataset
                source_df = results.get(source_key)
                if source_df is None or source_df.is_empty():
                    continue

                if binding.mode == "distinct":
                    extra_filters.extend(
                        _resolve_distinct_bindings_from_df(source_df, binding.mappings)
                    )
                elif binding.mode == "row_pair":
                    row_pair_df = _resolve_row_pair_filter_from_df(
                        source_df, binding.mappings
                    )

            has_bindings = bool(query_config.bindings)
            filter_values_count = len(extra_filters)

            async def _fetch(
                qc=query_config,
                k=dataset_key,
                ef=extra_filters,
                rpdf=row_pair_df,
                hb=has_bindings,
                fvc=filter_values_count,
            ):
                await _emit(
                    progress_callback,
                    "fetch_started",
                    dataset=k,
                    binding_resolved=hb,
                    filter_values_count=fvc,
                )
                df = await _execute_query_batched(
                    db, qc, k,
                    extra_filters=ef,
                    progress_callback=progress_callback,
                    control=control,
                    redis_store=redis_store,
                    run_id=run_id,
                )
                # Apply row-pair semi-join filter if needed
                if rpdf is not None and not rpdf.is_empty():
                    from easyweaver.queries.operations.binding import apply_row_pair_filter
                    df = apply_row_pair_filter(df, rpdf)
                return df

            task = asyncio.create_task(_fetch())
            in_flight[dataset_key] = task

    _start_ready()

    while len(completed) < total_datasets:
        if not in_flight:
            # No tasks running and not all completed - shouldn't happen with valid DAG
            missing = set(query_lookup) - completed
            raise ProcessExecutionError(
                f"Deadlock: no tasks in flight but datasets incomplete: {missing}"
            )

        # Wait for the first task to complete
        done, _ = await asyncio.wait(
            in_flight.values(),
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            # Find which key this task belongs to
            finished_key = None
            for k, t in in_flight.items():
                if t is task:
                    finished_key = k
                    break

            if finished_key is None:
                continue

            del in_flight[finished_key]

            if task.exception() is not None:
                # Cancel remaining tasks
                for t in in_flight.values():
                    t.cancel()
                exc = task.exception()
                raise ProcessExecutionError(
                    f"Query '{finished_key}' failed: {exc}",
                    details={"query_key": finished_key},
                ) from exc

            results[finished_key] = task.result()
            completed.add(finished_key)

            # Check for newly unblocked datasets
            newly_ready = get_ready_datasets(dag, completed)
            for nr_key in newly_ready:
                if nr_key in in_flight or nr_key in completed:
                    continue
                # Log waiting info for datasets that were waiting
                waiting_for = [
                    dep for dep in dag.get(nr_key, [])
                    if dep in dag and dep in completed
                ]
                if waiting_for:
                    await _emit(
                        progress_callback,
                        "fetch_waiting",
                        dataset=nr_key,
                        waiting_for=waiting_for,
                    )

            _start_ready()

    # ── Phase 2: Joining ──────────────────────────────────────────────
    await _emit(
        progress_callback,
        "phase",
        phase="joining",
        phase_index=2,
        total_phases=3,
    )

    for step_idx, step in enumerate(resolved_config.logics):
        if step.type == "join":
            left_df = results.get(step.left)
            right_df = results.get(step.right)
            if left_df is None:
                raise ProcessExecutionError(
                    f"Logic step '{step.key}': left reference '{step.left}' not found"
                )
            if right_df is None:
                raise ProcessExecutionError(
                    f"Logic step '{step.key}': right reference '{step.right}' not found"
                )

            await _emit(
                progress_callback,
                "join_progress",
                step_key=step.key,
                left=len(left_df),
                right=len(right_df),
                status="running",
                rows=0,
            )

            if left_df.is_empty() or right_df.is_empty():
                results[step.key] = pl.DataFrame()
                continue

            left_df, right_df = _coerce_join_keys(
                left_df, right_df, step.left_on, step.right_on
            )

            how_map = {"inner": "inner", "left": "left", "right": "right", "outer": "full"}
            how = how_map.get(step.join_type, "inner")

            joined = left_df.join(
                right_df,
                left_on=step.left_on,
                right_on=step.right_on,
                how=how,
                suffix="_right",
            )
            if step.select_columns:
                joined = select_columns(joined, step.select_columns)
            results[step.key] = joined

            await _emit(
                progress_callback,
                "join_progress",
                step_key=step.key,
                left=len(left_df),
                right=len(right_df),
                status="complete",
                rows=len(joined),
            )

            logger.info(
                "process_join_step",
                key=step.key,
                left=step.left,
                right=step.right,
                rows=len(joined),
            )

    # Determine final DataFrame
    if resolved_config.logics:
        final_key = resolved_config.logics[-1].key
        df = results.get(final_key)
        if df is None:
            raise ProcessExecutionError(f"Final logic step '{final_key}' produced no result")
    elif len(results) == 1:
        df = next(iter(results.values()))
    else:
        raise ProcessExecutionError(
            "Multiple queries without logic steps -- define joins to combine them"
        )

    # ── Phase 3: Transforms ───────────────────────────────────────────
    await _emit(
        progress_callback,
        "phase",
        phase="transforming",
        phase_index=3,
        total_phases=3,
    )

    total_transform_steps = (
        (1 if resolved_config.derived_columns else 0)
        + (1 if resolved_config.operations else 0)
        + (1 if resolved_config.transformations else 0)
    )
    current_step = 0

    # 3a. Apply derived columns
    if resolved_config.derived_columns:
        current_step += 1
        await _emit(
            progress_callback,
            "transform_progress",
            operation="derived_columns",
            step=current_step,
            total_steps=total_transform_steps,
        )
        derived_dicts = [d.model_dump() for d in resolved_config.derived_columns]
        df = apply_derived_columns(df, derived_dicts)

    # 3b. Apply operations (filters, group_by, distinct, sorts)
    if resolved_config.operations:
        current_step += 1
        await _emit(
            progress_callback,
            "transform_progress",
            operation="operations",
            step=current_step,
            total_steps=total_transform_steps,
        )
        ops = resolved_config.operations
        if ops.filters:
            filter_dicts = [f.model_dump() for f in ops.filters]
            df = apply_filters(df, filter_dicts, ops.filter_logic)
        if ops.group_by:
            df = apply_group_by(df, ops.group_by.model_dump())
        if ops.distinct:
            df = apply_distinct(df, ops.distinct.model_dump())
        if ops.sorts:
            sort_dicts = [s.model_dump() for s in ops.sorts]
            df = apply_sort(df, sort_dicts)

    # 3c. Apply transformations
    if resolved_config.transformations:
        current_step += 1
        await _emit(
            progress_callback,
            "transform_progress",
            operation="transformations",
            step=current_step,
            total_steps=total_transform_steps,
        )
        transform_dicts = [t.model_dump() for t in resolved_config.transformations]
        df = apply_transforms(df, transform_dicts)

    return df
