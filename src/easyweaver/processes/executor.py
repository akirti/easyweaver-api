import asyncio
import copy
import re

import polars as pl
import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.core.exceptions import ProcessExecutionError
from easyweaver.processes.schemas import ProcessConfig, ProcessQueryConfig
from easyweaver.queries.executor import (
    _coerce_join_keys,
    apply_sort,
    execute_single_source,
)
from easyweaver.queries.operations.filter import apply_filters
from easyweaver.queries.operations.transform import apply_transforms
from easyweaver.queries.schemas import QuerySourceConfig
from easyweaver.sources.service import get_source

logger = structlog.get_logger()

_PARAM_RE = re.compile(r"\{(\w+)\}")


def coerce_param_values(param_values: dict, param_defs: dict) -> dict:
    """Coerce param values to their declared types from param definitions."""
    coerced = dict(param_values)
    for name, value in coerced.items():
        defn = param_defs.get(name)
        if not defn:
            continue
        ptype = defn.get("type", "string") if isinstance(defn, dict) else getattr(defn, "type", "string")
        if ptype == "number" and not isinstance(value, (int, float)):
            try:
                # Try int first, then float
                coerced[name] = int(value) if "." not in str(value) else float(value)
            except (ValueError, TypeError):
                pass
        elif ptype == "boolean" and not isinstance(value, bool):
            if isinstance(value, str):
                coerced[name] = value.lower() in ("true", "1", "yes")
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


async def execute_process(
    config: ProcessConfig,
    param_values: dict,
    db: AsyncIOMotorDatabase,
) -> pl.DataFrame:
    """Execute a saved process configuration and return the final DataFrame."""
    # Resolve parameters in the config
    resolved_dict = resolve_params(config.model_dump(), param_values)
    resolved_config = ProcessConfig.model_validate(resolved_dict)

    results: dict[str, pl.DataFrame] = {}

    # 1. Execute all queries in parallel
    query_tasks = []
    query_keys = []

    for schema_name, queries in resolved_config.queries.items():
        for query_name, query_config in queries.items():
            key = f"{schema_name}.{query_name}"
            query_keys.append(key)
            query_tasks.append(_execute_query(db, query_config, key))

    if not query_tasks:
        raise ProcessExecutionError("Process has no queries to execute")

    query_results = await asyncio.gather(*query_tasks, return_exceptions=True)

    for key, result in zip(query_keys, query_results):
        if isinstance(result, Exception):
            raise ProcessExecutionError(
                f"Query '{key}' failed: {result}",
                details={"query_key": key},
            )
        results[key] = result

    # 2. Apply logic steps sequentially
    for step in resolved_config.logics:
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
            results[step.key] = joined
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
            "Multiple queries without logic steps — define joins to combine them"
        )

    # 3. Apply operations (filters + sorts)
    if resolved_config.operations:
        ops = resolved_config.operations
        if ops.filters:
            filter_dicts = [f.model_dump() for f in ops.filters]
            df = apply_filters(df, filter_dicts, ops.filter_logic)
        if ops.sorts:
            sort_dicts = [s.model_dump() for s in ops.sorts]
            df = apply_sort(df, sort_dicts)

    # 4. Apply transformations
    if resolved_config.transformations:
        transform_dicts = [t.model_dump() for t in resolved_config.transformations]
        df = apply_transforms(df, transform_dicts)

    return df


async def _execute_query(
    db: AsyncIOMotorDatabase,
    query_config: ProcessQueryConfig,
    key: str,
) -> pl.DataFrame:
    """Execute a single process query by looking up the source and running it."""
    source = await get_source(db, query_config.source_id)

    # Convert ProcessQueryConfig to QuerySourceConfig
    source_config = QuerySourceConfig(
        source_id=source.id,
        table=query_config.table,
        columns=query_config.columns,
        filters=[],
        filter_logic=query_config.filter_logic,
    )

    df = await execute_single_source(source, source_config)

    # Apply per-query filters post-execution if any
    if query_config.filters:
        filter_dicts = [f.model_dump() for f in query_config.filters]
        df = apply_filters(df, filter_dicts, query_config.filter_logic)

    logger.info("process_query_executed", key=key, rows=len(df))
    return df
