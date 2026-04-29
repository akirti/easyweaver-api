import asyncio
import time
from typing import Any

import polars as pl

from easyweaver.connectors.registry import get_connector
from easyweaver.processes.batch_adapter import adapt_batch_size
from easyweaver.queries.schemas import QuerySourceConfig, JoinConfig
from easyweaver.sources.models import DataSource
from easyweaver.sources.service import get_source_credentials


_MAX_CROSS_DATASET_VALUES = 10_000


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

async def _emit(callback, event_type: str, **data) -> None:
    """Fire a progress callback if one is provided."""
    if callback is not None:
        await callback(event_type, **data)


async def _resolve_value_from(store, value_from: dict) -> list:
    """Fetch unique values from a referenced dataset column."""
    from easyweaver.core.exceptions import QueryExecutionError

    ref_run_id = str(value_from["run_id"])
    ref_column = value_from["column"]

    ref_df = await store.get_result(ref_run_id)
    if ref_df is None:
        raise QueryExecutionError(
            f"Referenced dataset run {ref_run_id} not found or expired. "
            f"Re-run the source dataset before using cross-dataset filters."
        )
    if ref_column not in ref_df.columns:
        raise QueryExecutionError(
            f"Column '{ref_column}' not found in referenced dataset {ref_run_id}. "
            f"Available columns: {ref_df.columns}"
        )

    unique_values = (
        ref_df.select(pl.col(ref_column))
        .unique()
        .drop_nulls()
        .to_series()
        .to_list()
    )
    # Convert Polars/numpy scalars to native Python types
    unique_values = [v.item() if hasattr(v, "item") else v for v in unique_values]

    if len(unique_values) > _MAX_CROSS_DATASET_VALUES:
        raise QueryExecutionError(
            f"Cross-dataset filter references {len(unique_values)} unique values, "
            f"which exceeds the limit of {_MAX_CROSS_DATASET_VALUES}. Use a join instead."
        )
    return unique_values


async def resolve_cross_dataset_filters(
    store,
    filters: list[dict],
) -> list[dict]:
    """Resolve any cross-dataset value_from references in filters."""
    resolved = []
    for f in filters:
        f = dict(f)  # shallow copy
        value_from = f.pop("value_from", None)
        if value_from is not None and f["operator"] in ("in", "not_in"):
            f["value"] = await _resolve_value_from(store, value_from)
        resolved.append(f)
    return resolved


async def execute_single_source(
    source: DataSource,
    config: QuerySourceConfig,
    row_limit: int | None = None,
) -> pl.DataFrame:
    from easyweaver.settings import settings

    # Always enforce a row limit at the database level to prevent OOM on huge tables
    effective_limit = row_limit or settings.max_result_rows

    creds = get_source_credentials(source)
    connector = get_connector(source.source_type, creds)
    async with connector:
        rows = await connector.execute_query(
            table=config.table,
            columns=config.columns,
            filters=[f.model_dump() for f in config.filters],
            filter_logic=config.filter_logic,
            limit=effective_limit,
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


async def execute_single_source_batched(
    source: DataSource,
    config: QuerySourceConfig,
    row_limit: int | None = None,
    progress_callback=None,
    control: dict | None = None,
) -> pl.DataFrame:
    """Execute a single-source query with batched fetching and progress callbacks.

    Falls back to ``execute_single_source`` when the connector does not support
    batching.
    """
    from easyweaver.settings import settings

    effective_limit = row_limit or settings.max_result_rows

    creds = get_source_credentials(source)
    connector = get_connector(source.source_type, creds)

    if not connector.supports_batching:
        df = await execute_single_source(source, config, row_limit=effective_limit)
        await _emit(progress_callback, "fetch_complete", dataset="query", total_rows=len(df))
        return df

    # Batched fetch path
    ctrl = control or {}
    target_seconds = ctrl.get("target_batch_seconds", 10.0)
    adaptive_enabled = ctrl.get("adaptive_enabled", True)
    batch_size = 10_000

    filter_dicts = [f.model_dump() for f in config.filters] if config.filters else []
    all_rows: list[dict] = []
    batch_number = 0
    offset = 0
    last_key: Any = None

    async with connector:
        while True:
            # Check control flags
            if ctrl.get("cancelled", False):
                raise asyncio.CancelledError("Query cancelled by user")

            while ctrl.get("paused", False):
                await asyncio.sleep(0.5)
                if ctrl.get("cancelled", False):
                    raise asyncio.CancelledError("Query cancelled by user")

            effective_batch_size = ctrl.get("batch_size_override") or batch_size
            batch_number += 1
            t0 = time.monotonic()

            rows, has_more, last_key = await connector.execute_query_batched(
                table=config.table,
                columns=config.columns,
                filters=filter_dicts,
                filter_logic=config.filter_logic,
                batch_size=effective_batch_size,
                offset=offset,
                last_key=last_key,
            )

            batch_time = time.monotonic() - t0
            all_rows.extend(rows)
            offset += len(rows)

            await _emit(
                progress_callback,
                "fetch_progress",
                dataset="query",
                rows_fetched=len(all_rows),
                batch_number=batch_number,
                batch_size=effective_batch_size,
                batch_time_ms=round(batch_time * 1000),
                status="fetching",
            )

            if not has_more or not rows:
                break

            # Enforce row limit
            if len(all_rows) >= effective_limit:
                break

            # Adaptive batch sizing
            if adaptive_enabled and not ctrl.get("batch_size_override"):
                max_remaining = effective_limit - len(all_rows)
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
                        dataset="query",
                        old_batch_size=old_batch_size,
                        new_batch_size=batch_size,
                        reason="adaptive",
                    )

    df = pl.DataFrame(all_rows[:effective_limit]) if all_rows else pl.DataFrame()
    await _emit(progress_callback, "fetch_complete", dataset="query", total_rows=len(df))
    return df


async def execute_join(
    left_source: DataSource,
    right_source: DataSource,
    left_config: QuerySourceConfig,
    right_config: QuerySourceConfig,
    join_config: JoinConfig,
) -> pl.DataFrame:
    # Fetch both sides in parallel
    left_df, right_df = await asyncio.gather(
        execute_single_source(left_source, left_config),
        execute_single_source(right_source, right_config),
    )

    if left_df.is_empty() or right_df.is_empty():
        return pl.DataFrame()

    # Type-coerce join keys if needed
    left_df, right_df = _coerce_join_keys(
        left_df, right_df, join_config.left_on, join_config.right_on
    )

    # Normalize to lists for Polars
    left_on = join_config.left_on if isinstance(join_config.left_on, list) else [join_config.left_on]
    right_on = join_config.right_on if isinstance(join_config.right_on, list) else [join_config.right_on]

    # Map join types
    how_map = {"inner": "inner", "left": "left", "right": "right", "outer": "full"}
    how = how_map.get(join_config.join_type, "inner")

    result = left_df.join(
        right_df,
        left_on=left_on,
        right_on=right_on,
        how=how,
        suffix="_right",
    )
    return result


async def execute_join_from_results(
    store,
    left_run_id: str,
    right_run_id: str,
    join_config: JoinConfig,
) -> pl.DataFrame:
    """Join two previously-stored result DataFrames from Redis."""
    from easyweaver.core.exceptions import QueryExecutionError

    left_df = await store.get_result(left_run_id)
    right_df = await store.get_result(right_run_id)

    if left_df is None:
        raise QueryExecutionError(f"Results for run {left_run_id} not found or expired")
    if right_df is None:
        raise QueryExecutionError(f"Results for run {right_run_id} not found or expired")

    if left_df.is_empty() or right_df.is_empty():
        return pl.DataFrame()

    left_df, right_df = _coerce_join_keys(
        left_df, right_df, join_config.left_on, join_config.right_on
    )

    # Normalize to lists for Polars
    left_on = join_config.left_on if isinstance(join_config.left_on, list) else [join_config.left_on]
    right_on = join_config.right_on if isinstance(join_config.right_on, list) else [join_config.right_on]

    how_map = {"inner": "inner", "left": "left", "right": "right", "outer": "full"}
    how = how_map.get(join_config.join_type, "inner")

    return left_df.join(
        right_df,
        left_on=left_on,
        right_on=right_on,
        how=how,
        suffix="_right",
    )


def select_columns(df: pl.DataFrame, columns: list[str] | None) -> pl.DataFrame:
    """Select specific columns from a DataFrame. None or empty list means keep all."""
    if not columns:
        return df
    valid = [c for c in columns if c in df.columns]
    return df.select(valid) if valid else df


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    """Check if a Polars dtype is numeric (integer or float)."""
    return dtype.is_numeric() if hasattr(dtype, "is_numeric") else dtype in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    )


def _normalize_join_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Normalize a string join key: strip whitespace and leading zeros for numeric-like values."""
    if df.schema.get(col) == pl.Utf8:
        df = df.with_columns(
            pl.col(col)
            .str.strip_chars()
            .str.strip_chars_start("0")
            .alias(col)
        )
        # Restore empty strings (was all zeros like "000") to "0"
        df = df.with_columns(
            pl.when(pl.col(col) == "").then(pl.lit("0")).otherwise(pl.col(col)).alias(col)
        )
    return df


def _coerce_join_keys(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_on: str | list[str],
    right_on: str | list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Coerce join key columns to compatible types.

    When one side is numeric and the other is string, both are cast to string
    and normalized (stripped of leading zeros and whitespace) so that e.g.
    integer 123 matches string "00123".
    """
    left_cols = [left_on] if isinstance(left_on, str) else left_on
    right_cols = [right_on] if isinstance(right_on, str) else right_on
    for lc, rc in zip(left_cols, right_cols):
        left_type = left.schema.get(lc)
        right_type = right.schema.get(rc)
        if left_type is None or right_type is None:
            continue
        if left_type != right_type:
            # Determine if this is a numeric ↔ string mismatch needing normalization
            left_numeric = _is_numeric_dtype(left_type)
            right_numeric = _is_numeric_dtype(right_type)
            left_string = left_type == pl.Utf8
            right_string = right_type == pl.Utf8
            needs_normalize = (left_numeric and right_string) or (left_string and right_numeric)

            # Cast both to string
            left = left.with_columns(pl.col(lc).cast(pl.Utf8))
            right = right.with_columns(pl.col(rc).cast(pl.Utf8))

            # Normalize leading zeros when mixing numeric + string keys
            if needs_normalize:
                left = _normalize_join_col(left, lc)
                right = _normalize_join_col(right, rc)
    return left, right


def apply_sort(df: pl.DataFrame, sorts: list[dict]) -> pl.DataFrame:
    if not sorts:
        return df
    columns = [s["column"] for s in sorts]
    descending = [s.get("direction", "asc") == "desc" for s in sorts]
    return df.sort(columns, descending=descending)


def paginate_dataframe(
    df: pl.DataFrame, page: int, page_size: int
) -> tuple[list[dict[str, Any]], int]:
    total = len(df)
    offset = (page - 1) * page_size
    page_df = df.slice(offset, page_size)

    # Cast any temporal columns to string for JSON-safe serialisation
    temporal_cols = [
        c for c, t in page_df.schema.items()
        if t in (pl.Datetime, pl.Date, pl.Time, pl.Duration)
        or str(t).startswith("Datetime")
    ]
    if temporal_cols:
        page_df = page_df.with_columns(
            [pl.col(c).cast(pl.Utf8).alias(c) for c in temporal_cols]
        )

    rows = page_df.to_dicts()
    return rows, total
