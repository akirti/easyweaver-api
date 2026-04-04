import asyncio
from typing import Any

import polars as pl

from easyweaver.connectors.registry import get_connector
from easyweaver.queries.schemas import QuerySourceConfig, JoinConfig
from easyweaver.sources.models import DataSource
from easyweaver.sources.service import get_source_credentials


async def resolve_cross_dataset_filters(
    store,
    filters: list[dict],
) -> list[dict]:
    """Resolve any cross-dataset value_from references in filters.

    For filters with value_from={run_id, column}, reads the referenced
    dataset from Redis and replaces value_from with a concrete value list.
    """
    from easyweaver.core.exceptions import QueryExecutionError

    MAX_IN_VALUES = 10_000
    resolved = []
    for f in filters:
        f = dict(f)  # shallow copy
        value_from = f.pop("value_from", None)
        if value_from is not None and f["operator"] in ("in", "not_in"):
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
            unique_values = [
                v.item() if hasattr(v, "item") else v for v in unique_values
            ]

            if len(unique_values) > MAX_IN_VALUES:
                raise QueryExecutionError(
                    f"Cross-dataset filter references {len(unique_values)} unique values, "
                    f"which exceeds the limit of {MAX_IN_VALUES}. Use a join instead."
                )

            f["value"] = unique_values
        resolved.append(f)
    return resolved


async def execute_single_source(
    source: DataSource,
    config: QuerySourceConfig,
) -> pl.DataFrame:
    creds = get_source_credentials(source)
    connector = get_connector(source.source_type, creds)
    async with connector:
        rows = await connector.execute_query(
            table=config.table,
            columns=config.columns,
            filters=[f.model_dump() for f in config.filters],
            filter_logic=config.filter_logic,
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


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


def _coerce_join_keys(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_on: str | list[str],
    right_on: str | list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Coerce join key columns to compatible types (int ↔ string)."""
    left_cols = [left_on] if isinstance(left_on, str) else left_on
    right_cols = [right_on] if isinstance(right_on, str) else right_on
    for lc, rc in zip(left_cols, right_cols):
        left_type = left.schema.get(lc)
        right_type = right.schema.get(rc)
        if left_type is None or right_type is None:
            continue
        if left_type != right_type:
            left = left.with_columns(pl.col(lc).cast(pl.Utf8))
            right = right.with_columns(pl.col(rc).cast(pl.Utf8))
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
