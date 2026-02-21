import asyncio
import json
from typing import Any

import polars as pl

from easyweaver.connectors.registry import get_connector
from easyweaver.queries.schemas import QuerySourceConfig, JoinConfig
from easyweaver.sources.models import DataSource
from easyweaver.sources.service import get_source_credentials


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

    # Map join types
    how_map = {"inner": "inner", "left": "left", "right": "right", "outer": "full"}
    how = how_map.get(join_config.join_type, "inner")

    result = left_df.join(
        right_df,
        left_on=join_config.left_on,
        right_on=join_config.right_on,
        how=how,
        suffix="_right",
    )
    return result


def _coerce_join_keys(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_on: str,
    right_on: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Coerce join key columns to compatible types (int ↔ string)."""
    left_type = left.schema.get(left_on)
    right_type = right.schema.get(right_on)

    if left_type is None or right_type is None:
        return left, right

    if left_type != right_type:
        # Cast both to string for safe joining
        left = left.with_columns(pl.col(left_on).cast(pl.Utf8))
        right = right.with_columns(pl.col(right_on).cast(pl.Utf8))

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
    rows = page_df.to_dicts()
    return rows, total
