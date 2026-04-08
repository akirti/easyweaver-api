import polars as pl
import structlog

from easyweaver.core.exceptions import QueryExecutionError

logger = structlog.get_logger()

_MAX_DISTINCT_VALUES = 10_000
_MAX_ROW_PAIR_ROWS = 1000
_ROW_PAIR_WARN_THRESHOLD = 500


async def _get_source_df(store, source_run_id: str) -> pl.DataFrame:
    """Fetch source DataFrame from Redis, raising if not found."""
    source_df = await store.get_result(source_run_id)
    if source_df is None:
        raise QueryExecutionError(
            f"Binding source run {source_run_id} not found or expired. "
            f"Re-run the source dataset first."
        )
    return source_df


def _validate_columns(source_df: pl.DataFrame, columns: list[str], source_run_id: str) -> None:
    """Validate that all columns exist in the source DataFrame."""
    for col in columns:
        if col not in source_df.columns:
            raise QueryExecutionError(
                f"Column '{col}' not found in source dataset {source_run_id}. "
                f"Available: {source_df.columns}"
            )


def _extract_distinct_values(source_df: pl.DataFrame, src_col: str) -> list:
    """Extract unique non-null values from a column, converting to native Python types."""
    values = (
        source_df.select(pl.col(src_col))
        .unique()
        .drop_nulls()
        .to_series()
        .to_list()
    )
    values = [v.item() if hasattr(v, "item") else v for v in values]
    if len(values) > _MAX_DISTINCT_VALUES:
        raise QueryExecutionError(
            f"Binding column '{src_col}' has {len(values)} unique values, "
            f"exceeding limit of {_MAX_DISTINCT_VALUES}. Use a join instead."
        )
    return values


async def resolve_distinct_bindings(
    store, bindings: list[dict]
) -> list[dict]:
    """Resolve distinct-mode bindings into synthetic IN filter conditions."""
    filters = []
    for binding in bindings:
        if binding["mode"] != "distinct":
            continue
        source_run_id = str(binding["source_run_id"])
        source_df = await _get_source_df(store, source_run_id)
        if source_df.is_empty():
            logger.warning("binding_empty_source", run_id=source_run_id, mode="distinct")
            continue

        src_cols = [m["source_column"] for m in binding["mappings"]]
        _validate_columns(source_df, src_cols, source_run_id)

        for mapping in binding["mappings"]:
            values = _extract_distinct_values(source_df, mapping["source_column"])
            filters.append({"column": mapping["target_column"], "operator": "in", "value": values})
    return filters


async def resolve_row_pair_bindings(
    store, bindings: list[dict]
) -> pl.DataFrame | None:
    """Resolve row-pair bindings into a DataFrame for semi-join.

    Returns a DataFrame with columns renamed to target names, or None if
    no row-pair bindings exist.
    """
    for binding in bindings:
        if binding["mode"] != "row_pair":
            continue
        source_run_id = str(binding["source_run_id"])
        source_df = await _get_source_df(store, source_run_id)

        row_count = len(source_df)
        if row_count > _MAX_ROW_PAIR_ROWS:
            raise QueryExecutionError(
                f"Row-pair binding source has {row_count} rows, "
                f"exceeding limit of {_MAX_ROW_PAIR_ROWS}. "
                f"Filter the source dataset to reduce rows."
            )
        if row_count > _ROW_PAIR_WARN_THRESHOLD:
            logger.warning("row_pair_high_row_count", run_id=source_run_id, rows=row_count)
        if source_df.is_empty():
            return pl.DataFrame()

        mappings = binding["mappings"]
        src_cols = [m["source_column"] for m in mappings]
        tgt_cols = [m["target_column"] for m in mappings]
        _validate_columns(source_df, src_cols, source_run_id)

        rename_map = dict(zip(src_cols, tgt_cols))
        return source_df.select(src_cols).unique().rename(rename_map)

    return None


def apply_row_pair_filter(
    df: pl.DataFrame, pair_df: pl.DataFrame
) -> pl.DataFrame:
    """Apply row-pair binding as a semi-join."""
    if pair_df.is_empty():
        return pl.DataFrame(schema=df.schema)
    join_cols = pair_df.columns
    for col in join_cols:
        if col in df.columns and df.schema[col] != pair_df.schema[col]:
            pair_df = pair_df.with_columns(
                pl.col(col).cast(df.schema[col]).alias(col)
            )
    return df.join(pair_df, on=join_cols, how="semi")
