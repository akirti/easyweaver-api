import polars as pl
import structlog

logger = structlog.get_logger()

_AGG_MAP = {
    "count": lambda col: pl.col(col).count(),
    "sum": lambda col: pl.col(col).sum(),
    "avg": lambda col: pl.col(col).mean(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "count_distinct": lambda col: pl.col(col).n_unique(),
}


def apply_group_by(df: pl.DataFrame, spec: dict) -> pl.DataFrame:
    """Apply GROUP BY with aggregations to a DataFrame."""
    group_cols = [c for c in spec["group_columns"] if c in df.columns]
    if not group_cols:
        return df

    agg_exprs = []
    for agg in spec.get("aggregations", []):
        col = agg["column"]
        func = agg["function"]
        alias = agg.get("alias") or f"{func}_{col}"

        if col not in df.columns:
            logger.warning("group_by_skip_missing_column", column=col, function=func)
            continue
        if func not in _AGG_MAP:
            logger.warning("group_by_unknown_function", function=func)
            continue

        agg_exprs.append(_AGG_MAP[func](col).alias(alias))

    if not agg_exprs:
        return df

    return df.group_by(group_cols).agg(agg_exprs)
