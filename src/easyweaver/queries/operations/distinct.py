import polars as pl
import structlog

logger = structlog.get_logger()


def apply_distinct(df: pl.DataFrame, spec: dict) -> pl.DataFrame:
    """Apply distinct/deduplication to a DataFrame."""
    if not spec.get("enabled", True):
        return df

    columns = spec.get("columns")
    keep = spec.get("keep", "first")

    if columns:
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return df
        return df.unique(subset=valid_cols, keep=keep)
    else:
        return df.unique(keep=keep)
