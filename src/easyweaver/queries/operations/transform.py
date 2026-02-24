import polars as pl
import structlog

logger = structlog.get_logger()

_CAST_MAP: dict[str, pl.DataType] = {
    "string": pl.Utf8,
    "integer": pl.Int64,
    "float": pl.Float64,
    "boolean": pl.Boolean,
    "date": pl.Date,
    "datetime": pl.Datetime,
}


def apply_transforms(df: pl.DataFrame, transforms: list[dict]) -> pl.DataFrame:
    """Apply column transforms sequentially. Unknown columns or failed transforms are skipped."""
    for t in transforms:
        col = t["column"]
        ttype = t["type"]

        if col not in df.columns:
            logger.warning("transform_skip_missing_column", column=col, transform=ttype)
            continue

        try:
            if ttype == "rename":
                new_name = t.get("new_name")
                if new_name:
                    df = df.rename({col: new_name})
            elif ttype == "format_date":
                fmt = t.get("date_format")
                if fmt:
                    df = df.with_columns(pl.col(col).dt.strftime(fmt).alias(col))
            elif ttype == "round":
                decimals = t.get("decimals", 0)
                df = df.with_columns(pl.col(col).round(decimals).alias(col))
            elif ttype == "uppercase":
                df = df.with_columns(pl.col(col).str.to_uppercase().alias(col))
            elif ttype == "lowercase":
                df = df.with_columns(pl.col(col).str.to_lowercase().alias(col))
            elif ttype == "trim":
                df = df.with_columns(pl.col(col).str.strip_chars().alias(col))
            elif ttype == "cast":
                target = t.get("target_type")
                if target and target in _CAST_MAP:
                    df = df.with_columns(pl.col(col).cast(_CAST_MAP[target]).alias(col))
            else:
                logger.warning("transform_unknown_type", column=col, transform=ttype)
        except Exception:
            logger.warning("transform_failed", column=col, transform=ttype, exc_info=True)

    return df
