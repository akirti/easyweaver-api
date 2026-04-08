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


def _ensure_string_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Cast column to Utf8 if it isn't already, so string operations work."""
    if df.schema[col] != pl.Utf8:
        df = df.with_columns(pl.col(col).cast(pl.Utf8).alias(col))
    return df


def _apply_rename(df: pl.DataFrame, col: str, t: dict) -> pl.DataFrame:
    new_name = t.get("new_name")
    if new_name:
        df = df.rename({col: new_name})
    return df


def _apply_format_date(df: pl.DataFrame, col: str, t: dict) -> pl.DataFrame:
    fmt = t.get("date_format")
    if fmt:
        df = df.with_columns(pl.col(col).dt.strftime(fmt).alias(col))
    return df


def _apply_round(df: pl.DataFrame, col: str, t: dict) -> pl.DataFrame:
    decimals = t.get("decimals", 0)
    return df.with_columns(pl.col(col).round(decimals).alias(col))


def _apply_string_op(df: pl.DataFrame, col: str, op: str) -> pl.DataFrame:
    df = _ensure_string_col(df, col)
    if op == "uppercase":
        return df.with_columns(pl.col(col).str.to_uppercase().alias(col))
    if op == "lowercase":
        return df.with_columns(pl.col(col).str.to_lowercase().alias(col))
    # trim
    return df.with_columns(pl.col(col).str.strip_chars().alias(col))


def _apply_cast(df: pl.DataFrame, col: str, t: dict) -> pl.DataFrame:
    target = t.get("target_type")
    if not target or target not in _CAST_MAP:
        return df
    # For casting string → numeric, strip whitespace first
    if target in ("integer", "float") and df.schema[col] == pl.Utf8:
        df = df.with_columns(pl.col(col).str.strip_chars().alias(col))
    return df.with_columns(pl.col(col).cast(_CAST_MAP[target]).alias(col))


def _apply_strip_leading_zeros(df: pl.DataFrame, col: str) -> pl.DataFrame:
    df = _ensure_string_col(df, col)
    df = df.with_columns(
        pl.col(col).str.strip_chars().str.lstrip("0").alias(col)
    )
    # Restore empty strings (was all zeros like "000") to "0"
    return df.with_columns(
        pl.when(pl.col(col) == "").then(pl.lit("0")).otherwise(pl.col(col)).alias(col)
    )


def _apply_pad_left(df: pl.DataFrame, col: str, t: dict) -> pl.DataFrame:
    pad_char = t.get("pad_char", "0")
    pad_length = t.get("pad_length", 1)
    df = _ensure_string_col(df, col)
    return df.with_columns(
        pl.col(col).str.pad_start(pad_length, pad_char).alias(col)
    )


def _apply_replace(df: pl.DataFrame, col: str, t: dict) -> pl.DataFrame:
    find_str = t.get("find_str", "")
    replace_str = t.get("replace_str", "")
    df = _ensure_string_col(df, col)
    return df.with_columns(
        pl.col(col).str.replace_all(find_str, replace_str, literal=True).alias(col)
    )


def _apply_substring(df: pl.DataFrame, col: str, t: dict) -> pl.DataFrame:
    start = t.get("start", 0)
    length = t.get("length")
    df = _ensure_string_col(df, col)
    if length is not None:
        return df.with_columns(pl.col(col).str.slice(start, length).alias(col))
    return df.with_columns(pl.col(col).str.slice(start).alias(col))


_TRANSFORM_HANDLERS = {
    "rename": lambda df, col, t: _apply_rename(df, col, t),
    "format_date": lambda df, col, t: _apply_format_date(df, col, t),
    "round": lambda df, col, t: _apply_round(df, col, t),
    "uppercase": lambda df, col, t: _apply_string_op(df, col, "uppercase"),
    "lowercase": lambda df, col, t: _apply_string_op(df, col, "lowercase"),
    "trim": lambda df, col, t: _apply_string_op(df, col, "trim"),
    "cast": lambda df, col, t: _apply_cast(df, col, t),
    "strip_leading_zeros": lambda df, col, t: _apply_strip_leading_zeros(df, col),
    "pad_left": lambda df, col, t: _apply_pad_left(df, col, t),
    "replace": lambda df, col, t: _apply_replace(df, col, t),
    "substring": lambda df, col, t: _apply_substring(df, col, t),
}


def apply_transforms(df: pl.DataFrame, transforms: list[dict]) -> pl.DataFrame:
    """Apply column transforms sequentially. Unknown columns or failed transforms are skipped."""
    for t in transforms:
        col = t["column"]
        ttype = t["type"]

        if col not in df.columns:
            logger.warning("transform_skip_missing_column", column=col, transform=ttype)
            continue

        handler = _TRANSFORM_HANDLERS.get(ttype)
        if handler is None:
            logger.warning("transform_unknown_type", column=col, transform=ttype)
            continue

        try:
            df = handler(df, col, t)
        except Exception:
            logger.warning("transform_failed", column=col, transform=ttype, exc_info=True)

    return df
