import polars as pl

_PL_INT_TYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
_PL_FLOAT_TYPES = {pl.Float32, pl.Float64}


def _coerce_value(value, dtype: pl.DataType):
    """Coerce a filter value to match the Polars column dtype."""
    if value is None:
        return None
    dtype_base = type(dtype)
    if dtype_base in _PL_INT_TYPES:
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if dtype_base in _PL_FLOAT_TYPES:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if dtype_base is pl.Boolean:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "t", "yes")
        return bool(value)
    if dtype_base is pl.Utf8 or dtype_base is pl.String:
        # Coerce numeric/other values to string for string columns
        if not isinstance(value, str):
            return str(value)
    return value


def _coerce_list(values: list, dtype: pl.DataType) -> list:
    """Coerce a list of filter values to match the Polars column dtype."""
    return [_coerce_value(v, dtype) for v in values]


def _build_filter_expr(col: str, op: str, val, f: dict, dtype: pl.DataType) -> pl.Expr | None:
    """Build a single Polars filter expression for the given operator."""
    val = _coerce_value(val, dtype)
    if op == "eq":
        return pl.col(col) == val
    elif op == "neq":
        return pl.col(col) != val
    elif op == "gt":
        return pl.col(col) > val
    elif op == "lt":
        return pl.col(col) < val
    elif op == "gte":
        return pl.col(col) >= val
    elif op == "lte":
        return pl.col(col) <= val
    elif op == "like":
        return pl.col(col).cast(pl.Utf8).str.contains(str(val), literal=False)
    elif op == "is_null":
        return pl.col(col).is_null()
    elif op == "is_not_null":
        return pl.col(col).is_not_null()
    elif op == "in":
        values = val if isinstance(val, list) else []
        if values:
            return pl.col(col).is_in(_coerce_list(values, dtype))
        else:
            return pl.lit(False)
    elif op == "not_in":
        values = val if isinstance(val, list) else []
        if values:
            return ~pl.col(col).is_in(_coerce_list(values, dtype))
        return None
    elif op == "between":
        val2 = _coerce_value(f.get("value2"), dtype)
        if val is not None and val2 is not None:
            return pl.col(col).is_between(val, val2)
    return None


def apply_filters(df: pl.DataFrame, filters: list[dict], logic: str = "and") -> pl.DataFrame:
    expressions = []
    for f in filters:
        col = f["column"]
        op = f["operator"]
        val = f.get("value")

        if col not in df.columns:
            continue

        dtype = df.schema[col]
        expr = _build_filter_expr(col, op, val, f, dtype)
        if expr is not None:
            expressions.append(expr)

    if not expressions:
        return df

    if logic == "or":
        combined = expressions[0]
        for e in expressions[1:]:
            combined = combined | e
        return df.filter(combined)
    else:
        combined = expressions[0]
        for e in expressions[1:]:
            combined = combined & e
        return df.filter(combined)
