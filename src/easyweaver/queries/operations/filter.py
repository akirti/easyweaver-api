import polars as pl


def apply_filters(df: pl.DataFrame, filters: list[dict]) -> pl.DataFrame:
    for f in filters:
        col = f["column"]
        op = f["operator"]
        val = f.get("value")

        if col not in df.columns:
            continue

        if op == "eq":
            df = df.filter(pl.col(col) == val)
        elif op == "neq":
            df = df.filter(pl.col(col) != val)
        elif op == "gt":
            df = df.filter(pl.col(col) > val)
        elif op == "lt":
            df = df.filter(pl.col(col) < val)
        elif op == "gte":
            df = df.filter(pl.col(col) >= val)
        elif op == "lte":
            df = df.filter(pl.col(col) <= val)
        elif op == "like":
            df = df.filter(pl.col(col).str.contains(val))
        elif op == "is_null":
            df = df.filter(pl.col(col).is_null())
        elif op == "is_not_null":
            df = df.filter(pl.col(col).is_not_null())

    return df
