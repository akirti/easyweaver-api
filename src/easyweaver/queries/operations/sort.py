import polars as pl


def apply_sort(df: pl.DataFrame, sorts: list[dict]) -> pl.DataFrame:
    if not sorts:
        return df
    columns = [s["column"] for s in sorts if s["column"] in df.columns]
    descending = [
        s.get("direction", "asc") == "desc" for s in sorts if s["column"] in df.columns
    ]
    if not columns:
        return df
    return df.sort(columns, descending=descending)
