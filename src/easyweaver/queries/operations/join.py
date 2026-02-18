import polars as pl


def execute_polars_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_on: str,
    right_on: str,
    how: str = "inner",
) -> pl.DataFrame:
    how_map = {"inner": "inner", "left": "left", "right": "right", "outer": "full"}
    polars_how = how_map.get(how, "inner")

    # Coerce types for safe join
    if left.schema.get(left_on) != right.schema.get(right_on):
        left = left.with_columns(pl.col(left_on).cast(pl.Utf8))
        right = right.with_columns(pl.col(right_on).cast(pl.Utf8))

    return left.join(right, left_on=left_on, right_on=right_on, how=polars_how, suffix="_right")
