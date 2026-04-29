"""Comprehensive tests for easyweaver.queries.operations.join."""
import polars as pl
import pytest

from easyweaver.queries.operations.join import execute_polars_join


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def left_df(operations_data):
    return pl.DataFrame(operations_data["join_data"]["left_rows"])


@pytest.fixture
def right_df(operations_data):
    return pl.DataFrame(operations_data["join_data"]["right_rows"])


# ---------------------------------------------------------------------------
# Inner join
# ---------------------------------------------------------------------------


class TestInnerJoin:
    def test_inner_join_matching_rows(self, left_df, right_df):
        result = execute_polars_join(left_df, right_df, "id", "id", how="inner")
        # Only ids 1 and 2 are in both
        assert set(result["id"].to_list()) == {1, 2}
        assert len(result) == 2

    def test_inner_join_columns_present(self, left_df, right_df):
        result = execute_polars_join(left_df, right_df, "id", "id", how="inner")
        assert "name" in result.columns
        assert "dept" in result.columns

    def test_inner_join_no_matches_empty_result(self, left_df):
        right = pl.DataFrame({"id": [99, 100], "dept": ["X", "Y"]})
        result = execute_polars_join(left_df, right, "id", "id", how="inner")
        assert result.is_empty()


# ---------------------------------------------------------------------------
# Left join
# ---------------------------------------------------------------------------


class TestLeftJoin:
    def test_left_join_keeps_all_left_rows(self, left_df, right_df):
        result = execute_polars_join(left_df, right_df, "id", "id", how="left")
        assert len(result) == len(left_df)  # All 3 left rows

    def test_left_join_unmatched_right_is_null(self, left_df, right_df):
        result = execute_polars_join(left_df, right_df, "id", "id", how="left")
        # id=3 (Charlie) has no matching right row → dept should be null
        charlie = result.filter(pl.col("id") == 3)
        assert charlie["dept"].to_list() == [None]


# ---------------------------------------------------------------------------
# Right join
# ---------------------------------------------------------------------------


class TestRightJoin:
    def test_right_join_keeps_all_right_rows(self, left_df, right_df):
        result = execute_polars_join(left_df, right_df, "id", "id", how="right")
        assert len(result) == len(right_df)  # All 3 right rows

    def test_right_join_unmatched_left_is_null(self, left_df, right_df):
        result = execute_polars_join(left_df, right_df, "id", "id", how="right")
        # id=4 (Marketing) has no matching left row → name should be null
        marketing = result.filter(pl.col("id") == 4)
        assert marketing["name"].to_list() == [None]


# ---------------------------------------------------------------------------
# Outer (full) join
# ---------------------------------------------------------------------------


class TestOuterJoin:
    def test_outer_join_all_rows_present(self, left_df, right_df):
        result = execute_polars_join(left_df, right_df, "id", "id", how="outer")
        # ids: left has 1,2,3; right has 1,2,4 → union = 1,2,3,4
        all_ids = set(result["id"].drop_nulls().to_list())
        # In a full join both sides are coalesced; just check row count
        assert len(result) == 4

    def test_outer_join_includes_unmatched_from_both(self, left_df, right_df):
        result = execute_polars_join(left_df, right_df, "id", "id", how="outer")
        # In a Polars full join the right-only rows have id=null on the left side
        # and the original right key appears in "id_right".
        # id=3 (left-only) should appear in "id" column
        left_ids = set(result["id"].drop_nulls().to_list())
        assert 3 in left_ids
        # id=4 (right-only) appears in "id_right" (the right key column)
        right_ids = set(result["id_right"].drop_nulls().to_list())
        assert 4 in right_ids


# ---------------------------------------------------------------------------
# Unknown join type falls back to inner
# ---------------------------------------------------------------------------


class TestUnknownJoinType:
    def test_unknown_type_defaults_to_inner(self, left_df, right_df):
        result = execute_polars_join(left_df, right_df, "id", "id", how="cross_type_unknown")
        # Falls back to inner
        assert set(result["id"].to_list()) == {1, 2}


# ---------------------------------------------------------------------------
# Type mismatch coercion
# ---------------------------------------------------------------------------


class TestTypeMismatchCoercion:
    def test_int_vs_string_id_coerced(self, right_df):
        # Left has string ids, right has int ids
        left = pl.DataFrame({"id": ["1", "2", "3"], "name": ["Alice", "Bob", "Charlie"]})
        result = execute_polars_join(left, right_df, "id", "id", how="inner")
        # After coercion both become Utf8 → join on string "1"/"2"
        assert len(result) == 2

    def test_same_type_no_coercion(self, left_df, right_df):
        # Both have Int64 id columns
        result = execute_polars_join(left_df, right_df, "id", "id", how="inner")
        assert result["id"].dtype == pl.Int64

    def test_coercion_with_different_column_names(self):
        left = pl.DataFrame({"emp_id": [1, 2, 3], "name": ["A", "B", "C"]})
        right = pl.DataFrame({"dept_id": ["1", "2", "4"], "dept": ["X", "Y", "Z"]})
        result = execute_polars_join(left, right, "emp_id", "dept_id", how="inner")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Suffix on duplicate column names
# ---------------------------------------------------------------------------


class TestSuffix:
    def test_right_suffix_on_duplicate_column(self):
        left = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        right = pl.DataFrame({"id": [1, 2], "name": ["X", "Y"]})
        result = execute_polars_join(left, right, "id", "id", how="inner")
        # Duplicate "name" → right side becomes "name_right"
        assert "name_right" in result.columns
        assert "name" in result.columns
