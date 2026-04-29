"""Comprehensive tests for easyweaver.queries.operations.group_by."""
import polars as pl
import pytest

from easyweaver.queries.operations.group_by import apply_group_by


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def group_df(operations_data):
    return pl.DataFrame(operations_data["group_by_data"]["sample_rows"])


def _result_for_dept(result: pl.DataFrame, dept: str) -> dict:
    row = result.filter(pl.col("dept") == dept).to_dicts()
    assert len(row) == 1, f"Expected exactly one row for dept={dept!r}"
    return row[0]


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------


class TestAggregations:
    def test_count(self, group_df):
        spec = {
            "group_columns": ["dept"],
            "aggregations": [{"column": "name", "function": "count", "alias": "headcount"}],
        }
        result = apply_group_by(group_df, spec)
        eng = _result_for_dept(result, "engineering")
        sales = _result_for_dept(result, "sales")
        assert eng["headcount"] == 2
        assert sales["headcount"] == 3

    def test_sum(self, group_df):
        spec = {
            "group_columns": ["dept"],
            "aggregations": [{"column": "salary", "function": "sum", "alias": "total_salary"}],
        }
        result = apply_group_by(group_df, spec)
        eng = _result_for_dept(result, "engineering")
        sales = _result_for_dept(result, "sales")
        assert eng["total_salary"] == 220000
        assert sales["total_salary"] == 255000

    def test_avg(self, group_df):
        spec = {
            "group_columns": ["dept"],
            "aggregations": [{"column": "salary", "function": "avg", "alias": "avg_salary"}],
        }
        result = apply_group_by(group_df, spec)
        eng = _result_for_dept(result, "engineering")
        assert eng["avg_salary"] == pytest.approx(110000.0)

    def test_min(self, group_df):
        spec = {
            "group_columns": ["dept"],
            "aggregations": [{"column": "salary", "function": "min", "alias": "min_salary"}],
        }
        result = apply_group_by(group_df, spec)
        sales = _result_for_dept(result, "sales")
        assert sales["min_salary"] == 80000

    def test_max(self, group_df):
        spec = {
            "group_columns": ["dept"],
            "aggregations": [{"column": "salary", "function": "max", "alias": "max_salary"}],
        }
        result = apply_group_by(group_df, spec)
        sales = _result_for_dept(result, "sales")
        assert sales["max_salary"] == 90000

    def test_count_distinct(self, group_df):
        # Add duplicate names to test count_distinct properly
        extra = pl.DataFrame(
            {"dept": ["engineering"], "name": ["Alice"], "salary": [100000]}
        )
        df = pl.concat([group_df, extra])
        spec = {
            "group_columns": ["dept"],
            "aggregations": [{"column": "name", "function": "count_distinct", "alias": "unique_names"}],
        }
        result = apply_group_by(df, spec)
        eng = _result_for_dept(result, "engineering")
        # Alice appears twice; count_distinct should be 2 (Alice, Bob)
        assert eng["unique_names"] == 2


# ---------------------------------------------------------------------------
# Alias handling
# ---------------------------------------------------------------------------


class TestAlias:
    def test_default_alias_when_not_specified(self, group_df):
        spec = {
            "group_columns": ["dept"],
            "aggregations": [{"column": "salary", "function": "sum"}],
        }
        result = apply_group_by(group_df, spec)
        assert "sum_salary" in result.columns

    def test_custom_alias(self, group_df):
        spec = {
            "group_columns": ["dept"],
            "aggregations": [{"column": "salary", "function": "sum", "alias": "my_sum"}],
        }
        result = apply_group_by(group_df, spec)
        assert "my_sum" in result.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_missing_group_column_skipped(self, group_df):
        spec = {
            "group_columns": ["nonexistent_col"],
            "aggregations": [{"column": "salary", "function": "sum", "alias": "total"}],
        }
        # No valid group cols → return df unchanged
        result = apply_group_by(group_df, spec)
        assert result.shape == group_df.shape

    def test_missing_agg_column_skipped(self, group_df):
        spec = {
            "group_columns": ["dept"],
            "aggregations": [{"column": "nonexistent_col", "function": "sum", "alias": "x"}],
        }
        # No valid agg exprs → return df unchanged
        result = apply_group_by(group_df, spec)
        assert result.shape == group_df.shape

    def test_unknown_function_skipped(self, group_df):
        spec = {
            "group_columns": ["dept"],
            "aggregations": [{"column": "salary", "function": "median", "alias": "x"}],
        }
        # Unknown function → expression skipped → no valid aggs → return df unchanged
        result = apply_group_by(group_df, spec)
        assert result.shape == group_df.shape

    def test_no_aggregations_key(self, group_df):
        spec = {"group_columns": ["dept"]}
        result = apply_group_by(group_df, spec)
        assert result.shape == group_df.shape

    def test_empty_aggregations_list(self, group_df):
        spec = {"group_columns": ["dept"], "aggregations": []}
        result = apply_group_by(group_df, spec)
        assert result.shape == group_df.shape

    def test_multiple_group_columns(self):
        df = pl.DataFrame(
            {
                "dept": ["eng", "eng", "sales", "sales"],
                "region": ["US", "EU", "US", "EU"],
                "salary": [100, 110, 80, 90],
            }
        )
        spec = {
            "group_columns": ["dept", "region"],
            "aggregations": [{"column": "salary", "function": "sum", "alias": "total"}],
        }
        result = apply_group_by(df, spec)
        assert len(result) == 4

    def test_mix_valid_invalid_agg_cols(self, group_df):
        spec = {
            "group_columns": ["dept"],
            "aggregations": [
                {"column": "salary", "function": "sum", "alias": "total"},
                {"column": "missing_col", "function": "count", "alias": "cnt"},
            ],
        }
        result = apply_group_by(group_df, spec)
        # Only valid agg runs → "total" column present, "cnt" absent
        assert "total" in result.columns
        assert "cnt" not in result.columns
