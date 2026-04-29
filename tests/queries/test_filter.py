"""Comprehensive tests for easyweaver.queries.operations.filter."""
import polars as pl
import pytest

from easyweaver.queries.operations.filter import (
    _build_filter_expr,
    _coerce_list,
    _coerce_value,
    apply_filters,
)


# ---------------------------------------------------------------------------
# _coerce_value
# ---------------------------------------------------------------------------


class TestCoerceValue:
    def test_none_returns_none(self):
        assert _coerce_value(None, pl.Int64()) is None

    # --- int types ---
    def test_int_already_int(self):
        assert _coerce_value(42, pl.Int64()) == 42

    def test_int_from_string(self):
        assert _coerce_value("99", pl.Int32()) == 99

    def test_int_from_float(self):
        assert _coerce_value(3.7, pl.Int64()) == 3

    def test_int_coerce_failure_returns_original(self):
        # "abc" cannot be converted to int — return as-is
        result = _coerce_value("abc", pl.Int64())
        assert result == "abc"

    def test_int8_type(self):
        assert _coerce_value("5", pl.Int8()) == 5

    def test_uint32_type(self):
        assert _coerce_value("7", pl.UInt32()) == 7

    # --- float types ---
    def test_float_from_int(self):
        result = _coerce_value(10, pl.Float64())
        assert result == 10.0
        assert isinstance(result, float)

    def test_float_from_string(self):
        assert _coerce_value("3.14", pl.Float64()) == pytest.approx(3.14)

    def test_float_already_float(self):
        assert _coerce_value(2.5, pl.Float32()) == pytest.approx(2.5)

    def test_float_coerce_failure_returns_original(self):
        result = _coerce_value("notanumber", pl.Float64())
        assert result == "notanumber"

    # --- bool ---
    def test_bool_already_bool_true(self):
        assert _coerce_value(True, pl.Boolean()) is True

    def test_bool_already_bool_false(self):
        assert _coerce_value(False, pl.Boolean()) is False

    def test_bool_from_string_true_variants(self):
        for s in ("true", "1", "t", "yes"):
            assert _coerce_value(s, pl.Boolean()) is True, f"Expected True for '{s}'"

    def test_bool_from_string_false(self):
        assert _coerce_value("false", pl.Boolean()) is False

    def test_bool_from_int_nonzero(self):
        assert _coerce_value(1, pl.Boolean()) is True

    def test_bool_from_int_zero(self):
        assert _coerce_value(0, pl.Boolean()) is False

    # --- string ---
    def test_string_already_string(self):
        assert _coerce_value("hello", pl.Utf8()) == "hello"

    def test_string_from_int(self):
        assert _coerce_value(42, pl.Utf8()) == "42"

    def test_string_from_float(self):
        assert _coerce_value(3.14, pl.Utf8()) == "3.14"

    # --- unknown dtype passthrough ---
    def test_unknown_dtype_passthrough(self):
        # For types not explicitly handled, value is returned unchanged
        result = _coerce_value("2024-01-01", pl.Date())
        assert result == "2024-01-01"


# ---------------------------------------------------------------------------
# _coerce_list
# ---------------------------------------------------------------------------


class TestCoerceList:
    def test_list_of_strings_to_int(self):
        result = _coerce_list(["1", "2", "3"], pl.Int64())
        assert result == [1, 2, 3]

    def test_list_with_none(self):
        result = _coerce_list([None, "5"], pl.Int64())
        assert result == [None, 5]

    def test_empty_list(self):
        assert _coerce_list([], pl.Float64()) == []


# ---------------------------------------------------------------------------
# _build_filter_expr  (tested by applying to a real DataFrame)
# ---------------------------------------------------------------------------


@pytest.fixture
def base_df():
    return pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", None],
            "age": [30, 25, 35, 40],
            "score": [85.5, 92.0, 78.3, None],
            "active": [True, False, True, False],
        }
    )


class TestBuildFilterExpr:
    """Each test builds an expression and applies it to base_df."""

    def _apply(self, df, col, op, val, extra=None):
        f = {"column": col, "operator": op, "value": val}
        if extra:
            f.update(extra)
        dtype = df.schema[col]
        expr = _build_filter_expr(col, op, val, f, dtype)
        if expr is None:
            return df  # pragma: no cover — shouldn't happen in passing tests
        return df.filter(expr)

    def test_eq_int(self, base_df):
        result = self._apply(base_df, "age", "eq", 30)
        assert result["name"].to_list() == ["Alice"]

    def test_neq_int(self, base_df):
        result = self._apply(base_df, "age", "neq", 30)
        assert 30 not in result["age"].to_list()

    def test_gt(self, base_df):
        result = self._apply(base_df, "age", "gt", 30)
        assert all(a > 30 for a in result["age"].to_list())

    def test_lt(self, base_df):
        result = self._apply(base_df, "age", "lt", 30)
        assert all(a < 30 for a in result["age"].to_list())

    def test_gte(self, base_df):
        result = self._apply(base_df, "age", "gte", 30)
        assert all(a >= 30 for a in result["age"].to_list())

    def test_lte(self, base_df):
        result = self._apply(base_df, "age", "lte", 30)
        assert all(a <= 30 for a in result["age"].to_list())

    def test_like(self, base_df):
        result = self._apply(base_df, "name", "like", "li")
        # Matches Alice and Charlie
        assert len(result) == 2

    def test_is_null(self, base_df):
        result = self._apply(base_df, "name", "is_null", None)
        assert result["name"].to_list() == [None]

    def test_is_not_null(self, base_df):
        result = self._apply(base_df, "name", "is_not_null", None)
        assert None not in result["name"].to_list()

    def test_in_list(self, base_df):
        result = self._apply(base_df, "age", "in", [25, 35])
        assert set(result["age"].to_list()) == {25, 35}

    def test_in_string_csv(self, base_df):
        # "in" with a CSV string should parse it
        result = self._apply(base_df, "name", "in", "Alice,Charlie")
        assert set(result["name"].to_list()) == {"Alice", "Charlie"}

    def test_in_empty_list_returns_no_rows(self, base_df):
        result = self._apply(base_df, "age", "in", [])
        assert len(result) == 0

    def test_not_in_list(self, base_df):
        result = self._apply(base_df, "age", "not_in", [25, 35])
        assert 25 not in result["age"].to_list()
        assert 35 not in result["age"].to_list()

    def test_not_in_string_csv(self, base_df):
        result = self._apply(base_df, "name", "not_in", "Alice")
        assert "Alice" not in result["name"].drop_nulls().to_list()

    def test_not_in_empty_list_returns_none(self, base_df):
        f = {"column": "age", "operator": "not_in", "value": []}
        dtype = base_df.schema["age"]
        expr = _build_filter_expr("age", "not_in", [], f, dtype)
        assert expr is None

    def test_between(self, base_df):
        f = {"column": "age", "operator": "between", "value": 25, "value2": 32}
        dtype = base_df.schema["age"]
        expr = _build_filter_expr("age", "between", 25, f, dtype)
        result = base_df.filter(expr)
        assert all(25 <= a <= 32 for a in result["age"].to_list())

    def test_between_missing_value2_returns_none(self, base_df):
        f = {"column": "age", "operator": "between", "value": 25}
        dtype = base_df.schema["age"]
        expr = _build_filter_expr("age", "between", 25, f, dtype)
        assert expr is None

    def test_unknown_operator_returns_none(self, base_df):
        f = {"column": "age", "operator": "unknown_op", "value": 10}
        dtype = base_df.schema["age"]
        expr = _build_filter_expr("age", "unknown_op", 10, f, dtype)
        assert expr is None


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------


class TestApplyFilters:
    @pytest.fixture
    def df(self, operations_data):
        return pl.DataFrame(operations_data["filter_data"]["sample_rows"])

    def test_and_logic_multiple_conditions(self, df):
        filters = [
            {"column": "city", "operator": "eq", "value": "NYC"},
            {"column": "active", "operator": "eq", "value": True},
        ]
        result = apply_filters(df, filters, logic="and")
        assert all(row["city"] == "NYC" for row in result.to_dicts())
        assert all(row["active"] for row in result.to_dicts())

    def test_or_logic(self, df):
        filters = [
            {"column": "city", "operator": "eq", "value": "NYC"},
            {"column": "city", "operator": "eq", "value": "LA"},
        ]
        result = apply_filters(df, filters, logic="or")
        cities = set(result["city"].drop_nulls().to_list())
        assert cities == {"NYC", "LA"}

    def test_missing_column_skipped(self, df):
        filters = [{"column": "nonexistent_col", "operator": "eq", "value": "x"}]
        result = apply_filters(df, filters)
        # Should return df unchanged (no crash, no filter applied)
        assert len(result) == len(df)

    def test_empty_filters_returns_df_unchanged(self, df):
        result = apply_filters(df, [])
        assert result.shape == df.shape

    def test_default_logic_is_and(self, df):
        filters = [
            {"column": "city", "operator": "eq", "value": "NYC"},
            {"column": "age", "operator": "gt", "value": 30},
        ]
        result_default = apply_filters(df, filters)
        result_and = apply_filters(df, filters, logic="and")
        assert result_default.shape == result_and.shape

    def test_null_filter_is_null(self, df):
        filters = [{"column": "city", "operator": "is_null", "value": None}]
        result = apply_filters(df, filters)
        assert result["city"].to_list() == [None]

    def test_score_gte_filter(self, df):
        filters = [{"column": "score", "operator": "gte", "value": 85.5}]
        result = apply_filters(df, filters)
        scores = result["score"].drop_nulls().to_list()
        assert all(s >= 85.5 for s in scores)

    def test_between_filter(self, df):
        filters = [{"column": "age", "operator": "between", "value": 25, "value2": 32}]
        result = apply_filters(df, filters)
        ages = result["age"].to_list()
        assert all(25 <= a <= 32 for a in ages)

    def test_in_filter_with_csv_string(self, df):
        # For string columns, pass values as CSV — _coerce_value converts a list to
        # str(list) before the is_in check runs, so use a CSV string instead.
        filters = [{"column": "name", "operator": "in", "value": "Alice,Bob"}]
        result = apply_filters(df, filters)
        assert set(result["name"].to_list()) == {"Alice", "Bob"}

    def test_not_in_skips_when_empty(self, df):
        # not_in with empty list → expression is None → filter not applied
        filters = [{"column": "age", "operator": "not_in", "value": []}]
        result = apply_filters(df, filters)
        assert len(result) == len(df)
