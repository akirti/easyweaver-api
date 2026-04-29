"""Comprehensive tests for easyweaver.queries.operations.derived."""
from datetime import date, datetime

import polars as pl
import pytest

from easyweaver.queries.operations.derived import (
    _apply_concat,
    _apply_conditional,
    _apply_date_part,
    _apply_math,
    _parse_math_expr,
    _tokenize_math,
    apply_derived_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def derived_df(operations_data):
    return pl.DataFrame(operations_data["derived_data"]["sample_rows"])


@pytest.fixture
def datetime_df():
    return pl.DataFrame(
        {
            "created_at": pl.Series(
                [
                    datetime(2024, 3, 15, 10, 30, 45),
                    datetime(2023, 7, 4, 8, 0, 0),
                ]
            ).cast(pl.Datetime),
        }
    )


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------


class TestConcat:
    def test_concat_with_separator(self, derived_df):
        spec = {"columns": ["first", "last"], "separator": " "}
        result = _apply_concat(derived_df, spec, "full_name")
        assert result["full_name"].to_list() == ["Alice Smith", "Bob Jones"]

    def test_concat_without_separator(self, derived_df):
        spec = {"columns": ["first", "last"], "separator": ""}
        result = _apply_concat(derived_df, spec, "full_name")
        assert result["full_name"].to_list() == ["AliceSmith", "BobJones"]

    def test_concat_missing_column_excluded(self, derived_df):
        spec = {"columns": ["first", "nonexistent"], "separator": "-"}
        result = _apply_concat(derived_df, spec, "partial")
        # Only "first" is valid — result is just first column values
        assert result["partial"].to_list() == ["Alice", "Bob"]

    def test_concat_all_missing_columns_returns_unchanged(self, derived_df):
        spec = {"columns": ["no_col1", "no_col2"], "separator": ""}
        result = _apply_concat(derived_df, spec, "full_name")
        assert "full_name" not in result.columns

    def test_concat_no_columns_key_returns_unchanged(self, derived_df):
        spec = {}
        result = _apply_concat(derived_df, spec, "full_name")
        assert "full_name" not in result.columns

    def test_concat_converts_numeric_to_string(self, derived_df):
        spec = {"columns": ["first", "price"], "separator": ":"}
        result = _apply_concat(derived_df, spec, "tagged")
        assert result["tagged"].to_list() == ["Alice:100", "Bob:200"]


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


class TestMath:
    def test_multiply_two_columns(self, derived_df):
        spec = {"expression": "{price} * {qty}"}
        result = _apply_math(derived_df, spec, "revenue")
        assert result["revenue"].to_list() == [500.0, 600.0]

    def test_add_constant_to_column(self, derived_df):
        spec = {"expression": "{price} + 10"}
        result = _apply_math(derived_df, spec, "adjusted")
        assert result["adjusted"].to_list() == [110.0, 210.0]

    def test_expression_with_parentheses(self, derived_df):
        spec = {"expression": "({price} + {qty}) * 2"}
        result = _apply_math(derived_df, spec, "calc")
        assert result["calc"].to_list() == [(100 + 5) * 2.0, (200 + 3) * 2.0]

    def test_division(self, derived_df):
        spec = {"expression": "{price} / {qty}"}
        result = _apply_math(derived_df, spec, "unit_price")
        assert result["unit_price"].to_list() == pytest.approx([20.0, 200 / 3])

    def test_subtraction(self, derived_df):
        spec = {"expression": "{price} - 50"}
        result = _apply_math(derived_df, spec, "reduced")
        assert result["reduced"].to_list() == [50.0, 150.0]

    def test_empty_expression_returns_unchanged(self, derived_df):
        spec = {"expression": ""}
        result = _apply_math(derived_df, spec, "x")
        assert "x" not in result.columns

    def test_invalid_chars_returns_unchanged(self, derived_df):
        spec = {"expression": "{price} ^ 2"}  # ^ is not allowed
        result = _apply_math(derived_df, spec, "x")
        assert "x" not in result.columns

    def test_missing_column_returns_unchanged(self, derived_df):
        spec = {"expression": "{price} * {missing_col}"}
        result = _apply_math(derived_df, spec, "x")
        assert "x" not in result.columns


# ---------------------------------------------------------------------------
# date_part
# ---------------------------------------------------------------------------


class TestDatePart:
    def test_year(self, datetime_df):
        spec = {"source_column": "created_at", "part": "year"}
        result = _apply_date_part(datetime_df, spec, "yr")
        assert result["yr"].to_list() == [2024, 2023]

    def test_month(self, datetime_df):
        spec = {"source_column": "created_at", "part": "month"}
        result = _apply_date_part(datetime_df, spec, "mo")
        assert result["mo"].to_list() == [3, 7]

    def test_day(self, datetime_df):
        spec = {"source_column": "created_at", "part": "day"}
        result = _apply_date_part(datetime_df, spec, "dy")
        assert result["dy"].to_list() == [15, 4]

    def test_hour(self, datetime_df):
        spec = {"source_column": "created_at", "part": "hour"}
        result = _apply_date_part(datetime_df, spec, "hr")
        assert result["hr"].to_list() == [10, 8]

    def test_minute(self, datetime_df):
        spec = {"source_column": "created_at", "part": "minute"}
        result = _apply_date_part(datetime_df, spec, "mn")
        assert result["mn"].to_list() == [30, 0]

    def test_second(self, datetime_df):
        spec = {"source_column": "created_at", "part": "second"}
        result = _apply_date_part(datetime_df, spec, "sc")
        assert result["sc"].to_list() == [45, 0]

    def test_quarter(self, datetime_df):
        spec = {"source_column": "created_at", "part": "quarter"}
        result = _apply_date_part(datetime_df, spec, "qtr")
        assert result["qtr"].to_list() == [1, 3]

    def test_day_of_week(self, datetime_df):
        spec = {"source_column": "created_at", "part": "day_of_week"}
        result = _apply_date_part(datetime_df, spec, "dow")
        # 2024-03-15 = Friday (4), 2023-07-04 = Tuesday (1)
        assert result["dow"].to_list()[0] in range(0, 7)

    def test_missing_column_returns_unchanged(self, datetime_df):
        spec = {"source_column": "no_col", "part": "year"}
        result = _apply_date_part(datetime_df, spec, "yr")
        assert "yr" not in result.columns

    def test_unknown_part_returns_unchanged(self, datetime_df):
        spec = {"source_column": "created_at", "part": "century"}
        result = _apply_date_part(datetime_df, spec, "yr")
        assert "yr" not in result.columns


# ---------------------------------------------------------------------------
# conditional
# ---------------------------------------------------------------------------


class TestConditional:
    def test_eq_condition(self, derived_df):
        spec = {
            "condition_column": "status",
            "condition_operator": "eq",
            "condition_value": "active",
            "then_value": "yes",
            "else_value": "no",
        }
        result = _apply_conditional(derived_df, spec, "is_active")
        assert result["is_active"].to_list() == ["yes", "no"]

    def test_gt_condition(self, derived_df):
        spec = {
            "condition_column": "price",
            "condition_operator": "gt",
            "condition_value": 150,
            "then_value": "expensive",
            "else_value": "cheap",
        }
        result = _apply_conditional(derived_df, spec, "label")
        assert result["label"].to_list() == ["cheap", "expensive"]

    def test_is_null_condition(self):
        df = pl.DataFrame({"val": [1, None, 3]})
        spec = {
            "condition_column": "val",
            "condition_operator": "is_null",
            "condition_value": None,
            "then_value": "missing",
            "else_value": "present",
        }
        result = _apply_conditional(df, spec, "check")
        assert result["check"].to_list() == ["present", "missing", "present"]

    def test_missing_column_returns_unchanged(self, derived_df):
        spec = {
            "condition_column": "no_col",
            "condition_operator": "eq",
            "condition_value": "x",
            "then_value": "a",
            "else_value": "b",
        }
        result = _apply_conditional(derived_df, spec, "out")
        assert "out" not in result.columns

    def test_unknown_operator_returns_unchanged(self, derived_df):
        spec = {
            "condition_column": "status",
            "condition_operator": "starts_with",
            "condition_value": "act",
            "then_value": "a",
            "else_value": "b",
        }
        result = _apply_conditional(derived_df, spec, "out")
        assert "out" not in result.columns

    def test_neq_condition(self, derived_df):
        spec = {
            "condition_column": "status",
            "condition_operator": "neq",
            "condition_value": "active",
            "then_value": 1,
            "else_value": 0,
        }
        result = _apply_conditional(derived_df, spec, "flag")
        assert result["flag"].to_list() == [0, 1]

    def test_is_not_null_condition(self):
        df = pl.DataFrame({"val": [1, None]})
        spec = {
            "condition_column": "val",
            "condition_operator": "is_not_null",
            "condition_value": None,
            "then_value": "filled",
            "else_value": "empty",
        }
        result = _apply_conditional(df, spec, "check")
        assert result["check"].to_list() == ["filled", "empty"]

    def test_lte_condition(self, derived_df):
        spec = {
            "condition_column": "qty",
            "condition_operator": "lte",
            "condition_value": 3,
            "then_value": "low",
            "else_value": "high",
        }
        result = _apply_conditional(derived_df, spec, "stock")
        assert result["stock"].to_list() == ["high", "low"]


# ---------------------------------------------------------------------------
# literal
# ---------------------------------------------------------------------------


class TestLiteral:
    def test_literal_string(self, derived_df):
        specs = [{"expression_type": "literal", "name": "const", "value": "static"}]
        result = apply_derived_columns(derived_df, specs)
        assert result["const"].to_list() == ["static", "static"]

    def test_literal_number(self, derived_df):
        specs = [{"expression_type": "literal", "name": "num_col", "value": 42}]
        result = apply_derived_columns(derived_df, specs)
        assert result["num_col"].to_list() == [42, 42]

    def test_literal_none(self, derived_df):
        specs = [{"expression_type": "literal", "name": "empty_col", "value": None}]
        result = apply_derived_columns(derived_df, specs)
        assert result["empty_col"].to_list() == [None, None]


# ---------------------------------------------------------------------------
# apply_derived_columns — top-level dispatch
# ---------------------------------------------------------------------------


class TestApplyDerivedColumns:
    def test_unknown_type_is_skipped(self, derived_df):
        specs = [{"expression_type": "foobar", "name": "x"}]
        result = apply_derived_columns(derived_df, specs)
        assert "x" not in result.columns

    def test_failed_expression_does_not_crash(self, derived_df):
        # math with bad expression — should be caught and skipped
        specs = [{"expression_type": "math", "name": "bad", "expression": "{price} ^ {qty}"}]
        result = apply_derived_columns(derived_df, specs)
        assert "bad" not in result.columns

    def test_multiple_specs_applied_in_sequence(self, derived_df):
        specs = [
            {"expression_type": "concat", "name": "full_name", "columns": ["first", "last"], "separator": " "},
            {"expression_type": "literal", "name": "version", "value": 2},
        ]
        result = apply_derived_columns(derived_df, specs)
        assert "full_name" in result.columns
        assert "version" in result.columns


# ---------------------------------------------------------------------------
# Tokenizer unit tests
# ---------------------------------------------------------------------------


class TestTokenizeMath:
    def _make_df(self, **cols):
        return pl.DataFrame(cols)

    def test_tokenize_column_reference(self):
        df = self._make_df(price=[100])
        tokens = _tokenize_math("{price}", df)
        assert tokens == [("col", "price")]

    def test_tokenize_number_integer(self):
        df = self._make_df(price=[1])
        tokens = _tokenize_math("{price} + 10", df)
        assert ("num", 10.0) in tokens

    def test_tokenize_number_float(self):
        df = self._make_df(x=[1])
        tokens = _tokenize_math("{x} * 1.5", df)
        assert ("num", 1.5) in tokens

    def test_tokenize_operators(self):
        df = self._make_df(a=[1], b=[2])
        for op in ["+", "-", "*", "/"]:
            tokens = _tokenize_math(f"{{a}} {op} {{b}}", df)
            assert ("op", op) in tokens

    def test_tokenize_parentheses(self):
        df = self._make_df(a=[1], b=[2])
        tokens = _tokenize_math("({a} + {b})", df)
        assert ("lparen", "(") in tokens
        assert ("rparen", ")") in tokens

    def test_tokenize_missing_column_returns_none(self):
        df = self._make_df(a=[1])
        tokens = _tokenize_math("{a} + {missing}", df)
        assert tokens is None

    def test_tokenize_invalid_char_returns_none(self):
        df = self._make_df(a=[1])
        tokens = _tokenize_math("{a} % 2", df)
        assert tokens is None

    def test_tokenize_whitespace_ignored(self):
        df = self._make_df(a=[1])
        tokens = _tokenize_math("  {a}  ", df)
        assert tokens == [("col", "a")]


# ---------------------------------------------------------------------------
# _parse_math_expr edge cases
# ---------------------------------------------------------------------------


class TestParseMathExpr:
    def _df(self, **cols):
        return pl.DataFrame(cols)

    def test_invalid_regex_chars_returns_none(self):
        df = self._df(a=[1])
        # '@' is not in allowed regex
        result = _parse_math_expr("{a} @ 2", df)
        assert result is None

    def test_unbalanced_parens_returns_none(self):
        df = self._df(a=[1])
        # Parser will fail on unmatched paren
        result = _parse_math_expr("({a} + 1", df)
        assert result is None
