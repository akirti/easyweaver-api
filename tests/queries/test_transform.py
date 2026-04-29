"""Comprehensive tests for easyweaver.queries.operations.transform."""
import polars as pl
import pytest

from easyweaver.queries.operations.transform import apply_transforms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transform_df(operations_data):
    return pl.DataFrame(operations_data["transform_data"]["sample_rows"])


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------


class TestRename:
    def test_rename_column(self, transform_df):
        result = apply_transforms(
            transform_df, [{"column": "name", "type": "rename", "new_name": "full_name"}]
        )
        assert "full_name" in result.columns
        assert "name" not in result.columns

    def test_rename_no_new_name_is_noop(self, transform_df):
        result = apply_transforms(
            transform_df, [{"column": "name", "type": "rename"}]
        )
        assert "name" in result.columns

    def test_rename_missing_column_is_skipped(self, transform_df):
        result = apply_transforms(
            transform_df, [{"column": "no_such_col", "type": "rename", "new_name": "x"}]
        )
        assert "x" not in result.columns
        assert result.shape == transform_df.shape


# ---------------------------------------------------------------------------
# format_date
# ---------------------------------------------------------------------------


class TestFormatDate:
    def test_format_date_on_date_column(self):
        df = pl.DataFrame({"dt": pl.Series(["2024-03-15"]).str.strptime(pl.Date, "%Y-%m-%d")})
        result = apply_transforms(
            df, [{"column": "dt", "type": "format_date", "date_format": "%d/%m/%Y"}]
        )
        assert result["dt"].to_list() == ["15/03/2024"]

    def test_format_date_no_fmt_is_noop(self):
        df = pl.DataFrame({"dt": pl.Series(["2024-03-15"]).str.strptime(pl.Date, "%Y-%m-%d")})
        result = apply_transforms(df, [{"column": "dt", "type": "format_date"}])
        # No format supplied → column unchanged
        assert result.shape == df.shape


# ---------------------------------------------------------------------------
# round
# ---------------------------------------------------------------------------


class TestRound:
    def test_round_two_decimals(self, transform_df):
        result = apply_transforms(
            transform_df, [{"column": "amount", "type": "round", "decimals": 2}]
        )
        for val in result["amount"].to_list():
            # Value should have at most 2 decimal places
            assert round(val, 2) == pytest.approx(val)

    def test_round_zero_decimals(self, transform_df):
        result = apply_transforms(
            transform_df, [{"column": "amount", "type": "round", "decimals": 0}]
        )
        for val in result["amount"].to_list():
            assert val == int(val)

    def test_round_default_decimals(self, transform_df):
        # Default decimals=0 when not specified
        result = apply_transforms(transform_df, [{"column": "amount", "type": "round"}])
        for val in result["amount"].to_list():
            assert val == int(val)


# ---------------------------------------------------------------------------
# uppercase / lowercase / trim
# ---------------------------------------------------------------------------


class TestStringOps:
    def test_uppercase(self, transform_df):
        result = apply_transforms(
            transform_df, [{"column": "name", "type": "uppercase"}]
        )
        for v in result["name"].to_list():
            assert v == v.upper()

    def test_lowercase(self, transform_df):
        result = apply_transforms(
            transform_df, [{"column": "name", "type": "lowercase"}]
        )
        for v in result["name"].to_list():
            assert v == v.lower()

    def test_trim(self, transform_df):
        result = apply_transforms(
            transform_df, [{"column": "name", "type": "trim"}]
        )
        for v in result["name"].to_list():
            assert v == v.strip()

    def test_uppercase_on_numeric_col_casts_first(self, transform_df):
        # amount is Float64 — uppercase should cast to Utf8 then uppercase
        result = apply_transforms(
            transform_df, [{"column": "amount", "type": "uppercase"}]
        )
        assert result["amount"].dtype == pl.Utf8


# ---------------------------------------------------------------------------
# cast
# ---------------------------------------------------------------------------


class TestCast:
    def test_cast_string_to_int(self):
        df = pl.DataFrame({"val": ["10", "20", "30"]})
        result = apply_transforms(
            df, [{"column": "val", "type": "cast", "target_type": "integer"}]
        )
        assert result["val"].dtype == pl.Int64
        assert result["val"].to_list() == [10, 20, 30]

    def test_cast_string_to_float(self):
        df = pl.DataFrame({"val": ["1.5", "2.5"]})
        result = apply_transforms(
            df, [{"column": "val", "type": "cast", "target_type": "float"}]
        )
        assert result["val"].dtype == pl.Float64

    def test_cast_int_to_string(self):
        df = pl.DataFrame({"val": [1, 2, 3]})
        result = apply_transforms(
            df, [{"column": "val", "type": "cast", "target_type": "string"}]
        )
        assert result["val"].dtype == pl.Utf8

    def test_cast_unknown_target_type_is_noop(self):
        df = pl.DataFrame({"val": ["hello"]})
        result = apply_transforms(
            df, [{"column": "val", "type": "cast", "target_type": "jsonb"}]
        )
        assert result["val"].dtype == pl.Utf8

    def test_cast_no_target_type_is_noop(self):
        df = pl.DataFrame({"val": [1, 2]})
        result = apply_transforms(df, [{"column": "val", "type": "cast"}])
        assert result["val"].dtype == pl.Int64


# ---------------------------------------------------------------------------
# strip_leading_zeros
# ---------------------------------------------------------------------------


class TestStripLeadingZeros:
    def test_basic_strip(self, transform_df):
        result = apply_transforms(
            transform_df, [{"column": "code", "type": "strip_leading_zeros"}]
        )
        codes = result["code"].to_list()
        assert codes[0] == "123"
        assert codes[1] == "45"

    def test_all_zeros_becomes_zero(self):
        df = pl.DataFrame({"code": ["000", "0000"]})
        result = apply_transforms(
            df, [{"column": "code", "type": "strip_leading_zeros"}]
        )
        assert result["code"].to_list() == ["0", "0"]


# ---------------------------------------------------------------------------
# pad_left
# ---------------------------------------------------------------------------


class TestPadLeft:
    def test_pad_left_with_zeros(self):
        df = pl.DataFrame({"code": ["1", "23", "456"]})
        result = apply_transforms(
            df,
            [{"column": "code", "type": "pad_left", "pad_char": "0", "pad_length": 5}],
        )
        assert result["code"].to_list() == ["00001", "00023", "00456"]

    def test_pad_left_default_char(self):
        df = pl.DataFrame({"code": ["1"]})
        result = apply_transforms(
            df,
            [{"column": "code", "type": "pad_left", "pad_length": 3}],
        )
        assert result["code"].to_list() == ["001"]


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------


class TestReplace:
    def test_replace_substring(self, transform_df):
        result = apply_transforms(
            transform_df,
            [{"column": "text", "type": "replace", "find_str": "Hello", "replace_str": "Hi"}],
        )
        assert result["text"].to_list()[0] == "Hi World"

    def test_replace_all_occurrences(self):
        df = pl.DataFrame({"val": ["aabbaa"]})
        result = apply_transforms(
            df,
            [{"column": "val", "type": "replace", "find_str": "a", "replace_str": "x"}],
        )
        assert result["val"].to_list() == ["xxbbxx"]


# ---------------------------------------------------------------------------
# substring
# ---------------------------------------------------------------------------


class TestSubstring:
    def test_substring_with_length(self, transform_df):
        result = apply_transforms(
            transform_df,
            [{"column": "text", "type": "substring", "start": 0, "length": 5}],
        )
        assert result["text"].to_list()[0] == "Hello"

    def test_substring_without_length(self, transform_df):
        result = apply_transforms(
            transform_df,
            [{"column": "text", "type": "substring", "start": 6}],
        )
        assert result["text"].to_list()[0] == "World"

    def test_substring_from_start(self):
        df = pl.DataFrame({"val": ["abcdef"]})
        result = apply_transforms(
            df,
            [{"column": "val", "type": "substring", "start": 2, "length": 3}],
        )
        assert result["val"].to_list() == ["cde"]


# ---------------------------------------------------------------------------
# Unknown type / missing column / failed transform
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_transform_type_is_skipped(self, transform_df):
        original_shape = transform_df.shape
        result = apply_transforms(
            transform_df, [{"column": "name", "type": "nonexistent_op"}]
        )
        assert result.shape == original_shape

    def test_missing_column_is_skipped(self, transform_df):
        original_shape = transform_df.shape
        result = apply_transforms(
            transform_df, [{"column": "does_not_exist", "type": "uppercase"}]
        )
        assert result.shape == original_shape

    def test_failed_transform_does_not_crash(self):
        # round on a string column — Polars will raise; transform should be skipped
        df = pl.DataFrame({"val": ["hello"]})
        result = apply_transforms(df, [{"column": "val", "type": "round", "decimals": 2}])
        # Should return df without crashing (transform skipped on failure)
        assert "val" in result.columns

    def test_multiple_transforms_applied_in_order(self, transform_df):
        result = apply_transforms(
            transform_df,
            [
                {"column": "name", "type": "trim"},
                {"column": "name", "type": "uppercase"},
            ],
        )
        for v in result["name"].to_list():
            assert v == v.upper()
            assert v == v.strip()
