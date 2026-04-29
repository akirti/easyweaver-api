"""Comprehensive tests for easyweaver.queries.operations.sort."""
import polars as pl
import pytest

from easyweaver.queries.operations.sort import apply_sort


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sort_df():
    return pl.DataFrame(
        {
            "name": ["Charlie", "Alice", "Bob", "Alice"],
            "age": [35, 30, 25, 28],
            "score": [78.3, 85.5, 92.0, 88.0],
        }
    )


# ---------------------------------------------------------------------------
# Single-column sorts
# ---------------------------------------------------------------------------


class TestSingleColumnSort:
    def test_ascending_default(self, sort_df):
        sorts = [{"column": "age", "direction": "asc"}]
        result = apply_sort(sort_df, sorts)
        ages = result["age"].to_list()
        assert ages == sorted(ages)

    def test_descending(self, sort_df):
        sorts = [{"column": "age", "direction": "desc"}]
        result = apply_sort(sort_df, sorts)
        ages = result["age"].to_list()
        assert ages == sorted(ages, reverse=True)

    def test_ascending_string_column(self, sort_df):
        sorts = [{"column": "name", "direction": "asc"}]
        result = apply_sort(sort_df, sorts)
        names = result["name"].to_list()
        assert names == sorted(names)

    def test_descending_float_column(self, sort_df):
        sorts = [{"column": "score", "direction": "desc"}]
        result = apply_sort(sort_df, sorts)
        scores = result["score"].to_list()
        assert scores == sorted(scores, reverse=True)

    def test_direction_defaults_to_asc_when_missing(self, sort_df):
        sorts = [{"column": "age"}]
        result = apply_sort(sort_df, sorts)
        ages = result["age"].to_list()
        assert ages == sorted(ages)


# ---------------------------------------------------------------------------
# Multi-column sorts
# ---------------------------------------------------------------------------


class TestMultiColumnSort:
    def test_sort_by_name_then_age(self, sort_df):
        sorts = [
            {"column": "name", "direction": "asc"},
            {"column": "age", "direction": "asc"},
        ]
        result = apply_sort(sort_df, sorts)
        rows = result.to_dicts()
        # Alice rows should come before Bob, Bob before Charlie
        alice_rows = [r for r in rows if r["name"] == "Alice"]
        bob_rows = [r for r in rows if r["name"] == "Bob"]
        charlie_rows = [r for r in rows if r["name"] == "Charlie"]
        assert rows.index(alice_rows[0]) < rows.index(bob_rows[0])
        assert rows.index(bob_rows[0]) < rows.index(charlie_rows[0])
        # Alice rows should be sorted by age ascending
        alice_ages = [r["age"] for r in alice_rows]
        assert alice_ages == sorted(alice_ages)

    def test_sort_asc_then_desc(self, sort_df):
        sorts = [
            {"column": "name", "direction": "asc"},
            {"column": "age", "direction": "desc"},
        ]
        result = apply_sort(sort_df, sorts)
        alice_rows = [r for r in result.to_dicts() if r["name"] == "Alice"]
        # Alice appears twice; with desc age the older Alice (30) should be first
        assert alice_rows[0]["age"] == 30
        assert alice_rows[1]["age"] == 28


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_sorts_returns_unchanged(self, sort_df):
        result = apply_sort(sort_df, [])
        assert result.shape == sort_df.shape

    def test_missing_column_skipped(self, sort_df):
        sorts = [{"column": "nonexistent", "direction": "asc"}]
        result = apply_sort(sort_df, sorts)
        # No valid columns → return df unchanged
        assert result.shape == sort_df.shape

    def test_mix_valid_and_invalid_columns(self, sort_df):
        sorts = [
            {"column": "age", "direction": "asc"},
            {"column": "nonexistent", "direction": "asc"},
        ]
        result = apply_sort(sort_df, sorts)
        # Only valid columns are sorted
        ages = result["age"].to_list()
        assert ages == sorted(ages)

    def test_empty_dataframe_sorts_without_error(self):
        df = pl.DataFrame({"x": pl.Series([], dtype=pl.Int64)})
        sorts = [{"column": "x", "direction": "asc"}]
        result = apply_sort(df, sorts)
        assert result.is_empty()

    def test_single_row_sort(self):
        df = pl.DataFrame({"x": [42]})
        sorts = [{"column": "x", "direction": "desc"}]
        result = apply_sort(df, sorts)
        assert result["x"].to_list() == [42]

    def test_all_missing_columns_returns_unchanged(self, sort_df):
        sorts = [
            {"column": "no_col_1", "direction": "asc"},
            {"column": "no_col_2", "direction": "desc"},
        ]
        result = apply_sort(sort_df, sorts)
        assert result.shape == sort_df.shape
