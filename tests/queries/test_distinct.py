"""Comprehensive tests for easyweaver.queries.operations.distinct."""
import polars as pl
import pytest

from easyweaver.queries.operations.distinct import apply_distinct


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dup_df():
    """DataFrame with duplicate rows."""
    return pl.DataFrame(
        {
            "name": ["Alice", "Alice", "Bob", "Charlie", "Bob"],
            "dept": ["Eng", "Eng", "Sales", "Eng", "Sales"],
            "salary": [100, 100, 80, 90, 80],
        }
    )


# ---------------------------------------------------------------------------
# enabled / disabled
# ---------------------------------------------------------------------------


class TestEnabled:
    def test_enabled_true_deduplicates(self, dup_df):
        spec = {"enabled": True}
        result = apply_distinct(dup_df, spec)
        assert len(result) < len(dup_df)

    def test_enabled_false_returns_unchanged(self, dup_df):
        spec = {"enabled": False}
        result = apply_distinct(dup_df, spec)
        assert len(result) == len(dup_df)

    def test_enabled_defaults_to_true_when_missing(self, dup_df):
        # No "enabled" key → defaults to True (enabled)
        result = apply_distinct(dup_df, {})
        assert len(result) < len(dup_df)


# ---------------------------------------------------------------------------
# columns subset
# ---------------------------------------------------------------------------


class TestColumnSubset:
    def test_distinct_on_specific_columns(self, dup_df):
        spec = {"enabled": True, "columns": ["name"]}
        result = apply_distinct(dup_df, spec)
        # Deduplicated on "name" → 3 unique names
        names = result["name"].to_list()
        assert len(set(names)) == len(names)

    def test_distinct_on_multiple_columns(self, dup_df):
        spec = {"enabled": True, "columns": ["name", "dept"]}
        result = apply_distinct(dup_df, spec)
        # (Alice, Eng) x2 → deduplicated
        combos = [(r["name"], r["dept"]) for r in result.to_dicts()]
        assert len(set(combos)) == len(combos)

    def test_distinct_with_empty_columns_deduplicates_all(self, dup_df):
        spec = {"enabled": True, "columns": []}
        # Empty list is falsy → spec.get("columns") returns [] which is falsy
        # → falls into the else branch → df.unique(keep=keep) on ALL columns
        result = apply_distinct(dup_df, spec)
        # (Alice, Eng, 100) and (Bob, Sales, 80) appear twice; Charlie is unique
        # After full dedup: 3 unique rows
        assert len(result) == 3

    def test_distinct_missing_column_skipped(self, dup_df):
        spec = {"enabled": True, "columns": ["nonexistent_col"]}
        # No valid columns → return df unchanged
        result = apply_distinct(dup_df, spec)
        assert result.shape == dup_df.shape

    def test_distinct_mix_valid_invalid_columns(self, dup_df):
        spec = {"enabled": True, "columns": ["name", "nonexistent_col"]}
        # Only "name" is valid → distinct on ["name"]
        result = apply_distinct(dup_df, spec)
        names = result["name"].to_list()
        assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# keep mode
# ---------------------------------------------------------------------------


class TestKeepMode:
    def test_keep_first(self, dup_df):
        spec = {"enabled": True, "columns": ["name"], "keep": "first"}
        result = apply_distinct(dup_df, spec)
        assert len(result) == 3  # Alice, Bob, Charlie

    def test_keep_last(self, dup_df):
        spec = {"enabled": True, "columns": ["name"], "keep": "last"}
        result = apply_distinct(dup_df, spec)
        assert len(result) == 3

    def test_keep_any(self, dup_df):
        # Polars supports "any" (non-deterministic)
        spec = {"enabled": True, "columns": ["name"], "keep": "any"}
        result = apply_distinct(dup_df, spec)
        assert len(result) == 3

    def test_keep_none(self, dup_df):
        # keep="none" drops ALL duplicated records (keeps only truly unique)
        spec = {"enabled": True, "columns": ["name"], "keep": "none"}
        result = apply_distinct(dup_df, spec)
        # "Charlie" is the only name appearing once
        assert result["name"].to_list() == ["Charlie"]

    def test_default_keep_is_first(self, dup_df):
        spec = {"enabled": True, "columns": ["name"]}
        result = apply_distinct(dup_df, spec)
        # Default keep="first" → 3 rows
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Empty DataFrame
# ---------------------------------------------------------------------------


class TestEmptyDataFrame:
    def test_empty_df_stays_empty(self):
        df = pl.DataFrame({"x": pl.Series([], dtype=pl.Int64)})
        result = apply_distinct(df, {"enabled": True})
        assert result.is_empty()

    def test_empty_df_disabled(self):
        df = pl.DataFrame({"x": pl.Series([], dtype=pl.Int64)})
        result = apply_distinct(df, {"enabled": False})
        assert result.is_empty()
