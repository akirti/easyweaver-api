"""Comprehensive tests for easyweaver.queries.operations.binding."""
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

from easyweaver.core.exceptions import QueryExecutionError
from easyweaver.queries.operations.binding import (
    _extract_distinct_values,
    _get_source_df,
    apply_row_pair_filter,
    resolve_distinct_bindings,
    resolve_row_pair_bindings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_store(result: pl.DataFrame | None):
    """Create a mock store whose get_result always returns `result`."""
    store = MagicMock()
    store.get_result = AsyncMock(return_value=result)
    return store


def make_binding(mode: str, source_run_id: str, mappings: list[dict]) -> dict:
    return {"mode": mode, "source_run_id": source_run_id, "mappings": mappings}


# ---------------------------------------------------------------------------
# _get_source_df
# ---------------------------------------------------------------------------


class TestGetSourceDf:
    @pytest.mark.asyncio
    async def test_returns_df_when_found(self):
        df = pl.DataFrame({"a": [1, 2]})
        store = make_store(df)
        result = await _get_source_df(store, "run_1")
        assert result.shape == df.shape

    @pytest.mark.asyncio
    async def test_raises_when_not_found(self):
        store = make_store(None)
        with pytest.raises(QueryExecutionError, match="not found or expired"):
            await _get_source_df(store, "missing_run")


# ---------------------------------------------------------------------------
# _extract_distinct_values
# ---------------------------------------------------------------------------


class TestExtractDistinctValues:
    def test_basic_distinct(self):
        df = pl.DataFrame({"col": [1, 2, 2, 3, None]})
        values = _extract_distinct_values(df, "col")
        assert set(values) == {1, 2, 3}
        assert None not in values

    def test_empty_after_nulls_dropped(self):
        df = pl.DataFrame({"col": pl.Series([None, None], dtype=pl.Int64)})
        values = _extract_distinct_values(df, "col")
        assert values == []

    def test_exceeds_limit_raises(self):
        # Patch the limit for this test by building a large series
        from easyweaver.queries.operations import binding as binding_mod

        original = binding_mod._MAX_DISTINCT_VALUES
        binding_mod._MAX_DISTINCT_VALUES = 3
        try:
            df = pl.DataFrame({"col": [1, 2, 3, 4, 5]})
            with pytest.raises(QueryExecutionError, match="exceeding limit"):
                _extract_distinct_values(df, "col")
        finally:
            binding_mod._MAX_DISTINCT_VALUES = original


# ---------------------------------------------------------------------------
# resolve_distinct_bindings
# ---------------------------------------------------------------------------


class TestResolveDistinctBindings:
    @pytest.mark.asyncio
    async def test_basic_distinct_binding(self):
        source_df = pl.DataFrame({"dept": ["Engineering", "Sales", "Sales", "Marketing"]})
        store = make_store(source_df)
        binding = make_binding(
            mode="distinct",
            source_run_id="run_1",
            mappings=[{"source_column": "dept", "target_column": "department"}],
        )
        filters = await resolve_distinct_bindings(store, [binding])
        assert len(filters) == 1
        assert filters[0]["column"] == "department"
        assert filters[0]["operator"] == "in"
        assert set(filters[0]["value"]) == {"Engineering", "Sales", "Marketing"}

    @pytest.mark.asyncio
    async def test_skips_non_distinct_mode(self):
        store = make_store(pl.DataFrame({"x": [1]}))
        binding = make_binding(
            mode="row_pair",
            source_run_id="run_1",
            mappings=[{"source_column": "x", "target_column": "y"}],
        )
        filters = await resolve_distinct_bindings(store, [binding])
        assert filters == []

    @pytest.mark.asyncio
    async def test_empty_source_df_is_skipped(self):
        store = make_store(pl.DataFrame({"dept": pl.Series([], dtype=pl.Utf8)}))
        binding = make_binding(
            mode="distinct",
            source_run_id="run_1",
            mappings=[{"source_column": "dept", "target_column": "department"}],
        )
        filters = await resolve_distinct_bindings(store, [binding])
        assert filters == []

    @pytest.mark.asyncio
    async def test_missing_source_column_raises(self):
        source_df = pl.DataFrame({"dept": ["Engineering"]})
        store = make_store(source_df)
        binding = make_binding(
            mode="distinct",
            source_run_id="run_1",
            mappings=[{"source_column": "missing_col", "target_column": "x"}],
        )
        with pytest.raises(QueryExecutionError, match="not found in source dataset"):
            await resolve_distinct_bindings(store, [binding])

    @pytest.mark.asyncio
    async def test_multiple_mappings(self):
        source_df = pl.DataFrame({"dept": ["Eng", "Sales"], "region": ["US", "EU"]})
        store = make_store(source_df)
        binding = make_binding(
            mode="distinct",
            source_run_id="run_1",
            mappings=[
                {"source_column": "dept", "target_column": "department"},
                {"source_column": "region", "target_column": "region_code"},
            ],
        )
        filters = await resolve_distinct_bindings(store, [binding])
        assert len(filters) == 2
        cols = {f["column"] for f in filters}
        assert cols == {"department", "region_code"}


# ---------------------------------------------------------------------------
# resolve_row_pair_bindings
# ---------------------------------------------------------------------------


class TestResolveRowPairBindings:
    @pytest.mark.asyncio
    async def test_basic_row_pair(self):
        source_df = pl.DataFrame({"emp_id": [1, 2, 2], "dept_id": [10, 20, 20]})
        store = make_store(source_df)
        binding = make_binding(
            mode="row_pair",
            source_run_id="run_1",
            mappings=[
                {"source_column": "emp_id", "target_column": "employee_id"},
                {"source_column": "dept_id", "target_column": "department_id"},
            ],
        )
        result = await resolve_row_pair_bindings(store, [binding])
        assert result is not None
        assert "employee_id" in result.columns
        assert "department_id" in result.columns
        # unique() applied so duplicates are removed
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_too_many_rows_raises(self):
        from easyweaver.queries.operations import binding as binding_mod

        original = binding_mod._MAX_ROW_PAIR_ROWS
        binding_mod._MAX_ROW_PAIR_ROWS = 2
        try:
            source_df = pl.DataFrame({"x": [1, 2, 3]})
            store = make_store(source_df)
            binding = make_binding(
                mode="row_pair",
                source_run_id="run_1",
                mappings=[{"source_column": "x", "target_column": "x"}],
            )
            with pytest.raises(QueryExecutionError, match="exceeding limit"):
                await resolve_row_pair_bindings(store, [binding])
        finally:
            binding_mod._MAX_ROW_PAIR_ROWS = original

    @pytest.mark.asyncio
    async def test_empty_source_returns_empty_df(self):
        source_df = pl.DataFrame({"x": pl.Series([], dtype=pl.Int64)})
        store = make_store(source_df)
        binding = make_binding(
            mode="row_pair",
            source_run_id="run_1",
            mappings=[{"source_column": "x", "target_column": "x"}],
        )
        result = await resolve_row_pair_bindings(store, [binding])
        assert result is not None
        assert result.is_empty()

    @pytest.mark.asyncio
    async def test_no_row_pair_bindings_returns_none(self):
        store = make_store(pl.DataFrame({"a": [1]}))
        binding = make_binding(
            mode="distinct",
            source_run_id="run_1",
            mappings=[{"source_column": "a", "target_column": "b"}],
        )
        result = await resolve_row_pair_bindings(store, [binding])
        assert result is None

    @pytest.mark.asyncio
    async def test_high_row_count_logs_warning_but_succeeds(self, caplog):
        from easyweaver.queries.operations import binding as binding_mod

        original_warn = binding_mod._ROW_PAIR_WARN_THRESHOLD
        binding_mod._ROW_PAIR_WARN_THRESHOLD = 1
        try:
            source_df = pl.DataFrame({"x": [1, 2, 3]})
            store = make_store(source_df)
            binding = make_binding(
                mode="row_pair",
                source_run_id="run_1",
                mappings=[{"source_column": "x", "target_column": "x"}],
            )
            # Should not raise
            result = await resolve_row_pair_bindings(store, [binding])
            assert result is not None
        finally:
            binding_mod._ROW_PAIR_WARN_THRESHOLD = original_warn


# ---------------------------------------------------------------------------
# apply_row_pair_filter
# ---------------------------------------------------------------------------


class TestApplyRowPairFilter:
    def test_semi_join_filters_correctly(self):
        df = pl.DataFrame({"emp_id": [1, 2, 3, 4], "name": ["A", "B", "C", "D"]})
        pair_df = pl.DataFrame({"emp_id": [1, 3]})
        result = apply_row_pair_filter(df, pair_df)
        assert set(result["emp_id"].to_list()) == {1, 3}

    def test_empty_pair_df_returns_empty_schema(self):
        df = pl.DataFrame({"emp_id": [1, 2], "name": ["A", "B"]})
        pair_df = pl.DataFrame({"emp_id": pl.Series([], dtype=pl.Int64)})
        result = apply_row_pair_filter(df, pair_df)
        assert result.is_empty()
        assert result.schema == df.schema

    def test_type_mismatch_coerced(self):
        # df has Int64 emp_id, pair_df has Utf8 — coercion should make join work
        df = pl.DataFrame({"emp_id": [1, 2, 3]})
        pair_df = pl.DataFrame({"emp_id": ["1", "3"]})
        result = apply_row_pair_filter(df, pair_df)
        assert len(result) == 2

    def test_all_match(self):
        df = pl.DataFrame({"x": [10, 20, 30]})
        pair_df = pl.DataFrame({"x": [10, 20, 30]})
        result = apply_row_pair_filter(df, pair_df)
        assert len(result) == 3

    def test_no_match(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        pair_df = pl.DataFrame({"x": [99, 100]})
        result = apply_row_pair_filter(df, pair_df)
        assert result.is_empty()
