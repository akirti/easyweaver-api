"""Tests for easyweaver.queries.executor."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from easyweaver.core.exceptions import QueryExecutionError
from easyweaver.queries.executor import (
    apply_sort,
    execute_join,
    execute_join_from_results,
    execute_single_source,
    execute_single_source_batched,
    paginate_dataframe,
    resolve_cross_dataset_filters,
    select_columns,
)
from easyweaver.queries.schemas import FilterCondition, JoinConfig, QuerySourceConfig
from easyweaver.sources.models import DataSource


# ── Fixtures ──────────────────────────────────────────────────────────

_SOURCE_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_SOURCE_ID_B = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _make_source(source_type: str = "postgres") -> DataSource:
    return DataSource(
        id=_SOURCE_ID,
        name="test_source",
        source_type=source_type,
        encrypted_credentials="encrypted",
    )


def _make_config(table: str = "orders", columns=None, filters=None) -> QuerySourceConfig:
    return QuerySourceConfig(
        source_id=_SOURCE_ID,
        table=table,
        columns=columns,
        filters=filters or [],
    )


def _make_connector(supports_batching: bool = False, rows=None) -> MagicMock:
    """Build a mock connector with async context manager support."""
    connector = MagicMock()
    connector.supports_batching = supports_batching
    connector.execute_query = AsyncMock(return_value=rows or [])
    connector.__aenter__ = AsyncMock(return_value=connector)
    connector.__aexit__ = AsyncMock(return_value=None)
    return connector


def _make_batched_connector(batches) -> MagicMock:
    """Build a connector where execute_query_batched returns a sequence of batches.

    Each item in *batches* is (rows, has_more, last_key).
    """
    connector = MagicMock()
    connector.supports_batching = True
    connector.execute_query_batched = AsyncMock(side_effect=batches)
    connector.__aenter__ = AsyncMock(return_value=connector)
    connector.__aexit__ = AsyncMock(return_value=None)
    return connector


# ── select_columns ────────────────────────────────────────────────────


class TestSelectColumns:
    def test_none_returns_all(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = select_columns(df, None)
        assert result.columns == ["a", "b"]

    def test_empty_list_returns_all(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = select_columns(df, [])
        assert result.columns == ["a", "b"]

    def test_valid_subset(self):
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = select_columns(df, ["a", "c"])
        assert result.columns == ["a", "c"]

    def test_invalid_columns_dropped(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        result = select_columns(df, ["a", "nonexistent"])
        assert result.columns == ["a"]

    def test_all_invalid_returns_full_df(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        result = select_columns(df, ["x", "y"])
        # No valid columns → returns original df
        assert result.columns == ["a", "b"]


# ── apply_sort ────────────────────────────────────────────────────────


class TestApplySort:
    def test_empty_sorts_returns_unchanged(self):
        df = pl.DataFrame({"a": [3, 1, 2]})
        result = apply_sort(df, [])
        assert result["a"].to_list() == [3, 1, 2]

    def test_sort_asc(self):
        df = pl.DataFrame({"a": [3, 1, 2]})
        result = apply_sort(df, [{"column": "a", "direction": "asc"}])
        assert result["a"].to_list() == [1, 2, 3]

    def test_sort_desc(self):
        df = pl.DataFrame({"a": [3, 1, 2]})
        result = apply_sort(df, [{"column": "a", "direction": "desc"}])
        assert result["a"].to_list() == [3, 2, 1]

    def test_multi_column_sort(self):
        df = pl.DataFrame({"dept": ["b", "a", "b", "a"], "sal": [200, 150, 100, 300]})
        result = apply_sort(
            df,
            [
                {"column": "dept", "direction": "asc"},
                {"column": "sal", "direction": "desc"},
            ],
        )
        assert result["dept"].to_list() == ["a", "a", "b", "b"]
        assert result["sal"].to_list() == [300, 150, 200, 100]

    def test_sort_defaults_to_asc(self):
        df = pl.DataFrame({"a": [3, 1, 2]})
        result = apply_sort(df, [{"column": "a"}])
        assert result["a"].to_list() == [1, 2, 3]


# ── paginate_dataframe ────────────────────────────────────────────────


class TestPaginateDataframe:
    def test_first_page(self):
        df = pl.DataFrame({"id": list(range(10))})
        rows, total = paginate_dataframe(df, page=1, page_size=3)
        assert total == 10
        assert len(rows) == 3
        assert rows[0]["id"] == 0

    def test_second_page(self):
        df = pl.DataFrame({"id": list(range(10))})
        rows, total = paginate_dataframe(df, page=2, page_size=3)
        assert rows[0]["id"] == 3

    def test_last_partial_page(self):
        df = pl.DataFrame({"id": list(range(10))})
        rows, total = paginate_dataframe(df, page=4, page_size=3)
        # page 4: offset 9 → only 1 row
        assert len(rows) == 1

    def test_returns_total_count(self):
        df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        _, total = paginate_dataframe(df, page=1, page_size=2)
        assert total == 5

    def test_temporal_columns_cast_to_string(self):
        from datetime import date

        df = pl.DataFrame({"d": [date(2026, 1, 1), date(2026, 6, 15)]})
        rows, _ = paginate_dataframe(df, page=1, page_size=10)
        assert isinstance(rows[0]["d"], str)

    def test_datetime_column_cast_to_string(self):
        from datetime import datetime, timezone

        df = pl.DataFrame(
            {"ts": [datetime(2026, 1, 1, tzinfo=timezone.utc)]}
        ).with_columns(pl.col("ts").cast(pl.Datetime))
        rows, _ = paginate_dataframe(df, page=1, page_size=10)
        assert isinstance(rows[0]["ts"], str)

    def test_empty_page_beyond_end(self):
        df = pl.DataFrame({"id": [1, 2]})
        rows, total = paginate_dataframe(df, page=99, page_size=10)
        assert rows == []
        assert total == 2


# ── execute_single_source ─────────────────────────────────────────────


class TestExecuteSingleSource:
    @pytest.mark.anyio
    async def test_returns_dataframe_from_rows(self):
        source = _make_source()
        config = _make_config()
        sample_rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        connector = _make_connector(rows=sample_rows)

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            df = await execute_single_source(source, config)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "id" in df.columns

    @pytest.mark.anyio
    async def test_empty_result_returns_empty_dataframe(self):
        source = _make_source()
        config = _make_config()
        connector = _make_connector(rows=[])

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            df = await execute_single_source(source, config)

        assert df.is_empty()

    @pytest.mark.anyio
    async def test_applies_row_limit(self):
        source = _make_source()
        config = _make_config()
        connector = _make_connector(rows=[{"id": i} for i in range(5)])

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            await execute_single_source(source, config, row_limit=3)

        # The connector was called with limit=3
        call_kwargs = connector.execute_query.call_args[1]
        assert call_kwargs["limit"] == 3

    @pytest.mark.anyio
    async def test_uses_settings_limit_when_none_provided(self):
        source = _make_source()
        config = _make_config()
        connector = _make_connector()

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 500
            await execute_single_source(source, config, row_limit=None)

        call_kwargs = connector.execute_query.call_args[1]
        assert call_kwargs["limit"] == 500


# ── execute_single_source_batched ─────────────────────────────────────


class TestExecuteSingleSourceBatched:
    @pytest.mark.anyio
    async def test_falls_back_when_no_batching_support(self):
        source = _make_source()
        config = _make_config()
        connector = _make_connector(supports_batching=False, rows=[{"id": 1}])

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            df = await execute_single_source_batched(source, config)

        assert len(df) == 1
        connector.execute_query.assert_called_once()

    @pytest.mark.anyio
    async def test_fallback_emits_fetch_complete(self):
        source = _make_source()
        config = _make_config()
        connector = _make_connector(supports_batching=False, rows=[{"id": 1}])
        events = []

        async def cb(event_type, **data):
            events.append(event_type)

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            await execute_single_source_batched(source, config, progress_callback=cb)

        assert "fetch_complete" in events

    @pytest.mark.anyio
    async def test_batched_single_batch_has_more_false(self):
        source = _make_source()
        config = _make_config()
        batch_rows = [{"id": i} for i in range(5)]
        connector = _make_batched_connector([(batch_rows, False, None)])

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            df = await execute_single_source_batched(source, config)

        assert len(df) == 5

    @pytest.mark.anyio
    async def test_batched_multiple_batches(self):
        source = _make_source()
        config = _make_config()
        batch1 = [{"id": i} for i in range(3)]
        batch2 = [{"id": i} for i in range(3, 6)]
        connector = _make_batched_connector(
            [(batch1, True, 3), (batch2, False, None)]
        )

        events = []

        async def cb(event_type, **data):
            events.append((event_type, data))

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
            patch("easyweaver.queries.executor.adapt_batch_size", return_value=10_000),
        ):
            mock_settings.max_result_rows = 10_000
            df = await execute_single_source_batched(source, config, progress_callback=cb)

        assert len(df) == 6
        progress_events = [e for e in events if e[0] == "fetch_progress"]
        assert len(progress_events) == 2
        complete_events = [e for e in events if e[0] == "fetch_complete"]
        assert len(complete_events) == 1

    @pytest.mark.anyio
    async def test_cancel_raises_cancelled_error(self):
        source = _make_source()
        config = _make_config()

        batch_rows = [{"id": i} for i in range(3)]
        connector = _make_batched_connector([(batch_rows, True, 3)] * 10)

        control = {"cancelled": True}

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            with pytest.raises(asyncio.CancelledError, match="cancelled by user"):
                await execute_single_source_batched(source, config, control=control)

    @pytest.mark.anyio
    async def test_pause_then_cancel_during_pause(self):
        source = _make_source()
        config = _make_config()
        batch_rows = [{"id": i} for i in range(3)]
        connector = _make_batched_connector([(batch_rows, True, 3)] * 10)

        control = {"paused": True, "cancelled": False}

        async def cancel_after_delay():
            await asyncio.sleep(0.05)
            control["cancelled"] = True

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            with pytest.raises(asyncio.CancelledError):
                await asyncio.gather(
                    execute_single_source_batched(source, config, control=control),
                    cancel_after_delay(),
                    return_exceptions=False,
                )

    @pytest.mark.anyio
    async def test_row_limit_stops_fetch(self):
        source = _make_source()
        config = _make_config()
        batch_rows = [{"id": i} for i in range(5)]
        connector = _make_batched_connector([(batch_rows, True, 5)] * 10)

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            df = await execute_single_source_batched(source, config, row_limit=5)

        # Should stop once we have 5+ rows
        assert len(df) == 5

    @pytest.mark.anyio
    async def test_batch_size_override_used(self):
        source = _make_source()
        config = _make_config()
        batch_rows = [{"id": 1}]
        connector = _make_batched_connector([(batch_rows, False, None)])
        control = {"batch_size_override": 999}

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            await execute_single_source_batched(source, config, control=control)

        call_kwargs = connector.execute_query_batched.call_args[1]
        assert call_kwargs["batch_size"] == 999

    @pytest.mark.anyio
    async def test_adaptive_batch_size_emits_batch_adjusted(self):
        source = _make_source()
        config = _make_config()

        # Two batches: first is slow (simulated by patching adapt_batch_size)
        batch1 = [{"id": i} for i in range(10)]
        batch2 = [{"id": i} for i in range(10, 20)]
        connector = _make_batched_connector([(batch1, True, 10), (batch2, False, None)])

        events = []

        async def cb(event_type, **data):
            events.append((event_type, data))

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
            patch("easyweaver.queries.executor.adapt_batch_size", return_value=20_000),
        ):
            mock_settings.max_result_rows = 10_000
            # Default batch_size is 10_000, adapted returns 20_000 → different → event emitted
            await execute_single_source_batched(source, config, progress_callback=cb)

        batch_adj_events = [e for e in events if e[0] == "batch_adjusted"]
        assert len(batch_adj_events) >= 1
        assert batch_adj_events[0][1]["reason"] == "adaptive"

    @pytest.mark.anyio
    async def test_adaptive_disabled_when_override_set(self):
        source = _make_source()
        config = _make_config()
        batch_rows = [{"id": 1}]
        connector = _make_batched_connector([(batch_rows, False, None)])
        control = {"batch_size_override": 5000, "adaptive_enabled": False}

        events = []

        async def cb(event_type, **data):
            events.append(event_type)

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.queries.executor.adapt_batch_size") as mock_adapt,
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            await execute_single_source_batched(
                source, config, control=control, progress_callback=cb
            )

        # adapt_batch_size should NOT be called when override is set
        mock_adapt.assert_not_called()

    @pytest.mark.anyio
    async def test_empty_batch_stops_loop(self):
        source = _make_source()
        config = _make_config()
        connector = _make_batched_connector([([], False, None)])

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            df = await execute_single_source_batched(source, config)

        assert df.is_empty()


# ── execute_join ──────────────────────────────────────────────────────


class TestExecuteJoin:
    @pytest.mark.anyio
    async def test_inner_join(self):
        left_source = _make_source()
        right_source = DataSource(
            id=_SOURCE_ID_B,
            name="right",
            source_type="postgres",
            encrypted_credentials="enc",
        )
        left_config = _make_config("orders")
        right_config = QuerySourceConfig(
            source_id=_SOURCE_ID_B, table="customers"
        )
        join_config = JoinConfig(
            join_type="inner", left_on="cid", right_on="id"
        )

        left_rows = [{"cid": 1, "amount": 100}, {"cid": 2, "amount": 200}]
        right_rows = [{"id": 1, "name": "Alice"}, {"id": 3, "name": "Carol"}]

        left_connector = _make_connector(rows=left_rows)
        right_connector = _make_connector(rows=right_rows)
        connectors = iter([left_connector, right_connector])

        def get_conn(source_type, creds):
            return next(connectors)

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", side_effect=get_conn),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            df = await execute_join(
                left_source, right_source, left_config, right_config, join_config
            )

        # Inner join: only cid=1 matches id=1
        assert len(df) == 1
        assert "amount" in df.columns

    @pytest.mark.anyio
    async def test_left_empty_returns_empty_df(self):
        left_source = _make_source()
        right_source = DataSource(
            id=_SOURCE_ID_B, name="r", source_type="postgres", encrypted_credentials="enc"
        )
        join_config = JoinConfig(join_type="inner", left_on="id", right_on="id")
        left_connector = _make_connector(rows=[])
        right_connector = _make_connector(rows=[{"id": 1}])
        connectors = iter([left_connector, right_connector])

        def get_conn(source_type, creds):
            return next(connectors)

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", side_effect=get_conn),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            df = await execute_join(
                left_source, right_source,
                _make_config("t1"),
                QuerySourceConfig(source_id=_SOURCE_ID_B, table="t2"),
                join_config,
            )

        assert df.is_empty()

    @pytest.mark.anyio
    async def test_right_empty_returns_empty_df(self):
        left_source = _make_source()
        right_source = DataSource(
            id=_SOURCE_ID_B, name="r", source_type="postgres", encrypted_credentials="enc"
        )
        join_config = JoinConfig(join_type="inner", left_on="id", right_on="id")
        left_connector = _make_connector(rows=[{"id": 1}])
        right_connector = _make_connector(rows=[])
        connectors = iter([left_connector, right_connector])

        def get_conn(source_type, creds):
            return next(connectors)

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", side_effect=get_conn),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            df = await execute_join(
                left_source, right_source,
                _make_config("t1"),
                QuerySourceConfig(source_id=_SOURCE_ID_B, table="t2"),
                join_config,
            )

        assert df.is_empty()

    @pytest.mark.anyio
    async def test_numeric_string_key_coercion(self):
        """Integer left key and string right key should be normalized for join."""
        left_source = _make_source()
        right_source = DataSource(
            id=_SOURCE_ID_B, name="r", source_type="postgres", encrypted_credentials="enc"
        )
        join_config = JoinConfig(join_type="inner", left_on="id", right_on="id")
        left_rows = [{"id": 1, "val": "A"}]
        right_rows = [{"id": "00001", "extra": "X"}]
        left_connector = _make_connector(rows=left_rows)
        right_connector = _make_connector(rows=right_rows)
        connectors = iter([left_connector, right_connector])

        def get_conn(source_type, creds):
            return next(connectors)

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", side_effect=get_conn),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            df = await execute_join(
                left_source, right_source,
                _make_config("t1"),
                QuerySourceConfig(source_id=_SOURCE_ID_B, table="t2"),
                join_config,
            )

        assert len(df) == 1

    @pytest.mark.anyio
    async def test_join_type_mapping(self):
        """Outer join should use 'full' in polars."""
        left_source = _make_source()
        right_source = DataSource(
            id=_SOURCE_ID_B, name="r", source_type="postgres", encrypted_credentials="enc"
        )
        join_config = JoinConfig(join_type="outer", left_on="id", right_on="id")
        left_rows = [{"id": 1, "v": "A"}, {"id": 2, "v": "B"}]
        right_rows = [{"id": 1, "w": "X"}, {"id": 3, "w": "Z"}]
        left_connector = _make_connector(rows=left_rows)
        right_connector = _make_connector(rows=right_rows)
        connectors = iter([left_connector, right_connector])

        def get_conn(source_type, creds):
            return next(connectors)

        with (
            patch("easyweaver.queries.executor.get_source_credentials", return_value={}),
            patch("easyweaver.queries.executor.get_connector", side_effect=get_conn),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10_000
            df = await execute_join(
                left_source, right_source,
                _make_config("t1"),
                QuerySourceConfig(source_id=_SOURCE_ID_B, table="t2"),
                join_config,
            )

        # Full outer join: rows for id 1, 2, 3
        assert len(df) == 3


# ── execute_join_from_results ─────────────────────────────────────────


class TestExecuteJoinFromResults:
    @pytest.mark.anyio
    async def test_joins_stored_results(self):
        left_df = pl.DataFrame({"id": [1, 2], "v": ["A", "B"]})
        right_df = pl.DataFrame({"id": [1, 3], "w": ["X", "Z"]})
        store = MagicMock()
        store.get_result = AsyncMock(side_effect=[left_df, right_df])
        join_config = JoinConfig(join_type="inner", left_on="id", right_on="id")

        df = await execute_join_from_results(store, "run-left", "run-right", join_config)

        assert len(df) == 1

    @pytest.mark.anyio
    async def test_raises_if_left_not_found(self):
        store = MagicMock()
        store.get_result = AsyncMock(return_value=None)
        join_config = JoinConfig(join_type="inner", left_on="id", right_on="id")

        with pytest.raises(QueryExecutionError, match="run-left"):
            await execute_join_from_results(store, "run-left", "run-right", join_config)

    @pytest.mark.anyio
    async def test_raises_if_right_not_found(self):
        left_df = pl.DataFrame({"id": [1]})
        store = MagicMock()
        store.get_result = AsyncMock(side_effect=[left_df, None])
        join_config = JoinConfig(join_type="inner", left_on="id", right_on="id")

        with pytest.raises(QueryExecutionError, match="run-right"):
            await execute_join_from_results(store, "run-left", "run-right", join_config)

    @pytest.mark.anyio
    async def test_left_empty_returns_empty(self):
        left_df = pl.DataFrame({"id": []}, schema={"id": pl.Int64})
        right_df = pl.DataFrame({"id": [1]})
        store = MagicMock()
        store.get_result = AsyncMock(side_effect=[left_df, right_df])
        join_config = JoinConfig(join_type="inner", left_on="id", right_on="id")

        df = await execute_join_from_results(store, "l", "r", join_config)
        assert df.is_empty()

    @pytest.mark.anyio
    async def test_list_join_keys(self):
        left_df = pl.DataFrame({"a": [1], "b": ["x"], "v": [10]})
        right_df = pl.DataFrame({"a": [1], "b": ["x"], "w": [20]})
        store = MagicMock()
        store.get_result = AsyncMock(side_effect=[left_df, right_df])
        join_config = JoinConfig(join_type="inner", left_on=["a", "b"], right_on=["a", "b"])

        df = await execute_join_from_results(store, "l", "r", join_config)
        assert len(df) == 1


# ── resolve_cross_dataset_filters ────────────────────────────────────


class TestResolveCrossDatasetFilters:
    @pytest.mark.anyio
    async def test_plain_filters_pass_through(self):
        store = MagicMock()
        filters = [{"column": "x", "operator": "eq", "value": 1}]

        result = await resolve_cross_dataset_filters(store, filters)

        assert result == [{"column": "x", "operator": "eq", "value": 1}]

    @pytest.mark.anyio
    async def test_value_from_resolved_for_in_operator(self):
        ref_df = pl.DataFrame({"status": ["active", "pending", "active"]})
        store = MagicMock()
        store.get_result = AsyncMock(return_value=ref_df)

        run_id = str(uuid.uuid4())
        filters = [
            {
                "column": "status",
                "operator": "in",
                "value_from": {"run_id": run_id, "column": "status"},
            }
        ]

        result = await resolve_cross_dataset_filters(store, filters)

        assert "value_from" not in result[0]
        assert set(result[0]["value"]) == {"active", "pending"}

    @pytest.mark.anyio
    async def test_raises_if_ref_dataset_not_found(self):
        store = MagicMock()
        store.get_result = AsyncMock(return_value=None)

        run_id = str(uuid.uuid4())
        filters = [
            {
                "column": "x",
                "operator": "in",
                "value_from": {"run_id": run_id, "column": "x"},
            }
        ]

        with pytest.raises(QueryExecutionError, match="not found or expired"):
            await resolve_cross_dataset_filters(store, filters)

    @pytest.mark.anyio
    async def test_raises_if_column_not_in_ref_dataset(self):
        ref_df = pl.DataFrame({"other": [1, 2]})
        store = MagicMock()
        store.get_result = AsyncMock(return_value=ref_df)

        run_id = str(uuid.uuid4())
        filters = [
            {
                "column": "x",
                "operator": "in",
                "value_from": {"run_id": run_id, "column": "missing_col"},
            }
        ]

        with pytest.raises(QueryExecutionError, match="not found in referenced dataset"):
            await resolve_cross_dataset_filters(store, filters)

    @pytest.mark.anyio
    async def test_raises_when_too_many_unique_values(self):
        """Cross-dataset filter should fail if unique values exceed 10_000."""
        ref_df = pl.DataFrame({"id": list(range(10_001))})
        store = MagicMock()
        store.get_result = AsyncMock(return_value=ref_df)

        run_id = str(uuid.uuid4())
        filters = [
            {
                "column": "id",
                "operator": "in",
                "value_from": {"run_id": run_id, "column": "id"},
            }
        ]

        with pytest.raises(QueryExecutionError, match="exceeds the limit"):
            await resolve_cross_dataset_filters(store, filters)

    @pytest.mark.anyio
    async def test_value_from_only_resolved_for_in_not_in(self):
        """value_from on a non-in operator should not trigger resolution."""
        store = MagicMock()
        # Operator is 'eq' so value_from should NOT be used
        filters = [
            {
                "column": "x",
                "operator": "eq",
                "value": 5,
            }
        ]

        result = await resolve_cross_dataset_filters(store, filters)
        store.get_result.assert_not_called()
        assert result[0]["value"] == 5
