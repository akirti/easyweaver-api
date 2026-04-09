"""Tests for easyweaver.processes.executor — DAG-aware batched executor."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import polars as pl
import pytest

from easyweaver.core.exceptions import ProcessExecutionError
from easyweaver.processes.executor import (
    _resolve_distinct_bindings_from_df,
    _resolve_row_pair_filter_from_df,
    coerce_param_values,
    execute_process,
    resolve_params,
)
from easyweaver.processes.schemas import (
    BindingMapping,
    ProcessConfig,
    ProcessQueryBinding,
    ProcessQueryConfig,
)


# ── Parameter helpers (preserved from original) ──────────────────────

class TestCoerceParamValues:
    def test_fills_defaults(self):
        result = coerce_param_values(
            {},
            {"name": {"type": "string", "default": "world"}},
        )
        assert result == {"name": "world"}

    def test_coerces_number(self):
        result = coerce_param_values(
            {"limit": "42"},
            {"limit": {"type": "number"}},
        )
        assert result["limit"] == 42

    def test_coerces_boolean(self):
        result = coerce_param_values(
            {"active": "true"},
            {"active": {"type": "boolean"}},
        )
        assert result["active"] is True


class TestResolveParams:
    def test_simple_replacement(self):
        cfg = {"table": "{tbl}", "filters": [{"value": "{val}"}]}
        result = resolve_params(cfg, {"tbl": "orders", "val": 42})
        assert result == {"table": "orders", "filters": [{"value": 42}]}

    def test_partial_replacement(self):
        result = resolve_params({"query": "SELECT * FROM {tbl}"}, {"tbl": "users"})
        assert result == {"query": "SELECT * FROM users"}


# ── Binding resolution helpers ────────────────────────────────────────

class TestResolveDistinctBindingsFromDf:
    def test_basic(self):
        source_df = pl.DataFrame({"order_id": [1, 2, 2, 3]})
        mappings = [BindingMapping(source_column="order_id", target_column="oid")]
        filters = _resolve_distinct_bindings_from_df(source_df, mappings)
        assert len(filters) == 1
        assert filters[0]["column"] == "oid"
        assert filters[0]["operator"] == "in"
        assert sorted(filters[0]["value"]) == [1, 2, 3]

    def test_empty_source(self):
        source_df = pl.DataFrame({"order_id": pl.Series([], dtype=pl.Int64)})
        mappings = [BindingMapping(source_column="order_id", target_column="oid")]
        filters = _resolve_distinct_bindings_from_df(source_df, mappings)
        assert filters == []


class TestResolveRowPairFilterFromDf:
    def test_basic(self):
        source_df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        mappings = [BindingMapping(source_column="id", target_column="src_id")]
        result = _resolve_row_pair_filter_from_df(source_df, mappings)
        assert result is not None
        assert "src_id" in result.columns
        assert len(result) == 3

    def test_no_mappings(self):
        source_df = pl.DataFrame({"id": [1]})
        result = _resolve_row_pair_filter_from_df(source_df, [])
        assert result is None


# ── execute_process integration ───────────────────────────────────────

def _make_config(queries_dict, logics=None, derived_columns=None, operations=None, transformations=None):
    """Build a ProcessConfig from a simplified dict."""
    return ProcessConfig(
        queries=queries_dict,
        logics=logics or [],
        derived_columns=derived_columns or [],
        operations=operations,
        transformations=transformations or [],
    )


def _mock_connector(rows_data, supports_batching=False):
    """Create a mock connector that returns given rows."""
    connector = AsyncMock()
    connector.supports_batching = supports_batching
    connector.execute_query = AsyncMock(return_value=rows_data)
    connector.execute_query_batched = AsyncMock(return_value=(rows_data, False, None))
    connector.__aenter__ = AsyncMock(return_value=connector)
    connector.__aexit__ = AsyncMock(return_value=False)
    return connector


@pytest.fixture
def mock_decrypt():
    """Mock credential decryption."""
    with patch(
        "easyweaver.core.security.decrypt_credentials",
        return_value='{"host": "localhost"}',
    ):
        yield


class TestExecuteProcessSingleDataset:
    """Test single-dataset (no joins) execution path."""

    @pytest.mark.asyncio
    async def test_single_query_no_bindings(self, mock_decrypt):
        rows = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        connector = _mock_connector(rows)

        config = _make_config({
            "schema1": {
                "main": ProcessQueryConfig(
                    source_id="s1",
                    table="orders",
                    source_type="postgres",
                    encrypted_credentials="enc",
                ),
            },
        })

        with patch(
            "easyweaver.processes.executor.get_connector",
            return_value=connector,
        ):
            df = await execute_process(config, {})

        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]

    @pytest.mark.asyncio
    async def test_empty_queries_raises(self):
        config = _make_config({})
        with pytest.raises(ProcessExecutionError, match="no queries"):
            await execute_process(config, {})


class TestExecuteProcessWithBindings:
    """Test DAG-ordered execution with bindings between datasets."""

    @pytest.mark.asyncio
    async def test_linear_dependency(self, mock_decrypt):
        """Dataset B depends on dataset A via distinct binding."""
        orders_rows = [{"order_id": 1}, {"order_id": 2}]
        customers_rows = [{"oid": 1, "name": "Alice"}, {"oid": 2, "name": "Bob"}]

        orders_connector = _mock_connector(orders_rows)
        customers_connector = _mock_connector(customers_rows)

        config = _make_config({
            "orders": {
                "main": ProcessQueryConfig(
                    source_id="s1",
                    table="orders",
                    source_type="postgres",
                    encrypted_credentials="enc",
                ),
            },
            "customers": {
                "main": ProcessQueryConfig(
                    source_id="s2",
                    table="customers",
                    source_type="postgres",
                    encrypted_credentials="enc",
                    bindings=[
                        ProcessQueryBinding(
                            source_dataset="orders.main",
                            mode="distinct",
                            mappings=[BindingMapping(source_column="order_id", target_column="oid")],
                        ),
                    ],
                ),
            },
        }, logics=[{
            "key": "joined",
            "type": "join",
            "left": "orders.main",
            "right": "customers.main",
            "left_on": ["order_id"],
            "right_on": ["oid"],
            "join_type": "inner",
        }])

        call_count = {"n": 0}

        def connector_factory(source_type, creds):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return orders_connector
            return customers_connector

        with patch(
            "easyweaver.processes.executor.get_connector",
            side_effect=connector_factory,
        ):
            df = await execute_process(config, {})

        # Both connectors should have been called
        assert orders_connector.execute_query.called
        assert customers_connector.execute_query.called

    @pytest.mark.asyncio
    async def test_progress_callback_receives_phases(self, mock_decrypt):
        """Progress callback should receive phase events."""
        rows = [{"id": 1}]
        connector = _mock_connector(rows)

        config = _make_config({
            "schema1": {
                "main": ProcessQueryConfig(
                    source_id="s1",
                    table="t1",
                    source_type="postgres",
                    encrypted_credentials="enc",
                ),
            },
        })

        events = []

        async def callback(event_type, **data):
            events.append((event_type, data))

        with patch(
            "easyweaver.processes.executor.get_connector",
            return_value=connector,
        ):
            await execute_process(config, {}, progress_callback=callback)

        event_types = [e[0] for e in events]
        assert "phase" in event_types
        assert "fetch_started" in event_types
        assert "fetch_complete" in event_types


class TestExecuteProcessControl:
    """Test control dict for cancel/pause."""

    @pytest.mark.asyncio
    async def test_cancel_raises(self, mock_decrypt):
        """Setting cancelled=True should cause CancelledError."""
        connector = AsyncMock()
        connector.supports_batching = True
        connector.__aenter__ = AsyncMock(return_value=connector)
        connector.__aexit__ = AsyncMock(return_value=False)

        call_count = {"n": 0}

        async def batched_fetch(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ([{"id": i} for i in range(100)], True, 99)
            return ([], False, None)

        original_side_effect = batched_fetch

        async def cancel_after_first(*args, **kwargs):
            result = await original_side_effect(*args, **kwargs)
            control["cancelled"] = True
            return result

        connector.execute_query_batched = AsyncMock(side_effect=cancel_after_first)

        config = _make_config({
            "schema1": {
                "main": ProcessQueryConfig(
                    source_id="s1",
                    table="t1",
                    source_type="postgres",
                    encrypted_credentials="enc",
                ),
            },
        })

        control = {"cancelled": False, "paused": False, "adaptive_enabled": False}

        with patch(
            "easyweaver.processes.executor.get_connector",
            return_value=connector,
        ):
            with pytest.raises((ProcessExecutionError, asyncio.CancelledError)):
                await execute_process(config, {}, control=control)


class TestExecuteProcessBatched:
    """Test batched fetching path."""

    @pytest.mark.asyncio
    async def test_batched_multi_batch(self, mock_decrypt):
        """Connector with supports_batching=True should use batched path."""
        connector = AsyncMock()
        connector.supports_batching = True
        connector.__aenter__ = AsyncMock(return_value=connector)
        connector.__aexit__ = AsyncMock(return_value=False)

        batch1 = ([{"id": i} for i in range(10)], True, 9)
        batch2 = ([{"id": i} for i in range(10, 15)], False, 14)
        connector.execute_query_batched = AsyncMock(side_effect=[batch1, batch2])

        config = _make_config({
            "schema1": {
                "main": ProcessQueryConfig(
                    source_id="s1",
                    table="t1",
                    source_type="postgres",
                    encrypted_credentials="enc",
                ),
            },
        })

        with patch(
            "easyweaver.processes.executor.get_connector",
            return_value=connector,
        ):
            df = await execute_process(config, {}, control={"adaptive_enabled": False})

        assert len(df) == 15
        assert connector.execute_query_batched.call_count == 2
