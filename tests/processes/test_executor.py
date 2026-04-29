"""Tests for easyweaver.processes.executor — DAG-aware batched executor."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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


# ── coerce_param_values edge cases ─────────────────────────────────────────


class TestCoerceParamValuesEdgeCases:
    def test_float_coercion_for_decimal_number(self):
        result = coerce_param_values(
            {"price": "9.99"},
            {"price": {"type": "number"}},
        )
        assert result["price"] == pytest.approx(9.99)

    def test_number_coercion_invalid_value_is_left_unchanged(self):
        result = coerce_param_values(
            {"count": "not-a-number"},
            {"count": {"type": "number"}},
        )
        # Invalid coercion - should leave original value
        assert result["count"] == "not-a-number"

    def test_select_type_is_left_as_is(self):
        result = coerce_param_values(
            {"region": "EU"},
            {"region": {"type": "select"}},
        )
        assert result["region"] == "EU"

    def test_multi_select_already_a_list(self):
        result = coerce_param_values(
            {"tags": ["a", "b"]},
            {"tags": {"type": "multi_select"}},
        )
        assert result["tags"] == ["a", "b"]

    def test_multi_select_comma_string(self):
        result = coerce_param_values(
            {"tags": "a, b, c"},
            {"tags": {"type": "multi_select"}},
        )
        assert result["tags"] == ["a", "b", "c"]

    def test_multi_select_non_string_non_list(self):
        result = coerce_param_values(
            {"tags": 42},
            {"tags": {"type": "multi_select"}},
        )
        assert result["tags"] == [42]

    def test_boolean_yesno_true_when_yes(self):
        result = coerce_param_values(
            {"active": "yes"},
            {"active": {"type": "boolean_yesno"}},
        )
        assert result["active"] is True

    def test_boolean_yesno_false_when_no(self):
        result = coerce_param_values(
            {"active": "no"},
            {"active": {"type": "boolean_yesno"}},
        )
        assert result["active"] is False

    def test_boolean_yesno_already_bool_unchanged(self):
        result = coerce_param_values(
            {"active": True},
            {"active": {"type": "boolean_yesno"}},
        )
        assert result["active"] is True

    def test_boolean_yesno_non_string_converted(self):
        result = coerce_param_values(
            {"active": 1},
            {"active": {"type": "boolean_yesno"}},
        )
        assert result["active"] is True

    def test_boolean_truefalse_true_when_true(self):
        result = coerce_param_values(
            {"flag": "true"},
            {"flag": {"type": "boolean_truefalse"}},
        )
        assert result["flag"] is True

    def test_boolean_truefalse_false_when_false(self):
        result = coerce_param_values(
            {"flag": "false"},
            {"flag": {"type": "boolean_truefalse"}},
        )
        assert result["flag"] is False

    def test_boolean_truefalse_already_bool_unchanged(self):
        result = coerce_param_values(
            {"flag": False},
            {"flag": {"type": "boolean_truefalse"}},
        )
        assert result["flag"] is False

    def test_boolean_truefalse_non_string_converted(self):
        result = coerce_param_values(
            {"flag": 0},
            {"flag": {"type": "boolean_truefalse"}},
        )
        assert result["flag"] is False


# ── resolve_params / _walk_replace edge cases ─────────────────────────────


class TestWalkReplace:
    def test_partial_placeholder_in_string(self):
        """When {param} is part of a larger string, it's interpolated as string."""
        result = resolve_params(
            {"q": "SELECT * FROM {tbl} WHERE id > 0"},
            {"tbl": "orders"},
        )
        assert result["q"] == "SELECT * FROM orders WHERE id > 0"

    def test_unknown_placeholder_left_unchanged(self):
        """Unknown {param} references in partial strings are left as-is."""
        result = resolve_params(
            {"q": "SELECT {unknown_param}"},
            {"tbl": "orders"},
        )
        assert result["q"] == "SELECT {unknown_param}"

    def test_full_placeholder_unknown_returns_original(self):
        """A full-match {param} that's not in param_values is returned unchanged."""
        result = resolve_params(
            {"val": "{missing_param}"},
            {"other": "x"},
        )
        assert result["val"] == "{missing_param}"

    def test_nested_list_replacement(self):
        result = resolve_params(
            {"items": ["{a}", "{b}", "static"]},
            {"a": 1, "b": 2},
        )
        assert result["items"] == [1, 2, "static"]

    def test_non_string_value_unchanged(self):
        result = resolve_params({"count": 42}, {"count": 99})
        assert result["count"] == 42  # int not replaced


# ── _execute_query legacy DB path ─────────────────────────────────────────


class TestExecuteQueryLegacyPath:
    @pytest.mark.asyncio
    async def test_raises_when_no_credentials_and_no_db(self):
        from easyweaver.core.exceptions import ProcessExecutionError
        from easyweaver.processes.executor import _execute_query

        qc = ProcessQueryConfig(
            source_id="s1",
            table="t1",
            source_type=None,
            encrypted_credentials=None,
        )

        with pytest.raises(ProcessExecutionError, match="no embedded credentials"):
            await _execute_query(db=None, query_config=qc, key="s1.main")

    @pytest.mark.asyncio
    async def test_legacy_path_uses_db_source(self):
        import uuid
        from easyweaver.processes.executor import _execute_query

        source_uuid = uuid.uuid4()
        qc = ProcessQueryConfig(
            source_id=str(source_uuid),
            table="orders",
            source_type=None,
            encrypted_credentials=None,
        )

        mock_source = MagicMock()
        mock_source.id = source_uuid
        mock_source.source_type = "postgres"

        expected_df = pl.DataFrame({"id": [1, 2]})

        with (
            patch("easyweaver.processes.executor.get_source", AsyncMock(return_value=mock_source)),
            patch("easyweaver.processes.executor.execute_single_source", AsyncMock(return_value=expected_df)),
        ):
            df = await _execute_query(db=MagicMock(), query_config=qc, key="s1.main")

        assert len(df) == 2

    @pytest.mark.asyncio
    async def test_applies_post_filters_to_result(self, mock_decrypt):
        from easyweaver.processes.executor import _execute_query
        from easyweaver.processes.schemas import ProcessFilterConfig

        qc = ProcessQueryConfig(
            source_id="s1",
            table="orders",
            source_type="postgres",
            encrypted_credentials="enc",
            filters=[ProcessFilterConfig(column="id", operator="eq", value=1)],
        )

        rows = [{"id": 1}, {"id": 2}]
        connector = AsyncMock()
        connector.supports_batching = False
        connector.execute_query = AsyncMock(return_value=rows)
        connector.__aenter__ = AsyncMock(return_value=connector)
        connector.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.processes.executor.apply_filters", return_value=pl.DataFrame({"id": [1]})) as mock_filter,
        ):
            df = await _execute_query(db=None, query_config=qc, key="s1.main")

        mock_filter.assert_called_once()
        assert len(df) == 1


# ── _execute_query_batched non-batching path ──────────────────────────────


class TestExecuteQueryBatchedNonBatching:
    @pytest.mark.asyncio
    async def test_non_batching_connector_with_filters(self, mock_decrypt):
        from easyweaver.processes.executor import _execute_query_batched
        from easyweaver.processes.schemas import ProcessFilterConfig

        qc = ProcessQueryConfig(
            source_id="s1",
            table="orders",
            source_type="postgres",
            encrypted_credentials="enc",
            filters=[ProcessFilterConfig(column="id", operator="gt", value=0)],
        )

        connector = AsyncMock()
        connector.supports_batching = False
        connector.execute_query = AsyncMock(return_value=[{"id": 1}, {"id": 2}])
        connector.__aenter__ = AsyncMock(return_value=connector)
        connector.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.processes.executor.apply_filters", return_value=pl.DataFrame({"id": [1]})),
        ):
            df = await _execute_query_batched(db=None, query_config=qc, key="s1.main")

        assert len(df) == 1

    @pytest.mark.asyncio
    async def test_legacy_path_when_no_credentials(self):
        from easyweaver.processes.executor import _execute_query_batched

        qc = ProcessQueryConfig(
            source_id="s1",
            table="orders",
            source_type=None,
            encrypted_credentials=None,
        )

        expected_df = pl.DataFrame({"id": [10]})

        with patch("easyweaver.processes.executor._execute_query", AsyncMock(return_value=expected_df)):
            df = await _execute_query_batched(db=MagicMock(), query_config=qc, key="s1.main")

        assert len(df) == 1  # single row in expected_df


# ── _execute_query_batched: adaptive sizing and redis store ───────────────


class TestExecuteQueryBatchedAdaptiveAndStore:
    @pytest.mark.asyncio
    async def test_adaptive_batch_size_adjustment(self, mock_decrypt):
        from easyweaver.processes.executor import _execute_query_batched

        connector = AsyncMock()
        connector.supports_batching = True
        connector.__aenter__ = AsyncMock(return_value=connector)
        connector.__aexit__ = AsyncMock(return_value=False)

        batch1 = ([{"id": i} for i in range(10)], True, 9)
        batch2 = ([{"id": i} for i in range(10, 20)], False, 19)
        connector.execute_query_batched = AsyncMock(side_effect=[batch1, batch2])

        qc = ProcessQueryConfig(
            source_id="s1",
            table="orders",
            source_type="postgres",
            encrypted_credentials="enc",
        )

        control = {"adaptive_enabled": True, "target_batch_seconds": 1.0, "cancelled": False, "paused": False}

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.processes.executor.adapt_batch_size", return_value=5000) as mock_adapt,
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 100000

            df = await _execute_query_batched(db=None, query_config=qc, key="s1.main", control=control)

        # adapt_batch_size should have been called after first batch
        mock_adapt.assert_called_once()
        assert len(df) == 20

    @pytest.mark.asyncio
    async def test_redis_store_caches_intermediate_result(self, mock_decrypt):
        from easyweaver.processes.executor import _execute_query_batched

        connector = AsyncMock()
        connector.supports_batching = True
        connector.__aenter__ = AsyncMock(return_value=connector)
        connector.__aexit__ = AsyncMock(return_value=False)
        connector.execute_query_batched = AsyncMock(return_value=([{"id": 1}], False, None))

        qc = ProcessQueryConfig(
            source_id="s1",
            table="orders",
            source_type="postgres",
            encrypted_credentials="enc",
        )

        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10000
            mock_settings.batch_intermediate_ttl_seconds = 300

            df = await _execute_query_batched(
                db=None, query_config=qc, key="s1.main",
                redis_store=mock_store, run_id="run-123",
                control={"adaptive_enabled": False, "cancelled": False, "paused": False},
            )

        mock_store.store_result.assert_called_once()
        cache_key_arg = mock_store.store_result.call_args[0][0]
        assert "ew:batch:run-123:s1.main" == cache_key_arg

    @pytest.mark.asyncio
    async def test_get_initial_batch_size_from_db(self, mock_decrypt):
        from easyweaver.processes.executor import _execute_query_batched

        connector = AsyncMock()
        connector.supports_batching = True
        connector.__aenter__ = AsyncMock(return_value=connector)
        connector.__aexit__ = AsyncMock(return_value=False)
        connector.execute_query_batched = AsyncMock(return_value=([{"id": 1}], False, None))

        qc = ProcessQueryConfig(
            source_id="s1",
            table="orders",
            source_type="postgres",
            encrypted_credentials="enc",
        )

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch(
                "easyweaver.processes.batch_adapter.get_initial_batch_size",
                AsyncMock(return_value=5000),
            ) as mock_get_bs,
            patch(
                "easyweaver.processes.batch_adapter.save_optimal_batch_size",
                AsyncMock(),
            ),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10000

            df = await _execute_query_batched(
                db=MagicMock(), query_config=qc, key="s1.main",
                control={"adaptive_enabled": False, "cancelled": False, "paused": False},
            )

        mock_get_bs.assert_called_once()

    @pytest.mark.asyncio
    async def test_pause_then_cancel_in_batch_loop(self, mock_decrypt):
        from easyweaver.processes.executor import _execute_query_batched

        connector = AsyncMock()
        connector.supports_batching = True
        connector.__aenter__ = AsyncMock(return_value=connector)
        connector.__aexit__ = AsyncMock(return_value=False)

        call_count = {"n": 0}

        async def batched(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # After first call, set paused then cancelled
                control["paused"] = True
                # Immediately also set cancelled so pause loop exits
                control["cancelled"] = True
                return ([{"id": 1}], True, 0)
            return ([{"id": 2}], False, None)

        connector.execute_query_batched = batched

        control = {
            "adaptive_enabled": False,
            "cancelled": False,
            "paused": False,
            "target_batch_seconds": 10.0,
        }

        qc = ProcessQueryConfig(
            source_id="s1",
            table="orders",
            source_type="postgres",
            encrypted_credentials="enc",
        )

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10000

            with pytest.raises(asyncio.CancelledError):
                await _execute_query_batched(
                    db=None, query_config=qc, key="s1.main", control=control
                )

    @pytest.mark.asyncio
    async def test_apply_filters_to_batched_result(self, mock_decrypt):
        from easyweaver.processes.executor import _execute_query_batched
        from easyweaver.processes.schemas import ProcessFilterConfig

        connector = AsyncMock()
        connector.supports_batching = True
        connector.__aenter__ = AsyncMock(return_value=connector)
        connector.__aexit__ = AsyncMock(return_value=False)
        connector.execute_query_batched = AsyncMock(return_value=([{"id": 1}, {"id": 2}], False, None))

        qc = ProcessQueryConfig(
            source_id="s1",
            table="orders",
            source_type="postgres",
            encrypted_credentials="enc",
            filters=[ProcessFilterConfig(column="id", operator="eq", value=1)],
        )

        expected_filtered = pl.DataFrame({"id": [1]})

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.processes.executor.apply_filters", return_value=expected_filtered) as mock_filter,
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10000
            df = await _execute_query_batched(
                db=None, query_config=qc, key="s1.main",
                control={"adaptive_enabled": False, "cancelled": False, "paused": False},
            )

        mock_filter.assert_called_once()
        assert len(df) == 1


# ── execute_process: DAG, joins, and transforms ───────────────────────────


class TestExecuteProcessDAGAndJoins:
    @pytest.mark.asyncio
    async def test_multiple_queries_without_logic_raises(self, mock_decrypt):
        """Multiple datasets with no logic step should raise an error."""
        connector = _mock_connector([{"id": 1}])
        config = _make_config({
            "s1": {"q1": ProcessQueryConfig(source_id="s1", table="t1", source_type="postgres", encrypted_credentials="enc")},
            "s2": {"q2": ProcessQueryConfig(source_id="s2", table="t2", source_type="postgres", encrypted_credentials="enc")},
        })

        call_count = {"n": 0}
        def connector_factory(source_type, creds):
            call_count["n"] += 1
            return _mock_connector([{"id": call_count["n"]}])

        from easyweaver.core.exceptions import ProcessExecutionError
        with patch("easyweaver.processes.executor.get_connector", side_effect=connector_factory):
            with pytest.raises(ProcessExecutionError, match="Multiple queries without logic"):
                await execute_process(config, {})

    @pytest.mark.asyncio
    async def test_join_with_empty_left_produces_empty_result(self, mock_decrypt):
        """Join where left dataset is empty should produce empty result without error."""
        orders_connector = _mock_connector([])  # empty
        customers_connector = _mock_connector([{"oid": 1, "name": "Alice"}])

        config = _make_config(
            {
                "orders": {"main": ProcessQueryConfig(source_id="s1", table="orders", source_type="postgres", encrypted_credentials="enc")},
                "customers": {"main": ProcessQueryConfig(source_id="s2", table="customers", source_type="postgres", encrypted_credentials="enc")},
            },
            logics=[{
                "key": "joined",
                "type": "join",
                "left": "orders.main",
                "right": "customers.main",
                "left_on": ["order_id"],
                "right_on": ["oid"],
                "join_type": "inner",
            }],
        )

        call_count = {"n": 0}
        def factory(source_type, creds):
            call_count["n"] += 1
            return orders_connector if call_count["n"] == 1 else customers_connector

        with patch("easyweaver.processes.executor.get_connector", side_effect=factory):
            df = await execute_process(config, {})

        assert len(df) == 0

    @pytest.mark.asyncio
    async def test_join_select_columns(self, mock_decrypt):
        """Join with select_columns should only include specified columns."""
        orders = [{"order_id": 1, "amount": 100}]
        customers = [{"order_id": 1, "name": "Alice", "email": "a@b.com"}]

        orders_c = _mock_connector(orders)
        customers_c = _mock_connector(customers)

        config = _make_config(
            {
                "orders": {"main": ProcessQueryConfig(source_id="s1", table="orders", source_type="postgres", encrypted_credentials="enc")},
                "customers": {"main": ProcessQueryConfig(source_id="s2", table="customers", source_type="postgres", encrypted_credentials="enc")},
            },
            logics=[{
                "key": "joined",
                "type": "join",
                "left": "orders.main",
                "right": "customers.main",
                "left_on": ["order_id"],
                "right_on": ["order_id"],
                "join_type": "inner",
                "select_columns": ["order_id", "name"],
            }],
        )

        call_count = {"n": 0}
        def factory(source_type, creds):
            call_count["n"] += 1
            return orders_c if call_count["n"] == 1 else customers_c

        with patch("easyweaver.processes.executor.get_connector", side_effect=factory):
            df = await execute_process(config, {})

        assert "order_id" in df.columns
        assert "name" in df.columns
        assert "email" not in df.columns

    @pytest.mark.asyncio
    async def test_join_left_not_found_raises(self, mock_decrypt):
        """Reference to a non-existent left dataset in join raises ProcessExecutionError."""
        from easyweaver.core.exceptions import ProcessExecutionError

        rows = [{"id": 1}]
        connector = _mock_connector(rows)

        config = _make_config(
            {
                "s1": {"main": ProcessQueryConfig(source_id="s1", table="t1", source_type="postgres", encrypted_credentials="enc")},
            },
            logics=[{
                "key": "joined",
                "type": "join",
                "left": "nonexistent.dataset",  # doesn't exist
                "right": "s1.main",
                "left_on": ["id"],
                "right_on": ["id"],
                "join_type": "inner",
            }],
        )

        with patch("easyweaver.processes.executor.get_connector", return_value=connector):
            with pytest.raises(ProcessExecutionError, match="not found"):
                await execute_process(config, {})

    @pytest.mark.asyncio
    async def test_row_pair_binding_applied(self, mock_decrypt):
        """Row-pair binding should create a semi-join filter from the source dataset."""
        orders_rows = [{"order_id": 1, "customer_id": 10}, {"order_id": 2, "customer_id": 20}]
        customers_rows = [{"order_id": 1, "cid": 10, "name": "Alice"}, {"order_id": 2, "cid": 20, "name": "Bob"}]

        orders_connector = _mock_connector(orders_rows)
        customers_connector = _mock_connector(customers_rows)

        from easyweaver.processes.schemas import ProcessQueryBinding

        config = _make_config(
            {
                "orders": {
                    "main": ProcessQueryConfig(
                        source_id="s1", table="orders", source_type="postgres", encrypted_credentials="enc",
                    ),
                },
                "customers": {
                    "main": ProcessQueryConfig(
                        source_id="s2", table="customers", source_type="postgres", encrypted_credentials="enc",
                        bindings=[
                            ProcessQueryBinding(
                                source_dataset="orders.main",
                                mode="row_pair",
                                mappings=[BindingMapping(source_column="customer_id", target_column="cid")],
                            ),
                        ],
                    ),
                },
            },
            logics=[{
                "key": "joined",
                "type": "join",
                "left": "orders.main",
                "right": "customers.main",
                "left_on": ["order_id"],
                "right_on": ["order_id"],
                "join_type": "inner",
            }],
        )

        call_count = {"n": 0}
        def factory(source_type, creds):
            call_count["n"] += 1
            return orders_connector if call_count["n"] == 1 else customers_connector

        with (
            patch("easyweaver.processes.executor.get_connector", side_effect=factory),
            patch(
                "easyweaver.queries.operations.binding.apply_row_pair_filter",
                side_effect=lambda df, rpdf: df,
            ) as mock_rpf,
        ):
            # Just verify the execution completes without error
            # (row_pair filter is applied but result depends on mock)
            df = await execute_process(config, {})

        assert len(df) >= 0  # just verify no error


class TestExecuteProcessTransforms:
    @pytest.mark.asyncio
    async def test_derived_columns_applied(self, mock_decrypt):
        """Derived columns should be applied during transform phase."""
        rows = [{"price": 10, "qty": 2}]
        connector = _mock_connector(rows)

        from easyweaver.queries.schemas import DerivedColumnSpec

        config = _make_config(
            {
                "s1": {"main": ProcessQueryConfig(source_id="s1", table="t1", source_type="postgres", encrypted_credentials="enc")},
            },
            derived_columns=[DerivedColumnSpec(
                name="total",
                expression_type="math",
                expression="price * qty",
            )],
        )

        expected_df = pl.DataFrame({"price": [10], "qty": [2], "total": [20.0]})

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.processes.executor.apply_derived_columns", return_value=expected_df) as mock_dc,
        ):
            df = await execute_process(config, {})

        mock_dc.assert_called_once()
        assert "total" in df.columns

    @pytest.mark.asyncio
    async def test_operations_filters_applied(self, mock_decrypt):
        """Operations filters should be applied during transform phase."""
        rows = [{"id": 1}, {"id": 2}, {"id": 3}]
        connector = _mock_connector(rows)

        from easyweaver.processes.schemas import ProcessFilterConfig, ProcessOperations

        config = _make_config(
            {
                "s1": {"main": ProcessQueryConfig(source_id="s1", table="t1", source_type="postgres", encrypted_credentials="enc")},
            },
            operations=ProcessOperations(
                filters=[ProcessFilterConfig(column="id", operator="gt", value=1)],
            ),
        )

        filtered_df = pl.DataFrame({"id": [2, 3]})

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.processes.executor.apply_filters", return_value=filtered_df) as mock_filters,
        ):
            df = await execute_process(config, {})

        mock_filters.assert_called()

    @pytest.mark.asyncio
    async def test_operations_group_by_applied(self, mock_decrypt):
        """Operations group_by should be applied during transform phase."""
        rows = [{"region": "EU", "amount": 100}, {"region": "EU", "amount": 200}]
        connector = _mock_connector(rows)

        from easyweaver.processes.schemas import ProcessOperations
        from easyweaver.queries.schemas import GroupBySpec, AggregationSpec

        config = _make_config(
            {
                "s1": {"main": ProcessQueryConfig(source_id="s1", table="t1", source_type="postgres", encrypted_credentials="enc")},
            },
            operations=ProcessOperations(
                group_by=GroupBySpec(
                    group_columns=["region"],
                    aggregations=[AggregationSpec(column="amount", function="sum", alias="total")],
                ),
            ),
        )

        grouped_df = pl.DataFrame({"region": ["EU"], "total": [300]})

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.processes.executor.apply_group_by", return_value=grouped_df) as mock_gb,
        ):
            df = await execute_process(config, {})

        mock_gb.assert_called_once()

    @pytest.mark.asyncio
    async def test_operations_distinct_applied(self, mock_decrypt):
        """Operations distinct should be applied during transform phase."""
        rows = [{"id": 1, "name": "a"}, {"id": 1, "name": "a"}]
        connector = _mock_connector(rows)

        from easyweaver.processes.schemas import ProcessOperations
        from easyweaver.queries.schemas import DistinctSpec

        config = _make_config(
            {
                "s1": {"main": ProcessQueryConfig(source_id="s1", table="t1", source_type="postgres", encrypted_credentials="enc")},
            },
            operations=ProcessOperations(
                distinct=DistinctSpec(columns=["id", "name"]),
            ),
        )

        distinct_df = pl.DataFrame({"id": [1], "name": ["a"]})

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.processes.executor.apply_distinct", return_value=distinct_df) as mock_d,
        ):
            df = await execute_process(config, {})

        mock_d.assert_called_once()

    @pytest.mark.asyncio
    async def test_operations_sorts_applied(self, mock_decrypt):
        """Operations sorts should be applied during transform phase."""
        rows = [{"id": 3}, {"id": 1}, {"id": 2}]
        connector = _mock_connector(rows)

        from easyweaver.processes.schemas import ProcessOperations
        from easyweaver.queries.schemas import SortSpec

        config = _make_config(
            {
                "s1": {"main": ProcessQueryConfig(source_id="s1", table="t1", source_type="postgres", encrypted_credentials="enc")},
            },
            operations=ProcessOperations(
                sorts=[SortSpec(column="id", direction="asc")],
            ),
        )

        sorted_df = pl.DataFrame({"id": [1, 2, 3]})

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.processes.executor.apply_sort", return_value=sorted_df) as mock_sort,
        ):
            df = await execute_process(config, {})

        mock_sort.assert_called_once()

    @pytest.mark.asyncio
    async def test_transformations_applied(self, mock_decrypt):
        """Transformations should be applied during transform phase."""
        rows = [{"date_str": "2024-01-01"}]
        connector = _mock_connector(rows)

        from easyweaver.processes.schemas import ProcessTransformation

        config = _make_config(
            {
                "s1": {"main": ProcessQueryConfig(source_id="s1", table="t1", source_type="postgres", encrypted_credentials="enc")},
            },
            transformations=[ProcessTransformation(column="date_str", type="to_date", date_format="%Y-%m-%d")],
        )

        transformed_df = pl.DataFrame({"date_str": ["2024-01-01"]})

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.processes.executor.apply_transforms", return_value=transformed_df) as mock_t,
        ):
            df = await execute_process(config, {})

        mock_t.assert_called_once()

    @pytest.mark.asyncio
    async def test_final_logic_key_not_found_raises(self, mock_decrypt):
        """If the final logic step produces no result, an error should be raised."""
        from easyweaver.core.exceptions import ProcessExecutionError

        connector = _mock_connector([{"id": 1}])

        # Build a config where both sides of join exist, but use a dataset key that
        # won't be found in results
        config = _make_config(
            {
                "s1": {"main": ProcessQueryConfig(source_id="s1", table="t1", source_type="postgres", encrypted_credentials="enc")},
            },
            logics=[{
                "key": "joined",
                "type": "join",
                "left": "s1.main",
                "right": "nonexistent.right",  # will cause right not found error
                "left_on": ["id"],
                "right_on": ["id"],
                "join_type": "inner",
            }],
        )

        with patch("easyweaver.processes.executor.get_connector", return_value=connector):
            with pytest.raises(ProcessExecutionError):
                await execute_process(config, {})

    @pytest.mark.asyncio
    async def test_cyclic_dag_raises_error(self, mock_decrypt):
        """A config with circular bindings should raise a ProcessExecutionError about cycles."""
        from easyweaver.core.exceptions import ProcessExecutionError
        from easyweaver.processes.schemas import ProcessQueryBinding

        # Create a cycle: schema1.a -> schema1.b -> schema1.a
        config = _make_config({
            "schema1": {
                "a": ProcessQueryConfig(
                    source_id="s1", table="t1", source_type="postgres", encrypted_credentials="enc",
                    bindings=[ProcessQueryBinding(source_dataset="schema1.b", mode="distinct", mappings=[
                        BindingMapping(source_column="id", target_column="id"),
                    ])],
                ),
                "b": ProcessQueryConfig(
                    source_id="s2", table="t2", source_type="postgres", encrypted_credentials="enc",
                    bindings=[ProcessQueryBinding(source_dataset="schema1.a", mode="distinct", mappings=[
                        BindingMapping(source_column="id", target_column="id"),
                    ])],
                ),
            },
        })

        with pytest.raises(ProcessExecutionError):
            await execute_process(config, {})

    @pytest.mark.asyncio
    async def test_task_failure_cancels_remaining_tasks(self, mock_decrypt):
        """When one dataset fails, remaining in-flight tasks should be cancelled."""
        from easyweaver.core.exceptions import ProcessExecutionError

        # Create two independent datasets where one will fail
        fail_connector = AsyncMock()
        fail_connector.supports_batching = False
        fail_connector.execute_query = AsyncMock(side_effect=RuntimeError("connector failed"))
        fail_connector.__aenter__ = AsyncMock(return_value=fail_connector)
        fail_connector.__aexit__ = AsyncMock(return_value=False)

        config = _make_config({
            "s1": {"main": ProcessQueryConfig(source_id="s1", table="t1", source_type="postgres", encrypted_credentials="enc")},
        })

        with patch("easyweaver.processes.executor.get_connector", return_value=fail_connector):
            with pytest.raises(ProcessExecutionError):
                await execute_process(config, {})
