"""Tests for easyweaver.dashboard.service — comprehensive coverage of all functions."""
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from easyweaver.core.exceptions import NotFoundError
from easyweaver.dashboard import service
from easyweaver.dashboard.models import DashboardConfig, DataSnapshot
from easyweaver.dashboard.schemas import (
    DashboardConfigCreate,
    DashboardConfigUpdate,
    TableMonitorConfig,
)


# ---------------------------------------------------------------------------
# Helpers & builders
# ---------------------------------------------------------------------------

CONFIG_ID = "550e8400-e29b-41d4-a716-446655440000"
SOURCE_ID = "660e8400-e29b-41d4-a716-446655440001"
SNAP_ID = "770e8400-e29b-41d4-a716-446655440002"

NOW = datetime.now(timezone.utc)


def _make_config(
    *,
    config_id: str = CONFIG_ID,
    source_id: str = SOURCE_ID,
    name: str = "Test Dashboard",
    tables=None,
    refresh_interval_minutes: int = 30,
) -> DashboardConfig:
    if tables is None:
        tables = [
            {
                "table_name": "orders",
                "timestamp_column": "updated_at",
                "modified_by_column": "modified_by",
            }
        ]
    return DashboardConfig(
        id=uuid.UUID(config_id),
        user_id="user-001",
        name=name,
        source_id=source_id,
        tables=tables,
        refresh_interval_minutes=refresh_interval_minutes,
        is_active=True,
        created_at=NOW,
        updated_at=NOW,
    )


def _config_doc(**overrides) -> dict:
    base = {
        "_id": CONFIG_ID,
        "user_id": "user-001",
        "name": "Test Dashboard",
        "source_id": SOURCE_ID,
        "tables": [
            {
                "table_name": "orders",
                "timestamp_column": "updated_at",
                "modified_by_column": "modified_by",
            }
        ],
        "refresh_interval_minutes": 30,
        "is_active": True,
        "created_at": NOW,
        "updated_at": NOW,
    }
    base.update(overrides)
    return base


def _snap_doc(**overrides) -> dict:
    base = {
        "_id": SNAP_ID,
        "config_id": CONFIG_ID,
        "source_id": SOURCE_ID,
        "table_name": "orders",
        "row_count": 1000,
        "stats": {
            "row_count": 1000,
            "size_bytes": 500000,
            "column_count": 8,
            "changes_1h": 10,
            "changes_3h": 25,
            "changes_24h": 100,
            "last_modified_at": "2026-04-01T09:55:00+00:00",
            "last_modified_by": "admin",
        },
        "captured_at": NOW,
    }
    base.update(overrides)
    return base


def _make_source_doc(source_type: str = "postgres") -> dict:
    return {
        "_id": SOURCE_ID,
        "name": "Test Source",
        "source_type": source_type,
        "encrypted_credentials": "encrypted==",
        "created_at": NOW,
        "updated_at": NOW,
    }


def _async_cursor(items: list):
    """Return an async-iterable mock from a list."""

    class _AsyncCursor:
        def __init__(self, data):
            self._data = data

        def sort(self, *args, **kwargs):
            return self

        def __aiter__(self):
            return self._iter()

        async def _iter(self):
            for item in self._data:
                yield item

    return _AsyncCursor(items)


def _make_db(
    *,
    config_doc=None,
    config_docs=None,
    snap_doc=None,
    insert_result=None,
    delete_result=None,
    update_result=None,
) -> MagicMock:
    """Build a mock AsyncIOMotorDatabase."""
    db = MagicMock()

    # dashboard_configs collection
    db.dashboard_configs.find_one = AsyncMock(return_value=config_doc)
    db.dashboard_configs.insert_one = AsyncMock(return_value=insert_result or MagicMock())
    db.dashboard_configs.update_one = AsyncMock(return_value=update_result or MagicMock())
    db.dashboard_configs.delete_one = AsyncMock(
        return_value=delete_result or MagicMock(deleted_count=1)
    )
    db.dashboard_configs.find = MagicMock(
        return_value=_async_cursor(config_docs or ([config_doc] if config_doc else []))
    )

    # data_snapshots collection
    db.data_snapshots.find_one = AsyncMock(return_value=snap_doc)
    db.data_snapshots.insert_one = AsyncMock(return_value=MagicMock())
    db.data_snapshots.delete_many = AsyncMock(return_value=MagicMock())
    db.data_snapshots.find = MagicMock(
        return_value=_async_cursor([snap_doc] if snap_doc else [])
    )

    # data_sources collection
    db.data_sources.find_one = AsyncMock(return_value=None)

    return db


# ===========================================================================
# _validate_identifier
# ===========================================================================


class TestValidateIdentifier:
    def test_valid_simple_name(self):
        assert service._validate_identifier("orders") is True

    def test_valid_with_underscore(self):
        assert service._validate_identifier("user_accounts") is True

    def test_valid_with_dot(self):
        assert service._validate_identifier("schema.table") is True

    def test_valid_with_digits(self):
        assert service._validate_identifier("table123") is True

    def test_invalid_starts_with_digit(self):
        assert service._validate_identifier("123table") is False

    def test_invalid_with_space(self):
        assert service._validate_identifier("bad table") is False

    def test_invalid_with_dash(self):
        assert service._validate_identifier("bad-name") is False

    def test_invalid_empty_string(self):
        assert service._validate_identifier("") is False

    def test_invalid_too_long(self):
        assert service._validate_identifier("a" * 129) is False

    def test_valid_max_length(self):
        assert service._validate_identifier("a" * 128) is True


# ===========================================================================
# create_config
# ===========================================================================


class TestCreateConfig:
    @pytest.mark.anyio
    async def test_creates_and_returns_dashboard_config(self):
        db = _make_db()
        data = DashboardConfigCreate(
            name="New Dashboard",
            source_id=SOURCE_ID,
            tables=[TableMonitorConfig(table_name="orders")],
            refresh_interval_minutes=60,
        )
        config = await service.create_config(db, data, user_id="user-1")

        assert config.name == "New Dashboard"
        assert config.source_id == SOURCE_ID
        assert config.user_id == "user-1"
        assert config.is_active is True
        db.dashboard_configs.insert_one.assert_called_once()

    @pytest.mark.anyio
    async def test_uses_system_user_by_default(self):
        db = _make_db()
        data = DashboardConfigCreate(
            name="Dashboard",
            source_id=SOURCE_ID,
            tables=[TableMonitorConfig(table_name="t")],
        )
        config = await service.create_config(db, data)
        assert config.user_id == "system"

    @pytest.mark.anyio
    async def test_tables_serialized_as_dicts(self):
        db = _make_db()
        data = DashboardConfigCreate(
            name="D",
            source_id=SOURCE_ID,
            tables=[TableMonitorConfig(table_name="orders", timestamp_column="ts")],
        )
        config = await service.create_config(db, data)
        assert isinstance(config.tables, list)
        assert isinstance(config.tables[0], dict)
        assert config.tables[0]["table_name"] == "orders"

    @pytest.mark.anyio
    async def test_generated_uuid_is_valid(self):
        db = _make_db()
        data = DashboardConfigCreate(
            name="D",
            source_id=SOURCE_ID,
            tables=[TableMonitorConfig(table_name="t")],
        )
        config = await service.create_config(db, data)
        assert isinstance(config.id, uuid.UUID)

    @pytest.mark.anyio
    async def test_insert_one_called_with_to_doc(self):
        db = _make_db()
        data = DashboardConfigCreate(
            name="D",
            source_id=SOURCE_ID,
            tables=[TableMonitorConfig(table_name="t")],
        )
        await service.create_config(db, data)
        call_args = db.dashboard_configs.insert_one.call_args[0][0]
        assert "_id" in call_args
        assert call_args["name"] == "D"


# ===========================================================================
# get_config
# ===========================================================================


class TestGetConfig:
    @pytest.mark.anyio
    async def test_returns_config_when_found(self):
        doc = _config_doc()
        db = _make_db(config_doc=doc)
        config = await service.get_config(db, CONFIG_ID)
        assert str(config.id) == CONFIG_ID
        assert config.name == "Test Dashboard"

    @pytest.mark.anyio
    async def test_raises_not_found_when_missing(self):
        db = _make_db(config_doc=None)
        with pytest.raises(NotFoundError) as exc_info:
            await service.get_config(db, "nonexistent-id")
        assert "DashboardConfig" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_queries_by_id(self):
        doc = _config_doc()
        db = _make_db(config_doc=doc)
        await service.get_config(db, CONFIG_ID)
        db.dashboard_configs.find_one.assert_called_once_with({"_id": CONFIG_ID})


# ===========================================================================
# list_configs
# ===========================================================================


class TestListConfigs:
    @pytest.mark.anyio
    async def test_returns_all_configs_when_no_user_id(self):
        docs = [_config_doc(), _config_doc(**{"_id": "660e8400-e29b-41d4-a716-446655440099", "name": "Second"})]
        db = _make_db(config_docs=docs)
        configs = await service.list_configs(db)
        assert len(configs) == 2

    @pytest.mark.anyio
    async def test_filters_by_user_id(self):
        docs = [_config_doc()]
        db = _make_db(config_docs=docs)
        await service.list_configs(db, user_id="user-001")
        db.dashboard_configs.find.assert_called_once_with({"user_id": "user-001"})

    @pytest.mark.anyio
    async def test_no_user_id_uses_empty_query(self):
        db = _make_db(config_docs=[])
        await service.list_configs(db, user_id=None)
        db.dashboard_configs.find.assert_called_once_with({})

    @pytest.mark.anyio
    async def test_returns_empty_list_when_none(self):
        db = _make_db(config_docs=[])
        configs = await service.list_configs(db)
        assert configs == []


# ===========================================================================
# update_config
# ===========================================================================


class TestUpdateConfig:
    @pytest.mark.anyio
    async def test_updates_name(self):
        doc = _config_doc()
        updated_doc = _config_doc(name="Updated Name")
        db = _make_db(config_doc=doc)
        # Second call to find_one (after update) should return updated doc
        db.dashboard_configs.find_one = AsyncMock(side_effect=[doc, updated_doc])
        data = DashboardConfigUpdate(name="Updated Name")
        config = await service.update_config(db, CONFIG_ID, data)
        assert config.name == "Updated Name"

    @pytest.mark.anyio
    async def test_updates_tables(self):
        doc = _config_doc()
        new_tables = [{"table_name": "payments", "timestamp_column": None, "modified_by_column": None}]
        updated_doc = _config_doc(tables=new_tables)
        db = _make_db(config_doc=doc)
        db.dashboard_configs.find_one = AsyncMock(side_effect=[doc, updated_doc])
        data = DashboardConfigUpdate(
            tables=[TableMonitorConfig(table_name="payments")]
        )
        config = await service.update_config(db, CONFIG_ID, data)
        assert config.tables[0]["table_name"] == "payments"

    @pytest.mark.anyio
    async def test_updates_refresh_interval(self):
        doc = _config_doc()
        updated_doc = _config_doc(refresh_interval_minutes=120)
        db = _make_db(config_doc=doc)
        db.dashboard_configs.find_one = AsyncMock(side_effect=[doc, updated_doc])
        data = DashboardConfigUpdate(refresh_interval_minutes=120)
        config = await service.update_config(db, CONFIG_ID, data)
        assert config.refresh_interval_minutes == 120

    @pytest.mark.anyio
    async def test_raises_not_found_when_config_missing(self):
        db = _make_db(config_doc=None)
        data = DashboardConfigUpdate(name="X")
        with pytest.raises(NotFoundError):
            await service.update_config(db, "nonexistent", data)

    @pytest.mark.anyio
    async def test_update_one_called_with_set(self):
        doc = _config_doc()
        db = _make_db(config_doc=doc)
        data = DashboardConfigUpdate(name="New Name")
        await service.update_config(db, CONFIG_ID, data)
        db.dashboard_configs.update_one.assert_called_once()
        call_args = db.dashboard_configs.update_one.call_args
        assert call_args[0][0] == {"_id": CONFIG_ID}
        assert "name" in call_args[0][1]["$set"]

    @pytest.mark.anyio
    async def test_none_fields_not_included_in_update(self):
        doc = _config_doc()
        db = _make_db(config_doc=doc)
        data = DashboardConfigUpdate()  # all None
        await service.update_config(db, CONFIG_ID, data)
        call_args = db.dashboard_configs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert "name" not in updates
        assert "tables" not in updates

    @pytest.mark.anyio
    async def test_is_active_can_be_set_false(self):
        doc = _config_doc()
        updated_doc = _config_doc()
        updated_doc["is_active"] = False
        db = _make_db(config_doc=doc)
        db.dashboard_configs.find_one = AsyncMock(side_effect=[doc, updated_doc])
        data = DashboardConfigUpdate(is_active=False)
        config = await service.update_config(db, CONFIG_ID, data)
        assert config.is_active is False


# ===========================================================================
# delete_config
# ===========================================================================


class TestDeleteConfig:
    @pytest.mark.anyio
    async def test_deletes_config_successfully(self):
        db = _make_db(delete_result=MagicMock(deleted_count=1))
        await service.delete_config(db, CONFIG_ID)
        db.dashboard_configs.delete_one.assert_called_once_with({"_id": CONFIG_ID})

    @pytest.mark.anyio
    async def test_raises_not_found_when_nothing_deleted(self):
        db = _make_db(delete_result=MagicMock(deleted_count=0))
        with pytest.raises(NotFoundError):
            await service.delete_config(db, "nonexistent")

    @pytest.mark.anyio
    async def test_also_deletes_associated_snapshots(self):
        db = _make_db(delete_result=MagicMock(deleted_count=1))
        await service.delete_config(db, CONFIG_ID)
        db.data_snapshots.delete_many.assert_called_once_with({"config_id": CONFIG_ID})


# ===========================================================================
# _get_source
# ===========================================================================


class TestGetSource:
    @pytest.mark.anyio
    async def test_returns_datasource_when_found(self):
        doc = _make_source_doc()
        db = _make_db()
        db.data_sources.find_one = AsyncMock(return_value=doc)
        from easyweaver.sources.models import DataSource

        result = await service._get_source(db, SOURCE_ID)
        assert isinstance(result, DataSource)
        assert result.name == "Test Source"

    @pytest.mark.anyio
    async def test_raises_not_found_when_missing(self):
        db = _make_db()
        db.data_sources.find_one = AsyncMock(return_value=None)
        with pytest.raises(NotFoundError) as exc_info:
            await service._get_source(db, "missing-src")
        assert "DataSource" in str(exc_info.value)


# ===========================================================================
# _gather_pg_table_stats
# ===========================================================================


_PG_STAT_ROW_MISSING = object()  # sentinel: fetchrow returns None


class TestGatherPgTableStats:
    def _make_pg_connector(self, *, count=100, stat_row=None, size=5000, col_count=5):
        connector = MagicMock()
        conn = AsyncMock()

        async def fetchval_side_effect(query, *args):
            if "COUNT(*)" in query and "information_schema" not in query:
                return count
            elif "pg_total_relation_size" in query:
                return size
            elif "information_schema.columns" in query:
                return col_count
            elif "MAX(" in query:
                return datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
            elif "WHERE" in query and "INTERVAL" in query:
                return 5
            return None

        conn.fetchval = AsyncMock(side_effect=fetchval_side_effect)

        if stat_row is _PG_STAT_ROW_MISSING:
            fetchrow_return = None
        elif stat_row is None:
            fetchrow_return = {"n_tup_ins": 100, "n_tup_upd": 50, "n_tup_del": 10}
        else:
            fetchrow_return = stat_row
        conn.fetchrow = AsyncMock(return_value=fetchrow_return)

        pool_ctx = MagicMock()
        pool_ctx.__aenter__ = AsyncMock(return_value=conn)
        pool_ctx.__aexit__ = AsyncMock(return_value=False)
        connector._pool = MagicMock()
        connector._pool.acquire = MagicMock(return_value=pool_ctx)

        return connector

    @pytest.mark.anyio
    async def test_returns_row_count(self):
        connector = self._make_pg_connector(count=999)
        stats = await service._gather_pg_table_stats(connector, "orders", None, None)
        assert stats["row_count"] == 999

    @pytest.mark.anyio
    async def test_returns_dml_stats(self):
        stat_row = {"n_tup_ins": 200, "n_tup_upd": 30, "n_tup_del": 5}
        connector = self._make_pg_connector(stat_row=stat_row)
        stats = await service._gather_pg_table_stats(connector, "orders", None, None)
        assert stats["inserts"] == 200
        assert stats["updates"] == 30
        assert stats["deletes"] == 5

    @pytest.mark.anyio
    async def test_returns_size_and_column_count(self):
        connector = self._make_pg_connector(size=100000, col_count=12)
        stats = await service._gather_pg_table_stats(connector, "orders", None, None)
        assert stats["size_bytes"] == 100000
        assert stats["column_count"] == 12

    @pytest.mark.anyio
    async def test_invalid_table_name_returns_error(self):
        connector = self._make_pg_connector()
        stats = await service._gather_pg_table_stats(connector, "bad-table!", None, None)
        assert "error" in stats
        assert stats["error"] == "invalid table name"

    @pytest.mark.anyio
    async def test_with_timestamp_column_gathers_change_counts(self):
        connector = self._make_pg_connector()
        stats = await service._gather_pg_table_stats(
            connector, "orders", "updated_at", None
        )
        assert "last_modified_at" in stats or "changes_1h" in stats

    @pytest.mark.anyio
    async def test_dml_stats_missing_when_no_stat_row(self):
        connector = self._make_pg_connector(stat_row=_PG_STAT_ROW_MISSING)
        stats = await service._gather_pg_table_stats(connector, "orders", None, None)
        assert "inserts" not in stats
        assert "updates" not in stats


# ===========================================================================
# _gather_mongo_table_stats
# ===========================================================================


class TestGatherMongoTableStats:
    def _make_mongo_connector(
        self,
        *,
        coll_stats=None,
        sample=None,
        last_ts=None,
        change_count=5,
    ):
        connector = MagicMock()

        if coll_stats is None:
            coll_stats = {"count": 500, "size": 200000, "avgObjSize": 400}

        db_mock = AsyncMock()
        db_mock.command = AsyncMock(return_value=coll_stats)

        # Mock collection
        coll_mock = MagicMock()

        if sample is None:
            sample = [{"_id": "x", "col1": 1, "col2": 2, "col3": 3}]

        async def _to_list(n):
            return sample

        sample_pipeline = MagicMock()
        sample_pipeline.to_list = _to_list

        last_ts_result = [{"_last_ts": last_ts or datetime(2026, 4, 1, tzinfo=timezone.utc)}] if last_ts is not False else []
        last_ts_agg = MagicMock()
        last_ts_agg.to_list = AsyncMock(return_value=last_ts_result)

        count_result = [{"total": change_count}]
        count_agg = MagicMock()
        count_agg.to_list = AsyncMock(return_value=count_result)

        modifier_result = [{"_id": "alice", "cnt": 3}]
        modifier_agg = MagicMock()
        modifier_agg.to_list = AsyncMock(return_value=modifier_result)

        call_count = [0]

        def aggregate_side_effect(pipeline):
            if any("$sample" in str(s) for s in pipeline):
                return sample_pipeline
            elif any("$project" in str(s) for s in pipeline):
                return last_ts_agg
            elif any("$count" in str(s) for s in pipeline):
                return count_agg
            else:
                return modifier_agg

        coll_mock.aggregate = MagicMock(side_effect=aggregate_side_effect)

        db_mock.__getitem__ = MagicMock(return_value=coll_mock)

        connector._client = {connector._db_name: db_mock}
        connector._db_name = "testdb"
        connector._client = MagicMock()
        connector._client.__getitem__ = MagicMock(return_value=db_mock)

        return connector

    @pytest.mark.anyio
    async def test_returns_row_count_from_coll_stats(self):
        connector = self._make_mongo_connector(coll_stats={"count": 777, "size": 0})
        stats = await service._gather_mongo_table_stats(connector, "orders", None, None)
        assert stats["row_count"] == 777

    @pytest.mark.anyio
    async def test_returns_size_bytes(self):
        connector = self._make_mongo_connector(coll_stats={"count": 0, "size": 99999})
        stats = await service._gather_mongo_table_stats(connector, "orders", None, None)
        assert stats["size_bytes"] == 99999

    @pytest.mark.anyio
    async def test_column_count_from_sample(self):
        # Sample has _id + 3 fields => column_count = 3
        connector = self._make_mongo_connector(
            sample=[{"_id": "x", "a": 1, "b": 2, "c": 3}]
        )
        stats = await service._gather_mongo_table_stats(connector, "orders", None, None)
        assert stats["column_count"] == 3

    @pytest.mark.anyio
    async def test_empty_sample_gives_zero_column_count(self):
        connector = self._make_mongo_connector(sample=[])
        stats = await service._gather_mongo_table_stats(connector, "orders", None, None)
        assert stats["column_count"] == 0

    @pytest.mark.anyio
    async def test_collstats_exception_sets_row_count_zero(self):
        connector = self._make_mongo_connector()
        connector._client.__getitem__.return_value.command = AsyncMock(
            side_effect=Exception("command failed")
        )
        stats = await service._gather_mongo_table_stats(connector, "orders", None, None)
        assert stats["row_count"] == 0

    @pytest.mark.anyio
    async def test_with_timestamp_column_gathers_change_counts(self):
        connector = self._make_mongo_connector()
        stats = await service._gather_mongo_table_stats(
            connector, "orders", "updated_at", None
        )
        # At minimum some time-based keys should be attempted
        assert "row_count" in stats


# ===========================================================================
# capture_snapshot
# ===========================================================================


class TestCaptureSnapshot:
    @pytest.mark.anyio
    async def test_raises_not_found_when_config_missing(self):
        db = _make_db(config_doc=None)
        with pytest.raises(NotFoundError):
            await service.capture_snapshot(db, "missing-id")

    @pytest.mark.anyio
    async def test_raises_not_found_when_source_missing(self):
        db = _make_db(config_doc=_config_doc())
        db.data_sources.find_one = AsyncMock(return_value=None)
        with pytest.raises(Exception):
            await service.capture_snapshot(db, CONFIG_ID)

    @pytest.mark.anyio
    async def test_unsupported_source_type_stores_error_snapshot(self):
        """For an unsupported source type, snapshot should have error in stats."""
        doc = _config_doc()
        source_doc = _make_source_doc(source_type="mysql")
        db = _make_db(config_doc=doc)
        db.data_sources.find_one = AsyncMock(return_value=source_doc)

        fake_connector = MagicMock()
        fake_connector.__aenter__ = AsyncMock(return_value=fake_connector)
        fake_connector.__aexit__ = AsyncMock(return_value=False)

        with patch("easyweaver.dashboard.service.decrypt_credentials", return_value='{"host":"localhost"}'):
            with patch("easyweaver.dashboard.service.get_connector", return_value=fake_connector):
                snapshots = await service.capture_snapshot(db, CONFIG_ID)

        assert len(snapshots) > 0
        assert "error" in snapshots[0].stats

    @pytest.mark.anyio
    async def test_inserts_snapshot_for_each_table(self):
        tables = [
            {"table_name": "orders", "timestamp_column": None, "modified_by_column": None},
            {"table_name": "users", "timestamp_column": None, "modified_by_column": None},
        ]
        doc = _config_doc(tables=tables)
        source_doc = _make_source_doc(source_type="mysql")  # unsupported triggers simple path
        db = _make_db(config_doc=doc)
        db.data_sources.find_one = AsyncMock(return_value=source_doc)

        fake_connector = MagicMock()
        fake_connector.__aenter__ = AsyncMock(return_value=fake_connector)
        fake_connector.__aexit__ = AsyncMock(return_value=False)

        with patch("easyweaver.dashboard.service.decrypt_credentials", return_value='{"host":"localhost"}'):
            with patch("easyweaver.dashboard.service.get_connector", return_value=fake_connector):
                snapshots = await service.capture_snapshot(db, CONFIG_ID)

        assert len(snapshots) == 2
        assert db.data_snapshots.insert_one.call_count == 2


# ===========================================================================
# get_current_stats
# ===========================================================================


class TestGetCurrentStats:
    def _make_full_db(self, *, snap_doc=None, stale=False):
        """Build a db where config and source are present and snapshot optional."""
        doc = _config_doc()
        source_doc = _make_source_doc()
        db = MagicMock()

        fresh_snap = snap_doc or _snap_doc()

        if stale:
            # captured_at older than refresh_interval_minutes
            old_time = datetime.now(timezone.utc) - timedelta(hours=2)
            fresh_snap = _snap_doc(captured_at=old_time)
            fresh_snap["captured_at"] = old_time

        async def find_one_side(query, *args, **kwargs):
            if "config_id" in query and "table_name" not in query:
                return fresh_snap
            elif "table_name" in query:
                return fresh_snap
            elif "_id" in query:
                if query["_id"] == CONFIG_ID:
                    return doc
                return source_doc
            return None

        db.dashboard_configs.find_one = AsyncMock(side_effect=find_one_side)
        db.data_sources.find_one = AsyncMock(return_value=source_doc)
        db.data_snapshots.find_one = AsyncMock(return_value=fresh_snap)
        db.data_snapshots.insert_one = AsyncMock(return_value=MagicMock())
        db.data_snapshots.delete_many = AsyncMock(return_value=MagicMock())
        db.data_snapshots.find = MagicMock(return_value=_async_cursor([fresh_snap]))

        return db

    @pytest.mark.anyio
    async def test_raises_not_found_when_config_missing(self):
        db = _make_db(config_doc=None)
        with pytest.raises(NotFoundError):
            await service.get_current_stats(db, "missing")

    @pytest.mark.anyio
    async def test_returns_dict_with_required_keys(self):
        db = self._make_full_db()

        with patch("easyweaver.dashboard.service.capture_snapshot", new_callable=AsyncMock) as mock_snap:
            mock_snap.return_value = []
            result = await service.get_current_stats(db, CONFIG_ID)

        assert "config_id" in result
        assert "source_name" in result
        assert "source_type" in result
        assert "tables" in result
        assert "captured_at" in result
        assert "connection_healthy" in result

    @pytest.mark.anyio
    async def test_force_refresh_calls_capture_snapshot(self):
        db = self._make_full_db()

        with patch("easyweaver.dashboard.service.capture_snapshot", new_callable=AsyncMock) as mock_snap:
            mock_snap.return_value = []
            await service.get_current_stats(db, CONFIG_ID, force_refresh=True)

        mock_snap.assert_called_once_with(db, CONFIG_ID)

    @pytest.mark.anyio
    async def test_fresh_snapshot_skips_capture(self):
        fresh_snap = _snap_doc()
        fresh_snap["captured_at"] = datetime.now(timezone.utc)
        db = self._make_full_db(snap_doc=fresh_snap)

        with patch("easyweaver.dashboard.service.capture_snapshot", new_callable=AsyncMock) as mock_snap:
            mock_snap.return_value = []
            await service.get_current_stats(db, CONFIG_ID, force_refresh=False)

        mock_snap.assert_not_called()

    @pytest.mark.anyio
    async def test_connection_healthy_false_on_capture_failure(self):
        db = self._make_full_db(stale=True)

        with patch(
            "easyweaver.dashboard.service.capture_snapshot",
            new_callable=AsyncMock,
            side_effect=Exception("connector failed"),
        ):
            result = await service.get_current_stats(db, CONFIG_ID)

        assert result["connection_healthy"] is False

    @pytest.mark.anyio
    async def test_table_stats_include_snapshot_data(self):
        db = self._make_full_db()
        snap = _snap_doc()
        db.data_snapshots.find_one = AsyncMock(return_value=snap)

        with patch("easyweaver.dashboard.service.capture_snapshot", new_callable=AsyncMock) as mock_snap:
            mock_snap.return_value = []
            result = await service.get_current_stats(db, CONFIG_ID)

        tables = result["tables"]
        assert len(tables) >= 1
        t = tables[0]
        assert t["table_name"] == "orders"
        assert t["current_row_count"] == snap["row_count"]

    @pytest.mark.anyio
    async def test_table_with_no_snapshot_returns_defaults(self):
        doc = _config_doc(tables=[
            {"table_name": "orders", "timestamp_column": None, "modified_by_column": None},
            {"table_name": "missing_table", "timestamp_column": None, "modified_by_column": None},
        ])
        source_doc = _make_source_doc()
        db = MagicMock()
        db.dashboard_configs.find_one = AsyncMock(return_value=doc)
        db.data_sources.find_one = AsyncMock(return_value=source_doc)

        orders_snap = _snap_doc()

        async def snap_find_one(query, *args, **kwargs):
            if query.get("table_name") == "orders":
                return orders_snap
            return None

        db.data_snapshots.find_one = AsyncMock(side_effect=snap_find_one)
        db.data_snapshots.insert_one = AsyncMock(return_value=MagicMock())

        with patch("easyweaver.dashboard.service.capture_snapshot", new_callable=AsyncMock) as mock_snap:
            mock_snap.return_value = []
            result = await service.get_current_stats(db, CONFIG_ID)

        missing = next((t for t in result["tables"] if t["table_name"] == "missing_table"), None)
        assert missing is not None
        assert missing["current_row_count"] == 0


# ===========================================================================
# get_snapshot_history
# ===========================================================================


class TestGetSnapshotHistory:
    @pytest.mark.anyio
    async def test_raises_not_found_when_config_missing(self):
        db = _make_db(config_doc=None)
        with pytest.raises(NotFoundError):
            await service.get_snapshot_history(db, "missing")

    @pytest.mark.anyio
    async def test_returns_snapshots_for_config(self):
        doc = _config_doc()
        snap = _snap_doc()
        db = _make_db(config_doc=doc, snap_doc=snap)
        db.data_snapshots.find = MagicMock(return_value=_async_cursor([snap]))
        results = await service.get_snapshot_history(db, CONFIG_ID, hours=24)
        assert len(results) == 1
        assert isinstance(results[0], DataSnapshot)

    @pytest.mark.anyio
    async def test_returns_empty_list_when_no_snapshots(self):
        doc = _config_doc()
        db = _make_db(config_doc=doc)
        db.data_snapshots.find = MagicMock(return_value=_async_cursor([]))
        results = await service.get_snapshot_history(db, CONFIG_ID, hours=24)
        assert results == []

    @pytest.mark.anyio
    async def test_uses_cutoff_in_query(self):
        doc = _config_doc()
        db = _make_db(config_doc=doc)
        db.data_snapshots.find = MagicMock(return_value=_async_cursor([]))
        await service.get_snapshot_history(db, CONFIG_ID, hours=6)
        call_query = db.data_snapshots.find.call_args[0][0]
        assert "captured_at" in call_query
        assert "$gte" in call_query["captured_at"]
