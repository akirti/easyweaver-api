"""Tests for easyweaver.dependencies — DB/Redis/semaphore lifecycle."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import easyweaver.dependencies as deps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_globals():
    """Reset module-level globals to None before each test."""
    deps.motor_client = None
    deps._meta_db = None
    deps.redis_client = None
    deps._redis_pool = None
    deps._query_semaphore = None


# ---------------------------------------------------------------------------
# get_query_semaphore
# ---------------------------------------------------------------------------


class TestGetQuerySemaphore:
    def setup_method(self):
        _reset_globals()

    def test_returns_semaphore(self):
        sem = deps.get_query_semaphore()
        assert isinstance(sem, asyncio.Semaphore)

    def test_singleton_same_object(self):
        sem1 = deps.get_query_semaphore()
        sem2 = deps.get_query_semaphore()
        assert sem1 is sem2

    def test_uses_max_concurrent_queries_from_settings(self):
        with patch("easyweaver.dependencies.settings") as mock_settings:
            mock_settings.max_concurrent_queries = 5
            deps._query_semaphore = None  # force re-init
            sem = deps.get_query_semaphore()
            # asyncio.Semaphore stores value internally
            assert sem._value == 5


# ---------------------------------------------------------------------------
# get_db
# ---------------------------------------------------------------------------


class TestGetDb:
    def setup_method(self):
        _reset_globals()

    @pytest.mark.anyio
    async def test_raises_when_not_initialized(self):
        deps._meta_db = None
        with pytest.raises(AssertionError, match="MongoDB not initialized"):
            await deps.get_db()

    @pytest.mark.anyio
    async def test_returns_db_when_initialized(self):
        mock_db = MagicMock()
        deps._meta_db = mock_db
        result = await deps.get_db()
        assert result is mock_db


# ---------------------------------------------------------------------------
# get_meta_db
# ---------------------------------------------------------------------------


class TestGetMetaDb:
    def setup_method(self):
        _reset_globals()

    def test_raises_when_not_initialized(self):
        deps._meta_db = None
        with pytest.raises(AssertionError, match="MongoDB not initialized"):
            deps.get_meta_db()

    def test_returns_db_when_initialized(self):
        mock_db = MagicMock()
        deps._meta_db = mock_db
        result = deps.get_meta_db()
        assert result is mock_db


# ---------------------------------------------------------------------------
# get_redis
# ---------------------------------------------------------------------------


class TestGetRedis:
    def setup_method(self):
        _reset_globals()

    @pytest.mark.anyio
    async def test_raises_when_not_initialized(self):
        deps.redis_client = None
        with pytest.raises(AssertionError, match="Redis not initialized"):
            await deps.get_redis()

    @pytest.mark.anyio
    async def test_returns_client_when_initialized(self):
        mock_redis = MagicMock()
        deps.redis_client = mock_redis
        result = await deps.get_redis()
        assert result is mock_redis


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


class TestInitDb:
    def setup_method(self):
        _reset_globals()

    @pytest.mark.anyio
    async def test_sets_motor_client_and_meta_db(self):
        mock_collection = MagicMock()
        mock_collection.create_index = AsyncMock(return_value="index_name")
        mock_db = MagicMock()
        # All collection attribute accesses on the db return mock_collection
        mock_db.users = mock_collection
        mock_db.data_sources = mock_collection
        mock_db.query_runs = mock_collection
        mock_db.process_configurations = mock_collection
        mock_db.process_runs = mock_collection
        mock_db.dashboard_configs = mock_collection
        mock_db.data_snapshots = mock_collection
        mock_db.configurations = mock_collection
        mock_db.batch_size_history = mock_collection

        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        mock_client.easyweaver_meta = mock_db

        with patch("easyweaver.dependencies.AsyncIOMotorClient", return_value=mock_client):
            with patch("easyweaver.dependencies.settings") as mock_settings:
                mock_settings.mongo_url = "mongodb://localhost:27017"
                mock_settings.db_pool_max_size = 10
                mock_settings.db_pool_min_size = 1
                await deps.init_db()

        assert deps.motor_client is mock_client

    @pytest.mark.anyio
    async def test_creates_indexes(self):
        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        mock_collection = MagicMock()
        mock_collection.create_index = AsyncMock(return_value="idx")
        mock_db = MagicMock()
        for attr in [
            "users", "data_sources", "query_runs", "process_configurations",
            "process_runs", "dashboard_configs", "data_snapshots",
            "configurations", "batch_size_history",
        ]:
            setattr(mock_db, attr, mock_collection)
        mock_client.easyweaver_meta = mock_db

        with patch("easyweaver.dependencies.AsyncIOMotorClient", return_value=mock_client):
            with patch("easyweaver.dependencies.settings") as mock_settings:
                mock_settings.mongo_url = "mongodb://localhost:27017"
                mock_settings.db_pool_max_size = 10
                mock_settings.db_pool_min_size = 1
                await deps.init_db()

        # create_index was called at least once
        assert mock_collection.create_index.call_count > 0


# ---------------------------------------------------------------------------
# shutdown_db
# ---------------------------------------------------------------------------


class TestShutdownDb:
    def setup_method(self):
        _reset_globals()

    @pytest.mark.anyio
    async def test_closes_motor_client(self):
        mock_client = MagicMock()
        mock_client.close = MagicMock()
        deps.motor_client = mock_client
        deps._meta_db = MagicMock()

        await deps.shutdown_db()

        mock_client.close.assert_called_once()
        assert deps.motor_client is None
        assert deps._meta_db is None

    @pytest.mark.anyio
    async def test_noop_when_already_none(self):
        deps.motor_client = None
        # Should not raise
        await deps.shutdown_db()


# ---------------------------------------------------------------------------
# init_redis
# ---------------------------------------------------------------------------


class TestInitRedis:
    def setup_method(self):
        _reset_globals()

    @pytest.mark.anyio
    async def test_sets_redis_client(self):
        mock_pool = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with patch("easyweaver.dependencies.ConnectionPool") as mock_cp:
            mock_cp.from_url = MagicMock(return_value=mock_pool)
            with patch("easyweaver.dependencies.Redis", return_value=mock_redis):
                with patch("easyweaver.dependencies.settings") as mock_settings:
                    mock_settings.redis_url = "redis://localhost:6379/0"
                    mock_settings.redis_max_connections = 20
                    await deps.init_redis()

        assert deps.redis_client is mock_redis
        assert deps._redis_pool is mock_pool

    @pytest.mark.anyio
    async def test_ping_called_after_init(self):
        mock_pool = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with patch("easyweaver.dependencies.ConnectionPool") as mock_cp:
            mock_cp.from_url = MagicMock(return_value=mock_pool)
            with patch("easyweaver.dependencies.Redis", return_value=mock_redis):
                with patch("easyweaver.dependencies.settings") as mock_settings:
                    mock_settings.redis_url = "redis://localhost:6379/0"
                    mock_settings.redis_max_connections = 20
                    await deps.init_redis()

        mock_redis.ping.assert_called_once()


# ---------------------------------------------------------------------------
# shutdown_redis
# ---------------------------------------------------------------------------


class TestShutdownRedis:
    def setup_method(self):
        _reset_globals()

    @pytest.mark.anyio
    async def test_closes_redis_and_pool(self):
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.aclose = AsyncMock()

        deps.redis_client = mock_redis
        deps._redis_pool = mock_pool

        await deps.shutdown_redis()

        mock_redis.aclose.assert_called_once()
        mock_pool.aclose.assert_called_once()
        assert deps.redis_client is None
        assert deps._redis_pool is None

    @pytest.mark.anyio
    async def test_noop_when_already_none(self):
        deps.redis_client = None
        deps._redis_pool = None
        # Should not raise
        await deps.shutdown_redis()
