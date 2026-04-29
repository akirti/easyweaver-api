"""Tests for easyweaver.results.redis_store (RedisResultStore)"""
import io
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from easyweaver.results.redis_store import RedisResultStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(rows=3) -> pl.DataFrame:
    return pl.DataFrame({"id": list(range(rows)), "value": [f"v{i}" for i in range(rows)]})


def _df_to_parquet_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd", compression_level=3)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _key
# ---------------------------------------------------------------------------


class TestRedisResultStoreKey:
    def test_key_format(self):
        store = RedisResultStore(redis=MagicMock())
        assert store._key("run-123") == "query_result:run-123"

    def test_key_unique_per_run_id(self):
        store = RedisResultStore(redis=MagicMock())
        assert store._key("a") != store._key("b")

    def test_key_prefix(self):
        store = RedisResultStore(redis=MagicMock())
        key = store._key("any-id")
        assert key.startswith("query_result:")


# ---------------------------------------------------------------------------
# store_result
# ---------------------------------------------------------------------------


class TestStoreResult:
    @pytest.fixture
    def store(self):
        redis_mock = MagicMock()
        return RedisResultStore(redis=redis_mock)

    async def test_stores_dataframe_in_redis(self, store):
        df = _make_df()
        raw_redis_mock = AsyncMock()
        raw_redis_mock.setex = AsyncMock()
        raw_redis_mock.aclose = AsyncMock()

        with patch("easyweaver.results.redis_store.Redis") as MockRedis:
            MockRedis.from_url.return_value = raw_redis_mock
            with patch("easyweaver.results.redis_store.settings") as mock_settings:
                mock_settings.result_ttl_seconds = 3600
                mock_settings.redis_url = "redis://localhost:6379/0"
                await store.store_result("run-1", df)

        raw_redis_mock.setex.assert_called_once()
        call_args = raw_redis_mock.setex.call_args
        key, ttl, data = call_args.args
        assert key == "query_result:run-1"
        assert ttl == 3600
        assert isinstance(data, bytes)

    async def test_uses_custom_ttl(self, store):
        df = _make_df()
        raw_redis_mock = AsyncMock()
        raw_redis_mock.setex = AsyncMock()
        raw_redis_mock.aclose = AsyncMock()

        with patch("easyweaver.results.redis_store.Redis") as MockRedis:
            MockRedis.from_url.return_value = raw_redis_mock
            with patch("easyweaver.results.redis_store.settings") as mock_settings:
                mock_settings.redis_url = "redis://localhost:6379/0"
                await store.store_result("run-2", df, ttl=7200)

        call_args = raw_redis_mock.setex.call_args
        assert call_args.args[1] == 7200

    async def test_falls_back_to_settings_ttl_when_none(self, store):
        df = _make_df()
        raw_redis_mock = AsyncMock()
        raw_redis_mock.setex = AsyncMock()
        raw_redis_mock.aclose = AsyncMock()

        with patch("easyweaver.results.redis_store.Redis") as MockRedis:
            MockRedis.from_url.return_value = raw_redis_mock
            with patch("easyweaver.results.redis_store.settings") as mock_settings:
                mock_settings.result_ttl_seconds = 999
                mock_settings.redis_url = "redis://localhost:6379/0"
                await store.store_result("run-3", df, ttl=None)

        call_args = raw_redis_mock.setex.call_args
        assert call_args.args[1] == 999

    async def test_aclose_called_even_on_error(self, store):
        df = _make_df()
        raw_redis_mock = AsyncMock()
        raw_redis_mock.setex = AsyncMock(side_effect=RuntimeError("connection lost"))
        raw_redis_mock.aclose = AsyncMock()

        with patch("easyweaver.results.redis_store.Redis") as MockRedis:
            MockRedis.from_url.return_value = raw_redis_mock
            with patch("easyweaver.results.redis_store.settings") as mock_settings:
                mock_settings.result_ttl_seconds = 3600
                mock_settings.redis_url = "redis://localhost:6379/0"
                with pytest.raises(RuntimeError):
                    await store.store_result("run-err", df)

        raw_redis_mock.aclose.assert_called_once()

    async def test_stored_data_is_parquet(self, store):
        df = _make_df(5)
        captured_data = {}

        async def fake_setex(key, ttl, data):
            captured_data["data"] = data

        raw_redis_mock = AsyncMock()
        raw_redis_mock.setex = fake_setex
        raw_redis_mock.aclose = AsyncMock()

        with patch("easyweaver.results.redis_store.Redis") as MockRedis:
            MockRedis.from_url.return_value = raw_redis_mock
            with patch("easyweaver.results.redis_store.settings") as mock_settings:
                mock_settings.result_ttl_seconds = 3600
                mock_settings.redis_url = "redis://localhost"
                await store.store_result("run-4", df)

        # Verify the bytes are valid parquet
        recovered = pl.read_parquet(io.BytesIO(captured_data["data"]))
        assert recovered.shape == df.shape


# ---------------------------------------------------------------------------
# get_result
# ---------------------------------------------------------------------------


class TestGetResult:
    @pytest.fixture
    def store(self):
        return RedisResultStore(redis=MagicMock())

    async def test_returns_none_when_key_missing(self, store):
        raw_redis_mock = AsyncMock()
        raw_redis_mock.get = AsyncMock(return_value=None)
        raw_redis_mock.aclose = AsyncMock()

        with patch("easyweaver.results.redis_store.Redis") as MockRedis:
            MockRedis.from_url.return_value = raw_redis_mock
            with patch("easyweaver.results.redis_store.settings") as mock_settings:
                mock_settings.redis_url = "redis://localhost"
                result = await store.get_result("missing-run")

        assert result is None

    async def test_returns_dataframe_when_key_exists(self, store):
        df = _make_df(4)
        parquet_bytes = _df_to_parquet_bytes(df)

        raw_redis_mock = AsyncMock()
        raw_redis_mock.get = AsyncMock(return_value=parquet_bytes)
        raw_redis_mock.aclose = AsyncMock()

        with patch("easyweaver.results.redis_store.Redis") as MockRedis:
            MockRedis.from_url.return_value = raw_redis_mock
            with patch("easyweaver.results.redis_store.settings") as mock_settings:
                mock_settings.redis_url = "redis://localhost"
                result = await store.get_result("run-4")

        assert isinstance(result, pl.DataFrame)
        assert result.shape == df.shape

    async def test_aclose_called_after_get(self, store):
        raw_redis_mock = AsyncMock()
        raw_redis_mock.get = AsyncMock(return_value=None)
        raw_redis_mock.aclose = AsyncMock()

        with patch("easyweaver.results.redis_store.Redis") as MockRedis:
            MockRedis.from_url.return_value = raw_redis_mock
            with patch("easyweaver.results.redis_store.settings") as mock_settings:
                mock_settings.redis_url = "redis://localhost"
                await store.get_result("run-x")

        raw_redis_mock.aclose.assert_called_once()

    async def test_get_uses_correct_key(self, store):
        raw_redis_mock = AsyncMock()
        raw_redis_mock.get = AsyncMock(return_value=None)
        raw_redis_mock.aclose = AsyncMock()

        with patch("easyweaver.results.redis_store.Redis") as MockRedis:
            MockRedis.from_url.return_value = raw_redis_mock
            with patch("easyweaver.results.redis_store.settings") as mock_settings:
                mock_settings.redis_url = "redis://localhost"
                await store.get_result("specific-run")

        raw_redis_mock.get.assert_called_once_with("query_result:specific-run")

    async def test_aclose_called_on_error(self, store):
        raw_redis_mock = AsyncMock()
        raw_redis_mock.get = AsyncMock(side_effect=RuntimeError("redis down"))
        raw_redis_mock.aclose = AsyncMock()

        with patch("easyweaver.results.redis_store.Redis") as MockRedis:
            MockRedis.from_url.return_value = raw_redis_mock
            with patch("easyweaver.results.redis_store.settings") as mock_settings:
                mock_settings.redis_url = "redis://localhost"
                with pytest.raises(RuntimeError):
                    await store.get_result("run-err")

        raw_redis_mock.aclose.assert_called_once()

    async def test_roundtrip_store_then_get(self, store):
        """Integration-style: store then retrieve verifies data fidelity."""
        df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        parquet_bytes = _df_to_parquet_bytes(df)

        stored = {}

        async def fake_setex(key, ttl, data):
            stored[key] = data

        async def fake_get(key):
            return stored.get(key)

        raw_redis_mock = AsyncMock()
        raw_redis_mock.setex = fake_setex
        raw_redis_mock.get = fake_get
        raw_redis_mock.aclose = AsyncMock()

        with patch("easyweaver.results.redis_store.Redis") as MockRedis:
            MockRedis.from_url.return_value = raw_redis_mock
            with patch("easyweaver.results.redis_store.settings") as mock_settings:
                mock_settings.result_ttl_seconds = 3600
                mock_settings.redis_url = "redis://localhost"
                await store.store_result("roundtrip-run", df)
                result = await store.get_result("roundtrip-run")

        assert result is not None
        assert result.to_dicts() == df.to_dicts()


# ---------------------------------------------------------------------------
# delete_result
# ---------------------------------------------------------------------------


class TestDeleteResult:
    async def test_delete_calls_redis_delete(self):
        redis_mock = AsyncMock()
        redis_mock.delete = AsyncMock()
        store = RedisResultStore(redis=redis_mock)

        await store.delete_result("run-to-delete")

        redis_mock.delete.assert_called_once_with("query_result:run-to-delete")

    async def test_delete_uses_correct_key(self):
        redis_mock = AsyncMock()
        redis_mock.delete = AsyncMock()
        store = RedisResultStore(redis=redis_mock)

        await store.delete_result("my-special-run")

        call_args = redis_mock.delete.call_args
        assert call_args.args[0] == "query_result:my-special-run"
