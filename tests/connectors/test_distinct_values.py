"""Unit tests for get_distinct_values across all connectors."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

class _FakeAsyncCM:
    """Reusable async context manager that yields a fixed value."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        return False


# ── Postgres ─────────────────────────────────────────────────────────────────

@pytest.fixture
def pg_connector():
    from easyweaver.connectors.implementations.postgres import PostgresConnector

    connector = PostgresConnector.__new__(PostgresConnector)
    connector.credentials = {"user": "u", "password": "p", "host": "h", "port": 5432, "database": "d"}
    connector._column_types = {}
    connector._pk_cache = {}
    connector._temp_files = []

    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAsyncCM(mock_conn)
    connector._pool = mock_pool
    connector._mock_conn = mock_conn
    return connector


class TestPostgresDistinct:
    @pytest.mark.anyio
    async def test_returns_values_not_truncated(self, pg_connector):
        # Simulate 3 rows returned for limit=500 (under threshold)
        pg_connector._mock_conn.fetch = AsyncMock(
            return_value=[{"status": "active"}, {"status": "inactive"}, {"status": "pending"}]
        )
        result = await pg_connector.get_distinct_values("public.orders", "status", limit=500)
        assert result["values"] == ["active", "inactive", "pending"]
        assert result["truncated"] is False
        assert result["total_count"] == 3

    @pytest.mark.anyio
    async def test_truncated_when_over_limit(self, pg_connector):
        # limit=2, so we fetch 3; DB returns 3 rows -> truncated
        rows = [{"color": "blue"}, {"color": "green"}, {"color": "red"}]
        pg_connector._mock_conn.fetch = AsyncMock(return_value=rows)
        result = await pg_connector.get_distinct_values("public.items", "color", limit=2)
        assert result["values"] == ["blue", "green"]
        assert result["truncated"] is True
        assert result["total_count"] is None

    @pytest.mark.anyio
    async def test_empty_result(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[])
        result = await pg_connector.get_distinct_values("public.items", "name", limit=10)
        assert result["values"] == []
        assert result["truncated"] is False
        assert result["total_count"] == 0

    @pytest.mark.anyio
    async def test_sql_uses_quote_ident(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[])
        await pg_connector.get_distinct_values("myschema.mytable", "my col", limit=5)
        call_args = pg_connector._mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert '"my col"' in sql
        assert '"myschema"."mytable"' in sql
        assert "LIMIT 6" in sql


# ── MySQL ────────────────────────────────────────────────────────────────────

@pytest.fixture
def mysql_connector():
    from easyweaver.connectors.implementations.mysql import MySQLConnector

    connector = MySQLConnector.__new__(MySQLConnector)
    connector.credentials = {"user": "u", "password": "p", "host": "h", "port": 3306, "database": "d"}
    connector._column_types = {}
    connector._pk_cache = {}
    connector._temp_files = []

    mock_cursor = AsyncMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = _FakeAsyncCM(mock_cursor)
    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAsyncCM(mock_conn)
    connector._pool = mock_pool
    connector._mock_cursor = mock_cursor
    return connector


class TestMySQLDistinct:
    @pytest.mark.anyio
    async def test_returns_values(self, mysql_connector):
        mysql_connector._mock_cursor.fetchall = AsyncMock(
            return_value=[{"city": "Amsterdam"}, {"city": "Berlin"}]
        )
        result = await mysql_connector.get_distinct_values("d.users", "city", limit=500)
        assert result["values"] == ["Amsterdam", "Berlin"]
        assert result["truncated"] is False
        assert result["total_count"] == 2

    @pytest.mark.anyio
    async def test_truncated(self, mysql_connector):
        rows = [{"x": 1}, {"x": 2}, {"x": 3}]
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=rows)
        result = await mysql_connector.get_distinct_values("d.t", "x", limit=2)
        assert result["values"] == [1, 2]
        assert result["truncated"] is True


# ── DB2 ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def db2_connector():
    with patch("easyweaver.connectors.implementations.db2._HAS_IBM_DB", True), \
         patch("easyweaver.connectors.implementations.db2._require_ibm_db"):
        from easyweaver.connectors.implementations.db2 import DB2Connector

        connector = DB2Connector.__new__(DB2Connector)
        connector.credentials = {"user": "u", "password": "p", "host": "h", "port": 50000, "database": "d"}
        connector._column_types = {}
        connector._pk_cache = {}

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        connector._conn = mock_conn
        connector._mock_cursor = mock_cursor
        return connector


class TestDB2Distinct:
    @pytest.mark.anyio
    async def test_returns_values(self, db2_connector):
        db2_connector._mock_cursor.fetchall.return_value = [("A",), ("B",), ("C",)]
        result = await db2_connector.get_distinct_values("MYSCHEMA.MYTABLE", "COL1", limit=500)
        assert result["values"] == ["A", "B", "C"]
        assert result["truncated"] is False

    @pytest.mark.anyio
    async def test_truncated(self, db2_connector):
        db2_connector._mock_cursor.fetchall.return_value = [(1,), (2,), (3,)]
        result = await db2_connector.get_distinct_values("S.T", "X", limit=2)
        assert result["values"] == [1, 2]
        assert result["truncated"] is True

    @pytest.mark.anyio
    async def test_sql_uses_fetch_first(self, db2_connector):
        db2_connector._mock_cursor.fetchall.return_value = []
        await db2_connector.get_distinct_values("S.T", "C", limit=10)
        sql = db2_connector._mock_cursor.execute.call_args[0][0]
        assert "FETCH FIRST 11 ROWS ONLY" in sql


# ── MongoDB ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mongo_connector():
    from easyweaver.connectors.implementations.mongodb import MongoDBConnector

    connector = MongoDBConnector.__new__(MongoDBConnector)
    connector.credentials = {"host": "localhost", "database": "test"}
    connector._db_name = "test"

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_collection.distinct = AsyncMock(return_value=["b", None, "a", "c"])
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)
    mock_client.__getitem__ = MagicMock(return_value=mock_db)
    connector._client = mock_client
    return connector


class TestMongoDBDistinct:
    @pytest.mark.anyio
    async def test_filters_nulls_and_sorts(self, mongo_connector):
        result = await mongo_connector.get_distinct_values("users", "status", limit=500)
        assert result["values"] == ["a", "b", "c"]
        assert result["truncated"] is False
        assert result["total_count"] == 3

    @pytest.mark.anyio
    async def test_truncated(self, mongo_connector):
        result = await mongo_connector.get_distinct_values("users", "status", limit=2)
        assert result["values"] == ["a", "b"]
        assert result["truncated"] is True
        assert result["total_count"] is None


# ── File Source ──────────────────────────────────────────────────────────────

@pytest.fixture
def file_connector():
    from easyweaver.connectors.implementations.file_source import FileConnector

    connector = FileConnector.__new__(FileConnector)
    connector.credentials = {"gcp_path": "x", "file_format": "csv", "original_filename": "data.csv"}
    connector._gcp_path = "x"
    connector._file_format = "csv"
    connector._original_filename = "data.csv"
    connector._df = pl.DataFrame({
        "color": ["red", "blue", "red", "green", None, "blue"],
        "size": [1, 2, 3, 4, None, 5],
    })
    return connector


class TestFileDistinct:
    @pytest.mark.anyio
    async def test_returns_unique_sorted(self, file_connector):
        result = await file_connector.get_distinct_values("data", "color", limit=500)
        assert result["values"] == ["blue", "green", "red"]
        assert result["truncated"] is False
        assert result["total_count"] == 3

    @pytest.mark.anyio
    async def test_truncated(self, file_connector):
        result = await file_connector.get_distinct_values("data", "color", limit=2)
        assert result["values"] == ["blue", "green"]
        assert result["truncated"] is True

    @pytest.mark.anyio
    async def test_numeric_column(self, file_connector):
        result = await file_connector.get_distinct_values("data", "size", limit=500)
        assert result["values"] == [1, 2, 3, 4, 5]
        assert result["truncated"] is False


# ── REST API ─────────────────────────────────────────────────────────────────

class TestRestAPIDistinct:
    @pytest.mark.anyio
    async def test_returns_empty(self):
        from easyweaver.connectors.implementations.rest_api import RestAPIConnector

        connector = RestAPIConnector.__new__(RestAPIConnector)
        connector.credentials = {"base_url": "http://example.com", "data_path": "data"}
        result = await connector.get_distinct_values("endpoint", "col", limit=100)
        assert result == {"values": [], "truncated": False, "total_count": 0}
