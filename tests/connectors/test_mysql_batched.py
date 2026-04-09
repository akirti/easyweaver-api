"""Unit tests for MySQLConnector.execute_query_batched."""

from unittest.mock import AsyncMock, MagicMock

import pytest


class _FakeAsyncCM:
    """Reusable async context manager that yields a fixed value."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        return False


class _FakeCursorCM:
    """Async CM for cursor() that supports configurable responses per call."""

    def __init__(self, cursor):
        self._cursor = cursor

    async def __aenter__(self):
        return self._cursor

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def mysql_connector():
    """Create a MySQLConnector with mocked pool."""
    from easyweaver.connectors.implementations.mysql import MySQLConnector

    connector = MySQLConnector.__new__(MySQLConnector)
    connector.credentials = {"user": "u", "password": "p", "host": "h", "port": 3306, "database": "testdb"}
    connector._column_types = {}
    connector._pk_cache = {}
    connector._temp_files = []

    mock_cursor = AsyncMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = _FakeCursorCM(mock_cursor)

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAsyncCM(mock_conn)

    connector._pool = mock_pool
    connector._mock_cursor = mock_cursor
    connector._mock_conn = mock_conn
    return connector


class TestSupportsBatching:
    def test_supports_batching_is_true(self, mysql_connector):
        assert mysql_connector.supports_batching is True


class TestGetPrimaryKey:
    @pytest.mark.anyio
    async def test_pk_detected_via_show_keys(self, mysql_connector):
        mysql_connector._mock_cursor.fetchall = AsyncMock(
            return_value=[{"Column_name": "order_id"}]
        )

        pk = await mysql_connector._get_primary_key("testdb.orders")
        assert pk == "order_id"
        assert mysql_connector._pk_cache["testdb.orders"] == "order_id"

    @pytest.mark.anyio
    async def test_pk_cached(self, mysql_connector):
        mysql_connector._pk_cache["testdb.orders"] = "id"
        pk = await mysql_connector._get_primary_key("testdb.orders")
        assert pk == "id"

    @pytest.mark.anyio
    async def test_fallback_to_first_column(self, mysql_connector):
        call_count = [0]

        async def multi_fetchall():
            call_count[0] += 1
            if call_count[0] == 1:
                return []  # no PK
            return [{"Field": "created_at"}]

        mysql_connector._mock_cursor.fetchall = multi_fetchall

        pk = await mysql_connector._get_primary_key("testdb.orders")
        assert pk == "created_at"


class TestBuildWhereClause:
    def test_empty_filters(self, mysql_connector):
        clause, params = mysql_connector._build_where_clause([], {}, "and")
        assert clause == ""
        assert params == []

    def test_single_eq_filter(self, mysql_connector):
        filters = [{"column": "status", "operator": "eq", "value": "active"}]
        clause, params = mysql_connector._build_where_clause(
            filters, {"status": "varchar"}, "and"
        )
        assert "`status` = %s" in clause
        assert params == ["active"]

    def test_or_logic(self, mysql_connector):
        filters = [
            {"column": "a", "operator": "eq", "value": "x"},
            {"column": "b", "operator": "eq", "value": "y"},
        ]
        clause, params = mysql_connector._build_where_clause(
            filters, {"a": "varchar", "b": "varchar"}, "or"
        )
        assert " OR " in clause


class TestExecuteQueryBatched:
    @pytest.mark.anyio
    async def test_sql_generation_no_offset(self, mysql_connector):
        """First batch (no last_key, offset=0) should have no OFFSET."""
        mysql_connector._pk_cache["testdb.t"] = "id"

        captured = []

        async def mock_execute(query, params=None):
            captured.append(query)

        mysql_connector._mock_cursor.execute = mock_execute
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[])

        rows, has_more, last_pk = await mysql_connector.execute_query_batched(
            table="testdb.t", batch_size=100, offset=0
        )

        sql = captured[0]
        assert "ORDER BY `id`" in sql
        assert "LIMIT 101" in sql
        assert "OFFSET" not in sql

    @pytest.mark.anyio
    async def test_offset_fallback(self, mysql_connector):
        """When last_key is None and offset > 0, OFFSET is appended."""
        mysql_connector._pk_cache["testdb.t"] = "id"

        captured = []

        async def mock_execute(query, params=None):
            captured.append(query)

        mysql_connector._mock_cursor.execute = mock_execute
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[])

        rows, has_more, last_pk = await mysql_connector.execute_query_batched(
            table="testdb.t", batch_size=100, offset=50
        )

        sql = captured[0]
        assert "OFFSET 50" in sql

    @pytest.mark.anyio
    async def test_keyset_pagination_with_last_key(self, mysql_connector):
        """When last_key is provided, WHERE pk > %s is used and no OFFSET."""
        mysql_connector._pk_cache["testdb.t"] = "id"

        captured = []
        captured_params = []

        async def mock_execute(query, params=None):
            captured.append(query)
            captured_params.append(params)

        mysql_connector._mock_cursor.execute = mock_execute
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[])

        rows, has_more, last_pk = await mysql_connector.execute_query_batched(
            table="testdb.t", batch_size=100, offset=50, last_key=42
        )

        sql = captured[0]
        assert "`id` > %s" in sql
        assert "OFFSET" not in sql
        assert 42 in captured_params[0]

    @pytest.mark.anyio
    async def test_has_more_true(self, mysql_connector):
        mysql_connector._pk_cache["testdb.t"] = "id"

        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(
            return_value=[{"id": i} for i in range(11)]
        )

        rows, has_more, last_pk = await mysql_connector.execute_query_batched(
            table="testdb.t", batch_size=10, offset=0
        )
        assert has_more is True
        assert len(rows) == 10
        assert last_pk == 9

    @pytest.mark.anyio
    async def test_has_more_false(self, mysql_connector):
        mysql_connector._pk_cache["testdb.t"] = "id"

        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(
            return_value=[{"id": i} for i in range(5)]
        )

        rows, has_more, last_pk = await mysql_connector.execute_query_batched(
            table="testdb.t", batch_size=10, offset=0
        )
        assert has_more is False
        assert len(rows) == 5
        assert last_pk == 4

    @pytest.mark.anyio
    async def test_returns_3_tuple(self, mysql_connector):
        mysql_connector._pk_cache["testdb.t"] = "id"

        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[])

        result = await mysql_connector.execute_query_batched(
            table="testdb.t", batch_size=10, offset=0
        )
        assert len(result) == 3
        rows, has_more, last_pk = result
        assert rows == []
        assert has_more is False
        assert last_pk is None

    @pytest.mark.anyio
    async def test_filters_in_query(self, mysql_connector):
        mysql_connector._pk_cache["testdb.t"] = "id"
        mysql_connector._column_types["testdb.t"] = {"status": "varchar"}

        captured = []

        async def mock_execute(query, params=None):
            captured.append(query)

        mysql_connector._mock_cursor.execute = mock_execute
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[])

        await mysql_connector.execute_query_batched(
            table="testdb.t",
            filters=[{"column": "status", "operator": "eq", "value": "active"}],
            batch_size=10,
        )

        sql = captured[0]
        assert "WHERE" in sql
        assert "`status` = %s" in sql
        assert "ORDER BY `id`" in sql

    @pytest.mark.anyio
    async def test_column_selection(self, mysql_connector):
        mysql_connector._pk_cache["testdb.t"] = "id"

        captured = []

        async def mock_execute(query, params=None):
            captured.append(query)

        mysql_connector._mock_cursor.execute = mock_execute
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[])

        await mysql_connector.execute_query_batched(
            table="testdb.t",
            columns=["name", "email"],
            batch_size=10,
        )

        sql = captured[0]
        assert "`name`, `email`" in sql
