"""Extended tests for MySQLConnector — covering execute_query, get_schema,
preview_table, test_connection, _coerce_value, _build_where_clause edge cases,
_quote_ident, _qualified_table_name, connect/disconnect, _get_column_types."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Shared async CM helpers ──────────────────────────────────────────────────

class _FakeAsyncCM:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        return False


class _FakeCursorCM:
    def __init__(self, cursor):
        self._cursor = cursor

    async def __aenter__(self):
        return self._cursor

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def mysql_connector():
    from easyweaver.connectors.implementations.mysql import MySQLConnector

    connector = MySQLConnector.__new__(MySQLConnector)
    connector.credentials = {
        "user": "u", "password": "p", "host": "h", "port": 3306, "database": "testdb"
    }
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


# ── _quote_ident ─────────────────────────────────────────────────────────────

class TestQuoteIdent:
    def test_simple_name(self):
        from easyweaver.connectors.implementations.mysql import MySQLConnector
        assert MySQLConnector._quote_ident("user") == "`user`"

    def test_name_with_backtick(self):
        from easyweaver.connectors.implementations.mysql import MySQLConnector
        result = MySQLConnector._quote_ident("col`name")
        assert result == "`col``name`"

    def test_name_with_spaces(self):
        from easyweaver.connectors.implementations.mysql import MySQLConnector
        result = MySQLConnector._quote_ident("my col")
        assert result == "`my col`"


# ── _qualified_table_name ────────────────────────────────────────────────────

class TestQualifiedTableName:
    def test_schema_dot_table(self):
        from easyweaver.connectors.implementations.mysql import MySQLConnector
        result = MySQLConnector._qualified_table_name("mydb.mytable")
        assert result == "`mydb`.`mytable`"

    def test_no_schema(self):
        from easyweaver.connectors.implementations.mysql import MySQLConnector
        result = MySQLConnector._qualified_table_name("mytable")
        assert result == "`mytable`"


# ── _coerce_value ────────────────────────────────────────────────────────────

class TestCoerceValue:
    def test_none_returns_none(self, mysql_connector):
        assert mysql_connector._coerce_value(None, "varchar") is None

    def test_int_type_coerces_string(self, mysql_connector):
        assert mysql_connector._coerce_value("10", "int") == 10

    def test_int_passthrough(self, mysql_connector):
        assert mysql_connector._coerce_value(5, "bigint") == 5

    def test_float_type_coerces(self, mysql_connector):
        result = mysql_connector._coerce_value("3.14", "float")
        assert abs(result - 3.14) < 0.001

    def test_text_passthrough(self, mysql_connector):
        assert mysql_connector._coerce_value("hello", "varchar") == "hello"

    def test_coerce_fails_gracefully(self, mysql_connector):
        result = mysql_connector._coerce_value("abc", "int")
        assert result == "abc"


# ── _build_where_clause edge cases ──────────────────────────────────────────

class TestBuildWhereClauseEdgeCases:
    def test_neq_filter(self, mysql_connector):
        filters = [{"column": "status", "operator": "neq", "value": "inactive"}]
        clause, params = mysql_connector._build_where_clause(filters, {"status": "varchar"}, "and")
        assert "`status` != %s" in clause
        assert params == ["inactive"]

    def test_gt_filter(self, mysql_connector):
        filters = [{"column": "age", "operator": "gt", "value": "18"}]
        clause, params = mysql_connector._build_where_clause(filters, {"age": "int"}, "and")
        assert "`age` > %s" in clause
        assert params == [18]

    def test_lt_filter(self, mysql_connector):
        filters = [{"column": "price", "operator": "lt", "value": "100"}]
        clause, params = mysql_connector._build_where_clause(filters, {"price": "decimal"}, "and")
        assert "`price` < %s" in clause

    def test_gte_filter(self, mysql_connector):
        filters = [{"column": "score", "operator": "gte", "value": "50"}]
        clause, params = mysql_connector._build_where_clause(filters, {"score": "int"}, "and")
        assert "`score` >= %s" in clause

    def test_lte_filter(self, mysql_connector):
        filters = [{"column": "score", "operator": "lte", "value": "100"}]
        clause, params = mysql_connector._build_where_clause(filters, {"score": "int"}, "and")
        assert "`score` <= %s" in clause

    def test_like_filter_wraps_value(self, mysql_connector):
        filters = [{"column": "name", "operator": "like", "value": "john"}]
        clause, params = mysql_connector._build_where_clause(filters, {"name": "varchar"}, "and")
        assert "LIKE %s" in clause
        assert params == ["%john%"]

    def test_is_null_filter(self, mysql_connector):
        filters = [{"column": "deleted_at", "operator": "is_null"}]
        clause, params = mysql_connector._build_where_clause(filters, {}, "and")
        assert "`deleted_at` IS NULL" in clause
        assert params == []

    def test_is_not_null_filter(self, mysql_connector):
        filters = [{"column": "name", "operator": "is_not_null"}]
        clause, params = mysql_connector._build_where_clause(filters, {}, "and")
        assert "`name` IS NOT NULL" in clause

    def test_in_filter(self, mysql_connector):
        filters = [{"column": "id", "operator": "in", "value": [1, 2, 3]}]
        clause, params = mysql_connector._build_where_clause(filters, {"id": "int"}, "and")
        assert "IN" in clause
        assert len(params) == 3

    def test_in_empty_list_produces_false(self, mysql_connector):
        filters = [{"column": "id", "operator": "in", "value": []}]
        clause, params = mysql_connector._build_where_clause(filters, {"id": "int"}, "and")
        assert "FALSE" in clause

    def test_in_comma_string_splits(self, mysql_connector):
        filters = [{"column": "id", "operator": "in", "value": "1, 2, 3"}]
        clause, params = mysql_connector._build_where_clause(filters, {"id": "int"}, "and")
        assert "IN" in clause
        assert len(params) == 3

    def test_not_in_filter(self, mysql_connector):
        filters = [{"column": "status", "operator": "not_in", "value": ["a", "b"]}]
        clause, params = mysql_connector._build_where_clause(filters, {"status": "varchar"}, "and")
        assert "NOT IN" in clause
        assert len(params) == 2

    def test_not_in_empty_list_produces_true(self, mysql_connector):
        filters = [{"column": "id", "operator": "not_in", "value": []}]
        clause, params = mysql_connector._build_where_clause(filters, {"id": "int"}, "and")
        assert "TRUE" in clause

    def test_between_filter(self, mysql_connector):
        filters = [{"column": "price", "operator": "between", "value": "10", "value2": "50"}]
        clause, params = mysql_connector._build_where_clause(filters, {"price": "decimal"}, "and")
        assert "BETWEEN" in clause
        assert len(params) == 2


# ── _get_column_types ─────────────────────────────────────────────────────────

class TestGetColumnTypes:
    @pytest.mark.anyio
    async def test_fetches_from_db(self, mysql_connector):
        mysql_connector._mock_cursor.fetchall = AsyncMock(
            return_value=[
                {"COLUMN_NAME": "id", "DATA_TYPE": "int"},
                {"COLUMN_NAME": "name", "DATA_TYPE": "varchar"},
            ]
        )
        types = await mysql_connector._get_column_types("testdb.users")
        assert types == {"id": "int", "name": "varchar"}
        assert mysql_connector._column_types["testdb.users"] == types

    @pytest.mark.anyio
    async def test_uses_cache_on_second_call(self, mysql_connector):
        mysql_connector._column_types["testdb.users"] = {"id": "int"}
        types = await mysql_connector._get_column_types("testdb.users")
        assert types == {"id": "int"}
        mysql_connector._mock_cursor.fetchall.assert_not_called()


# ── execute_query ─────────────────────────────────────────────────────────────

class TestExecuteQuery:
    @pytest.mark.anyio
    async def test_basic_query_all_columns(self, mysql_connector):
        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(
            return_value=[{"id": 1, "name": "Alice"}]
        )
        result = await mysql_connector.execute_query(table="testdb.users")
        assert len(result) == 1
        captured_sql = mysql_connector._mock_cursor.execute.call_args[0][0]
        assert "SELECT *" in captured_sql
        assert "`testdb`.`users`" in captured_sql

    @pytest.mark.anyio
    async def test_column_selection(self, mysql_connector):
        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[])
        await mysql_connector.execute_query(table="testdb.users", columns=["id", "name"])
        sql = mysql_connector._mock_cursor.execute.call_args[0][0]
        assert "`id`, `name`" in sql

    @pytest.mark.anyio
    async def test_with_filter(self, mysql_connector):
        mysql_connector._column_types["testdb.users"] = {"status": "varchar"}
        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[])
        await mysql_connector.execute_query(
            table="testdb.users",
            filters=[{"column": "status", "operator": "eq", "value": "active"}],
        )
        sql = mysql_connector._mock_cursor.execute.call_args[0][0]
        assert "WHERE" in sql
        assert "`status` = %s" in sql

    @pytest.mark.anyio
    async def test_with_sort_asc(self, mysql_connector):
        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[])
        await mysql_connector.execute_query(
            table="testdb.users",
            sort=[{"column": "name", "direction": "asc"}],
        )
        sql = mysql_connector._mock_cursor.execute.call_args[0][0]
        assert "`name` ASC" in sql

    @pytest.mark.anyio
    async def test_with_sort_desc(self, mysql_connector):
        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[])
        await mysql_connector.execute_query(
            table="testdb.users",
            sort=[{"column": "created_at", "direction": "desc"}],
        )
        sql = mysql_connector._mock_cursor.execute.call_args[0][0]
        assert "`created_at` DESC" in sql

    @pytest.mark.anyio
    async def test_with_limit(self, mysql_connector):
        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[])
        await mysql_connector.execute_query(table="testdb.users", limit=25)
        sql = mysql_connector._mock_cursor.execute.call_args[0][0]
        assert "LIMIT 25" in sql


# ── get_schema ────────────────────────────────────────────────────────────────

class TestGetSchema:
    @pytest.mark.anyio
    async def test_groups_columns_by_table(self, mysql_connector):
        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[
            {
                "TABLE_NAME": "users",
                "COLUMN_NAME": "id",
                "DATA_TYPE": "int",
                "IS_NULLABLE": "NO",
                "is_primary_key": 1,
                "row_estimate": 100,
            },
            {
                "TABLE_NAME": "users",
                "COLUMN_NAME": "name",
                "DATA_TYPE": "varchar",
                "IS_NULLABLE": "YES",
                "is_primary_key": 0,
                "row_estimate": 100,
            },
        ])
        result = await mysql_connector.get_schema()
        assert len(result) == 1
        assert result[0]["name"] == "testdb.users"
        assert len(result[0]["columns"]) == 2

    @pytest.mark.anyio
    async def test_nullable_flag(self, mysql_connector):
        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[
            {
                "TABLE_NAME": "t",
                "COLUMN_NAME": "c",
                "DATA_TYPE": "varchar",
                "IS_NULLABLE": "YES",
                "is_primary_key": 0,
                "row_estimate": 0,
            },
        ])
        result = await mysql_connector.get_schema()
        assert result[0]["columns"][0]["nullable"] is True


# ── preview_table ─────────────────────────────────────────────────────────────

class TestPreviewTable:
    @pytest.mark.anyio
    async def test_returns_preview_dict(self, mysql_connector):
        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(
            return_value=[{"id": 1, "name": "Alice"}]
        )
        result = await mysql_connector.preview_table("testdb.users", limit=10)
        assert "columns" in result
        assert "rows" in result
        assert "total_sampled" in result
        assert result["total_sampled"] == 1

    @pytest.mark.anyio
    async def test_empty_table(self, mysql_connector):
        mysql_connector._mock_cursor.execute = AsyncMock()
        mysql_connector._mock_cursor.fetchall = AsyncMock(return_value=[])
        result = await mysql_connector.preview_table("testdb.users")
        assert result["columns"] == []
        assert result["rows"] == []


# ── connect / disconnect ──────────────────────────────────────────────────────

class TestConnectDisconnect:
    @pytest.mark.anyio
    async def test_connect_creates_pool(self):
        from easyweaver.connectors.implementations.mysql import MySQLConnector

        connector = MySQLConnector.__new__(MySQLConnector)
        connector.credentials = {
            "user": "u", "password": "p", "host": "h", "port": 3306, "database": "d"
        }
        connector._pool = None
        connector._column_types = {}
        connector._pk_cache = {}
        connector._temp_files = []

        mock_pool = MagicMock()

        with patch("aiomysql.create_pool", AsyncMock(return_value=mock_pool)):
            await connector.connect()

        assert connector._pool is mock_pool

    @pytest.mark.anyio
    async def test_disconnect_closes_pool(self, mysql_connector):
        mock_pool = mysql_connector._pool
        mock_pool.close = MagicMock()
        mock_pool.wait_closed = AsyncMock()
        await mysql_connector.disconnect()
        mock_pool.close.assert_called_once()
        assert mysql_connector._pool is None


# ── test_connection ───────────────────────────────────────────────────────────

class TestTestConnection:
    @pytest.mark.anyio
    async def test_success(self):
        from easyweaver.connectors.implementations.mysql import MySQLConnector

        connector = MySQLConnector.__new__(MySQLConnector)
        connector.credentials = {
            "user": "u", "password": "p", "host": "h", "port": 3306, "database": "d"
        }
        connector._pool = None
        connector._column_types = {}
        connector._pk_cache = {}
        connector._temp_files = []

        mock_cursor = AsyncMock()
        mock_cursor.execute = AsyncMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = _FakeCursorCM(mock_cursor)
        mock_pool = MagicMock()
        mock_pool.acquire.return_value = _FakeAsyncCM(mock_conn)
        mock_pool.close = MagicMock()
        mock_pool.wait_closed = AsyncMock()

        with patch("aiomysql.create_pool", AsyncMock(return_value=mock_pool)):
            result = await connector.test_connection()

        assert result["success"] is True

    @pytest.mark.anyio
    async def test_failure(self):
        from easyweaver.connectors.implementations.mysql import MySQLConnector

        connector = MySQLConnector.__new__(MySQLConnector)
        connector.credentials = {
            "user": "u", "password": "p", "host": "h", "port": 3306, "database": "d"
        }
        connector._pool = None
        connector._column_types = {}
        connector._pk_cache = {}
        connector._temp_files = []

        with patch("aiomysql.create_pool", AsyncMock(side_effect=Exception("Connection refused"))):
            result = await connector.test_connection()

        assert result["success"] is False
        assert "Connection refused" in result["message"]
