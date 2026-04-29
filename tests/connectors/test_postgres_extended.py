"""Extended tests for PostgresConnector — covering execute_query, get_schema,
preview_table, test_connection, _coerce_value, _build_where_clause edge cases,
_quote_ident, _qualified_table_name, connect/disconnect, _get_column_types."""

from __future__ import annotations

import base64
import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Shared async CM helper ──────────────────────────────────────────────────

class _FakeAsyncCM:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def pg_connector():
    from easyweaver.connectors.implementations.postgres import PostgresConnector

    connector = PostgresConnector.__new__(PostgresConnector)
    connector.credentials = {
        "user": "u", "password": "p", "host": "h", "port": 5432, "database": "d"
    }
    connector._column_types = {}
    connector._pk_cache = {}
    connector._temp_files = []

    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAsyncCM(mock_conn)
    connector._pool = mock_pool
    connector._mock_conn = mock_conn
    return connector


# ── _quote_ident ─────────────────────────────────────────────────────────────

class TestQuoteIdent:
    def test_simple_name(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector
        assert PostgresConnector._quote_ident("user") == '"user"'

    def test_name_with_spaces(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector
        assert PostgresConnector._quote_ident("my col") == '"my col"'

    def test_name_with_embedded_quotes(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector
        result = PostgresConnector._quote_ident('col"name')
        assert result == '"col""name"'

    def test_special_characters(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector
        result = PostgresConnector._quote_ident("my-column!")
        assert result == '"my-column!"'


# ── _qualified_table_name ────────────────────────────────────────────────────

class TestQualifiedTableName:
    def test_schema_dot_table(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector
        result = PostgresConnector._qualified_table_name("myschema.mytable")
        assert result == '"myschema"."mytable"'

    def test_no_schema_defaults_to_public(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector
        result = PostgresConnector._qualified_table_name("mytable")
        assert result == '"public"."mytable"'

    def test_special_chars_in_schema(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector
        result = PostgresConnector._qualified_table_name("my schema.my table")
        assert result == '"my schema"."my table"'


# ── _coerce_value ────────────────────────────────────────────────────────────

class TestCoerceValue:
    def test_none_returns_none(self, pg_connector):
        assert pg_connector._coerce_value(None, "text") is None

    def test_integer_type_coerces_string(self, pg_connector):
        assert pg_connector._coerce_value("42", "integer") == 42

    def test_integer_type_passthrough(self, pg_connector):
        assert pg_connector._coerce_value(5, "smallint") == 5

    def test_float_type_coerces(self, pg_connector):
        result = pg_connector._coerce_value("3.14", "double precision")
        assert abs(result - 3.14) < 0.001

    def test_float_type_from_int(self, pg_connector):
        assert pg_connector._coerce_value(3, "numeric") == 3.0

    def test_boolean_true_strings(self, pg_connector):
        for val in ("true", "1", "t", "yes", "True", "YES"):
            assert pg_connector._coerce_value(val, "boolean") is True

    def test_boolean_false_strings(self, pg_connector):
        for val in ("false", "0", "no", "False"):
            assert pg_connector._coerce_value(val, "boolean") is False

    def test_boolean_passthrough(self, pg_connector):
        assert pg_connector._coerce_value(True, "boolean") is True

    def test_text_passthrough(self, pg_connector):
        assert pg_connector._coerce_value("hello", "text") == "hello"

    def test_coerce_fails_gracefully(self, pg_connector):
        # "not_a_number" should come back unchanged when coercion fails
        result = pg_connector._coerce_value("not_a_number", "integer")
        assert result == "not_a_number"


# ── _build_where_clause edge cases ──────────────────────────────────────────

class TestBuildWhereClauseEdgeCases:
    def test_neq_filter(self, pg_connector):
        filters = [{"column": "status", "operator": "neq", "value": "inactive"}]
        clause, params, _ = pg_connector._build_where_clause(filters, {"status": "text"}, "and")
        assert '"status" != $1' in clause
        assert params == ["inactive"]

    def test_gt_filter(self, pg_connector):
        filters = [{"column": "age", "operator": "gt", "value": "18"}]
        clause, params, _ = pg_connector._build_where_clause(filters, {"age": "integer"}, "and")
        assert '"age" > $1' in clause
        assert params == [18]

    def test_lt_filter(self, pg_connector):
        filters = [{"column": "price", "operator": "lt", "value": "100"}]
        clause, params, _ = pg_connector._build_where_clause(filters, {"price": "numeric"}, "and")
        assert '"price" < $1' in clause

    def test_gte_filter(self, pg_connector):
        filters = [{"column": "score", "operator": "gte", "value": "50"}]
        clause, params, _ = pg_connector._build_where_clause(filters, {"score": "integer"}, "and")
        assert '"score" >= $1' in clause

    def test_lte_filter(self, pg_connector):
        filters = [{"column": "score", "operator": "lte", "value": "100"}]
        clause, params, _ = pg_connector._build_where_clause(filters, {"score": "integer"}, "and")
        assert '"score" <= $1' in clause

    def test_like_filter_wraps_value(self, pg_connector):
        filters = [{"column": "name", "operator": "like", "value": "john"}]
        clause, params, _ = pg_connector._build_where_clause(filters, {"name": "text"}, "and")
        assert "ILIKE $1" in clause
        assert params == ["%john%"]

    def test_is_null_filter(self, pg_connector):
        filters = [{"column": "deleted_at", "operator": "is_null"}]
        clause, params, _ = pg_connector._build_where_clause(filters, {}, "and")
        assert '"deleted_at" IS NULL' in clause
        assert params == []

    def test_is_not_null_filter(self, pg_connector):
        filters = [{"column": "name", "operator": "is_not_null"}]
        clause, params, _ = pg_connector._build_where_clause(filters, {}, "and")
        assert '"name" IS NOT NULL' in clause

    def test_in_empty_list_produces_false(self, pg_connector):
        filters = [{"column": "id", "operator": "in", "value": []}]
        clause, params, _ = pg_connector._build_where_clause(filters, {"id": "integer"}, "and")
        assert "FALSE" in clause
        assert params == []

    def test_in_comma_string_splits(self, pg_connector):
        filters = [{"column": "id", "operator": "in", "value": "1, 2, 3"}]
        clause, params, _ = pg_connector._build_where_clause(filters, {"id": "integer"}, "and")
        assert "IN" in clause
        assert len(params) == 3

    def test_not_in_filter(self, pg_connector):
        filters = [{"column": "status", "operator": "not_in", "value": ["a", "b"]}]
        clause, params, _ = pg_connector._build_where_clause(filters, {"status": "text"}, "and")
        assert "NOT IN" in clause
        assert len(params) == 2

    def test_not_in_empty_list_produces_true(self, pg_connector):
        filters = [{"column": "id", "operator": "not_in", "value": []}]
        clause, params, _ = pg_connector._build_where_clause(filters, {"id": "integer"}, "and")
        assert "TRUE" in clause

    def test_between_filter(self, pg_connector):
        filters = [{"column": "price", "operator": "between", "value": "10", "value2": "50"}]
        clause, params, _ = pg_connector._build_where_clause(filters, {"price": "numeric"}, "and")
        assert "BETWEEN" in clause
        assert len(params) == 2


# ── _get_column_types ────────────────────────────────────────────────────────

class TestGetColumnTypes:
    @pytest.mark.anyio
    async def test_fetches_from_db(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(
            return_value=[
                {"column_name": "id", "data_type": "integer"},
                {"column_name": "name", "data_type": "text"},
            ]
        )
        types = await pg_connector._get_column_types("public.users")
        assert types == {"id": "integer", "name": "text"}
        assert pg_connector._column_types["public.users"] == types

    @pytest.mark.anyio
    async def test_uses_cache_on_second_call(self, pg_connector):
        pg_connector._column_types["public.users"] = {"id": "integer"}
        types = await pg_connector._get_column_types("public.users")
        assert types == {"id": "integer"}
        # fetch should not be called since we have a cache hit
        pg_connector._mock_conn.fetch.assert_not_called()

    @pytest.mark.anyio
    async def test_no_schema_defaults_to_public(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[])
        await pg_connector._get_column_types("mytable")
        # Should have been called with 'public' as schema
        call_args = pg_connector._mock_conn.fetch.call_args
        assert call_args[0][1] == "public"
        assert call_args[0][2] == "mytable"


# ── execute_query ────────────────────────────────────────────────────────────

class TestExecuteQuery:
    @pytest.mark.anyio
    async def test_basic_query_all_columns(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(
            return_value=[{"id": 1, "name": "Alice"}]
        )
        result = await pg_connector.execute_query(table="public.users")
        assert result == [{"id": 1, "name": "Alice"}]
        sql = pg_connector._mock_conn.fetch.call_args[0][0]
        assert "SELECT *" in sql
        assert '"public"."users"' in sql

    @pytest.mark.anyio
    async def test_column_selection(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[])
        await pg_connector.execute_query(table="public.users", columns=["id", "name"])
        sql = pg_connector._mock_conn.fetch.call_args[0][0]
        assert '"id", "name"' in sql

    @pytest.mark.anyio
    async def test_with_filter(self, pg_connector):
        # Mock _get_column_types to return cached types
        pg_connector._column_types["public.users"] = {"status": "text"}
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[])
        await pg_connector.execute_query(
            table="public.users",
            filters=[{"column": "status", "operator": "eq", "value": "active"}],
        )
        sql = pg_connector._mock_conn.fetch.call_args[0][0]
        assert "WHERE" in sql
        assert '"status" = $1' in sql

    @pytest.mark.anyio
    async def test_with_sort_asc(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[])
        await pg_connector.execute_query(
            table="public.users",
            sort=[{"column": "name", "direction": "asc"}],
        )
        sql = pg_connector._mock_conn.fetch.call_args[0][0]
        assert '"name" ASC' in sql

    @pytest.mark.anyio
    async def test_with_sort_desc(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[])
        await pg_connector.execute_query(
            table="public.users",
            sort=[{"column": "created_at", "direction": "desc"}],
        )
        sql = pg_connector._mock_conn.fetch.call_args[0][0]
        assert '"created_at" DESC' in sql

    @pytest.mark.anyio
    async def test_with_limit(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[])
        await pg_connector.execute_query(table="public.users", limit=25)
        sql = pg_connector._mock_conn.fetch.call_args[0][0]
        assert "LIMIT 25" in sql

    @pytest.mark.anyio
    async def test_returns_list_of_dicts(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(
            return_value=[{"id": 1}, {"id": 2}]
        )
        result = await pg_connector.execute_query(table="public.t")
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)


# ── get_schema ───────────────────────────────────────────────────────────────

class TestGetSchema:
    @pytest.mark.anyio
    async def test_groups_columns_by_table(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[
            {
                "table_schema": "public",
                "table_name": "users",
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": "NO",
                "is_primary_key": True,
                "row_estimate": 100,
            },
            {
                "table_schema": "public",
                "table_name": "users",
                "column_name": "name",
                "data_type": "text",
                "is_nullable": "YES",
                "is_primary_key": False,
                "row_estimate": 100,
            },
        ])
        result = await pg_connector.get_schema()
        assert len(result) == 1
        table = result[0]
        assert table["name"] == "public.users"
        assert table["row_estimate"] == 100
        assert len(table["columns"]) == 2

    @pytest.mark.anyio
    async def test_multiple_tables(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[
            {
                "table_schema": "public", "table_name": "a",
                "column_name": "id", "data_type": "integer",
                "is_nullable": "NO", "is_primary_key": True, "row_estimate": 10,
            },
            {
                "table_schema": "public", "table_name": "b",
                "column_name": "id", "data_type": "integer",
                "is_nullable": "NO", "is_primary_key": True, "row_estimate": 5,
            },
        ])
        result = await pg_connector.get_schema()
        names = [t["name"] for t in result]
        assert "public.a" in names
        assert "public.b" in names

    @pytest.mark.anyio
    async def test_nullable_flag(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[
            {
                "table_schema": "s", "table_name": "t",
                "column_name": "c", "data_type": "text",
                "is_nullable": "YES", "is_primary_key": False, "row_estimate": 0,
            },
        ])
        result = await pg_connector.get_schema()
        assert result[0]["columns"][0]["nullable"] is True


# ── preview_table ─────────────────────────────────────────────────────────────

class TestPreviewTable:
    @pytest.mark.anyio
    async def test_returns_preview_dict(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(
            return_value=[{"id": 1, "name": "Alice"}]
        )
        result = await pg_connector.preview_table("public.users", limit=10)
        assert "columns" in result
        assert "rows" in result
        assert "total_sampled" in result
        assert result["total_sampled"] == 1

    @pytest.mark.anyio
    async def test_empty_table(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[])
        result = await pg_connector.preview_table("public.users")
        assert result["columns"] == []
        assert result["rows"] == []


# ── get_table_schema ──────────────────────────────────────────────────────────

class TestGetTableSchema:
    @pytest.mark.anyio
    async def test_returns_matching_table(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[
            {
                "table_schema": "public", "table_name": "users",
                "column_name": "id", "data_type": "integer",
                "is_nullable": "NO", "is_primary_key": True, "row_estimate": 50,
            },
        ])
        result = await pg_connector.get_table_schema("public.users")
        assert result["name"] == "public.users"

    @pytest.mark.anyio
    async def test_returns_empty_for_missing_table(self, pg_connector):
        pg_connector._mock_conn.fetch = AsyncMock(return_value=[])
        result = await pg_connector.get_table_schema("public.nonexistent")
        assert result["columns"] == []


# ── connect / disconnect ──────────────────────────────────────────────────────

class TestConnectDisconnect:
    @pytest.mark.anyio
    async def test_connect_creates_pool(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector

        connector = PostgresConnector.__new__(PostgresConnector)
        connector.credentials = {
            "user": "u", "password": "p", "host": "h", "port": 5432, "database": "d"
        }
        connector._pool = None
        connector._column_types = {}
        connector._pk_cache = {}
        connector._temp_files = []

        mock_pool = MagicMock()

        with patch("asyncpg.create_pool", AsyncMock(return_value=mock_pool)):
            await connector.connect()

        assert connector._pool is mock_pool

    @pytest.mark.anyio
    async def test_disconnect_closes_pool(self, pg_connector):
        mock_pool = pg_connector._pool
        mock_pool.close = AsyncMock()
        await pg_connector.disconnect()
        mock_pool.close.assert_called_once()
        assert pg_connector._pool is None

    @pytest.mark.anyio
    async def test_disconnect_noop_when_no_pool(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector

        connector = PostgresConnector.__new__(PostgresConnector)
        connector.credentials = {"user": "u", "password": "p", "host": "h", "port": 5432, "database": "d"}
        connector._pool = None
        connector._column_types = {}
        connector._pk_cache = {}
        connector._temp_files = []
        # Should not raise
        await connector.disconnect()


# ── test_connection ───────────────────────────────────────────────────────────

class TestTestConnection:
    @pytest.mark.anyio
    async def test_success(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector

        connector = PostgresConnector.__new__(PostgresConnector)
        connector.credentials = {
            "user": "u", "password": "p", "host": "h", "port": 5432, "database": "d"
        }
        connector._pool = None
        connector._column_types = {}
        connector._pk_cache = {}
        connector._temp_files = []

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_pool = MagicMock()
        mock_pool.acquire.return_value = _FakeAsyncCM(mock_conn)
        mock_pool.close = AsyncMock()

        with patch("asyncpg.create_pool", AsyncMock(return_value=mock_pool)):
            result = await connector.test_connection()

        assert result["success"] is True
        assert result["latency_ms"] is not None

    @pytest.mark.anyio
    async def test_failure(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector

        connector = PostgresConnector.__new__(PostgresConnector)
        connector.credentials = {
            "user": "u", "password": "p", "host": "h", "port": 5432, "database": "d"
        }
        connector._pool = None
        connector._column_types = {}
        connector._pk_cache = {}
        connector._temp_files = []

        with patch("asyncpg.create_pool", AsyncMock(side_effect=Exception("Connection refused"))):
            result = await connector.test_connection()

        assert result["success"] is False
        assert "Connection refused" in result["message"]


# ── SSL context building ──────────────────────────────────────────────────────

class TestBuildSSLContext:
    def test_disable_returns_none(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector

        connector = PostgresConnector.__new__(PostgresConnector)
        connector.credentials = {"ssl_mode": "disable"}
        connector._temp_files = []
        assert connector._build_ssl_context() is None

    def test_require_without_certs_no_hostname_check(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector

        connector = PostgresConnector.__new__(PostgresConnector)
        connector.credentials = {"ssl_mode": "require"}
        connector._temp_files = []
        ctx = connector._build_ssl_context()
        assert ctx is not None
        assert ctx.check_hostname is False

    def test_missing_ssl_mode_defaults_to_disable(self):
        from easyweaver.connectors.implementations.postgres import PostgresConnector

        connector = PostgresConnector.__new__(PostgresConnector)
        connector.credentials = {}
        connector._temp_files = []
        # Default ssl_mode is "disable"
        assert connector._build_ssl_context() is None
