"""Extended tests for DB2Connector — covering execute_query, get_schema,
preview_table, _coerce_value, _build_where_clause edge cases,
_quote_ident, _qualified_table_name, _get_column_types."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock ibm_db_dbi before importing
mock_ibm_db_dbi = MagicMock()
sys.modules["ibm_db_dbi"] = mock_ibm_db_dbi


@pytest.fixture
def db2_connector():
    from easyweaver.connectors.implementations.db2 import DB2Connector

    connector = DB2Connector.__new__(DB2Connector)
    connector.credentials = {
        "user": "u", "password": "p", "host": "h", "port": 50000, "database": "mydb"
    }
    connector._conn = MagicMock()
    connector._column_types = {}
    connector._pk_cache = {}
    return connector


# ── _quote_ident ─────────────────────────────────────────────────────────────

class TestQuoteIdent:
    def test_simple_name(self):
        from easyweaver.connectors.implementations.db2 import DB2Connector
        assert DB2Connector._quote_ident("COLNAME") == '"COLNAME"'

    def test_embedded_quotes(self):
        from easyweaver.connectors.implementations.db2 import DB2Connector
        result = DB2Connector._quote_ident('COL"NAME')
        assert result == '"COL""NAME"'


# ── _qualified_table_name ────────────────────────────────────────────────────

class TestQualifiedTableName:
    def test_schema_dot_table(self):
        from easyweaver.connectors.implementations.db2 import DB2Connector
        result = DB2Connector._qualified_table_name("MYSCHEMA.MYTABLE")
        assert result == '"MYSCHEMA"."MYTABLE"'

    def test_no_schema(self):
        from easyweaver.connectors.implementations.db2 import DB2Connector
        result = DB2Connector._qualified_table_name("MYTABLE")
        assert result == '"MYTABLE"'


# ── _coerce_value ────────────────────────────────────────────────────────────

class TestCoerceValue:
    def test_none_returns_none(self, db2_connector):
        assert db2_connector._coerce_value(None, "varchar") is None

    def test_int_type_coerces_string(self, db2_connector):
        assert db2_connector._coerce_value("42", "integer") == 42

    def test_int_passthrough(self, db2_connector):
        assert db2_connector._coerce_value(5, "bigint") == 5

    def test_float_type_coerces(self, db2_connector):
        result = db2_connector._coerce_value("3.14", "decimal")
        assert abs(result - 3.14) < 0.001

    def test_boolean_true_string(self, db2_connector):
        assert db2_connector._coerce_value("true", "boolean") is True

    def test_boolean_false_string(self, db2_connector):
        assert db2_connector._coerce_value("false", "boolean") is False

    def test_text_passthrough(self, db2_connector):
        assert db2_connector._coerce_value("hello", "varchar") == "hello"


# ── _build_where_clause edge cases ──────────────────────────────────────────

class TestBuildWhereClauseEdgeCases:
    def test_neq_filter(self, db2_connector):
        filters = [{"column": "STATUS", "operator": "neq", "value": "inactive"}]
        clause, params = db2_connector._build_where_clause(filters, {"STATUS": "varchar"}, "and")
        assert '"STATUS" != ?' in clause
        assert params == ["inactive"]

    def test_gt_filter(self, db2_connector):
        filters = [{"column": "AGE", "operator": "gt", "value": "18"}]
        clause, params = db2_connector._build_where_clause(filters, {"AGE": "integer"}, "and")
        assert '"AGE" > ?' in clause
        assert params == [18]

    def test_lt_filter(self, db2_connector):
        filters = [{"column": "PRICE", "operator": "lt", "value": "100"}]
        clause, params = db2_connector._build_where_clause(filters, {"PRICE": "decimal"}, "and")
        assert '"PRICE" < ?' in clause

    def test_gte_filter(self, db2_connector):
        filters = [{"column": "SCORE", "operator": "gte", "value": "50"}]
        clause, params = db2_connector._build_where_clause(filters, {"SCORE": "integer"}, "and")
        assert '"SCORE" >= ?' in clause

    def test_lte_filter(self, db2_connector):
        filters = [{"column": "SCORE", "operator": "lte", "value": "100"}]
        clause, params = db2_connector._build_where_clause(filters, {"SCORE": "integer"}, "and")
        assert '"SCORE" <= ?' in clause

    def test_like_filter(self, db2_connector):
        filters = [{"column": "NAME", "operator": "like", "value": "john"}]
        clause, params = db2_connector._build_where_clause(filters, {"NAME": "varchar"}, "and")
        assert "LIKE ?" in clause
        assert params == ["%john%"]

    def test_is_null_filter(self, db2_connector):
        filters = [{"column": "DELETED_AT", "operator": "is_null"}]
        clause, params = db2_connector._build_where_clause(filters, {}, "and")
        assert '"DELETED_AT" IS NULL' in clause
        assert params == []

    def test_is_not_null_filter(self, db2_connector):
        filters = [{"column": "NAME", "operator": "is_not_null"}]
        clause, params = db2_connector._build_where_clause(filters, {}, "and")
        assert '"NAME" IS NOT NULL' in clause

    def test_in_filter(self, db2_connector):
        filters = [{"column": "ID", "operator": "in", "value": [1, 2, 3]}]
        clause, params = db2_connector._build_where_clause(filters, {"ID": "integer"}, "and")
        assert "IN" in clause
        assert len(params) == 3

    def test_in_empty_list_produces_1_0(self, db2_connector):
        filters = [{"column": "ID", "operator": "in", "value": []}]
        clause, params = db2_connector._build_where_clause(filters, {"ID": "integer"}, "and")
        assert "1=0" in clause

    def test_in_comma_string_splits(self, db2_connector):
        filters = [{"column": "ID", "operator": "in", "value": "1, 2, 3"}]
        clause, params = db2_connector._build_where_clause(filters, {"ID": "integer"}, "and")
        assert len(params) == 3

    def test_not_in_filter(self, db2_connector):
        filters = [{"column": "STATUS", "operator": "not_in", "value": ["a", "b"]}]
        clause, params = db2_connector._build_where_clause(filters, {"STATUS": "varchar"}, "and")
        assert "NOT IN" in clause
        assert len(params) == 2

    def test_not_in_empty_list_produces_1_1(self, db2_connector):
        filters = [{"column": "ID", "operator": "not_in", "value": []}]
        clause, params = db2_connector._build_where_clause(filters, {"ID": "integer"}, "and")
        assert "1=1" in clause

    def test_between_filter(self, db2_connector):
        filters = [{"column": "PRICE", "operator": "between", "value": "10", "value2": "50"}]
        clause, params = db2_connector._build_where_clause(filters, {"PRICE": "decimal"}, "and")
        assert "BETWEEN" in clause
        assert len(params) == 2

    def test_col_prefix_applied(self, db2_connector):
        filters = [{"column": "STATUS", "operator": "eq", "value": "active"}]
        clause, params = db2_connector._build_where_clause(
            filters, {"STATUS": "varchar"}, "and", col_prefix="t."
        )
        assert 't."STATUS" = ?' in clause


# ── _get_column_types ─────────────────────────────────────────────────────────

class TestGetColumnTypes:
    @pytest.mark.anyio
    async def test_fetches_from_db(self, db2_connector):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("ID", "INTEGER"),
            ("NAME", "VARCHAR"),
        ]
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        types = await db2_connector._get_column_types("MYSCHEMA.USERS")
        assert types == {"ID": "integer", "NAME": "varchar"}
        assert db2_connector._column_types["MYSCHEMA.USERS"] == types

    @pytest.mark.anyio
    async def test_uses_cache(self, db2_connector):
        db2_connector._column_types["MYSCHEMA.USERS"] = {"ID": "integer"}
        types = await db2_connector._get_column_types("MYSCHEMA.USERS")
        assert types == {"ID": "integer"}

    @pytest.mark.anyio
    async def test_no_schema_uses_current(self, db2_connector):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        await db2_connector._get_column_types("MYTABLE")
        # Should not raise; uses CURRENT SCHEMA path
        assert "MYTABLE" in db2_connector._column_types


# ── execute_query ─────────────────────────────────────────────────────────────

class TestExecuteQuery:
    @pytest.mark.anyio
    async def test_basic_query(self, db2_connector):
        mock_cursor = MagicMock()
        mock_cursor.execute = MagicMock()
        mock_cursor.description = [("ID ",), ("NAME ",)]
        mock_cursor.fetchall.return_value = [(1, "Alice"), (2, "Bob")]
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        result = await db2_connector.execute_query(table="MYSCHEMA.USERS")
        assert len(result) == 2
        assert result[0]["ID"] == 1
        assert result[0]["NAME"] == "Alice"

    @pytest.mark.anyio
    async def test_column_selection(self, db2_connector):
        mock_cursor = MagicMock()
        captured_queries = []

        def mock_execute(query, params=None):
            captured_queries.append(query)

        mock_cursor.execute = mock_execute
        mock_cursor.description = [("NAME ",)]
        mock_cursor.fetchall.return_value = [("Alice",)]
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        result = await db2_connector.execute_query(
            table="MYSCHEMA.USERS", columns=["NAME"]
        )
        sql = captured_queries[0]
        assert '"NAME"' in sql

    @pytest.mark.anyio
    async def test_with_sort(self, db2_connector):
        mock_cursor = MagicMock()
        captured_queries = []

        def mock_execute(query, params=None):
            captured_queries.append(query)

        mock_cursor.execute = mock_execute
        mock_cursor.description = [("ID ",)]
        mock_cursor.fetchall.return_value = []
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        await db2_connector.execute_query(
            table="MYSCHEMA.USERS",
            sort=[{"column": "NAME", "direction": "asc"}],
        )
        sql = captured_queries[0]
        assert '"NAME" ASC' in sql

    @pytest.mark.anyio
    async def test_with_limit_uses_fetch_first(self, db2_connector):
        mock_cursor = MagicMock()
        captured_queries = []

        def mock_execute(query, params=None):
            captured_queries.append(query)

        mock_cursor.execute = mock_execute
        mock_cursor.description = [("ID ",)]
        mock_cursor.fetchall.return_value = []
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        await db2_connector.execute_query(table="MYSCHEMA.USERS", limit=10)
        sql = captured_queries[0]
        assert "FETCH FIRST 10 ROWS ONLY" in sql


# ── get_schema ────────────────────────────────────────────────────────────────

class TestGetSchema:
    @pytest.mark.anyio
    async def test_groups_columns_by_table(self, db2_connector):
        """get_schema uses asyncio.to_thread; the inner function builds dicts via description.
        Columns come from description[0] entries (with trailing space, matching real DB2 driver).
        """
        mock_cursor = MagicMock()
        mock_cursor.execute = MagicMock()
        # DB2 driver returns column names with trailing space — the code does desc[0]
        mock_cursor.description = [
            ("TABSCHEMA",), ("TABNAME",), ("COLNAME",), ("TYPENAME",),
            ("NULLS",), ("KEYSEQ",), ("CARD",),
        ]
        mock_cursor.fetchall.return_value = [
            ("MYSCHEMA ", "USERS ", "ID ", "INTEGER ", "N", 1, 100),
            ("MYSCHEMA ", "USERS ", "NAME ", "VARCHAR ", "Y", None, 100),
        ]
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        result = await db2_connector.get_schema()
        assert len(result) == 1
        assert result[0]["name"] == "MYSCHEMA.USERS"
        assert len(result[0]["columns"]) == 2

    @pytest.mark.anyio
    async def test_primary_key_detected(self, db2_connector):
        mock_cursor = MagicMock()
        mock_cursor.execute = MagicMock()
        mock_cursor.description = [
            ("TABSCHEMA",), ("TABNAME",), ("COLNAME",), ("TYPENAME",),
            ("NULLS",), ("KEYSEQ",), ("CARD",),
        ]
        mock_cursor.fetchall.return_value = [
            ("S ", "T ", "ID ", "INTEGER ", "N", 1, 50),
        ]
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        result = await db2_connector.get_schema()
        col = result[0]["columns"][0]
        assert col["primary_key"] is True


# ── preview_table ─────────────────────────────────────────────────────────────

class TestPreviewTable:
    @pytest.mark.anyio
    async def test_returns_preview(self, db2_connector):
        mock_cursor = MagicMock()
        mock_cursor.execute = MagicMock()
        mock_cursor.description = [("ID ",), ("NAME ",)]
        mock_cursor.fetchall.return_value = [(1, "Alice"), (2, "Bob")]
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        result = await db2_connector.preview_table("MYSCHEMA.USERS", limit=10)
        assert "rows" in result
        assert len(result["rows"]) == 2
        assert result["total_sampled"] == 2

    @pytest.mark.anyio
    async def test_uses_fetch_first(self, db2_connector):
        mock_cursor = MagicMock()
        captured_queries = []

        def mock_execute(query, params=None):
            captured_queries.append(query)

        mock_cursor.execute = mock_execute
        mock_cursor.description = [("ID ",)]
        mock_cursor.fetchall.return_value = []
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        await db2_connector.preview_table("MYSCHEMA.USERS", limit=5)
        sql = captured_queries[0]
        assert "FETCH FIRST 5 ROWS ONLY" in sql


# ── connection string ─────────────────────────────────────────────────────────

class TestConnectionString:
    def test_builds_correct_dsn(self, db2_connector):
        cs = db2_connector._connection_string()
        assert "DATABASE=mydb" in cs
        assert "HOSTNAME=h" in cs
        assert "PORT=50000" in cs
        assert "UID=u" in cs
        assert "PWD=p" in cs
