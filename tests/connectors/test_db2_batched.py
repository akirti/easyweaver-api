"""Unit tests for DB2Connector.execute_query_batched."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock ibm_db_dbi before importing the connector
mock_ibm_db_dbi = MagicMock()
sys.modules["ibm_db_dbi"] = mock_ibm_db_dbi


@pytest.fixture
def db2_connector():
    """Create a DB2Connector with a mocked connection."""
    # Re-import to pick up the mock
    from easyweaver.connectors.implementations.db2 import DB2Connector

    connector = DB2Connector.__new__(DB2Connector)
    connector.credentials = {"user": "u", "password": "p", "host": "h", "port": 50000, "database": "d"}
    connector._conn = MagicMock()
    connector._column_types = {}
    connector._pk_cache = {}
    return connector


class TestSupportsBatching:
    def test_supports_batching_is_true(self, db2_connector):
        assert db2_connector.supports_batching is True


class TestGetPrimaryKey:
    @pytest.mark.anyio
    async def test_pk_detected_from_syscat(self, db2_connector):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("ORDER_ID ",)
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        pk = await db2_connector._get_primary_key("MYSCHEMA.ORDERS")
        assert pk == "ORDER_ID"
        assert db2_connector._pk_cache["MYSCHEMA.ORDERS"] == "ORDER_ID"

    @pytest.mark.anyio
    async def test_pk_cached(self, db2_connector):
        db2_connector._pk_cache["MYSCHEMA.T"] = "ID"
        pk = await db2_connector._get_primary_key("MYSCHEMA.T")
        assert pk == "ID"

    @pytest.mark.anyio
    async def test_fallback_to_first_column(self, db2_connector):
        call_count = [0]

        def mock_cursor_factory():
            call_count[0] += 1
            cursor = MagicMock()
            if call_count[0] == 1:
                cursor.fetchone.return_value = None  # no PK
            else:
                cursor.fetchone.return_value = ("CREATED_AT ",)
            return cursor

        db2_connector._conn.cursor = mock_cursor_factory

        pk = await db2_connector._get_primary_key("MYSCHEMA.T")
        assert pk == "CREATED_AT"


class TestBuildWhereClause:
    def test_empty_filters(self, db2_connector):
        clause, params = db2_connector._build_where_clause([], {}, "and")
        assert clause == ""
        assert params == []

    def test_single_eq_filter_no_prefix(self, db2_connector):
        filters = [{"column": "STATUS", "operator": "eq", "value": "active"}]
        clause, params = db2_connector._build_where_clause(
            filters, {"STATUS": "varchar"}, "and"
        )
        assert '"STATUS" = ?' in clause
        assert params == ["active"]

    def test_single_eq_filter_with_prefix(self, db2_connector):
        filters = [{"column": "STATUS", "operator": "eq", "value": "active"}]
        clause, params = db2_connector._build_where_clause(
            filters, {"STATUS": "varchar"}, "and", col_prefix="t."
        )
        assert 't."STATUS" = ?' in clause
        assert params == ["active"]

    def test_or_logic(self, db2_connector):
        filters = [
            {"column": "A", "operator": "eq", "value": "x"},
            {"column": "B", "operator": "eq", "value": "y"},
        ]
        clause, params = db2_connector._build_where_clause(
            filters, {"A": "varchar", "B": "varchar"}, "or", col_prefix="t."
        )
        assert " OR " in clause


class TestExecuteQueryBatched:
    @pytest.mark.anyio
    async def test_row_number_sql_pattern(self, db2_connector):
        db2_connector._pk_cache["MYSCHEMA.T"] = "ID"

        captured_queries = []
        mock_cursor = MagicMock()

        def mock_execute(query, params=None):
            captured_queries.append(query)

        mock_cursor.execute = mock_execute
        mock_cursor.description = [("ID ",), ("NAME ",)]
        mock_cursor.fetchall.return_value = []
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        rows, has_more, last_pk = await db2_connector.execute_query_batched(
            table="MYSCHEMA.T", batch_size=100, offset=200
        )

        sql = captured_queries[0]
        assert "ROW_NUMBER()" in sql
        assert 'ORDER BY t."ID"' in sql
        assert "rn__ >= 201" in sql
        assert "rn__ <= 301" in sql

    @pytest.mark.anyio
    async def test_has_more_true(self, db2_connector):
        db2_connector._pk_cache["MYSCHEMA.T"] = "ID"

        mock_cursor = MagicMock()
        mock_cursor.execute = MagicMock()
        mock_cursor.description = [("ID ",), ("NAME ",), ("rn__ ",)]
        # 6 rows for batch_size=5
        mock_cursor.fetchall.return_value = [
            (i, f"n{i}", i + 1) for i in range(6)
        ]
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        rows, has_more, last_pk = await db2_connector.execute_query_batched(
            table="MYSCHEMA.T", batch_size=5, offset=0
        )
        assert has_more is True
        assert len(rows) == 5
        # rn__ column should be stripped
        assert "rn__" not in rows[0]
        assert last_pk == 4  # last row's ID value

    @pytest.mark.anyio
    async def test_has_more_false(self, db2_connector):
        db2_connector._pk_cache["MYSCHEMA.T"] = "ID"

        mock_cursor = MagicMock()
        mock_cursor.execute = MagicMock()
        mock_cursor.description = [("ID ",), ("rn__ ",)]
        mock_cursor.fetchall.return_value = [(i, i + 1) for i in range(3)]
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        rows, has_more, last_pk = await db2_connector.execute_query_batched(
            table="MYSCHEMA.T", batch_size=5, offset=0
        )
        assert has_more is False
        assert len(rows) == 3
        assert last_pk == 2

    @pytest.mark.anyio
    async def test_rn_column_stripped_from_results(self, db2_connector):
        db2_connector._pk_cache["MYSCHEMA.T"] = "ID"

        mock_cursor = MagicMock()
        mock_cursor.execute = MagicMock()
        mock_cursor.description = [("ID ",), ("NAME ",), ("rn__ ",)]
        mock_cursor.fetchall.return_value = [(1, "alice", 1)]
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        rows, _, last_pk = await db2_connector.execute_query_batched(
            table="MYSCHEMA.T", batch_size=10, offset=0
        )
        assert rows == [{"ID": 1, "NAME": "alice"}]

    @pytest.mark.anyio
    async def test_returns_3_tuple(self, db2_connector):
        db2_connector._pk_cache["MYSCHEMA.T"] = "ID"

        mock_cursor = MagicMock()
        mock_cursor.execute = MagicMock()
        mock_cursor.description = [("ID ",)]
        mock_cursor.fetchall.return_value = []
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        result = await db2_connector.execute_query_batched(
            table="MYSCHEMA.T", batch_size=10, offset=0
        )
        assert len(result) == 3
        rows, has_more, last_pk = result
        assert rows == []
        assert has_more is False
        assert last_pk is None

    @pytest.mark.anyio
    async def test_filters_in_where_clause(self, db2_connector):
        db2_connector._pk_cache["MYSCHEMA.T"] = "ID"
        db2_connector._column_types["MYSCHEMA.T"] = {"STATUS": "varchar"}

        captured_queries = []
        mock_cursor = MagicMock()

        def mock_execute(query, params=None):
            captured_queries.append(query)

        mock_cursor.execute = mock_execute
        mock_cursor.description = [("ID ",)]
        mock_cursor.fetchall.return_value = []
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        await db2_connector.execute_query_batched(
            table="MYSCHEMA.T",
            filters=[{"column": "STATUS", "operator": "eq", "value": "active"}],
            batch_size=10,
        )

        sql = captured_queries[0]
        assert "WHERE" in sql
        assert 't."STATUS" = ?' in sql

    @pytest.mark.anyio
    async def test_column_selection_with_t_prefix(self, db2_connector):
        db2_connector._pk_cache["MYSCHEMA.T"] = "ID"

        captured_queries = []
        mock_cursor = MagicMock()

        def mock_execute(query, params=None):
            captured_queries.append(query)

        mock_cursor.execute = mock_execute
        mock_cursor.description = [("NAME ",), ("rn__ ",)]
        mock_cursor.fetchall.return_value = []
        mock_cursor.close = MagicMock()
        db2_connector._conn.cursor.return_value = mock_cursor

        await db2_connector.execute_query_batched(
            table="MYSCHEMA.T",
            columns=["NAME", "EMAIL"],
            batch_size=10,
        )

        sql = captured_queries[0]
        assert 't."NAME"' in sql
        assert 't."EMAIL"' in sql
