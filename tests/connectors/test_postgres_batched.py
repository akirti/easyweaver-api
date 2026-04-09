"""Unit tests for PostgresConnector.execute_query_batched."""

from contextlib import asynccontextmanager
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


@pytest.fixture
def pg_connector():
    """Create a PostgresConnector with a mocked pool."""
    from easyweaver.connectors.implementations.postgres import PostgresConnector

    connector = PostgresConnector.__new__(PostgresConnector)
    connector.credentials = {"user": "u", "password": "p", "host": "h", "port": 5432, "database": "d"}
    connector._column_types = {}
    connector._pk_cache = {}
    connector._temp_files = []

    # Build a mock pool whose .acquire() returns an async CM
    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value = _FakeAsyncCM(mock_conn)
    connector._pool = mock_pool
    connector._mock_conn = mock_conn  # expose for tests
    return connector


class TestSupportsBatching:
    def test_supports_batching_is_true(self, pg_connector):
        assert pg_connector.supports_batching is True


class TestGetPrimaryKey:
    @pytest.mark.anyio
    async def test_pk_detected_from_information_schema(self, pg_connector):
        pg_connector._mock_conn.fetchrow = AsyncMock(return_value={"column_name": "user_id"})

        pk = await pg_connector._get_primary_key("public.users")
        assert pk == "user_id"
        assert pg_connector._pk_cache["public.users"] == "user_id"

    @pytest.mark.anyio
    async def test_pk_cached_on_second_call(self, pg_connector):
        pg_connector._pk_cache["public.users"] = "id"
        pk = await pg_connector._get_primary_key("public.users")
        assert pk == "id"

    @pytest.mark.anyio
    async def test_fallback_to_first_column_when_no_pk(self, pg_connector):
        pg_connector._mock_conn.fetchrow = AsyncMock(
            side_effect=[None, {"column_name": "name"}]
        )

        pk = await pg_connector._get_primary_key("public.users")
        assert pk == "name"


class TestBuildWhereClause:
    def test_empty_filters(self, pg_connector):
        clause, params, next_idx = pg_connector._build_where_clause([], {}, "and")
        assert clause == ""
        assert params == []
        assert next_idx == 1

    def test_single_eq_filter(self, pg_connector):
        filters = [{"column": "status", "operator": "eq", "value": "active"}]
        clause, params, next_idx = pg_connector._build_where_clause(
            filters, {"status": "text"}, "and"
        )
        assert '"status" = $1' in clause
        assert params == ["active"]
        assert next_idx == 2

    def test_or_logic(self, pg_connector):
        filters = [
            {"column": "a", "operator": "eq", "value": "x"},
            {"column": "b", "operator": "eq", "value": "y"},
        ]
        clause, params, _ = pg_connector._build_where_clause(
            filters, {"a": "text", "b": "text"}, "or"
        )
        assert " OR " in clause

    def test_in_filter(self, pg_connector):
        filters = [{"column": "id", "operator": "in", "value": [1, 2, 3]}]
        clause, params, next_idx = pg_connector._build_where_clause(
            filters, {"id": "integer"}, "and"
        )
        assert "IN" in clause
        assert len(params) == 3
        assert next_idx == 4

    def test_start_idx(self, pg_connector):
        filters = [{"column": "x", "operator": "eq", "value": "v"}]
        clause, params, next_idx = pg_connector._build_where_clause(
            filters, {"x": "text"}, "and", start_idx=5
        )
        assert "$5" in clause
        assert next_idx == 6


class TestExecuteQueryBatched:
    @pytest.mark.anyio
    async def test_basic_query_no_offset(self, pg_connector):
        """First batch (no last_key, offset=0) should have no OFFSET clause."""
        pg_connector._pk_cache["public.orders"] = "order_id"

        captured = []

        async def mock_fetch(query, *params):
            captured.append(query)
            return []

        pg_connector._mock_conn.fetch = mock_fetch

        rows, has_more, last_pk = await pg_connector.execute_query_batched(
            table="public.orders", batch_size=100, offset=0
        )

        sql = captured[0]
        assert 'ORDER BY "order_id"' in sql
        assert "LIMIT 101" in sql
        assert "OFFSET" not in sql

    @pytest.mark.anyio
    async def test_offset_fallback(self, pg_connector):
        """When last_key is None and offset > 0, OFFSET is appended."""
        pg_connector._pk_cache["public.orders"] = "order_id"

        captured = []

        async def mock_fetch(query, *params):
            captured.append(query)
            return []

        pg_connector._mock_conn.fetch = mock_fetch

        rows, has_more, last_pk = await pg_connector.execute_query_batched(
            table="public.orders", batch_size=100, offset=200
        )

        sql = captured[0]
        assert "OFFSET 200" in sql

    @pytest.mark.anyio
    async def test_keyset_pagination_with_last_key(self, pg_connector):
        """When last_key is provided, WHERE pk > $N is used and no OFFSET."""
        pg_connector._pk_cache["public.orders"] = "order_id"

        captured_queries = []
        captured_params = []

        async def mock_fetch(query, *params):
            captured_queries.append(query)
            captured_params.append(params)
            return []

        pg_connector._mock_conn.fetch = mock_fetch

        rows, has_more, last_pk = await pg_connector.execute_query_batched(
            table="public.orders", batch_size=100, offset=200, last_key=42
        )

        sql = captured_queries[0]
        assert '"order_id" > $1' in sql
        assert "OFFSET" not in sql
        assert 42 in captured_params[0]

    @pytest.mark.anyio
    async def test_keyset_with_filters(self, pg_connector):
        """Keyset pagination combined with filters uses AND."""
        pg_connector._pk_cache["public.t"] = "id"
        pg_connector._column_types["public.t"] = {"status": "text", "id": "integer"}

        captured = []

        async def mock_fetch(query, *params):
            captured.append(query)
            return []

        pg_connector._mock_conn.fetch = mock_fetch

        await pg_connector.execute_query_batched(
            table="public.t",
            filters=[{"column": "status", "operator": "eq", "value": "active"}],
            batch_size=10,
            last_key=99,
        )

        sql = captured[0]
        assert "WHERE" in sql
        assert '"status" = $1' in sql
        assert '"id" > $2' in sql
        assert " AND " in sql

    @pytest.mark.anyio
    async def test_has_more_true_when_extra_row(self, pg_connector):
        pg_connector._pk_cache["public.t"] = "id"

        async def mock_fetch(query, *params):
            return [{"id": i, "name": f"n{i}"} for i in range(6)]

        pg_connector._mock_conn.fetch = mock_fetch

        rows, has_more, last_pk = await pg_connector.execute_query_batched(
            table="public.t", batch_size=5, offset=0
        )
        assert has_more is True
        assert len(rows) == 5
        assert last_pk == 4  # last row's id

    @pytest.mark.anyio
    async def test_has_more_false_when_exact_or_fewer(self, pg_connector):
        pg_connector._pk_cache["public.t"] = "id"

        async def mock_fetch(query, *params):
            return [{"id": i} for i in range(3)]

        pg_connector._mock_conn.fetch = mock_fetch

        rows, has_more, last_pk = await pg_connector.execute_query_batched(
            table="public.t", batch_size=5, offset=0
        )
        assert has_more is False
        assert len(rows) == 3
        assert last_pk == 2

    @pytest.mark.anyio
    async def test_returns_3_tuple(self, pg_connector):
        """Return value is always a 3-tuple (rows, has_more, last_key)."""
        pg_connector._pk_cache["public.t"] = "id"

        async def mock_fetch(query, *params):
            return []

        pg_connector._mock_conn.fetch = mock_fetch

        result = await pg_connector.execute_query_batched(
            table="public.t", batch_size=10, offset=0
        )
        assert len(result) == 3
        rows, has_more, last_pk = result
        assert rows == []
        assert has_more is False
        assert last_pk is None

    @pytest.mark.anyio
    async def test_filters_included_in_query(self, pg_connector):
        pg_connector._pk_cache["public.t"] = "id"
        pg_connector._column_types["public.t"] = {"status": "text", "id": "integer"}

        captured = []

        async def mock_fetch(query, *params):
            captured.append(query)
            return []

        pg_connector._mock_conn.fetch = mock_fetch

        await pg_connector.execute_query_batched(
            table="public.t",
            filters=[{"column": "status", "operator": "eq", "value": "active"}],
            batch_size=10,
        )

        sql = captured[0]
        assert "WHERE" in sql
        assert '"status" = $1' in sql
        assert 'ORDER BY "id"' in sql

    @pytest.mark.anyio
    async def test_column_selection(self, pg_connector):
        pg_connector._pk_cache["public.t"] = "id"

        captured = []

        async def mock_fetch(query, *params):
            captured.append(query)
            return []

        pg_connector._mock_conn.fetch = mock_fetch

        await pg_connector.execute_query_batched(
            table="public.t",
            columns=["name", "email"],
            batch_size=10,
        )

        sql = captured[0]
        assert '"name", "email"' in sql

    @pytest.mark.anyio
    async def test_empty_result(self, pg_connector):
        pg_connector._pk_cache["public.t"] = "id"

        async def mock_fetch(query, *params):
            return []

        pg_connector._mock_conn.fetch = mock_fetch

        rows, has_more, last_pk = await pg_connector.execute_query_batched(
            table="public.t", batch_size=10, offset=0
        )
        assert rows == []
        assert has_more is False
        assert last_pk is None

    @pytest.mark.anyio
    async def test_or_filter_logic(self, pg_connector):
        pg_connector._pk_cache["public.t"] = "id"
        pg_connector._column_types["public.t"] = {"a": "text", "b": "text"}

        captured = []

        async def mock_fetch(query, *params):
            captured.append(query)
            return []

        pg_connector._mock_conn.fetch = mock_fetch

        await pg_connector.execute_query_batched(
            table="public.t",
            filters=[
                {"column": "a", "operator": "eq", "value": "x"},
                {"column": "b", "operator": "eq", "value": "y"},
            ],
            filter_logic="or",
            batch_size=10,
        )

        sql = captured[0]
        assert " OR " in sql
