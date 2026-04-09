"""Unit tests for MongoDBConnector.execute_query_batched."""

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


@pytest.fixture
def mongo_connector():
    """Create a MongoDBConnector with a mocked client."""
    from easyweaver.connectors.implementations.mongodb import MongoDBConnector

    connector = MongoDBConnector.__new__(MongoDBConnector)
    connector.credentials = {"database": "testdb"}
    connector._client = MagicMock()
    connector._db_name = "testdb"
    return connector


def _make_cursor(docs):
    """Create a mock async cursor that returns docs from to_list()."""
    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.skip = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
    cursor.to_list = AsyncMock(return_value=docs)
    return cursor


class TestSupportsBatching:
    def test_supports_batching_is_true(self, mongo_connector):
        assert mongo_connector.supports_batching is True


class TestExecuteQueryBatched:
    @pytest.mark.anyio
    async def test_sort_skip_limit_chain(self, mongo_connector):
        """Verify that sort(_id:1), skip(offset), limit(batch_size+1) are called."""
        mock_coll = MagicMock()
        mock_cursor = _make_cursor([])
        mock_coll.find.return_value = mock_cursor

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        rows, has_more, last_pk = await mongo_connector.execute_query_batched(
            table="orders", batch_size=100, offset=50
        )

        mock_coll.find.assert_called_once()
        mock_cursor.sort.assert_called_once_with("_id", 1)
        mock_cursor.skip.assert_called_once_with(50)
        mock_cursor.limit.assert_called_once_with(101)

    @pytest.mark.anyio
    async def test_keyset_pagination_no_skip(self, mongo_connector):
        """When last_key is provided, skip() should NOT be called."""
        mock_coll = MagicMock()
        mock_cursor = _make_cursor([])
        mock_coll.find.return_value = mock_cursor

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        rows, has_more, last_pk = await mongo_connector.execute_query_batched(
            table="orders", batch_size=100, offset=50, last_key="507f1f77bcf86cd799439011"
        )

        mock_cursor.sort.assert_called_once_with("_id", 1)
        mock_cursor.skip.assert_not_called()
        mock_cursor.limit.assert_called_once_with(101)

        # Verify the filter includes _id > condition
        call_args = mock_coll.find.call_args
        mongo_filter = call_args[0][0]
        assert "$gt" in str(mongo_filter)

    @pytest.mark.anyio
    async def test_has_more_true_when_extra_doc(self, mongo_connector):
        # Return 6 docs for batch_size=5
        docs = [{"_id": f"id{i}", "name": f"n{i}"} for i in range(6)]

        mock_coll = MagicMock()
        mock_cursor = _make_cursor(docs)
        mock_coll.find.return_value = mock_cursor

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        rows, has_more, last_pk = await mongo_connector.execute_query_batched(
            table="users", batch_size=5, offset=0
        )
        assert has_more is True
        assert len(rows) == 5
        assert last_pk == "id4"  # _id of last returned row

    @pytest.mark.anyio
    async def test_has_more_false_when_fewer(self, mongo_connector):
        docs = [{"_id": f"id{i}", "name": f"n{i}"} for i in range(3)]

        mock_coll = MagicMock()
        mock_cursor = _make_cursor(docs)
        mock_coll.find.return_value = mock_cursor

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        rows, has_more, last_pk = await mongo_connector.execute_query_batched(
            table="users", batch_size=5, offset=0
        )
        assert has_more is False
        assert len(rows) == 3
        assert last_pk == "id2"

    @pytest.mark.anyio
    async def test_returns_3_tuple(self, mongo_connector):
        mock_coll = MagicMock()
        mock_cursor = _make_cursor([])
        mock_coll.find.return_value = mock_cursor

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        result = await mongo_connector.execute_query_batched(
            table="empty_coll", batch_size=10, offset=0
        )
        assert len(result) == 3
        rows, has_more, last_pk = result
        assert rows == []
        assert has_more is False
        assert last_pk is None

    @pytest.mark.anyio
    async def test_filters_passed_to_find(self, mongo_connector):
        mock_coll = MagicMock()
        mock_cursor = _make_cursor([])
        mock_coll.find.return_value = mock_cursor

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        await mongo_connector.execute_query_batched(
            table="users",
            filters=[{"column": "status", "operator": "eq", "value": "active"}],
            batch_size=10,
        )

        call_args = mock_coll.find.call_args
        mongo_filter = call_args[0][0]
        assert "$and" in mongo_filter
        assert {"status": "active"} in mongo_filter["$and"]

    @pytest.mark.anyio
    async def test_column_projection(self, mongo_connector):
        mock_coll = MagicMock()
        mock_cursor = _make_cursor([])
        mock_coll.find.return_value = mock_cursor

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        await mongo_connector.execute_query_batched(
            table="users",
            columns=["name", "email"],
            batch_size=10,
        )

        call_args = mock_coll.find.call_args
        projection = call_args[0][1]
        assert projection == {"name": 1, "email": 1}

    @pytest.mark.anyio
    async def test_empty_collection(self, mongo_connector):
        mock_coll = MagicMock()
        mock_cursor = _make_cursor([])
        mock_coll.find.return_value = mock_cursor

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        rows, has_more, last_pk = await mongo_connector.execute_query_batched(
            table="empty_coll", batch_size=10, offset=0
        )
        assert rows == []
        assert has_more is False
        assert last_pk is None

    @pytest.mark.anyio
    async def test_or_filter_logic(self, mongo_connector):
        mock_coll = MagicMock()
        mock_cursor = _make_cursor([])
        mock_coll.find.return_value = mock_cursor

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        await mongo_connector.execute_query_batched(
            table="users",
            filters=[
                {"column": "status", "operator": "eq", "value": "active"},
                {"column": "role", "operator": "eq", "value": "admin"},
            ],
            filter_logic="or",
            batch_size=10,
        )

        call_args = mock_coll.find.call_args
        mongo_filter = call_args[0][0]
        assert "$or" in mongo_filter
