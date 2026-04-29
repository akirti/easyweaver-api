"""Extended tests for MongoDBConnector — covering execute_query, get_schema,
preview_table, test_connection, _build_filter, _flatten_doc, _infer_columns,
_make_serializable, _uri, connect/disconnect, get_table_schema."""

from __future__ import annotations

from datetime import datetime, date, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bson import ObjectId


@pytest.fixture
def mongo_connector():
    from easyweaver.connectors.implementations.mongodb import MongoDBConnector

    connector = MongoDBConnector.__new__(MongoDBConnector)
    connector.credentials = {
        "host": "localhost", "port": 27017, "database": "testdb"
    }
    connector._db_name = "testdb"

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)
    mock_client.__getitem__ = MagicMock(return_value=mock_db)
    connector._client = mock_client
    connector._mock_db = mock_db
    connector._mock_collection = mock_collection
    return connector


# ── _uri ──────────────────────────────────────────────────────────────────────

class TestUri:
    def test_basic_uri_with_user_password(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        connector = MongoDBConnector.__new__(MongoDBConnector)
        connector.credentials = {
            "host": "myhost", "port": 27017, "user": "admin", "password": "secret",
            "database": "testdb", "auth_database": "admin",
        }
        connector._db_name = "testdb"
        uri = connector._uri()
        assert "admin:secret@myhost" in uri
        assert "27017" in uri

    def test_uri_without_credentials(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        connector = MongoDBConnector.__new__(MongoDBConnector)
        connector.credentials = {"host": "myhost", "database": "testdb"}
        connector._db_name = "testdb"
        uri = connector._uri()
        assert "myhost" in uri
        assert "@" not in uri

    def test_connection_string_passthrough(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        connector = MongoDBConnector.__new__(MongoDBConnector)
        connector.credentials = {
            "connection_string": "mongodb://user:pass@cluster.example.com/mydb",
            "database": "mydb",
        }
        connector._db_name = "mydb"
        uri = connector._uri()
        assert uri == "mongodb://user:pass@cluster.example.com/mydb"

    def test_srv_scheme(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        connector = MongoDBConnector.__new__(MongoDBConnector)
        connector.credentials = {
            "scheme": "mongodb+srv",
            "host": "cluster0.example.com",
            "user": "admin", "password": "secret",
            "database": "testdb", "auth_database": "admin",
        }
        connector._db_name = "testdb"
        uri = connector._uri()
        assert uri.startswith("mongodb+srv://")

    def test_strips_protocol_prefix_from_host(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        connector = MongoDBConnector.__new__(MongoDBConnector)
        connector.credentials = {
            "host": "mongodb://myhost",
            "database": "testdb",
        }
        connector._db_name = "testdb"
        uri = connector._uri()
        # Should not double the protocol
        assert "mongodb://mongodb://" not in uri


# ── _build_filter ─────────────────────────────────────────────────────────────

class TestBuildFilter:
    def test_eq_filter(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "status", "operator": "eq", "value": "active"}]
        )
        assert result == {"$and": [{"status": "active"}]}

    def test_neq_filter(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "status", "operator": "neq", "value": "inactive"}]
        )
        assert result == {"$and": [{"status": {"$ne": "inactive"}}]}

    def test_gt_filter(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "age", "operator": "gt", "value": 18}]
        )
        assert result == {"$and": [{"age": {"$gt": 18}}]}

    def test_lt_filter(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "age", "operator": "lt", "value": 65}]
        )
        assert result == {"$and": [{"age": {"$lt": 65}}]}

    def test_gte_filter(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "score", "operator": "gte", "value": 50}]
        )
        assert result == {"$and": [{"score": {"$gte": 50}}]}

    def test_lte_filter(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "score", "operator": "lte", "value": 100}]
        )
        assert result == {"$and": [{"score": {"$lte": 100}}]}

    def test_like_filter_uses_regex(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "name", "operator": "like", "value": "john"}]
        )
        cond = result["$and"][0]
        assert "$regex" in cond["name"]
        assert "$options" in cond["name"]
        assert cond["name"]["$options"] == "i"

    def test_is_null_filter(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "deleted_at", "operator": "is_null"}]
        )
        cond = result["$and"][0]
        assert "$or" in cond

    def test_is_not_null_filter(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "name", "operator": "is_not_null"}]
        )
        assert result == {"$and": [{"name": {"$exists": True, "$ne": None}}]}

    def test_in_filter(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "status", "operator": "in", "value": ["a", "b"]}]
        )
        assert result == {"$and": [{"status": {"$in": ["a", "b"]}}]}

    def test_in_filter_comma_string(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "status", "operator": "in", "value": "a, b, c"}]
        )
        cond = result["$and"][0]
        assert "$in" in cond["status"]
        assert len(cond["status"]["$in"]) == 3

    def test_not_in_filter(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "status", "operator": "not_in", "value": ["x", "y"]}]
        )
        assert result == {"$and": [{"status": {"$nin": ["x", "y"]}}]}

    def test_between_filter(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "price", "operator": "between", "value": 10, "value2": 50}]
        )
        cond = result["$and"][0]["price"]
        assert cond["$gte"] == 10
        assert cond["$lte"] == 50

    def test_or_logic(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [
                {"column": "a", "operator": "eq", "value": "x"},
                {"column": "b", "operator": "eq", "value": "y"},
            ],
            logic="or",
        )
        assert "$or" in result

    def test_empty_filters_returns_empty_dict(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter([])
        assert result == {}

    def test_id_string_coerced_to_objectid(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        oid_str = "507f1f77bcf86cd799439011"
        result = MongoDBConnector._build_filter(
            [{"column": "_id", "operator": "eq", "value": oid_str}]
        )
        cond = result["$and"][0]["_id"]
        assert isinstance(cond, ObjectId)

    def test_nested_col_dot_notation(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._build_filter(
            [{"column": "address.city", "operator": "eq", "value": "NYC"}]
        )
        assert result == {"$and": [{"address.city": "NYC"}]}


# ── _col_to_mongo_path ────────────────────────────────────────────────────────

class TestColToMongoPath:
    def test_removes_numeric_index(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        assert MongoDBConnector._col_to_mongo_path("addresses[0].line1") == "addresses.line1"

    def test_removes_empty_index(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        assert MongoDBConnector._col_to_mongo_path("addresses[].line1") == "addresses.line1"

    def test_plain_column_unchanged(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        assert MongoDBConnector._col_to_mongo_path("name") == "name"

    def test_dot_notation_unchanged(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        assert MongoDBConnector._col_to_mongo_path("user.name") == "user.name"


# ── _flatten_doc ──────────────────────────────────────────────────────────────

class TestFlattenDoc:
    def test_flat_doc_unchanged(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        doc = {"name": "Alice", "age": 30}
        result = MongoDBConnector._flatten_doc(doc)
        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_nested_dict_flattened(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        doc = {"address": {"city": "NYC", "zip": "10001"}}
        result = MongoDBConnector._flatten_doc(doc)
        assert result["address.city"] == "NYC"
        assert result["address.zip"] == "10001"

    def test_array_of_dicts_uses_indexed_keys(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        doc = {"items": [{"name": "a"}, {"name": "b"}]}
        result = MongoDBConnector._flatten_doc(doc)
        assert result["items[0].name"] == "a"
        assert result["items[1].name"] == "b"

    def test_array_of_primitives_joined(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        doc = {"tags": ["x", "y", "z"]}
        result = MongoDBConnector._flatten_doc(doc)
        assert result["tags"] == "x, y, z"

    def test_id_converted_to_string(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        oid = ObjectId("507f1f77bcf86cd799439011")
        doc = {"_id": oid}
        result = MongoDBConnector._flatten_doc(doc)
        assert result["_id"] == str(oid)


# ── _make_serializable ────────────────────────────────────────────────────────

class TestMakeSerializable:
    def test_objectid_to_str(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        oid = ObjectId("507f1f77bcf86cd799439011")
        assert MongoDBConnector._make_serializable(oid) == str(oid)

    def test_datetime_to_isoformat(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = MongoDBConnector._make_serializable(dt)
        assert isinstance(result, str)
        assert "2024-01-15" in result

    def test_date_to_isoformat(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        d = date(2024, 1, 15)
        result = MongoDBConnector._make_serializable(d)
        assert result == "2024-01-15"

    def test_dict_to_json_string(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        result = MongoDBConnector._make_serializable({"key": "val"})
        import json
        assert json.loads(result) == {"key": "val"}

    def test_plain_value_passthrough(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        assert MongoDBConnector._make_serializable("hello") == "hello"
        assert MongoDBConnector._make_serializable(42) == 42


# ── _infer_columns ────────────────────────────────────────────────────────────

class TestInferColumns:
    def test_infers_types_from_docs(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        docs = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        columns = MongoDBConnector._infer_columns(docs)
        col_map = {c["name"]: c["type"] for c in columns}
        assert col_map["name"] in ("string", "str")
        assert col_map["age"] in ("integer", "int")

    def test_excludes_id_field(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        docs = [{"_id": ObjectId(), "name": "Alice"}]
        columns = MongoDBConnector._infer_columns(docs)
        names = [c["name"] for c in columns]
        assert "_id" not in names

    def test_mixed_types_marked_mixed(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        docs = [{"field": "text"}, {"field": 42}]
        columns = MongoDBConnector._infer_columns(docs)
        col = next(c for c in columns if c["name"] == "field")
        assert col["type"] == "mixed"

    def test_empty_docs_returns_empty_list(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        assert MongoDBConnector._infer_columns([]) == []


# ── execute_query ─────────────────────────────────────────────────────────────

class TestExecuteQuery:
    def _make_cursor(self, docs):
        cursor = MagicMock()
        cursor.sort = MagicMock(return_value=cursor)
        cursor.limit = MagicMock(return_value=cursor)
        cursor.to_list = AsyncMock(return_value=docs)
        return cursor

    @pytest.mark.anyio
    async def test_basic_query(self, mongo_connector):
        docs = [{"_id": "id1", "name": "Alice"}]
        cursor = self._make_cursor(docs)
        mongo_connector._mock_collection.find.return_value = cursor

        result = await mongo_connector.execute_query(table="users")
        assert len(result) == 1
        assert result[0]["name"] == "Alice"

    @pytest.mark.anyio
    async def test_with_projection(self, mongo_connector):
        cursor = self._make_cursor([])
        mongo_connector._mock_collection.find.return_value = cursor

        await mongo_connector.execute_query(table="users", columns=["name", "email"])
        call_args = mongo_connector._mock_collection.find.call_args
        projection = call_args[0][1]
        assert projection == {"name": 1, "email": 1}

    @pytest.mark.anyio
    async def test_with_filter(self, mongo_connector):
        cursor = self._make_cursor([])
        mongo_connector._mock_collection.find.return_value = cursor

        await mongo_connector.execute_query(
            table="users",
            filters=[{"column": "status", "operator": "eq", "value": "active"}],
        )
        call_args = mongo_connector._mock_collection.find.call_args
        mongo_filter = call_args[0][0]
        assert "$and" in mongo_filter

    @pytest.mark.anyio
    async def test_with_sort_asc(self, mongo_connector):
        cursor = self._make_cursor([])
        cursor.sort = MagicMock(return_value=cursor)
        mongo_connector._mock_collection.find.return_value = cursor

        await mongo_connector.execute_query(
            table="users",
            sort=[{"column": "name", "direction": "asc"}],
        )
        cursor.sort.assert_called_once_with([("name", 1)])

    @pytest.mark.anyio
    async def test_with_sort_desc(self, mongo_connector):
        cursor = self._make_cursor([])
        cursor.sort = MagicMock(return_value=cursor)
        mongo_connector._mock_collection.find.return_value = cursor

        await mongo_connector.execute_query(
            table="users",
            sort=[{"column": "created_at", "direction": "desc"}],
        )
        cursor.sort.assert_called_once_with([("created_at", -1)])

    @pytest.mark.anyio
    async def test_with_limit(self, mongo_connector):
        cursor = self._make_cursor([])
        cursor.limit = MagicMock(return_value=cursor)
        mongo_connector._mock_collection.find.return_value = cursor

        await mongo_connector.execute_query(table="users", limit=10)
        cursor.limit.assert_called_once_with(10)


# ── get_schema ────────────────────────────────────────────────────────────────

class TestGetSchema:
    @pytest.mark.anyio
    async def test_returns_collections(self, mongo_connector):
        mock_db = mongo_connector._mock_db
        mock_db.list_collection_names = AsyncMock(return_value=["users", "orders"])

        mock_users_coll = MagicMock()
        mock_users_coll.estimated_document_count = AsyncMock(return_value=100)
        agg_cursor = MagicMock()
        agg_cursor.to_list = AsyncMock(return_value=[{"name": "Alice", "age": 30}])
        mock_users_coll.aggregate = MagicMock(return_value=agg_cursor)

        mock_orders_coll = MagicMock()
        mock_orders_coll.estimated_document_count = AsyncMock(return_value=50)
        agg_cursor2 = MagicMock()
        agg_cursor2.to_list = AsyncMock(return_value=[])
        mock_orders_coll.aggregate = MagicMock(return_value=agg_cursor2)

        def getitem(key):
            if key == "users":
                return mock_users_coll
            return mock_orders_coll

        mock_db.__getitem__ = MagicMock(side_effect=getitem)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        result = await mongo_connector.get_schema()
        names = [t["name"] for t in result]
        assert "users" in names
        assert "orders" in names

    @pytest.mark.anyio
    async def test_skips_system_collections(self, mongo_connector):
        mock_db = mongo_connector._mock_db
        mock_db.list_collection_names = AsyncMock(
            return_value=["users", "system.users"]
        )

        mock_coll = MagicMock()
        mock_coll.estimated_document_count = AsyncMock(return_value=10)
        agg_cursor = MagicMock()
        agg_cursor.to_list = AsyncMock(return_value=[])
        mock_coll.aggregate = MagicMock(return_value=agg_cursor)

        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        result = await mongo_connector.get_schema()
        names = [t["name"] for t in result]
        assert "system.users" not in names


# ── get_table_schema ──────────────────────────────────────────────────────────

class TestGetTableSchema:
    @pytest.mark.anyio
    async def test_returns_schema_for_collection(self, mongo_connector):
        mock_db = mongo_connector._mock_db
        mock_coll = MagicMock()
        mock_coll.estimated_document_count = AsyncMock(return_value=10)
        agg_cursor = MagicMock()
        agg_cursor.to_list = AsyncMock(return_value=[{"name": "Alice"}])
        mock_coll.aggregate = MagicMock(return_value=agg_cursor)
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        result = await mongo_connector.get_table_schema("users")
        assert result["name"] == "users"
        assert result["row_estimate"] == 10


# ── preview_table ─────────────────────────────────────────────────────────────

class TestPreviewTable:
    @pytest.mark.anyio
    async def test_returns_preview_dict(self, mongo_connector):
        mock_db = mongo_connector._mock_db
        mock_coll = MagicMock()

        find_cursor = MagicMock()
        find_cursor.limit = MagicMock(return_value=find_cursor)
        find_cursor.to_list = AsyncMock(return_value=[{"_id": "id1", "name": "Alice"}])
        mock_coll.find = MagicMock(return_value=find_cursor)

        mock_db.__getitem__ = MagicMock(return_value=mock_coll)
        mongo_connector._client.__getitem__ = MagicMock(return_value=mock_db)

        result = await mongo_connector.preview_table("users", limit=50)
        assert "rows" in result
        assert "columns" in result
        assert "total_sampled" in result


# ── connect / disconnect ──────────────────────────────────────────────────────

class TestConnectDisconnect:
    @pytest.mark.anyio
    async def test_disconnect_clears_client(self, mongo_connector):
        await mongo_connector.disconnect()
        assert mongo_connector._client is None

    @pytest.mark.anyio
    async def test_disconnect_noop_when_no_client(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        connector = MongoDBConnector.__new__(MongoDBConnector)
        connector.credentials = {"database": "testdb"}
        connector._db_name = "testdb"
        connector._client = None
        # Should not raise
        await connector.disconnect()


# ── test_connection ───────────────────────────────────────────────────────────

class TestTestConnection:
    @pytest.mark.anyio
    async def test_success(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        connector = MongoDBConnector.__new__(MongoDBConnector)
        connector.credentials = {"host": "localhost", "database": "testdb"}
        connector._db_name = "testdb"
        connector._client = None

        mock_client = MagicMock()
        mock_client.admin = MagicMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        mock_client.close = MagicMock()

        with patch(
            "easyweaver.connectors.implementations.mongodb.AsyncIOMotorClient",
            return_value=mock_client,
        ):
            result = await connector.test_connection()

        assert result["success"] is True

    @pytest.mark.anyio
    async def test_failure(self):
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        connector = MongoDBConnector.__new__(MongoDBConnector)
        connector.credentials = {"host": "localhost", "database": "testdb"}
        connector._db_name = "testdb"
        connector._client = None

        mock_client = MagicMock()
        mock_client.admin = MagicMock()
        mock_client.admin.command = AsyncMock(side_effect=Exception("Network error"))
        mock_client.close = MagicMock()

        with patch(
            "easyweaver.connectors.implementations.mongodb.AsyncIOMotorClient",
            return_value=mock_client,
        ):
            result = await connector.test_connection()

        assert result["success"] is False
        assert "Network error" in result["message"]
