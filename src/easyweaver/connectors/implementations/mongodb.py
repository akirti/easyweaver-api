import json
import time
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from easyweaver.connectors.base import BaseConnector

# MongoDB type mapping
_BSON_TYPE_MAP = {
    "int": "integer",
    "long": "integer",
    "double": "float",
    "decimal": "decimal",
    "string": "string",
    "bool": "boolean",
    "date": "datetime",
    "objectId": "objectId",
    "array": "array",
    "object": "object",
    "null": "null",
}


class MongoDBConnector(BaseConnector):
    def __init__(self, credentials: dict[str, Any]):
        super().__init__(credentials)
        self._client: AsyncIOMotorClient | None = None
        self._db_name = credentials.get("database", "test")

    def _uri(self) -> str:
        c = self.credentials
        # If a direct connection string is provided, use it as-is
        connection_string = c.get("connection_string", "")
        if connection_string:
            return connection_string

        host = c.get("host", "localhost")
        port = c.get("port")
        user = c.get("user", "")
        password = c.get("password", "")
        auth_db = c.get("auth_database", "admin")
        scheme = c.get("scheme", "mongodb")

        # Strip any protocol prefix the user may have included in the host
        clean_host = host.replace("mongodb+srv://", "").replace("mongodb://", "").rstrip("/")

        is_srv = scheme == "mongodb+srv"

        if is_srv:
            if user and password:
                return f"mongodb+srv://{user}:{password}@{clean_host}/{self._db_name}?authSource={auth_db}"
            return f"mongodb+srv://{clean_host}/{self._db_name}"
        else:
            actual_port = port or 27017
            if user and password:
                return f"mongodb://{user}:{password}@{clean_host}:{actual_port}/{self._db_name}?authSource={auth_db}"
            return f"mongodb://{clean_host}:{actual_port}/{self._db_name}"

    @property
    def _db(self):
        assert self._client
        return self._client[self._db_name]

    async def connect(self) -> None:
        self._client = AsyncIOMotorClient(self._uri())
        await self._client.admin.command("ping")

    async def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    async def test_connection(self) -> dict[str, Any]:
        start = time.monotonic()
        try:
            client = AsyncIOMotorClient(self._uri(), serverSelectionTimeoutMS=5000)
            await client.admin.command("ping")
            latency = round((time.monotonic() - start) * 1000, 1)
            client.close()
            return {"success": True, "latency_ms": latency, "message": "Connection successful"}
        except Exception as e:
            return {"success": False, "latency_ms": None, "message": str(e)}

    async def get_schema(self) -> list[dict[str, Any]]:
        collections = await self._db.list_collection_names()
        result = []
        for coll_name in sorted(collections):
            if coll_name.startswith("system."):
                continue
            coll = self._db[coll_name]
            count = await coll.estimated_document_count()
            # Sample documents to infer schema
            sample = await coll.aggregate([{"$sample": {"size": 100}}]).to_list(100)
            columns = self._infer_columns(sample)
            result.append(
                {
                    "name": coll_name,
                    "row_estimate": count,
                    "columns": columns,
                }
            )
        return result

    async def get_table_schema(self, table_name: str) -> dict[str, Any]:
        coll = self._db[table_name]
        count = await coll.estimated_document_count()
        sample = await coll.aggregate([{"$sample": {"size": 100}}]).to_list(100)
        columns = self._infer_columns(sample)
        return {"name": table_name, "row_estimate": count, "columns": columns}

    async def preview_table(self, table_name: str, limit: int = 50) -> dict[str, Any]:
        coll = self._db[table_name]
        docs = await coll.find().limit(limit).to_list(limit)
        rows = self._serialize_docs(docs)
        columns = self._infer_columns(docs)
        return {"columns": columns, "rows": rows, "total_sampled": len(rows)}

    async def execute_query(
        self,
        table: str,
        columns: list[str] | None = None,
        filters: list[dict] | None = None,
        sort: list[dict] | None = None,
        limit: int | None = None,
        filter_logic: str = "and",
    ) -> list[dict[str, Any]]:
        coll = self._db[table]
        mongo_filter = self._build_filter(filters, filter_logic) if filters else {}
        projection = {c.replace("[].", "."): 1 for c in columns} if columns else None
        cursor = coll.find(mongo_filter, projection)

        if sort:
            sort_list = []
            for s in sort:
                direction = -1 if s.get("direction", "asc") == "desc" else 1
                sort_list.append((s["column"], direction))
            cursor = cursor.sort(sort_list)

        if limit:
            cursor = cursor.limit(limit)

        docs = await cursor.to_list(limit or 10000)
        return self._serialize_docs(docs)

    @staticmethod
    def _build_filter(filters: list[dict], logic: str = "and") -> dict:
        conditions = []
        for f in filters:
            col = f["column"].replace("[].", ".")
            op = f["operator"]
            val = f.get("value")
            if op == "eq":
                conditions.append({col: val})
            elif op == "neq":
                conditions.append({col: {"$ne": val}})
            elif op == "gt":
                conditions.append({col: {"$gt": val}})
            elif op == "lt":
                conditions.append({col: {"$lt": val}})
            elif op == "gte":
                conditions.append({col: {"$gte": val}})
            elif op == "lte":
                conditions.append({col: {"$lte": val}})
            elif op == "like":
                conditions.append({col: {"$regex": val, "$options": "i"}})
            elif op == "is_null":
                conditions.append({"$or": [{col: None}, {col: {"$exists": False}}]})
            elif op == "is_not_null":
                conditions.append({col: {"$exists": True, "$ne": None}})
            elif op == "in":
                values = val if isinstance(val, list) else []
                conditions.append({col: {"$in": values}})
            elif op == "not_in":
                values = val if isinstance(val, list) else []
                conditions.append({col: {"$nin": values}})
            elif op == "between":
                val2 = f.get("value2")
                conditions.append({col: {"$gte": val, "$lte": val2}})
        if not conditions:
            return {}
        key = "$or" if logic == "or" else "$and"
        return {key: conditions}

    @staticmethod
    def _infer_columns(docs: list[dict], max_depth: int = 3) -> list[dict[str, Any]]:
        """Infer column names and types, recursing into nested dicts."""
        field_types: dict[str, set[str]] = {}

        def _collect(obj: dict, prefix: str = "", depth: int = 0):
            for key, value in obj.items():
                if key == "_id" and not prefix:
                    continue
                full_key = f"{prefix}{key}" if prefix else key
                if isinstance(value, dict) and depth < max_depth:
                    _collect(value, f"{full_key}.", depth + 1)
                elif (
                    isinstance(value, list)
                    and value
                    and isinstance(value[0], dict)
                    and depth < max_depth
                ):
                    _collect(value[0], f"{full_key}[].", depth + 1)
                else:
                    if full_key not in field_types:
                        field_types[full_key] = set()
                    t = type(value).__name__
                    field_types[full_key].add(t)

        for doc in docs:
            _collect(doc)

        columns = []
        type_map = {
            "str": "string",
            "int": "integer",
            "float": "float",
            "bool": "boolean",
            "list": "array",
            "NoneType": "null",
        }
        for name, types in sorted(field_types.items()):
            types_no_null = types - {"NoneType"}
            primary_type = (
                next(iter(types_no_null))
                if len(types_no_null) == 1
                else ("mixed" if types_no_null else "null")
            )
            columns.append(
                {
                    "name": name,
                    "type": type_map.get(primary_type, primary_type),
                    "nullable": True,
                    "primary_key": False,
                }
            )
        return columns

    @staticmethod
    def _make_serializable(v: Any) -> Any:
        """Convert non-JSON-serializable types to strings."""
        if isinstance(v, ObjectId):
            return str(v)
        if isinstance(v, (dict, list)):
            return json.dumps(v, default=str)
        return v

    @staticmethod
    def _flatten_doc(doc: dict, prefix: str = "", max_depth: int = 3, depth: int = 0) -> dict:
        """Flatten nested dicts and arrays into dot-path keys matching column names."""
        flat: dict[str, Any] = {}
        for k, v in doc.items():
            key = f"{prefix}{k}" if prefix else k
            if k == "_id" and not prefix:
                flat[k] = str(v)
            elif isinstance(v, dict) and depth < max_depth:
                flat.update(MongoDBConnector._flatten_doc(v, f"{key}.", max_depth, depth + 1))
            elif isinstance(v, list) and v and isinstance(v[0], dict) and depth < max_depth:
                # Array of dicts: extract each sub-field across all elements
                all_sub_keys: set[str] = set()
                for elem in v:
                    if isinstance(elem, dict):
                        all_sub_keys.update(elem.keys())
                for sub_key in sorted(all_sub_keys):
                    arr_key = f"{key}[].{sub_key}"
                    vals = [elem.get(sub_key) for elem in v if isinstance(elem, dict)]
                    # Collapse to single value if all same, else comma-join
                    primitives = [
                        MongoDBConnector._make_serializable(x) for x in vals if x is not None
                    ]
                    if len(primitives) == 1:
                        flat[arr_key] = primitives[0]
                    elif primitives:
                        flat[arr_key] = ", ".join(str(x) for x in primitives)
                    else:
                        flat[arr_key] = None
            else:
                flat[key] = MongoDBConnector._make_serializable(v)
        return flat

    @staticmethod
    def _serialize_docs(docs: list[dict]) -> list[dict[str, Any]]:
        return [MongoDBConnector._flatten_doc(doc) for doc in docs]
