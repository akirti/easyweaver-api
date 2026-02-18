import time
from typing import Any

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
        host = c.get("host", "localhost")
        port = c.get("port", 27017)
        user = c.get("user", "")
        password = c.get("password", "")
        auth_db = c.get("auth_database", "admin")
        if user and password:
            return f"mongodb://{user}:{password}@{host}:{port}/{self._db_name}?authSource={auth_db}"
        return f"mongodb://{host}:{port}/{self._db_name}"

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
            result.append({
                "name": coll_name,
                "row_estimate": count,
                "columns": columns,
            })
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
    ) -> list[dict[str, Any]]:
        coll = self._db[table]
        mongo_filter = self._build_filter(filters) if filters else {}
        projection = {c: 1 for c in columns} if columns else None
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
    def _build_filter(filters: list[dict]) -> dict:
        conditions = []
        for f in filters:
            col = f["column"]
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
                conditions.append({col: None})
            elif op == "is_not_null":
                conditions.append({col: {"$ne": None}})
        return {"$and": conditions} if conditions else {}

    @staticmethod
    def _infer_columns(docs: list[dict]) -> list[dict[str, Any]]:
        field_types: dict[str, set[str]] = {}
        for doc in docs:
            for key, value in doc.items():
                if key == "_id":
                    continue
                if key not in field_types:
                    field_types[key] = set()
                t = type(value).__name__
                field_types[key].add(t)

        columns = []
        for name, types in sorted(field_types.items()):
            primary_type = next(iter(types)) if len(types) == 1 else "mixed"
            type_map = {"str": "string", "int": "integer", "float": "float", "bool": "boolean"}
            columns.append({
                "name": name,
                "type": type_map.get(primary_type, primary_type),
                "nullable": True,
                "primary_key": False,
            })
        return columns

    @staticmethod
    def _serialize_docs(docs: list[dict]) -> list[dict[str, Any]]:
        rows = []
        for doc in docs:
            row = {}
            for k, v in doc.items():
                if k == "_id":
                    row[k] = str(v)
                else:
                    row[k] = v
            rows.append(row)
        return rows
