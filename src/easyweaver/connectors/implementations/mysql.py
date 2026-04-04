import time
from typing import Any

import aiomysql

from easyweaver.connectors.base import BaseConnector

_MYSQL_INT_TYPES = {"tinyint", "smallint", "mediumint", "int", "bigint"}
_MYSQL_FLOAT_TYPES = {"float", "double", "decimal", "numeric"}


class MySQLConnector(BaseConnector):
    def __init__(self, credentials: dict[str, Any]):
        super().__init__(credentials)
        self._pool: aiomysql.Pool | None = None
        self._column_types: dict[str, dict[str, str]] = {}

    @staticmethod
    def _qualified_table_name(table: str) -> str:
        """Return a properly quoted schema.table SQL identifier."""
        if '.' in table:
            schema, tbl = table.split('.', 1)
            return f'`{schema}`.`{tbl}`'
        return f'`{table}`'

    async def connect(self) -> None:
        c = self.credentials
        self._pool = await aiomysql.create_pool(
            host=c["host"],
            port=c.get("port", 3306),
            user=c["user"],
            password=c["password"],
            db=c["database"],
            minsize=1,
            maxsize=5,
            autocommit=True,
        )

    async def disconnect(self) -> None:
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def test_connection(self) -> dict[str, Any]:
        c = self.credentials
        start = time.monotonic()
        try:
            pool = await aiomysql.create_pool(
                host=c["host"],
                port=c.get("port", 3306),
                user=c["user"],
                password=c["password"],
                db=c["database"],
                minsize=1,
                maxsize=1,
                autocommit=True,
            )
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
            pool.close()
            await pool.wait_closed()
            latency = round((time.monotonic() - start) * 1000, 1)
            return {"success": True, "latency_ms": latency, "message": "Connection successful"}
        except Exception as e:
            return {"success": False, "latency_ms": None, "message": str(e)}

    async def get_schema(self) -> list[dict[str, Any]]:
        assert self._pool
        query = """
            SELECT
                t.TABLE_NAME,
                c.COLUMN_NAME,
                c.DATA_TYPE,
                c.IS_NULLABLE,
                CASE WHEN k.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END as is_primary_key,
                t.TABLE_ROWS as row_estimate
            FROM information_schema.TABLES t
            JOIN information_schema.COLUMNS c
                ON t.TABLE_NAME = c.TABLE_NAME AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
            LEFT JOIN information_schema.KEY_COLUMN_USAGE k
                ON k.TABLE_NAME = c.TABLE_NAME
                AND k.COLUMN_NAME = c.COLUMN_NAME
                AND k.TABLE_SCHEMA = c.TABLE_SCHEMA
                AND k.CONSTRAINT_NAME = 'PRIMARY'
            WHERE t.TABLE_SCHEMA = %s AND t.TABLE_TYPE = 'BASE TABLE'
            ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, (self.credentials["database"],))
                rows = await cur.fetchall()

        db_name = self.credentials["database"]
        tables: dict[str, dict] = {}
        for row in rows:
            tn = f"{db_name}.{row['TABLE_NAME']}"
            if tn not in tables:
                tables[tn] = {
                    "name": tn,
                    "row_estimate": row["row_estimate"] or 0,
                    "columns": [],
                }
            tables[tn]["columns"].append(
                {
                    "name": row["COLUMN_NAME"],
                    "type": row["DATA_TYPE"],
                    "nullable": row["IS_NULLABLE"] == "YES",
                    "primary_key": bool(row["is_primary_key"]),
                }
            )
        return list(tables.values())

    async def get_table_schema(self, table_name: str) -> dict[str, Any]:
        tables = await self.get_schema()
        for t in tables:
            if t["name"] == table_name:
                return t
        return {"name": table_name, "columns": [], "row_estimate": 0}

    async def preview_table(self, table_name: str, limit: int = 50) -> dict[str, Any]:
        assert self._pool
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                sql_table = self._qualified_table_name(table_name)
                await cur.execute(f"SELECT * FROM {sql_table} LIMIT %s", (limit,))
                rows = await cur.fetchall()
                columns = [{"name": k, "type": "text"} for k in rows[0].keys()] if rows else []
                return {
                    "columns": columns,
                    "rows": [dict(r) for r in rows],
                    "total_sampled": len(rows),
                }

    async def _get_column_types(self, table: str) -> dict[str, str]:
        if table in self._column_types:
            return self._column_types[table]
        assert self._pool
        if '.' in table:
            db, tbl = table.split('.', 1)
        else:
            db, tbl = self.credentials["database"], table
        query = """
            SELECT COLUMN_NAME, DATA_TYPE
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, (db, tbl))
                rows = await cur.fetchall()
        mapping = {r["COLUMN_NAME"]: r["DATA_TYPE"] for r in rows}
        self._column_types[table] = mapping
        return mapping

    def _coerce_value(self, value: Any, mysql_type: str) -> Any:
        if value is None:
            return None
        if mysql_type in _MYSQL_INT_TYPES:
            if isinstance(value, int):
                return value
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        if mysql_type in _MYSQL_FLOAT_TYPES:
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        if mysql_type == "tinyint":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "t", "yes")
            return bool(value)
        return value

    async def execute_query(
        self,
        table: str,
        columns: list[str] | None = None,
        filters: list[dict] | None = None,
        sort: list[dict] | None = None,
        limit: int | None = None,
        filter_logic: str = "and",
    ) -> list[dict[str, Any]]:
        assert self._pool

        col_types = await self._get_column_types(table) if filters else {}

        col_clause = ", ".join(f"`{c}`" for c in columns) if columns else "*"
        sql_table = self._qualified_table_name(table)
        query = f"SELECT {col_clause} FROM {sql_table}"
        params: list[Any] = []

        if filters:
            clauses = []
            for f in filters:
                op = f["operator"]
                col_name = f["column"]
                mysql_type = col_types.get(col_name, "varchar")
                if op == "eq":
                    clauses.append(f"`{col_name}` = %s")
                    params.append(self._coerce_value(f["value"], mysql_type))
                elif op == "neq":
                    clauses.append(f"`{col_name}` != %s")
                    params.append(self._coerce_value(f["value"], mysql_type))
                elif op == "gt":
                    clauses.append(f"`{col_name}` > %s")
                    params.append(self._coerce_value(f["value"], mysql_type))
                elif op == "lt":
                    clauses.append(f"`{col_name}` < %s")
                    params.append(self._coerce_value(f["value"], mysql_type))
                elif op == "gte":
                    clauses.append(f"`{col_name}` >= %s")
                    params.append(self._coerce_value(f["value"], mysql_type))
                elif op == "lte":
                    clauses.append(f"`{col_name}` <= %s")
                    params.append(self._coerce_value(f["value"], mysql_type))
                elif op == "like":
                    clauses.append(f"`{col_name}` LIKE %s")
                    params.append(f"%{f['value']}%")
                elif op == "is_null":
                    clauses.append(f"`{col_name}` IS NULL")
                elif op == "is_not_null":
                    clauses.append(f"`{col_name}` IS NOT NULL")
                elif op == "in":
                    values = f.get("value", [])
                    if not values:
                        clauses.append("FALSE")
                    else:
                        coerced = [self._coerce_value(v, mysql_type) for v in values]
                        placeholders = ", ".join("%s" for _ in coerced)
                        clauses.append(f"`{col_name}` IN ({placeholders})")
                        params.extend(coerced)
                elif op == "not_in":
                    values = f.get("value", [])
                    if not values:
                        clauses.append("TRUE")
                    else:
                        coerced = [self._coerce_value(v, mysql_type) for v in values]
                        placeholders = ", ".join("%s" for _ in coerced)
                        clauses.append(f"`{col_name}` NOT IN ({placeholders})")
                        params.extend(coerced)
                elif op == "between":
                    clauses.append(f"`{col_name}` BETWEEN %s AND %s")
                    params.append(self._coerce_value(f["value"], mysql_type))
                    params.append(self._coerce_value(f["value2"], mysql_type))
            if clauses:
                joiner = " OR " if filter_logic == "or" else " AND "
                query += " WHERE " + joiner.join(clauses)

        if sort:
            order_parts = []
            for s in sort:
                direction = "DESC" if s.get("direction", "asc") == "desc" else "ASC"
                order_parts.append(f"`{s['column']}` {direction}")
            query += " ORDER BY " + ", ".join(order_parts)

        if limit:
            query += f" LIMIT {int(limit)}"

        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, tuple(params) if params else None)
                rows = await cur.fetchall()
                return [dict(r) for r in rows]
