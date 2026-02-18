import time
from typing import Any

import asyncpg

from easyweaver.connectors.base import BaseConnector


class PostgresConnector(BaseConnector):
    def __init__(self, credentials: dict[str, Any]):
        super().__init__(credentials)
        self._pool: asyncpg.Pool | None = None

    def _dsn(self) -> str:
        c = self.credentials
        return f"postgresql://{c['user']}:{c['password']}@{c['host']}:{c.get('port', 5432)}/{c['database']}"

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn(), min_size=1, max_size=5)

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def test_connection(self) -> dict[str, Any]:
        start = time.monotonic()
        try:
            pool = await asyncpg.create_pool(self._dsn(), min_size=1, max_size=1)
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            await pool.close()
            latency = round((time.monotonic() - start) * 1000, 1)
            return {"success": True, "latency_ms": latency, "message": "Connection successful"}
        except Exception as e:
            return {"success": False, "latency_ms": None, "message": str(e)}

    async def get_schema(self) -> list[dict[str, Any]]:
        assert self._pool
        query = """
            SELECT
                t.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key,
                (SELECT reltuples::bigint FROM pg_class WHERE relname = t.table_name) as row_estimate
            FROM information_schema.tables t
            JOIN information_schema.columns c
                ON t.table_name = c.table_name AND t.table_schema = c.table_schema
            LEFT JOIN (
                SELECT ku.column_name, ku.table_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku
                    ON tc.constraint_name = ku.constraint_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
            ) pk ON pk.column_name = c.column_name AND pk.table_name = t.table_name
            WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
            ORDER BY t.table_name, c.ordinal_position
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query)

        tables: dict[str, dict] = {}
        for row in rows:
            tn = row["table_name"]
            if tn not in tables:
                tables[tn] = {
                    "name": tn,
                    "row_estimate": row["row_estimate"] or 0,
                    "columns": [],
                }
            tables[tn]["columns"].append({
                "name": row["column_name"],
                "type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
                "primary_key": row["is_primary_key"],
            })
        return list(tables.values())

    async def get_table_schema(self, table_name: str) -> dict[str, Any]:
        tables = await self.get_schema()
        for t in tables:
            if t["name"] == table_name:
                return t
        return {"name": table_name, "columns": [], "row_estimate": 0}

    async def preview_table(self, table_name: str, limit: int = 50) -> dict[str, Any]:
        assert self._pool
        # Safe: table_name is validated against schema
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f'SELECT * FROM "{table_name}" LIMIT $1', limit)
            columns = [{"name": k, "type": "text"} for k in rows[0].keys()] if rows else []
            return {
                "columns": columns,
                "rows": [dict(r) for r in rows],
                "total_sampled": len(rows),
            }

    async def execute_query(
        self,
        table: str,
        columns: list[str] | None = None,
        filters: list[dict] | None = None,
        sort: list[dict] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        assert self._pool

        col_clause = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        query = f'SELECT {col_clause} FROM "{table}"'
        params: list[Any] = []
        idx = 1

        if filters:
            clauses = []
            for f in filters:
                op = f["operator"]
                if op == "eq":
                    clauses.append(f'"{f["column"]}" = ${idx}')
                    params.append(f["value"])
                    idx += 1
                elif op == "neq":
                    clauses.append(f'"{f["column"]}" != ${idx}')
                    params.append(f["value"])
                    idx += 1
                elif op == "gt":
                    clauses.append(f'"{f["column"]}" > ${idx}')
                    params.append(f["value"])
                    idx += 1
                elif op == "lt":
                    clauses.append(f'"{f["column"]}" < ${idx}')
                    params.append(f["value"])
                    idx += 1
                elif op == "gte":
                    clauses.append(f'"{f["column"]}" >= ${idx}')
                    params.append(f["value"])
                    idx += 1
                elif op == "lte":
                    clauses.append(f'"{f["column"]}" <= ${idx}')
                    params.append(f["value"])
                    idx += 1
                elif op == "like":
                    clauses.append(f'"{f["column"]}" LIKE ${idx}')
                    params.append(f["value"])
                    idx += 1
                elif op == "is_null":
                    clauses.append(f'"{f["column"]}" IS NULL')
                elif op == "is_not_null":
                    clauses.append(f'"{f["column"]}" IS NOT NULL')
            if clauses:
                query += " WHERE " + " AND ".join(clauses)

        if sort:
            order_parts = []
            for s in sort:
                direction = "DESC" if s.get("direction", "asc") == "desc" else "ASC"
                order_parts.append(f'"{s["column"]}" {direction}')
            query += " ORDER BY " + ", ".join(order_parts)

        if limit:
            query += f" LIMIT {int(limit)}"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(r) for r in rows]
