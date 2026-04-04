import base64
import ssl
import tempfile
import time
from typing import Any

import asyncpg

from easyweaver.connectors.base import BaseConnector


_PG_INT_TYPES = {"smallint", "integer", "bigint"}
_PG_FLOAT_TYPES = {"real", "double precision", "numeric", "money"}


class PostgresConnector(BaseConnector):
    def __init__(self, credentials: dict[str, Any]):
        super().__init__(credentials)
        self._pool: asyncpg.Pool | None = None
        self._column_types: dict[str, dict[str, str]] = {}  # table -> {col: type}
        self._temp_files: list[str] = []  # track temp cert files for cleanup

    @staticmethod
    def _qualified_table_name(table: str) -> str:
        """Return a properly quoted schema.table SQL identifier."""
        if '.' in table:
            schema, tbl = table.split('.', 1)
            return f'"{schema}"."{tbl}"'
        return f'"public"."{table}"'

    def _dsn(self) -> str:
        c = self.credentials
        return f"postgresql://{c['user']}:{c['password']}@{c['host']}:{c.get('port', 5432)}/{c['database']}"

    def _build_ssl_context(self) -> ssl.SSLContext | bool | None:
        """Build SSL context from credentials if SSL is configured."""
        c = self.credentials
        ssl_mode = c.get("ssl_mode", "disable")

        if ssl_mode == "disable":
            return None

        # For 'require' without certs, just enable SSL without verification
        if ssl_mode == "require" and not c.get("ssl_ca_cert"):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx

        # For verify-ca / verify-full or when certs are provided
        ctx = ssl.create_default_context()

        if ssl_mode == "verify-full":
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
        elif ssl_mode == "verify-ca":
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_REQUIRED
        else:
            # prefer / require with certs
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        # Write base64-encoded certificates to temp files
        if c.get("ssl_ca_cert"):
            ca_path = self._write_temp_cert(c["ssl_ca_cert"])
            ctx.load_verify_locations(ca_path)
            if ssl_mode in ("verify-ca", "verify-full"):
                ctx.verify_mode = ssl.CERT_REQUIRED

        if c.get("ssl_client_cert") and c.get("ssl_client_key"):
            cert_path = self._write_temp_cert(c["ssl_client_cert"])
            key_path = self._write_temp_cert(c["ssl_client_key"])
            ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
        elif c.get("ssl_client_cert"):
            cert_path = self._write_temp_cert(c["ssl_client_cert"])
            ctx.load_cert_chain(certfile=cert_path)

        return ctx

    def _write_temp_cert(self, b64_content: str) -> str:
        """Decode base64 cert content and write to a temp file. Returns the file path."""
        content = base64.b64decode(b64_content)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
        tmp.write(content)
        tmp.close()
        self._temp_files.append(tmp.name)
        return tmp.name

    def _cleanup_temp_files(self) -> None:
        import os
        for path in self._temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._temp_files.clear()

    async def connect(self) -> None:
        ssl_ctx = self._build_ssl_context()
        kwargs: dict[str, Any] = {"min_size": 1, "max_size": 5}
        if ssl_ctx is not None:
            kwargs["ssl"] = ssl_ctx
        self._pool = await asyncpg.create_pool(self._dsn(), **kwargs)

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._cleanup_temp_files()

    async def test_connection(self) -> dict[str, Any]:
        start = time.monotonic()
        try:
            ssl_ctx = self._build_ssl_context()
            kwargs: dict[str, Any] = {"min_size": 1, "max_size": 1}
            if ssl_ctx is not None:
                kwargs["ssl"] = ssl_ctx
            pool = await asyncpg.create_pool(self._dsn(), **kwargs)
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            await pool.close()
            self._cleanup_temp_files()
            latency = round((time.monotonic() - start) * 1000, 1)
            return {"success": True, "latency_ms": latency, "message": "Connection successful"}
        except Exception as e:
            self._cleanup_temp_files()
            return {"success": False, "latency_ms": None, "message": str(e)}

    async def get_schema(self) -> list[dict[str, Any]]:
        assert self._pool
        query = """
            SELECT
                t.table_schema,
                t.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key,
                COALESCE(s.n_live_tup, 0) as row_estimate
            FROM information_schema.tables t
            JOIN information_schema.columns c
                ON t.table_name = c.table_name AND t.table_schema = c.table_schema
            LEFT JOIN (
                SELECT ku.column_name, ku.table_name, ku.table_schema
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku
                    ON tc.constraint_name = ku.constraint_name
                    AND tc.table_schema = ku.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
            ) pk ON pk.column_name = c.column_name
                AND pk.table_name = t.table_name
                AND pk.table_schema = t.table_schema
            LEFT JOIN pg_stat_user_tables s
                ON s.schemaname = t.table_schema AND s.relname = t.table_name
            WHERE t.table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                AND t.table_type = 'BASE TABLE'
            ORDER BY t.table_schema, t.table_name, c.ordinal_position
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query)

        tables: dict[str, dict] = {}
        for row in rows:
            tn = f"{row['table_schema']}.{row['table_name']}"
            if tn not in tables:
                tables[tn] = {
                    "name": tn,
                    "row_estimate": row["row_estimate"] or 0,
                    "columns": [],
                }
            tables[tn]["columns"].append(
                {
                    "name": row["column_name"],
                    "type": row["data_type"],
                    "nullable": row["is_nullable"] == "YES",
                    "primary_key": row["is_primary_key"],
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
        # Safe: table_name is validated against schema
        sql_table = self._qualified_table_name(table_name)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f'SELECT * FROM {sql_table} LIMIT $1', limit)
            columns = [{"name": k, "type": "text"} for k in rows[0].keys()] if rows else []
            return {
                "columns": columns,
                "rows": [dict(r) for r in rows],
                "total_sampled": len(rows),
            }

    async def _get_column_types(self, table: str) -> dict[str, str]:
        """Fetch and cache column name→data_type mapping for a table."""
        if table in self._column_types:
            return self._column_types[table]
        assert self._pool
        if '.' in table:
            schema, tbl = table.split('.', 1)
        else:
            schema, tbl = 'public', table
        query = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, schema, tbl)
        mapping = {r["column_name"]: r["data_type"] for r in rows}
        self._column_types[table] = mapping
        return mapping

    def _coerce_value(self, value: Any, pg_type: str) -> Any:
        """Coerce a filter value to match the Postgres column type."""
        if value is None:
            return None
        if pg_type in _PG_INT_TYPES:
            if isinstance(value, int):
                return value
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        if pg_type in _PG_FLOAT_TYPES:
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        if pg_type == "boolean":
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

        col_clause = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        sql_table = self._qualified_table_name(table)
        query = f'SELECT {col_clause} FROM {sql_table}'
        params: list[Any] = []
        idx = 1

        if filters:
            clauses = []
            for f in filters:
                op = f["operator"]
                col_name = f["column"]
                pg_type = col_types.get(col_name, "text")
                if op == "eq":
                    clauses.append(f'"{col_name}" = ${idx}')
                    params.append(self._coerce_value(f["value"], pg_type))
                    idx += 1
                elif op == "neq":
                    clauses.append(f'"{col_name}" != ${idx}')
                    params.append(self._coerce_value(f["value"], pg_type))
                    idx += 1
                elif op == "gt":
                    clauses.append(f'"{col_name}" > ${idx}')
                    params.append(self._coerce_value(f["value"], pg_type))
                    idx += 1
                elif op == "lt":
                    clauses.append(f'"{col_name}" < ${idx}')
                    params.append(self._coerce_value(f["value"], pg_type))
                    idx += 1
                elif op == "gte":
                    clauses.append(f'"{col_name}" >= ${idx}')
                    params.append(self._coerce_value(f["value"], pg_type))
                    idx += 1
                elif op == "lte":
                    clauses.append(f'"{col_name}" <= ${idx}')
                    params.append(self._coerce_value(f["value"], pg_type))
                    idx += 1
                elif op == "like":
                    clauses.append(f'"{col_name}" ILIKE ${idx}')
                    params.append(f"%{f['value']}%")
                    idx += 1
                elif op == "is_null":
                    clauses.append(f'"{col_name}" IS NULL')
                elif op == "is_not_null":
                    clauses.append(f'"{col_name}" IS NOT NULL')
                elif op == "in":
                    values = f.get("value", [])
                    if not values:
                        clauses.append("FALSE")
                    else:
                        coerced = [self._coerce_value(v, pg_type) for v in values]
                        placeholders = ", ".join(f"${idx + i}" for i in range(len(coerced)))
                        clauses.append(f'"{col_name}" IN ({placeholders})')
                        params.extend(coerced)
                        idx += len(coerced)
                elif op == "not_in":
                    values = f.get("value", [])
                    if not values:
                        clauses.append("TRUE")
                    else:
                        coerced = [self._coerce_value(v, pg_type) for v in values]
                        placeholders = ", ".join(f"${idx + i}" for i in range(len(coerced)))
                        clauses.append(f'"{col_name}" NOT IN ({placeholders})')
                        params.extend(coerced)
                        idx += len(coerced)
                elif op == "between":
                    clauses.append(f'"{col_name}" BETWEEN ${idx} AND ${idx + 1}')
                    params.append(self._coerce_value(f["value"], pg_type))
                    params.append(self._coerce_value(f["value2"], pg_type))
                    idx += 2
            if clauses:
                joiner = " OR " if filter_logic == "or" else " AND "
                query += " WHERE " + joiner.join(clauses)

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
