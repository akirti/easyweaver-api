import base64
import ssl
import tempfile
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
        self._pk_cache: dict[str, str] = {}  # table -> primary key column
        self._temp_files: list[str] = []

    @property
    def supports_batching(self) -> bool:
        return True

    @staticmethod
    def _qualified_table_name(table: str) -> str:
        """Return a properly quoted schema.table SQL identifier."""
        if '.' in table:
            schema, tbl = table.split('.', 1)
            return f'`{schema}`.`{tbl}`'
        return f'`{table}`'

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        """Build SSL context from credentials if SSL is configured."""
        c = self.credentials
        ssl_mode = c.get("ssl_mode", "disable")

        if ssl_mode == "disable":
            return None

        # For 'require' without certs, enable SSL without verification
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
        """Decode base64 cert content and write to a temp file."""
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

    def _pool_kwargs(self) -> dict[str, Any]:
        """Build common pool kwargs including SSL if configured."""
        c = self.credentials
        kwargs: dict[str, Any] = {
            "host": c["host"],
            "port": c.get("port", 3306),
            "user": c["user"],
            "password": c["password"],
            "db": c["database"],
            "autocommit": True,
        }
        ssl_ctx = self._build_ssl_context()
        if ssl_ctx is not None:
            kwargs["ssl"] = ssl_ctx
        return kwargs

    async def connect(self) -> None:
        self._pool = await aiomysql.create_pool(
            **self._pool_kwargs(),
            minsize=1,
            maxsize=5,
        )

    async def disconnect(self) -> None:
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
        self._cleanup_temp_files()

    async def test_connection(self) -> dict[str, Any]:
        start = time.monotonic()
        try:
            pool = await aiomysql.create_pool(
                **self._pool_kwargs(),
                minsize=1,
                maxsize=1,
            )
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
            pool.close()
            await pool.wait_closed()
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

    def _build_where_clause(
        self, filters: list[dict], col_types: dict, filter_logic: str
    ) -> tuple[str, list[Any]]:
        """Build WHERE clause. MySQL uses %s placeholders."""
        if not filters:
            return "", []
        clauses: list[str] = []
        params: list[Any] = []
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
            return " WHERE " + joiner.join(clauses), params
        return "", []

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

        where_clause, params = self._build_where_clause(
            filters or [], col_types, filter_logic
        )
        query += where_clause

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

    async def _get_primary_key(self, table: str) -> str:
        """Detect the primary key column for a table (cached).

        Falls back to the first column if no PK is found.
        """
        if table in self._pk_cache:
            return self._pk_cache[table]
        assert self._pool
        if '.' in table:
            _db, tbl = table.split('.', 1)
        else:
            tbl = table
        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    f"SHOW KEYS FROM {self._qualified_table_name(table)} WHERE Key_name = 'PRIMARY'"
                )
                rows = await cur.fetchall()
        if rows:
            pk = rows[0]["Column_name"]
        else:
            # Fall back to first column
            async with self._pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(
                        f"SHOW COLUMNS FROM {self._qualified_table_name(table)}"
                    )
                    cols = await cur.fetchall()
            pk = cols[0]["Field"] if cols else "id"
        self._pk_cache[table] = pk
        return pk

    async def execute_query_batched(
        self,
        table: str,
        columns: list[str] | None = None,
        filters: list[dict] | None = None,
        filter_logic: str = "and",
        batch_size: int = 10_000,
        offset: int = 0,
        last_key: Any = None,
    ) -> tuple[list[dict[str, Any]], bool, Any]:
        assert self._pool

        pk = await self._get_primary_key(table)
        col_types = await self._get_column_types(table) if filters else {}

        col_clause = ", ".join(f"`{c}`" for c in columns) if columns else "*"
        sql_table = self._qualified_table_name(table)
        query = f"SELECT {col_clause} FROM {sql_table}"

        where_clause, params = self._build_where_clause(
            filters or [], col_types, filter_logic
        )

        # Keyset pagination: when last_key is provided, add pk > last_key
        if last_key is not None:
            pk_condition = f"`{pk}` > %s"
            params.append(last_key)
            if where_clause:
                query += where_clause + f" AND {pk_condition}"
            else:
                query += f" WHERE {pk_condition}"
        else:
            query += where_clause

        query += f" ORDER BY `{pk}` LIMIT {int(batch_size) + 1}"

        # Fall back to OFFSET when no keyset key and offset > 0
        if last_key is None and offset > 0:
            query += f" OFFSET {int(offset)}"

        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, tuple(params) if params else None)
                rows = await cur.fetchall()

        result = [dict(r) for r in rows]
        if len(result) > batch_size:
            result = result[:batch_size]
            last_pk = result[-1][pk] if result else None
            return result, True, last_pk
        last_pk = result[-1][pk] if result else None
        return result, False, last_pk
