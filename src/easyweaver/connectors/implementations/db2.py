import asyncio
import time
from typing import Any

from easyweaver.connectors.base import BaseConnector

try:
    import ibm_db_dbi

    _HAS_IBM_DB = True
except ImportError:
    _HAS_IBM_DB = False

_DB2_INT_TYPES = {"smallint", "integer", "bigint"}
_DB2_FLOAT_TYPES = {"real", "double", "decimal", "numeric", "decfloat"}


def _require_ibm_db() -> None:
    if not _HAS_IBM_DB:
        raise ImportError(
            "ibm-db is required for DB2 support. "
            "Install it with: pip install ibm-db  (requires a C compiler and IBM CLIDRIVER)."
        )


class DB2Connector(BaseConnector):
    def __init__(self, credentials: dict[str, Any]):
        _require_ibm_db()
        super().__init__(credentials)
        self._conn: Any = None
        self._column_types: dict[str, dict[str, str]] = {}

    @staticmethod
    def _qualified_table_name(table: str) -> str:
        """Return a properly quoted schema.table SQL identifier."""
        if '.' in table:
            schema, tbl = table.split('.', 1)
            return f'"{schema}"."{tbl}"'
        return f'"{table}"'

    def _connection_string(self) -> str:
        c = self.credentials
        return (
            f"DATABASE={c['database']};"
            f"HOSTNAME={c['host']};"
            f"PORT={c.get('port', 50000)};"
            f"PROTOCOL=TCPIP;"
            f"UID={c['user']};"
            f"PWD={c['password']};"
        )

    def _connect_sync(self) -> None:
        self._conn = ibm_db_dbi.connect(self._connection_string(), "", "")

    def _disconnect_sync(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    async def connect(self) -> None:
        await asyncio.to_thread(self._connect_sync)

    async def disconnect(self) -> None:
        await asyncio.to_thread(self._disconnect_sync)

    async def test_connection(self) -> dict[str, Any]:
        start = time.monotonic()
        try:

            def _test() -> None:
                conn = ibm_db_dbi.connect(self._connection_string(), "", "")
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM SYSIBM.SYSDUMMY1")
                cursor.fetchone()
                cursor.close()
                conn.close()

            await asyncio.to_thread(_test)
            latency = round((time.monotonic() - start) * 1000, 1)
            return {"success": True, "latency_ms": latency, "message": "Connection successful"}
        except Exception as e:
            return {"success": False, "latency_ms": None, "message": str(e)}

    async def get_schema(self) -> list[dict[str, Any]]:
        assert self._conn

        def _fetch() -> list[dict]:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT t.TABSCHEMA, t.TABNAME, c.COLNAME, c.TYPENAME, c.NULLS, c.KEYSEQ, t.CARD
                FROM SYSCAT.TABLES t
                JOIN SYSCAT.COLUMNS c
                    ON t.TABSCHEMA = c.TABSCHEMA AND t.TABNAME = c.TABNAME
                WHERE t.TABSCHEMA NOT IN ('SYSCAT', 'SYSIBM', 'SYSSTAT', 'SYSTOOLS')
                    AND t.TYPE = 'T'
                ORDER BY t.TABSCHEMA, t.TABNAME, c.COLNO
                """
            )
            cols = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
            return [dict(zip(cols, row)) for row in rows]

        rows = await asyncio.to_thread(_fetch)

        tables: dict[str, dict] = {}
        for row in rows:
            tn = f"{row['TABSCHEMA'].strip()}.{row['TABNAME'].strip()}"
            if tn not in tables:
                tables[tn] = {
                    "name": tn,
                    "row_estimate": max(int(row["CARD"]), 0) if row["CARD"] is not None else 0,
                    "columns": [],
                }
            tables[tn]["columns"].append(
                {
                    "name": row["COLNAME"].strip(),
                    "type": row["TYPENAME"].strip().lower(),
                    "nullable": row["NULLS"] == "Y",
                    "primary_key": (row["KEYSEQ"] or 0) > 0,
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
        assert self._conn

        def _fetch() -> dict[str, Any]:
            cursor = self._conn.cursor()
            sql_table = DB2Connector._qualified_table_name(table_name)
            cursor.execute(f'SELECT * FROM {sql_table} FETCH FIRST {int(limit)} ROWS ONLY')
            cols = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
            columns = [{"name": c, "type": "text"} for c in cols]
            return {
                "columns": columns,
                "rows": [dict(zip(cols, row)) for row in rows],
                "total_sampled": len(rows),
            }

        return await asyncio.to_thread(_fetch)

    async def _get_column_types(self, table: str) -> dict[str, str]:
        if table in self._column_types:
            return self._column_types[table]
        assert self._conn

        if '.' in table:
            schema, tbl = table.split('.', 1)
        else:
            schema, tbl = None, table

        def _fetch() -> dict[str, str]:
            cursor = self._conn.cursor()
            if schema is not None:
                cursor.execute(
                    "SELECT COLNAME, TYPENAME FROM SYSCAT.COLUMNS "
                    "WHERE TABSCHEMA = ? AND TABNAME = ?",
                    (schema, tbl),
                )
            else:
                cursor.execute(
                    "SELECT COLNAME, TYPENAME FROM SYSCAT.COLUMNS "
                    "WHERE TABSCHEMA = CURRENT SCHEMA AND TABNAME = ?",
                    (tbl,),
                )
            rows = cursor.fetchall()
            cursor.close()
            return {r[0].strip(): r[1].strip().lower() for r in rows}

        mapping = await asyncio.to_thread(_fetch)
        self._column_types[table] = mapping
        return mapping

    def _coerce_value(self, value: Any, db2_type: str) -> Any:
        if value is None:
            return None
        if db2_type in _DB2_INT_TYPES:
            if isinstance(value, int):
                return value
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        if db2_type in _DB2_FLOAT_TYPES:
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        if db2_type == "boolean":
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
        assert self._conn

        col_types = await self._get_column_types(table) if filters else {}

        col_clause = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        sql_table = self._qualified_table_name(table)
        query = f'SELECT {col_clause} FROM {sql_table}'
        params: list[Any] = []

        if filters:
            clauses = []
            for f in filters:
                op = f["operator"]
                col_name = f["column"]
                db2_type = col_types.get(col_name, "varchar")
                if op == "eq":
                    clauses.append(f'"{col_name}" = ?')
                    params.append(self._coerce_value(f["value"], db2_type))
                elif op == "neq":
                    clauses.append(f'"{col_name}" != ?')
                    params.append(self._coerce_value(f["value"], db2_type))
                elif op == "gt":
                    clauses.append(f'"{col_name}" > ?')
                    params.append(self._coerce_value(f["value"], db2_type))
                elif op == "lt":
                    clauses.append(f'"{col_name}" < ?')
                    params.append(self._coerce_value(f["value"], db2_type))
                elif op == "gte":
                    clauses.append(f'"{col_name}" >= ?')
                    params.append(self._coerce_value(f["value"], db2_type))
                elif op == "lte":
                    clauses.append(f'"{col_name}" <= ?')
                    params.append(self._coerce_value(f["value"], db2_type))
                elif op == "like":
                    clauses.append(f'"{col_name}" LIKE ?')
                    params.append(f"%{f['value']}%")
                elif op == "is_null":
                    clauses.append(f'"{col_name}" IS NULL')
                elif op == "is_not_null":
                    clauses.append(f'"{col_name}" IS NOT NULL')
                elif op == "in":
                    values = f.get("value", [])
                    if not values:
                        clauses.append("1=0")
                    else:
                        coerced = [self._coerce_value(v, db2_type) for v in values]
                        placeholders = ", ".join("?" for _ in coerced)
                        clauses.append(f'"{col_name}" IN ({placeholders})')
                        params.extend(coerced)
                elif op == "not_in":
                    values = f.get("value", [])
                    if not values:
                        clauses.append("1=1")
                    else:
                        coerced = [self._coerce_value(v, db2_type) for v in values]
                        placeholders = ", ".join("?" for _ in coerced)
                        clauses.append(f'"{col_name}" NOT IN ({placeholders})')
                        params.extend(coerced)
                elif op == "between":
                    clauses.append(f'"{col_name}" BETWEEN ? AND ?')
                    params.append(self._coerce_value(f["value"], db2_type))
                    params.append(self._coerce_value(f["value2"], db2_type))
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
            query += f" FETCH FIRST {int(limit)} ROWS ONLY"

        def _execute() -> list[dict[str, Any]]:
            cursor = self._conn.cursor()
            cursor.execute(query, tuple(params) if params else None)
            cols = [desc[0].strip() for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
            return [dict(zip(cols, row)) for row in rows]

        return await asyncio.to_thread(_execute)
