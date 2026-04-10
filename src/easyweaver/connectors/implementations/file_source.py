import asyncio
import io
import time
from typing import Any

import polars as pl
import structlog

from easyweaver.connectors.base import BaseConnector

logger = structlog.get_logger()

_POLARS_TYPE_MAP = {
    pl.Int8: "integer",
    pl.Int16: "integer",
    pl.Int32: "integer",
    pl.Int64: "integer",
    pl.UInt8: "integer",
    pl.UInt16: "integer",
    pl.UInt32: "integer",
    pl.UInt64: "integer",
    pl.Float32: "float",
    pl.Float64: "float",
    pl.Boolean: "boolean",
    pl.Date: "date",
    pl.Datetime: "datetime",
    pl.Time: "time",
    pl.Utf8: "text",
    pl.String: "text",
}


def _polars_type_to_str(dtype: pl.DataType) -> str:
    return _POLARS_TYPE_MAP.get(type(dtype), "text")


def _parse_file(data: bytes, file_format: str) -> pl.DataFrame:
    buf = io.BytesIO(data)
    if file_format == "csv":
        return pl.read_csv(buf, infer_schema_length=1000)
    elif file_format == "json":
        text = data.decode("utf-8").strip()
        if text.startswith("["):
            return pl.read_json(io.BytesIO(data))
        else:
            return pl.read_ndjson(io.BytesIO(data))
    elif file_format == "xlsx":
        return pl.read_excel(buf, engine="openpyxl")
    elif file_format == "xls":
        return pl.read_excel(buf, engine="xlrd")
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


class FileConnector(BaseConnector):
    def __init__(self, credentials: dict[str, Any]):
        super().__init__(credentials)
        self._df: pl.DataFrame | None = None
        self._gcp_path: str = credentials["gcp_path"]
        self._file_format: str = credentials["file_format"]
        self._original_filename: str = credentials.get("original_filename", "data")

    def _download_and_parse(self) -> pl.DataFrame:
        from easyweaver.storage.gcs_client import get_gcs_client

        client = get_gcs_client()
        blob = client._bucket.blob(self._gcp_path)
        data = blob.download_as_bytes()
        return _parse_file(data, self._file_format)

    async def connect(self) -> None:
        self._df = await asyncio.to_thread(self._download_and_parse)
        logger.info(
            "file_source_connected",
            path=self._gcp_path,
            rows=len(self._df),
            columns=len(self._df.columns),
        )

    async def disconnect(self) -> None:
        self._df = None

    async def test_connection(self) -> dict[str, Any]:
        start = time.monotonic()
        try:
            df = await asyncio.to_thread(self._download_and_parse)
            latency = round((time.monotonic() - start) * 1000, 1)
            return {
                "success": True,
                "latency_ms": latency,
                "message": f"File parsed: {len(df)} rows, {len(df.columns)} columns",
            }
        except Exception as e:
            return {"success": False, "latency_ms": None, "message": str(e)}

    async def get_schema(self) -> list[dict[str, Any]]:
        assert self._df is not None
        columns = []
        for col_name in self._df.columns:
            dtype = self._df[col_name].dtype
            columns.append(
                {
                    "name": col_name,
                    "type": _polars_type_to_str(dtype),
                    "nullable": self._df[col_name].null_count() > 0,
                    "primary_key": False,
                }
            )
        table_name = self._original_filename.rsplit(".", 1)[0]
        return [
            {
                "name": table_name,
                "row_estimate": len(self._df),
                "columns": columns,
            }
        ]

    async def get_table_schema(self, table_name: str) -> dict[str, Any]:
        tables = await self.get_schema()
        return tables[0]

    async def preview_table(self, table_name: str, limit: int = 50) -> dict[str, Any]:
        assert self._df is not None
        preview_df = self._df.head(limit)
        columns = [
            {"name": col, "type": _polars_type_to_str(self._df[col].dtype)}
            for col in self._df.columns
        ]
        return {
            "columns": columns,
            "rows": preview_df.to_dicts(),
            "total_sampled": len(preview_df),
        }

    async def get_distinct_values(
        self, table: str, column: str, limit: int = 500
    ) -> dict[str, Any]:
        assert self._df is not None
        values = (
            self._df[column]
            .drop_nulls()
            .unique()
            .sort()
            .head(limit + 1)
            .to_list()
        )
        truncated = len(values) > limit
        if truncated:
            values = values[:limit]
        return {
            "values": values,
            "truncated": truncated,
            "total_count": len(values) if not truncated else None,
        }

    async def execute_query(
        self,
        table: str,
        columns: list[str] | None = None,
        filters: list[dict] | None = None,
        sort: list[dict] | None = None,
        limit: int | None = None,
        filter_logic: str = "and",
    ) -> list[dict[str, Any]]:
        assert self._df is not None
        df = self._df

        if columns:
            valid_cols = [c for c in columns if c in df.columns]
            if valid_cols:
                df = df.select(valid_cols)

        if sort:
            sort_cols = []
            sort_desc = []
            for s in sort:
                if s["column"] in df.columns:
                    sort_cols.append(s["column"])
                    sort_desc.append(s.get("direction", "asc") == "desc")
            if sort_cols:
                df = df.sort(sort_cols, descending=sort_desc)

        if limit:
            df = df.head(limit)

        return df.to_dicts()
