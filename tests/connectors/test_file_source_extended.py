"""Extended tests for FileConnector — covering execute_query, get_schema,
preview_table, get_table_schema, test_connection, connect/disconnect,
_parse_file, _polars_type_to_str."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest


@pytest.fixture
def file_connector():
    from easyweaver.connectors.implementations.file_source import FileConnector

    connector = FileConnector.__new__(FileConnector)
    connector.credentials = {
        "gcp_path": "gs://bucket/data.csv",
        "file_format": "csv",
        "original_filename": "data.csv",
    }
    connector._gcp_path = "gs://bucket/data.csv"
    connector._file_format = "csv"
    connector._original_filename = "data.csv"
    connector._df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [90.5, 85.0, 78.3, 92.1, 88.7],
        "active": [True, False, True, True, False],
        "city": ["NYC", "LA", "NYC", "Chicago", None],
    })
    return connector


# ── get_schema ────────────────────────────────────────────────────────────────

class TestGetSchema:
    @pytest.mark.anyio
    async def test_returns_columns_with_types(self, file_connector):
        result = await file_connector.get_schema()
        assert len(result) == 1
        table = result[0]
        assert table["name"] == "data"  # filename without extension
        col_map = {c["name"]: c["type"] for c in table["columns"]}
        assert col_map["id"] == "integer"
        assert col_map["name"] == "text"
        assert col_map["score"] == "float"
        assert col_map["active"] == "boolean"

    @pytest.mark.anyio
    async def test_row_estimate_equals_df_length(self, file_connector):
        result = await file_connector.get_schema()
        assert result[0]["row_estimate"] == 5

    @pytest.mark.anyio
    async def test_nullable_column_detected(self, file_connector):
        result = await file_connector.get_schema()
        col_map = {c["name"]: c["nullable"] for c in result[0]["columns"]}
        assert col_map["city"] is True
        assert col_map["id"] is False


# ── get_table_schema ──────────────────────────────────────────────────────────

class TestGetTableSchema:
    @pytest.mark.anyio
    async def test_returns_first_table(self, file_connector):
        result = await file_connector.get_table_schema("data")
        assert result["name"] == "data"
        assert len(result["columns"]) == 5


# ── preview_table ─────────────────────────────────────────────────────────────

class TestPreviewTable:
    @pytest.mark.anyio
    async def test_returns_all_rows_when_limit_larger(self, file_connector):
        result = await file_connector.preview_table("data", limit=100)
        assert len(result["rows"]) == 5
        assert result["total_sampled"] == 5

    @pytest.mark.anyio
    async def test_respects_limit(self, file_connector):
        result = await file_connector.preview_table("data", limit=3)
        assert len(result["rows"]) == 3
        assert result["total_sampled"] == 3

    @pytest.mark.anyio
    async def test_includes_column_schema(self, file_connector):
        result = await file_connector.preview_table("data", limit=10)
        assert "columns" in result
        assert len(result["columns"]) == 5


# ── execute_query ─────────────────────────────────────────────────────────────

class TestExecuteQuery:
    @pytest.mark.anyio
    async def test_returns_all_rows(self, file_connector):
        result = await file_connector.execute_query(table="data")
        assert len(result) == 5

    @pytest.mark.anyio
    async def test_column_selection(self, file_connector):
        result = await file_connector.execute_query(
            table="data", columns=["id", "name"]
        )
        assert all("score" not in row for row in result)
        assert all("id" in row for row in result)

    @pytest.mark.anyio
    async def test_with_limit(self, file_connector):
        result = await file_connector.execute_query(table="data", limit=2)
        assert len(result) == 2

    @pytest.mark.anyio
    async def test_with_sort_asc(self, file_connector):
        result = await file_connector.execute_query(
            table="data",
            sort=[{"column": "name", "direction": "asc"}],
        )
        names = [r["name"] for r in result]
        assert names == sorted(names)

    @pytest.mark.anyio
    async def test_with_sort_desc(self, file_connector):
        result = await file_connector.execute_query(
            table="data",
            sort=[{"column": "score", "direction": "desc"}],
        )
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.anyio
    async def test_invalid_column_selection_ignored(self, file_connector):
        # "nonexistent" is not in the df, should be silently skipped
        result = await file_connector.execute_query(
            table="data",
            columns=["id", "nonexistent"],
        )
        assert all("id" in row for row in result)

    @pytest.mark.anyio
    async def test_returns_dicts(self, file_connector):
        result = await file_connector.execute_query(table="data")
        assert all(isinstance(row, dict) for row in result)


# ── get_distinct_values ───────────────────────────────────────────────────────

class TestGetDistinctValues:
    @pytest.mark.anyio
    async def test_filters_nulls(self, file_connector):
        result = await file_connector.get_distinct_values("data", "city", limit=500)
        assert None not in result["values"]
        assert len(result["values"]) == 3  # NYC, LA, Chicago

    @pytest.mark.anyio
    async def test_values_sorted(self, file_connector):
        result = await file_connector.get_distinct_values("data", "city", limit=500)
        vals = result["values"]
        assert vals == sorted(vals)

    @pytest.mark.anyio
    async def test_not_truncated_when_under_limit(self, file_connector):
        result = await file_connector.get_distinct_values("data", "city", limit=500)
        assert result["truncated"] is False
        assert result["total_count"] == 3

    @pytest.mark.anyio
    async def test_truncated_when_over_limit(self, file_connector):
        result = await file_connector.get_distinct_values("data", "city", limit=2)
        assert result["truncated"] is True
        assert len(result["values"]) == 2
        assert result["total_count"] is None


# ── connect / disconnect ──────────────────────────────────────────────────────

class TestConnectDisconnect:
    @pytest.mark.anyio
    async def test_disconnect_clears_df(self, file_connector):
        await file_connector.disconnect()
        assert file_connector._df is None

    @pytest.mark.anyio
    async def test_connect_calls_download_and_parse(self):
        from easyweaver.connectors.implementations.file_source import FileConnector

        connector = FileConnector.__new__(FileConnector)
        connector.credentials = {
            "gcp_path": "bucket/data.csv",
            "file_format": "csv",
            "original_filename": "data.csv",
        }
        connector._gcp_path = "bucket/data.csv"
        connector._file_format = "csv"
        connector._original_filename = "data.csv"
        connector._df = None

        mock_df = pl.DataFrame({"a": [1, 2, 3]})

        with patch.object(connector, "_download_and_parse", return_value=mock_df):
            with patch("asyncio.to_thread", new=AsyncMock(return_value=mock_df)):
                await connector.connect()

        assert connector._df is not None


# ── test_connection ───────────────────────────────────────────────────────────

class TestTestConnection:
    @pytest.mark.anyio
    async def test_success(self):
        from easyweaver.connectors.implementations.file_source import FileConnector

        connector = FileConnector.__new__(FileConnector)
        connector.credentials = {
            "gcp_path": "bucket/data.csv",
            "file_format": "csv",
            "original_filename": "data.csv",
        }
        connector._gcp_path = "bucket/data.csv"
        connector._file_format = "csv"
        connector._original_filename = "data.csv"
        connector._df = None

        mock_df = pl.DataFrame({"a": [1, 2, 3]})

        with patch("asyncio.to_thread", new=AsyncMock(return_value=mock_df)):
            result = await connector.test_connection()

        assert result["success"] is True
        assert "3 rows" in result["message"]

    @pytest.mark.anyio
    async def test_failure(self):
        from easyweaver.connectors.implementations.file_source import FileConnector

        connector = FileConnector.__new__(FileConnector)
        connector.credentials = {
            "gcp_path": "bucket/data.csv",
            "file_format": "csv",
            "original_filename": "data.csv",
        }
        connector._gcp_path = "bucket/data.csv"
        connector._file_format = "csv"
        connector._original_filename = "data.csv"
        connector._df = None

        with patch("asyncio.to_thread", new=AsyncMock(side_effect=Exception("File not found"))):
            result = await connector.test_connection()

        assert result["success"] is False
        assert "File not found" in result["message"]


# ── _parse_file ───────────────────────────────────────────────────────────────

class TestParseFile:
    def test_parse_csv(self):
        from easyweaver.connectors.implementations.file_source import _parse_file

        csv_data = b"id,name\n1,Alice\n2,Bob\n"
        df = _parse_file(csv_data, "csv")
        assert df.shape[0] == 2
        assert "id" in df.columns

    def test_parse_json_array(self):
        from easyweaver.connectors.implementations.file_source import _parse_file

        json_data = b'[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'
        df = _parse_file(json_data, "json")
        assert df.shape[0] == 2

    def test_parse_ndjson(self):
        from easyweaver.connectors.implementations.file_source import _parse_file

        ndjson_data = b'{"id": 1, "name": "Alice"}\n{"id": 2, "name": "Bob"}\n'
        df = _parse_file(ndjson_data, "json")
        assert df.shape[0] == 2

    def test_unsupported_format_raises(self):
        from easyweaver.connectors.implementations.file_source import _parse_file

        with pytest.raises(ValueError, match="Unsupported file format"):
            _parse_file(b"data", "parquet")


# ── _polars_type_to_str ───────────────────────────────────────────────────────

class TestPolarsTypeToStr:
    def test_integer_types(self):
        from easyweaver.connectors.implementations.file_source import _polars_type_to_str

        for dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            assert _polars_type_to_str(dtype()) == "integer"

    def test_float_types(self):
        from easyweaver.connectors.implementations.file_source import _polars_type_to_str

        for dtype in [pl.Float32, pl.Float64]:
            assert _polars_type_to_str(dtype()) == "float"

    def test_boolean(self):
        from easyweaver.connectors.implementations.file_source import _polars_type_to_str

        assert _polars_type_to_str(pl.Boolean()) == "boolean"

    def test_string_types(self):
        from easyweaver.connectors.implementations.file_source import _polars_type_to_str

        assert _polars_type_to_str(pl.String()) == "text"

    def test_unknown_type_defaults_to_text(self):
        from easyweaver.connectors.implementations.file_source import _polars_type_to_str

        # Categorical is not in the POLARS_TYPE_MAP so falls back to "text"
        assert _polars_type_to_str(pl.Categorical()) == "text"
