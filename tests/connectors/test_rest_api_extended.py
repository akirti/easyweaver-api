"""Comprehensive tests for RestAPIConnector — covering _authenticate, execute_query,
get_schema, preview_table, get_table_schema, test_connection, _fetch_endpoint,
_extract_path, _flatten_dict, _infer_columns_from_rows, connect/disconnect."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Module-level helpers ──────────────────────────────────────────────────────

def _make_connector(credentials: dict | None = None):
    from easyweaver.connectors.implementations.rest_api import RestAPIConnector

    creds = credentials or {
        "base_url": "https://api.example.com",
        "endpoints": {
            "users": {"path": "/users", "method": "GET"},
        },
    }
    connector = RestAPIConnector.__new__(RestAPIConnector)
    connector.credentials = creds
    connector._auth_headers = {}
    connector._base_url = creds.get("base_url", "https://api.example.com").rstrip("/")
    connector._endpoints = creds.get("endpoints", {})
    connector._extra_headers = creds.get("headers", {})
    connector._client = None
    return connector


def _make_mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.json = MagicMock(return_value=json_data)
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


# ── _extract_path ─────────────────────────────────────────────────────────────

class TestExtractPath:
    def test_empty_path_returns_data(self):
        from easyweaver.connectors.implementations.rest_api import _extract_path

        assert _extract_path({"key": "val"}, "") == {"key": "val"}

    def test_single_key(self):
        from easyweaver.connectors.implementations.rest_api import _extract_path

        assert _extract_path({"data": [1, 2, 3]}, "data") == [1, 2, 3]

    def test_nested_path(self):
        from easyweaver.connectors.implementations.rest_api import _extract_path

        data = {"meta": {"results": [{"id": 1}]}}
        assert _extract_path(data, "meta.results") == [{"id": 1}]

    def test_list_index_access(self):
        from easyweaver.connectors.implementations.rest_api import _extract_path

        assert _extract_path([10, 20, 30], "1") == 20

    def test_invalid_path_raises(self):
        from easyweaver.connectors.implementations.rest_api import _extract_path

        with pytest.raises(KeyError):
            _extract_path({"a": 1}, "b")


# ── _flatten_dict ─────────────────────────────────────────────────────────────

class TestFlattenDict:
    def test_flat_dict_unchanged(self):
        from easyweaver.connectors.implementations.rest_api import _flatten_dict

        result = _flatten_dict({"name": "Alice", "age": 30})
        assert result == {"name": "Alice", "age": 30}

    def test_nested_dict(self):
        from easyweaver.connectors.implementations.rest_api import _flatten_dict

        result = _flatten_dict({"address": {"city": "NYC", "zip": "10001"}})
        assert result["address.city"] == "NYC"
        assert result["address.zip"] == "10001"

    def test_list_of_dicts_indexed(self):
        from easyweaver.connectors.implementations.rest_api import _flatten_dict

        result = _flatten_dict({"items": [{"name": "a"}, {"name": "b"}]})
        assert result["items[0].name"] == "a"
        assert result["items[1].name"] == "b"

    def test_list_of_primitives_kept_as_is(self):
        from easyweaver.connectors.implementations.rest_api import _flatten_dict

        result = _flatten_dict({"tags": [1, 2, 3]})
        assert result["tags"] == [1, 2, 3]


# ── _infer_columns_from_rows ──────────────────────────────────────────────────

class TestInferColumnsFromRows:
    def test_basic_type_inference(self):
        from easyweaver.connectors.implementations.rest_api import _infer_columns_from_rows

        rows = [{"name": "Alice", "age": 30, "active": True}]
        cols = _infer_columns_from_rows(rows)
        col_map = {c["name"]: c["type"] for c in cols}
        assert col_map["name"] == "text"
        assert col_map["age"] == "integer"
        assert col_map["active"] == "boolean"

    def test_mixed_types_produce_mixed(self):
        from easyweaver.connectors.implementations.rest_api import _infer_columns_from_rows

        rows = [{"val": "text"}, {"val": 42}]
        cols = _infer_columns_from_rows(rows)
        col = next(c for c in cols if c["name"] == "val")
        assert col["type"] == "mixed"

    def test_nullable_field_detected(self):
        from easyweaver.connectors.implementations.rest_api import _infer_columns_from_rows

        rows = [{"name": "Alice"}, {"name": None}]
        cols = _infer_columns_from_rows(rows)
        col = next(c for c in cols if c["name"] == "name")
        assert col["nullable"] is True

    def test_empty_rows_returns_empty(self):
        from easyweaver.connectors.implementations.rest_api import _infer_columns_from_rows

        assert _infer_columns_from_rows([]) == []


# ── _authenticate ─────────────────────────────────────────────────────────────

class TestAuthenticate:
    @pytest.mark.anyio
    async def test_none_auth_no_headers(self):
        connector = _make_connector()
        connector.credentials["auth_type"] = "none"
        connector._client = MagicMock()
        await connector._authenticate()
        assert connector._auth_headers == {}

    @pytest.mark.anyio
    async def test_bearer_auth_sets_header(self):
        connector = _make_connector()
        connector.credentials["auth_type"] = "bearer"
        connector.credentials["bearer_token"] = "mytoken123"
        connector._client = MagicMock()
        await connector._authenticate()
        assert connector._auth_headers == {"Authorization": "Bearer mytoken123"}

    @pytest.mark.anyio
    async def test_basic_auth_sets_header(self):
        import base64
        connector = _make_connector()
        connector.credentials["auth_type"] = "basic"
        connector.credentials["basic_user"] = "user"
        connector.credentials["basic_password"] = "pass"
        connector._client = MagicMock()
        await connector._authenticate()
        expected = base64.b64encode(b"user:pass").decode()
        assert connector._auth_headers == {"Authorization": f"Basic {expected}"}

    @pytest.mark.anyio
    async def test_api_key_auth(self):
        connector = _make_connector()
        connector.credentials["auth_type"] = "api_key"
        connector.credentials["api_key_header"] = "X-API-Key"
        connector.credentials["api_key_value"] = "secret-key"
        connector._client = MagicMock()
        await connector._authenticate()
        assert connector._auth_headers == {"X-API-Key": "secret-key"}

    @pytest.mark.anyio
    async def test_api_key_default_header(self):
        connector = _make_connector()
        connector.credentials["auth_type"] = "api_key"
        connector.credentials["api_key_value"] = "my-key"
        connector._client = MagicMock()
        await connector._authenticate()
        assert "X-API-Key" in connector._auth_headers


# ── _request_headers ─────────────────────────────────────────────────────────

class TestRequestHeaders:
    def test_merges_extra_and_auth_headers(self):
        connector = _make_connector()
        connector._extra_headers = {"X-Custom": "value"}
        connector._auth_headers = {"Authorization": "Bearer token"}
        headers = connector._request_headers()
        assert headers["X-Custom"] == "value"
        assert headers["Authorization"] == "Bearer token"

    def test_auth_overrides_extra(self):
        connector = _make_connector()
        connector._extra_headers = {"Authorization": "Basic old"}
        connector._auth_headers = {"Authorization": "Bearer new"}
        headers = connector._request_headers()
        assert headers["Authorization"] == "Bearer new"


# ── _fetch_endpoint ───────────────────────────────────────────────────────────

class TestFetchEndpoint:
    @pytest.mark.anyio
    async def test_returns_list_of_rows(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        resp = _make_mock_response([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        mock_client.request = AsyncMock(return_value=resp)
        connector._client = mock_client

        rows = await connector._fetch_endpoint({"path": "/users", "method": "GET"})
        assert len(rows) == 2
        assert rows[0]["id"] == 1

    @pytest.mark.anyio
    async def test_returns_single_dict_wrapped(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        resp = _make_mock_response({"id": 1, "name": "Alice"})
        mock_client.request = AsyncMock(return_value=resp)
        connector._client = mock_client

        rows = await connector._fetch_endpoint({"path": "/user/1", "method": "GET"})
        assert len(rows) == 1

    @pytest.mark.anyio
    async def test_applies_data_path(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        resp = _make_mock_response({"data": {"results": [{"id": 1}]}})
        mock_client.request = AsyncMock(return_value=resp)
        connector._client = mock_client

        rows = await connector._fetch_endpoint({
            "path": "/api",
            "method": "GET",
            "data_path": "data.results",
        })
        assert len(rows) == 1
        assert rows[0]["id"] == 1

    @pytest.mark.anyio
    async def test_primitive_response_wrapped(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        resp = _make_mock_response(42)
        mock_client.request = AsyncMock(return_value=resp)
        connector._client = mock_client

        rows = await connector._fetch_endpoint({"path": "/count"})
        assert rows == [{"value": 42}]


# ── execute_query ─────────────────────────────────────────────────────────────

class TestExecuteQuery:
    @pytest.mark.anyio
    async def test_basic_query(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        resp = _make_mock_response([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        mock_client.request = AsyncMock(return_value=resp)
        connector._client = mock_client

        result = await connector.execute_query(table="users")
        assert len(result) == 2

    @pytest.mark.anyio
    async def test_column_selection(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        resp = _make_mock_response([{"id": 1, "name": "Alice", "email": "a@b.com"}])
        mock_client.request = AsyncMock(return_value=resp)
        connector._client = mock_client

        result = await connector.execute_query(table="users", columns=["id", "name"])
        assert "email" not in result[0]
        assert "id" in result[0]

    @pytest.mark.anyio
    async def test_with_limit(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        resp = _make_mock_response([{"id": i} for i in range(10)])
        mock_client.request = AsyncMock(return_value=resp)
        connector._client = mock_client

        result = await connector.execute_query(table="users", limit=3)
        assert len(result) == 3

    @pytest.mark.anyio
    async def test_with_sort_asc(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        resp = _make_mock_response([
            {"id": 2, "name": "Bob"},
            {"id": 1, "name": "Alice"},
        ])
        mock_client.request = AsyncMock(return_value=resp)
        connector._client = mock_client

        result = await connector.execute_query(
            table="users",
            sort=[{"column": "name", "direction": "asc"}],
        )
        # Alice should come before Bob
        assert result[0]["name"] == "Alice"

    @pytest.mark.anyio
    async def test_unknown_endpoint_raises(self):
        connector = _make_connector()
        with pytest.raises(ValueError, match="Unknown endpoint"):
            await connector.execute_query(table="nonexistent")


# ── get_schema ────────────────────────────────────────────────────────────────

class TestGetSchema:
    @pytest.mark.anyio
    async def test_returns_endpoints_as_tables(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        resp = _make_mock_response([{"id": 1, "name": "Alice"}])
        mock_client.request = AsyncMock(return_value=resp)
        connector._client = mock_client

        result = await connector.get_schema()
        assert len(result) == 1
        assert result[0]["name"] == "users"
        assert result[0]["row_estimate"] == 1

    @pytest.mark.anyio
    async def test_failed_endpoint_returns_empty_schema(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=Exception("Network error"))
        connector._client = mock_client

        result = await connector.get_schema()
        assert result[0]["columns"] == []
        assert result[0]["row_estimate"] == 0


# ── get_table_schema ──────────────────────────────────────────────────────────

class TestGetTableSchema:
    @pytest.mark.anyio
    async def test_returns_schema_for_endpoint(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        resp = _make_mock_response([{"id": 1, "name": "Alice"}])
        mock_client.request = AsyncMock(return_value=resp)
        connector._client = mock_client

        result = await connector.get_table_schema("users")
        assert result["name"] == "users"

    @pytest.mark.anyio
    async def test_returns_empty_for_unknown_endpoint(self):
        connector = _make_connector()
        result = await connector.get_table_schema("nonexistent")
        assert result["columns"] == []

    @pytest.mark.anyio
    async def test_handles_fetch_error(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=Exception("fail"))
        connector._client = mock_client

        result = await connector.get_table_schema("users")
        assert result["columns"] == []


# ── preview_table ─────────────────────────────────────────────────────────────

class TestPreviewTable:
    @pytest.mark.anyio
    async def test_returns_preview(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        resp = _make_mock_response([{"id": i} for i in range(10)])
        mock_client.request = AsyncMock(return_value=resp)
        connector._client = mock_client

        result = await connector.preview_table("users", limit=5)
        assert len(result["rows"]) == 5
        assert result["total_sampled"] == 5

    @pytest.mark.anyio
    async def test_unknown_endpoint_returns_empty(self):
        connector = _make_connector()
        result = await connector.preview_table("nonexistent")
        assert result == {"columns": [], "rows": [], "total_sampled": 0}


# ── connect / disconnect ──────────────────────────────────────────────────────

class TestConnectDisconnect:
    @pytest.mark.anyio
    async def test_connect_creates_client(self):
        connector = _make_connector({"base_url": "https://api.example.com"})
        connector.credentials["auth_type"] = "none"

        with patch("httpx.AsyncClient") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.post = AsyncMock()
            mock_cls.return_value = mock_instance
            await connector.connect()

        assert connector._client is mock_instance

    @pytest.mark.anyio
    async def test_disconnect_closes_client(self):
        connector = _make_connector()
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        connector._client = mock_client

        await connector.disconnect()

        mock_client.aclose.assert_called_once()
        assert connector._client is None
        assert connector._auth_headers == {}

    @pytest.mark.anyio
    async def test_disconnect_noop_when_no_client(self):
        connector = _make_connector()
        connector._client = None
        # Should not raise
        await connector.disconnect()


# ── test_connection ───────────────────────────────────────────────────────────

class TestTestConnection:
    @pytest.mark.anyio
    async def test_success_no_endpoints(self):
        connector = _make_connector({"base_url": "https://api.example.com"})
        connector.credentials["auth_type"] = "none"
        connector._endpoints = {}

        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await connector.test_connection()

        assert result["success"] is True

    @pytest.mark.anyio
    async def test_success_with_endpoint(self):
        connector = _make_connector()
        connector.credentials["auth_type"] = "none"

        mock_resp = _make_mock_response([])
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)
        mock_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await connector.test_connection()

        assert result["success"] is True

    @pytest.mark.anyio
    async def test_failure(self):
        connector = _make_connector()
        connector.credentials["auth_type"] = "none"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=Exception("Timeout"))
        mock_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await connector.test_connection()

        assert result["success"] is False
        assert "Timeout" in result["message"]
