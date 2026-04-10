import base64
import time
from typing import Any

import httpx
import structlog

from easyweaver.connectors.base import BaseConnector

logger = structlog.get_logger()


def _extract_path(data: Any, path: str) -> Any:
    """Navigate nested dict/list using dot-notation path.

    e.g. "data.results" -> data["data"]["results"]
    """
    if not path:
        return data
    for key in path.split("."):
        if isinstance(data, dict):
            data = data[key]
        elif isinstance(data, list) and key.isdigit():
            data = data[int(key)]
        else:
            raise KeyError(f"Cannot navigate path segment '{key}' in {type(data)}")
    return data


def _flatten_dict(obj: dict, prefix: str = "", max_depth: int = 3, depth: int = 0) -> dict:
    """Flatten nested dicts into dot-path keys, similar to MongoDB connector."""
    flat: dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict) and depth < max_depth:
            flat.update(_flatten_dict(v, f"{key}.", max_depth, depth + 1))
        elif isinstance(v, list) and v and isinstance(v[0], dict) and depth < max_depth:
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    flat.update(_flatten_dict(item, f"{key}[{i}].", max_depth, depth + 1))
        else:
            flat[key] = v
    return flat


def _infer_columns_from_rows(rows: list[dict]) -> list[dict[str, Any]]:
    """Infer column schema from a sample of flattened rows."""
    field_types: dict[str, set[str]] = {}
    for row in rows:
        for k, v in row.items():
            if k not in field_types:
                field_types[k] = set()
            field_types[k].add(type(v).__name__)

    type_map = {
        "str": "text",
        "int": "integer",
        "float": "float",
        "bool": "boolean",
        "NoneType": "null",
        "list": "array",
        "dict": "object",
    }
    columns = []
    for name, types in sorted(field_types.items()):
        types_no_null = types - {"NoneType"}
        primary = (
            next(iter(types_no_null))
            if len(types_no_null) == 1
            else ("mixed" if types_no_null else "null")
        )
        columns.append(
            {
                "name": name,
                "type": type_map.get(primary, primary),
                "nullable": "NoneType" in types,
                "primary_key": False,
            }
        )
    return columns


class RestAPIConnector(BaseConnector):
    def __init__(self, credentials: dict[str, Any]):
        super().__init__(credentials)
        self._auth_headers: dict[str, str] = {}
        self._base_url: str = credentials["base_url"].rstrip("/")
        self._endpoints: dict[str, dict] = credentials.get("endpoints", {})
        self._extra_headers: dict[str, str] = credentials.get("headers", {})
        self._client: httpx.AsyncClient | None = None

    async def _authenticate(self) -> None:
        auth_type = self.credentials.get("auth_type", "none")

        if auth_type == "none":
            pass
        elif auth_type == "bearer":
            token = self.credentials["bearer_token"]
            self._auth_headers["Authorization"] = f"Bearer {token}"
        elif auth_type == "basic":
            user = self.credentials["basic_user"]
            password = self.credentials["basic_password"]
            encoded = base64.b64encode(f"{user}:{password}".encode()).decode()
            self._auth_headers["Authorization"] = f"Basic {encoded}"
        elif auth_type == "api_key":
            header = self.credentials.get("api_key_header", "X-API-Key")
            value = self.credentials["api_key_value"]
            self._auth_headers[header] = value
        elif auth_type == "oauth2_client_credentials":
            token_url = self.credentials["oauth2_token_url"]
            client_id = self.credentials["oauth2_client_id"]
            client_secret = self.credentials["oauth2_client_secret"]
            scope = self.credentials.get("oauth2_scope", "")
            assert self._client
            payload = {"grant_type": "client_credentials"}
            if scope:
                payload["scope"] = scope
            resp = await self._client.post(
                token_url,
                data=payload,
                auth=(client_id, client_secret),
            )
            resp.raise_for_status()
            token_data = resp.json()
            self._auth_headers["Authorization"] = f"Bearer {token_data['access_token']}"
        elif auth_type == "login":
            login_url = self.credentials["login_url"]
            login_body = self.credentials.get("login_body", {})
            token_path = self.credentials.get("login_token_path", "access_token")
            assert self._client
            resp = await self._client.post(login_url, json=login_body)
            resp.raise_for_status()
            token = _extract_path(resp.json(), token_path)
            self._auth_headers["Authorization"] = f"Bearer {token}"

        logger.info("rest_api_authenticated", auth_type=auth_type)

    def _request_headers(self) -> dict[str, str]:
        headers = {**self._extra_headers, **self._auth_headers}
        return headers

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        await self._authenticate()

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self._auth_headers = {}

    async def test_connection(self) -> dict[str, Any]:
        start = time.monotonic()
        try:
            client = httpx.AsyncClient(timeout=10.0)
            self._client = client
            await self._authenticate()

            if self._endpoints:
                first_ep = next(iter(self._endpoints.values()))
                url = self._base_url + first_ep.get("path", "/")
                method = first_ep.get("method", "GET").upper()
                resp = await client.request(method, url, headers=self._request_headers())
                resp.raise_for_status()

            await client.aclose()
            self._client = None
            latency = round((time.monotonic() - start) * 1000, 1)
            return {"success": True, "latency_ms": latency, "message": "Connection successful"}
        except Exception as e:
            if self._client:
                await self._client.aclose()
                self._client = None
            return {"success": False, "latency_ms": None, "message": str(e)}

    async def _fetch_endpoint(self, ep_config: dict) -> list[dict]:
        assert self._client
        url = self._base_url + ep_config.get("path", "/")
        method = ep_config.get("method", "GET").upper()
        resp = await self._client.request(method, url, headers=self._request_headers())
        resp.raise_for_status()
        data = resp.json()

        data_path = ep_config.get("data_path", "")
        if data_path:
            data = _extract_path(data, data_path)

        if isinstance(data, list):
            return [_flatten_dict(row) if isinstance(row, dict) else {"value": row} for row in data]
        elif isinstance(data, dict):
            return [_flatten_dict(data)]
        return [{"value": data}]

    async def get_schema(self) -> list[dict[str, Any]]:
        result = []
        for name, ep_config in self._endpoints.items():
            try:
                rows = await self._fetch_endpoint(ep_config)
                sample = rows[:100]
                columns = _infer_columns_from_rows(sample) if sample else []
                result.append(
                    {
                        "name": name,
                        "row_estimate": len(rows),
                        "columns": columns,
                    }
                )
            except Exception as e:
                logger.warning("rest_api_schema_error", endpoint=name, error=str(e))
                result.append(
                    {
                        "name": name,
                        "row_estimate": 0,
                        "columns": [],
                    }
                )
        return result

    async def get_table_schema(self, table_name: str) -> dict[str, Any]:
        ep_config = self._endpoints.get(table_name)
        if not ep_config:
            return {"name": table_name, "columns": [], "row_estimate": 0}
        try:
            rows = await self._fetch_endpoint(ep_config)
            sample = rows[:100]
            columns = _infer_columns_from_rows(sample) if sample else []
            return {"name": table_name, "row_estimate": len(rows), "columns": columns}
        except Exception:
            return {"name": table_name, "columns": [], "row_estimate": 0}

    async def preview_table(self, table_name: str, limit: int = 50) -> dict[str, Any]:
        ep_config = self._endpoints.get(table_name)
        if not ep_config:
            return {"columns": [], "rows": [], "total_sampled": 0}
        rows = await self._fetch_endpoint(ep_config)
        sample = rows[:limit]
        columns = _infer_columns_from_rows(sample) if sample else []
        return {"columns": columns, "rows": sample, "total_sampled": len(sample)}

    async def get_distinct_values(
        self, table: str, column: str, limit: int = 500
    ) -> dict[str, Any]:
        return {"values": [], "truncated": False, "total_count": 0}

    async def execute_query(
        self,
        table: str,
        columns: list[str] | None = None,
        filters: list[dict] | None = None,
        sort: list[dict] | None = None,
        limit: int | None = None,
        filter_logic: str = "and",
    ) -> list[dict[str, Any]]:
        ep_config = self._endpoints.get(table)
        if not ep_config:
            raise ValueError(f"Unknown endpoint: {table}")

        rows = await self._fetch_endpoint(ep_config)

        if columns:
            rows = [{k: r.get(k) for k in columns} for r in rows]

        if sort:
            for s in reversed(sort):
                col = s["column"]
                reverse = s.get("direction", "asc") == "desc"
                rows.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=reverse)

        if limit:
            rows = rows[:limit]

        return rows
