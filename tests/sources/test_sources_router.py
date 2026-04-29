"""Tests for easyweaver.sources.router using TestClient."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyweaver.core.exceptions import NotFoundError
from easyweaver.core.middleware import setup_middleware
from easyweaver.dependencies import get_db
from easyweaver.sources.models import DataSource
from easyweaver.sources.router import router

_FERNET_KEY = Fernet.generate_key().decode()


def _make_app() -> FastAPI:
    app = FastAPI()
    setup_middleware(app)
    app.include_router(router, prefix="/sources")
    return app


def _make_source(source_id: str | None = None, name: str = "Test Source") -> DataSource:
    from easyweaver.core.security import encrypt_credentials

    uid = uuid.UUID(source_id) if source_id else uuid.uuid4()
    encrypted = encrypt_credentials(json.dumps({"type": "postgres", "host": "h", "password": "p"}))
    return DataSource(
        id=uid,
        name=name,
        source_type="postgres",
        encrypted_credentials=encrypted,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture(autouse=True)
def patch_fernet(monkeypatch):
    from easyweaver.core import security
    import easyweaver.settings as settings_module

    monkeypatch.setattr(settings_module.settings, "fernet_key", _FERNET_KEY)
    security._fernet = None
    yield
    security._fernet = None


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def app(mock_db):
    _app = _make_app()
    _app.dependency_overrides[get_db] = lambda: mock_db
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /sources
# ---------------------------------------------------------------------------


class TestListSourcesEndpoint:
    def test_returns_empty_list(self, client):
        with patch("easyweaver.sources.router.service.list_sources", new=AsyncMock(return_value=[])):
            resp = client.get("/sources")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_list_of_sources(self, client):
        sources = [_make_source(), _make_source()]
        with patch("easyweaver.sources.router.service.list_sources", new=AsyncMock(return_value=sources)):
            resp = client.get("/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all("id" in s for s in data)
        assert all("name" in s for s in data)

    def test_response_excludes_credentials(self, client):
        source = _make_source()
        with patch("easyweaver.sources.router.service.list_sources", new=AsyncMock(return_value=[source])):
            resp = client.get("/sources")
        data = resp.json()[0]
        assert "encrypted_credentials" not in data
        assert "credentials" not in data


# ---------------------------------------------------------------------------
# GET /sources/{source_id}
# ---------------------------------------------------------------------------


class TestGetSourceEndpoint:
    def test_returns_source_when_found(self, client):
        sid = str(uuid.uuid4())
        source = _make_source(sid)
        with patch("easyweaver.sources.router.service.get_source", new=AsyncMock(return_value=source)):
            resp = client.get(f"/sources/{sid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == sid

    def test_returns_404_when_not_found(self, client):
        sid = str(uuid.uuid4())
        with patch(
            "easyweaver.sources.router.service.get_source",
            new=AsyncMock(side_effect=NotFoundError("DataSource", sid)),
        ):
            resp = client.get(f"/sources/{sid}")
        assert resp.status_code == 404

    def test_returns_422_for_invalid_uuid(self, client):
        resp = client.get("/sources/not-a-uuid")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /sources
# ---------------------------------------------------------------------------


class TestCreateSourceEndpoint:
    def test_create_source_returns_201(self, client):
        source = _make_source(name="New Source")
        with patch("easyweaver.sources.router.service.create_source", new=AsyncMock(return_value=source)):
            resp = client.post(
                "/sources",
                json={
                    "name": "New Source",
                    "source_type": "postgres",
                    "credentials": {
                        "type": "postgres",
                        "host": "localhost",
                        "port": 5432,
                        "database": "testdb",
                        "user": "user",
                        "password": "pass",
                    },
                },
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "New Source"

    def test_create_source_missing_name_returns_422(self, client):
        resp = client.post(
            "/sources",
            json={
                "source_type": "postgres",
                "credentials": {"type": "postgres", "host": "h", "database": "db", "user": "u", "password": "p"},
            },
        )
        assert resp.status_code == 422

    def test_create_source_invalid_type_returns_422(self, client):
        resp = client.post(
            "/sources",
            json={
                "name": "Invalid",
                "source_type": "oracle",
                "credentials": {"type": "postgres", "host": "h", "database": "db", "user": "u", "password": "p"},
            },
        )
        assert resp.status_code == 422

    def test_create_mongodb_source(self, client):
        source = _make_source(name="Mongo Source")
        source.source_type = "mongodb"
        with patch("easyweaver.sources.router.service.create_source", new=AsyncMock(return_value=source)):
            resp = client.post(
                "/sources",
                json={
                    "name": "Mongo Source",
                    "source_type": "mongodb",
                    "credentials": {
                        "type": "mongodb",
                        "connection_string": "mongodb://localhost:27017",
                        "database": "testdb",
                    },
                },
            )
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# PUT /sources/{source_id}
# ---------------------------------------------------------------------------


class TestUpdateSourceEndpoint:
    def test_update_name_returns_200(self, client):
        sid = str(uuid.uuid4())
        source = _make_source(sid, name="Updated Name")
        with patch("easyweaver.sources.router.service.update_source", new=AsyncMock(return_value=source)):
            resp = client.put(f"/sources/{sid}", json={"name": "Updated Name"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Name"

    def test_update_not_found_returns_404(self, client):
        sid = str(uuid.uuid4())
        with patch(
            "easyweaver.sources.router.service.update_source",
            new=AsyncMock(side_effect=NotFoundError("DataSource", sid)),
        ):
            resp = client.put(f"/sources/{sid}", json={"name": "New"})
        assert resp.status_code == 404

    def test_update_invalid_uuid_returns_422(self, client):
        resp = client.put("/sources/invalid-uuid", json={"name": "New"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /sources/{source_id}
# ---------------------------------------------------------------------------


class TestDeleteSourceEndpoint:
    def test_delete_returns_204(self, client):
        sid = str(uuid.uuid4())
        with patch("easyweaver.sources.router.service.delete_source", new=AsyncMock(return_value=None)):
            resp = client.delete(f"/sources/{sid}")
        assert resp.status_code == 204

    def test_delete_not_found_returns_404(self, client):
        sid = str(uuid.uuid4())
        with patch(
            "easyweaver.sources.router.service.delete_source",
            new=AsyncMock(side_effect=NotFoundError("DataSource", sid)),
        ):
            resp = client.delete(f"/sources/{sid}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /sources/{source_id}/test
# ---------------------------------------------------------------------------


class TestConnectionTestEndpoint:
    def test_test_connection_success(self, client):
        sid = str(uuid.uuid4())
        test_result = {"success": True, "latency_ms": 12.5, "message": "Connected"}
        with patch(
            "easyweaver.sources.router.service.test_source_connection",
            new=AsyncMock(return_value=test_result),
        ):
            resp = client.post(f"/sources/{sid}/test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_test_connection_not_found_returns_404(self, client):
        sid = str(uuid.uuid4())
        with patch(
            "easyweaver.sources.router.service.test_source_connection",
            new=AsyncMock(side_effect=NotFoundError("DataSource", sid)),
        ):
            resp = client.post(f"/sources/{sid}/test")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /sources/{source_id}/schema
# ---------------------------------------------------------------------------


class TestGetSchemaEndpoint:
    def test_get_schema_returns_schema(self, client):
        sid = str(uuid.uuid4())
        source = _make_source(sid)
        schema = {"tables": ["users", "orders"]}

        mock_connector = MagicMock()
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)
        mock_connector.get_schema = AsyncMock(return_value=schema)

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("easyweaver.sources.router.service.get_source", new=AsyncMock(return_value=source)), \
             patch("easyweaver.sources.router.service.get_source_credentials", return_value={"host": "h"}), \
             patch("easyweaver.sources.router.get_redis", new=AsyncMock(return_value=mock_redis)), \
             patch("easyweaver.connectors.registry.get_connector", return_value=mock_connector):
            resp = client.get(f"/sources/{sid}/schema")

        assert resp.status_code == 200
        assert resp.json() == schema

    def test_get_schema_serves_from_cache(self, client):
        sid = str(uuid.uuid4())
        cached_schema = {"tables": ["cached_table"]}

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_schema))

        with patch("easyweaver.sources.router.get_redis", new=AsyncMock(return_value=mock_redis)):
            resp = client.get(f"/sources/{sid}/schema")

        assert resp.status_code == 200
        assert resp.json() == cached_schema


# ---------------------------------------------------------------------------
# GET /sources/{source_id}/tables/{table}/columns/{column}/distinct
# ---------------------------------------------------------------------------


class TestDistinctValuesEndpoint:
    def test_returns_distinct_values(self, client):
        sid = str(uuid.uuid4())
        source = _make_source(sid)
        distinct_result = {"values": ["a", "b", "c"], "truncated": False, "total_count": 3}

        mock_connector = MagicMock()
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)
        mock_connector.get_distinct_values = AsyncMock(return_value=distinct_result)

        with patch("easyweaver.sources.router.service.get_source", new=AsyncMock(return_value=source)), \
             patch("easyweaver.sources.router.service.get_source_credentials", return_value={}), \
             patch("easyweaver.connectors.registry.get_connector", return_value=mock_connector):
            resp = client.get(f"/sources/{sid}/tables/public.users/columns/status/distinct")

        assert resp.status_code == 200
        assert resp.json()["values"] == ["a", "b", "c"]

    def test_limit_capped_at_5000(self, client):
        sid = str(uuid.uuid4())
        source = _make_source(sid)
        captured = {}

        class CapturingConnector:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get_distinct_values(self, table, column, limit):
                captured["limit"] = limit
                return {"values": [], "truncated": False, "total_count": 0}

        with patch("easyweaver.sources.router.service.get_source", new=AsyncMock(return_value=source)), \
             patch("easyweaver.sources.router.service.get_source_credentials", return_value={}), \
             patch("easyweaver.connectors.registry.get_connector", return_value=CapturingConnector()):
            resp = client.get(f"/sources/{sid}/tables/t/columns/c/distinct?limit=99999")

        assert resp.status_code == 200
        assert captured["limit"] == 5000


# ---------------------------------------------------------------------------
# GET /sources/{source_id}/schema/{table_name}
# ---------------------------------------------------------------------------


class TestGetTableSchemaEndpoint:
    def test_returns_table_schema(self, client):
        sid = str(uuid.uuid4())
        source = _make_source(sid)
        table_schema = {"columns": [{"name": "id", "type": "integer"}]}

        mock_connector = MagicMock()
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)
        mock_connector.get_table_schema = AsyncMock(return_value=table_schema)

        with patch("easyweaver.sources.router.service.get_source", new=AsyncMock(return_value=source)), \
             patch("easyweaver.sources.router.service.get_source_credentials", return_value={"host": "h"}), \
             patch("easyweaver.connectors.registry.get_connector", return_value=mock_connector):
            resp = client.get(f"/sources/{sid}/schema/users")

        assert resp.status_code == 200
        assert resp.json() == table_schema

    def test_passes_table_name_to_connector(self, client):
        sid = str(uuid.uuid4())
        source = _make_source(sid)
        captured = {}

        mock_connector = MagicMock()
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)

        async def capture_table(table_name):
            captured["table"] = table_name
            return {"columns": []}

        mock_connector.get_table_schema = capture_table

        with patch("easyweaver.sources.router.service.get_source", new=AsyncMock(return_value=source)), \
             patch("easyweaver.sources.router.service.get_source_credentials", return_value={}), \
             patch("easyweaver.connectors.registry.get_connector", return_value=mock_connector):
            resp = client.get(f"/sources/{sid}/schema/my_table")

        assert resp.status_code == 200
        assert captured["table"] == "my_table"

    def test_source_not_found_returns_404(self, client):
        sid = str(uuid.uuid4())
        from easyweaver.core.exceptions import NotFoundError

        with patch(
            "easyweaver.sources.router.service.get_source",
            new=AsyncMock(side_effect=NotFoundError("DataSource", sid)),
        ):
            resp = client.get(f"/sources/{sid}/schema/users")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /sources/{source_id}/preview/{table_name}
# ---------------------------------------------------------------------------


class TestPreviewTableEndpoint:
    def test_returns_preview_rows(self, client):
        sid = str(uuid.uuid4())
        source = _make_source(sid)
        preview_data = {"rows": [{"id": 1, "name": "Alice"}], "total": 1}

        mock_connector = MagicMock()
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)
        mock_connector.preview_table = AsyncMock(return_value=preview_data)

        with patch("easyweaver.sources.router.service.get_source", new=AsyncMock(return_value=source)), \
             patch("easyweaver.sources.router.service.get_source_credentials", return_value={}), \
             patch("easyweaver.connectors.registry.get_connector", return_value=mock_connector):
            resp = client.get(f"/sources/{sid}/preview/users")

        assert resp.status_code == 200
        assert resp.json() == preview_data

    def test_passes_table_name_to_preview(self, client):
        sid = str(uuid.uuid4())
        source = _make_source(sid)
        captured = {}

        mock_connector = MagicMock()
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)

        async def capture_preview(table_name):
            captured["table"] = table_name
            return {"rows": [], "total": 0}

        mock_connector.preview_table = capture_preview

        with patch("easyweaver.sources.router.service.get_source", new=AsyncMock(return_value=source)), \
             patch("easyweaver.sources.router.service.get_source_credentials", return_value={}), \
             patch("easyweaver.connectors.registry.get_connector", return_value=mock_connector):
            resp = client.get(f"/sources/{sid}/preview/orders")

        assert resp.status_code == 200
        assert captured["table"] == "orders"

    def test_source_not_found_returns_404(self, client):
        sid = str(uuid.uuid4())
        from easyweaver.core.exceptions import NotFoundError

        with patch(
            "easyweaver.sources.router.service.get_source",
            new=AsyncMock(side_effect=NotFoundError("DataSource", sid)),
        ):
            resp = client.get(f"/sources/{sid}/preview/users")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /sources/upload
# ---------------------------------------------------------------------------


class TestUploadFileSourceEndpoint:
    def test_unsupported_file_type_returns_400(self, client):
        import io

        resp = client.post(
            "/sources/upload",
            data={"name": "My File"},
            files={"file": ("data.pdf", io.BytesIO(b"pdf content"), "application/pdf")},
        )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_valid_csv_upload_creates_source(self, client):
        import io

        source = _make_source(name="CSV Source")
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_gcs = MagicMock()
        mock_gcs._bucket = mock_bucket

        with patch("easyweaver.sources.router.service.create_source", new=AsyncMock(return_value=source)), \
             patch("easyweaver.storage.gcs_client.get_gcs_client", return_value=mock_gcs):
            resp = client.post(
                "/sources/upload",
                data={"name": "CSV Source"},
                files={"file": ("data.csv", io.BytesIO(b"id,name\n1,Alice"), "text/csv")},
            )

        assert resp.status_code == 201

    def test_json_file_accepted(self, client):
        import io

        source = _make_source(name="JSON Source")
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_gcs = MagicMock()
        mock_gcs._bucket = mock_bucket

        with patch("easyweaver.sources.router.service.create_source", new=AsyncMock(return_value=source)), \
             patch("easyweaver.storage.gcs_client.get_gcs_client", return_value=mock_gcs):
            resp = client.post(
                "/sources/upload",
                data={"name": "JSON Source"},
                files={"file": ("data.json", io.BytesIO(b'[{"id": 1}]'), "application/json")},
            )

        assert resp.status_code == 201

    def test_xlsx_file_accepted(self, client):
        import io

        source = _make_source(name="Excel Source")
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_gcs = MagicMock()
        mock_gcs._bucket = mock_bucket

        with patch("easyweaver.sources.router.service.create_source", new=AsyncMock(return_value=source)), \
             patch("easyweaver.storage.gcs_client.get_gcs_client", return_value=mock_gcs):
            resp = client.post(
                "/sources/upload",
                data={"name": "Excel Source"},
                files={"file": ("data.xlsx", io.BytesIO(b"fake xlsx"), "application/vnd.openxmlformats")},
            )

        assert resp.status_code == 201

    def test_file_without_extension_returns_400(self, client):
        import io

        resp = client.post(
            "/sources/upload",
            data={"name": "No Ext"},
            files={"file": ("datafile", io.BytesIO(b"content"), "application/octet-stream")},
        )
        assert resp.status_code == 400
