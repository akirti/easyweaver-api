"""Tests for easyweaver.lookups.router using TestClient."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyweaver.core.middleware import setup_middleware
from easyweaver.dependencies import get_db
from easyweaver.lookups.router import router
from easyweaver.processes.models import ProcessConfiguration


def _make_app() -> FastAPI:
    app = FastAPI()
    setup_middleware(app)
    app.include_router(router, prefix="/lookups")
    return app


def _make_process_config(process_id: str, params: dict | None = None) -> ProcessConfiguration:
    return ProcessConfiguration(
        id=uuid.uuid4(),
        name="Test Process",
        user_id="user-123",
        params=params or {},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


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
# GET /lookups/{process_id}
# ---------------------------------------------------------------------------


class TestGetLookupsEndpoint:
    def test_returns_lookup_doc_when_found(self, client):
        proc_id = "proc-123"
        doc = {
            "process_id": proc_id,
            "lookups": {"status": ["active", "inactive"]},
            "references": [],
        }
        with patch("easyweaver.lookups.router.service.get_lookups", new=AsyncMock(return_value=doc)):
            resp = client.get(f"/lookups/{proc_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["process_id"] == proc_id
        assert "status" in data["lookups"]

    def test_returns_404_when_no_lookups(self, client):
        with patch("easyweaver.lookups.router.service.get_lookups", new=AsyncMock(return_value=None)):
            resp = client.get("/lookups/nonexistent-proc")

        assert resp.status_code == 404
        assert "No lookups found" in resp.json()["detail"]

    def test_process_id_is_passed_to_service(self, client):
        proc_id = "my-special-proc"
        with patch(
            "easyweaver.lookups.router.service.get_lookups",
            new=AsyncMock(return_value=None),
        ) as mock_get:
            client.get(f"/lookups/{proc_id}")

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][1] == proc_id

    def test_returns_lookups_with_multiple_params(self, client):
        proc_id = "multi-proc"
        doc = {
            "process_id": proc_id,
            "lookups": {
                "status": ["active", "inactive"],
                "region": ["north", "south", "east", "west"],
            },
            "references": [
                {"source_id": "s1", "table": "t1", "column": "status"},
                {"source_id": "s1", "table": "t1", "column": "region"},
            ],
        }
        with patch("easyweaver.lookups.router.service.get_lookups", new=AsyncMock(return_value=doc)):
            resp = client.get(f"/lookups/{proc_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["lookups"]) == 2
        assert len(data["references"]) == 2


# ---------------------------------------------------------------------------
# POST /lookups/{process_id}/refresh
# ---------------------------------------------------------------------------


class TestRefreshLookupsEndpoint:
    def test_refresh_returns_updated_lookups(self, client):
        proc_id = "proc-to-refresh"
        config = _make_process_config(
            proc_id,
            params={
                "status": {
                    "type": "select",
                    "options_source": {"source_id": "s1", "table": "t", "column": "c"},
                }
            },
        )
        updated_doc = {
            "process_id": proc_id,
            "lookups": {"status": ["a", "b"]},
            "references": [{"source_id": "s1", "table": "t", "column": "c"}],
        }

        with patch("easyweaver.lookups.router.process_service.get_configuration", new=AsyncMock(return_value=config)), \
             patch(
                 "easyweaver.lookups.router.service.build_lookups_from_params",
                 new=AsyncMock(return_value=({"status": ["a", "b"]}, [{"source_id": "s1", "table": "t", "column": "c"}])),
             ), \
             patch("easyweaver.lookups.router.service.upsert_lookups", new=AsyncMock(return_value=updated_doc)):
            resp = client.post(f"/lookups/{proc_id}/refresh")

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data["lookups"]

    def test_refresh_process_not_found_returns_404(self, client):
        from easyweaver.core.exceptions import NotFoundError

        with patch(
            "easyweaver.lookups.router.process_service.get_configuration",
            new=AsyncMock(side_effect=NotFoundError("ProcessConfiguration", "missing-proc")),
        ):
            resp = client.post("/lookups/missing-proc/refresh")

        assert resp.status_code == 404

    def test_refresh_with_no_params_returns_400(self, client):
        proc_id = "no-params-proc"
        config = _make_process_config(proc_id, params={})

        with patch("easyweaver.lookups.router.process_service.get_configuration", new=AsyncMock(return_value=config)):
            resp = client.post(f"/lookups/{proc_id}/refresh")

        assert resp.status_code == 400
        assert "no parameters" in resp.json()["detail"].lower()

    def test_refresh_with_no_select_params_returns_400(self, client):
        proc_id = "no-select-proc"
        config = _make_process_config(
            proc_id,
            params={
                "start_date": {"type": "date"},
                "count": {"type": "number"},
            },
        )

        with patch("easyweaver.lookups.router.process_service.get_configuration", new=AsyncMock(return_value=config)), \
             patch(
                 "easyweaver.lookups.router.service.build_lookups_from_params",
                 new=AsyncMock(return_value=({}, [])),
             ):
            resp = client.post(f"/lookups/{proc_id}/refresh")

        assert resp.status_code == 400
        assert "select" in resp.json()["detail"].lower()

    def test_refresh_calls_upsert_with_correct_args(self, client):
        proc_id = "upsert-test-proc"
        config = _make_process_config(
            proc_id,
            params={
                "status": {
                    "type": "select",
                    "options_source": {"source_id": "s1", "table": "t", "column": "c"},
                }
            },
        )
        lookups = {"status": ["x", "y"]}
        references = [{"source_id": "s1", "table": "t", "column": "c"}]
        updated_doc = {"process_id": proc_id, "lookups": lookups, "references": references}
        upsert_calls = []

        async def mock_upsert(db, pid, lkps, refs):
            upsert_calls.append((pid, lkps, refs))
            return updated_doc

        with patch("easyweaver.lookups.router.process_service.get_configuration", new=AsyncMock(return_value=config)), \
             patch(
                 "easyweaver.lookups.router.service.build_lookups_from_params",
                 new=AsyncMock(return_value=(lookups, references)),
             ), \
             patch("easyweaver.lookups.router.service.upsert_lookups", new=mock_upsert):
            resp = client.post(f"/lookups/{proc_id}/refresh")

        assert resp.status_code == 200
        assert len(upsert_calls) == 1
        assert upsert_calls[0][0] == proc_id
        assert upsert_calls[0][1] == lookups
