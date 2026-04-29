"""Tests for easyweaver.dashboard.router — TestClient for all endpoints."""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyweaver.core.exceptions import NotFoundError
from easyweaver.dashboard.models import DashboardConfig, DataSnapshot
from easyweaver.dashboard.router import router
from easyweaver.dependencies import get_db

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

NOW = datetime.now(timezone.utc)
CONFIG_ID = "550e8400-e29b-41d4-a716-446655440000"
SOURCE_ID = "660e8400-e29b-41d4-a716-446655440001"


def _make_config(**overrides) -> DashboardConfig:
    defaults = dict(
        id=uuid.UUID(CONFIG_ID),
        user_id="user-001",
        name="Test Dashboard",
        source_id=SOURCE_ID,
        tables=[{"table_name": "orders", "timestamp_column": None, "modified_by_column": None}],
        refresh_interval_minutes=30,
        is_active=True,
        created_at=NOW,
        updated_at=NOW,
    )
    defaults.update(overrides)
    return DashboardConfig(**defaults)


def _make_snapshot(**overrides) -> DataSnapshot:
    defaults = dict(
        id=uuid.UUID("770e8400-e29b-41d4-a716-446655440002"),
        config_id=CONFIG_ID,
        source_id=SOURCE_ID,
        table_name="orders",
        row_count=1000,
        stats={"changes_1h": 5},
        captured_at=NOW,
    )
    defaults.update(overrides)
    return DataSnapshot(**defaults)


def _make_app(mock_db=None) -> FastAPI:
    app = FastAPI()

    async def _get_db_override():
        return mock_db or MagicMock()

    app.dependency_overrides[get_db] = _get_db_override
    app.include_router(router, prefix="/api/v1/dashboard")
    return app


def _make_client(mock_db=None) -> TestClient:
    app = _make_app(mock_db)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# list_configs — GET /api/v1/dashboard/configs
# ---------------------------------------------------------------------------


class TestListConfigs:
    def test_returns_200_empty_list(self):
        with patch("easyweaver.dashboard.service.list_configs", new_callable=AsyncMock) as m:
            m.return_value = []
            client = _make_client()
            resp = client.get("/api/v1/dashboard/configs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_configs_list(self):
        config = _make_config()
        with patch("easyweaver.dashboard.service.list_configs", new_callable=AsyncMock) as m:
            m.return_value = [config]
            client = _make_client()
            resp = client.get("/api/v1/dashboard/configs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == CONFIG_ID
        assert data[0]["name"] == "Test Dashboard"

    def test_filters_by_user_id_query_param(self):
        with patch("easyweaver.dashboard.service.list_configs", new_callable=AsyncMock) as m:
            m.return_value = []
            client = _make_client()
            resp = client.get("/api/v1/dashboard/configs?user_id=user-001")
        assert resp.status_code == 200
        m.assert_called_once()
        assert m.call_args.kwargs["user_id"] == "user-001" or m.call_args[0][1] == "user-001"

    def test_multiple_configs(self):
        configs = [
            _make_config(id=uuid.UUID(CONFIG_ID), name="A"),
            _make_config(id=uuid.uuid4(), name="B"),
        ]
        with patch("easyweaver.dashboard.service.list_configs", new_callable=AsyncMock) as m:
            m.return_value = configs
            client = _make_client()
            resp = client.get("/api/v1/dashboard/configs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


# ---------------------------------------------------------------------------
# create_config — POST /api/v1/dashboard/configs
# ---------------------------------------------------------------------------


class TestCreateConfig:
    _payload = {
        "name": "New Dashboard",
        "source_id": SOURCE_ID,
        "tables": [{"table_name": "orders", "timestamp_column": None, "modified_by_column": None}],
        "refresh_interval_minutes": 60,
    }

    def test_returns_201_on_success(self):
        config = _make_config(name="New Dashboard")
        with patch("easyweaver.dashboard.service.create_config", new_callable=AsyncMock) as m:
            m.return_value = config
            client = _make_client()
            resp = client.post("/api/v1/dashboard/configs", json=self._payload)
        assert resp.status_code == 201

    def test_response_body_contains_config(self):
        config = _make_config(name="New Dashboard")
        with patch("easyweaver.dashboard.service.create_config", new_callable=AsyncMock) as m:
            m.return_value = config
            client = _make_client()
            resp = client.post("/api/v1/dashboard/configs", json=self._payload)
        data = resp.json()
        assert data["name"] == "New Dashboard"
        assert data["source_id"] == SOURCE_ID

    def test_returns_422_for_missing_name(self):
        client = _make_client()
        resp = client.post(
            "/api/v1/dashboard/configs",
            json={"source_id": SOURCE_ID, "tables": [{"table_name": "t"}]},
        )
        assert resp.status_code == 422

    def test_returns_422_for_empty_tables(self):
        client = _make_client()
        resp = client.post(
            "/api/v1/dashboard/configs",
            json={"name": "D", "source_id": SOURCE_ID, "tables": []},
        )
        assert resp.status_code == 422

    def test_service_called_with_correct_args(self):
        config = _make_config()
        with patch("easyweaver.dashboard.service.create_config", new_callable=AsyncMock) as m:
            m.return_value = config
            client = _make_client()
            client.post("/api/v1/dashboard/configs", json=self._payload)
        m.assert_called_once()


# ---------------------------------------------------------------------------
# get_config — GET /api/v1/dashboard/configs/{config_id}
# ---------------------------------------------------------------------------


class TestGetConfig:
    def test_returns_200_when_found(self):
        config = _make_config()
        with patch("easyweaver.dashboard.service.get_config", new_callable=AsyncMock) as m:
            m.return_value = config
            client = _make_client()
            resp = client.get(f"/api/v1/dashboard/configs/{CONFIG_ID}")
        assert resp.status_code == 200
        assert resp.json()["id"] == CONFIG_ID

    def test_returns_404_when_not_found(self):
        with patch("easyweaver.dashboard.service.get_config", new_callable=AsyncMock) as m:
            m.side_effect = NotFoundError("DashboardConfig", CONFIG_ID)
            app = FastAPI()

            async def _get_db_override():
                return MagicMock()

            app.dependency_overrides[get_db] = _get_db_override

            from easyweaver.core.middleware import setup_middleware
            setup_middleware(app)
            app.include_router(router, prefix="/api/v1/dashboard")
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get(f"/api/v1/dashboard/configs/{CONFIG_ID}")
        assert resp.status_code == 404

    def test_config_id_passed_to_service(self):
        config = _make_config()
        with patch("easyweaver.dashboard.service.get_config", new_callable=AsyncMock) as m:
            m.return_value = config
            client = _make_client()
            client.get(f"/api/v1/dashboard/configs/{CONFIG_ID}")
        m.assert_called_once()
        assert m.call_args[0][1] == CONFIG_ID


# ---------------------------------------------------------------------------
# update_config — PUT /api/v1/dashboard/configs/{config_id}
# ---------------------------------------------------------------------------


class TestUpdateConfig:
    def test_returns_200_on_success(self):
        config = _make_config(name="Updated")
        with patch("easyweaver.dashboard.service.update_config", new_callable=AsyncMock) as m:
            m.return_value = config
            client = _make_client()
            resp = client.put(
                f"/api/v1/dashboard/configs/{CONFIG_ID}",
                json={"name": "Updated"},
            )
        assert resp.status_code == 200

    def test_returns_updated_config(self):
        config = _make_config(name="Updated Name")
        with patch("easyweaver.dashboard.service.update_config", new_callable=AsyncMock) as m:
            m.return_value = config
            client = _make_client()
            resp = client.put(
                f"/api/v1/dashboard/configs/{CONFIG_ID}",
                json={"name": "Updated Name"},
            )
        assert resp.json()["name"] == "Updated Name"

    def test_returns_404_when_not_found(self):
        with patch("easyweaver.dashboard.service.update_config", new_callable=AsyncMock) as m:
            m.side_effect = NotFoundError("DashboardConfig", CONFIG_ID)
            app = FastAPI()

            async def _get_db_override():
                return MagicMock()

            app.dependency_overrides[get_db] = _get_db_override
            from easyweaver.core.middleware import setup_middleware
            setup_middleware(app)
            app.include_router(router, prefix="/api/v1/dashboard")
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.put(
                f"/api/v1/dashboard/configs/{CONFIG_ID}",
                json={"name": "X"},
            )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# delete_config — DELETE /api/v1/dashboard/configs/{config_id}
# ---------------------------------------------------------------------------


class TestDeleteConfig:
    def test_returns_204_on_success(self):
        with patch("easyweaver.dashboard.service.delete_config", new_callable=AsyncMock) as m:
            m.return_value = None
            client = _make_client()
            resp = client.delete(f"/api/v1/dashboard/configs/{CONFIG_ID}")
        assert resp.status_code == 204

    def test_returns_404_when_not_found(self):
        with patch("easyweaver.dashboard.service.delete_config", new_callable=AsyncMock) as m:
            m.side_effect = NotFoundError("DashboardConfig", CONFIG_ID)
            app = FastAPI()

            async def _get_db_override():
                return MagicMock()

            app.dependency_overrides[get_db] = _get_db_override
            from easyweaver.core.middleware import setup_middleware
            setup_middleware(app)
            app.include_router(router, prefix="/api/v1/dashboard")
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.delete(f"/api/v1/dashboard/configs/{CONFIG_ID}")
        assert resp.status_code == 404

    def test_service_called_with_config_id(self):
        with patch("easyweaver.dashboard.service.delete_config", new_callable=AsyncMock) as m:
            m.return_value = None
            client = _make_client()
            client.delete(f"/api/v1/dashboard/configs/{CONFIG_ID}")
        m.assert_called_once()
        assert m.call_args[0][1] == CONFIG_ID


# ---------------------------------------------------------------------------
# get_stats — GET /api/v1/dashboard/configs/{config_id}/stats
# ---------------------------------------------------------------------------


class TestGetStats:
    _stats_response = {
        "config_id": CONFIG_ID,
        "source_name": "Test DB",
        "source_type": "postgres",
        "tables": [
            {
                "table_name": "orders",
                "current_row_count": 1000,
                "changes_1h": 5,
                "changes_3h": 15,
                "changes_24h": 50,
                "last_modified_at": None,
                "last_modified_by": None,
                "column_count": 8,
                "size_bytes": 500000,
            }
        ],
        "captured_at": NOW.isoformat(),
        "connection_healthy": True,
    }

    def test_returns_200_with_stats(self):
        with patch("easyweaver.dashboard.service.get_current_stats", new_callable=AsyncMock) as m:
            m.return_value = self._stats_response
            client = _make_client()
            resp = client.get(f"/api/v1/dashboard/configs/{CONFIG_ID}/stats")
        assert resp.status_code == 200

    def test_response_contains_expected_fields(self):
        with patch("easyweaver.dashboard.service.get_current_stats", new_callable=AsyncMock) as m:
            m.return_value = self._stats_response
            client = _make_client()
            resp = client.get(f"/api/v1/dashboard/configs/{CONFIG_ID}/stats")
        data = resp.json()
        assert "config_id" in data
        assert "tables" in data
        assert "connection_healthy" in data


# ---------------------------------------------------------------------------
# refresh_stats — POST /api/v1/dashboard/configs/{config_id}/refresh
# ---------------------------------------------------------------------------


class TestRefreshStats:
    def test_returns_200_on_refresh(self):
        stats = {
            "config_id": CONFIG_ID,
            "source_name": "DB",
            "source_type": "postgres",
            "tables": [],
            "captured_at": NOW.isoformat(),
            "connection_healthy": True,
        }
        with patch("easyweaver.dashboard.service.get_current_stats", new_callable=AsyncMock) as m:
            m.return_value = stats
            client = _make_client()
            resp = client.post(f"/api/v1/dashboard/configs/{CONFIG_ID}/refresh")
        assert resp.status_code == 200

    def test_calls_get_current_stats_with_force_refresh(self):
        stats = {
            "config_id": CONFIG_ID,
            "source_name": "DB",
            "source_type": "postgres",
            "tables": [],
            "captured_at": NOW.isoformat(),
            "connection_healthy": True,
        }
        with patch("easyweaver.dashboard.service.get_current_stats", new_callable=AsyncMock) as m:
            m.return_value = stats
            client = _make_client()
            client.post(f"/api/v1/dashboard/configs/{CONFIG_ID}/refresh")
        m.assert_called_once()
        # force_refresh=True should be passed
        call_kwargs = m.call_args
        assert True in call_kwargs[0] or call_kwargs[1].get("force_refresh") is True or call_kwargs[0][2] is True


# ---------------------------------------------------------------------------
# get_history — GET /api/v1/dashboard/configs/{config_id}/history
# ---------------------------------------------------------------------------


class TestGetHistory:
    def test_returns_200_with_snapshots(self):
        snap = _make_snapshot()
        with patch("easyweaver.dashboard.service.get_snapshot_history", new_callable=AsyncMock) as m:
            m.return_value = [snap]
            client = _make_client()
            resp = client.get(f"/api/v1/dashboard/configs/{CONFIG_ID}/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "snapshots" in data
        assert "total" in data
        assert data["total"] == 1

    def test_returns_empty_snapshots(self):
        with patch("easyweaver.dashboard.service.get_snapshot_history", new_callable=AsyncMock) as m:
            m.return_value = []
            client = _make_client()
            resp = client.get(f"/api/v1/dashboard/configs/{CONFIG_ID}/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["snapshots"] == []

    def test_accepts_hours_query_param(self):
        with patch("easyweaver.dashboard.service.get_snapshot_history", new_callable=AsyncMock) as m:
            m.return_value = []
            client = _make_client()
            resp = client.get(
                f"/api/v1/dashboard/configs/{CONFIG_ID}/history?hours=48"
            )
        assert resp.status_code == 200
        m.assert_called_once()
        call_args = m.call_args
        assert 48 in call_args[0] or call_args[1].get("hours") == 48

    def test_snapshot_records_have_correct_structure(self):
        snap = _make_snapshot()
        with patch("easyweaver.dashboard.service.get_snapshot_history", new_callable=AsyncMock) as m:
            m.return_value = [snap]
            client = _make_client()
            resp = client.get(f"/api/v1/dashboard/configs/{CONFIG_ID}/history")
        record = resp.json()["snapshots"][0]
        assert "id" in record
        assert "config_id" in record
        assert "table_name" in record
        assert "row_count" in record
        assert "captured_at" in record
