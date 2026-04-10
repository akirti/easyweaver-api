"""Unit tests for the GET /{source_id}/tables/{table}/columns/{column}/distinct endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyweaver.dependencies import get_db
from easyweaver.sources.router import router


# Build a minimal FastAPI app with just the sources router
_app = FastAPI()
_app.include_router(router, prefix="/api/v1/sources")

# Override get_db to return a mock
_mock_db = MagicMock()
_app.dependency_overrides[get_db] = lambda: _mock_db


class _FakeSource:
    source_type = "postgres"


class _FakeConnector:
    def __init__(self, result):
        self._result = result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get_distinct_values(self, table, column, limit):
        return self._result


@pytest.fixture
def client():
    return TestClient(_app)


@pytest.fixture
def source_id():
    return str(uuid.uuid4())


class TestDistinctEndpoint:
    def test_returns_distinct_values(self, client, source_id):
        expected = {"values": ["a", "b", "c"], "truncated": False, "total_count": 3}
        fake_source = _FakeSource()

        with patch("easyweaver.sources.router.service") as mock_service, \
             patch("easyweaver.connectors.registry.get_connector") as mock_get_connector:
            mock_service.get_source = AsyncMock(return_value=fake_source)
            mock_service.get_source_credentials = MagicMock(return_value={"host": "h"})
            mock_get_connector.return_value = _FakeConnector(expected)

            resp = client.get(f"/api/v1/sources/{source_id}/tables/public.users/columns/status/distinct")
            assert resp.status_code == 200
            assert resp.json() == expected

    def test_limit_capped_at_5000(self, client, source_id):
        expected = {"values": [], "truncated": False, "total_count": 0}
        fake_source = _FakeSource()

        with patch("easyweaver.sources.router.service") as mock_service, \
             patch("easyweaver.connectors.registry.get_connector") as mock_get_connector:
            mock_service.get_source = AsyncMock(return_value=fake_source)
            mock_service.get_source_credentials = MagicMock(return_value={"host": "h"})

            captured_limit = {}

            class _CapturingConnector(_FakeConnector):
                async def get_distinct_values(self, table, column, limit):
                    captured_limit["limit"] = limit
                    return self._result

            mock_get_connector.return_value = _CapturingConnector(expected)

            resp = client.get(
                f"/api/v1/sources/{source_id}/tables/t/columns/c/distinct?limit=9999"
            )
            assert resp.status_code == 200
            assert captured_limit["limit"] == 5000

    def test_default_limit_is_500(self, client, source_id):
        expected = {"values": [], "truncated": False, "total_count": 0}
        fake_source = _FakeSource()

        with patch("easyweaver.sources.router.service") as mock_service, \
             patch("easyweaver.connectors.registry.get_connector") as mock_get_connector:
            mock_service.get_source = AsyncMock(return_value=fake_source)
            mock_service.get_source_credentials = MagicMock(return_value={"host": "h"})

            captured_limit = {}

            class _CapturingConnector(_FakeConnector):
                async def get_distinct_values(self, table, column, limit):
                    captured_limit["limit"] = limit
                    return self._result

            mock_get_connector.return_value = _CapturingConnector(expected)

            resp = client.get(
                f"/api/v1/sources/{source_id}/tables/t/columns/c/distinct"
            )
            assert resp.status_code == 200
            assert captured_limit["limit"] == 500
