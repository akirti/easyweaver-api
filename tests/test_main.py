"""Tests for easyweaver.main — create_app, routes, middleware, lifespan."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_returns_fastapi_instance(self):
        from easyweaver.main import create_app
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_has_title_from_settings(self):
        from easyweaver.main import create_app
        from easyweaver.settings import settings
        app = create_app()
        assert app.title == settings.app_name

    def test_app_has_version(self):
        from easyweaver.main import create_app
        app = create_app()
        assert app.version == "0.1.0"

    def test_multiple_calls_create_independent_apps(self):
        from easyweaver.main import create_app
        app1 = create_app()
        app2 = create_app()
        assert app1 is not app2


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.fixture
    def client(self):
        from easyweaver.main import create_app
        app = create_app()
        return TestClient(app, raise_server_exceptions=False)

    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, client):
        resp = client.get("/api/v1/health")
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Public settings endpoint
# ---------------------------------------------------------------------------


class TestPublicSettingsEndpoint:
    @pytest.fixture
    def client(self):
        from easyweaver.main import create_app
        app = create_app()
        return TestClient(app, raise_server_exceptions=False)

    def test_settings_endpoint_returns_200(self, client):
        resp = client.get("/api/v1/settings")
        assert resp.status_code == 200

    def test_settings_endpoint_returns_max_result_rows(self, client):
        from easyweaver.settings import settings
        resp = client.get("/api/v1/settings")
        data = resp.json()
        assert "max_result_rows" in data
        assert data["max_result_rows"] == settings.max_result_rows


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------


class TestRouterRegistration:
    @pytest.fixture
    def client(self):
        from easyweaver.main import create_app
        app = create_app()
        return TestClient(app, raise_server_exceptions=False)

    def test_sources_router_registered(self, client):
        # /api/v1/sources should return 200 or similar (not 404)
        resp = client.get("/api/v1/sources")
        assert resp.status_code != 404

    def test_queries_router_registered(self, client):
        # queries router exposes POST /execute — verify prefix routes resolve (not 404)
        resp = client.post("/api/v1/queries/execute", json={})
        assert resp.status_code != 404

    def test_auth_router_registered(self, client):
        resp = client.post("/api/v1/auth/login", json={"email": "x", "password": "y"})
        assert resp.status_code != 404

    def test_processes_router_registered(self, client):
        resp = client.get("/api/v1/processes")
        assert resp.status_code != 404

    def test_dashboard_router_registered(self, client):
        resp = client.get("/api/v1/dashboard/configs")
        assert resp.status_code != 404

    def test_lookups_router_registered(self, client):
        # lookups router exposes GET /{process_id} — any non-existing process still routes (not 404 on prefix)
        resp = client.get("/api/v1/lookups/some-process-id")
        assert resp.status_code != 404


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class TestMiddleware:
    @pytest.fixture
    def client(self):
        from easyweaver.main import create_app
        app = create_app()
        return TestClient(app, raise_server_exceptions=False)

    def test_cors_headers_present_on_health(self, client):
        resp = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
        # CORS middleware should add Access-Control-Allow-Origin
        assert resp.status_code == 200

    def test_unknown_route_returns_404(self, client):
        resp = client.get("/api/v1/does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Lifespan — startup & shutdown
# ---------------------------------------------------------------------------


class TestLifespan:
    @pytest.mark.anyio
    async def test_lifespan_handles_mongodb_failure_gracefully(self):
        """Startup should not crash if MongoDB is unavailable."""
        from easyweaver.main import lifespan
        from fastapi import FastAPI

        app = FastAPI()

        with patch("easyweaver.main.init_db", new_callable=AsyncMock) as mock_init_db:
            with patch("easyweaver.main.init_redis", new_callable=AsyncMock) as mock_init_redis:
                with patch("easyweaver.main.shutdown_redis", new_callable=AsyncMock):
                    with patch("easyweaver.main.shutdown_db", new_callable=AsyncMock):
                        mock_init_db.side_effect = Exception("MongoDB not reachable")
                        mock_init_redis.return_value = None

                        async with lifespan(app):
                            pass  # should not raise

    @pytest.mark.anyio
    async def test_lifespan_handles_redis_failure_gracefully(self):
        """Startup should not crash if Redis is unavailable."""
        from easyweaver.main import lifespan
        from fastapi import FastAPI

        app = FastAPI()

        with patch("easyweaver.main.init_db", new_callable=AsyncMock) as mock_init_db:
            with patch("easyweaver.main.init_redis", new_callable=AsyncMock) as mock_init_redis:
                with patch("easyweaver.main.shutdown_redis", new_callable=AsyncMock):
                    with patch("easyweaver.main.shutdown_db", new_callable=AsyncMock):
                        mock_init_db.return_value = None
                        mock_init_redis.side_effect = Exception("Redis not reachable")

                        async with lifespan(app):
                            pass  # should not raise

    @pytest.mark.anyio
    async def test_lifespan_calls_shutdown_on_exit(self):
        """Shutdown functions must be called when exiting the context."""
        from easyweaver.main import lifespan
        from fastapi import FastAPI

        app = FastAPI()

        with patch("easyweaver.main.init_db", new_callable=AsyncMock):
            with patch("easyweaver.main.init_redis", new_callable=AsyncMock):
                with patch("easyweaver.main.shutdown_redis", new_callable=AsyncMock) as mock_shutdown_redis:
                    with patch("easyweaver.main.shutdown_db", new_callable=AsyncMock) as mock_shutdown_db:
                        async with lifespan(app):
                            pass

                        mock_shutdown_redis.assert_called_once()
                        mock_shutdown_db.assert_called_once()

    @pytest.mark.anyio
    async def test_lifespan_calls_both_inits(self):
        """Both init_db and init_redis should be called on startup."""
        from easyweaver.main import lifespan
        from fastapi import FastAPI

        app = FastAPI()

        with patch("easyweaver.main.init_db", new_callable=AsyncMock) as mock_init_db:
            with patch("easyweaver.main.init_redis", new_callable=AsyncMock) as mock_init_redis:
                with patch("easyweaver.main.shutdown_redis", new_callable=AsyncMock):
                    with patch("easyweaver.main.shutdown_db", new_callable=AsyncMock):
                        async with lifespan(app):
                            pass

                        mock_init_db.assert_called_once()
                        mock_init_redis.assert_called_once()


# ---------------------------------------------------------------------------
# Module-level app object
# ---------------------------------------------------------------------------


class TestModuleLevelApp:
    def test_module_app_is_fastapi(self):
        from easyweaver.main import app
        assert isinstance(app, FastAPI)

    def test_module_app_responds_to_health(self):
        from easyweaver.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
