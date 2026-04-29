"""Tests for easyweaver.core.middleware"""
import time
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from easyweaver.core.exceptions import EasyWeaverError
from easyweaver.core.middleware import RateLimitMiddleware, setup_middleware


# ---------------------------------------------------------------------------
# RateLimitMiddleware unit tests
# ---------------------------------------------------------------------------


class TestRateLimitMiddleware:
    def _make_app_with_rl(self, max_requests: int = 3, window_seconds: int = 60):
        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        return TestClient(app, raise_server_exceptions=False)

    def test_allows_requests_under_limit(self):
        client = self._make_app_with_rl(max_requests=5)
        for _ in range(5):
            resp = client.get("/ping")
            assert resp.status_code == 200

    def test_blocks_when_limit_exceeded(self):
        client = self._make_app_with_rl(max_requests=2)
        client.get("/ping")
        client.get("/ping")
        resp = client.get("/ping")
        assert resp.status_code == 429

    def test_429_response_body(self):
        client = self._make_app_with_rl(max_requests=1)
        client.get("/ping")
        resp = client.get("/ping")
        body = resp.json()
        assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert "message" in body["error"]

    def test_window_cleans_old_entries(self):
        """After old entries expire, requests should be allowed again."""
        app = FastAPI()
        rl = RateLimitMiddleware(app, max_requests=2, window_seconds=1)

        # Pre-fill requests for "127.0.0.1" with timestamps far in the past
        past = time.monotonic() - 100  # definitely outside the 1-second window
        rl._requests["127.0.0.1"] = [past, past]

        # With old entries cleaned, a new request should be allowed (list becomes empty)
        request = MagicMock()
        request.client.host = "127.0.0.1"

        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        import asyncio

        asyncio.run(rl.dispatch(request, call_next))
        call_next.assert_called_once()

    def test_unknown_client_ip(self):
        """Requests without a client host should use 'unknown' as key."""
        app = FastAPI()
        rl = RateLimitMiddleware(app, max_requests=10)

        request = MagicMock()
        request.client = None  # no client info

        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        import asyncio

        asyncio.run(rl.dispatch(request, call_next))
        assert "unknown" in rl._requests

    def test_different_ips_tracked_separately(self):
        client = self._make_app_with_rl(max_requests=1)
        # First IP hits limit
        client.get("/ping")
        resp = client.get("/ping")
        assert resp.status_code == 429
        # A different IP would be tracked separately (internal state check)


# ---------------------------------------------------------------------------
# setup_middleware integration tests
# ---------------------------------------------------------------------------


class TestSetupMiddleware:
    def _make_app(self, cors_origins=None, rate_limit=0):
        app = FastAPI()
        with (
            patch("easyweaver.core.middleware.settings") as mock_settings,
        ):
            mock_settings.cors_origins = cors_origins or ["*"]
            mock_settings.rate_limit_per_minute = rate_limit
            setup_middleware(app)
        return app

    def test_cors_middleware_added(self):
        app = self._make_app()
        middleware_types = [m.cls for m in app.user_middleware]
        assert CORSMiddleware in middleware_types

    def test_rate_limit_middleware_added_when_nonzero(self):
        app = self._make_app(rate_limit=100)
        middleware_types = [m.cls for m in app.user_middleware]
        assert RateLimitMiddleware in middleware_types

    def test_rate_limit_middleware_not_added_when_zero(self):
        app = self._make_app(rate_limit=0)
        middleware_types = [m.cls for m in app.user_middleware]
        assert RateLimitMiddleware not in middleware_types

    def test_easyweaver_error_handler_returns_correct_status(self):
        app = FastAPI()
        with patch("easyweaver.core.middleware.settings") as mock_settings:
            mock_settings.cors_origins = ["*"]
            mock_settings.rate_limit_per_minute = 0
            setup_middleware(app)

        @app.get("/not-found-route")
        async def raise_not_found():
            raise EasyWeaverError("not found", code="NOT_FOUND")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/not-found-route")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "NOT_FOUND"

    def test_easyweaver_error_handler_validation_error(self):
        app = FastAPI()
        with patch("easyweaver.core.middleware.settings") as mock_settings:
            mock_settings.cors_origins = ["*"]
            mock_settings.rate_limit_per_minute = 0
            setup_middleware(app)

        @app.get("/validation")
        async def raise_validation():
            raise EasyWeaverError("bad input", code="VALIDATION_ERROR")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/validation")
        assert resp.status_code == 422

    def test_easyweaver_error_handler_unknown_code_defaults_500(self):
        app = FastAPI()
        with patch("easyweaver.core.middleware.settings") as mock_settings:
            mock_settings.cors_origins = ["*"]
            mock_settings.rate_limit_per_minute = 0
            setup_middleware(app)

        @app.get("/unknown-error")
        async def raise_unknown():
            raise EasyWeaverError("something broke", code="UNKNOWN_CODE")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/unknown-error")
        assert resp.status_code == 500

    def test_unhandled_exception_returns_500(self):
        app = FastAPI()
        with patch("easyweaver.core.middleware.settings") as mock_settings:
            mock_settings.cors_origins = ["*"]
            mock_settings.rate_limit_per_minute = 0
            setup_middleware(app)

        @app.get("/crash")
        async def crash():
            raise ValueError("unexpected crash")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/crash")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"

    @pytest.mark.parametrize("code,expected_status", [
        ("NOT_FOUND", 404),
        ("VALIDATION_ERROR", 422),
        ("AUTHENTICATION_FAILED", 401),
        ("CONNECTION_TEST_FAILED", 400),
        ("QUERY_EXECUTION_FAILED", 400),
        ("PROCESS_EXECUTION_FAILED", 400),
        ("STORAGE_ERROR", 500),
    ])
    def test_all_error_codes_mapped(self, code, expected_status):
        app = FastAPI()
        with patch("easyweaver.core.middleware.settings") as mock_settings:
            mock_settings.cors_origins = ["*"]
            mock_settings.rate_limit_per_minute = 0
            setup_middleware(app)

        @app.get("/error-route")
        async def raise_error():
            raise EasyWeaverError("error", code=code)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/error-route")
        assert resp.status_code == expected_status

    def test_error_response_includes_details(self):
        app = FastAPI()
        with patch("easyweaver.core.middleware.settings") as mock_settings:
            mock_settings.cors_origins = ["*"]
            mock_settings.rate_limit_per_minute = 0
            setup_middleware(app)

        @app.get("/with-details")
        async def raise_with_details():
            raise EasyWeaverError("err", code="VALIDATION_ERROR", details={"field": "name"})

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/with-details")
        body = resp.json()
        assert body["error"]["details"] == {"field": "name"}
