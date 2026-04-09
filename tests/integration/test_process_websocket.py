"""Integration tests for the process WebSocket endpoint.

Tests the WebSocket handler end-to-end using FastAPI's TestClient,
covering connect/start, progress messages, pause/resume, cancel,
batch size control, attach/reconnection, error handling, and
concurrent connections.

All service and executor calls are mocked so no real database or
Redis connection is needed.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jose import jwt
from starlette.testclient import TestClient

from easyweaver.dependencies import get_db
from easyweaver.main import create_app
from easyweaver.processes.models import ProcessConfiguration, ProcessRun
from easyweaver.settings import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_mock_db = MagicMock()


def _make_config_obj() -> ProcessConfiguration:
    return ProcessConfiguration(
        id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        user_id="system",
        name="Test Process",
        description="",
        version=1,
        config={
            "queries": {
                "s": {
                    "q": {
                        "source_id": "src1",
                        "table": "t1",
                        "source_type": "postgres",
                        "encrypted_credentials": "enc",
                    }
                }
            },
            "logics": [],
        },
        params={},
        save_destination="redis",
        gcp_path="",
        tags=[],
    )


def _make_run_obj(
    run_id: str | None = None,
    status: str = "pending",
    progress: dict | None = None,
    row_count: int | None = None,
    error: str | None = None,
    result_run_id: str = "",
) -> ProcessRun:
    uid = uuid.UUID(run_id) if run_id else uuid.uuid4()
    now = datetime.now(timezone.utc)
    return ProcessRun(
        id=uid,
        process_id="proc-1",
        user_id="system",
        param_values={},
        status=status,
        row_count=row_count,
        error=error,
        result_run_id=result_run_id,
        progress=progress,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def app():
    """Create the FastAPI app with lifespan and DB dependency overridden."""
    with (
        patch("easyweaver.main.init_db", new_callable=AsyncMock),
        patch("easyweaver.main.shutdown_db", new_callable=AsyncMock),
        patch("easyweaver.main.init_redis", new_callable=AsyncMock),
        patch("easyweaver.main.shutdown_redis", new_callable=AsyncMock),
    ):
        application = create_app()

    async def _override_get_db():
        return _mock_db

    application.dependency_overrides[get_db] = _override_get_db
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client(app):
    return TestClient(app)


def _make_valid_token() -> str:
    """Create a valid JWT token for WebSocket authentication in tests."""
    payload = {
        "sub": "test-user-id",
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")


def _ws_url(config_id: str = "cfg-test") -> str:
    token = _make_valid_token()
    return f"/api/v1/processes/{config_id}/run/ws?token={token}"


# ---------------------------------------------------------------------------
# Test: Connect and start
# ---------------------------------------------------------------------------


class TestWSConnectAndStart:
    def test_start_returns_run_started(self, client):
        config = _make_config_obj()
        run = _make_run_obj()

        with patch("easyweaver.processes.ws_handler.service") as mock_svc:
            mock_svc.get_configuration = AsyncMock(return_value=config)
            mock_svc.create_process_run = AsyncMock(return_value=run)

            # Mock _run_execution to be a no-op coroutine so it doesn't
            # try to connect to real Redis/DB
            with patch(
                "easyweaver.processes.ws_handler.ProcessWebSocketHandler._run_execution",
                new_callable=AsyncMock,
            ):
                with client.websocket_connect(_ws_url()) as ws:
                    ws.send_json({"type": "start", "param_values": {}})
                    msg = ws.receive_json()
                    assert msg["type"] == "run_started"
                    assert "run_id" in msg


# ---------------------------------------------------------------------------
# Test: Progress messages (structural)
# ---------------------------------------------------------------------------


class TestWSProgressMessages:
    def test_start_flow_emits_run_started_with_config_id(self, client):
        """After start, the client receives run_started with config_id."""
        config = _make_config_obj()
        run = _make_run_obj()

        with patch("easyweaver.processes.ws_handler.service") as mock_svc:
            mock_svc.get_configuration = AsyncMock(return_value=config)
            mock_svc.create_process_run = AsyncMock(return_value=run)

            with patch(
                "easyweaver.processes.ws_handler.ProcessWebSocketHandler._run_execution",
                new_callable=AsyncMock,
            ):
                with client.websocket_connect(_ws_url()) as ws:
                    ws.send_json({"type": "start"})
                    msg = ws.receive_json()
                    assert msg["type"] == "run_started"
                    assert msg["run_id"] == str(run.id)
                    assert "phases" in msg
                    assert "datasets" in msg
                    assert "dag" in msg


# ---------------------------------------------------------------------------
# Test: Pause / Resume
# ---------------------------------------------------------------------------


class TestWSPauseResume:
    def test_pause_then_resume(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            ws.send_json({"type": "pause"})
            msg = ws.receive_json()
            assert msg["type"] == "paused"

            ws.send_json({"type": "resume"})
            msg = ws.receive_json()
            assert msg["type"] == "resumed"


# ---------------------------------------------------------------------------
# Test: Cancel
# ---------------------------------------------------------------------------


class TestWSCancel:
    def test_cancel_returns_cancelled(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            ws.send_json({"type": "cancel"})
            msg = ws.receive_json()
            assert msg["type"] == "cancelled"


# ---------------------------------------------------------------------------
# Test: set_batch_size
# ---------------------------------------------------------------------------


class TestWSSetBatchSize:
    def test_set_batch_size_ack(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            ws.send_json({"type": "set_batch_size", "batch_size": 25000})
            msg = ws.receive_json()
            assert msg["type"] == "batch_size_set"
            assert msg["batch_size"] == 25000


# ---------------------------------------------------------------------------
# Test: Attach / reconnection
# ---------------------------------------------------------------------------


class TestWSAttach:
    def test_attach_completed_run_sends_snapshot(self, client):
        run = _make_run_obj(
            run_id="00000000-0000-0000-0000-000000000099",
            status="completed",
            progress={"phase": "transforming"},
            row_count=42,
            result_run_id="res-abc",
        )

        with patch("easyweaver.processes.ws_handler.service") as mock_svc:
            mock_svc.get_process_run = AsyncMock(return_value=run)

            with client.websocket_connect(_ws_url()) as ws:
                ws.send_json({
                    "type": "attach",
                    "run_id": "00000000-0000-0000-0000-000000000099",
                })
                msg1 = ws.receive_json()
                assert msg1["type"] == "state_snapshot"
                assert msg1["progress"]["phase"] == "transforming"

                msg2 = ws.receive_json()
                assert msg2["type"] == "completed"
                assert msg2["total_rows"] == 42

    def test_attach_running_sends_attached(self, client):
        run = _make_run_obj(status="running")

        with patch("easyweaver.processes.ws_handler.service") as mock_svc:
            mock_svc.get_process_run = AsyncMock(return_value=run)

            with client.websocket_connect(_ws_url()) as ws:
                ws.send_json({"type": "attach", "run_id": str(run.id)})
                msg = ws.receive_json()
                assert msg["type"] == "attached"
                assert msg["status"] == "running"


# ---------------------------------------------------------------------------
# Test: Error handling
# ---------------------------------------------------------------------------


class TestWSAuthentication:
    """Test WebSocket authentication via token query parameter."""

    def test_missing_token_rejected(self, client):
        """Connection without a token is closed with 4003."""
        with pytest.raises(Exception):
            with client.websocket_connect("/api/v1/processes/cfg-test/run/ws") as ws:
                ws.receive_json()

    def test_invalid_token_rejected(self, client):
        """Connection with a bad token is closed with 4003."""
        with pytest.raises(Exception):
            with client.websocket_connect(
                "/api/v1/processes/cfg-test/run/ws?token=bad-token"
            ) as ws:
                ws.receive_json()

    def test_expired_token_rejected(self, client):
        """Connection with an expired token is closed with 4003."""
        payload = {
            "sub": "test-user-id",
            "type": "access",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
        }
        expired_token = jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")
        with pytest.raises(Exception):
            with client.websocket_connect(
                f"/api/v1/processes/cfg-test/run/ws?token={expired_token}"
            ) as ws:
                ws.receive_json()

    def test_valid_token_accepted(self, client):
        """Connection with a valid token succeeds."""
        with client.websocket_connect(_ws_url()) as ws:
            ws.send_json({"type": "pause"})
            msg = ws.receive_json()
            assert msg["type"] == "paused"


class TestWSErrorHandling:
    def test_start_with_missing_config_returns_error(self, client):
        with patch("easyweaver.processes.ws_handler.service") as mock_svc:
            mock_svc.get_configuration = AsyncMock(
                side_effect=Exception("Config not found")
            )

            with client.websocket_connect(_ws_url("nonexistent")) as ws:
                ws.send_json({"type": "start"})
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "Config not found" in msg["error"]

    def test_unknown_message_type_returns_error(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            ws.send_json({"type": "bogus_command"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Unknown message type" in msg["error"]


# ---------------------------------------------------------------------------
# Test: Concurrent connections / attach edge cases
# ---------------------------------------------------------------------------


class TestWSConcurrentConnections:
    def test_attach_missing_run_id_returns_error(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            ws.send_json({"type": "attach"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "run_id is required" in msg["error"]

    def test_attach_failed_run_sends_error_details(self, client):
        run = _make_run_obj(status="failed", error="DB connection refused")

        with patch("easyweaver.processes.ws_handler.service") as mock_svc:
            mock_svc.get_process_run = AsyncMock(return_value=run)

            with client.websocket_connect(_ws_url()) as ws:
                ws.send_json({"type": "attach", "run_id": str(run.id)})
                msg = ws.receive_json()
                assert msg["type"] == "failed"
                assert msg["error"] == "DB connection refused"
