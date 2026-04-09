"""Tests for easyweaver.processes.ws_handler — ProcessWebSocketHandler."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from easyweaver.processes.models import ProcessConfiguration, ProcessRun
from easyweaver.processes.ws_handler import (
    ProcessWebSocketHandler,
    _active_handlers,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_mock_db() -> MagicMock:
    """Create a mock AsyncIOMotorDatabase."""
    db = MagicMock()
    db.process_runs = MagicMock()
    db.process_runs.update_one = AsyncMock()
    db.process_runs.find_one = AsyncMock(return_value=None)
    db.process_configurations = MagicMock()
    db.process_configurations.find_one = AsyncMock(return_value=None)
    return db


def _make_mock_ws() -> MagicMock:
    """Create a mock WebSocket."""
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock()
    return ws


def _make_config(config_id: str = "cfg-1") -> ProcessConfiguration:
    uid = uuid.UUID("12345678-1234-1234-1234-123456789abc")
    now = datetime.now(timezone.utc)
    return ProcessConfiguration(
        id=uid,
        user_id="system",
        name="Test Config",
        description="",
        version=1,
        config={"queries": {}, "logics": []},
        params={},
        save_destination="redis",
        gcp_path="",
        tags=[],
        created_at=now,
        updated_at=now,
    )


def _make_run(
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


def _make_handler(ws=None, db=None, config_id="cfg-1") -> ProcessWebSocketHandler:
    ws = ws or _make_mock_ws()
    db = db or _make_mock_db()
    return ProcessWebSocketHandler(ws, config_id, db)


# ── Test: Dispatch ────────────────────────────────────────────────────


class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_unknown_type_sends_error(self):
        handler = _make_handler()
        await handler._dispatch({"type": "unknown_msg"})

        handler.ws.send_json.assert_called_once()
        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "error"
        assert "Unknown message type" in msg["error"]

    @pytest.mark.asyncio
    async def test_dispatch_routes_pause(self):
        handler = _make_handler()
        await handler._dispatch({"type": "pause"})

        handler.ws.send_json.assert_called_once()
        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "paused"

    @pytest.mark.asyncio
    async def test_dispatch_routes_resume(self):
        handler = _make_handler()
        handler.control["paused"] = True
        await handler._dispatch({"type": "resume"})

        handler.ws.send_json.assert_called_once()
        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "resumed"

    @pytest.mark.asyncio
    async def test_dispatch_routes_cancel(self):
        handler = _make_handler()
        await handler._dispatch({"type": "cancel"})

        handler.ws.send_json.assert_called_once()
        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "cancelled"

    @pytest.mark.asyncio
    async def test_dispatch_missing_type(self):
        handler = _make_handler()
        await handler._dispatch({})

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "error"


# ── Test: Pause / Resume ─────────────────────────────────────────────


class TestPauseResume:
    @pytest.mark.asyncio
    async def test_pause_sets_control(self):
        handler = _make_handler()
        assert handler.control["paused"] is False

        await handler._handle_pause({})
        assert handler.control["paused"] is True

    @pytest.mark.asyncio
    async def test_resume_clears_control(self):
        handler = _make_handler()
        handler.control["paused"] = True

        await handler._handle_resume({})
        assert handler.control["paused"] is False


# ── Test: Batch size / target seconds ─────────────────────────────────


class TestBatchControls:
    @pytest.mark.asyncio
    async def test_set_batch_size(self):
        handler = _make_handler()

        await handler._handle_set_batch_size({"batch_size": 5000})
        assert handler.control["batch_size_override"] == 5000
        assert handler.control["adaptive_enabled"] is False

    @pytest.mark.asyncio
    async def test_set_target_seconds_enables_adaptive(self):
        handler = _make_handler()
        handler.control["batch_size_override"] = 5000
        handler.control["adaptive_enabled"] = False

        await handler._handle_set_target_seconds({"target_seconds": 15.0})

        assert handler.control["target_batch_seconds"] == 15.0
        assert handler.control["adaptive_enabled"] is True
        assert handler.control["batch_size_override"] is None

    @pytest.mark.asyncio
    async def test_set_target_seconds_default(self):
        handler = _make_handler()

        await handler._handle_set_target_seconds({})
        assert handler.control["target_batch_seconds"] == 10.0


# ── Test: Cancel ──────────────────────────────────────────────────────


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_sets_flag(self):
        handler = _make_handler()

        await handler._handle_cancel({})
        assert handler.control["cancelled"] is True

    @pytest.mark.asyncio
    async def test_cancel_cancels_task(self):
        handler = _make_handler()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel = MagicMock()
        handler._execution_task = mock_task

        await handler._handle_cancel({})

        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_ignores_done_task(self):
        handler = _make_handler()
        mock_task = MagicMock()
        mock_task.done.return_value = True
        mock_task.cancel = MagicMock()
        handler._execution_task = mock_task

        await handler._handle_cancel({})

        mock_task.cancel.assert_not_called()


# ── Test: Start ───────────────────────────────────────────────────────


class TestStart:
    @pytest.mark.asyncio
    async def test_start_creates_run_and_sends_started(self):
        handler = _make_handler()
        config = _make_config()
        run = _make_run()

        with (
            patch("easyweaver.processes.ws_handler.service") as mock_service,
            patch("asyncio.create_task") as mock_create_task,
        ):
            mock_service.get_configuration = AsyncMock(return_value=config)
            mock_service.create_process_run = AsyncMock(return_value=run)

            # Patch create_task to capture but not run the coroutine
            captured_coro = None
            def capture_task(coro):
                nonlocal captured_coro
                captured_coro = coro
                mock_task = MagicMock()
                mock_task.done.return_value = False
                return mock_task
            mock_create_task.side_effect = capture_task

            await handler._handle_start({"type": "start", "param_values": {"x": 1}, "max_rows": 500})

            mock_service.get_configuration.assert_called_once_with(handler.db, "cfg-1")
            mock_service.create_process_run.assert_called_once()

            assert handler.run_id == str(run.id)

            # Check run_started message
            sent_msgs = [call[0][0] for call in handler.ws.send_json.call_args_list]
            started = [m for m in sent_msgs if m.get("type") == "run_started"]
            assert len(started) == 1
            assert started[0]["run_id"] == str(run.id)

            # Clean up captured coroutine
            if captured_coro is not None:
                captured_coro.close()

    @pytest.mark.asyncio
    async def test_start_registers_in_active_handlers(self):
        handler = _make_handler()
        config = _make_config()
        run = _make_run()

        with (
            patch("easyweaver.processes.ws_handler.service") as mock_service,
            patch("asyncio.create_task") as mock_create_task,
        ):
            mock_service.get_configuration = AsyncMock(return_value=config)
            mock_service.create_process_run = AsyncMock(return_value=run)

            captured_coro = None
            def capture_task(coro):
                nonlocal captured_coro
                captured_coro = coro
                mock_task = MagicMock()
                mock_task.done.return_value = False
                return mock_task
            mock_create_task.side_effect = capture_task

            await handler._handle_start({"type": "start"})

            assert _active_handlers.get(str(run.id)) is handler

            # Clean up
            if str(run.id) in _active_handlers:
                del _active_handlers[str(run.id)]
            if captured_coro is not None:
                captured_coro.close()

    @pytest.mark.asyncio
    async def test_start_config_not_found_sends_error(self):
        handler = _make_handler()

        with patch("easyweaver.processes.ws_handler.service") as mock_service:
            mock_service.get_configuration = AsyncMock(
                side_effect=Exception("Config not found")
            )

            await handler._handle_start({"type": "start"})

            msg = handler.ws.send_json.call_args[0][0]
            assert msg["type"] == "error"
            assert "Config not found" in msg["error"]

    @pytest.mark.asyncio
    async def test_start_applies_target_seconds(self):
        handler = _make_handler()
        config = _make_config()
        run = _make_run()

        with (
            patch("easyweaver.processes.ws_handler.service") as mock_service,
            patch("asyncio.create_task") as mock_ct,
        ):
            mock_service.get_configuration = AsyncMock(return_value=config)
            mock_service.create_process_run = AsyncMock(return_value=run)

            captured_coro = None
            def capture_task(coro):
                nonlocal captured_coro
                captured_coro = coro
                return MagicMock(done=MagicMock(return_value=False))
            mock_ct.side_effect = capture_task

            await handler._handle_start({"type": "start", "target_batch_seconds": 20.0})

            assert handler.control["target_batch_seconds"] == 20.0

            # Clean up
            if str(run.id) in _active_handlers:
                del _active_handlers[str(run.id)]
            if captured_coro is not None:
                captured_coro.close()


# ── Test: Attach ──────────────────────────────────────────────────────


class TestAttach:
    @pytest.mark.asyncio
    async def test_attach_missing_run_id(self):
        handler = _make_handler()

        await handler._handle_attach({"type": "attach"})

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "error"
        assert "run_id is required" in msg["error"]

    @pytest.mark.asyncio
    async def test_attach_run_not_found(self):
        handler = _make_handler()

        with patch("easyweaver.processes.ws_handler.service") as mock_service:
            mock_service.get_process_run = AsyncMock(
                side_effect=Exception("Run not found")
            )

            await handler._handle_attach({"type": "attach", "run_id": "non-existent"})

            msg = handler.ws.send_json.call_args[0][0]
            assert msg["type"] == "error"

    @pytest.mark.asyncio
    async def test_attach_completed_run_sends_snapshot_and_status(self):
        handler = _make_handler()
        run = _make_run(
            status="completed",
            progress={"phase": "transforming", "datasets": {}},
            row_count=42,
            result_run_id="result-abc",
        )

        with patch("easyweaver.processes.ws_handler.service") as mock_service:
            mock_service.get_process_run = AsyncMock(return_value=run)

            await handler._handle_attach({"type": "attach", "run_id": str(run.id)})

            sent_msgs = [call[0][0] for call in handler.ws.send_json.call_args_list]

            # Should have state_snapshot
            snapshots = [m for m in sent_msgs if m["type"] == "state_snapshot"]
            assert len(snapshots) == 1
            assert snapshots[0]["progress"]["phase"] == "transforming"

            # Should have completed
            completed = [m for m in sent_msgs if m["type"] == "completed"]
            assert len(completed) == 1
            assert completed[0]["total_rows"] == 42

    @pytest.mark.asyncio
    async def test_attach_failed_run_sends_error(self):
        handler = _make_handler()
        run = _make_run(status="failed", error="Something broke")

        with patch("easyweaver.processes.ws_handler.service") as mock_service:
            mock_service.get_process_run = AsyncMock(return_value=run)

            await handler._handle_attach({"type": "attach", "run_id": str(run.id)})

            sent_msgs = [call[0][0] for call in handler.ws.send_json.call_args_list]
            failed = [m for m in sent_msgs if m["type"] == "failed"]
            assert len(failed) == 1
            assert failed[0]["error"] == "Something broke"

    @pytest.mark.asyncio
    async def test_attach_running_sends_attached(self):
        handler = _make_handler()
        run = _make_run(status="running")

        with patch("easyweaver.processes.ws_handler.service") as mock_service:
            mock_service.get_process_run = AsyncMock(return_value=run)

            await handler._handle_attach({"type": "attach", "run_id": str(run.id)})

            sent_msgs = [call[0][0] for call in handler.ws.send_json.call_args_list]
            attached = [m for m in sent_msgs if m["type"] == "attached"]
            assert len(attached) == 1
            assert attached[0]["status"] == "running"

    @pytest.mark.asyncio
    async def test_attach_wires_observer_on_active_handler(self):
        """When an active handler exists for a run_id, the new WS should be added as observer."""
        active_handler = _make_handler()
        active_handler.run_id = "test-run-id"
        _active_handlers["test-run-id"] = active_handler

        new_ws = _make_mock_ws()
        new_handler = _make_handler(ws=new_ws)
        run = _make_run(status="running")

        try:
            with patch("easyweaver.processes.ws_handler.service") as mock_service:
                mock_service.get_process_run = AsyncMock(return_value=run)

                await new_handler._handle_attach({"type": "attach", "run_id": "test-run-id"})

                assert new_ws in active_handler._observers
        finally:
            if "test-run-id" in _active_handlers:
                del _active_handlers["test-run-id"]

    @pytest.mark.asyncio
    async def test_attach_no_progress_skips_snapshot(self):
        handler = _make_handler()
        run = _make_run(status="running", progress=None)

        with patch("easyweaver.processes.ws_handler.service") as mock_service:
            mock_service.get_process_run = AsyncMock(return_value=run)

            await handler._handle_attach({"type": "attach", "run_id": str(run.id)})

            sent_msgs = [call[0][0] for call in handler.ws.send_json.call_args_list]
            snapshots = [m for m in sent_msgs if m["type"] == "state_snapshot"]
            assert len(snapshots) == 0


# ── Test: Send / Broadcast ────────────────────────────────────────────


class TestSendBroadcast:
    @pytest.mark.asyncio
    async def test_send_ignores_disconnect(self):
        handler = _make_handler()
        handler.ws.send_json = AsyncMock(side_effect=RuntimeError("disconnected"))

        # Should not raise
        await handler._send({"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_observers(self):
        handler = _make_handler()
        obs1 = _make_mock_ws()
        obs2 = _make_mock_ws()
        handler._observers = [obs1, obs2]

        await handler._broadcast({"type": "progress"})

        handler.ws.send_json.assert_called_once()
        obs1.send_json.assert_called_once()
        obs2.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_observers(self):
        handler = _make_handler()
        dead_obs = _make_mock_ws()
        dead_obs.send_json = AsyncMock(side_effect=RuntimeError("disconnected"))
        live_obs = _make_mock_ws()
        handler._observers = [dead_obs, live_obs]

        await handler._broadcast({"type": "progress"})

        assert dead_obs not in handler._observers
        assert live_obs in handler._observers


# ── Test: Progress callback ───────────────────────────────────────────


class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_callback_broadcasts(self):
        handler = _make_handler()
        obs = _make_mock_ws()
        handler._observers = [obs]

        await handler._progress_callback("fetch_progress", dataset="orders", rows_fetched=100)

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "fetch_progress"
        assert msg["dataset"] == "orders"
        assert msg["rows_fetched"] == 100

        obs_msg = obs.send_json.call_args[0][0]
        assert obs_msg["type"] == "fetch_progress"


# ── Test: Handle (main loop) ─────────────────────────────────────────


class TestHandle:
    @pytest.mark.asyncio
    async def test_handle_accepts_and_loops(self):
        """handle() should accept the WS, receive messages, and exit on disconnect."""
        from fastapi import WebSocketDisconnect

        ws = _make_mock_ws()
        call_count = 0

        async def receive_then_disconnect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"type": "pause"}
            raise WebSocketDisconnect()

        ws.receive_json = AsyncMock(side_effect=receive_then_disconnect)
        handler = _make_handler(ws=ws)

        await handler.handle()

        ws.accept.assert_called_once()
        assert handler.control["paused"] is True

    @pytest.mark.asyncio
    async def test_handle_immediate_disconnect(self):
        """If client disconnects right away, handle() exits cleanly."""
        from fastapi import WebSocketDisconnect

        ws = _make_mock_ws()
        ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())
        handler = _make_handler(ws=ws)

        await handler.handle()

        ws.accept.assert_called_once()


# ── Test: Disconnection (execution continues) ────────────────────────


class TestDisconnection:
    @pytest.mark.asyncio
    async def test_send_failure_does_not_stop_execution(self):
        """If the WS send fails (client disconnected), _run_execution should continue."""
        handler = _make_handler()
        handler.ws.send_json = AsyncMock(side_effect=RuntimeError("closed"))

        # _send should swallow the error
        await handler._send({"type": "test"})
        # No exception = test passes

    @pytest.mark.asyncio
    async def test_broadcast_continues_on_primary_failure(self):
        """Even if primary WS fails, observers should still receive."""
        handler = _make_handler()
        handler.ws.send_json = AsyncMock(side_effect=RuntimeError("closed"))
        obs = _make_mock_ws()
        handler._observers = [obs]

        await handler._broadcast({"type": "progress"})

        obs.send_json.assert_called_once()
