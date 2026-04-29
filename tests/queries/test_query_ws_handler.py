"""Tests for easyweaver.queries.ws_handler — QueryWebSocketHandler."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect

from easyweaver.queries.models import QueryRun
from easyweaver.queries.ws_handler import (
    QueryWebSocketHandler,
    _active_query_handlers,
)


# ── Helpers ───────────────────────────────────────────────────────────

_RUN_ID = "12345678-1234-1234-1234-123456789abc"
_SOURCE_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_NOW = datetime(2026, 4, 29, 10, 0, 0, tzinfo=timezone.utc)


def _make_mock_db() -> MagicMock:
    db = MagicMock()
    db.query_runs = MagicMock()
    db.query_runs.update_one = AsyncMock()
    db.query_runs.find_one = AsyncMock(return_value=None)
    return db


def _make_mock_ws() -> MagicMock:
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock()
    return ws


def _make_run(run_id: str = _RUN_ID, status: str = "pending") -> QueryRun:
    return QueryRun(
        id=uuid.UUID(run_id),
        config='{"type": "single"}',
        status=status,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_handler(ws=None, db=None) -> QueryWebSocketHandler:
    ws = ws or _make_mock_ws()
    db = db or _make_mock_db()
    return QueryWebSocketHandler(ws, db)


def _valid_start_msg() -> dict:
    return {
        "type": "start",
        "request": {
            "type": "single",
            "left": {
                "source_id": _SOURCE_ID,
                "table": "orders",
                "columns": None,
                "filters": [],
                "filter_logic": "and",
            },
        },
    }


# ── Test: Initialization ──────────────────────────────────────────────


class TestInit:
    def test_initial_control_values(self):
        handler = _make_handler()
        assert handler.control["paused"] is False
        assert handler.control["cancelled"] is False
        assert handler.control["adaptive_enabled"] is True
        assert handler.control["batch_size_override"] is None
        assert handler.control["target_batch_seconds"] == 10.0

    def test_run_id_is_none_initially(self):
        handler = _make_handler()
        assert handler.run_id is None

    def test_execution_task_is_none_initially(self):
        handler = _make_handler()
        assert handler._execution_task is None


# ── Test: Dispatch ────────────────────────────────────────────────────


class TestDispatch:
    @pytest.mark.anyio
    async def test_unknown_message_type_sends_error(self):
        handler = _make_handler()
        await handler._dispatch({"type": "unknown_xyz"})

        handler.ws.send_json.assert_called_once()
        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "error"
        assert "Unknown message type" in msg["message"]

    @pytest.mark.anyio
    async def test_missing_type_sends_error(self):
        handler = _make_handler()
        await handler._dispatch({})

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "error"

    @pytest.mark.anyio
    async def test_dispatch_routes_pause(self):
        handler = _make_handler()
        await handler._dispatch({"type": "pause"})

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "paused"

    @pytest.mark.anyio
    async def test_dispatch_routes_resume(self):
        handler = _make_handler()
        handler.control["paused"] = True
        await handler._dispatch({"type": "resume"})

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "resumed"

    @pytest.mark.anyio
    async def test_dispatch_routes_cancel(self):
        handler = _make_handler()
        await handler._dispatch({"type": "cancel"})

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "cancelled"

    @pytest.mark.anyio
    async def test_dispatch_routes_set_batch_size(self):
        handler = _make_handler()
        await handler._dispatch({"type": "set_batch_size", "batch_size": 2000})

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "batch_size_set"

    @pytest.mark.anyio
    async def test_dispatch_routes_set_target_seconds(self):
        handler = _make_handler()
        await handler._dispatch({"type": "set_target_seconds", "target_seconds": 15.0})

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "target_seconds_set"


# ── Test: Handle (main loop) ──────────────────────────────────────────


class TestHandle:
    @pytest.mark.anyio
    async def test_accepts_and_dispatches(self):
        ws = _make_mock_ws()
        call_count = 0

        async def receive_side():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"type": "pause"}
            raise WebSocketDisconnect()

        ws.receive_json = AsyncMock(side_effect=receive_side)
        handler = _make_handler(ws=ws)

        await handler.handle()

        ws.accept.assert_called_once()
        assert handler.control["paused"] is True

    @pytest.mark.anyio
    async def test_immediate_disconnect_exits_cleanly(self):
        ws = _make_mock_ws()
        ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())
        handler = _make_handler(ws=ws)

        await handler.handle()

        ws.accept.assert_called_once()


# ── Test: _handle_start ───────────────────────────────────────────────


class TestHandleStart:
    @pytest.mark.anyio
    async def test_missing_request_sends_error(self):
        handler = _make_handler()
        await handler._handle_start({"type": "start"})

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "error"
        assert "request" in msg["message"].lower()

    @pytest.mark.anyio
    async def test_invalid_request_sends_error(self):
        handler = _make_handler()
        await handler._handle_start({"type": "start", "request": {"invalid": "data"}})

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "error"
        assert "Invalid QueryRequest" in msg["message"]

    @pytest.mark.anyio
    async def test_valid_start_creates_run_and_sends_run_started(self):
        handler = _make_handler()
        run = _make_run()

        with (
            patch("easyweaver.queries.ws_handler.service") as mock_service,
            patch("asyncio.create_task") as mock_create_task,
        ):
            mock_service.create_query_run = AsyncMock(return_value=run)
            captured_coro = None

            def capture_task(coro):
                nonlocal captured_coro
                captured_coro = coro
                return MagicMock(done=MagicMock(return_value=False))

            mock_create_task.side_effect = capture_task

            await handler._handle_start(_valid_start_msg())

        mock_service.create_query_run.assert_called_once()
        assert handler.run_id == str(run.id)

        sent_msgs = [c[0][0] for c in handler.ws.send_json.call_args_list]
        started = [m for m in sent_msgs if m.get("type") == "run_started"]
        assert len(started) == 1
        assert started[0]["run_id"] == str(run.id)

        if captured_coro is not None:
            captured_coro.close()

    @pytest.mark.anyio
    async def test_start_sets_target_seconds_from_msg(self):
        handler = _make_handler()
        run = _make_run()

        with (
            patch("easyweaver.queries.ws_handler.service") as mock_service,
            patch("asyncio.create_task") as mock_create_task,
        ):
            mock_service.create_query_run = AsyncMock(return_value=run)
            captured_coro = None

            def capture_task(coro):
                nonlocal captured_coro
                captured_coro = coro
                return MagicMock(done=MagicMock(return_value=False))

            mock_create_task.side_effect = capture_task

            msg = _valid_start_msg()
            msg["target_batch_seconds"] = 20.0
            await handler._handle_start(msg)

        assert handler.control["target_batch_seconds"] == 20.0

        if captured_coro is not None:
            captured_coro.close()

    @pytest.mark.anyio
    async def test_start_registers_in_active_handlers(self):
        handler = _make_handler()
        run = _make_run()

        with (
            patch("easyweaver.queries.ws_handler.service") as mock_service,
            patch("asyncio.create_task") as mock_create_task,
        ):
            mock_service.create_query_run = AsyncMock(return_value=run)
            captured_coro = None

            def capture_task(coro):
                nonlocal captured_coro
                captured_coro = coro
                return MagicMock(done=MagicMock(return_value=False))

            mock_create_task.side_effect = capture_task

            await handler._handle_start(_valid_start_msg())

        run_id_str = str(run.id)
        assert _active_query_handlers.get(run_id_str) is handler

        # Cleanup
        _active_query_handlers.pop(run_id_str, None)
        if captured_coro is not None:
            captured_coro.close()

    @pytest.mark.anyio
    async def test_start_run_started_message_has_expected_keys(self):
        handler = _make_handler()
        run = _make_run()

        with (
            patch("easyweaver.queries.ws_handler.service") as mock_service,
            patch("asyncio.create_task") as mock_create_task,
        ):
            mock_service.create_query_run = AsyncMock(return_value=run)
            captured_coro = None

            def capture_task(coro):
                nonlocal captured_coro
                captured_coro = coro
                return MagicMock(done=MagicMock(return_value=False))

            mock_create_task.side_effect = capture_task

            await handler._handle_start(_valid_start_msg())

        sent_msgs = [c[0][0] for c in handler.ws.send_json.call_args_list]
        started = next(m for m in sent_msgs if m.get("type") == "run_started")
        assert "phases" in started
        assert "datasets" in started
        assert "dag" in started

        _active_query_handlers.pop(str(run.id), None)
        if captured_coro is not None:
            captured_coro.close()


# ── Test: _handle_pause / _handle_resume ─────────────────────────────


class TestPauseResume:
    @pytest.mark.anyio
    async def test_pause_sets_flag(self):
        handler = _make_handler()
        await handler._handle_pause({})
        assert handler.control["paused"] is True

    @pytest.mark.anyio
    async def test_pause_sends_paused(self):
        handler = _make_handler()
        await handler._handle_pause({})
        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "paused"

    @pytest.mark.anyio
    async def test_pause_persists_control_to_db_when_run_exists(self):
        handler = _make_handler()
        handler.run_id = _RUN_ID

        with patch("easyweaver.queries.ws_handler.service") as mock_service:
            mock_service.update_query_run_progress = AsyncMock()
            await handler._handle_pause({})

        mock_service.update_query_run_progress.assert_called_once_with(
            handler.db, _RUN_ID, control=handler.control
        )

    @pytest.mark.anyio
    async def test_pause_does_not_call_db_when_no_run_id(self):
        handler = _make_handler()
        # run_id is None initially

        with patch("easyweaver.queries.ws_handler.service") as mock_service:
            mock_service.update_query_run_progress = AsyncMock()
            await handler._handle_pause({})

        mock_service.update_query_run_progress.assert_not_called()

    @pytest.mark.anyio
    async def test_resume_clears_flag(self):
        handler = _make_handler()
        handler.control["paused"] = True
        await handler._handle_resume({})
        assert handler.control["paused"] is False

    @pytest.mark.anyio
    async def test_resume_sends_resumed(self):
        handler = _make_handler()
        await handler._handle_resume({})
        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "resumed"

    @pytest.mark.anyio
    async def test_resume_persists_control_when_run_exists(self):
        handler = _make_handler()
        handler.run_id = _RUN_ID

        with patch("easyweaver.queries.ws_handler.service") as mock_service:
            mock_service.update_query_run_progress = AsyncMock()
            await handler._handle_resume({})

        mock_service.update_query_run_progress.assert_called_once()


# ── Test: _handle_set_batch_size ──────────────────────────────────────


class TestSetBatchSize:
    @pytest.mark.anyio
    async def test_sets_override_and_disables_adaptive(self):
        handler = _make_handler()
        await handler._handle_set_batch_size({"batch_size": 5000})

        assert handler.control["batch_size_override"] == 5000
        assert handler.control["adaptive_enabled"] is False

    @pytest.mark.anyio
    async def test_sends_batch_size_set_response(self):
        handler = _make_handler()
        await handler._handle_set_batch_size({"batch_size": 5000})

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "batch_size_set"
        assert msg["batch_size"] == 5000

    @pytest.mark.anyio
    async def test_none_batch_size_clears_override(self):
        handler = _make_handler()
        handler.control["batch_size_override"] = 5000
        await handler._handle_set_batch_size({"batch_size": None})

        assert handler.control["batch_size_override"] is None


# ── Test: _handle_set_target_seconds ─────────────────────────────────


class TestSetTargetSeconds:
    @pytest.mark.anyio
    async def test_sets_target_seconds_and_enables_adaptive(self):
        handler = _make_handler()
        handler.control["batch_size_override"] = 5000
        handler.control["adaptive_enabled"] = False

        await handler._handle_set_target_seconds({"target_seconds": 15.0})

        assert handler.control["target_batch_seconds"] == 15.0
        assert handler.control["adaptive_enabled"] is True
        assert handler.control["batch_size_override"] is None

    @pytest.mark.anyio
    async def test_default_target_seconds_is_10(self):
        handler = _make_handler()
        await handler._handle_set_target_seconds({})

        assert handler.control["target_batch_seconds"] == 10.0

    @pytest.mark.anyio
    async def test_sends_target_seconds_set_response(self):
        handler = _make_handler()
        await handler._handle_set_target_seconds({"target_seconds": 20.0})

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "target_seconds_set"
        assert msg["target_seconds"] == 20.0


# ── Test: _handle_cancel ──────────────────────────────────────────────


class TestCancel:
    @pytest.mark.anyio
    async def test_sets_cancelled_flag(self):
        handler = _make_handler()
        await handler._handle_cancel({})
        assert handler.control["cancelled"] is True

    @pytest.mark.anyio
    async def test_sends_cancelled_response(self):
        handler = _make_handler()
        await handler._handle_cancel({})
        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "cancelled"

    @pytest.mark.anyio
    async def test_cancels_running_task(self):
        handler = _make_handler()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel = MagicMock()
        handler._execution_task = mock_task

        await handler._handle_cancel({})

        mock_task.cancel.assert_called_once()

    @pytest.mark.anyio
    async def test_does_not_cancel_done_task(self):
        handler = _make_handler()
        mock_task = MagicMock()
        mock_task.done.return_value = True
        mock_task.cancel = MagicMock()
        handler._execution_task = mock_task

        await handler._handle_cancel({})

        mock_task.cancel.assert_not_called()

    @pytest.mark.anyio
    async def test_persists_control_when_run_exists(self):
        handler = _make_handler()
        handler.run_id = _RUN_ID

        with patch("easyweaver.queries.ws_handler.service") as mock_service:
            mock_service.update_query_run_progress = AsyncMock()
            await handler._handle_cancel({})

        mock_service.update_query_run_progress.assert_called_once_with(
            handler.db, _RUN_ID, control=handler.control
        )

    @pytest.mark.anyio
    async def test_cancel_with_no_task(self):
        handler = _make_handler()
        # _execution_task is None — should not raise
        await handler._handle_cancel({})
        assert handler.control["cancelled"] is True


# ── Test: _send helper ────────────────────────────────────────────────


class TestSend:
    @pytest.mark.anyio
    async def test_sends_json_to_ws(self):
        handler = _make_handler()
        await handler._send({"type": "test", "value": 42})

        handler.ws.send_json.assert_called_once_with({"type": "test", "value": 42})

    @pytest.mark.anyio
    async def test_swallows_send_exception(self):
        handler = _make_handler()
        handler.ws.send_json = AsyncMock(side_effect=RuntimeError("disconnected"))

        # Should not raise
        await handler._send({"type": "test"})


# ── Test: _progress_callback ──────────────────────────────────────────


class TestProgressCallback:
    @pytest.mark.anyio
    async def test_sends_event_to_ws(self):
        handler = _make_handler()
        handler.run_id = _RUN_ID

        with patch("easyweaver.queries.ws_handler.service") as mock_service:
            mock_service.update_query_run_progress = AsyncMock()
            await handler._progress_callback(
                "fetch_progress", dataset="query", rows_fetched=100
            )

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["type"] == "fetch_progress"
        assert msg["dataset"] == "query"
        assert msg["rows_fetched"] == 100

    @pytest.mark.anyio
    async def test_persists_on_fetch_complete(self):
        handler = _make_handler()
        handler.run_id = _RUN_ID

        with patch("easyweaver.queries.ws_handler.service") as mock_service:
            mock_service.update_query_run_progress = AsyncMock()
            await handler._progress_callback(
                "fetch_complete", dataset="query", total_rows=50
            )

        mock_service.update_query_run_progress.assert_called_once()
        call_kwargs = mock_service.update_query_run_progress.call_args
        progress_dict = call_kwargs[1].get("progress") or call_kwargs[0][2]
        assert progress_dict["last_event"] == "fetch_complete"

    @pytest.mark.anyio
    async def test_persists_on_completed(self):
        handler = _make_handler()
        handler.run_id = _RUN_ID

        with patch("easyweaver.queries.ws_handler.service") as mock_service:
            mock_service.update_query_run_progress = AsyncMock()
            await handler._progress_callback("completed", total_rows=50)

        mock_service.update_query_run_progress.assert_called_once()

    @pytest.mark.anyio
    async def test_persists_on_error(self):
        handler = _make_handler()
        handler.run_id = _RUN_ID

        with patch("easyweaver.queries.ws_handler.service") as mock_service:
            mock_service.update_query_run_progress = AsyncMock()
            await handler._progress_callback("error", message="Something failed")

        mock_service.update_query_run_progress.assert_called_once()

    @pytest.mark.anyio
    async def test_no_persist_on_fetch_progress(self):
        """fetch_progress is not a key event — should not be persisted."""
        handler = _make_handler()
        handler.run_id = _RUN_ID

        with patch("easyweaver.queries.ws_handler.service") as mock_service:
            mock_service.update_query_run_progress = AsyncMock()
            await handler._progress_callback(
                "fetch_progress", rows_fetched=100
            )

        mock_service.update_query_run_progress.assert_not_called()

    @pytest.mark.anyio
    async def test_persist_failure_caught_by_except_block(self):
        """DB update failures are caught in the except block of _progress_callback."""
        import structlog

        handler = _make_handler()
        handler.run_id = _RUN_ID

        # Patch both the service (to fail) and logger.warning (to avoid structlog kwarg clash)
        with (
            patch("easyweaver.queries.ws_handler.service") as mock_service,
            patch("easyweaver.queries.ws_handler.logger") as mock_logger,
        ):
            mock_service.update_query_run_progress = AsyncMock(
                side_effect=Exception("DB error")
            )
            mock_logger.warning = MagicMock()

            # Should not raise — exception is caught and logged
            await handler._progress_callback("fetch_complete", total_rows=10)

        # WS should still have received the send
        handler.ws.send_json.assert_called_once()
        # logger.warning should have been called with the error info
        mock_logger.warning.assert_called_once()

    @pytest.mark.anyio
    async def test_callback_sends_all_data_kwargs(self):
        handler = _make_handler()
        handler.run_id = _RUN_ID

        with patch("easyweaver.queries.ws_handler.service") as mock_service:
            mock_service.update_query_run_progress = AsyncMock()
            await handler._progress_callback(
                "batch_adjusted",
                old_batch_size=10000,
                new_batch_size=20000,
                reason="adaptive",
            )

        msg = handler.ws.send_json.call_args[0][0]
        assert msg["old_batch_size"] == 10000
        assert msg["new_batch_size"] == 20000
        assert msg["reason"] == "adaptive"


# ── Test: _progress_callback with no run_id ───────────────────────────


class TestProgressCallbackNoRunId:
    @pytest.mark.anyio
    async def test_still_sends_to_ws_when_no_run_id(self):
        handler = _make_handler()
        # run_id is None

        with patch("easyweaver.queries.ws_handler.service") as mock_service:
            mock_service.update_query_run_progress = AsyncMock()
            await handler._progress_callback(
                "fetch_complete", total_rows=5
            )

        # Should still send to WS
        handler.ws.send_json.assert_called_once()
        # But DB persist will be called (run_id is None though, so it will persist with None)
        # The actual persist call passes run_id=None — service handles it
