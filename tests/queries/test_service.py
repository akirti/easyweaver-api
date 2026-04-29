"""Tests for easyweaver.queries.service — CRUD operations."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from easyweaver.core.exceptions import NotFoundError
from easyweaver.queries import service
from easyweaver.queries.models import QueryRun
from easyweaver.queries.schemas import QueryRequest, QuerySourceConfig


# ── Helpers ───────────────────────────────────────────────────────────

_RUN_ID = "12345678-1234-1234-1234-123456789abc"
_SOURCE_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_NOW = datetime(2026, 4, 29, 10, 0, 0, tzinfo=timezone.utc)


def _make_mock_db(run_doc: dict | None = None) -> MagicMock:
    db = MagicMock()
    db.query_runs = MagicMock()
    db.query_runs.insert_one = AsyncMock()
    db.query_runs.update_one = AsyncMock()
    db.query_runs.find_one = AsyncMock(return_value=run_doc)
    return db


def _make_run_doc(
    run_id: str = _RUN_ID,
    status: str = "pending",
    **kwargs,
) -> dict:
    return {
        "_id": run_id,
        "config": '{"type": "single"}',
        "status": status,
        "row_count": None,
        "error": None,
        "progress": None,
        "control": None,
        "created_at": _NOW,
        "updated_at": _NOW,
        **kwargs,
    }


def _make_query_request() -> QueryRequest:
    return QueryRequest(
        type="single",
        left=QuerySourceConfig(
            source_id=uuid.UUID(_SOURCE_ID),
            table="orders",
        ),
    )


# ── Tests: create_query_run ───────────────────────────────────────────


class TestCreateQueryRun:
    @pytest.mark.anyio
    async def test_creates_doc_in_db(self):
        db = _make_mock_db()
        request = _make_query_request()

        run = await service.create_query_run(db, request)

        db.query_runs.insert_one.assert_called_once()
        inserted_doc = db.query_runs.insert_one.call_args[0][0]
        assert inserted_doc["status"] == "pending"
        assert "_id" in inserted_doc
        assert "config" in inserted_doc

    @pytest.mark.anyio
    async def test_returns_query_run_with_pending_status(self):
        db = _make_mock_db()
        request = _make_query_request()

        run = await service.create_query_run(db, request)

        assert isinstance(run, QueryRun)
        assert run.status == "pending"
        assert isinstance(run.id, uuid.UUID)

    @pytest.mark.anyio
    async def test_config_is_serialized_json(self):
        db = _make_mock_db()
        request = _make_query_request()

        run = await service.create_query_run(db, request)

        assert isinstance(run.config, str)
        import json
        parsed = json.loads(run.config)
        assert parsed["type"] == "single"

    @pytest.mark.anyio
    async def test_timestamps_are_set(self):
        db = _make_mock_db()
        request = _make_query_request()

        run = await service.create_query_run(db, request)

        assert run.created_at is not None
        assert run.updated_at is not None
        assert run.created_at.tzinfo is not None


# ── Tests: get_query_run ──────────────────────────────────────────────


class TestGetQueryRun:
    @pytest.mark.anyio
    async def test_found_by_uuid(self):
        doc = _make_run_doc()
        db = _make_mock_db(run_doc=doc)

        run = await service.get_query_run(db, uuid.UUID(_RUN_ID))

        db.query_runs.find_one.assert_called_once_with({"_id": _RUN_ID})
        assert run.id == uuid.UUID(_RUN_ID)
        assert run.status == "pending"

    @pytest.mark.anyio
    async def test_found_by_string_id(self):
        doc = _make_run_doc()
        db = _make_mock_db(run_doc=doc)

        run = await service.get_query_run(db, _RUN_ID)

        db.query_runs.find_one.assert_called_once_with({"_id": _RUN_ID})
        assert run.id == uuid.UUID(_RUN_ID)

    @pytest.mark.anyio
    async def test_not_found_raises_not_found_error(self):
        db = _make_mock_db(run_doc=None)

        with pytest.raises(NotFoundError) as exc_info:
            await service.get_query_run(db, _RUN_ID)

        assert "QueryRun" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_returns_query_run_instance(self):
        doc = _make_run_doc(status="completed", row_count=50)
        db = _make_mock_db(run_doc=doc)

        run = await service.get_query_run(db, _RUN_ID)

        assert isinstance(run, QueryRun)
        assert run.status == "completed"
        assert run.row_count == 50


# ── Tests: update_query_run ───────────────────────────────────────────


class TestUpdateQueryRun:
    @pytest.mark.anyio
    async def test_updates_status(self):
        doc = _make_run_doc(status="running")
        db = _make_mock_db(run_doc=doc)

        run = await service.update_query_run(db, _RUN_ID, status="running")

        db.query_runs.update_one.assert_called_once()
        call_args = db.query_runs.update_one.call_args
        assert call_args[0][0] == {"_id": _RUN_ID}
        updates = call_args[0][1]["$set"]
        assert updates["status"] == "running"

    @pytest.mark.anyio
    async def test_updates_row_count(self):
        doc = _make_run_doc(status="completed", row_count=100)
        db = _make_mock_db(run_doc=doc)

        await service.update_query_run(db, _RUN_ID, row_count=100)

        call_args = db.query_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert updates["row_count"] == 100

    @pytest.mark.anyio
    async def test_updates_error(self):
        doc = _make_run_doc(status="failed", error="timeout")
        db = _make_mock_db(run_doc=doc)

        await service.update_query_run(db, _RUN_ID, error="timeout")

        call_args = db.query_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert updates["error"] == "timeout"

    @pytest.mark.anyio
    async def test_always_sets_updated_at(self):
        doc = _make_run_doc()
        db = _make_mock_db(run_doc=doc)

        await service.update_query_run(db, _RUN_ID)

        call_args = db.query_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert "updated_at" in updates

    @pytest.mark.anyio
    async def test_none_status_not_included_in_update(self):
        doc = _make_run_doc()
        db = _make_mock_db(run_doc=doc)

        await service.update_query_run(db, _RUN_ID, status=None)

        call_args = db.query_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert "status" not in updates

    @pytest.mark.anyio
    async def test_returns_updated_run(self):
        doc = _make_run_doc(status="completed")
        db = _make_mock_db(run_doc=doc)

        run = await service.update_query_run(db, _RUN_ID, status="completed")

        assert isinstance(run, QueryRun)

    @pytest.mark.anyio
    async def test_accepts_uuid_run_id(self):
        doc = _make_run_doc()
        db = _make_mock_db(run_doc=doc)

        await service.update_query_run(db, uuid.UUID(_RUN_ID), status="running")

        call_args = db.query_runs.update_one.call_args
        filter_doc = call_args[0][0]
        assert filter_doc == {"_id": _RUN_ID}


# ── Tests: update_query_run_progress ─────────────────────────────────


class TestUpdateQueryRunProgress:
    @pytest.mark.anyio
    async def test_updates_progress(self):
        db = _make_mock_db()
        progress = {"last_event": "fetch_progress", "rows_fetched": 5000}

        await service.update_query_run_progress(db, _RUN_ID, progress=progress)

        db.query_runs.update_one.assert_called_once()
        call_args = db.query_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert updates["progress"] == progress

    @pytest.mark.anyio
    async def test_updates_control(self):
        db = _make_mock_db()
        control = {"paused": True, "cancelled": False}

        await service.update_query_run_progress(db, _RUN_ID, control=control)

        call_args = db.query_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert updates["control"] == control

    @pytest.mark.anyio
    async def test_updates_both_progress_and_control(self):
        db = _make_mock_db()
        progress = {"rows_fetched": 1000}
        control = {"paused": False}

        await service.update_query_run_progress(db, _RUN_ID, progress=progress, control=control)

        call_args = db.query_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert updates["progress"] == progress
        assert updates["control"] == control

    @pytest.mark.anyio
    async def test_always_sets_updated_at(self):
        db = _make_mock_db()

        await service.update_query_run_progress(db, _RUN_ID, progress={})

        call_args = db.query_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert "updated_at" in updates

    @pytest.mark.anyio
    async def test_none_progress_not_included(self):
        db = _make_mock_db()

        await service.update_query_run_progress(db, _RUN_ID, control={"paused": True})

        call_args = db.query_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert "progress" not in updates

    @pytest.mark.anyio
    async def test_none_control_not_included(self):
        db = _make_mock_db()

        await service.update_query_run_progress(db, _RUN_ID, progress={"x": 1})

        call_args = db.query_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert "control" not in updates

    @pytest.mark.anyio
    async def test_returns_none(self):
        db = _make_mock_db()

        result = await service.update_query_run_progress(db, _RUN_ID)

        assert result is None
