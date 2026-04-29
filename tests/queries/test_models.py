"""Tests for easyweaver.queries.models — QueryRun dataclass."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from easyweaver.queries.models import QueryRun


# ── Helpers ───────────────────────────────────────────────────────────

_RUN_ID = "12345678-1234-1234-1234-123456789abc"
_NOW = datetime(2026, 4, 29, 10, 0, 0, tzinfo=timezone.utc)


def _minimal_doc() -> dict:
    return {
        "_id": _RUN_ID,
        "config": '{"type": "single"}',
        "status": "pending",
        "row_count": None,
        "error": None,
        "progress": None,
        "control": None,
        "created_at": _NOW,
        "updated_at": _NOW,
    }


# ── Tests: from_doc ───────────────────────────────────────────────────


class TestFromDoc:
    def test_minimal_doc(self):
        doc = _minimal_doc()
        run = QueryRun.from_doc(doc)
        assert run.id == uuid.UUID(_RUN_ID)
        assert run.config == '{"type": "single"}'
        assert run.status == "pending"
        assert run.row_count is None
        assert run.error is None
        assert run.progress is None
        assert run.control is None
        assert run.created_at == _NOW
        assert run.updated_at == _NOW

    def test_completed_doc_with_all_fields(self):
        doc = _minimal_doc()
        doc.update(
            {
                "status": "completed",
                "row_count": 42,
                "error": None,
                "progress": {"last_event": "fetch_complete", "total_rows": 42},
                "control": {"paused": False, "cancelled": False},
            }
        )
        run = QueryRun.from_doc(doc)
        assert run.status == "completed"
        assert run.row_count == 42
        assert run.progress == {"last_event": "fetch_complete", "total_rows": 42}
        assert run.control == {"paused": False, "cancelled": False}

    def test_failed_doc_with_error(self):
        doc = _minimal_doc()
        doc["status"] = "failed"
        doc["error"] = "Connection refused"
        run = QueryRun.from_doc(doc)
        assert run.status == "failed"
        assert run.error == "Connection refused"

    def test_status_defaults_to_pending_when_missing(self):
        doc = _minimal_doc()
        del doc["status"]
        run = QueryRun.from_doc(doc)
        assert run.status == "pending"

    def test_created_at_defaults_when_missing(self):
        doc = _minimal_doc()
        del doc["created_at"]
        del doc["updated_at"]
        run = QueryRun.from_doc(doc)
        # Should get current time — just assert it is a datetime
        assert isinstance(run.created_at, datetime)
        assert isinstance(run.updated_at, datetime)

    def test_progress_and_control_optional(self):
        doc = _minimal_doc()
        # Omit optional fields entirely (use .get() with None default)
        del doc["progress"]
        del doc["control"]
        run = QueryRun.from_doc(doc)
        assert run.progress is None
        assert run.control is None


# ── Tests: to_doc ─────────────────────────────────────────────────────


class TestToDoc:
    def test_round_trip(self):
        run = QueryRun(
            id=uuid.UUID(_RUN_ID),
            config='{"type": "single"}',
            status="completed",
            row_count=10,
            error=None,
            progress={"last_event": "fetch_complete"},
            control={"paused": False},
            created_at=_NOW,
            updated_at=_NOW,
        )
        doc = run.to_doc()
        assert doc["_id"] == _RUN_ID
        assert doc["config"] == '{"type": "single"}'
        assert doc["status"] == "completed"
        assert doc["row_count"] == 10
        assert doc["error"] is None
        assert doc["progress"] == {"last_event": "fetch_complete"}
        assert doc["control"] == {"paused": False}
        assert doc["created_at"] == _NOW
        assert doc["updated_at"] == _NOW

    def test_id_stored_as_string(self):
        run = QueryRun(
            id=uuid.UUID(_RUN_ID),
            config="{}",
            created_at=_NOW,
            updated_at=_NOW,
        )
        doc = run.to_doc()
        assert doc["_id"] == _RUN_ID
        assert isinstance(doc["_id"], str)

    def test_null_fields_preserved(self):
        run = QueryRun(
            id=uuid.UUID(_RUN_ID),
            config="{}",
            created_at=_NOW,
            updated_at=_NOW,
        )
        doc = run.to_doc()
        assert "row_count" in doc
        assert doc["row_count"] is None
        assert "error" in doc
        assert doc["error"] is None


# ── Tests: Default values ─────────────────────────────────────────────


class TestDefaults:
    def test_default_status_is_pending(self):
        run = QueryRun(id=uuid.uuid4(), config="{}")
        assert run.status == "pending"

    def test_default_optional_fields_are_none(self):
        run = QueryRun(id=uuid.uuid4(), config="{}")
        assert run.row_count is None
        assert run.error is None
        assert run.progress is None
        assert run.control is None

    def test_default_timestamps_are_utc(self):
        run = QueryRun(id=uuid.uuid4(), config="{}")
        assert run.created_at.tzinfo is not None
        assert run.updated_at.tzinfo is not None
