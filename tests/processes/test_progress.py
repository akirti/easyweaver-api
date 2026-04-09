"""Tests for easyweaver.processes.progress — ProcessProgressTracker."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from easyweaver.processes.progress import ProcessProgressTracker


def _make_mock_db() -> MagicMock:
    """Create a mock AsyncIOMotorDatabase with a process_runs collection."""
    db = MagicMock()
    db.process_runs = MagicMock()
    db.process_runs.update_one = AsyncMock()
    return db


# ── Initialisation ────────────────────────────────────────────────────


class TestInit:
    def test_default_state(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)
        snap = tracker.get_snapshot()

        assert snap["phase"] == "pending"
        assert snap["phase_index"] == 0
        assert snap["total_phases"] == 3
        assert snap["datasets"] == {}
        assert snap["current_operation"] is None
        assert snap["paused"] is False
        assert "updated_at" in snap


# ── set_phase ─────────────────────────────────────────────────────────


class TestSetPhase:
    @pytest.mark.asyncio
    async def test_updates_phase(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        await tracker.set_phase("fetching", 1)

        snap = tracker.get_snapshot()
        assert snap["phase"] == "fetching"
        assert snap["phase_index"] == 1

    @pytest.mark.asyncio
    async def test_persists_to_db(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        await tracker.set_phase("joining", 2)

        db.process_runs.update_one.assert_called()
        call_args = db.process_runs.update_one.call_args
        assert call_args[0][0] == {"_id": "run-1"}
        assert "$set" in call_args[0][1]
        assert "progress" in call_args[0][1]["$set"]


# ── init_dataset ──────────────────────────────────────────────────────


class TestInitDataset:
    @pytest.mark.asyncio
    async def test_registers_dataset(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        await tracker.init_dataset("orders.main", depends_on=["customers.main"])

        snap = tracker.get_snapshot()
        ds = snap["datasets"]["orders.main"]
        assert ds["status"] == "waiting"
        assert ds["depends_on"] == ["customers.main"]
        assert ds["rows_fetched"] == 0
        assert ds["batch_number"] == 0

    @pytest.mark.asyncio
    async def test_no_depends(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        await tracker.init_dataset("orders.main")

        snap = tracker.get_snapshot()
        assert snap["datasets"]["orders.main"]["depends_on"] == []


# ── update_dataset ────────────────────────────────────────────────────


class TestUpdateDataset:
    @pytest.mark.asyncio
    async def test_updates_fields(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        await tracker.init_dataset("orders.main")
        await tracker.update_dataset("orders.main", rows_fetched=500, batch_number=3, batch_size=200)

        snap = tracker.get_snapshot()
        ds = snap["datasets"]["orders.main"]
        assert ds["rows_fetched"] == 500
        assert ds["batch_number"] == 3
        assert ds["batch_size"] == 200

    @pytest.mark.asyncio
    async def test_unknown_dataset_logs_warning(self):
        """Updating an unregistered dataset should not raise."""
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        # Should not raise — just logs a warning
        await tracker.update_dataset("nonexistent", rows_fetched=10)

        # Only the initial state persisted (no dataset added)
        snap = tracker.get_snapshot()
        assert "nonexistent" not in snap["datasets"]


# ── set_dataset_status ────────────────────────────────────────────────


class TestSetDatasetStatus:
    @pytest.mark.asyncio
    async def test_sets_status(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        await tracker.init_dataset("orders.main")
        await tracker.set_dataset_status("orders.main", "fetching")

        snap = tracker.get_snapshot()
        assert snap["datasets"]["orders.main"]["status"] == "fetching"


# ── set_paused ────────────────────────────────────────────────────────


class TestSetPaused:
    @pytest.mark.asyncio
    async def test_pause_and_resume(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        await tracker.set_paused(True)
        assert tracker.get_snapshot()["paused"] is True

        await tracker.set_paused(False)
        assert tracker.get_snapshot()["paused"] is False

    @pytest.mark.asyncio
    async def test_persist_called_on_pause(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        await tracker.set_paused(True)
        db.process_runs.update_one.assert_called()


# ── set_current_operation ─────────────────────────────────────────────


class TestSetCurrentOperation:
    @pytest.mark.asyncio
    async def test_set_and_clear(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        await tracker.set_current_operation("join: orders x customers")
        assert tracker.get_snapshot()["current_operation"] == "join: orders x customers"

        await tracker.set_current_operation(None)
        assert tracker.get_snapshot()["current_operation"] is None


# ── get_snapshot ──────────────────────────────────────────────────────


class TestGetSnapshot:
    def test_returns_deep_copy(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        snap1 = tracker.get_snapshot()
        snap1["phase"] = "mutated"

        snap2 = tracker.get_snapshot()
        assert snap2["phase"] == "pending"  # original unchanged


# ── _persist ──────────────────────────────────────────────────────────


class TestPersist:
    @pytest.mark.asyncio
    async def test_uses_set_on_progress_field(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        await tracker.set_phase("fetching", 1)

        call_args = db.process_runs.update_one.call_args
        filter_doc, update_doc = call_args[0]
        assert filter_doc == {"_id": "run-1"}
        assert list(update_doc.keys()) == ["$set"]
        assert list(update_doc["$set"].keys()) == ["progress"]
        progress = update_doc["$set"]["progress"]
        assert progress["phase"] == "fetching"

    @pytest.mark.asyncio
    async def test_updates_timestamp_on_each_persist(self):
        db = _make_mock_db()
        tracker = ProcessProgressTracker("run-1", db)

        await tracker.set_phase("fetching", 1)
        ts1 = tracker.get_snapshot()["updated_at"]

        await tracker.set_phase("joining", 2)
        ts2 = tracker.get_snapshot()["updated_at"]

        assert ts1 <= ts2
