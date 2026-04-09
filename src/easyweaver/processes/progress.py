"""Progress tracking for process runs.

Manages in-memory progress state and persists it to MongoDB
so that clients can poll or receive real-time updates via WebSocket.
"""

from __future__ import annotations

import copy
import time
from datetime import datetime, timezone

import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = structlog.get_logger()

# Minimum interval (seconds) between MongoDB writes for non-critical events.
_DEBOUNCE_INTERVAL = 1.0


class ProcessProgressTracker:
    """Tracks the progress of a single process run.

    Maintains an in-memory ``_state`` dict and persists it to the
    ``process_runs`` collection's ``progress`` field after each mutation.

    To avoid excessive MongoDB writes, persistence is debounced: at most one
    write per ``_DEBOUNCE_INTERVAL`` seconds for non-critical updates.
    Phase changes and completion are always persisted immediately.
    """

    def __init__(self, run_id: str, db: AsyncIOMotorDatabase):
        self.run_id = run_id
        self.db = db
        self._last_persist: float = 0.0
        self._state: dict = {
            "phase": "pending",
            "phase_index": 0,
            "total_phases": 3,
            "datasets": {},
            "current_operation": None,
            "paused": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Phase management ──────────────────────────────────────────────

    async def set_phase(self, phase: str, phase_index: int) -> None:
        """Update the current phase (e.g. fetching / joining / transforming)."""
        self._state["phase"] = phase
        self._state["phase_index"] = phase_index
        await self._persist(force=True)

    # ── Dataset tracking ──────────────────────────────────────────────

    async def init_dataset(self, key: str, depends_on: list[str] | None = None) -> None:
        """Register a dataset in the progress state."""
        self._state["datasets"][key] = {
            "status": "waiting",
            "depends_on": depends_on or [],
            "rows_fetched": 0,
            "batch_number": 0,
            "batch_size": 0,
            "error": None,
        }
        await self._persist(force=True)

    async def update_dataset(self, key: str, **kwargs) -> None:
        """Update arbitrary fields on a dataset's progress entry."""
        ds = self._state["datasets"].get(key)
        if ds is None:
            logger.warning("progress_update_unknown_dataset", run_id=self.run_id, key=key)
            return
        ds.update(kwargs)
        await self._persist()

    async def set_dataset_status(self, key: str, status: str) -> None:
        """Quick status update (waiting / fetching / paused / completed / failed)."""
        await self.update_dataset(key, status=status)

    # ── Global controls ───────────────────────────────────────────────

    async def set_paused(self, paused: bool) -> None:
        """Update pause state."""
        self._state["paused"] = paused
        await self._persist(force=True)

    async def set_current_operation(self, operation: str | None) -> None:
        """Update the current transform/join operation name."""
        self._state["current_operation"] = operation
        await self._persist()

    # ── Snapshot / persistence ────────────────────────────────────────

    def get_snapshot(self) -> dict:
        """Return a deep copy of the current progress state."""
        return copy.deepcopy(self._state)

    async def _persist(self, *, force: bool = False) -> None:
        """Write ``self._state`` to MongoDB: ``process_runs.progress`` field.

        Uses ``$set`` on the ``progress`` field only (not full-doc replacement).

        When *force* is ``False`` the write is skipped if less than
        ``_DEBOUNCE_INTERVAL`` seconds have passed since the last write,
        keeping MongoDB load manageable during high-frequency fetch_progress
        events.
        """
        now = time.monotonic()
        if not force and (now - self._last_persist) < _DEBOUNCE_INTERVAL:
            return
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        await self.db.process_runs.update_one(
            {"_id": self.run_id},
            {"$set": {"progress": self._state}},
        )
        self._last_persist = now
