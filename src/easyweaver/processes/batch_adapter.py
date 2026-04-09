"""Adaptive batch sizing for streamed/batched data fetches.

Adjusts batch sizes dynamically based on measured throughput so each batch
completes close to a configurable target time (default 10 s).  Historical
optimal sizes are persisted in MongoDB so subsequent runs start near their
ideal batch size.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorDatabase


def _get_settings():  # noqa: ANN202
    """Lazy import to avoid triggering Settings() at module load time."""
    from easyweaver.settings import settings

    return settings

# ── Connector-aware defaults ──────────────────────────────────────────
_CONNECTOR_DEFAULTS: dict[str, int] = {
    "postgres": 15_000,
    "mysql": 10_000,
    "db2": 5_000,
    "mongodb": 10_000,
}
_FALLBACK_DEFAULT: int = 10_000


# ── Public API ────────────────────────────────────────────────────────

def adapt_batch_size(
    current_batch_size: int,
    batch_time_seconds: float,
    target_seconds: float,
    max_rows_remaining: int,
) -> int:
    """Return the next batch size adapted from measured throughput.

    Uses 50 % dampening toward the ideal size to avoid oscillation, then
    clamps the result to ``[batch_min_size, batch_max_size]`` and ensures
    it does not exceed *max_rows_remaining*.
    """
    s = _get_settings()

    # Guard against degenerate inputs
    if current_batch_size <= 0 or batch_time_seconds <= 0 or target_seconds <= 0:
        clamped = max(s.batch_min_size, min(s.batch_max_size, current_batch_size))
        if max_rows_remaining > 0:
            clamped = min(clamped, max_rows_remaining)
        return max(clamped, s.batch_min_size)

    rows_per_second = current_batch_size / batch_time_seconds
    ideal = rows_per_second * target_seconds

    # Dampen: move 50 % toward the ideal size
    new = current_batch_size + 0.5 * (ideal - current_batch_size)

    # Clamp
    new = int(max(s.batch_min_size, min(s.batch_max_size, new)))

    # Don't exceed remaining rows
    if max_rows_remaining > 0:
        new = min(new, max_rows_remaining)

    return max(new, s.batch_min_size)


async def get_initial_batch_size(
    source_id: str,
    connector_type: str,
    table: str,
    db: AsyncIOMotorDatabase,
) -> int:
    """Return the best starting batch size for the given source/table.

    Checks the ``batch_size_history`` collection first; falls back to
    connector-aware defaults.
    """
    doc = await db["batch_size_history"].find_one(
        {"source_id": source_id, "connector_type": connector_type, "table": table}
    )
    if doc and "optimal_batch_size" in doc:
        return int(doc["optimal_batch_size"])

    return _CONNECTOR_DEFAULTS.get(connector_type, _FALLBACK_DEFAULT)


async def save_optimal_batch_size(
    source_id: str,
    connector_type: str,
    table: str,
    batch_sizes: list[int],
    batch_times: list[float],
    target_seconds: float,
    db: AsyncIOMotorDatabase,
) -> None:
    """Persist the optimal batch size derived from a completed fetch.

    Only "stable" batches (time within 20 % of *target_seconds*) are
    considered.  If none are stable the function returns without writing.
    """
    if not batch_sizes or not batch_times or len(batch_sizes) != len(batch_times):
        return

    lower = target_seconds * 0.8
    upper = target_seconds * 1.2

    stable_sizes: list[int] = []
    stable_rps: list[float] = []

    for size, time in zip(batch_sizes, batch_times):
        if time <= 0 or size <= 0:
            continue
        if lower <= time <= upper:
            stable_sizes.append(size)
            stable_rps.append(size / time)

    if not stable_sizes:
        return

    optimal = int(statistics.median(stable_sizes))
    avg_rps = sum(stable_rps) / len(stable_rps)

    await db["batch_size_history"].update_one(
        {"source_id": source_id, "connector_type": connector_type, "table": table},
        {
            "$set": {
                "source_id": source_id,
                "connector_type": connector_type,
                "table": table,
                "optimal_batch_size": optimal,
                "avg_rows_per_second": avg_rps,
                "last_updated": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )
