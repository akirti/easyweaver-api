"""Tests for easyweaver.processes.batch_adapter."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Default settings stub used by most tests
_SETTINGS_STUB = SimpleNamespace(
    batch_min_size=1_000,
    batch_max_size=100_000,
)

# Patch target for the lazy settings helper
_SETTINGS_PATCH = "easyweaver.processes.batch_adapter._get_settings"


# ── Helper to import functions after settings is mockable ────────────
# We import at module level; the lazy _get_settings() means this is safe.
from easyweaver.processes.batch_adapter import (
    adapt_batch_size,
    get_initial_batch_size,
    save_optimal_batch_size,
)


# =====================================================================
# adapt_batch_size
# =====================================================================


class TestAdaptBatchSize:
    """Tests for the synchronous adapt_batch_size function."""

    # -- Normal adaptation & dampening --------------------------------

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_normal_adaptation_faster_than_target(self, _mock):
        """Batch finished in 5 s with 10 000 rows -> throughput 2000 r/s.
        Ideal for 10 s target = 20 000.  Dampened = 10000 + 0.5*(20000-10000) = 15000."""
        result = adapt_batch_size(10_000, 5.0, 10.0, 1_000_000)
        assert result == 15_000

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_normal_adaptation_slower_than_target(self, _mock):
        """Batch finished in 20 s with 10 000 rows -> throughput 500 r/s.
        Ideal = 5000.  Dampened = 10000 + 0.5*(5000-10000) = 7500."""
        result = adapt_batch_size(10_000, 20.0, 10.0, 1_000_000)
        assert result == 7_500

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_on_target(self, _mock):
        """Batch finished exactly on target -> size should stay the same."""
        result = adapt_batch_size(10_000, 10.0, 10.0, 1_000_000)
        assert result == 10_000

    # -- Clamping -----------------------------------------------------

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_clamp_to_max(self, _mock):
        # Very fast batch -> ideal wants to exceed max
        result = adapt_batch_size(90_000, 1.0, 10.0, 1_000_000)
        assert result <= 100_000

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_clamp_to_min(self, _mock):
        # Very slow batch -> ideal wants to go below min
        result = adapt_batch_size(1_500, 100.0, 10.0, 1_000_000)
        assert result >= 1_000

    # -- max_rows_remaining cap ---------------------------------------

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_capped_by_remaining_rows(self, _mock):
        """Result should not exceed max_rows_remaining."""
        result = adapt_batch_size(10_000, 5.0, 10.0, 3_000)
        assert result == 3_000

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_remaining_rows_zero_treated_as_no_cap(self, _mock):
        """max_rows_remaining <= 0 means no cap from that parameter."""
        result = adapt_batch_size(10_000, 5.0, 10.0, 0)
        assert result == 15_000

    # -- Edge cases ---------------------------------------------------

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_zero_batch_time(self, _mock):
        """batch_time <= 0 should return a clamped fallback, not crash."""
        result = adapt_batch_size(10_000, 0.0, 10.0, 1_000_000)
        assert 1_000 <= result <= 100_000

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_negative_batch_time(self, _mock):
        result = adapt_batch_size(10_000, -1.0, 10.0, 1_000_000)
        assert 1_000 <= result <= 100_000

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_zero_current_batch_size(self, _mock):
        result = adapt_batch_size(0, 5.0, 10.0, 1_000_000)
        assert result >= 1_000

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_negative_current_batch_size(self, _mock):
        result = adapt_batch_size(-500, 5.0, 10.0, 1_000_000)
        assert result >= 1_000

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_remaining_less_than_min_returns_min(self, _mock):
        """When remaining rows < batch_min_size, we still return min."""
        result = adapt_batch_size(10_000, 10.0, 10.0, 500)
        assert result == 1_000

    @patch(_SETTINGS_PATCH, return_value=_SETTINGS_STUB)
    def test_dampening_is_half(self, _mock):
        """Verify the 50% dampening formula exactly."""
        # throughput = 20000/10 = 2000, ideal = 2000*10 = 20000
        # dampened = 20000 + 0.5*(20000-20000) = 20000
        result = adapt_batch_size(20_000, 10.0, 10.0, 1_000_000)
        assert result == 20_000


# =====================================================================
# get_initial_batch_size
# =====================================================================


class TestGetInitialBatchSize:
    """Tests for the async get_initial_batch_size function."""

    @staticmethod
    def _make_db(find_one_result):
        collection = AsyncMock()
        collection.find_one.return_value = find_one_result
        db = MagicMock()
        db.__getitem__ = MagicMock(return_value=collection)
        return db, collection

    @pytest.mark.asyncio
    async def test_found_in_db(self):
        db, collection = self._make_db({"optimal_batch_size": 12_345})
        result = await get_initial_batch_size("src1", "postgres", "orders", db)
        assert result == 12_345
        collection.find_one.assert_awaited_once_with(
            {"source_id": "src1", "connector_type": "postgres", "table": "orders"}
        )

    @pytest.mark.asyncio
    async def test_not_found_postgres(self):
        db, _ = self._make_db(None)
        assert await get_initial_batch_size("s", "postgres", "t", db) == 15_000

    @pytest.mark.asyncio
    async def test_not_found_mysql(self):
        db, _ = self._make_db(None)
        assert await get_initial_batch_size("s", "mysql", "t", db) == 10_000

    @pytest.mark.asyncio
    async def test_not_found_db2(self):
        db, _ = self._make_db(None)
        assert await get_initial_batch_size("s", "db2", "t", db) == 5_000

    @pytest.mark.asyncio
    async def test_not_found_mongodb(self):
        db, _ = self._make_db(None)
        assert await get_initial_batch_size("s", "mongodb", "t", db) == 10_000

    @pytest.mark.asyncio
    async def test_not_found_unknown_connector(self):
        db, _ = self._make_db(None)
        assert await get_initial_batch_size("s", "oracle", "t", db) == 10_000

    @pytest.mark.asyncio
    async def test_doc_missing_optimal_field(self):
        """Document exists but lacks optimal_batch_size -> use default."""
        db, _ = self._make_db({"source_id": "s"})
        assert await get_initial_batch_size("s", "postgres", "t", db) == 15_000


# =====================================================================
# save_optimal_batch_size
# =====================================================================


class TestSaveOptimalBatchSize:
    """Tests for the async save_optimal_batch_size function."""

    @staticmethod
    def _make_db():
        collection = AsyncMock()
        db = MagicMock()
        db.__getitem__ = MagicMock(return_value=collection)
        return db, collection

    @pytest.mark.asyncio
    async def test_stable_batches_upserted(self):
        """Stable batches (within 20% of target) produce an upsert."""
        db, collection = self._make_db()
        # target = 10.  Stable range = [8, 12].
        sizes = [8_000, 10_000, 12_000]
        times = [9.0, 10.0, 11.0]  # all within [8, 12]

        await save_optimal_batch_size("s1", "postgres", "t1", sizes, times, 10.0, db)

        collection.update_one.assert_awaited_once()
        call_args = collection.update_one.call_args
        filt = call_args[0][0]
        update = call_args[0][1]
        assert filt == {"source_id": "s1", "connector_type": "postgres", "table": "t1"}
        assert update["$set"]["optimal_batch_size"] == 10_000  # median
        assert update["$set"]["avg_rows_per_second"] > 0
        assert call_args[1]["upsert"] is True

    @pytest.mark.asyncio
    async def test_no_stable_batches_no_upsert(self):
        """When no batches fall within 20% of target, nothing is written."""
        db, collection = self._make_db()
        sizes = [10_000, 10_000]
        times = [1.0, 50.0]  # way outside [8, 12]

        await save_optimal_batch_size("s1", "postgres", "t1", sizes, times, 10.0, db)
        collection.update_one.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_lists_no_upsert(self):
        db, collection = self._make_db()
        await save_optimal_batch_size("s1", "postgres", "t1", [], [], 10.0, db)
        collection.update_one.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mismatched_lists_no_upsert(self):
        db, collection = self._make_db()
        await save_optimal_batch_size("s1", "postgres", "t1", [1000], [], 10.0, db)
        collection.update_one.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_filters_out_zero_time_batches(self):
        """Batches with time <= 0 are ignored even if in range."""
        db, collection = self._make_db()
        sizes = [10_000, 10_000]
        times = [0.0, 10.0]

        await save_optimal_batch_size("s1", "postgres", "t1", sizes, times, 10.0, db)

        call_args = collection.update_one.call_args
        update = call_args[0][1]
        assert update["$set"]["optimal_batch_size"] == 10_000

    @pytest.mark.asyncio
    async def test_median_with_even_count(self):
        """Median of even-count list works (statistics.median averages middle two)."""
        db, collection = self._make_db()
        sizes = [8_000, 10_000, 11_000, 12_000]
        times = [9.0, 10.0, 10.5, 11.0]  # all stable

        await save_optimal_batch_size("s1", "postgres", "t1", sizes, times, 10.0, db)

        call_args = collection.update_one.call_args
        update = call_args[0][1]
        # median of [8000, 10000, 11000, 12000] = 10500
        assert update["$set"]["optimal_batch_size"] == 10_500

    @pytest.mark.asyncio
    async def test_last_updated_is_set(self):
        db, collection = self._make_db()
        sizes = [10_000]
        times = [10.0]

        await save_optimal_batch_size("s1", "postgres", "t1", sizes, times, 10.0, db)

        call_args = collection.update_one.call_args
        update = call_args[0][1]
        ts = update["$set"]["last_updated"]
        assert isinstance(ts, datetime)

    @pytest.mark.asyncio
    async def test_single_stable_batch(self):
        """A single stable batch should still result in an upsert."""
        db, collection = self._make_db()
        sizes = [5_000, 10_000]  # only second is stable
        times = [1.0, 10.0]

        await save_optimal_batch_size("s1", "mysql", "t1", sizes, times, 10.0, db)

        collection.update_one.assert_awaited_once()
        call_args = collection.update_one.call_args
        update = call_args[0][1]
        assert update["$set"]["optimal_batch_size"] == 10_000

    @pytest.mark.asyncio
    async def test_avg_rows_per_second_calculation(self):
        """Verify the average rows/second is computed correctly."""
        db, collection = self._make_db()
        sizes = [8_000, 12_000]
        times = [8.0, 12.0]  # both stable, rps = 1000 each

        await save_optimal_batch_size("s1", "postgres", "t1", sizes, times, 10.0, db)

        call_args = collection.update_one.call_args
        update = call_args[0][1]
        assert update["$set"]["avg_rows_per_second"] == pytest.approx(1000.0)
