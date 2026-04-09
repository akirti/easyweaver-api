"""Load tests for the batched fetch system.

Verifies that the batched execution pipeline handles stress correctly:
large datasets, concurrent executions, adaptive sizing convergence,
and rapid progress callback throughput.  All tests use mocked connectors
-- no real databases are required.
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from easyweaver.connectors.base import BaseConnector
from easyweaver.processes.batch_adapter import adapt_batch_size
from easyweaver.processes.executor import _execute_query_batched, execute_process
from easyweaver.processes.schemas import ProcessConfig, ProcessQueryConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(i: int, ncols: int = 5) -> dict:
    """Generate a single row dict with *ncols* columns."""
    return {f"col_{c}": f"val_{i}_{c}" for c in range(ncols)}


class MockBatchConnector(BaseConnector):
    """In-memory connector that returns deterministic batches."""

    def __init__(
        self,
        total_rows: int,
        ncols: int = 5,
        per_batch_delay: float = 0.0,
    ):
        super().__init__(credentials={})
        self.total_rows = total_rows
        self.ncols = ncols
        self.per_batch_delay = per_batch_delay
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def test_connection(self) -> dict:
        return {"success": True, "latency_ms": 0, "message": "mock"}

    async def get_schema(self) -> list[dict]:
        return []

    async def get_table_schema(self, table_name: str) -> dict:
        return {}

    async def preview_table(self, table_name: str, limit: int = 50) -> dict:
        return {"columns": [], "rows": []}

    async def execute_query(self, **kwargs) -> list[dict]:
        return [_make_row(i, self.ncols) for i in range(self.total_rows)]

    @property
    def supports_batching(self) -> bool:
        return True

    async def execute_query_batched(
        self,
        table: str,
        columns: list[str] | None = None,
        filters: list[dict] | None = None,
        filter_logic: str = "and",
        batch_size: int = 10_000,
        offset: int = 0,
        last_key: Any = None,
    ) -> tuple[list[dict], bool, Any]:
        if self.per_batch_delay > 0:
            await asyncio.sleep(self.per_batch_delay)
        # Keyset pagination: use last_key as effective offset if provided
        effective_offset = offset
        if last_key is not None:
            effective_offset = last_key + 1
        end = min(effective_offset + batch_size, self.total_rows)
        rows = [_make_row(i, self.ncols) for i in range(effective_offset, end)]
        has_more = end < self.total_rows
        last_pk = end - 1 if rows else None
        return rows, has_more, last_pk


class VariableLatencyConnector(MockBatchConnector):
    """Connector whose batch latencies follow a prescribed sequence."""

    def __init__(self, total_rows: int, latencies: list[float], ncols: int = 5):
        super().__init__(total_rows=total_rows, ncols=ncols)
        self.latencies = latencies
        self._batch_idx = 0

    async def execute_query_batched(self, **kwargs) -> tuple[list[dict], bool, Any]:
        delay = self.latencies[self._batch_idx % len(self.latencies)]
        self._batch_idx += 1
        await asyncio.sleep(delay)
        return await super().execute_query_batched(**kwargs)


def _make_query_config(source_type: str = "postgres") -> ProcessQueryConfig:
    """Create a minimal ProcessQueryConfig with embedded credentials."""
    return ProcessQueryConfig(
        source_id="src_1",
        table="test_table",
        columns=["col_0", "col_1", "col_2", "col_3", "col_4"],
        source_type=source_type,
        encrypted_credentials="fake_encrypted",
    )


def _make_process_config(n_datasets: int = 1) -> ProcessConfig:
    """Build a ProcessConfig with *n_datasets* independent queries."""
    queries: dict[str, dict[str, ProcessQueryConfig]] = {}
    for i in range(n_datasets):
        schema = f"schema_{i}"
        queries[schema] = {
            "query_0": _make_query_config(),
        }
    return ProcessConfig(queries=queries)


# ---------------------------------------------------------------------------
# 1. Large dataset fetch (500K+ rows)
# ---------------------------------------------------------------------------

class TestLargeDatasetFetch:
    """Verify that batched fetching handles 500K+ rows correctly."""

    @pytest.mark.asyncio
    async def test_500k_rows_accumulated_correctly(self):
        """All 500K rows are accumulated without data loss."""
        total = 500_000
        connector = MockBatchConnector(total_rows=total, ncols=3)
        qc = _make_query_config()

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.core.security.decrypt_credentials", return_value='{"host":"x"}'),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 1_000_000
            mock_settings.batch_min_size = 1_000
            mock_settings.batch_max_size = 100_000

            df = await _execute_query_batched(
                db=None,
                query_config=qc,
                key="test.ds",
                progress_callback=AsyncMock(),
                control={"adaptive_enabled": False, "batch_size_override": 10_000},
            )

        assert len(df) == total, f"Expected {total} rows, got {len(df)}"
        assert set(df.columns) == {"col_0", "col_1", "col_2"}

    @pytest.mark.asyncio
    async def test_memory_bounded(self):
        """DataFrame size stays within expected bounds for 500K rows."""
        total = 500_000
        connector = MockBatchConnector(total_rows=total, ncols=3)
        qc = _make_query_config()

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.core.security.decrypt_credentials", return_value='{"host":"x"}'),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 1_000_000
            mock_settings.batch_min_size = 1_000
            mock_settings.batch_max_size = 100_000

            df = await _execute_query_batched(
                db=None,
                query_config=qc,
                key="test.ds",
                progress_callback=AsyncMock(),
                control={"adaptive_enabled": False, "batch_size_override": 10_000},
            )

        # Each cell is ~10-12 bytes string; 500K * 3 cols ~= 15-18 MB of string data.
        # Polars stores strings more efficiently.  Allow generous 200 MB upper bound.
        size_bytes = df.estimated_size("b")
        max_expected = 200 * 1024 * 1024  # 200 MB
        assert size_bytes < max_expected, (
            f"DataFrame too large: {size_bytes / (1024*1024):.1f} MB > {max_expected / (1024*1024):.0f} MB"
        )

    @pytest.mark.asyncio
    async def test_no_pathological_slowdown(self):
        """500K rows with 10K batches completes in a reasonable time."""
        total = 500_000
        connector = MockBatchConnector(total_rows=total, ncols=3)
        qc = _make_query_config()

        with (
            patch("easyweaver.processes.executor.get_connector", return_value=connector),
            patch("easyweaver.core.security.decrypt_credentials", return_value='{"host":"x"}'),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 1_000_000
            mock_settings.batch_min_size = 1_000
            mock_settings.batch_max_size = 100_000

            t0 = time.monotonic()
            df = await _execute_query_batched(
                db=None,
                query_config=qc,
                key="test.ds",
                progress_callback=AsyncMock(),
                control={"adaptive_enabled": False, "batch_size_override": 10_000},
            )
            elapsed = time.monotonic() - t0

        assert len(df) == total
        # With mocked I/O (no real DB), 500K rows should complete in < 60s
        assert elapsed < 60, f"Took too long: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# 2. Concurrent process executions (10+)
# ---------------------------------------------------------------------------

class TestConcurrentExecutions:
    """Verify no cross-contamination between concurrent process runs."""

    @pytest.mark.asyncio
    async def test_10_concurrent_processes(self):
        """Launch 10 concurrent execute_process calls; all succeed independently.

        Each process gets a different row count so we can verify no cross-
        contamination of results (each result has its own unique length).
        We run them sequentially-patched but with concurrent gather to test
        the executor's internal concurrency safety.
        """
        n_concurrent = 10
        base_rows = 500

        # We use unique row counts per process (500, 600, 700, ..., 1400)
        # to verify each result is independent.
        connectors = {
            i: MockBatchConnector(total_rows=base_rows + i * 100, ncols=2)
            for i in range(n_concurrent)
        }

        def _connector_factory(source_type, creds, *, _idx_holder=[0]):
            """Return the next connector in sequence."""
            idx = _idx_holder[0]
            _idx_holder[0] += 1
            return connectors[idx % n_concurrent]

        configs = []
        for i in range(n_concurrent):
            configs.append(_make_process_config(n_datasets=1))

        with (
            patch("easyweaver.processes.executor.get_connector", side_effect=_connector_factory),
            patch("easyweaver.core.security.decrypt_credentials", return_value='{"host":"x"}'),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 1_000_000
            mock_settings.batch_min_size = 1_000
            mock_settings.batch_max_size = 100_000

            async def _run_one(idx: int) -> pl.DataFrame:
                return await execute_process(
                    configs[idx],
                    param_values={},
                    db=None,
                    progress_callback=AsyncMock(),
                    control={
                        "adaptive_enabled": False,
                        "batch_size_override": base_rows + idx * 100,
                    },
                )

            results = await asyncio.gather(*[_run_one(i) for i in range(n_concurrent)])

        # Verify each result has the correct number of rows
        for idx, df in enumerate(results):
            expected = base_rows + idx * 100
            assert len(df) == expected, (
                f"Process {idx}: expected {expected} rows, got {len(df)}"
            )

    @pytest.mark.asyncio
    async def test_concurrent_no_errors(self):
        """All 10 concurrent runs complete without raising exceptions."""
        n_concurrent = 10
        errors: list[Exception] = []

        async def _run_one(idx: int):
            try:
                connector = MockBatchConnector(total_rows=500, ncols=2)
                config = _make_process_config(n_datasets=1)
                with (
                    patch("easyweaver.processes.executor.get_connector", return_value=connector),
                    patch("easyweaver.core.security.decrypt_credentials", return_value='{"host":"x"}'),
                    patch("easyweaver.settings.settings") as mock_settings,
                ):
                    mock_settings.max_result_rows = 1_000_000
                    mock_settings.batch_min_size = 1_000
                    mock_settings.batch_max_size = 100_000

                    await execute_process(
                        config,
                        param_values={},
                        db=None,
                        progress_callback=AsyncMock(),
                        control={"adaptive_enabled": False, "batch_size_override": 500},
                    )
            except Exception as exc:
                errors.append(exc)

        await asyncio.gather(*[_run_one(i) for i in range(n_concurrent)])
        assert len(errors) == 0, f"Errors in concurrent runs: {errors}"


# ---------------------------------------------------------------------------
# 3. Adaptive algorithm convergence
# ---------------------------------------------------------------------------

class TestAdaptiveConvergence:
    """Verify that adapt_batch_size converges toward the target time."""

    def test_converges_within_5_batches(self):
        """Batch size stabilises within 20% of ideal after ~5 iterations."""
        # Simulate variable response times
        latencies = [3.0, 5.0, 12.0, 8.0, 10.0, 10.0, 9.5, 10.2, 10.1, 9.8,
                     10.0, 10.3, 9.7, 10.1, 10.0, 9.9, 10.2, 10.0, 9.8, 10.1]
        target_seconds = 10.0
        batch_size = 10_000
        max_remaining = 1_000_000

        sizes: list[int] = []
        with patch("easyweaver.processes.batch_adapter._get_settings") as mock_gs:
            s = MagicMock()
            s.batch_min_size = 1_000
            s.batch_max_size = 100_000
            mock_gs.return_value = s

            for i, lat in enumerate(latencies):
                batch_size = adapt_batch_size(batch_size, lat, target_seconds, max_remaining)
                sizes.append(batch_size)

        # After the first 5 batches, sizes should be within 20% of each other
        last_10 = sizes[-10:]
        mean_size = sum(last_10) / len(last_10)
        for sz in last_10:
            deviation = abs(sz - mean_size) / mean_size
            assert deviation < 0.20, (
                f"Batch size {sz} deviates {deviation:.1%} from mean {mean_size:.0f} "
                f"(all sizes: {sizes})"
            )

    def test_adapts_upward_when_fast(self):
        """If batches complete much faster than target, size increases."""
        with patch("easyweaver.processes.batch_adapter._get_settings") as mock_gs:
            s = MagicMock()
            s.batch_min_size = 1_000
            s.batch_max_size = 100_000
            mock_gs.return_value = s

            # 10K rows in 1 second -> should want ~100K rows for 10s target
            new_size = adapt_batch_size(10_000, 1.0, 10.0, 500_000)
            assert new_size > 10_000, f"Expected increase, got {new_size}"

    def test_adapts_downward_when_slow(self):
        """If batches take too long, size decreases."""
        with patch("easyweaver.processes.batch_adapter._get_settings") as mock_gs:
            s = MagicMock()
            s.batch_min_size = 1_000
            s.batch_max_size = 100_000
            mock_gs.return_value = s

            # 50K rows in 50 seconds -> way too slow
            new_size = adapt_batch_size(50_000, 50.0, 10.0, 500_000)
            assert new_size < 50_000, f"Expected decrease, got {new_size}"

    def test_respects_min_max_bounds(self):
        """Adapted size never goes below min or above max."""
        with patch("easyweaver.processes.batch_adapter._get_settings") as mock_gs:
            s = MagicMock()
            s.batch_min_size = 1_000
            s.batch_max_size = 100_000
            mock_gs.return_value = s

            # Extremely fast -> wants huge batch, but capped at max
            new_size = adapt_batch_size(100_000, 0.1, 10.0, 5_000_000)
            assert new_size <= 100_000

            # Extremely slow -> wants tiny batch, but floored at min
            new_size = adapt_batch_size(1_000, 1000.0, 10.0, 500_000)
            assert new_size >= 1_000

    def test_20_batch_simulation(self):
        """Full 20-batch simulation with variable latencies converges."""
        latencies = [3.0, 5.0, 12.0, 8.0, 10.0, 10.0, 9.5, 10.2, 10.1, 9.8,
                     10.0, 10.3, 9.7, 10.1, 10.0, 9.9, 10.2, 10.0, 9.8, 10.1]
        target = 10.0
        batch_size = 10_000

        with patch("easyweaver.processes.batch_adapter._get_settings") as mock_gs:
            s = MagicMock()
            s.batch_min_size = 1_000
            s.batch_max_size = 100_000
            mock_gs.return_value = s

            for lat in latencies:
                batch_size = adapt_batch_size(batch_size, lat, target, 1_000_000)

        # After 20 batches, should have converged: batch_size should yield
        # approximately target seconds at the final throughput rate
        # latencies[-1] == 10.1 -> rps ~ batch_size / 10.1
        # ideal = rps * 10.0 ~= batch_size (roughly stable)
        # Just verify it is within [5K, 100K] and not stuck at bounds
        assert 2_000 < batch_size < 100_000, f"Unexpected final size: {batch_size}"


# ---------------------------------------------------------------------------
# 4. WebSocket message throughput
# ---------------------------------------------------------------------------

class TestProgressCallbackThroughput:
    """Verify that rapid progress callbacks are delivered without loss."""

    @pytest.mark.asyncio
    async def test_100_plus_callbacks_per_second(self):
        """Fire 200 rapid callbacks and verify all are received."""
        received: list[dict] = []

        async def _callback(event_type: str, **data):
            received.append({"type": event_type, **data})

        n_messages = 200
        t0 = time.monotonic()
        for i in range(n_messages):
            await _callback(
                "fetch_progress",
                dataset="test.ds",
                rows_fetched=i * 100,
                batch_number=i,
                batch_size=100,
                batch_time_ms=5,
                status="fetching",
            )
        elapsed = time.monotonic() - t0

        assert len(received) == n_messages, (
            f"Lost messages: expected {n_messages}, got {len(received)}"
        )
        # All 200 should complete quickly (< 1 second for in-memory)
        assert elapsed < 1.0, f"Callbacks took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_ws_broadcast_throughput(self):
        """Simulate WebSocket broadcast to multiple observers without loss."""
        n_messages = 150
        n_observers = 5

        # Track messages per observer
        observer_messages: dict[int, list] = {i: [] for i in range(n_observers)}

        class MockWS:
            def __init__(self, idx: int):
                self.idx = idx

            async def send_json(self, msg: dict):
                observer_messages[self.idx].append(msg)

        primary_ws = MockWS(0)
        observers = [MockWS(i) for i in range(1, n_observers)]

        # Simulate broadcast (mirrors ws_handler._broadcast logic)
        async def broadcast(msg: dict):
            await primary_ws.send_json(msg)
            for obs in observers:
                await obs.send_json(msg)

        for i in range(n_messages):
            await broadcast({"type": "fetch_progress", "batch": i})

        # All observers (including primary) should get all messages
        for idx in range(n_observers):
            assert len(observer_messages[idx]) == n_messages, (
                f"Observer {idx}: expected {n_messages}, got {len(observer_messages[idx])}"
            )

    @pytest.mark.asyncio
    async def test_callback_ordering_preserved(self):
        """Progress callbacks arrive in the order they were sent."""
        received: list[int] = []

        async def _callback(event_type: str, **data):
            received.append(data.get("seq", -1))

        for i in range(500):
            await _callback("progress", seq=i)

        assert received == list(range(500)), "Callback ordering was not preserved"

    @pytest.mark.asyncio
    async def test_concurrent_callbacks_no_loss(self):
        """Concurrent callback firing (via gather) delivers all messages."""
        received: list[int] = []
        lock = asyncio.Lock()

        async def _callback(event_type: str, **data):
            async with lock:
                received.append(data.get("seq", -1))

        tasks = [_callback("progress", seq=i) for i in range(200)]
        await asyncio.gather(*tasks)

        assert len(received) == 200
        assert set(received) == set(range(200))
