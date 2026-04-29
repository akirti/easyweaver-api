"""Tests for easyweaver.tasks.query_tasks — execute_query_task and _execute."""
import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# _run_async helper
# ---------------------------------------------------------------------------


class TestRunAsync:
    def test_runs_coroutine_to_completion(self):
        from easyweaver.tasks.query_tasks import _run_async

        async def coro():
            return 42

        result = _run_async(coro())
        assert result == 42

    def test_closes_loop_after_completion(self):
        import asyncio
        from easyweaver.tasks.query_tasks import _run_async

        loops_seen = []

        async def coro():
            loops_seen.append(asyncio.get_event_loop())
            return "done"

        _run_async(coro())
        assert loops_seen[0].is_closed()

    def test_propagates_exceptions(self):
        from easyweaver.tasks.query_tasks import _run_async

        async def bad_coro():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            _run_async(bad_coro())

    def test_closes_loop_even_on_exception(self):
        import asyncio
        from easyweaver.tasks.query_tasks import _run_async

        loops_seen = []

        async def bad_coro():
            loops_seen.append(asyncio.get_event_loop())
            raise RuntimeError("error")

        with pytest.raises(RuntimeError):
            _run_async(bad_coro())
        assert loops_seen[0].is_closed()


# ---------------------------------------------------------------------------
# Helpers for _execute tests
# ---------------------------------------------------------------------------


def _make_single_request() -> str:
    return json.dumps({
        "type": "single",
        "left": {
            "source_id": "550e8400-e29b-41d4-a716-446655440001",
            "table": "orders",
            "columns": [],
            "filters": [],
            "aggregations": [],
            "derived_columns": [],
            "group_by": [],
        },
        "right": None,
        "join": None,
        "sort": [],
        "limit": None,
    })


def _make_execute_patches(mock_redis, mock_df=None, mock_store=None, fail_source=False, row_limit=100_000):
    """Return an ExitStack with all _execute dependencies patched.

    Since _execute uses local imports, we patch at the source module level.
    """
    if mock_df is None:
        mock_df = pl.DataFrame({"id": [1, 2]})
    if mock_store is None:
        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()

    mock_db_ctx = MagicMock()
    mock_db_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
    mock_db_ctx.__aexit__ = AsyncMock(return_value=False)

    stack = ExitStack()

    # redis.asyncio.Redis is imported locally inside _execute — patch at source
    mock_redis_module = stack.enter_context(patch("redis.asyncio.Redis"))
    mock_redis_module.from_url = MagicMock(return_value=mock_redis)

    # RedisResultStore imported locally from easyweaver.results.redis_store
    stack.enter_context(patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store))

    # async_session imported locally from easyweaver.dependencies (doesn't exist yet — use create=True)
    stack.enter_context(patch("easyweaver.dependencies.async_session", return_value=mock_db_ctx, create=True))

    # update_query_run imported locally from easyweaver.queries.service
    stack.enter_context(patch("easyweaver.queries.service.update_query_run", new_callable=AsyncMock))

    if fail_source:
        stack.enter_context(
            patch(
                "easyweaver.sources.service.get_source",
                new_callable=AsyncMock,
                side_effect=Exception("source not found"),
            )
        )
    else:
        get_src = stack.enter_context(
            patch("easyweaver.sources.service.get_source", new_callable=AsyncMock)
        )
        get_src.return_value = MagicMock()
        stack.enter_context(
            patch(
                "easyweaver.queries.executor.execute_single_source",
                new_callable=AsyncMock,
                return_value=mock_df,
            )
        )
        stack.enter_context(
            patch("easyweaver.queries.executor.apply_sort", return_value=mock_df)
        )

    mock_settings = stack.enter_context(patch("easyweaver.settings.settings"))
    mock_settings.redis_url = "redis://localhost"
    mock_settings.max_result_rows = row_limit

    return stack


# ---------------------------------------------------------------------------
# _execute — single source path
# ---------------------------------------------------------------------------


class TestExecuteSingleSource:
    @pytest.mark.anyio
    async def test_publishes_running_on_start(self):
        from easyweaver.tasks.query_tasks import _execute

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()
        mock_redis.aclose = AsyncMock()

        stack = _make_execute_patches(mock_redis)
        with stack:
            await _execute("run-1", _make_single_request())

        calls = mock_redis.publish.call_args_list
        statuses = [json.loads(c[0][1])["status"] for c in calls]
        assert "running" in statuses

    @pytest.mark.anyio
    async def test_publishes_completed_on_success(self):
        from easyweaver.tasks.query_tasks import _execute

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()
        mock_redis.aclose = AsyncMock()

        stack = _make_execute_patches(mock_redis)
        with stack:
            await _execute("run-1", _make_single_request())

        calls = mock_redis.publish.call_args_list
        statuses = [json.loads(c[0][1])["status"] for c in calls]
        assert "completed" in statuses

    @pytest.mark.anyio
    async def test_publishes_failed_on_exception(self):
        from easyweaver.tasks.query_tasks import _execute

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()
        mock_redis.aclose = AsyncMock()

        stack = _make_execute_patches(mock_redis, fail_source=True)
        with stack:
            await _execute("run-err", _make_single_request())

        calls = mock_redis.publish.call_args_list
        statuses = [json.loads(c[0][1])["status"] for c in calls]
        assert "failed" in statuses

    @pytest.mark.anyio
    async def test_redis_closed_in_finally(self):
        from easyweaver.tasks.query_tasks import _execute

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()
        mock_redis.aclose = AsyncMock()

        stack = _make_execute_patches(mock_redis)
        with stack:
            await _execute("run-1", _make_single_request())

        mock_redis.aclose.assert_called_once()

    @pytest.mark.anyio
    async def test_redis_closed_even_on_failure(self):
        from easyweaver.tasks.query_tasks import _execute

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()
        mock_redis.aclose = AsyncMock()

        stack = _make_execute_patches(mock_redis, fail_source=True)
        with stack:
            await _execute("run-err", _make_single_request())

        mock_redis.aclose.assert_called_once()

    @pytest.mark.anyio
    async def test_result_stored_in_redis(self):
        from easyweaver.tasks.query_tasks import _execute

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()
        mock_redis.aclose = AsyncMock()

        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()

        stack = _make_execute_patches(mock_redis, mock_store=mock_store)
        with stack:
            await _execute("run-store", _make_single_request())

        mock_store.store_result.assert_called_once()
        call_args = mock_store.store_result.call_args
        assert call_args[0][0] == "run-store"

    @pytest.mark.anyio
    async def test_enforces_row_limit(self):
        from easyweaver.tasks.query_tasks import _execute

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()
        mock_redis.aclose = AsyncMock()

        big_df = pl.DataFrame({"id": list(range(1000))})
        stored_dfs = []

        mock_store = MagicMock()

        async def capture_store(run_id, df):
            stored_dfs.append(df)

        mock_store.store_result = capture_store

        # Use _make_execute_patches with row_limit=10; but store is the capturing one
        stack = _make_execute_patches(mock_redis, mock_df=big_df, mock_store=mock_store, row_limit=10)
        with stack:
            await _execute("run-limit", _make_single_request())

        assert len(stored_dfs) == 1
        assert len(stored_dfs[0]) == 10

    @pytest.mark.anyio
    async def test_update_query_run_called_with_completed(self):
        """update_query_run should be called with status=completed on success."""
        from easyweaver.tasks.query_tasks import _execute

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()
        mock_redis.aclose = AsyncMock()

        mock_update = AsyncMock()
        mock_df = pl.DataFrame({"id": [1]})
        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()

        # Use the standard patch helper, but override update_query_run separately
        stack = _make_execute_patches(mock_redis, mock_df=mock_df, mock_store=mock_store)
        with stack:
            # Re-patch update_query_run at its source module to our mock
            with patch("easyweaver.queries.service.update_query_run", mock_update):
                await _execute("run-complete", _make_single_request())

        all_args = [str(call) for call in mock_update.call_args_list]
        assert any("completed" in a for a in all_args)


# ---------------------------------------------------------------------------
# execute_query_task — Celery task wrapper
# ---------------------------------------------------------------------------


class TestExecuteQueryTask:
    def test_task_is_registered(self):
        from easyweaver.tasks.celery_app import celery_app
        from easyweaver.tasks import query_tasks  # noqa: F401 — registers task
        assert "easyweaver.execute_query" in celery_app.tasks

    def test_task_calls_run_async(self):
        from easyweaver.tasks.query_tasks import execute_query_task

        with patch("easyweaver.tasks.query_tasks._run_async") as mock_run:
            mock_run.return_value = None
            execute_query_task("run-task-1", _make_single_request())
        mock_run.assert_called_once()

    def test_task_passes_execute_coroutine(self):
        """_run_async should receive a coroutine from _execute."""
        import inspect
        from easyweaver.tasks.query_tasks import execute_query_task

        coros_seen = []

        def capture_coro(coro):
            coros_seen.append(coro)

        with patch("easyweaver.tasks.query_tasks._run_async", side_effect=capture_coro):
            execute_query_task("run-task-2", _make_single_request())

        assert len(coros_seen) == 1
        assert inspect.iscoroutine(coros_seen[0])
        coros_seen[0].close()  # clean up
