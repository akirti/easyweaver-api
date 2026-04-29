"""Tests for easyweaver.queries.router — FastAPI endpoint tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyweaver.queries.router import router
from easyweaver.queries.models import QueryRun


# ── App setup ─────────────────────────────────────────────────────────

app = FastAPI()
app.include_router(router, prefix="/queries")


# ── Helpers ───────────────────────────────────────────────────────────

_RUN_ID = "12345678-1234-1234-1234-123456789abc"
_RUN_ID_2 = "22345678-1234-1234-1234-123456789abc"
_SOURCE_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
_NOW = datetime(2026, 4, 29, 10, 0, 0, tzinfo=timezone.utc)


def _make_run(
    run_id: str = _RUN_ID,
    status: str = "pending",
    row_count: int | None = None,
    error: str | None = None,
) -> QueryRun:
    return QueryRun(
        id=uuid.UUID(run_id),
        config='{"type": "single"}',
        status=status,
        row_count=row_count,
        error=error,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_mock_db(run: QueryRun | None = None) -> MagicMock:
    db = MagicMock()
    db.query_runs = MagicMock()
    db.query_runs.insert_one = AsyncMock()
    db.query_runs.update_one = AsyncMock()
    if run is not None:
        db.query_runs.find_one = AsyncMock(return_value=run.to_doc())
    else:
        db.query_runs.find_one = AsyncMock(return_value=None)
    return db


def _single_query_payload(source_id: str = _SOURCE_ID) -> dict:
    return {
        "type": "single",
        "left": {
            "source_id": source_id,
            "table": "orders",
            "columns": None,
            "filters": [],
            "filter_logic": "and",
        },
    }


def _join_results_payload(
    left_run_id: str = _RUN_ID,
    right_run_id: str = _RUN_ID_2,
) -> dict:
    return {
        "left_run_id": left_run_id,
        "right_run_id": right_run_id,
        "join": {
            "join_type": "inner",
            "left_on": "id",
            "right_on": "id",
        },
    }


def _get_client(mock_db: MagicMock, raise_server_exceptions: bool = True) -> TestClient:
    from easyweaver.dependencies import get_db
    app.dependency_overrides[get_db] = lambda: mock_db
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


# ── POST /execute ─────────────────────────────────────────────────────


class TestExecuteEndpoint:
    def test_creates_run_returns_202(self):
        pending_run = _make_run(status="pending")
        mock_db = _make_mock_db(run=pending_run)

        async def _noop(*args, **kwargs):
            pass

        try:
            client = _get_client(mock_db)
            with (
                patch(
                    "easyweaver.queries.router.service.create_query_run",
                    new=AsyncMock(return_value=pending_run),
                ),
                patch(
                    "easyweaver.queries.router._execute_inline",
                    side_effect=_noop,
                ),
            ):
                resp = client.post("/queries/execute", json=_single_query_payload())
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 202
        data = resp.json()
        assert data["id"] == _RUN_ID
        assert data["status"] == "pending"

    def test_invalid_request_returns_422(self):
        mock_db = _make_mock_db()
        try:
            client = _get_client(mock_db)
            resp = client.post("/queries/execute", json={"type": "single"})  # missing 'left'
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 422


# ── GET /runs/{run_id} ────────────────────────────────────────────────


class TestGetRunEndpoint:
    def test_returns_run_status(self):
        run = _make_run(status="completed", row_count=5)
        mock_db = _make_mock_db(run=run)

        try:
            client = _get_client(mock_db)
            with patch(
                "easyweaver.queries.router.service.get_query_run",
                new=AsyncMock(return_value=run),
            ):
                resp = client.get(f"/queries/runs/{_RUN_ID}")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["row_count"] == 5

    def test_not_found_run_returns_error(self):
        from easyweaver.core.exceptions import NotFoundError

        mock_db = _make_mock_db()

        try:
            client = _get_client(mock_db, raise_server_exceptions=False)
            with patch(
                "easyweaver.queries.router.service.get_query_run",
                new=AsyncMock(side_effect=NotFoundError("QueryRun", _RUN_ID)),
            ):
                resp = client.get(f"/queries/runs/{_RUN_ID}", follow_redirects=True)
        finally:
            app.dependency_overrides.clear()

        # The NotFoundError is not caught by default; it propagates as 500
        assert resp.status_code != 200

    def test_returns_run_fields(self):
        run = _make_run(status="pending")
        mock_db = _make_mock_db(run=run)

        try:
            client = _get_client(mock_db)
            with patch(
                "easyweaver.queries.router.service.get_query_run",
                new=AsyncMock(return_value=run),
            ):
                resp = client.get(f"/queries/runs/{_RUN_ID}")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "status" in data
        assert "created_at" in data


# ── GET /runs/{run_id}/results ────────────────────────────────────────


class TestGetResultsEndpoint:
    def test_not_completed_returns_empty_results(self):
        run = _make_run(status="running")
        mock_db = _make_mock_db(run=run)
        mock_redis = AsyncMock()
        mock_store = MagicMock()
        mock_store.get_result = AsyncMock(return_value=None)

        try:
            client = _get_client(mock_db)
            with (
                patch(
                    "easyweaver.queries.router.service.get_query_run",
                    new=AsyncMock(return_value=run),
                ),
                patch(
                    "easyweaver.queries.router.get_redis",
                    new=AsyncMock(return_value=mock_redis),
                ),
                patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            ):
                resp = client.get(f"/queries/runs/{_RUN_ID}/results")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["rows"] == []
        assert data["total"] == 0

    def test_completed_returns_paginated_results(self):
        run = _make_run(status="completed", row_count=3)
        sample_df = pl.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        mock_db = _make_mock_db(run=run)
        mock_redis = AsyncMock()
        mock_store = MagicMock()
        mock_store.get_result = AsyncMock(return_value=sample_df)

        try:
            client = _get_client(mock_db)
            with (
                patch(
                    "easyweaver.queries.router.service.get_query_run",
                    new=AsyncMock(return_value=run),
                ),
                patch(
                    "easyweaver.queries.router.get_redis",
                    new=AsyncMock(return_value=mock_redis),
                ),
                patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            ):
                resp = client.get(f"/queries/runs/{_RUN_ID}/results?page=1&page_size=10")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["rows"]) == 3

    def test_result_not_in_store_returns_empty(self):
        run = _make_run(status="completed", row_count=0)
        mock_db = _make_mock_db(run=run)
        mock_redis = AsyncMock()
        mock_store = MagicMock()
        mock_store.get_result = AsyncMock(return_value=None)

        try:
            client = _get_client(mock_db)
            with (
                patch(
                    "easyweaver.queries.router.service.get_query_run",
                    new=AsyncMock(return_value=run),
                ),
                patch(
                    "easyweaver.queries.router.get_redis",
                    new=AsyncMock(return_value=mock_redis),
                ),
                patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            ):
                resp = client.get(f"/queries/runs/{_RUN_ID}/results")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_sort_applied_when_sort_column_specified(self):
        run = _make_run(status="completed", row_count=3)
        sample_df = pl.DataFrame({"id": [3, 1, 2], "name": ["C", "A", "B"]})
        mock_db = _make_mock_db(run=run)
        mock_redis = AsyncMock()
        mock_store = MagicMock()
        mock_store.get_result = AsyncMock(return_value=sample_df)

        try:
            client = _get_client(mock_db)
            with (
                patch(
                    "easyweaver.queries.router.service.get_query_run",
                    new=AsyncMock(return_value=run),
                ),
                patch(
                    "easyweaver.queries.router.get_redis",
                    new=AsyncMock(return_value=mock_redis),
                ),
                patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            ):
                resp = client.get(
                    f"/queries/runs/{_RUN_ID}/results?sort_column=id&sort_direction=asc"
                )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        ids = [row["id"] for row in data["rows"]]
        assert ids == [1, 2, 3]

    def test_pagination_params_forwarded(self):
        run = _make_run(status="completed", row_count=10)
        sample_df = pl.DataFrame({"id": list(range(10))})
        mock_db = _make_mock_db(run=run)
        mock_redis = AsyncMock()
        mock_store = MagicMock()
        mock_store.get_result = AsyncMock(return_value=sample_df)

        try:
            client = _get_client(mock_db)
            with (
                patch(
                    "easyweaver.queries.router.service.get_query_run",
                    new=AsyncMock(return_value=run),
                ),
                patch(
                    "easyweaver.queries.router.get_redis",
                    new=AsyncMock(return_value=mock_redis),
                ),
                patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            ):
                resp = client.get(f"/queries/runs/{_RUN_ID}/results?page=2&page_size=3")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 2
        assert data["page_size"] == 3
        assert len(data["rows"]) == 3

    def test_total_pages_calculated(self):
        run = _make_run(status="completed", row_count=10)
        sample_df = pl.DataFrame({"id": list(range(10))})
        mock_db = _make_mock_db(run=run)
        mock_redis = AsyncMock()
        mock_store = MagicMock()
        mock_store.get_result = AsyncMock(return_value=sample_df)

        try:
            client = _get_client(mock_db)
            with (
                patch(
                    "easyweaver.queries.router.service.get_query_run",
                    new=AsyncMock(return_value=run),
                ),
                patch(
                    "easyweaver.queries.router.get_redis",
                    new=AsyncMock(return_value=mock_redis),
                ),
                patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            ):
                resp = client.get(f"/queries/runs/{_RUN_ID}/results?page=1&page_size=3")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_pages"] == 4  # ceil(10/3)


# ── POST /runs/{run_id}/cancel ────────────────────────────────────────


class TestCancelEndpoint:
    def test_cancel_returns_cancelled_status(self):
        run = _make_run(status="cancelled")
        mock_db = _make_mock_db(run=run)

        try:
            client = _get_client(mock_db)
            with patch(
                "easyweaver.queries.router.service.update_query_run",
                new=AsyncMock(return_value=run),
            ):
                resp = client.post(f"/queries/runs/{_RUN_ID}/cancel")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"
        assert data["id"] == _RUN_ID


# ── GET /runs/{run_id}/export ─────────────────────────────────────────


class TestExportEndpoint:
    def test_export_csv(self):
        run = _make_run(status="completed", row_count=2)
        sample_df = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        mock_db = _make_mock_db(run=run)
        mock_redis = AsyncMock()
        mock_store = MagicMock()
        mock_store.get_result = AsyncMock(return_value=sample_df)

        try:
            client = _get_client(mock_db)
            with (
                patch(
                    "easyweaver.queries.router.service.get_query_run",
                    new=AsyncMock(return_value=run),
                ),
                patch(
                    "easyweaver.queries.router.get_redis",
                    new=AsyncMock(return_value=mock_redis),
                ),
                patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
                patch("easyweaver.settings.settings") as mock_settings,
            ):
                mock_settings.max_export_rows = 10_000
                resp = client.get(f"/queries/runs/{_RUN_ID}/export")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        content = resp.text
        assert "id" in content
        assert "Alice" in content

    def test_export_row_limit_enforced(self):
        run = _make_run(status="completed", row_count=10)
        sample_df = pl.DataFrame({"id": list(range(10))})
        mock_db = _make_mock_db(run=run)
        mock_redis = AsyncMock()
        mock_store = MagicMock()
        mock_store.get_result = AsyncMock(return_value=sample_df)

        try:
            client = _get_client(mock_db)
            with (
                patch(
                    "easyweaver.queries.router.service.get_query_run",
                    new=AsyncMock(return_value=run),
                ),
                patch(
                    "easyweaver.queries.router.get_redis",
                    new=AsyncMock(return_value=mock_redis),
                ),
                patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
                patch("easyweaver.settings.settings") as mock_settings,
            ):
                mock_settings.max_export_rows = 3
                resp = client.get(f"/queries/runs/{_RUN_ID}/export")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        lines = [l for l in resp.text.strip().split("\n") if l]
        # header + 3 data rows = 4 lines
        assert len(lines) == 4

    def test_export_not_found_raises_error(self):
        run = _make_run(status="completed")
        mock_db = _make_mock_db(run=run)
        mock_redis = AsyncMock()
        mock_store = MagicMock()
        mock_store.get_result = AsyncMock(return_value=None)

        try:
            client = _get_client(mock_db, raise_server_exceptions=False)
            with (
                patch(
                    "easyweaver.queries.router.service.get_query_run",
                    new=AsyncMock(return_value=run),
                ),
                patch(
                    "easyweaver.queries.router.get_redis",
                    new=AsyncMock(return_value=mock_redis),
                ),
                patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
                patch("easyweaver.settings.settings") as mock_settings,
            ):
                mock_settings.max_export_rows = 10_000
                resp = client.get(
                    f"/queries/runs/{_RUN_ID}/export",
                    follow_redirects=True,
                )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code != 200


# ── POST /join-results ────────────────────────────────────────────────


class TestJoinResultsEndpoint:
    def test_join_returns_202(self):
        left_run = _make_run(_RUN_ID, status="completed")
        right_run = _make_run(_RUN_ID_2, status="completed")
        mock_db = _make_mock_db()

        async def _noop(*args, **kwargs):
            pass

        def get_run_side_effect(db, run_id):
            if str(run_id) == _RUN_ID:
                return left_run
            return right_run

        try:
            client = _get_client(mock_db)
            with (
                patch(
                    "easyweaver.queries.router.service.get_query_run",
                    new=AsyncMock(side_effect=get_run_side_effect),
                ),
                patch(
                    "easyweaver.queries.router._execute_join_results_inline",
                    side_effect=_noop,
                ),
            ):
                resp = client.post("/queries/join-results", json=_join_results_payload())
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 202

    def test_join_fails_if_left_not_completed(self):
        left_run = _make_run(_RUN_ID, status="running")
        right_run = _make_run(_RUN_ID_2, status="completed")
        mock_db = _make_mock_db()

        def get_run_side_effect(db, run_id):
            if str(run_id) == _RUN_ID:
                return left_run
            return right_run

        try:
            client = _get_client(mock_db, raise_server_exceptions=False)
            with patch(
                "easyweaver.queries.router.service.get_query_run",
                new=AsyncMock(side_effect=get_run_side_effect),
            ):
                resp = client.post(
                    "/queries/join-results",
                    json=_join_results_payload(),
                    follow_redirects=True,
                )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code != 202

    def test_join_fails_if_right_not_completed(self):
        left_run = _make_run(_RUN_ID, status="completed")
        right_run = _make_run(_RUN_ID_2, status="failed")
        mock_db = _make_mock_db()

        def get_run_side_effect(db, run_id):
            if str(run_id) == _RUN_ID:
                return left_run
            return right_run

        try:
            client = _get_client(mock_db, raise_server_exceptions=False)
            with patch(
                "easyweaver.queries.router.service.get_query_run",
                new=AsyncMock(side_effect=get_run_side_effect),
            ):
                resp = client.post(
                    "/queries/join-results",
                    json=_join_results_payload(),
                    follow_redirects=True,
                )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code != 202

    def test_join_inserts_new_run_doc(self):
        left_run = _make_run(_RUN_ID, status="completed")
        right_run = _make_run(_RUN_ID_2, status="completed")
        mock_db = _make_mock_db()

        async def _noop(*args, **kwargs):
            pass

        def get_run_side_effect(db, run_id):
            if str(run_id) == _RUN_ID:
                return left_run
            return right_run

        try:
            client = _get_client(mock_db)
            with (
                patch(
                    "easyweaver.queries.router.service.get_query_run",
                    new=AsyncMock(side_effect=get_run_side_effect),
                ),
                patch(
                    "easyweaver.queries.router._execute_join_results_inline",
                    side_effect=_noop,
                ),
            ):
                resp = client.post("/queries/join-results", json=_join_results_payload())
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 202
        mock_db.query_runs.insert_one.assert_called_once()


# ── WebSocket /run/ws — auth validation ───────────────────────────────


class TestWebSocketAuth:
    def test_no_token_rejected(self):
        """Server closes WebSocket when no token is provided."""
        client = TestClient(app)
        # Use raise_server_exceptions=False so we can inspect the close
        with pytest.raises(Exception):
            with client.websocket_connect("/queries/run/ws") as ws:
                ws.receive_json()

    def test_invalid_token_rejected(self):
        """Server closes WebSocket when token is invalid."""
        from easyweaver.core.exceptions import AuthenticationError

        client = TestClient(app)
        with (
            patch(
                "easyweaver.auth.service.decode_token",
                side_effect=AuthenticationError("bad token"),
            ),
            pytest.raises(Exception),
        ):
            with client.websocket_connect("/queries/run/ws?token=bad") as ws:
                ws.receive_json()


# ── _execute_inline background task ───────────────────────────────────


class TestExecuteInline:
    """Direct tests of the _execute_inline background coroutine."""

    @pytest.mark.anyio
    async def test_single_query_happy_path(self):
        """Single-type query completes successfully and stores result."""
        import polars as pl
        from easyweaver.queries.router import _execute_inline
        from easyweaver.queries.schemas import QueryRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})

        request = QueryRequest.model_validate(
            {
                "type": "single",
                "left": {
                    "source_id": _SOURCE_ID,
                    "table": "orders",
                    "filters": [],
                    "filter_logic": "and",
                },
            }
        )

        mock_source = MagicMock()
        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_db = _make_mock_db(_make_run(run_id))
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.queries.router.service.create_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=mock_db),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_single_source", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.sources.service.get_source", new=AsyncMock(return_value=mock_source)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_inline(run_id, request)

        mock_store.store_result.assert_called_once()

    @pytest.mark.anyio
    async def test_join_query_calls_execute_join(self):
        """Join-type query calls execute_join (not execute_single_source)."""
        import polars as pl
        from easyweaver.queries.router import _execute_inline
        from easyweaver.queries.schemas import QueryRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"id": [1]})

        request = QueryRequest.model_validate(
            {
                "type": "join",
                "left": {
                    "source_id": _SOURCE_ID,
                    "table": "orders",
                    "filters": [],
                    "filter_logic": "and",
                },
                "right": {
                    "source_id": _SOURCE_ID,
                    "table": "customers",
                    "filters": [],
                    "filter_logic": "and",
                },
                "join": {"join_type": "inner", "left_on": "id", "right_on": "id"},
            }
        )

        mock_source = MagicMock()
        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_db = _make_mock_db(_make_run(run_id))
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        execute_join_mock = AsyncMock(return_value=df_result)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=mock_db),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_join", new=execute_join_mock),
            patch("easyweaver.sources.service.get_source", new=AsyncMock(return_value=mock_source)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_inline(run_id, request)

        execute_join_mock.assert_called_once()

    @pytest.mark.anyio
    async def test_timeout_marks_run_as_failed(self):
        """asyncio.TimeoutError causes run to be marked failed."""
        import asyncio as _asyncio
        from easyweaver.queries.router import _execute_inline
        from easyweaver.queries.schemas import QueryRequest

        run_id = str(uuid.uuid4())
        request = QueryRequest.model_validate(
            {
                "type": "single",
                "left": {
                    "source_id": _SOURCE_ID,
                    "table": "orders",
                    "filters": [],
                    "filter_logic": "and",
                },
            }
        )

        update_mock = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=update_mock),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("asyncio.wait_for", side_effect=_asyncio.TimeoutError()),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 1
            mock_settings.max_result_rows = 10_000

            await _execute_inline(run_id, request)

        # Last update_query_run call should be with status="failed"
        last_call = update_mock.call_args
        assert last_call[1].get("status") == "failed" or last_call[0][2] == "failed"

    @pytest.mark.anyio
    async def test_exception_marks_run_as_failed(self):
        """Generic exception causes run to be marked failed with error message."""
        from easyweaver.queries.router import _execute_inline
        from easyweaver.queries.schemas import QueryRequest

        run_id = str(uuid.uuid4())
        request = QueryRequest.model_validate(
            {
                "type": "single",
                "left": {
                    "source_id": _SOURCE_ID,
                    "table": "orders",
                    "filters": [],
                    "filter_logic": "and",
                },
            }
        )

        update_mock = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=update_mock),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.sources.service.get_source", side_effect=RuntimeError("db down")),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_inline(run_id, request)

        # Should have called update with status=failed
        calls = update_mock.call_args_list
        failed_calls = [c for c in calls if c[1].get("status") == "failed"]
        assert len(failed_calls) >= 1

    @pytest.mark.anyio
    async def test_row_limit_truncates_result(self):
        """Results exceeding max_result_rows are truncated before storing."""
        import polars as pl
        from easyweaver.queries.router import _execute_inline
        from easyweaver.queries.schemas import QueryRequest

        run_id = str(uuid.uuid4())
        # 10 rows but limit is 3
        df_result = pl.DataFrame({"id": list(range(10))})
        captured = {}

        request = QueryRequest.model_validate(
            {
                "type": "single",
                "left": {
                    "source_id": _SOURCE_ID,
                    "table": "orders",
                    "filters": [],
                    "filter_logic": "and",
                },
            }
        )

        mock_source = MagicMock()
        mock_store = MagicMock()

        async def capture_store(rid, df):
            captured["df"] = df

        mock_store.store_result = capture_store
        mock_db = _make_mock_db(_make_run(run_id))
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=mock_db),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_single_source", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.sources.service.get_source", new=AsyncMock(return_value=mock_source)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 3

            await _execute_inline(run_id, request)

        assert "df" in captured
        assert len(captured["df"]) == 3

    @pytest.mark.anyio
    async def test_transforms_applied_when_present(self):
        """Transforms are applied to the dataframe before storing."""
        import polars as pl
        from easyweaver.queries.router import _execute_inline
        from easyweaver.queries.schemas import QueryRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"name": ["alice", "bob"]})

        request = QueryRequest.model_validate(
            {
                "type": "single",
                "left": {
                    "source_id": _SOURCE_ID,
                    "table": "orders",
                    "filters": [],
                    "filter_logic": "and",
                },
                "transforms": [{"column": "name", "type": "uppercase"}],
            }
        )

        mock_source = MagicMock()
        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        apply_transforms_mock = MagicMock(return_value=df_result)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_single_source", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.sources.service.get_source", new=AsyncMock(return_value=mock_source)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.queries.operations.transform.apply_transforms", apply_transforms_mock),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_inline(run_id, request)

        apply_transforms_mock.assert_called_once()

    @pytest.mark.anyio
    async def test_sort_applied_when_present(self):
        """Sort is applied to the dataframe before storing."""
        import polars as pl
        from easyweaver.queries.router import _execute_inline
        from easyweaver.queries.schemas import QueryRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"id": [3, 1, 2]})

        request = QueryRequest.model_validate(
            {
                "type": "single",
                "left": {
                    "source_id": _SOURCE_ID,
                    "table": "orders",
                    "filters": [],
                    "filter_logic": "and",
                },
                "sort": [{"column": "id", "direction": "asc"}],
            }
        )

        mock_source = MagicMock()
        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        apply_sort_mock = MagicMock(return_value=df_result)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_single_source", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.sources.service.get_source", new=AsyncMock(return_value=mock_source)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.queries.executor.apply_sort", apply_sort_mock),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_inline(run_id, request)

        apply_sort_mock.assert_called_once()

    @pytest.mark.anyio
    async def test_group_by_applied_when_present(self):
        """group_by is applied when specified in request."""
        import polars as pl
        from easyweaver.queries.router import _execute_inline
        from easyweaver.queries.schemas import QueryRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"category": ["A", "B"], "total": [10, 20]})

        request = QueryRequest.model_validate(
            {
                "type": "single",
                "left": {
                    "source_id": _SOURCE_ID,
                    "table": "orders",
                    "filters": [],
                    "filter_logic": "and",
                },
                "group_by": {
                    "group_columns": ["category"],
                    "aggregations": [{"column": "total", "function": "sum"}],
                },
            }
        )

        mock_source = MagicMock()
        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        group_by_mock = MagicMock(return_value=df_result)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_single_source", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.sources.service.get_source", new=AsyncMock(return_value=mock_source)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.queries.operations.group_by.apply_group_by", group_by_mock),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_inline(run_id, request)

        group_by_mock.assert_called_once()

    @pytest.mark.anyio
    async def test_distinct_applied_when_present(self):
        """distinct is applied when specified in request."""
        import polars as pl
        from easyweaver.queries.router import _execute_inline
        from easyweaver.queries.schemas import QueryRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"id": [1, 2]})

        request = QueryRequest.model_validate(
            {
                "type": "single",
                "left": {
                    "source_id": _SOURCE_ID,
                    "table": "orders",
                    "filters": [],
                    "filter_logic": "and",
                },
                "distinct": {"enabled": True, "columns": ["id"]},
            }
        )

        mock_source = MagicMock()
        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        distinct_mock = MagicMock(return_value=df_result)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_single_source", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.sources.service.get_source", new=AsyncMock(return_value=mock_source)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.queries.operations.distinct.apply_distinct", distinct_mock),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_inline(run_id, request)

        distinct_mock.assert_called_once()


# ── _execute_join_results_inline background task ──────────────────────


class TestExecuteJoinResultsInline:
    """Direct tests of the _execute_join_results_inline background coroutine."""

    def _make_join_results_request(
        self,
        left_run_id: str = _RUN_ID,
        right_run_id: str = _RUN_ID_2,
    ):
        from easyweaver.queries.schemas import JoinResultsRequest

        return JoinResultsRequest.model_validate(
            {
                "left_run_id": left_run_id,
                "right_run_id": right_run_id,
                "join": {"join_type": "inner", "left_on": "id", "right_on": "id"},
            }
        )

    @pytest.mark.anyio
    async def test_happy_path_stores_result(self):
        """Successful join stores result and marks run completed."""
        import polars as pl
        from easyweaver.queries.router import _execute_join_results_inline

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"id": [1, 2]})
        request = self._make_join_results_request()

        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_store.get_result = AsyncMock(return_value=df_result)
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)
        update_mock = AsyncMock()

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=update_mock),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_join_from_results", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_join_results_inline(run_id, request)

        mock_store.store_result.assert_called_once()
        completed_calls = [
            c for c in update_mock.call_args_list if c[1].get("status") == "completed"
        ]
        assert len(completed_calls) >= 1

    @pytest.mark.anyio
    async def test_timeout_marks_run_failed(self):
        """TimeoutError causes run to be marked failed."""
        import asyncio as _asyncio
        from easyweaver.queries.router import _execute_join_results_inline

        run_id = str(uuid.uuid4())
        request = self._make_join_results_request()

        update_mock = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=update_mock),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("asyncio.wait_for", side_effect=_asyncio.TimeoutError()),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 1
            mock_settings.max_result_rows = 10_000

            await _execute_join_results_inline(run_id, request)

        failed_calls = [c for c in update_mock.call_args_list if c[1].get("status") == "failed"]
        assert len(failed_calls) >= 1

    @pytest.mark.anyio
    async def test_exception_marks_run_failed(self):
        """Generic exception causes run to be marked failed."""
        from easyweaver.queries.router import _execute_join_results_inline

        run_id = str(uuid.uuid4())
        request = self._make_join_results_request()

        update_mock = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=update_mock),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch(
                "easyweaver.queries.executor.execute_join_from_results",
                side_effect=RuntimeError("join exploded"),
            ),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_join_results_inline(run_id, request)

        failed_calls = [c for c in update_mock.call_args_list if c[1].get("status") == "failed"]
        assert len(failed_calls) >= 1

    @pytest.mark.anyio
    async def test_row_limit_enforced(self):
        """Results exceeding max_result_rows are truncated."""
        import polars as pl
        from easyweaver.queries.router import _execute_join_results_inline

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"id": list(range(20))})
        request = self._make_join_results_request()

        captured = {}
        mock_store = MagicMock()

        async def capture_store(rid, df):
            captured["df"] = df

        mock_store.store_result = capture_store
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_join_from_results", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 5

            await _execute_join_results_inline(run_id, request)

        assert "df" in captured
        assert len(captured["df"]) == 5

    @pytest.mark.anyio
    async def test_sort_applied_when_present(self):
        """Sort is applied in join results execution."""
        import polars as pl
        from easyweaver.queries.router import _execute_join_results_inline
        from easyweaver.queries.schemas import JoinResultsRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"id": [3, 1, 2]})

        request = JoinResultsRequest.model_validate(
            {
                "left_run_id": _RUN_ID,
                "right_run_id": _RUN_ID_2,
                "join": {"join_type": "inner", "left_on": "id", "right_on": "id"},
                "sort": [{"column": "id", "direction": "asc"}],
            }
        )

        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        apply_sort_mock = MagicMock(return_value=df_result)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_join_from_results", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.queries.executor.apply_sort", apply_sort_mock),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_join_results_inline(run_id, request)

        apply_sort_mock.assert_called_once()

    @pytest.mark.anyio
    async def test_transforms_applied_when_present(self):
        """Transforms are applied in join results execution."""
        import polars as pl
        from easyweaver.queries.router import _execute_join_results_inline
        from easyweaver.queries.schemas import JoinResultsRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"name": ["alice"]})

        request = JoinResultsRequest.model_validate(
            {
                "left_run_id": _RUN_ID,
                "right_run_id": _RUN_ID_2,
                "join": {"join_type": "inner", "left_on": "id", "right_on": "id"},
                "transforms": [{"column": "name", "type": "uppercase"}],
            }
        )

        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        apply_transforms_mock = MagicMock(return_value=df_result)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_join_from_results", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.queries.operations.transform.apply_transforms", apply_transforms_mock),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_join_results_inline(run_id, request)

        apply_transforms_mock.assert_called_once()

    @pytest.mark.anyio
    async def test_select_columns_applied_when_present(self):
        """select_columns is applied when specified in join results request."""
        import polars as pl
        from easyweaver.queries.router import _execute_join_results_inline
        from easyweaver.queries.schemas import JoinResultsRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"id": [1], "name": ["a"], "extra": [99]})

        request = JoinResultsRequest.model_validate(
            {
                "left_run_id": _RUN_ID,
                "right_run_id": _RUN_ID_2,
                "join": {"join_type": "inner", "left_on": "id", "right_on": "id"},
                "select_columns": ["id", "name"],
            }
        )

        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        select_mock = MagicMock(return_value=df_result.select(["id", "name"]))

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_join_from_results", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.queries.executor.select_columns", select_mock),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_join_results_inline(run_id, request)

        select_mock.assert_called_once()


class TestExecuteInlineBindings:
    """Tests for binding resolution inside _execute_inline."""

    @pytest.mark.anyio
    async def test_distinct_bindings_applied_when_filters_returned(self):
        """When resolve_distinct_bindings returns filters they are applied."""
        import polars as pl
        from easyweaver.queries.router import _execute_inline
        from easyweaver.queries.schemas import QueryRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})

        # Build a request with a data binding
        request = QueryRequest.model_validate(
            {
                "type": "single",
                "left": {
                    "source_id": _SOURCE_ID,
                    "table": "orders",
                    "filters": [],
                    "filter_logic": "and",
                },
                "bindings": [
                    {
                        "source_run_id": _RUN_ID,
                        "mode": "distinct",
                        "mappings": [{"source_column": "id", "target_column": "id"}],
                    }
                ],
            }
        )

        mock_source = MagicMock()
        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        distinct_filters = [{"column": "id", "operator": "in", "value": [1, 2]}]
        apply_filters_mock = MagicMock(return_value=df_result.head(2))

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_single_source", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.sources.service.get_source", new=AsyncMock(return_value=mock_source)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch(
                "easyweaver.queries.operations.binding.resolve_distinct_bindings",
                new=AsyncMock(return_value=distinct_filters),
            ),
            patch(
                "easyweaver.queries.operations.binding.resolve_row_pair_bindings",
                new=AsyncMock(return_value=None),
            ),
            patch("easyweaver.queries.operations.filter.apply_filters", apply_filters_mock),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_inline(run_id, request)

        apply_filters_mock.assert_called_once()

    @pytest.mark.anyio
    async def test_row_pair_bindings_applied_when_pair_df_returned(self):
        """When resolve_row_pair_bindings returns a df, apply_row_pair_filter is called."""
        import polars as pl
        from easyweaver.queries.router import _execute_inline
        from easyweaver.queries.schemas import QueryRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"id": [1, 2, 3]})
        pair_df = pl.DataFrame({"id": [1, 2]})

        request = QueryRequest.model_validate(
            {
                "type": "single",
                "left": {
                    "source_id": _SOURCE_ID,
                    "table": "orders",
                    "filters": [],
                    "filter_logic": "and",
                },
                "bindings": [
                    {
                        "source_run_id": _RUN_ID,
                        "mode": "row_pair",
                        "mappings": [{"source_column": "id", "target_column": "id"}],
                    }
                ],
            }
        )

        mock_source = MagicMock()
        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        apply_row_pair_mock = MagicMock(return_value=df_result.head(2))

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_single_source", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.sources.service.get_source", new=AsyncMock(return_value=mock_source)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch(
                "easyweaver.queries.operations.binding.resolve_distinct_bindings",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "easyweaver.queries.operations.binding.resolve_row_pair_bindings",
                new=AsyncMock(return_value=pair_df),
            ),
            patch("easyweaver.queries.operations.binding.apply_row_pair_filter", apply_row_pair_mock),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_inline(run_id, request)

        apply_row_pair_mock.assert_called_once()


class TestExecuteJoinResultsInlineBindings:
    """Tests for binding resolution inside _execute_join_results_inline."""

    @pytest.mark.anyio
    async def test_bindings_applied_in_join_results(self):
        """Bindings are resolved and applied in join results execution path."""
        import polars as pl
        from easyweaver.queries.router import _execute_join_results_inline
        from easyweaver.queries.schemas import JoinResultsRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"id": [1, 2, 3]})
        distinct_filters = [{"column": "id", "operator": "in", "value": [1, 2]}]

        request = JoinResultsRequest.model_validate(
            {
                "left_run_id": _RUN_ID,
                "right_run_id": _RUN_ID_2,
                "join": {"join_type": "inner", "left_on": "id", "right_on": "id"},
                "bindings": [
                    {
                        "source_run_id": _RUN_ID,
                        "mode": "distinct",
                        "mappings": [{"source_column": "id", "target_column": "id"}],
                    }
                ],
            }
        )

        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)
        apply_filters_mock = MagicMock(return_value=df_result.head(2))

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_join_from_results", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch(
                "easyweaver.queries.operations.binding.resolve_distinct_bindings",
                new=AsyncMock(return_value=distinct_filters),
            ),
            patch(
                "easyweaver.queries.operations.binding.resolve_row_pair_bindings",
                new=AsyncMock(return_value=None),
            ),
            patch("easyweaver.queries.operations.filter.apply_filters", apply_filters_mock),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_join_results_inline(run_id, request)

        apply_filters_mock.assert_called_once()

    @pytest.mark.anyio
    async def test_group_by_applied_in_join_results(self):
        """group_by is applied in join results execution."""
        import polars as pl
        from easyweaver.queries.router import _execute_join_results_inline
        from easyweaver.queries.schemas import JoinResultsRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"category": ["A"], "total": [10]})

        request = JoinResultsRequest.model_validate(
            {
                "left_run_id": _RUN_ID,
                "right_run_id": _RUN_ID_2,
                "join": {"join_type": "inner", "left_on": "id", "right_on": "id"},
                "group_by": {
                    "group_columns": ["category"],
                    "aggregations": [{"column": "total", "function": "sum"}],
                },
            }
        )

        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        group_by_mock = MagicMock(return_value=df_result)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_join_from_results", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.queries.operations.group_by.apply_group_by", group_by_mock),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_join_results_inline(run_id, request)

        group_by_mock.assert_called_once()

    @pytest.mark.anyio
    async def test_distinct_applied_in_join_results(self):
        """distinct is applied in join results execution."""
        import polars as pl
        from easyweaver.queries.router import _execute_join_results_inline
        from easyweaver.queries.schemas import JoinResultsRequest

        run_id = str(uuid.uuid4())
        df_result = pl.DataFrame({"id": [1, 2]})

        request = JoinResultsRequest.model_validate(
            {
                "left_run_id": _RUN_ID,
                "right_run_id": _RUN_ID_2,
                "join": {"join_type": "inner", "left_on": "id", "right_on": "id"},
                "distinct": {"enabled": True, "columns": ["id"]},
            }
        )

        mock_store = MagicMock()
        mock_store.store_result = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_semaphore = MagicMock()
        mock_semaphore.__aenter__ = AsyncMock(return_value=None)
        mock_semaphore.__aexit__ = AsyncMock(return_value=None)

        distinct_mock = MagicMock(return_value=df_result)

        with (
            patch("easyweaver.queries.router.service.update_query_run", new=AsyncMock()),
            patch("easyweaver.dependencies.get_meta_db", return_value=MagicMock()),
            patch("easyweaver.dependencies.get_query_semaphore", return_value=mock_semaphore),
            patch("easyweaver.queries.executor.execute_join_from_results", new=AsyncMock(return_value=df_result)),
            patch("easyweaver.results.redis_store.RedisResultStore", return_value=mock_store),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
            patch("easyweaver.queries.operations.distinct.apply_distinct", distinct_mock),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6380"
            mock_settings.query_timeout_seconds = 30
            mock_settings.max_result_rows = 10_000

            await _execute_join_results_inline(run_id, request)

        distinct_mock.assert_called_once()
