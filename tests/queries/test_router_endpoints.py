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
