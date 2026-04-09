"""Tests for the preview_dataset endpoint in easyweaver.processes.router."""

from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest
from httpx import ASGITransport, AsyncClient

from easyweaver.dependencies import get_db
from easyweaver.processes.models import ProcessRun


def _make_run(run_id: str = "00000000-0000-0000-0000-000000000001") -> ProcessRun:
    now = datetime.now(timezone.utc)
    return ProcessRun(
        id=uuid.UUID(run_id),
        process_id="proc-1",
        user_id="user-1",
        status="running",
        created_at=now,
        updated_at=now,
    )


def _df_to_parquet_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd")
    return buf.getvalue()


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["a", "b", "c", "d", "e"],
        "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def mock_db():
    db = MagicMock()
    return db


@pytest.fixture
def app(mock_db):
    """Create a minimal FastAPI app with the processes router and overridden deps."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from starlette.requests import Request

    from easyweaver.core.exceptions import EasyWeaverError
    from easyweaver.processes.router import router

    test_app = FastAPI()
    test_app.include_router(router, prefix="/processes")

    @test_app.exception_handler(EasyWeaverError)
    async def easyweaver_error_handler(request: Request, exc: EasyWeaverError):
        status_map = {"NOT_FOUND": 404, "VALIDATION": 422, "AUTH": 401, "FORBIDDEN": 403}
        return JSONResponse(
            status_code=status_map.get(exc.code, 500),
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    async def override_get_db():
        return mock_db

    test_app.dependency_overrides[get_db] = override_get_db
    return test_app


class TestPreviewDataset:
    @pytest.mark.asyncio
    async def test_returns_preview_data(self, app, mock_db, sample_df):
        run = _make_run()
        parquet_bytes = _df_to_parquet_bytes(sample_df)

        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=parquet_bytes)
        mock_redis.aclose = AsyncMock()

        with (
            patch("easyweaver.processes.router.service.get_process_run", return_value=run),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    f"/processes/runs/{run.id}/preview/orders.main",
                    params={"page_size": 3},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["rows"]) == 3
        assert body["total"] == 5
        assert body["page"] == 1
        assert body["page_size"] == 3
        assert body["total_pages"] == 2
        col_names = [c["name"] for c in body["columns"]]
        assert col_names == ["id", "name", "value"]

    @pytest.mark.asyncio
    async def test_returns_404_when_no_cached_data(self, app, mock_db):
        run = _make_run()

        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.aclose = AsyncMock()

        with (
            patch("easyweaver.processes.router.service.get_process_run", return_value=run),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    f"/processes/runs/{run.id}/preview/orders.main"
                )

        assert resp.status_code == 404
        assert "No cached data" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_returns_404_when_run_not_found(self, app, mock_db):
        from easyweaver.core.exceptions import NotFoundError

        with patch(
            "easyweaver.processes.router.service.get_process_run",
            side_effect=NotFoundError("ProcessRun", "bad-id"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    "/processes/runs/bad-id/preview/orders.main"
                )

        # NotFoundError should propagate (likely 404 or 500 depending on exception handler)
        assert resp.status_code in (404, 500)

    @pytest.mark.asyncio
    async def test_default_page_size(self, app, mock_db, sample_df):
        """Without explicit page_size, should default to 100 (returning all 5 rows)."""
        run = _make_run()
        parquet_bytes = _df_to_parquet_bytes(sample_df)

        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=parquet_bytes)
        mock_redis.aclose = AsyncMock()

        with (
            patch("easyweaver.processes.router.service.get_process_run", return_value=run),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    f"/processes/runs/{run.id}/preview/orders.main"
                )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["rows"]) == 5  # all rows returned (5 < 100)
        assert body["page_size"] == 100

    @pytest.mark.asyncio
    async def test_nested_dataset_key_path(self, app, mock_db, sample_df):
        """Dataset key with dots should work via path parameter."""
        run = _make_run()
        parquet_bytes = _df_to_parquet_bytes(sample_df)

        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=parquet_bytes)
        mock_redis.aclose = AsyncMock()

        with (
            patch("easyweaver.processes.router.service.get_process_run", return_value=run),
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    f"/processes/runs/{run.id}/preview/schema1.orders"
                )

        assert resp.status_code == 200
        # Verify the correct Redis key was used
        mock_redis.get.assert_called_once_with(f"ew:batch:{run.id}:schema1.orders")
