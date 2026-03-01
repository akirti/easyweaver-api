import structlog
from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.dashboard import service
from easyweaver.dashboard.schemas import (
    DashboardConfigCreate,
    DashboardConfigResponse,
    DashboardConfigUpdate,
    DashboardStatsResponse,
    SnapshotHistoryResponse,
    SnapshotRecord,
)
from easyweaver.dependencies import get_db

logger = structlog.get_logger()
router = APIRouter()


def _config_to_response(config) -> dict:
    """Convert a DashboardConfig dataclass to a response dict."""
    return {
        "id": str(config.id),
        "user_id": config.user_id,
        "name": config.name,
        "source_id": config.source_id,
        "tables": config.tables,
        "refresh_interval_minutes": config.refresh_interval_minutes,
        "is_active": config.is_active,
        "created_at": config.created_at,
        "updated_at": config.updated_at,
    }


# ---------------------------------------------------------------------------
# Config CRUD
# ---------------------------------------------------------------------------


@router.get("/configs", response_model=list[DashboardConfigResponse])
async def list_configs(
    user_id: str | None = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    configs = await service.list_configs(db, user_id=user_id)
    return [_config_to_response(c) for c in configs]


@router.post("/configs", response_model=DashboardConfigResponse, status_code=201)
async def create_config(
    data: DashboardConfigCreate,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    config = await service.create_config(db, data, user_id="system")
    return _config_to_response(config)


@router.get("/configs/{config_id}", response_model=DashboardConfigResponse)
async def get_config(
    config_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    config = await service.get_config(db, config_id)
    return _config_to_response(config)


@router.put("/configs/{config_id}", response_model=DashboardConfigResponse)
async def update_config(
    config_id: str,
    data: DashboardConfigUpdate,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    config = await service.update_config(db, config_id, data)
    return _config_to_response(config)


@router.delete("/configs/{config_id}", status_code=204)
async def delete_config(
    config_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    await service.delete_config(db, config_id)


# ---------------------------------------------------------------------------
# Stats & snapshots
# ---------------------------------------------------------------------------


@router.get("/configs/{config_id}/stats", response_model=DashboardStatsResponse)
async def get_stats(
    config_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    return await service.get_current_stats(db, config_id)


@router.post("/configs/{config_id}/refresh", response_model=DashboardStatsResponse)
async def refresh_stats(
    config_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    return await service.get_current_stats(db, config_id, force_refresh=True)


@router.get("/configs/{config_id}/history", response_model=SnapshotHistoryResponse)
async def get_history(
    config_id: str,
    hours: int = 24,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    snapshots = await service.get_snapshot_history(db, config_id, hours=hours)
    records = [
        SnapshotRecord(
            id=str(s.id),
            config_id=s.config_id,
            source_id=s.source_id,
            table_name=s.table_name,
            row_count=s.row_count,
            stats=s.stats,
            captured_at=s.captured_at,
        )
        for s in snapshots
    ]
    return SnapshotHistoryResponse(snapshots=records, total=len(records))


