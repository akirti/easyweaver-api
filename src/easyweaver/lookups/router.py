from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.dependencies import get_db
from easyweaver.lookups import service
from easyweaver.processes import service as process_service

router = APIRouter()


@router.get("/{process_id}")
async def get_lookups(
    process_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """Return cached lookup values for a saved process."""
    doc = await service.get_lookups(db, process_id)
    if not doc:
        raise HTTPException(status_code=404, detail="No lookups found for this process")
    return doc


@router.post("/{process_id}/refresh")
async def refresh_lookups(
    process_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """Re-fetch distinct values from live sources and update the lookup cache."""
    config = await process_service.get_configuration(db, process_id)
    if not config.params:
        raise HTTPException(status_code=400, detail="Process has no parameters")

    lookups, references = await service.build_lookups_from_params(db, config.params)
    if not lookups:
        raise HTTPException(
            status_code=400,
            detail="No select/multi_select params with options_source found",
        )

    result = await service.upsert_lookups(db, process_id, lookups, references)
    return result
