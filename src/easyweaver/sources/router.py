import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from easyweaver.dependencies import get_db
from easyweaver.sources import service
from easyweaver.sources.schemas import (
    SourceCreate,
    SourceUpdate,
    SourceResponse,
    ConnectionTestResponse,
)

router = APIRouter()


@router.get("", response_model=list[SourceResponse])
async def list_sources(db: AsyncSession = Depends(get_db)):
    sources = await service.list_sources(db)
    return sources


@router.post("", response_model=SourceResponse, status_code=201)
async def create_source(data: SourceCreate, db: AsyncSession = Depends(get_db)):
    return await service.create_source(db, data)


@router.get("/{source_id}", response_model=SourceResponse)
async def get_source(source_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    return await service.get_source(db, source_id)


@router.put("/{source_id}", response_model=SourceResponse)
async def update_source(
    source_id: uuid.UUID, data: SourceUpdate, db: AsyncSession = Depends(get_db)
):
    return await service.update_source(db, source_id, data)


@router.delete("/{source_id}", status_code=204)
async def delete_source(source_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    await service.delete_source(db, source_id)


@router.post("/{source_id}/test", response_model=ConnectionTestResponse)
async def test_connection(source_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await service.test_source_connection(db, source_id)
    return result
