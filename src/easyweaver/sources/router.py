import json
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from easyweaver.dependencies import get_db, get_redis
from easyweaver.sources import service
from easyweaver.sources.schemas import (
    SourceCreate,
    SourceUpdate,
    SourceResponse,
    ConnectionTestResponse,
)
from easyweaver.settings import settings

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


@router.get("/{source_id}/schema")
async def get_schema(source_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    redis = await get_redis()
    cache_key = f"schema:{source_id}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    source = await service.get_source(db, source_id)
    creds = service.get_source_credentials(source)
    from easyweaver.connectors.registry import get_connector

    connector = get_connector(source.source_type, creds)
    async with connector:
        schema = await connector.get_schema()

    await redis.setex(cache_key, settings.schema_cache_ttl_seconds, json.dumps(schema))
    return schema


@router.get("/{source_id}/schema/{table_name}")
async def get_table_schema(
    source_id: uuid.UUID, table_name: str, db: AsyncSession = Depends(get_db)
):
    source = await service.get_source(db, source_id)
    creds = service.get_source_credentials(source)
    from easyweaver.connectors.registry import get_connector

    connector = get_connector(source.source_type, creds)
    async with connector:
        return await connector.get_table_schema(table_name)


@router.get("/{source_id}/preview/{table_name}")
async def preview_table(
    source_id: uuid.UUID, table_name: str, db: AsyncSession = Depends(get_db)
):
    source = await service.get_source(db, source_id)
    creds = service.get_source_credentials(source)
    from easyweaver.connectors.registry import get_connector

    connector = get_connector(source.source_type, creds)
    async with connector:
        return await connector.preview_table(table_name)
