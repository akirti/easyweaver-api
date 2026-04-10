import json
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.dependencies import get_db, get_redis
from easyweaver.sources import service
from easyweaver.sources.schemas import (
    FileCredentials,
    SourceCreate,
    SourceUpdate,
    SourceResponse,
    ConnectionTestResponse,
)
from easyweaver.settings import settings

router = APIRouter()


@router.get("", response_model=list[SourceResponse])
async def list_sources(db: AsyncIOMotorDatabase = Depends(get_db)):
    sources = await service.list_sources(db)
    return sources


@router.post("", response_model=SourceResponse, status_code=201)
async def create_source(data: SourceCreate, db: AsyncIOMotorDatabase = Depends(get_db)):
    return await service.create_source(db, data)


@router.get("/{source_id}", response_model=SourceResponse)
async def get_source(source_id: uuid.UUID, db: AsyncIOMotorDatabase = Depends(get_db)):
    return await service.get_source(db, source_id)


@router.put("/{source_id}", response_model=SourceResponse)
async def update_source(
    source_id: uuid.UUID, data: SourceUpdate, db: AsyncIOMotorDatabase = Depends(get_db)
):
    return await service.update_source(db, source_id, data)


@router.delete("/{source_id}", status_code=204)
async def delete_source(source_id: uuid.UUID, db: AsyncIOMotorDatabase = Depends(get_db)):
    await service.delete_source(db, source_id)


@router.post("/{source_id}/test", response_model=ConnectionTestResponse)
async def test_connection(source_id: uuid.UUID, db: AsyncIOMotorDatabase = Depends(get_db)):
    result = await service.test_source_connection(db, source_id)
    return result


@router.get("/{source_id}/schema")
async def get_schema(source_id: uuid.UUID, db: AsyncIOMotorDatabase = Depends(get_db)):
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
    source_id: uuid.UUID, table_name: str, db: AsyncIOMotorDatabase = Depends(get_db)
):
    source = await service.get_source(db, source_id)
    creds = service.get_source_credentials(source)
    from easyweaver.connectors.registry import get_connector

    connector = get_connector(source.source_type, creds)
    async with connector:
        return await connector.get_table_schema(table_name)


@router.get("/{source_id}/preview/{table_name}")
async def preview_table(
    source_id: uuid.UUID, table_name: str, db: AsyncIOMotorDatabase = Depends(get_db)
):
    source = await service.get_source(db, source_id)
    creds = service.get_source_credentials(source)
    from easyweaver.connectors.registry import get_connector

    connector = get_connector(source.source_type, creds)
    async with connector:
        return await connector.preview_table(table_name)


@router.get("/{source_id}/tables/{table:path}/columns/{column}/distinct")
async def get_column_distinct_values(
    source_id: uuid.UUID,
    table: str,
    column: str,
    limit: int = 500,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if limit > 5000:
        limit = 5000
    source = await service.get_source(db, source_id)
    creds = service.get_source_credentials(source)
    from easyweaver.connectors.registry import get_connector

    connector = get_connector(source.source_type, creds)
    async with connector:
        result = await connector.get_distinct_values(table, column, limit)
    return result


_ALLOWED_FILE_EXTENSIONS = {".csv", ".json", ".xlsx", ".xls"}
_EXTENSION_TO_FORMAT = {".csv": "csv", ".json": "json", ".xlsx": "xlsx", ".xls": "xls"}


@router.post("/upload", response_model=SourceResponse, status_code=201)
async def upload_file_source(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    filename = file.filename or "data.csv"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_FILE_EXTENSIONS))}",
        )

    file_format = _EXTENSION_TO_FORMAT[ext]
    source_id = uuid.uuid4()
    gcp_path = f"file_uploads/{source_id}/{filename}"

    import asyncio

    from easyweaver.storage.gcs_client import get_gcs_client

    content = await file.read()
    client = get_gcs_client()
    await asyncio.to_thread(
        lambda: client._bucket.blob(gcp_path).upload_from_string(
            content, content_type=file.content_type or "application/octet-stream"
        )
    )

    file_creds = FileCredentials(
        gcp_path=gcp_path,
        file_format=file_format,
        original_filename=filename,
    )
    data = SourceCreate(
        name=name,
        source_type="file",
        credentials=file_creds,
    )
    return await service.create_source(db, data)
