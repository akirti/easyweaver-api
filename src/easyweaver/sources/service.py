import json
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from easyweaver.connectors.registry import get_connector
from easyweaver.core.exceptions import NotFoundError
from easyweaver.core.security import encrypt_credentials, decrypt_credentials
from easyweaver.sources.models import DataSource
from easyweaver.sources.schemas import SourceCreate, SourceUpdate


async def list_sources(db: AsyncSession) -> list[DataSource]:
    result = await db.execute(select(DataSource).order_by(DataSource.created_at.desc()))
    return list(result.scalars().all())


async def get_source(db: AsyncSession, source_id: uuid.UUID) -> DataSource:
    result = await db.execute(select(DataSource).where(DataSource.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        raise NotFoundError("DataSource", source_id)
    return source


async def create_source(db: AsyncSession, data: SourceCreate) -> DataSource:
    encrypted = encrypt_credentials(data.credentials.model_dump_json())
    source = DataSource(
        name=data.name,
        source_type=data.source_type,
        encrypted_credentials=encrypted,
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


async def update_source(db: AsyncSession, source_id: uuid.UUID, data: SourceUpdate) -> DataSource:
    source = await get_source(db, source_id)
    if data.name is not None:
        source.name = data.name
    if data.credentials is not None:
        source.encrypted_credentials = encrypt_credentials(data.credentials.model_dump_json())
    await db.commit()
    await db.refresh(source)
    return source


async def delete_source(db: AsyncSession, source_id: uuid.UUID) -> None:
    source = await get_source(db, source_id)
    await db.delete(source)
    await db.commit()


async def test_source_connection(db: AsyncSession, source_id: uuid.UUID) -> dict:
    source = await get_source(db, source_id)
    creds = json.loads(decrypt_credentials(source.encrypted_credentials))
    connector = get_connector(source.source_type, creds)
    return await connector.test_connection()


def get_source_credentials(source: DataSource) -> dict:
    return json.loads(decrypt_credentials(source.encrypted_credentials))
