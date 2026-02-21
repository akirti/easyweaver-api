import json
import uuid
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.connectors.registry import get_connector
from easyweaver.core.exceptions import NotFoundError
from easyweaver.core.security import encrypt_credentials, decrypt_credentials
from easyweaver.sources.models import DataSource
from easyweaver.sources.schemas import SourceCreate, SourceUpdate


async def list_sources(db: AsyncIOMotorDatabase) -> list[DataSource]:
    cursor = db.data_sources.find().sort("created_at", -1)
    return [DataSource.from_doc(doc) async for doc in cursor]


async def get_source(db: AsyncIOMotorDatabase, source_id: uuid.UUID | str) -> DataSource:
    sid = str(source_id)
    doc = await db.data_sources.find_one({"_id": sid})
    if not doc:
        raise NotFoundError("DataSource", source_id)
    return DataSource.from_doc(doc)


async def create_source(db: AsyncIOMotorDatabase, data: SourceCreate) -> DataSource:
    encrypted = encrypt_credentials(data.credentials.model_dump_json())
    now = datetime.now(timezone.utc)
    source = DataSource(
        id=uuid.uuid4(),
        name=data.name,
        source_type=data.source_type,
        encrypted_credentials=encrypted,
        created_at=now,
        updated_at=now,
    )
    await db.data_sources.insert_one(source.to_doc())
    return source


async def update_source(
    db: AsyncIOMotorDatabase, source_id: uuid.UUID, data: SourceUpdate
) -> DataSource:
    source = await get_source(db, source_id)
    updates: dict = {"updated_at": datetime.now(timezone.utc)}
    if data.name is not None:
        updates["name"] = data.name
    if data.credentials is not None:
        updates["encrypted_credentials"] = encrypt_credentials(data.credentials.model_dump_json())
    await db.data_sources.update_one({"_id": str(source_id)}, {"$set": updates})
    return await get_source(db, source_id)


async def delete_source(db: AsyncIOMotorDatabase, source_id: uuid.UUID) -> None:
    result = await db.data_sources.delete_one({"_id": str(source_id)})
    if result.deleted_count == 0:
        raise NotFoundError("DataSource", source_id)


async def test_source_connection(db: AsyncIOMotorDatabase, source_id: uuid.UUID) -> dict:
    source = await get_source(db, source_id)
    creds = json.loads(decrypt_credentials(source.encrypted_credentials))
    connector = get_connector(source.source_type, creds)
    return await connector.test_connection()


def get_source_credentials(source: DataSource) -> dict:
    return json.loads(decrypt_credentials(source.encrypted_credentials))
