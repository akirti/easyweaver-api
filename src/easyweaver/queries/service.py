import uuid
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.core.exceptions import NotFoundError
from easyweaver.queries.models import QueryRun
from easyweaver.queries.schemas import QueryRequest


async def create_query_run(db: AsyncIOMotorDatabase, request: QueryRequest) -> QueryRun:
    now = datetime.now(timezone.utc)
    run = QueryRun(
        id=uuid.uuid4(),
        config=request.model_dump_json(),
        status="pending",
        created_at=now,
        updated_at=now,
    )
    await db.query_runs.insert_one(run.to_doc())
    return run


async def get_query_run(db: AsyncIOMotorDatabase, run_id: uuid.UUID | str) -> QueryRun:
    rid = str(run_id)
    doc = await db.query_runs.find_one({"_id": rid})
    if not doc:
        raise NotFoundError("QueryRun", run_id)
    return QueryRun.from_doc(doc)


async def update_query_run(
    db: AsyncIOMotorDatabase,
    run_id: uuid.UUID | str,
    status: str | None = None,
    row_count: int | None = None,
    error: str | None = None,
) -> QueryRun:
    rid = str(run_id)
    updates: dict = {"updated_at": datetime.now(timezone.utc)}
    if status:
        updates["status"] = status
    if row_count is not None:
        updates["row_count"] = row_count
    if error is not None:
        updates["error"] = error
    await db.query_runs.update_one({"_id": rid}, {"$set": updates})
    return await get_query_run(db, rid)
