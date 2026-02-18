import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from easyweaver.core.exceptions import NotFoundError
from easyweaver.queries.models import QueryRun
from easyweaver.queries.schemas import QueryRequest


async def create_query_run(db: AsyncSession, request: QueryRequest) -> QueryRun:
    run = QueryRun(
        config=request.model_dump_json(),
        status="pending",
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


async def get_query_run(db: AsyncSession, run_id: uuid.UUID) -> QueryRun:
    result = await db.execute(select(QueryRun).where(QueryRun.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise NotFoundError("QueryRun", run_id)
    return run


async def update_query_run(
    db: AsyncSession,
    run_id: uuid.UUID,
    status: str | None = None,
    row_count: int | None = None,
    error: str | None = None,
) -> QueryRun:
    values: dict = {"updated_at": datetime.utcnow()}
    if status:
        values["status"] = status
    if row_count is not None:
        values["row_count"] = row_count
    if error is not None:
        values["error"] = error
    await db.execute(update(QueryRun).where(QueryRun.id == run_id).values(**values))
    await db.commit()
    return await get_query_run(db, run_id)
