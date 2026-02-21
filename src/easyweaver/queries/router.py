import asyncio
import csv
import io
import uuid

import structlog
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.dependencies import get_db, get_redis
from easyweaver.queries import service
from easyweaver.queries.schemas import (
    QueryRequest,
    QueryRunResponse,
    QueryResultsResponse,
)

logger = structlog.get_logger()
router = APIRouter()


async def _execute_inline(run_id: str, request: QueryRequest):
    """Execute query inline (no Celery) using a background asyncio task."""
    from easyweaver.dependencies import get_meta_db
    from easyweaver.queries.executor import (
        execute_single_source,
        execute_join,
        apply_sort,
    )
    from easyweaver.sources.service import get_source
    from easyweaver.results.redis_store import RedisResultStore
    from easyweaver.settings import settings
    from redis.asyncio import Redis

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    store = RedisResultStore(redis)

    db = get_meta_db()
    try:
        await service.update_query_run(db, run_id, status="running")

        if request.type == "single":
            source = await get_source(db, request.left.source_id)
            df = await execute_single_source(source, request.left)
        else:
            assert request.right is not None and request.join is not None
            left_source = await get_source(db, request.left.source_id)
            right_source = await get_source(db, request.right.source_id)
            df = await execute_join(
                left_source, right_source, request.left, request.right, request.join
            )

        # Apply sort
        if request.sort:
            df = apply_sort(df, [s.model_dump() for s in request.sort])

        # Enforce row limit
        if len(df) > settings.max_result_rows:
            df = df.head(settings.max_result_rows)

        # Store result
        await store.store_result(run_id, df)
        await service.update_query_run(db, run_id, status="completed", row_count=len(df))
        logger.info("query_completed", run_id=run_id, rows=len(df))

    except Exception as e:
        logger.exception("query_execution_failed", run_id=run_id, error=str(e))
        await service.update_query_run(db, run_id, status="failed", error=str(e))
    finally:
        await redis.aclose()


@router.post("/execute", response_model=QueryRunResponse, status_code=202)
async def execute_query(request: QueryRequest, db: AsyncIOMotorDatabase = Depends(get_db)):
    run = await service.create_query_run(db, request)
    # Execute inline as a background task (no Celery needed for dev)
    asyncio.create_task(_execute_inline(str(run.id), request))
    return run


@router.get("/runs/{run_id}", response_model=QueryRunResponse)
async def get_query_run(run_id: uuid.UUID, db: AsyncIOMotorDatabase = Depends(get_db)):
    return await service.get_query_run(db, run_id)


@router.get("/runs/{run_id}/results", response_model=QueryResultsResponse)
async def get_query_results(
    run_id: uuid.UUID,
    page: int = 1,
    page_size: int = 50,
    sort_column: str | None = None,
    sort_direction: str | None = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    from easyweaver.results.redis_store import RedisResultStore

    redis = await get_redis()
    store = RedisResultStore(redis)

    run = await service.get_query_run(db, run_id)
    if run.status != "completed":
        return QueryResultsResponse(
            columns=[], rows=[], total=0, page=page, page_size=page_size, total_pages=0
        )

    df = await store.get_result(str(run_id))
    if df is None:
        return QueryResultsResponse(
            columns=[], rows=[], total=0, page=page, page_size=page_size, total_pages=0
        )

    # Apply sorting if requested
    if sort_column and sort_column in df.columns:
        from easyweaver.queries.executor import apply_sort

        df = apply_sort(df, [{"column": sort_column, "direction": sort_direction or "asc"}])

    from easyweaver.queries.executor import paginate_dataframe

    rows, total = paginate_dataframe(df, page, page_size)
    total_pages = (total + page_size - 1) // page_size
    columns = [{"name": c, "type": str(df.schema[c])} for c in df.columns]

    return QueryResultsResponse(
        columns=columns,
        rows=rows,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.post("/runs/{run_id}/cancel")
async def cancel_query(run_id: uuid.UUID, db: AsyncIOMotorDatabase = Depends(get_db)):
    run = await service.update_query_run(db, run_id, status="cancelled")
    return {"status": "cancelled", "id": str(run.id)}


@router.get("/runs/{run_id}/export")
async def export_results(
    run_id: uuid.UUID,
    format: str = "csv",
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    from easyweaver.results.redis_store import RedisResultStore

    redis = await get_redis()
    store = RedisResultStore(redis)

    run = await service.get_query_run(db, run_id)
    df = await store.get_result(str(run_id))
    if df is None:
        from easyweaver.core.exceptions import NotFoundError

        raise NotFoundError("QueryResult", run_id)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=df.columns)
    writer.writeheader()
    writer.writerows(df.to_dicts())

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=query_{run_id}.csv"},
    )


@router.websocket("/ws/{run_id}")
async def query_progress_ws(websocket: WebSocket, run_id: str):
    await websocket.accept()
    redis = await get_redis()
    pubsub = redis.pubsub()
    channel = f"query_progress:{run_id}"
    await pubsub.subscribe(channel)
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_text(message["data"])
                if '"status":"completed"' in message["data"] or '"status":"failed"' in message["data"]:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
