import csv
import io
import uuid

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from easyweaver.dependencies import get_db, get_redis
from easyweaver.queries import service
from easyweaver.queries.schemas import (
    QueryRequest,
    QueryRunResponse,
    QueryResultsResponse,
)
from easyweaver.tasks.query_tasks import execute_query_task

router = APIRouter()


@router.post("/execute", response_model=QueryRunResponse, status_code=202)
async def execute_query(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    run = await service.create_query_run(db, request)
    # Dispatch to Celery
    execute_query_task.delay(str(run.id), request.model_dump_json())
    return run


@router.get("/runs/{run_id}", response_model=QueryRunResponse)
async def get_query_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    return await service.get_query_run(db, run_id)


@router.get("/runs/{run_id}/results", response_model=QueryResultsResponse)
async def get_query_results(
    run_id: uuid.UUID,
    page: int = 1,
    page_size: int = 50,
    sort_column: str | None = None,
    sort_direction: str | None = None,
    db: AsyncSession = Depends(get_db),
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
async def cancel_query(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    run = await service.update_query_run(db, run_id, status="cancelled")
    return {"status": "cancelled", "id": str(run.id)}


@router.get("/runs/{run_id}/export")
async def export_results(
    run_id: uuid.UUID,
    format: str = "csv",
    db: AsyncSession = Depends(get_db),
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
