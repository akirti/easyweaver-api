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
    JoinResultsRequest,
    QueryRunResponse,
    QueryResultsResponse,
)

logger = structlog.get_logger()
router = APIRouter()


async def _execute_inline(run_id: str, request: QueryRequest):
    """Execute query inline (no Celery) using a background asyncio task."""
    from easyweaver.dependencies import get_meta_db, get_query_semaphore
    from easyweaver.queries.executor import (
        execute_single_source,
        execute_join,
        apply_sort,
        resolve_cross_dataset_filters,
    )
    from easyweaver.sources.service import get_source
    from easyweaver.results.redis_store import RedisResultStore
    from easyweaver.settings import settings
    from redis.asyncio import Redis

    semaphore = get_query_semaphore()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    store = RedisResultStore(redis)

    db = get_meta_db()
    try:
        async with semaphore:
            await service.update_query_run(db, run_id, status="running")

            async def _run_query():
                # Resolve cross-dataset filter references
                from easyweaver.queries.schemas import FilterCondition as FC

                left_config = request.left
                if any(f.value_from for f in left_config.filters):
                    resolved = await resolve_cross_dataset_filters(
                        store, [f.model_dump() for f in left_config.filters]
                    )
                    left_config = left_config.model_copy(
                        update={"filters": [FC(**fd) for fd in resolved]}
                    )

                if request.type == "single":
                    source = await get_source(db, left_config.source_id)
                    return await execute_single_source(source, left_config)
                else:
                    assert request.right is not None and request.join is not None
                    right_config = request.right
                    if any(f.value_from for f in right_config.filters):
                        resolved_right = await resolve_cross_dataset_filters(
                            store, [f.model_dump() for f in right_config.filters]
                        )
                        right_config = right_config.model_copy(
                            update={"filters": [FC(**fd) for fd in resolved_right]}
                        )
                    left_source = await get_source(db, left_config.source_id)
                    right_source = await get_source(db, right_config.source_id)
                    return await execute_join(
                        left_source, right_source, left_config, right_config, request.join
                    )

            df = await asyncio.wait_for(
                _run_query(), timeout=settings.query_timeout_seconds
            )

            # Resolve data bindings
            if request.bindings:
                from easyweaver.queries.operations.binding import (
                    resolve_distinct_bindings,
                    resolve_row_pair_bindings,
                    apply_row_pair_filter,
                )
                from easyweaver.queries.operations.filter import apply_filters

                distinct_filters = await resolve_distinct_bindings(
                    store, [b.model_dump() for b in request.bindings]
                )
                if distinct_filters:
                    df = apply_filters(df, distinct_filters, "and")

                pair_df = await resolve_row_pair_bindings(
                    store, [b.model_dump() for b in request.bindings]
                )
                if pair_df is not None:
                    df = apply_row_pair_filter(df, pair_df)

            # Apply transforms (cast, strip zeros, etc.) before sort
            if request.transforms:
                from easyweaver.queries.operations.transform import apply_transforms
                df = apply_transforms(df, [t.model_dump() for t in request.transforms])

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

    except asyncio.TimeoutError:
        msg = f"Query timed out after {settings.query_timeout_seconds}s"
        logger.warning("query_timeout", run_id=run_id)
        await service.update_query_run(db, run_id, status="failed", error=msg)
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
    from easyweaver.settings import settings

    redis = await get_redis()
    store = RedisResultStore(redis)

    run = await service.get_query_run(db, run_id)
    df = await store.get_result(str(run_id))
    if df is None:
        from easyweaver.core.exceptions import NotFoundError

        raise NotFoundError("QueryResult", run_id)

    # Enforce export row limit
    if len(df) > settings.max_export_rows:
        df = df.head(settings.max_export_rows)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=df.columns)
    writer.writeheader()
    writer.writerows(df.to_dicts())

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=query_{run_id}.csv"},
    )


async def _execute_join_results_inline(run_id: str, request: JoinResultsRequest):
    """Join two existing result sets as a background task."""
    from easyweaver.dependencies import get_meta_db, get_query_semaphore
    from easyweaver.queries.executor import execute_join_from_results, apply_sort, select_columns
    from easyweaver.queries.operations.filter import apply_filters
    from easyweaver.queries.operations.transform import apply_transforms
    from easyweaver.queries.operations.derived import apply_derived_columns
    from easyweaver.queries.operations.group_by import apply_group_by
    from easyweaver.queries.operations.distinct import apply_distinct
    from easyweaver.results.redis_store import RedisResultStore
    from easyweaver.settings import settings
    from redis.asyncio import Redis

    semaphore = get_query_semaphore()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    store = RedisResultStore(redis)
    db = get_meta_db()

    try:
        async with semaphore:
            await service.update_query_run(db, run_id, status="running")

            async def _run_join():
                df = await execute_join_from_results(
                    store, str(request.left_run_id), str(request.right_run_id), request.join
                )

                if request.select_columns:
                    df = select_columns(df, request.select_columns)

                if request.derived_columns:
                    df = apply_derived_columns(
                        df, [d.model_dump() for d in request.derived_columns]
                    )

                if request.filters:
                    df = apply_filters(
                        df, [f.model_dump() for f in request.filters], request.filter_logic
                    )

                if request.bindings:
                    from easyweaver.queries.operations.binding import (
                        resolve_distinct_bindings,
                        resolve_row_pair_bindings,
                        apply_row_pair_filter,
                    )

                    distinct_filters = await resolve_distinct_bindings(
                        store, [b.model_dump() for b in request.bindings]
                    )
                    if distinct_filters:
                        df = apply_filters(df, distinct_filters, "and")

                    pair_df = await resolve_row_pair_bindings(
                        store, [b.model_dump() for b in request.bindings]
                    )
                    if pair_df is not None:
                        df = apply_row_pair_filter(df, pair_df)

                if request.group_by:
                    df = apply_group_by(df, request.group_by.model_dump())

                if request.distinct:
                    df = apply_distinct(df, request.distinct.model_dump())

                if request.sort:
                    df = apply_sort(df, [s.model_dump() for s in request.sort])

                if request.transforms:
                    df = apply_transforms(df, [t.model_dump() for t in request.transforms])

                return df

            df = await asyncio.wait_for(
                _run_join(), timeout=settings.query_timeout_seconds
            )

            if len(df) > settings.max_result_rows:
                df = df.head(settings.max_result_rows)

            await store.store_result(run_id, df)
            await service.update_query_run(db, run_id, status="completed", row_count=len(df))
            logger.info("join_results_completed", run_id=run_id, rows=len(df))

    except asyncio.TimeoutError:
        msg = f"Join timed out after {settings.query_timeout_seconds}s"
        logger.warning("join_timeout", run_id=run_id)
        await service.update_query_run(db, run_id, status="failed", error=msg)
    except Exception as e:
        logger.exception("join_results_failed", run_id=run_id, error=str(e))
        await service.update_query_run(db, run_id, status="failed", error=str(e))
    finally:
        await redis.aclose()


@router.post("/join-results", response_model=QueryRunResponse, status_code=202)
async def join_results(
    request: JoinResultsRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """Join two previously-executed result sets."""
    from easyweaver.core.exceptions import ValidationError as EWValidationError
    from easyweaver.queries.models import QueryRun as QueryRunModel
    from datetime import datetime, timezone
    import uuid as _uuid

    left_run = await service.get_query_run(db, request.left_run_id)
    right_run = await service.get_query_run(db, request.right_run_id)

    if left_run.status != "completed":
        raise EWValidationError(f"Left dataset is not completed (status: {left_run.status})")
    if right_run.status != "completed":
        raise EWValidationError(f"Right dataset is not completed (status: {right_run.status})")

    now = datetime.now(timezone.utc)
    run = QueryRunModel(
        id=_uuid.uuid4(),
        config=request.model_dump_json(),
        status="pending",
        created_at=now,
        updated_at=now,
    )
    await db.query_runs.insert_one(run.to_doc())

    asyncio.create_task(_execute_join_results_inline(str(run.id), request))
    return run


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
