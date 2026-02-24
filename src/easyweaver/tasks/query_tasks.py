import asyncio

import structlog

from easyweaver.tasks.celery_app import celery_app
from easyweaver.tasks.progress import publish_progress

logger = structlog.get_logger()


def _run_async(coro):
    """Run async code from Celery sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="easyweaver.execute_query")
def execute_query_task(self, run_id: str, config_json: str):
    """Execute a query (single or join) and store results in Redis."""
    _run_async(_execute(run_id, config_json))


async def _execute(run_id: str, config_json: str):
    from easyweaver.dependencies import async_session
    from easyweaver.queries.schemas import QueryRequest
    from easyweaver.queries.service import update_query_run
    from easyweaver.queries.executor import (
        execute_single_source,
        execute_join,
        apply_sort,
    )
    from easyweaver.sources.service import get_source
    from easyweaver.results.redis_store import RedisResultStore
    from redis.asyncio import Redis
    from easyweaver.settings import settings

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    store = RedisResultStore(redis)

    async with async_session() as db:
        try:
            await update_query_run(db, run_id, status="running")
            await publish_progress(redis, run_id, "running", "Starting query execution")

            request = QueryRequest.model_validate_json(config_json)

            if request.type == "single":
                source = await get_source(db, request.left.source_id)
                df = await execute_single_source(source, request.left)
            else:
                assert request.right is not None and request.join is not None
                left_source = await get_source(db, request.left.source_id)
                right_source = await get_source(db, request.right.source_id)
                await publish_progress(redis, run_id, "running", "Fetching data from sources")
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
            await update_query_run(db, run_id, status="completed", row_count=len(df))
            await publish_progress(redis, run_id, "completed", f"Done. {len(df)} rows.")

        except Exception as e:
            logger.exception("query_execution_failed", run_id=run_id, error=str(e))
            await update_query_run(db, run_id, status="failed", error=str(e))
            await publish_progress(redis, run_id, "failed", str(e))
        finally:
            await redis.aclose()
