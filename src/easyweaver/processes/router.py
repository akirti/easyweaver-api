import asyncio
import uuid

import structlog
from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.dependencies import get_db, get_redis
from easyweaver.processes import service
from easyweaver.processes.schemas import (
    ProcessConfigurationCreate,
    ProcessConfigurationResponse,
    ProcessConfigurationUpdate,
    ProcessRunHistoryResponse,
    ProcessRunRequest,
    ProcessRunResponse,
)
from easyweaver.queries.schemas import QueryResultsResponse

logger = structlog.get_logger()
router = APIRouter()


def _config_to_response(config) -> dict:
    """Convert a ProcessConfiguration dataclass to a response dict."""
    return {
        "id": str(config.id),
        "user_id": config.user_id,
        "name": config.name,
        "description": config.description,
        "version": config.version,
        "config": config.config,
        "params": config.params,
        "save_destination": config.save_destination,
        "gcp_path": config.gcp_path,
        "tags": config.tags,
        "created_at": config.created_at,
        "updated_at": config.updated_at,
    }


def _run_to_response(run) -> dict:
    """Convert a ProcessRun dataclass to a response dict."""
    return {
        "id": str(run.id),
        "process_id": run.process_id,
        "status": run.status,
        "param_values": run.param_values,
        "row_count": run.row_count,
        "error": run.error,
        "result_gcp_path": run.result_gcp_path,
        "result_run_id": run.result_run_id,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


# --- Run endpoints (must be before /{config_id} to avoid path conflicts) ---


@router.get("/runs/{run_id}", response_model=ProcessRunResponse)
async def get_run(
    run_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    run = await service.get_process_run(db, run_id)
    return _run_to_response(run)


@router.get("/runs/{run_id}/results", response_model=QueryResultsResponse)
async def get_run_results(
    run_id: str,
    page: int = 1,
    page_size: int = 50,
    sort_column: str | None = None,
    sort_direction: str | None = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    from easyweaver.results.redis_store import RedisResultStore

    redis = await get_redis()
    store = RedisResultStore(redis)

    run = await service.get_process_run(db, run_id)
    if run.status != "completed" or not run.result_run_id:
        return QueryResultsResponse(
            columns=[], rows=[], total=0, page=page, page_size=page_size, total_pages=0
        )

    df = await store.get_result(run.result_run_id)
    if df is None:
        return QueryResultsResponse(
            columns=[], rows=[], total=0, page=page, page_size=page_size, total_pages=0
        )

    if sort_column and sort_column in df.columns:
        from easyweaver.queries.executor import apply_sort

        df = apply_sort(df, [{"column": sort_column, "direction": sort_direction or "asc"}])

    from easyweaver.queries.executor import paginate_dataframe

    rows, total = paginate_dataframe(df, page, page_size)
    total_pages = (total + page_size - 1) // page_size
    columns = [{"name": c, "type": str(df.schema[c])} for c in df.columns]

    return QueryResultsResponse(
        columns=columns, rows=rows, total=total, page=page,
        page_size=page_size, total_pages=total_pages,
    )


@router.post("/runs/{run_id}/save-results")
async def save_results_to_gcp(
    run_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    from easyweaver.results.redis_store import RedisResultStore

    redis = await get_redis()
    store = RedisResultStore(redis)

    run = await service.get_process_run(db, run_id)
    df = await store.get_result(run.result_run_id)
    if df is None:
        from easyweaver.core.exceptions import NotFoundError

        raise NotFoundError("ProcessRunResult", run_id)

    gcp_path = service.save_results_to_gcp(run, df)
    await service.update_process_run(db, run_id, result_gcp_path=gcp_path)
    return {"gcp_path": gcp_path}


@router.post("/runs/{run_id}/reload")
async def reload_results_from_gcp(
    run_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    from easyweaver.results.redis_store import RedisResultStore

    redis = await get_redis()
    store = RedisResultStore(redis)

    run = await service.get_process_run(db, run_id)
    if not run.result_gcp_path:
        from easyweaver.core.exceptions import NotFoundError

        raise NotFoundError("GCPResult", run_id)

    df = service.load_results_from_gcp(run.result_gcp_path)
    await store.store_result(run.result_run_id, df)
    return {"status": "ok", "row_count": len(df)}


# --- Configuration endpoints ---


@router.get("", response_model=list[ProcessConfigurationResponse])
async def list_configurations(
    user_id: str | None = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    configs = await service.list_configurations(db, user_id=user_id)
    return [_config_to_response(c) for c in configs]


@router.post("", response_model=ProcessConfigurationResponse, status_code=201)
async def create_configuration(
    data: ProcessConfigurationCreate,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    config = await service.create_configuration(db, data, user_id="system")
    return _config_to_response(config)


@router.get("/{config_id}", response_model=ProcessConfigurationResponse)
async def get_configuration(
    config_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    config = await service.get_configuration(db, config_id)
    return _config_to_response(config)


@router.put("/{config_id}", response_model=ProcessConfigurationResponse)
async def update_configuration(
    config_id: str,
    data: ProcessConfigurationUpdate,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    config = await service.update_configuration(db, config_id, data)
    return _config_to_response(config)


@router.delete("/{config_id}", status_code=204)
async def delete_configuration(
    config_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    await service.delete_configuration(db, config_id)


@router.post("/{config_id}/run", response_model=ProcessRunResponse, status_code=202)
async def run_process(
    config_id: str,
    request: ProcessRunRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    config = await service.get_configuration(db, config_id)
    run = await service.create_process_run(
        db, process_id=str(config.id), user_id="system", param_values=request.param_values
    )
    asyncio.create_task(
        _execute_process_inline(
            str(run.id), config_id, request.param_values, request.save_results_to_gcp
        )
    )
    return _run_to_response(run)


@router.get("/{config_id}/runs", response_model=ProcessRunHistoryResponse)
async def list_runs(
    config_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    runs = await service.list_process_runs(db, config_id)
    return ProcessRunHistoryResponse(
        runs=[ProcessRunResponse(**_run_to_response(r)) for r in runs],
        total=len(runs),
    )


# --- Background execution ---


async def _execute_process_inline(
    run_id: str, config_id: str, param_values: dict, save_to_gcp: bool
):
    """Execute a process in the background."""
    from easyweaver.dependencies import get_meta_db, get_query_semaphore
    from easyweaver.processes.executor import execute_process
    from easyweaver.processes.schemas import ProcessConfig
    from easyweaver.results.redis_store import RedisResultStore
    from easyweaver.settings import settings
    from redis.asyncio import Redis

    semaphore = get_query_semaphore()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    store = RedisResultStore(redis)
    db = get_meta_db()

    try:
        async with semaphore:
            await service.update_process_run(db, run_id, status="running")

            config = await service.get_configuration(db, config_id)
            process_config = ProcessConfig.model_validate(config.config)

            df = await asyncio.wait_for(
                execute_process(process_config, param_values, db),
                timeout=settings.query_timeout_seconds,
            )

            # Enforce row limit
            if len(df) > settings.max_result_rows:
                df = df.head(settings.max_result_rows)

            # Store in Redis
            result_run_id = str(uuid.uuid4())
            await store.store_result(result_run_id, df)

            update_kwargs: dict = {
                "status": "completed",
                "row_count": len(df),
                "result_run_id": result_run_id,
            }

            # Optionally save to GCP
            if save_to_gcp:
                try:
                    run = await service.get_process_run(db, run_id)
                    gcp_path = service.save_results_to_gcp(run, df)
                    update_kwargs["result_gcp_path"] = gcp_path
                except Exception as e:
                    logger.warning("gcs_save_failed", run_id=run_id, error=str(e))

            await service.update_process_run(db, run_id, **update_kwargs)
            logger.info("process_completed", run_id=run_id, rows=len(df))

    except asyncio.TimeoutError:
        msg = f"Process timed out after {settings.query_timeout_seconds}s"
        logger.warning("process_timeout", run_id=run_id)
        await service.update_process_run(db, run_id, status="failed", error=msg)
    except Exception as e:
        logger.exception("process_execution_failed", run_id=run_id, error=str(e))
        await service.update_process_run(db, run_id, status="failed", error=str(e))
    finally:
        await redis.aclose()
