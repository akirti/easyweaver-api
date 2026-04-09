import asyncio
import copy
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, WebSocket
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
    """Convert a ProcessConfiguration dataclass to a response dict.

    Strips encrypted_credentials from query configs to avoid leaking secrets.
    """
    config_data = copy.deepcopy(config.config)
    for _schema_name, schema_queries in config_data.get("queries", {}).items():
        for _query_name, qc in schema_queries.items():
            qc.pop("encrypted_credentials", None)

    return {
        "id": str(config.id),
        "user_id": config.user_id,
        "name": config.name,
        "description": config.description,
        "version": config.version,
        "config": config_data,
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
        "progress": run.progress,
        "control": run.control,
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


@router.get("/runs/{run_id}/preview/{dataset_key:path}")
async def preview_dataset(
    run_id: str,
    dataset_key: str,
    page_size: int = 100,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """Return a preview of intermediate cached data for a dataset during a run.

    Reads from the Redis key ``ew:batch:{run_id}:{dataset_key}`` which stores
    the intermediate Parquet-encoded DataFrame produced by the batched fetch.
    """
    import io

    import polars as pl
    from redis.asyncio import Redis

    from easyweaver.settings import settings

    # Verify the run exists
    await service.get_process_run(db, run_id)

    redis_key = f"ew:batch:{run_id}:{dataset_key}"

    # Use raw (non-decoded) redis to read binary parquet data
    raw_redis = Redis.from_url(settings.redis_url, decode_responses=False)
    try:
        data = await raw_redis.get(redis_key)
    finally:
        await raw_redis.aclose()

    if data is None:
        raise HTTPException(status_code=404, detail=f"No cached data for dataset '{dataset_key}'")

    df = pl.read_parquet(io.BytesIO(data))
    rows = df.head(page_size).to_dicts()
    columns = [{"name": c, "type": str(df.schema[c])} for c in df.columns]
    total = len(df)
    total_pages = (total + page_size - 1) // page_size

    return QueryResultsResponse(
        columns=columns,
        rows=rows,
        total=total,
        page=1,
        page_size=page_size,
        total_pages=total_pages,
    )


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


@router.websocket("/{config_id}/run/ws")
async def run_process_ws(
    websocket: WebSocket,
    config_id: str,
    token: str | None = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """WebSocket endpoint for real-time process execution with progress updates.

    Requires a valid JWT token passed as a ``token`` query parameter.
    Supports both EasyWeaver-native and admin-panel bridge tokens.
    """
    from easyweaver.auth.service import decode_token
    from easyweaver.core.exceptions import AuthenticationError
    from easyweaver.processes.ws_handler import ProcessWebSocketHandler

    # Validate token before accepting the connection
    if not token:
        await websocket.close(code=4003, reason="Forbidden: token required")
        return

    try:
        decode_token(token)
    except (AuthenticationError, Exception):
        await websocket.close(code=4003, reason="Forbidden: invalid token")
        return

    handler = ProcessWebSocketHandler(websocket, config_id, db)
    await handler.handle()


@router.post("/{config_id}/run", response_model=ProcessRunResponse, status_code=202)
async def run_process(
    config_id: str,
    request: ProcessRunRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    from easyweaver.settings import settings

    if request.max_rows > settings.max_result_rows:
        raise HTTPException(
            status_code=422,
            detail=f"max_rows ({request.max_rows}) exceeds system limit ({settings.max_result_rows})",
        )

    config = await service.get_configuration(db, config_id)
    run = await service.create_process_run(
        db, process_id=str(config.id), user_id="system", param_values=request.param_values
    )
    asyncio.create_task(
        _execute_process_inline(
            str(run.id),
            config_id,
            request.param_values,
            request.save_results_to_gcp,
            config_source=request.config_source,
            max_rows=request.max_rows,
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


@router.post("/{config_id}/refresh-credentials", response_model=ProcessConfigurationResponse)
async def refresh_credentials(
    config_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    config = await service.refresh_process_credentials(db, config_id)
    return _config_to_response(config)


# --- Background execution ---


async def _execute_process_inline(
    run_id: str,
    config_id: str,
    param_values: dict,
    save_to_gcp: bool,
    config_source: str = "auto",
    max_rows: int = 1000,
):
    """Execute a process in the background.

    config_source controls where to load the config from:
      - "gcp": load directly from GCS (self-sufficient, no MongoDB needed for execution)
      - "mongodb": load from MongoDB (legacy behavior)
      - "auto": prefer GCP if available, fall back to MongoDB
    """
    from easyweaver.dependencies import get_meta_db, get_query_semaphore
    from easyweaver.processes.executor import coerce_param_values, execute_process
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

            use_gcp = False
            config = await service.get_configuration(db, config_id)

            if config_source == "gcp" or (
                config_source == "auto"
                and config.gcp_path
                and config.save_destination in ("gcp", "both")
            ):
                # Try loading from GCP for self-sufficient execution
                try:
                    gcp_doc = service.load_config_from_gcp(config.gcp_path)
                    process_config = ProcessConfig.model_validate(gcp_doc.get("config", {}))
                    param_defs = gcp_doc.get("params", {})
                    use_gcp = True
                except Exception as e:
                    if config_source == "gcp":
                        raise
                    logger.warning(
                        "gcp_config_load_fallback",
                        config_id=config_id,
                        error=str(e),
                    )

            if not use_gcp:
                # MongoDB path (legacy)
                process_config = ProcessConfig.model_validate(config.config)
                param_defs = config.params

            coerced_params = coerce_param_values(param_values, param_defs)

            exec_db = None if use_gcp else db
            df = await asyncio.wait_for(
                execute_process(process_config, coerced_params, db=exec_db),
                timeout=settings.query_timeout_seconds,
            )

            # Enforce row limit (user-requested max_rows, capped by system limit)
            effective_limit = min(max_rows, settings.max_result_rows)
            if len(df) > effective_limit:
                df = df.head(effective_limit)

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
