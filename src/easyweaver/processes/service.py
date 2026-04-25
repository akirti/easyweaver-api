import copy
import uuid
from datetime import datetime, timezone

import polars as pl
import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.core.exceptions import NotFoundError
from easyweaver.processes.models import ProcessConfiguration, ProcessRun
from easyweaver.processes.schemas import ProcessConfigurationCreate, ProcessConfigurationUpdate

logger = structlog.get_logger()


# --- ProcessConfiguration CRUD ---


async def list_configurations(
    db: AsyncIOMotorDatabase, user_id: str | None = None
) -> list[ProcessConfiguration]:
    query = {"user_id": user_id} if user_id else {}
    cursor = db.process_configurations.find(query).sort("created_at", -1)
    return [ProcessConfiguration.from_doc(doc) async for doc in cursor]


async def get_configuration(
    db: AsyncIOMotorDatabase, config_id: str | uuid.UUID
) -> ProcessConfiguration:
    cid = str(config_id)
    doc = await db.process_configurations.find_one({"_id": cid})
    if not doc:
        raise NotFoundError("ProcessConfiguration", config_id)
    return ProcessConfiguration.from_doc(doc)


async def embed_source_details(db: AsyncIOMotorDatabase, config_dict: dict) -> dict:
    """Enrich config with source_name, source_type, encrypted_credentials from DataSources."""
    from easyweaver.sources.service import get_source

    config_dict = copy.deepcopy(config_dict)

    # Collect unique source_ids
    source_ids: set[str] = set()
    queries = config_dict.get("queries", {})
    for _schema_name, schema_queries in queries.items():
        for _query_name, qc in schema_queries.items():
            sid = qc.get("source_id")
            if sid:
                source_ids.add(sid)

    # Batch-fetch DataSource docs
    sources: dict = {}
    for sid in source_ids:
        try:
            sources[sid] = await get_source(db, sid)
        except NotFoundError:
            logger.warning("embed_source_details_source_not_found", source_id=sid)

    # Embed details into each query config
    for _schema_name, schema_queries in queries.items():
        for _query_name, qc in schema_queries.items():
            sid = qc.get("source_id")
            source = sources.get(sid) if sid else None
            if source:
                qc["source_name"] = source.name
                qc["source_type"] = source.source_type
                qc["encrypted_credentials"] = source.encrypted_credentials

    config_dict["config_version"] = 2
    return config_dict


async def create_configuration(
    db: AsyncIOMotorDatabase, data: ProcessConfigurationCreate, user_id: str
) -> ProcessConfiguration:
    now = datetime.now(timezone.utc)
    config_dict = await embed_source_details(db, data.config.model_dump())
    config = ProcessConfiguration(
        id=uuid.uuid4(),
        user_id=user_id,
        name=data.name,
        description=data.description,
        version=1,
        config=config_dict,
        params={k: v.model_dump() for k, v in data.params.items()},
        save_destination=data.save_destination,
        gcp_path=data.gcp_path,
        tags=data.tags,
        created_at=now,
        updated_at=now,
    )
    await db.process_configurations.insert_one(config.to_doc())
    logger.info("process_configuration_created", id=str(config.id), name=config.name)

    # Build and cache lookup data for select/multi_select params
    if config.params:
        try:
            from easyweaver.lookups.service import build_lookups_from_params, upsert_lookups

            lookups, references = await build_lookups_from_params(db, config.params)
            if lookups:
                await upsert_lookups(db, str(config.id), lookups, references)
        except Exception as e:
            logger.warning("lookup_build_on_save_failed", id=str(config.id), error=str(e))

    # Save config JSON to GCP if requested
    if data.save_destination in ("gcp", "both"):
        try:
            gcp_path = f"process_configurations/{config.id}/v{config.version}/config.json"
            save_config_to_gcp(config)
            config.gcp_path = gcp_path
            await db.process_configurations.update_one(
                {"_id": str(config.id)}, {"$set": {"gcp_path": gcp_path}}
            )
        except Exception as e:
            logger.warning("gcs_config_save_failed", id=str(config.id), error=str(e))

    return config


async def update_configuration(
    db: AsyncIOMotorDatabase,
    config_id: str | uuid.UUID,
    data: ProcessConfigurationUpdate,
) -> ProcessConfiguration:
    cid = str(config_id)
    await get_configuration(db, cid)  # ensure exists
    updates: dict = {"updated_at": datetime.now(timezone.utc)}
    if data.name is not None:
        updates["name"] = data.name
    if data.description is not None:
        updates["description"] = data.description
    if data.params is not None:
        updates["params"] = {k: v.model_dump() for k, v in data.params.items()}
    if data.config is not None:
        updates["config"] = await embed_source_details(db, data.config.model_dump())
    if data.save_destination is not None:
        updates["save_destination"] = data.save_destination
    if data.gcp_path is not None:
        updates["gcp_path"] = data.gcp_path
    if data.tags is not None:
        updates["tags"] = data.tags

    # Increment version
    await db.process_configurations.update_one(
        {"_id": cid}, {"$set": updates, "$inc": {"version": 1}}
    )
    updated = await get_configuration(db, cid)

    # Rebuild lookup cache if params were updated
    if data.params is not None and updated.params:
        try:
            from easyweaver.lookups.service import build_lookups_from_params, upsert_lookups

            lookups, references = await build_lookups_from_params(db, updated.params)
            if lookups:
                await upsert_lookups(db, cid, lookups, references)
        except Exception as e:
            logger.warning("lookup_rebuild_on_update_failed", id=cid, error=str(e))

    # Save updated config to GCP if destination requires it
    dest = data.save_destination or updated.save_destination
    if dest in ("gcp", "both"):
        try:
            save_config_to_gcp(updated)
            gcp_path = f"process_configurations/{updated.id}/v{updated.version}/config.json"
            if updated.gcp_path != gcp_path:
                await db.process_configurations.update_one(
                    {"_id": cid}, {"$set": {"gcp_path": gcp_path}}
                )
                updated.gcp_path = gcp_path
        except Exception as e:
            logger.warning("gcs_config_save_failed", id=cid, error=str(e))

    return updated


async def delete_configuration(db: AsyncIOMotorDatabase, config_id: str | uuid.UUID) -> None:
    cid = str(config_id)
    result = await db.process_configurations.delete_one({"_id": cid})
    if result.deleted_count == 0:
        raise NotFoundError("ProcessConfiguration", config_id)
    # Clean up lookup cache
    try:
        from easyweaver.lookups.service import delete_lookups

        await delete_lookups(db, cid)
    except Exception:
        pass
    logger.info("process_configuration_deleted", id=cid)


# --- ProcessRun CRUD ---


async def create_process_run(
    db: AsyncIOMotorDatabase, process_id: str, user_id: str, param_values: dict
) -> ProcessRun:
    now = datetime.now(timezone.utc)
    run = ProcessRun(
        id=uuid.uuid4(),
        process_id=process_id,
        user_id=user_id,
        param_values=param_values,
        status="pending",
        created_at=now,
        updated_at=now,
    )
    await db.process_runs.insert_one(run.to_doc())
    return run


async def get_process_run(db: AsyncIOMotorDatabase, run_id: str | uuid.UUID) -> ProcessRun:
    rid = str(run_id)
    doc = await db.process_runs.find_one({"_id": rid})
    if not doc:
        raise NotFoundError("ProcessRun", run_id)
    return ProcessRun.from_doc(doc)


async def update_process_run(
    db: AsyncIOMotorDatabase,
    run_id: str | uuid.UUID,
    status: str | None = None,
    row_count: int | None = None,
    error: str | None = None,
    result_gcp_path: str | None = None,
    result_run_id: str | None = None,
    control: dict | None = None,
) -> ProcessRun:
    rid = str(run_id)
    updates: dict = {"updated_at": datetime.now(timezone.utc)}
    if status is not None:
        updates["status"] = status
    if row_count is not None:
        updates["row_count"] = row_count
    if error is not None:
        updates["error"] = error
    if result_gcp_path is not None:
        updates["result_gcp_path"] = result_gcp_path
    if result_run_id is not None:
        updates["result_run_id"] = result_run_id
    if control is not None:
        updates["control"] = control
    await db.process_runs.update_one({"_id": rid}, {"$set": updates})
    return await get_process_run(db, rid)


async def list_process_runs(
    db: AsyncIOMotorDatabase, process_id: str
) -> list[ProcessRun]:
    cursor = db.process_runs.find({"process_id": process_id}).sort("created_at", -1)
    return [ProcessRun.from_doc(doc) async for doc in cursor]


async def refresh_process_credentials(
    db: AsyncIOMotorDatabase, config_id: str | uuid.UUID
) -> ProcessConfiguration:
    """Re-fetch current creds from sources, re-embed, increment version, re-save to GCP."""
    config = await get_configuration(db, config_id)
    cid = str(config_id)

    enriched = await embed_source_details(db, config.config)
    now = datetime.now(timezone.utc)
    await db.process_configurations.update_one(
        {"_id": cid},
        {"$set": {"config": enriched, "updated_at": now}, "$inc": {"version": 1}},
    )
    updated = await get_configuration(db, cid)

    # Re-save to GCP if destination requires it
    if updated.save_destination in ("gcp", "both"):
        try:
            save_config_to_gcp(updated)
            gcp_path = f"process_configurations/{updated.id}/v{updated.version}/config.json"
            if updated.gcp_path != gcp_path:
                await db.process_configurations.update_one(
                    {"_id": cid}, {"$set": {"gcp_path": gcp_path}}
                )
                updated.gcp_path = gcp_path
        except Exception as e:
            logger.warning("gcs_config_save_failed", id=cid, error=str(e))

    logger.info("process_credentials_refreshed", id=cid, version=updated.version)
    return updated


def load_config_from_gcp(gcp_path: str) -> dict:
    """Download process configuration JSON from GCS."""
    from easyweaver.storage.gcs_client import get_gcs_client

    client = get_gcs_client()
    return client.download_json(gcp_path)


# --- GCS helpers ---


def save_config_to_gcp(config: ProcessConfiguration) -> None:
    """Save process configuration JSON to GCS."""
    from easyweaver.storage.gcs_client import get_gcs_client

    client = get_gcs_client()
    path = f"process_configurations/{config.id}/v{config.version}/config.json"
    client.upload_json(path, config.to_doc())
    logger.info("process_config_saved_to_gcp", id=str(config.id), path=path)


def save_results_to_gcp(run: ProcessRun, df: pl.DataFrame) -> str:
    """Save process run results as Parquet to GCS. Returns the GCP path."""
    from easyweaver.storage.gcs_client import get_gcs_client

    client = get_gcs_client()
    path = f"process_runs/{run.process_id}/{run.id}/results.parquet"
    client.upload_parquet(path, df)
    logger.info("process_results_saved_to_gcp", run_id=str(run.id), path=path, rows=len(df))
    return path


def load_results_from_gcp(gcp_path: str) -> pl.DataFrame:
    """Download process run results from GCS."""
    from easyweaver.storage.gcs_client import get_gcs_client

    client = get_gcs_client()
    return client.download_parquet(gcp_path)
