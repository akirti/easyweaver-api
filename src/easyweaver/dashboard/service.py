import json
import re
import uuid
from datetime import datetime, timedelta, timezone

import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.connectors.registry import get_connector
from easyweaver.core.exceptions import NotFoundError
from easyweaver.core.security import decrypt_credentials
from easyweaver.dashboard.models import DashboardConfig, DataSnapshot
from easyweaver.dashboard.schemas import DashboardConfigCreate, DashboardConfigUpdate

logger = structlog.get_logger()

_VALID_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")


def _validate_identifier(name: str) -> bool:
    """Validate that a name is a safe SQL/collection identifier."""
    return bool(_VALID_IDENTIFIER.match(name)) and len(name) <= 128


# ---------------------------------------------------------------------------
# Config CRUD
# ---------------------------------------------------------------------------


async def create_config(
    db: AsyncIOMotorDatabase, data: DashboardConfigCreate, user_id: str = "system"
) -> DashboardConfig:
    now = datetime.now(timezone.utc)
    config = DashboardConfig(
        id=uuid.uuid4(),
        user_id=user_id,
        name=data.name,
        source_id=data.source_id,
        tables=[t.model_dump() for t in data.tables],
        refresh_interval_minutes=data.refresh_interval_minutes,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    await db.dashboard_configs.insert_one(config.to_doc())
    return config


async def get_config(db: AsyncIOMotorDatabase, config_id: str) -> DashboardConfig:
    doc = await db.dashboard_configs.find_one({"_id": config_id})
    if not doc:
        raise NotFoundError("DashboardConfig", config_id)
    return DashboardConfig.from_doc(doc)


async def list_configs(
    db: AsyncIOMotorDatabase, user_id: str | None = None
) -> list[DashboardConfig]:
    query: dict = {}
    if user_id:
        query["user_id"] = user_id
    cursor = db.dashboard_configs.find(query).sort("created_at", -1)
    return [DashboardConfig.from_doc(doc) async for doc in cursor]


async def update_config(
    db: AsyncIOMotorDatabase, config_id: str, data: DashboardConfigUpdate
) -> DashboardConfig:
    await get_config(db, config_id)  # ensure exists
    updates: dict = {"updated_at": datetime.now(timezone.utc)}
    if data.name is not None:
        updates["name"] = data.name
    if data.tables is not None:
        updates["tables"] = [t.model_dump() for t in data.tables]
    if data.refresh_interval_minutes is not None:
        updates["refresh_interval_minutes"] = data.refresh_interval_minutes
    if data.is_active is not None:
        updates["is_active"] = data.is_active
    await db.dashboard_configs.update_one({"_id": config_id}, {"$set": updates})
    return await get_config(db, config_id)


async def delete_config(db: AsyncIOMotorDatabase, config_id: str) -> None:
    result = await db.dashboard_configs.delete_one({"_id": config_id})
    if result.deleted_count == 0:
        raise NotFoundError("DashboardConfig", config_id)
    # Clean up associated snapshots
    await db.data_snapshots.delete_many({"config_id": config_id})


# ---------------------------------------------------------------------------
# Stats gathering
# ---------------------------------------------------------------------------


async def _get_source(db: AsyncIOMotorDatabase, source_id: str):
    """Fetch a data source document from MongoDB."""
    from easyweaver.sources.models import DataSource

    doc = await db.data_sources.find_one({"_id": source_id})
    if not doc:
        raise NotFoundError("DataSource", source_id)
    return DataSource.from_doc(doc)


async def _gather_pg_table_stats(
    connector, table_name: str, timestamp_column: str | None, modified_by_column: str | None
) -> dict:
    """Gather stats for a single PostgreSQL table."""
    stats: dict = {}
    pool = connector._pool
    assert pool, "PostgreSQL connector must be connected"

    if not _validate_identifier(table_name):
        logger.warning("invalid_table_name", table_name=table_name)
        return {"error": "invalid table name"}

    async with pool.acquire() as conn:
        quoted_table = f'"{table_name}"'

        # Actual row count via COUNT(*)
        actual_count = await conn.fetchval(f"SELECT COUNT(*) FROM {quoted_table}")
        stats["row_count"] = actual_count or 0

        # DML stats from pg_stat_user_tables
        row = await conn.fetchrow(
            "SELECT n_tup_ins, n_tup_upd, n_tup_del "
            "FROM pg_stat_user_tables WHERE relname = $1",
            table_name,
        )
        if row:
            stats["inserts"] = row["n_tup_ins"]
            stats["updates"] = row["n_tup_upd"]
            stats["deletes"] = row["n_tup_del"]

        # Table size
        size_val = await conn.fetchval(
            "SELECT pg_total_relation_size($1)", table_name
        )
        stats["size_bytes"] = size_val

        # Column count
        col_count = await conn.fetchval(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = $1",
            table_name,
        )
        stats["column_count"] = col_count or 0

        # Time-based change counts (if timestamp column configured)
        if timestamp_column and _validate_identifier(timestamp_column):
            quoted_col = f'"{timestamp_column}"'

            # Last modified timestamp
            try:
                last_ts = await conn.fetchval(
                    f"SELECT MAX({quoted_col}) FROM {quoted_table}"
                )
                if last_ts:
                    stats["last_modified_at"] = last_ts.isoformat()
            except Exception as e:
                logger.warning(
                    "pg_last_modified_failed",
                    table=table_name, error=str(e),
                )
            for label, hours in [("changes_1h", 1), ("changes_3h", 3), ("changes_24h", 24)]:
                try:
                    cnt = await conn.fetchval(
                        f"SELECT COUNT(*) FROM {quoted_table} "
                        f"WHERE {quoted_col} > NOW() - INTERVAL '{hours} hours'"
                    )
                    stats[label] = cnt
                except Exception as e:
                    logger.warning(
                        "pg_change_count_failed",
                        table=table_name, column=timestamp_column, hours=hours, error=str(e),
                    )

            # Last modifier
            if modified_by_column and _validate_identifier(modified_by_column):
                quoted_mod = f'"{modified_by_column}"'
                try:
                    row = await conn.fetchrow(
                        f"SELECT {quoted_mod} as modifier, COUNT(*) as cnt "
                        f"FROM {quoted_table} "
                        f"WHERE {quoted_col} > NOW() - INTERVAL '1 hour' "
                        f"GROUP BY {quoted_mod} ORDER BY cnt DESC LIMIT 1"
                    )
                    if row:
                        stats["last_modified_by"] = str(row["modifier"])
                except Exception as e:
                    logger.warning(
                        "pg_modifier_query_failed",
                        table=table_name, error=str(e),
                    )

    return stats


async def _gather_mongo_table_stats(
    connector, table_name: str, timestamp_column: str | None, modified_by_column: str | None
) -> dict:
    """Gather stats for a single MongoDB collection."""
    stats: dict = {}
    db = connector._client[connector._db_name]

    try:
        coll_stats = await db.command("collStats", table_name)
        stats["row_count"] = coll_stats.get("count", 0)
        stats["size_bytes"] = coll_stats.get("size", 0)
        stats["avg_obj_size"] = coll_stats.get("avgObjSize", 0)
    except Exception as e:
        logger.warning("mongo_collstats_failed", collection=table_name, error=str(e))
        stats["row_count"] = 0

    # Column count from sampling
    coll = db[table_name]
    sample = await coll.aggregate([{"$sample": {"size": 1}}]).to_list(1)
    if sample:
        stats["column_count"] = len(sample[0]) - 1  # exclude _id
    else:
        stats["column_count"] = 0

    # Time-based change counts
    if timestamp_column:
        now = datetime.now(timezone.utc)

        # Last modified timestamp
        try:
            pipeline = [
                {"$sort": {timestamp_column: -1}},
                {"$limit": 1},
                {"$project": {"_last_ts": f"${timestamp_column}"}},
            ]
            result = await coll.aggregate(pipeline).to_list(1)
            if result and result[0].get("_last_ts"):
                ts_val = result[0]["_last_ts"]
                if hasattr(ts_val, "isoformat"):
                    stats["last_modified_at"] = ts_val.isoformat()
                else:
                    stats["last_modified_at"] = str(ts_val)
        except Exception as e:
            logger.warning(
                "mongo_last_modified_failed",
                collection=table_name, error=str(e),
            )

        for label, hours in [("changes_1h", 1), ("changes_3h", 3), ("changes_24h", 24)]:
            try:
                cutoff = now - timedelta(hours=hours)
                pipeline = [
                    {"$match": {timestamp_column: {"$gt": cutoff}}},
                    {"$count": "total"},
                ]
                result = await coll.aggregate(pipeline).to_list(1)
                stats[label] = result[0]["total"] if result else 0
            except Exception as e:
                logger.warning(
                    "mongo_change_count_failed",
                    collection=table_name, column=timestamp_column,
                    hours=hours, error=str(e),
                )

        # Last modifier
        if modified_by_column:
            try:
                cutoff = now - timedelta(hours=1)
                pipeline = [
                    {"$match": {timestamp_column: {"$gt": cutoff}}},
                    {"$group": {"_id": f"${modified_by_column}", "cnt": {"$sum": 1}}},
                    {"$sort": {"cnt": -1}},
                    {"$limit": 1},
                ]
                result = await coll.aggregate(pipeline).to_list(1)
                if result:
                    stats["last_modified_by"] = str(result[0]["_id"])
            except Exception as e:
                logger.warning(
                    "mongo_modifier_query_failed",
                    collection=table_name, error=str(e),
                )

    return stats


async def capture_snapshot(db: AsyncIOMotorDatabase, config_id: str) -> list[DataSnapshot]:
    """Connect to the source and gather fresh stats for each monitored table."""
    config = await get_config(db, config_id)
    source = await _get_source(db, config.source_id)
    creds = json.loads(decrypt_credentials(source.encrypted_credentials))
    connector = get_connector(source.source_type, creds)

    snapshots: list[DataSnapshot] = []
    now = datetime.now(timezone.utc)

    try:
        async with connector:
            for table_cfg in config.tables:
                table_name = table_cfg["table_name"]
                ts_col = table_cfg.get("timestamp_column")
                mod_col = table_cfg.get("modified_by_column")

                if source.source_type == "postgres":
                    stats = await _gather_pg_table_stats(connector, table_name, ts_col, mod_col)
                elif source.source_type == "mongodb":
                    stats = await _gather_mongo_table_stats(
                        connector, table_name, ts_col, mod_col
                    )
                else:
                    stats = {"row_count": 0, "error": f"Unsupported source type: {source.source_type}"}

                snapshot = DataSnapshot(
                    id=uuid.uuid4(),
                    config_id=config_id,
                    source_id=config.source_id,
                    table_name=table_name,
                    row_count=stats.get("row_count", 0),
                    stats=stats,
                    captured_at=now,
                )
                await db.data_snapshots.insert_one(snapshot.to_doc())
                snapshots.append(snapshot)

    except Exception as e:
        logger.exception("capture_snapshot_failed", config_id=config_id, error=str(e))
        raise

    logger.info("snapshot_captured", config_id=config_id, tables=len(snapshots))
    return snapshots


async def get_current_stats(
    db: AsyncIOMotorDatabase, config_id: str, force_refresh: bool = False
) -> dict:
    """Return current stats, refreshing if stale or forced."""
    config = await get_config(db, config_id)
    source = await _get_source(db, config.source_id)

    stale_threshold = datetime.now(timezone.utc) - timedelta(
        minutes=config.refresh_interval_minutes
    )

    # Check latest snapshot age
    latest = await db.data_snapshots.find_one(
        {"config_id": config_id},
        sort=[("captured_at", -1)],
    )

    captured_at = latest["captured_at"] if latest else None
    if captured_at and captured_at.tzinfo is None:
        captured_at = captured_at.replace(tzinfo=timezone.utc)
    if force_refresh or not captured_at or captured_at < stale_threshold:
        try:
            snapshots = await capture_snapshot(db, config_id)
            connection_healthy = True
        except Exception:
            connection_healthy = False
            snapshots = []
    else:
        connection_healthy = True
        snapshots = None  # use existing

    # Build table stats from latest snapshots
    table_stats = []
    for table_cfg in config.tables:
        table_name = table_cfg["table_name"]
        snap_doc = await db.data_snapshots.find_one(
            {"config_id": config_id, "table_name": table_name},
            sort=[("captured_at", -1)],
        )
        if snap_doc:
            s = snap_doc.get("stats", {})
            table_stats.append({
                "table_name": table_name,
                "current_row_count": snap_doc.get("row_count", 0),
                "changes_1h": s.get("changes_1h"),
                "changes_3h": s.get("changes_3h"),
                "changes_24h": s.get("changes_24h"),
                "last_modified_at": s.get("last_modified_at"),
                "last_modified_by": s.get("last_modified_by"),
                "column_count": s.get("column_count", 0),
                "size_bytes": s.get("size_bytes"),
            })
        else:
            table_stats.append({
                "table_name": table_name,
                "current_row_count": 0,
            })

    captured_at = (
        snapshots[0].captured_at
        if snapshots
        else (latest["captured_at"] if latest else datetime.now(timezone.utc))
    )

    return {
        "config_id": config_id,
        "source_name": source.name,
        "source_type": source.source_type,
        "tables": table_stats,
        "captured_at": captured_at,
        "connection_healthy": connection_healthy,
    }


async def get_snapshot_history(
    db: AsyncIOMotorDatabase, config_id: str, hours: int = 24
) -> list[DataSnapshot]:
    """Return snapshots for the past N hours."""
    await get_config(db, config_id)  # ensure config exists
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cursor = db.data_snapshots.find(
        {"config_id": config_id, "captured_at": {"$gte": cutoff}}
    ).sort("captured_at", -1)
    return [DataSnapshot.from_doc(doc) async for doc in cursor]
