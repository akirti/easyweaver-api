import asyncio

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from redis.asyncio import ConnectionPool, Redis

from easyweaver.settings import settings

motor_client: AsyncIOMotorClient | None = None
_meta_db: AsyncIOMotorDatabase | None = None

redis_client: Redis | None = None
_redis_pool: ConnectionPool | None = None

# Semaphore to limit concurrent query executions
_query_semaphore: asyncio.Semaphore | None = None


def get_query_semaphore() -> asyncio.Semaphore:
    """Get the query concurrency semaphore (lazy init)."""
    global _query_semaphore
    if _query_semaphore is None:
        _query_semaphore = asyncio.Semaphore(settings.max_concurrent_queries)
    return _query_semaphore


async def init_db():
    """Initialize MongoDB connection and create indexes."""
    global motor_client, _meta_db
    motor_client = AsyncIOMotorClient(
        settings.mongo_url,
        maxPoolSize=settings.db_pool_max_size,
        minPoolSize=settings.db_pool_min_size,
    )
    _meta_db = motor_client.easyweaver_meta
    # Verify connection
    await motor_client.admin.command("ping")
    # Ensure indexes
    await _meta_db.users.create_index("email", unique=True)
    await _meta_db.data_sources.create_index("created_at")
    await _meta_db.query_runs.create_index("created_at")
    await _meta_db.process_configurations.create_index("created_at")
    await _meta_db.process_configurations.create_index("user_id")
    await _meta_db.process_runs.create_index("created_at")
    await _meta_db.process_runs.create_index("process_id")
    await _meta_db.dashboard_configs.create_index("user_id")
    await _meta_db.dashboard_configs.create_index("created_at")
    await _meta_db.data_snapshots.create_index("config_id")
    await _meta_db.data_snapshots.create_index("captured_at")
    await _meta_db.data_snapshots.create_index([("config_id", 1), ("captured_at", -1)])
    await _meta_db.configurations.create_index("type")
    await _meta_db.configurations.create_index("process_id")
    await _meta_db.batch_size_history.create_index(
        [("source_id", 1), ("connector_type", 1), ("table", 1)]
    )


def shutdown_db():
    global motor_client, _meta_db
    if motor_client:
        motor_client.close()
        motor_client = None
        _meta_db = None


async def init_redis():
    global redis_client, _redis_pool
    _redis_pool = ConnectionPool.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        decode_responses=True,
    )
    redis_client = Redis(connection_pool=_redis_pool)
    await redis_client.ping()


async def shutdown_redis():
    global redis_client, _redis_pool
    if redis_client:
        await redis_client.aclose()
        redis_client = None
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None


def get_meta_db() -> AsyncIOMotorDatabase:
    """Get the motor database instance directly (for background tasks)."""
    assert _meta_db is not None, "MongoDB not initialized"
    return _meta_db


def get_db() -> AsyncIOMotorDatabase:
    """FastAPI dependency that provides the motor database."""
    assert _meta_db is not None, "MongoDB not initialized"
    return _meta_db


def get_redis() -> Redis:
    assert redis_client is not None, "Redis not initialized"
    return redis_client
