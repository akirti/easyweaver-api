from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from redis.asyncio import Redis

from easyweaver.settings import settings

motor_client: AsyncIOMotorClient | None = None
_meta_db: AsyncIOMotorDatabase | None = None

redis_client: Redis | None = None


async def init_db():
    """Initialize MongoDB connection and create indexes."""
    global motor_client, _meta_db
    motor_client = AsyncIOMotorClient(settings.mongo_url)
    _meta_db = motor_client.easyweaver_meta
    # Verify connection
    await motor_client.admin.command("ping")
    # Ensure indexes
    await _meta_db.users.create_index("email", unique=True)
    await _meta_db.data_sources.create_index("created_at")
    await _meta_db.query_runs.create_index("created_at")


async def shutdown_db():
    global motor_client, _meta_db
    if motor_client:
        motor_client.close()
        motor_client = None
        _meta_db = None


async def init_redis():
    global redis_client
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    await redis_client.ping()


async def shutdown_redis():
    global redis_client
    if redis_client:
        await redis_client.aclose()
        redis_client = None


def get_meta_db() -> AsyncIOMotorDatabase:
    """Get the motor database instance directly (for background tasks)."""
    assert _meta_db is not None, "MongoDB not initialized"
    return _meta_db


async def get_db() -> AsyncIOMotorDatabase:
    """FastAPI dependency that provides the motor database."""
    assert _meta_db is not None, "MongoDB not initialized"
    return _meta_db


async def get_redis() -> Redis:
    assert redis_client is not None, "Redis not initialized"
    return redis_client
