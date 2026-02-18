from typing import AsyncGenerator

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from easyweaver.settings import settings

engine = create_async_engine(settings.database_url, echo=settings.debug)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

redis_client: Redis | None = None


async def init_db():
    """Verify DB connection on startup."""
    async with engine.connect() as conn:
        await conn.execute(
            __import__("sqlalchemy").text("SELECT 1")
        )


async def shutdown_db():
    await engine.dispose()


async def init_redis():
    global redis_client
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    await redis_client.ping()


async def shutdown_redis():
    global redis_client
    if redis_client:
        await redis_client.aclose()
        redis_client = None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


async def get_redis() -> Redis:
    assert redis_client is not None, "Redis not initialized"
    return redis_client
