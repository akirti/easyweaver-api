import io

import polars as pl
from redis.asyncio import Redis

from easyweaver.results.store import ResultStore
from easyweaver.settings import settings


class RedisResultStore(ResultStore):
    def __init__(self, redis: Redis):
        self._redis = redis

    def _key(self, run_id: str) -> str:
        return f"query_result:{run_id}"

    async def store_result(self, run_id: str, df: pl.DataFrame, ttl: int | None = None) -> None:
        ttl = ttl or settings.result_ttl_seconds
        buf = io.BytesIO()
        df.write_parquet(buf, compression="zstd", compression_level=3)
        # Use a separate non-decoded connection for binary data
        raw_redis = Redis.from_url(settings.redis_url, decode_responses=False)
        try:
            await raw_redis.setex(self._key(run_id), ttl, buf.getvalue())
        finally:
            await raw_redis.aclose()

    async def get_result(self, run_id: str) -> pl.DataFrame | None:
        raw_redis = Redis.from_url(settings.redis_url, decode_responses=False)
        try:
            data = await raw_redis.get(self._key(run_id))
        finally:
            await raw_redis.aclose()
        if data is None:
            return None
        return pl.read_parquet(io.BytesIO(data))

    async def delete_result(self, run_id: str) -> None:
        await self._redis.delete(self._key(run_id))
