import json

from redis.asyncio import Redis


async def publish_progress(redis: Redis, run_id: str, status: str, message: str):
    channel = f"query_progress:{run_id}"
    payload = json.dumps({"run_id": run_id, "status": status, "message": message})
    await redis.publish(channel, payload)
