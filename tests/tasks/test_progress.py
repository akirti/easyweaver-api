"""Tests for easyweaver.tasks.progress — publish_progress."""
import json
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from easyweaver.tasks.progress import publish_progress


# ---------------------------------------------------------------------------
# publish_progress
# ---------------------------------------------------------------------------


class TestPublishProgress:
    @pytest.mark.anyio
    async def test_publishes_to_correct_channel(self):
        redis = AsyncMock()
        redis.publish = AsyncMock()

        await publish_progress(redis, "run-123", "running", "Starting")

        redis.publish.assert_called_once()
        channel = redis.publish.call_args[0][0]
        assert channel == "query_progress:run-123"

    @pytest.mark.anyio
    async def test_payload_is_valid_json(self):
        redis = AsyncMock()
        redis.publish = AsyncMock()

        await publish_progress(redis, "run-123", "running", "Some message")

        payload_str = redis.publish.call_args[0][1]
        payload = json.loads(payload_str)
        assert isinstance(payload, dict)

    @pytest.mark.anyio
    async def test_payload_contains_run_id(self):
        redis = AsyncMock()
        redis.publish = AsyncMock()

        await publish_progress(redis, "run-abc", "completed", "Done")

        payload = json.loads(redis.publish.call_args[0][1])
        assert payload["run_id"] == "run-abc"

    @pytest.mark.anyio
    async def test_payload_contains_status(self):
        redis = AsyncMock()
        redis.publish = AsyncMock()

        await publish_progress(redis, "run-1", "failed", "Error occurred")

        payload = json.loads(redis.publish.call_args[0][1])
        assert payload["status"] == "failed"

    @pytest.mark.anyio
    async def test_payload_contains_message(self):
        redis = AsyncMock()
        redis.publish = AsyncMock()

        await publish_progress(redis, "run-1", "running", "Processing 500 rows")

        payload = json.loads(redis.publish.call_args[0][1])
        assert payload["message"] == "Processing 500 rows"

    @pytest.mark.anyio
    async def test_channel_uses_run_id_suffix(self):
        redis = AsyncMock()
        redis.publish = AsyncMock()

        await publish_progress(redis, "my-unique-run", "completed", "Done")

        channel = redis.publish.call_args[0][0]
        assert channel.endswith("my-unique-run")

    @pytest.mark.anyio
    async def test_channel_prefix_is_query_progress(self):
        redis = AsyncMock()
        redis.publish = AsyncMock()

        await publish_progress(redis, "run-999", "running", "msg")

        channel = redis.publish.call_args[0][0]
        assert channel.startswith("query_progress:")

    @pytest.mark.anyio
    async def test_different_run_ids_use_different_channels(self):
        redis = AsyncMock()
        redis.publish = AsyncMock()

        await publish_progress(redis, "run-A", "running", "msg")
        await publish_progress(redis, "run-B", "running", "msg")

        calls = redis.publish.call_args_list
        channels = [c[0][0] for c in calls]
        assert channels[0] != channels[1]
        assert "run-A" in channels[0]
        assert "run-B" in channels[1]

    @pytest.mark.anyio
    async def test_all_statuses_are_published(self):
        redis = AsyncMock()
        redis.publish = AsyncMock()

        for status in ["running", "completed", "failed", "paused", "cancelled"]:
            await publish_progress(redis, "run-1", status, "msg")

        assert redis.publish.call_count == 5

    @pytest.mark.anyio
    async def test_payload_has_exactly_three_keys(self):
        redis = AsyncMock()
        redis.publish = AsyncMock()

        await publish_progress(redis, "run-1", "running", "msg")

        payload = json.loads(redis.publish.call_args[0][1])
        assert set(payload.keys()) == {"run_id", "status", "message"}

    @pytest.mark.anyio
    async def test_empty_message_is_published(self):
        redis = AsyncMock()
        redis.publish = AsyncMock()

        await publish_progress(redis, "run-1", "running", "")

        payload = json.loads(redis.publish.call_args[0][1])
        assert payload["message"] == ""

    @pytest.mark.anyio
    async def test_redis_publish_awaited(self):
        redis = MagicMock()
        redis.publish = AsyncMock(return_value=1)

        await publish_progress(redis, "run-1", "running", "msg")

        redis.publish.assert_awaited_once()
