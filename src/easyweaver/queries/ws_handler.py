"""WebSocket handler for real-time query execution with progress.

Accepts a WebSocket connection and dispatches incoming messages to start,
pause, resume, cancel, or adjust batch sizing.  Progress events from the
batched executor are forwarded to the client in real time.

Modelled after ``easyweaver.processes.ws_handler`` but simplified: no DAG,
no params, no GCP.
"""

from __future__ import annotations

import asyncio
import uuid

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.queries import service
from easyweaver.queries.schemas import QueryRequest

logger = structlog.get_logger()

# Module-level registry so that reconnecting clients can find a running handler.
_active_query_handlers: dict[str, "QueryWebSocketHandler"] = {}


class QueryWebSocketHandler:
    """Manages a single WebSocket connection for query execution."""

    def __init__(self, websocket: WebSocket, db: AsyncIOMotorDatabase):
        self.ws = websocket
        self.db = db
        self.run_id: str | None = None
        self.control: dict = {
            "paused": False,
            "batch_size_override": None,
            "target_batch_seconds": 10.0,
            "adaptive_enabled": True,
            "cancelled": False,
        }
        self._execution_task: asyncio.Task | None = None
        self._observers: list[WebSocket] = []

    # ── Public entry point ────────────────────────────────────────────

    async def handle(self) -> None:
        """Accept the connection and enter the receive-dispatch loop."""
        await self.ws.accept()
        try:
            while True:
                data = await self.ws.receive_json()
                await self._dispatch(data)
        except WebSocketDisconnect:
            pass  # Client disconnected; execution continues in background

    # ── Message dispatch ──────────────────────────────────────────────

    _HANDLERS = {
        "start": "_handle_start",
        "pause": "_handle_pause",
        "resume": "_handle_resume",
        "set_batch_size": "_handle_set_batch_size",
        "set_target_seconds": "_handle_set_target_seconds",
        "cancel": "_handle_cancel",
    }

    async def _dispatch(self, msg: dict) -> None:
        """Route an incoming message to the appropriate handler by type."""
        msg_type = msg.get("type")
        handler_name = self._HANDLERS.get(msg_type)
        if handler_name is None:
            await self._send({"type": "error", "error": f"Unknown message type: {msg_type}"})
            return
        handler = getattr(self, handler_name)
        await handler(msg)

    # ── Individual message handlers ───────────────────────────────────

    async def _handle_start(self, msg: dict) -> None:
        """Start a new query execution."""
        request_data = msg.get("request")
        if not request_data:
            await self._send({"type": "error", "error": "'request' payload is required"})
            return

        target_seconds = msg.get("target_batch_seconds")
        if target_seconds is not None:
            self.control["target_batch_seconds"] = target_seconds

        try:
            request = QueryRequest.model_validate(request_data)
        except Exception as exc:
            await self._send({"type": "error", "error": f"Invalid QueryRequest: {exc}"})
            return

        run = await service.create_query_run(self.db, request)
        self.run_id = str(run.id)

        # Register in module-level map
        _active_query_handlers[self.run_id] = self

        await self._send({
            "type": "run_started",
            "run_id": self.run_id,
            "phases": ["fetching"],
            "datasets": ["query"],
            "dag": {},
        })

        # Launch background execution task
        self._execution_task = asyncio.create_task(
            self._run_execution(request)
        )

    async def _handle_pause(self, msg: dict) -> None:
        self.control["paused"] = True
        if self.run_id:
            await service.update_query_run_progress(self.db, self.run_id, control=self.control)
        await self._send({"type": "paused"})

    async def _handle_resume(self, msg: dict) -> None:
        self.control["paused"] = False
        if self.run_id:
            await service.update_query_run_progress(self.db, self.run_id, control=self.control)
        await self._send({"type": "resumed"})

    async def _handle_set_batch_size(self, msg: dict) -> None:
        self.control["batch_size_override"] = msg.get("batch_size")
        self.control["adaptive_enabled"] = False
        await self._send({"type": "batch_size_set", "batch_size": msg.get("batch_size")})

    async def _handle_set_target_seconds(self, msg: dict) -> None:
        self.control["target_batch_seconds"] = msg.get("target_seconds", 10.0)
        self.control["adaptive_enabled"] = True
        self.control["batch_size_override"] = None
        await self._send({"type": "target_seconds_set", "target_seconds": self.control["target_batch_seconds"]})

    async def _handle_cancel(self, msg: dict) -> None:
        self.control["cancelled"] = True
        if self.run_id:
            await service.update_query_run_progress(self.db, self.run_id, control=self.control)
        if self._execution_task and not self._execution_task.done():
            self._execution_task.cancel()
        await self._send({"type": "cancelled"})

    # ── WebSocket send helpers ────────────────────────────────────────

    async def _send(self, msg: dict) -> None:
        """Send JSON to the primary WebSocket, ignoring errors on disconnect."""
        try:
            await self.ws.send_json(msg)
        except Exception:
            pass  # Client may have disconnected

    async def _broadcast(self, msg: dict) -> None:
        """Send to primary WS and all observer WS connections."""
        await self._send(msg)
        dead: list[WebSocket] = []
        for obs in self._observers:
            try:
                await obs.send_json(msg)
            except Exception:
                dead.append(obs)
        for d in dead:
            self._observers.remove(d)

    # ── Progress callback ─────────────────────────────────────────────

    async def _progress_callback(self, event_type: str, **data) -> None:
        """Callback passed to execute_single_source_batched; forwards events to WS.

        Progress is persisted to MongoDB on key events so that clients can
        recover state after reconnection.
        """
        await self._broadcast({"type": event_type, **data})

        # Persist progress on key events
        if event_type in ("fetch_complete", "completed", "error"):
            try:
                await service.update_query_run_progress(
                    self.db,
                    self.run_id,
                    progress={"last_event": event_type, **data},
                )
            except Exception:
                logger.warning(
                    "query_progress_persist_error",
                    run_id=self.run_id,
                    event=event_type,
                    exc_info=True,
                )

    # ── Background execution ──────────────────────────────────────────

    async def _run_execution(self, request: QueryRequest) -> None:
        """Background task that runs the query and sends completion/error."""
        from easyweaver.dependencies import get_meta_db, get_query_semaphore
        from easyweaver.queries.executor import (
            execute_single_source_batched,
            execute_join,
            apply_sort,
            resolve_cross_dataset_filters,
        )
        from easyweaver.sources.service import get_source
        from easyweaver.results.redis_store import RedisResultStore
        from easyweaver.settings import settings
        from redis.asyncio import Redis

        semaphore = get_query_semaphore()
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        store = RedisResultStore(redis)
        run_id = self.run_id
        db = self.db

        try:
            async with semaphore:
                await service.update_query_run(db, run_id, status="running")

                # Send phase event
                await self._broadcast({
                    "type": "phase",
                    "phase": "fetching",
                    "phase_index": 1,
                    "total_phases": 1,
                })

                # Send fetch_started event
                await self._broadcast({
                    "type": "fetch_started",
                    "dataset": "query",
                })

                # Resolve cross-dataset filter references
                from easyweaver.queries.schemas import FilterCondition as FC

                left_config = request.left
                if any(f.value_from for f in left_config.filters):
                    resolved = await resolve_cross_dataset_filters(
                        store, [f.model_dump() for f in left_config.filters]
                    )
                    left_config = left_config.model_copy(
                        update={"filters": [FC(**fd) for fd in resolved]}
                    )

                if request.type == "single":
                    source = await get_source(db, left_config.source_id)
                    df = await asyncio.wait_for(
                        execute_single_source_batched(
                            source,
                            left_config,
                            row_limit=settings.max_result_rows,
                            progress_callback=self._progress_callback,
                            control=self.control,
                        ),
                        timeout=settings.query_timeout_seconds,
                    )
                else:
                    # Join queries: fall back to non-batched execution
                    assert request.right is not None and request.join is not None
                    right_config = request.right
                    if any(f.value_from for f in right_config.filters):
                        resolved_right = await resolve_cross_dataset_filters(
                            store, [f.model_dump() for f in right_config.filters]
                        )
                        right_config = right_config.model_copy(
                            update={"filters": [FC(**fd) for fd in resolved_right]}
                        )
                    left_source = await get_source(db, left_config.source_id)
                    right_source = await get_source(db, right_config.source_id)
                    df = await asyncio.wait_for(
                        execute_join(
                            left_source, right_source,
                            left_config, right_config,
                            request.join,
                        ),
                        timeout=settings.query_timeout_seconds,
                    )
                    # Emit fetch_complete for join since there's no batched callback
                    await self._progress_callback("fetch_complete", dataset="query", total_rows=len(df))

                # Resolve data bindings
                if request.bindings:
                    from easyweaver.queries.operations.binding import (
                        resolve_distinct_bindings,
                        resolve_row_pair_bindings,
                        apply_row_pair_filter,
                    )
                    from easyweaver.queries.operations.filter import apply_filters

                    distinct_filters = await resolve_distinct_bindings(
                        store, [b.model_dump() for b in request.bindings]
                    )
                    if distinct_filters:
                        df = apply_filters(df, distinct_filters, "and")

                    pair_df = await resolve_row_pair_bindings(
                        store, [b.model_dump() for b in request.bindings]
                    )
                    if pair_df is not None:
                        df = apply_row_pair_filter(df, pair_df)

                # Apply transforms
                if request.transforms:
                    from easyweaver.queries.operations.transform import apply_transforms
                    df = apply_transforms(df, [t.model_dump() for t in request.transforms])

                # Apply group_by
                if request.group_by:
                    from easyweaver.queries.operations.group_by import apply_group_by
                    df = apply_group_by(df, request.group_by.model_dump())

                # Apply distinct
                if request.distinct:
                    from easyweaver.queries.operations.distinct import apply_distinct
                    df = apply_distinct(df, request.distinct.model_dump())

                # Apply sort
                if request.sort:
                    df = apply_sort(df, [s.model_dump() for s in request.sort])

                # Enforce row limit
                if len(df) > settings.max_result_rows:
                    df = df.head(settings.max_result_rows)

                # Store result in Redis
                result_run_id = str(uuid.uuid4())
                await store.store_result(run_id, df)

                await service.update_query_run(
                    db, run_id, status="completed", row_count=len(df),
                )
                logger.info("query_completed_ws", run_id=run_id, rows=len(df))

                await self._broadcast({
                    "type": "completed",
                    "run_id": run_id,
                    "total_rows": len(df),
                })

        except asyncio.CancelledError:
            logger.info("query_cancelled_ws", run_id=run_id)
            await service.update_query_run(db, run_id, status="cancelled")
            await self._broadcast({"type": "cancelled", "run_id": run_id})

        except asyncio.TimeoutError:
            msg = f"Query timed out after {settings.query_timeout_seconds}s"
            logger.warning("query_timeout_ws", run_id=run_id)
            await service.update_query_run(db, run_id, status="failed", error=msg)
            await self._broadcast({"type": "error", "run_id": run_id, "error": msg})

        except Exception as e:
            logger.exception("query_execution_failed_ws", run_id=run_id, error=str(e))
            await service.update_query_run(db, run_id, status="failed", error=str(e))
            await self._broadcast({"type": "error", "run_id": run_id, "error": str(e)})

        finally:
            await redis.aclose()
            # Unregister from active handlers
            if run_id and run_id in _active_query_handlers:
                del _active_query_handlers[run_id]
