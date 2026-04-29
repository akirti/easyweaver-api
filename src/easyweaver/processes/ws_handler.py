"""WebSocket handler for real-time process execution.

Accepts a WebSocket connection for a given process configuration and
dispatches incoming messages to start, attach, pause, resume, cancel,
or adjust batch sizing.  Progress events from the executor are forwarded
to the client in real time.
"""

from __future__ import annotations

import asyncio
import uuid

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.processes import service
from easyweaver.processes.dag import build_dag
from easyweaver.processes.progress import ProcessProgressTracker
from easyweaver.processes.schemas import ProcessConfig

logger = structlog.get_logger()

# Module-level registry so that reconnecting clients can find a running handler
# by run_id and wire into its progress stream.
_active_handlers: dict[str, "ProcessWebSocketHandler"] = {}


class ProcessWebSocketHandler:
    """Manages a single WebSocket connection for process execution."""

    def __init__(self, websocket: WebSocket, config_id: str, db: AsyncIOMotorDatabase):
        self.ws = websocket
        self.config_id = config_id
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
        self._progress_tracker: ProcessProgressTracker | None = None
        # Additional WS clients observing this handler (for reconnection)
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
        "attach": "_handle_attach",
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
        """Start a new process execution."""
        param_values = msg.get("param_values", {})
        max_rows = msg.get("max_rows", 1000)
        save_to_gcp = msg.get("save_results_to_gcp", False)
        config_source = msg.get("config_source", "auto")
        target_seconds = msg.get("target_batch_seconds")

        if target_seconds is not None:
            self.control["target_batch_seconds"] = target_seconds

        try:
            config = await service.get_configuration(self.db, self.config_id)
        except Exception as exc:
            await self._send({"type": "error", "error": str(exc)})
            return

        run = await service.create_process_run(
            self.db,
            process_id=str(config.id),
            user_id="system",
            param_values=param_values,
        )
        self.run_id = str(run.id)
        self._progress_tracker = ProcessProgressTracker(self.run_id, self.db)

        # Register in module-level map for reconnection
        _active_handlers[self.run_id] = self

        # Build DAG and extract dataset keys for the frontend
        from easyweaver.processes.executor import resolve_params, coerce_param_values

        process_config = ProcessConfig.model_validate(config.config)
        coerced_params = coerce_param_values(param_values, config.params)
        resolved_dict = resolve_params(process_config.model_dump(), coerced_params)
        resolved_config = ProcessConfig.model_validate(resolved_dict)

        dag = build_dag(resolved_config.queries)
        dataset_keys = []
        for schema_name, queries in resolved_config.queries.items():
            for query_name in queries:
                dataset_keys.append(f"{schema_name}.{query_name}")

        await self._send({
            "type": "run_started",
            "run_id": self.run_id,
            "phases": ["fetching", "joining", "transforming"],
            "datasets": dataset_keys,
            "dag": dag,
        })

        # Launch background execution task
        self._execution_task = asyncio.create_task(
            self._run_execution(config, param_values, max_rows, save_to_gcp, config_source)
        )

    async def _handle_attach(self, msg: dict) -> None:
        """Attach to an existing running process (reconnection)."""
        run_id = msg.get("run_id")
        if not run_id:
            await self._send({"type": "error", "error": "run_id is required for attach"})
            return

        # Check if there is a live handler for this run
        active = _active_handlers.get(run_id)
        if active is not None:
            # Wire this WS as an observer on the active handler
            active._observers.append(self.ws)
            self.run_id = run_id

        # Load the run from MongoDB to get current state
        try:
            run = await service.get_process_run(self.db, run_id)
        except Exception as exc:
            await self._send({"type": "error", "error": str(exc)})
            return

        # Send current progress snapshot
        if run.progress:
            await self._send({"type": "state_snapshot", "run_id": str(run.id), "progress": run.progress, "control": run.control or {}})

        # If run is terminal, send final status
        if run.status in ("completed", "failed", "cancelled"):
            final_msg: dict = {"type": run.status, "run_id": str(run.id)}
            if run.status == "completed":
                final_msg["total_rows"] = run.row_count
                final_msg["result_run_id"] = run.result_run_id
            elif run.status == "failed":
                final_msg["error"] = run.error
            await self._send(final_msg)
        else:
            await self._send({"type": "attached", "run_id": str(run.id), "status": run.status})

    async def _handle_pause(self, _msg: dict) -> None:
        self.control["paused"] = True
        if self.run_id:
            await service.update_process_run(self.db, self.run_id, control=self.control)
        if self._progress_tracker:
            await self._progress_tracker.set_paused(True)
        await self._send({"type": "paused"})

    async def _handle_resume(self, _msg: dict) -> None:
        self.control["paused"] = False
        if self.run_id:
            await service.update_process_run(self.db, self.run_id, control=self.control)
        if self._progress_tracker:
            await self._progress_tracker.set_paused(False)
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

    async def _handle_cancel(self, _msg: dict) -> None:
        self.control["cancelled"] = True
        if self.run_id:
            await service.update_process_run(self.db, self.run_id, control=self.control)
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
        """Callback passed to execute_process; forwards events to WS clients.

        In addition to broadcasting via WebSocket, progress is persisted to
        MongoDB through ``ProcessProgressTracker`` so that reconnecting
        clients can replay the latest state.
        """
        await self._broadcast({"type": event_type, **data})

        tracker = self._progress_tracker
        if tracker is None:
            return

        try:
            if event_type == "phase":
                await tracker.set_phase(data.get("phase", ""), data.get("phase_index", 0))

            elif event_type in ("fetch_started", "fetch_waiting"):
                dataset = data.get("dataset", "")
                await tracker.init_dataset(dataset, depends_on=data.get("depends_on"))
                status = "fetching" if event_type == "fetch_started" else "waiting"
                await tracker.set_dataset_status(dataset, status)

            elif event_type == "fetch_progress":
                dataset = data.get("dataset", "")
                await tracker.update_dataset(
                    dataset,
                    rows_fetched=data.get("rows_fetched", 0),
                    batch_number=data.get("batch_number", 0),
                    batch_size=data.get("batch_size", 0),
                    status="fetching",
                )

            elif event_type == "fetch_complete":
                dataset = data.get("dataset", "")
                await tracker.set_dataset_status(dataset, "completed")
                await tracker.update_dataset(dataset, rows_fetched=data.get("total_rows", 0))

            elif event_type in ("join_progress", "transform_progress"):
                await tracker.set_current_operation(data.get("operation"))

        except Exception:
            logger.warning("progress_tracker_error", run_id=self.run_id, event=event_type, exc_info=True)

    # ── Background execution ──────────────────────────────────────────

    async def _run_execution(
        self,
        config,
        param_values: dict,
        max_rows: int,
        save_to_gcp: bool,
        config_source: str,
    ) -> None:
        """Background task that runs the process and sends completion/error."""
        from easyweaver.dependencies import get_query_semaphore
        from easyweaver.processes.executor import coerce_param_values, execute_process
        from easyweaver.processes.schemas import ProcessConfig
        from easyweaver.results.redis_store import RedisResultStore
        from easyweaver.settings import settings
        from redis.asyncio import Redis

        semaphore = get_query_semaphore()
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        store = RedisResultStore(redis)
        run_id = self.run_id

        try:
            async with semaphore:
                await service.update_process_run(self.db, run_id, status="running")

                use_gcp = False

                if config_source == "gcp" or (
                    config_source == "auto"
                    and config.gcp_path
                    and config.save_destination in ("gcp", "both")
                ):
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
                            config_id=self.config_id,
                            error=str(e),
                        )

                if not use_gcp:
                    process_config = ProcessConfig.model_validate(config.config)
                    param_defs = config.params

                coerced_params = coerce_param_values(param_values, param_defs)

                exec_db = None if use_gcp else self.db
                df = await asyncio.wait_for(
                    execute_process(
                        process_config,
                        coerced_params,
                        db=exec_db,
                        progress_callback=self._progress_callback,
                        control=self.control,
                        run_id=run_id,
                    ),
                    timeout=settings.query_timeout_seconds,
                )

                # Enforce row limit
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

                if save_to_gcp:
                    try:
                        run = await service.get_process_run(self.db, run_id)
                        gcp_path = service.save_results_to_gcp(run, df)
                        update_kwargs["result_gcp_path"] = gcp_path
                    except Exception as e:
                        logger.warning("gcs_save_failed", run_id=run_id, error=str(e))

                await service.update_process_run(self.db, run_id, **update_kwargs)
                logger.info("process_completed_ws", run_id=run_id, rows=len(df))

                await self._broadcast({
                    "type": "completed",
                    "run_id": run_id,
                    "total_rows": len(df),
                    "result_run_id": result_run_id,
                })

        except asyncio.CancelledError:
            logger.info("process_cancelled_ws", run_id=run_id)
            await service.update_process_run(self.db, run_id, status="cancelled")
            await self._broadcast({"type": "cancelled", "run_id": run_id})
            raise

        except asyncio.TimeoutError:
            msg = f"Process timed out after {settings.query_timeout_seconds}s"
            logger.warning("process_timeout_ws", run_id=run_id)
            await service.update_process_run(self.db, run_id, status="failed", error=msg)
            await self._broadcast({"type": "error", "run_id": run_id, "error": msg})

        except Exception as e:
            logger.exception("process_execution_failed_ws", run_id=run_id, error=str(e))
            await service.update_process_run(self.db, run_id, status="failed", error=str(e))
            await self._broadcast({"type": "error", "run_id": run_id, "error": str(e)})

        finally:
            await redis.aclose()
            # Unregister from active handlers
            if run_id and run_id in _active_handlers:
                del _active_handlers[run_id]
