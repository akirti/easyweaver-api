"""Tests for easyweaver.processes.router — all HTTP endpoints and WS auth."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from easyweaver.processes.models import ProcessConfiguration, ProcessRun


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_config(name: str = "Test Config", config_id: str | None = None) -> ProcessConfiguration:
    uid = uuid.UUID(config_id) if config_id else uuid.uuid4()
    now = datetime.now(timezone.utc)
    return ProcessConfiguration(
        id=uid,
        user_id="system",
        name=name,
        description="A test config",
        version=1,
        config={"queries": {}, "logics": []},
        params={},
        save_destination="redis",
        gcp_path="",
        tags=[],
        created_at=now,
        updated_at=now,
    )


def _make_run(
    process_id: str = "proc-1",
    status: str = "pending",
    run_id: str | None = None,
) -> ProcessRun:
    uid = uuid.UUID(run_id) if run_id else uuid.uuid4()
    now = datetime.now(timezone.utc)
    return ProcessRun(
        id=uid,
        process_id=process_id,
        user_id="system",
        param_values={},
        status=status,
        row_count=None,
        error=None,
        result_run_id="",
        result_gcp_path="",
        progress=None,
        control=None,
        created_at=now,
        updated_at=now,
    )


def _make_mock_db() -> MagicMock:
    db = MagicMock()
    db.process_configurations = MagicMock()
    db.process_configurations.find_one = AsyncMock(return_value=None)
    db.process_configurations.insert_one = AsyncMock()
    db.process_configurations.update_one = AsyncMock()
    db.process_configurations.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    db.process_runs = MagicMock()
    db.process_runs.find_one = AsyncMock(return_value=None)
    db.process_runs.insert_one = AsyncMock()
    db.process_runs.update_one = AsyncMock()
    return db


# ── _config_to_response / _run_to_response ────────────────────────────────────


class TestConfigToResponse:
    def test_converts_config_to_dict(self):
        from easyweaver.processes.router import _config_to_response

        config = _make_config()
        result = _config_to_response(config)
        assert result["id"] == str(config.id)
        assert result["name"] == config.name
        assert result["user_id"] == "system"
        assert result["version"] == 1

    def test_strips_encrypted_credentials(self):
        from easyweaver.processes.router import _config_to_response

        config = _make_config()
        config.config = {
            "queries": {
                "schema1": {
                    "q1": {
                        "source_id": "src-1",
                        "table": "users",
                        "encrypted_credentials": "secret-stuff",
                    }
                }
            },
            "logics": [],
        }
        result = _config_to_response(config)
        q1 = result["config"]["queries"]["schema1"]["q1"]
        assert "encrypted_credentials" not in q1

    def test_non_encrypted_fields_preserved(self):
        from easyweaver.processes.router import _config_to_response

        config = _make_config()
        config.config = {
            "queries": {
                "schema1": {
                    "q1": {
                        "source_id": "src-1",
                        "table": "users",
                        "columns": ["id", "name"],
                    }
                }
            },
            "logics": [],
        }
        result = _config_to_response(config)
        q1 = result["config"]["queries"]["schema1"]["q1"]
        assert q1["table"] == "users"
        assert q1["columns"] == ["id", "name"]


class TestRunToResponse:
    def test_converts_run_to_dict(self):
        from easyweaver.processes.router import _run_to_response

        run = _make_run()
        result = _run_to_response(run)
        assert result["id"] == str(run.id)
        assert result["process_id"] == run.process_id
        assert result["status"] == "pending"
        assert result["param_values"] == {}


# ── GET /runs/{run_id} ────────────────────────────────────────────────────────


class TestGetRun:
    @pytest.mark.anyio
    async def test_returns_run_when_found(self):
        from easyweaver.processes import router

        run = _make_run(status="completed")
        db = _make_mock_db()

        with patch.object(router.service, "get_process_run", AsyncMock(return_value=run)):
            result = await router.get_run(str(run.id), db=db)

        assert result["id"] == str(run.id)
        assert result["status"] == "completed"

    @pytest.mark.anyio
    async def test_raises_when_not_found(self):
        from easyweaver.processes import router
        from easyweaver.core.exceptions import NotFoundError

        db = _make_mock_db()

        with patch.object(
            router.service, "get_process_run",
            AsyncMock(side_effect=NotFoundError("ProcessRun", "bad-id"))
        ):
            with pytest.raises(NotFoundError):
                await router.get_run("bad-id", db=db)


# ── GET /{config_id} ─────────────────────────────────────────────────────────


class TestGetConfiguration:
    @pytest.mark.anyio
    async def test_returns_config_when_found(self):
        from easyweaver.processes import router

        config = _make_config(name="My Config")
        db = _make_mock_db()

        with patch.object(router.service, "get_configuration", AsyncMock(return_value=config)):
            result = await router.get_configuration(str(config.id), db=db)

        assert result["id"] == str(config.id)
        assert result["name"] == "My Config"

    @pytest.mark.anyio
    async def test_raises_when_not_found(self):
        from easyweaver.processes import router
        from easyweaver.core.exceptions import NotFoundError

        db = _make_mock_db()

        with patch.object(
            router.service, "get_configuration",
            AsyncMock(side_effect=NotFoundError("ProcessConfiguration", "bad-id"))
        ):
            with pytest.raises(NotFoundError):
                await router.get_configuration("bad-id", db=db)


# ── GET / (list configurations) ───────────────────────────────────────────────


class TestListConfigurations:
    @pytest.mark.anyio
    async def test_returns_list(self):
        from easyweaver.processes import router

        configs = [_make_config("A"), _make_config("B")]
        db = _make_mock_db()

        with patch.object(router.service, "list_configurations", AsyncMock(return_value=configs)):
            result = await router.list_configurations(db=db)

        assert len(result) == 2
        names = [r["name"] for r in result]
        assert "A" in names
        assert "B" in names

    @pytest.mark.anyio
    async def test_empty_list(self):
        from easyweaver.processes import router

        db = _make_mock_db()

        with patch.object(router.service, "list_configurations", AsyncMock(return_value=[])):
            result = await router.list_configurations(db=db)

        assert result == []

    @pytest.mark.anyio
    async def test_filters_by_user_id(self):
        from easyweaver.processes import router

        config = _make_config("User Config")
        db = _make_mock_db()

        mock_list = AsyncMock(return_value=[config])
        with patch.object(router.service, "list_configurations", mock_list):
            result = await router.list_configurations(user_id="user-123", db=db)

        mock_list.assert_called_once_with(db, user_id="user-123")
        assert len(result) == 1


# ── POST / (create configuration) ─────────────────────────────────────────────


class TestCreateConfiguration:
    @pytest.mark.anyio
    async def test_creates_and_returns_config(self):
        from easyweaver.processes import router
        from easyweaver.processes.schemas import ProcessConfigurationCreate, ProcessConfig

        config = _make_config("New Config")
        db = _make_mock_db()

        data = ProcessConfigurationCreate(
            name="New Config",
            config=ProcessConfig(queries={}, logics=[]),
        )

        with patch.object(router.service, "create_configuration", AsyncMock(return_value=config)):
            result = await router.create_configuration(data, db=db)

        assert result["name"] == "New Config"
        assert "id" in result


# ── PUT /{config_id} (update configuration) ───────────────────────────────────


class TestUpdateConfiguration:
    @pytest.mark.anyio
    async def test_updates_and_returns_config(self):
        from easyweaver.processes import router
        from easyweaver.processes.schemas import ProcessConfigurationUpdate

        config = _make_config("Updated Name")
        db = _make_mock_db()
        data = ProcessConfigurationUpdate(name="Updated Name")

        with patch.object(router.service, "update_configuration", AsyncMock(return_value=config)):
            result = await router.update_configuration(str(config.id), data, db=db)

        assert result["name"] == "Updated Name"

    @pytest.mark.anyio
    async def test_raises_when_not_found(self):
        from easyweaver.processes import router
        from easyweaver.processes.schemas import ProcessConfigurationUpdate
        from easyweaver.core.exceptions import NotFoundError

        db = _make_mock_db()
        data = ProcessConfigurationUpdate(name="New Name")

        with patch.object(
            router.service, "update_configuration",
            AsyncMock(side_effect=NotFoundError("ProcessConfiguration", "bad-id"))
        ):
            with pytest.raises(NotFoundError):
                await router.update_configuration("bad-id", data, db=db)


# ── DELETE /{config_id} ───────────────────────────────────────────────────────


class TestDeleteConfiguration:
    @pytest.mark.anyio
    async def test_deletes_successfully(self):
        from easyweaver.processes import router

        config = _make_config()
        db = _make_mock_db()

        with patch.object(router.service, "delete_configuration", AsyncMock(return_value=None)):
            # Should not raise
            await router.delete_configuration(str(config.id), db=db)

    @pytest.mark.anyio
    async def test_raises_when_not_found(self):
        from easyweaver.processes import router
        from easyweaver.core.exceptions import NotFoundError

        db = _make_mock_db()

        with patch.object(
            router.service, "delete_configuration",
            AsyncMock(side_effect=NotFoundError("ProcessConfiguration", "bad-id"))
        ):
            with pytest.raises(NotFoundError):
                await router.delete_configuration("bad-id", db=db)


# ── POST /{config_id}/run ─────────────────────────────────────────────────────


class TestRunProcess:
    @pytest.mark.anyio
    async def test_creates_run_and_returns_pending(self):
        from easyweaver.processes import router
        from easyweaver.processes.schemas import ProcessRunRequest

        config = _make_config()
        run = _make_run(process_id=str(config.id), status="pending")
        db = _make_mock_db()

        request = ProcessRunRequest(param_values={"x": 1}, max_rows=100)

        # settings is imported inside the function body — patch its actual module
        with (
            patch.object(router.service, "get_configuration", AsyncMock(return_value=config)),
            patch.object(router.service, "create_process_run", AsyncMock(return_value=run)),
            patch("asyncio.create_task", MagicMock()),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10000
            result = await router.run_process(str(config.id), request, db=db)

        assert result["id"] == str(run.id)
        assert result["status"] == "pending"

    @pytest.mark.anyio
    async def test_exceeds_max_rows_raises_422(self):
        from easyweaver.processes import router
        from easyweaver.processes.schemas import ProcessRunRequest
        from fastapi import HTTPException

        db = _make_mock_db()
        request = ProcessRunRequest(param_values={}, max_rows=9999999)

        with patch("easyweaver.settings.settings") as mock_settings:
            mock_settings.max_result_rows = 1000
            with pytest.raises(HTTPException) as exc_info:
                await router.run_process("some-config-id", request, db=db)

        assert exc_info.value.status_code == 422
        assert "max_rows" in str(exc_info.value.detail)

    @pytest.mark.anyio
    async def test_config_not_found_propagates(self):
        from easyweaver.processes import router
        from easyweaver.processes.schemas import ProcessRunRequest
        from easyweaver.core.exceptions import NotFoundError

        db = _make_mock_db()
        request = ProcessRunRequest(max_rows=100)

        with (
            patch.object(
                router.service, "get_configuration",
                AsyncMock(side_effect=NotFoundError("ProcessConfiguration", "bad-id"))
            ),
            patch("easyweaver.settings.settings") as mock_settings,
        ):
            mock_settings.max_result_rows = 10000
            with pytest.raises(NotFoundError):
                await router.run_process("bad-id", request, db=db)


# ── GET /{config_id}/runs ─────────────────────────────────────────────────────


class TestListRuns:
    @pytest.mark.anyio
    async def test_returns_run_history(self):
        from easyweaver.processes import router

        runs = [_make_run(status="completed"), _make_run(status="failed")]
        db = _make_mock_db()

        with patch.object(router.service, "list_process_runs", AsyncMock(return_value=runs)):
            result = await router.list_runs("proc-1", db=db)

        assert result.total == 2
        assert len(result.runs) == 2

    @pytest.mark.anyio
    async def test_empty_run_history(self):
        from easyweaver.processes import router

        db = _make_mock_db()

        with patch.object(router.service, "list_process_runs", AsyncMock(return_value=[])):
            result = await router.list_runs("proc-1", db=db)

        assert result.total == 0
        assert result.runs == []


# ── WS /{config_id}/run/ws — auth validation ─────────────────────────────────


class TestRunProcessWsAuth:
    @pytest.mark.anyio
    async def test_missing_token_closes_with_4003(self):
        from easyweaver.processes import router

        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()
        db = _make_mock_db()

        await router.run_process_ws(mock_ws, "cfg-1", token=None, db=db)

        mock_ws.close.assert_called_once_with(code=4003, reason="Forbidden: token required")

    @pytest.mark.anyio
    async def test_invalid_token_closes_with_4003(self):
        from easyweaver.processes import router

        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()
        db = _make_mock_db()

        # decode_token is imported inside the WS function — patch its actual location
        with patch("easyweaver.auth.service.decode_token", side_effect=Exception("bad token")):
            await router.run_process_ws(mock_ws, "cfg-1", token="invalid.token.here", db=db)

        mock_ws.close.assert_called_once_with(code=4003, reason="Forbidden: invalid token")

    @pytest.mark.anyio
    async def test_valid_token_creates_handler(self):
        from easyweaver.processes import router

        mock_ws = MagicMock()
        mock_ws.accept = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.receive_json = AsyncMock(side_effect=Exception("disconnect"))
        db = _make_mock_db()

        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock()

        with (
            patch("easyweaver.auth.service.decode_token", return_value={"sub": "user-1"}),
            patch("easyweaver.processes.ws_handler.ProcessWebSocketHandler", return_value=mock_handler),
        ):
            await router.run_process_ws(mock_ws, "cfg-1", token="valid.jwt.token", db=db)

        mock_handler.handle.assert_called_once()
        # close should NOT be called with 4003
        for call in mock_ws.close.call_args_list:
            assert call[1].get("code", None) != 4003


# ── Service layer unit tests (process service CRUD) ───────────────────────────


class TestProcessService:
    @pytest.mark.anyio
    async def test_list_configurations_empty(self):
        from easyweaver.processes import service

        mock_db = MagicMock()

        async def async_iter():
            return
            yield  # make it an async generator

        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.__aiter__ = MagicMock(return_value=async_iter())
        mock_db.process_configurations.find = MagicMock(return_value=mock_cursor)

        result = await service.list_configurations(mock_db)
        assert result == []

    @pytest.mark.anyio
    async def test_get_configuration_not_found_raises(self):
        from easyweaver.processes import service
        from easyweaver.core.exceptions import NotFoundError

        mock_db = MagicMock()
        mock_db.process_configurations.find_one = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await service.get_configuration(mock_db, "nonexistent-id")

    @pytest.mark.anyio
    async def test_get_configuration_returns_dataclass(self):
        from easyweaver.processes import service

        config = _make_config()
        mock_db = MagicMock()
        mock_db.process_configurations.find_one = AsyncMock(return_value=config.to_doc())

        result = await service.get_configuration(mock_db, str(config.id))
        assert isinstance(result, ProcessConfiguration)
        assert result.name == config.name

    @pytest.mark.anyio
    async def test_create_process_run_returns_pending(self):
        from easyweaver.processes import service

        mock_db = MagicMock()
        mock_db.process_runs.insert_one = AsyncMock()

        run = await service.create_process_run(mock_db, "proc-1", "system", {"x": 1})
        assert run.status == "pending"
        assert run.process_id == "proc-1"
        assert run.param_values == {"x": 1}

    @pytest.mark.anyio
    async def test_get_process_run_not_found_raises(self):
        from easyweaver.processes import service
        from easyweaver.core.exceptions import NotFoundError

        mock_db = MagicMock()
        mock_db.process_runs.find_one = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await service.get_process_run(mock_db, "nonexistent-run-id")

    @pytest.mark.anyio
    async def test_get_process_run_returns_dataclass(self):
        from easyweaver.processes import service

        run = _make_run(status="running")
        mock_db = MagicMock()
        mock_db.process_runs.find_one = AsyncMock(return_value=run.to_doc())

        result = await service.get_process_run(mock_db, str(run.id))
        assert isinstance(result, ProcessRun)
        assert result.status == "running"

    @pytest.mark.anyio
    async def test_delete_configuration_raises_when_not_found(self):
        from easyweaver.processes import service
        from easyweaver.core.exceptions import NotFoundError

        mock_db = MagicMock()
        mock_db.process_configurations.delete_one = AsyncMock(
            return_value=MagicMock(deleted_count=0)
        )

        # delete_lookups is imported inside the function body; patch at its actual location
        with patch("easyweaver.lookups.service.delete_lookups", AsyncMock()):
            with pytest.raises(NotFoundError):
                await service.delete_configuration(mock_db, "nonexistent-id")

    @pytest.mark.anyio
    async def test_delete_configuration_succeeds(self):
        from easyweaver.processes import service

        mock_db = MagicMock()
        mock_db.process_configurations.delete_one = AsyncMock(
            return_value=MagicMock(deleted_count=1)
        )

        with patch("easyweaver.lookups.service.delete_lookups", AsyncMock()):
            # Should not raise
            await service.delete_configuration(mock_db, "some-id")

    @pytest.mark.anyio
    async def test_update_process_run_updates_fields(self):
        from easyweaver.processes import service

        run = _make_run(status="pending")
        updated_run = _make_run(status="running")
        mock_db = MagicMock()
        mock_db.process_runs.update_one = AsyncMock()
        mock_db.process_runs.find_one = AsyncMock(return_value=updated_run.to_doc())

        result = await service.update_process_run(mock_db, str(run.id), status="running")
        assert result.status == "running"

    @pytest.mark.anyio
    async def test_list_process_runs_returns_list(self):
        from easyweaver.processes import service

        mock_db = MagicMock()

        runs_data = [_make_run(status="completed"), _make_run(status="failed")]

        async def async_iter_runs():
            for r in runs_data:
                yield r.to_doc()

        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.__aiter__ = MagicMock(return_value=async_iter_runs())
        mock_db.process_runs.find = MagicMock(return_value=mock_cursor)

        result = await service.list_process_runs(mock_db, "proc-1")
        assert len(result) == 2


# ── _config_to_response does not mutate original ──────────────────────────────


class TestConfigToResponseImmutability:
    def test_does_not_mutate_original_config(self):
        from easyweaver.processes.router import _config_to_response

        config = _make_config()
        config.config = {
            "queries": {
                "s": {
                    "q": {
                        "source_id": "src-1",
                        "table": "t",
                        "encrypted_credentials": "secret",
                    }
                }
            },
            "logics": [],
        }
        original_creds = config.config["queries"]["s"]["q"]["encrypted_credentials"]
        _config_to_response(config)
        # Original should still have the credentials
        assert config.config["queries"]["s"]["q"]["encrypted_credentials"] == original_creds
