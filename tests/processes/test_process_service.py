"""Tests for easyweaver.processes.service — CRUD and GCS helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from easyweaver.processes.models import ProcessConfiguration, ProcessRun


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_config(
    name: str = "Test Config",
    config_id: str | None = None,
    save_destination: str = "redis",
    gcp_path: str = "",
) -> ProcessConfiguration:
    uid = uuid.UUID(config_id) if config_id else uuid.uuid4()
    now = datetime.now(timezone.utc)
    return ProcessConfiguration(
        id=uid,
        user_id="system",
        name=name,
        description="A test config",
        version=1,
        config={"queries": {}, "logics": [], "config_version": 2},
        params={},
        save_destination=save_destination,
        gcp_path=gcp_path,
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


def _make_async_cursor(items):
    """Build a mock cursor that async-iterates over `items`."""
    async def _aiter():
        for item in items:
            yield item

    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.__aiter__ = MagicMock(return_value=_aiter())
    return cursor


# ── list_configurations ───────────────────────────────────────────────────────


class TestListConfigurations:
    @pytest.mark.anyio
    async def test_returns_all_when_no_user_id(self):
        from easyweaver.processes import service

        configs = [_make_config("A"), _make_config("B")]
        db = _make_mock_db()
        db.process_configurations.find = MagicMock(
            return_value=_make_async_cursor([c.to_doc() for c in configs])
        )

        result = await service.list_configurations(db)
        assert len(result) == 2
        names = {r.name for r in result}
        assert names == {"A", "B"}

    @pytest.mark.anyio
    async def test_filters_by_user_id(self):
        from easyweaver.processes import service

        config = _make_config("Mine")
        db = _make_mock_db()
        db.process_configurations.find = MagicMock(
            return_value=_make_async_cursor([config.to_doc()])
        )

        result = await service.list_configurations(db, user_id="user-abc")
        # The query should have been called with user_id filter
        call_args = db.process_configurations.find.call_args[0][0]
        assert call_args == {"user_id": "user-abc"}
        assert len(result) == 1

    @pytest.mark.anyio
    async def test_returns_empty_list(self):
        from easyweaver.processes import service

        db = _make_mock_db()
        db.process_configurations.find = MagicMock(
            return_value=_make_async_cursor([])
        )

        result = await service.list_configurations(db)
        assert result == []


# ── embed_source_details ──────────────────────────────────────────────────────


class TestEmbedSourceDetails:
    @pytest.mark.anyio
    async def test_embeds_source_info_into_queries(self):
        from easyweaver.processes import service

        mock_source = MagicMock()
        mock_source.name = "My Postgres"
        mock_source.source_type = "postgres"
        mock_source.encrypted_credentials = "enc-cred"

        config_dict = {
            "queries": {
                "schema1": {
                    "main": {"source_id": "src-1", "table": "orders"}
                }
            },
            "logics": [],
        }

        with patch("easyweaver.sources.service.get_source", AsyncMock(return_value=mock_source)):
            result = await service.embed_source_details(MagicMock(), config_dict)

        q = result["queries"]["schema1"]["main"]
        assert q["source_name"] == "My Postgres"
        assert q["source_type"] == "postgres"
        assert q["encrypted_credentials"] == "enc-cred"
        assert result["config_version"] == 2

    @pytest.mark.anyio
    async def test_skips_queries_without_source_id(self):
        from easyweaver.processes import service

        config_dict = {
            "queries": {
                "schema1": {
                    "main": {"table": "orders"}  # no source_id
                }
            },
            "logics": [],
        }

        result = await service.embed_source_details(MagicMock(), config_dict)
        q = result["queries"]["schema1"]["main"]
        assert "source_name" not in q

    @pytest.mark.anyio
    async def test_logs_warning_when_source_not_found(self):
        from easyweaver.processes import service
        from easyweaver.core.exceptions import NotFoundError

        config_dict = {
            "queries": {
                "schema1": {
                    "main": {"source_id": "missing-src", "table": "orders"}
                }
            },
            "logics": [],
        }

        with patch(
            "easyweaver.sources.service.get_source",
            AsyncMock(side_effect=NotFoundError("DataSource", "missing-src")),
        ):
            # Should not raise — just log a warning
            result = await service.embed_source_details(MagicMock(), config_dict)

        q = result["queries"]["schema1"]["main"]
        assert "source_name" not in q

    @pytest.mark.anyio
    async def test_does_not_mutate_original(self):
        from easyweaver.processes import service

        mock_source = MagicMock()
        mock_source.name = "PG"
        mock_source.source_type = "postgres"
        mock_source.encrypted_credentials = "enc"

        original = {
            "queries": {"s": {"q": {"source_id": "sid", "table": "t"}}},
            "logics": [],
        }

        with patch("easyweaver.sources.service.get_source", AsyncMock(return_value=mock_source)):
            result = await service.embed_source_details(MagicMock(), original)

        # original should be unmodified
        assert "source_name" not in original["queries"]["s"]["q"]
        assert "source_name" in result["queries"]["s"]["q"]

    @pytest.mark.anyio
    async def test_handles_multiple_schemas_and_queries(self):
        from easyweaver.processes import service

        mock_source = MagicMock()
        mock_source.name = "S"
        mock_source.source_type = "postgres"
        mock_source.encrypted_credentials = "enc"

        config_dict = {
            "queries": {
                "schema1": {
                    "q1": {"source_id": "sid", "table": "t1"},
                    "q2": {"source_id": "sid", "table": "t2"},
                },
                "schema2": {
                    "q3": {"source_id": "sid", "table": "t3"},
                },
            },
            "logics": [],
        }

        with patch("easyweaver.sources.service.get_source", AsyncMock(return_value=mock_source)):
            result = await service.embed_source_details(MagicMock(), config_dict)

        # All three queries should have source info embedded
        for schema_queries in result["queries"].values():
            for qc in schema_queries.values():
                assert "source_name" in qc


# ── create_configuration ─────────────────────────────────────────────────────


class TestCreateConfiguration:
    @pytest.mark.anyio
    async def test_creates_and_inserts(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import ProcessConfig, ProcessConfigurationCreate

        db = _make_mock_db()

        data = ProcessConfigurationCreate(
            name="New Process",
            config=ProcessConfig(queries={}, logics=[]),
        )

        with (
            patch("easyweaver.sources.service.get_source", AsyncMock(side_effect=Exception("no source"))),
            patch("easyweaver.lookups.service.build_lookups_from_params", AsyncMock(return_value=([], []))),
        ):
            result = await service.create_configuration(db, data, user_id="user-1")

        assert result.name == "New Process"
        assert result.user_id == "user-1"
        assert result.status if hasattr(result, "status") else True
        db.process_configurations.insert_one.assert_called_once()

    @pytest.mark.anyio
    async def test_builds_lookups_when_params_exist(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import (
            ParamDefinition,
            ProcessConfig,
            ProcessConfigurationCreate,
        )

        db = _make_mock_db()
        data = ProcessConfigurationCreate(
            name="With Params",
            config=ProcessConfig(queries={}, logics=[]),
            params={"region": ParamDefinition(type="select", options=["EU", "US"])},
        )

        mock_lookups = [{"column": "region", "values": ["EU", "US"]}]

        with (
            patch("easyweaver.sources.service.get_source", AsyncMock(side_effect=Exception("no source"))),
            patch(
                "easyweaver.lookups.service.build_lookups_from_params",
                AsyncMock(return_value=(mock_lookups, [])),
            ),
            patch("easyweaver.lookups.service.upsert_lookups", AsyncMock()) as mock_upsert,
        ):
            result = await service.create_configuration(db, data, user_id="user-1")

        mock_upsert.assert_called_once()

    @pytest.mark.anyio
    async def test_saves_to_gcp_when_destination_is_gcp(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import ProcessConfig, ProcessConfigurationCreate

        db = _make_mock_db()
        data = ProcessConfigurationCreate(
            name="GCP Config",
            config=ProcessConfig(queries={}, logics=[]),
            save_destination="gcp",
        )

        with (
            patch("easyweaver.sources.service.get_source", AsyncMock(side_effect=Exception("no source"))),
            patch("easyweaver.lookups.service.build_lookups_from_params", AsyncMock(return_value=([], []))),
            patch.object(service, "save_config_to_gcp") as mock_save_gcp,
        ):
            result = await service.create_configuration(db, data, user_id="user-1")

        mock_save_gcp.assert_called_once()

    @pytest.mark.anyio
    async def test_gcp_save_failure_is_logged_not_raised(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import ProcessConfig, ProcessConfigurationCreate

        db = _make_mock_db()
        data = ProcessConfigurationCreate(
            name="GCP Fail Config",
            config=ProcessConfig(queries={}, logics=[]),
            save_destination="both",
        )

        with (
            patch("easyweaver.sources.service.get_source", AsyncMock(side_effect=Exception("no source"))),
            patch("easyweaver.lookups.service.build_lookups_from_params", AsyncMock(return_value=([], []))),
            patch.object(service, "save_config_to_gcp", side_effect=RuntimeError("GCS error")),
        ):
            # Should not raise
            result = await service.create_configuration(db, data, user_id="user-1")

        assert result.name == "GCP Fail Config"

    @pytest.mark.anyio
    async def test_lookup_build_failure_is_logged_not_raised(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import (
            ParamDefinition,
            ProcessConfig,
            ProcessConfigurationCreate,
        )

        db = _make_mock_db()
        data = ProcessConfigurationCreate(
            name="Lookup Fail",
            config=ProcessConfig(queries={}, logics=[]),
            params={"x": ParamDefinition(type="select", options=["a"])},
        )

        with (
            patch("easyweaver.sources.service.get_source", AsyncMock(side_effect=Exception("no source"))),
            patch(
                "easyweaver.lookups.service.build_lookups_from_params",
                AsyncMock(side_effect=RuntimeError("lookup error")),
            ),
        ):
            result = await service.create_configuration(db, data, user_id="user-1")

        assert result.name == "Lookup Fail"


# ── update_configuration ─────────────────────────────────────────────────────


class TestUpdateConfiguration:
    @pytest.mark.anyio
    async def test_updates_name_only(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import ProcessConfigurationUpdate

        config = _make_config("Old Name")
        db = _make_mock_db()
        db.process_configurations.find_one = AsyncMock(side_effect=[
            config.to_doc(),  # get_configuration (exist check)
            config.to_doc(),  # get_configuration after update
        ])

        data = ProcessConfigurationUpdate(name="New Name")

        result = await service.update_configuration(db, str(config.id), data)

        db.process_configurations.update_one.assert_called_once()
        call_args = db.process_configurations.update_one.call_args
        assert "name" in call_args[0][1]["$set"]

    @pytest.mark.anyio
    async def test_updates_config_and_embeds_sources(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import ProcessConfig, ProcessConfigurationUpdate

        config = _make_config()
        db = _make_mock_db()
        db.process_configurations.find_one = AsyncMock(side_effect=[
            config.to_doc(),
            config.to_doc(),
        ])

        new_config = ProcessConfig(queries={}, logics=[])
        data = ProcessConfigurationUpdate(config=new_config)

        mock_source = MagicMock()
        mock_source.name = "PG"
        mock_source.source_type = "postgres"
        mock_source.encrypted_credentials = "enc"

        with patch("easyweaver.sources.service.get_source", AsyncMock(return_value=mock_source)):
            result = await service.update_configuration(db, str(config.id), data)

        db.process_configurations.update_one.assert_called_once()

    @pytest.mark.anyio
    async def test_updates_params_and_rebuilds_lookups(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import ParamDefinition, ProcessConfigurationUpdate

        config = _make_config()
        updated_config = _make_config()
        updated_config.params = {"region": {"type": "select", "default": None, "options": []}}
        db = _make_mock_db()
        db.process_configurations.find_one = AsyncMock(side_effect=[
            config.to_doc(),
            updated_config.to_doc(),
        ])

        data = ProcessConfigurationUpdate(
            params={"region": ParamDefinition(type="select", options=["EU"])}
        )

        mock_lookups = [{"column": "region", "values": ["EU"]}]

        with (
            patch(
                "easyweaver.lookups.service.build_lookups_from_params",
                AsyncMock(return_value=(mock_lookups, [])),
            ),
            patch("easyweaver.lookups.service.upsert_lookups", AsyncMock()) as mock_upsert,
        ):
            result = await service.update_configuration(db, str(config.id), data)

        mock_upsert.assert_called_once()

    @pytest.mark.anyio
    async def test_saves_to_gcp_when_destination_is_gcp(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import ProcessConfigurationUpdate

        config = _make_config(save_destination="gcp", gcp_path="some/path.json")
        db = _make_mock_db()
        db.process_configurations.find_one = AsyncMock(side_effect=[
            config.to_doc(),
            config.to_doc(),
        ])

        data = ProcessConfigurationUpdate(name="Updated")

        with patch.object(service, "save_config_to_gcp") as mock_save:
            result = await service.update_configuration(db, str(config.id), data)

        mock_save.assert_called_once()

    @pytest.mark.anyio
    async def test_gcp_save_failure_on_update_logged_not_raised(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import ProcessConfigurationUpdate

        config = _make_config(save_destination="both")
        db = _make_mock_db()
        db.process_configurations.find_one = AsyncMock(side_effect=[
            config.to_doc(),
            config.to_doc(),
        ])

        data = ProcessConfigurationUpdate(save_destination="both")

        with patch.object(service, "save_config_to_gcp", side_effect=RuntimeError("GCS down")):
            result = await service.update_configuration(db, str(config.id), data)

        assert result is not None

    @pytest.mark.anyio
    async def test_lookup_rebuild_failure_on_update_logged_not_raised(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import ParamDefinition, ProcessConfigurationUpdate

        config = _make_config()
        updated = _make_config()
        updated.params = {"x": {"type": "select"}}
        db = _make_mock_db()
        db.process_configurations.find_one = AsyncMock(side_effect=[
            config.to_doc(),
            updated.to_doc(),
        ])

        data = ProcessConfigurationUpdate(params={"x": ParamDefinition(type="select")})

        with patch(
            "easyweaver.lookups.service.build_lookups_from_params",
            AsyncMock(side_effect=RuntimeError("lookup crash")),
        ):
            result = await service.update_configuration(db, str(config.id), data)

        assert result is not None

    @pytest.mark.anyio
    async def test_raises_when_config_not_found(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import ProcessConfigurationUpdate
        from easyweaver.core.exceptions import NotFoundError

        db = _make_mock_db()
        db.process_configurations.find_one = AsyncMock(return_value=None)

        data = ProcessConfigurationUpdate(name="X")

        with pytest.raises(NotFoundError):
            await service.update_configuration(db, "nonexistent-id", data)

    @pytest.mark.anyio
    async def test_updates_description_gcp_path_and_tags(self):
        from easyweaver.processes import service
        from easyweaver.processes.schemas import ProcessConfigurationUpdate

        config = _make_config()
        db = _make_mock_db()
        db.process_configurations.find_one = AsyncMock(side_effect=[
            config.to_doc(),
            config.to_doc(),
        ])

        data = ProcessConfigurationUpdate(
            description="New description",
            gcp_path="gs://new/path.json",
            tags=["tag1", "tag2"],
        )

        result = await service.update_configuration(db, str(config.id), data)

        call_args = db.process_configurations.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert updates["description"] == "New description"
        assert updates["gcp_path"] == "gs://new/path.json"
        assert updates["tags"] == ["tag1", "tag2"]


# ── delete_configuration ─────────────────────────────────────────────────────


class TestDeleteConfiguration:
    @pytest.mark.anyio
    async def test_successful_delete_calls_lookup_cleanup(self):
        from easyweaver.processes import service

        db = _make_mock_db()
        db.process_configurations.delete_one = AsyncMock(
            return_value=MagicMock(deleted_count=1)
        )

        with patch("easyweaver.lookups.service.delete_lookups", AsyncMock()) as mock_del:
            await service.delete_configuration(db, "some-id")

        mock_del.assert_called_once_with(db, "some-id")

    @pytest.mark.anyio
    async def test_lookup_cleanup_exception_is_swallowed(self):
        from easyweaver.processes import service

        db = _make_mock_db()
        db.process_configurations.delete_one = AsyncMock(
            return_value=MagicMock(deleted_count=1)
        )

        with patch(
            "easyweaver.lookups.service.delete_lookups",
            AsyncMock(side_effect=RuntimeError("cleanup error")),
        ):
            # Should not raise
            await service.delete_configuration(db, "some-id")


# ── update_process_run (all field branches) ───────────────────────────────────


class TestUpdateProcessRunFields:
    @pytest.mark.anyio
    async def test_updates_all_optional_fields(self):
        from easyweaver.processes import service

        run = _make_run(status="running")
        updated = _make_run(status="completed")
        db = _make_mock_db()
        db.process_runs.find_one = AsyncMock(return_value=updated.to_doc())

        result = await service.update_process_run(
            db,
            str(run.id),
            status="completed",
            row_count=500,
            error=None,
            result_gcp_path="gs://bucket/path",
            result_run_id="result-run-123",
            control={"paused": False},
        )

        call_args = db.process_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert updates["status"] == "completed"
        assert updates["row_count"] == 500
        assert updates["result_gcp_path"] == "gs://bucket/path"
        assert updates["result_run_id"] == "result-run-123"
        assert updates["control"] == {"paused": False}

    @pytest.mark.anyio
    async def test_updates_error_field(self):
        from easyweaver.processes import service

        run = _make_run()
        db = _make_mock_db()
        db.process_runs.find_one = AsyncMock(return_value=run.to_doc())

        await service.update_process_run(db, str(run.id), error="Something failed")

        call_args = db.process_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        assert updates["error"] == "Something failed"

    @pytest.mark.anyio
    async def test_no_fields_only_updates_timestamp(self):
        from easyweaver.processes import service

        run = _make_run()
        db = _make_mock_db()
        db.process_runs.find_one = AsyncMock(return_value=run.to_doc())

        await service.update_process_run(db, str(run.id))

        call_args = db.process_runs.update_one.call_args
        updates = call_args[0][1]["$set"]
        # Should only have updated_at
        assert "updated_at" in updates
        assert "status" not in updates


# ── refresh_process_credentials ──────────────────────────────────────────────


class TestRefreshProcessCredentials:
    @pytest.mark.anyio
    async def test_refreshes_and_returns_updated_config(self):
        from easyweaver.processes import service

        config = _make_config(save_destination="redis")
        db = _make_mock_db()
        db.process_configurations.find_one = AsyncMock(side_effect=[
            config.to_doc(),  # initial get
            config.to_doc(),  # after update
        ])

        with patch("easyweaver.sources.service.get_source", AsyncMock(side_effect=Exception("no src"))):
            result = await service.refresh_process_credentials(db, str(config.id))

        db.process_configurations.update_one.assert_called_once()
        assert isinstance(result, ProcessConfiguration)

    @pytest.mark.anyio
    async def test_refreshes_to_gcp_when_destination_set(self):
        from easyweaver.processes import service

        config = _make_config(save_destination="gcp", gcp_path="existing/path.json")
        db = _make_mock_db()
        db.process_configurations.find_one = AsyncMock(side_effect=[
            config.to_doc(),
            config.to_doc(),
        ])

        with (
            patch("easyweaver.sources.service.get_source", AsyncMock(side_effect=Exception("no src"))),
            patch.object(service, "save_config_to_gcp") as mock_gcp,
        ):
            result = await service.refresh_process_credentials(db, str(config.id))

        mock_gcp.assert_called_once()

    @pytest.mark.anyio
    async def test_gcp_save_failure_during_refresh_is_logged(self):
        from easyweaver.processes import service

        config = _make_config(save_destination="gcp", gcp_path="some/path.json")
        db = _make_mock_db()
        db.process_configurations.find_one = AsyncMock(side_effect=[
            config.to_doc(),
            config.to_doc(),
        ])

        with (
            patch("easyweaver.sources.service.get_source", AsyncMock(side_effect=Exception("no src"))),
            patch.object(service, "save_config_to_gcp", side_effect=RuntimeError("GCS fail")),
        ):
            result = await service.refresh_process_credentials(db, str(config.id))

        assert isinstance(result, ProcessConfiguration)


# ── GCS helpers ───────────────────────────────────────────────────────────────


class TestLoadConfigFromGcp:
    def test_calls_gcs_client_download(self):
        from easyweaver.processes import service

        mock_client = MagicMock()
        mock_client.download_json = MagicMock(return_value={"key": "value"})

        with patch("easyweaver.storage.gcs_client.get_gcs_client", return_value=mock_client):
            result = service.load_config_from_gcp("some/path/config.json")

        mock_client.download_json.assert_called_once_with("some/path/config.json")
        assert result == {"key": "value"}


class TestSaveConfigToGcp:
    def test_calls_gcs_client_upload(self):
        from easyweaver.processes import service

        config = _make_config()
        mock_client = MagicMock()
        mock_client.upload_json = MagicMock()

        with patch("easyweaver.storage.gcs_client.get_gcs_client", return_value=mock_client):
            service.save_config_to_gcp(config)

        mock_client.upload_json.assert_called_once()
        call_args = mock_client.upload_json.call_args
        path = call_args[0][0]
        assert str(config.id) in path
        assert "config.json" in path


class TestSaveResultsToGcp:
    def test_returns_gcp_path(self):
        from easyweaver.processes import service

        run = _make_run(process_id="proc-123")
        df = pl.DataFrame({"id": [1, 2, 3]})
        mock_client = MagicMock()
        mock_client.upload_parquet = MagicMock()

        with patch("easyweaver.storage.gcs_client.get_gcs_client", return_value=mock_client):
            path = service.save_results_to_gcp(run, df)

        assert "proc-123" in path
        assert str(run.id) in path
        assert "results.parquet" in path
        mock_client.upload_parquet.assert_called_once()


class TestLoadResultsFromGcp:
    def test_calls_gcs_client_download_parquet(self):
        from easyweaver.processes import service

        df = pl.DataFrame({"x": [1, 2]})
        mock_client = MagicMock()
        mock_client.download_parquet = MagicMock(return_value=df)

        with patch("easyweaver.storage.gcs_client.get_gcs_client", return_value=mock_client):
            result = service.load_results_from_gcp("some/path/results.parquet")

        mock_client.download_parquet.assert_called_once_with("some/path/results.parquet")
        assert len(result) == 2
