"""Tests for easyweaver.lookups.service."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from easyweaver.lookups.service import (
    _doc_id,
    build_lookups_from_params,
    delete_lookups,
    get_lookups,
    upsert_lookups,
)
from easyweaver.sources.models import DataSource

_FERNET_KEY = Fernet.generate_key().decode()


@pytest.fixture(autouse=True)
def patch_fernet(monkeypatch):
    from easyweaver.core import security
    import easyweaver.settings as settings_module

    monkeypatch.setattr(settings_module.settings, "fernet_key", _FERNET_KEY)
    security._fernet = None
    yield
    security._fernet = None


# ---------------------------------------------------------------------------
# _doc_id
# ---------------------------------------------------------------------------


class TestDocId:
    def test_format_is_process_id_lookups(self):
        pid = "my-process-123"
        assert _doc_id(pid) == "my-process-123_lookups"

    def test_different_process_ids_give_different_doc_ids(self):
        assert _doc_id("a") != _doc_id("b")


# ---------------------------------------------------------------------------
# get_lookups
# ---------------------------------------------------------------------------


class TestGetLookups:
    @pytest.mark.anyio
    async def test_returns_none_when_not_found(self):
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.find_one = AsyncMock(return_value=None)

        result = await get_lookups(mock_db, "proc-1")
        assert result is None

    @pytest.mark.anyio
    async def test_returns_doc_without_id_field(self):
        proc_id = "proc-1"
        doc = {
            "_id": _doc_id(proc_id),
            "process_id": proc_id,
            "lookups": {"param1": ["a", "b"]},
            "references": [],
        }
        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(return_value=doc)
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        result = await get_lookups(mock_db, proc_id)
        assert result is not None
        assert "_id" not in result
        assert result["process_id"] == proc_id

    @pytest.mark.anyio
    async def test_queries_with_correct_doc_id(self):
        proc_id = "proc-abc"
        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        await get_lookups(mock_db, proc_id)
        mock_collection.find_one.assert_called_once_with({"_id": _doc_id(proc_id)})


# ---------------------------------------------------------------------------
# upsert_lookups
# ---------------------------------------------------------------------------


class TestUpsertLookups:
    @pytest.mark.anyio
    async def test_inserts_when_no_existing_doc(self):
        proc_id = "new-proc"
        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_collection.insert_one = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        lookups = {"param1": ["x", "y"]}
        references = [{"source_id": "s1", "table": "t1", "column": "c1"}]

        result = await upsert_lookups(mock_db, proc_id, lookups, references)

        mock_collection.insert_one.assert_called_once()
        assert result["process_id"] == proc_id
        assert result["lookups"] == lookups
        assert "_id" not in result

    @pytest.mark.anyio
    async def test_updates_when_existing_doc(self):
        proc_id = "existing-proc"
        existing_doc = {
            "_id": _doc_id(proc_id),
            "process_id": proc_id,
            "lookups": {"old_param": ["old"]},
            "references": [],
        }
        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(return_value=existing_doc)
        mock_collection.update_one = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        new_lookups = {"param1": ["a", "b"]}
        references = []
        result = await upsert_lookups(mock_db, proc_id, new_lookups, references)

        mock_collection.update_one.assert_called_once()
        # Should not call insert_one
        mock_collection.insert_one.assert_not_called() if hasattr(mock_collection.insert_one, "assert_not_called") else None

    @pytest.mark.anyio
    async def test_inserted_doc_has_created_at(self):
        proc_id = "new-proc-2"
        inserted = []
        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_collection.insert_one = AsyncMock(side_effect=lambda doc: inserted.append(doc))
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        await upsert_lookups(mock_db, proc_id, {}, [])

        assert len(inserted) == 1
        assert "created_at" in inserted[0]

    @pytest.mark.anyio
    async def test_result_excludes_id(self):
        proc_id = "proc-no-id"
        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(return_value=None)
        mock_collection.insert_one = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        result = await upsert_lookups(mock_db, proc_id, {}, [])
        assert "_id" not in result

    @pytest.mark.anyio
    async def test_update_sets_correct_fields(self):
        proc_id = "update-proc"
        existing_doc = {
            "_id": _doc_id(proc_id),
            "process_id": proc_id,
            "lookups": {},
            "references": [],
        }
        update_calls = []
        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(return_value=existing_doc)
        mock_collection.update_one = AsyncMock(side_effect=lambda q, u: update_calls.append((q, u)))
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        new_lookups = {"p": ["v1", "v2"]}
        new_refs = [{"source_id": "s", "table": "t", "column": "c"}]
        await upsert_lookups(mock_db, proc_id, new_lookups, new_refs)

        assert len(update_calls) == 1
        set_op = update_calls[0][1]["$set"]
        assert set_op["lookups"] == new_lookups
        assert set_op["references"] == new_refs
        assert "updated_at" in set_op


# ---------------------------------------------------------------------------
# delete_lookups
# ---------------------------------------------------------------------------


class TestDeleteLookups:
    @pytest.mark.anyio
    async def test_deletes_with_correct_doc_id(self):
        proc_id = "del-proc"
        mock_collection = MagicMock()
        mock_collection.delete_one = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        await delete_lookups(mock_db, proc_id)
        mock_collection.delete_one.assert_called_once_with({"_id": _doc_id(proc_id)})

    @pytest.mark.anyio
    async def test_delete_does_not_raise_when_not_found(self):
        mock_collection = MagicMock()
        mock_collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=0))
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        # Should not raise
        await delete_lookups(mock_db, "nonexistent-proc")


# ---------------------------------------------------------------------------
# build_lookups_from_params
# ---------------------------------------------------------------------------


class TestBuildLookupsFromParams:
    @pytest.mark.anyio
    async def test_empty_params_returns_empty(self):
        mock_db = MagicMock()
        lookups, refs = await build_lookups_from_params(mock_db, {})
        assert lookups == {}
        assert refs == []

    @pytest.mark.anyio
    async def test_non_select_params_are_skipped(self):
        mock_db = MagicMock()
        params = {
            "start_date": {"type": "date", "options_source": {"source_id": "s", "table": "t", "column": "c"}},
            "count": {"type": "number"},
        }
        lookups, refs = await build_lookups_from_params(mock_db, params)
        assert lookups == {}
        assert refs == []

    @pytest.mark.anyio
    async def test_select_without_options_source_is_skipped(self):
        mock_db = MagicMock()
        params = {
            "status": {"type": "select"},  # No options_source
        }
        lookups, refs = await build_lookups_from_params(mock_db, params)
        assert lookups == {}

    @pytest.mark.anyio
    async def test_select_with_missing_required_source_fields_is_skipped(self):
        mock_db = MagicMock()
        params = {
            "status": {
                "type": "select",
                "options_source": {"source_id": "s1"},  # Missing table and column
            }
        }
        lookups, refs = await build_lookups_from_params(mock_db, params)
        assert lookups == {}

    @pytest.mark.anyio
    async def test_fetches_values_for_select_param(self):
        from easyweaver.core.security import encrypt_credentials

        source_id = str(uuid.uuid4())
        fake_source = DataSource(
            id=uuid.UUID(source_id),
            name="Test",
            source_type="postgres",
            encrypted_credentials=encrypt_credentials(
                json.dumps({"type": "postgres", "host": "h", "password": "p"})
            ),
        )
        distinct_result = {"values": ["active", "inactive"], "truncated": False, "total_count": 2}

        mock_connector = MagicMock()
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)
        mock_connector.get_distinct_values = AsyncMock(return_value=distinct_result)

        mock_db = MagicMock()
        params = {
            "status": {
                "type": "select",
                "options_source": {"source_id": source_id, "table": "users", "column": "status"},
            }
        }

        with patch("easyweaver.lookups.service.get_source", new=AsyncMock(return_value=fake_source)), \
             patch("easyweaver.lookups.service.get_source_credentials", return_value={"host": "h"}), \
             patch("easyweaver.lookups.service.get_connector", return_value=mock_connector):
            lookups, refs = await build_lookups_from_params(mock_db, params)

        assert lookups["status"] == ["active", "inactive"]
        assert len(refs) == 1
        assert refs[0]["source_id"] == source_id

    @pytest.mark.anyio
    async def test_multi_select_param_is_included(self):
        from easyweaver.core.security import encrypt_credentials

        source_id = str(uuid.uuid4())
        fake_source = DataSource(
            id=uuid.UUID(source_id),
            name="Test",
            source_type="postgres",
            encrypted_credentials=encrypt_credentials(
                json.dumps({"type": "postgres", "host": "h", "password": "p"})
            ),
        )
        distinct_result = {"values": ["a", "b", "c"], "truncated": False, "total_count": 3}

        mock_connector = MagicMock()
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)
        mock_connector.get_distinct_values = AsyncMock(return_value=distinct_result)

        mock_db = MagicMock()
        params = {
            "categories": {
                "type": "multi_select",
                "options_source": {"source_id": source_id, "table": "items", "column": "category"},
            }
        }

        with patch("easyweaver.lookups.service.get_source", new=AsyncMock(return_value=fake_source)), \
             patch("easyweaver.lookups.service.get_source_credentials", return_value={}), \
             patch("easyweaver.lookups.service.get_connector", return_value=mock_connector):
            lookups, refs = await build_lookups_from_params(mock_db, params)

        assert "categories" in lookups
        assert lookups["categories"] == ["a", "b", "c"]

    @pytest.mark.anyio
    async def test_deduplicates_references(self):
        """Two params pointing to same source/table/column should produce one reference entry."""
        from easyweaver.core.security import encrypt_credentials

        source_id = str(uuid.uuid4())
        fake_source = DataSource(
            id=uuid.UUID(source_id),
            name="Test",
            source_type="postgres",
            encrypted_credentials=encrypt_credentials(
                json.dumps({"type": "postgres", "host": "h", "password": "p"})
            ),
        )
        distinct_result = {"values": ["x"], "truncated": False, "total_count": 1}

        mock_connector = MagicMock()
        mock_connector.__aenter__ = AsyncMock(return_value=mock_connector)
        mock_connector.__aexit__ = AsyncMock(return_value=None)
        mock_connector.get_distinct_values = AsyncMock(return_value=distinct_result)

        mock_db = MagicMock()
        params = {
            "status_a": {
                "type": "select",
                "options_source": {"source_id": source_id, "table": "t", "column": "c"},
            },
            "status_b": {
                "type": "select",
                "options_source": {"source_id": source_id, "table": "t", "column": "c"},
            },
        }

        with patch("easyweaver.lookups.service.get_source", new=AsyncMock(return_value=fake_source)), \
             patch("easyweaver.lookups.service.get_source_credentials", return_value={}), \
             patch("easyweaver.lookups.service.get_connector", return_value=mock_connector):
            lookups, refs = await build_lookups_from_params(mock_db, params)

        # Both params get values, but only one reference entry
        assert "status_a" in lookups
        assert "status_b" in lookups
        assert len(refs) == 1

    @pytest.mark.anyio
    async def test_connector_failure_stores_empty_list(self):
        """If connector fails for a param, its lookup value should be an empty list."""
        source_id = str(uuid.uuid4())
        mock_db = MagicMock()
        params = {
            "bad_param": {
                "type": "select",
                "options_source": {"source_id": source_id, "table": "t", "column": "c"},
            }
        }

        with patch("easyweaver.lookups.service.get_source", new=AsyncMock(side_effect=Exception("Connection failed"))):
            lookups, refs = await build_lookups_from_params(mock_db, params)

        assert "bad_param" in lookups
        assert lookups["bad_param"] == []

    @pytest.mark.anyio
    async def test_uses_max_options_from_param_def(self):
        from easyweaver.core.security import encrypt_credentials

        source_id = str(uuid.uuid4())
        fake_source = DataSource(
            id=uuid.UUID(source_id),
            name="Test",
            source_type="postgres",
            encrypted_credentials=encrypt_credentials(
                json.dumps({"type": "postgres", "host": "h", "password": "p"})
            ),
        )
        captured = {}

        class CapConnector:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get_distinct_values(self, table, column, limit):
                captured["limit"] = limit
                return {"values": [], "truncated": False, "total_count": 0}

        mock_db = MagicMock()
        params = {
            "status": {
                "type": "select",
                "options_source": {"source_id": source_id, "table": "t", "column": "c"},
                "max_options": 100,
            }
        }

        with patch("easyweaver.lookups.service.get_source", new=AsyncMock(return_value=fake_source)), \
             patch("easyweaver.lookups.service.get_source_credentials", return_value={}), \
             patch("easyweaver.lookups.service.get_connector", return_value=CapConnector()):
            await build_lookups_from_params(mock_db, params)

        assert captured.get("limit") == 100
