"""Tests for easyweaver.sources.service."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from easyweaver.core.exceptions import NotFoundError
from easyweaver.sources.models import DataSource
from easyweaver.sources.schemas import (
    PostgresCredentials,
    SourceCreate,
    SourceUpdate,
)
from easyweaver.sources.service import (
    create_source,
    delete_source,
    get_source,
    get_source_credentials,
    list_sources,
    test_source_connection as svc_test_source_connection,
    update_source,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fernet_key() -> str:
    return Fernet.generate_key().decode()


def _make_source(source_id: str | None = None) -> DataSource:
    uid = uuid.UUID(source_id) if source_id else uuid.uuid4()
    from cryptography.fernet import Fernet as _Fernet

    key = _FERNET_KEY
    fernet = _Fernet(key.encode())
    creds_json = json.dumps({"type": "postgres", "host": "localhost", "port": 5432, "database": "testdb", "user": "u", "password": "p", "ssl_mode": "disable"})
    encrypted = fernet.encrypt(creds_json.encode()).decode()
    return DataSource(
        id=uid,
        name="Test Source",
        source_type="postgres",
        encrypted_credentials=encrypted,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_source_doc(source_id: str | None = None) -> dict:
    uid = source_id or str(uuid.uuid4())
    return {
        "_id": uid,
        "name": "Test Source",
        "source_type": "postgres",
        "encrypted_credentials": "encrypted_placeholder",
        "metadata": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def _postgres_creds() -> PostgresCredentials:
    return PostgresCredentials(
        host="localhost",
        port=5432,
        database="testdb",
        user="testuser",
        password="testpass",
    )


_FERNET_KEY = Fernet.generate_key().decode()


@pytest.fixture(autouse=True)
def patch_fernet(monkeypatch):
    """Patch the fernet key so encryption/decryption works in tests."""
    from easyweaver.core import security
    import easyweaver.settings as settings_module

    monkeypatch.setattr(settings_module.settings, "fernet_key", _FERNET_KEY)
    # Reset cached fernet so it picks up the new key
    security._fernet = None
    yield
    security._fernet = None


# ---------------------------------------------------------------------------
# list_sources
# ---------------------------------------------------------------------------


class TestListSources:
    @pytest.mark.anyio
    async def test_returns_empty_list_when_no_sources(self):
        mock_db = MagicMock()

        async def empty_async_iter():
            return
            yield  # make it an async generator

        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.__aiter__ = MagicMock(return_value=empty_async_iter())
        mock_db.data_sources.find = MagicMock(return_value=mock_cursor)

        result = await list_sources(mock_db)
        assert result == []

    @pytest.mark.anyio
    async def test_returns_list_of_data_sources(self):
        docs = [_make_source_doc(), _make_source_doc()]
        mock_db = MagicMock()

        async def doc_iter():
            for d in docs:
                yield d

        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.__aiter__ = MagicMock(return_value=doc_iter())
        mock_db.data_sources.find = MagicMock(return_value=mock_cursor)

        result = await list_sources(mock_db)
        assert len(result) == 2
        assert all(isinstance(s, DataSource) for s in result)

    @pytest.mark.anyio
    async def test_sorts_by_created_at_desc(self):
        mock_db = MagicMock()

        async def empty_async_iter():
            return
            yield

        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.__aiter__ = MagicMock(return_value=empty_async_iter())
        mock_db.data_sources.find = MagicMock(return_value=mock_cursor)

        await list_sources(mock_db)
        mock_cursor.sort.assert_called_once_with("created_at", -1)


# ---------------------------------------------------------------------------
# get_source
# ---------------------------------------------------------------------------


class TestGetSource:
    @pytest.mark.anyio
    async def test_returns_data_source_when_found(self):
        sid = str(uuid.uuid4())
        doc = _make_source_doc(sid)
        mock_db = MagicMock()
        mock_db.data_sources.find_one = AsyncMock(return_value=doc)

        result = await get_source(mock_db, sid)
        assert isinstance(result, DataSource)
        assert str(result.id) == sid

    @pytest.mark.anyio
    async def test_raises_not_found_when_missing(self):
        mock_db = MagicMock()
        mock_db.data_sources.find_one = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await get_source(mock_db, str(uuid.uuid4()))

    @pytest.mark.anyio
    async def test_accepts_uuid_object(self):
        uid = uuid.uuid4()
        doc = _make_source_doc(str(uid))
        mock_db = MagicMock()
        mock_db.data_sources.find_one = AsyncMock(return_value=doc)

        result = await get_source(mock_db, uid)
        assert result.id == uid

    @pytest.mark.anyio
    async def test_queries_with_string_id(self):
        sid = str(uuid.uuid4())
        doc = _make_source_doc(sid)
        mock_db = MagicMock()
        mock_db.data_sources.find_one = AsyncMock(return_value=doc)

        await get_source(mock_db, sid)
        mock_db.data_sources.find_one.assert_called_once_with({"_id": sid})


# ---------------------------------------------------------------------------
# create_source
# ---------------------------------------------------------------------------


class TestCreateSource:
    @pytest.mark.anyio
    async def test_creates_source_and_returns_it(self):
        mock_db = MagicMock()
        mock_db.data_sources.insert_one = AsyncMock()

        data = SourceCreate(
            name="My PG Source",
            source_type="postgres",
            credentials=_postgres_creds(),
        )
        result = await create_source(mock_db, data)

        assert isinstance(result, DataSource)
        assert result.name == "My PG Source"
        assert result.source_type == "postgres"
        mock_db.data_sources.insert_one.assert_called_once()

    @pytest.mark.anyio
    async def test_credentials_are_encrypted(self):
        mock_db = MagicMock()
        inserted_docs = []
        mock_db.data_sources.insert_one = AsyncMock(side_effect=lambda doc: inserted_docs.append(doc))

        data = SourceCreate(
            name="My PG Source",
            source_type="postgres",
            credentials=_postgres_creds(),
        )
        await create_source(mock_db, data)

        assert len(inserted_docs) == 1
        encrypted = inserted_docs[0]["encrypted_credentials"]
        # Should be encrypted — not the raw JSON
        assert "testpass" not in encrypted
        # Should be decryptable
        from easyweaver.core.security import decrypt_credentials
        decrypted = decrypt_credentials(encrypted)
        creds_dict = json.loads(decrypted)
        assert creds_dict["password"] == "testpass"

    @pytest.mark.anyio
    async def test_assigns_new_uuid(self):
        mock_db = MagicMock()
        mock_db.data_sources.insert_one = AsyncMock()

        data = SourceCreate(
            name="Source 1",
            source_type="postgres",
            credentials=_postgres_creds(),
        )
        s1 = await create_source(mock_db, data)
        s2 = await create_source(mock_db, data)
        assert s1.id != s2.id


# ---------------------------------------------------------------------------
# update_source
# ---------------------------------------------------------------------------


class TestUpdateSource:
    @pytest.mark.anyio
    async def test_updates_name(self):
        sid = uuid.uuid4()
        doc = _make_source_doc(str(sid))
        mock_db = MagicMock()
        mock_db.data_sources.find_one = AsyncMock(return_value=doc)
        mock_db.data_sources.update_one = AsyncMock()

        update_data = SourceUpdate(name="New Name")
        result = await update_source(mock_db, sid, update_data)

        mock_db.data_sources.update_one.assert_called_once()
        call_args = mock_db.data_sources.update_one.call_args
        assert call_args[0][1]["$set"]["name"] == "New Name"

    @pytest.mark.anyio
    async def test_raises_not_found_for_missing_source(self):
        mock_db = MagicMock()
        mock_db.data_sources.find_one = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await update_source(mock_db, uuid.uuid4(), SourceUpdate(name="New"))

    @pytest.mark.anyio
    async def test_updates_credentials_when_provided(self):
        sid = uuid.uuid4()
        doc = _make_source_doc(str(sid))
        mock_db = MagicMock()
        mock_db.data_sources.find_one = AsyncMock(return_value=doc)
        mock_db.data_sources.update_one = AsyncMock()

        new_creds = _postgres_creds()
        new_creds.password = "newpass"
        update_data = SourceUpdate(credentials=new_creds)
        await update_source(mock_db, sid, update_data)

        call_args = mock_db.data_sources.update_one.call_args
        assert "encrypted_credentials" in call_args[0][1]["$set"]

    @pytest.mark.anyio
    async def test_no_name_update_when_none(self):
        sid = uuid.uuid4()
        doc = _make_source_doc(str(sid))
        mock_db = MagicMock()
        mock_db.data_sources.find_one = AsyncMock(return_value=doc)
        mock_db.data_sources.update_one = AsyncMock()

        update_data = SourceUpdate()  # both None
        await update_source(mock_db, sid, update_data)

        call_args = mock_db.data_sources.update_one.call_args
        assert "name" not in call_args[0][1]["$set"]


# ---------------------------------------------------------------------------
# delete_source
# ---------------------------------------------------------------------------


class TestDeleteSource:
    @pytest.mark.anyio
    async def test_deletes_existing_source(self):
        sid = uuid.uuid4()
        mock_result = MagicMock()
        mock_result.deleted_count = 1
        mock_db = MagicMock()
        mock_db.data_sources.delete_one = AsyncMock(return_value=mock_result)

        await delete_source(mock_db, sid)
        mock_db.data_sources.delete_one.assert_called_once_with({"_id": str(sid)})

    @pytest.mark.anyio
    async def test_raises_not_found_when_nothing_deleted(self):
        mock_result = MagicMock()
        mock_result.deleted_count = 0
        mock_db = MagicMock()
        mock_db.data_sources.delete_one = AsyncMock(return_value=mock_result)

        with pytest.raises(NotFoundError):
            await delete_source(mock_db, uuid.uuid4())


# ---------------------------------------------------------------------------
# get_source_credentials
# ---------------------------------------------------------------------------


class TestGetSourceCredentials:
    def test_decrypts_credentials(self):
        from easyweaver.core.security import encrypt_credentials

        creds = {"type": "postgres", "host": "db.example.com", "password": "secret"}
        encrypted = encrypt_credentials(json.dumps(creds))

        source = DataSource(
            id=uuid.uuid4(),
            name="Test",
            source_type="postgres",
            encrypted_credentials=encrypted,
        )
        result = get_source_credentials(source)
        assert result["password"] == "secret"
        assert result["host"] == "db.example.com"

    def test_returns_dict(self):
        from easyweaver.core.security import encrypt_credentials

        creds = {"type": "postgres", "host": "h", "password": "p"}
        encrypted = encrypt_credentials(json.dumps(creds))
        source = DataSource(
            id=uuid.uuid4(),
            name="Test",
            source_type="postgres",
            encrypted_credentials=encrypted,
        )
        result = get_source_credentials(source)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# test_source_connection
# ---------------------------------------------------------------------------


class TestTestSourceConnection:
    @pytest.mark.anyio
    async def test_calls_connector_test_connection(self):
        sid = uuid.uuid4()
        doc = _make_source_doc(str(sid))

        from easyweaver.core.security import encrypt_credentials
        creds_json = json.dumps({"type": "postgres", "host": "h", "port": 5432, "database": "db", "user": "u", "password": "p", "ssl_mode": "disable"})
        doc["encrypted_credentials"] = encrypt_credentials(creds_json)

        mock_db = MagicMock()
        mock_db.data_sources.find_one = AsyncMock(return_value=doc)

        mock_connector = AsyncMock()
        mock_connector.test_connection = AsyncMock(return_value={"success": True, "latency_ms": 5.0, "message": "ok"})

        with patch("easyweaver.sources.service.get_connector", return_value=mock_connector):
            result = await svc_test_source_connection(mock_db, sid)

        assert result["success"] is True
        mock_connector.test_connection.assert_called_once()
