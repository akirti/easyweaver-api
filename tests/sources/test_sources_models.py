"""Tests for easyweaver.sources.models.DataSource."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from easyweaver.sources.models import DataSource


def _make_source_doc(source_id: str | None = None) -> dict:
    uid = source_id or str(uuid.uuid4())
    return {
        "_id": uid,
        "name": "Test Source",
        "source_type": "postgres",
        "encrypted_credentials": "encrypted_creds",
        "metadata": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


class TestDataSourceFromDoc:
    def test_from_doc_creates_data_source(self):
        doc = _make_source_doc()
        source = DataSource.from_doc(doc)
        assert isinstance(source, DataSource)

    def test_from_doc_parses_uuid(self):
        uid = str(uuid.uuid4())
        doc = _make_source_doc(uid)
        source = DataSource.from_doc(doc)
        assert source.id == uuid.UUID(uid)

    def test_from_doc_sets_name(self):
        doc = _make_source_doc()
        doc["name"] = "My Custom Source"
        source = DataSource.from_doc(doc)
        assert source.name == "My Custom Source"

    def test_from_doc_sets_source_type(self):
        doc = _make_source_doc()
        doc["source_type"] = "mongodb"
        source = DataSource.from_doc(doc)
        assert source.source_type == "mongodb"

    def test_from_doc_sets_encrypted_credentials(self):
        doc = _make_source_doc()
        doc["encrypted_credentials"] = "some_encrypted_value"
        source = DataSource.from_doc(doc)
        assert source.encrypted_credentials == "some_encrypted_value"

    def test_from_doc_reads_metadata(self):
        doc = _make_source_doc()
        doc["metadata"] = "extra info"
        source = DataSource.from_doc(doc)
        assert source.metadata_ == "extra info"

    def test_from_doc_defaults_metadata_to_none(self):
        doc = _make_source_doc()
        del doc["metadata"]
        source = DataSource.from_doc(doc)
        assert source.metadata_ is None

    def test_from_doc_sets_timestamps(self):
        now = datetime.now(timezone.utc)
        doc = _make_source_doc()
        doc["created_at"] = now
        doc["updated_at"] = now
        source = DataSource.from_doc(doc)
        assert source.created_at == now
        assert source.updated_at == now


class TestDataSourceToDoc:
    def test_to_doc_returns_dict(self):
        source = DataSource(
            id=uuid.uuid4(),
            name="Test",
            source_type="postgres",
            encrypted_credentials="enc",
        )
        doc = source.to_doc()
        assert isinstance(doc, dict)

    def test_to_doc_uses_string_id(self):
        uid = uuid.uuid4()
        source = DataSource(
            id=uid,
            name="Test",
            source_type="postgres",
            encrypted_credentials="enc",
        )
        doc = source.to_doc()
        assert doc["_id"] == str(uid)

    def test_to_doc_includes_required_fields(self):
        source = DataSource(
            id=uuid.uuid4(),
            name="Test",
            source_type="postgres",
            encrypted_credentials="enc",
        )
        doc = source.to_doc()
        assert "name" in doc
        assert "source_type" in doc
        assert "encrypted_credentials" in doc
        assert "metadata" in doc
        assert "created_at" in doc
        assert "updated_at" in doc

    def test_to_doc_metadata_key_is_metadata(self):
        source = DataSource(
            id=uuid.uuid4(),
            name="Test",
            source_type="postgres",
            encrypted_credentials="enc",
            metadata_="custom meta",
        )
        doc = source.to_doc()
        assert doc["metadata"] == "custom meta"

    def test_to_doc_round_trip(self):
        uid = uuid.uuid4()
        source = DataSource(
            id=uid,
            name="Round Trip",
            source_type="mysql",
            encrypted_credentials="encrypted_data",
            metadata_="info",
        )
        doc = source.to_doc()
        restored = DataSource.from_doc(doc)
        assert restored.id == uid
        assert restored.name == source.name
        assert restored.source_type == source.source_type
        assert restored.encrypted_credentials == source.encrypted_credentials
        assert restored.metadata_ == source.metadata_
