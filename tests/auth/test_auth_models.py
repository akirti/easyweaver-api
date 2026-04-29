"""Tests for easyweaver.auth.models.User."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from easyweaver.auth.models import User


def _make_user_doc(user_id: str | None = None) -> dict:
    uid = user_id or str(uuid.uuid4())
    return {
        "_id": uid,
        "email": "user@example.com",
        "hashed_password": "$2b$12$hashedpassword",
        "display_name": "Test User",
        "role": "admin",
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


class TestUserFromDoc:
    def test_from_doc_creates_user(self):
        doc = _make_user_doc()
        user = User.from_doc(doc)
        assert isinstance(user, User)
        assert user.email == "user@example.com"

    def test_from_doc_parses_uuid(self):
        uid = str(uuid.uuid4())
        doc = _make_user_doc(uid)
        user = User.from_doc(doc)
        assert user.id == uuid.UUID(uid)

    def test_from_doc_sets_role(self):
        doc = _make_user_doc()
        doc["role"] = "viewer"
        user = User.from_doc(doc)
        assert user.role == "viewer"

    def test_from_doc_defaults_role_to_admin(self):
        doc = _make_user_doc()
        del doc["role"]
        user = User.from_doc(doc)
        assert user.role == "admin"

    def test_from_doc_sets_is_active(self):
        doc = _make_user_doc()
        doc["is_active"] = False
        user = User.from_doc(doc)
        assert user.is_active is False

    def test_from_doc_defaults_is_active_to_true(self):
        doc = _make_user_doc()
        del doc["is_active"]
        user = User.from_doc(doc)
        assert user.is_active is True

    def test_from_doc_sets_timestamps(self):
        now = datetime.now(timezone.utc)
        doc = _make_user_doc()
        doc["created_at"] = now
        doc["updated_at"] = now
        user = User.from_doc(doc)
        assert user.created_at == now
        assert user.updated_at == now


class TestUserToDoc:
    def test_to_doc_returns_dict(self):
        user = User(
            id=uuid.uuid4(),
            email="user@example.com",
            hashed_password="$2b$12$hashed",
            display_name="Test User",
        )
        doc = user.to_doc()
        assert isinstance(doc, dict)

    def test_to_doc_uses_string_id(self):
        uid = uuid.uuid4()
        user = User(
            id=uid,
            email="user@example.com",
            hashed_password="$2b$12$hashed",
            display_name="Test User",
        )
        doc = user.to_doc()
        assert doc["_id"] == str(uid)

    def test_to_doc_includes_all_fields(self):
        user = User(
            id=uuid.uuid4(),
            email="user@example.com",
            hashed_password="$2b$12$hashed",
            display_name="Test User",
            role="viewer",
            is_active=True,
        )
        doc = user.to_doc()
        assert "email" in doc
        assert "hashed_password" in doc
        assert "display_name" in doc
        assert "role" in doc
        assert "is_active" in doc
        assert "created_at" in doc
        assert "updated_at" in doc

    def test_to_doc_round_trip(self):
        uid = uuid.uuid4()
        user = User(
            id=uid,
            email="roundtrip@example.com",
            hashed_password="$2b$12$hashed",
            display_name="Round Trip",
            role="admin",
            is_active=True,
        )
        doc = user.to_doc()
        restored = User.from_doc(doc)
        assert restored.id == uid
        assert restored.email == user.email
        assert restored.display_name == user.display_name
        assert restored.role == user.role
        assert restored.is_active == user.is_active

    def test_to_doc_default_role_is_admin(self):
        user = User(
            id=uuid.uuid4(),
            email="user@example.com",
            hashed_password="$2b$12$hashed",
            display_name="Test User",
        )
        doc = user.to_doc()
        assert doc["role"] == "admin"


class TestUserDefaults:
    def test_default_role_is_admin(self):
        user = User(
            id=uuid.uuid4(),
            email="user@example.com",
            hashed_password="$2b$12$hashed",
            display_name="Test User",
        )
        assert user.role == "admin"

    def test_default_is_active_is_true(self):
        user = User(
            id=uuid.uuid4(),
            email="user@example.com",
            hashed_password="$2b$12$hashed",
            display_name="Test User",
        )
        assert user.is_active is True

    def test_created_at_defaults_to_now(self):
        before = datetime.now(timezone.utc)
        user = User(
            id=uuid.uuid4(),
            email="user@example.com",
            hashed_password="$2b$12$hashed",
            display_name="Test User",
        )
        after = datetime.now(timezone.utc)
        assert before <= user.created_at <= after
