"""Tests for easyweaver.auth.router using TestClient."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyweaver.auth.dependencies import get_current_user
from easyweaver.auth.models import User
from easyweaver.auth.router import router
from easyweaver.auth.service import hash_password
from easyweaver.core.exceptions import AuthenticationError, ValidationError
from easyweaver.core.middleware import setup_middleware
from easyweaver.dependencies import get_db


def _make_app() -> FastAPI:
    app = FastAPI()
    setup_middleware(app)
    app.include_router(router, prefix="/auth")
    return app


def _make_user(
    email: str = "test@example.com",
    is_active: bool = True,
    user_id: str | None = None,
) -> User:
    return User(
        id=uuid.UUID(user_id) if user_id else uuid.uuid4(),
        email=email,
        hashed_password=hash_password("TestPassword123!"),
        display_name="Test User",
        role="admin",
        is_active=is_active,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def app(mock_db):
    _app = _make_app()
    _app.dependency_overrides[get_db] = lambda: mock_db
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------


class TestRegisterEndpoint:
    def test_register_success_returns_201(self, client, mock_db):
        user = _make_user(email="new@example.com")
        with patch("easyweaver.auth.service.register_user", new=AsyncMock(return_value=user)):
            resp = client.post(
                "/auth/register",
                json={"email": "new@example.com", "password": "Pass123!", "display_name": "New User"},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == "new@example.com"
        assert "id" in data
        assert "hashed_password" not in data

    def test_register_duplicate_email_returns_422(self, client, mock_db):
        with patch(
            "easyweaver.auth.service.register_user",
            new=AsyncMock(side_effect=ValidationError("Email already registered")),
        ):
            resp = client.post(
                "/auth/register",
                json={"email": "existing@example.com", "password": "Pass123!", "display_name": "User"},
            )
        assert resp.status_code == 422

    def test_register_missing_fields_returns_422(self, client):
        resp = client.post("/auth/register", json={"email": "bad@example.com"})
        assert resp.status_code == 422

    def test_register_invalid_email_returns_422(self, client):
        resp = client.post(
            "/auth/register",
            json={"email": "not-an-email", "password": "Pass123!", "display_name": "User"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------


class TestLoginEndpoint:
    def test_login_success_returns_tokens(self, client, mock_db):
        user = _make_user()
        with patch("easyweaver.auth.service.authenticate_user", new=AsyncMock(return_value=user)):
            resp = client.post(
                "/auth/login",
                json={"email": "test@example.com", "password": "TestPassword123!"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials_returns_401(self, client, mock_db):
        with patch(
            "easyweaver.auth.service.authenticate_user",
            new=AsyncMock(side_effect=AuthenticationError("Invalid email or password")),
        ):
            resp = client.post(
                "/auth/login",
                json={"email": "test@example.com", "password": "WrongPassword!"},
            )
        assert resp.status_code == 401

    def test_login_missing_fields_returns_422(self, client):
        resp = client.post("/auth/login", json={"email": "test@example.com"})
        assert resp.status_code == 422

    def test_login_returns_tokens_with_string_type(self, client, mock_db):
        user = _make_user()
        with patch("easyweaver.auth.service.authenticate_user", new=AsyncMock(return_value=user)):
            resp = client.post(
                "/auth/login",
                json={"email": "test@example.com", "password": "TestPassword123!"},
            )
        data = resp.json()
        assert isinstance(data["access_token"], str)
        assert isinstance(data["refresh_token"], str)


# ---------------------------------------------------------------------------
# POST /auth/refresh
# ---------------------------------------------------------------------------


class TestRefreshEndpoint:
    def test_refresh_with_valid_refresh_token(self, client, mock_db):
        from easyweaver.auth.service import create_refresh_token

        user = _make_user()
        refresh_tok = create_refresh_token(str(user.id))

        with patch("easyweaver.auth.service.get_user_by_id", new=AsyncMock(return_value=user)):
            resp = client.post("/auth/refresh", json={"refresh_token": refresh_tok})

        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_with_access_token_returns_401(self, client, mock_db):
        from easyweaver.auth.service import create_access_token

        user = _make_user()
        access_tok = create_access_token(str(user.id))

        resp = client.post("/auth/refresh", json={"refresh_token": access_tok})
        assert resp.status_code == 401

    def test_refresh_with_invalid_token_returns_401(self, client):
        resp = client.post("/auth/refresh", json={"refresh_token": "not.a.valid.token"})
        assert resp.status_code == 401

    def test_refresh_user_not_found_returns_401(self, client, mock_db):
        from easyweaver.auth.service import create_refresh_token

        uid = str(uuid.uuid4())
        refresh_tok = create_refresh_token(uid)

        with patch("easyweaver.auth.service.get_user_by_id", new=AsyncMock(return_value=None)):
            resp = client.post("/auth/refresh", json={"refresh_token": refresh_tok})

        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------


class TestMeEndpoint:
    def test_me_returns_current_user(self, app, mock_db):
        user = _make_user()
        app.dependency_overrides[get_current_user] = lambda: user
        client = TestClient(app)

        resp = client.get("/auth/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == user.email
        assert data["role"] == user.role
        assert "hashed_password" not in data

    def test_me_without_auth_returns_401(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_me_response_excludes_sensitive_fields(self, app, mock_db):
        user = _make_user()
        app.dependency_overrides[get_current_user] = lambda: user
        client = TestClient(app)

        resp = client.get("/auth/me")
        data = resp.json()
        assert "hashed_password" not in data
        assert "id" in data
        assert "email" in data
        assert "display_name" in data
        assert "role" in data
        assert "is_active" in data
