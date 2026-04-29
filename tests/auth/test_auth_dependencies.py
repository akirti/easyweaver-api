"""Tests for easyweaver.auth.dependencies."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jose import jwt

from easyweaver.auth.dependencies import get_current_user
from easyweaver.auth.models import User
from easyweaver.auth.service import create_access_token, create_refresh_token, hash_password
from easyweaver.core.exceptions import AuthenticationError
from easyweaver.settings import settings


def _make_mock_credentials(token: str):
    creds = MagicMock()
    creds.credentials = token
    return creds


def _make_user(user_id: str | None = None, is_active: bool = True) -> User:
    uid = uuid.UUID(user_id) if user_id else uuid.uuid4()
    return User(
        id=uid,
        email="dep@example.com",
        hashed_password=hash_password("Password123!"),
        display_name="Dep User",
        role="admin",
        is_active=is_active,
    )


class TestGetCurrentUser:
    @pytest.mark.anyio
    async def test_raises_when_no_credentials(self):
        mock_db = MagicMock()
        with pytest.raises(AuthenticationError, match="Not authenticated"):
            await get_current_user(credentials=None, db=mock_db)

    @pytest.mark.anyio
    async def test_valid_easyweaver_token_returns_user(self):
        user = _make_user()
        token = create_access_token(str(user.id))
        creds = _make_mock_credentials(token)
        mock_db = MagicMock()

        with patch("easyweaver.auth.dependencies.get_user_by_id", new=AsyncMock(return_value=user)):
            result = await get_current_user(credentials=creds, db=mock_db)

        assert result.email == user.email

    @pytest.mark.anyio
    async def test_refresh_token_raises_invalid_token_type(self):
        user = _make_user()
        refresh_token = create_refresh_token(str(user.id))
        creds = _make_mock_credentials(refresh_token)
        mock_db = MagicMock()

        with pytest.raises(AuthenticationError, match="Invalid token type"):
            await get_current_user(credentials=creds, db=mock_db)

    @pytest.mark.anyio
    async def test_invalid_token_raises_authentication_error(self):
        creds = _make_mock_credentials("invalid.token.value")
        mock_db = MagicMock()

        with pytest.raises(AuthenticationError):
            await get_current_user(credentials=creds, db=mock_db)

    @pytest.mark.anyio
    async def test_user_not_found_raises_authentication_error(self):
        token = create_access_token(str(uuid.uuid4()))
        creds = _make_mock_credentials(token)
        mock_db = MagicMock()

        with patch("easyweaver.auth.dependencies.get_user_by_id", new=AsyncMock(return_value=None)):
            with pytest.raises(AuthenticationError, match="User not found or inactive"):
                await get_current_user(credentials=creds, db=mock_db)

    @pytest.mark.anyio
    async def test_inactive_user_raises_authentication_error(self):
        user = _make_user(is_active=False)
        token = create_access_token(str(user.id))
        creds = _make_mock_credentials(token)
        mock_db = MagicMock()

        with patch("easyweaver.auth.dependencies.get_user_by_id", new=AsyncMock(return_value=user)):
            with pytest.raises(AuthenticationError, match="User not found or inactive"):
                await get_current_user(credentials=creds, db=mock_db)

    @pytest.mark.anyio
    async def test_admin_panel_token_builds_user_without_db(self):
        """Admin panel token: user is built from claims, no DB lookup needed."""
        admin_secret = "test-admin-secret"
        expire = datetime.now(timezone.utc) + timedelta(hours=1)
        raw_id = "507f1f77bcf86cd799439011"  # MongoDB-style ObjectId string
        token = jwt.encode(
            {
                "user_id": raw_id,
                "email": "admin@easylife.com",
                "type": "access",
                "exp": expire,
            },
            admin_secret,
            algorithm="HS256",
        )
        creds = _make_mock_credentials(token)
        mock_db = MagicMock()

        with patch.object(settings, "admin_jwt_secret_key", admin_secret), \
             patch.object(settings, "admin_jwt_audience", ""), \
             patch.object(settings, "admin_jwt_issuer", ""):
            result = await get_current_user(credentials=creds, db=mock_db)

        assert result.email == "admin@easylife.com"
        assert result.role == "admin"
        assert result.is_active is True

    @pytest.mark.anyio
    async def test_admin_panel_token_uuid_raw_id_preserved(self):
        """Admin panel token: UUID raw_id is used directly without transformation."""
        admin_secret = "test-admin-secret"
        expire = datetime.now(timezone.utc) + timedelta(hours=1)
        raw_uuid = str(uuid.uuid4())
        token = jwt.encode(
            {
                "user_id": raw_uuid,
                "email": "adminuuid@easylife.com",
                "type": "access",
                "exp": expire,
            },
            admin_secret,
            algorithm="HS256",
        )
        creds = _make_mock_credentials(token)
        mock_db = MagicMock()

        with patch.object(settings, "admin_jwt_secret_key", admin_secret), \
             patch.object(settings, "admin_jwt_audience", ""), \
             patch.object(settings, "admin_jwt_issuer", ""):
            result = await get_current_user(credentials=creds, db=mock_db)

        assert result.id == uuid.UUID(raw_uuid)

    @pytest.mark.anyio
    async def test_admin_panel_non_access_token_raises_error(self):
        """Admin panel token with type != access should raise AuthenticationError."""
        admin_secret = "test-admin-secret"
        expire = datetime.now(timezone.utc) + timedelta(hours=1)
        token = jwt.encode(
            {
                "user_id": "some-id",
                "email": "admin@easylife.com",
                "type": "refresh",  # Wrong type
                "exp": expire,
            },
            admin_secret,
            algorithm="HS256",
        )
        creds = _make_mock_credentials(token)
        mock_db = MagicMock()

        with patch.object(settings, "admin_jwt_secret_key", admin_secret), \
             patch.object(settings, "admin_jwt_audience", ""), \
             patch.object(settings, "admin_jwt_issuer", ""):
            with pytest.raises(AuthenticationError, match="Invalid token type"):
                await get_current_user(credentials=creds, db=mock_db)

    @pytest.mark.anyio
    async def test_token_missing_sub_claim_raises_error(self):
        """Easyweaver token without sub claim should raise AuthenticationError."""
        # Manually craft a token without 'sub'
        expire = datetime.now(timezone.utc) + timedelta(hours=1)
        token = jwt.encode(
            {"type": "access", "exp": expire},
            settings.jwt_secret_key,
            algorithm="HS256",
        )
        creds = _make_mock_credentials(token)
        mock_db = MagicMock()

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await get_current_user(credentials=creds, db=mock_db)
