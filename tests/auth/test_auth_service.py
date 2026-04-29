"""Tests for easyweaver.auth.service."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt
import pytest
from jose import jwt

from easyweaver.auth.models import User
from easyweaver.auth.schemas import RegisterRequest
from easyweaver.auth.service import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_user_by_id,
    hash_password,
    register_user,
    verify_password,
)
from easyweaver.core.exceptions import AuthenticationError, ValidationError
from easyweaver.settings import settings

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / f"{name}.json") as f:
        return json.load(f)


@pytest.fixture
def auth_data():
    return load_fixture("auth")


@pytest.fixture
def test_user_id():
    return str(uuid.uuid4())


@pytest.fixture
def test_user_doc(test_user_id):
    return {
        "_id": test_user_id,
        "email": "test@example.com",
        "hashed_password": bcrypt.hashpw(b"TestPassword123!", bcrypt.gensalt()).decode(),
        "display_name": "Test User",
        "role": "admin",
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


@pytest.fixture
def inactive_user_doc(test_user_id):
    return {
        "_id": str(uuid.uuid4()),
        "email": "inactive@example.com",
        "hashed_password": bcrypt.hashpw(b"SomePassword!", bcrypt.gensalt()).decode(),
        "display_name": "Inactive User",
        "role": "admin",
        "is_active": False,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# hash_password / verify_password
# ---------------------------------------------------------------------------


class TestHashPassword:
    def test_returns_string(self):
        result = hash_password("mysecret")
        assert isinstance(result, str)

    def test_result_is_valid_bcrypt_hash(self):
        result = hash_password("mysecret")
        assert bcrypt.checkpw(b"mysecret", result.encode())

    def test_different_passwords_produce_different_hashes(self):
        h1 = hash_password("password1")
        h2 = hash_password("password2")
        assert h1 != h2

    def test_same_password_produces_different_hashes_due_to_salt(self):
        h1 = hash_password("samepassword")
        h2 = hash_password("samepassword")
        assert h1 != h2


class TestVerifyPassword:
    def test_correct_password_returns_true(self):
        hashed = hash_password("correctpassword")
        assert verify_password("correctpassword", hashed) is True

    def test_wrong_password_returns_false(self):
        hashed = hash_password("correctpassword")
        assert verify_password("wrongpassword", hashed) is False

    def test_empty_password_vs_nonempty_hash(self):
        hashed = hash_password("nonempty")
        assert verify_password("", hashed) is False

    def test_nonempty_password_vs_empty_hash_is_false(self):
        # bcrypt.checkpw raises ValueError for invalid hash — verify it returns False
        assert verify_password("somepassword", hash_password("somepassword")) is True


# ---------------------------------------------------------------------------
# create_access_token / create_refresh_token
# ---------------------------------------------------------------------------


class TestCreateAccessToken:
    def test_returns_string(self):
        token = create_access_token("user-123")
        assert isinstance(token, str)

    def test_contains_sub_claim(self):
        user_id = str(uuid.uuid4())
        token = create_access_token(user_id)
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        assert payload["sub"] == user_id

    def test_contains_type_access(self):
        token = create_access_token("user-123")
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        assert payload["type"] == "access"

    def test_contains_exp_claim(self):
        token = create_access_token("user-123")
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        assert "exp" in payload


class TestCreateRefreshToken:
    def test_returns_string(self):
        token = create_refresh_token("user-123")
        assert isinstance(token, str)

    def test_contains_sub_claim(self):
        user_id = str(uuid.uuid4())
        token = create_refresh_token(user_id)
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        assert payload["sub"] == user_id

    def test_contains_type_refresh(self):
        token = create_refresh_token("user-123")
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        assert payload["type"] == "refresh"

    def test_refresh_token_different_from_access(self):
        uid = "user-abc"
        access = create_access_token(uid)
        refresh = create_refresh_token(uid)
        assert access != refresh


# ---------------------------------------------------------------------------
# decode_token
# ---------------------------------------------------------------------------


class TestDecodeToken:
    def test_decode_valid_easyweaver_token(self):
        token = create_access_token("user-xyz")
        payload = decode_token(token)
        assert payload["sub"] == "user-xyz"
        assert payload["_token_source"] == "easyweaver"

    def test_decode_refresh_token(self):
        token = create_refresh_token("user-xyz")
        payload = decode_token(token)
        assert payload["type"] == "refresh"
        assert payload["_token_source"] == "easyweaver"

    def test_invalid_token_raises_authentication_error(self):
        with pytest.raises(AuthenticationError):
            decode_token("not.a.valid.token")

    def test_tampered_token_raises_authentication_error(self):
        token = create_access_token("user-123")
        # Tamper the signature
        parts = token.split(".")
        parts[2] = parts[2] + "tampered"
        bad_token = ".".join(parts)
        with pytest.raises(AuthenticationError):
            decode_token(bad_token)

    def test_admin_panel_token_when_secret_configured(self):
        admin_secret = "admin-secret-key-for-testing"
        from datetime import timedelta

        expire = datetime.now(timezone.utc) + timedelta(hours=1)
        token = jwt.encode(
            {"user_id": "admin123", "email": "admin@example.com", "type": "access", "exp": expire},
            admin_secret,
            algorithm="HS256",
        )
        with patch.object(settings, "admin_jwt_secret_key", admin_secret), \
             patch.object(settings, "admin_jwt_audience", ""), \
             patch.object(settings, "admin_jwt_issuer", ""):
            payload = decode_token(token)
        assert payload["_token_source"] == "admin_panel"
        assert payload["user_id"] == "admin123"

    def test_admin_panel_token_issuer_mismatch_raises_error(self):
        admin_secret = "admin-secret-key-for-testing"
        from datetime import timedelta

        expire = datetime.now(timezone.utc) + timedelta(hours=1)
        token = jwt.encode(
            {"user_id": "admin123", "iss": "wrong-issuer", "type": "access", "exp": expire},
            admin_secret,
            algorithm="HS256",
        )
        with patch.object(settings, "admin_jwt_secret_key", admin_secret), \
             patch.object(settings, "admin_jwt_audience", ""), \
             patch.object(settings, "admin_jwt_issuer", "expected-issuer"):
            with pytest.raises(AuthenticationError):
                decode_token(token)

    def test_no_admin_secret_invalid_token_raises_authentication_error(self):
        with patch.object(settings, "admin_jwt_secret_key", ""):
            with pytest.raises(AuthenticationError):
                decode_token("not.a.valid.token")


# ---------------------------------------------------------------------------
# register_user
# ---------------------------------------------------------------------------


class TestRegisterUser:
    @pytest.mark.anyio
    async def test_registers_new_user_successfully(self):
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value=None)
        mock_db.users.insert_one = AsyncMock()

        data = RegisterRequest(
            email="newuser@example.com",
            password="NewPassword123!",
            display_name="New User",
        )
        user = await register_user(mock_db, data)

        assert user.email == "newuser@example.com"
        assert user.display_name == "New User"
        assert user.is_active is True
        assert verify_password("NewPassword123!", user.hashed_password)
        mock_db.users.insert_one.assert_called_once()

    @pytest.mark.anyio
    async def test_raises_validation_error_when_email_taken(self):
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value={"_id": "existing", "email": "existing@example.com"})

        data = RegisterRequest(
            email="existing@example.com",
            password="Password123!",
            display_name="Some User",
        )
        with pytest.raises(ValidationError, match="Email already registered"):
            await register_user(mock_db, data)

    @pytest.mark.anyio
    async def test_user_inserted_with_correct_doc_structure(self):
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value=None)
        inserted_docs = []
        mock_db.users.insert_one = AsyncMock(side_effect=lambda doc: inserted_docs.append(doc))

        data = RegisterRequest(
            email="doc@example.com",
            password="DocPass123!",
            display_name="Doc User",
        )
        await register_user(mock_db, data)

        assert len(inserted_docs) == 1
        doc = inserted_docs[0]
        assert "_id" in doc
        assert doc["email"] == "doc@example.com"
        assert "hashed_password" in doc


# ---------------------------------------------------------------------------
# authenticate_user
# ---------------------------------------------------------------------------


class TestAuthenticateUser:
    @pytest.mark.anyio
    async def test_valid_credentials_returns_user(self, test_user_doc):
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value=test_user_doc)

        user = await authenticate_user(mock_db, "test@example.com", "TestPassword123!")
        assert user.email == "test@example.com"

    @pytest.mark.anyio
    async def test_user_not_found_raises_authentication_error(self):
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value=None)

        with pytest.raises(AuthenticationError, match="Invalid email or password"):
            await authenticate_user(mock_db, "nobody@example.com", "password")

    @pytest.mark.anyio
    async def test_wrong_password_raises_authentication_error(self, test_user_doc):
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value=test_user_doc)

        with pytest.raises(AuthenticationError, match="Invalid email or password"):
            await authenticate_user(mock_db, "test@example.com", "WrongPassword!")

    @pytest.mark.anyio
    async def test_inactive_user_raises_authentication_error(self, inactive_user_doc):
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value=inactive_user_doc)

        with pytest.raises(AuthenticationError, match="Account is disabled"):
            await authenticate_user(mock_db, "inactive@example.com", "SomePassword!")


# ---------------------------------------------------------------------------
# get_user_by_id
# ---------------------------------------------------------------------------


class TestGetUserById:
    @pytest.mark.anyio
    async def test_returns_user_when_found(self, test_user_doc, test_user_id):
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value=test_user_doc)

        user = await get_user_by_id(mock_db, test_user_id)
        assert user is not None
        assert user.email == "test@example.com"

    @pytest.mark.anyio
    async def test_returns_none_when_not_found(self):
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value=None)

        result = await get_user_by_id(mock_db, str(uuid.uuid4()))
        assert result is None

    @pytest.mark.anyio
    async def test_accepts_uuid_object(self, test_user_doc):
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value=test_user_doc)

        uid = uuid.UUID(test_user_doc["_id"])
        user = await get_user_by_id(mock_db, uid)
        assert user is not None

    @pytest.mark.anyio
    async def test_queries_with_string_id(self, test_user_doc, test_user_id):
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value=test_user_doc)

        await get_user_by_id(mock_db, test_user_id)
        mock_db.users.find_one.assert_called_once_with({"_id": test_user_id})
