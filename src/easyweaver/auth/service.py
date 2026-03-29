import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.auth.models import User
from easyweaver.auth.schemas import RegisterRequest
from easyweaver.core.exceptions import AuthenticationError, ValidationError
from easyweaver.settings import settings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    return jwt.encode(
        {"sub": user_id, "exp": expire, "type": "access"},
        settings.jwt_secret_key,
        algorithm="HS256",
    )


def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_expire_days)
    return jwt.encode(
        {"sub": user_id, "exp": expire, "type": "refresh"},
        settings.jwt_secret_key,
        algorithm="HS256",
    )


def decode_token(token: str) -> dict:
    # Try easyweaver's own JWT first
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        payload["_token_source"] = "easyweaver"
        return payload
    except JWTError:
        pass

    # Try admin-panel JWT if shared secret is configured
    if settings.admin_jwt_secret_key:
        try:
            decode_opts: dict = {"algorithms": ["HS256"]}
            # Only validate issuer/audience if explicitly configured
            if settings.admin_jwt_audience:
                decode_opts["audience"] = settings.admin_jwt_audience
            else:
                decode_opts["options"] = {"verify_aud": False}
            payload = jwt.decode(
                token,
                settings.admin_jwt_secret_key,
                **decode_opts,
            )
            # Verify issuer manually if configured (jose doesn't have issuer= param)
            if settings.admin_jwt_issuer and payload.get("iss") != settings.admin_jwt_issuer:
                raise JWTError(f"Invalid issuer: {payload.get('iss')}")
            payload["_token_source"] = "admin_panel"
            return payload
        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {e}")

    raise AuthenticationError("Invalid token")


async def register_user(db: AsyncIOMotorDatabase, data: RegisterRequest) -> User:
    existing = await db.users.find_one({"email": data.email})
    if existing:
        raise ValidationError("Email already registered")

    now = datetime.now(timezone.utc)
    user = User(
        id=uuid.uuid4(),
        email=data.email,
        hashed_password=hash_password(data.password),
        display_name=data.display_name,
        created_at=now,
        updated_at=now,
    )
    await db.users.insert_one(user.to_doc())
    return user


async def authenticate_user(db: AsyncIOMotorDatabase, email: str, password: str) -> User:
    doc = await db.users.find_one({"email": email})
    if not doc:
        raise AuthenticationError("Invalid email or password")
    user = User.from_doc(doc)
    if not verify_password(password, user.hashed_password):
        raise AuthenticationError("Invalid email or password")
    if not user.is_active:
        raise AuthenticationError("Account is disabled")
    return user


async def get_user_by_id(db: AsyncIOMotorDatabase, user_id: uuid.UUID | str) -> User | None:
    uid = str(user_id)
    doc = await db.users.find_one({"_id": uid})
    if not doc:
        return None
    return User.from_doc(doc)
