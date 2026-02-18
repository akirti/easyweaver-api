import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from easyweaver.auth.models import User
from easyweaver.auth.schemas import RegisterRequest
from easyweaver.core.exceptions import AuthenticationError, ValidationError
from easyweaver.settings import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


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
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {e}")


async def register_user(db: AsyncSession, data: RegisterRequest) -> User:
    existing = await db.execute(select(User).where(User.email == data.email))
    if existing.scalar_one_or_none():
        raise ValidationError("Email already registered")

    user = User(
        email=data.email,
        hashed_password=hash_password(data.password),
        display_name=data.display_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(password, user.hashed_password):
        raise AuthenticationError("Invalid email or password")
    if not user.is_active:
        raise AuthenticationError("Account is disabled")
    return user


async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
