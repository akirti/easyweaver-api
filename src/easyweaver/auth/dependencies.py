import uuid

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.auth.models import User
from easyweaver.auth.service import decode_token, get_user_by_id
from easyweaver.core.exceptions import AuthenticationError
from easyweaver.dependencies import get_db

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> User:
    if credentials is None:
        raise AuthenticationError("Not authenticated")

    payload = decode_token(credentials.credentials)
    if payload.get("type") != "access":
        raise AuthenticationError("Invalid token type")

    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid token")

    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise AuthenticationError("User not found or inactive")

    return user
