from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from easyweaver.auth import service
from easyweaver.auth.dependencies import get_current_user
from easyweaver.auth.models import User
from easyweaver.auth.schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    UserResponse,
)
from easyweaver.dependencies import get_db

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    return await service.register_user(db, data)


@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await service.authenticate_user(db, data.email, data.password)
    return TokenResponse(
        access_token=service.create_access_token(str(user.id)),
        refresh_token=service.create_refresh_token(str(user.id)),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(data: RefreshRequest, db: AsyncSession = Depends(get_db)):
    payload = service.decode_token(data.refresh_token)
    if payload.get("type") != "refresh":
        from easyweaver.core.exceptions import AuthenticationError

        raise AuthenticationError("Invalid refresh token")
    user = await service.get_user_by_id(db, payload["sub"])
    if not user:
        from easyweaver.core.exceptions import AuthenticationError

        raise AuthenticationError("User not found")
    return TokenResponse(
        access_token=service.create_access_token(str(user.id)),
        refresh_token=service.create_refresh_token(str(user.id)),
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return current_user
