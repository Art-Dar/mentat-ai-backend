from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.security import create_access_token, verify_password
from app.models.user import AuthIdentity, AuthProvider
from app.schemas.auth import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth")


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    email = payload.email.strip().lower()

    identity = await db.scalar(
        select(AuthIdentity).where(
            AuthIdentity.provider == AuthProvider.PASSWORD,
            AuthIdentity.provider_user_id == email,
        )
    )
    # verify even when no identity exists-the response doesn't reveal whether the address is registered
    if identity is None or not verify_password(payload.password, identity.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    return TokenResponse(access_token=create_access_token(str(identity.user_id)))
