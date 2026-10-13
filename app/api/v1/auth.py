import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user
from app.core.security import (
    create_access_token,
    hash_password,
    verify_google_access_token,
    verify_google_id_token,
    verify_password,
)
from app.models.user import AuthIdentity, AuthProvider, User
from app.schemas.auth import (
    ConnectResponse,
    GoogleAuthRequest,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)

router = APIRouter(prefix="/auth")


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    email = payload.email.strip().lower()

    existing = await db.scalar(
        select(AuthIdentity).where(
            AuthIdentity.provider == AuthProvider.PASSWORD,
            AuthIdentity.provider_user_id == email,
        )
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists",
        )

    user = User(email=email, email_verified=False)
    user.identities.append(
        AuthIdentity(
            provider=AuthProvider.PASSWORD,
            provider_user_id=email,
            password_hash=hash_password(payload.password),
        )
    )
    db.add(user)
    await db.commit()

    return TokenResponse(access_token=create_access_token(str(user.id)))


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    email = payload.email.strip().lower()

    identity = await db.scalar(
        select(AuthIdentity).where(
            AuthIdentity.provider == AuthProvider.PASSWORD,
            AuthIdentity.provider_user_id == email,
        )
    )
    # verify even when no identity exists ->
    # the response doesn't reveal whether the address is registered
    if identity is None or not verify_password(payload.password, identity.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    return TokenResponse(access_token=create_access_token(str(identity.user_id)))


def _verify_google_claims(payload: GoogleAuthRequest) -> dict:
    try:
        if payload.id_token:
            return verify_google_id_token(payload.id_token)
        return verify_google_access_token(payload.access_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token",
        ) from exc


@router.post("/google", response_model=TokenResponse)
async def google_login(payload: GoogleAuthRequest, db: AsyncSession = Depends(get_db)):
    claims = _verify_google_claims(payload)
    google_sub = claims["sub"]
    email = claims["email"].strip().lower()

    identity = await db.scalar(
        select(AuthIdentity).where(
            AuthIdentity.provider == AuthProvider.GOOGLE,
            AuthIdentity.provider_user_id == google_sub,
        )
    )
    if identity is not None:
        return TokenResponse(access_token=create_access_token(str(identity.user_id)))

    # No Google identity yet. If the email is already registered under a
    # different identity (password, most likely), don't auto-link —
    # the person has to log in that way first and connect Google explicitly
    # via /auth/google/connect.
    if await db.scalar(select(User).where(User.email == email)) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "An account with that email already exists. Log in with your "
                "password, then connect Google from your account."
            ),
        )

    user = User(email=email, email_verified=bool(claims.get("email_verified", False)))
    user.identities.append(
        AuthIdentity(
            provider=AuthProvider.GOOGLE,
            provider_user_id=google_sub,
            password_hash=None,
        )
    )
    db.add(user)
    await db.commit()

    return TokenResponse(access_token=create_access_token(str(user.id)))


@router.post("/google/connect", response_model=ConnectResponse, status_code=status.HTTP_201_CREATED)
async def connect_google(
    payload: GoogleAuthRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
):
    claims = _verify_google_claims(payload)
    google_sub = claims["sub"]

    existing = await db.scalar(
        select(AuthIdentity).where(
            AuthIdentity.provider == AuthProvider.GOOGLE,
            AuthIdentity.provider_user_id == google_sub,
        )
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="That Google account is already connected to a user",
        )

    db.add(
        AuthIdentity(
            user_id=uuid.UUID(current_user_id),
            provider=AuthProvider.GOOGLE,
            provider_user_id=google_sub,
            password_hash=None,
        )
    )
    await db.commit()

    return ConnectResponse(provider="google")
