from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import httpx
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from jose import jwt

from app.core.config import settings

GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(subject: str) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(
        payload, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm
    )


def decode_token(token: str) -> str:
    payload = jwt.decode(
        token, settings.jwt_secret.get_secret_value(), algorithms=[settings.jwt_algorithm]
    )
    return payload["sub"]


def verify_google_id_token(token: str) -> dict[str, Any]:
    """
    Verify a Google Identity Services ID token against our client ID and
    return its claims (sub, email, email_verified, ...).

    Raises ValueError if the token is invalid, expired, or was issued for a
    different client.
    """
    return google_id_token.verify_oauth2_token(
        token, google_requests.Request(), settings.google_client_id
    )


def verify_google_access_token(access_token: str) -> dict[str, Any]:
    """
    Verify a Google OAuth access token — the kind `chrome.identity.getAuthToken`
    returns — via Google's tokeninfo endpoint, and return claims shaped like
    an ID token's (sub, email, email_verified).

    Unlike an ID token, an access token isn't a signed JWT we can check
    locally, so we ask Google directly. tokeninfo confirms the token is
    still valid and tells us who it was issued to (`aud`) — we still have to
    check that ourselves, since a valid-but-wrong-audience token (minted for
    some other app) would otherwise be accepted.

    Raises ValueError if the token is invalid, expired, or was issued for a
    different client.
    """
    response = httpx.get(GOOGLE_TOKENINFO_URL, params={"access_token": access_token}, timeout=5.0)
    if response.status_code != 200:
        raise ValueError("Invalid or expired Google access token")

    claims = response.json()
    if claims.get("aud") != settings.google_client_id:
        raise ValueError("Google access token was issued for a different client")

    return {
        "sub": claims["sub"],
        "email": claims.get("email", ""),
        "email_verified": claims.get("email_verified") in ("true", True),
    }
