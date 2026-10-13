import asyncio

import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.core.deps import get_current_user
from app.core.security import (
    create_access_token,
    decode_token,
    hash_password,
    verify_google_access_token,
    verify_password,
)


class _FakeTokeninfoResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_hash_and_verify_password_roundtrip():
    hashed = hash_password("my-secret")

    assert verify_password("my-secret", hashed)
    assert not verify_password("wrong-password", hashed)


def test_create_and_decode_token_roundtrip():
    token = create_access_token(subject="artem")

    assert decode_token(token) == "artem"


def test_get_current_user_rejects_invalid_token():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(get_current_user(token="not-a-real-token"))

    assert exc_info.value.status_code == 401


def test_verify_google_access_token_returns_claims(monkeypatch):
    monkeypatch.setattr(
        "app.core.security.httpx.get",
        lambda *a, **k: _FakeTokeninfoResponse(
            200,
            {
                "aud": settings.google_client_id,
                "sub": "123",
                "email": "a@example.com",
                "email_verified": "true",
            },
        ),
    )

    claims = verify_google_access_token("whatever")

    assert claims == {"sub": "123", "email": "a@example.com", "email_verified": True}


def test_verify_google_access_token_rejects_wrong_audience(monkeypatch):
    monkeypatch.setattr(
        "app.core.security.httpx.get",
        lambda *a, **k: _FakeTokeninfoResponse(
            200, {"aud": "some-other-client", "sub": "123", "email": "a@example.com"}
        ),
    )

    with pytest.raises(ValueError):
        verify_google_access_token("whatever")


def test_verify_google_access_token_rejects_non_200(monkeypatch):
    monkeypatch.setattr(
        "app.core.security.httpx.get",
        lambda *a, **k: _FakeTokeninfoResponse(400, {"error": "invalid_token"}),
    )

    with pytest.raises(ValueError):
        verify_google_access_token("whatever")
