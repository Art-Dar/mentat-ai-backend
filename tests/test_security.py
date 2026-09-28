import asyncio

import pytest
from fastapi import HTTPException

from app.core.deps import get_current_user
from app.core.security import create_access_token, decode_token, hash_password, verify_password


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
