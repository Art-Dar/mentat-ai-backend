from sqlalchemy import select

from app.models.user import AuthIdentity, AuthProvider, User


def test_login_success(client, seeded_user):
    email, password = seeded_user

    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})

    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_login_wrong_password(client, seeded_user):
    email, _ = seeded_user

    response = client.post(
        "/api/v1/auth/login", json={"email": email, "password": "wrong-password"}
    )

    assert response.status_code == 401


def test_login_unknown_user(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "does-not-exist@example.com", "password": "whatever"},
    )

    assert response.status_code == 401


def test_register_creates_user_and_signs_in(client, db_session, register_email):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": register_email, "password": "correct-horse-battery-staple"},
    )

    assert response.status_code == 201
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"

    user = db_session.scalar(select(User).where(User.email == register_email))
    assert user is not None
    identity = db_session.scalar(select(AuthIdentity).where(AuthIdentity.user_id == user.id))
    assert identity.provider == AuthProvider.PASSWORD
    assert identity.provider_user_id == register_email


def test_register_normalizes_email(client, db_session, register_email):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"  {register_email.upper()}  ",
            "password": "correct-horse-battery-staple",
        },
    )

    assert response.status_code == 201
    assert db_session.scalar(select(User).where(User.email == register_email)) is not None


def test_register_duplicate_email(client, seeded_user):
    email, _ = seeded_user

    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "correct-horse-battery-staple"},
    )

    assert response.status_code == 409


def test_register_password_too_short(client, register_email):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": register_email, "password": "short1"},
    )

    assert response.status_code == 422
