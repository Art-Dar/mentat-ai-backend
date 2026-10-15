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


def test_register_rejects_email_already_used_by_google_account(
    client, db_session, seeded_google_user
):
    email, _ = seeded_google_user

    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "correct-horse-battery-staple"},
    )

    assert response.status_code == 409
    # no password got attached, and no duplicate user was created
    users = db_session.scalars(select(User).where(User.email == email)).all()
    assert len(users) == 1
    assert (
        db_session.scalar(
            select(AuthIdentity).where(
                AuthIdentity.provider == AuthProvider.PASSWORD,
                AuthIdentity.provider_user_id == email,
            )
        )
        is None
    )


def _mock_google_token(monkeypatch, claims: dict):
    monkeypatch.setattr("app.api.v1.auth.verify_google_id_token", lambda token: claims)


def test_google_login_creates_new_user_and_signs_in(
    client, db_session, register_email, monkeypatch
):
    google_sub = f"google-sub-{register_email}"
    _mock_google_token(
        monkeypatch, {"sub": google_sub, "email": register_email, "email_verified": True}
    )

    response = client.post("/api/v1/auth/google", json={"id_token": "whatever"})

    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"

    user = db_session.scalar(select(User).where(User.email == register_email))
    assert user is not None
    identity = db_session.scalar(select(AuthIdentity).where(AuthIdentity.user_id == user.id))
    assert identity.provider == AuthProvider.GOOGLE
    assert identity.provider_user_id == google_sub
    assert identity.password_hash is None


def test_google_login_existing_identity_signs_in(client, seeded_google_user, monkeypatch):
    email, google_sub = seeded_google_user
    _mock_google_token(monkeypatch, {"sub": google_sub, "email": email, "email_verified": True})

    response = client.post("/api/v1/auth/google", json={"id_token": "whatever"})

    assert response.status_code == 200
    assert "access_token" in response.json()


def test_google_login_rejects_email_already_registered_with_password(
    client, db_session, seeded_user, monkeypatch
):
    email, _ = seeded_user
    _mock_google_token(
        monkeypatch, {"sub": "a-new-google-sub", "email": email, "email_verified": True}
    )

    response = client.post("/api/v1/auth/google", json={"id_token": "whatever"})

    assert response.status_code == 409
    # no Google identity got linked behind the scenes
    assert (
        db_session.scalar(
            select(AuthIdentity).where(
                AuthIdentity.provider == AuthProvider.GOOGLE,
                AuthIdentity.provider_user_id == "a-new-google-sub",
            )
        )
        is None
    )


def test_google_login_invalid_token_returns_401(client, monkeypatch):
    def _raise_invalid(token):
        raise ValueError("invalid token")

    monkeypatch.setattr("app.api.v1.auth.verify_google_id_token", _raise_invalid)

    response = client.post("/api/v1/auth/google", json={"id_token": "garbage"})

    assert response.status_code == 401


def test_connect_google_links_identity_to_current_user(
    client, db_session, seeded_user, monkeypatch
):
    email, password = seeded_user
    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    google_sub = "connect-sub-1"
    _mock_google_token(monkeypatch, {"sub": google_sub, "email": email, "email_verified": True})

    response = client.post(
        "/api/v1/auth/google/connect", json={"id_token": "whatever"}, headers=headers
    )

    assert response.status_code == 201
    assert response.json() == {"provider": "google", "connected": True}

    user = db_session.scalar(select(User).where(User.email == email))
    identity = db_session.scalar(
        select(AuthIdentity).where(
            AuthIdentity.provider == AuthProvider.GOOGLE,
            AuthIdentity.provider_user_id == google_sub,
        )
    )
    assert identity.user_id == user.id


def test_connect_google_rejects_already_connected_identity(
    client, seeded_user, seeded_google_user, monkeypatch
):
    email, password = seeded_user
    _, google_sub = seeded_google_user  # already connected to a different user

    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    _mock_google_token(monkeypatch, {"sub": google_sub, "email": email, "email_verified": True})

    response = client.post(
        "/api/v1/auth/google/connect", json={"id_token": "whatever"}, headers=headers
    )

    assert response.status_code == 409


def test_connect_google_requires_auth(client):
    response = client.post("/api/v1/auth/google/connect", json={"id_token": "whatever"})

    assert response.status_code == 401


def _mock_google_access_token(monkeypatch, claims: dict):
    monkeypatch.setattr("app.api.v1.auth.verify_google_access_token", lambda token: claims)


def test_google_login_with_access_token_creates_user(
    client, db_session, register_email, monkeypatch
):
    # exercises the chrome.identity.getAuthToken path, not the ID-token one
    _mock_google_access_token(
        monkeypatch, {"sub": f"chrome-sub-{register_email}", "email": register_email}
    )

    response = client.post("/api/v1/auth/google", json={"access_token": "whatever"})

    assert response.status_code == 200
    assert db_session.scalar(select(User).where(User.email == register_email)) is not None


def test_google_request_rejects_both_token_types(client):
    response = client.post(
        "/api/v1/auth/google",
        json={"id_token": "a", "access_token": "b"},
    )

    assert response.status_code == 422


def test_google_request_rejects_neither_token_type(client):
    response = client.post("/api/v1/auth/google", json={})

    assert response.status_code == 422


def test_set_password_lets_google_user_log_in_with_a_password_afterwards(
    client, seeded_google_user, monkeypatch
):
    email, google_sub = seeded_google_user
    _mock_google_token(monkeypatch, {"sub": google_sub, "email": email, "email_verified": True})
    login = client.post("/api/v1/auth/google", json={"id_token": "whatever"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    response = client.post(
        "/api/v1/auth/set-password",
        json={"password": "correct-horse-battery-staple"},
        headers=headers,
    )

    assert response.status_code == 201
    assert response.json() == {"provider": "password", "connected": True}

    # the whole point: password login now works for this account
    password_login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "correct-horse-battery-staple"},
    )
    assert password_login.status_code == 200


def test_set_password_rejects_if_already_has_one(client, seeded_user):
    email, password = seeded_user
    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    response = client.post(
        "/api/v1/auth/set-password",
        json={"password": "another-correct-horse-battery"},
        headers=headers,
    )

    assert response.status_code == 409


def test_set_password_requires_auth(client):
    response = client.post(
        "/api/v1/auth/set-password", json={"password": "correct-horse-battery-staple"}
    )

    assert response.status_code == 401


def test_set_password_too_short(client, seeded_google_user, monkeypatch):
    email, google_sub = seeded_google_user
    _mock_google_token(monkeypatch, {"sub": google_sub, "email": email, "email_verified": True})
    login = client.post("/api/v1/auth/google", json={"id_token": "whatever"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    response = client.post(
        "/api/v1/auth/set-password", json={"password": "short1"}, headers=headers
    )

    assert response.status_code == 422
