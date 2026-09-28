def test_login_success(client, seeded_user):
    username, password = seeded_user

    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})

    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_login_wrong_password(client, seeded_user):
    username, _ = seeded_user

    response = client.post(
        "/api/v1/auth/login", json={"username": username, "password": "wrong-password"}
    )

    assert response.status_code == 401


def test_login_unknown_user(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "does-not-exist", "password": "whatever"},
    )

    assert response.status_code == 401
