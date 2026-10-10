import uuid

import pytest
from sqlalchemy import delete, select

from app.models import Document, DocumentSource, Tag, User


@pytest.fixture
def auth_headers(client, seeded_user):
    email, password = seeded_user
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def document(db_session, seeded_user):
    """A document owned by the seeded user, removed afterwards."""
    email, _ = seeded_user
    user = db_session.scalar(select(User).where(User.email == email))
    doc = Document(
        user_id=user.id,
        source=DocumentSource.NOTE,
        content="a note that needs tagging",
        title="Tagged note",
    )
    db_session.add(doc)
    db_session.commit()
    document_id = doc.id

    yield document_id

    db_session.execute(delete(Document).where(Document.id == document_id))
    db_session.commit()


@pytest.fixture(autouse=True)
def clean_tags(db_session):
    """Tags are global, so they outlive the document that created them."""
    yield
    db_session.execute(delete(Tag).where(Tag.name.ilike("api:%")))
    db_session.commit()


def t(name: str) -> str:
    return f"api:{name}"


# attaching


def test_posting_tags_attaches_them(client, auth_headers, document):
    response = client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": [t("python"), t("fastapi")]},
    )

    assert response.status_code == 201
    assert {item["tag"]["name"] for item in response.json()} == {t("python"), t("fastapi")}


def test_api_tags_are_always_user_assigned(client, auth_headers, document):
    """
    AUTO belongs to the pipeline. If the API could write it, the extension
    could demote a tag the user applied by hand.
    """
    response = client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": [t("python")]},
    )

    link = response.json()[0]
    assert link["assigned_by"] == "user"
    assert link["confidence"] is None


def test_the_response_carries_the_full_tag_list(client, auth_headers, document):
    """So the client renders from one response instead of merging two."""
    client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": [t("first")]},
    )
    response = client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": [t("second")]},
    )

    assert {item["tag"]["name"] for item in response.json()} == {t("first"), t("second")}


def test_posting_the_same_tag_twice_is_not_a_conflict(client, auth_headers, document):
    """Re-tagging by accident is constant; failing the user for it is rude."""
    client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": [t("python")]},
    )
    response = client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": [t("PYTHON")]},
    )

    assert response.status_code == 201
    assert len(response.json()) == 1


def test_a_blank_tag_name_is_rejected(client, auth_headers, document):
    response = client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": ["   "]},
    )

    assert response.status_code == 422


def test_an_empty_names_list_is_rejected(client, auth_headers, document):
    response = client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": []},
    )

    assert response.status_code == 422


def test_a_misspelled_field_name_is_rejected(client, auth_headers, document):
    """
    Sending "tags" instead of "names" must 422 on the missing required
    field — nothing is attached by accident.
    """
    response = client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"tags": [t("python")]},
    )

    assert response.status_code == 422


def test_an_extra_field_is_rejected(client, auth_headers, document):
    """
    RequestSchema's extra="forbid", on its own.

    `names` is valid here, so the only thing that can reject this request
    is the unknown key. Without that distinction the test would pass on
    the missing-field error instead and prove nothing about extra="forbid".
    """
    response = client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": [t("python")], "colour": "#FF0000"},
    )

    assert response.status_code == 422


# reading


def test_reading_a_documents_tags(client, auth_headers, document):
    client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": [t("zebra"), t("Apple")]},
    )

    response = client.get(f"/api/v1/documents/{document}/tags", headers=auth_headers)

    assert response.status_code == 200
    assert [item["tag"]["name"] for item in response.json()] == [t("Apple"), t("zebra")]


def test_a_document_with_no_tags_returns_an_empty_list(client, auth_headers, document):
    response = client.get(f"/api/v1/documents/{document}/tags", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == []


def test_listing_all_tags_includes_usage_counts(client, auth_headers, document):
    client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": [t("counted")]},
    )

    response = client.get("/api/v1/tags", headers=auth_headers)

    assert response.status_code == 200
    counts = {item["name"]: item["document_count"] for item in response.json()}
    assert counts[t("counted")] == 1


# detaching


def test_deleting_a_tag_detaches_it(client, auth_headers, document):
    client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": [t("python"), t("fastapi")]},
    )

    response = client.delete(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        params={"name": t("PYTHON")},  # case-insensitive
    )

    assert response.status_code == 204
    remaining = client.get(f"/api/v1/documents/{document}/tags", headers=auth_headers).json()
    assert {item["tag"]["name"] for item in remaining} == {t("fastapi")}


def test_deleting_a_tag_that_is_not_attached_is_404(client, auth_headers, document):
    response = client.delete(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        params={"name": t("never-attached")},
    )

    assert response.status_code == 404


def test_a_tag_name_with_a_slash_survives_the_round_trip(client, auth_headers, document):
    """
    Why the name is a query parameter and not a path segment: proxies
    normalise %2F back into a separator, so "ml/nlp" in a path would stop
    the route matching.
    """
    client.post(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        json={"names": [t("ml/nlp")]},
    )

    response = client.delete(
        f"/api/v1/documents/{document}/tags",
        headers=auth_headers,
        params={"name": t("ml/nlp")},
    )

    assert response.status_code == 204


# missing documents and auth


def test_tagging_a_missing_document_is_404(client, auth_headers):
    """The FK would reject it anyway, but as an opaque 500."""
    response = client.post(
        f"/api/v1/documents/{uuid.uuid4()}/tags",
        headers=auth_headers,
        json={"names": [t("python")]},
    )

    assert response.status_code == 404


def test_reading_tags_of_a_missing_document_is_404(client, auth_headers):
    response = client.get(f"/api/v1/documents/{uuid.uuid4()}/tags", headers=auth_headers)

    assert response.status_code == 404


@pytest.mark.parametrize("method", ["get", "post", "delete"])
def test_tag_endpoints_require_authentication(client, document, method):
    url = f"/api/v1/documents/{document}/tags"
    call = getattr(client, method)
    response = call(url, json={"names": ["x"]}) if method == "post" else call(url)

    assert response.status_code == 401


def test_listing_tags_requires_authentication(client):
    assert client.get("/api/v1/tags").status_code == 401