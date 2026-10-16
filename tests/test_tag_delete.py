import uuid

import pytest
from sqlalchemy import func, select

from app.models import Document, DocumentSource, DocumentTag, IngestionStatus, Tag, User


@pytest.fixture
def auth_headers(client, seeded_user):
    email, password = seeded_user
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def make_tag(db_session, seeded_user):
    """A tag, optionally attached to N fresh documents. Cleaned up after."""
    tags: list[uuid.UUID] = []
    documents: list[uuid.UUID] = []
    email, _ = seeded_user

    def build(name: str = "demo", *, documents_using: int = 0) -> Tag:
        user = db_session.execute(select(User).where(User.email == email)).scalars().one()
        tag = Tag(id=uuid.uuid4(), name=f"{name}-{uuid.uuid4().hex[:6]}")
        db_session.add(tag)
        db_session.flush()
        tags.append(tag.id)

        for _ in range(documents_using):
            document = Document(
                id=uuid.uuid4(),
                user_id=user.id,
                source=DocumentSource.NOTE,
                status=IngestionStatus.COMPLETED,
                content="a note that carries the tag",
                title="note",
            )
            db_session.add(document)
            db_session.flush()
            documents.append(document.id)
            db_session.add(DocumentTag(document_id=document.id, tag_id=tag.id))

        db_session.commit()
        return tag

    yield build

    for document_id in documents:
        document = db_session.get(Document, document_id)
        if document is not None:
            db_session.delete(document)
    for tag_id in tags:
        tag = db_session.get(Tag, tag_id)
        if tag is not None:
            db_session.delete(tag)
    db_session.commit()


def links_for(db_session, tag_id: uuid.UUID) -> int:
    return db_session.execute(
        select(func.count()).select_from(DocumentTag).where(DocumentTag.tag_id == tag_id)
    ).scalar_one()


def test_an_unused_tag_deletes_cleanly(client, auth_headers, make_tag, db_session):
    """The sidebar case: a tag showing 0 is clutter and should just go."""
    tag = make_tag("demo")
    # Captured as plain values first: once the API has deleted the row,
    # touching the ORM instance makes SQLAlchemy refresh it and raise
    # ObjectDeletedError, which would be a test bug reported as a failure.
    tag_id, tag_name = tag.id, tag.name

    response = client.delete(f"/api/v1/tags/{tag_id}", headers=auth_headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["documents_affected"] == 0
    assert body["name"] == tag_name
    assert uuid.UUID(body["tag_id"]) == tag_id

    db_session.expire_all()
    assert db_session.get(Tag, tag_id) is None


def test_the_deleted_tag_leaves_the_list(client, auth_headers, make_tag):
    tag_id = make_tag("demo").id
    before = {t["id"] for t in client.get("/api/v1/tags", headers=auth_headers).json()}
    assert str(tag_id) in before

    client.delete(f"/api/v1/tags/{tag_id}", headers=auth_headers)

    after = {t["id"] for t in client.get("/api/v1/tags", headers=auth_headers).json()}
    assert str(tag_id) not in after


def test_a_tag_in_use_is_refused_with_409_and_a_count(
    client, auth_headers, make_tag, db_session
):
    """
    Not a 422: the request is well formed, it conflicts with the state.
    The count is in the message because the user is being asked to decide,
    and "this is on 3 documents" is the whole basis for that decision.
    """
    tag = make_tag("work", documents_using=3)

    response = client.delete(f"/api/v1/tags/{tag.id}", headers=auth_headers)

    assert response.status_code == 409
    assert "3 document" in response.json()["error"]["message"]

    db_session.expire_all()
    assert db_session.get(Tag, tag.id) is not None, "a refused delete must change nothing"
    assert links_for(db_session, tag.id) == 3


def test_force_deletes_the_tag_and_its_links(client, auth_headers, make_tag, db_session):
    tag = make_tag("work", documents_using=3)
    tag_id = tag.id

    response = client.delete(f"/api/v1/tags/{tag_id}?force=true", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["documents_affected"] == 3

    db_session.expire_all()
    assert db_session.get(Tag, tag_id) is None
    assert links_for(db_session, tag_id) == 0, "the link rows must cascade away"


def test_force_does_not_delete_the_documents_themselves(
    client, auth_headers, make_tag, db_session
):
    """
    The cascade runs from tag to link, never from tag to document. If this
    ever fails, deleting a tag is destroying a user's saved pages.
    """
    tag_id = make_tag("work", documents_using=2).id
    document_ids = list(
        db_session.execute(
            select(DocumentTag.document_id).where(DocumentTag.tag_id == tag_id)
        ).scalars().all()
    )
    assert len(document_ids) == 2

    client.delete(f"/api/v1/tags/{tag_id}?force=true", headers=auth_headers)

    db_session.expire_all()
    for document_id in document_ids:
        assert db_session.get(Document, document_id) is not None, (
            "deleting a tag must never delete the documents it was on"
        )


def test_other_tags_on_the_same_document_survive(
    client, auth_headers, make_tag, db_session
):
    doomed_id = make_tag("doomed", documents_using=1).id
    keeper_id = make_tag("keeper").id
    document_id = db_session.execute(
        select(DocumentTag.document_id).where(DocumentTag.tag_id == doomed_id)
    ).scalars().one()
    db_session.add(DocumentTag(document_id=document_id, tag_id=keeper_id))
    db_session.commit()

    client.delete(f"/api/v1/tags/{doomed_id}?force=true", headers=auth_headers)

    db_session.expire_all()
    assert db_session.get(Tag, keeper_id) is not None
    assert links_for(db_session, keeper_id) == 1


def test_a_missing_tag_is_404(client, auth_headers):
    response = client.delete(f"/api/v1/tags/{uuid.uuid4()}", headers=auth_headers)
    assert response.status_code == 404


def test_deleting_twice_is_404_the_second_time(client, auth_headers, make_tag):
    """No crash on a double-click, and no pretending the second one worked."""
    tag_id = make_tag("demo").id

    assert client.delete(f"/api/v1/tags/{tag_id}", headers=auth_headers).status_code == 200
    assert client.delete(f"/api/v1/tags/{tag_id}", headers=auth_headers).status_code == 404


def test_authentication_is_required(client, make_tag):
    response = client.delete(f"/api/v1/tags/{make_tag('demo').id}")
    assert response.status_code in (401, 403)


def test_a_malformed_id_is_422_not_500(client, auth_headers):
    response = client.delete("/api/v1/tags/not-a-uuid", headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.parametrize("flag", ["true", "True", "1"])
def test_force_accepts_the_usual_spellings(client, auth_headers, make_tag, flag):
    """
    The frontend will send whatever its HTTP client produces for a boolean.
    All of these must mean the same thing, or the confirm dialog silently
    does nothing.
    """
    tag_id = make_tag("work", documents_using=1).id
    response = client.delete(f"/api/v1/tags/{tag_id}?force={flag}", headers=auth_headers)
    assert response.status_code == 200, f"force={flag} was not understood"


def test_force_false_still_refuses(client, auth_headers, make_tag):
    tag = make_tag("work", documents_using=1)
    response = client.delete(f"/api/v1/tags/{tag.id}?force=false", headers=auth_headers)
    assert response.status_code == 409