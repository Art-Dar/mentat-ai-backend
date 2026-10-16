import uuid

import boto3
import pytest
from moto import mock_aws

from app.services.storage.base import ObjectNotFound, StorageError, storage_key
from app.services.storage.fake_store import FakeFileStore
from app.services.storage.s3_store import S3FileStore

BUCKET = "mentat-documents"
PDF = b"%PDF-1.7\nhello\n%%EOF\n"


@pytest.fixture
def store():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        # No injected client: this exercises the real client-building path,
        # including _BOTO_CONFIG. Injecting one would have tested boto3's
        # defaults instead of ours — which is how the SigV2 presigned URL
        # went unnoticed the first time.
        yield S3FileStore(
            bucket=BUCKET,
            endpoint_url="",
            access_key="test",
            secret_key="test",
            region="us-east-1",
        )


@pytest.mark.anyio
async def test_put_then_get_round_trips(store):
    key = storage_key(uuid.uuid4(), uuid.uuid4())
    await store.put(key, PDF, content_type="application/pdf")
    assert await store.get(key) == PDF


@pytest.mark.anyio
async def test_content_type_is_stored(store):
    key = storage_key(uuid.uuid4(), uuid.uuid4())
    await store.put(key, PDF, content_type="application/pdf")
    head = boto3.client("s3", region_name="us-east-1").head_object(Bucket=BUCKET, Key=key)
    assert head["ContentType"] == "application/pdf"


@pytest.mark.anyio
async def test_missing_key_raises_object_not_found(store):
    with pytest.raises(ObjectNotFound):
        await store.get("documents/nope/nope.pdf")


@pytest.mark.anyio
async def test_delete_removes_the_object(store):
    key = storage_key(uuid.uuid4(), uuid.uuid4())
    await store.put(key, PDF, content_type="application/pdf")
    await store.delete(key)
    with pytest.raises(ObjectNotFound):
        await store.get(key)


@pytest.mark.anyio
async def test_deleting_a_missing_key_is_not_an_error(store):
    """S3 delete is idempotent, which is what makes the retry safe."""
    await store.delete("documents/nope/nope.pdf")


@pytest.mark.anyio
async def test_presigned_url_points_at_the_key(store):
    key = storage_key(uuid.uuid4(), uuid.uuid4())
    await store.put(key, PDF, content_type="application/pdf")
    url = await store.url_for(key, expires_in=60)
    assert key in url
    assert "X-Amz-Signature" in url
    assert "X-Amz-Expires=60" in url


@pytest.mark.anyio
async def test_an_unreachable_store_raises_storage_error():
    """Connection refused is the store being down, not a bad request."""
    broken = S3FileStore(
        bucket=BUCKET,
        endpoint_url="http://127.0.0.1:1",
        access_key="x",
        secret_key="y",
        region="auto",
    )
    with pytest.raises(StorageError):
        await broken.put("k", PDF, content_type="application/pdf")


@pytest.mark.anyio
async def test_ensure_bucket_creates_it():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        store = S3FileStore(bucket="brand-new", client=client)
        await store.ensure_bucket()
        assert "brand-new" in [b["Name"] for b in client.list_buckets()["Buckets"]]


# the key scheme


def test_key_is_scoped_by_user_and_document():
    user, doc = uuid.uuid4(), uuid.uuid4()
    assert storage_key(user, doc) == f"documents/{user}/{doc}.pdf"


def test_key_never_contains_a_client_filename():
    """Path traversal: the key is built from ids we generated, full stop."""
    key = storage_key(uuid.uuid4(), uuid.uuid4())
    assert ".." not in key
    assert key.count("/") == 2


def test_extension_is_configurable_for_images():
    key = storage_key(uuid.uuid4(), uuid.uuid4(), extension=".png")
    assert key.endswith(".png")
    assert ".." not in key


# the fake matches the real one


@pytest.mark.anyio
async def test_the_fake_behaves_like_the_real_store(store):
    """
    Every assertion here runs against BOTH, so a test written against the
    fake cannot pass for a reason S3 would not reproduce.
    """
    for impl in (store, FakeFileStore()):
        key = storage_key(uuid.uuid4(), uuid.uuid4())

        with pytest.raises(ObjectNotFound):
            await impl.get(key)

        await impl.put(key, PDF, content_type="application/pdf")
        assert await impl.get(key) == PDF
        assert key in await impl.url_for(key)

        await impl.delete(key)
        await impl.delete(key)  # idempotent in both
        with pytest.raises(ObjectNotFound):
            await impl.get(key)
