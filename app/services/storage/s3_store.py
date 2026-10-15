import functools
import logging
from typing import Any

import anyio.to_thread
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import settings
from app.services.storage.base import (
    DEFAULT_URL_TTL_SECONDS,
    ObjectNotFound,
    StorageError,
)

logger = logging.getLogger(__name__)

# The set differs by
_NOT_FOUND = {"404", "NoSuchKey", "NoSuchBucket", "NotFound"}

_BOTO_CONFIG = Config(
    # NOT the default everywhere. Depending on region and endpoint, boto3
    # will happily sign with SigV2, which produces a presigned URL
    signature_version="s3v4",
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required",
    retries={"max_attempts": 3, "mode": "standard"},
    # Keeps a hung store from holding an API worker open indefinitely.
    connect_timeout=5,
    read_timeout=30,
)


class S3FileStore:
    def __init__(
        self,
        *,
        bucket: str | None = None,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
        client: Any = None,
    ) -> None:
        self._bucket = bucket or settings.storage_bucket

        if client is not None:
            # Injected in tests so the suite never reaches a network.
            self._client = client
            return

        endpoint = endpoint_url if endpoint_url is not None else settings.storage_endpoint_url
        self._client = boto3.client(
            "s3",
            # Empty means real AWS, which infers its own endpoint. MinIO
            # and R2 both require this to be set.
            endpoint_url=endpoint or None,
            aws_access_key_id=access_key or settings.storage_access_key,
            aws_secret_access_key=(
                secret_key
                if secret_key is not None
                else settings.storage_secret_key.get_secret_value()
            ),
            # R2 ignores regions but SigV4 still needs one in the signature;
            # "auto" is what Cloudflare documents.
            region_name=region or settings.storage_region,
            config=_BOTO_CONFIG,
        )

    async def _call(self, method: str, **kwargs: Any) -> Any:
        func = functools.partial(getattr(self._client, method), **kwargs)
        try:
            return await anyio.to_thread.run_sync(func)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in _NOT_FOUND:
                raise ObjectNotFound(f"no object for that key ({code})") from exc
            raise StorageError(f"{method} failed: {code or exc}") from exc
        except BotoCoreError as exc:
            # Connection refused, DNS failure, timeout — the store is down
            # rather than the request being wrong.
            raise StorageError(f"{method} could not reach the store: {exc}") from exc

    async def put(self, key: str, data: bytes, *, content_type: str) -> None:
        await self._call(
            "put_object",
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    async def get(self, key: str) -> bytes:
        response = await self._call("get_object", Bucket=self._bucket, Key=key)
        # .read() on the streaming body is itself blocking I/O.
        return await anyio.to_thread.run_sync(response["Body"].read)

    async def delete(self, key: str) -> None:
        # S3 delete is idempotent: removing a key that is not there is a
        # success, not a 404
        await self._call("delete_object", Bucket=self._bucket, Key=key)

    async def url_for(self, key: str, *, expires_in: int = DEFAULT_URL_TTL_SECONDS) -> str:
        return await self._call(
            "generate_presigned_url",
            ClientMethod="get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    async def ensure_bucket(self) -> None:
        """
        Create the bucket if it does not exist.
        """
        try:
            await self._call("head_bucket", Bucket=self._bucket)
        except (ObjectNotFound, StorageError):
            await self._call("create_bucket", Bucket=self._bucket)


def get_file_store() -> S3FileStore:
    """The store the API uses. Replace in tests by passing one in."""
    return S3FileStore()
