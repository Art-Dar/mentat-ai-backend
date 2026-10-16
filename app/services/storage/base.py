import uuid
from typing import Protocol, runtime_checkable

# How long a download URL stays valid. Long enough to click, short enough
# that a URL copied out of a browser's history is useless by the time
# anyone finds it. These are receipts and bank statements.
DEFAULT_URL_TTL_SECONDS = 900  # 15 minutes


class StorageError(RuntimeError):
    """The object store could not satisfy the request."""


class ObjectNotFound(StorageError):
    """No object at that key."""


@runtime_checkable
class FileStore(Protocol):
    """What the ingestion layer needs from a place to put bytes."""

    async def put(self, key: str, data: bytes, *, content_type: str) -> None: ...

    async def get(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None: ...

    async def url_for(self, key: str, *, expires_in: int = DEFAULT_URL_TTL_SECONDS) -> str:
        """
        A URL the browser can fetch directly.
        """
        ...


def storage_key(user_id: uuid.UUID, document_id: uuid.UUID, *, extension: str = "pdf") -> str:
    """
    The original filename is display metadata on the row; it
    never reaches a path.
    """
    return f"documents/{user_id}/{document_id}.{extension.lstrip('.')}"
