import uuid

from app.services.storage.base import DEFAULT_URL_TTL_SECONDS, ObjectNotFound


class FakeFileStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []

    async def put(self, key: str, data: bytes, *, content_type: str) -> None:
        self.calls.append(("put", key))
        self.objects[key] = data
        self.content_types[key] = content_type

    async def get(self, key: str) -> bytes:
        self.calls.append(("get", key))
        try:
            return self.objects[key]
        except KeyError as exc:
            raise ObjectNotFound(key) from exc

    async def delete(self, key: str) -> None:
        self.calls.append(("delete", key))
        # Idempotent, matching S3: deleting what is not there succeeds.
        self.objects.pop(key, None)
        self.content_types.pop(key, None)

    async def url_for(self, key: str, *, expires_in: int = DEFAULT_URL_TTL_SECONDS) -> str:
        self.calls.append(("url_for", key))
        if key not in self.objects:
            raise ObjectNotFound(key)
        return f"https://fake-store.invalid/{key}?expires_in={expires_in}"


class BrokenFileStore:
    """Stands in for R2 being unreachable, so the endpoint's failure path runs."""

    async def put(self, key: str, data: bytes, *, content_type: str) -> None:
        from app.services.storage.base import StorageError

        raise StorageError("object store unavailable")

    async def get(self, key: str) -> bytes:
        raise ObjectNotFound(key)

    async def delete(self, key: str) -> None:
        return None

    async def url_for(self, key: str, *, expires_in: int = DEFAULT_URL_TTL_SECONDS) -> str:
        raise ObjectNotFound(key)


def unused_key() -> str:
    return f"documents/{uuid.uuid4()}/{uuid.uuid4()}.pdf"