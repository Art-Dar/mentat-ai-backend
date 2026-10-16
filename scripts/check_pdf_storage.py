import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> project root
INVOKED_FROM = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import argparse
import asyncio
import hashlib
import uuid

from sqlalchemy import select

from app.core.config import settings
from app.core.db import AsyncSessionLocal
from app.models import Document, DocumentSource, IngestionStatus, User
from app.services.ingestion.extractors.pdf import (
    PdfExtractionError,
    extract_pdf_text_async,
)
from app.services.storage.base import (
    ObjectNotFound,
    StorageError,
    storage_key,
)
from app.services.storage.s3_store import S3FileStore

"""
Prove that a real PDF goes into Cloudflare R2 and Postgres together, and
that the two still agree afterwards.
"""

OK = "  ok   "
BAD = "  FAIL "
INFO = "       "


def _from_invocation_dir(raw: Path) -> Path:
    """
    Resolve a path the user typed against the directory they typed it in.
    """
    return raw if raw.is_absolute() else (INVOKED_FROM / raw)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


async def _pick_user(db, email: str | None) -> User:
    stmt = select(User).order_by(User.created_at)
    if email:
        stmt = stmt.where(User.email == email)
    user = (await db.execute(stmt)).scalars().first()
    if user is None:
        raise SystemExit(
            f"No user{' with email ' + email if email else ''} in the database. "
            "Run: poetry run python -m scripts.seed_users"
        )
    return user


async def ingest(path: Path, email: str | None) -> None:
    path = _from_invocation_dir(path)
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    data = path.read_bytes()
    print(f"\nfile   {path.name}  ({len(data):,} bytes, sha256 {_digest(data)})")
    print(f"bucket {settings.storage_bucket} @ {settings.storage_endpoint_url or 'AWS'}\n")

    # 1. extraction — no database, no network
    try:
        extraction = await extract_pdf_text_async(data)
    except PdfExtractionError as exc:
        raise SystemExit(f"{BAD}extraction refused the file: {exc}") from exc

    print(f"{OK}extracted {extraction.page_count} page(s), "
          f"{len(extraction.text):,} chars")
    if extraction.needs_ocr:
        print(f"{INFO}pages needing OCR: {list(extraction.pages_needing_ocr)}"
              f"{'  (every page — this is a scan)' if extraction.is_fully_scanned else ''}")
    else:
        print(f"{INFO}every page has a usable text layer")

    store = S3FileStore()

    async with AsyncSessionLocal() as db:
        user = await _pick_user(db, email)
        document_id = uuid.uuid4()
        key = storage_key(user.id, document_id)
        print(f"{INFO}owner {user.email}")
        print(f"{INFO}key   {key}\n")

        # 2. bytes first — same ordering the endpoint uses
        try:
            await store.put(key, data, content_type="application/pdf")
        except StorageError as exc:
            raise SystemExit(f"{BAD}could not write to the bucket: {exc}") from exc
        print(f"{OK}object written to R2")

        # 3. then the row
        db.add(
            Document(
                id=document_id,
                user_id=user.id,
                source=DocumentSource.PDF,
                content=extraction.text,
                title=path.stem,
                status=IngestionStatus.PENDING,
                storage_key=key,
                file_size=len(data),
                original_filename=path.name,
                page_map=[
                    {
                        "index": p.index,
                        "start_char": p.start_char,
                        "end_char": p.end_char,
                        "needs_ocr": p.needs_ocr,
                    }
                    for p in extraction.pages
                ],
            )
        )
        await db.commit()
        print(f"{OK}row committed, id {document_id}")

    print("\nverifying the two agree:\n")
    await _verify(document_id, expected=data)
    print(f"\nclean up with:\n  poetry run python -m scripts.check_pdf_storage "
          f"delete {document_id}\n")


async def _verify(document_id: uuid.UUID, expected: bytes | None = None) -> bool:
    """Read the row back in a fresh session and reconcile it with the bucket."""
    store = S3FileStore()
    good = True

    async with AsyncSessionLocal() as db:
        document = await db.get(Document, document_id)

    if document is None:
        print(f"{BAD}no document row with that id")
        return False
    print(f"{OK}row found: status={document.status.value} "
          f"size={document.file_size:,} pages={len(document.page_map or [])}")

    if not document.storage_key:
        print(f"{BAD}the row has no storage_key — nothing points at an object")
        return False

    try:
        fetched = await store.get(document.storage_key)
    except ObjectNotFound:
        print(f"{BAD}storage_key points at an object that is NOT in the bucket")
        print(f"{INFO}this is the orphaned-row case: {document.storage_key}")
        return False
    except StorageError as exc:
        print(f"{BAD}could not read the object: {exc}")
        return False

    print(f"{OK}object fetched back, {len(fetched):,} bytes, sha256 {_digest(fetched)}")

    if document.file_size != len(fetched):
        print(f"{BAD}file_size on the row ({document.file_size}) != object "
              f"({len(fetched)})")
        good = False

    if expected is not None:
        if fetched == expected:
            print(f"{OK}bytes are identical to the file on disk")
        else:
            print(f"{BAD}bytes differ from the file on disk — corruption in transit")
            good = False

    # The guarantee citations rely on: page offsets index the stored text.
    content = document.content or ""
    for page in document.page_map or []:
        slice_ = content[page["start_char"]: page["end_char"]]
        if len(slice_) != page["end_char"] - page["start_char"]:
            print(f"{BAD}page {page['index']} offsets fall outside content")
            good = False
            break
    else:
        print(f"{OK}every page offset indexes the stored content")

    try:
        url = await store.url_for(document.storage_key, expires_in=900)
        print(f"{OK}presigned URL issued (valid 15 min) — open it to see the PDF:")
        print(f"\n{url}\n")
    except StorageError as exc:
        print(f"{BAD}could not sign a URL: {exc}")
        good = False

    print("RESULT: storage and database agree" if good else "RESULT: THEY DISAGREE")
    return good


async def show(document_id: uuid.UUID) -> None:
    print()
    await _verify(document_id)


async def listing(limit: int) -> None:
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(Document)
                .where(Document.storage_key.is_not(None))
                .order_by(Document.created_at.desc())
                .limit(limit)
            )
        ).scalars().all()

    if not rows:
        print("\nNo documents with a stored file yet.\n")
        return

    print(f"\n{'id':38} {'status':11} {'pages':>5} {'size':>10}  title")
    for d in rows:
        print(f"{str(d.id):38} {d.status.value:11} "
              f"{len(d.page_map or []):>5} {d.file_size or 0:>10,}  "
              f"{(d.title or '')[:30]}")
    print()


async def delete(document_id: uuid.UUID) -> None:
    store = S3FileStore()
    async with AsyncSessionLocal() as db:
        document = await db.get(Document, document_id)
        if document is None:
            raise SystemExit("No document with that id")
        key = document.storage_key
        # Row first, then the object: the row is what the user sees, and a
        # leftover object is collectable garbage. The reverse leaves a
        # visible document whose file 404s.
        await db.delete(document)
        await db.commit()
    print(f"{OK}row deleted")

    if key:
        try:
            await store.delete(key)
            print(f"{OK}object deleted ({key})")
        except StorageError as exc:
            print(f"{BAD}row is gone but the object remains: {exc}")


_EXAMPLES = """
examples:
  python -m scripts.check_pdf_storage ingest docs/sample-invoice.pdf
  python -m scripts.check_pdf_storage list
  python -m scripts.check_pdf_storage show 6d154105-8c35-4b7c-bfda-a153fcac7c9e
  python -m scripts.check_pdf_storage delete 6d154105-8c35-4b7c-bfda-a153fcac7c9e

needs: docker compose up -d db, alembic upgrade head, a seeded user, and
the STORAGE_* keys in .env
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="check_pdf_storage",
        description="Reconcile a PDF in object storage against its database row.",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="push a PDF through R2 and the database")
    p_ingest.add_argument("path", type=Path)
    p_ingest.add_argument("--email", help="owner; defaults to the oldest user")

    p_list = sub.add_parser("list", help="documents that have a stored file")
    p_list.add_argument("--limit", type=int, default=15)

    p_show = sub.add_parser("show", help="re-check one document against the bucket")
    p_show.add_argument("document_id", type=uuid.UUID)

    p_del = sub.add_parser("delete", help="remove the row and its object")
    p_del.add_argument("document_id", type=uuid.UUID)

    # Launched with no arguments at all — PyCharm's green arrow, or a
    # double-click. argparse would print three lines of usage and exit 2;
    # the full help with examples is more use to whoever did that.
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.command in {"ingest", "show", "delete"} and not settings.storage_access_key:
        sys.exit("STORAGE_ACCESS_KEY is empty — check your .env")

    if args.command == "ingest":
        asyncio.run(ingest(args.path, args.email))
    elif args.command == "list":
        asyncio.run(listing(args.limit))
    elif args.command == "show":
        asyncio.run(show(args.document_id))
    elif args.command == "delete":
        asyncio.run(delete(args.document_id))


if __name__ == "__main__":
    main()
