"""
Run a real photo through OCR, Cloudflare R2 and Postgres, and check the
three still agree afterwards.

    poetry run python -m scripts.check_image_ocr read docs/receipt.jpg
    poetry run python -m scripts.check_image_ocr ingest docs/receipt.jpg
    poetry run python -m scripts.check_image_ocr list
    poetry run python -m scripts.check_image_ocr show <document-id>
    poetry run python -m scripts.check_image_ocr delete <document-id>

`read` is the one to reach for first: OCR only, no database, no bucket, no
credentials. It is how you find out whether a photo is readable before
deciding anything else, and it is safe to run in a loop while you work out
what lighting and angle your demo needs.

`ingest` writes to whatever STORAGE_BUCKET points at. Keep it on
mentat-dev; `delete` removes both the row and the object.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVOKED_FROM = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import argparse  # noqa: E402
import asyncio  # noqa: E402
import hashlib  # noqa: E402
import uuid  # noqa: E402

from sqlalchemy import select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.db import AsyncSessionLocal  # noqa: E402
from app.models import Document, DocumentSource, IngestionStatus, User  # noqa: E402
from app.services.ingestion.extractors.ocr import (  # noqa: E402
    DEFAULT_LANGUAGE,
    LOW_CONFIDENCE,
    MIN_CHARS,
    OcrEngineUnavailable,
    OcrError,
    extract_image_text_async,
)
from app.services.storage.base import (  # noqa: E402
    ObjectNotFound,
    StorageError,
    storage_key,
)
from app.services.storage.s3_store import S3FileStore  # noqa: E402

OK = "  ok   "
BAD = "  FAIL "
WARN = "  warn "
INFO = "       "

# Matches _IMAGE_MAGIC in app/api/v1/files.py. Duplicated rather than
# imported so this script does not drag the FastAPI app into a process that
# only needs the extractor — but if you add a format there, add it here.
_MAGIC: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"\xff\xd8\xff", "jpg", "image/jpeg"),
    (b"II*\x00", "tiff", "image/tiff"),
    (b"MM\x00*", "tiff", "image/tiff"),
)
_WEBP = ("webp", "image/webp")


def _from_invocation_dir(raw: Path) -> Path:
    return raw if raw.is_absolute() else (INVOKED_FROM / raw)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _image_format(data: bytes) -> tuple[str, str] | None:
    """(extension, media type) from the bytes — never from the filename."""
    for magic, extension, media_type in _MAGIC:
        if data.startswith(magic):
            return extension, media_type
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _WEBP
    return None


def _load(path: Path) -> tuple[bytes, str, str]:
    path = _from_invocation_dir(path)
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")
    data = path.read_bytes()
    image_format = _image_format(data)
    if image_format is None:
        raise SystemExit(
            f"{path.name} is not a PNG, JPEG, WEBP or TIFF. "
            "The endpoint checks the first bytes, not the filename, so it "
            "would answer 415 for this file too."
        )
    extension, media_type = image_format
    print(f"\nfile   {path.name}  ({len(data):,} bytes, {extension}, "
          f"sha256 {_digest(data)})")
    return data, extension, media_type


def _install_hint() -> str:
    """
    The right command for whoever is reading this.

    pytesseract is a wrapper; the engine is a binary pip cannot install, so
    the first thing anyone hits on a fresh machine is this message. Telling
    a Windows user to run apt-get wastes their next ten minutes.
    """
    if sys.platform == "win32":
        return (
            "This is the host's fault, not the photo's. Install it with:\n"
            "       winget install UB-Mannheim.TesseractOCR\n"
            "       then reopen the terminal so PATH picks it up, and check "
            "with: tesseract --version"
        )
    if sys.platform == "darwin":
        return "This is the host's fault, not the photo's. Install it with: brew install tesseract"
    return (
        "This is the host's fault, not the photo's. Install it with: "
        "sudo apt-get install -y tesseract-ocr"
    )


async def _recognise(data: bytes, language: str):
    try:
        return await extract_image_text_async(data, language=language)
    except OcrEngineUnavailable as exc:
        raise SystemExit(f"{BAD}{exc}\n{INFO}{_install_hint()}") from exc
    except OcrError as exc:
        raise SystemExit(f"{BAD}the image could not be read: {exc}") from exc


def _report(extraction) -> None:
    """What OCR made of the image, in the terms the API reports."""
    confidence = extraction.mean_confidence
    shown = f"{confidence:.1f}" if confidence is not None else "n/a"
    print(f"{OK}recognised {extraction.word_count} word(s), "
          f"{extraction.char_count} chars, mean confidence {shown}")
    print(f"{INFO}language: {extraction.language}")

    if not extraction.has_text:
        print(f"{BAD}under {MIN_CHARS} characters — the endpoint would answer 422")
        print(f"{INFO}try a sharper, straight-on photo in better light")
    elif extraction.is_low_confidence:
        print(f"{WARN}below the {LOW_CONFIDENCE} confidence threshold: accepted, "
              f"but flagged to the client as low_confidence")
    else:
        print(f"{OK}clean read — the endpoint would accept this without a warning")


def _show_text(extraction, *, boxes: int) -> None:
    print("\n--- recognised text " + "-" * 50)
    print(extraction.text or "(nothing)")
    print("-" * 70)

    if boxes and extraction.words:
        worst = sorted(extraction.words, key=lambda w: w.confidence)[:boxes]
        print(f"\nthe {len(worst)} least confident words "
              f"(these are what a citation would point at):\n")
        print(f"  {'word':24} {'conf':>6}  {'x':>5} {'y':>5} {'w':>5} {'h':>5}")
        for word in worst:
            print(f"  {word.text[:24]:24} {word.confidence:6.1f}  "
                  f"{word.left:5} {word.top:5} {word.width:5} {word.height:5}")


async def read(path: Path, language: str, boxes: int) -> None:
    """OCR only. No database, no bucket, no credentials."""
    data, _, _ = _load(path)
    extraction = await _recognise(data, language)
    _report(extraction)
    _show_text(extraction, boxes=boxes)
    print()


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


async def ingest(path: Path, email: str | None, language: str) -> None:
    data, extension, media_type = _load(path)
    print(f"bucket {settings.storage_bucket} @ {settings.storage_endpoint_url or 'AWS'}\n")

    extraction = await _recognise(data, language)
    _report(extraction)

    if not extraction.has_text:
        raise SystemExit(
            f"\n{INFO}Nothing stored. The endpoint refuses before writing, so "
            f"this script does too."
        )

    store = S3FileStore()

    async with AsyncSessionLocal() as db:
        user = await _pick_user(db, email)
        document_id = uuid.uuid4()
        key = storage_key(user.id, document_id, extension=extension)
        print(f"\n{INFO}owner {user.email}")
        print(f"{INFO}key   {key}\n")

        try:
            await store.put(key, data, content_type=media_type)
        except StorageError as exc:
            raise SystemExit(f"{BAD}could not write to the bucket: {exc}") from exc
        print(f"{OK}object written to R2")

        db.add(
            Document(
                id=document_id,
                user_id=user.id,
                source=DocumentSource.IMAGE,
                content=extraction.text,
                title=path.stem,
                status=IngestionStatus.PENDING,
                storage_key=key,
                file_size=len(data),
                original_filename=path.name,
                page_map=[
                    {
                        "index": 0,
                        "start_char": 0,
                        "end_char": len(extraction.text),
                        "needs_ocr": False,
                        "source": "ocr",
                        "language": extraction.language,
                        "word_count": extraction.word_count,
                        "mean_confidence": extraction.mean_confidence,
                    }
                ],
            )
        )
        await db.commit()
        print(f"{OK}row committed, id {document_id}")

    print("\nverifying the three agree:\n")
    await _verify(document_id, expected=data)
    _show_text(extraction, boxes=0)
    print(f"\nclean up with:\n  poetry run python -m scripts.check_image_ocr "
          f"delete {document_id}\n")


async def _verify(document_id: uuid.UUID, expected: bytes | None = None) -> bool:
    store = S3FileStore()
    good = True

    async with AsyncSessionLocal() as db:
        document = await db.get(Document, document_id)

    if document is None:
        print(f"{BAD}no document row with that id")
        return False
    if document.source != DocumentSource.IMAGE:
        print(f"{WARN}that document is a {document.source.value}, not an image")

    print(f"{OK}row found: status={document.status.value} "
          f"size={document.file_size:,} chars={len(document.content or '')}")

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
        print(f"{BAD}file_size on the row ({document.file_size}) != object ({len(fetched)})")
        good = False

    if expected is not None:
        if fetched == expected:
            print(f"{OK}bytes are identical to the file on disk")
        else:
            print(f"{BAD}bytes differ from the file on disk — corruption in transit")
            good = False

    # The stored object must still be a decodable image of the format its
    # key claims. A PDF saved under a .jpg key would serve as a broken
    # image in the browser and nothing else would notice.
    extension = document.storage_key.rsplit(".", 1)[-1]
    actual_format = _image_format(fetched)
    actual = actual_format[0] if actual_format else None
    if actual is None:
        print(f"{BAD}the stored object is not a recognisable image any more")
        good = False
    elif actual != extension:
        print(f"{BAD}key says .{extension} but the bytes are {actual}")
        good = False
    else:
        print(f"{OK}stored bytes really are a {actual}, matching the key")

    entries = document.page_map or []
    content = document.content or ""
    if len(entries) != 1:
        print(f"{BAD}expected one page_map entry for an image, found {len(entries)}")
        good = False
    else:
        entry = entries[0]
        if content[entry["start_char"]: entry["end_char"]] != content:
            print(f"{BAD}page_map offsets do not cover the stored content")
            good = False
        else:
            confidence = entry.get("mean_confidence")
            shown = f"{confidence:.1f}" if confidence is not None else "n/a"
            print(f"{OK}page_map covers the content "
                  f"({entry.get('word_count')} words, confidence {shown}, "
                  f"source={entry.get('source')})")

    try:
        url = await store.url_for(document.storage_key, expires_in=900)
        print(f"{OK}presigned URL issued (valid 15 min) — open it to see the photo:")
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
                .where(Document.source == DocumentSource.IMAGE)
                .where(Document.storage_key.is_not(None))
                .order_by(Document.created_at.desc())
                .limit(limit)
            )
        ).scalars().all()

    if not rows:
        print("\nNo image documents with a stored file yet.\n")
        return

    print(f"\n{'id':38} {'status':11} {'conf':>6} {'words':>6} {'size':>10}  title")
    for d in rows:
        entry = (d.page_map or [{}])[0]
        confidence = entry.get("mean_confidence")
        shown = f"{confidence:.1f}" if isinstance(confidence, int | float) else "n/a"
        print(f"{str(d.id):38} {d.status.value:11} {shown:>6} "
              f"{entry.get('word_count', 0):>6} {d.file_size or 0:>10,}  "
              f"{(d.title or '')[:28]}")
    print()


async def delete(document_id: uuid.UUID) -> None:
    store = S3FileStore()
    async with AsyncSessionLocal() as db:
        document = await db.get(Document, document_id)
        if document is None:
            raise SystemExit("No document with that id")
        key = document.storage_key
        # Row first, then the object — a leftover object is collectable
        # garbage, a leftover row is a document whose file 404s.
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
  python -m scripts.check_image_ocr read   docs/receipt.jpg      (no DB, no bucket)
  python -m scripts.check_image_ocr read   docs/receipt.jpg --boxes 10
  python -m scripts.check_image_ocr ingest docs/receipt.jpg
  python -m scripts.check_image_ocr ingest docs/receipt.jpg --language eng+ukr
  python -m scripts.check_image_ocr list
  python -m scripts.check_image_ocr show   <document-id>
  python -m scripts.check_image_ocr delete <document-id>

`read` needs only the tesseract binary. The rest also need the database
(docker compose up -d db, alembic upgrade head, a seeded user) and the
STORAGE_* keys in .env
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="check_image_ocr",
        description="Run a photo through OCR, object storage and the database.",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_read = sub.add_parser("read", help="OCR only — no database, no bucket")
    p_read.add_argument("path", type=Path)
    p_read.add_argument("--language", default=DEFAULT_LANGUAGE)
    p_read.add_argument(
        "--boxes",
        type=int,
        default=0,
        metavar="N",
        help="also print the N least confident words and their boxes",
    )

    p_ingest = sub.add_parser("ingest", help="push a photo through R2 and the database")
    p_ingest.add_argument("path", type=Path)
    p_ingest.add_argument("--email", help="owner; defaults to the oldest user")
    p_ingest.add_argument("--language", default=DEFAULT_LANGUAGE)

    p_list = sub.add_parser("list", help="image documents that have a stored file")
    p_list.add_argument("--limit", type=int, default=15)

    p_show = sub.add_parser("show", help="re-check one document against the bucket")
    p_show.add_argument("document_id", type=uuid.UUID)

    p_del = sub.add_parser("delete", help="remove the row and its object")
    p_del.add_argument("document_id", type=uuid.UUID)

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    # `read` deliberately does not need credentials — that is the point of it.
    if args.command in {"ingest", "show", "delete"} and not settings.storage_access_key:
        sys.exit("STORAGE_ACCESS_KEY is empty — check your .env")

    if args.command == "read":
        asyncio.run(read(args.path, args.language, args.boxes))
    elif args.command == "ingest":
        asyncio.run(ingest(args.path, args.email, args.language))
    elif args.command == "list":
        asyncio.run(listing(args.limit))
    elif args.command == "show":
        asyncio.run(show(args.document_id))
    elif args.command == "delete":
        asyncio.run(delete(args.document_id))


if __name__ == "__main__":
    main()