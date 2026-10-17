"""
Re-chunk and re-embed existing documents to the current settings.

    python scripts/backfill_chunks.py --dry-run
    python scripts/backfill_chunks.py
    python scripts/backfill_chunks.py --chunk-size 600 --chunk-overlap 100
    python scripts/backfill_chunks.py --all --limit 50
"""

import argparse
import asyncio
import logging
import pathlib
import sys
import time

# Run from anywhere: the project root is two levels up from this file,
# and without this the imports below fail depending on the launcher.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.db import AsyncSessionLocal  # noqa: E402
from app.models import Chunk, Document, IngestionStatus  # noqa: E402
from app.services.ingestion.backfill import documents_needing_rechunk  # noqa: E402
from app.services.ingestion.chunking import chunk_text  # noqa: E402
from app.services.ingestion.embedder import get_embedder  # noqa: E402
from app.services.ingestion.lexical_embedder import LexicalEmbedder  # noqa: E402
from app.services.ingestion.pipeline import rechunk_and_embed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("backfill")

# voyage-4 bills per token; four characters per token is the usual
# English approximation and is close enough to tell a 2-dollar run from
# a 200-dollar one, which is the decision this number supports.
CHARS_PER_TOKEN = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"target chunk size (default: settings.chunk_size = {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"target overlap (default: {settings.chunk_overlap})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change, touch nothing, spend nothing",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="re-embed every document, including ones already at the target "
        "chunking — for when the embedding MODEL changed, not the chunking",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N documents")
    parser.add_argument(
        "--user-id", default=None, help="restrict to one user's documents"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the local lexical embedder instead of the real one — for "
        "local development and demos with no API key. The vectors are NOT "
        "comparable with production ones, so a corpus backfilled offline "
        "must be backfilled again before it is used with the real embedder",
    )
    return parser.parse_args()


def _candidates_query(args: argparse.Namespace):
    """The selection lives in the service layer, where it is tested."""
    return documents_needing_rechunk(
        args.chunk_size,
        args.chunk_overlap,
        include_all=args.all,
        user_id=args.user_id,
    )


async def plan(args: argparse.Namespace) -> None:
    """Report the size of the job without changing anything."""
    async with AsyncSessionLocal() as db:
        total = (
            await db.execute(
                select(func.count()).select_from(Document).where(
                    Document.status == IngestionStatus.COMPLETED
                )
            )
        ).scalar_one()

        documents = (await db.execute(_candidates_query(args))).scalars().all()
        if args.limit:
            documents = documents[: args.limit]

        current_chunks = (
            await db.execute(select(func.count()).select_from(Chunk))
        ).scalar_one()

        characters = sum(len(d.content or "") for d in documents)
        projected = sum(
            len(chunk_text(d.content or "", args.chunk_size, args.chunk_overlap))
            for d in documents
        )

    logger.info("Target chunking: size=%d overlap=%d", args.chunk_size, args.chunk_overlap)
    logger.info("Completed documents in the library: %d", total)
    logger.info("Documents needing work:             %d", len(documents))
    logger.info("")
    logger.info("Chunks now (whole library):         %d", current_chunks)
    logger.info("Chunks these documents would make:  %d", projected)
    logger.info("")
    # Overlap means the embedded text exceeds the document text, and by
    # how much is exactly what changes when overlap is swept — so it is
    # the number worth printing, not the raw character count.
    embedded_chars = int(characters * (1 + args.chunk_overlap / max(args.chunk_size, 1)))
    logger.info("Text to embed:        ~%s characters", f"{embedded_chars:,}")
    logger.info("Rough token estimate: ~%s tokens", f"{embedded_chars // CHARS_PER_TOKEN:,}")
    logger.info("")
    logger.info("Re-run without --dry-run to apply. This calls the embedding API")
    logger.info("and is billed; it is safe to interrupt and resume.")


async def backfill(args: argparse.Namespace) -> int:
    documents = []
    async with AsyncSessionLocal() as db:
        documents = [d.id for d in (await db.execute(_candidates_query(args))).scalars()]
    if args.limit:
        documents = documents[: args.limit]

    if not documents:
        logger.info("Nothing to do: every document is already at the target chunking.")
        return 0

    logger.info("Re-chunking %d document(s) to size=%d overlap=%d",
                len(documents), args.chunk_size, args.chunk_overlap)

    if args.offline:
        # The lexical embedder derives its weights from the corpus, so
        # it has to see the whole thing before it can embed any of it.
        async with AsyncSessionLocal() as db:
            corpus = [
                d.content or ""
                for d in (
                    await db.execute(
                        select(Document).where(Document.content.isnot(None))
                    )
                ).scalars()
            ]
        embedder = LexicalEmbedder().fit(corpus)
        logger.warning(
            "OFFLINE: embedding with the local lexical model. These vectors "
            "are not comparable with %s and must be rebuilt before the real "
            "embedder is used again.",
            settings.embedding_model,
        )
    else:
        embedder = get_embedder()

    done = failed = 0
    started = time.monotonic()

    for position, document_id in enumerate(documents, start=1):
        # A session per document, so a failure rolls back ONE document
        # and the rest of the run is unaffected. The embedding call is
        # inside the transaction: the chunks and their vectors commit
        # together or not at all, and a document is never left with
        # chunks that have no embeddings — which would make it silently
        # unsearchable while still looking COMPLETED.
        async with AsyncSessionLocal() as db:
            try:
                document = await db.get(Document, document_id)
                if document is None:
                    continue  # deleted since the plan was made

                chunks = await rechunk_and_embed(
                    db,
                    document,
                    embedder,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
                await db.commit()
                done += 1
                logger.info(
                    "  [%d/%d] %s → %d chunks",
                    position,
                    len(documents),
                    (document.title or str(document_id))[:48],
                    len(chunks),
                )
            except Exception as exc:
                await db.rollback()
                failed += 1
                logger.error(
                    "  [%d/%d] %s FAILED: %s: %s",
                    position,
                    len(documents),
                    document_id,
                    type(exc).__name__,
                    exc,
                )
                # Keep going. One document whose text upsets the
                # embedder should not strand the other four hundred —
                # and the rollback means it still has its OLD chunks, so
                # it is stale rather than broken.

    elapsed = time.monotonic() - started
    logger.info("")
    logger.info("Done: %d re-chunked, %d failed, %.1fs", done, failed, elapsed)
    if failed:
        logger.info("Failed documents kept their previous chunks and are still")
        logger.info("searchable. Re-run to retry only those.")
    return 1 if failed else 0


def main() -> int:
    args = parse_args()

    if args.chunk_overlap >= args.chunk_size:
        logger.error("overlap (%d) must be smaller than chunk size (%d)",
                     args.chunk_overlap, args.chunk_size)
        return 2

    if args.dry_run:
        asyncio.run(plan(args))
        return 0
    return asyncio.run(backfill(args))


if __name__ == "__main__":
    raise SystemExit(main())
