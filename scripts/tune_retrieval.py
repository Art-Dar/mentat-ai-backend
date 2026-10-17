"""
Sweep chunk size, overlap and k against a labelled gold set.

    python scripts/tune_retrieval.py --gold eval/goldset.json --offline
    python scripts/tune_retrieval.py --gold eval/goldset.json \
        --chunk-sizes 400,600,1000,1500 --top-k 3,5,10,20
    python scripts/tune_retrieval.py --gold eval/goldset.json --json results.json

The loop structure is the whole point:

    for each (chunk_size, overlap):       ← re-chunks and RE-EMBEDS: slow, costs money
        for each k:                       ← re-queries only: fast, free
            score

--offline uses a local TF-IDF embedder instead of Voyage.

THIS SCRIPT WRITES TO THE DATABASE. It re-chunks the documents it
evaluates, repeatedly.
"""

import argparse
import asyncio
import json
import logging
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.db import AsyncSessionLocal  # noqa: E402
from app.models import Document, IngestionStatus  # noqa: E402
from app.services.evaluation.goldset import (  # noqa: E402
    GoldSetError,
    check_spans_exist,
    load_goldset,
)
from app.services.evaluation.metrics import (  # noqa: E402
    EvaluationResult,
    format_table,
    score_question,
    usable,
)
from app.services.ingestion.embedder import get_embedder  # noqa: E402
from app.services.ingestion.lexical_embedder import LexicalEmbedder  # noqa: E402
from app.services.ingestion.pipeline import rechunk_and_embed  # noqa: E402
from app.services.retrieval.search import search_by_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("tune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gold", required=True, help="path to the gold set JSON")
    parser.add_argument("--user-id", required=True, help="whose documents to evaluate")
    parser.add_argument(
        "--chunk-sizes",
        default="400,600,1000,1500",
        help="comma-separated chunk sizes to try",
    )
    parser.add_argument(
        "--overlaps",
        default=None,
        help="comma-separated overlaps; default is 15%% of each chunk size, "
        "which keeps the embedding cost comparable between settings",
    )
    parser.add_argument("--top-k", default="3,5,10,20", help="comma-separated k values")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the local TF-IDF embedder instead of the real one",
    )
    parser.add_argument("--json", default=None, help="also write results here")
    return parser.parse_args()


def _ints(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


async def _corpus(user_id: str) -> dict[str, str]:
    async with AsyncSessionLocal() as db:
        rows = (
            (
                await db.execute(
                    select(Document).where(
                        Document.user_id == user_id,
                        Document.status == IngestionStatus.COMPLETED,
                        Document.content.isnot(None),
                    )
                )
            )
            .scalars()
            .all()
        )
    return {str(d.id): (d.content or "") for d in rows}


async def _reindex(user_id: str, chunk_size: int, overlap: int, embedder) -> int:
    """Re-chunk and re-embed the whole corpus at one setting."""
    async with AsyncSessionLocal() as db:
        documents = (
            (
                await db.execute(
                    select(Document).where(
                        Document.user_id == user_id,
                        Document.status == IngestionStatus.COMPLETED,
                        Document.content.isnot(None),
                    )
                )
            )
            .scalars()
            .all()
        )

        total = 0
        for document in documents:
            chunks = await rechunk_and_embed(
                db, document, embedder, chunk_size=chunk_size, chunk_overlap=overlap
            )
            total += len(chunks)
        await db.commit()
    return total


async def _evaluate(user_id, gold, chunk_size, overlap, k, embedder, chunk_count):
    results = []
    async with AsyncSessionLocal() as db:
        for question in gold:
            started = time.perf_counter()
            hits = await search_by_text(
                db,
                question.question,
                user_id=user_id,
                embedder=embedder,
                top_k=k,
                # No floor: a similarity cut-off is a THIRD tunable, and
                # mixing it in here would make it impossible to say
                # whether a change came from k or from the threshold.
                min_similarity=-1.0,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            results.append(
                score_question(question, [hit.content for hit in hits], latency_ms=latency_ms)
            )

    return EvaluationResult(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        top_k=k,
        results=tuple(results),
        chunk_count=chunk_count,
    )


async def run(args: argparse.Namespace) -> int:
    try:
        gold = load_goldset(args.gold)
    except GoldSetError as exc:
        logger.error("gold set: %s", exc)
        return 2

    corpus = await _corpus(args.user_id)
    if not corpus:
        logger.error("no completed documents for user %s", args.user_id)
        return 2

    # Before anything is measured: a label with a typo scores zero under
    # every setting, which does not change the ranking but drags every
    # number down — and six months later nobody can tell whether recall
    # of 0.62 was the retrieval or the typo.
    unmatched = check_spans_exist(gold, corpus)
    if unmatched:
        logger.error("These gold answers match no document in the corpus:")
        for line in unmatched:
            logger.error("  %s", line)
        logger.error("")
        logger.error("Fix the labels before measuring anything.")
        return 2

    logger.info(
        "%d questions, %d documents, embedder=%s",
        len(gold),
        len(corpus),
        "lexical (OFFLINE)" if args.offline else settings.embedding_model,
    )
    logger.info("")

    chunk_sizes = _ints(args.chunk_sizes)
    overlaps = _ints(args.overlaps) if args.overlaps else None
    ks = _ints(args.top_k)

    results: list[EvaluationResult] = []
    for chunk_size in chunk_sizes:
        # Overlap scales with chunk size by default so that every setting
        # embeds roughly the same amount of text. A fixed overlap would
        # make small chunks far more expensive per document, and the
        # comparison would partly be measuring that.
        for overlap in overlaps or [int(chunk_size * 0.15)]:
            if overlap >= chunk_size:
                logger.warning("skipping size=%d overlap=%d", chunk_size, overlap)
                continue

            if args.offline:
                # Fit on the corpus chunked at THIS setting, so the
                # embedder sees the same text distribution the index
                # will hold.
                from app.services.ingestion.chunking import chunk_text

                pieces = [
                    piece.text
                    for text in corpus.values()
                    for piece in chunk_text(text, chunk_size, overlap)
                ]
                embedder = LexicalEmbedder().fit(pieces + list(corpus.values()))
            else:
                embedder = get_embedder()

            started = time.monotonic()
            chunk_count = await _reindex(args.user_id, chunk_size, overlap, embedder)
            logger.info(
                "indexed size=%d overlap=%d → %d chunks (%.1fs)",
                chunk_size,
                overlap,
                chunk_count,
                time.monotonic() - started,
            )

            for k in ks:
                results.append(
                    await _evaluate(
                        args.user_id, gold, chunk_size, overlap, k, embedder, chunk_count
                    )
                )

    logger.info("")
    logger.info(format_table(results))
    logger.info("")

    # Degenerate rows — where k is large enough to return most of the
    # library — are excluded from the conclusion. They score a perfect
    # recall by construction, so including them makes every sweep
    # recommend the largest k, which is the one answer that is certainly
    # wrong.
    candidates = [r for r in usable(results) if not r.exceeds_prompt_budget]
    if not candidates:
        candidates = usable(results)
    if not candidates:
        logger.warning(
            "Every setting retrieved most of the corpus, so recall is 1.000 "
            "by construction and this sweep has no opinion."
        )
        logger.warning(
            "The corpus is too small for these k values, or k is too large. "
            "Add more documents — including ones that do NOT answer any gold "
            "question — or sweep smaller k."
        )
        return 1

    best_mrr = max(candidates, key=lambda r: r.mrr)
    best_recall = max(candidates, key=lambda r: (r.recall, -r.mean_context_chars))
    logger.info(
        "Best MRR:    size=%d overlap=%d k=%d (MRR %.3f, recall %.3f)",
        best_mrr.chunk_size,
        best_mrr.chunk_overlap,
        best_mrr.top_k,
        best_mrr.mrr,
        best_mrr.recall,
    )
    logger.info(
        "Best recall: size=%d overlap=%d k=%d (recall %.3f, %.0f ctx chars)",
        best_recall.chunk_size,
        best_recall.chunk_overlap,
        best_recall.top_k,
        best_recall.recall,
        best_recall.mean_context_chars,
    )

    if best_mrr.misses:
        logger.info("")
        logger.info(
            "Still missed by the best setting (%d of %d):", len(best_mrr.misses), best_mrr.questions
        )
        for miss in best_mrr.misses:
            label = f" [{', '.join(miss.tags)}]" if miss.tags else ""
            logger.info("  %s%s", miss.question, label)
        logger.info("")
        logger.info("These are the questions to look at: an aggregate says how")
        logger.info("good a setting is, the misses say what to fix next.")

    tags = sorted({tag for r in results for tag in r.tags})
    if tags:
        logger.info("")
        logger.info("Recall by category for the best-MRR setting:")
        for tag in tags:
            value = best_mrr.recall_for_tag(tag)
            if value is not None:
                logger.info("  %-20s %.3f", tag, value)

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(
                [
                    {
                        "chunk_size": r.chunk_size,
                        "chunk_overlap": r.chunk_overlap,
                        "top_k": r.top_k,
                        "recall": r.recall,
                        "mrr": r.mrr,
                        "hit_at_1": r.hit_at_1,
                        "mean_context_chars": r.mean_context_chars,
                        "median_latency_ms": r.median_latency_ms,
                        "p95_latency_ms": r.p95_latency_ms,
                        "chunk_count": r.chunk_count,
                        "questions": r.questions,
                        "offline": args.offline,
                    }
                    for r in results
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("")
        logger.info("Wrote %s", args.json)

    logger.info("")
    logger.info(
        "The corpus is now chunked at size=%d overlap=%d — the LAST setting",
        results[-1].chunk_size,
        results[-1].chunk_overlap,
    )
    logger.info("tried, not the best one. Run scripts/backfill_chunks.py to put it")
    logger.info("back to whatever settings.chunk_size says.")
    if args.offline:
        logger.info("")
        logger.info("OFFLINE RUN — these numbers come from a lexical embedder, not")
        logger.info(
            "%s. Re-run without --offline before changing a setting.", settings.embedding_model
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
