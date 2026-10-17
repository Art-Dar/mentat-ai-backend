import statistics
from dataclasses import dataclass

from app.services.answering.answering_prompt import MAX_CONTEXT_CHARS
from app.services.evaluation.goldset import GoldQuestion


@dataclass(frozen=True)
class QuestionResult:
    """What one question scored under one setting."""

    question: str
    # 1-based rank of the first retrieved chunk containing an answer
    # span, or None when none of them did.
    hit_rank: int | None
    retrieved: int
    context_chars: int
    latency_ms: float
    tags: tuple[str, ...] = ()

    @property
    def hit(self) -> bool:
        return self.hit_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.hit_rank is None else 1.0 / self.hit_rank


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate scores for one (chunk_size, overlap, k) setting."""

    chunk_size: int
    chunk_overlap: int
    top_k: int
    results: tuple[QuestionResult, ...]
    # Corpus shape, carried because it explains the rest: halving the
    # chunk size roughly doubles the chunk count, which is what the
    # embedding bill and the index size follow.
    chunk_count: int = 0

    @property
    def questions(self) -> int:
        return len(self.results)

    @property
    def recall(self) -> float:
        """Fraction of questions whose answer was retrieved at all."""
        if not self.results:
            return 0.0
        return sum(r.hit for r in self.results) / len(self.results)

    @property
    def mrr(self) -> float:
        """
        Mean reciprocal rank: 1.0 if the answer is always first, 0.5 if
        always second. Misses contribute zero rather than being dropped,
        so a setting cannot improve its MRR by failing more often.
        """
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    @property
    def hit_at_1(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.hit_rank == 1 for r in self.results) / len(self.results)

    @property
    def mean_context_chars(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.context_chars for r in self.results)

    @property
    def median_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return statistics.median(r.latency_ms for r in self.results)

    @property
    def p95_latency_ms(self) -> float:
        """
        The tail, because the median hides it. A user notices the slow
        question, not the average one.
        """
        if not self.results:
            return 0.0
        ordered = sorted(r.latency_ms for r in self.results)
        return ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]

    def recall_for_tag(self, tag: str) -> float | None:
        """
        Tuning that lifts the average while destroying one category is
        common and invisible in a single number — smaller chunks usually
        help precise lookups and hurt questions whose answer is spread
        over several paragraphs, and the average can stay flat while both
        move.
        """
        subset = [r for r in self.results if tag in r.tags]
        if not subset:
            return None
        return sum(r.hit for r in subset) / len(subset)

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(sorted({tag for r in self.results for tag in r.tags}))

    @property
    def exceeds_prompt_budget(self) -> bool:
        """
        True when the retrieved text is larger than the prompt can hold.
        """
        return self.mean_context_chars > MAX_CONTEXT_CHARS

    @property
    def misses(self) -> tuple[QuestionResult, ...]:
        """
        The questions this setting failed on.
        """
        return tuple(r for r in self.results if not r.hit)

    @property
    def degenerate(self) -> bool:
        """
        True when k is large enough relative to the corpus that recall is
        measuring nothing.
        At k >= chunk_count, retrieval returns the entire library and
        recall is 1.000 by construction. The failure is quiet and
        convincing
        """
        return self.chunk_count > 0 and self.top_k >= self.chunk_count / 2


def score_question(
    gold: GoldQuestion,
    retrieved_texts: list[str],
    *,
    latency_ms: float,
) -> QuestionResult:
    """
    Find the rank of the first retrieved chunk containing an answer span.
    """
    hit_rank = None
    for rank, text in enumerate(retrieved_texts, start=1):
        if gold.matches(text):
            hit_rank = rank
            break

    return QuestionResult(
        question=gold.question,
        hit_rank=hit_rank,
        retrieved=len(retrieved_texts),
        context_chars=sum(len(t) for t in retrieved_texts),
        latency_ms=latency_ms,
        tags=gold.tags,
    )


def format_table(results: list[EvaluationResult]) -> str:
    """A fixed-width comparison, because this is read in a terminal."""
    header = (
        f"{'chunk':>6} {'ovl':>4} {'k':>3} │ {'recall':>6} {'MRR':>6} "
        f"{'hit@1':>6} │ {'ctx':>7} {'p95 ms':>7} {'chunks':>7}"
    )
    lines = [header, "─" * len(header)]
    for r in results:
        if r.degenerate:
            flag = "  ← k covers the corpus"
        elif r.exceeds_prompt_budget:
            flag = "  ← over the prompt budget"
        else:
            flag = ""
        lines.append(
            f"{r.chunk_size:>6} {r.chunk_overlap:>4} {r.top_k:>3} │ "
            f"{r.recall:>6.3f} {r.mrr:>6.3f} {r.hit_at_1:>6.3f} │ "
            f"{r.mean_context_chars:>7.0f} {r.p95_latency_ms:>7.1f} "
            f"{r.chunk_count:>7}{flag}"
        )
    return "\n".join(lines)


def usable(results: list[EvaluationResult]) -> list[EvaluationResult]:
    """
    Degenerate rows are kept in the table so the shape is visible, and
    excluded from "best setting" so the answer is never "retrieve
    everything".
    """
    return [r for r in results if not r.degenerate]


__all__ = [
    "EvaluationResult",
    "QuestionResult",
    "format_table",
    "score_question",
    "usable",
]
