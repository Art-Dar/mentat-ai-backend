import json

import pytest

from app.services.answering.answering_prompt import MAX_CONTEXT_CHARS
from app.services.evaluation.goldset import (
    GoldQuestion,
    GoldSetError,
    check_spans_exist,
    load_goldset,
    normalize,
)
from app.services.evaluation.metrics import (
    EvaluationResult,
    format_table,
    score_question,
    usable,
)


def gold(question="when is it due?", spans=("net 14 days from the invoice",), tags=()):
    return GoldQuestion(question=question, answer_spans=tuple(spans), tags=tuple(tags))


def result(hit_ranks, *, chunk_size=1000, overlap=150, k=5, chunks=200, ctx=2000, tags=()):
    return EvaluationResult(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        top_k=k,
        chunk_count=chunks,
        results=tuple(
            score_question(
                gold(question=f"q{i}", tags=tags),
                # A list of chunk texts where the one at `rank` contains
                # the span, built so the rank being measured is the rank
                # the fixture asked for.
                [
                    (
                        "net 14 days from the invoice"
                        if rank is not None and j + 1 == rank
                        else "unrelated filler text about something else entirely"
                    )
                    for j in range(k)
                ],
                latency_ms=10.0 + i,
            )
            for i, rank in enumerate(hit_ranks)
        ),
    )


@pytest.mark.parametrize(
    "chunk",
    [
        "Payment terms: net 14 days from the invoice date.",
        "Payment terms: NET 14 DAYS FROM THE INVOICE date.",
        "Payment terms: net 14 days\n   from the invoice date.",
        "Payment terms: net 14 days from the invoice\tdate.",
    ],
)
def test_a_span_matches_through_case_and_whitespace(chunk):
    """
    Chunk text keeps the document's line breaks and nobody writing a
    label reproduces them. A label that is right but scores zero drags
    every setting down equally, which does not change the ranking but
    does make the absolute numbers a lie.
    """
    assert gold().matches(chunk)


def test_unicode_is_normalised_before_matching():
    """
    A label typed with a plain hyphen against a PDF's en-dash is the
    single most common reason a correct label scores nothing.
    """
    question = gold(spans=("net 14–21 days from issue",))
    assert question.matches("terms are net 14–21 days from issue")
    assert normalize("net 14–21 days") == normalize("NET  14–21\nDAYS")


def test_a_chunk_without_the_span_does_not_match():
    assert not gold().matches("Payment terms: thirty days from month end.")


def test_spans_are_alternatives_not_a_conjunction():
    """
    The same fact is often phrased twice in a document — a table row and
    a sentence. Requiring both would punish a retrieval that found a
    perfectly good copy.
    """
    question = gold(spans=("TOTAL DUE: 8556.00", "total payable is 8556"))

    assert question.matches("... TOTAL DUE: 8556.00 UAH ...")
    assert question.matches("... the total payable is 8556 hryvnia ...")


def write(tmp_path, payload):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_valid_gold_set_loads(tmp_path):
    questions = load_goldset(
        write(
            tmp_path,
            [
                {
                    "question": "when is it due?",
                    "answer_spans": ["net 14 days from the invoice"],
                    "tags": ["invoice"],
                }
            ],
        )
    )

    assert len(questions) == 1
    assert questions[0].tags == ("invoice",)


def test_a_question_with_no_spans_is_refused(tmp_path):
    """
    Unscoreable. Counting it as a miss would understate every setting by
    the same amount, which looks like a real result.
    """
    with pytest.raises(GoldSetError, match="answer_spans"):
        load_goldset(write(tmp_path, [{"question": "when is it due?"}]))


def test_a_span_too_short_to_identify_anything_is_refused(tmp_path):
    """
    "480.00" appears in almost any invoice chunk, so it scores a hit for
    a retrieval that found nothing relevant. Refused loudly rather than
    quietly inflating recall for every setting — a wrong number that
    looks stable is worse than one that looks wrong.
    """
    with pytest.raises(GoldSetError, match="too short"):
        load_goldset(write(tmp_path, [{"question": "what is the total?", "answer_spans": ["480"]}]))


def test_a_single_span_may_be_given_as_a_string(tmp_path):
    questions = load_goldset(
        write(
            tmp_path,
            [{"question": "q", "answer_spans": "net 14 days from the invoice"}],
        )
    )
    assert questions[0].answer_spans == ("net 14 days from the invoice",)


@pytest.mark.parametrize(
    "payload", [[], [{"answer_spans": ["net 14 days from the invoice"]}], {"a": 1}]
)
def test_malformed_gold_sets_are_refused(tmp_path, payload):
    with pytest.raises(GoldSetError):
        load_goldset(write(tmp_path, payload))


def test_a_missing_file_names_itself(tmp_path):
    with pytest.raises(GoldSetError, match="no gold set"):
        load_goldset(tmp_path / "nope.json")


def test_labels_that_match_nothing_in_the_corpus_are_reported():
    """
    Run before every sweep. A typo in a label scores zero under every
    setting; six months later nobody can tell whether recall of 0.62 was
    the retrieval or the typo.
    """
    questions = (
        gold(question="real", spans=("net 14 days from the invoice",)),
        gold(question="typo", spans=("net 41 days from the invoice",)),
    )

    unmatched = check_spans_exist(
        questions, {"invoice": "Payment terms: net 14 days from the invoice date."}
    )

    assert len(unmatched) == 1
    assert "typo" in unmatched[0]


def test_recall_counts_questions_whose_answer_was_retrieved():
    assert result([1, 2, None, None]).recall == 0.5


def test_mrr_rewards_the_answer_being_near_the_top():
    """
    The metric that matters for this pipeline: the prompt is filled
    best-first and truncated, so a chunk at rank 30 may never reach the
    model even though it was "retrieved".
    """
    assert result([1, 1]).mrr == 1.0
    assert result([2, 2]).mrr == 0.5
    assert result([1, 2]).mrr == pytest.approx(0.75)


def test_a_miss_contributes_zero_to_mrr_rather_than_being_dropped():
    """
    Otherwise a setting could improve its MRR by failing more often —
    retrieve nothing except the one easy question and score 1.0.
    """
    assert result([1, None]).mrr == 0.5
    assert result([1]).mrr == 1.0


def test_hit_at_one_is_not_the_same_as_recall():
    scored = result([1, 3, 3, None])
    assert scored.hit_at_1 == 0.25
    assert scored.recall == 0.75


def test_context_cost_is_counted():
    """The speed-and-cost half of "balance relevance and speed"."""
    assert result([1], k=5).mean_context_chars > 0
    assert result([1], k=20).mean_context_chars > result([1], k=5).mean_context_chars


def test_recall_can_be_sliced_by_category():
    """
    Tuning that lifts the average while destroying one category is
    common and invisible in a single number.
    """
    scored = result([1, None], tags=("receipt",))
    assert scored.recall_for_tag("receipt") == 0.5
    assert scored.recall_for_tag("invoice") is None


def test_a_k_that_covers_the_corpus_is_marked_degenerate():
    """
    The failure that produced a wrong answer on the first real run: a
    four-document corpus chunked at 2000 held five chunks, so k=5
    retrieved the entire library and scored a perfect recall. The
    conclusion was "k retrieved everything", not "this chunk size is
    better".
    """
    assert result([1], k=5, chunks=5).degenerate is True
    assert result([1], k=5, chunks=8).degenerate is True
    assert result([1], k=5, chunks=200).degenerate is False


def test_degenerate_settings_cannot_win():
    """
    Kept in the table so the shape is visible, excluded from the
    conclusion so the answer is never "retrieve everything".
    """
    honest = result([1, None], k=5, chunks=200)
    cheating = result([1, 1], k=50, chunks=60)

    assert cheating.recall > honest.recall
    assert usable([honest, cheating]) == [honest]


def test_retrieval_larger_than_the_prompt_budget_is_flagged():
    """
    Found by reading the first real sweep: at chunk=2000 k=20 the
    retrieved context averaged 34,000 characters against a budget of
    24,000. select_citations silently drops what does not fit, so a
    third of what this harness called "retrieved" never reached the
    model.
    """
    under = result([1], ctx=1000)
    assert under.exceeds_prompt_budget is False

    over = EvaluationResult(
        chunk_size=2000,
        chunk_overlap=300,
        top_k=20,
        chunk_count=200,
        results=(score_question(gold(), ["x" * (MAX_CONTEXT_CHARS + 1000)], latency_ms=1.0),),
    )
    assert over.exceeds_prompt_budget is True


def test_the_misses_are_reported_not_only_counted():
    """An aggregate says how good a setting is; the misses say what to fix."""
    scored = result([1, None, None])

    assert len(scored.misses) == 2
    assert all(not m.hit for m in scored.misses)


def test_the_table_flags_what_cannot_be_trusted():
    table = format_table([result([1], k=5, chunks=6), result([1], k=5, chunks=400)])
    assert "k covers the corpus" in table
    assert table.count("k covers the corpus") == 1


def test_an_empty_result_does_not_divide_by_zero():
    empty = EvaluationResult(chunk_size=1000, chunk_overlap=150, top_k=5, results=())
    assert empty.recall == 0.0
    assert empty.mrr == 0.0
    assert empty.p95_latency_ms == 0.0


def test_the_first_matching_chunk_sets_the_rank_not_the_best_one():
    """
    The prompt is truncated best-first, so what matters is how early the
    answer appears. A later duplicate adds nothing.
    """
    scored = score_question(
        gold(),
        [
            "filler",
            "Payment terms: net 14 days from the invoice date.",
            "net 14 days from the invoice date, again",
        ],
        latency_ms=5.0,
    )
    assert scored.hit_rank == 2
    assert scored.reciprocal_rank == 0.5


def test_no_match_anywhere_is_a_miss_not_a_crash():
    scored = score_question(gold(), ["nothing", "relevant", "here"], latency_ms=5.0)

    assert scored.hit_rank is None
    assert scored.hit is False
    assert scored.reciprocal_rank == 0.0


def test_retrieving_nothing_is_a_miss():
    scored = score_question(gold(), [], latency_ms=5.0)
    assert scored.hit is False
    assert scored.context_chars == 0
