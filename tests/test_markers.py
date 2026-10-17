import uuid

import pytest

from app.models import DocumentSource
from app.services.answering.markers import GroundedAnswer, ground_answer, used_markers
from app.services.retrieval.citations import Citation


def citation(marker: str, title: str = "Invoice 2026-0042") -> Citation:
    return Citation(
        marker=marker,
        document_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        chunk_index=0,
        title=title,
        source=DocumentSource.PDF,
        locator="page 1",
        pages=(1,),
        page_count=2,
        start_char=0,
        end_char=100,
        from_ocr=False,
        ocr_confidence=None,
        similarity=0.8,
        url=None,
        quote="Total: 480.00 UAH. Payment terms: net 14 days.",
    )


THREE = (citation("[1]"), citation("[2]"), citation("[3]"))


def test_an_answer_that_cites_properly_is_left_alone():
    text = "Payment is due in 14 days [1] and the total is 480.00 UAH [2]."
    grounded = ground_answer(text, THREE)

    assert grounded.text == text
    assert grounded.invented == ()
    assert grounded.is_grounded is True


def test_cited_and_uncited_split_the_sources():
    """
    What lets a UI say "also found in" honestly instead of implying the
    answer drew on all three sources when it used two.
    """
    grounded = ground_answer("Due in 14 days [1], total 480.00 [3].", THREE)

    assert [c.marker for c in grounded.cited] == ["[1]", "[3]"]
    assert [c.marker for c in grounded.uncited] == ["[2]"]


def test_cited_keeps_retrieval_order_not_order_of_mention():
    """
    The panel is ranked by how well each source matched the question. The
    model mentioning the third one first does not make it the best match,
    and reordering the list to follow the prose would quietly reshuffle
    the ranking the user is shown.
    """
    grounded = ground_answer("First [3], then [1].", THREE)

    assert [c.marker for c in grounded.cited] == ["[1]", "[3]"]


def test_markers_are_not_renumbered_when_one_is_skipped():
    """
    Citing [1] and [3] leaves a gap in the footnote list, and that gap
    stays. Closing it would mean rewriting the prose to point at
    different sources — the one thing a verification layer must not do.
    """
    grounded = ground_answer("Total 480.00 [3].", THREE)

    assert "[3]" in grounded.text
    assert [c.marker for c in grounded.cited] == ["[3]"]


def test_a_marker_for_a_source_that_does_not_exist_is_removed():
    grounded = ground_answer("The vendor is Kyiv Coffee Roasters [7].", THREE)

    assert "[7]" not in grounded.text
    assert grounded.invented == ("[7]",)
    assert grounded.is_grounded is False


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # Before punctuation: the space goes with the marker.
        ("Due in net 14 days [9].", "Due in net 14 days."),
        # Between two words: exactly one space survives.
        ("Due in [9] 14 days.", "Due in 14 days."),
        ("Paid [9] on time [8] yesterday.", "Paid on time yesterday."),
        # At the very start of the answer.
        ("[9] The total is 480.00.", "The total is 480.00."),
        # Adjacent to a real one.
        ("Total 480.00 [1][9].", "Total 480.00 [1]."),
        ("Total 480.00 [9][1].", "Total 480.00 [1]."),
    ],
)
def test_removing_a_marker_leaves_the_sentence_correctly_spaced(answer, expected):
    """
    "daysand" and "days ." are both obviously broken to a reader, and a
    visibly broken answer is not a trustworthy one. The spacing is
    decided at the match, where both sides are known.
    """
    assert ground_answer(answer, THREE).text == expected


def test_marker_zero_is_invented():
    """Markers are 1-based; [0] resolves to nothing."""
    grounded = ground_answer("The total is 480.00 [0].", THREE)

    assert grounded.invented == ("[0]",)
    assert grounded.text == "The total is 480.00."


def test_an_answer_citing_nothing_is_reported_as_ungrounded():
    """
    Not an error — "I could not find that in your documents" is the
    correct answer to an unanswerable question and cites nothing. But the
    caller has to know, so it does not describe the result as verifiable
    or draw a sources panel beside prose that references none.
    """
    grounded = ground_answer("I could not find that in your documents.", THREE)

    assert grounded.is_grounded is False
    assert grounded.cited == ()
    assert len(grounded.uncited) == 3
    assert grounded.invented == ()


def test_an_invented_marker_is_reported_once_however_often_it_appears():
    grounded = ground_answer("Both [9] and again [9] and once more [9].", THREE)

    assert grounded.invented == ("[9]",)


def test_the_comma_form_is_normalised_rather_than_dropped():
    """
    Models write "[1, 2]" however the prompt is worded.
    """
    grounded = ground_answer("Both sources agree [1, 2].", THREE)

    assert grounded.text == "Both sources agree [1][2]."
    assert [c.marker for c in grounded.cited] == ["[1]", "[2]"]
    assert grounded.invented == ()


def test_a_comma_form_with_one_bad_number_keeps_only_the_good_one():
    grounded = ground_answer("Sources agree [2, 9].", THREE)

    assert grounded.text == "Sources agree [2]."
    assert [c.marker for c in grounded.cited] == ["[2]"]
    assert grounded.invented == ("[9]",)


def test_a_comma_form_of_entirely_bad_numbers_is_removed():
    grounded = ground_answer("Sources agree [8, 9].", THREE)

    assert grounded.text == "Sources agree."
    assert grounded.invented == ("[8]", "[9]")


def test_whitespace_inside_a_marker_is_tolerated():
    assert ground_answer("Total 480.00 [ 1 ].", THREE).text == "Total 480.00 [1]."


def test_used_markers_reads_both_forms_in_order_of_appearance():
    assert used_markers("First [2], then [1, 3], then [2] again.") == (
        "[2]",
        "[1]",
        "[3]",
    )


def test_grounding_an_already_grounded_answer_changes_nothing():
    once = ground_answer("Due in 14 days [1], total 480.00 [9].", THREE)
    twice = ground_answer(once.text, THREE)

    assert twice.text == once.text
    assert twice.invented == ()


def test_a_reconciled_answer_is_immutable():
    """
    Pins the design decision, not the language feature.
    """
    assert GroundedAnswer.__dataclass_params__.frozen, (
        "GroundedAnswer must stay frozen: ground_answer's guarantee is that "
        "every marker in `text` resolves, and editing either field afterwards "
        "silently breaks it"
    )



def test_the_result_is_frozen():
    """An answer that has been reconciled should not then be edited."""
    grounded = ground_answer("Total [1].", THREE)
    assert isinstance(grounded, GroundedAnswer)
    with pytest.raises((AttributeError, TypeError)):
        grounded.text = "something else"  # type: ignore[misc]
