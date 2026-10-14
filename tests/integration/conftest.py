import pytest

from app.services.ingestion.fake_embedder import FakeEmbedder
from app.services.tagging.fake_tagger import FakeTagger


@pytest.fixture(scope="session")
def keep_rows(request) -> bool:
    return bool(request.config.getoption("--keep-rows"))


@pytest.fixture(autouse=True)
def offline_providers(request, monkeypatch):
    """
    Keep the integration tests off the network.

    process_document falls back to the REAL provider whenever an argument
    is omitted, and every call in test_pipeline.py and test_search.py
    omits `tagger`:

        await process_document(document_id, embedder=FakeEmbedder())

    so each one builds a live GroqTagger: a real HTTP client opened inside
    that test's event loop, and a real API call. On the free tier the
    rate-limited ones then sleep 3s and 6s inside the retry loop, in tests
    that are supposed to be hermetic and fast. _assign_tags swallows the
    resulting failure, so nothing in the assertions ever said so.

    pipeline_isolation in tests/conftest.py already does this for
    get_embedder, for exactly this reason. This closes the same hole for
    get_tagger, and repeats the embedder patch so a future call site that
    forgets `embedder=` cannot reach Voyage either.

    Mark a test `real_providers` to opt out — only tests that are ABOUT
    provider construction should.
    """
    if request.node.get_closest_marker("real_providers"):
        return

    monkeypatch.setattr("app.services.ingestion.pipeline.get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("app.services.ingestion.pipeline.get_tagger", lambda: FakeTagger())