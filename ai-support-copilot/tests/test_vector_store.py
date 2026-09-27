import pytest

from copilot.ingest import Chunk
from copilot.vector_store import VectorStore


def _sample_chunks():
    return [
        Chunk(text="To reset a password open the admin console and send a reset link.", source="faq.md", chunk_index=0),
        Chunk(text="Cancel a subscription from the billing tab of the admin console.", source="faq.md", chunk_index=1),
        Chunk(text="Escalate unresolved bugs to the engineering on-call channel.", source="faq.md", chunk_index=2),
    ]


def test_search_returns_most_relevant_chunk_first():
    store = VectorStore()
    store.index(_sample_chunks())

    results = store.search("How do I reset a password?", top_k=2)

    assert len(results) >= 1
    assert "reset" in results[0].chunk.text.lower()


def test_search_before_index_raises():
    store = VectorStore()
    with pytest.raises(RuntimeError):
        store.search("anything")


def test_index_empty_chunks_raises():
    store = VectorStore()
    with pytest.raises(ValueError):
        store.index([])
