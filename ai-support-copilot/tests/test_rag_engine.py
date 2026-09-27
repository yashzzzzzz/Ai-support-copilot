import os

from copilot.ingest import Chunk
from copilot.rag_engine import answer_question
from copilot.vector_store import VectorStore


def _store_with_faq():
    chunks = [
        Chunk(text="To reset a password open the admin console and send a reset link.", source="faq.md", chunk_index=0),
        Chunk(text="Cancel a subscription from the billing tab of the admin console.", source="faq.md", chunk_index=1),
    ]
    store = VectorStore()
    store.index(chunks)
    return store


def test_extractive_mode_used_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = _store_with_faq()

    result = answer_question("How do I reset a password?", store)

    assert result.mode == "extractive"
    assert "reset" in result.answer.lower()
    assert result.sources == ["faq.md"]


def test_no_matching_chunks_returns_fallback_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = VectorStore()
    store.index([Chunk(text="zzz completely unrelated zzz", source="faq.md", chunk_index=0)])

    # A query using entirely different vocabulary should score at/near zero.
    result = answer_question("quantum thermodynamics", store, top_k=1)

    assert result.mode == "extractive"
    assert result.answer  # never empty, even on a poor match
