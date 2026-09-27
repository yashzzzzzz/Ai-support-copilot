"""Orchestrates retrieve -> prompt-build -> generate.

Falls back to extractive mode (return the best-matching passage verbatim,
with no LLM call) whenever no ANTHROPIC_API_KEY is configured, so the
retrieval logic is always testable without a network call or API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .vector_store import ScoredChunk, VectorStore

SYSTEM_PROMPT = (
    "You are a support assistant. Answer the user's question using ONLY the "
    "context passages provided below. If the answer is not contained in the "
    "context, say you don't have enough information. Cite the source file "
    "name(s) you used."
)


@dataclass
class RagAnswer:
    answer: str
    sources: list[str]
    mode: str  # "llm" or "extractive"


def _build_context(chunks: list[ScoredChunk]) -> str:
    return "\n\n".join(f"[Source: {c.chunk.source}]\n{c.chunk.text}" for c in chunks)


def _extractive_answer(chunks: list[ScoredChunk]) -> RagAnswer:
    if not chunks:
        return RagAnswer(answer="I don't have enough information to answer that.", sources=[], mode="extractive")
    best = chunks[0]
    return RagAnswer(answer=best.chunk.text, sources=[best.chunk.source], mode="extractive")


def _llm_answer(question: str, chunks: list[ScoredChunk], *, client) -> RagAnswer:
    context = _build_context(chunks)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    sources = sorted({c.chunk.source for c in chunks})
    return RagAnswer(answer=text, sources=sources, mode="llm")


def answer_question(question: str, store: VectorStore, *, top_k: int = 3) -> RagAnswer:
    chunks = store.search(question, top_k=top_k)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _extractive_answer(chunks)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        return _llm_answer(question, chunks, client=client)
    except Exception:
        # Any API/import failure degrades gracefully to extractive mode
        # rather than breaking the user-facing answer.
        return _extractive_answer(chunks)
