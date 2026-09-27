"""A dependency-light in-memory vector store.

Uses TF-IDF + cosine similarity by default so the whole retrieval pipeline is
testable offline with no API key and no network call. The `embed_fn` hook
lets you swap in real embeddings (Voyage, OpenAI, sentence-transformers) for
production without touching the retrieval logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore:
    """In-memory TF-IDF vector store. Swap `embed_fn` for real embeddings later."""

    def __init__(self, embed_fn: Callable[[list[str]], "object"] | None = None) -> None:
        self._chunks: list[Chunk] = []
        self._vectorizer = TfidfVectorizer()
        self._matrix = None
        self._embed_fn = embed_fn  # reserved for a future real-embedding backend

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot index an empty chunk list")
        self._chunks = chunks
        texts = [c.text for c in chunks]
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, *, top_k: int = 3) -> list[ScoredChunk]:
        if self._matrix is None or not self._chunks:
            raise RuntimeError("VectorStore.index() must be called before search()")

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        ranked = sorted(zip(self._chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [ScoredChunk(chunk=c, score=float(s)) for c, s in ranked[:top_k] if s > 0]
