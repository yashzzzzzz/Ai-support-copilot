"""Ingest stage: read documents and split them into overlapping chunks that
are small enough to embed and retrieve individually, but large enough to
carry useful context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_index: int


def _split_text(text: str, *, chunk_size: int = 60, overlap: int = 15) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_documents(docs_dir: str | Path, *, extensions: tuple[str, ...] = (".md", ".txt")) -> list[Chunk]:
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    all_chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*")):
        if path.suffix not in extensions:
            continue
        text = path.read_text(encoding="utf-8")
        for i, chunk_text in enumerate(_split_text(text)):
            all_chunks.append(Chunk(text=chunk_text, source=path.name, chunk_index=i))

    return all_chunks
