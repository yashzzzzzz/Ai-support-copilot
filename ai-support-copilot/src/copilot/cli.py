"""Command-line entry point: `copilot ask "question" [--docs sample_docs]`"""

from __future__ import annotations

import click
from dotenv import load_dotenv

from .ingest import load_documents
from .rag_engine import answer_question
from .vector_store import VectorStore

load_dotenv()


@click.group()
def main() -> None:
    """AI Support Copilot CLI."""


@main.command()
@click.argument("question")
@click.option("--docs", default="sample_docs", help="Directory of documents to search.")
@click.option("--top-k", default=3, help="Number of passages to retrieve.")
def ask(question: str, docs: str, top_k: int) -> None:
    """Ask QUESTION against the documents in --docs."""
    chunks = load_documents(docs)
    store = VectorStore()
    store.index(chunks)

    result = answer_question(question, store, top_k=top_k)
    click.echo(f"\nAnswer ({result.mode} mode):\n{result.answer}")
    if result.sources:
        click.echo(f"\nSources: {', '.join(result.sources)}")


if __name__ == "__main__":
    main()
