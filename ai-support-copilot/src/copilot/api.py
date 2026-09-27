"""Flask API so the copilot can be embedded in a client's existing tools
(Slack bot, internal support portal, etc.)."""

from __future__ import annotations

from flask import Flask, jsonify, request

from .ingest import load_documents
from .rag_engine import answer_question
from .vector_store import VectorStore

app = Flask(__name__)

_store = VectorStore()
_docs_loaded = False


def _ensure_index(docs_dir: str = "sample_docs") -> None:
    global _docs_loaded
    if not _docs_loaded:
        chunks = load_documents(docs_dir)
        _store.index(chunks)
        _docs_loaded = True


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"error": "field 'question' is required"}), 400

    _ensure_index()
    result = answer_question(question, _store)
    return jsonify({"answer": result.answer, "sources": result.sources, "mode": result.mode})


if __name__ == "__main__":
    app.run(debug=True)
