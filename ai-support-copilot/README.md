# AI Support Copilot

A small, production-shaped Retrieval-Augmented Generation (RAG) service: it ingests a
company's own documents (FAQs, playbooks, runbooks), retrieves the most relevant passages
for a question, and asks an LLM to answer **grounded only in those passages** — so answers
are traceable back to a source document instead of hallucinated.

Built to demonstrate the kind of client-facing AI engineering a Forward Deployed AI
Engineer ships: ingest a client's own data, wire it into an LLM, expose it as an API, and
keep it testable and observable.

## Why RAG instead of just calling the LLM

Client documents change constantly and are usually private. RAG lets you answer questions
against **today's** documents without retraining anything, and — critically for a
client-facing tool — every answer can cite which document it came from.

## Architecture

```
 docs/*.md,*.txt        1. Ingest              2. Retrieve             3. Generate
 ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
 │  Raw          │ --> │  Chunk +      │ --> │  Similarity   │ --> │  LLM answers  │
 │  documents    │      │  embed        │      │  search       │      │  grounded in  │
 │               │      │  (VectorStore)│      │  top-k chunks │      │  retrieved    │
 └──────────────┘      └──────────────┘      └──────────────┘      │  passages     │
                                                                     └──────────────┘
```

- **ingest.py** — reads `.md` / `.txt` files, splits them into overlapping chunks.
- **vector_store.py** — a dependency-light in-memory vector store (TF-IDF by default,
  swappable for real embeddings) with cosine-similarity search.
- **rag_engine.py** — orchestrates retrieve → prompt-build → generate, and returns the
  answer plus the source chunks it was grounded in.
- **api.py** — a Flask API (`POST /ask`) so the copilot can be embedded in a client's
  existing tools (Slack bot, internal portal, etc.).
- **cli.py** — `copilot ask "How do I reset a customer's password?"`

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY

# Ask a question against the sample FAQ docs
python -m copilot.cli ask "How do I reset a customer's password?"

# Or run the API
python -m copilot.api
curl -X POST localhost:5000/ask -H "Content-Type: application/json" \
  -d '{"question": "How do I reset a customer'\''s password?"}'
```

If `ANTHROPIC_API_KEY` isn't set, the engine falls back to an extractive mode (returns the
most relevant passage verbatim) so the retrieval half of the system is fully testable
without any API key or network access.

## Testing

```bash
pytest -v
```

## Project layout

```
ai-support-copilot/
├── src/copilot/
│   ├── ingest.py
│   ├── vector_store.py
│   ├── rag_engine.py
│   ├── api.py
│   └── cli.py
├── sample_docs/faq.md
├── tests/
└── .github/workflows/ci.yml
```

## Roadmap

- [ ] Swap TF-IDF for real embeddings (Voyage / OpenAI / sentence-transformers)
- [ ] Persist the vector store (SQLite / pgvector) instead of in-memory
- [ ] Add conversation memory for multi-turn support chats

## License

MIT
