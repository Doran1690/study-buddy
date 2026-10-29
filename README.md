# 📚 Study Buddy: chat with your course material

Upload lecture notes or course PDFs, ask questions in **Hebrew or English**, and get answers
grounded in your material with **page-level citations**. Generate **practice exams** from
any document with one click.

Built as a Retrieval-Augmented Generation (RAG) system with FastAPI, Chroma, multilingual
sentence embeddings, and Claude.

<!-- TODO: add a screenshot / GIF here -->

## How it works

```
             ┌──────────── Indexing (once per PDF) ────────────┐
PDF ──► PyMuPDF ──► page-aware chunking ──► multilingual embeddings ──► Chroma
                                                                         │
             ┌──────────── Question time ─────────────┐                  │
Question ──► embed ──► top-k similarity search ◄─────────────────────────┘
                              │
                              ▼
            numbered excerpts + question ──► Claude ──► answer with [1][2] citations
```

1. **Extraction**: text is pulled from each page and the page number is kept.
2. **Chunking**: pages are split into ~1000-character chunks with overlap, preferring
   paragraph and sentence boundaries so ideas aren't cut mid-thought.
3. **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` maps Hebrew and English
   text into the same vector space, so a Hebrew question can match English notes.
4. **Retrieval**: cosine similarity search in Chroma returns the top 5 chunks.
5. **Generation**: Claude answers only from the numbered excerpts, cites them inline,
   and says so when the material doesn't contain the answer.
6. **Quiz mode**: chunks sampled evenly across the document are sent to Claude with a
   JSON schema (structured outputs), so the response is always a valid quiz.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Async, typed request validation, auto docs at `/docs` |
| PDF parsing | PyMuPDF | Fast and keeps page boundaries |
| Embeddings | sentence-transformers (multilingual) | Runs locally and free, supports Hebrew |
| Vector DB | Chroma (persistent) | Zero-ops and metadata filtering by document |
| LLM | Claude API | Grounded answers and structured JSON output |
| Frontend | Vanilla HTML/JS (RTL) | No build step |
| Tests | pytest with fake embedder and fake LLM | Fast and run without an API key |

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your ANTHROPIC_API_KEY in .env
uvicorn app.main:app --reload
```

Open http://localhost:8000. Interactive API docs are at http://localhost:8000/docs.

The first upload downloads the embedding model (~470MB), so it takes a minute.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/documents` | Upload a PDF (multipart `file`) |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{name}` | Remove a document |
| `POST` | `/api/ask` | `{"question": "...", "document": null}` → answer + sources |
| `POST` | `/api/quiz` | `{"document": "...", "num_questions": 5}` → multiple-choice quiz |

## Tests

```bash
pytest
```

The tests use a deterministic fake embedder and a fake Claude client, so they run in
seconds with no network access or API key.

## Roadmap

- [ ] Stream answers token-by-token
- [ ] Highlight the cited passage inside an embedded PDF viewer
- [ ] Hybrid search (BM25 + vectors) and re-ranking
- [ ] Retrieval evaluation set (hit-rate / MRR) to compare chunking strategies
- [ ] OCR for scanned PDFs
- [ ] Docker image and cloud deployment
