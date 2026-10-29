# 📚 Study Buddy: chat with your course material

[![tests](https://github.com/Doran1690/study-buddy/actions/workflows/tests.yml/badge.svg)](https://github.com/Doran1690/study-buddy/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.12+-blue)

Upload lecture notes or course PDFs, ask questions in **Hebrew or English**, and get answers
grounded in your material with **page-level citations**. Generate **practice exams** from
any document with one click.

Built as a Retrieval-Augmented Generation (RAG) system with FastAPI, Chroma, multilingual
sentence embeddings, and Claude.

<!-- Screenshot: save it as docs/screenshot.png, then remove these comment markers:
![Study Buddy answering a question with page citations](docs/screenshot.png)
-->

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

1. **Extraction and cleaning**: text is pulled from each page, keeping the page number.
   Noise common in slide decks is removed: lines duplicated by text-shadow effects, and
   headers or footers that repeat on most pages.
2. **Chunking**: pages are split into ~1000-character chunks with overlap, preferring
   paragraph and sentence boundaries so ideas aren't cut mid-thought.
   Consecutive short pages (like slides) are merged, so a chunk never holds just a title.
3. **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` maps Hebrew and English
   text into the same vector space, so a Hebrew question can match English notes.
4. **Retrieval**, with two modes:
   - **Full context**: when the material being searched is small (under ~60K characters),
     all of it is sent in reading order. Broad questions like "summarize this file" need
     the whole document, which top-k search can't provide.
   - **Similarity search**: for larger material, cosine similarity search in Chroma
     returns the top 5 chunks.
5. **Generation**: Claude answers only from the numbered excerpts, cites them inline,
   and says so when the material doesn't contain the answer.
6. **Quiz mode**: chunks sampled evenly across one document, or across all of them, are
   sent to Claude with a JSON schema (structured outputs), so the response is always a
   valid quiz. Each question points back to its document and page.

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
| `POST` | `/api/quiz` | `{"document": null, "num_questions": 5}` → multiple-choice quiz (`null` = all documents) |

## Tests

```bash
pytest
```

The tests use a deterministic fake embedder and a fake Claude client, so they run in
seconds with no network access or API key.

## Lessons learned

Testing on real course material surfaced problems the unit tests didn't:

- **Slide decks are noisy.** An 84-slide lecture exported from PowerPoint contained every
  line twice because of a text-shadow effect, plus a copyright footer on every page. Half
  the extracted text was noise. A cleaning step (dedupe adjacent lines, drop lines that
  appear on most pages) cut it from 32K to 16K characters without losing content.
- **Top-k retrieval can't answer broad questions.** "Summarize this file" returned five
  disconnected slides. Retrieval now switches modes: small material is sent whole, and
  similarity search is used only when the material is too large for that.
- **Tiny chunks lose meaning.** One chunk per slide meant many chunks were just a title.
  Consecutive short pages are now merged into one chunk.
- **A race condition on startup.** Two requests arriving while the embedding model was
  still downloading both tried to open the Chroma database and crashed it. Initialization
  is now guarded by a lock, with a regression test that reproduces the timing.

## Roadmap

- [ ] Stream answers token-by-token
- [ ] Highlight the cited passage inside an embedded PDF viewer
- [ ] Hybrid search (BM25 + vectors) and re-ranking
- [ ] Retrieval evaluation set (hit-rate / MRR) to compare chunking strategies
- [ ] OCR for scanned PDFs
- [ ] Docker image and cloud deployment
