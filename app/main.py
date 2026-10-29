import threading
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.chunking import chunk_pages
from app.pdf_loader import extract_pages
from app.rag import RefusedError, StudyAssistant
from app.vector_store import VectorStore

STATIC_DIR = Path(__file__).parent.parent / "static"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

app = FastAPI(title="Study Buddy", description="Chat with your course PDFs and generate practice quizzes.")


# FastAPI runs sync dependencies in a thread pool, so the first requests can arrive
# concurrently. lru_cache alone doesn't stop two threads from building the store at
# once, and Chroma fails when two clients open the same folder simultaneously.
_init_lock = threading.RLock()


@lru_cache
def _build_store() -> VectorStore:
    return VectorStore()


@lru_cache
def _build_assistant() -> StudyAssistant:
    return StudyAssistant(get_store())


def get_store() -> VectorStore:
    with _init_lock:
        return _build_store()


def get_assistant() -> StudyAssistant:
    with _init_lock:
        return _build_assistant()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document: str | None = None


class QuizRequest(BaseModel):
    document: str
    num_questions: int = Field(default=5, ge=1, le=15)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/documents")
def list_documents(store: VectorStore = Depends(get_store)):
    return {"documents": store.documents()}


@app.post("/api/documents")
async def upload_document(file: UploadFile, store: VectorStore = Depends(get_store)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large (max 30MB).")
    try:
        pages = extract_pages(data)
    except Exception:
        raise HTTPException(400, "Could not read this PDF.")
    if not pages:
        raise HTTPException(400, "No text found. Scanned PDFs need OCR first.")
    count = store.add(file.filename, chunk_pages(pages))
    return {"document": file.filename, "pages": len(pages), "chunks": count}


@app.delete("/api/documents/{name}")
def delete_document(name: str, store: VectorStore = Depends(get_store)):
    store.delete(name)
    return {"deleted": name}


@app.post("/api/ask")
def ask(req: AskRequest, assistant: StudyAssistant = Depends(get_assistant)):
    try:
        return assistant.ask(req.question, req.document)
    except RefusedError as e:
        raise HTTPException(422, str(e))


@app.post("/api/quiz")
def quiz(req: QuizRequest, assistant: StudyAssistant = Depends(get_assistant)):
    try:
        return {"questions": assistant.quiz(req.document, req.num_questions)}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except RefusedError as e:
        raise HTTPException(422, str(e))
