"""End-to-end API tests with a fake embedder and a fake Claude client (no network, no API key)."""

import hashlib
import json
import re
from types import SimpleNamespace

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import main
from app.chunking import chunk_pages
from app.rag import StudyAssistant
from app.vector_store import VectorStore


class FakeEmbedder:
    """Bag-of-words hashed into 64 dims: similar words give similar vectors."""

    def embed(self, texts):
        vectors = []
        for t in texts:
            v = [0.0] * 64
            for word in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
            norm = sum(x * x for x in v) ** 0.5 or 1
            vectors.append([x / norm for x in v])
        return vectors


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "format" in kwargs["output_config"]:
            text = json.dumps({"questions": [{
                "question": "What do plants make?", "options": ["Glucose", "Iron", "Salt", "Gold"],
                "correct_index": 0, "explanation": "Photosynthesis makes glucose.", "page": 1,
            }]})
        else:
            text = "Plants make glucose from sunlight [1]."
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


@pytest.fixture
def client(tmp_path):
    store = VectorStore(embedder=FakeEmbedder(), path=str(tmp_path))
    messages = FakeMessages()
    assistant = StudyAssistant(store, client=SimpleNamespace(beta=SimpleNamespace(messages=messages)))
    main.app.dependency_overrides[main.get_store] = lambda: store
    main.app.dependency_overrides[main.get_assistant] = lambda: assistant
    yield TestClient(main.app), messages
    main.app.dependency_overrides.clear()


def upload(c, name="bio.pdf"):
    pdf = make_pdf(["Photosynthesis turns sunlight into glucose.", "Mitochondria produce energy."])
    return c.post("/api/documents", files={"file": (name, pdf, "application/pdf")})


def test_upload_and_list(client):
    c, _ = client
    r = upload(c)
    assert r.status_code == 200
    assert r.json()["pages"] == 2
    assert c.get("/api/documents").json() == {"documents": ["bio.pdf"]}


def test_rejects_non_pdf(client):
    c, _ = client
    r = c.post("/api/documents", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_small_material_is_sent_whole(client):
    c, messages = client
    upload(c)
    body = c.post("/api/ask", json={"question": "Summarize the file"}).json()
    assert body["mode"] == "full"
    assert "glucose" in body["answer"]
    prompt = messages.calls[0]["messages"][0]["content"]
    assert "Photosynthesis" in prompt and "Mitochondria" in prompt  # both pages sent
    assert [s["id"] for s in body["sources"]] == [1]  # only the cited source is listed


def test_large_material_uses_similarity_search(client, monkeypatch):
    monkeypatch.setattr("app.config.FULL_CONTEXT_CHARS", 0)
    # Small chunks so each page becomes its own chunk.
    monkeypatch.setattr(main, "chunk_pages", lambda pages: chunk_pages(pages, size=30))
    c, messages = client
    upload(c)
    body = c.post("/api/ask", json={"question": "What does photosynthesis make from sunlight?"}).json()
    assert body["mode"] == "search"
    assert body["sources"][0]["pages"] == "1"  # most relevant chunk ranked first


def test_ask_without_documents(client):
    c, messages = client
    r = c.post("/api/ask", json={"question": "anything"})
    assert r.json()["sources"] == []
    assert messages.calls == []


def test_quiz(client):
    c, _ = client
    upload(c)
    r = c.post("/api/quiz", json={"document": "bio.pdf"})
    assert r.json()["questions"][0]["options"][0] == "Glucose"


def test_quiz_unknown_document(client):
    c, _ = client
    assert c.post("/api/quiz", json={"document": "nope.pdf"}).status_code == 404


def test_delete(client):
    c, _ = client
    upload(c)
    c.delete("/api/documents/bio.pdf")
    assert c.get("/api/documents").json() == {"documents": []}


def test_store_is_built_once_under_concurrent_requests(monkeypatch, tmp_path):
    """Regression: concurrent first requests used to open Chroma twice and crash."""
    import threading
    import time

    built = []

    def slow_store():
        time.sleep(0.2)  # like downloading the embedding model on first use
        built.append(1)
        return VectorStore(embedder=FakeEmbedder(), path=str(tmp_path))

    monkeypatch.setattr(main, "VectorStore", slow_store)
    main._build_store.cache_clear()
    try:
        results = []
        threads = [threading.Thread(target=lambda: results.append(main.get_store())) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(built) == 1
        assert len({id(s) for s in results}) == 1
    finally:
        main._build_store.cache_clear()
