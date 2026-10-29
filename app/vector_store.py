"""Embeds chunks and stores them in a persistent Chroma collection."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import chromadb

from app.chunking import Chunk
from app.config import DATA_DIR, EMBEDDING_MODEL, TOP_K


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        # Imported lazily: torch is slow to load and tests use a fake embedder.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


@lru_cache
def default_embedder() -> Embedder:
    return SentenceTransformerEmbedder()


@dataclass
class SearchResult:
    text: str
    page: int
    document: str
    score: float  # cosine similarity, higher is better
    page_end: int | None = None

    def __post_init__(self):
        if self.page_end is None:
            self.page_end = self.page

    @property
    def pages(self) -> str:
        return str(self.page) if self.page == self.page_end else f"{self.page}-{self.page_end}"


def _result(text: str, meta: dict, score: float = 1.0) -> SearchResult:
    return SearchResult(
        text=text, page=meta["page"], document=meta["document"], score=score,
        page_end=meta.get("page_end"),  # absent in documents indexed before page merging
    )


class VectorStore:
    def __init__(self, embedder: Embedder | None = None, path: str | None = None):
        self.embedder = embedder or default_embedder()
        client = chromadb.PersistentClient(path=path or str(DATA_DIR / "chroma"))
        self.collection = client.get_or_create_collection(
            "chunks", metadata={"hnsw:space": "cosine"}
        )

    def add(self, document: str, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self.delete(document)  # re-uploading a file replaces it
        self.collection.add(
            ids=[f"{document}::{c.index}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self.embedder.embed([c.text for c in chunks]),
            metadatas=[{"document": document, "page": c.page, "page_end": c.page_end} for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, k: int = TOP_K, document: str | None = None) -> list[SearchResult]:
        res = self.collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=k,
            where={"document": document} if document else None,
        )
        return [
            _result(text, meta, score=1 - dist)
            for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def all_chunks(self, document: str | None = None) -> list[SearchResult]:
        """Every chunk in reading order (by document, then position)."""
        res = self.collection.get(where={"document": document} if document else None)
        rows = sorted(
            zip(res["ids"], res["documents"], res["metadatas"]),
            key=lambda r: (r[2]["document"], int(r[0].rsplit("::", 1)[1])),
        )
        return [_result(text, meta) for _, text, meta in rows]

    def sample(self, document: str, n: int = 8) -> list[SearchResult]:
        """Evenly spaced chunks across a document, used to generate quizzes."""
        rows = self.all_chunks(document)
        step = max(1, len(rows) // n)
        return rows[::step][:n]

    def documents(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["document"] for m in metas})

    def delete(self, document: str) -> None:
        self.collection.delete(where={"document": document})
