"""Split page text into overlapping chunks that remember which page they came from."""

import re
from dataclasses import dataclass

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.pdf_loader import Page


@dataclass
class Chunk:
    text: str
    page: int  # first page the chunk covers
    index: int  # position within the document
    page_end: int | None = None  # last page, when short pages were merged

    def __post_init__(self):
        if self.page_end is None:
            self.page_end = self.page


def _normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on paragraph/sentence boundaries where possible, falling back to hard cuts."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = _normalize(text)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer to break at a paragraph, then a sentence, then a space.
            window = text[start:end]
            for sep in ("\n\n", ". ", "? ", "! ", "\n", " "):
                cut = window.rfind(sep)
                if cut > size // 2:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(pages: list[Page], size: int = CHUNK_SIZE) -> list[Chunk]:
    """Long pages are split; consecutive short pages (e.g. slides) are merged so each
    chunk carries enough context to be useful on its own."""
    chunks: list[Chunk] = []
    buffer: list[Page] = []

    def flush():
        if buffer:
            text = "\n\n".join(p.text for p in buffer)
            chunks.append(Chunk(text, buffer[0].number, len(chunks), buffer[-1].number))
            buffer.clear()

    for page in pages:
        text = _normalize(page.text)
        if len(text) > size:
            flush()
            for piece in split_text(text, size=size, overlap=min(CHUNK_OVERLAP, size // 4)):
                chunks.append(Chunk(piece, page.number, len(chunks)))
            continue
        if buffer and sum(len(p.text) for p in buffer) + len(text) > size:
            flush()
        buffer.append(Page(page.number, text))
    flush()
    return chunks
