"""Split page text into overlapping chunks that remember which page they came from."""

import re
from dataclasses import dataclass

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.pdf_loader import Page


@dataclass
class Chunk:
    text: str
    page: int
    index: int  # position within the document


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


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for piece in split_text(page.text):
            chunks.append(Chunk(text=piece, page=page.number, index=len(chunks)))
    return chunks
