"""Extract text from a PDF, page by page."""

from dataclasses import dataclass

import pymupdf


@dataclass
class Page:
    number: int  # 1-based
    text: str


def extract_pages(pdf_bytes: bytes) -> list[Page]:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [Page(number=i + 1, text=page.get_text()) for i, page in enumerate(doc)]
    return [p for p in pages if p.text.strip()]
