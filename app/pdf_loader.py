"""Extract text from a PDF, page by page, and strip the noise slide decks carry."""

import re
from collections import Counter
from dataclasses import dataclass

import pymupdf


@dataclass
class Page:
    number: int  # 1-based
    text: str


def _key(line: str) -> str:
    """Line identity ignoring whitespace and punctuation, which shadow copies often shift."""
    return re.sub(r"[\W_]+", "", line)


def dedupe_lines(text: str) -> str:
    """Drop lines that repeat the previous line. Slides exported with a text shadow
    contain every text box twice, so each line shows up back to back."""
    kept: list[str] = []
    for line in text.splitlines():
        if kept and _key(line) == _key(kept[-1]):
            continue
        kept.append(line)
    return "\n".join(kept)


def remove_boilerplate(pages: list[Page], min_pages: int = 4, ratio: float = 0.5) -> list[Page]:
    """Remove lines (headers, footers, copyright notices) that appear on most pages."""
    if len(pages) < min_pages:
        return pages
    counts = Counter(k for p in pages for k in {_key(line) for line in p.text.splitlines()} if k)
    common = {k for k, n in counts.items() if n >= ratio * len(pages) and not k.isdigit()}
    return [
        Page(p.number, "\n".join(line for line in p.text.splitlines() if _key(line) not in common))
        for p in pages
    ]


def extract_pages(pdf_bytes: bytes) -> list[Page]:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [Page(number=i + 1, text=dedupe_lines(page.get_text())) for i, page in enumerate(doc)]
    pages = remove_boilerplate(pages)
    return [p for p in pages if p.text.strip()]
