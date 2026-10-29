import pytest

from app.chunking import chunk_pages, split_text
from app.pdf_loader import Page


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"Sentence number {i}." for i in range(200))
    chunks = split_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert "Sentence number 0." in chunks[0]
    assert "Sentence number 199." in chunks[-1]


def test_chunks_prefer_sentence_boundaries():
    text = " ".join(f"Sentence number {i}." for i in range(200))
    for c in split_text(text, size=200, overlap=40)[:-1]:
        assert c.endswith(".")


def test_hebrew_text():
    text = "זהו משפט בעברית. " * 100
    chunks = split_text(text, size=150, overlap=20)
    assert len(chunks) > 1
    assert all("משפט" in c for c in chunks)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("abc", size=10, overlap=10)


def test_chunk_pages_keeps_page_numbers():
    pages = [Page(1, "First page. " * 200), Page(2, "Second page.")]
    chunks = chunk_pages(pages)
    assert chunks[0].page == 1
    assert chunks[-1].page == 2
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_short_pages_are_merged():
    pages = [Page(i, f"Slide {i} title.") for i in range(1, 11)]
    chunks = chunk_pages(pages, size=60)
    assert len(chunks) < 10
    assert chunks[0].page == 1 and chunks[0].page_end > 1
    assert chunks[-1].page_end == 10
    # Every slide's text survives exactly once.
    joined = "\n".join(c.text for c in chunks)
    assert all(joined.count(f"Slide {i} title.") == 1 for i in range(1, 11))


def test_long_page_flushes_pending_short_pages():
    pages = [Page(1, "Short."), Page(2, "Long sentence here. " * 100), Page(3, "End.")]
    chunks = chunk_pages(pages, size=200)
    assert (chunks[0].page, chunks[0].page_end, chunks[0].text) == (1, 1, "Short.")
    assert all(c.page == 2 for c in chunks[1:-1])
    assert (chunks[-1].page, chunks[-1].text) == (3, "End.")
