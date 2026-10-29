from app.pdf_loader import Page, dedupe_lines, remove_boilerplate


def test_dedupe_shadow_lines():
    # Real pattern from a slide deck exported with text shadows.
    text = "מורכבות\nמורכבות\n/ \n /\n?\n?מורכבות\nמורכבות\n1\n1\n2\n2"
    assert dedupe_lines(text) == "מורכבות\n/ \n?\n?מורכבות\n1\n2"


def test_dedupe_keeps_non_adjacent_repeats():
    assert dedupe_lines("a\nb\na") == "a\nb\na"


def test_remove_boilerplate_footer():
    pages = [Page(i, f"Topic {i}\nSE intro - Copyright Someone\n{i}") for i in range(1, 7)]
    cleaned = remove_boilerplate(pages)
    assert all("Copyright" not in p.text for p in cleaned)
    assert cleaned[2].text == "Topic 3\n3"  # content and page numbers stay


def test_remove_boilerplate_skips_short_documents():
    pages = [Page(1, "Footer"), Page(2, "Footer")]
    assert remove_boilerplate(pages) == pages
