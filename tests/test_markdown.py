"""Stage 3 and 4 behaviour: figure substitution and running-chrome removal."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf2md import markdown as md
from pdf2md.ocr import Figure


def fig(page: int, link: str, *, description="", caption="", error="", category="image") -> Figure:
    return Figure(
        page=page,
        index=1,
        category=category,
        path=Path(link),
        link=link,
        caption=caption,
        box_norm=[0, 0, 10, 10],
        box_px=[0, 0, 10, 10],
        description=description,
        error=error,
    )


# --------------------------------------------------------------------------- #
# Stage 3: substitution
# --------------------------------------------------------------------------- #
def test_substitute_replaces_placeholder_with_description():
    pages = ["intro\n\n![image](figures/page_0001_fig01_image.png)\n\nFigure 1 | caption"]
    figures = [fig(1, "figures/page_0001_fig01_image.png", description="Diagram alur R-SWA.")]

    out, n = md.substitute_figures(pages, figures)

    assert n == 1
    assert "Diagram alur R-SWA." in out[0]
    assert "**Deskripsi gambar (image, halaman 1):**" in out[0]
    # Surrounding text is untouched.
    assert out[0].startswith("intro")
    assert out[0].endswith("Figure 1 | caption")


def test_substitute_can_drop_the_image_link():
    pages = ["![image](figures/a.png)"]
    figures = [fig(1, "figures/a.png", description="Grafik latensi.")]

    kept, _ = md.substitute_figures(pages, figures, keep_image_link=True)
    dropped, _ = md.substitute_figures(pages, figures, keep_image_link=False)

    assert "![" in kept[0]
    assert "![" not in dropped[0]
    assert "Grafik latensi." in dropped[0]


def test_substitute_marks_failed_description_instead_of_silently_dropping():
    pages = ["![image](figures/a.png)"]
    figures = [fig(1, "figures/a.png", error="429 rate limit")]

    out, n = md.substitute_figures(pages, figures)

    assert n == 1
    assert "429 rate limit" in out[0]


def test_substitute_leaves_unknown_images_alone():
    pages = ["![image](figures/missing.png)"]
    out, n = md.substitute_figures(pages, [fig(1, "figures/other.png", description="x")])
    assert n == 0
    assert out[0] == "![image](figures/missing.png)"


def test_inline_image_reference_is_not_treated_as_a_figure_block():
    # Only a line that is *entirely* an image link is a figure placeholder.
    pages = ["lihat ![image](figures/a.png) di atas"]
    out, n = md.substitute_figures(pages, [fig(1, "figures/a.png", description="d")])
    assert n == 0
    assert out[0] == pages[0]


# --------------------------------------------------------------------------- #
# Stage 4: running header/footer removal
# --------------------------------------------------------------------------- #
def _doc(n: int) -> list[str]:
    return [f"Unlimited OCR Works\n\nisi halaman {i}\n\n{i}" for i in range(1, n + 1)]


def test_repeated_header_and_page_number_are_removed():
    pages = _doc(6)

    chrome = md.find_chrome(pages)
    cleaned, report = md.strip_chrome(pages, chrome)

    assert all("Unlimited OCR Works" not in p for p in cleaned)
    assert [p.strip() for p in cleaned] == [f"isi halaman {i}" for i in range(1, 7)]
    assert report.removed_count == 12


def test_footer_with_varying_page_number_still_matches():
    pages = [f"body {i}\n\nPage {i} of 5" for i in range(1, 6)]

    chrome = md.find_chrome(pages)
    cleaned, _ = md.strip_chrome(pages, chrome, drop_page_numbers=False)

    assert all("Page" not in p for p in cleaned)


def test_body_text_matching_a_header_mid_page_survives():
    # "Unlimited OCR Works" repeats as a header AND appears mid-body on page 1.
    pages = [
        "Unlimited OCR Works\n\nintro\n\nUnlimited OCR Works adalah model.\n\nlanjutan\n\n1",
        "Unlimited OCR Works\n\nbody\n\nlagi\n\n2",
        "Unlimited OCR Works\n\nbody\n\nlagi\n\n3",
        "Unlimited OCR Works\n\nbody\n\nlagi\n\n4",
    ]

    chrome = md.find_chrome(pages)
    cleaned, _ = md.strip_chrome(pages, chrome)

    assert "Unlimited OCR Works adalah model." in cleaned[0]
    assert not cleaned[1].startswith("Unlimited OCR Works")


def test_short_document_is_not_self_triggered():
    # Two pages cannot establish a repetition pattern; nothing may be removed.
    pages = ["Judul\n\nisi a", "Judul\n\nisi b"]

    chrome = md.find_chrome(pages, min_pages=3)
    cleaned, report = md.strip_chrome(pages, chrome)

    assert not chrome
    assert report.removed_count == 0
    assert cleaned == pages


def test_model_labelled_chrome_is_trusted_without_repetition():
    pages = ["Baidu百度\n\nisi a", "lain\n\nisi b"]
    # Only page 1 has it, so frequency alone would keep it.
    chrome = md.find_chrome(pages, extra=[{"Baidu百度"}, set()])
    cleaned, report = md.strip_chrome(pages, chrome)

    assert "Baidu百度" not in cleaned[0]
    assert report.removed_count == 1


def test_markdown_content_is_protected_from_cleanup():
    table = "<table><tr><td>a</td></tr></table>"
    pages = [f"{table}\n\nbody {i}\n\n{i}" for i in range(1, 6)]

    chrome = md.find_chrome(pages)
    cleaned, _ = md.strip_chrome(pages, chrome)

    assert all(table in p for p in cleaned)


def test_figure_block_is_never_stripped_as_chrome():
    # A figure at the top of every page must survive: repeated, but content.
    block = "![image](figures/a.png)"
    pages = [f"{block}\n\nbody {i}" for i in range(1, 6)]

    chrome = md.find_chrome(pages)
    cleaned, _ = md.strip_chrome(pages, chrome)

    assert all(block in p for p in cleaned)


@pytest.mark.parametrize("line", ["7", "- 7 -", "Page 7", "7 / 14", "vii", "Halaman 12"])
def test_page_number_shapes_detected(line):
    assert md.PAGE_NUM_RE.match(line)


@pytest.mark.parametrize("line", ["3.4.2 KV cache", "Tabel 1", "R-SWA", "2025 edition"])
def test_non_page_number_lines_rejected(line):
    assert not md.PAGE_NUM_RE.match(line)


def test_collapse_blank_lines():
    assert md.collapse_blank_lines("a\n\n\n\nb") == "a\n\nb\n"
