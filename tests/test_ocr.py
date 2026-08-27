"""Stage 1 parsing: grounding markers -> markdown, figure boxes, chrome labels."""

from __future__ import annotations

from pdf2md import ocr

RAW_PAGE = (
    "<|ref|>header<|/ref|><|det|>[100, 20, 900, 45]<|/det|>Unlimited OCR Works\n"
    "<|ref|>title<|/ref|><|det|>[100, 60, 900, 110]<|/det|>3.4.3. Kernel study\n"
    "<|ref|>text<|/ref|><|det|>[100, 120, 900, 300]<|/det|>As shown in Figure 3, we plot\n"
    "the per-call duration.\n"
    "<|ref|>chart<|/ref|><|det|>[499, 601, 878, 804]<|/det|>"
    "<|ref|>image_caption<|/ref|><|det|>[400, 810, 900, 840]<|/det|>Figure 3 | The latency.\n"
    "<|ref|>page_number<|/ref|><|det|>[490, 950, 510, 970]<|/det|>7\n"
)


def test_parse_regions_finds_every_grounded_block():
    regions = ocr.parse_regions(RAW_PAGE)
    cats = [r["category"] for r in regions]

    assert cats == ["header", "title", "text", "chart", "image_caption", "page_number"]
    assert regions[3]["box"] == (499, 601, 878, 804)


def test_remove_det_emits_image_link_for_a_linked_figure():
    links = {(499, 601, 878, 804): "figures/page_0007_fig01_chart.png"}

    out = ocr.remove_det(RAW_PAGE, links)

    assert "![chart](figures/page_0007_fig01_chart.png)" in out
    assert "<|det|>" not in out and "<|ref|>" not in out
    # A wrapped block keeps its continuation line inside the same block.
    assert "As shown in Figure 3, we plot\nthe per-call duration." in out


def test_remove_det_drops_unlinked_figure_region():
    out = ocr.remove_det(RAW_PAGE, {})
    assert "![" not in out
    # The caption still survives as ordinary text.
    assert "Figure 3 | The latency." in out


def test_chrome_text_reports_model_labelled_header_and_page_number():
    assert ocr.chrome_text(ocr.parse_regions(RAW_PAGE)) == {"Unlimited OCR Works", "7"}


def test_denormalize_scales_per_axis():
    assert ocr.denormalize((0, 0, 1000, 1000), 800, 1200) == (0, 0, 800, 1200)
    assert ocr.denormalize((500, 250, 750, 500), 800, 1200) == (400, 300, 600, 600)


def test_ungrounded_output_still_yields_blocks():
    # Some pages come back with no markers at all; blank lines must delimit.
    out = ocr.remove_det("baris satu\nbaris dua\n\nparagraf dua\n")
    assert out == "baris satu\nbaris dua\n\nparagraf dua"


def test_crop_figures_writes_one_png_per_figure(tmp_path):
    from PIL import Image

    page = tmp_path / "page_0007.png"
    Image.new("RGB", (1000, 1000), "white").save(page)

    figures = ocr.crop_figures(
        page, ocr.parse_regions(RAW_PAGE), tmp_path / "figures", 7, pad=6
    )

    assert len(figures) == 1
    fig = figures[0]
    assert fig.category == "chart"
    assert fig.page == 7
    assert fig.link == "figures/page_0007_fig01_chart.png"
    assert fig.path.is_file()
    # Caption is picked up from the neighbouring caption region below the figure.
    assert fig.caption == "Figure 3 | The latency."
    with Image.open(fig.path) as img:
        # 499..878 wide, 601..804 tall on a 1000px page, plus 6px pad each side.
        assert img.size == (391, 215)


def test_degenerate_box_is_skipped(tmp_path):
    from PIL import Image

    page = tmp_path / "p.png"
    Image.new("RGB", (100, 100), "white").save(page)
    raw = "<|ref|>image<|/ref|><|det|>[500, 500, 501, 501]<|/det|>\n"

    assert ocr.crop_figures(page, ocr.parse_regions(raw), tmp_path / "f", 1, pad=0) == []
