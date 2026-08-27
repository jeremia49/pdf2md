"""End-to-end pipeline over a synthetic PDF with both endpoints stubbed.

Both network stages are replaced at their call boundary (`ocr_one` and
`VisionDescriber.describe`), so the test exercises the real rasterizer, the real
figure cropping, the real substitution, and the real cleanup.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from pdf2md import ocr as ocr_mod
from pdf2md import pipeline as pipe
from pdf2md import vision as vision_mod
from pdf2md.config import CleanupSettings, OcrSettings, Settings, VisionSettings

PAGES = 5


def _raw(page: int) -> str:
    """One page of grounded OCR output; page 2 carries a figure."""
    lines = [
        f"<|ref|>header<|/ref|><|det|>[100, 20, 900, 45]<|/det|>Unlimited OCR Works",
        f"<|ref|>text<|/ref|><|det|>[100, 120, 900, 300]<|/det|>Isi halaman {page}.",
    ]
    if page == 2:
        lines.append("<|ref|>chart<|/ref|><|det|>[200, 400, 800, 700]<|/det|>")
        lines.append(
            "<|ref|>image_caption<|/ref|><|det|>[200, 710, 800, 740]<|/det|>"
            "Figure 1 | Latensi kernel."
        )
    lines.append(f"<|ref|>page_number<|/ref|><|det|>[490, 950, 510, 970]<|/det|>{page}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(PAGES):
        page = doc.new_page()
        page.insert_text((72, 72), f"halaman {i + 1}")
    out = tmp_path / "sample.pdf"
    doc.save(out)
    doc.close()
    return out


@pytest.fixture
def settings() -> Settings:
    return Settings(
        # 100 DPI keeps the synthetic render fast; cropping is resolution-agnostic.
        ocr=OcrSettings(dpi=100, concurrency=2),
        vision=VisionSettings(api_key="test", concurrency=2),
        cleanup=CleanupSettings(min_pages=3),
    )


@pytest.fixture
def stub(monkeypatch):
    """Stub both endpoints. Returns the call log for assertions."""
    calls = {"ocr": [], "vision": []}

    def fake_ocr_one(base_url, api_key, model, image: Path, *a, **kw):
        page = int(image.stem.split("_")[-1])
        calls["ocr"].append(page)
        return _raw(page)

    def fake_describe(self, image: Path, prompt: str, context: str = ""):
        calls["vision"].append((image.name, context))
        return "Grafik latensi kernel terhadap panjang decoding, tren datar."

    monkeypatch.setattr(pipe, "run_ocr", pipe.run_ocr)  # keep the real orchestrator
    monkeypatch.setattr(ocr_mod, "ocr_one", fake_ocr_one)
    monkeypatch.setattr(vision_mod.VisionDescriber, "describe", fake_describe)
    return calls


def test_pipeline_end_to_end(pdf, tmp_path, settings, stub):
    result = pipe.run_pipeline(pdf, tmp_path / "work", settings)

    assert result.page_count == PAGES
    assert sorted(stub["ocr"]) == list(range(1, PAGES + 1))

    # Stage 2 saw exactly the one cropped figure, with its caption as context.
    assert len(result.figures) == 1
    assert len(stub["vision"]) == 1
    name, context = stub["vision"][0]
    assert name == "page_0002_fig01_chart.png"
    assert context == "Figure 1 | Latensi kernel."

    # Stage 3: the placeholder became a description.
    assert result.substituted == 1
    assert result.described == 1
    assert "Grafik latensi kernel" in result.markdown
    assert "**Deskripsi gambar (chart, halaman 2):**" in result.markdown

    # Stage 4: the repeated header and every page number are gone.
    assert "Unlimited OCR Works" not in result.markdown
    assert result.cleanup.removed_count == 2 * PAGES

    # Body text survived, in page order.
    body = [line for line in result.markdown.splitlines() if line.startswith("Isi halaman")]
    assert body == [f"Isi halaman {i}." for i in range(1, PAGES + 1)]

    # The artifacts on disk match what the UI shows.
    assert (tmp_path / "work" / "output.md").read_text(encoding="utf-8") == result.markdown
    manifest = json.loads((tmp_path / "work" / "figures.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1
    assert manifest[0]["file"] == "figures/page_0002_fig01_chart.png"
    assert manifest[0]["page"] == 2
    assert "Grafik latensi kernel" in manifest[0]["description"]
    assert manifest[0]["error"] == ""


def test_failed_page_degrades_instead_of_aborting(pdf, tmp_path, settings, monkeypatch):
    def flaky(base_url, api_key, model, image: Path, *a, **kw):
        page = int(image.stem.split("_")[-1])
        if page == 3:
            raise ocr_mod.FatalRequestError("400: bad request")
        return _raw(page)

    monkeypatch.setattr(ocr_mod, "ocr_one", flaky)
    monkeypatch.setattr(
        vision_mod.VisionDescriber, "describe", lambda self, i, p, context="": "desc"
    )

    result = pipe.run_pipeline(pdf, tmp_path / "work", settings)

    assert len(result.failures) == 1
    assert "halaman 3" in result.failures[0]
    assert "GAGAL diproses" in result.markdown
    assert "Isi halaman 4." in result.markdown


def test_failed_figure_keeps_the_document_usable(pdf, tmp_path, settings, monkeypatch):
    monkeypatch.setattr(ocr_mod, "ocr_one", lambda b, k, m, image, *a, **kw: _raw(
        int(image.stem.split("_")[-1])
    ))

    def boom(self, image, prompt, context=""):
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr(vision_mod.VisionDescriber, "describe", boom)

    result = pipe.run_pipeline(pdf, tmp_path / "work", settings)

    assert result.described == 0
    assert result.substituted == 1
    assert "429 rate limit" in result.markdown
    assert result.figure_errors


def test_all_pages_failing_raises(pdf, tmp_path, settings, monkeypatch):
    monkeypatch.setattr(
        ocr_mod,
        "ocr_one",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("connection refused")),
    )

    with pytest.raises(RuntimeError, match="Semua halaman gagal"):
        pipe.run_pipeline(pdf, tmp_path / "work", settings)


def test_progress_covers_every_stage_and_reaches_one(pdf, tmp_path, settings, stub):
    ticks: list[pipe.Progress] = []
    pipe.run_pipeline(pdf, tmp_path / "work", settings, progress=ticks.append)

    assert {t.stage for t in ticks} == set(pipe.STAGE_ORDER)
    assert all(0.0 <= t.stage_fraction <= 1.0 for t in ticks)

    # Every stage must end fully complete, or the bar stalls below 100%.
    tracker = pipe.ProgressTracker()
    for tick in ticks:
        tracker.update(tick)
    assert tracker.overall == pytest.approx(1.0)
    assert all(tracker.is_done(stage) for stage in pipe.STAGE_ORDER)

    # OCR ticks count pages monotonically: 1/5 .. 5/5.
    ocr_ticks = [t for t in ticks if t.stage == "ocr" and t.done]
    assert [t.done for t in ocr_ticks] == list(range(1, PAGES + 1))
    assert all(t.total == PAGES for t in ocr_ticks)


def test_overall_progress_never_decreases_under_overlap(pdf, tmp_path, settings, stub):
    # OCR and vision report interleaved, and the vision denominator grows as figures
    # are discovered, so a naive bar would jump backward mid-run.
    tracker = pipe.ProgressTracker()
    seen: list[float] = []

    def record(tick: pipe.Progress) -> None:
        tracker.update(tick)
        seen.append(tracker.overall)

    pipe.run_pipeline(pdf, tmp_path / "work", settings, progress=record)

    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


def test_vision_starts_before_ocr_finishes(pdf, tmp_path, settings, monkeypatch):
    """The point of the overlap: a figure is described while OCR is still running.

    Page 2 carries the figure and returns immediately; pages 3+ block until the
    description has been requested. If the pipeline still batched descriptions after
    OCR, those pages would block forever and the test would time out.
    """
    described = threading.Event()
    ocr_done: list[int] = []

    def slow_ocr(base_url, api_key, model, image: Path, *a, **kw):
        page = int(image.stem.split("_")[-1])
        if page > 2:
            # Wait for the vision call triggered by page 2's figure.
            assert described.wait(timeout=30), "vision never started during OCR"
        ocr_done.append(page)
        return _raw(page)

    def fake_describe(self, image, prompt, context=""):
        described.set()
        return "deskripsi"

    monkeypatch.setattr(ocr_mod, "ocr_one", slow_ocr)
    monkeypatch.setattr(vision_mod.VisionDescriber, "describe", fake_describe)

    result = pipe.run_pipeline(pdf, tmp_path / "work", settings)

    assert described.is_set()
    assert result.described == 1
    assert len(ocr_done) == PAGES


def test_figures_are_returned_in_page_order(pdf, tmp_path, settings, monkeypatch):
    # Descriptions complete out of order under concurrency; the manifest and UI list
    # must still read top-to-bottom through the document.
    def every_page_has_a_figure(base_url, api_key, model, image: Path, *a, **kw):
        page = int(image.stem.split("_")[-1])
        return (
            f"<|ref|>text<|/ref|><|det|>[100, 120, 900, 300]<|/det|>Isi halaman {page}.\n"
            "<|ref|>chart<|/ref|><|det|>[200, 400, 800, 700]<|/det|>\n"
        )

    def jittered(self, image, prompt, context=""):
        # Later pages finish first, so completion order is the reverse of page order.
        page = int(image.name.split("_")[1])
        time.sleep(0.02 * (PAGES - page))
        return f"deskripsi {page}"

    monkeypatch.setattr(ocr_mod, "ocr_one", every_page_has_a_figure)
    monkeypatch.setattr(vision_mod.VisionDescriber, "describe", jittered)

    result = pipe.run_pipeline(pdf, tmp_path / "work", settings)

    assert [f.page for f in result.figures] == list(range(1, PAGES + 1))
    assert result.described == PAGES
