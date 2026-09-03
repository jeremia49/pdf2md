"""Pipeline orchestration: render -> OCR -> describe -> substitute -> clean.

The UI never calls the stage modules directly; it calls `run_pipeline` and renders
whatever the progress callback reports. That keeps stage ordering and the failure
policy (a bad page or figure degrades the output, never aborts the run) in one
place, and makes the pipeline testable without Streamlit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import markdown as md
from .config import Settings
from .ocr import Figure, pdf_to_images, run_ocr
from .vision import DescriptionPool, VisionDescriber

STAGES = (
    ("render", "Render halaman PDF"),
    ("ocr", "OCR layout (Unlimited-OCR)"),
    ("vision", "Deskripsi gambar (vision LLM)"),
    ("substitute", "Substitusi placeholder gambar"),
    ("cleanup", "Hapus header/footer berulang"),
)
STAGE_ORDER = {key: i for i, (key, _) in enumerate(STAGES)}
STAGE_LABELS = dict(STAGES)

# `ocr` and `vision` run concurrently, so overall progress cannot be "stage index
# plus fraction of the current stage" -- a vision tick would otherwise report less
# progress than an OCR tick that already happened. Instead every stage owns a slice
# of the bar and contributes its own completion independently. Weights are rough
# wall-clock shares: OCR dominates, vision overlaps it, the text stages are cheap.
STAGE_WEIGHTS = {
    "render": 0.10,
    "ocr": 0.50,
    "vision": 0.30,
    "substitute": 0.05,
    "cleanup": 0.05,
}
# Stages that make progress at the same time; the UI must not mark one "done"
# merely because a sibling reported.
CONCURRENT_STAGES = frozenset({"ocr", "vision"})


@dataclass(slots=True)
class Progress:
    """One progress tick. `total` is 0 when a stage has no countable unit yet.

    A `vision` tick's `total` is provisional: figures are discovered page by page,
    so it grows while OCR runs.
    """

    stage: str
    label: str
    done: int
    total: int
    message: str = ""

    @property
    def stage_fraction(self) -> float:
        """Completion of this stage alone, 0..1."""
        if not self.total:
            return 0.0
        return min(1.0, self.done / self.total)


ProgressFn = Callable[[Progress], None]


class ProgressTracker:
    """Folds a stream of per-stage ticks into monotonic overall state.

    Overlapped stages report interleaved, and a `vision` total grows as figures are
    discovered, so a naive bar would jump backward. This keeps the highest fraction
    each stage has reported and never lets the overall value decrease.
    """

    def __init__(self) -> None:
        self._stage: dict[str, float] = {}
        self._overall = 0.0
        self.latest: dict[str, Progress] = {}

    def update(self, tick: Progress) -> None:
        self.latest[tick.stage] = tick
        prior = self._stage.get(tick.stage, 0.0)
        self._stage[tick.stage] = max(prior, tick.stage_fraction)
        total = sum(STAGE_WEIGHTS[key] * value for key, value in self._stage.items())
        self._overall = max(self._overall, min(1.0, total))

    def complete(self, stage: str) -> None:
        """Mark a stage fully done, e.g. one that has no countable unit."""
        self.update(Progress(stage, STAGE_LABELS[stage], 1, 1))

    @property
    def overall(self) -> float:
        return self._overall

    def is_done(self, stage: str) -> bool:
        return self._stage.get(stage, 0.0) >= 1.0

    def started(self, stage: str) -> bool:
        return stage in self._stage


@dataclass(slots=True)
class PipelineResult:
    markdown: str
    pages: list[str]
    figures: list[Figure]
    page_count: int
    described: int
    substituted: int
    cleanup: md.CleanupReport
    failures: list[str] = field(default_factory=list)
    workdir: Path | None = None

    @property
    def figure_errors(self) -> list[str]:
        return [f"{f.link}: {f.error}" for f in self.figures if f.error]


def _emit(
    progress: ProgressFn | None, stage: str, done: int, total: int, message: str = ""
) -> None:
    if progress:
        progress(Progress(stage, STAGE_LABELS[stage], done, total, message))


def run_pipeline(
    pdf_path: Path,
    workdir: Path,
    settings: Settings,
    progress: ProgressFn | None = None,
    describe: bool = True,
) -> PipelineResult:
    """Full PDF -> Markdown run. Artifacts land under `workdir`.

    When `describe` is False, the vision stage is skipped: figures are still
    detected and their placeholders remain in the markdown, but no description
    is generated. This is the "pure OCR" mode.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    pages_dir = workdir / "pages"
    figures_dir = workdir / "figures"

    # --- Stage 1a: rasterize -------------------------------------------------
    _emit(progress, "render", 0, 1, f"membaca {pdf_path.name}")
    images = pdf_to_images(pdf_path, settings.ocr.dpi, pages_dir)
    if not images:
        raise ValueError(f"{pdf_path.name} tidak punya halaman yang bisa dirender")
    _emit(
        progress,
        "render",
        len(images),
        len(images),
        f"{len(images)} halaman @ {settings.ocr.dpi} DPI",
    )

    # --- Stages 1b + 2: OCR and figure description, overlapped ---------------
    # The vision pool is opened BEFORE OCR starts, and run_ocr feeds it each page's
    # crops as they appear. On a document whose figures cluster early, this hides
    # nearly all of the description latency behind the remaining OCR.
    if describe:
        describer = VisionDescriber(
            base_url=settings.vision.base_url,
            api_key=settings.vision.api_key,
            model=settings.vision.model,
            timeout=settings.vision.timeout,
            temperature=settings.vision.temperature,
            max_tokens=settings.vision.max_tokens,
            retries=settings.vision.retries,
        )
        pool = DescriptionPool(
            describer,
            settings.vision.prompt,
            concurrency=settings.vision.concurrency,
            progress=lambda done, total, msg: _emit(
                progress, "vision", done, total, msg
            ),
        )
        on_figures = pool.submit
    else:
        # Pure OCR mode: no vision endpoint, figures stay undescribed.
        pool = None
        on_figures = None

    _emit(progress, "ocr", 0, len(images), "mengirim halaman ke Unlimited-OCR")
    if describe:
        _emit(progress, "vision", 0, 0, "menunggu gambar pertama")
    try:
        ocr = run_ocr(
            images,
            figures_dir,
            base_url=settings.ocr.base_url,
            api_key=settings.ocr.api_key,
            model=settings.ocr.model,
            max_tokens=settings.ocr.max_tokens,
            timeout=settings.ocr.timeout,
            concurrency=settings.ocr.concurrency,
            retries=settings.ocr.retries,
            figure_pad=settings.ocr.figure_pad,
            progress=lambda done, total, msg: _emit(progress, "ocr", done, total, msg),
            on_figures=on_figures,
        )
    finally:
        # Workers hold an open HTTP client; drain them even if OCR blew up, or the
        # threads outlive the run and the error surfaces with descriptions still
        # in flight.
        figures = pool.join() if pool else []

    if all(page is None for page in ocr.pages):
        raise RuntimeError(
            "Semua halaman gagal di-OCR. Periksa OCR_BASE_URL/OCR_MODEL. "
            + (ocr.failures[0] if ocr.failures else "")
        )

    described = sum(1 for f in figures if f.description)
    # done/total, not a bare count: a document with no figures must still drive the
    # vision stage to 1.0, and total=0 would read as "no progress" forever.
    if describe:
        _emit(
            progress,
            "vision",
            len(figures) or 1,
            len(figures) or 1,
            f"{described}/{len(figures)} gambar dideskripsikan"
            if figures
            else "tidak ada gambar",
        )

    # --- Stage 3: substitute placeholders -----------------------------------
    # A failed page is a marker string, not None, from here on: substitution and
    # cleanup both operate on real text.
    text_pages = [
        page if page is not None else f"<!-- halaman {i + 1} GAGAL diproses -->"
        for i, page in enumerate(ocr.pages)
    ]
    _emit(progress, "substitute", 0, len(figures) or 1, "menyisipkan deskripsi gambar")
    text_pages, substituted = md.substitute_figures(
        text_pages, figures, keep_image_link=settings.keep_image_link
    )
    # The stage is finished once substitution returns, even when `substituted` is
    # below the figure count: a figure whose placeholder never made it into the
    # markdown has nothing to replace, and that is reported separately.
    _emit(
        progress,
        "substitute",
        len(figures) or 1,
        len(figures) or 1,
        f"{substituted} placeholder diganti",
    )

    # --- Stage 4: strip repeated headers/footers -----------------------------
    total_pages = len(text_pages)
    report = md.CleanupReport(removed_lines=[], removed_count=0)
    if settings.cleanup.enabled:
        _emit(progress, "cleanup", 0, total_pages, "mendeteksi header/footer berulang")
        chrome = md.find_chrome(
            text_pages,
            min_ratio=settings.cleanup.min_ratio,
            min_pages=settings.cleanup.min_pages,
            zone_lines=settings.cleanup.zone_lines,
            extra=ocr.chrome_lines,
        )
        text_pages, report = md.strip_chrome(
            text_pages,
            chrome,
            zone_lines=settings.cleanup.zone_lines,
            drop_page_numbers=settings.cleanup.drop_page_numbers,
        )
    _emit(
        progress,
        "cleanup",
        total_pages,
        total_pages,
        f"{report.removed_count} baris header/footer dihapus",
    )

    document = md.collapse_blank_lines(
        md.PAGE_BREAK.join(p for p in text_pages if p.strip())
    )
    (workdir / "output.md").write_text(document, encoding="utf-8")
    # Manifest mirrors ocrpdf's figures.json, plus the description each figure got,
    # so a run can be inspected (or a bad description spotted) after the fact.
    (workdir / "figures.json").write_text(
        json.dumps([f.manifest_entry() for f in figures], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return PipelineResult(
        markdown=document,
        pages=text_pages,
        figures=figures,
        page_count=total_pages,
        described=described,
        substituted=substituted,
        cleanup=report,
        failures=ocr.failures,
        workdir=workdir,
    )
