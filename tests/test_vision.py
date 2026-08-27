"""DescriptionPool: incremental submission, failure isolation, ordering.

The pool is the piece that lets figure description overlap OCR, so its contract is
tested directly rather than only through the pipeline: submissions arrive in waves
while earlier work is still in flight.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from pdf2md.ocr import Figure
from pdf2md.vision import DescriptionPool


def fig(page: int, index: int = 1, caption: str = "") -> Figure:
    name = f"page_{page:04d}_fig{index:02d}_chart.png"
    return Figure(
        page=page,
        index=index,
        category="chart",
        path=Path(name),
        link=f"figures/{name}",
        caption=caption,
        box_norm=[0, 0, 10, 10],
        box_px=[0, 0, 10, 10],
    )


class FakeDescriber:
    """Stands in for VisionChatClient; records calls, optionally blocks or fails."""

    def __init__(self, *, fail_on: set[str] | None = None, delay: float = 0.0) -> None:
        self.fail_on = fail_on or set()
        self.delay = delay
        self.calls: list[tuple[str, str, str]] = []
        self.gate = threading.Event()
        self.gate.set()
        self._lock = threading.Lock()

    def describe(self, image: Path, prompt: str, context: str = "") -> str:
        self.gate.wait(timeout=30)
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.calls.append((image.name, prompt, context))
        if image.name in self.fail_on:
            raise RuntimeError("429 rate limit")
        return f"deskripsi {image.name}"


def test_pool_describes_every_submitted_figure():
    describer = FakeDescriber()
    pool = DescriptionPool(describer, "jelaskan")

    pool.submit([fig(1), fig(2)])
    figures = pool.join()

    assert [f.description for f in figures] == [
        "deskripsi page_0001_fig01_chart.png",
        "deskripsi page_0002_fig01_chart.png",
    ]
    assert all(not f.error for f in figures)


def test_pool_accepts_submissions_in_waves_while_working():
    # This is the overlap: OCR hands over page 1's figures, then page 2's, while the
    # first batch is still in flight.
    describer = FakeDescriber()
    describer.gate.clear()  # hold workers so the second wave lands mid-flight
    pool = DescriptionPool(describer, "jelaskan", concurrency=2)

    pool.submit([fig(1)])
    assert pool.submitted == 1
    pool.submit([fig(2), fig(3)])
    assert pool.submitted == 3

    describer.gate.set()
    figures = pool.join()

    assert len(figures) == 3
    assert all(f.description for f in figures)


def test_pool_returns_figures_in_page_order_not_completion_order():
    # Later pages finish first; the manifest must still read top-to-bottom.
    class Reversed(FakeDescriber):
        def describe(self, image: Path, prompt: str, context: str = "") -> str:
            page = int(image.name.split("_")[1])
            time.sleep(0.02 * (4 - page))
            return f"deskripsi {page}"

    pool = DescriptionPool(Reversed(), "jelaskan", concurrency=4)
    pool.submit([fig(3), fig(1), fig(2)])

    assert [f.page for f in pool.join()] == [1, 2, 3]


def test_pool_orders_multiple_figures_within_a_page():
    pool = DescriptionPool(FakeDescriber(), "jelaskan", concurrency=3)
    pool.submit([fig(2, 2), fig(1, 2), fig(2, 1), fig(1, 1)])

    assert [(f.page, f.index) for f in pool.join()] == [(1, 1), (1, 2), (2, 1), (2, 2)]


def test_one_failing_figure_does_not_sink_the_others():
    describer = FakeDescriber(fail_on={"page_0002_fig01_chart.png"})
    pool = DescriptionPool(describer, "jelaskan", concurrency=2)
    pool.submit([fig(1), fig(2), fig(3)])

    figures = pool.join()

    failed = [f for f in figures if f.error]
    assert len(failed) == 1
    assert failed[0].page == 2
    assert "429 rate limit" in failed[0].error
    assert not failed[0].description
    assert [f.page for f in figures if f.description] == [1, 3]


def test_caption_is_passed_as_context():
    describer = FakeDescriber()
    pool = DescriptionPool(describer, "jelaskan")
    pool.submit([fig(1, caption="Figure 1 | Latensi kernel.")])
    pool.join()

    _, prompt, context = describer.calls[0]
    assert prompt == "jelaskan"
    assert context == "Figure 1 | Latensi kernel."


def test_progress_totals_grow_as_figures_are_discovered():
    # A tick's total is a lower bound while OCR is still submitting, so the UI must
    # be able to see the denominator rise rather than assume it is final.
    describer = FakeDescriber()
    describer.gate.clear()
    ticks: list[tuple[int, int]] = []
    pool = DescriptionPool(
        describer,
        "jelaskan",
        concurrency=2,
        progress=lambda done, total, msg: ticks.append((done, total)),
    )

    pool.submit([fig(1)])
    pool.submit([fig(2)])
    describer.gate.set()
    pool.join()

    totals = [total for _, total in ticks]
    assert totals == sorted(totals), "total must never shrink"
    assert max(totals) == 2
    # Final tick accounts for every figure.
    assert max(done for done, _ in ticks) == 2


def test_join_is_safe_with_no_submissions():
    pool = DescriptionPool(FakeDescriber(), "jelaskan")
    assert pool.join() == []


def test_describe_calls_run_in_parallel():
    """Figures are described simultaneously, not one after another.

    A Barrier of N only releases when N calls are genuinely in flight at the same
    moment. A serial pool would deadlock here and time out, so this cannot pass by
    accident.
    """
    workers = 4
    barrier = threading.Barrier(workers, timeout=10)
    live = 0
    peak = 0
    lock = threading.Lock()

    class Simultaneous:
        def describe(self, image: Path, prompt: str, context: str = "") -> str:
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            try:
                barrier.wait()
            finally:
                with lock:
                    live -= 1
            return "ok"

    pool = DescriptionPool(Simultaneous(), "jelaskan", concurrency=workers)
    pool.submit([fig(i) for i in range(1, workers + 1)])
    figures = pool.join()

    assert peak == workers
    assert all(f.description == "ok" for f in figures)


def test_concurrency_setting_caps_parallel_calls():
    # The sidebar's "Paralel gambar" must actually bound in-flight requests, or a
    # rate-limited endpoint cannot be tamed.
    limit = 2
    live = 0
    peak = 0
    lock = threading.Lock()

    class Counting:
        def describe(self, image: Path, prompt: str, context: str = "") -> str:
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.05)
            with lock:
                live -= 1
            return "ok"

    pool = DescriptionPool(Counting(), "jelaskan", concurrency=limit)
    pool.submit([fig(i) for i in range(1, 7)])
    figures = pool.join()

    assert peak <= limit
    assert len(figures) == 6
    assert all(f.description for f in figures)


def test_pool_reports_queued_before_completed():
    describer = FakeDescriber()
    describer.gate.clear()
    messages: list[str] = []
    pool = DescriptionPool(
        describer,
        "jelaskan",
        progress=lambda done, total, msg: messages.append(msg),
    )

    pool.submit([fig(1)])
    # A queued figure is visible immediately: the UI shows work pending, not silence.
    assert any("antrean" in m for m in messages)

    describer.gate.set()
    pool.join()
    assert any("selesai" in m for m in messages)
