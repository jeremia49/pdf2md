"""Stages 3 and 4: substitute figure descriptions, strip repeated page chrome.

Stage 3 rewrites every `![alt](figures/...)` placeholder with the vision model's
description, keyed on the link path, which stage 1 guarantees to be unique per
figure.

Stage 4 removes running headers/footers. Detection is statistical and works per
page, which is why the pipeline joins pages only after cleaning: a header is a
line that repeats near the edge of most pages. That is a property of the page
list, unrecoverable once pages are concatenated.

Two match strengths, because they carry different false-positive risk:

  * exact  -- the line repeats verbatim. Safe anywhere in the edge zone.
  * folded -- the line repeats once digit runs are ignored, which is how a footer
    like "Page 7 of 14" repeats. Digit folding also collapses genuinely distinct
    body lines ("Bab 1", "Bab 2"), so it is only trusted on the very first or last
    content line of a page, where body text does not live.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .ocr import Figure

IMAGE_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<link>[^)]+)\)\s*$")

# Bare page numbers: "7", "- 7 -", "Page 7", "7 / 14", "vii".
PAGE_NUM_RE = re.compile(
    r"^(?:[-–—\s|]*)(?:page|hal|halaman|pág|pagina)?[\s.:-]*"
    r"(?:\d{1,4}(?:\s*[/|of]{1,2}\s*\d{1,4})?|[ivxlcdm]{1,7})"
    r"(?:[-–—\s|.]*)$",
    re.IGNORECASE,
)

PAGE_BREAK = "\n\n"


@dataclass(slots=True)
class CleanupReport:
    removed_lines: list[str]
    removed_count: int


@dataclass(slots=True)
class Chrome:
    """Normalized keys that identify running headers/footers."""

    exact: set[str] = field(default_factory=set)
    folded: set[str] = field(default_factory=set)

    def __bool__(self) -> bool:
        return bool(self.exact or self.folded)


def _exact_key(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def _folded_key(line: str) -> str:
    """Exact key with every digit run replaced, so "Page 3" == "Page 12"."""
    return re.sub(r"\d+", "#", _exact_key(line))


def _is_protected(line: str) -> bool:
    """Lines that must survive cleanup even if they repeat."""
    stripped = line.strip()
    if not stripped:
        return True
    # Markdown images, tables, headings, math, and list items are content.
    return stripped.startswith(("!", "|", "<table", "#", r"\[", r"\(", "-", "*", ">"))


def _content_indices(lines: list[str]) -> list[int]:
    return [i for i, ln in enumerate(lines) if ln.strip()]


def _zones(lines: list[str], zone_lines: int) -> tuple[set[int], set[int]]:
    """(edge zone, outermost) real line indices.

    Blank lines carry no content but do shift positions, so zones are measured over
    content lines and mapped back to real indices.
    """
    idx = _content_indices(lines)
    if not idx:
        return set(), set()
    zone = set(idx[:zone_lines]) | set(idx[-zone_lines:])
    return zone, {idx[0], idx[-1]}


def find_chrome(
    pages: list[str],
    *,
    min_ratio: float = 0.6,
    min_pages: int = 3,
    zone_lines: int = 3,
    extra: list[set[str]] | None = None,
) -> Chrome:
    """Lines that behave like running headers or footers.

    `extra` holds per-page lines the OCR model itself labelled header/footer/page
    number. Those are trusted with no repetition threshold, because an explicit
    label is stronger evidence than a frequency count.
    """
    chrome = Chrome()
    for page_extra in extra or []:
        for line in page_extra:
            if line.strip():
                chrome.exact.add(_exact_key(line))
                chrome.folded.add(_folded_key(line))

    usable = [p for p in pages if p]
    if len(usable) < max(2, min_pages):
        return chrome

    exact: Counter[str] = Counter()
    folded: Counter[str] = Counter()
    for page in usable:
        lines = page.splitlines()
        zone, outermost = _zones(lines, zone_lines)
        # Per-page sets, so a line repeated twice on one page counts once.
        exact.update(
            {_exact_key(lines[i]) for i in zone if not _is_protected(lines[i])}
        )
        folded.update(
            {_folded_key(lines[i]) for i in outermost if not _is_protected(lines[i])}
        )

    threshold = max(min_pages, int(len(usable) * min_ratio))
    chrome.exact |= {key for key, n in exact.items() if key and n >= threshold}
    chrome.folded |= {key for key, n in folded.items() if key and n >= threshold}
    return chrome


def strip_chrome(
    pages: list[str],
    chrome: Chrome,
    *,
    zone_lines: int = 3,
    drop_page_numbers: bool = True,
) -> tuple[list[str], CleanupReport]:
    """Drop chrome lines from each page's leading/trailing zone.

    Confining removal to the zones is what keeps a body sentence that happens to
    match a header from being deleted mid-page.
    """
    removed: Counter[str] = Counter()
    cleaned: list[str] = []

    for page in pages:
        lines = page.splitlines()
        zone, outermost = _zones(lines, zone_lines)
        keep: list[str] = []
        for i, line in enumerate(lines):
            if i in zone and not _is_protected(line):
                stripped = line.strip()
                hit = _exact_key(line) in chrome.exact
                if not hit and i in outermost:
                    hit = _folded_key(line) in chrome.folded
                if not hit and drop_page_numbers and i in outermost:
                    hit = bool(PAGE_NUM_RE.match(stripped))
                if hit:
                    removed[stripped] += 1
                    continue
            keep.append(line)
        cleaned.append("\n".join(keep).strip())

    report = CleanupReport(
        removed_lines=[f"{line}  ({n}x)" for line, n in removed.most_common()],
        removed_count=sum(removed.values()),
    )
    return cleaned, report


def figure_block(fig: Figure, *, keep_image_link: bool) -> str:
    """Markdown that replaces one figure placeholder."""
    body = fig.description.strip()
    if not body:
        # Vision call failed or returned nothing: keep the caption so the reader
        # still knows a figure stood here, and say why there is no description.
        note = f"deskripsi gagal: {fig.error}" if fig.error else "deskripsi tidak tersedia"
        body = f"*[{note}]*"

    label = f"**Deskripsi gambar ({fig.category}, halaman {fig.page}):**"
    # A transcribed table (or any multi-line answer) must start on its own line:
    # a `|---|` row glued after the bold label is not a table to any renderer.
    joiner = "\n\n" if "\n" in body else " "

    parts: list[str] = []
    if keep_image_link:
        parts.append(f"![{fig.category}]({fig.link})")
    parts.append(f"{label}{joiner}{body}")
    return "\n\n".join(parts)


def substitute_figures(
    pages: list[str],
    figures: list[Figure],
    *,
    keep_image_link: bool = True,
) -> tuple[list[str], int]:
    """Replace each `![...](link)` line with its figure block. Returns (pages, n)."""
    by_link = {fig.link: fig for fig in figures}
    replaced = 0
    out: list[str] = []

    for page in pages:
        lines: list[str] = []
        for line in page.splitlines():
            m = IMAGE_RE.match(line)
            fig = by_link.get(m.group("link")) if m else None
            if fig is None:
                lines.append(line)
                continue
            lines.append(figure_block(fig, keep_image_link=keep_image_link))
            replaced += 1
        out.append("\n".join(lines))
    return out, replaced


def collapse_blank_lines(text: str) -> str:
    """At most one blank line between blocks, so removals leave no holes."""
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
