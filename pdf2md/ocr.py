"""Stage 1: Unlimited-OCR document parsing.

Ported from `ocrpdf/ocr_client.py`. The server-side decode recipe is not optional
-- three pieces must be right or the model returns empty output or loops forever:

  * the prompt must begin with the literal "<image>"
  * skip_special_tokens must be False (grounding tokens are part of the output)
  * ngram_size / window_size must be passed per request via vllm_xargs

Figure regions are croppable: the model grounds every block with a bounding box,
so each image/chart is saved as its own PNG for stage 2 to describe.
"""

from __future__ import annotations

import base64
import mimetypes
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# One page per request means every request is single-image, i.e. gundam (crop)
# mode with a 128-token n-gram window. The 1024 window only applies when several
# images share one request, which this client never does.
NGRAM_SIZE = 35
WINDOW_SINGLE = 128

# Region categories the model actually emits, censused over every <|det|> marker
# of a 14-page paper: header, title, text, image, image_caption, page_number,
# equation, chart, table, ref_text. Note that "chart" is a figure too -- dropping
# only "image" leaks vector charts into the text stream.
FIGURE_CATS = frozenset({"image", "chart", "figure", "diagram", "graph", "plot"})
CAPTION_CATS = frozenset(
    {"image_caption", "figure_caption", "chart_caption", "table_caption"}
)
# Categories that are page chrome by the model's own labelling. Stage 4 also
# removes chrome statistically, but a labelled header needs no statistics.
CHROME_CATS = frozenset({"header", "footer", "page_number", "page_footer", "page_header"})

# <|det|> boxes are normalized per axis to 0..1000, NOT pixels.
DET_SCALE = 1000.0

# Two grounding shapes occur in the wild: paired markers
# (<|ref|>cat<|/ref|><|det|>[x1,y1,x2,y2]<|/det|>) and the category inside
# <|det|>. Handle both.
BLOCK_RE = re.compile(
    r"^(?:<\|ref\|>(.*?)<\|/ref\|>)?\s*<\|det\|>(.*?)<\|/det\|>\s*(.*)$", re.DOTALL
)
INLINE_REF_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
INLINE_DET_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
# A <|ref|>cat<|/ref|> glued to a <|det|> is a nested BLOCK marker -- a category
# label plus a box, not prose. Its label must vanish with the box; unwrapping it
# as inline text yields junk like "image_captionFigure 3 | The latency."
NESTED_BLOCK_RE = re.compile(
    r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>", re.DOTALL
)
# Several markers can sit back-to-back on one line: a chart and its caption are
# emitted as "<|det|>chart [...]<|/det|><|det|>image_caption [...]". The leading
# <|ref|> group is optional because the category lives either there or inside
# <|det|>; matching it here also keeps it out of the previous region's text.
ANY_DET_RE = re.compile(
    r"(?:<\|ref\|>(?P<ref>[^<]*?)<\|/ref\|>)?\s*"
    r"<\|det\|>(?P<cat>[^\[\]]*?)\s*\[(?P<box>[^\]]*)\]\s*<\|/det\|>",
    re.DOTALL,
)


class FatalRequestError(RuntimeError):
    """A 4xx from the server: the request itself is wrong, so do not retry."""


@dataclass(slots=True)
class Figure:
    """One cropped figure, carried from stage 1 to stage 3."""

    page: int
    index: int
    category: str
    path: Path
    link: str
    caption: str
    box_norm: list[int]
    box_px: list[int]
    description: str = ""
    error: str = ""

    def manifest_entry(self) -> dict:
        return {
            "page": self.page,
            "category": self.category,
            "file": self.link,
            "caption": self.caption,
            "box_norm": self.box_norm,
            "box_px": self.box_px,
            "description": self.description,
            "error": self.error,
        }


@dataclass(slots=True)
class OcrResult:
    pages: list[str | None]
    figures: list[Figure]
    chrome_lines: list[set[str]]
    failures: list[str] = field(default_factory=list)


def _box(payload: str) -> tuple[int, int, int, int] | None:
    """Pull the 4 leading integers out of a <|det|> payload."""
    nums = [int(n) for n in re.findall(r"-?\d+", payload)[:4]]
    return (nums[0], nums[1], nums[2], nums[3]) if len(nums) == 4 else None


def _clean(text: str) -> str:
    """Strip grounding markup from prose.

    Nested block markers go first: a <|ref|>label<|/ref|><|det|>box<|/det|> pair is
    a category label, so both halves must go. A lone <|ref|> is real text and is
    only unwrapped.
    """
    text = NESTED_BLOCK_RE.sub("", text)
    return INLINE_DET_RE.sub("", INLINE_REF_RE.sub(r"\1", text)).strip()


def parse_regions(raw: str) -> list[dict]:
    """Every grounded region in emission order: category, box, trailing text."""
    marks = list(ANY_DET_RE.finditer(raw))
    regions = []
    for i, m in enumerate(marks):
        box = _box(m.group("box"))
        if box is None:
            continue
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        # The category lives in <|ref|> when the checkpoint emits paired markers,
        # otherwise as the leading word inside <|det|>.
        category = (m.group("ref") or m.group("cat") or "").strip().lower()
        regions.append(
            {
                "category": category,
                "box": box,
                "text": _clean(raw[m.end() : stop]),
            }
        )
    return regions


def denormalize(box, width: int, height: int) -> tuple[int, int, int, int]:
    """0..1000 per-axis box -> pixel box in an image of the given size."""
    x1, y1, x2, y2 = box
    return (
        round(x1 / DET_SCALE * width),
        round(y1 / DET_SCALE * height),
        round(x2 / DET_SCALE * width),
        round(y2 / DET_SCALE * height),
    )


def chrome_text(regions: list[dict]) -> set[str]:
    """Text of regions the model itself labelled as header/footer/page number."""
    return {
        r["text"] for r in regions if r["category"] in CHROME_CATS and r["text"]
    }


def remove_det(raw: str, figure_links: dict | None = None) -> str:
    """Drop grounding tokens, join lines in a block, blank-line between blocks.

    A figure region carries no text. When ``figure_links`` maps its box to a
    relative path, emit a Markdown image reference in its place so the document
    keeps the figure's position; otherwise drop it.
    """
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            # A blank line ends the current block. Without this, ungrounded
            # output (no <|det|> markers at all) collapses into one block.
            if cur:
                blocks.append(cur)
            cur = None
            continue
        m = BLOCK_RE.match(line)
        if m:
            ref, det, rest = m.group(1), m.group(2), m.group(3)
            category = (ref or det.split("[", 1)[0]).strip().lower()
            if cur is not None:
                blocks.append(cur)
            cur = None
            content = _clean(rest)
            if category in FIGURE_CATS:
                box = _box(det)
                link = (figure_links or {}).get(box)
                if link:
                    # Alt text is the category, never the caption: a caption can
                    # carry "]" and break the link, and it is already emitted as
                    # its own paragraph right below. Keeps parity with the
                    # existing figures.json/outputmarket artifacts.
                    blocks.append([f"![{category}]({link})"])
                if content:
                    cur = [content]
                continue
            cur = [content] if content else []
            continue
        content = _clean(line)
        if not content:
            continue
        if cur is None:
            cur = []
        cur.append(content)
    if cur is not None:
        blocks.append(cur)
    return "\n\n".join("\n".join(b) for b in blocks if b).strip()


def pdf_to_images(pdf_path: Path, dpi: int, out_dir: Path) -> list[Path]:
    import fitz  # PyMuPDF

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pages = []
    try:
        for i, page in enumerate(doc):
            out = out_dir / f"page_{i + 1:04d}.png"
            page.get_pixmap(matrix=mat).save(out)
            pages.append(out)
    finally:
        doc.close()
    return pages


def encode_image(path: Path) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


# One requests.Session shared across threads corrupts TLS records through a
# Cloudflare tunnel (observed: SSLV3_ALERT_BAD_RECORD_MAC under 4 workers). Give
# every worker its own session and drop a poisoned pool on failure.
_local = threading.local()


def _session(api_key: str) -> requests.Session:
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        if api_key:
            session.headers["Authorization"] = f"Bearer {api_key}"
        _local.session = session
    return session


def ocr_one(
    base_url: str,
    api_key: str,
    model: str,
    image: Path,
    window_size: int,
    max_tokens: int,
    timeout: int,
    retries: int = 4,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                # The literal "<image>" prefix is mandatory.
                "content": [
                    {"type": "text", "text": "<image>document parsing."},
                    encode_image(image),
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "skip_special_tokens": False,
        "vllm_xargs": {"ngram_size": NGRAM_SIZE, "window_size": window_size},
    }
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _session(api_key).post(url, json=payload, timeout=timeout)
            if 400 <= resp.status_code < 500:
                raise FatalRequestError(f"{resp.status_code}: {resp.text[:400]}")
            if resp.status_code >= 500:
                raise RuntimeError(f"{resp.status_code}: {resp.text[:200]}")
            return resp.json()["choices"][0]["message"]["content"] or ""
        except FatalRequestError:
            raise
        except Exception as exc:
            last = exc
        _local.session = None
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gave up after {retries} attempts: {last}")


def _nearest_caption(figure: dict, captions: list[dict]) -> str:
    """Caption whose box sits closest below the figure, if any."""
    fx1, _, fx2, fy2 = figure["box"]
    best, best_gap = "", 10**9
    for cap in captions:
        cx1, cy1, cx2, _ = cap["box"]
        overlap = min(fx2, cx2) - max(fx1, cx1)
        gap = cy1 - fy2
        if overlap > 0 and 0 <= gap < best_gap:
            best, best_gap = cap["text"], gap
    return best


def crop_figures(
    page_image: Path,
    regions: list[dict],
    out_dir: Path,
    page: int,
    pad: int = 6,
) -> list[Figure]:
    """Save every figure region as its own PNG.

    Figures in a paper are often vector drawings, not embedded raster images, so
    pulling them out of the PDF object tree misses them. Cropping the rendered
    page from the model's own boxes catches both.
    """
    from PIL import Image

    figures = [r for r in regions if r["category"] in FIGURE_CATS]
    if not figures:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    captions = [r for r in regions if r["category"] in CAPTION_CATS]
    entries: list[Figure] = []
    with Image.open(page_image) as src:
        width, height = src.size
        for n, region in enumerate(figures, 1):
            x1, y1, x2, y2 = denormalize(region["box"], width, height)
            box = (
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(width, x2 + pad),
                min(height, y2 + pad),
            )
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                continue  # degenerate box, nothing worth describing
            name = f"page_{page:04d}_fig{n:02d}_{region['category']}.png"
            path = out_dir / name
            src.crop(box).save(path)
            entries.append(
                Figure(
                    page=page,
                    index=n,
                    category=region["category"],
                    path=path,
                    link=f"figures/{name}",
                    # Caption text may trail the figure marker, or arrive as a
                    # separate caption region; prefer the inline one.
                    caption=region["text"] or _nearest_caption(region, captions),
                    box_norm=list(region["box"]),
                    box_px=list(box),
                )
            )
    return entries


ProgressFn = Callable[[int, int, str], None]
FigureSink = Callable[[list["Figure"]], None]


def run_ocr(
    images: list[Path],
    figures_dir: Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout: int,
    concurrency: int,
    retries: int,
    figure_pad: int,
    progress: ProgressFn | None = None,
    on_figures: FigureSink | None = None,
) -> OcrResult:
    """OCR every page concurrently, crop figures, keep page order.

    Progress fires once per finished page with (done, total, message).

    `on_figures` is called with each page's freshly cropped figures, from this
    loop's thread, as soon as that page is parsed. That is what lets figure
    description overlap with the remaining OCR instead of waiting for the whole
    document. It is never called for a page with no figures.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(images)
    # Indexed, not append-order: pages finish out of order under concurrency and
    # the document must stay in page order.
    pages: list[str | None] = [None] * total
    per_page_figures: list[list[Figure]] = [[] for _ in images]
    chrome: list[set[str]] = [set() for _ in images]
    failures: list[str] = []
    done = 0

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(
                ocr_one,
                base_url,
                api_key,
                model,
                img,
                WINDOW_SINGLE,
                max_tokens,
                timeout,
                retries,
            ): i
            for i, img in enumerate(images)
        }
        for future in as_completed(futures):
            i = futures[future]
            done += 1
            try:
                raw = future.result()
            except Exception as exc:  # one bad page must not sink the batch
                failures.append(f"halaman {i + 1}: {exc}")
                if progress:
                    progress(done, total, f"halaman {i + 1} GAGAL: {exc}")
                continue

            regions = parse_regions(raw)
            entries = crop_figures(
                images[i], regions, figures_dir, i + 1, pad=figure_pad
            )
            links = {tuple(e.box_norm): e.link for e in entries}
            per_page_figures[i] = entries
            chrome[i] = chrome_text(regions)
            pages[i] = remove_det(raw, links)
            if entries and on_figures:
                # Hand off before reporting progress: the vision stage should be
                # working on these crops by the time the UI redraws.
                on_figures(entries)
            if progress:
                note = f", {len(entries)} gambar" if entries else ""
                progress(done, total, f"halaman {i + 1} selesai{note}")

    flat = [f for page in per_page_figures for f in page]
    return OcrResult(pages=pages, figures=flat, chrome_lines=chrome, failures=failures)
