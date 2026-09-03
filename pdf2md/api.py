"""HTTP API over the same pipeline the Streamlit app drives.

`POST /convert` takes a PDF and answers with the finished Markdown, so a caller
never needs the UI:

    curl -F file=@paper.pdf http://127.0.0.1:8080/convert -o paper.md

`POST /convert/stream` answers the same upload with a live NDJSON feed: one JSON
object per line, a `progress` event per pipeline tick (the same per-stage view the
Streamlit app paints), then a single terminal event carrying the whole Markdown.

`POST /convert/ocr` and `POST /convert/ocr/stream` are the pure-OCR variants: they
skip the vision LLM stage entirely, so figure placeholders remain in the markdown
without descriptions. Use these when you don't need figure understanding or want
faster processing.

Design notes:

* Settings come from `.env` once at import (`Settings.from_env()`), because a
  server has no sidebar. A handful of per-request query overrides exist for the
  values that legitimately vary per document (`dpi`, `cleanup`,
  `keep_image_link`); everything else, including endpoints and API keys, stays
  process-wide.
* `run_pipeline` is blocking and thread-parallel inside, so it runs in a worker
  thread and a semaphore caps how many runs happen at once. Excess requests
  queue instead of multiplying `OCR_CONCURRENCY` onto the OCR endpoint.
* The work directory is temporary and removed once the response is built: the
  Markdown and the figure manifest are returned by value, nothing is served
  from disk afterwards.
* The stream endpoint relays progress through a thread-safe queue instead of an
  async queue: the capacity limiter that bounds concurrent runs is contextvar
  state, and anyio's `task_group` + `to_thread` would only see a copied context.
  Moving the limiter through `start_blocking_portal` would fix that but cost an
  extra thread-per-process dance; a `queue.Queue` plus a worker thread is simpler
  and just as correct.
"""

from __future__ import annotations

import asyncio
import json
import queue
import re
import secrets
import shutil
import functools
import tempfile
import threading
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

import anyio
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .config import ApiSettings, Settings, load_env
from .ocr import Figure
from .pipeline import (
    CONCURRENT_STAGES,
    STAGE_LABELS,
    PipelineResult,
    Progress,
    ProgressTracker,
    run_pipeline,
)

load_env()
API = ApiSettings.from_env()
SETTINGS = Settings.from_env()

CHUNK = 1 << 20
MAX_UPLOAD = API.max_upload_mb * 1024 * 1024
# A PDF must start with the version header. Rejecting here keeps a mislabelled
# upload from becoming a PyMuPDF stack trace.
PDF_MAGIC = b"%PDF-"
# Filename characters that survive into the response: word chars (unicode letters
# included), space, dash, dot, parens. Everything else is folded to "_", which also
# defuses path traversal from the multipart filename.
UNSAFE_NAME_RE = re.compile(r"[^\w\-. ()]+", re.UNICODE)
NON_ASCII_RE = re.compile(r"[^A-Za-z0-9._-]+")

app = FastAPI(
    title="pdf2md",
    version="0.1.0",
    summary="PDF → Markdown: Unlimited-OCR layout parsing + vision-LLM figure descriptions",
)

# Bound concurrent pipeline runs. anyio's limiter works with the threadpool call
# below, so a queued request holds no worker thread while it waits.
_slots = anyio.CapacityLimiter(API.max_concurrent)
# The stream endpoint runs on plain threads, where anyio's limiter release would
# mix thread- and task-owner bookkeeping; a plain semaphore is its equivalent.
_stream_slots = threading.Semaphore(API.max_concurrent)


def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Header auth, only when `API_KEY` is set. Empty key means an open server."""
    if not API.api_key:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, API.api_key):
        raise HTTPException(status_code=401, detail="X-API-Key tidak ada atau salah")


class FigureOut(BaseModel):
    """One figure as the caller sees it: no crop file, since none is served."""

    page: int
    category: str
    file: str
    caption: str
    description: str
    error: str

    @classmethod
    def of(cls, fig: Figure) -> "FigureOut":
        return cls(
            page=fig.page,
            category=fig.category,
            file=fig.link,
            caption=fig.caption,
            description=fig.description,
            error=fig.error,
        )


class ConvertOut(BaseModel):
    filename: str
    markdown: str
    page_count: int
    figure_count: int
    described: int
    substituted: int
    chrome_removed: int
    # A page that failed OCR or a figure that failed description degrades the
    # document instead of failing the request; the caller sees both here.
    page_failures: list[str] = Field(default_factory=list)
    figure_errors: list[str] = Field(default_factory=list)
    figures: list[FigureOut] = Field(default_factory=list)


def _safe_pdf_name(raw: str | None) -> str:
    """Multipart filename -> a bare, traversal-free `*.pdf` name."""
    name = Path(raw or "").name.strip()
    name = UNSAFE_NAME_RE.sub("_", name).strip(". ") or "document"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def _content_disposition(md_name: str) -> str:
    """RFC 6266 header: ASCII fallback plus the real UTF-8 name."""
    ascii_name = NON_ASCII_RE.sub("_", md_name) or "output.md"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(md_name)}"


async def _read_capped(upload: UploadFile) -> bytes:
    """Buffer the upload, refusing anything past `API_MAX_UPLOAD_MB`."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(CHUNK):
        total += len(chunk)
        if total > MAX_UPLOAD:
            raise HTTPException(
                status_code=413,
                detail=f"PDF lebih besar dari batas {API.max_upload_mb} MB",
            )
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(status_code=400, detail="File kosong")
    data = b"".join(chunks)
    if not data.startswith(PDF_MAGIC):
        raise HTTPException(status_code=415, detail="File bukan PDF")
    return data


def _settings_for(
    dpi: int | None, cleanup: bool | None, keep_image_link: bool | None
) -> Settings:
    """Env defaults with the per-request overrides layered on top."""
    settings = SETTINGS
    if dpi is not None:
        settings = replace(settings, ocr=replace(settings.ocr, dpi=dpi))
    if cleanup is not None:
        settings = replace(settings, cleanup=replace(settings.cleanup, enabled=cleanup))
    if keep_image_link is not None:
        settings = replace(settings, keep_image_link=keep_image_link)
    return settings


def _convert(
    pdf_bytes: bytes, name: str, settings: Settings, describe: bool = True
) -> PipelineResult:
    """Blocking half of a request: write the upload, run the pipeline, drop the workdir.

    Runs on a worker thread. The result carries the document and the figure
    manifest by value, so the temporary directory is safe to remove here.
    """
    workdir = Path(tempfile.mkdtemp(prefix="pdf2md_api_"))
    try:
        pdf = workdir / name
        pdf.write_bytes(pdf_bytes)
        return run_pipeline(pdf, workdir, settings, describe=describe)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _convert_out(result: PipelineResult, md_name: str) -> ConvertOut:
    """The JSON shape of a finished run, shared by `/convert` and the stream."""
    return ConvertOut(
        filename=md_name,
        markdown=result.markdown,
        page_count=result.page_count,
        figure_count=len(result.figures),
        described=result.described,
        substituted=result.substituted,
        chrome_removed=result.cleanup.removed_count,
        page_failures=result.failures,
        figure_errors=result.figure_errors,
        figures=[FigureOut.of(f) for f in result.figures],
    )


def _outcome(status: int, message: str) -> dict[str, Any]:
    """Terminal error event of a `/convert/stream` feed."""
    return {"status": status, "detail": message}


def _convert_streaming(
    pdf_bytes: bytes,
    name: str,
    settings: Settings,
    ticks: queue.Queue,
    describe: bool = True,
) -> None:
    """`_convert` plus progress: runs on a plain thread and reports via `ticks`.

    The terminal item is always `("done", PipelineResult)` or
    `("error", {status, detail})`, so the async reader can stop after one
    terminal event without waiting for the thread to be reaped.
    """
    workdir = Path(tempfile.mkdtemp(prefix="pdf2md_api_"))
    try:
        pdf = workdir / name
        pdf.write_bytes(pdf_bytes)
        result = run_pipeline(
            pdf, workdir, settings, progress=ticks.put, describe=describe
        )
        ticks.put(("done", result))
    except ValueError as exc:
        ticks.put(("error", _outcome(400, str(exc))))
    except RuntimeError as exc:
        ticks.put(("error", _outcome(502, str(exc))))
    except Exception as exc:  # unexpected: still terminate the feed cleanly
        ticks.put(("error", _outcome(500, f"Pipeline gagal: {exc}")))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _stream_worker(
    pdf_bytes: bytes,
    name: str,
    settings: Settings,
    ticks: queue.Queue,
    describe: bool = True,
) -> None:
    """Hold a concurrency slot for the whole run, then convert with progress."""
    _stream_slots.acquire()
    try:
        _convert_streaming(pdf_bytes, name, settings, ticks, describe)
    finally:
        _stream_slots.release()


def _stages_view(tracker: ProgressTracker, tick: Progress) -> dict[str, Any]:
    """One progress event: the raw tick plus the per-stage snapshot the Streamlit
    app paints as rows."""
    stages: dict[str, Any] = {}
    for key, label in STAGE_LABELS.items():
        seen = tracker.latest.get(key)
        if seen is None:
            stages[key] = {"label": label, "status": "pending"}
            continue
        counter = f"{seen.done}/{seen.total}" if seen.total else "—"
        if key in CONCURRENT_STAGES and seen.total and not tracker.is_done(key):
            # A vision total grows while figures are still being found; mark the
            # denominator as provisional, like the Streamlit "+" suffix.
            counter += "+"
        stages[key] = {
            "label": label,
            "status": "done" if tracker.is_done(key) else "running",
            "done": seen.done,
            "total": seen.total,
            "counter": counter,
            "message": seen.message,
        }
    active = " + ".join(
        STAGE_LABELS[k]
        for k in ("ocr", "vision")
        if tracker.started(k) and not tracker.is_done(k)
    )
    return {
        "type": "progress",
        "stage": tick.stage,
        "label": tick.label,
        "done": tick.done,
        "total": tick.total,
        "message": tick.message,
        "overall": round(tracker.overall, 4),
        "active": active or tick.label,
        "stages": stages,
    }


async def _stream_run(
    pdf_bytes: bytes, name: str, settings: Settings, describe: bool = True
) -> AsyncIterator[str]:
    """Drain the worker's queue into an NDJSON feed.

    The slot is acquired inside the worker thread (see the module docstring), so
    a queued request parks one executor thread while it waits in `queue.get`.
    """
    ticks: queue.Queue = queue.Queue()
    threading.Thread(
        target=_stream_worker,
        args=(pdf_bytes, name, settings, ticks, describe),
        daemon=True,
    ).start()
    tracker = ProgressTracker()
    loop = asyncio.get_running_loop()
    md_name = f"{Path(name).stem}.md"
    while True:
        # run_in_executor keeps the blocking queue.get off the event loop. On a
        # client disconnect the generator is abandoned and one executor thread
        # stays parked in get() until the run ends -- which the daemon worker
        # guarantees, removing its workdir in its own finally.
        item = await loop.run_in_executor(None, ticks.get)
        if isinstance(item, Progress):
            tracker.update(item)
            yield json.dumps(_stages_view(tracker, item), ensure_ascii=False) + "\n"
            continue
        kind, payload = item
        if kind == "error":
            yield json.dumps({"type": "error", **payload}, ensure_ascii=False) + "\n"
            return
        yield (
            json.dumps(
                {"type": "result"} | _convert_out(payload, md_name).model_dump(),
                ensure_ascii=False,
            )
            + "\n"
        )
        return


@app.get("/health")
def health() -> dict:
    """Liveness plus the effective model config. Never echoes an API key."""
    return {
        "status": "ok",
        "ocr": {
            "base_url": SETTINGS.ocr.base_url,
            "model": SETTINGS.ocr.model,
            "dpi": SETTINGS.ocr.dpi,
        },
        "vision": {
            "base_url": SETTINGS.vision.base_url,
            "model": SETTINGS.vision.model,
        },
        "cleanup_enabled": SETTINGS.cleanup.enabled,
        "keep_image_link": SETTINGS.keep_image_link,
        "auth_required": bool(API.api_key),
        "max_upload_mb": API.max_upload_mb,
        "max_concurrent": API.max_concurrent,
    }


@app.post(
    "/convert",
    dependencies=[Depends(require_key)],
    response_model=None,
    responses={
        200: {
            "content": {
                "text/markdown": {"schema": {"type": "string"}},
                "application/json": {"schema": ConvertOut.model_json_schema()},
            }
        }
    },
)
async def convert(
    file: Annotated[UploadFile, File(description="PDF yang mau dikonversi")],
    format: Annotated[str, Query(pattern="^(md|json)$")] = "md",
    dpi: Annotated[int | None, Query(ge=72, le=600)] = None,
    cleanup: Annotated[bool | None, Query()] = None,
    keep_image_link: Annotated[bool | None, Query()] = None,
) -> Response:
    """PDF in, Markdown out.

    `format=md` (default) answers with the document itself as a `.md` attachment;
    run counters ride along as `X-Pdf2md-*` headers. `format=json` answers with
    the same Markdown plus the figure manifest and every non-fatal failure.
    """
    pdf_bytes = await _read_capped(file)
    name = _safe_pdf_name(file.filename)
    settings = _settings_for(dpi, cleanup, keep_image_link)

    try:
        result = await anyio.to_thread.run_sync(
            _convert, pdf_bytes, name, settings, limiter=_slots
        )
    except ValueError as exc:
        # Unrenderable/empty document: the upload is at fault.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Every page failed OCR: the upstream endpoint is at fault, not the caller.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    md_name = f"{Path(name).stem}.md"
    if format == "json":
        return JSONResponse(_convert_out(result, md_name).model_dump())

    return Response(
        content=result.markdown.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": _content_disposition(md_name),
            "X-Pdf2md-Pages": str(result.page_count),
            "X-Pdf2md-Figures": str(len(result.figures)),
            "X-Pdf2md-Described": str(result.described),
            "X-Pdf2md-Substituted": str(result.substituted),
            "X-Pdf2md-Chrome-Removed": str(result.cleanup.removed_count),
            "X-Pdf2md-Page-Failures": str(len(result.failures)),
        },
    )


@app.post(
    "/convert/stream",
    dependencies=[Depends(require_key)],
    response_model=None,
    response_class=StreamingResponse,
    responses={
        200: {"content": {"application/x-ndjson": {"schema": {"type": "string"}}}}
    },
)
async def convert_stream(
    file: Annotated[UploadFile, File(description="PDF yang mau dikonversi")],
    dpi: Annotated[int | None, Query(ge=72, le=600)] = None,
    cleanup: Annotated[bool | None, Query()] = None,
    keep_image_link: Annotated[bool | None, Query()] = None,
) -> StreamingResponse:
    """PDF in, live progress out, Markdown last.

    Streams NDJSON: a `progress` event per pipeline tick (`stage`, `done/total`,
    `overall` 0..1, and `stages` — the per-stage snapshot the Streamlit UI
    shows), then one terminal event. On success that is `type=result` with the
    whole Markdown plus the figure manifest, the same payload as
    `/convert?format=json`. On failure it is `type=error` with the HTTP
    `status` the one-shot endpoint would have returned; upload validation keeps
    its plain status codes, since no feed exists yet.
    """
    pdf_bytes = await _read_capped(file)
    name = _safe_pdf_name(file.filename)
    settings = _settings_for(dpi, cleanup, keep_image_link)
    return StreamingResponse(
        _stream_run(pdf_bytes, name, settings),
        media_type="application/x-ndjson",
        # The feed is live; buffering proxies must not sit on it.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post(
    "/convert/ocr",
    dependencies=[Depends(require_key)],
    response_model=None,
    responses={
        200: {
            "content": {
                "text/markdown": {"schema": {"type": "string"}},
                "application/json": {"schema": ConvertOut.model_json_schema()},
            }
        }
    },
)
async def convert_ocr_only(
    file: Annotated[UploadFile, File(description="PDF yang mau dikonversi")],
    format: Annotated[str, Query(pattern="^(md|json)$")] = "md",
    dpi: Annotated[int | None, Query(ge=72, le=600)] = None,
    cleanup: Annotated[bool | None, Query()] = None,
    keep_image_link: Annotated[bool | None, Query()] = None,
) -> Response:
    """Pure OCR mode: PDF in, Markdown out, without vision LLM descriptions.

    Same as `/convert` but skips the figure description stage. Figures are still
    detected and their placeholders remain in the markdown (or are replaced with
    a note that description was skipped, depending on `keep_image_link`).
    """
    pdf_bytes = await _read_capped(file)
    name = _safe_pdf_name(file.filename)
    settings = _settings_for(dpi, cleanup, keep_image_link)

    try:
        result = await anyio.to_thread.run_sync(
            functools.partial(_convert, pdf_bytes, name, settings, describe=False), limiter=_slots
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    md_name = f"{Path(name).stem}.md"
    if format == "json":
        return JSONResponse(_convert_out(result, md_name).model_dump())

    return Response(
        content=result.markdown.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": _content_disposition(md_name),
            "X-Pdf2md-Pages": str(result.page_count),
            "X-Pdf2md-Figures": str(len(result.figures)),
            "X-Pdf2md-Described": str(result.described),
            "X-Pdf2md-Substituted": str(result.substituted),
            "X-Pdf2md-Chrome-Removed": str(result.cleanup.removed_count),
            "X-Pdf2md-Page-Failures": str(len(result.failures)),
        },
    )


@app.post(
    "/convert/ocr/stream",
    dependencies=[Depends(require_key)],
    response_model=None,
    response_class=StreamingResponse,
    responses={
        200: {"content": {"application/x-ndjson": {"schema": {"type": "string"}}}}
    },
)
async def convert_ocr_only_stream(
    file: Annotated[UploadFile, File(description="PDF yang mau dikonversi")],
    dpi: Annotated[int | None, Query(ge=72, le=600)] = None,
    cleanup: Annotated[bool | None, Query()] = None,
    keep_image_link: Annotated[bool | None, Query()] = None,
) -> StreamingResponse:
    """Pure OCR mode with live progress: PDF in, NDJSON feed out.

    Same as `/convert/stream` but skips the vision stage. The `vision` stage in
    progress events will show as pending (never started).
    """
    pdf_bytes = await _read_capped(file)
    name = _safe_pdf_name(file.filename)
    settings = _settings_for(dpi, cleanup, keep_image_link)
    return StreamingResponse(
        _stream_run(pdf_bytes, name, settings, describe=False),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
