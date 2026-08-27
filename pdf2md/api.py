"""HTTP API over the same pipeline the Streamlit app drives.

`POST /convert` takes a PDF and answers with the finished Markdown, so a caller
never needs the UI:

    curl -F file=@paper.pdf http://127.0.0.1:8080/convert -o paper.md

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
"""

from __future__ import annotations

import re
import secrets
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

import anyio
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .config import ApiSettings, Settings, load_env
from .ocr import Figure
from .pipeline import PipelineResult, run_pipeline

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


def _settings_for(dpi: int | None, cleanup: bool | None, keep_image_link: bool | None) -> Settings:
    """Env defaults with the per-request overrides layered on top."""
    settings = SETTINGS
    if dpi is not None:
        settings = replace(settings, ocr=replace(settings.ocr, dpi=dpi))
    if cleanup is not None:
        settings = replace(settings, cleanup=replace(settings.cleanup, enabled=cleanup))
    if keep_image_link is not None:
        settings = replace(settings, keep_image_link=keep_image_link)
    return settings


def _convert(pdf_bytes: bytes, name: str, settings: Settings) -> PipelineResult:
    """Blocking half of a request: write the upload, run the pipeline, drop the workdir.

    Runs on a worker thread. The result carries the document and the figure
    manifest by value, so the temporary directory is safe to remove here.
    """
    workdir = Path(tempfile.mkdtemp(prefix="pdf2md_api_"))
    try:
        pdf = workdir / name
        pdf.write_bytes(pdf_bytes)
        return run_pipeline(pdf, workdir, settings)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.get("/health")
def health() -> dict:
    """Liveness plus the effective model config. Never echoes an API key."""
    return {
        "status": "ok",
        "ocr": {"base_url": SETTINGS.ocr.base_url, "model": SETTINGS.ocr.model, "dpi": SETTINGS.ocr.dpi},
        "vision": {"base_url": SETTINGS.vision.base_url, "model": SETTINGS.vision.model},
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
        return JSONResponse(
            ConvertOut(
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
            ).model_dump()
        )

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
