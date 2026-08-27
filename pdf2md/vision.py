"""Stage 2: describe every cropped figure with a vision LLM.

Ported from `ocrgeneral/client.py`, narrowed to what the pipeline needs: local
image -> base64 data URL -> one chat completion -> description text.

Figures are described as OCR discovers them, not in a batch afterwards, so the
vision endpoint starts on page 1's chart while later pages are still being parsed.
A single bad figure never raises: the error lands on the Figure so stage 3 can fall
back to the original caption.
"""

from __future__ import annotations

import base64
import mimetypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

from openai import OpenAI

from .ocr import Figure

ProgressFn = Callable[[int, int, str], None]


def encode_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None or not mime.startswith("image/"):
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class VisionDescriber:
    """Thin wrapper over the OpenAI SDK for image -> description."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 300.0,
        temperature: float = 0.2,
        max_tokens: int = 600,
        retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = max(1, retries)
        self._client = OpenAI(
            base_url=base_url or None,
            api_key=api_key or "missing",
            timeout=timeout,
            max_retries=0,  # retries are handled here, with our own backoff
        )

    def describe(self, image: Path, prompt: str, context: str = "") -> str:
        """One figure -> description text. Raises after the last retry."""
        text = prompt
        if context:
            # The OCR caption is the single most useful hint the model can get:
            # it names the figure and often states what it is meant to show.
            text = f"{prompt}\n\nKeterangan gambar dari dokumen (untuk konteks):\n{context}"

        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": encode_data_url(image)},
                                },
                            ],
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except FileNotFoundError:
                raise
            except Exception as exc:
                last = exc
                if attempt < self.retries - 1:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"gagal setelah {self.retries} percobaan: {last}")


class DescriptionPool:
    """Describes figures as they arrive, instead of waiting for a complete list.

    OCR discovers figures page by page, so the vision endpoint can start working on
    page 1's chart while page 14 is still being parsed. Submissions come from the
    OCR loop (one thread); completions land on worker threads, so the counters are
    lock-guarded.

    `total` grows as figures are discovered, so a tick's `total` is a lower bound,
    not a known whole, until OCR stops submitting. Callers rendering a progress bar
    must treat it as provisional.
    """

    def __init__(
        self,
        describer: VisionDescriber,
        prompt: str,
        *,
        concurrency: int = 4,
        progress: ProgressFn | None = None,
    ) -> None:
        self._describer = describer
        self._prompt = prompt
        self._progress = progress
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, concurrency), thread_name_prefix="vision"
        )
        self._lock = threading.Lock()
        self._figures: list[Figure] = []
        self._done = 0

    def submit(self, figures: Iterable[Figure]) -> None:
        """Queue figures for description. Safe to call while others are in flight."""
        for fig in figures:
            with self._lock:
                self._figures.append(fig)
                total = len(self._figures)
            self._tick(self._done, total, f"{fig.path.name} dalam antrean")
            self._pool.submit(self._run, fig)

    def _run(self, fig: Figure) -> None:
        try:
            fig.description = self._describer.describe(fig.path, self._prompt, fig.caption)
            message = f"{fig.path.name} selesai"
        except Exception as exc:  # one bad figure must not sink the batch
            fig.error = str(exc)
            message = f"{fig.path.name} GAGAL: {exc}"
        with self._lock:
            self._done += 1
            done, total = self._done, len(self._figures)
        self._tick(done, total, message)

    def _tick(self, done: int, total: int, message: str) -> None:
        if self._progress:
            self._progress(done, total, message)

    @property
    def submitted(self) -> int:
        with self._lock:
            return len(self._figures)

    def join(self) -> list[Figure]:
        """Wait for every queued figure, then return them in page order.

        Page order, not completion order: the manifest and the UI figure list read
        top-to-bottom through the document.
        """
        self._pool.shutdown(wait=True)
        with self._lock:
            return sorted(self._figures, key=lambda f: (f.page, f.index))
