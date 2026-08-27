"""Runtime settings: `.env` defaults, overridable from the Streamlit sidebar.

Settings are dataclasses rather than module globals so each Streamlit rerun builds
a fresh config; nothing process-wide leaks between pipeline runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_FIGURE_PROMPT = (
    "Kamu menganalisis satu gambar yang dipotong dari sebuah dokumen PDF. "
    "Jelaskan isi dan maksud gambar itu dalam bahasa Indonesia secara padat dan faktual, "
    "2-5 kalimat. Sebutkan jenis visualnya (diagram alur, grafik, skema arsitektur, foto, "
    "tangkapan layar), label/komponen yang terbaca, hubungan antar komponen, serta tren atau "
    "angka penting bila ada. Untuk grafik, sebutkan sumbu, satuan, dan arah trennya. "
    "Jangan menambahkan informasi yang tidak terlihat pada gambar. "
    "Jawab hanya dengan deskripsinya: tanpa pembuka, tanpa heading, tanpa bullet."
)


def load_env() -> Path | None:
    """Load `pdf2md/.env` into the process env. Returns the file used, if any."""
    if ENV_PATH.is_file():
        load_dotenv(ENV_PATH, override=False)
        return ENV_PATH
    return None


def _env(key: str, default: str) -> str:
    value = os.environ.get(key, "")
    return value if value.strip() else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    return _env(key, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class OcrSettings:
    """Unlimited-OCR vLLM endpoint (stage 1: layout parsing)."""

    base_url: str = "http://127.0.0.1:8000"
    api_key: str = ""
    model: str = "baidu/Unlimited-OCR"
    dpi: int = 300
    # max_tokens + prompt_tokens must stay below max_model_len (32768). A 300 DPI
    # A4 page already costs ~2.7k prompt tokens, so 32768 here is rejected with 400.
    max_tokens: int = 8192
    timeout: int = 1800
    concurrency: int = 4
    retries: int = 4
    figure_pad: int = 6

    # `slots=True` replaces class attributes with slot descriptors, so `cls.dpi`
    # is a descriptor, not 300. Defaults must be read off an instance.
    @classmethod
    def from_env(cls) -> "OcrSettings":
        d = cls()
        return cls(
            base_url=_env("OCR_BASE_URL", d.base_url),
            api_key=_env("OCR_API_KEY", d.api_key),
            model=_env("OCR_MODEL", d.model),
            dpi=_env_int("OCR_DPI", d.dpi),
            max_tokens=_env_int("OCR_MAX_TOKENS", d.max_tokens),
            timeout=_env_int("OCR_TIMEOUT", d.timeout),
            concurrency=_env_int("OCR_CONCURRENCY", d.concurrency),
            retries=_env_int("OCR_RETRIES", d.retries),
            figure_pad=_env_int("OCR_FIGURE_PAD", d.figure_pad),
        )


@dataclass(slots=True)
class VisionSettings:
    """OpenAI-compatible vision endpoint (stage 2: figure understanding)."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    timeout: float = 300.0
    temperature: float = 0.2
    max_tokens: int = 600
    concurrency: int = 4
    retries: int = 3
    prompt: str = DEFAULT_FIGURE_PROMPT

    @classmethod
    def from_env(cls) -> "VisionSettings":
        d = cls()
        return cls(
            base_url=_env("VISION_BASE_URL", d.base_url),
            api_key=_env("VISION_API_KEY", d.api_key),
            model=_env("VISION_MODEL", d.model),
            timeout=_env_float("VISION_TIMEOUT", d.timeout),
            temperature=_env_float("VISION_TEMPERATURE", d.temperature),
            max_tokens=_env_int("VISION_MAX_TOKENS", d.max_tokens),
            concurrency=_env_int("VISION_CONCURRENCY", d.concurrency),
            retries=_env_int("VISION_RETRIES", d.retries),
            prompt=_env("VISION_PROMPT", d.prompt),
        )


@dataclass(slots=True)
class CleanupSettings:
    """Repeated header/footer stripping (stage 4)."""

    enabled: bool = True
    # A line must repeat on at least this fraction of pages to count as chrome.
    min_ratio: float = 0.6
    # ...and on at least this many pages, so a 2-page doc cannot self-trigger.
    min_pages: int = 3
    # How many leading/trailing lines of a page are eligible.
    zone_lines: int = 3
    # Drop bare page numbers even when they never repeat verbatim.
    drop_page_numbers: bool = True

    @classmethod
    def from_env(cls) -> "CleanupSettings":
        d = cls()
        return cls(
            enabled=_env_bool("CLEANUP_ENABLED", d.enabled),
            min_ratio=_env_float("CLEANUP_MIN_RATIO", d.min_ratio),
            min_pages=_env_int("CLEANUP_MIN_PAGES", d.min_pages),
            zone_lines=_env_int("CLEANUP_ZONE_LINES", d.zone_lines),
            drop_page_numbers=_env_bool("CLEANUP_DROP_PAGE_NUMBERS", d.drop_page_numbers),
        )


@dataclass(slots=True)
class Settings:
    ocr: OcrSettings = field(default_factory=OcrSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    cleanup: CleanupSettings = field(default_factory=CleanupSettings)
    # Emit the figure description inline and keep the ![](...) link, or replace it.
    keep_image_link: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        return cls(
            ocr=OcrSettings.from_env(),
            vision=VisionSettings.from_env(),
            cleanup=CleanupSettings.from_env(),
            keep_image_link=_env_bool("KEEP_IMAGE_LINK", True),
        )
