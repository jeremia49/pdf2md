"""Runtime settings: `.env` defaults, overridable from the Streamlit sidebar.

Settings are dataclasses rather than module globals so each Streamlit rerun builds
a fresh config; nothing process-wide leaks between pipeline runs.

`ApiSettings` is deliberately *not* part of `Settings`: it configures the HTTP
server process (auth, upload cap, run slots), not a single pipeline run, and is
read once at startup instead of per request.
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
    "Jangan menambahkan informasi yang tidak terlihat pada gambar.\n\n"
    "Bila gambar memuat tabel (seluruh gambar berupa tabel, atau ada tabel di dalamnya), "
    "kamu WAJIB mentranskripsikan tabel itu sebagai tabel markdown GFM, bukan meringkasnya:\n"
    "- Salin setiap sel apa adanya, termasuk angka, satuan, dan tanda; jangan membulatkan, "
    "menghitung ulang, mengurutkan ulang, atau memotong baris/kolom.\n"
    "- Baris pertama adalah header, diikuti baris pemisah `|---|`. Bila tabel tidak punya "
    "header, pakai header kosong dengan jumlah kolom yang sama.\n"
    "- Setiap baris tabel dimulai dan diakhiri dengan `|`, dan berdiri di barisnya sendiri. "
    "Mulai tabel pada baris baru, jangan menempel pada kalimat.\n"
    "- Sel gabungan (merge) diulang nilainya pada tiap kolom/baris yang dicakup; sel kosong "
    "ditulis kosong; teks yang tidak terbaca ditulis `?`.\n"
    "- Ganti karakter `|` di dalam sel dengan `\\|`, dan ganti pergantian baris di dalam sel "
    "dengan `<br>`.\n"
    "- Tulis 1-2 kalimat deskripsi lebih dulu, lalu tabelnya. Catatan kaki atau satuan di "
    "luar tabel ditulis sebagai baris teks setelah tabel.\n\n"
    "Selain tabel, jawab hanya dengan deskripsinya: tanpa pembuka, tanpa heading, tanpa "
    "bullet, tanpa blok kode."
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
class ApiSettings:
    """HTTP surface in `pdf2md/api.py`. Process-wide, read once at import."""

    # Empty means no authentication: every caller may convert. Set API_KEY to
    # require the `X-API-Key` header.
    api_key: str = ""
    max_upload_mb: int = 50
    # Each run already fans out to OCR_CONCURRENCY + VISION_CONCURRENCY threads,
    # so concurrent requests are queued rather than multiplied onto the endpoints.
    max_concurrent: int = 2

    @classmethod
    def from_env(cls) -> "ApiSettings":
        d = cls()
        return cls(
            api_key=_env("API_KEY", d.api_key),
            max_upload_mb=_env_int("API_MAX_UPLOAD_MB", d.max_upload_mb),
            max_concurrent=max(1, _env_int("API_MAX_CONCURRENT", d.max_concurrent)),
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
