"""Settings resolution: dataclass defaults, .env overrides, bad values."""

from __future__ import annotations

import pytest

from pdf2md.config import CleanupSettings, OcrSettings, Settings, VisionSettings

# `slots=True` turns class attributes into slot descriptors, so a `from_env` that
# reads `cls.dpi` silently yields a descriptor object instead of 300. That shipped
# once and only surfaced as a Streamlit widget type error, so pin the real types.
NUMERIC = [
    (OcrSettings, "dpi", int),
    (OcrSettings, "max_tokens", int),
    (OcrSettings, "timeout", int),
    (OcrSettings, "concurrency", int),
    (OcrSettings, "retries", int),
    (OcrSettings, "figure_pad", int),
    (VisionSettings, "timeout", float),
    (VisionSettings, "temperature", float),
    (VisionSettings, "max_tokens", int),
    (VisionSettings, "concurrency", int),
    (VisionSettings, "retries", int),
    (CleanupSettings, "min_ratio", float),
    (CleanupSettings, "min_pages", int),
    (CleanupSettings, "zone_lines", int),
]


@pytest.mark.parametrize("cls,name,kind", NUMERIC)
def test_from_env_yields_real_numbers(cls, name, kind, monkeypatch):
    monkeypatch.delenv(f"{cls.__name__}", raising=False)
    value = getattr(cls.from_env(), name)
    assert isinstance(value, kind), f"{cls.__name__}.{name} is {type(value)}"


def test_from_env_matches_dataclass_defaults_when_env_is_empty(monkeypatch):
    for key in ("OCR_DPI", "OCR_MODEL", "VISION_MODEL", "CLEANUP_MIN_RATIO"):
        monkeypatch.delenv(key, raising=False)

    assert OcrSettings.from_env().dpi == OcrSettings().dpi
    assert OcrSettings.from_env().model == OcrSettings().model
    assert VisionSettings.from_env().model == VisionSettings().model
    assert CleanupSettings.from_env().min_ratio == CleanupSettings().min_ratio


def test_env_overrides_are_applied_and_typed(monkeypatch):
    monkeypatch.setenv("OCR_BASE_URL", "http://ocr.internal:9000")
    monkeypatch.setenv("OCR_DPI", "150")
    monkeypatch.setenv("VISION_MODEL", "qwen2.5-vl")
    monkeypatch.setenv("VISION_TEMPERATURE", "0.9")
    monkeypatch.setenv("CLEANUP_ENABLED", "0")
    monkeypatch.setenv("KEEP_IMAGE_LINK", "0")

    settings = Settings.from_env()

    assert settings.ocr.base_url == "http://ocr.internal:9000"
    assert settings.ocr.dpi == 150
    assert settings.vision.model == "qwen2.5-vl"
    assert settings.vision.temperature == pytest.approx(0.9)
    assert settings.cleanup.enabled is False
    assert settings.keep_image_link is False


def test_blank_env_value_falls_back_to_default(monkeypatch):
    # An unfilled `.env` line (`OCR_MODEL=`) must not blank out the model name.
    monkeypatch.setenv("OCR_MODEL", "   ")
    assert OcrSettings.from_env().model == OcrSettings().model


def test_unparsable_numeric_env_falls_back_instead_of_crashing(monkeypatch):
    monkeypatch.setenv("OCR_DPI", "banyak")
    monkeypatch.setenv("VISION_TEMPERATURE", "panas")
    assert OcrSettings.from_env().dpi == OcrSettings().dpi
    assert VisionSettings.from_env().temperature == VisionSettings().temperature


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("no", False)],
)
def test_bool_env_shapes(monkeypatch, raw, expected):
    monkeypatch.setenv("CLEANUP_ENABLED", raw)
    assert CleanupSettings.from_env().enabled is expected
