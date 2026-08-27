"""HTTP contract of `pdf2md.api`: upload validation, auth, both output formats.

The pipeline itself is covered by `test_pipeline.py`; here it is replaced at the
`api._convert` boundary so the tests exercise routing, validation, header/JSON
shaping and error mapping without rendering a PDF.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import pytest

from pdf2md import api as api_mod
from pdf2md.markdown import CleanupReport
from pdf2md.ocr import Figure
from pdf2md.pipeline import PipelineResult

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

PDF = b"%PDF-1.7\n%%EOF\n"


def _result() -> PipelineResult:
    fig = Figure(
        page=2,
        index=1,
        category="chart",
        path=Path("figures/page_0002_fig01_chart.png"),
        link="figures/page_0002_fig01_chart.png",
        caption="Figure 1 | Latensi kernel.",
        box_norm=[200, 400, 800, 700],
        box_px=[20, 40, 80, 70],
        description="Grafik latensi kernel, tren datar.",
    )
    return PipelineResult(
        markdown="# Judul\n\nIsi halaman 1.",
        pages=["# Judul", "Isi halaman 1."],
        figures=[fig],
        page_count=2,
        described=1,
        substituted=1,
        cleanup=CleanupReport(removed_lines=["Unlimited OCR Works"], removed_count=3),
        failures=["halaman 4: timeout"],
    )


@pytest.fixture
def client(monkeypatch):
    """Client with the pipeline stubbed. Yields (client, call log)."""
    calls: list[tuple[bytes, str, object]] = []

    def fake_convert(pdf_bytes, name, settings):
        calls.append((pdf_bytes, name, settings))
        return _result()

    monkeypatch.setattr(api_mod, "_convert", fake_convert)
    with TestClient(api_mod.app) as c:
        yield c, calls


def _post(client, *, name="paper.pdf", data=PDF, **params):
    return client.post(
        "/convert", files={"file": (name, data, "application/pdf")}, params=params
    )


def test_markdown_response_carries_the_document_and_counters(client):
    c, calls = client
    r = _post(c)

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert r.text == "# Judul\n\nIsi halaman 1."
    assert "paper.md" in r.headers["content-disposition"]
    assert r.headers["x-pdf2md-pages"] == "2"
    assert r.headers["x-pdf2md-figures"] == "1"
    assert r.headers["x-pdf2md-described"] == "1"
    assert r.headers["x-pdf2md-chrome-removed"] == "3"
    # A page that failed OCR degrades the document; it must not fail the request.
    assert r.headers["x-pdf2md-page-failures"] == "1"
    assert calls[0][0] == PDF


def test_json_response_exposes_the_manifest_and_failures(client):
    c, _ = client
    r = _post(c, format="json")

    body = r.json()
    assert r.status_code == 200
    assert body["filename"] == "paper.md"
    assert body["markdown"].startswith("# Judul")
    assert body["page_failures"] == ["halaman 4: timeout"]
    assert body["figure_count"] == 1
    assert body["figures"][0]["file"] == "figures/page_0002_fig01_chart.png"
    assert body["figures"][0]["description"].startswith("Grafik latensi")
    assert body["figures"][0]["error"] == ""


def test_query_overrides_reach_the_pipeline_settings(client):
    c, calls = client
    _post(c, dpi=150, cleanup="false", keep_image_link="true")

    settings = calls[0][2]
    assert settings.ocr.dpi == 150
    assert settings.cleanup.enabled is False
    assert settings.keep_image_link is True
    # Overriding one run must not mutate the process-wide defaults.
    assert api_mod.SETTINGS.ocr.dpi != 150 or api_mod.SETTINGS.cleanup.enabled is True


def test_defaults_are_untouched_env_settings(client):
    c, calls = client
    _post(c)

    settings = calls[0][2]
    assert settings.ocr.dpi == api_mod.SETTINGS.ocr.dpi
    assert settings.cleanup.enabled == api_mod.SETTINGS.cleanup.enabled


def test_non_pdf_upload_is_rejected(client):
    c, calls = client
    r = _post(c, name="notes.txt", data=b"halo, ini bukan pdf")

    assert r.status_code == 415
    assert not calls  # never reached the pipeline


def test_empty_upload_is_rejected(client):
    c, _ = client
    assert _post(c, data=b"").status_code == 400


def test_upload_over_the_cap_is_rejected(client, monkeypatch):
    c, calls = client
    monkeypatch.setattr(api_mod, "MAX_UPLOAD", 32)
    r = _post(c, data=PDF + b"x" * 64)

    assert r.status_code == 413
    assert not calls


def test_bad_format_is_a_validation_error(client):
    c, _ = client
    assert _post(c, format="docx").status_code == 422


def test_traversal_in_the_filename_cannot_escape_the_workdir(client):
    c, calls = client
    r = _post(c, name="../../etc/passwd.pdf")

    name = calls[0][1]
    assert "/" not in name and "\\" not in name and ".." not in name
    assert name.endswith(".pdf")
    assert "passwd.md" in r.headers["content-disposition"]


def test_extensionless_upload_still_yields_a_md_filename(client):
    c, calls = client
    r = _post(c, name="scan")

    assert calls[0][1] == "scan.pdf"
    assert "scan.md" in r.headers["content-disposition"]


def test_non_ascii_filename_gets_both_disposition_forms(client):
    c, _ = client
    r = _post(c, name="laporan riset ekonomi.pdf")

    disposition = r.headers["content-disposition"]
    assert 'filename="laporan_riset_ekonomi.md"' in disposition
    assert unquote(disposition.split("filename*=UTF-8''")[1]) == "laporan riset ekonomi.md"


def test_unrenderable_pdf_is_the_callers_fault(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(
        api_mod,
        "_convert",
        lambda *a: (_ for _ in ()).throw(ValueError("paper.pdf tidak punya halaman")),
    )
    r = _post(c)

    assert r.status_code == 400
    assert "tidak punya halaman" in r.json()["detail"]


def test_every_page_failing_is_reported_as_upstream_failure(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(
        api_mod,
        "_convert",
        lambda *a: (_ for _ in ()).throw(RuntimeError("Semua halaman gagal di-OCR.")),
    )
    r = _post(c)

    assert r.status_code == 502
    assert "Semua halaman gagal" in r.json()["detail"]


def test_health_reports_config_without_leaking_keys(client):
    c, _ = client
    body = c.get("/health").json()

    assert body["status"] == "ok"
    assert body["ocr"]["model"] == api_mod.SETTINGS.ocr.model
    assert "api_key" not in repr(body)
    assert api_mod.SETTINGS.vision.api_key == "" or api_mod.SETTINGS.vision.api_key not in repr(body)


def test_api_key_is_enforced_only_when_configured(client, monkeypatch):
    c, calls = client
    monkeypatch.setattr(api_mod, "API", api_mod.ApiSettings(api_key="rahasia"))

    assert _post(c).status_code == 401
    assert not calls

    ok = c.post(
        "/convert",
        files={"file": ("paper.pdf", PDF, "application/pdf")},
        headers={"X-API-Key": "rahasia"},
    )
    assert ok.status_code == 200

    wrong = c.post(
        "/convert",
        files={"file": ("paper.pdf", PDF, "application/pdf")},
        headers={"X-API-Key": "salah"},
    )
    assert wrong.status_code == 401


def test_open_server_needs_no_header(client):
    c, _ = client
    assert not api_mod.API.api_key, "test .env must not set API_KEY"
    assert _post(c).status_code == 200
