"""Streamlit UI for the PDF -> Markdown pipeline.

Progress is reported per stage and per unit within the stage ("halaman 7/14"), so
a long OCR run never looks stalled. The pipeline itself runs in a worker thread and
pushes ticks through a queue: Streamlit widgets can only be touched from the script
thread, so the main thread drains the queue and does the drawing.
"""

from __future__ import annotations

import queue
import shutil
import tempfile
import threading
import traceback
from dataclasses import replace
from pathlib import Path

import streamlit as st

from pdf2md.config import (
    DEFAULT_FIGURE_PROMPT,
    ENV_PATH,
    Settings,
    load_env,
)
from pdf2md.pipeline import (
    CONCURRENT_STAGES,
    STAGE_LABELS,
    STAGES,
    PipelineResult,
    Progress,
    ProgressTracker,
    run_pipeline,
)

st.set_page_config(page_title="PDF → Markdown", page_icon="📄", layout="wide")


@st.cache_resource
def _env_defaults() -> tuple[Settings, bool]:
    """Read `.env` once per process; the sidebar layers overrides on top."""
    used = load_env()
    return Settings.from_env(), used is not None


def sidebar_settings(defaults: Settings, env_found: bool) -> Settings:
    st.sidebar.header("Setelan model")
    if env_found:
        st.sidebar.caption(f"Default dibaca dari `{ENV_PATH.name}`")
    else:
        st.sidebar.warning(
            f"`{ENV_PATH.name}` tidak ditemukan. Salin `.env.example` → `.env`, "
            "atau isi nilainya di sini."
        )

    with st.sidebar.expander("1 · OCR dokumen (Unlimited-OCR)", expanded=True):
        ocr = replace(
            defaults.ocr,
            base_url=st.text_input("OCR base URL", defaults.ocr.base_url, key="ocr_url"),
            model=st.text_input("OCR model", defaults.ocr.model, key="ocr_model"),
            api_key=st.text_input(
                "OCR API key", defaults.ocr.api_key, type="password", key="ocr_key"
            ),
            dpi=st.number_input(
                "DPI render", 72, 600, defaults.ocr.dpi, step=50, key="ocr_dpi"
            ),
            max_tokens=st.number_input(
                "Max tokens / halaman",
                512,
                32768,
                defaults.ocr.max_tokens,
                step=512,
                key="ocr_maxtok",
                help="max_tokens + prompt harus di bawah max_model_len (32768).",
            ),
            concurrency=st.slider(
                "Paralel halaman", 1, 16, defaults.ocr.concurrency, key="ocr_conc"
            ),
            timeout=st.number_input(
                "Timeout (detik)", 30, 7200, defaults.ocr.timeout, step=30, key="ocr_to"
            ),
        )

    with st.sidebar.expander("2 · Deskripsi gambar (vision LLM)", expanded=True):
        vision = replace(
            defaults.vision,
            base_url=st.text_input(
                "Vision base URL",
                defaults.vision.base_url,
                key="vis_url",
                help="Pakai root /v1; SDK menambahkan /chat/completions.",
            ),
            model=st.text_input("Vision model", defaults.vision.model, key="vis_model"),
            api_key=st.text_input(
                "Vision API key", defaults.vision.api_key, type="password", key="vis_key"
            ),
            temperature=st.slider(
                "Temperature", 0.0, 1.0, defaults.vision.temperature, 0.05, key="vis_temp"
            ),
            max_tokens=st.number_input(
                "Max tokens / gambar",
                64,
                4096,
                defaults.vision.max_tokens,
                step=64,
                key="vis_maxtok",
            ),
            concurrency=st.slider(
                "Paralel gambar", 1, 16, defaults.vision.concurrency, key="vis_conc"
            ),
            prompt=st.text_area(
                "Prompt deskripsi",
                defaults.vision.prompt or DEFAULT_FIGURE_PROMPT,
                height=160,
                key="vis_prompt",
            ),
        )

    with st.sidebar.expander("4 · Header/footer berulang"):
        cleanup = replace(
            defaults.cleanup,
            enabled=st.toggle(
                "Hapus otomatis", defaults.cleanup.enabled, key="cl_on"
            ),
            min_ratio=st.slider(
                "Ambang kemunculan (rasio halaman)",
                0.2,
                1.0,
                defaults.cleanup.min_ratio,
                0.05,
                key="cl_ratio",
                help="Baris harus muncul di minimal proporsi halaman ini.",
            ),
            zone_lines=st.slider(
                "Baris zona atas/bawah", 1, 6, defaults.cleanup.zone_lines, key="cl_zone"
            ),
            drop_page_numbers=st.toggle(
                "Hapus nomor halaman", defaults.cleanup.drop_page_numbers, key="cl_pn"
            ),
        )

    keep_link = st.sidebar.toggle(
        "Simpan link gambar di markdown",
        defaults.keep_image_link,
        key="keep_link",
        help="Nonaktifkan agar placeholder gambar diganti deskripsi saja.",
    )

    return Settings(ocr=ocr, vision=vision, cleanup=cleanup, keep_image_link=keep_link)


def _worker(pdf: Path, workdir: Path, settings: Settings, ticks: queue.Queue) -> None:
    """Run the pipeline off-thread, funnelling progress and outcome into `ticks`."""
    try:
        result = run_pipeline(pdf, workdir, settings, progress=ticks.put)
        ticks.put(("done", result))
    except Exception as exc:
        ticks.put(("error", (exc, traceback.format_exc())))


def execute(pdf_bytes: bytes, name: str, settings: Settings) -> PipelineResult:
    """Drive the pipeline and paint live progress. Raises on pipeline failure."""
    workdir = Path(tempfile.mkdtemp(prefix="pdf2md_"))
    pdf = workdir / name
    pdf.write_bytes(pdf_bytes)

    ticks: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_worker, args=(pdf, workdir, settings, ticks), daemon=True
    )

    bar = st.progress(0.0, text="Menyiapkan…")
    stage_slots = {key: st.empty() for key, _ in STAGES}
    for i, (key, label) in enumerate(STAGES, 1):
        stage_slots[key].markdown(f"◻︎ **{i}. {label}** — menunggu")

    tracker = ProgressTracker()

    def paint() -> None:
        """Redraw every stage row from tracker state.

        Stage rows are driven by the tracker, not by "a later stage reported, so
        earlier ones must be done": OCR and vision overlap, so a vision tick says
        nothing about whether OCR has finished.
        """
        for n, (key, label) in enumerate(STAGES, 1):
            tick = tracker.latest.get(key)
            if tick is None:
                continue
            counter = f"{tick.done}/{tick.total}" if tick.total else "—"
            if key in CONCURRENT_STAGES and tick.total:
                # A vision total grows while figures are still being found, so say
                # so rather than implying the denominator is final.
                counter += "+" if not tracker.is_done(key) else ""
            icon = "✅" if tracker.is_done(key) else "⏳"
            suffix = f" · {tick.message}" if tick.message else ""
            stage_slots[key].markdown(f"{icon} **{n}. {label}** — {counter}{suffix}")

    thread.start()
    result: PipelineResult | None = None

    while True:
        item = ticks.get()  # blocks; the worker always sends a terminal item
        if isinstance(item, Progress):
            tracker.update(item)
            paint()
            active = " + ".join(
                STAGE_LABELS[k]
                for k in ("ocr", "vision")
                if tracker.started(k) and not tracker.is_done(k)
            )
            bar.progress(tracker.overall, text=active or item.label)
            continue

        kind, payload = item
        if kind == "error":
            exc, tb = payload
            bar.progress(tracker.overall, text="Gagal")
            st.error(f"Pipeline gagal: {exc}")
            with st.expander("Traceback"):
                st.code(tb)
            shutil.rmtree(workdir, ignore_errors=True)
            raise exc
        result = payload
        break

    thread.join()
    for i, (key, label) in enumerate(STAGES, 1):
        stage_slots[key].markdown(f"✅ **{i}. {label}** — selesai")
    bar.progress(1.0, text="Selesai")
    return result


def render_result(result: PipelineResult) -> None:
    cols = st.columns(4)
    cols[0].metric("Halaman", result.page_count)
    cols[1].metric("Gambar dideskripsikan", f"{result.described}/{len(result.figures)}")
    cols[2].metric("Placeholder diganti", result.substituted)
    cols[3].metric("Baris chrome dihapus", result.cleanup.removed_count)

    if result.failures:
        st.warning("Halaman gagal di-OCR:\n\n" + "\n".join(f"- {f}" for f in result.failures))
    if result.figure_errors:
        st.warning(
            "Gambar gagal dideskripsikan:\n\n"
            + "\n".join(f"- {e}" for e in result.figure_errors)
        )
    if result.cleanup.removed_lines:
        with st.expander(f"Header/footer yang dihapus ({len(result.cleanup.removed_lines)} pola)"):
            st.write("\n".join(f"- `{line}`" for line in result.cleanup.removed_lines))

    st.download_button(
        "⬇️ Unduh markdown",
        result.markdown,
        file_name="output.md",
        mime="text/markdown",
        type="primary",
    )

    tab_copy, tab_preview, tab_figs = st.tabs(["Markdown (copy)", "Pratinjau", "Gambar"])
    with tab_copy:
        # st.code gives a one-click copy button and no markdown interpretation.
        st.code(result.markdown, language="markdown")
    with tab_preview:
        st.markdown(result.markdown)
    with tab_figs:
        if not result.figures:
            st.info("Tidak ada gambar terdeteksi di dokumen ini.")
        for fig in result.figures:
            left, right = st.columns([1, 2])
            with left:
                st.image(str(fig.path), caption=f"hal {fig.page} · {fig.category}")
            with right:
                if fig.caption:
                    st.caption(fig.caption)
                st.write(fig.description or f"_Gagal: {fig.error or 'tidak ada hasil'}_")


def main() -> None:
    st.title("📄 PDF → Markdown")
    st.caption(
        "Unlimited-OCR membaca layout, lalu vision LLM menjelaskan setiap gambar, "
        "placeholder gambar diganti deskripsinya, dan header/footer berulang dibuang."
    )

    defaults, env_found = _env_defaults()
    settings = sidebar_settings(defaults, env_found)

    upload = st.file_uploader("Unggah PDF", type=["pdf"])
    if upload is None:
        st.info("Unggah sebuah PDF untuk memulai.")
        return

    st.write(f"**{upload.name}** · {upload.size / 1_048_576:.2f} MB")
    if not st.button("Proses", type="primary"):
        # A finished run survives reruns (widget clicks) via session state.
        if "result" in st.session_state:
            render_result(st.session_state["result"])
        return

    if not settings.vision.api_key:
        st.warning(
            "Vision API key kosong. Tahap deskripsi gambar akan gagal kecuali "
            "endpoint-mu memang tidak butuh key."
        )

    try:
        result = execute(upload.getvalue(), upload.name, settings)
    except Exception:
        return  # already surfaced by execute()
    st.session_state["result"] = result
    st.success("Selesai. Markdown siap dicopy di bawah.")
    render_result(result)


main()
