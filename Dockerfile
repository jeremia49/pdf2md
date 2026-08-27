# API webserver image: `pdf2md.api:app` behind uvicorn.
#
# Two stages so the runtime layer carries no uv, no build tooling and no dev/ui
# deps. The venv is built against the lockfile, then copied as-is.
#
#   docker build -t pdf2md-api .
#   docker run --rm -p 8080:8080 --env-file .env pdf2md-api
#
# Streamlit is NOT in this image: it lives in the `ui` dependency group, which the
# install below skips. Model endpoints are read from the environment at startup.

FROM python:3.13-slim-bookworm AS build

# astral-sh/uv publishes `<version>-python<X.Y>-<distro>` tags only for some
# combinations; the bare version tag is the binary-only image, which is what this
# copies. Same pin as the local toolchain, one Python base for both stages.
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency layer: only the manifests, so editing pdf2md/ never reinvalidates it.
# `--no-default-groups` drops both `dev` (pytest, httpx) and `ui` (streamlit,
# pandas, pyarrow) -- the API needs none of them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups


FROM python:3.13-slim-bookworm AS runtime

# PyMuPDF and Pillow ship manylinux wheels, so no compiler or image libraries are
# needed at runtime; only certificates, for HTTPS vision endpoints.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 pdf2md

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # `package = false`, so the source tree itself must be importable.
    PYTHONPATH=/app \
    # Uploads and page renders go to a tmpfs-friendly path the app user owns.
    TMPDIR=/tmp

WORKDIR /app

COPY --from=build /app/.venv /app/.venv
COPY pdf2md/ /app/pdf2md/

USER pdf2md

EXPOSE 8080

# `/health` needs no auth, so this works whether or not API_KEY is set.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status == 200 else 1)"]

# One worker on purpose: each request already fans out to OCR_CONCURRENCY +
# VISION_CONCURRENCY threads, and API_MAX_CONCURRENT caps parallel runs inside the
# process. Scale with replicas, not workers.
CMD ["uvicorn", "pdf2md.api:app", "--host", "0.0.0.0", "--port", "8080"]
