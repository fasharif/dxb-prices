# syntax=docker/dockerfile:1
# Targets:
#   dev  - full environment (pipeline, tests, linters); mount the repo at /app
#   api  - FastAPI service; mount a trained model directory at /model
#   ui   - Streamlit page that calls the API
#
# Base images come from the AWS ECR mirror of the Docker Official Images
# (same images, no Docker Hub pull-rate limit).

FROM ghcr.io/astral-sh/uv:0.12.19 AS uv

FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm AS base
# LightGBM's Linux wheel links against the GNU OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS dev
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --all-extras --no-install-project
CMD ["bash"]

FROM base AS api-build
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM base AS ui-build
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra ui --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra ui --no-editable

FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /home/app

FROM runtime AS api
COPY --from=api-build /opt/venv /opt/venv
ENV DXB_MODEL_DIR=/model
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"]
CMD ["uvicorn", "dxb_prices.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS ui
COPY --from=ui-build /opt/venv /opt/venv
COPY src/dxb_prices/ui/streamlit_app.py ./streamlit_app.py
ENV DXB_API_URL=http://api:8000
USER app
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false", "--client.toolbarMode=viewer"]
