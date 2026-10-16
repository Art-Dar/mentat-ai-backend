# syntax=docker/dockerfile:1

# ---- builder: resolve deps into a venv with Poetry ----
FROM python:3.12-slim AS builder

ENV POETRY_VERSION=2.5.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"
ENV PATH="${POETRY_HOME}/bin:${PATH}"

WORKDIR /app

# Copy only the dependency manifest first so this layer stays cached
# while app code changes. README.md is included too since [project] in
# pyproject.toml declares it and `poetry check` validates it exists.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry check --lock \
    && poetry install --no-root --only main

# ---- runtime: slim image with just the venv + app code ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:${PATH}"

# The OCR engine, for app/services/ingestion/extractors/ocr.py.
#
# pytesseract is only a wrapper: it shells out to the `tesseract` binary,
# which is not a Python package and so cannot arrive through poetry.lock.
# Without this layer every call to /ingest/image answers 503 with "Text
# recognition is unavailable" — correct behaviour, but not useful.
#
# In the RUNTIME stage, not the builder: everything in the builder is
# thrown away except /app/.venv, so a binary installed there never reaches
# the running container.
#
# Before `USER app`, because apt-get needs root. Early in the stage,
# because this layer changes far less often than the application code
# copied below it and should stay cached across ordinary rebuilds.
#
# --no-install-recommends is safe here: tesseract-ocr lists
# tesseract-ocr-eng and tesseract-ocr-osd under Depends, not Recommends,
# so the English training data comes in regardless. (Measured with
# apt-cache show on 5.3.4; verify in your own image with the command in
# the comment at the bottom of this file.)
#
# tesseract-ocr-ukr is for Ukrainian receipts, used when a request passes
# language=ukr or eng+ukr. Drop it if you want a smaller image and only
# ever OCR English.
#
# Pillow needs no apt packages: its manylinux wheels bundle their own
# codecs (libjpeg, libtiff, libwebp, freetype) inside pillow.libs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-ukr \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY scripts ./scripts
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=2)" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# After building, check the engine actually landed and can see its data:
#
#   docker compose build api
#   docker compose run --rm --entrypoint sh api -c \ 'tesseract --version && tesseract --list-langs'
#
# You want eng, osd and ukr in the list. If the build itself fails at
# `poetry check --lock`, the lock file is stale — run `poetry lock` and
# commit it; the image build enforces the same thing your CI does.