# syntax=docker/dockerfile:1.7
# GigaAM ASR — self-hosted образ (linux/amd64, CPU-only torch, БЕЗ весов внутри образа).
#
# Платформа задаётся на этапе сборки (`docker build --platform linux/amd64 ...`, см. Makefile),
# а НЕ зашита в FROM — так образ остаётся multi-arch, а на amd64-хосте собирается нативно.
# torch/torchaudio ставятся как CPU-колёса из индекса download.pytorch.org/whl/cpu
# (через [tool.uv.sources] + маркер sys_platform=='linux' в pyproject.toml) — без CUDA/nvidia.
# Веса GigaAM скачиваются при первом старте в смонтированный volume (MODELS_DIR=/data/models).

# ---- builder: ставим зависимости в изолированный /app/.venv через uv ----
FROM python:3.12-slim AS builder

# uv из официального образа (пин версии = воспроизводимая сборка).
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

# git — для git-зависимости gigaam (uv клонирует репозиторий и собирает пакет).
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

# UV_COMPILE_BYTECODE — .pyc на этапе сборки (быстрее старт); UV_LINK_MODE=copy — без hardlink
# (кэш-маунт на другой ФС); UV_PYTHON_DOWNLOADS=0 — использовать системный CPython базового образа.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 1) Слой зависимостей: кэшируется, пока не изменятся pyproject.toml/uv.lock.
#    --no-install-project — сам пакет ставится ниже, отдельным тонким слоем (после кода).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2) Код приложения + установка самого пакета.
COPY gigaam_api ./gigaam_api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- runtime: тонкий образ только с ffmpeg + готовым .venv ----
FROM python:3.12-slim AS runtime

# ffmpeg/ffprobe обязательны (декодирование аудио + probe длительности); ca-certificates — для
# HTTPS-скачивания весов с CDN при первом старте.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Непривилегированный пользователь с фиксированным UID/GID 1000. ВАЖНО: хостовый каталог
# ./models, смонтированный в /data/models, должен принадлежать UID 1000 (chown 1000:1000),
# иначе non-root не сможет писать веса в volume. См. README → «Деплой (Docker Compose)».
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin app

# PATH к .venv; MODELS_DIR — кэш весов GigaAM (volume); XDG_CACHE_HOME в volume —
# страховка, чтобы случайный кэш не улетал в недоступный для non-root ~/.cache (Silero
# встроен в пакет, проект не использует HF Hub/torch.hub).
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/data/models \
    XDG_CACHE_HOME=/data/models/.cache

# Точка монтирования volume; владелец — app (UID 1000), чтобы non-root мог писать веса.
RUN mkdir -p /data/models && chown -R 1000:1000 /data/models

WORKDIR /app
COPY --from=builder --chown=1000:1000 /app/.venv /app/.venv
COPY --from=builder --chown=1000:1000 /app/gigaam_api /app/gigaam_api

USER 1000
# 8000 — API, 8001 — веб-панель управления.
EXPOSE 8000 8001

# Healthcheck без curl (его нет в slim) — тонкий запрос к /health через stdlib urllib.
# start-period щедрый: первый старт может скачивать веса (минуты на медленном канале),
# пока они не загружены, uvicorn ещё не отвечает. В compose start_period задан отдельно.
# ВАЖНО: проверяем только код 200, не поле loaded — выгруженная по простою модель это норма.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status==200 else 1)"]

# Прод-запуск: единая точка входа поднимает API (8000) и панель управления (8001).
CMD ["python", "-m", "gigaam_api.serve"]
