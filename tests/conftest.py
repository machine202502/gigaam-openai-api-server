"""Общие фикстуры тестов: изолируют Settings от окружения и состояние логирования."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

# Все переменные окружения, которые читает Settings.
_SETTINGS_ENV_VARS = (
    "MODEL",
    "DEVICE",
    "API_KEY",
    "MODELS_DIR",
    "QUANTIZE_INT8",
    "BATCH_SIZE",
    "NUM_THREADS",
    "MAX_UPLOAD_MB",
    "MAX_AUDIO_SECONDS",
    "MAX_QUEUE",
    "IDLE_UNLOAD_SECONDS",
    "VAD_MIN_DURATION",
    "VAD_MAX_DURATION",
    "VAD_STRICT_LIMIT",
    "VAD_NEW_CHUNK_THRESHOLD",
    "VAD_THRESHOLD",
    "HOST",
    "PORT",
    "UI_ENABLED",
    "UI_HOST",
    "UI_PORT",
    "LOG_LEVEL",
    "LOG_JSON",
    "DEFAULT_RESPONSE_FORMAT",
    "ALLOWED_MODELS",
)


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Изолировать Settings: запуск из пустого каталога (без внешнего `.env`),
    очистка всех переменных окружения сервиса и кэша get_settings."""
    monkeypatch.chdir(tmp_path)
    for var in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    from gigaam_api.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Сохранить и восстановить состояние корневого логгера, чтобы setup_logging
    в одном тесте не протекал в другие."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
