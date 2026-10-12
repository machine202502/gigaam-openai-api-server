"""Тесты конфигурации Settings."""

import pytest
from pydantic import ValidationError

from gigaam_api.config import Settings, get_settings


def test_defaults() -> None:
    s = Settings()
    assert s.MODEL == "v3_ctc"
    assert s.DEVICE == "auto"
    assert s.API_KEY == ""
    assert s.PORT == 8000
    assert s.IDLE_UNLOAD_SECONDS == 300
    assert s.UI_ENABLED is True
    assert s.UI_PORT == 8001
    assert s.LOG_LEVEL == "INFO"
    assert s.LOG_JSON is False
    assert s.DEFAULT_RESPONSE_FORMAT == "json"
    assert s.MAX_AUDIO_SECONDS == 36000
    assert s.MAX_QUEUE == 8
    assert s.VAD_MAX_DURATION == 22.0
    assert s.ALLOWED_MODELS == ["v3_ctc", "v3_e2e_ctc", "v3_rnnt", "v3_e2e_rnnt"]


def test_allowed_models_parsed_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    # Пробелы вокруг элементов и пустые значения отбрасываются.
    monkeypatch.setenv("ALLOWED_MODELS", "v3_ctc, v3_rnnt , v3_e2e_ctc,")
    s = Settings()
    assert s.ALLOWED_MODELS == ["v3_ctc", "v3_rnnt", "v3_e2e_ctc"]


def test_invalid_device_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "gpu")
    with pytest.raises(ValidationError):
        Settings()


def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "TRACE")
    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_max_queue_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_QUEUE", "3")
    assert Settings().MAX_QUEUE == 3


def test_max_queue_zero_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # MAX_QUEUE=0 отвергал бы все запросы (503 на всё) — должно быть отклонено валидацией.
    monkeypatch.setenv("MAX_QUEUE", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_idle_unload_negative_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDLE_UNLOAD_SECONDS", "-1")
    with pytest.raises(ValidationError):
        Settings()


def test_idle_unload_zero_means_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDLE_UNLOAD_SECONDS", "0")
    assert Settings().IDLE_UNLOAD_SECONDS == 0
