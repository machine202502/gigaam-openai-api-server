"""Тесты ModelManager: ленивая загрузка, выгрузка по простою, ручное управление."""

import time
from collections.abc import Callable, Iterator

import pytest

import gigaam_api.asr.gigaam_engine as ge
from gigaam_api.asr.engine import ASRResult, EngineInfo, SegmentTS
from gigaam_api.asr.model_manager import ModelManager
from gigaam_api.config import Settings


class _FakeEngine:
    """Двойник GigaAMEngine: считает созданные экземпляры, чтобы проверять перезагрузку."""

    created = 0

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.MODEL
        self.device = "cpu"
        type(self).created += 1

    def transcribe(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ASRResult:
        return ASRResult(text="ok", duration=1.0, segments=[SegmentTS("ok", 0.0, 1.0)])

    def iter_segments(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[SegmentTS]:
        yield SegmentTS("ok", 0.0, 1.0)

    def info(self) -> EngineInfo:
        return {"model": self.model_name, "device": self.device, "loaded": True}


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeEngine.created = 0
    monkeypatch.setattr(ge, "GigaAMEngine", _FakeEngine)


def test_lazy_load_on_first_use() -> None:
    manager = ModelManager(Settings())
    assert manager.is_loaded is False
    assert manager.info()["loaded"] is False

    engine = manager.load()

    assert isinstance(engine, _FakeEngine)
    assert manager.is_loaded is True
    assert manager.info()["loaded"] is True
    assert manager.device == "cpu"


def test_transcribe_loads_and_touches() -> None:
    manager = ModelManager(Settings())
    manager._last_used = time.monotonic() - 1000  # имитируем долгий простой

    result = manager.transcribe("a.wav", word_timestamps=False)

    assert result.text == "ok"
    assert manager.is_loaded is True
    assert manager.idle_for_seconds() < 1.0  # использование сбросило таймер


def test_iter_segments_loads_and_touches() -> None:
    manager = ModelManager(Settings())
    manager._last_used = time.monotonic() - 1000

    segments = list(manager.iter_segments("a.wav", word_timestamps=True))

    assert [s.text for s in segments] == ["ok"]
    assert manager.idle_for_seconds() < 1.0


def test_unload_frees_and_is_idempotent() -> None:
    manager = ModelManager(Settings())
    manager.load()

    assert manager.unload() is True
    assert manager.is_loaded is False
    assert manager.unload() is False  # повторная выгрузка — no-op


def test_unload_if_idle_respects_threshold() -> None:
    manager = ModelManager(Settings())
    manager.load()

    manager.idle_seconds = 60
    assert manager.unload_if_idle() is False  # только что использовали
    assert manager.is_loaded is True

    manager._last_used = time.monotonic() - 120  # простой больше порога
    assert manager.unload_if_idle() is True
    assert manager.is_loaded is False


def test_unload_if_idle_disabled_with_zero() -> None:
    manager = ModelManager(Settings())
    manager.load()
    manager.idle_seconds = 0
    manager._last_used = time.monotonic() - 10000

    assert manager.unload_if_idle() is False
    assert manager.is_loaded is True


def test_reload_after_unload_creates_new_engine() -> None:
    manager = ModelManager(Settings())
    first = manager.load()
    manager.unload()
    second = manager.load()

    assert first is not second
    assert _FakeEngine.created == 2


def test_idle_seconds_setter_rejects_negative() -> None:
    manager = ModelManager(Settings())
    with pytest.raises(ValueError, match="idle_seconds"):
        manager.idle_seconds = -1


def test_status_snapshot() -> None:
    manager = ModelManager(Settings())
    manager.load()
    manager.idle_seconds = 42

    status = manager.status()

    assert status["model"] == "v3_ctc"
    assert status["loaded"] is True
    assert status["idle_seconds"] == 42
    assert isinstance(status["idle_for_seconds"], float)
