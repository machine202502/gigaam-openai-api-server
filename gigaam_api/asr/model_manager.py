"""Управление жизненным циклом модели: ленивая загрузка, выгрузка по простою, ручное управление.

Идея: сервис может крутиться постоянно (Docker), но держать веса в памяти только когда модель
реально нужна. `ModelManager` реализует контракт `ASREngine` и прозрачно для обработчиков:

- загружает модель при первом обращении (`ensure_loaded`), если её нет в памяти;
- отмечает время последнего использования на каждом запросе;
- выгружает модель, если она простаивала дольше `idle_seconds` (`unload_if_idle`, дёргается
  сторожевым таймером) — либо принудительно по кнопке в UI;
- при следующем запросе снова загружает (веса берутся из кэша `MODELS_DIR`, без сети).

Все переходы загрузка/выгрузка и сам инференс идут через единственный воркер `Runner`, поэтому
выгрузка не может произойти посреди инференса.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
from collections.abc import Callable, Iterator

from gigaam_api.asr.engine import ASREngine, ASRResult, EngineInfo, SegmentTS
from gigaam_api.config import Settings
from gigaam_api.runner import Runner

logger = logging.getLogger(__name__)


class ModelManager:
    """Обёртка над ASREngine с ленивой загрузкой и выгрузкой по простою.

    Реализует `ASREngine` (runtime_checkable Protocol), поэтому может подменять движок в
    `app.state.engine` без изменений в HTTP-слое.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._engine: ASREngine | None = None
        self._last_used = time.monotonic()
        self._idle_seconds = settings.IDLE_UNLOAD_SECONDS
        # `model_name`/`device` — часть контракта ASREngine; device уточняется при загрузке.
        self.model_name = settings.MODEL
        self.device: str = settings.DEVICE

    # --- состояние -------------------------------------------------------------------

    @property
    def engine(self) -> ASREngine | None:
        """Загруженный движок или None (без блокировки — читаем атомарно)."""
        return self._engine

    @property
    def is_loaded(self) -> bool:
        """Загружена ли модель сейчас."""
        return self._engine is not None

    @property
    def idle_seconds(self) -> int:
        """Порог простоя (секунды); 0 — автовыгрузка выключена."""
        return self._idle_seconds

    @idle_seconds.setter
    def idle_seconds(self, value: int) -> None:
        if value < 0:
            raise ValueError("idle_seconds must be >= 0")
        with self._lock:
            self._idle_seconds = value
        logger.info("idle unload threshold set to %ds", value)

    def idle_for_seconds(self) -> float:
        """Сколько секунд прошло с последнего использования модели."""
        return time.monotonic() - self._last_used

    def touch(self) -> None:
        """Отметить модель как использованную (сбросить таймер простоя)."""
        with self._lock:
            self._last_used = time.monotonic()

    def status(self) -> dict[str, object]:
        """Снимок состояния для UI."""
        return {
            "model": self.model_name,
            "device": self.device,
            "loaded": self.is_loaded,
            "idle_seconds": self._idle_seconds,
            "idle_for_seconds": round(self.idle_for_seconds(), 1),
        }

    # --- загрузка/выгрузка -----------------------------------------------------------

    def load(self) -> ASREngine:
        """Гарантировать загрузку модели и вернуть движок (блокирующе, из воркера)."""
        return self._ensure_loaded()

    def _ensure_loaded(self) -> ASREngine:
        with self._lock:
            engine = self._engine
            if engine is None:
                # Ленивый импорт: тянем torch/gigaam только при реальной загрузке.
                from gigaam_api.asr.gigaam_engine import GigaAMEngine

                engine = GigaAMEngine(self._settings)
                self._engine = engine
                self.device = engine.device
                logger.info("model loaded into memory: %s device=%s", self.model_name, self.device)
            self._last_used = time.monotonic()
            return engine

    def unload(self) -> bool:
        """Выгрузить модель из памяти принудительно. True — если что-то выгрузили."""
        with self._lock:
            if self._engine is None:
                return False
            self._engine = None
        gc.collect()
        self._release_torch_cache()
        logger.info("model unloaded from memory: %s", self.model_name)
        return True

    def unload_if_idle(self) -> bool:
        """Выгрузить модель, только если она простаивала дольше порога. True — если выгрузили."""
        with self._lock:
            if self._engine is None:
                return False
            idle = self._idle_seconds
            if idle <= 0 or self.idle_for_seconds() < idle:
                return False
            self._engine = None
        gc.collect()
        self._release_torch_cache()
        logger.info("model unloaded after %ds idle: %s", idle, self.model_name)
        return True

    @staticmethod
    def _release_torch_cache() -> None:
        """По возможности освободить CUDA-кэш (на CPU — no-op)."""
        try:
            import torch
        except Exception:  # pragma: no cover - torch обязателен в рантайме
            return
        if torch.cuda.is_available():  # pragma: no cover - на CPU/в CI недоступно
            torch.cuda.empty_cache()

    # --- контракт ASREngine ----------------------------------------------------------

    def transcribe(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ASRResult:
        engine = self._ensure_loaded()
        try:
            return engine.transcribe(
                wav_path, word_timestamps=word_timestamps, cancel_check=cancel_check
            )
        finally:
            self.touch()

    def iter_segments(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[SegmentTS]:
        # Тело генератора выполнится на первом next() — уже в потоке воркера.
        engine = self._ensure_loaded()
        try:
            yield from engine.iter_segments(
                wav_path, word_timestamps=word_timestamps, cancel_check=cancel_check
            )
        finally:
            self.touch()

    def info(self) -> EngineInfo:
        return {"model": self.model_name, "device": self.device, "loaded": self.is_loaded}


async def idle_watchdog(manager: ModelManager, runner: Runner, *, interval: float = 15.0) -> None:
    """Фоновый сторожевой цикл: периодически выгружает простаивающую модель.

    Выгрузка отправляется в единственный воркер `Runner` и там ПЕРЕпроверяет простой —
    поэтому не может выгрузить модель во время инференса или сразу после нового запроса.
    Выключенный порог (`idle_seconds == 0`) — no-op.
    """
    while True:
        await asyncio.sleep(interval)
        if manager.idle_seconds <= 0 or not manager.is_loaded:
            continue
        if runner.inflight > 0:
            continue
        try:
            await asyncio.wrap_future(runner.submit(manager.unload_if_idle))
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - защитная сеть, не должна срабатывать
            logger.exception("idle unload failed")
