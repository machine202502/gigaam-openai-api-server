"""Сериализация блокирующего инференса: один воркер + лимит очереди (backpressure).

`Runner` гарантирует, что одновременно выполняется не более одного инференса (цикл событий
не блокируется) и отклоняет запросы сверх `MAX_QUEUE` через `QueueFullError` (→ 503),
вместо того чтобы молча ждать часами. Ничего не знает о модели и HTTP.
"""

import asyncio
import functools
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class QueueFullError(Exception):
    """Очередь инференса переполнена (принято ≥ MAX_QUEUE) — backpressure (→ 503)."""


class Runner:
    """Один `ThreadPoolExecutor(max_workers=1)` сериализует инференс; счётчик принятых
    (в очереди + в работе) ограничивает нагрузку."""

    def __init__(self, max_queue: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._max_queue = max_queue
        self._inflight = 0
        # счётчик принятых меняют и цикл событий (run/try_acquire), и поток воркера
        # (release из done-callback потока) → защищаем блокировкой.
        self._lock = threading.Lock()

    @property
    def inflight(self) -> int:
        """Сколько задач сейчас принято (в очереди + в работе)."""
        with self._lock:
            return self._inflight

    @property
    def max_queue(self) -> int:
        """Настроенный лимит очереди (`MAX_QUEUE`)."""
        return self._max_queue

    def try_acquire(self) -> None:
        """Занять слот (backpressure). При переполнении (принято ≥ MAX_QUEUE) → QueueFullError.

        Синхронный — вызывается в обработчике ДО `StreamingResponse`, чтобы 503 ушёл
        до заголовков (иначе поток уже был бы `200`)."""

        with self._lock:
            if self._inflight >= self._max_queue:
                logger.warning(
                    "queue full: inflight=%d max_queue=%d", self._inflight, self._max_queue
                )
                raise QueueFullError(f"inference queue is full (max_queue={self._max_queue})")
            self._inflight += 1
            logger.debug("inference enqueued: inflight=%d", self._inflight)

    def release(self) -> None:
        """Освободить слот (пара к `try_acquire`)."""
        with self._lock:
            self._inflight -= 1
            logger.debug("inference finished: inflight=%d", self._inflight)

    def submit(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        """Отправить блокирующую `fn` в тот же единственный воркер (сериализация сохраняется).

        НЕ отслеживает принятые сам — это делает вызывающий (поток) через
        `try_acquire`/`release`, так как слот держится всё время потока, а не только до
        возврата из `submit`."""

        return self._executor.submit(functools.partial(fn, *args, **kwargs))

    async def run(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Выполнить блокирующую `fn` в единственном воркере. При переполнении → QueueFullError."""
        self.try_acquire()
        try:
            loop = asyncio.get_running_loop()
            partial = functools.partial(fn, *args, **kwargs)
            return await loop.run_in_executor(self._executor, partial)
        finally:
            self.release()

    def shutdown(self) -> None:
        """Остановить пул (вызывать в lifespan на завершении)."""
        # cancel_futures=True отменяет задачи в очереди; уже запущенный поток доживёт до конца.
        self._executor.shutdown(wait=False, cancel_futures=True)
