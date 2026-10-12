"""Сборка SSE-событий для `stream=true`.

Контракт (сверен с OpenAI Speech-to-text, 06.2026):
- на каждый сегмент → `{"type":"transcript.text.delta","delta":"<текст>"}`;
- в конце → `{"type":"transcript.text.done","text":"<полный текст>"}` → `data: [DONE]`;
- ошибка в источнике → `{"type":"error","error":{...}}` и закрытие (без `[DONE]`).

Инвариант (как в OpenAI): конкатенация всех `delta` == `done.text` == синхронный ответ.
Разделитель-пробел «уезжает» в начало следующего delta (у первого нет ведущего пробела),
так что `"".join(delta) == " ".join(тексты сегментов)`.

Здесь — только рендеринг событий (чистый async-генератор над async-итератором сегментов).
Мост «блокирующий генератор движка → async» + heartbeat живут в `stream_segments` (см. ниже).
"""

import asyncio
import json
import logging
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator

from gigaam_api.asr.engine import SegmentTS
from gigaam_api.runner import Runner

logger = logging.getLogger(__name__)

# Интервал heartbeat: на медленном CPU один батч считается минутами без событий —
# периодический SSE-комментарий держит соединение против idle-таймаутов прокси.
STREAM_HEARTBEAT_SECONDS = 15.0

_HEARTBEAT_COMMENT = ": keep-alive\n\n"  # SSE-комментарий: клиенты игнорируют, держит соединение


class _Heartbeat:
    """Маркер «событий пока нет» — рендерится в SSE-комментарий против idle-таймаутов прокси."""


HEARTBEAT = _Heartbeat()

# Элемент потока: либо готовый сегмент, либо маркер heartbeat (когда событий ещё нет).
StreamItem = SegmentTS | _Heartbeat


def format_sse(data: dict[str, object] | str) -> str:
    """Собрать один SSE-фрейм `data: <payload>\\n\\n`.

    `str` возвращается как есть (терминатор `[DONE]`); `dict` сериализуется в компактный
    JSON с сохранением кириллицы (`ensure_ascii=False`).
    """
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


def _error_event(exc: Exception) -> dict[str, object]:
    """OpenAI-формат события ошибки в середине потока."""
    return {
        "type": "error",
        "error": {"message": str(exc), "type": "api_error", "param": None, "code": None},
    }


async def sse_transcription(
    items: AsyncIterator[StreamItem], *, request_id: str
) -> AsyncIterator[str]:
    """Рендеринг потока сегментов/heartbeat в SSE-фреймы.

    `CancelledError`/`GeneratorExit` (клиент отключился) НЕ ловятся намеренно —
    передаём очистку наверх; ловим только настоящие ошибки источника → событие error.
    """
    parts: list[str] = []
    try:
        async for item in items:
            if isinstance(item, _Heartbeat):
                yield _HEARTBEAT_COMMENT
                continue
            delta = item.text if not parts else " " + item.text
            parts.append(item.text)
            logger.debug(
                "request_id=%s delta idx=%d len=%d", request_id, len(parts) - 1, len(delta)
            )
            yield format_sse({"type": "transcript.text.delta", "delta": delta})
        full_text = " ".join(parts)
        logger.info(
            "request_id=%s stream done segments=%d chars=%d", request_id, len(parts), len(full_text)
        )
        yield format_sse({"type": "transcript.text.done", "text": full_text})
        yield format_sse("[DONE]")
    except Exception as exc:
        logger.exception("request_id=%s error in the SSE stream", request_id)
        yield format_sse(_error_event(exc))


class _Done:
    """Внутренний маркер очереди: продюсер исчерпал сегменты."""


_DONE = _Done()


async def stream_segments(
    runner: Runner,
    make_iter: Callable[[], Iterator[SegmentTS]],
    *,
    request_id: str,
    heartbeat_interval: float,
    cancel: threading.Event,
    on_done: Callable[[], None],
) -> AsyncGenerator[StreamItem, None]:
    """Мост: блокирующий генератор сегментов (в воркере Runner) → async-итератор.

    Сериализация инференса сохраняется — продюсер попадает в тот же единственный воркер
    (`runner.submit`), а не во временный поток. Между потоком и циклом событий —
    `asyncio.Queue` (наполняется через `call_soon_threadsafe`); если сегмент не пришёл
    за `heartbeat_interval`, отдаём `HEARTBEAT` (рендерится в SSE-комментарий против
    idle-таймаутов прокси).

    Жизненный цикл:
    - `on_done` (освободить слот + удалить временный файл) вызывается, когда продюсер
      РЕАЛЬНО закончил (воркер свободен) — через done-callback фьючера, а не когда
      потребитель закончил читать;
    - `cancel` выставляется в `finally` (потребитель ушёл/отменён) → `iter_segments`
      остановится между батчами;
    - ошибка продюсера передаётся потребителю (→ `sse_transcription` выдаёт событие error).
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[SegmentTS | _Done | Exception] = asyncio.Queue()

    def produce() -> None:
        try:
            for segment in make_iter():
                loop.call_soon_threadsafe(queue.put_nowait, segment)
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)
        # любая ошибка инференса → потребителю, поток не умирает молча
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)

    future = runner.submit(produce)
    future.add_done_callback(lambda _f: on_done())  # воркер свободен → освободить слот + очистка
    logger.info("request_id=%s stream started", request_id)
    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
            except TimeoutError:
                yield HEARTBEAT
                continue
            if isinstance(message, _Done):
                return
            if isinstance(message, Exception):
                raise message
            yield message
    finally:
        cancel.set()
