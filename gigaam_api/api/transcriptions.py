"""POST /v1/audio/transcriptions — OpenAI-совместимый эндпоинт (sync + SSE-стриминг).

Загрузка пишется на диск чанками (не держим весь файл в RAM); инференс —
через Runner (сериализация + backpressure).

Две ветки:
- `stream=true` И формат ∈ {json,text} → SSE: сегменты по мере готовности (delta) →
  done → [DONE]. backpressure (try_acquire→503) проверяется ДО StreamingResponse;
  владение временным файлом передаётся потоку (очистка в on_done, когда воркер свободен).
- иначе (sync, или verbose/srt/vtt+stream → синхронный fallback) → полный ответ.
  Отмена longform кооперативная: watcher на request.is_disconnected() выставляет
  threading.Event, переданный как cancel_check; в потоке отмена приходит от
  самого StreamingResponse.
"""

import asyncio
import contextlib
import logging
import os
import tempfile
import threading
import uuid
from collections.abc import Iterator
from typing import Annotated, cast

import anyio
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from gigaam_api.asr import formats
from gigaam_api.asr.engine import (
    ASREngine,
    ASRResult,
    AudioTooLongError,
    InferenceCancelledError,
    SegmentTS,
)
from gigaam_api.audio import probe_duration
from gigaam_api.auth import require_auth
from gigaam_api.config import Settings, get_settings
from gigaam_api.errors import PayloadTooLargeError, error_response
from gigaam_api.runner import Runner
from gigaam_api.streaming import STREAM_HEARTBEAT_SECONDS, sse_transcription, stream_segments

logger = logging.getLogger(__name__)
router = APIRouter()

_VALID_FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}
_CHUNK = 1 << 20  # 1 МиБ


async def _watch_disconnect(request: Request, event: threading.Event) -> None:
    """Опрос отключения клиента; при отключении — взвести флаг cancelled.

    Запускается ТОЛЬКО как дочерняя задача anyio task group и снимается через
    `cancel_scope.cancel()`. Использовать raw `asyncio.create_task` + `task.cancel()`
    НЕЛЬЗЯ: `Request.is_disconnected()` держит внутри `anyio.CancelScope`, и raw
    отмена конфликтует с ним — задача не завершается, `await task` deadlock (ADR этапа 04).
    """
    while not await request.is_disconnected():
        await anyio.sleep(1.0)
    event.set()


def _render(result: ASRResult, fmt: str, granularities: set[str], request_id: str) -> Response:
    """Рендер ASRResult в ответ по формату (fmt уже провалидирован вызывающим)."""
    logger.debug("request_id=%s render format=%s chars=%d", request_id, fmt, len(result.text))
    if fmt == "json":
        return JSONResponse(content=formats.to_json(result))
    if fmt == "verbose_json":
        return JSONResponse(content=formats.to_verbose_json(result, granularities=granularities))
    if fmt == "text":
        return PlainTextResponse(content=formats.to_text(result))
    if fmt == "srt":
        return PlainTextResponse(
            content=formats.to_srt(result), media_type="text/plain; charset=utf-8"
        )
    return PlainTextResponse(content=formats.to_vtt(result), media_type="text/vtt; charset=utf-8")


@router.post("/v1/audio/transcriptions", dependencies=[Depends(require_auth)])
async def transcriptions(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File()],
    model: Annotated[str | None, Form()] = None,
    response_format: Annotated[str | None, Form()] = None,
    timestamp_granularities: Annotated[
        list[str] | None, Form(alias="timestamp_granularities[]")
    ] = None,
    language: Annotated[str | None, Form()] = None,  # принимается и игнорируется (GigaAM — RU)
    stream: Annotated[bool, Form()] = False,  # json/text → SSE; verbose/srt/vtt → sync fallback
    prompt: Annotated[str | None, Form()] = None,  # принимается и игнорируется
    temperature: Annotated[float | None, Form()] = None,  # принимается и игнорируется
) -> Response:
    request_id = uuid.uuid4().hex
    fmt = response_format or settings.DEFAULT_RESPONSE_FORMAT
    logger.info(
        "request_id=%s model=%s response_format=%s file=%s stream=%s",
        request_id,
        model,
        fmt,
        file.filename,
        stream,
    )

    if model is not None and model not in settings.ALLOWED_MODELS:
        return error_response(
            400,
            f"model '{model}' is not available",
            "invalid_request_error",
            param="model",
        )
    if fmt not in _VALID_FORMATS:
        return error_response(
            400,
            f"response_format '{fmt}' is not supported",
            "invalid_request_error",
            param="response_format",
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    tmp_path: str | None = None
    streamed = False  # True → владение временным файлом передано потоку
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(_CHUNK):
                size += len(chunk)
                if max_bytes > 0 and size > max_bytes:
                    raise PayloadTooLargeError(
                        f"upload exceeds MAX_UPLOAD_MB={settings.MAX_UPLOAD_MB}"
                    )
                tmp.write(chunk)
        logger.debug("request_id=%s tmp=%s size=%d", request_id, tmp_path, size)

        loop = asyncio.get_running_loop()
        duration = await loop.run_in_executor(None, probe_duration, tmp_path)
        if settings.MAX_AUDIO_SECONDS > 0 and duration > settings.MAX_AUDIO_SECONDS:
            raise AudioTooLongError(
                f"audio {duration:.1f}s exceeds MAX_AUDIO_SECONDS={settings.MAX_AUDIO_SECONDS}"
            )

        word_timestamps = bool(timestamp_granularities and "word" in timestamp_granularities)
        granularities = set(timestamp_granularities) if timestamp_granularities else {"segment"}
        logger.debug(
            "request_id=%s duration=%.2fs word_timestamps=%s",
            request_id,
            duration,
            word_timestamps,
        )

        engine = cast(ASREngine, request.app.state.engine)
        runner = cast(Runner, request.app.state.runner)

        # SSE-стриминг только для json/text; verbose/srt/vtt с stream=true → синхронный
        # fallback (полный результат), а не 400 (терпимее к клиентам, ADR этапа 05).
        if stream and fmt in {"json", "text"}:
            assert tmp_path is not None  # the upload wrote the file above
            path = tmp_path

            # backpressure ДО StreamingResponse: 503 уходит без заголовков
            # (иначе поток уже был бы 200).
            runner.try_acquire()
            # с этого момента временный файл принадлежит потоку (очистка в _cleanup)
            streamed = True
            cancel_event = threading.Event()

            def _make_iter() -> Iterator[SegmentTS]:
                return engine.iter_segments(
                    path, word_timestamps=word_timestamps, cancel_check=cancel_event.is_set
                )

            def _cleanup() -> None:
                # Вызывается, когда продюсер реально закончил (воркер свободен).
                runner.release()
                with contextlib.suppress(OSError):
                    os.unlink(path)
                logger.debug(
                    "request_id=%s stream cleanup: slot released, temp removed", request_id
                )

            segments = stream_segments(
                runner,
                _make_iter,
                request_id=request_id,
                heartbeat_interval=STREAM_HEARTBEAT_SECONDS,
                cancel=cancel_event,
                on_done=_cleanup,
            )
            logger.info("request_id=%s stream=true response_format=%s → SSE", request_id, fmt)
            return StreamingResponse(
                sse_transcription(segments, request_id=request_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # --- Синхронная ветка (stream=false ИЛИ verbose/srt/vtt → fallback) ---
        # Наблюдатель отключения и инференс живут в одной anyio task group: когда
        # клиент отключается, наблюдатель выставляет cancel_event, и longform
        # останавливается между батчами.
        # Результат инференса захватываем ВНУТРИ группы и извлекаем СНАРУЖИ, чтобы
        # исключения (например, QueueFullError→503) не упаковывались в ExceptionGroup.
        cancel_event = threading.Event()
        result: ASRResult | None = None
        inference_error: BaseException | None = None
        async with anyio.create_task_group() as tg:
            tg.start_soon(_watch_disconnect, request, cancel_event)
            try:
                result = await runner.run(
                    engine.transcribe,
                    tmp_path,
                    word_timestamps=word_timestamps,
                    cancel_check=cancel_event.is_set,
                )
            # ловим, чтобы снять наблюдателя и избежать ExceptionGroup
            except Exception as exc:
                inference_error = exc
            finally:
                tg.cancel_scope.cancel()

        if isinstance(inference_error, InferenceCancelledError):
            logger.info("request_id=%s inference cancelled (client disconnected)", request_id)
            return Response(status_code=499)
        if inference_error is not None:
            raise inference_error
        assert result is not None
        return _render(result, fmt, granularities, request_id)
    finally:
        # При стриминге временный файл нужен воркеру после возврата из обработчика;
        # _cleanup удалит его.
        if tmp_path is not None and not streamed:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
