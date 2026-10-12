"""Точка входа FastAPI-приложения.

`create_app()` остаётся «лёгким» (без torch): модель (`ModelManager`) и `Runner` создаются в
lifespan (или принимаются готовым `AppContext` от лаунчера). Роутеры (/health,
/v1/audio/transcriptions, /v1/models) и обработчики ошибок OpenAI подключаются при создании.

Модель загружается при старте (fail-fast) и далее может выгружаться по простою сторожевым
таймером; при следующем запросе загружается снова прозрачно для клиента.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from gigaam_api import __version__
from gigaam_api.api.health import router as health_router
from gigaam_api.api.models import router as models_router
from gigaam_api.api.transcriptions import router as transcriptions_router
from gigaam_api.asr.model_manager import idle_watchdog
from gigaam_api.config import get_settings
from gigaam_api.context import AppContext, build_context
from gigaam_api.errors import register_exception_handlers
from gigaam_api.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def create_app(context: AppContext | None = None) -> FastAPI:
    settings = get_settings()
    setup_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Контекст либо общий (передан лаунчером), либо создаём свой (тесты/одиночный запуск).
        ctx = context or build_context(settings)
        app.state.context = ctx
        # Менеджер реализует ASREngine — HTTP-слой работает с ним как с движком.
        app.state.engine = ctx.manager
        app.state.runner = ctx.runner
        logger.info(
            "starting gigaam-api v%s | model=%s device=%s idle_unload=%ds log_level=%s",
            __version__,
            settings.MODEL,
            settings.DEVICE,
            ctx.manager.idle_seconds,
            settings.LOG_LEVEL,
        )
        try:
            # Загружаем модель при старте (fail-fast + прогрев весов).
            ctx.manager.load()
        except Exception:
            logger.exception(
                "ASR model failed to load — the application will not start (fail fast)"
            )
            raise
        logger.info(
            "ASR engine ready: model=%s device=%s max_queue=%d",
            ctx.manager.model_name,
            ctx.manager.device,
            settings.MAX_QUEUE,
        )
        # Сторожевой таймер: выгружает простаивающую модель (no-op при idle_seconds == 0).
        watchdog = asyncio.create_task(idle_watchdog(ctx.manager, ctx.runner))
        try:
            yield
        finally:
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog
            ctx.runner.shutdown()
            ctx.manager.unload()
            app.state.runner = None
            app.state.engine = None
            logger.info("ASR engine and runner released")

    app = FastAPI(title="GigaAM ASR", version=__version__, lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(transcriptions_router)
    app.include_router(models_router)
    return app


app = create_app()
