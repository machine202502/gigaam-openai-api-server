"""Единая точка запуска сервиса: API (основной порт) + веб-панель управления (UI_PORT).

Запускает два сервера uvicorn в одном процессе:
- основной API — в главном потоке (стандартная обработка сигналов);
- панель управления — в фоновом потоке (там uvicorn не перехватывает сигналы).

Оба используют общий `AppContext`, поэтому панель управляет той же моделью, что и API.

Запуск: `python -m gigaam_api.serve` (то же делает Docker-образ и `make run`).
"""

import logging
import threading

import uvicorn

from gigaam_api.admin import create_admin_app
from gigaam_api.config import get_settings
from gigaam_api.context import build_context
from gigaam_api.logging_setup import setup_logging
from gigaam_api.main import create_app

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    setup_logging(settings)
    context = build_context(settings)
    main_app = create_app(context)

    if settings.UI_ENABLED:
        # Панель — в фоновом потоке: uvicorn в не-главном потоке не перехватывает сигналы.
        admin_app = create_admin_app(context)
        admin_config = uvicorn.Config(
            admin_app,
            host=settings.UI_HOST,
            port=settings.UI_PORT,
            log_config=None,
            access_log=False,
        )
        admin_server = uvicorn.Server(admin_config)
        threading.Thread(target=admin_server.run, name="gigaam-admin", daemon=True).start()
        logger.info("admin UI on http://%s:%d", settings.UI_HOST, settings.UI_PORT)

    config = uvicorn.Config(main_app, host=settings.HOST, port=settings.PORT, log_config=None)
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
