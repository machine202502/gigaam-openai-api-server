"""Общий рантайм-контекст приложения.

Один экземпляр `AppContext` разделяется между основным приложением (порт API) и веб-панелью
управления (отдельный порт): обоим нужен доступ к одному и тому же `ModelManager` и `Runner`,
чтобы панель могла выгружать/загружать ту же модель, что используется при инференсе.
"""

from dataclasses import dataclass

from gigaam_api.asr.model_manager import ModelManager
from gigaam_api.config import Settings
from gigaam_api.runner import Runner


@dataclass
class AppContext:
    """Настройки + общие `Runner` и `ModelManager`."""

    settings: Settings
    runner: Runner
    manager: ModelManager


def build_context(settings: Settings) -> AppContext:
    """Собрать контекст из настроек."""
    return AppContext(
        settings=settings,
        runner=Runner(max_queue=settings.MAX_QUEUE),
        manager=ModelManager(settings),
    )
