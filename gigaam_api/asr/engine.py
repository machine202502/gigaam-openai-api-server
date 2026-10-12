"""Контракт ASR-движка: типы результатов, Protocol и исключения уровня движка.

Типы намеренно не зависят от gigaam — HTTP- и форматный слои импортируют их,
не зная о конкретном инференс-бэкенде.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable


class EngineInfo(TypedDict):
    """Снимок состояния движка для GET /health."""

    model: str
    device: str
    loaded: bool


@dataclass(frozen=True)
class WordTS:
    """Слово с таймстемпами (секунды от начала аудио)."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class SegmentTS:
    """Сегмент транскрипта; `words` заполняется при word-уровневой гранулярности."""

    text: str
    start: float
    end: float
    words: list[WordTS] | None = None


@dataclass(frozen=True)
class ASRResult:
    """Результат распознавания: полный текст, длительность и сегменты.

    Для короткого аудио (≤25с) `segments` — один сегмент `[0, duration]`.
    """

    text: str
    duration: float
    segments: list[SegmentTS]


class AudioTooLongError(Exception):
    """Аудио длиннее лимита MAX_AUDIO_SECONDS.

    Порог longform 25с обрабатывается внутри движка.
    """


class InferenceCancelledError(Exception):
    """Инференс прерван по запросу (клиент отключился между longform-батчами)."""


@runtime_checkable
class ASREngine(Protocol):
    """Контракт инференса; ничего не знает о HTTP.

    Реализация (`GigaAMEngine`) держит загруженную модель и сериализуется снаружи.
    `runtime_checkable` нужен, чтобы /health мог сузить тип `app.state.engine` через
    isinstance без импорта gigaam.
    """

    model_name: str
    device: str

    def transcribe(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ASRResult: ...

    def iter_segments(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[SegmentTS]: ...

    def info(self) -> EngineInfo: ...
