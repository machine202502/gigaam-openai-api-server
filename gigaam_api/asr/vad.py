"""Silero VAD + алгоритм чанкинга для longform-распознавания.

`merge_intervals_to_chunks` — чистая функция, дословный порт логики merge из
gigaam/vad_utils.py::segment_audio_file (_update_segments + цикл). Меняется только
источник speech-интервалов: pyannote → Silero.

Импорты silero/torch — ленивые (внутри функций), чтобы импорт модуля только ради
чистой функции чанкинга не тянул тяжёлый стек. Веса Silero встроены в пакет (без сети).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


def load_vad() -> object:
    """Загрузить Silero VAD один раз (JIT из пакета; без сети/кэша).

    Возвращаем непрозрачный handle (передаём в `speech_intervals`); JIT-стек —
    тот же, что у GigaAM (torch) — без onnxruntime, чтобы не плодить пулы потоков
    на слабых CPU (напр. ~4 ядра; см. ADR в CLAUDE.md).

    Защита причины: silero_vad вызывает `torch.set_num_threads(1)` при импорте
    (silero_vad/model.py, уровень модуля — срабатывает при первом импорте в процессе).
    Мы сохраняем число потоков ДО импорта silero и восстанавливаем после — чтобы загрузка
    VAD не имела глобального побочного эффекта; иначе это молча прибивает ВЕСЬ инференс
    к одному потоку (NUM_THREADS игнорируется, ~3x медленнее на 4-ядерном CPU).
    Сохранение ДОЛЖНО идти до импорта: перехват после прочитал бы уже испорченную 1.
    """
    import torch

    saved_threads = torch.get_num_threads()
    try:
        from silero_vad import load_silero_vad

        return load_silero_vad()
    finally:
        torch.set_num_threads(saved_threads)


def speech_intervals(
    wav: Tensor, model: object, *, threshold: float, sampling_rate: int = 16000
) -> list[tuple[float, float]]:
    """Вернуть речевые интервалы `(start, end)` в секундах через Silero.

    `wav` — float32 моно 16кГц (весь сигнал; пик памяти на этой стадии).
    Формат идентичен pyannote: список `(start, end)` в секундах.
    """
    from silero_vad import get_speech_timestamps

    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sampling_rate,
        threshold=threshold,
        return_seconds=True,
    )
    return [(float(ts["start"]), float(ts["end"])) for ts in timestamps]


def merge_intervals_to_chunks(
    intervals: list[tuple[float, float]],
    audio_duration: float,
    *,
    min_duration: float,
    max_duration: float,
    strict_limit: float,
    new_chunk_threshold: float,
) -> list[tuple[float, float]]:
    """Слить речевые интервалы в чанки и вернуть их границы (в секундах).

    Порт `segment_audio_file` без нарезки сигнала: чистая функция от интервалов и
    параметров (движок делает нарезку сам). Чанки длиннее `strict_limit`
    режутся на равные части (`int(d/strict_limit)+1`), как в upstream.
    """
    boundaries: list[tuple[float, float]] = []
    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0

    def _flush(curr_start: float, curr_end: float, curr_duration: float) -> None:
        if curr_duration > strict_limit:
            max_segments = int(curr_duration / strict_limit) + 1
            segment_duration = curr_duration / max_segments
            curr_end = curr_start + segment_duration
            for _ in range(max_segments - 1):
                boundaries.append((curr_start, curr_end))
                curr_start = curr_end
                curr_end += segment_duration
        boundaries.append((curr_start, curr_end))

    for raw_start, raw_end in intervals:
        start = max(0.0, raw_start)
        end = min(audio_duration, raw_end)
        if curr_duration == 0.0:
            curr_start = start
        elif curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration or curr_duration > min_duration
        ):
            _flush(curr_start, curr_end, curr_duration)
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _flush(curr_start, curr_end, curr_duration)

    return boundaries
