"""Работа со звуком: probe длительности (ffprobe) и декодирование в int16 16кГц моно (ffmpeg).

Короткий путь распознавания делегирует декодирование самому gigaam (`model.transcribe`
внутри вызывает ffmpeg, см. gigaam/preprocess.py::load_audio). Longform-путь декодирует
сам через `decode_to_int16_16k_mono` (int16 экономит память на длинных файлах).

torch импортируется **лениво** внутри `decode_to_int16_16k_mono`: модуль остаётся
torch-free, чтобы импорт audio.py из HTTP-слоя не тянул torch.
"""

from __future__ import annotations

import logging
import subprocess
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class AudioDecodeError(Exception):
    """Не удалось прочитать/декодировать аудио (битый/неподдерживаемый файл)."""


class AudioToolNotFoundError(Exception):
    """ffprobe/ffmpeg не найдены в PATH — проблема окружения на сервере (→ 500)."""


def probe_duration(path: str) -> float:
    """Вернуть длительность аудио в секундах через `ffprobe`.

    ffprobe надёжно определяет длительность для любого формата, поддерживаемого
    ffmpeg. Любая ошибка (нет файла, битый вход, неизвестная длительность) →
    AudioDecodeError, чтобы наружу не летели сырые ошибки subprocess.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise AudioToolNotFoundError("ffprobe not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        logger.warning("ffprobe could not read audio: %s", path)
        raise AudioDecodeError(f"failed to read audio: {path}") from exc

    raw = proc.stdout.strip()
    try:
        duration = float(raw)
    except ValueError as exc:
        raise AudioDecodeError(f"ffprobe returned an invalid duration: {raw!r}") from exc

    logger.debug("probe_duration %s -> %.3fs", path, duration)
    return duration


def decode_to_int16_16k_mono(path: str) -> Tensor:
    """Декодировать аудио в 1-D **int16** torch.Tensor (16кГц моно) через ffmpeg.

    Как gigaam `load_audio`, но возвращаем int16 (не float): на длинных файлах это
    вдвое снижает потребление памяти (~1.15 ГБ/10ч против 2.3 ГБ во float). Во float
    переводим по чанкам при батчинге. Тензор только для чтения (делит память с буфером
    ffmpeg) — инференс делает копии (`.float()`), сам буфер не мутируется.
    """
    import torch

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError as exc:
        raise AudioToolNotFoundError("ffmpeg not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg could not decode audio: %s", path)
        raise AudioDecodeError(f"failed to decode audio: {path}") from exc

    with warnings.catch_warnings():
        # torch.frombuffer на bytes даёт read-only тензор → UserWarning, как в gigaam.
        warnings.simplefilter("ignore", category=UserWarning)
        wav: Tensor = torch.frombuffer(raw, dtype=torch.int16)

    logger.debug(
        "decode_to_int16_16k_mono %s -> %d samples (%.1f MB int16)",
        path,
        wav.numel(),
        wav.numel() * 2 / 1024 / 1024,
    )
    return wav
