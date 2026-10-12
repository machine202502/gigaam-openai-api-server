"""Юнит-тесты обёрток Silero VAD: load_vad + speech_intervals.

Используют настоящую JIT-модель Silero (входит в пакет silero-vad, без сети) —
поэтому это честный тест без моков. Путь с речью (реальная речь → непустые
интервалы) покрыт интеграционным тестом longform.
"""

import pytest
import torch

from gigaam_api.asr.vad import load_vad, speech_intervals


def test_load_vad_returns_usable_model() -> None:
    model = load_vad()
    assert model is not None


def test_load_vad_restores_torch_thread_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_vad не должен утекать побочный эффект импорта silero."""

    # silero_vad выполняет torch.set_num_threads(1) при *импорте*
    # (silero_vad/model.py, уровень модуля) — срабатывает один раз, при первом импорте в процессе.
    # load_vad должен запомнить число потоков ДО импорта silero и восстановить его после, иначе он
    # молча прибьёт весь последующий инференс к одному потоку (NUM_THREADS игнорируется,
    # ~в 3 раза медленнее на 4-ядерном CPU; см. ADR).
    import sys

    for name in [m for m in sys.modules if m == "silero_vad" or m.startswith("silero_vad.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    original = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        load_vad()
        assert torch.get_num_threads() == 4
    finally:
        torch.set_num_threads(original)


def test_speech_intervals_on_silence_returns_empty_list() -> None:
    model = load_vad()
    wav = torch.zeros(16000 * 2)  # 2с тишины, float32 16kHz → речи нет
    intervals = speech_intervals(wav, model, threshold=0.5)
    assert intervals == []


def test_speech_intervals_returns_list_of_float_pairs() -> None:
    # Контракт возврата не зависит от наличия речи: list[(float, float)].
    model = load_vad()
    wav = torch.zeros(16000).float()
    intervals = speech_intervals(wav, model, threshold=0.5)
    assert isinstance(intervals, list)
    assert all(
        isinstance(pair, tuple)
        and len(pair) == 2
        and isinstance(pair[0], float)
        and isinstance(pair[1], float)
        for pair in intervals
    )
