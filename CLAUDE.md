# CLAUDE.md — GigaAM ASR (OpenAI-совместимый сервис)

Сервис распознавания **русской речи** на базе [GigaAM](https://github.com/salute-developers/GigaAM),
выставляющий **OpenAI-совместимый** API (`POST /v1/audio/transcriptions`). Self-hosted в Docker на
любом CPU-хосте (Linux-сервер / NAS / мини-ПК), разработка — на macOS. Этот файл — гайд по проекту;
источник истины поведения — сам код.

## Критические правила (первопричины — не нарушать)

1. **Прод — CPU-only.** Docker на Mac работает в Linux-ВМ без Metal → GPU нет. MPS работает только
   на **нативном** `uv`-запуске, никогда в Docker. Целевой хост — self-hosted CPU-машина.
2. **Longform делаем сами** — Silero VAD + портированный чанкинг GigaAM. **Никогда не вызывать
   `model.transcribe_longform`** (тянет pyannote); нигде не должно быть `import pyannote`.
3. **Всегда оборачивать прямой `model.forward`/`model._decode` в `torch.inference_mode()`**
   (longform `_iter_chunks` и любой новый инференс-путь этапа 07). Без этого autograd +
   inference-tensor rotary-кэш короткого пути → `RuntimeError: Inference tensors cannot be saved
   for backward` (воспроизводится только short→long на одном экземпляре; заперто тестом
   `tests/integration/test_short_then_long_real.py`).
4. **Linux torch — CPU-колёса** (`2.12.0+cpu`, через `index+marker` в `pyproject.toml`). Никогда
   не возвращать CUDA в Linux-резолв — это добавит гигабайты nvidia/triton в образ.
5. **Образ самодостаточен:** вызывать ffmpeg/ffprobe из PATH образа (встроен через apt;
   `gigaam_api/audio.py`). Никогда не полагаться на хостовые бинарники — на хосте ffmpeg может
   отсутствовать.
6. **MPS на Mac** может требоваться `PYTORCH_ENABLE_MPS_FALLBACK=1` (GigaAM на MPS не тестирован upstream).
7. **Сервис батчевый, не realtime:** 10ч аудио = часы CPU-счёта (мин. 2 ядра, рек. 4). Длинные
   файлы — через `stream=true`.
8. **`load_vad()` должен восстанавливать число потоков torch.** Silero вызывает `torch.set_num_threads(1)`
   при импорте (`silero_vad/model.py`, уровень модуля), что молча прибивает ВЕСЬ инференс к одному
   потоку (`NUM_THREADS` игнорируется, ~3.3× медленнее на 4-ядерном CPU). `load_vad` сохраняет/
   восстанавливает счётчик, чтобы загрузка VAD не имела глобального побочного эффекта; заперто
   `tests/unit/test_vad_silero.py::test_load_vad_restores_torch_thread_count`. Эффективное число потоков
   логируется при загрузке движка (`threads=N`) для выявления регрессии.

## Команды (Makefile)

- `make install` — `uv sync`.
- `make run` — локальный запуск сервиса + веб-панели (python -m gigaam_api.serve).
- `make run-api` — только API с автоперезагрузкой (uvicorn --reload).
- `make check` — `lint` + `format-check` + `mypy(strict)` + unit — быстрый цикл.
- `make pre-commit` — `check` + integration-тесты; **должен быть зелёным после каждой задачи**.
  (Цель Makefile, а не инструмент pre-commit.)
- `make test` / `make test-integration` — юнит / тесты с реальной моделью.
- `make download-weights-local` — прогреть веса нативно (uv, без Docker) в `MODELS_DIR`.
- `make build-docker` / `up` / `down` / `logs` — Docker (удобство dev; прод использует `docker compose`).
- `make download-weights` — прогреть веса через Docker-профиль `tools`.
- Подцели: `lint`, `format`, `format-check`, `typecheck`, `coverage`, `clean`.

## Конфигурация

pydantic-settings загружается из `.env` (имена полей = имена переменных окружения). Источники истины:

- **Полная справочная таблица** — раздел «Конфигурация» в README.
- **Редактируемый шаблон с дефолтами** — `.env.example`.
- **Схема/валидация** — `gigaam_api/config.py` (`Settings`).

Ключевые переменные: `MODEL` (`v3_ctc`), `API_KEY` (пусто ⇒ auth выключен), `DEVICE` (`auto`→cuda/mps/cpu),
`MODELS_DIR`, `IDLE_UNLOAD_SECONDS` (300), `UI_PORT` (8001), `MAX_QUEUE` (8), `BATCH_SIZE` (4),
`NUM_THREADS` (4), `MAX_AUDIO_SECONDS` (36000), `MAX_UPLOAD_MB` (2048), `VAD_*` настройки чанкинга,
`ALLOWED_MODELS` (CSV).

## Конвенции

- **mypy strict**, **TDD** (тест → код), чистые функции для `formats`/VAD-чанкинга.
- **Тестируем прагматично:** покрываем рискованную/ключевую логику и happy path; не гонимся за
  покрытием и не пишем тесты ради тестов.
- **YAGNI:** удалять неиспользуемый/мёртвый код сразу (без закомментированных блоков). Единственное
  осознанное исключение — абстракция `ASREngine`, оставленная под ONNX.
- Никаких сетевых вызовов при импорте; скачивание весов — только в lifespan.
- Логирование через stdlib `logging`, уровень из `LOG_LEVEL`; debug-логи в ключевых точках.
- Держать **CLAUDE.md и README актуальными** в той же задаче, что меняет поведение/команды/API/
  конфигурацию; записывать новые архитектурные решения в таблицу ниже.
- Запускать **`make pre-commit` (зелёный) после каждой задачи** и в конце каждого этапа.
- **Не коммитить и не пушить без явного запроса.**

## Архитектурные решения (ADR-лог)

> Добавляйте новые архитектурные решения сюда строкой `решение · причина` — для переиспользования
> опыта между сессиями. Это живой раздел.

| Решение | Причина |
|---|---|
| Инференс-бэкенд — **PyTorch** за `ASREngine` Protocol (ONNX опционально, этап 07) | Полный API GigaAM из коробки; одна кодовая база для cpu/mps/cuda. |
| Длинное аудио VAD — **Silero VAD JIT** (`load_silero_vad(onnx=False)`), не pyannote, не onnxruntime | Без HF_TOKEN/лицензий; единый стек torch; onnxruntime по умолчанию создаёт потоки на все ядра → переподписка против пула torch на слабых CPU. Встроен в pip-пакет (без сети); переключение на ONNX потом = 1 строка. |
| **Python 3.12** + пакетный менеджер **uv** | Совместимость с torch/MPS; 3.14 слишком свежий. |
| Модель через `MODEL` env, **CTC по умолчанию** | CTC быстрее на целевом CPU-хосте. |
| Auth — **Bearer-ключ из `API_KEY`** (пусто ⇒ выключен) | Совместимость с OpenAI-клиентами. |
| Веса — **скачиваются при первом старте в volume `MODELS_DIR`**, не запечены в образ | Лёгкий образ. |
| API-поверхность — sync `POST /v1/audio/transcriptions` (+ опционально `stream=true` SSE), `GET /v1/models`, `GET /health` | Стандарт OpenAI; translations/WebSocket вне scope. |
| Сборка пакета — **hatchling** (редактируемый через `uv sync`) | `gigaam_api` импортируем в pytest/mypy/uvicorn. |
| Пины — **`>=` в `pyproject.toml`, точные пины в `uv.lock`**; gigaam git-pinned (`rev 559d88d6`) в `[tool.uv.sources]` как «голая» зависимость `gigaam` | Идиома uv: воспроизводимость через lock, осознанные апгрейды через `uv lock --upgrade`. |
| **`DEVICE=auto` резолвим сами** (cuda→mps→cpu), явное устройство передаём в `load_model` | Собственный auto у GigaAM пропускает MPS (cuda→cpu); MPS нужен на dev-Mac. |
| CSV-настройки (`ALLOWED_MODELS`) — **`Annotated[..., NoDecode]` + `field_validator`** | pydantic-settings парсит `list` как JSON; NoDecode + `split(",")` даёт CSV. |
| `ASREngine` — **`@runtime_checkable` Protocol** (`transcribe`/`info`/`iter_segments`); `/health` сужает тип движка без импорта gigaam/torch | «HTTP ⟂ inference»: HTTP-слой без torch; движок импортируется лениво в lifespan. |
| mypy — **per-module `ignore_missing_imports`** для `gigaam.*`/`silero_vad.*` | Нет py.typed/stubs; точечный override лучше широкого `# type: ignore`. |
| Маршрутизация по длительности **внутри движка**: `>MAX_AUDIO_SECONDS`→`AudioTooLongError`; `≤25s`→short (делегирование `model.transcribe`); иначе→longform. Near-boundary `ValueError "too long"`→fallback в longform | gigaam меряет по сэмплам, probe — по секундам → у границы 25с longform лучше падения; горячий короткий путь не трогаем. |
| Longform — порт `gigaam vad_utils.segment_audio_file`: **чистая `merge_intervals_to_chunks` (только границы)** + нарезка/батчинг в движке; инференс через `model.forward`/`model._decode`; слова `+seg_start`, `round(...,3)` | Чистая логика merge unit-тестируется изолированно; никогда не вызываем upstream `transcribe_longform` (тянет pyannote). |
| Longform-память — декодирование в **int16 `torch.Tensor`** (`torch.frombuffer`, без numpy); float-копия только для стадии VAD, сразу освобождается | int16 вдвое снижает память (~1.15 ГБ/10ч); float-пик — на VAD, не на батчах; `audio.py` без torch для HTTP-слоя (ленивые импорты torch). |
| Longform-инференс (`forward`/`_decode`) **должен идти под `torch.inference_mode()`** | Иначе autograd + inference-tensor rotary-кэш короткого пути → `RuntimeError: Inference tensors cannot be saved for backward`; регрессия заперта `test_short_then_long_real.py`. |
| Отмена longform при отключении — **кооперативная** `cancel_check` проверяется на батч → `InferenceCancelledError`; API следит через `request.is_disconnected()`. Короткий путь не отменяем | Задачу ThreadPool нельзя прервать; один воркер ⇒ брошенный longform заблокирует очередь всем. |
| Backpressure — единый ключ **`MAX_QUEUE`** (принято = в очереди + в работе); сверх → `QueueFullError`→**503**. **Без таймаута запроса** | Таймаут порезал бы легитимные многочасовые файлы (RTF≥1); брошенные задачи обрабатываются отменой, не таймаутом. |
| Маппинг ошибок разделяет причину: `AudioToolNotFoundError` (нет ffmpeg/ffprobe)→**500** `api_error`; `AudioDecodeError` (битый/неподдерживаемый файл)→**400** `invalid_request_error`. Без 415 | Настоящий OpenAI возвращает 400 `invalid_request_error` на битое аудио, никогда 415; один код для клиентских и серверных причин неверен. |
| Детали совместимости с OpenAI — `timestamp_granularities[]` form alias; verbose `seek=0` + per-segment `compression_ratio`; `/v1/models` эхом возвращает `ALLOWED_MODELS` | Совпадает с каноничным wire-форматом OpenAI-клиента. |
| `compression_ratio` — **байты/байты** `len(b)/len(zlib.compress(b))`, `b=text.encode()` | Whisper считает байты; счёт кириллических символов (2 байта каждый) вдвое уменьшил бы ratio и никогда не триггерил бы порог галлюцинации >2.4. |
| Наблюдатель отключения — **`anyio.create_task_group()` + `cancel_scope.cancel()`** (не raw `asyncio` cancel); результат инференса захватывается внутри группы и извлекается снаружи | `Request.is_disconnected()` держит anyio CancelScope; raw-asyncio cancel конфликтует → deadlock запроса. Захват снаружи сохраняет `QueueFullError`→503 (не обёрнуто в ExceptionGroup→500). |
| Семантика SSE delta — **префиксный пробел**: первый delta = `seg0.text`, остальные = `" "+segN.text`; `done.text=" ".join(...)`. Инвариант `"".join(delta)==done.text==sync` | Универсальный инвариант OpenAI-стриминга: конкатенация дельт в точности воспроизводит финальный текст. |
| Мост блокирующий-`iter_segments`→async — **`asyncio.Queue` + `loop.call_soon_threadsafe`**, продюсер в `Runner.submit` (тот же единственный воркер), без `maxsize` | Сохраняет сериализацию инференса, никогда не блокирует цикл; heartbeat через `wait_for(queue.get(), 15s)` (безопасно отменять свою корутину), а не через отмену чужого генератора. |
| Стриминговый backpressure — **`try_acquire()` ДО `StreamingResponse`** (503 без заголовков); `release()` в done-callback продюсера; `_inflight` под блокировкой | Async-генератор откладывает тело после 200 → решение о 503 должно быть принято раньше; release по завершению продюсера = inflight отслеживает занятость воркера, а не скорость чтения клиента. |
| **Владение временным файлом передаётся потоку** — `finally` обработчика не удаляет; `_cleanup` (done-callback продюсера) удаляет после завершения чтения воркером | Обработчик возвращается немедленно → его `finally` удалил бы файл до чтения инференсом. |
| Отмена потока — **`cancel_event.set()` в `finally` мостового генератора** (Starlette отменяет генератор при отключении); `sse_transcription` ловит `Exception`→`error` event, но пропускает `CancelledError`/`GeneratorExit` | `iter_segments` останавливается между батчами (та же гранулярность, что и sync); отключение = только очистка. |
| `verbose_json`/`srt`/`vtt` + `stream=true` → **синхронный fallback** (стриминг только при `fmt in {json,text}`), не 400 | Большинство клиентов шлют `stream=true` по умолчанию с `verbose_json`; 400 сломало бы их. |
| `iter_segments` — **общий батчевый цикл `_iter_chunks`** (+ `_prepare_longform`), переиспользуемый sync `_transcribe_longform`; часть протокола `ASREngine` | Единый источник longform-логики (DRY); ≤25с делегирует короткому пути и yield-ит один сегмент. |
| CPU-torch — **`index+marker` в `pyproject.toml`**: `[[tool.uv.index]] pytorch-cpu` (`explicit`) + torch/torchaudio с маркером `sys_platform=='linux'`. Один `uv.lock`: Mac `2.12.0` (MPS), Linux `2.12.0+cpu` | Идиома uv; побочным эффектом весь стек CUDA (nvidia/triton) выпадает из Linux-резолва. Dockerfile — просто `uv sync --frozen`. |
| **`onnx` переопределён на `>=1.21.0`** через `[tool.uv] override-dependencies` | `onnx` транзитивен через gigaam, который жёстко пинит `onnx==1.19.*` (открытые high/medium GHSA-уязвимости, исправлены в 1.21.0); этот пин делал Dependabot security-обновления неразрешимыми. Безопасно: gigaam не делает «голый» `import onnx` (только onnxruntime + `torch.onnx.export`), а этот сервис никогда не вызывает ONNX-экспорт; зелёный `make pre-commit` вкл. real-inference тесты подтверждает. |
| Образ — **multi-stage `python:3.12-slim`**: builder (uv + git; слой зависимостей, затем код) + тонкий runtime (ffmpeg из apt, non-root UID/GID 1000, healthcheck через stdlib urllib, `XDG_CACHE_HOME=/data/models/.cache`). Платформа — на этапе сборки (`--platform`), не в `FROM` | Самодостаточность (ffmpeg может отсутствовать на хосте); кэш зависимостей отдельно от кода; multi-arch + быстрая нативная arm64-валидация на Mac. |
| **Silero встроен в pip-пакет** ⇒ volume нужен **только** для весов GigaAM (`MODELS_DIR`) | Без HF Hub / torch.hub; `XDG_CACHE_HOME`→volume оставлен только как non-root страховка. |
| Тестовые фикстуры — закоммичены `tests/integration/data/ru_short_sample.wav` (~11с) и `ru_long_sample.wav` (40с, вырезан из GigaAM `long_example.wav`); integration-тесты идут на **cpu** и пропускаются без весов/сети | Отслеживаемая реальная русская речь (длинный файл с паузами ⇒ >1 чанка); имена ≠ одноразовый `example.wav`, который отбрасывает `.gitignore`; cpu = детерминизм + прод-паритет. |
| Деплой — **`docker-compose.yml` + `docker compose` CLI; без `make` в проде**. Первый `up` скачивает веса (`healthcheck start_period 600s`); опциональный `tools`-профиль `download-weights` для прогрева | Прод-путь должен «просто работать» из compose (в т.ч. из-за UI NAS). Цели `make` — только dev-удобство на Mac. |
| Язык проекта — **русский** (комментарии/docstring + сообщения логов/ошибок на английском); ruff RUF001/002/003 отключены | Код и документация на русском. Транскрипты русской речи в тестах — как тестовые данные ASR. |
| **`VAD_THRESHOLD` дефолт 0.5 → 0.25** | Эмпирически (pooled WER на 14 файлах FLEURS-ru, чистый + шум 15/5дБ, v3_ctc и v3_e2e_rnnt): 0.2–0.3 снижает WER на 10–23% относительно 0.5 — ниже порог = меньше ложных пауз = больше контекста на чанк. Пол: 0.2 — при ~0.1 VAD перестаёт видеть паузы → блоки >30с → арифметические порезы посреди слова. |
| **Без overlap/snap постобработки границ чанков** (YAGNI) | Измерено: естественная русская речь никогда не имеет >30с непрерывных блоков (макс ~5–6с при пороге ≥0.3) → арифметический порез, режущий слова, почти никогда не срабатывает; selective-overlap == baseline. Energy-snap небезопасен (нельзя отличить внутрисловный провал энергии от паузы; в худшем случае вредит). Если домен станет без пауз (пение/плотная речь), добавить **selective overlap+stitch на ARITH-границах**, а не snap. |
| **`NUM_THREADS` восстанавливается в `load_vad()`** (silero `set_num_threads(1)` при импорте затирал → весь инференс однонитевой) | Измерено в Linux/OpenMP-контейнере (4 ядра, прод-паритет): молчаливый затир прибивал инференс к 1 потоку; восстановление `NUM_THREADS=4` урезало wall-clock в 3.3× (RTF 0.125→0.038) на файле 754с, почти линейно числу ядер, переподписка (8 потоков на 4 ядра = медленнее). Исправлено в причине (VAD-загрузчик без побочных эффектов), а не переупорядочиванием init движка. **Параллельная обработка чанков отвергнута как YAGNI**: torch intra-op уже насыщает ядра; дополнительная конкурентность только переподписывает. Пересмотреть только на многоядерном хосте, где один `forward` упирается ниже числа ядер. |
| Пин gigaam **`6e4b027`→`559d88d6`** (v0.1.0→v0.2.0) — исправления batched-encoder (#75 conv-padding mask, #81 SDPA all-masked-row NaN) | Наш longform-путь батчит zero-padded чанки через `model.forward`, применимы оба. Граф зависимостей не изменился (override onnx всё ещё нужен; pyannote всё ещё только `longform`-extra; без новых/удалённых пакетов); `model.py` не тронут → `forward`/`_decode`/`transcribe`/`_dtype` идентичны. Эмпирически **идентичный вывод** до/после на реальных RU-файлах + **0/10** состязательных mid-word 3–15× padded срезов (v3_ctc и v3_e2e_rnnt): расхождение — на уровне логитов, ниже порога greedy argmax-flip → страховка корректности, а не измеримое изменение WER. Бонус (не используется): `multilingual_*` модели (scope только RU), retry скачивания весов. Зелёный `make pre-commit`. |
| **Автовыгрузка модели по простою** (`ModelManager`) + веб-панель управления на отдельном порту | Сервис крутится постоянно в Docker; если модель не используется дольше `IDLE_UNLOAD_SECONDS` (по умолчанию 300с) — выгружается из памяти, при следующем запросе загружается прозрачно. `ModelManager` реализует `ASREngine` и подменяет движок в `app.state.engine`. Сторожевой таймер (`idle_watchdog`) через единый воркер `Runner` сериализует выгрузку с инференсом. Веб-панель (`gigaam_api/admin.py`) на порту `UI_PORT` (8001) — в одном процессе с API, через фоновый поток uvicorn; делит `AppContext` с основным приложением. UI: HTML/JS без зависимостей, позволяет менять порог выгрузки и загружать/выгружать модель вручную. Auth панели — тот же Bearer-ключ `API_KEY`. |
| Язык проекта сменён на **русский**; ruff RUF001/002/003 отключены | Комментарии, docstring и документация — на русском. Сообщения логов/ошибок оставлены на английском (операционные строки, не комментарии). README_ru.md удалён (дубликат), README.md — русский. |

<!-- Добавляйте новые решения строкой выше этой подсказки. -->

## Справка по GigaAM

[GigaAM](https://github.com/salute-developers/GigaAM) — upstream-библиотека русской ASR, которую оборачивает этот сервис; git-pinned зависимость (`rev 559d88d6` в `[tool.uv.sources]`), не vendor-нутая. Переиспользуем её инференс (`model.transcribe`, `forward`/`_decode`) и портировали её VAD-чанкинг. Ключевые ссылки:
[`gigaam/model.py`](https://github.com/salute-developers/GigaAM/blob/559d88d6/gigaam/model.py),
[`gigaam/vad_utils.py`](https://github.com/salute-developers/GigaAM/blob/559d88d6/gigaam/vad_utils.py)
(алгоритм чанкинга), `gigaam/decoding.py`, `gigaam/__init__.py` (загрузка/кэш весов).