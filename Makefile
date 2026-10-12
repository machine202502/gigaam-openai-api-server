# Makefile — команды разработки GigaAM ASR.
# `make pre-commit` запускается ПОСЛЕ КАЖДОЙ ЗАДАЧИ и должен быть зелёным.
# Это цель Makefile, а не инструмент pre-commit.

HOST ?= 0.0.0.0
PORT ?= 8000

IMAGE ?= gigaam-api:latest

.PHONY: install run run-api download-weights-local lint format format-check typecheck test test-integration coverage check pre-commit clean \
        build-docker up down logs download-weights

install:  ## Установить зависимости (uv sync)
	uv sync

run:  ## Локальный запуск сервиса + веб-панели управления (см. .env: UI_PORT)
	uv run python -m gigaam_api.serve

run-api:  ## Локальный запуск только API с автоперезагрузкой (uvicorn --reload)
	uv run uvicorn gigaam_api.main:app --host $(HOST) --port $(PORT) --reload

download-weights-local:  ## Прогреть веса нативно (uv, без Docker) в MODELS_DIR из .env
	uv run python -m gigaam_api.download_weights

lint:  ## ruff check
	uv run ruff check .

format:  ## ruff format (применить)
	uv run ruff format .

format-check:  ## ruff format --check
	uv run ruff format --check .

typecheck:  ## mypy (strict)
	uv run mypy gigaam_api tests

test:  ## Юнит-тесты (без integration)
	uv run pytest -m "not integration"

test-integration:  ## Интеграционные тесты (реальная модель/сеть). Код выхода 5 («нет тестов») считается успехом.
	uv run pytest -m integration || [ $$? -eq 5 ]

coverage:  ## Отчёт покрытия (без порога)
	uv run pytest -m "not integration" --cov=gigaam_api --cov-report=term-missing

check: lint format-check typecheck test  ## Быстрый цикл: lint + format-check + mypy + unit

pre-commit: lint format-check typecheck test test-integration  ## Вся пачка тестов всех типов подряд

clean:  ## Удалить кэши инструментов
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# --- Docker/деплой (этап 06) ---
# Удобство для разработки на Mac. В проде деплой идёт через docker-compose.yml +
# `docker compose` (без make). Прод-образ — linux/amd64 (на Mac собирается через
# эмуляцию; нативно на amd64-хосте).

build-docker:  ## Собрать прод-образ (linux/amd64)
	docker build --platform linux/amd64 -t $(IMAGE) .

up:  ## docker compose up -d (запустить сервис)
	docker compose up -d

down:  ## docker compose down (остановить сервис)
	docker compose down

logs:  ## docker compose logs -f (логи сервиса)
	docker compose logs -f

download-weights:  ## Прогреть веса в ./models (разовый контейнер, без старта сервиса)
	docker compose --profile tools run --rm download-weights
