"""Веб-панель управления моделью — на отдельном порту (`UI_PORT`).

Одна страница: показывает состояние (загружена ли модель, устройство, простой, очередь),
позволяет задать порог автовыгрузки (`IDLE_UNLOAD_SECONDS`) и вручную загрузить/выгрузить модель.
Панель работает в том же процессе, что и основной сервис, и делит с ним общий `AppContext`,
поэтому кнопки действуют именно на ту модель, что используется при инференсе.

Эндпоинты (`/api/*`) защищены тем же Bearer-ключом `API_KEY`, что и API (если он задан).
HTML-страница отдаётся без auth, чтобы можно было ввести ключ в браузере.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from gigaam_api.auth import require_auth
from gigaam_api.context import AppContext
from gigaam_api.errors import register_exception_handlers

logger = logging.getLogger(__name__)


class IdleConfig(BaseModel):
    """Тело запроса на смену порога автовыгрузки."""

    idle_seconds: int = Field(ge=0)


async def _submit[T](context: AppContext, fn: Callable[[], T]) -> T:
    """Выполнить блокирующую операцию в единственном воркере (сериализация с инференсом)."""
    return await asyncio.wrap_future(context.runner.submit(fn))


def _status(context: AppContext) -> dict[str, Any]:
    """Снимок состояния модели и очереди для UI."""
    return {
        **context.manager.status(),
        "inflight": context.runner.inflight,
        "max_queue": context.runner.max_queue,
        "ui_port": context.settings.UI_PORT,
    }


def create_admin_app(context: AppContext) -> FastAPI:
    """Собрать приложение веб-панели, привязанное к общему контексту."""
    app = FastAPI(title="GigaAM ASR — управление", docs_url=None, redoc_url=None)
    register_exception_handlers(app)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    @app.get("/api/status", dependencies=[Depends(require_auth)])
    def status() -> dict[str, Any]:
        return _status(context)

    @app.post("/api/idle", dependencies=[Depends(require_auth)])
    def set_idle(config: IdleConfig) -> dict[str, Any]:
        context.manager.idle_seconds = config.idle_seconds
        return _status(context)

    @app.post("/api/unload", dependencies=[Depends(require_auth)])
    async def unload() -> dict[str, Any]:
        unloaded = await _submit(context, context.manager.unload)
        logger.info("manual unload via UI: unloaded=%s", unloaded)
        return _status(context)

    @app.post("/api/reload", dependencies=[Depends(require_auth)])
    async def reload_model() -> dict[str, Any]:
        await _submit(context, context.manager.load)
        logger.info("manual load via UI")
        return _status(context)

    return app


# HTML/JS — одна строковая константа (лишние зависимости не нужны).
# E501 в этом файле отключён per-file-ignores в pyproject.toml.
_INDEX_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GigaAM ASR — управление</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 0; padding: 24px; max-width: 720px; }
  h1 { font-size: 20px; margin: 0 0 16px; }
  .card { border: 1px solid #8884; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 4px 0; }
  .label { opacity: .7; min-width: 190px; }
  .val { font-weight: 600; }
  button { padding: 8px 14px; border-radius: 6px; border: 1px solid #8886; cursor: pointer; }
  input { padding: 8px; border-radius: 6px; border: 1px solid #8886; min-width: 120px; }
  #msg { min-height: 1.2em; opacity: .8; }
  .hint { opacity: .6; font-size: 13px; }
</style>
</head>
<body>
<h1>GigaAM ASR — управление моделью</h1>

<div class="card">
  <div class="row"><span class="label">Статус</span><span class="val" id="loaded">…</span></div>
  <div class="row"><span class="label">Модель</span><span class="val" id="model">…</span></div>
  <div class="row"><span class="label">Устройство</span><span class="val" id="device">…</span></div>
  <div class="row"><span class="label">Простой</span><span class="val" id="idle_for">…</span></div>
  <div class="row"><span class="label">Порог выгрузки, с</span><span class="val" id="idle_seconds">…</span></div>
  <div class="row"><span class="label">Очередь (в работе)</span><span class="val" id="queue">…</span></div>
</div>

<div class="card">
  <div class="row">
    <span class="label">Выгружать после простоя, с</span>
    <input id="idle_input" type="number" min="0" step="30">
    <button onclick="saveIdle()">Сохранить</button>
  </div>
  <div class="hint">0 — автовыгрузка выключена (модель всегда в памяти).</div>
</div>

<div class="card">
  <div class="row">
    <button onclick="doReload()">Загрузить модель</button>
    <button onclick="doUnload()">Выгрузить модель сейчас</button>
  </div>
  <div class="hint">Загрузка может занять время (чтение весов из кэша).</div>
</div>

<div class="card">
  <div class="row">
    <span class="label">API-ключ (если задан API_KEY)</span>
    <input id="key" type="password" placeholder="Bearer-ключ">
    <button onclick="saveKey()">Сохранить ключ</button>
  </div>
</div>

<div id="msg"></div>

<script>
const $ = (id) => document.getElementById(id);
function authHeaders() {
  const key = localStorage.getItem('gigaam_key') || '';
  return key ? {'Authorization': 'Bearer ' + key} : {};
}
async function api(path, opts) {
  opts = opts || {};
  opts.headers = Object.assign({}, authHeaders(), opts.headers || {});
  const resp = await fetch(path, opts);
  if (!resp.ok) throw new Error(resp.status + ': ' + (await resp.text()));
  return resp.json();
}
function msg(text) {
  $('msg').textContent = text;
  setTimeout(() => { $('msg').textContent = ''; }, 4000);
}
async function refresh() {
  const s = await api('/api/status');
  $('loaded').textContent = s.loaded ? 'загружена' : 'выгружена';
  $('model').textContent = s.model;
  $('device').textContent = s.device;
  $('idle_for').textContent = Math.round(s.idle_for_seconds) + ' с';
  $('idle_seconds').textContent = s.idle_seconds + ' с';
  $('queue').textContent = s.inflight + ' / ' + s.max_queue;
  const input = $('idle_input');
  if (document.activeElement !== input) input.value = s.idle_seconds;
}
async function saveIdle() {
  const idle_seconds = parseInt($('idle_input').value, 10);
  await api('/api/idle', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({idle_seconds: idle_seconds}),
  });
  msg('Порог сохранён');
  await refresh();
}
async function doReload() {
  msg('Загружаю модель…');
  await api('/api/reload', {method: 'POST'});
  msg('Модель загружена');
  await refresh();
}
async function doUnload() {
  await api('/api/unload', {method: 'POST'});
  msg('Модель выгружена');
  await refresh();
}
function saveKey() {
  localStorage.setItem('gigaam_key', $('key').value);
  msg('Ключ сохранён');
}
$('key').value = localStorage.getItem('gigaam_key') || '';
refresh().catch((e) => msg(e.message));
setInterval(() => refresh().catch(() => {}), 3000);
</script>
</body>
</html>
"""
