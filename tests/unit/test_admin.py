"""Тесты веб-панели управления: статус, смена порога автовыгрузки, загрузка/выгрузка."""

from collections.abc import Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gigaam_api.asr.gigaam_engine as ge
from gigaam_api.admin import create_admin_app
from gigaam_api.asr.engine import ASRResult, EngineInfo, SegmentTS
from gigaam_api.asr.model_manager import ModelManager
from gigaam_api.config import Settings, get_settings
from gigaam_api.context import build_context


class _FakeEngine:
    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.MODEL
        self.device = "cpu"

    def transcribe(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ASRResult:
        raise NotImplementedError

    def iter_segments(
        self,
        wav_path: str,
        *,
        word_timestamps: bool,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[SegmentTS]:
        raise NotImplementedError

    def info(self) -> EngineInfo:
        return {"model": self.model_name, "device": self.device, "loaded": True}


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ge, "GigaAMEngine", _FakeEngine)


def _build(settings: Settings | None = None) -> tuple[FastAPI, ModelManager]:
    context = build_context(settings or Settings())
    context.manager.load()
    return create_admin_app(context), context.manager


def test_index_serves_html() -> None:
    app, _ = _build()
    resp = TestClient(app).get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "управление" in resp.text.lower()


def test_status_reports_loaded() -> None:
    app, _ = _build(Settings(IDLE_UNLOAD_SECONDS=120))
    body = TestClient(app).get("/api/status").json()
    assert body["loaded"] is True
    assert body["model"] == "v3_ctc"
    assert body["idle_seconds"] == 120
    assert body["max_queue"] == 8


def test_set_idle_updates_threshold() -> None:
    app, manager = _build()
    resp = TestClient(app).post("/api/idle", json={"idle_seconds": 30})
    assert resp.status_code == 200
    assert resp.json()["idle_seconds"] == 30
    assert manager.idle_seconds == 30


def test_set_idle_rejects_negative() -> None:
    app, _ = _build()
    resp = TestClient(app).post("/api/idle", json={"idle_seconds": -5})
    assert resp.status_code == 400


def test_manual_unload_and_reload() -> None:
    app, manager = _build()
    client = TestClient(app)

    assert client.post("/api/unload").json()["loaded"] is False
    assert manager.is_loaded is False

    assert client.post("/api/reload").json()["loaded"] is True
    assert manager.is_loaded is True


def test_auth_required_when_api_key_set() -> None:
    app, _ = _build()
    app.dependency_overrides[get_settings] = lambda: Settings(API_KEY="secret")
    client = TestClient(app)

    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"Authorization": "Bearer secret"}).status_code == 200
