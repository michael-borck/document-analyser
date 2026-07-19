"""Tests for the BYOK AI provider fold (document-lens Tauri migration §3.2).

Covers the store round-trip and the /ai/* routes. Keys use the plaintext
fallback here because the test environment forces the null keyring backend
(PYTHON_KEYRING_BACKEND) — the same code path a keyring-less host would take.
Each test isolates the on-disk config so the developer's real config is never
touched.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from document_analyser.ai import store


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch):
    """Point the store at a throwaway config file for every test."""
    cfg = tmp_path / "ai-providers.json"
    monkeypatch.setattr(store, "_config_path", lambda: cfg)
    yield


class TestStore:
    def test_default_snapshot_lists_seven_providers(self):
        snap = store.get_providers()
        assert len(snap["providers"]) == 7
        assert snap["active"] is None
        assert {p["id"] for p in snap["providers"]} == {
            "anthropic", "openai", "gemini", "grok", "openai-compat", "ollama", "ollama-bearer",
        }

    def test_save_reveal_active_clear_roundtrip(self):
        store.save_provider("openai", "https://api.openai.com/v1", "gpt-4o", "sk-secret")
        prov = _by_id(store.get_providers(), "openai")
        assert prov["hasKey"] is True
        assert prov["model"] == "gpt-4o"
        assert store.reveal_key("openai") == "sk-secret"

        store.set_active_provider("openai")
        assert store.get_providers()["active"] == "openai"

        # key="" clears; key omitted (None) would leave it untouched.
        store.save_provider("openai", "https://api.openai.com/v1", "gpt-4o", "")
        assert _by_id(store.get_providers(), "openai")["hasKey"] is False

    def test_save_with_none_key_preserves_existing(self):
        store.save_provider("anthropic", "https://api.anthropic.com", "claude", "sk-keep")
        store.save_provider("anthropic", "https://api.anthropic.com", "claude-2", None)
        assert store.reveal_key("anthropic") == "sk-keep"  # untouched
        assert _by_id(store.get_providers(), "anthropic")["model"] == "claude-2"

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError):
            store.save_provider("nope", "x", None, None)


class TestRoutes:
    def test_get_providers(self, client: TestClient):
        r = client.get("/ai/providers")
        assert r.status_code == 200
        assert len(r.json()["providers"]) == 7

    def test_save_then_reveal_via_http(self, client: TestClient):
        r = client.post(
            "/ai/providers/grok",
            json={"baseUrl": "https://api.x.ai/v1", "model": "grok-2", "key": "xai-abc"},
        )
        assert r.status_code == 200
        assert _by_id(r.json(), "grok")["hasKey"] is True

        r = client.post("/ai/reveal/grok")
        assert r.json()["key"] == "xai-abc"

    def test_chat_without_active_provider_returns_error(self, client: TestClient):
        r = client.post("/ai/chat", json={"system": "s", "user": "u"})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert "active AI provider" in body["error"]


def _by_id(snapshot: dict, provider_id: str) -> dict:
    return next(p for p in snapshot["providers"] if p["id"] == provider_id)
