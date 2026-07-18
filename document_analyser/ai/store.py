"""BYOK (bring-your-own-key) AI provider store — Python port of the Electron
desktop host's electron/ai-providers.ts.

Part of the Tauri migration (document-lens plan §3.2): the AI provider config
and LLM calls move OUT of the desktop shell and INTO this backend so neither
the Electron main process nor a Rust core has to own key storage and CORS-bound
HTTP. The renderer talks to the /ai/* routes over the authenticated loopback
connection.

Keys are stored in the OS keychain via `keyring` (the Python equivalent of
Electron's safeStorage). When no keychain backend is available (e.g. a headless
Linux box with no keyring), we fall back to plaintext in the config file and
report `encryption_available = False` so the UI can warn — exactly mirroring the
Electron `plain:` fallback.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import keyring
import platformdirs

ApiShape = Literal["anthropic", "openai", "gemini"]
KeyMode = Literal["required", "optional", "none"]

# Keychain namespace. One entry per provider id.
_KEYRING_SERVICE = "document-lens-ai"


@dataclass(frozen=True)
class ProviderPreset:
    id: str
    label: str
    shape: ApiShape
    default_base_url: str
    key_mode: KeyMode


# Seven providers collapse into three API shapes — identical to ai-providers.ts.
PROVIDER_PRESETS: list[ProviderPreset] = [
    ProviderPreset("anthropic", "Anthropic", "anthropic", "https://api.anthropic.com", "required"),
    ProviderPreset("openai", "OpenAI", "openai", "https://api.openai.com/v1", "required"),
    ProviderPreset("gemini", "Google Gemini", "gemini", "https://generativelanguage.googleapis.com/v1beta", "required"),
    ProviderPreset("grok", "Grok (xAI)", "openai", "https://api.x.ai/v1", "required"),
    ProviderPreset("openai-compat", "OpenAI-compatible", "openai", "", "optional"),
    ProviderPreset("ollama", "Ollama (local)", "openai", "http://localhost:11434/v1", "none"),
    ProviderPreset("ollama-bearer", "Ollama + Bearer", "openai", "http://localhost:11434/v1", "required"),
]

_PRESET_BY_ID = {p.id: p for p in PROVIDER_PRESETS}


def preset_for(provider_id: str) -> ProviderPreset:
    preset = _PRESET_BY_ID.get(provider_id)
    if preset is None:
        raise ValueError(f"Unknown AI provider: {provider_id}")
    return preset


# --- Persistence -----------------------------------------------------------
#
# Non-secret config (baseUrl, model, active) lives in a JSON file. Secrets live
# in the OS keychain when available; otherwise a base64 "plain" blob is kept in
# the same JSON as the documented fallback.


def _config_path() -> Path:
    d = Path(platformdirs.user_config_dir("document-lens", appauthor=False))
    d.mkdir(parents=True, exist_ok=True)
    return d / "ai-providers.json"


def _load_config() -> dict:
    try:
        raw = _config_path().read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {"active": None, "providers": {}}
    return {"active": parsed.get("active"), "providers": parsed.get("providers") or {}}


def _save_config(config: dict) -> None:
    _config_path().write_text(json.dumps(config, indent=2), encoding="utf-8")


def encryption_available() -> bool:
    """True when a real OS keychain backend is present (not the fail/null one)."""
    module = type(keyring.get_keyring()).__module__
    return "fail" not in module and "null" not in module


def _store_key(provider_id: str, plain: str) -> dict:
    """Persist a key. Returns the JSON-side blob to record (None when the secret
    went to the keychain, a 'plain:'-prefixed blob otherwise)."""
    if encryption_available():
        try:
            keyring.set_password(_KEYRING_SERVICE, provider_id, plain)
            return {"keyBlob": None}
        except keyring.errors.KeyringError:
            pass  # fall through to plaintext
    return {"keyBlob": "plain:" + base64.b64encode(plain.encode("utf-8")).decode("ascii")}


def _clear_key(provider_id: str) -> None:
    try:
        keyring.delete_password(_KEYRING_SERVICE, provider_id)
    except keyring.errors.PasswordDeleteError:
        pass


def _read_key(provider_id: str, key_blob: str | None) -> str | None:
    """Reveal a stored key: keychain first, then the plaintext fallback blob."""
    if encryption_available():
        try:
            found = keyring.get_password(_KEYRING_SERVICE, provider_id)
            if found is not None:
                return found
        except keyring.errors.KeyringError:
            pass
    if key_blob and key_blob.startswith("plain:"):
        return base64.b64decode(key_blob[len("plain:"):]).decode("utf-8")
    return None


def _has_key(provider_id: str, stored: dict) -> bool:
    if stored.get("keyBlob"):
        return True
    if encryption_available():
        try:
            return keyring.get_password(_KEYRING_SERVICE, provider_id) is not None
        except keyring.errors.KeyringError:
            return False
    return False


# --- Public API (mirrors the exported functions of ai-providers.ts) --------


def get_providers() -> dict:
    """Snapshot returned to the renderer — never includes raw keys."""
    config = _load_config()
    providers = []
    for preset in PROVIDER_PRESETS:
        stored = config["providers"].get(preset.id) or {}
        providers.append(
            {
                "id": preset.id,
                "label": preset.label,
                "shape": preset.shape,
                "keyMode": preset.key_mode,
                "baseUrl": stored.get("baseUrl") or preset.default_base_url,
                "model": stored.get("model"),
                "hasKey": _has_key(preset.id, stored),
            }
        )
    return {
        "active": config["active"],
        "encryptionAvailable": encryption_available(),
        "providers": providers,
    }


def save_provider(provider_id: str, base_url: str, model: str | None, key: str | None) -> dict:
    """Save a provider's settings. `key is None` leaves the stored key
    untouched; `key == ''` clears it; a non-empty string replaces it."""
    preset_for(provider_id)  # validate
    config = _load_config()
    prev = config["providers"].get(provider_id) or {}
    key_blob = prev.get("keyBlob")

    if key is not None:
        if key == "":
            _clear_key(provider_id)
            key_blob = None
        else:
            _clear_key(provider_id)  # drop any stale keychain entry first
            key_blob = _store_key(provider_id, key)["keyBlob"]

    config["providers"][provider_id] = {"baseUrl": base_url, "model": model, "keyBlob": key_blob}
    _save_config(config)
    return get_providers()


def set_active_provider(provider_id: str | None) -> dict:
    config = _load_config()
    config["active"] = provider_id
    _save_config(config)
    return get_providers()


def reveal_key(provider_id: str) -> str | None:
    stored = _load_config()["providers"].get(provider_id) or {}
    return _read_key(provider_id, stored.get("keyBlob"))


@dataclass(frozen=True)
class ResolvedProvider:
    shape: ApiShape
    base_url: str
    key: str | None


def resolve(provider_id: str, draft_base_url: str | None = None, draft_key: str | None = None) -> ResolvedProvider:
    """Resolve to concrete (shape, baseUrl, key), letting an unsaved draft
    override the stored config so the user can test before saving."""
    preset = preset_for(provider_id)
    stored = _load_config()["providers"].get(provider_id) or {}
    base_url = (draft_base_url or stored.get("baseUrl") or preset.default_base_url).rstrip("/")
    if draft_key is not None:
        key = draft_key or None
    else:
        key = _read_key(provider_id, stored.get("keyBlob"))
    return ResolvedProvider(shape=preset.shape, base_url=base_url, key=key)


def active_selection() -> tuple[str, str] | None:
    """The active (provider_id, model), or None if not fully configured."""
    config = _load_config()
    active = config["active"]
    if not active:
        return None
    model = (config["providers"].get(active) or {}).get("model")
    if not model:
        return None
    return active, model
