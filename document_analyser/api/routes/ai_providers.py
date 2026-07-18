"""BYOK AI provider routes for the desktop host (document-lens).

Folded in during the Tauri migration (plan §3.2). These endpoints replace the
Electron main process's ai:* IPC handlers; the renderer calls them over the
authenticated loopback connection. Raw keys never leave the backend except via
the explicit /ai/reveal endpoint (the Settings "show key" toggle).

Naming: request fields use the same camelCase the renderer already used across
the IPC boundary (baseUrl, maxTokens), so the client shape is unchanged.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from document_analyser.ai import client, store

router = APIRouter(prefix="/ai", tags=["ai-providers"])


class SaveProviderBody(BaseModel):
    baseUrl: str
    model: str | None = None
    # Absent -> leave key untouched; "" -> clear; non-empty -> replace.
    key: str | None = None


class SetActiveBody(BaseModel):
    id: str | None = None


class DraftBody(BaseModel):
    baseUrl: str | None = None
    key: str | None = None


class ChatBody(BaseModel):
    system: str
    user: str
    maxTokens: int = 1024


@router.get("/providers")
async def get_providers() -> dict:
    return store.get_providers()


@router.post("/providers/{provider_id}")
async def save_provider(provider_id: str, body: SaveProviderBody) -> dict:
    try:
        return store.save_provider(provider_id, body.baseUrl, body.model, body.key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/active")
async def set_active(body: SetActiveBody) -> dict:
    return store.set_active_provider(body.id)


@router.post("/reveal/{provider_id}")
async def reveal_key(provider_id: str) -> dict:
    return {"key": store.reveal_key(provider_id)}


async def _test(provider_id: str, draft: DraftBody | None) -> dict:
    """Connection test == list models: one request that proves reachability AND
    key validity. Returns {ok, models?|error} rather than raising, matching
    testConnection() in ai-providers.ts."""
    try:
        resolved = store.resolve(
            provider_id,
            draft_base_url=draft.baseUrl if draft else None,
            draft_key=draft.key if draft else None,
        )
        models = await client.list_models(resolved)
        return {"ok": True, "models": models}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 — surface any provider/network error to the UI
        return {"ok": False, "error": str(e)}


@router.post("/test/{provider_id}")
async def test_connection(provider_id: str, draft: DraftBody | None = None) -> dict:
    return await _test(provider_id, draft)


@router.post("/models/{provider_id}")
async def list_models(provider_id: str, draft: DraftBody | None = None) -> dict:
    return await _test(provider_id, draft)


@router.post("/chat")
async def chat(body: ChatBody) -> dict:
    """One-shot chat via the active provider + its selected model."""
    selection = store.active_selection()
    if selection is None:
        return {"ok": False, "error": "No active AI provider with a model selected. Configure one in Settings → AI provider."}
    provider_id, model = selection
    try:
        resolved = store.resolve(provider_id)
        text = await client.chat(resolved, model, body.system, body.user, body.maxTokens)
        if not text:
            return {"ok": False, "error": "The model returned an empty response."}
        return {"ok": True, "text": text, "provider": store.preset_for(provider_id).label, "model": model}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
