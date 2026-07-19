"""LLM HTTP client — Python port of the network half of ai-providers.ts.

Three API shapes: anthropic (Messages), openai (Chat Completions — also Grok,
OpenAI-compatible, and both Ollama modes), and gemini (Google GenAI). All calls
run here in the backend so raw keys never reach the renderer and browser CORS
does not apply.
"""

from __future__ import annotations

from urllib.parse import quote

import httpx

from document_analyser.ai.store import ResolvedProvider

_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


async def list_models(p: ResolvedProvider) -> list[str]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as http:
        if p.shape == "anthropic":
            res = await http.get(
                f"{p.base_url}/v1/models",
                headers={"x-api-key": p.key or "", "anthropic-version": "2023-06-01"},
            )
            _raise_for_status(res)
            data = res.json().get("data") or []
            return [m["id"] for m in data]

        if p.shape == "gemini":
            res = await http.get(f"{p.base_url}/models?key={quote(p.key or '')}")
            _raise_for_status(res)
            models = res.json().get("models") or []
            return [m["name"].removeprefix("models/") for m in models]

        # openai shape (OpenAI, Grok, compat, Ollama)
        headers = {"Authorization": f"Bearer {p.key}"} if p.key else {}
        res = await http.get(f"{p.base_url}/models", headers=headers)
        _raise_for_status(res)
        data = res.json().get("data") or []
        return [m["id"] for m in data]


async def chat(p: ResolvedProvider, model: str, system: str, user: str, max_tokens: int) -> str:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as http:
        if p.shape == "anthropic":
            res = await http.post(
                f"{p.base_url}/v1/messages",
                headers={
                    "x-api-key": p.key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            _raise_for_status(res)
            blocks = res.json().get("content") or []
            return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()

        if p.shape == "gemini":
            res = await http.post(
                f"{p.base_url}/models/{quote(model)}:generateContent?key={quote(p.key or '')}",
                headers={"content-type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                },
            )
            _raise_for_status(res)
            candidates = res.json().get("candidates") or []
            if not candidates:
                return ""
            parts = (candidates[0].get("content") or {}).get("parts") or []
            return "".join(part.get("text", "") for part in parts).strip()

        # openai shape
        headers = {"content-type": "application/json"}
        if p.key:
            headers["Authorization"] = f"Bearer {p.key}"
        res = await http.post(
            f"{p.base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        _raise_for_status(res)
        choices = res.json().get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content", "").strip()


def _raise_for_status(res: httpx.Response) -> None:
    """Raise a message that surfaces the provider's error body (truncated),
    matching the errBody() detail in ai-providers.ts."""
    if res.is_success:
        return
    body = res.text[:300]
    raise RuntimeError(f"HTTP {res.status_code}: {body}")
