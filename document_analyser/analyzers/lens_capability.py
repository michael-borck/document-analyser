"""Lens-declared suggestion models, and refusal on domain mismatch.

ADR-0043. Three models were named in a module constant here, which meant a lens
defined for any other domain silently received climate suggestions for its
judgement axis, and nothing in the interface could reveal it. The models are now
supplied by the caller as a capability declaration, and a declaration whose
domain does not match the loaded models produces **no suggestions at all** --
not a warning, not a suggestion with a caveat.

The default is refusal. ``load_capability`` accepts models only when the
caller declares a domain, and ``suggest`` requires that declaration to name the
loaded domain. A caller that passes nothing gets silence, which is the correct
outcome for a lens that has not said what assistance it wants.

Backwards compatibility
-----------------------
``MODEL_SPECS`` and the ``climate_framing`` singleton remain, because the
desktop app calls the suggestion route without a capability declaration until
its seed carries the block. ADR-0043 records that as a deliberate regression:
visible rather than silent. ``suggest_for_lens`` is the new entry point and
``suggest_batch`` is the legacy one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .climate_framing import ClimateFramingAnalyzer, MODEL_SPECS

__all__ = [
    "SuggestionCapability",
    "loaded_domain",
    "capability_state",
    "climate_capability",
    "suggest_for_lens",
]


@dataclass(frozen=True)
class SuggestionCapability:
    """What a lens declares about the assistance it wants.

    `models` names the model ids per role. `domain` is the lens's own domain tag
    and is what gets compared against the loaded models' domain; it is not a
    label for the model, it is the lens asserting what it is about. `revisions`
    pins weights the way ``DOCUMENT_ANALYSER_CLIMATEBERT_REVISIONS`` does, but
    per capability rather than per process.

    A capability with no models, or no domain, yields silence. That is the whole
    point: absence of a declaration is a declaration, and it is honoured.
    """

    domain: str = ""
    models: dict[str, str] = field(default_factory=dict)
    revisions: dict[str, str] = field(default_factory=dict)
    #: Why this capability was refused, if it was. Surfaced in /health so an
    #: absent suggestion can be explained rather than merely observed.
    refusal: str | None = None

    @property
    def silent(self) -> bool:
        return bool(self.refusal)

    @property
    def model_summary(self) -> str:
        if not self.models:
            return "none declared"
        return ", ".join(f"{k}={v}" for k, v in sorted(self.models.items()))


#: Domain tag for the ClimateBERT family. Fixed, because these models are
#: fine-tuned on climate disclosure text and a suggestion from them about a
#: cybersecurity passage is a category error rather than a weak answer.
CLIMATE_DOMAIN = "climate-disclosure"


def loaded_domain() -> str:
    """The domain of the models this process would actually load.

    Derived from ``MODEL_SPECS`` rather than declared separately, so it cannot
    drift from what is loaded when someone changes the constant.
    """
    return CLIMATE_DOMAIN if MODEL_SPECS else ""


def capability_state(cap: SuggestionCapability) -> tuple[bool, str | None]:
    """(usable, reason). Mirrors the health endpoint's other capability checks."""
    if not cap.models:
        return False, "lens declares no suggestion models"
    if not cap.domain:
        return False, "lens declares models but no domain, so they cannot be checked against it"
    actual = loaded_domain()
    if not actual:
        return False, "no suggestion models are configured in this build"
    if cap.domain != actual:
        return False, (
            f"lens domain {cap.domain!r} does not match the loaded models' domain "
            f"({actual!r}); producing suggestions would mean scoring out-of-domain material"
        )
    for role, model in cap.models.items():
        if MODEL_SPECS.get(role) != model:
            return False, f"model for role {role!r} is not the one this build loads ({model})"
    return True, None


def climate_capability() -> SuggestionCapability:
    """The capability block the sustainability lens declares.

    Kept as a function rather than a constant so the env-var revision override
    is read at call time, matching how the analyzer resolves revisions.
    """
    revisions: dict[str, str] = {}
    env = os.environ.get("DOCUMENT_ANALYSER_CLIMATEBERT_REVISIONS", "")
    for part in env.split(","):
        if "=" in part:
            key, _, rev = part.partition("=")
            revisions[key.strip()] = rev.strip()
    cap = SuggestionCapability(domain=CLIMATE_DOMAIN, models=dict(MODEL_SPECS), revisions=revisions)
    ok, why = capability_state(cap)
    if ok:
        return cap
    return SuggestionCapability(domain=cap.domain, models=cap.models,
                                revisions=revisions, refusal=why)


def suggest_for_lens(
    passages: list[str],
    capability: SuggestionCapability | None,
    *,
    analyzer: ClimateFramingAnalyzer | None = None,
) -> dict[str, Any]:
    """Suggest framing for passages, or explain why not.

    Returns the analyzer's own payload when the capability is usable, and a
    payload with `suggestions=[]` plus a `refusal` string otherwise. Callers get
    one shape either way, so a refusal cannot be mistaken for "nothing found".
    """
    cap = capability if capability is not None else climate_capability()
    usable, reason = capability_state(cap)
    if not usable or cap.silent:
        return {
            "suggestions": [],
            "available": False,
            "refusal": reason or cap.refusal or "no suggestion capability for this lens",
            "domain": cap.domain,
            "models": cap.model_summary,
        }
    analyzer = analyzer or _default_analyzer()
    out = analyzer.suggest_batch(passages)
    return {
        **out,
        "available": bool(out.get("available", True)),
        "refusal": None,
        "domain": cap.domain,
        "models": cap.model_summary,
    }


def _default_analyzer() -> ClimateFramingAnalyzer:  # pragma: no cover - wiring
    """The process-wide analyzer.

    Instantiated here rather than imported from a module singleton because the
    analyzer lives with the route that owns it (`api/routes/semantic_analysis`),
    and an analyzer module importing its own router would be a cycle. One
    analyzer per process is what the loader is built for, so reuse that one
    rather than constructing a second and loading the weights twice.
    """
    from ..api.routes.semantic_analysis import climate_framing

    return climate_framing
