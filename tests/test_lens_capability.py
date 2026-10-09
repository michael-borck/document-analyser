"""Tests for the lens-declared suggestion capability (ADR-0043).

These pin the refusal behaviour rather than the model's accuracy. The claim
under test is that a lens gets no suggestions from another domain's models, and
that the absence is explicable -- both of which are easy to break silently and
impossible to notice without a test.
"""

from __future__ import annotations


from document_analyser.analyzers.climate_framing import MODEL_SPECS
from document_analyser.analyzers.lens_capability import (
    CLIMATE_DOMAIN,
    SuggestionCapability,
    capability_state,
    climate_capability,
    suggest_for_lens,
)


class TestCapabilityDeclaration:
    def test_the_sustainability_lens_is_usable(self):
        cap = climate_capability()
        usable, reason = capability_state(cap)
        assert usable, reason
        assert cap.domain == CLIMATE_DOMAIN

    def test_a_capability_with_no_models_is_silent(self):
        usable, reason = capability_state(SuggestionCapability(domain="cybersecurity"))
        assert not usable
        assert "no suggestion models" in reason

    def test_a_capability_with_no_domain_is_refused(self):
        """Models without a domain cannot be checked against one."""
        usable, reason = capability_state(SuggestionCapability(domain="", models=dict(MODEL_SPECS)))
        assert not usable
        assert "no domain" in reason


class TestDomainMismatch:
    """The failure ADR-0043 exists to prevent."""

    def test_a_foreign_domain_is_refused(self):
        cap = SuggestionCapability(domain="cybersecurity", models=dict(MODEL_SPECS))
        usable, reason = capability_state(cap)
        assert not usable
        assert "does not match" in reason

    def test_the_refusal_explains_itself(self):
        cap = SuggestionCapability(domain="cybersecurity", models=dict(MODEL_SPECS))
        _, reason = capability_state(cap)
        assert "cybersecurity" in reason and "climate-disclosure" in reason

    def test_a_model_this_build_does_not_load_is_refused(self):
        cap = SuggestionCapability(
            domain=CLIMATE_DOMAIN,
            models={**MODEL_SPECS, "target": "some-other-org/netzero-v2"},
        )
        usable, reason = capability_state(cap)
        assert not usable
        assert "not the one this build loads" in reason


class TestSuggestForLens:
    def test_a_foreign_lens_gets_no_suggestions_and_a_reason(self, monkeypatch):
        calls: list = []
        monkeypatch.setattr(
            "document_analyser.analyzers.lens_capability._default_analyzer",
            lambda: (_ for _ in ()).throw(AssertionError("analyzer must not be reached")),
        )
        out = suggest_for_lens(
            ["a passage"], SuggestionCapability(domain="cybersecurity", models=dict(MODEL_SPECS))
        )
        assert out["suggestions"] == []
        assert out["available"] is False
        assert out["refusal"] and "does not match" in out["refusal"]
        assert not calls

    def test_no_capability_at_all_is_silence_not_a_crash(self, monkeypatch):
        """A caller that declares nothing must not reach the model loader.

        Also keeps this test from downloading the ClimateBERT weights: the point
        is the refusal path, not the model's opinion of a passage.
        """
        monkeypatch.setattr(
            "document_analyser.analyzers.lens_capability._default_analyzer",
            lambda: (_ for _ in ()).throw(AssertionError("analyzer must not be reached")),
        )
        out = suggest_for_lens(["a passage"], SuggestionCapability())
        assert out["suggestions"] == []
        assert out["available"] is False
        assert out["refusal"]

    def test_the_payload_always_has_the_same_shape(self):
        """A refusal must be distinguishable from 'nothing found'."""
        out = suggest_for_lens(["a passage"], SuggestionCapability(domain="cybersecurity", models=dict(MODEL_SPECS)))
        for key in ("suggestions", "available", "refusal", "domain", "models"):
            assert key in out
