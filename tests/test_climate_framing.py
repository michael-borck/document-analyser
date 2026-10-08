#!/usr/bin/env python3
"""ClimateBERT framing analyzer tests (ADR-0039, ML rung).

Unit tests run against FAKE pipelines (label-mapping + gating logic, no
downloads). The real-model test is marked slow — opt in with
``pytest -m slow`` (downloads ~300 MB of pinned weights on first run).
"""

import pytest

from document_analyser.analyzers.climate_framing import ClimateFramingAnalyzer


class FakePipeline:
    """Stands in for a transformers text-classification pipeline."""

    def __init__(self, label: str, score: float, id2label: dict[int, str]) -> None:
        self.label = label
        self.score = score
        self.model = type(
            "M", (), {"config": type("C", (), {"id2label": id2label})()}
        )()

    def __call__(self, text: str, **kwargs) -> list[dict]:
        _ = text
        return [{"label": self.label, "score": self.score}]


CLIMATE_LABELS = {0: "Not climate related", 1: "Climate related"}
TARGET_LABELS = {0: "No Target", 1: "Net Zero Target", 2: "Reduction Target"}
COMMITMENT_LABELS = {0: "No Specific Commitment", 1: "Specific Commitment"}


def make_analyzer(
    detector_label: str = "LABEL_1",
    detector_score: float = 0.95,
    target_label: str = "LABEL_1",
    target_score: float = 0.88,
    commitment_label: str = "LABEL_1",
) -> ClimateFramingAnalyzer:
    analyzer = ClimateFramingAnalyzer()
    analyzer._pipelines = {  # noqa: SLF001 — test seam
        "detector": FakePipeline(detector_label, detector_score, CLIMATE_LABELS),
        "target": FakePipeline(target_label, target_score, TARGET_LABELS),
        "commitment": FakePipeline(commitment_label, 0.9, COMMITMENT_LABELS),
    }
    return analyzer


def test_climate_passage_with_netzero_target_suggests_aspirational():
    analyzer = make_analyzer()
    out = analyzer.suggest_batch(["We are committed to achieving net zero across our operations."])
    assert out["available"] is True
    result = out["results"][0]
    assert result["climate"] is True
    assert result["target"] == "net-zero"
    assert result["framing_value"] == "1"
    assert "climatebert/netzero-reduction@" in result["model_revision"]


def test_reduction_target_also_suggests_aspirational():
    analyzer = make_analyzer(target_label="LABEL_2")
    out = analyzer.suggest_batch(["Emissions will fall as the program delivers."])
    assert out["results"][0]["framing_value"] == "1"
    assert out["results"][0]["target"] == "reduction"


def test_no_target_stays_unsuggested():
    analyzer = make_analyzer(target_label="LABEL_0")
    out = analyzer.suggest_batch(["The weather was mentioned in the risk section."])
    result = out["results"][0]
    assert result["framing_value"] is None


def test_low_confidence_target_does_not_suggest():
    analyzer = make_analyzer(target_score=0.3)
    out = analyzer.suggest_batch(["We are committed to net zero."])
    assert out["results"][0]["framing_value"] is None


def test_non_climate_passage_is_gated_before_the_target_model():
    analyzer = make_analyzer(detector_label="LABEL_0", detector_score=0.99)
    out = analyzer.suggest_batch(["Quarterly revenue rose modestly."])
    result = out["results"][0]
    assert result["climate"] is False
    assert result["framing_value"] is None


def test_unavailable_models_degrade_gracefully():
    analyzer = ClimateFramingAnalyzer()  # no pipelines loaded, transformers may exist
    analyzer._load = lambda: None  # type: ignore[method-assign] — force unavailable
    out = analyzer.suggest_batch(["anything"])
    assert out["available"] is False
    assert out["results"] == [None]


@pytest.mark.slow
def test_real_models_load_and_suggest():
    """Opt-in: downloads the pinned ClimateBERT weights on first run."""
    analyzer = ClimateFramingAnalyzer()
    out = analyzer.suggest_batch(
        [
            "We are committed to net zero by 2035, cutting scope 1 and 2 emissions.",
            "The board approved the dividend.",
        ]
    )
    assert out["available"] is True
    climate_hit, non_climate = out["results"]
    assert climate_hit is not None and climate_hit["climate"] is True
    assert non_climate is not None and non_climate["climate"] is False
