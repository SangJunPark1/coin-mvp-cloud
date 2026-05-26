from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelScore:
    probability: float
    confidence_adjustment: float
    notes: list[str]
    features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_entry_with_feature_model(decision_input: dict[str, Any]) -> ModelScore:
    """Small deterministic model layer with the same interface a neural model can replace.

    This is intentionally transparent until enough labeled trade outcomes exist.
    Once the journal has enough examples, this function can load learned weights
    or be replaced by an ONNX/Torch inference wrapper without changing callers.
    """

    features = extract_features(decision_input)
    logit = -0.35
    logit += features["signal_confidence"] * 1.8
    logit += min(features["expected_upside_pct"], 3.0) * 0.22
    logit -= min(features["expected_downside_pct"], 3.0) * 0.32
    logit += features["recent_volatility_pct"] * 0.10
    logit += features["btc_momentum_pct"] * 0.35
    logit += features["news_sentiment_score"] * 0.45
    logit += features["chart_quality_score"] * 0.75
    logit -= min(features["news_risk_headline_count"], 5.0) * 0.08
    logit += setup_bias(str(decision_input.get("candidate", {}).get("signal_reason", "")))
    probability = 1.0 / (1.0 + math.exp(-logit))
    notes = []
    if features["news_risk_headline_count"] >= 3 and features["news_sentiment_score"] < 0:
        notes.append("News model detected elevated headline risk.")
    if features["expected_downside_pct"] >= features["expected_upside_pct"]:
        notes.append("Feature model sees poor upside/downside balance.")
    if features["chart_quality_score"] >= 0.65:
        notes.append("Chart feature model detected a high-quality technical setup.")
    confidence_adjustment = (probability - 0.5) * 0.18
    return ModelScore(
        probability=probability,
        confidence_adjustment=confidence_adjustment,
        notes=notes,
        features=features,
    )


def extract_features(decision_input: dict[str, Any]) -> dict[str, float]:
    candidate = decision_input.get("candidate", {})
    context = decision_input.get("market_context", {})
    return {
        "signal_confidence": as_float(candidate.get("signal_confidence")),
        "expected_upside_pct": as_float(candidate.get("expected_upside_pct")),
        "expected_downside_pct": as_float(candidate.get("expected_downside_pct")),
        "recent_volatility_pct": as_float(candidate.get("recent_volatility_pct")),
        "btc_momentum_pct": as_float(context.get("btc_momentum_pct")),
        "news_sentiment_score": as_float(context.get("news_sentiment_score")),
        "news_risk_headline_count": as_float(context.get("news_risk_headline_count")),
        "chart_quality_score": as_float(candidate.get("chart_quality_score")),
    }


def setup_bias(reason: str) -> float:
    normalized = reason.lower()
    if "trend breakout setup" in normalized:
        return 0.10
    if "pullback continuation setup" in normalized:
        return 0.14
    if "range rebound setup" in normalized:
        return -0.02
    if "bollinger rebound setup" in normalized:
        return -0.06
    if "chart ai setup" in normalized:
        return 0.16
    return 0.0


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
