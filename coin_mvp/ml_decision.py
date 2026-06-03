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
    logit += min(features["reward_risk_ratio"], 4.0) * 0.20
    logit += features["recent_volatility_pct"] * 0.10
    logit += features["btc_momentum_pct"] * 0.35
    logit += features["news_sentiment_score"] * 0.45
    logit += features["community_sentiment_score"] * 0.35
    logit += features["chart_quality_score"] * 0.35
    logit += features["impulse_quality_score"] * 0.95
    logit += features["opportunity_edge_score"] * 0.70
    logit -= min(features["news_risk_headline_count"], 5.0) * 0.08
    logit -= min(features["community_risk_count"], 6.0) * 0.045
    reason = str(decision_input.get("candidate", {}).get("signal_reason", ""))
    logit += setup_bias(reason)
    logit += setup_quality_adjustment(reason, features)
    if str(decision_input.get("market_context", {}).get("market_mode", "neutral")).lower() == "neutral":
        logit -= 0.18
    probability = 1.0 / (1.0 + math.exp(-logit))
    notes = []
    if features["news_risk_headline_count"] >= 3 and features["news_sentiment_score"] < 0:
        notes.append("News model detected elevated headline risk.")
    if features["community_risk_count"] >= 5 and features["community_sentiment_score"] < 0:
        notes.append("Community model detected panic-heavy retail sentiment.")
    if features["community_sentiment_score"] > 0.2:
        notes.append("Community attention model detected positive retail momentum.")
    if features["expected_downside_pct"] >= features["expected_upside_pct"]:
        notes.append("Feature model sees poor upside/downside balance.")
    if features["impulse_quality_score"] < 0.58:
        notes.append("Impulse model sees weak follow-through quality.")
    if features["rsi"] > 64.0:
        notes.append("Momentum is near an overheated RSI zone.")
    if features["reward_risk_ratio"] < 2.0:
        notes.append("Reward/risk ratio is below the upgraded model target.")
    if features["opportunity_edge_score"] >= 0.72:
        notes.append("Opportunity model detected a high-conviction profit window.")
    if features["chart_quality_score"] >= 0.65 and features["impulse_quality_score"] >= 0.55:
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
    chart = candidate.get("chart_features", {})
    expected_upside = as_float(candidate.get("expected_upside_pct"))
    expected_downside = as_float(candidate.get("expected_downside_pct"))
    momentum_3 = as_float(chart.get("momentum_3_pct"))
    momentum_8 = as_float(chart.get("momentum_8_pct"))
    volume_ratio = as_float(chart.get("volume_ratio"))
    close_position = as_float(chart.get("close_position"))
    rsi = as_float(chart.get("rsi"))
    impulse_quality = impulse_quality_score(momentum_3, momentum_8, volume_ratio, close_position, rsi)
    reward_risk = expected_upside / expected_downside if expected_downside > 0 else 0.0
    opportunity_edge = opportunity_edge_score(
        expected_upside,
        reward_risk,
        impulse_quality,
        momentum_3,
        momentum_8,
        volume_ratio,
        close_position,
        rsi,
    )
    return {
        "signal_confidence": as_float(candidate.get("signal_confidence")),
        "expected_upside_pct": expected_upside,
        "expected_downside_pct": expected_downside,
        "recent_volatility_pct": as_float(candidate.get("recent_volatility_pct")),
        "btc_momentum_pct": as_float(context.get("btc_momentum_pct")),
        "news_sentiment_score": as_float(context.get("news_sentiment_score")),
        "news_risk_headline_count": as_float(context.get("news_risk_headline_count")),
        "community_sentiment_score": as_float(context.get("community_sentiment_score")),
        "community_risk_count": as_float(context.get("community_risk_count")),
        "chart_quality_score": as_float(candidate.get("chart_quality_score")),
        "reward_risk_ratio": reward_risk,
        "momentum_3_pct": momentum_3,
        "momentum_8_pct": momentum_8,
        "volume_ratio": volume_ratio,
        "close_position": close_position,
        "rsi": rsi,
        "impulse_quality_score": impulse_quality,
        "opportunity_edge_score": opportunity_edge,
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


def impulse_quality_score(momentum_3: float, momentum_8: float, volume_ratio: float, close_position: float, rsi: float) -> float:
    score = 0.0
    score += min(max(momentum_3, 0.0) / 0.8, 1.0) * 0.24
    score += min(max(momentum_8, 0.0) / 1.2, 1.0) * 0.24
    score += min(max(volume_ratio - 1.0, 0.0) / 2.5, 1.0) * 0.22
    score += min(max(close_position - 0.55, 0.0) / 0.45, 1.0) * 0.16
    if 42.0 <= rsi <= 62.0:
        score += 0.14
    elif 62.0 < rsi <= 66.0:
        score += 0.04
    return min(score, 1.0)


def opportunity_edge_score(
    expected_upside_pct: float,
    reward_risk_ratio: float,
    impulse_quality: float,
    momentum_3_pct: float,
    momentum_8_pct: float,
    volume_ratio: float,
    close_position: float,
    rsi: float,
) -> float:
    """Score whether a setup deserves capital, not just permission.

    The prior model mostly rejected weak entries. This score explicitly looks
    for the combination that can pay for fees and losers: enough upside,
    enough reward/risk, fresh impulse, volume confirmation, and a strong close.
    """

    score = 0.0
    score += min(max(expected_upside_pct - 1.6, 0.0) / 1.8, 1.0) * 0.20
    score += min(max(reward_risk_ratio - 1.8, 0.0) / 2.0, 1.0) * 0.22
    score += impulse_quality * 0.28
    score += min(max(momentum_3_pct, 0.0) / 0.9, 1.0) * 0.10
    score += min(max(momentum_8_pct, 0.0) / 1.4, 1.0) * 0.08
    score += min(max(volume_ratio - 1.2, 0.0) / 2.8, 1.0) * 0.07
    score += min(max(close_position - 0.65, 0.0) / 0.35, 1.0) * 0.05
    if rsi > 66.0:
        score -= 0.12
    if rsi < 38.0 and momentum_3_pct < 0.35:
        score -= 0.08
    return min(max(score, 0.0), 1.0)


def setup_quality_adjustment(reason: str, features: dict[str, float]) -> float:
    normalized = reason.lower()
    adjustment = 0.0
    if "micro recovery setup" in normalized:
        if features["momentum_8_pct"] < 0.85:
            adjustment -= 0.22
        if features["expected_upside_pct"] < 2.2:
            adjustment -= 0.18
    if "trend breakout setup" in normalized:
        if features["momentum_3_pct"] < 0.45:
            adjustment -= 0.22
        if features["rsi"] > 63.5:
            adjustment -= 0.14
        if features["opportunity_edge_score"] >= 0.72:
            adjustment += 0.16
    if "chart ai setup" in normalized:
        if features["impulse_quality_score"] < 0.60:
            adjustment -= 0.24
        if features["rsi"] > 64.0:
            adjustment -= 0.18
        if features["opportunity_edge_score"] >= 0.76:
            adjustment += 0.14
    if "range rebound setup" in normalized:
        adjustment -= 0.16
        if features["expected_upside_pct"] < 2.0:
            adjustment -= 0.22
    return adjustment


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
