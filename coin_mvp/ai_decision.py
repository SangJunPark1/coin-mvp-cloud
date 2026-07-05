from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from .config import AiDecisionConfig, StrategyConfig
from .market_context import DecisionContext
from .ml_decision import score_entry_with_feature_model
from .models import Candle, Signal
from .strategy import chart_feature_snapshot, estimate_expected_downside_pct, estimate_signal_expected_upside_pct, recent_volatility_pct


@dataclass(frozen=True)
class DecisionReview:
    action: str
    grade: str
    confidence: float
    thesis: str
    invalidation: str
    expected_upside_pct: float
    expected_downside_pct: float
    risk_notes: list[str]
    source: str = "local"
    input_snapshot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def review_entry_candidate(
    signal: Signal,
    candles: list[Candle],
    context: DecisionContext,
    strategy: StrategyConfig,
    ai_config: AiDecisionConfig,
) -> DecisionReview:
    """Review a deterministic entry signal before the risk layer can approve it.

    The current implementation is a local structured reviewer. It creates the
    same JSON-shaped output that an external AI provider should later return.
    """

    decision_input = build_decision_input(signal, candles, context, strategy)
    fallback_source = "local" if ai_config.provider.lower() == "local" else f"local-fallback:{ai_config.provider}"
    fallback_reason: str | None = None
    if ai_config.enabled and ai_config.provider.lower() == "openai":
        external, fallback_reason = try_openai_review(decision_input, ai_config)
        if external is not None:
            return external

    expected_upside = float(decision_input["candidate"]["expected_upside_pct"])
    expected_downside = estimate_expected_downside_pct(candles, strategy.stop_loss_pct, strategy.stop_volatility_multiplier)
    model_score = score_entry_with_feature_model(decision_input)
    risk_notes: list[str] = []

    if not ai_config.enabled:
        return DecisionReview(
            action="buy",
            grade="A",
            confidence=signal.confidence,
            thesis="AI decision layer disabled; deterministic strategy signal accepted.",
            invalidation=f"Exit if stop loss reaches {strategy.stop_loss_pct:.2f}% or trend breaks.",
            expected_upside_pct=expected_upside,
            expected_downside_pct=expected_downside,
            risk_notes=[],
            source="disabled",
            input_snapshot=decision_input,
        )

    confidence = min(0.95, max(0.0, signal.confidence + model_score.confidence_adjustment) * context.score_multiplier)
    reason_text = str(decision_input.get("candidate", {}).get("signal_reason", "")).lower()
    is_daily_participation = "daily participation setup" in reason_text
    if expected_upside < strategy.min_expected_upside_pct:
        risk_notes.append("Expected upside is below the configured minimum.")
    if expected_downside >= expected_upside:
        risk_notes.append("Expected downside is greater than or equal to expected upside.")
    if context.fear_greed_value is not None and context.fear_greed_value >= 85:
        risk_notes.append("Sentiment is overheated.")
    if context.fear_greed_value is not None and context.fear_greed_value <= 15:
        risk_notes.append("Market stress is high.")
    if not context.allows_entries:
        risk_notes.append("Market context does not allow new entries.")
    risk_notes.extend(model_score.notes)
    if fallback_reason:
        risk_notes.append(f"OpenAI fallback reason: {fallback_reason}")

    grade = decision_grade(confidence, expected_upside, expected_downside, context.allows_entries)
    hard_block = upgraded_model_hard_block(decision_input, model_score.features)

    if hard_block:
        action = "hold"
        risk_notes.append(hard_block)
        thesis = "Upgraded AI model rejected the candidate despite the deterministic setup."
    elif not context.allows_entries and not is_daily_participation:
        action = "pause"
        thesis = "Market context is unfavorable, so the candidate should not be opened."
    elif is_daily_participation:
        action = "buy"
        thesis = "Daily participation engine chose the best tradable market, so the AI layer allows the entry instead of sitting in cash."
    elif expected_downside >= expected_upside:
        action = "hold"
        thesis = "Candidate risk/reward is unfavorable because downside is at least as large as upside."
    elif risk_notes and confidence < ai_config.min_confidence + 0.15:
        action = "hold"
        thesis = "Candidate has a deterministic signal, but the risk/reward evidence is not strong enough."
    elif model_score.probability < required_model_probability(decision_input, context):
        action = "hold"
        thesis = "Upgraded feature model score is too weak for a new entry."
    elif confidence < ai_config.min_confidence or grade == "C":
        action = "hold"
        thesis = "Candidate grade is below the entry threshold."
    else:
        action = "buy"
        thesis = "Candidate has strategy, market, news, and feature-model evidence above the entry threshold."

    return DecisionReview(
        action=action,
        grade=grade,
        confidence=confidence,
        thesis=thesis,
        invalidation=f"Exit if stop loss reaches {strategy.stop_loss_pct:.2f}%, time stop triggers, or BTC context deteriorates.",
        expected_upside_pct=expected_upside,
        expected_downside_pct=expected_downside,
        risk_notes=risk_notes,
        source=fallback_source,
        input_snapshot={**decision_input, "local_model": model_score.to_dict()},
    )


def build_decision_input(
    signal: Signal,
    candles: list[Candle],
    context: DecisionContext,
    strategy: StrategyConfig,
) -> dict[str, Any]:
    closes = [candle.close for candle in candles]
    latest = closes[-1] if closes else 0.0
    recent_high = max((candle.high for candle in candles[-30:]), default=latest)
    recent_low = min((candle.low for candle in candles[-30:]), default=latest)
    chart_features = chart_feature_snapshot(candles, strategy.rsi_period)
    return {
        "schema_version": "decision-input-v1",
        "candidate": {
            "market": candles[-1].market if candles else "",
            "latest_price": latest,
            "signal_side": signal.side.value if hasattr(signal.side, "value") else str(signal.side),
            "signal_reason": signal.reason,
            "signal_confidence": signal.confidence,
            "expected_upside_pct": estimate_signal_expected_upside_pct(candles, signal, strategy),
            "expected_downside_pct": estimate_expected_downside_pct(candles, strategy.stop_loss_pct, strategy.stop_volatility_multiplier),
            "recent_high": recent_high,
            "recent_low": recent_low,
            "recent_volatility_pct": recent_volatility_pct(candles),
            "chart_quality_score": chart_quality_score(chart_features),
            "chart_features": chart_features,
        },
        "market_context": context.to_dict(),
        "strategy_limits": {
            "target_upside_pct": strategy.target_upside_pct,
            "min_expected_upside_pct": strategy.min_expected_upside_pct,
            "take_profit_pct": strategy.take_profit_pct,
            "stop_loss_pct": strategy.stop_loss_pct,
            "max_entry_rsi": strategy.max_entry_rsi,
        },
    }


def chart_quality_score(features: dict[str, float]) -> float:
    if not features:
        return 0.0
    score = 0.0
    score += min(max(features["momentum_3_pct"], 0.0) / 2.5, 0.22)
    score += min(max(features["volume_ratio"] - 0.8, 0.0) / 2.8, 0.20)
    score += min(max(features["close_position"] - 0.45, 0.0) / 1.2, 0.18)
    if 38.0 <= features["rsi"] <= 68.0:
        score += 0.14
    if features["ema9_gap_pct"] >= -0.2 and features["ema21_gap_pct"] >= -0.45:
        score += 0.16
    if features["range_expansion_ratio"] >= 1.1:
        score += 0.10
    return min(score, 1.0)


def decision_grade(confidence: float, expected_upside: float, expected_downside: float, allows_entries: bool) -> str:
    if not allows_entries or expected_downside >= expected_upside:
        return "C"
    if confidence >= 0.75 and expected_upside >= 2.0:
        return "A"
    if confidence >= 0.60 and expected_upside >= 1.2:
        return "B"
    return "C"


def required_model_probability(decision_input: dict[str, Any], context: DecisionContext) -> float:
    reason = str(decision_input.get("candidate", {}).get("signal_reason", "")).lower()
    mode = str(getattr(context, "market_mode", "neutral")).lower()
    required = 0.66
    if mode == "neutral":
        required = 0.74
    elif mode == "risk_off":
        required = 0.82
    elif mode == "risk_on":
        required = 0.62
    if "composite engine setup" in reason:
        required = min(required, 0.68 if mode in {"risk_on", "neutral"} else 0.74)
    if "regime ensemble setup" in reason:
        required = min(required, 0.66 if mode in {"risk_on", "neutral"} else 0.74)
    if "range rebound setup" in reason:
        required += 0.06
    if "micro recovery setup" in reason:
        required += 0.04
    if "chart ai setup" in reason:
        required += 0.03
    if "daily participation setup" in reason:
        required = min(required, 0.50 if mode != "capital_protect" else 0.9)
    if "capitulation rebound" in reason:
        required = min(required, 0.70 if mode in {"panic_rebound", "risk_off"} else 0.76)
    return min(required, 0.9)


def upgraded_model_hard_block(decision_input: dict[str, Any], features: dict[str, float]) -> str:
    reason = str(decision_input.get("candidate", {}).get("signal_reason", "")).lower()
    mode = str(decision_input.get("market_context", {}).get("market_mode", "neutral")).lower()
    if "regime ensemble setup" in reason:
        if features.get("reward_risk_ratio", 0.0) < 1.65:
            return "AI hard block: regime ensemble reward/risk is too weak."
        if features.get("close_position", 0.0) < 0.54:
            return "AI hard block: regime ensemble close quality is weak."
        if features.get("volume_ratio", 0.0) < 1.0:
            return "AI hard block: regime ensemble volume is too thin."
        if features.get("rsi", 0.0) > 70.0:
            return "AI hard block: regime ensemble entry is overheated."
        return ""
    if "daily participation setup" in reason:
        if mode == "capital_protect":
            return "AI hard block: daily participation is blocked in capital-protect mode."
        strategy_limits = decision_input.get("strategy_limits", {})
        configured_min_upside = float(strategy_limits.get("min_expected_upside_pct", 0.75) or 0.75)
        if features.get("expected_upside_pct", 0.0) < max(0.75, configured_min_upside):
            return "AI hard block: daily participation expected upside is too small."
        if features.get("reward_risk_ratio", 0.0) < 1.55:
            return "AI hard block: daily participation reward/risk is too weak."
        if features.get("volume_ratio", 0.0) < 0.65:
            return "AI hard block: daily participation volume is too thin."
        if features.get("close_position", 0.0) < 0.60:
            return "AI hard block: daily participation close quality is weak."
        if features.get("momentum_8_pct", 0.0) < -0.55 and features.get("momentum_3_pct", 0.0) < 0.25:
            return "AI hard block: daily participation has weak bounce after negative momentum."
        if features.get("rsi", 0.0) >= 95.0:
            return "AI hard block: daily participation is extremely overheated."
        if features.get("momentum_8_pct", 0.0) < -4.0 and features.get("close_position", 0.0) < 0.18:
            return "AI hard block: daily participation is in active breakdown."
        return ""
    if "capitulation rebound" in reason:
        if mode not in {"panic_rebound", "risk_off", "neutral"}:
            return "AI hard block: capitulation rebound is for weak-market regimes only."
        if features.get("reward_risk_ratio", 0.0) < 2.2:
            return "AI hard block: capitulation rebound reward/risk is too weak."
        if features.get("volume_ratio", 0.0) < 2.4:
            return "AI hard block: capitulation rebound volume is too thin."
        if features.get("close_position", 0.0) < 0.72:
            return "AI hard block: capitulation rebound close position is weak."
        if features.get("momentum_3_pct", 0.0) < 0.18:
            return "AI hard block: capitulation rebound bounce is too weak."
        if not 24.0 <= features.get("rsi", 0.0) <= 50.0:
            return "AI hard block: capitulation rebound RSI is outside the panic-rebound zone."
        if features.get("community_sentiment_score", 0.0) < -0.45 and features.get("community_risk_count", 0.0) >= 6:
            return "AI hard block: community panic is still too negative for a rebound."
        return ""
    if "composite engine setup" in reason:
        if features.get("reward_risk_ratio", 0.0) < 1.75:
            return "AI hard block: composite engine reward/risk is too weak."
        if features.get("volume_ratio", 0.0) < 1.25:
            return "AI hard block: composite engine volume is too thin."
        if features.get("close_position", 0.0) < 0.56:
            return "AI hard block: composite engine close position is weak."
        if "qullamaggie ucl breakout" in reason:
            if features.get("momentum_3_pct", 0.0) < 0.12:
                return "AI hard block: UCL breakout impulse is too weak."
            if features.get("rsi", 0.0) > 69.0:
                return "AI hard block: UCL breakout RSI is too hot."
        if "lcl recovery rebound" in reason:
            if not 22.0 <= features.get("rsi", 0.0) <= 58.0:
                return "AI hard block: LCL rebound RSI is outside recovery range."
            if features.get("momentum_3_pct", 0.0) < 0.02:
                return "AI hard block: LCL rebound has not recovered yet."
        return ""
    if features.get("reward_risk_ratio", 0.0) < 2.15:
        return "AI hard block: reward/risk ratio is below 2.15."
    if features.get("impulse_quality_score", 0.0) < 0.58:
        return "AI hard block: impulse quality is too weak."
    if "micro recovery setup" in reason:
        if mode not in {"risk_on", "neutral"}:
            return "AI hard block: micro recovery is blocked outside risk-on/neutral mode."
        if features.get("close_position", 0.0) < 0.65:
            return "AI hard block: micro recovery close position is weak."
        if mode == "neutral" and features.get("reward_risk_ratio", 0.0) >= 2.8:
            if (
                features.get("close_position", 0.0) >= 0.92
                and features.get("momentum_3_pct", 0.0) >= 0.35
                and features.get("volume_ratio", 0.0) >= 1.45
                and features.get("expected_upside_pct", 0.0) >= 2.2
                and features.get("rsi", 0.0) <= 55.0
            ):
                return ""
        if features.get("momentum_8_pct", 0.0) < 0.55:
            return "AI hard block: micro recovery follow-through momentum is weak."
        if features.get("expected_upside_pct", 0.0) < 2.8:
            return "AI hard block: micro recovery expected upside is too small."
        if features.get("rsi", 0.0) > 60.0:
            return "AI hard block: micro recovery RSI is too hot."
    if "trend breakout setup" in reason:
        if mode != "risk_on" and features.get("expected_upside_pct", 0.0) < 2.9:
            return "AI hard block: neutral/risk-off trend breakout needs at least 2.9% expected upside."
        if mode != "risk_on" and features.get("volume_ratio", 0.0) < 1.55:
            return "AI hard block: neutral/risk-off trend breakout volume is not decisive enough."
        if mode != "risk_on" and features.get("volume_ratio", 0.0) > 3.5 and features.get("rsi", 0.0) > 64.0:
            return "AI hard block: neutral/risk-off trend breakout is too crowded."
        if features.get("volume_ratio", 0.0) > 5.0 and features.get("rsi", 0.0) > 61.5:
            return "AI hard block: trend breakout looks like late blow-off volume."
        if features.get("rsi", 0.0) > 64.0 and features.get("volume_ratio", 0.0) < 2.0:
            return "AI hard block: trend breakout RSI is hot without decisive volume."
        if features.get("momentum_3_pct", 0.0) < 0.45 and features.get("expected_upside_pct", 0.0) < 2.4:
            return "AI hard block: trend breakout short impulse is weak."
        if features.get("rsi", 0.0) > 66.0:
            return "AI hard block: trend breakout RSI is too hot."
        if features.get("rsi", 0.0) > 64.0 and features.get("expected_upside_pct", 0.0) < 2.8:
            return "AI hard block: trend breakout RSI is hot without enough upside."
    if "chart ai setup" in reason and features.get("close_position", 0.0) < 0.72:
        return "AI hard block: chart setup close position is weak."
    return ""


def try_openai_review(decision_input: dict[str, Any], ai_config: AiDecisionConfig) -> tuple[DecisionReview | None, str | None]:
    api_key = os.environ.get(ai_config.api_key_env)
    if not api_key:
        return None, f"missing environment variable {ai_config.api_key_env}"
    try:
        payload = {
            "model": ai_config.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are a conservative crypto paper-trading decision reviewer. "
                        "Return only the structured decision. Never ignore risk limits."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(decision_input, ensure_ascii=False),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "paper_trade_decision_review",
                    "strict": True,
                    "schema": decision_review_schema(),
                }
            },
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        parsed = extract_openai_json(response_payload)
        return decision_from_mapping(parsed, source="openai", input_snapshot=decision_input), None
    except Exception as exc:
        reason = summarize_openai_error(exc)
        print(f"OpenAI decision review failed: {reason}")
        return None, reason


def summarize_openai_error(exc: Exception) -> str:
    status = getattr(exc, "code", None)
    body = ""
    try:
        body_bytes = exc.read()  # type: ignore[attr-defined]
        body = body_bytes.decode("utf-8", errors="replace")[:500]
    except Exception:
        body = str(exc)[:500]
    if status:
        return f"HTTP {status}: {body}"
    return body


def decision_review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action",
            "grade",
            "confidence",
            "thesis",
            "invalidation",
            "expected_upside_pct",
            "expected_downside_pct",
            "risk_notes",
        ],
        "properties": {
            "action": {"type": "string", "enum": ["buy", "hold", "sell", "pause", "resume"]},
            "grade": {"type": "string", "enum": ["A", "B", "C"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "thesis": {"type": "string"},
            "invalidation": {"type": "string"},
            "expected_upside_pct": {"type": "number"},
            "expected_downside_pct": {"type": "number"},
            "risk_notes": {"type": "array", "items": {"type": "string"}},
        },
    }


def extract_openai_json(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("output_parsed"), dict):
        return payload["output_parsed"]
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("parsed"), dict):
                return content["parsed"]
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return json.loads(text)
    text = payload.get("output_text")
    if isinstance(text, str) and text.strip():
        return json.loads(text)
    raise ValueError("OpenAI response did not contain structured JSON.")


def decision_from_mapping(payload: dict[str, Any], source: str, input_snapshot: dict[str, Any]) -> DecisionReview:
    return DecisionReview(
        action=str(payload["action"]),
        grade=str(payload.get("grade", "B")),
        confidence=float(payload["confidence"]),
        thesis=str(payload["thesis"]),
        invalidation=str(payload["invalidation"]),
        expected_upside_pct=float(payload["expected_upside_pct"]),
        expected_downside_pct=float(payload["expected_downside_pct"]),
        risk_notes=[str(note) for note in payload.get("risk_notes", [])],
        source=source,
        input_snapshot=input_snapshot,
    )
