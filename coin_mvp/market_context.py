from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import StrategyConfig
from .data import UpbitPublicDataSource
from .models import Candle
from .news import fetch_community_signal, fetch_crypto_news_signal
from .strategy import recent_volatility_pct

KST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class DecisionContext:
    allows_entries: bool
    reason: str
    score_multiplier: float
    btc_momentum_pct: float
    btc_volatility_pct: float
    market_mode: str = "neutral"
    mode_reason: str = ""
    session_label: str = ""
    position_fraction_multiplier: float = 1.0
    fear_greed_value: int | None = None
    fear_greed_label: str = "unknown"
    onchain_tx_count: int | None = None
    global_market_cap_change_pct: float | None = None
    btc_dominance_pct: float | None = None
    binance_btcusdt_change_pct: float | None = None
    binance_btcusdt_quote_volume: float | None = None
    news_sentiment_score: float | None = None
    news_risk_headline_count: int = 0
    news_positive_headline_count: int = 0
    news_headlines: list[str] | None = None
    community_sentiment_score: float | None = None
    community_risk_count: int = 0
    community_positive_count: int = 0
    community_headlines: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_decision_context(data_source: UpbitPublicDataSource, config: StrategyConfig) -> DecisionContext:
    btc_candles = data_source.get_recent_candles("KRW-BTC", max(config.btc_long_window + 5, 30))
    btc_momentum = candle_momentum_pct(btc_candles, lookback=5)
    btc_volatility = recent_volatility_pct(btc_candles, lookback=20)
    fear_value, fear_label = fetch_fear_greed()
    tx_count = fetch_blockchain_tx_count()
    global_change, btc_dominance = fetch_coingecko_global()
    binance_change, binance_quote_volume = fetch_binance_btcusdt_24h()
    news_signal = fetch_news_signal()
    community_signal = fetch_local_community_signal()
    session_label, session_bias = current_session_label()

    allows = True
    reasons = [f"BTC momentum {btc_momentum:.2f}%", f"BTC volatility {btc_volatility:.2f}%"]
    mode_score = session_bias

    if btc_momentum < config.min_btc_momentum_pct:
        mode_score -= 1.0
        reasons.append("BTC trend weak")
    elif btc_momentum >= 0.25:
        mode_score += 1.5
    elif btc_momentum >= 0.0:
        mode_score += 0.6
    if fear_value is not None:
        reasons.append(f"fear-greed {fear_value} {fear_label}")
        if fear_value >= 85:
            mode_score -= 0.8
            reasons.append("market greed overheated")
        elif fear_value <= 15:
            mode_score -= 1.0
            reasons.append("market fear stressed")
    if tx_count is not None:
        reasons.append(f"BTC tx {tx_count}")
    if global_change is not None:
        reasons.append(f"global cap 24h {global_change:.2f}%")
        if global_change < -3.0:
            mode_score -= 1.3
            reasons.append("global crypto market weak")
        elif global_change < -1.0:
            mode_score -= 0.5
        elif global_change > 1.0:
            mode_score += 0.8
    if btc_dominance is not None:
        reasons.append(f"BTC dominance {btc_dominance:.2f}%")
    if binance_change is not None:
        reasons.append(f"Binance BTCUSDT 24h {binance_change:.2f}%")
        if binance_change < -2.0:
            mode_score -= 0.8
            reasons.append("global BTC pair weak")
        elif binance_change > 1.0:
            mode_score += 0.5
    if news_signal is not None:
        reasons.append(
            f"news sentiment {news_signal.sentiment_score:.2f} "
            f"({news_signal.positive_headline_count}+/{news_signal.risk_headline_count}-)"
        )
        if news_signal.risk_headline_count >= 3 and news_signal.sentiment_score < -0.15:
            mode_score -= 0.7
            reasons.append("news risk elevated")
        elif news_signal.sentiment_score > 0.15:
            mode_score += 0.35
    if community_signal is not None:
        reasons.append(
            f"community sentiment {community_signal.sentiment_score:.2f} "
            f"({community_signal.positive_headline_count}+/{community_signal.risk_headline_count}-)"
        )
        if community_signal.risk_headline_count >= 5 and community_signal.sentiment_score < -0.20:
            mode_score -= 0.55
            reasons.append("community panic elevated")
        elif community_signal.sentiment_score > 0.18:
            mode_score += 0.45
            reasons.append("community attention positive")

    market_mode, mode_multiplier, position_multiplier = classify_market_mode(mode_score, btc_momentum, global_change, binance_change)
    reasons.append(f"session {session_label}")
    reasons.append(f"mode {market_mode} score {mode_score:.2f}")

    return DecisionContext(
        allows_entries=allows,
        reason="; ".join(reasons),
        score_multiplier=mode_multiplier,
        btc_momentum_pct=btc_momentum,
        btc_volatility_pct=btc_volatility,
        market_mode=market_mode,
        mode_reason=f"score {mode_score:.2f}; session {session_label}",
        session_label=session_label,
        position_fraction_multiplier=position_multiplier,
        fear_greed_value=fear_value,
        fear_greed_label=fear_label,
        onchain_tx_count=tx_count,
        global_market_cap_change_pct=global_change,
        btc_dominance_pct=btc_dominance,
        binance_btcusdt_change_pct=binance_change,
        binance_btcusdt_quote_volume=binance_quote_volume,
        news_sentiment_score=news_signal.sentiment_score if news_signal is not None else None,
        news_risk_headline_count=news_signal.risk_headline_count if news_signal is not None else 0,
        news_positive_headline_count=news_signal.positive_headline_count if news_signal is not None else 0,
        news_headlines=news_signal.latest_headlines if news_signal is not None else None,
        community_sentiment_score=community_signal.sentiment_score if community_signal is not None else None,
        community_risk_count=community_signal.risk_headline_count if community_signal is not None else 0,
        community_positive_count=community_signal.positive_headline_count if community_signal is not None else 0,
        community_headlines=community_signal.latest_headlines if community_signal is not None else None,
    )


def fetch_news_signal():
    try:
        return fetch_crypto_news_signal(timeout_seconds=3, max_items=10)
    except Exception:
        return None


def fetch_local_community_signal():
    try:
        return fetch_community_signal(timeout_seconds=3, max_items=16)
    except Exception:
        return None


def current_session_label(now: datetime | None = None) -> tuple[str, float]:
    now = now or datetime.now(KST)
    hour = now.astimezone(KST).hour
    if 8 <= hour < 11:
        return "asia_morning_check", 0.2
    if 11 <= hour < 15:
        return "korea_midday", -0.1
    if 15 <= hour < 19:
        return "europe_prepare", 0.2
    if 19 <= hour < 22:
        return "evening_liquidity", 0.3
    if 22 <= hour or hour < 2:
        return "us_overlap", 0.4
    return "late_quiet", -0.2


def classify_market_mode(
    score: float,
    btc_momentum_pct: float,
    global_change_pct: float | None,
    binance_change_pct: float | None,
) -> tuple[str, float, float]:
    severe_global = global_change_pct is not None and global_change_pct <= -3.0
    severe_btc = binance_change_pct is not None and binance_change_pct <= -2.5
    if score <= -2.2 and (btc_momentum_pct < -0.35 or severe_global or severe_btc):
        return "panic_rebound", 0.74, 0.55
    if score >= 2.0:
        return "risk_on", 1.18, 1.15
    if score >= -0.4:
        return "neutral", 1.0, 1.0
    return "risk_off", 0.82, 0.7


def candle_momentum_pct(candles: list[Candle], lookback: int) -> float:
    if len(candles) <= lookback or candles[-1 - lookback].close <= 0:
        return 0.0
    return (candles[-1].close / candles[-1 - lookback].close - 1.0) * 100.0


def fetch_fear_greed() -> tuple[int | None, str]:
    try:
        payload = fetch_json("https://api.alternative.me/fng/?limit=1", timeout_seconds=4)
        first = payload.get("data", [{}])[0]
        return int(first["value"]), str(first.get("value_classification", "unknown"))
    except Exception:
        return None, "unavailable"


def fetch_blockchain_tx_count() -> int | None:
    try:
        payload = fetch_json("https://api.blockchain.info/stats", timeout_seconds=4)
        value = payload.get("n_tx")
        return int(value) if value is not None else None
    except Exception:
        return None


def fetch_coingecko_global() -> tuple[float | None, float | None]:
    try:
        payload = fetch_json("https://api.coingecko.com/api/v3/global", timeout_seconds=4)
        data = payload.get("data", {})
        change = data.get("market_cap_change_percentage_24h_usd")
        dominance = data.get("market_cap_percentage", {}).get("btc")
        return maybe_float(change), maybe_float(dominance)
    except Exception:
        return None, None


def fetch_binance_btcusdt_24h() -> tuple[float | None, float | None]:
    try:
        payload = fetch_json("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout_seconds=4)
        return maybe_float(payload.get("priceChangePercent")), maybe_float(payload.get("quoteVolume"))
    except Exception:
        return None, None


def fetch_json(url: str, timeout_seconds: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "coin-paper-simulation/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
