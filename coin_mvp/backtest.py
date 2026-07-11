from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import watch_multi as watch_multi_module
from .config import AppConfig, PathConfig, load_config
from .data import SampleMarketDataSource, UpbitPublicDataSource
from .market_context import DecisionContext, candle_momentum_pct, classify_market_mode
from .models import Candle, OrderbookSnapshot
from .report import calculate_metrics, read_events, read_trades, render_report
from .strategy import market_breadth_ratio, recent_volatility_pct, required_candle_count
from .watch_multi import MultiMarketTradingApp


@dataclass(frozen=True)
class BacktestSummary:
    ticks: int
    markets: list[str]
    entry_count: int
    exit_count: int
    candidate_ticks: int
    total_candidates: int
    total_realized_pnl: float
    return_pct: float
    win_rate: float
    payoff_ratio: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    max_consecutive_losses: int
    open_position_count: int
    top_blocked_reasons: list[dict[str, Any]]
    verdict: str
    report_path: str
    trade_path: str
    event_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReplayDataSource:
    def __init__(self, candles_by_unit: dict[int, dict[str, list[Candle]]], markets: list[str], start_index: int) -> None:
        self.candles_by_unit = candles_by_unit
        self.markets = markets
        self.current_index = start_index

    def set_index(self, index: int) -> None:
        self.current_index = index

    def get_top_krw_markets(self, count: int, min_trade_price_krw: float = 0.0) -> list[str]:
        return self.markets[:count]

    def get_recent_candles(self, market: str, count: int, unit_minutes: int | None = None) -> list[Candle]:
        unit = unit_minutes or 1
        candles = self.candles_by_unit[unit][market]
        if unit == 1:
            end = min(self.current_index + 1, len(candles))
        else:
            end = min(max(count, self.current_index // unit + 1), len(candles))
        start = max(0, end - count)
        result = candles[start:end]
        if len(result) < count and result:
            result = [result[0]] * (count - len(result)) + result
        return result

    def get_orderbook_snapshot(self, market: str) -> OrderbookSnapshot:
        latest = self.get_recent_candles(market, 1)[-1]
        spread = max(latest.close * 0.0006, 0.01)
        volume = max(latest.volume, 1.0)
        return OrderbookSnapshot(
            market=market,
            timestamp=latest.timestamp,
            best_bid_price=latest.close - spread / 2.0,
            best_bid_size=volume,
            best_ask_price=latest.close + spread / 2.0,
            best_ask_size=max(volume * 0.95, 0.001),
            total_bid_size=volume * 8.0,
            total_ask_size=volume * 7.6,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recent candles and verify strategy behavior.")
    parser.add_argument("--config", default="config.cloud.json")
    parser.add_argument("--source", choices=["sample", "upbit"], default="sample")
    parser.add_argument("--top-markets", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=60)
    parser.add_argument("--step-minutes", type=int, default=1)
    parser.add_argument("--history-count", type=int, default=200)
    parser.add_argument("--output", default="reports/backtest_latest.html")
    parser.add_argument("--summary-output", default="reports/backtest_latest.json")
    parser.add_argument("--request-delay", type=float, default=0.0)
    args = parser.parse_args()

    summary = run_backtest(
        config_path=args.config,
        source=args.source,
        top_markets=args.top_markets,
        ticks=args.ticks,
        step_minutes=args.step_minutes,
        history_count=args.history_count,
        output=Path(args.output),
        summary_output=Path(args.summary_output),
        request_delay=args.request_delay,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


def run_backtest(
    config_path: str | Path = "config.cloud.json",
    source: str = "sample",
    top_markets: int = 8,
    ticks: int = 60,
    step_minutes: int = 1,
    history_count: int = 200,
    output: Path = Path("reports/backtest_latest.html"),
    summary_output: Path = Path("reports/backtest_latest.json"),
    request_delay: float = 0.0,
) -> BacktestSummary:
    config = make_backtest_config(load_config(config_path), output)
    reset_backtest_files(config, output, summary_output)
    candles_by_unit, markets = load_replay_candles(source, top_markets, history_count, config)
    warmup = min(max(required_candle_count(config.strategy), 30), history_count - 2)
    replay = ReplayDataSource(candles_by_unit, markets, start_index=warmup)
    app = MultiMarketTradingApp(config, replay, markets, request_delay=request_delay)

    original_context = watch_multi_module.collect_decision_context
    watch_multi_module.collect_decision_context = lambda _data_source, _strategy: historical_backtest_context(
        replay,
        config,
    )
    try:
        step = max(1, step_minutes)
        last_index = min(history_count - 1, warmup + max(1, ticks) * step)
        completed = 0
        app.journal.event(
            "backtest_started",
            {
                "source": source,
                "markets": markets,
                "history_count": history_count,
                "warmup": warmup,
                "step_minutes": step,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        for index in range(warmup, last_index, step):
            replay.set_index(index)
            app.run_tick(completed + 1)
            completed += 1
        app.journal.event(
            "backtest_finished",
            {
                "ticks": completed,
                "cash": app.broker.cash,
                "equity": app.broker.equity(app.last_prices),
                "positions": {market: asdict(position) for market, position in app.broker.positions.items()},
                "risk": asdict(app.risk.state),
            },
        )
    finally:
        watch_multi_module.collect_decision_context = original_context

    trades = read_trades(config.paths.trade_journal)
    events = read_events(config.paths.event_log)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(trades, events), encoding="utf-8")
    summary = summarize_backtest(trades, events, config, markets, completed, output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def make_backtest_config(config: AppConfig, output: Path) -> AppConfig:
    stem = output.stem
    strategy = replace(
        config.strategy,
        long_trend_ema_window=min(config.strategy.long_trend_ema_window, 120),
        enable_bollinger_rebound_filter=config.strategy.enable_bollinger_rebound_filter,
    )
    risk = replace(config.risk, max_entries_per_day=max(config.risk.max_entries_per_day, 999))
    return replace(
        config,
        poll_seconds=0,
        strategy=strategy,
        risk=risk,
        ai_decision=replace(config.ai_decision, enabled=True, provider="local"),
        paths=PathConfig(
            trade_journal=Path("data") / f"{stem}_trades.csv",
            event_log=Path("logs") / f"{stem}_events.jsonl",
            state_file=Path("data") / f"{stem}_state.json",
        ),
    )


def reset_backtest_files(config: AppConfig, output: Path, summary_output: Path) -> None:
    for path in (config.paths.trade_journal, config.paths.event_log, config.paths.state_file, output, summary_output):
        if path.exists() and path.is_file():
            path.unlink()


def load_replay_candles(
    source: str,
    top_markets: int,
    history_count: int,
    config: AppConfig,
) -> tuple[dict[int, dict[str, list[Candle]]], list[str]]:
    if source == "sample":
        markets = [f"KRW-SAMPLE{i + 1}" for i in range(top_markets)]
        sample = SampleMarketDataSource()
        unit_one = {market: sample.get_recent_candles(market, history_count) for market in markets}
        return {1: unit_one, 5: compress_unit(unit_one, 5), 15: compress_unit(unit_one, 15), 60: compress_unit(unit_one, 60)}, markets

    data_source = UpbitPublicDataSource()
    markets = data_source.get_top_krw_markets(top_markets, min_trade_price_krw=config.strategy.min_price_krw)
    candles_by_unit: dict[int, dict[str, list[Candle]]] = {1: {}, 5: {}, 15: {}, 60: {}}
    for market in markets:
        candles_by_unit[1][market] = data_source.get_recent_candles(market, history_count, unit_minutes=1)
        candles_by_unit[5][market] = data_source.get_recent_candles(market, max(40, history_count // 5), unit_minutes=5)
        candles_by_unit[15][market] = data_source.get_recent_candles(market, max(30, history_count // 15), unit_minutes=15)
        candles_by_unit[60][market] = data_source.get_recent_candles(market, max(30, history_count // 60), unit_minutes=60)
    return candles_by_unit, markets


def compress_unit(unit_one: dict[str, list[Candle]], unit: int) -> dict[str, list[Candle]]:
    compressed: dict[str, list[Candle]] = {}
    for market, candles in unit_one.items():
        grouped = []
        for start in range(0, len(candles), unit):
            chunk = candles[start : start + unit]
            if not chunk:
                continue
            grouped.append(
                Candle(
                    market=market,
                    timestamp=chunk[-1].timestamp,
                    open=chunk[0].open,
                    high=max(candle.high for candle in chunk),
                    low=min(candle.low for candle in chunk),
                    close=chunk[-1].close,
                    volume=sum(candle.volume for candle in chunk),
                )
            )
        compressed[market] = grouped
    return compressed


def static_backtest_context() -> DecisionContext:
    return DecisionContext(
        allows_entries=True,
        reason="backtest static context",
        score_multiplier=1.0,
        btc_momentum_pct=0.0,
        btc_volatility_pct=0.0,
        market_mode="neutral",
        mode_reason="offline replay",
        session_label="backtest",
        position_fraction_multiplier=1.0,
        news_sentiment_score=0.0,
        news_risk_headline_count=0,
        news_positive_headline_count=0,
        news_headlines=[],
    )


def historical_backtest_context(replay: ReplayDataSource, config: AppConfig) -> DecisionContext:
    """Rebuild the market regime at the replay timestamp without future data."""
    lookback = max(config.strategy.long_window, 30)
    reference_market = "KRW-BTC" if "KRW-BTC" in replay.candles_by_unit.get(1, {}) else replay.markets[0]
    btc_candles = replay.get_recent_candles(reference_market, lookback)
    btc_momentum = candle_momentum_pct(btc_candles, lookback=5)
    btc_volatility = recent_volatility_pct(btc_candles, lookback=20)
    candles_by_market = {
        market: replay.get_recent_candles(market, lookback)
        for market in replay.markets
    }
    breadth = market_breadth_ratio(
        candles_by_market,
        config.strategy.short_window,
        config.strategy.long_window,
        min(config.strategy.long_trend_ema_window, lookback),
    )

    mode_score = 0.0
    if btc_momentum >= 0.25:
        mode_score += 1.5
    elif btc_momentum >= 0.0:
        mode_score += 0.6
    else:
        mode_score -= 1.0
    if breadth >= 0.65:
        mode_score += 0.8
    elif breadth >= 0.50:
        mode_score += 0.4
    elif breadth < 0.30:
        mode_score -= 1.0
    if btc_volatility > 1.2 and btc_momentum < 0:
        mode_score -= 0.5

    market_mode, score_multiplier, position_multiplier = classify_market_mode(
        mode_score,
        btc_momentum,
        None,
        None,
    )
    reason = (
        f"historical BTC momentum {btc_momentum:.2f}%; "
        f"volatility {btc_volatility:.2f}%; breadth {breadth:.2f}; "
        f"mode {market_mode} score {mode_score:.2f}"
    )
    return DecisionContext(
        allows_entries=True,
        reason=reason,
        score_multiplier=score_multiplier,
        btc_momentum_pct=btc_momentum,
        btc_volatility_pct=btc_volatility,
        market_mode=market_mode,
        mode_reason=f"historical replay score {mode_score:.2f}",
        session_label="historical_replay",
        position_fraction_multiplier=position_multiplier,
        news_sentiment_score=0.0,
        news_risk_headline_count=0,
        news_positive_headline_count=0,
        news_headlines=[],
    )


def summarize_backtest(
    trades,
    events: list[dict[str, Any]],
    config: AppConfig,
    markets: list[str],
    ticks: int,
    report_path: Path,
) -> BacktestSummary:
    metrics = calculate_metrics(trades)
    entries = [trade for trade in trades if trade.side == "buy"]
    exits = [trade for trade in trades if trade.side == "sell"]
    finished = next((event for event in reversed(events) if event.get("event") == "backtest_finished"), {})
    payload = finished.get("payload", {}) if isinstance(finished, dict) else {}
    positions = payload.get("positions", {}) if isinstance(payload, dict) else {}
    open_position_count = len(positions) if isinstance(positions, dict) else 0
    total_candidates, candidate_ticks = candidate_stats(events)
    top_blocked = top_blocked_reasons(events)
    total_realized = float(metrics["total_realized"])
    return_pct = total_realized / config.starting_cash * 100.0 if config.starting_cash else 0.0
    verdict = backtest_verdict(int(metrics["exit_count"]), total_realized, float(metrics["profit_factor"]), float(metrics["max_drawdown"]))
    return BacktestSummary(
        ticks=ticks,
        markets=markets,
        entry_count=len(entries),
        exit_count=len(exits),
        candidate_ticks=candidate_ticks,
        total_candidates=total_candidates,
        total_realized_pnl=total_realized,
        return_pct=return_pct,
        win_rate=float(metrics["win_rate"]),
        payoff_ratio=float(metrics["payoff_ratio"]),
        profit_factor=float(metrics["profit_factor"]),
        expectancy=float(metrics["expectancy"]),
        max_drawdown=float(metrics["max_drawdown"]),
        max_consecutive_losses=int(metrics["max_consecutive_losses"]),
        open_position_count=open_position_count,
        top_blocked_reasons=top_blocked,
        verdict=verdict,
        report_path=str(report_path),
        trade_path=str(config.paths.trade_journal),
        event_path=str(config.paths.event_log),
    )


def candidate_stats(events: list[dict[str, Any]]) -> tuple[int, int]:
    total = 0
    ticks = 0
    for event in events:
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        count = payload.get("candidate_count", payload.get("candidates"))
        if isinstance(count, int):
            total += count
            if count > 0:
                ticks += 1
    return total, ticks


def top_blocked_reasons(events: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for event in events:
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        blocked = payload.get("blocked_reasons", {})
        if not isinstance(blocked, dict):
            continue
        for reason, count in blocked.items():
            counts[str(reason)] = counts.get(str(reason), 0) + int(count)
    return [
        {"reason": reason, "count": count}
        for reason, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


def backtest_verdict(exit_count: int, total_realized: float, profit_factor: float, max_drawdown: float) -> str:
    if exit_count < 10:
        return "insufficient_sample"
    if total_realized > 0 and profit_factor >= 1.2 and max_drawdown > -30_000:
        return "pass"
    if total_realized > 0:
        return "watch"
    return "fail"


if __name__ == "__main__":
    main()
