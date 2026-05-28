from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from datetime import timedelta, timezone
from pathlib import Path

from .ai_decision import review_entry_candidate
from .broker import PortfolioPaperBroker
from .config import AppConfig, StrategyConfig, load_config
from .data import UpbitPublicDataSource, sleep_between_ticks
from .journal import Journal
from .market_context import collect_decision_context
from .models import Candle, Side, Signal
from .risk import RiskManager
from .strategy import (
    MovingAverageStrategy,
    bearish_crash_candle_risk,
    bollinger_lower_rebound_quality,
    btc_regime_allows_entries,
    calculate_ema,
    calculate_rsi,
    chart_feature_snapshot,
    estimate_expected_downside_pct,
    estimate_signal_expected_upside_pct,
    latest_volume_ratio,
    market_breadth_ratio,
    mean,
    required_candle_count,
    volatility_adjusted_position_fraction,
)
from .watch import refresh_report

KST = timezone(timedelta(hours=9))


class StrategyPerformanceGate:
    def __init__(self, trade_path: Path) -> None:
        self.trade_path = trade_path
        self._signature: tuple[int, int] | None = None
        self._round_trips: list[dict[str, object]] = []

    def round_trips(self) -> list[dict[str, object]]:
        signature = self._file_signature()
        if signature == self._signature:
            return self._round_trips
        self._signature = signature
        self._round_trips = self._load_round_trips()
        return self._round_trips

    def _file_signature(self) -> tuple[int, int]:
        if not self.trade_path.exists():
            return (0, 0)
        stat = self.trade_path.stat()
        return (int(stat.st_mtime), int(stat.st_size))

    def _load_round_trips(self) -> list[dict[str, object]]:
        if not self.trade_path.exists():
            return []
        open_entries: dict[str, dict[str, str]] = {}
        pairs: list[dict[str, object]] = []
        with self.trade_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                market = str(row.get("market") or "")
                side = str(row.get("side") or "")
                if not market:
                    continue
                if side == "buy":
                    open_entries[market] = row
                elif side == "sell":
                    entry = open_entries.pop(market, None)
                    if entry is None:
                        continue
                    try:
                        pnl = float(row.get("realized_pnl") or 0.0)
                    except ValueError:
                        continue
                    pairs.append(
                        {
                            "market": market,
                            "strategy": strategy_name_from_reason(str(entry.get("reason") or "")),
                            "reason_bucket": reason_bucket_from_reason(str(entry.get("reason") or "")),
                            "pnl": pnl,
                        }
                    )
        return pairs


class MultiMarketTradingApp:
    def __init__(self, config: AppConfig, data_source: UpbitPublicDataSource, markets: list[str], request_delay: float) -> None:
        self.config = config
        self.data_source = data_source
        self.markets = markets
        self.request_delay = request_delay
        self.broker = PortfolioPaperBroker(
            starting_cash=config.starting_cash,
            fee_rate=config.fee_rate,
            slippage_bps=config.slippage_bps,
        )
        self.strategy = MovingAverageStrategy(config.strategy)
        self.risk = RiskManager(config.risk, starting_equity=config.starting_cash)
        self.journal = Journal(config.paths.trade_journal, config.paths.event_log)
        self.position_entry_tick: dict[str, int] = {}
        self.position_entry_strategy: dict[str, str] = {}
        self.last_prices: dict[str, float] = {}
        self.market_reentry_until_tick: dict[str, int] = {}
        self.market_stopout_ticks: dict[str, list[int]] = {}
        self.last_decision_context: dict[str, object] = {}
        self.performance_gate = StrategyPerformanceGate(config.paths.trade_journal)

    def run_tick(self, tick: int) -> None:
        self.risk.refresh_halt(self.broker.equity(self.last_prices), tick=tick)
        for market in list(self.broker.open_markets()):
            self._manage_open_position(tick, market)
        if not self.config.risk.new_entries_enabled:
            self.journal.event(
                "market_scan",
                {
                    "tick": tick,
                    "markets_scanned": 0,
                    "candidates": 0,
                    "reason": "new entries disabled",
                    "risk": self.risk.state,
                },
            )
            return
        if self.risk.state.halted:
            self.journal.event(
                "market_scan",
                {
                    "tick": tick,
                    "markets_scanned": 0,
                    "candidates": 0,
                    "reason": self.risk.state.halt_reason,
                    "risk": self.risk.state,
                },
            )
            return
        if len(self.broker.open_markets()) >= self.config.risk.max_open_positions:
            return
        self._scan_and_enter(tick)

    def _manage_open_position(self, tick: int, market: str) -> None:
        candles = self.data_source.get_recent_candles(market, required_candle_count(self.config.strategy))
        latest_price = candles[-1].close
        self.last_prices[market] = latest_price
        position = self.broker.get_position(market)
        self.broker.mark_peak(market, latest_price)
        equity = self.broker.equity(self.last_prices)
        self.risk.ensure_trading_day(candles[-1].timestamp, equity)
        signal = self._position_management_signal(tick, market, candles, latest_price, position)
        if signal.side == Side.HOLD:
            signal = self.strategy.generate(candles, position)
        signal = self._apply_rebound_exit_grace(tick, market, latest_price, signal)
        signal = self._apply_small_loss_trend_hold(tick, market, latest_price, signal, position)
        signal = self._apply_time_stop(tick, market, latest_price, signal, position)
        position_fraction = volatility_adjusted_position_fraction(candles, self.config.strategy)
        approved, risk_reason = self.risk.approve(signal, equity, position_fraction, tick=tick)
        self._log_tick(tick, market, latest_price, equity, signal, approved, risk_reason)
        if not approved:
            if self.risk.state.halted:
                self._force_exit_if_needed(tick, market, latest_price)
            return
        if signal.side == Side.SELL:
            fill = self.broker.sell_fraction(market, signal.price, signal.size_fraction, signal.reason)
            if fill is not None:
                self.risk.record_fill(fill)
                self.journal.trade(fill)
                self.journal.event("fill", {"tick": tick, "fill": fill, "risk": self.risk.state})
                if fill.realized_pnl < 0:
                    self._register_stopout(market, tick)
                    self.market_reentry_until_tick[market] = tick + self.config.strategy.reentry_cooldown_ticks
                if not self.broker.get_position(market).is_open:
                    self.position_entry_tick.pop(market, None)
                    self.position_entry_strategy.pop(market, None)

    def _scan_and_enter(self, tick: int) -> None:
        context = collect_decision_context(self.data_source, self.config.strategy)
        self.last_decision_context = context.to_dict()
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        performance_ok, performance_reason = self._recent_performance_allows_entries()
        if not performance_ok:
            self.journal.event(
                "market_scan",
                {
                    "tick": tick,
                    "markets_scanned": 0,
                    "candidates": 0,
                    "reason": performance_reason,
                    "decision_context": context.to_dict(),
                    "risk": self.risk.state,
                },
            )
            return
        market_health_ok, market_health_reason = self._recent_market_stopouts_allow_entries(tick)
        if not market_health_ok:
            self.journal.event(
                "market_scan",
                {
                    "tick": tick,
                    "markets_scanned": 0,
                    "candidates": 0,
                    "reason": market_health_reason,
                    "decision_context": context.to_dict(),
                    "risk": self.risk.state,
                },
            )
            return
        if not context.allows_entries:
            self.journal.event(
                "market_scan",
                {
                    "tick": tick,
                    "markets_scanned": 0,
                    "candidates": 0,
                    "reason": context.reason,
                    "decision_context": context.to_dict(),
                    "risk": self.risk.state,
                },
            )
            return

        candidates = []
        blocked_reasons: dict[str, int] = {}
        blocked_samples: list[dict[str, object]] = []
        first_candles: list[Candle] | None = None
        candles_by_market: dict[str, list[Candle]] = {}
        for market in self.markets:
            if self.broker.get_position(market).is_open:
                continue
            try:
                candles = self.data_source.get_recent_candles(market, required_candle_count(self.config.strategy))
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
            except Exception as exc:
                self.journal.event("market_scan_error", {"tick": tick, "market": market, "error": repr(exc)})
                continue
            if first_candles is None:
                first_candles = candles
            candles_by_market[market] = candles
            self.last_prices[market] = candles[-1].close
            trend_screen_reason, trend_screen_penalty = self._universe_trend_signal(candles)
            blocked, blocked_reason = self._entry_time_block(candles[-1].timestamp)
            if blocked:
                blocked_reasons[blocked_reason] = blocked_reasons.get(blocked_reason, 0) + 1
                if len(blocked_samples) < 20:
                    blocked_samples.append(
                        {
                            "market": market,
                            "reason": blocked_reason,
                            "price": candles[-1].close,
                        }
                    )
                continue
            blocked, filter_reason, filter_penalty = self._entry_market_filters(tick, market, candles)
            if blocked:
                blocked_reasons[filter_reason] = blocked_reasons.get(filter_reason, 0) + 1
                if len(blocked_samples) < 20:
                    blocked_samples.append({"market": market, "reason": filter_reason, "price": candles[-1].close})
                continue
            signal = self.strategy.generate(candles, self.broker.get_position(market))
            if signal.side != Side.BUY:
                signal = self._bollinger_rebound_entry_signal(candles, filter_reason, signal)
            if signal.side == Side.BUY:
                strategy_gate_ok, strategy_gate_reason = self._candidate_adaptive_gate(market, signal, context)
                if not strategy_gate_ok:
                    blocked_reasons[strategy_gate_reason] = blocked_reasons.get(strategy_gate_reason, 0) + 1
                    if len(blocked_samples) < 20:
                        blocked_samples.append({"market": market, "reason": strategy_gate_reason, "price": candles[-1].close})
                    continue
                recovery_ok, recovery_reason = self._validated_recovery_ok(candles, signal)
                if not recovery_ok:
                    blocked_reasons[recovery_reason] = blocked_reasons.get(recovery_reason, 0) + 1
                    if len(blocked_samples) < 20:
                        blocked_samples.append({"market": market, "reason": recovery_reason, "price": candles[-1].close})
                    continue
                rr_ok, rr_reason = self._reward_risk_ok(candles, signal)
                if not rr_ok:
                    blocked_reasons[rr_reason] = blocked_reasons.get(rr_reason, 0) + 1
                    if len(blocked_samples) < 20:
                        blocked_samples.append({"market": market, "reason": rr_reason, "price": candles[-1].close})
                    continue
                penalty = (
                    self._recent_stopout_penalty(market, tick)
                    + self._entry_filter_penalty_for_signal(signal, filter_reason, filter_penalty)
                    + trend_screen_penalty
                )
                candidates.append((candidate_score(candles, signal, self.config.strategy, penalty=penalty) * context.score_multiplier, market, candles, signal))
            else:
                reason = signal.reason
                if trend_screen_reason != "universe trend bonus":
                    reason = f"{reason}; {trend_screen_reason}"
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
                if len(blocked_samples) < 20:
                    blocked_samples.append(
                        {
                            "market": market,
                            "reason": reason,
                            "price": candles[-1].close,
                        }
                    )

        if first_candles is None:
            raise RuntimeError("No market data was available during multi-market scan.")

        equity = self.broker.equity(self.last_prices)
        self.risk.ensure_trading_day(first_candles[-1].timestamp, equity)
        breadth_ratio = market_breadth_ratio(
            candles_by_market,
            self.config.strategy.short_window,
            self.config.strategy.long_window,
            self.config.strategy.long_window,
        )
        breadth_penalty = 0.0
        if breadth_ratio < self.config.strategy.min_market_breadth_ratio:
            shortage = self.config.strategy.min_market_breadth_ratio - breadth_ratio
            breadth_penalty = min(0.35, shortage * 0.8)
        if not candidates:
            self.journal.event(
                "market_scan",
                {
                    "tick": tick,
                    "markets_scanned": len(self.markets),
                    "candidates": 0,
                    "reason": "no entry condition" if breadth_penalty <= 0 else f"no entry condition; breadth penalty {breadth_penalty:.2f}",
                    "decision_context": context.to_dict(),
                    "blocked_reasons": blocked_reasons,
                    "blocked_samples": blocked_samples,
                    "market_breadth_ratio": breadth_ratio,
                    "market_breadth_penalty": breadth_penalty,
                    "risk": self.risk.state,
                },
            )
            return

        candidates.sort(key=lambda item: item[0], reverse=True)
        adjusted_candidates = [
            (score - self._breadth_penalty_for_candidate(score, signal, breadth_penalty), market, candles, signal)
            for score, market, candles, signal in candidates
        ]
        adjusted_candidates.sort(key=lambda item: item[0], reverse=True)
        filled_count = 0
        max_new_entries = self._max_new_entries_for_context(context)
        for score, market, candles, signal in adjusted_candidates:
            if score <= 0:
                continue
            equity = self.broker.equity(self.last_prices)
            score_floor = self._candidate_score_floor(context, equity)
            if score < score_floor:
                reason = f"candidate score below floor: {score:.2f} < {score_floor:.2f}"
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
                continue
            if len(self.broker.open_markets()) >= self.config.risk.max_open_positions:
                break
            latest_price = candles[-1].close
            self.last_prices[market] = latest_price
            equity = self.broker.equity(self.last_prices)
            position_fraction = self._position_fraction_for_context(candles, context, equity, signal, market, score)
            decision_review = review_entry_candidate(signal, candles, context, self.config.strategy, self.config.ai_decision)
            if decision_review.action != "buy":
                self._log_tick(
                    tick,
                    market,
                    latest_price,
                    equity,
                    signal,
                    False,
                    f"ai decision blocked: {decision_review.action}",
                    score=score,
                    candidates=len(candidates),
                    btc_regime=context.reason,
                    decision_context=context.to_dict(),
                    ai_decision=decision_review.to_dict(),
                    blocked_reasons=blocked_reasons,
                    blocked_samples=blocked_samples,
                    market_breadth_ratio=breadth_ratio,
                    breadth_penalty=breadth_penalty,
                )
                continue
            approved, risk_reason = self.risk.approve(signal, equity, position_fraction, tick=tick)
            self._log_tick(
                tick,
                market,
                latest_price,
                equity,
                signal,
                approved,
                risk_reason,
                score=score,
                candidates=len(candidates),
                btc_regime=context.reason,
                decision_context=context.to_dict(),
                ai_decision=decision_review.to_dict(),
                blocked_reasons=blocked_reasons,
                blocked_samples=blocked_samples,
                market_breadth_ratio=breadth_ratio,
                breadth_penalty=breadth_penalty,
            )
            if not approved:
                continue

            invested = self.broker.invested_value(self.last_prices)
            total_budget_remaining = max(0.0, equity * self.config.risk.max_total_position_fraction - invested)
            max_position_cash = equity * self.config.risk.max_position_fraction
            desired_cash = equity * position_fraction
            if desired_cash < self.config.risk.min_trade_cash_krw:
                desired_cash = min(self.config.risk.min_trade_cash_krw, max_position_cash)
            cash_to_use = min(
                desired_cash,
                max_position_cash,
                total_budget_remaining,
                self.broker.cash,
            )
            if cash_to_use < self.config.risk.min_trade_cash_krw:
                self.journal.event(
                    "fill_skipped",
                    {
                        "tick": tick,
                        "market": market,
                        "signal": signal,
                        "reason": f"trade cash below minimum: {cash_to_use:.0f} < {self.config.risk.min_trade_cash_krw:.0f}",
                    },
                )
                continue
            fill = self.broker.buy(market, signal.price, cash_to_use, f"{signal.reason}; {context.reason}; selected from top-volume scan")
            if fill is None:
                self.journal.event("fill_skipped", {"tick": tick, "signal": signal, "market": market})
                continue
            self.risk.record_fill(fill)
            self.journal.trade(fill)
            self.journal.event("fill", {"tick": tick, "fill": fill, "risk": self.risk.state})
            self.position_entry_tick[market] = tick
            self.position_entry_strategy[market] = self._entry_strategy_name(signal)
            filled_count += 1
            if filled_count >= max_new_entries:
                break
        if filled_count == 0:
            self.journal.event(
                "market_scan",
                {
                    "tick": tick,
                    "markets_scanned": len(self.markets),
                    "candidates": len(candidates),
                    "reason": "candidates reviewed but no fill",
                    "decision_context": context.to_dict(),
                    "blocked_reasons": blocked_reasons,
                    "blocked_samples": blocked_samples,
                    "market_breadth_ratio": breadth_ratio,
                    "market_breadth_penalty": breadth_penalty,
                    "risk": self.risk.state,
                },
            )

    def _entry_time_block(self, timestamp) -> tuple[bool, str]:
        blocked_hours = set(self.config.strategy.blocked_entry_hours_kst)
        if not blocked_hours:
            return False, ""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        hour = timestamp.astimezone(KST).hour
        if hour in blocked_hours:
            return True, f"blocked entry hour: {hour:02d} KST"
        return False, ""

    def _entry_market_filters(self, tick: int, market: str, candles: list[Candle]) -> tuple[bool, str, float]:
        latest_price = candles[-1].close
        if latest_price < self.config.strategy.min_price_krw:
            return True, f"price below floor: {latest_price:.2f} KRW", 0.0
        reentry_until = self.market_reentry_until_tick.get(market)
        if reentry_until is not None and tick < reentry_until:
            return True, f"reentry cooldown active until tick {reentry_until}", 0.0
        self._prune_stopouts(market, tick)
        if len(self.market_stopout_ticks.get(market, [])) >= self.config.strategy.max_recent_stopouts_per_market:
            return True, f"recent stopouts limit reached: {market}", 0.0
        crash_risk, crash_reason = bearish_crash_candle_risk(candles, self.config.strategy)
        if crash_risk:
            return True, crash_reason, 0.0
        orderbook = self.data_source.get_orderbook_snapshot(market)
        spread_limit = self.config.strategy.max_orderbook_spread_bps
        if orderbook.spread_bps > spread_limit * 2.0:
            return True, f"wide spread: {orderbook.spread_bps:.1f}bps", 0.0
        spread_penalty = 0.0
        if orderbook.spread_bps > spread_limit:
            spread_penalty = orderbook_spread_penalty(orderbook.spread_bps, spread_limit)
        imbalance_limit = self.config.strategy.min_orderbook_imbalance
        imbalance_penalty = 0.0
        if orderbook.imbalance_ratio < imbalance_limit:
            imbalance_penalty = orderbook_imbalance_penalty(orderbook.imbalance_ratio, imbalance_limit)
        mtf_ok, mtf_reason, mtf_penalty = self._five_minute_trend_ok(market)
        if not mtf_ok:
            return True, mtf_reason, 0.0
        bollinger_ok, bollinger_reason, bollinger_penalty = self._multi_timeframe_bollinger_ok(market)
        total_penalty = spread_penalty + imbalance_penalty + mtf_penalty
        reasons: list[str] = []
        if spread_penalty > 0.0:
            reasons.append(f"spread penalty: {orderbook.spread_bps:.1f}bps")
        if imbalance_penalty > 0.0:
            reasons.append(f"imbalance penalty: {orderbook.imbalance_ratio:.2f}")
        if mtf_reason != "5m trend ok":
            reasons.append(mtf_reason)
        if bollinger_reason != "bollinger filter disabled":
            if bollinger_ok:
                total_penalty += bollinger_penalty
            else:
                total_penalty += max(self.config.strategy.bollinger_filter_penalty, 0.08)
            reasons.append(bollinger_reason)
        reason = "; ".join(reasons) if reasons else mtf_reason
        return False, reason, total_penalty

    def _reward_risk_ok(self, candles: list[Candle], signal: Signal) -> tuple[bool, str]:
        expected_upside = estimate_signal_expected_upside_pct(candles, signal, self.config.strategy)
        expected_downside = estimate_expected_downside_pct(
            candles,
            self.config.strategy.stop_loss_pct,
            self.config.strategy.stop_volatility_multiplier,
        )
        roundtrip_cost = self._roundtrip_cost_pct()
        net_edge = expected_upside - expected_downside - roundtrip_cost
        if expected_upside <= 0:
            return False, "reward-risk blocked: no expected upside"
        if net_edge < self.config.strategy.min_net_edge_pct:
            return False, f"reward-risk blocked: net edge {net_edge:.2f}% < {self.config.strategy.min_net_edge_pct:.2f}%"
        downside_to_upside = expected_downside / expected_upside
        max_downside_to_upside = self._max_downside_to_upside_for_signal(signal)
        if downside_to_upside > max_downside_to_upside:
            reward_risk = expected_upside / expected_downside if expected_downside > 0 else 999.0
            required = 1.0 / max_downside_to_upside
            return False, f"reward-risk blocked: {reward_risk:.2f}R < {required:.2f}R"
        return True, f"reward-risk ok: upside {expected_upside:.2f}%, downside {expected_downside:.2f}%, net edge {net_edge:.2f}%"

    def _validated_recovery_ok(self, candles: list[Candle], signal: Signal) -> tuple[bool, str]:
        if len(candles) < max(6, self.config.strategy.long_window):
            return False, "validated recovery blocked: not enough candles"
        closes = [candle.close for candle in candles]
        latest = candles[-1]
        previous = candles[-2]
        lookback = min(5, len(closes) - 1)
        recovery_momentum = ((latest.close / closes[-1 - lookback]) - 1.0) * 100.0 if closes[-1 - lookback] > 0 else 0.0
        one_candle_recovery = ((latest.close / previous.close) - 1.0) * 100.0 if previous.close > 0 else 0.0
        short_ma = mean(closes[-self.config.strategy.short_window :])
        long_ma = mean(closes[-self.config.strategy.long_window :])
        volume_ratio = latest_volume_ratio(candles, lookback=10)
        candle_range = latest.high - latest.low
        close_position = 1.0 if candle_range <= 0 else (latest.close - latest.low) / candle_range
        recovered_price = latest.close >= short_ma or latest.close >= previous.close
        strong_recovery = recovery_momentum >= self.config.strategy.min_validated_recovery_pct or one_candle_recovery >= self.config.strategy.min_validated_recovery_pct
        if "trend breakout setup" in signal.reason:
            if recovery_momentum < self.config.strategy.min_validated_recovery_pct and one_candle_recovery < 0.05:
                return False, (
                    "validated recovery blocked: "
                    f"trend recovery {recovery_momentum:.2f}%, one-candle {one_candle_recovery:.2f}%"
                )
            if volume_ratio < self.config.strategy.min_volume_ratio:
                return False, f"validated recovery blocked: trend volume {volume_ratio:.2f}x"
            if close_position < 0.55:
                return False, f"validated recovery blocked: weak trend close position {close_position:.2f}"
            return True, (
                f"validated trend ok: recovery {recovery_momentum:.2f}%, "
                f"one-candle {one_candle_recovery:.2f}%, volume {volume_ratio:.2f}x"
            )
        if "micro recovery setup" in signal.reason:
            if recovery_momentum < 0.20 or one_candle_recovery < 0.10:
                return False, (
                    "validated recovery blocked: "
                    f"micro recovery {recovery_momentum:.2f}%, one-candle {one_candle_recovery:.2f}%"
                )
            if volume_ratio < max(1.6, self.config.strategy.min_volume_ratio):
                return False, f"validated recovery blocked: micro volume {volume_ratio:.2f}x"
            if close_position < 0.42:
                return False, f"validated recovery blocked: weak micro close position {close_position:.2f}"
            rsi = calculate_rsi(closes, self.config.strategy.rsi_period)
            if rsi is not None and rsi > 66.0:
                return False, f"validated recovery blocked: micro RSI {rsi:.1f}"
            return True, (
                f"validated micro recovery ok: recovery {recovery_momentum:.2f}%, "
                f"one-candle {one_candle_recovery:.2f}%, volume {volume_ratio:.2f}x"
            )
        if "chart ai setup" in signal.reason:
            if not recovered_price and one_candle_recovery < -0.08:
                return False, (
                    "validated recovery blocked: "
                    f"chart recovery {recovery_momentum:.2f}%, one-candle {one_candle_recovery:.2f}%"
                )
            features = chart_feature_snapshot(candles, self.config.strategy.rsi_period)
            if features and "momentum ignition" in signal.reason:
                if features["momentum_8_pct"] < 0.28:
                    return False, f"validated recovery blocked: chart momentum8 {features['momentum_8_pct']:.2f}%"
                if features["recent_high_gap_pct"] > 0.50:
                    return False, f"validated recovery blocked: chart high gap {features['recent_high_gap_pct']:.2f}%"
            if features and "pullback reclaim" in signal.reason:
                if features["momentum_3_pct"] < 0.25:
                    return False, f"validated recovery blocked: chart pullback momentum3 {features['momentum_3_pct']:.2f}%"
                if features["rsi"] > 58.0:
                    return False, f"validated recovery blocked: chart pullback RSI {features['rsi']:.1f}"
            if features and "volatility expansion" in signal.reason:
                if features["momentum_3_pct"] < 0.35 or features["momentum_8_pct"] < 0.35:
                    return False, (
                        "validated recovery blocked: chart expansion momentum "
                        f"{features['momentum_3_pct']:.2f}%/{features['momentum_8_pct']:.2f}%"
                    )
                if features["volume_ratio"] < max(2.0, self.config.strategy.min_volume_ratio * 1.5):
                    return False, f"validated recovery blocked: chart expansion volume {features['volume_ratio']:.2f}x"
            if volume_ratio < max(0.85, self.config.strategy.min_volume_ratio * 0.68):
                return False, f"validated recovery blocked: chart volume {volume_ratio:.2f}x"
            if close_position < 0.46:
                return False, f"validated recovery blocked: weak chart close position {close_position:.2f}"
            return True, (
                f"validated chart ai ok: recovery {recovery_momentum:.2f}%, "
                f"one-candle {one_candle_recovery:.2f}%, volume {volume_ratio:.2f}x"
            )
        if not recovered_price or not strong_recovery:
            return False, (
                "validated recovery blocked: "
                f"recovery {recovery_momentum:.2f}%, one-candle {one_candle_recovery:.2f}%"
            )
        if latest.close < long_ma * 0.985 and "trend breakout setup" in signal.reason:
            return False, "validated recovery blocked: trend entry below long MA buffer"
        if volume_ratio < self.config.strategy.min_volume_ratio:
            return False, f"validated recovery blocked: volume {volume_ratio:.2f}x"
        if close_position < 0.45:
            return False, f"validated recovery blocked: weak close position {close_position:.2f}"
        return True, (
            f"validated recovery ok: recovery {recovery_momentum:.2f}%, "
            f"one-candle {one_candle_recovery:.2f}%, volume {volume_ratio:.2f}x"
        )

    def _roundtrip_cost_pct(self) -> float:
        fee_pct = self.config.fee_rate * 2.0 * 100.0
        slippage_pct = self.config.slippage_bps * 2.0 / 100.0
        return fee_pct + slippage_pct

    def _position_fraction_for_context(
        self,
        candles: list[Candle],
        context,
        equity: float | None = None,
        signal: Signal | None = None,
        market: str | None = None,
        score: float | None = None,
    ) -> float:
        base = volatility_adjusted_position_fraction(candles, self.config.strategy)
        multiplier = float(getattr(context, "position_fraction_multiplier", 1.0) or 1.0)
        multiplier *= self._drawdown_exposure_multiplier(equity)
        if self.risk.state.consecutive_losses >= 2:
            multiplier *= 0.7
        elif self.risk.state.consecutive_losses == 1:
            multiplier *= 0.85
        if signal is not None and self.config.risk.adaptive_position_sizing:
            multiplier *= self._performance_position_multiplier(market or "", self._entry_strategy_name(signal))
            multiplier *= self._reason_bucket_position_multiplier(signal)
            multiplier *= self._conviction_position_multiplier(signal, score)
        return min(self.config.risk.max_position_fraction, max(0.0, base * multiplier))

    def _conviction_position_multiplier(self, signal: Signal, score: float | None) -> float:
        if score is None:
            return 1.0
        edge = score - self.config.risk.min_candidate_score
        reason = signal.reason.lower()
        if "chart ai setup: pullback reclaim" in reason:
            return 0.82
        if "chart ai setup" in reason:
            return 0.95
        if "micro recovery setup" in reason:
            if signal.confidence >= 0.66 and edge >= 0.08:
                return 0.95
            return 0.78
        if signal.confidence >= 0.76 and edge >= 0.18:
            return 1.35
        if signal.confidence >= 0.70 and edge >= 0.10:
            return 1.22
        if signal.confidence >= 0.64 and edge >= 0.04:
            return 1.08
        if edge < 0.02:
            return 0.88
        return 1.0

    def _max_downside_to_upside_for_signal(self, signal: Signal) -> float:
        reason = signal.reason.lower()
        configured = self.config.risk.max_expected_downside_to_upside_ratio
        if "trend breakout setup" in reason:
            return max(configured, 0.72)
        if "pullback continuation setup" in reason:
            return max(configured, 0.68)
        if "micro recovery setup" in reason:
            return max(configured, 0.62)
        if "chart ai setup" in reason:
            return max(configured, 0.58)
        return configured

    def _entry_filter_penalty_for_signal(self, signal: Signal, filter_reason: str, filter_penalty: float) -> float:
        if filter_penalty <= 0:
            return 0.0
        if "trend breakout setup" in signal.reason and "bollinger filter blocked" in filter_reason:
            bollinger_failure_penalty = max(self.config.strategy.bollinger_filter_penalty, 0.08)
            return max(0.0, filter_penalty - bollinger_failure_penalty)
        return filter_penalty

    def _breadth_penalty_for_candidate(self, score: float, signal: Signal, breadth_penalty: float) -> float:
        if breadth_penalty <= 0:
            return 0.0
        if signal.confidence >= 0.72 and score >= self.config.risk.min_candidate_score + 0.14:
            return breadth_penalty * 0.35
        if signal.confidence >= 0.64 and score >= self.config.risk.min_candidate_score + 0.08:
            return breadth_penalty * 0.6
        return breadth_penalty

    def _max_new_entries_for_context(self, context) -> int:
        configured = max(1, self.config.risk.max_new_entries_per_tick)
        if self.risk.state.entries_today >= 3:
            return 1
        mode = str(getattr(context, "market_mode", "neutral"))
        if mode == "risk_on":
            return configured
        return 1

    def _candidate_score_floor(self, context, equity: float | None = None) -> float:
        mode = str(getattr(context, "market_mode", "neutral"))
        base = self.config.risk.min_candidate_score
        if mode == "risk_on":
            base = max(0.0, base - 0.05)
        elif mode == "risk_off":
            base += 0.10
        elif mode == "capital_protect":
            return 999.0
        drawdown_pct = self._period_drawdown_pct(equity)
        if drawdown_pct <= -1.5:
            base += 0.10
        elif drawdown_pct <= -0.8:
            base += 0.05
        if self.risk.state.consecutive_losses >= 2:
            base += 0.12
        elif self.risk.state.consecutive_losses == 1:
            base += 0.05
        if self.risk.state.entries_today >= 12:
            base += 0.14
        elif self.risk.state.entries_today >= 6:
            base += 0.10
        elif self.risk.state.entries_today >= 3:
            base += 0.06
        return base

    def _candidate_adaptive_gate(self, market: str, signal: Signal, context) -> tuple[bool, str]:
        strategy_name = self._entry_strategy_name(signal)
        mode_ok, mode_reason = self._market_mode_allows_strategy(strategy_name, context)
        if not mode_ok:
            return False, mode_reason
        strategy_ok, strategy_reason = self._strategy_performance_allows_entry(strategy_name)
        if not strategy_ok:
            return False, strategy_reason
        reason_ok, reason = self._reason_bucket_performance_allows_entry(signal)
        if not reason_ok:
            return False, reason
        market_ok, market_reason = self._market_performance_allows_entry(market)
        if not market_ok:
            return False, market_reason
        return True, "adaptive gate ok"

    def _market_mode_allows_strategy(self, strategy_name: str, context) -> tuple[bool, str]:
        mode = str(getattr(context, "market_mode", "neutral"))
        if mode == "capital_protect":
            return False, "market mode blocked: capital protect"
        if mode == "risk_off" and strategy_name in {"trend", "micro_recovery"}:
            return False, f"market mode blocked: {mode} rejects {strategy_name}"
        return True, "market mode ok"

    def _strategy_performance_allows_entry(self, strategy_name: str) -> tuple[bool, str]:
        sample_size = self.config.risk.strategy_exit_sample_size
        if sample_size <= 0:
            return True, "strategy performance gate disabled"
        pnls = [
            float(item["pnl"])
            for item in self.performance_gate.round_trips()
            if item.get("strategy") == strategy_name
        ][-sample_size:]
        if len(pnls) < sample_size:
            return True, f"strategy performance warming up: {strategy_name} {len(pnls)}/{sample_size}"
        expectancy, profit_factor, loss_rate = performance_stats(pnls)
        if expectancy < self.config.risk.min_strategy_expectancy_krw:
            return False, f"strategy disabled: {strategy_name} expectancy {expectancy:.0f} KRW"
        if loss_rate > self.config.risk.max_strategy_loss_rate:
            return False, f"strategy disabled: {strategy_name} loss rate {loss_rate:.0%}"
        return True, f"strategy performance ok: {strategy_name} pf {profit_factor:.2f}"

    def _market_performance_allows_entry(self, market: str) -> tuple[bool, str]:
        sample_size = self.config.risk.market_exit_sample_size
        if sample_size <= 0:
            return True, "market performance gate disabled"
        pnls = [
            float(item["pnl"])
            for item in self.performance_gate.round_trips()
            if item.get("market") == market
        ][-sample_size:]
        if len(pnls) < sample_size:
            return True, f"market performance warming up: {market} {len(pnls)}/{sample_size}"
        expectancy, _profit_factor, loss_rate = performance_stats(pnls)
        if expectancy < self.config.risk.min_market_expectancy_krw:
            return False, f"market disabled: {market} expectancy {expectancy:.0f} KRW"
        if loss_rate > self.config.risk.max_market_loss_rate:
            return False, f"market disabled: {market} loss rate {loss_rate:.0%}"
        return True, f"market performance ok: {market}"

    def _reason_bucket_performance_allows_entry(self, signal: Signal) -> tuple[bool, str]:
        sample_size = self.config.risk.reason_exit_sample_size
        if sample_size <= 0:
            return True, "reason performance gate disabled"
        bucket = self._entry_reason_bucket(signal)
        pnls = [
            float(item["pnl"])
            for item in self.performance_gate.round_trips()
            if item.get("reason_bucket") == bucket
        ][-sample_size:]
        if len(pnls) < sample_size:
            return True, f"reason performance warming up: {bucket} {len(pnls)}/{sample_size}"
        expectancy, profit_factor, loss_rate = performance_stats(pnls)
        if expectancy < self.config.risk.min_reason_expectancy_krw:
            return False, f"reason disabled: {bucket} expectancy {expectancy:.0f} KRW"
        if loss_rate > self.config.risk.max_reason_loss_rate:
            return False, f"reason disabled: {bucket} loss rate {loss_rate:.0%}"
        return True, f"reason performance ok: {bucket} pf {profit_factor:.2f}"

    def _performance_position_multiplier(self, market: str, strategy_name: str) -> float:
        trips = self.performance_gate.round_trips()
        strategy_sample = [float(item["pnl"]) for item in trips if item.get("strategy") == strategy_name][-self.config.risk.strategy_exit_sample_size :]
        market_sample = [float(item["pnl"]) for item in trips if item.get("market") == market][-self.config.risk.market_exit_sample_size :]
        multiplier = 1.0
        for sample in (strategy_sample, market_sample):
            if len(sample) < 2:
                continue
            expectancy, profit_factor, loss_rate = performance_stats(sample)
            if expectancy > 0 and profit_factor >= 1.35 and loss_rate <= 0.5:
                multiplier *= 1.08
            elif expectancy < 0 or profit_factor < 1.0 or loss_rate >= 0.67:
                multiplier *= 0.72
            elif profit_factor < 1.15:
                multiplier *= 0.88
        return max(0.45, min(1.18, multiplier))

    def _reason_bucket_position_multiplier(self, signal: Signal) -> float:
        sample_size = self.config.risk.reason_exit_sample_size
        if sample_size <= 0:
            return 1.0
        bucket = self._entry_reason_bucket(signal)
        sample = [
            float(item["pnl"])
            for item in self.performance_gate.round_trips()
            if item.get("reason_bucket") == bucket
        ][-sample_size:]
        if len(sample) < 2:
            return 1.0
        expectancy, profit_factor, loss_rate = performance_stats(sample)
        if expectancy > 0 and profit_factor >= 1.45 and loss_rate <= 0.4:
            return 1.12
        if expectancy < 0 or profit_factor < 1.0 or loss_rate >= 0.67:
            return 0.58
        if profit_factor < 1.15:
            return 0.82
        return 1.0

    def _recent_performance_allows_entries(self) -> tuple[bool, str]:
        sample_size = self.config.risk.recent_exit_sample_size
        if sample_size <= 0:
            return True, "recent performance gate disabled"
        pnls = self._recent_exit_pnls(sample_size)
        if len(pnls) < sample_size:
            return True, f"recent performance sample warming up: {len(pnls)}/{sample_size}"
        wins = [value for value in pnls if value > 0]
        losses = [value for value in pnls if value < 0]
        expectancy = sum(pnls) / len(pnls)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
        loss_rate = len(losses) / len(pnls)
        if expectancy < self.config.risk.min_recent_expectancy_krw:
            return False, (
                "recent performance gate: "
                f"expectancy {expectancy:.0f} KRW < {self.config.risk.min_recent_expectancy_krw:.0f} KRW"
            )
        if profit_factor < self.config.risk.min_recent_profit_factor:
            return False, (
                "recent performance gate: "
                f"profit factor {profit_factor:.2f} < {self.config.risk.min_recent_profit_factor:.2f}"
            )
        if loss_rate > self.config.risk.max_recent_loss_rate:
            return False, (
                "recent performance gate: "
                f"loss rate {loss_rate:.0%} > {self.config.risk.max_recent_loss_rate:.0%}"
            )
        return True, (
            "recent performance ok: "
            f"expectancy {expectancy:.0f} KRW, profit factor {profit_factor:.2f}, loss rate {loss_rate:.0%}"
        )

    def _recent_exit_pnls(self, sample_size: int) -> list[float]:
        path = self.config.paths.trade_journal
        if not path.exists():
            return []
        values: list[float] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("side") != "sell":
                    continue
                try:
                    values.append(float(row.get("realized_pnl") or 0.0))
                except ValueError:
                    continue
        return values[-sample_size:]

    def _recent_market_stopouts_allow_entries(self, tick: int) -> tuple[bool, str]:
        lookback = max(1, min(self.config.strategy.stopout_lookback_ticks, 96))
        recent_markets = []
        for market in list(self.market_stopout_ticks):
            self._prune_stopouts(market, tick)
            ticks = self.market_stopout_ticks.get(market, [])
            if any(tick - value <= lookback for value in ticks):
                recent_markets.append(market)
        if len(recent_markets) >= 3:
            return False, f"market stopout cluster blocked: {len(recent_markets)} markets in {lookback} ticks"
        return True, "market stopout cluster ok"

    def _period_drawdown_pct(self, equity: float | None) -> float:
        if equity is None or self.risk.state.starting_equity <= 0:
            return 0.0
        return (equity / self.risk.state.starting_equity - 1.0) * 100.0

    def _drawdown_exposure_multiplier(self, equity: float | None) -> float:
        drawdown_pct = self._period_drawdown_pct(equity)
        if drawdown_pct <= -1.5:
            return 0.7
        if drawdown_pct <= -0.8:
            return 0.85
        if drawdown_pct >= 1.0:
            return 1.08
        return 1.0

    def _bollinger_rebound_entry_signal(self, candles: list[Candle], filter_reason: str, fallback: Signal) -> Signal:
        if not self.config.strategy.enable_bollinger_rebound_filter:
            return fallback
        if "bollinger lower rebound" not in filter_reason:
            return fallback
        latest_price = candles[-1].close
        expected_upside_pct = estimate_signal_expected_upside_pct(candles, fallback, self.config.strategy)
        if expected_upside_pct < self.config.strategy.bollinger_min_expected_upside_pct:
            reason = f"{fallback.reason}; bollinger rebound skipped: expected upside {expected_upside_pct:.2f}%"
            return Signal(fallback.side, reason, latest_price, fallback.confidence, fallback.size_fraction)
        confidence = min(0.74, 0.58 + min(expected_upside_pct / 30.0, 0.08))
        return Signal(
            Side.BUY,
            f"bollinger rebound setup: expected upside {expected_upside_pct:.2f}%; {filter_reason}",
            latest_price,
            confidence,
        )

    def _universe_trend_signal(self, candles: list[Candle]) -> tuple[str, float]:
        closes = [candle.close for candle in candles]
        if len(closes) < self.config.strategy.long_window:
            return "universe trend penalty: not enough candles", 0.18
        latest_price = closes[-1]
        short_ma = mean(closes[-self.config.strategy.short_window :])
        long_ma = mean(closes[-self.config.strategy.long_window :])
        long_trend_ema = calculate_ema(closes, self.config.strategy.long_trend_ema_window)
        if long_trend_ema is not None and latest_price < long_trend_ema:
            ema_gap_pct = ((latest_price / long_trend_ema) - 1.0) * 100.0 if long_trend_ema > 0 else 0.0
            if ema_gap_pct < -2.0:
                return f"universe trend penalty: deeply below EMA{self.config.strategy.long_trend_ema_window}", 0.16
            return f"universe trend penalty: below EMA{self.config.strategy.long_trend_ema_window} {ema_gap_pct:.2f}%", 0.06
        if short_ma <= long_ma:
            return f"universe trend penalty: short {short_ma:.3f} <= long {long_ma:.3f}", 0.14
        if latest_price <= long_ma:
            return f"universe trend penalty: price {latest_price:.3f} <= long {long_ma:.3f}", 0.12
        return "universe trend bonus", -0.05

    def _five_minute_trend_ok(self, market: str) -> tuple[bool, str, float]:
        candles = self.data_source.get_recent_candles(
            market,
            max(self.config.strategy.five_minute_long_window + 3, self.config.strategy.long_trend_ema_window // 5 if self.config.strategy.long_trend_ema_window else 0),
            unit_minutes=5,
        )
        closes = [candle.close for candle in candles]
        if len(closes) < self.config.strategy.five_minute_long_window:
            return False, "5m trend unavailable", 0.0
        latest_price = closes[-1]
        short_ma = mean(closes[-self.config.strategy.five_minute_short_window :])
        long_ma = mean(closes[-self.config.strategy.five_minute_long_window :])
        tolerance = self.config.strategy.five_minute_trend_tolerance_pct / 100.0
        min_short_ma = long_ma * (1.0 - tolerance)
        min_latest_price = short_ma * (1.0 - tolerance)
        momentum_pct = ((latest_price / closes[-min(4, len(closes) - 1) - 1]) - 1.0) * 100.0 if len(closes) > 4 else 0.0
        short_ma_shortfall = max(0.0, (min_short_ma - short_ma) / long_ma) if long_ma > 0 else 0.0
        latest_price_shortfall = max(0.0, (min_latest_price - latest_price) / short_ma) if short_ma > 0 else 0.0
        trend_shortfall = max(short_ma_shortfall, latest_price_shortfall)
        if trend_shortfall > 0.0:
            if trend_shortfall >= 0.01:
                return False, "5m trend weak", 0.0
            penalty = five_minute_trend_penalty(trend_shortfall)
            return True, f"5m trend penalty: {trend_shortfall * 100.0:.2f}%", penalty
        if momentum_pct < self.config.strategy.min_five_minute_momentum_pct:
            penalty = five_minute_momentum_penalty(momentum_pct, self.config.strategy.min_five_minute_momentum_pct)
            return True, f"5m momentum penalty: {momentum_pct:.2f}%", penalty
        return True, "5m trend ok", 0.0

    def _multi_timeframe_bollinger_ok(self, market: str) -> tuple[bool, str, float]:
        if not self.config.strategy.enable_bollinger_rebound_filter:
            return True, "bollinger filter disabled", 0.0
        count = self.config.strategy.bollinger_window + self.config.strategy.bollinger_prior_touch_lookback + 3
        checks = []
        for unit in (15, 60):
            candles = self.data_source.get_recent_candles(market, count, unit_minutes=unit)
            if self.request_delay > 0:
                time.sleep(self.request_delay)
            ok, reason = bollinger_lower_rebound_quality(
                candles,
                self.config.strategy.bollinger_window,
                self.config.strategy.bollinger_stddev,
                self.config.strategy.bollinger_touch_tolerance_pct,
                self.config.strategy.bollinger_prior_touch_lookback,
            )
            checks.append((unit, ok, reason))
        confirmations = sum(1 for _, ok, _ in checks if ok)
        required = min(len(checks), self.config.strategy.bollinger_min_confirmations)
        passed = [f"{unit}m {reason}" for unit, ok, reason in checks if ok]
        blocked = [f"{unit}m {reason}" for unit, ok, reason in checks if not ok]
        if confirmations >= required:
            missing_penalty = max(0, len(checks) - confirmations) * self.config.strategy.bollinger_filter_penalty
            reason = "; ".join(passed + blocked)
            return True, reason, missing_penalty
        return False, "; ".join(blocked), 0.0

    def _position_management_signal(self, tick: int, market: str, candles: list[Candle], latest_price: float, position) -> Signal:
        if not position.is_open:
            return Signal(Side.HOLD, "no position", latest_price, 0.0)
        avg_price = position.avg_price
        pnl_pct = (latest_price / avg_price - 1.0) * 100.0 if avg_price > 0 else 0.0
        dynamic_stop_loss_pct = estimate_expected_downside_pct(candles, self.config.strategy.stop_loss_pct, self.config.strategy.stop_volatility_multiplier)
        if pnl_pct <= -dynamic_stop_loss_pct:
            return Signal(Side.SELL, f"stop loss reached: {pnl_pct:.2f}%", latest_price, 0.95)
        if not position.partial_exit_taken and pnl_pct >= self.config.strategy.partial_take_profit_pct:
            return Signal(
                Side.SELL,
                f"partial take profit reached: {pnl_pct:.2f}%",
                latest_price,
                0.85,
                size_fraction=self.config.strategy.partial_take_profit_fraction,
            )
        peak_price = max(position.peak_price, latest_price)
        peak_pnl_pct = (peak_price / avg_price - 1.0) * 100.0 if avg_price > 0 else 0.0
        breakeven_floor_pct = self._roundtrip_cost_pct() * 0.75
        if peak_pnl_pct >= self.config.strategy.breakeven_trigger_pct and pnl_pct <= breakeven_floor_pct:
            return Signal(
                Side.SELL,
                f"breakeven stop reached: peak {peak_pnl_pct:.2f}%, pnl {pnl_pct:.2f}%",
                latest_price,
                0.82,
            )
        peak_drawdown_pct = (latest_price / peak_price - 1.0) * 100.0 if peak_price > 0 else 0.0
        if pnl_pct > 0 and peak_drawdown_pct <= -self.config.strategy.trailing_stop_pct:
            return Signal(Side.SELL, f"trailing stop reached: {peak_drawdown_pct:.2f}%", latest_price, 0.75)
        return Signal(Side.HOLD, "position managed", latest_price, 0.2)

    def _register_stopout(self, market: str, tick: int) -> None:
        ticks = self.market_stopout_ticks.setdefault(market, [])
        ticks.append(tick)
        self._prune_stopouts(market, tick)

    def _prune_stopouts(self, market: str, tick: int) -> None:
        lookback = self.config.strategy.stopout_lookback_ticks
        ticks = self.market_stopout_ticks.get(market, [])
        if not ticks:
            return
        self.market_stopout_ticks[market] = [value for value in ticks if tick - value <= lookback]

    def _recent_stopout_penalty(self, market: str, tick: int) -> float:
        self._prune_stopouts(market, tick)
        count = len(self.market_stopout_ticks.get(market, []))
        return min(0.3, count * 0.08)

    def _force_exit_if_needed(self, tick: int, market: str, latest_price: float) -> None:
        if not self.broker.get_position(market).is_open:
            return
        fill = self.broker.sell_all(market, latest_price, f"forced exit: {self.risk.state.halt_reason}")
        if fill is not None:
            self.risk.record_fill(fill)
            self.journal.trade(fill)
            self.journal.event("forced_exit", {"tick": tick, "fill": fill, "risk": self.risk.state})
            self.position_entry_tick.pop(market, None)
            self.position_entry_strategy.pop(market, None)

    def _entry_strategy_name(self, signal: Signal) -> str:
        if "bollinger rebound setup" in signal.reason:
            return "bollinger_rebound"
        if "range rebound setup" in signal.reason:
            return "range_rebound"
        if "chart ai setup" in signal.reason:
            return "chart_ai"
        if "pullback continuation setup" in signal.reason:
            return "pullback"
        if "micro recovery setup" in signal.reason:
            return "micro_recovery"
        return "trend"

    def _entry_reason_bucket(self, signal: Signal) -> str:
        return reason_bucket_from_reason(signal.reason)

    def _apply_range_rebound_exit_grace(
        self,
        tick: int,
        market: str | float,
        latest_price: float | Signal,
        signal: Signal | None = None,
    ) -> Signal:
        return self._apply_rebound_exit_grace(tick, market, latest_price, signal)

    def _apply_rebound_exit_grace(
        self,
        tick: int,
        market: str | float,
        latest_price: float | Signal,
        signal: Signal | None = None,
    ) -> Signal:
        if signal is None:
            signal = latest_price  # type: ignore[assignment]
            latest_price = market  # type: ignore[assignment]
            market = "KRW-BTC"
        assert isinstance(signal, Signal)
        if signal.side != Side.SELL or signal.reason != "trend break":
            return signal
        market_name = str(market)
        entry_strategy = self._entry_strategy_for(market_name)
        if entry_strategy not in {"range_rebound", "bollinger_rebound"}:
            return signal
        entry_tick = self._entry_tick_for(market_name)
        if entry_tick is None:
            return signal
        if entry_strategy == "bollinger_rebound":
            grace_ticks = self.config.strategy.bollinger_trend_break_grace_ticks
        else:
            grace_ticks = self.config.strategy.range_rebound_trend_break_grace_ticks
        if grace_ticks <= 0:
            return signal
        held_ticks = tick - entry_tick
        if held_ticks <= grace_ticks:
            strategy_label = entry_strategy.replace("_", " ")
            return Signal(
                Side.HOLD,
                f"{strategy_label} grace active: held {held_ticks} ticks, suppress trend break",
                float(latest_price),
                0.2,
            )
        return signal

    def _btc_regime(self) -> tuple[bool, str, float]:
        candles = self.data_source.get_recent_candles("KRW-BTC", required_candle_count(self.config.strategy))
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        return btc_regime_allows_entries(candles, self.config.strategy)

    def _apply_time_stop(self, tick: int, market: str, latest_price: float, signal: Signal, position) -> Signal:
        if signal.side == Side.SELL or not position.is_open:
            return signal
        entry_tick = self._entry_tick_for(market)
        if entry_tick is None:
            return signal
        max_ticks = self.config.strategy.time_stop_ticks
        if max_ticks <= 0:
            return signal
        held_ticks = tick - entry_tick
        pnl_pct = (latest_price / position.avg_price - 1.0) * 100.0
        if held_ticks >= max_ticks and pnl_pct <= self.config.strategy.time_stop_min_pnl_pct:
            return Signal(
                Side.SELL,
                f"time stop reached: held {held_ticks} ticks, pnl {pnl_pct:.2f}%",
                latest_price,
                0.7,
            )
        return signal

    def _apply_small_loss_trend_hold(self, tick: int, market: str, latest_price: float, signal: Signal, position) -> Signal:
        if signal.side != Side.SELL or signal.reason != "trend break" or not position.is_open:
            return signal
        entry_tick = self._entry_tick_for(market)
        if entry_tick is None:
            return signal
        pnl_pct = (latest_price / position.avg_price - 1.0) * 100.0 if position.avg_price > 0 else 0.0
        if pnl_pct >= 0.25:
            return Signal(Side.SELL, f"trend break profit lock: {pnl_pct:.2f}%", latest_price, signal.confidence)
        held_ticks = tick - entry_tick
        if pnl_pct > -self.config.strategy.stop_loss_pct * 0.65:
            return Signal(
                Side.HOLD,
                f"trend break watch: held {held_ticks} ticks, pnl {pnl_pct:.2f}%",
                latest_price,
                0.2,
            )
        return signal

    def _log_tick(
        self,
        tick: int,
        market: str,
        price: float,
        equity: float,
        signal: Signal,
        approved: bool,
        risk_reason: str,
        score: float | None = None,
        candidates: int | None = None,
        btc_regime: str | None = None,
        decision_context: dict[str, object] | None = None,
        ai_decision: dict[str, object] | None = None,
        blocked_reasons: dict[str, int] | None = None,
        blocked_samples: list[dict[str, object]] | None = None,
        market_breadth_ratio: float | None = None,
        breadth_penalty: float | None = None,
    ) -> None:
        payload = {
            "tick": tick,
            "market": market,
            "price": price,
            "equity": equity,
            "cash": self.broker.cash,
            "positions": {market: asdict(position) for market, position in self.broker.positions.items()},
            "last_prices": self.last_prices,
            "signal": signal,
            "approved": approved,
            "risk_reason": risk_reason,
            "risk": self.risk.state,
        }
        if score is not None:
            payload["candidate_score"] = score
        if candidates is not None:
            payload["candidate_count"] = candidates
        if btc_regime is not None:
            payload["btc_regime"] = btc_regime
        if decision_context is not None:
            payload["decision_context"] = decision_context
        if ai_decision is not None:
            payload["ai_decision"] = ai_decision
        if blocked_reasons is not None:
            payload["blocked_reasons"] = blocked_reasons
        if blocked_samples is not None:
            payload["blocked_samples"] = blocked_samples
        if market_breadth_ratio is not None:
            payload["market_breadth_ratio"] = market_breadth_ratio
        if breadth_penalty is not None:
            payload["market_breadth_penalty"] = breadth_penalty
        self.journal.event("tick", payload)

    def _entry_tick_for(self, market: str) -> int | None:
        if isinstance(self.position_entry_tick, dict):
            return self.position_entry_tick.get(market)
        if isinstance(self.position_entry_tick, int):
            return self.position_entry_tick
        return None

    def _entry_strategy_for(self, market: str) -> str | None:
        if isinstance(self.position_entry_strategy, dict):
            return self.position_entry_strategy.get(market)
        if isinstance(self.position_entry_strategy, str):
            return self.position_entry_strategy
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-market Upbit paper observation.")
    parser.add_argument("--config", default="config.lowload.json")
    parser.add_argument("--top-markets", type=int, default=30)
    parser.add_argument("--ticks", type=int, default=864)
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--output", default="reports/latest_report.html")
    parser.add_argument("--request-delay", type=float, default=0.18, help="Delay between Upbit candle requests during a scan.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_source = UpbitPublicDataSource()
    markets = data_source.get_top_krw_markets(args.top_markets, min_trade_price_krw=config.strategy.min_price_krw)
    app = MultiMarketTradingApp(config, data_source, markets, request_delay=args.request_delay)
    output = Path(args.output)

    app.journal.event(
        "watch_started",
        {
            "mode": config.mode,
            "source": "upbit",
            "market_mode": "top_krw_markets",
            "markets": markets,
            "request_delay": args.request_delay,
            "ticks": args.ticks,
            "report_every": args.report_every,
        },
    )
    refresh_report(config.paths.trade_journal, config.paths.event_log, output)

    for tick in range(1, args.ticks + 1):
        try:
            app.run_tick(tick)
            if tick % args.report_every == 0:
                refresh_report(config.paths.trade_journal, config.paths.event_log, output)
                print(f"Report refreshed at tick {tick}: {output}")
        except Exception as exc:
            app.journal.event("watch_error", {"tick": tick, "error": repr(exc)})
            refresh_report(config.paths.trade_journal, config.paths.event_log, output)
            raise
        sleep_between_ticks(config.poll_seconds, "upbit")

    app.journal.event(
        "watch_finished",
        {
            "cash": app.broker.cash,
            "position": asdict(app.broker.position),
            "positions": {market: asdict(position) for market, position in app.broker.positions.items()},
            "equity": app.broker.equity(app.last_prices),
            "risk": asdict(app.risk.state),
            "markets": markets,
        },
    )
    refresh_report(config.paths.trade_journal, config.paths.event_log, output)


def candidate_score(candles: list[Candle], signal: Signal, config: StrategyConfig, penalty: float = 0.0) -> float:
    closes = [candle.close for candle in candles]
    momentum = (closes[-1] / closes[-5] - 1.0) if len(closes) >= 5 and closes[-5] else 0.0
    volume_value = candles[-1].close * candles[-1].volume
    pullback_risk = max(0.0, (max(closes[-5:]) / closes[-1] - 1.0)) if len(closes) >= 5 and closes[-1] else 0.0
    expected_upside = estimate_signal_expected_upside_pct(candles, signal, config) / 100.0
    expected_downside = estimate_expected_downside_pct(candles, config.stop_loss_pct, config.stop_volatility_multiplier) / 100.0
    net_edge = expected_upside - expected_downside
    chart = chart_feature_snapshot(candles, config.rsi_period)
    chart_quality = 0.0
    if chart:
        chart_quality += min(max(chart["momentum_3_pct"], 0.0) / 100.0, 0.035)
        chart_quality += min(max(chart["volume_ratio"] - 0.8, 0.0) * 0.035, 0.045)
        chart_quality += min(max(chart["close_position"] - 0.45, 0.0) * 0.08, 0.04)
        if "chart ai setup" in signal.reason.lower():
            chart_quality += 0.04
    return (
        signal.confidence
        + momentum
        + expected_upside
        + max(net_edge, -0.03)
        + chart_quality
        - expected_downside * 0.6
        - pullback_risk
        + min(volume_value / 1_000_000_000_000.0, 0.25)
        - penalty
    )


def five_minute_momentum_penalty(momentum_pct: float, min_required_pct: float) -> float:
    shortfall = max(0.0, min_required_pct - momentum_pct)
    return min(0.18, 0.06 + shortfall * 0.6)


def five_minute_trend_penalty(shortfall_ratio: float) -> float:
    return min(0.2, 0.06 + shortfall_ratio * 8.0)


def orderbook_spread_penalty(spread_bps: float, max_spread_bps: float) -> float:
    excess = max(0.0, spread_bps - max_spread_bps)
    return min(0.22, 0.05 + excess / max(1.0, max_spread_bps) * 0.18)


def orderbook_imbalance_penalty(imbalance_ratio: float, min_imbalance_ratio: float) -> float:
    shortfall = max(0.0, min_imbalance_ratio - imbalance_ratio)
    return min(0.2, 0.05 + shortfall / max(0.1, min_imbalance_ratio) * 0.2)


def strategy_name_from_reason(reason: str) -> str:
    text = reason.lower()
    if "bollinger rebound setup" in text:
        return "bollinger_rebound"
    if "range rebound setup" in text:
        return "range_rebound"
    if "pullback continuation setup" in text:
        return "pullback"
    if "micro recovery setup" in text:
        return "micro_recovery"
    if "chart ai setup" in text:
        return "chart_ai"
    return "trend"


def reason_bucket_from_reason(reason: str) -> str:
    text = reason.lower()
    if "chart ai setup: pullback reclaim" in text:
        return "chart_ai_pullback_reclaim"
    if "chart ai setup: momentum ignition" in text:
        return "chart_ai_momentum_ignition"
    if "chart ai setup" in text:
        return "chart_ai_other"
    if "micro recovery setup" in text:
        return "micro_recovery"
    if "pullback continuation setup" in text:
        if "below ema200" in text:
            return "pullback_below_ema200"
        return "pullback_continuation"
    if "trend breakout setup" in text:
        if "momentum 0." in text:
            return "trend_low_momentum"
        return "trend_breakout"
    if "range rebound setup" in text:
        return "range_rebound"
    if "bollinger rebound setup" in text:
        return "bollinger_rebound"
    return strategy_name_from_reason(reason)


def performance_stats(pnls: list[float]) -> tuple[float, float, float]:
    if not pnls:
        return 0.0, 0.0, 0.0
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    expectancy = sum(pnls) / len(pnls)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
    loss_rate = len(losses) / len(pnls)
    return expectancy, profit_factor, loss_rate


if __name__ == "__main__":
    main()
