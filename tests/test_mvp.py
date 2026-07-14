import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from coin_mvp.broker import PaperBroker
from coin_mvp.ai_decision import extract_openai_json, review_entry_candidate
from coin_mvp.backtest import backtest_verdict, candidate_stats, top_blocked_reasons
from coin_mvp.cloud_tick import resume_state
from coin_mvp.config import AiDecisionConfig, AppConfig, PathConfig, RiskConfig, StrategyConfig
from coin_mvp.data import UpbitPublicDataSource
from coin_mvp.market_context import DecisionContext, maybe_float
from coin_mvp.ml_decision import opportunity_edge_score, score_entry_with_feature_model
from coin_mvp.models import Candle, Fill, OrderbookSnapshot, Position, Side, Signal
from coin_mvp.news import score_headlines
from coin_mvp.report import calculate_max_consecutive_losses, calculate_max_drawdown
from coin_mvp.risk import RiskManager
from coin_mvp.strategy import (
    MovingAverageStrategy,
    bollinger_lower_rebound_quality,
    btc_regime_allows_entries,
    calculate_ema,
    chart_feature_snapshot,
    estimate_expected_downside_pct,
    estimate_expected_upside_pct,
    market_breadth_ratio,
    volatility_adjusted_position_fraction,
)
from coin_mvp.watch_multi import (
    MultiMarketTradingApp,
    five_minute_momentum_penalty,
    five_minute_trend_penalty,
    orderbook_imbalance_penalty,
    orderbook_spread_penalty,
    opportunity_score_for_signal,
    performance_stats,
    reason_bucket_from_reason,
    strategy_name_from_reason,
)


class PaperBrokerTest(unittest.TestCase):
    def test_buy_and_sell_updates_cash_and_position(self):
        broker = PaperBroker("KRW-BTC", starting_cash=1_000_000, fee_rate=0.0005, slippage_bps=0)

        buy = broker.buy(price=50_000_000, cash_to_use=200_000, reason="test buy")
        self.assertIsNotNone(buy)
        self.assertGreater(broker.position.qty, 0)
        self.assertLess(broker.cash, 1_000_000)

        sell = broker.sell_all(price=51_000_000, reason="test sell")
        self.assertIsNotNone(sell)
        self.assertEqual(broker.position.qty, 0)
        self.assertGreater(sell.realized_pnl, 0)

    def test_partial_sell_keeps_remaining_position(self):
        broker = PaperBroker("KRW-BTC", starting_cash=1_000_000, fee_rate=0.0005, slippage_bps=0)
        broker.buy(price=50_000_000, cash_to_use=200_000, reason="test buy")

        fill = broker.sell_fraction(price=50_500_000, fraction=0.5, reason="partial")

        self.assertIsNotNone(fill)
        self.assertGreater(broker.position.qty, 0)
        self.assertTrue(broker.position.partial_exit_taken)


class RiskManagerTest(unittest.TestCase):
    def test_daily_loss_halts_trading(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=1.0,
                daily_loss_limit_pct=1.0,
                max_entries_per_day=3,
                max_position_fraction=0.25,
                max_consecutive_losses=2,
            ),
            starting_equity=1_000_000,
        )

        approved, reason = risk.approve(
            Signal(Side.BUY, "test", price=100.0),
            current_equity=989_000,
            position_fraction=0.2,
        )
        self.assertFalse(approved)
        self.assertIn("daily loss limit", reason)

    def test_sell_is_allowed_after_entry_limit(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=1.0,
                daily_loss_limit_pct=1.0,
                max_entries_per_day=1,
                max_position_fraction=0.25,
                max_consecutive_losses=2,
            ),
            starting_equity=1_000_000,
        )
        risk.state.entries_today = 1

        approved, reason = risk.approve(
            Signal(Side.SELL, "exit", price=100.0),
            current_equity=1_000_000,
            position_fraction=0.2,
        )
        self.assertTrue(approved)
        self.assertIn("risk-reducing exit", reason)

    def test_new_entry_pause_blocks_buys_but_allows_sells(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=1.0,
                daily_loss_limit_pct=1.0,
                max_entries_per_day=3,
                max_position_fraction=0.25,
                max_consecutive_losses=2,
                new_entries_enabled=False,
            ),
            starting_equity=1_000_000,
        )

        approved, reason = risk.approve(Signal(Side.BUY, "entry", price=100.0), 1_000_000, 0.2)
        self.assertFalse(approved)
        self.assertEqual("new entries disabled", reason)

        approved, reason = risk.approve(Signal(Side.SELL, "exit", price=100.0), 1_000_000, 0.2)
        self.assertTrue(approved)
        self.assertIn("risk-reducing exit", reason)

    def test_minimum_equity_floor_halts_entries(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=3.0,
                daily_loss_limit_pct=20.0,
                max_entries_per_day=3,
                max_position_fraction=0.25,
                max_consecutive_losses=2,
                min_equity_krw=900_000,
            ),
            starting_equity=1_000_000,
        )

        approved, reason = risk.approve(Signal(Side.BUY, "entry", price=100.0), 899_000, 0.2, tick=1)

        self.assertFalse(approved)
        self.assertIn("minimum equity floor reached", reason)
        self.assertTrue(risk.state.halted)

    def test_new_24_hour_period_resets_target_base(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=3.0,
                daily_loss_limit_pct=5.0,
                max_entries_per_day=12,
                max_position_fraction=0.35,
                max_consecutive_losses=4,
            ),
            starting_equity=1_000_000,
        )
        risk.ensure_trading_day(datetime(2026, 4, 19, 0, 0, tzinfo=timezone.utc), 1_000_000)
        risk.state.entries_today = 3
        risk.state.halted = True
        risk.state.halt_reason = "daily profit target reached: 3.00%"

        risk.ensure_trading_day(datetime(2026, 4, 20, 0, 1, tzinfo=timezone.utc), 1_030_000)

        self.assertEqual(risk.state.starting_equity, 1_030_000)
        self.assertEqual(risk.state.entries_today, 0)
        self.assertFalse(risk.state.halted)

    def test_halt_cooldown_releases_trading(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=3.0,
                daily_loss_limit_pct=1.0,
                max_entries_per_day=12,
                max_position_fraction=0.35,
                max_consecutive_losses=4,
                halt_cooldown_ticks=2,
            ),
            starting_equity=1_000_000,
        )

        approved, reason = risk.approve(Signal(Side.BUY, "test", price=100.0), 989_000, 0.2, tick=10)
        self.assertFalse(approved)
        self.assertIn("daily loss limit", reason)

        approved, _ = risk.approve(Signal(Side.BUY, "test", price=100.0), 1_000_000, 0.2, tick=12)
        self.assertTrue(approved)

    def test_daily_loss_cooldown_rebases_equity_after_pause(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=3.0,
                daily_loss_limit_pct=1.0,
                max_entries_per_day=12,
                max_position_fraction=0.35,
                max_consecutive_losses=4,
                halt_cooldown_ticks=2,
            ),
            starting_equity=1_000_000,
        )

        approved, reason = risk.approve(Signal(Side.BUY, "test", price=100.0), 989_000, 0.2, tick=10)
        self.assertFalse(approved)
        self.assertIn("daily loss limit", reason)

        approved, _ = risk.approve(Signal(Side.BUY, "test", price=100.0), 989_000, 0.2, tick=12)
        self.assertTrue(approved)
        self.assertEqual(989_000, risk.state.starting_equity)
        self.assertFalse(risk.state.halted)

    def test_consecutive_loss_cooldown_resets_loss_counter(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=3.0,
                daily_loss_limit_pct=5.0,
                max_entries_per_day=12,
                max_position_fraction=0.35,
                max_consecutive_losses=4,
                consecutive_loss_cooldown_ticks=2,
            ),
            starting_equity=1_000_000,
        )
        risk.state.consecutive_losses = 4

        approved, reason = risk.approve(Signal(Side.BUY, "test", price=100.0), 1_000_000, 0.2, tick=10)
        self.assertFalse(approved)
        self.assertEqual("max consecutive losses reached", reason)

        approved, _ = risk.approve(Signal(Side.BUY, "test", price=100.0), 1_000_000, 0.2, tick=12)
        self.assertTrue(approved)
        self.assertEqual(0, risk.state.consecutive_losses)

    def test_record_fill_halts_immediately_on_max_consecutive_losses(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=3.0,
                daily_loss_limit_pct=5.0,
                max_entries_per_day=12,
                max_position_fraction=0.35,
                max_consecutive_losses=3,
                consecutive_loss_cooldown_ticks=12,
            ),
            starting_equity=1_000_000,
        )
        loss_fill = Fill(
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
            market="KRW-BTC",
            side=Side.SELL,
            price=100.0,
            qty=1.0,
            fee=0.0,
            cash_after=999_900.0,
            position_qty_after=0.0,
            realized_pnl=-100.0,
            reason="test loss",
        )

        risk.record_fill(loss_fill, tick=10)
        risk.record_fill(loss_fill, tick=11)
        risk.record_fill(loss_fill, tick=12)

        self.assertTrue(risk.state.halted)
        self.assertEqual("max consecutive losses reached", risk.state.halt_reason)
        self.assertEqual(24, risk.state.halt_until_tick)

    def test_consecutive_loss_cooldown_can_be_shortened_after_state_was_saved(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=3.0,
                daily_loss_limit_pct=5.0,
                max_entries_per_day=12,
                max_position_fraction=0.35,
                max_consecutive_losses=4,
                consecutive_loss_cooldown_ticks=2,
            ),
            starting_equity=1_000_000,
        )
        risk.state.halted = True
        risk.state.halt_reason = "max consecutive losses reached"
        risk.state.halt_started_tick = 209
        risk.state.halt_until_tick = 221
        risk.state.consecutive_losses = 5

        approved, _ = risk.approve(Signal(Side.BUY, "test", price=100.0), 1_000_000, 0.2, tick=216)
        self.assertTrue(approved)
        self.assertIsNone(risk.state.halt_until_tick)
        self.assertEqual(0, risk.state.consecutive_losses)


class ReportMetricsTest(unittest.TestCase):
    def test_resume_state_clears_halt_without_resetting_cash(self):
        state = {
            "tick": 65,
            "equity": 989_696.0,
            "broker": {"cash": 989_696.0, "positions": {}},
            "risk": {
                "starting_equity": 1_000_000.0,
                "consecutive_losses": 1,
                "halted": True,
                "halt_reason": "daily loss limit reached: -1.03%",
                "halt_started_tick": 40,
                "halt_until_tick": 44,
            },
        }

        resumed = resume_state(state)

        self.assertEqual(989_696.0, resumed["broker"]["cash"])
        self.assertEqual(989_696.0, resumed["risk"]["starting_equity"])
        self.assertFalse(resumed["risk"]["halted"])
        self.assertEqual("", resumed["risk"]["halt_reason"])

    def test_max_drawdown_uses_cumulative_realized_pnl(self):
        self.assertEqual(calculate_max_drawdown([1000, -300, -500, 200]), -800)

    def test_max_consecutive_losses(self):
        self.assertEqual(calculate_max_consecutive_losses([1000, -1, -2, 3, -4, -5, -6]), 3)

    def test_backtest_candidate_and_blocked_reason_summary(self):
        events = [
            {"payload": {"candidates": 2, "blocked_reasons": {"weak bounce": 2}}},
            {"payload": {"candidate_count": 1, "blocked_reasons": {"weak bounce": 1, "wide spread": 1}}},
            {"payload": {"candidates": 0}},
        ]

        total, ticks = candidate_stats(events)
        blocked = top_blocked_reasons(events, limit=2)

        self.assertEqual(total, 3)
        self.assertEqual(ticks, 2)
        self.assertEqual(blocked[0]["reason"], "weak bounce")

    def test_backtest_verdict_marks_small_samples(self):
        self.assertEqual(backtest_verdict(3, 1000.0, 2.0, -100.0), "insufficient_sample")
        self.assertEqual(backtest_verdict(12, -1000.0, 0.8, -2000.0), "fail")


class StrategyFilterTest(unittest.TestCase):
    def test_entry_blocks_overextended_move(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            max_recent_momentum_pct=1.0,
            min_volume_ratio=0.5,
            long_trend_ema_window=0,
        )
        candles = make_candles([100, 101, 102, 103, 104, 112], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertIn("overextended", signal.reason)

    def test_btc_regime_blocks_weak_trend(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            btc_short_window=3,
            btc_long_window=5,
            min_btc_momentum_pct=-0.5,
        )
        candles = make_candles([100, 99, 98, 97, 96, 94], volume=10.0)

        allowed, reason, _ = btc_regime_allows_entries(candles, config)

        self.assertFalse(allowed)
        self.assertIn("btc regime blocked", reason)

    def test_entry_blocks_high_rsi(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            max_entry_rsi=70.0,
            min_volume_ratio=0.5,
            max_recent_momentum_pct=10.0,
            long_trend_ema_window=0,
        )
        candles = make_candles([100, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertIn("overextended: RSI", signal.reason)

    def test_volatility_reduces_position_fraction(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            target_recent_volatility_pct=1.0,
            min_volatility_position_fraction=0.4,
        )
        candles = make_candles([100, 105, 97, 108, 96, 110, 95, 112, 94, 115, 93, 118, 92, 120, 91, 122, 90, 124, 89, 126, 88], volume=10.0)

        fraction = volatility_adjusted_position_fraction(candles, config)

        self.assertLess(fraction, config.position_fraction)
        self.assertGreaterEqual(fraction, config.position_fraction * config.min_volatility_position_fraction)

    def test_long_ema_does_not_hard_block_shallow_breakout(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_volume_ratio=0.5,
            long_trend_ema_window=8,
        )
        candles = make_candles([150, 140, 130, 100, 101, 102, 103, 104], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertNotIn("long trend filter blocked", signal.reason)

    def test_generate_surfaces_ma_alignment_failure_reason(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_volume_ratio=0.5,
            long_trend_ema_window=0,
        )
        candles = make_candles([105, 104, 103, 102, 101, 100], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertIn("ma alignment failed", signal.reason)

    def test_generate_surfaces_price_below_long_ma_reason(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_volume_ratio=0.5,
            long_trend_ema_window=0,
        )
        candles = make_candles([95, 96, 97, 110, 100, 98], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertIn("price below long MA", signal.reason)

    def test_range_rebound_helper_can_identify_weak_trend_bounce(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_recent_momentum_pct=0.05,
            max_recent_momentum_pct=4.0,
            min_volume_ratio=1.2,
            long_trend_ema_window=8,
            rsi_period=5,
            enable_range_rebound=True,
            range_rebound_lookback=8,
            range_rebound_max_distance_from_low_pct=2.5,
            range_rebound_max_ema_gap_pct=3.0,
            range_rebound_min_bounce_pct=0.1,
            range_rebound_min_volume_ratio=0.9,
            range_rebound_min_expected_upside_pct=1.0,
            range_rebound_min_rsi=30.0,
            range_rebound_max_entry_rsi=60.0,
        )
        candles = make_variable_candles(
            [103, 102, 101, 100.5, 100, 99.5, 99, 100.2],
            [10, 10, 10, 10, 10, 10, 10, 16],
        )

        strategy = MovingAverageStrategy(config)
        signal, _ = strategy._range_rebound_signal(candles, 99.5, 100.0)  # noqa: SLF001

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.side, Side.BUY)
        self.assertIn("range rebound setup", signal.reason)

    def test_cost_aware_edge_rejects_flat_low_edge_market(self):
        config = StrategyConfig(
            short_window=5,
            long_window=20,
            take_profit_pct=0.72,
            stop_loss_pct=0.34,
            position_fraction=0.72,
            min_volume_ratio=0.8,
            min_expected_upside_pct=0.85,
            min_net_edge_pct=0.20,
        )
        closes = [
            100.0,
            100.02,
            99.98,
            100.01,
            100.0,
            100.03,
            99.99,
            100.02,
            100.01,
            100.0,
            100.02,
            99.98,
            100.01,
            100.0,
            100.03,
            99.99,
            100.02,
            100.01,
            100.0,
            100.02,
            99.98,
            100.01,
            100.0,
            100.02,
            100.01,
        ]
        signal = MovingAverageStrategy(config).generate(make_candles(closes, volume=10.0), Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertIn("cost-aware edge blocked", signal.reason)

    def test_cost_aware_edge_buys_valid_breakout_after_costs(self):
        config = StrategyConfig(
            short_window=5,
            long_window=20,
            take_profit_pct=0.72,
            stop_loss_pct=0.34,
            position_fraction=0.72,
            min_volume_ratio=0.8,
            min_expected_upside_pct=0.75,
            min_net_edge_pct=0.12,
            target_upside_pct=2.2,
            max_entry_rsi=74.0,
        )
        closes = [
            102.0,
            102.2,
            102.0,
            102.3,
            102.1,
            102.4,
            102.2,
            102.5,
            102.3,
            102.6,
            102.4,
            102.7,
            102.5,
            102.8,
            102.6,
            102.9,
            102.7,
            103.0,
            102.8,
            103.1,
            102.9,
            103.2,
            103.0,
            103.3,
            103.9,
        ]
        volumes = [10.0] * (len(closes) - 1) + [22.0]
        signal = MovingAverageStrategy(config).generate(make_variable_candles(closes, volumes), Position())

        self.assertEqual(signal.side, Side.BUY)
        self.assertIn("cost-aware edge setup", signal.reason)

    def test_weak_micro_recovery_is_rejected_without_bollinger(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_recent_momentum_pct=1.0,
            min_volume_ratio=1.0,
            min_expected_upside_pct=0.8,
            long_trend_ema_window=0,
            rsi_period=5,
        )
        candles = make_variable_candles(
            [100, 99.8, 99.6, 99.5, 99.4, 98.5, 99.0, 99.3, 99.55, 99.7],
            [10, 10, 10, 10, 10, 10, 10, 11, 12, 20],
        )

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertNotIn("micro recovery setup", signal.reason)

    def test_range_rebound_rejects_when_far_from_recent_low(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_recent_momentum_pct=0.05,
            max_recent_momentum_pct=4.0,
            min_volume_ratio=1.2,
            long_trend_ema_window=8,
            enable_range_rebound=True,
            range_rebound_lookback=8,
            range_rebound_max_distance_from_low_pct=1.0,
            range_rebound_max_ema_gap_pct=3.0,
            range_rebound_min_bounce_pct=0.1,
            range_rebound_min_volume_ratio=0.9,
            range_rebound_min_expected_upside_pct=1.0,
            range_rebound_min_rsi=30.0,
            range_rebound_max_entry_rsi=60.0,
        )
        candles = make_variable_candles(
            [120, 116, 112, 108, 104, 100, 98, 101],
            [10, 10, 10, 10, 10, 10, 10, 14],
        )

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)

    def test_calculate_ema_returns_value_when_enough_closes(self):
        self.assertIsNotNone(calculate_ema([1, 2, 3, 4, 5], 5))
        self.assertIsNone(calculate_ema([1, 2, 3], 5))

    def test_estimated_upside_caps_at_target(self):
        candles = make_candles([100, 102, 101, 104, 103, 108], volume=10.0)

        upside = estimate_expected_upside_pct(candles, target_upside_pct=3.0)

        self.assertLessEqual(upside, 3.0)
        self.assertGreater(upside, 0.0)

    def test_estimated_downside_respects_stop_floor(self):
        candles = make_candles([100, 99, 101, 98, 102, 100], volume=10.0)

        downside = estimate_expected_downside_pct(candles, stop_loss_pct=1.0, volatility_multiplier=1.1)

        self.assertGreaterEqual(downside, 1.0)

    def test_market_breadth_ratio_counts_passing_markets(self):
        bullish = make_candles([100, 101, 102, 103, 104, 105], volume=10.0)
        bearish = make_candles([105, 104, 103, 102, 101, 100], volume=10.0)

        ratio = market_breadth_ratio({"A": bullish, "B": bearish}, short_window=3, long_window=5, ema_window=0)

        self.assertEqual(ratio, 0.5)

    def test_bollinger_lower_rebound_requires_prior_touch_and_recovery(self):
        candles = []
        for index, close in enumerate([100, 100, 100, 100, 100, 94, 93, 93]):
            low = close
            if index in {5, 6}:
                low = close - 5
            candles.append(
                Candle(
                    market="KRW-BTC",
                    timestamp=datetime(2026, 4, 20, index, tzinfo=timezone.utc),
                    open=close,
                    high=close,
                    low=low,
                    close=close,
                    volume=10.0,
                )
            )

        ok, reason = bollinger_lower_rebound_quality(candles, 5, 2.0, 1.0, 2)

        self.assertTrue(ok)
        self.assertIn("bollinger lower rebound", reason)

    def test_local_ai_decision_blocks_low_confidence(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_expected_upside_pct=0.5,
            target_upside_pct=3.0,
        )
        context = DecisionContext(True, "test context", 1.0, 0.2, 1.0)
        decision = review_entry_candidate(
            Signal(Side.BUY, "test", price=100.0, confidence=0.2),
            make_candles([100, 101, 102, 103, 104, 105], volume=10.0),
            context,
            config,
            AiDecisionConfig(enabled=True, min_confidence=0.55),
        )

        self.assertEqual(decision.action, "hold")

    def test_local_ai_decision_blocks_unfavorable_risk_reward(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_expected_upside_pct=0.5,
            target_upside_pct=3.0,
        )
        context = DecisionContext(True, "test context", 1.0, 0.2, 1.0)
        decision = review_entry_candidate(
            Signal(Side.BUY, "test", price=100.0, confidence=0.9),
            make_candles([100.0, 100.2, 100.1, 100.3, 100.2, 100.25], volume=10.0),
            context,
            config,
            AiDecisionConfig(enabled=True, min_confidence=0.55),
        )

        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.grade, "C")
        self.assertTrue(any("downside" in note.lower() for note in decision.risk_notes))

    def test_upgraded_ai_blocks_weak_micro_recovery_impulse(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=0.65,
            position_fraction=0.2,
            min_expected_upside_pct=0.5,
            target_upside_pct=3.0,
        )
        context = DecisionContext(True, "neutral test context", 1.0, 0.0, 0.2, market_mode="neutral")
        candles = make_variable_candles(
            [352.0, 353.0, 354.0, 354.0, 354.0, 354.0, 354.0, 354.0, 354.0, 355.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 46.0],
        )

        decision = review_entry_candidate(
            Signal(
                Side.BUY,
                "micro recovery setup: momentum 0.57%; one-candle 0.28%; close position 0.50; volume 4.61x; expected follow-through 1.99%; RSI 50.0",
                price=355.0,
                confidence=0.68,
            ),
            candles,
            context,
            config,
            AiDecisionConfig(enabled=True, min_confidence=0.55),
        )

        self.assertEqual(decision.action, "hold")
        self.assertTrue(any("AI hard block" in note for note in decision.risk_notes))

    def test_orderbook_snapshot_metrics(self):
        snapshot = OrderbookSnapshot(
            market="KRW-BTC",
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
            best_bid_price=100.0,
            best_bid_size=5.0,
            best_ask_price=100.1,
            best_ask_size=4.0,
            total_bid_size=12.0,
            total_ask_size=10.0,
        )

        self.assertGreater(snapshot.spread_bps, 0)
        self.assertGreater(snapshot.imbalance_ratio, 1.0)

    def test_five_minute_momentum_penalty_grows_with_shortfall(self):
        mild = five_minute_momentum_penalty(-0.05, 0.0)
        worse = five_minute_momentum_penalty(-0.25, 0.0)

        self.assertGreater(mild, 0.0)
        self.assertGreater(worse, mild)
        self.assertLessEqual(worse, 0.18)

    def test_five_minute_trend_penalty_grows_with_shortfall(self):
        mild = five_minute_trend_penalty(0.002)
        worse = five_minute_trend_penalty(0.008)

        self.assertGreater(mild, 0.0)
        self.assertGreater(worse, mild)

    def test_orderbook_spread_penalty_grows_with_excess_spread(self):
        mild = orderbook_spread_penalty(20.0, 18.0)
        worse = orderbook_spread_penalty(30.0, 18.0)

        self.assertGreater(mild, 0.0)
        self.assertGreater(worse, mild)

    def test_orderbook_imbalance_penalty_grows_with_shortfall(self):
        mild = orderbook_imbalance_penalty(0.82, 0.85)
        worse = orderbook_imbalance_penalty(0.70, 0.85)

        self.assertGreater(mild, 0.0)
        self.assertGreater(worse, mild)
        self.assertLessEqual(worse, 0.2)

    def test_openai_response_json_extraction(self):
        payload = {"output": [{"content": [{"text": "{\"action\":\"hold\"}"}]}]}

        parsed = extract_openai_json(payload)

        self.assertEqual(parsed["action"], "hold")

    def test_market_context_float_parser_is_soft(self):
        self.assertEqual(maybe_float("1.25"), 1.25)
        self.assertIsNone(maybe_float("not-a-number"))

    def test_stablecoin_markets_are_excluded_from_trading_universe(self):
        self.assertIn("KRW-USDT", UpbitPublicDataSource.EXCLUDED_TRADING_MARKETS)

    def test_news_headline_scoring_detects_risk(self):
        signal = score_headlines([
            "Bitcoin rally expands as ETF inflow rises",
            "Crypto exchange hack triggers market selloff",
        ])

        self.assertEqual(signal.headline_count, 2)
        self.assertGreater(signal.risk_headline_count, 0)

    def test_local_feature_model_uses_news_features(self):
        payload = {
            "candidate": {
                "signal_reason": "pullback continuation setup",
                "signal_confidence": 0.72,
                "expected_upside_pct": 1.8,
                "expected_downside_pct": 0.6,
                "recent_volatility_pct": 0.4,
            },
            "market_context": {
                "btc_momentum_pct": 0.2,
                "news_sentiment_score": 0.4,
                "news_risk_headline_count": 0,
            },
        }

        score = score_entry_with_feature_model(payload)

        self.assertGreater(score.probability, 0.5)


class RangeReboundExitGraceTest(unittest.TestCase):
    def test_bollinger_filter_can_create_entry_candidate(self):
        app = make_test_app()
        app.config = AppConfig(
            mode=app.config.mode,
            market=app.config.market,
            poll_seconds=app.config.poll_seconds,
            starting_cash=app.config.starting_cash,
            fee_rate=app.config.fee_rate,
            slippage_bps=app.config.slippage_bps,
            strategy=StrategyConfig(
                short_window=5,
                long_window=20,
                take_profit_pct=1.0,
                stop_loss_pct=1.0,
                position_fraction=0.2,
                enable_bollinger_rebound_filter=True,
            ),
            risk=app.config.risk,
            ai_decision=app.config.ai_decision,
            paths=app.config.paths,
        )
        fallback = Signal(Side.HOLD, "weak bounce", 100.0, 0.1)

        signal = app._bollinger_rebound_entry_signal(
            make_candles([100, 99, 98, 97, 96, 97], volume=10.0),
            "15m bollinger lower rebound: distance 0.50%; 60m bollinger filter blocked",
            fallback,
        )

        self.assertEqual(signal.side, Side.BUY)
        self.assertIn("bollinger rebound setup", signal.reason)

    def test_bollinger_low_upside_keeps_original_hold_reason(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            strategy=replace(
                app.config.strategy,
                enable_bollinger_rebound_filter=True,
                bollinger_min_expected_upside_pct=1.5,
            ),
        )
        fallback = Signal(Side.HOLD, "ma alignment failed", 100.0, 0.1)

        signal = app._bollinger_rebound_entry_signal(
            make_candles([100, 99.99, 100, 99.99, 100, 100.01], volume=10.0),
            "15m bollinger lower rebound: distance 0.50%",
            fallback,
        )

        self.assertEqual(signal.side, Side.HOLD)
        self.assertIn("ma alignment failed", signal.reason)
        self.assertIn("bollinger rebound skipped", signal.reason)

    def test_trend_reward_risk_uses_strategy_specific_threshold(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            risk=replace(app.config.risk, max_expected_downside_to_upside_ratio=0.38),
            strategy=replace(app.config.strategy, min_net_edge_pct=0.0, stop_loss_pct=0.55, stop_volatility_multiplier=0.55),
        )
        signal = Signal(Side.BUY, "trend breakout setup", 105.0, 0.7)

        ok, reason = app._reward_risk_ok(make_candles([100, 101, 102, 103, 104, 105], volume=10.0), signal)

        self.assertTrue(ok, reason)

    def test_range_rebound_grace_suppresses_early_trend_break(self):
        app = make_test_app(range_rebound_trend_break_grace_ticks=2)
        app.position_entry_strategy = "range_rebound"
        app.position_entry_tick = 10
        app.broker.position = Position(qty=1.0, avg_price=100.0, peak_price=100.0)

        suppressed = app._apply_range_rebound_exit_grace(11, 99.0, Signal(Side.SELL, "trend break", 99.0, 0.6))

        self.assertEqual(suppressed.side, Side.HOLD)
        self.assertIn("range rebound grace active", suppressed.reason)

    def test_range_rebound_grace_expires_after_configured_ticks(self):
        app = make_test_app(range_rebound_trend_break_grace_ticks=2)
        app.position_entry_strategy = "range_rebound"
        app.position_entry_tick = 10
        app.broker.position = Position(qty=1.0, avg_price=100.0, peak_price=100.0)
        original = Signal(Side.SELL, "trend break", 99.0, 0.6)

        allowed = app._apply_range_rebound_exit_grace(13, 99.0, original)

        self.assertEqual(allowed, original)

    def test_trend_entries_do_not_get_rebound_grace(self):
        app = make_test_app(range_rebound_trend_break_grace_ticks=2)
        app.position_entry_strategy = "trend"
        app.position_entry_tick = 10
        app.broker.position = Position(qty=1.0, avg_price=100.0, peak_price=100.0)
        original = Signal(Side.SELL, "trend break", 99.0, 0.6)

        allowed = app._apply_range_rebound_exit_grace(11, 99.0, original)

        self.assertEqual(allowed, original)

    def test_bollinger_rebound_grace_suppresses_early_trend_break(self):
        app = make_test_app(range_rebound_trend_break_grace_ticks=2)
        app.position_entry_strategy = {"KRW-BTC": "bollinger_rebound"}
        app.position_entry_tick = {"KRW-BTC": 10}
        app.broker.position = Position(qty=1.0, avg_price=100.0, peak_price=100.0)

        suppressed = app._apply_rebound_exit_grace(11, "KRW-BTC", 99.0, Signal(Side.SELL, "trend break", 99.0, 0.6))

        self.assertEqual(suppressed.side, Side.HOLD)
        self.assertIn("bollinger rebound grace active", suppressed.reason)

    def test_validated_recovery_rejects_flat_rebound(self):
        app = make_test_app()
        candles = make_candles([100.0] * 20, volume=10.0)
        signal = Signal(Side.BUY, "bollinger rebound setup", 100.0, 0.7)

        ok, reason = app._validated_recovery_ok(candles, signal)

        self.assertFalse(ok)
        self.assertIn("validated recovery blocked", reason)

    def test_validated_recovery_accepts_confirmed_recovery(self):
        app = make_test_app()
        closes = [100, 99.8, 99.6, 99.4, 99.2, 99.0, 98.8, 98.6, 98.4, 98.2, 98.0, 98.1, 98.2, 98.3, 98.4, 98.5, 98.7, 98.9, 99.1, 99.4]
        candles = make_variable_candles(closes, [10.0] * 19 + [18.0])
        signal = Signal(Side.BUY, "bollinger rebound setup", 99.4, 0.7)

        ok, reason = app._validated_recovery_ok(candles, signal)

        self.assertTrue(ok)
        self.assertIn("validated recovery ok", reason)

    def test_bollinger_filter_failure_is_penalty_not_hard_block(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            strategy=replace(
                app.config.strategy,
                enable_bollinger_rebound_filter=True,
                bollinger_filter_penalty=0.04,
            ),
        )
        app.data_source.get_orderbook_snapshot = lambda *_args, **_kwargs: OrderbookSnapshot(
            market="KRW-BTC",
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
            best_bid_price=99.96,
            best_bid_size=1.0,
            best_ask_price=100.04,
            best_ask_size=1.0,
            total_bid_size=10.0,
            total_ask_size=9.5,
        )
        app._five_minute_trend_ok = lambda _market: (True, "5m trend ok", 0.0)
        app._multi_timeframe_bollinger_ok = lambda _market: (False, "15m bollinger filter blocked", 0.0)

        blocked, reason, penalty = app._entry_market_filters(1, "KRW-BTC", make_candles([100.0] * 20, volume=10.0))

        self.assertFalse(blocked)
        self.assertIn("bollinger filter blocked", reason)
        self.assertGreaterEqual(penalty, 0.08)

    def test_candidate_score_floor_rises_after_losses_and_drawdown(self):
        app = make_test_app()
        context = DecisionContext(
            allows_entries=True,
            reason="test",
            score_multiplier=1.0,
            btc_momentum_pct=0.0,
            btc_volatility_pct=0.1,
            market_mode="neutral",
        )
        app.config = AppConfig(
            mode=app.config.mode,
            market=app.config.market,
            poll_seconds=app.config.poll_seconds,
            starting_cash=app.config.starting_cash,
            fee_rate=app.config.fee_rate,
            slippage_bps=app.config.slippage_bps,
            strategy=app.config.strategy,
            risk=RiskConfig(
                daily_profit_target_pct=3.0,
                daily_loss_limit_pct=5.0,
                max_entries_per_day=48,
                max_position_fraction=0.42,
                max_consecutive_losses=4,
                min_candidate_score=0.62,
            ),
            ai_decision=app.config.ai_decision,
            paths=app.config.paths,
        )
        app.risk.config = app.config.risk
        app.risk.state.consecutive_losses = 2
        app.risk.state.entries_today = 13

        floor = app._candidate_score_floor(context, equity=985_000)

        self.assertGreater(floor, 0.89)

    def test_drawdown_and_losses_reduce_position_fraction(self):
        app = make_test_app()
        context = DecisionContext(
            allows_entries=True,
            reason="test",
            score_multiplier=1.0,
            btc_momentum_pct=0.0,
            btc_volatility_pct=0.1,
            market_mode="neutral",
        )
        candles = make_candles([100, 101, 102, 103, 104, 105], volume=10.0)
        app.risk.state.consecutive_losses = 2

        reduced = app._position_fraction_for_context(candles, context, equity=985_000)
        normal = app._position_fraction_for_context(candles, context, equity=1_000_000)

        self.assertLess(reduced, normal)

    def test_bollinger_failure_penalty_does_not_tax_trend_entries(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            strategy=replace(
                app.config.strategy,
                enable_bollinger_rebound_filter=True,
                bollinger_filter_penalty=0.04,
            ),
        )
        signal = Signal(Side.BUY, "trend breakout setup; momentum 0.8%", 100.0, 0.7)

        penalty = app._entry_filter_penalty_for_signal(
            signal,
            "15m bollinger filter blocked: not near lower band",
            0.08,
        )

        self.assertEqual(penalty, 0.0)

    def test_strong_candidate_gets_smaller_breadth_penalty(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            risk=replace(app.config.risk, min_candidate_score=0.44),
        )
        signal = Signal(Side.BUY, "uptrend filter passed", 100.0, 0.74)

        penalty = app._breadth_penalty_for_candidate(0.62, signal, 0.10)

        self.assertLess(penalty, 0.10)

    def test_breakeven_stop_uses_peak_profit(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            strategy=replace(app.config.strategy, breakeven_trigger_pct=0.6),
        )
        position = Position(qty=1.0, avg_price=100.0, peak_price=101.0)
        candles = make_candles([100.5, 101.0, 100.2, 100.05, 100.01], volume=10.0)

        signal = app._position_management_signal(1, "KRW-BTC", candles, 100.01, position)

        self.assertEqual(signal.side, Side.SELL)
        self.assertIn("breakeven stop reached", signal.reason)

    def test_full_take_profit_exits_before_partial(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            strategy=replace(app.config.strategy, target_upside_pct=3.0, partial_take_profit_pct=1.0),
        )
        position = Position(qty=1.0, avg_price=100.0, peak_price=103.8)
        candles = make_candles([100.0, 101.0, 102.0, 103.0, 103.8], volume=10.0)

        signal = app._position_management_signal(1, "KRW-BTC", candles, 103.8, position)

        self.assertEqual(signal.side, Side.SELL)
        self.assertEqual(1.0, signal.size_fraction)
        self.assertIn("full take profit reached", signal.reason)

    def test_post_partial_profit_floor_protects_remaining_position(self):
        app = make_test_app()
        position = Position(qty=1.0, avg_price=100.0, peak_price=103.0, partial_exit_taken=True)
        candles = make_candles([100.0, 102.0, 103.0, 101.0, 100.3], volume=10.0)

        signal = app._position_management_signal(1, "KRW-BTC", candles, 100.3, position)

        self.assertEqual(signal.side, Side.SELL)
        self.assertIn("post-partial profit floor", signal.reason)

    def test_recent_negative_expectancy_blocks_new_entries(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            risk=replace(
                app.config.risk,
                recent_exit_sample_size=3,
                min_recent_expectancy_krw=0.0,
                min_recent_profit_factor=1.05,
                max_recent_loss_rate=0.70,
            ),
        )
        with TemporaryDirectory() as tmp:
            trade_path = Path(tmp) / "trades.csv"
            trade_path.write_text(
                "\n".join(
                    [
                        "timestamp,market,side,price,qty,fee,cash_after,position_qty_after,realized_pnl,reason",
                        "2026-05-25T00:00:00+00:00,KRW-BTC,sell,100,1,1,999000,0,-1000,stop",
                        "2026-05-25T00:01:00+00:00,KRW-ETH,sell,100,1,1,998000,0,-1000,stop",
                        "2026-05-25T00:02:00+00:00,KRW-XRP,sell,100,1,1,999500,0,500,target",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            app.config = replace(app.config, paths=replace(app.config.paths, trade_journal=trade_path))

            ok, reason = app._recent_performance_allows_entries()

        self.assertFalse(ok)
        self.assertIn("recent performance gate", reason)

    def test_recent_stopout_cluster_blocks_market_entries(self):
        app = make_test_app()
        app.market_stopout_ticks = {
            "KRW-A": [10],
            "KRW-B": [11],
            "KRW-C": [12],
        }

        ok, reason = app._recent_market_stopouts_allow_entries(20)

        self.assertFalse(ok)
        self.assertIn("market stopout cluster blocked", reason)

    def test_trend_breakout_requires_strong_close_position(self):
        strategy = MovingAverageStrategy(
            StrategyConfig(
                short_window=3,
                long_window=5,
                take_profit_pct=1.0,
                stop_loss_pct=1.0,
                position_fraction=0.2,
                min_recent_momentum_pct=0.1,
                min_volume_ratio=1.0,
                min_expected_upside_pct=0.1,
            )
        )
        candles = make_variable_candles(
            [100, 100.2, 100.4, 100.8, 101.0, 101.4, 101.8],
            [10, 10, 10, 10, 10, 10, 20],
        )
        weak_latest = replace(candles[-1], high=104.0, low=100.0, close=101.8)
        candles = [*candles[:-1], weak_latest]

        signal = strategy.generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertIn("weak breakout close position", signal.reason)

    def test_strategy_performance_gate_blocks_losing_strategy(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            risk=replace(
                app.config.risk,
                strategy_exit_sample_size=3,
                min_strategy_expectancy_krw=0.0,
                max_strategy_loss_rate=0.8,
            ),
        )
        with TemporaryDirectory() as tmp:
            trade_path = Path(tmp) / "trades.csv"
            trade_path.write_text(
                "\n".join(
                    [
                        "timestamp,market,side,price,qty,fee,cash_after,position_qty_after,realized_pnl,reason",
                        "2026-05-25T00:00:00+00:00,KRW-A,buy,100,1,1,900000,1,0,trend breakout setup: momentum 1%",
                        "2026-05-25T00:01:00+00:00,KRW-A,sell,99,1,1,999000,0,-1000,stop",
                        "2026-05-25T00:02:00+00:00,KRW-B,buy,100,1,1,900000,1,0,trend breakout setup: momentum 1%",
                        "2026-05-25T00:03:00+00:00,KRW-B,sell,99,1,1,999000,0,-1000,stop",
                        "2026-05-25T00:04:00+00:00,KRW-C,buy,100,1,1,900000,1,0,trend breakout setup: momentum 1%",
                        "2026-05-25T00:05:00+00:00,KRW-C,sell,101,1,1,1000500,0,500,target",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            app.config = replace(app.config, paths=replace(app.config.paths, trade_journal=trade_path))
            app.performance_gate = app.performance_gate.__class__(trade_path)

            ok, reason = app._strategy_performance_allows_entry("trend")

        self.assertFalse(ok)
        self.assertIn("strategy disabled", reason)

    def test_reason_bucket_gate_blocks_repeated_losing_setup(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            risk=replace(
                app.config.risk,
                reason_exit_sample_size=2,
                min_reason_expectancy_krw=0.0,
                max_reason_loss_rate=0.8,
            ),
        )
        with TemporaryDirectory() as tmp:
            trade_path = Path(tmp) / "trades.csv"
            trade_path.write_text(
                "\n".join(
                    [
                        "timestamp,market,side,price,qty,fee,cash_after,position_qty_after,realized_pnl,reason",
                        "2026-05-25T00:00:00+00:00,KRW-A,buy,100,1,1,900000,1,0,chart ai setup: pullback reclaim",
                        "2026-05-25T00:01:00+00:00,KRW-A,sell,99,1,1,999000,0,-1000,stop",
                        "2026-05-25T00:02:00+00:00,KRW-B,buy,100,1,1,900000,1,0,chart ai setup: pullback reclaim",
                        "2026-05-25T00:03:00+00:00,KRW-B,sell,98,1,1,998000,0,-2000,stop",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            app.performance_gate = app.performance_gate.__class__(trade_path)

            ok, reason = app._reason_bucket_performance_allows_entry(
                Signal(Side.BUY, "chart ai setup: pullback reclaim", 100.0, 0.7)
            )

        self.assertFalse(ok)
        self.assertIn("reason disabled: chart_ai_pullback_reclaim", reason)

    def test_reason_bucket_position_multiplier_reduces_losing_setup(self):
        app = make_test_app()
        app.config = replace(app.config, risk=replace(app.config.risk, reason_exit_sample_size=3))
        with TemporaryDirectory() as tmp:
            trade_path = Path(tmp) / "trades.csv"
            trade_path.write_text(
                "\n".join(
                    [
                        "timestamp,market,side,price,qty,fee,cash_after,position_qty_after,realized_pnl,reason",
                        "2026-05-25T00:00:00+00:00,KRW-A,buy,100,1,1,900000,1,0,micro recovery setup",
                        "2026-05-25T00:01:00+00:00,KRW-A,sell,99,1,1,999000,0,-1000,stop",
                        "2026-05-25T00:02:00+00:00,KRW-B,buy,100,1,1,900000,1,0,micro recovery setup",
                        "2026-05-25T00:03:00+00:00,KRW-B,sell,99,1,1,999000,0,-1000,stop",
                        "2026-05-25T00:04:00+00:00,KRW-C,buy,100,1,1,900000,1,0,micro recovery setup",
                        "2026-05-25T00:05:00+00:00,KRW-C,sell,101,1,1,1000200,0,200,target",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            app.performance_gate = app.performance_gate.__class__(trade_path)

            multiplier = app._reason_bucket_position_multiplier(Signal(Side.BUY, "micro recovery setup", 100.0, 0.7))

        self.assertLess(multiplier, 1.0)

    def test_market_mode_allows_micro_recovery_in_neutral(self):
        app = make_test_app()
        context = DecisionContext(
            allows_entries=True,
            reason="test",
            score_multiplier=1.0,
            btc_momentum_pct=0.0,
            btc_volatility_pct=0.1,
            market_mode="neutral",
        )

        ok, reason = app._market_mode_allows_strategy("micro_recovery", context)

        self.assertTrue(ok)
        self.assertEqual("market mode ok", reason)

    def test_performance_position_multiplier_reduces_weak_market(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            risk=replace(app.config.risk, market_exit_sample_size=3, strategy_exit_sample_size=3),
        )
        with TemporaryDirectory() as tmp:
            trade_path = Path(tmp) / "trades.csv"
            trade_path.write_text(
                "\n".join(
                    [
                        "timestamp,market,side,price,qty,fee,cash_after,position_qty_after,realized_pnl,reason",
                        "2026-05-25T00:00:00+00:00,KRW-BAD,buy,100,1,1,900000,1,0,trend breakout setup",
                        "2026-05-25T00:01:00+00:00,KRW-BAD,sell,99,1,1,999000,0,-1000,stop",
                        "2026-05-25T00:02:00+00:00,KRW-BAD,buy,100,1,1,900000,1,0,trend breakout setup",
                        "2026-05-25T00:03:00+00:00,KRW-BAD,sell,99,1,1,999000,0,-1000,stop",
                        "2026-05-25T00:04:00+00:00,KRW-BAD,buy,100,1,1,900000,1,0,trend breakout setup",
                        "2026-05-25T00:05:00+00:00,KRW-BAD,sell,101,1,1,1000200,0,200,target",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            app.performance_gate = app.performance_gate.__class__(trade_path)

            multiplier = app._performance_position_multiplier("KRW-BAD", "trend")

        self.assertLess(multiplier, 1.0)

    def test_high_conviction_candidate_gets_larger_position_fraction(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            risk=replace(app.config.risk, min_candidate_score=0.50),
        )
        context = DecisionContext(
            allows_entries=True,
            reason="test",
            score_multiplier=1.0,
            btc_momentum_pct=0.0,
            btc_volatility_pct=0.1,
            market_mode="risk_on",
        )
        candles = make_candles([100, 101, 102, 103, 104, 105], volume=10.0)
        signal = Signal(Side.BUY, "trend breakout setup", 105.0, 0.78)

        normal = app._position_fraction_for_context(candles, context, equity=1_000_000, signal=signal, market="KRW-BTC", score=0.51)
        high = app._position_fraction_for_context(candles, context, equity=1_000_000, signal=signal, market="KRW-BTC", score=0.72)

        self.assertGreater(high, normal)

    def test_strategy_name_from_reason_and_performance_stats(self):
        self.assertEqual("cost_aware_edge", strategy_name_from_reason("cost-aware edge setup: breakout"))
        self.assertEqual("cost_edge_breakout", reason_bucket_from_reason("cost-aware edge setup: breakout"))
        self.assertEqual("pullback", strategy_name_from_reason("pullback continuation setup: trend 0.3%"))
        self.assertEqual("chart_ai_momentum_ignition", reason_bucket_from_reason("chart ai setup: momentum ignition"))
        expectancy, profit_factor, loss_rate = performance_stats([1000.0, -500.0, 500.0])
        self.assertGreater(expectancy, 0.0)
        self.assertEqual(1 / 3, loss_rate)
        self.assertGreater(profit_factor, 1.0)

    def test_chart_ai_signal_can_create_candidate_from_technical_features(self):
        strategy = MovingAverageStrategy(
            StrategyConfig(
                short_window=5,
                long_window=20,
                take_profit_pct=1.0,
                stop_loss_pct=1.0,
                position_fraction=0.2,
                min_recent_momentum_pct=2.0,
                min_volume_ratio=2.0,
                min_expected_upside_pct=0.8,
                max_entry_rsi=78.0,
                min_validated_recovery_pct=2.0,
            )
        )
        closes = [
            100.0,
            99.7,
            99.4,
            99.6,
            99.2,
            99.5,
            99.1,
            99.4,
            99.0,
            99.3,
            99.1,
            99.6,
            99.3,
            99.9,
            99.5,
            100.1,
            99.8,
            100.4,
            100.0,
            100.7,
            100.3,
            101.0,
        ]
        volumes = [10.0] * (len(closes) - 1) + [24.0]
        candles = make_variable_candles(closes, volumes)

        closes_for_ma = [candle.close for candle in candles]
        signal = strategy._chart_ai_signal(  # noqa: SLF001 - this verifies the chart model path directly.
            candles,
            sum(closes_for_ma[-strategy.config.short_window :]) / strategy.config.short_window,
            sum(closes_for_ma[-strategy.config.long_window :]) / strategy.config.long_window,
        )

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.side, Side.BUY)
        self.assertIn("chart ai setup", signal.reason)

    def test_chart_quality_feature_boosts_local_model(self):
        payload = {
            "candidate": {
                "signal_confidence": 0.56,
                "expected_upside_pct": 1.4,
                "expected_downside_pct": 0.7,
                "recent_volatility_pct": 0.4,
                "chart_quality_score": 0.8,
                "signal_reason": "chart ai setup: momentum ignition",
            },
            "market_context": {
                "btc_momentum_pct": 0.0,
                "news_sentiment_score": 0.0,
                "news_risk_headline_count": 0,
            },
        }

        score = score_entry_with_feature_model(payload)

        self.assertGreater(score.probability, 0.6)
        self.assertIn("chart_quality_score", score.features)

    def test_opportunity_edge_boosts_profit_seeking_setups(self):
        weak = opportunity_edge_score(
            expected_upside_pct=1.8,
            reward_risk_ratio=1.9,
            impulse_quality=0.45,
            momentum_3_pct=0.25,
            momentum_8_pct=0.35,
            volume_ratio=1.4,
            close_position=0.62,
            rsi=55.0,
        )
        strong = opportunity_edge_score(
            expected_upside_pct=3.2,
            reward_risk_ratio=3.4,
            impulse_quality=0.82,
            momentum_3_pct=0.75,
            momentum_8_pct=1.15,
            volume_ratio=3.4,
            close_position=0.92,
            rsi=57.0,
        )

        self.assertLess(weak, 0.45)
        self.assertGreater(strong, 0.72)

    def test_candidate_score_rewards_high_edge_setup(self):
        config = StrategyConfig(short_window=5, long_window=20, take_profit_pct=1.0, stop_loss_pct=0.65, position_fraction=0.2)
        strong_candles = make_variable_candles(
            [
                100.0,
                100.1,
                100.0,
                100.2,
                100.5,
                100.9,
                101.4,
                101.9,
                102.5,
                103.0,
                103.5,
                104.0,
                104.6,
                105.2,
                105.8,
                106.4,
                107.0,
                107.7,
                108.4,
                109.2,
            ],
            [10.0] * 17 + [32.0, 36.0, 42.0],
        )
        weak_candles = make_variable_candles(
            [
                100.0,
                100.2,
                100.1,
                100.2,
                100.3,
                100.2,
                100.4,
                100.3,
                100.4,
                100.5,
                100.4,
                100.5,
                100.6,
                100.5,
                100.6,
                100.7,
                100.6,
                100.7,
                100.8,
                100.7,
            ],
            [10.0] * 20,
        )
        strong_signal = Signal(Side.BUY, "trend breakout setup: momentum 0.75%; volume 3.40x; expected follow-through 3.20%", 0.76)
        weak_signal = Signal(Side.BUY, "trend breakout setup: momentum 0.25%; volume 1.10x; expected follow-through 1.80%", 0.76)

        self.assertGreater(
            opportunity_score_for_signal(strong_candles, strong_signal, config),
            opportunity_score_for_signal(weak_candles, weak_signal, config),
        )


def make_candles(closes: list[float], volume: float) -> list[Candle]:
    base_time = datetime(2026, 4, 20, tzinfo=timezone.utc)
    return [
        Candle(
            market="KRW-BTC",
            timestamp=base_time + timedelta(minutes=index),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=volume,
        )
        for index, close in enumerate(closes)
    ]


def make_variable_candles(closes: list[float], volumes: list[float]) -> list[Candle]:
    base_time = datetime(2026, 4, 20, tzinfo=timezone.utc)
    return [
        Candle(
            market="KRW-BTC",
            timestamp=base_time + timedelta(minutes=index),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=volumes[index],
        )
        for index, close in enumerate(closes)
    ]


def make_intrabar_candles(open_price: float, high: float, low: float, close: float) -> list[Candle]:
    base = make_candles([100.0] * 19, volume=10.0)
    return base + [
        Candle(
            market="KRW-BTC",
            timestamp=datetime(2026, 4, 20, 19, tzinfo=timezone.utc),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=10.0,
        )
    ]


class RegimeEnsembleGateTest(unittest.TestCase):
    def test_regime_reason_is_classified_by_setup(self):
        app = make_test_app()

        self.assertEqual(
            "regime_reversal",
            app._entry_strategy_name(Signal(Side.BUY, "regime ensemble setup: lcl reclaim reversal", 100.0)),
        )
        self.assertEqual(
            "regime_breakout",
            app._entry_strategy_name(Signal(Side.BUY, "regime ensemble setup: ucl volatility breakout", 100.0)),
        )
        self.assertEqual(
            "regime_trend",
            app._entry_strategy_name(Signal(Side.BUY, "regime ensemble setup: ema pullback reclaim", 100.0)),
        )

    def test_neutral_mode_allows_reversal_but_rejects_trend_chasing(self):
        app = make_test_app()
        context = DecisionContext(
            allows_entries=True,
            reason="test",
            score_multiplier=1.0,
            btc_momentum_pct=0.0,
            btc_volatility_pct=0.1,
            market_mode="neutral",
        )

        reversal_ok, _ = app._market_mode_allows_strategy("regime_reversal", context)
        breakout_ok, _ = app._market_mode_allows_strategy("regime_breakout", context)
        trend_ok, _ = app._market_mode_allows_strategy("regime_trend", context)

        self.assertTrue(reversal_ok)
        self.assertFalse(breakout_ok)
        self.assertFalse(trend_ok)

    def test_risk_off_rejects_reversal(self):
        app = make_test_app()
        context = DecisionContext(
            allows_entries=True,
            reason="test",
            score_multiplier=0.82,
            btc_momentum_pct=-0.2,
            btc_volatility_pct=0.5,
            market_mode="risk_off",
        )

        allowed, reason = app._market_mode_allows_strategy("regime_reversal", context)

        self.assertFalse(allowed)
        self.assertIn("risk_off", reason)

    def test_daily_participation_allows_risk_off_quota_entry(self):
        app = make_test_app()
        context = DecisionContext(
            allows_entries=True,
            reason="test",
            score_multiplier=0.82,
            btc_momentum_pct=-0.2,
            btc_volatility_pct=0.5,
            market_mode="risk_off",
        )

        allowed, reason = app._market_mode_allows_strategy("daily_participation", context)

        self.assertTrue(allowed)
        self.assertEqual("market mode ok", reason)

    def test_daily_participation_candidate_respects_quality_until_min_entries_met(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            strategy=replace(
                app.config.strategy,
                stop_loss_pct=0.10,
                stop_volatility_multiplier=0.55,
                min_price_krw=300.0,
                max_orderbook_spread_bps=24.0,
            ),
            risk=replace(app.config.risk, min_entries_per_day=4, min_candidate_score=0.44),
        )
        closes = [
            1000,
            1004,
            1001,
            1005,
            1002,
            1006,
            1003,
            1007,
            1004,
            1008,
            1005,
            1009,
            1006,
            1010,
            1007,
            1011,
            1008,
            1012,
            1009,
            1015,
        ]
        candles = make_variable_candles(closes, [10.0] * len(closes))
        app.data_source.get_orderbook_snapshot = lambda *_args, **_kwargs: OrderbookSnapshot(
            market="KRW-TEST",
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
            best_bid_price=1014.8,
            best_bid_size=1.0,
            best_ask_price=1015.2,
            best_ask_size=1.0,
            total_bid_size=10.0,
            total_ask_size=10.0,
        )
        app.data_source.get_recent_candles = lambda *_args, **_kwargs: candles
        context = DecisionContext(
            allows_entries=True,
            reason="test",
            score_multiplier=1.0,
            btc_momentum_pct=0.0,
            btc_volatility_pct=0.1,
            market_mode="neutral",
        )

        candidate = app._daily_participation_candidate(1, {"KRW-TEST": candles}, context, 0.0)

        self.assertIsNone(candidate)

        app.risk.state.entries_today = 4
        self.assertIsNone(app._daily_participation_candidate(2, {"KRW-TEST": candles}, context, 0.0))

    def test_intrabar_exit_uses_candle_low_for_stop(self):
        app = make_test_app()
        app.config = replace(app.config, strategy=replace(app.config.strategy, stop_loss_pct=0.5))
        candles = make_intrabar_candles(open_price=100.0, high=100.2, low=99.4, close=99.8)
        position = Position(qty=10.0, avg_price=100.0, peak_price=100.0)

        signal = app._intrabar_position_signal(candles, position)

        self.assertEqual(signal.side, Side.SELL)
        self.assertIn("intrabar stop loss", signal.reason)
        self.assertAlmostEqual(signal.price, 99.5)

    def test_intrabar_exit_uses_candle_high_for_partial_profit(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            strategy=replace(
                app.config.strategy,
                partial_take_profit_pct=0.4,
                partial_take_profit_fraction=0.7,
                take_profit_pct=1.5,
            ),
        )
        candles = make_intrabar_candles(open_price=100.0, high=100.5, low=99.9, close=100.1)
        position = Position(qty=10.0, avg_price=100.0, peak_price=100.0)

        signal = app._intrabar_position_signal(candles, position)

        self.assertEqual(signal.side, Side.SELL)
        self.assertIn("intrabar partial take profit", signal.reason)
        self.assertAlmostEqual(signal.price, 100.4)
        self.assertAlmostEqual(signal.size_fraction, 0.7)

    def test_daily_participation_uses_lower_cash_floor_in_risk_off(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            risk=replace(
                app.config.risk,
                min_trade_cash_krw=680_000,
                defensive_min_trade_cash_krw=260_000,
                panic_min_trade_cash_krw=180_000,
            ),
        )
        signal = Signal(Side.BUY, "daily participation setup: quota 1/4", 100.0)
        risk_off = DecisionContext(
            allows_entries=True,
            reason="weak market",
            score_multiplier=0.82,
            btc_momentum_pct=-0.4,
            btc_volatility_pct=0.5,
            market_mode="risk_off",
            global_market_cap_change_pct=-2.8,
            binance_btcusdt_change_pct=-2.1,
        )
        panic = replace(risk_off, market_mode="panic_rebound", global_market_cap_change_pct=-4.2)

        self.assertEqual(app._min_trade_cash_for_context(risk_off, signal), 260_000)
        self.assertEqual(app._min_trade_cash_for_context(panic, signal), 180_000)
        self.assertEqual(app._min_trade_cash_for_context(risk_off, Signal(Side.BUY, "trend breakout setup", 100.0)), 680_000)

    def test_risk_off_daily_participation_reduces_position_fraction(self):
        app = make_test_app()
        app.config = replace(
            app.config,
            strategy=replace(app.config.strategy, position_fraction=0.78, min_volatility_position_fraction=0.82),
            risk=replace(
                app.config.risk,
                max_position_fraction=0.82,
                min_trade_cash_krw=680_000,
                defensive_min_trade_cash_krw=260_000,
                panic_min_trade_cash_krw=180_000,
            ),
        )
        candles = make_variable_candles([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], [10.0] * 10)
        neutral = DecisionContext(
            allows_entries=True,
            reason="neutral",
            score_multiplier=1.0,
            btc_momentum_pct=0.0,
            btc_volatility_pct=0.2,
            market_mode="neutral",
        )
        risk_off = replace(
            neutral,
            market_mode="risk_off",
            position_fraction_multiplier=0.7,
            btc_momentum_pct=-0.45,
            global_market_cap_change_pct=-3.0,
            binance_btcusdt_change_pct=-2.2,
        )
        signal = Signal(Side.BUY, "daily participation setup: quota 1/4", candles[-1].close, 0.6)

        neutral_fraction = app._position_fraction_for_context(candles, neutral, equity=1_000_000, signal=signal, score=0.55)
        risk_off_fraction = app._position_fraction_for_context(candles, risk_off, equity=1_000_000, signal=signal, score=0.55)

        self.assertLess(risk_off_fraction, neutral_fraction * 0.55)


class DummyDataSource:
    def get_recent_candles(self, *args, **kwargs):
        raise NotImplementedError

    def get_orderbook_snapshot(self, *args, **kwargs):
        raise NotImplementedError

    def get_top_krw_markets(self, *args, **kwargs):
        return []


def make_test_app(range_rebound_trend_break_grace_ticks: int = 2) -> MultiMarketTradingApp:
    config = AppConfig(
        mode="paper",
        market="KRW-BTC",
        poll_seconds=0,
        starting_cash=1_000_000.0,
        fee_rate=0.0005,
        slippage_bps=5.0,
        strategy=StrategyConfig(
            short_window=5,
            long_window=20,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            range_rebound_trend_break_grace_ticks=range_rebound_trend_break_grace_ticks,
        ),
        risk=RiskConfig(
            daily_profit_target_pct=3.0,
            daily_loss_limit_pct=5.0,
            max_entries_per_day=12,
            max_position_fraction=0.35,
            max_consecutive_losses=4,
        ),
        ai_decision=AiDecisionConfig(enabled=False),
        paths=PathConfig(
            trade_journal=Path("data/test_trades.csv"),
            event_log=Path("logs/test_events.jsonl"),
            state_file=Path("data/test_state.json"),
        ),
    )
    return MultiMarketTradingApp(config, DummyDataSource(), ["KRW-BTC"], request_delay=0.0)


if __name__ == "__main__":
    unittest.main()
