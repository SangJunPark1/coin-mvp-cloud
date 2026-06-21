from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StrategyConfig:
    short_window: int
    long_window: int
    take_profit_pct: float
    stop_loss_pct: float
    position_fraction: float
    min_recent_momentum_pct: float = 0.05
    max_recent_momentum_pct: float = 4.0
    min_volume_ratio: float = 1.05
    max_ma_distance_pct: float = 6.0
    rsi_period: int = 14
    max_entry_rsi: float = 72.0
    target_recent_volatility_pct: float = 1.5
    min_volatility_position_fraction: float = 0.4
    long_trend_ema_window: int = 200
    time_stop_ticks: int = 12
    time_stop_min_pnl_pct: float = 0.0
    btc_short_window: int = 5
    btc_long_window: int = 20
    min_btc_momentum_pct: float = -0.7
    min_expected_upside_pct: float = 0.8
    min_validated_recovery_pct: float = 0.08
    min_net_edge_pct: float = 0.35
    target_upside_pct: float = 3.0
    blocked_entry_hours_kst: tuple[int, ...] = ()
    reentry_cooldown_ticks: int = 6
    stop_volatility_multiplier: float = 1.1
    breakeven_trigger_pct: float = 0.8
    partial_take_profit_pct: float = 1.0
    partial_take_profit_fraction: float = 0.5
    trailing_stop_pct: float = 0.8
    min_orderbook_imbalance: float = 0.95
    max_orderbook_spread_bps: float = 12.0
    min_market_breadth_ratio: float = 0.35
    min_price_krw: float = 100.0
    max_recent_stopouts_per_market: int = 2
    stopout_lookback_ticks: int = 144
    five_minute_short_window: int = 3
    five_minute_long_window: int = 6
    min_five_minute_momentum_pct: float = 0.0
    five_minute_trend_tolerance_pct: float = 0.2
    enable_range_rebound: bool = True
    range_rebound_lookback: int = 12
    range_rebound_max_distance_from_low_pct: float = 3.0
    range_rebound_max_ema_gap_pct: float = 3.5
    range_rebound_min_bounce_pct: float = 0.15
    range_rebound_min_volume_ratio: float = 0.9
    range_rebound_min_expected_upside_pct: float = 1.0
    range_rebound_min_rsi: float = 35.0
    range_rebound_max_entry_rsi: float = 55.0
    range_rebound_trend_break_grace_ticks: int = 2
    bollinger_trend_break_grace_ticks: int = 2
    enable_bollinger_rebound_filter: bool = False
    bollinger_window: int = 20
    bollinger_stddev: float = 2.0
    bollinger_touch_tolerance_pct: float = 0.6
    bollinger_prior_touch_lookback: int = 2
    bollinger_filter_penalty: float = 0.08
    bollinger_min_confirmations: int = 2
    bollinger_min_expected_upside_pct: float = 0.3
    enable_crash_candle_filter: bool = True
    crash_candle_lookback: int = 3
    crash_candle_body_pct: float = 1.2
    crash_candle_volume_ratio: float = 1.4
    crash_candle_break_lookback: int = 12
    regime_ensemble_only: bool = False


@dataclass(frozen=True)
class RiskConfig:
    daily_profit_target_pct: float
    daily_loss_limit_pct: float
    max_entries_per_day: int
    max_position_fraction: float
    max_consecutive_losses: int
    min_entries_per_day: int = 0
    new_entries_enabled: bool = True
    min_equity_krw: float = 0.0
    halt_cooldown_ticks: int = 6
    consecutive_loss_cooldown_ticks: int = 12
    max_expected_downside_to_upside_ratio: float = 1.0
    max_open_positions: int = 4
    max_total_position_fraction: float = 0.95
    max_new_entries_per_tick: int = 1
    min_trade_cash_krw: float = 0.0
    min_candidate_score: float = 0.0
    recent_exit_sample_size: int = 20
    min_recent_expectancy_krw: float = 0.0
    min_recent_profit_factor: float = 1.05
    max_recent_loss_rate: float = 0.58
    strategy_exit_sample_size: int = 6
    market_exit_sample_size: int = 4
    min_strategy_expectancy_krw: float = 0.0
    min_market_expectancy_krw: float = 0.0
    max_strategy_loss_rate: float = 0.67
    max_market_loss_rate: float = 0.75
    reason_exit_sample_size: int = 3
    min_reason_expectancy_krw: float = 0.0
    max_reason_loss_rate: float = 0.67
    adaptive_position_sizing: bool = True


@dataclass(frozen=True)
class AiDecisionConfig:
    enabled: bool = True
    provider: str = "local"
    min_confidence: float = 0.55
    openai_model: str = "gpt-5.4-mini"
    api_key_env: str = "OPENAI_API_KEY"


@dataclass(frozen=True)
class PathConfig:
    trade_journal: Path
    event_log: Path
    state_file: Path


@dataclass(frozen=True)
class AppConfig:
    mode: str
    market: str
    poll_seconds: int
    starting_cash: float
    fee_rate: float
    slippage_bps: float
    strategy: StrategyConfig
    risk: RiskConfig
    ai_decision: AiDecisionConfig
    paths: PathConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent

    strategy = _require_mapping(raw, "strategy")
    risk = _require_mapping(raw, "risk")
    ai_decision = raw.get("ai_decision", {})
    if not isinstance(ai_decision, dict):
        raise ValueError("ai_decision must be an object when provided.")
    paths = _require_mapping(raw, "paths")

    app = AppConfig(
        mode=str(raw.get("mode", "paper")).lower(),
        market=str(raw["market"]),
        poll_seconds=int(raw.get("poll_seconds", 15)),
        starting_cash=float(raw["starting_cash"]),
        fee_rate=float(raw.get("fee_rate", 0.0005)),
        slippage_bps=float(raw.get("slippage_bps", 5)),
        strategy=StrategyConfig(
            short_window=int(strategy["short_window"]),
            long_window=int(strategy["long_window"]),
            take_profit_pct=float(strategy["take_profit_pct"]),
            stop_loss_pct=float(strategy["stop_loss_pct"]),
            position_fraction=float(strategy["position_fraction"]),
            min_recent_momentum_pct=float(strategy.get("min_recent_momentum_pct", 0.05)),
            max_recent_momentum_pct=float(strategy.get("max_recent_momentum_pct", 4.0)),
            min_volume_ratio=float(strategy.get("min_volume_ratio", 1.05)),
            max_ma_distance_pct=float(strategy.get("max_ma_distance_pct", 6.0)),
            rsi_period=int(strategy.get("rsi_period", 14)),
            max_entry_rsi=float(strategy.get("max_entry_rsi", 72.0)),
            target_recent_volatility_pct=float(strategy.get("target_recent_volatility_pct", 1.5)),
            min_volatility_position_fraction=float(strategy.get("min_volatility_position_fraction", 0.4)),
            long_trend_ema_window=int(strategy.get("long_trend_ema_window", 200)),
            time_stop_ticks=int(strategy.get("time_stop_ticks", 12)),
            time_stop_min_pnl_pct=float(strategy.get("time_stop_min_pnl_pct", 0.0)),
            btc_short_window=int(strategy.get("btc_short_window", strategy.get("short_window", 5))),
            btc_long_window=int(strategy.get("btc_long_window", strategy.get("long_window", 20))),
            min_btc_momentum_pct=float(strategy.get("min_btc_momentum_pct", -0.7)),
            min_expected_upside_pct=float(strategy.get("min_expected_upside_pct", 0.8)),
            min_validated_recovery_pct=float(strategy.get("min_validated_recovery_pct", 0.08)),
            min_net_edge_pct=float(strategy.get("min_net_edge_pct", 0.35)),
            target_upside_pct=float(strategy.get("target_upside_pct", risk.get("daily_profit_target_pct", 3.0))),
            blocked_entry_hours_kst=tuple(int(value) for value in strategy.get("blocked_entry_hours_kst", [])),
            reentry_cooldown_ticks=int(strategy.get("reentry_cooldown_ticks", 6)),
            stop_volatility_multiplier=float(strategy.get("stop_volatility_multiplier", 1.1)),
            breakeven_trigger_pct=float(strategy.get("breakeven_trigger_pct", 0.8)),
            partial_take_profit_pct=float(strategy.get("partial_take_profit_pct", 1.0)),
            partial_take_profit_fraction=float(strategy.get("partial_take_profit_fraction", 0.5)),
            trailing_stop_pct=float(strategy.get("trailing_stop_pct", 0.8)),
            min_orderbook_imbalance=float(strategy.get("min_orderbook_imbalance", 0.95)),
            max_orderbook_spread_bps=float(strategy.get("max_orderbook_spread_bps", 12.0)),
            min_market_breadth_ratio=float(strategy.get("min_market_breadth_ratio", 0.35)),
            min_price_krw=float(strategy.get("min_price_krw", 100.0)),
            max_recent_stopouts_per_market=int(strategy.get("max_recent_stopouts_per_market", 2)),
            stopout_lookback_ticks=int(strategy.get("stopout_lookback_ticks", 144)),
            five_minute_short_window=int(strategy.get("five_minute_short_window", 3)),
            five_minute_long_window=int(strategy.get("five_minute_long_window", 6)),
            min_five_minute_momentum_pct=float(strategy.get("min_five_minute_momentum_pct", 0.0)),
            five_minute_trend_tolerance_pct=float(strategy.get("five_minute_trend_tolerance_pct", 0.2)),
            enable_range_rebound=bool(strategy.get("enable_range_rebound", True)),
            range_rebound_lookback=int(strategy.get("range_rebound_lookback", 12)),
            range_rebound_max_distance_from_low_pct=float(strategy.get("range_rebound_max_distance_from_low_pct", 3.0)),
            range_rebound_max_ema_gap_pct=float(strategy.get("range_rebound_max_ema_gap_pct", 3.5)),
            range_rebound_min_bounce_pct=float(strategy.get("range_rebound_min_bounce_pct", 0.15)),
            range_rebound_min_volume_ratio=float(strategy.get("range_rebound_min_volume_ratio", 0.9)),
            range_rebound_min_expected_upside_pct=float(strategy.get("range_rebound_min_expected_upside_pct", 1.0)),
            range_rebound_min_rsi=float(strategy.get("range_rebound_min_rsi", 35.0)),
            range_rebound_max_entry_rsi=float(strategy.get("range_rebound_max_entry_rsi", 55.0)),
            range_rebound_trend_break_grace_ticks=int(strategy.get("range_rebound_trend_break_grace_ticks", 2)),
            bollinger_trend_break_grace_ticks=int(strategy.get("bollinger_trend_break_grace_ticks", strategy.get("range_rebound_trend_break_grace_ticks", 2))),
            enable_bollinger_rebound_filter=bool(strategy.get("enable_bollinger_rebound_filter", False)),
            bollinger_window=int(strategy.get("bollinger_window", 20)),
            bollinger_stddev=float(strategy.get("bollinger_stddev", 2.0)),
            bollinger_touch_tolerance_pct=float(strategy.get("bollinger_touch_tolerance_pct", 0.6)),
            bollinger_prior_touch_lookback=int(strategy.get("bollinger_prior_touch_lookback", 2)),
            bollinger_filter_penalty=float(strategy.get("bollinger_filter_penalty", 0.08)),
            bollinger_min_confirmations=int(strategy.get("bollinger_min_confirmations", 2)),
            bollinger_min_expected_upside_pct=float(strategy.get("bollinger_min_expected_upside_pct", 0.3)),
            enable_crash_candle_filter=bool(strategy.get("enable_crash_candle_filter", True)),
            crash_candle_lookback=int(strategy.get("crash_candle_lookback", 3)),
            crash_candle_body_pct=float(strategy.get("crash_candle_body_pct", 1.2)),
            crash_candle_volume_ratio=float(strategy.get("crash_candle_volume_ratio", 1.4)),
            crash_candle_break_lookback=int(strategy.get("crash_candle_break_lookback", 12)),
            regime_ensemble_only=bool(strategy.get("regime_ensemble_only", False)),
        ),
        risk=RiskConfig(
            daily_profit_target_pct=float(risk["daily_profit_target_pct"]),
            daily_loss_limit_pct=float(risk["daily_loss_limit_pct"]),
            max_entries_per_day=int(risk.get("max_entries_per_day", risk.get("max_trades_per_day", 3))),
            min_entries_per_day=int(risk.get("min_entries_per_day", 0)),
            max_position_fraction=float(risk["max_position_fraction"]),
            max_consecutive_losses=int(risk["max_consecutive_losses"]),
            new_entries_enabled=bool(risk.get("new_entries_enabled", True)),
            min_equity_krw=float(risk.get("min_equity_krw", 0.0)),
            halt_cooldown_ticks=int(risk.get("halt_cooldown_ticks", 6)),
            consecutive_loss_cooldown_ticks=int(risk.get("consecutive_loss_cooldown_ticks", risk.get("halt_cooldown_ticks", 6))),
            max_expected_downside_to_upside_ratio=float(risk.get("max_expected_downside_to_upside_ratio", 1.0)),
            max_open_positions=int(risk.get("max_open_positions", 4)),
            max_total_position_fraction=float(risk.get("max_total_position_fraction", 0.95)),
            max_new_entries_per_tick=int(risk.get("max_new_entries_per_tick", 1)),
            min_trade_cash_krw=float(risk.get("min_trade_cash_krw", 0.0)),
            min_candidate_score=float(risk.get("min_candidate_score", 0.0)),
            recent_exit_sample_size=int(risk.get("recent_exit_sample_size", 20)),
            min_recent_expectancy_krw=float(risk.get("min_recent_expectancy_krw", 0.0)),
            min_recent_profit_factor=float(risk.get("min_recent_profit_factor", 1.05)),
            max_recent_loss_rate=float(risk.get("max_recent_loss_rate", 0.58)),
            strategy_exit_sample_size=int(risk.get("strategy_exit_sample_size", 6)),
            market_exit_sample_size=int(risk.get("market_exit_sample_size", 4)),
            min_strategy_expectancy_krw=float(risk.get("min_strategy_expectancy_krw", 0.0)),
            min_market_expectancy_krw=float(risk.get("min_market_expectancy_krw", 0.0)),
            max_strategy_loss_rate=float(risk.get("max_strategy_loss_rate", 0.67)),
            max_market_loss_rate=float(risk.get("max_market_loss_rate", 0.75)),
            reason_exit_sample_size=int(risk.get("reason_exit_sample_size", 3)),
            min_reason_expectancy_krw=float(risk.get("min_reason_expectancy_krw", 0.0)),
            max_reason_loss_rate=float(risk.get("max_reason_loss_rate", 0.67)),
            adaptive_position_sizing=bool(risk.get("adaptive_position_sizing", True)),
        ),
        ai_decision=AiDecisionConfig(
            enabled=bool(ai_decision.get("enabled", True)),
            provider=str(ai_decision.get("provider", "local")),
            min_confidence=float(ai_decision.get("min_confidence", 0.55)),
            openai_model=str(ai_decision.get("openai_model", "gpt-5.4-mini")),
            api_key_env=str(ai_decision.get("api_key_env", "OPENAI_API_KEY")),
        ),
        paths=PathConfig(
            trade_journal=_resolve_path(base_dir, paths["trade_journal"]),
            event_log=_resolve_path(base_dir, paths["event_log"]),
            state_file=_resolve_path(base_dir, paths["state_file"]),
        ),
    )
    _validate_config(app)
    return app


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing object field: {key}")
    return value


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _validate_config(config: AppConfig) -> None:
    if config.mode != "paper":
        raise ValueError("Only paper mode is implemented in this MVP.")
    if config.starting_cash <= 0:
        raise ValueError("starting_cash must be positive.")
    if config.strategy.short_window <= 0:
        raise ValueError("short_window must be positive.")
    if config.strategy.long_window <= config.strategy.short_window:
        raise ValueError("long_window must be greater than short_window.")
    if not 0 < config.strategy.position_fraction <= 1:
        raise ValueError("position_fraction must be between 0 and 1.")
    if config.strategy.btc_long_window <= config.strategy.btc_short_window:
        raise ValueError("btc_long_window must be greater than btc_short_window.")
    if config.strategy.min_volume_ratio <= 0:
        raise ValueError("min_volume_ratio must be positive.")
    if config.strategy.max_recent_momentum_pct <= config.strategy.min_recent_momentum_pct:
        raise ValueError("max_recent_momentum_pct must be greater than min_recent_momentum_pct.")
    if config.strategy.max_ma_distance_pct <= 0:
        raise ValueError("max_ma_distance_pct must be positive.")
    if config.strategy.rsi_period < 2:
        raise ValueError("rsi_period must be at least 2.")
    if not 0 < config.strategy.max_entry_rsi <= 100:
        raise ValueError("max_entry_rsi must be between 0 and 100.")
    if config.strategy.target_recent_volatility_pct <= 0:
        raise ValueError("target_recent_volatility_pct must be positive.")
    if not 0 < config.strategy.min_volatility_position_fraction <= 1:
        raise ValueError("min_volatility_position_fraction must be between 0 and 1.")
    if config.strategy.long_trend_ema_window < 0:
        raise ValueError("long_trend_ema_window must not be negative.")
    if config.strategy.long_trend_ema_window > 200:
        raise ValueError("long_trend_ema_window must be 200 or lower for the Upbit candle API.")
    if config.strategy.time_stop_ticks < 0:
        raise ValueError("time_stop_ticks must not be negative.")
    if config.strategy.min_expected_upside_pct < 0:
        raise ValueError("min_expected_upside_pct must not be negative.")
    if config.strategy.min_validated_recovery_pct < 0:
        raise ValueError("min_validated_recovery_pct must not be negative.")
    if config.strategy.min_net_edge_pct < 0:
        raise ValueError("min_net_edge_pct must not be negative.")
    if config.strategy.target_upside_pct <= 0:
        raise ValueError("target_upside_pct must be positive.")
    if any(hour < 0 or hour > 23 for hour in config.strategy.blocked_entry_hours_kst):
        raise ValueError("blocked_entry_hours_kst must contain hours between 0 and 23.")
    if config.strategy.reentry_cooldown_ticks < 0:
        raise ValueError("reentry_cooldown_ticks must not be negative.")
    if config.strategy.stop_volatility_multiplier <= 0:
        raise ValueError("stop_volatility_multiplier must be positive.")
    if config.strategy.breakeven_trigger_pct < 0:
        raise ValueError("breakeven_trigger_pct must not be negative.")
    if config.strategy.partial_take_profit_pct < 0:
        raise ValueError("partial_take_profit_pct must not be negative.")
    if not 0 < config.strategy.partial_take_profit_fraction <= 1:
        raise ValueError("partial_take_profit_fraction must be between 0 and 1.")
    if config.strategy.trailing_stop_pct < 0:
        raise ValueError("trailing_stop_pct must not be negative.")
    if config.strategy.min_orderbook_imbalance <= 0:
        raise ValueError("min_orderbook_imbalance must be positive.")
    if config.strategy.max_orderbook_spread_bps < 0:
        raise ValueError("max_orderbook_spread_bps must not be negative.")
    if not 0 <= config.strategy.min_market_breadth_ratio <= 1:
        raise ValueError("min_market_breadth_ratio must be between 0 and 1.")
    if config.strategy.min_price_krw < 0:
        raise ValueError("min_price_krw must not be negative.")
    if config.strategy.max_recent_stopouts_per_market < 0:
        raise ValueError("max_recent_stopouts_per_market must not be negative.")
    if config.strategy.stopout_lookback_ticks < 0:
        raise ValueError("stopout_lookback_ticks must not be negative.")
    if config.strategy.five_minute_short_window <= 0:
        raise ValueError("five_minute_short_window must be positive.")
    if config.strategy.five_minute_long_window <= config.strategy.five_minute_short_window:
        raise ValueError("five_minute_long_window must be greater than five_minute_short_window.")
    if config.strategy.five_minute_trend_tolerance_pct < 0:
        raise ValueError("five_minute_trend_tolerance_pct must not be negative.")
    if config.strategy.range_rebound_lookback < 3:
        raise ValueError("range_rebound_lookback must be at least 3.")
    if config.strategy.range_rebound_max_distance_from_low_pct < 0:
        raise ValueError("range_rebound_max_distance_from_low_pct must not be negative.")
    if config.strategy.range_rebound_max_ema_gap_pct < 0:
        raise ValueError("range_rebound_max_ema_gap_pct must not be negative.")
    if config.strategy.range_rebound_min_volume_ratio <= 0:
        raise ValueError("range_rebound_min_volume_ratio must be positive.")
    if config.strategy.range_rebound_min_expected_upside_pct < 0:
        raise ValueError("range_rebound_min_expected_upside_pct must not be negative.")
    if not 0 <= config.strategy.range_rebound_min_rsi <= 100:
        raise ValueError("range_rebound_min_rsi must be between 0 and 100.")
    if not 0 <= config.strategy.range_rebound_max_entry_rsi <= 100:
        raise ValueError("range_rebound_max_entry_rsi must be between 0 and 100.")
    if config.strategy.range_rebound_trend_break_grace_ticks < 0:
        raise ValueError("range_rebound_trend_break_grace_ticks must not be negative.")
    if config.strategy.bollinger_trend_break_grace_ticks < 0:
        raise ValueError("bollinger_trend_break_grace_ticks must not be negative.")
    if config.strategy.bollinger_window < 5:
        raise ValueError("bollinger_window must be at least 5.")
    if config.strategy.bollinger_stddev <= 0:
        raise ValueError("bollinger_stddev must be positive.")
    if config.strategy.bollinger_touch_tolerance_pct < 0:
        raise ValueError("bollinger_touch_tolerance_pct must not be negative.")
    if config.strategy.bollinger_prior_touch_lookback < 0:
        raise ValueError("bollinger_prior_touch_lookback must not be negative.")
    if config.strategy.bollinger_filter_penalty < 0:
        raise ValueError("bollinger_filter_penalty must not be negative.")
    if config.strategy.bollinger_min_confirmations < 1:
        raise ValueError("bollinger_min_confirmations must be at least 1.")
    if config.strategy.bollinger_min_expected_upside_pct < 0:
        raise ValueError("bollinger_min_expected_upside_pct must not be negative.")
    if config.strategy.crash_candle_lookback < 1:
        raise ValueError("crash_candle_lookback must be at least 1.")
    if config.strategy.crash_candle_body_pct < 0:
        raise ValueError("crash_candle_body_pct must not be negative.")
    if config.strategy.crash_candle_volume_ratio <= 0:
        raise ValueError("crash_candle_volume_ratio must be positive.")
    if config.strategy.crash_candle_break_lookback < 2:
        raise ValueError("crash_candle_break_lookback must be at least 2.")
    if not 0 < config.risk.max_position_fraction <= 1:
        raise ValueError("max_position_fraction must be between 0 and 1.")
    if config.risk.max_open_positions < 1:
        raise ValueError("max_open_positions must be at least 1.")
    if not 0 < config.risk.max_total_position_fraction <= 1:
        raise ValueError("max_total_position_fraction must be between 0 and 1.")
    if config.risk.max_entries_per_day < 1:
        raise ValueError("max_entries_per_day must be at least 1.")
    if config.risk.min_entries_per_day < 0:
        raise ValueError("min_entries_per_day must not be negative.")
    if config.risk.min_entries_per_day > config.risk.max_entries_per_day:
        raise ValueError("min_entries_per_day must not exceed max_entries_per_day.")
    if config.risk.min_equity_krw < 0:
        raise ValueError("min_equity_krw must not be negative.")
    if config.risk.max_new_entries_per_tick < 1:
        raise ValueError("max_new_entries_per_tick must be at least 1.")
    if config.risk.min_trade_cash_krw < 0:
        raise ValueError("min_trade_cash_krw must not be negative.")
    if config.risk.min_candidate_score < 0:
        raise ValueError("min_candidate_score must not be negative.")
    if config.risk.recent_exit_sample_size < 0:
        raise ValueError("recent_exit_sample_size must not be negative.")
    if config.risk.min_recent_profit_factor < 0:
        raise ValueError("min_recent_profit_factor must not be negative.")
    if not 0 <= config.risk.max_recent_loss_rate <= 1:
        raise ValueError("max_recent_loss_rate must be between 0 and 1.")
    if config.risk.strategy_exit_sample_size < 0:
        raise ValueError("strategy_exit_sample_size must not be negative.")
    if config.risk.market_exit_sample_size < 0:
        raise ValueError("market_exit_sample_size must not be negative.")
    if not 0 <= config.risk.max_strategy_loss_rate <= 1:
        raise ValueError("max_strategy_loss_rate must be between 0 and 1.")
    if not 0 <= config.risk.max_market_loss_rate <= 1:
        raise ValueError("max_market_loss_rate must be between 0 and 1.")
    if config.risk.halt_cooldown_ticks < 0:
        raise ValueError("halt_cooldown_ticks must not be negative.")
    if config.risk.consecutive_loss_cooldown_ticks < 0:
        raise ValueError("consecutive_loss_cooldown_ticks must not be negative.")
    if config.risk.max_expected_downside_to_upside_ratio <= 0:
        raise ValueError("max_expected_downside_to_upside_ratio must be positive.")
    if not 0 <= config.ai_decision.min_confidence <= 1:
        raise ValueError("ai_decision.min_confidence must be between 0 and 1.")
