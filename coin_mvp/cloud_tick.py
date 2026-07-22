from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig, load_config
from .cloud_storage import get_cloud_storage, runtime_config
from .data import UpbitPublicDataSource
from .models import Position
from .report import read_events, read_trades, render_compact_report
from .risk import RiskState
from .watch_multi import MultiMarketTradingApp

KST = timezone(timedelta(hours=9))
MAX_CATCH_UP_TICKS = 12


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one or more cloud-style paper trading ticks.")
    parser.add_argument("--config", default="config.cloud.json")
    parser.add_argument("--top-markets", type=int, default=30)
    parser.add_argument("--request-delay", type=float, default=0.35)
    parser.add_argument("--ticks", type=int, default=1, help="Run this many one-shot cloud ticks.")
    parser.add_argument("--output", action="append", default=[])
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    result = run_cloud_ticks(
        config_path=args.config,
        top_markets=args.top_markets,
        request_delay=args.request_delay,
        ticks=args.ticks,
        outputs=[Path(value) for value in args.output] or [Path("docs/index.html")],
        reset=args.reset,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_cloud_ticks(
    config_path: str | Path = "config.cloud.json",
    top_markets: int = 30,
    request_delay: float = 0.35,
    ticks: int = 1,
    outputs: list[Path] | None = None,
    reset: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    loaded_config = load_config(config_path)
    storage = get_cloud_storage()
    config = runtime_config(loaded_config, storage)
    outputs = outputs or [Path("docs/index.html")]
    if reset:
        storage.reset(config)
        reset_outputs(outputs)
        if ticks <= 0:
            refresh_outputs(config, outputs)
            storage.persist(config)
            return {
                "ok": True,
                "reset": True,
                "tick": 0,
                "cash": config.starting_cash,
                "equity": config.starting_cash,
                "positions": {},
                "risk": {
                    "starting_equity": config.starting_cash,
                    "entries_today": 0,
                    "exits_today": 0,
                    "consecutive_losses": 0,
                    "halted": False,
                    "halt_reason": "",
                },
                "outputs": [str(output) for output in outputs],
                "storage": "remote" if storage.enabled else "local",
            }
    else:
        storage.hydrate(config)

    state = load_state(config.paths.state_file)
    if resume and state:
        state = resume_state(state)
        config.paths.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        if ticks <= 0:
            refresh_outputs(config, outputs)
            storage.persist(config)
            return {
                "ok": True,
                "resumed": True,
                "tick": int(state.get("tick", 0)),
                "cash": state.get("broker", {}).get("cash"),
                "equity": state.get("equity"),
                "positions": state.get("broker", {}).get("positions", {}),
                "risk": state.get("risk", {}),
                "outputs": [str(output) for output in outputs],
                "storage": "remote" if storage.enabled else "local",
            }
    cooldown = int(config.poll_seconds)
    if state and cooldown > 0 and not reset and not resume:
        last_run_at = parse_kst_time(str(state.get("last_run_at") or ""))
        now = datetime.now(KST)
        if last_run_at is not None and now < last_run_at + timedelta(seconds=cooldown):
            return {
                "ok": True,
                "skipped": True,
                "reason": f"tick cooldown active: {cooldown}s",
                "next_run_at": (last_run_at + timedelta(seconds=cooldown)).isoformat(timespec="seconds"),
                "tick": int(state.get("tick", 0)),
                "cash": state.get("broker", {}).get("cash"),
                "equity": state.get("equity"),
                "positions": state.get("broker", {}).get("positions", {}),
                "risk": state.get("risk", {}),
                "outputs": [str(output) for output in outputs],
                "storage": "remote" if storage.enabled else "local",
            }

    effective_top_markets = min(max(6, top_markets), 12)
    data_source = UpbitPublicDataSource()
    markets = data_source.get_top_krw_markets(effective_top_markets, min_trade_price_krw=config.strategy.min_price_krw)
    now = datetime.now(KST)
    last_run_at = parse_kst_time(str(state.get("last_run_at") or "")) if state else None
    catch_up_times, skipped_catch_up = missed_tick_times(
        last_run_at,
        now,
        cooldown,
        max_ticks=MAX_CATCH_UP_TICKS,
    )
    app = MultiMarketTradingApp(config, data_source, markets, request_delay=request_delay)
    if state:
        apply_state(app, state)
    else:
        app.journal.event(
            "cloud_started",
            {
                "started_at": datetime.now(KST).isoformat(timespec="seconds"),
                "starting_cash": config.starting_cash,
                "markets": markets,
                "requested_top_markets": top_markets,
                "effective_top_markets": effective_top_markets,
                "mode": "one_tick_cron",
            },
        )

    previous_tick = int(state.get("tick", 0)) if state else 0
    completed_tick = previous_tick
    caught_up = 0
    if catch_up_times:
        caught_up = run_historical_catch_up(
            app,
            data_source,
            markets,
            catch_up_times,
            previous_tick,
        )
        completed_tick += caught_up
        app.journal.event(
            "cloud_catch_up",
            {
                "from": catch_up_times[0].isoformat(timespec="seconds"),
                "to": catch_up_times[-1].isoformat(timespec="seconds"),
                "processed": caught_up,
                "skipped_older": skipped_catch_up,
                "cadence_seconds": cooldown,
                "method": "point_in_time_candle_replay",
            },
        )
    for tick in range(completed_tick + 1, completed_tick + max(1, ticks) + 1):
        app.run_tick(tick)
        completed_tick = tick

    save_state(config.paths.state_file, app, completed_tick, markets)
    app.journal.event("state_snapshot", build_state_snapshot(app, completed_tick, markets))
    trim_runtime_logs(config, max_events=180)
    refresh_outputs(config, outputs)
    storage.persist(config)
    return {
        "ok": True,
        "tick": completed_tick,
        "catch_up_ticks": caught_up,
        "catch_up_skipped_older": skipped_catch_up,
        "cash": app.broker.cash,
        "equity": app.broker.equity(app.last_prices),
        "positions": {market: asdict(position) for market, position in app.broker.positions.items()},
        "risk": asdict(app.risk.state),
        "outputs": [str(output) for output in outputs],
        "storage": "remote" if storage.enabled else "local",
    }


def missed_tick_times(
    last_run_at: datetime | None,
    now: datetime,
    cadence_seconds: int,
    max_ticks: int = MAX_CATCH_UP_TICKS,
) -> tuple[list[datetime], int]:
    """Return missed scheduled instants, excluding the current live invocation."""
    if last_run_at is None or cadence_seconds <= 0 or max_ticks <= 0:
        return [], 0
    cadence = timedelta(seconds=cadence_seconds)
    elapsed_slots = int((now - last_run_at).total_seconds() // cadence_seconds)
    missed_count = max(0, elapsed_slots - 1)
    if missed_count == 0:
        return [], 0
    all_times = [last_run_at + cadence * slot for slot in range(1, missed_count + 1)]
    skipped = max(0, len(all_times) - max_ticks)
    return all_times[-max_ticks:], skipped


def run_historical_catch_up(
    app: MultiMarketTradingApp,
    live_source: UpbitPublicDataSource,
    markets: list[str],
    scheduled_times: list[datetime],
    previous_tick: int,
) -> int:
    """Replay missed ticks from candles known at each scheduled instant."""
    from . import watch_multi as watch_multi_module
    from .backtest import ReplayDataSource, historical_backtest_context
    from .strategy import required_candle_count

    newest = scheduled_times[-1].astimezone(timezone.utc)
    oldest = scheduled_times[0].astimezone(timezone.utc)
    span_minutes = max(1, int((newest - oldest).total_seconds() // 60))
    one_minute_count = min(400, max(required_candle_count(app.config.strategy) + span_minutes + 10, 240))
    candles_by_unit: dict[int, dict[str, list[Any]]] = {1: {}, 5: {}, 15: {}, 60: {}}
    for market in markets:
        candles_by_unit[1][market] = live_source.get_recent_candles(market, one_minute_count, unit_minutes=1)
        candles_by_unit[5][market] = live_source.get_recent_candles(market, 100, unit_minutes=5)
        candles_by_unit[15][market] = live_source.get_recent_candles(market, 100, unit_minutes=15)
        candles_by_unit[60][market] = live_source.get_recent_candles(market, 100, unit_minutes=60)

    reference = candles_by_unit[1][markets[0]]
    replay = ReplayDataSource(candles_by_unit, markets, start_index=0)
    original_source = app.data_source
    original_context = watch_multi_module.collect_decision_context
    app.data_source = replay
    watch_multi_module.collect_decision_context = lambda _source, _strategy: historical_backtest_context(replay, app.config)
    completed = 0
    try:
        for scheduled_at in scheduled_times:
            asof = scheduled_at.astimezone(timezone.utc)
            eligible = [
                index
                for index, candle in enumerate(reference)
                if candle.timestamp + timedelta(minutes=1) <= asof
            ]
            if not eligible:
                continue
            replay.set_index(eligible[-1])
            app.run_tick(previous_tick + completed + 1)
            completed += 1
    finally:
        app.data_source = original_source
        watch_multi_module.collect_decision_context = original_context
    return completed


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resume_state(state: dict[str, Any]) -> dict[str, Any]:
    risk = state.get("risk")
    if not isinstance(risk, dict):
        risk = {}
        state["risk"] = risk
    equity = state.get("equity")
    try:
        current_equity = float(equity)
    except (TypeError, ValueError):
        current_equity = float(state.get("broker", {}).get("cash", 0.0) or 0.0)
    if current_equity > 0:
        risk["starting_equity"] = current_equity
    risk["halted"] = False
    risk["halt_reason"] = ""
    risk["halt_started_tick"] = None
    risk["halt_until_tick"] = None
    risk["consecutive_losses"] = min(int(risk.get("consecutive_losses", 0) or 0), 1)
    state["last_run_at"] = datetime.now(KST).isoformat(timespec="seconds")
    return state


def parse_kst_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def apply_state(app: MultiMarketTradingApp, state: dict[str, Any]) -> None:
    broker = state.get("broker", {})
    app.broker.cash = float(broker.get("cash", app.broker.cash))
    app.broker.realized_pnl = float(broker.get("realized_pnl", app.broker.realized_pnl))
    app.broker.positions = load_positions(broker)
    last_prices = state.get("last_prices", {})
    if isinstance(last_prices, dict):
        app.last_prices = {str(key): float(value) for key, value in last_prices.items() if value is not None}

    risk = state.get("risk", {})
    app.risk.state = RiskState(
        starting_equity=float(risk.get("starting_equity", app.risk.state.starting_equity)),
        day_key=str(risk.get("day_key", "")),
        entries_today=int(risk.get("entries_today", 0)),
        exits_today=int(risk.get("exits_today", 0)),
        consecutive_losses=int(risk.get("consecutive_losses", 0)),
        halted=bool(risk.get("halted", False)),
        halt_reason=str(risk.get("halt_reason", "")),
        period_started_at=str(risk.get("period_started_at", "")),
        halt_started_tick=int(risk["halt_started_tick"]) if risk.get("halt_started_tick") is not None else None,
        halt_until_tick=int(risk["halt_until_tick"]) if risk.get("halt_until_tick") is not None else None,
    )

    app.position_entry_tick = int_dict(state.get("position_entry_tick", {}))
    app.position_entry_strategy = str_dict(state.get("position_entry_strategy", {}))
    app.market_reentry_until_tick = int_dict(state.get("market_reentry_until_tick", {}))
    last_context = state.get("last_decision_context", {})
    if isinstance(last_context, dict):
        app.last_decision_context = last_context
    stopout_ticks = state.get("market_stopout_ticks", {})
    if isinstance(stopout_ticks, dict):
        app.market_stopout_ticks = {
            str(key): [int(item) for item in value if item is not None]
            for key, value in stopout_ticks.items()
            if isinstance(value, list)
        }


def load_positions(broker: dict[str, Any]) -> dict[str, Position]:
    positions = broker.get("positions")
    if isinstance(positions, dict):
        return {
            str(market): Position(
                qty=float(position.get("qty", 0.0)),
                avg_price=float(position.get("avg_price", 0.0)),
                peak_price=float(position.get("peak_price", 0.0)),
                partial_exit_taken=bool(position.get("partial_exit_taken", False)),
            )
            for market, position in positions.items()
            if isinstance(position, dict) and float(position.get("qty", 0.0)) > 0
        }

    position = broker.get("position", {})
    market = str(broker.get("market") or "")
    if isinstance(position, dict) and market and float(position.get("qty", 0.0)) > 0:
        return {
            market: Position(
                qty=float(position.get("qty", 0.0)),
                avg_price=float(position.get("avg_price", 0.0)),
                peak_price=float(position.get("peak_price", 0.0)),
                partial_exit_taken=bool(position.get("partial_exit_taken", False)),
            )
        }
    return {}


def save_state(path: Path, app: MultiMarketTradingApp, tick: int, markets: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_run_at": datetime.now(KST).isoformat(timespec="seconds"),
        "tick": tick,
        "markets": markets,
        "last_prices": app.last_prices,
        "position_entry_tick": app.position_entry_tick,
        "position_entry_strategy": app.position_entry_strategy,
        "market_reentry_until_tick": app.market_reentry_until_tick,
        "market_stopout_ticks": app.market_stopout_ticks,
        "last_decision_context": app.last_decision_context,
        "broker": {
            "cash": app.broker.cash,
            "realized_pnl": app.broker.realized_pnl,
            "positions": {market: asdict(position) for market, position in app.broker.positions.items()},
        },
        "equity": app.broker.equity(app.last_prices),
        "risk": asdict(app.risk.state),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_state_snapshot(app: MultiMarketTradingApp, tick: int, markets: list[str]) -> dict[str, Any]:
    return {
        "tick": tick,
        "markets": markets,
        "cash": app.broker.cash,
        "equity": app.broker.equity(app.last_prices),
        "positions": {market: asdict(position) for market, position in app.broker.positions.items()},
        "last_prices": app.last_prices,
        "decision_context": app.last_decision_context,
        "risk": asdict(app.risk.state),
    }


def refresh_outputs(config: AppConfig, outputs: list[Path]) -> None:
    html = render_compact_report(read_trades(config.paths.trade_journal), read_events(config.paths.event_log)[-180:])
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")


def trim_runtime_logs(config: AppConfig, max_events: int = 180, max_trades: int = 120) -> None:
    trim_jsonl(config.paths.event_log, max_events)
    trim_csv(config.paths.trade_journal, max_trades)


def trim_jsonl(path: Path, keep: int) -> None:
    if keep <= 0 or not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= keep:
        return
    path.write_text("\n".join(lines[-keep:]) + "\n", encoding="utf-8")


def trim_csv(path: Path, keep: int) -> None:
    if keep <= 0 or not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= keep + 1:
        return
    header = lines[:1]
    rows = lines[1:]
    path.write_text("\n".join(header + rows[-keep:]) + "\n", encoding="utf-8")


def reset_outputs(outputs: list[Path]) -> None:
    paths = [*outputs]
    for path in paths:
        if path.exists() and path.is_file():
            path.unlink()


def int_dict(value: Any) -> dict[str, int]:
    if isinstance(value, int):
        return {"KRW-BTC": value}
    if not isinstance(value, dict):
        return {}
    return {str(key): int(item) for key, item in value.items() if item is not None}


def str_dict(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        return {"KRW-BTC": value}
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if item is not None}


if __name__ == "__main__":
    main()
