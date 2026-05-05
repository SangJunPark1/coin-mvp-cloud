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
from .report import read_events, read_trades, render_report
from .risk import RiskState
from .watch_multi import MultiMarketTradingApp

KST = timezone(timedelta(hours=9))


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
) -> dict[str, Any]:
    loaded_config = load_config(config_path)
    storage = get_cloud_storage()
    config = runtime_config(loaded_config, storage)
    outputs = outputs or [Path("docs/index.html")]
    if reset:
        storage.reset(config)
        reset_outputs(outputs)
    else:
        storage.hydrate(config)

    data_source = UpbitPublicDataSource()
    markets = data_source.get_top_krw_markets(top_markets, min_trade_price_krw=config.strategy.min_price_krw)
    app = MultiMarketTradingApp(config, data_source, markets, request_delay=request_delay)
    state = load_state(config.paths.state_file)
    if state:
        apply_state(app, state)
    else:
        app.journal.event(
            "cloud_started",
            {
                "started_at": datetime.now(KST).isoformat(timespec="seconds"),
                "starting_cash": config.starting_cash,
                "markets": markets,
                "mode": "one_tick_cron",
            },
        )

    previous_tick = int(state.get("tick", 0)) if state else 0
    completed_tick = previous_tick
    for tick in range(previous_tick + 1, previous_tick + max(1, ticks) + 1):
        app.run_tick(tick)
        completed_tick = tick

    save_state(config.paths.state_file, app, completed_tick, markets)
    refresh_outputs(config, outputs)
    storage.persist(config)
    return {
        "ok": True,
        "tick": completed_tick,
        "cash": app.broker.cash,
        "equity": app.broker.equity(app.last_prices),
        "positions": {market: asdict(position) for market, position in app.broker.positions.items()},
        "risk": asdict(app.risk.state),
        "outputs": [str(output) for output in outputs],
        "storage": "remote" if storage.enabled else "local",
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
        "broker": {
            "cash": app.broker.cash,
            "realized_pnl": app.broker.realized_pnl,
            "positions": {market: asdict(position) for market, position in app.broker.positions.items()},
        },
        "equity": app.broker.equity(app.last_prices),
        "risk": asdict(app.risk.state),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def refresh_outputs(config: AppConfig, outputs: list[Path]) -> None:
    html = render_report(read_trades(config.paths.trade_journal), read_events(config.paths.event_log))
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")


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
