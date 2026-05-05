from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from typing import Any

from coin_mvp.cloud_storage import get_cloud_storage, runtime_config
from coin_mvp.config import load_config
from coin_mvp.report import read_events, read_trades, render_report


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        try:
            loaded_config = load_config(os.environ.get("COIN_MVP_CONFIG", "config.cloud.json"))
            storage = get_cloud_storage()
            config = runtime_config(loaded_config, storage)
            storage.hydrate(config)
            trades = read_trades(config.paths.trade_journal)
            events = read_events(config.paths.event_log)
            events = append_state_snapshot(events, load_state(config.paths.state_file))
            html = render_report(trades, events)
            self._send_html(html)
        except Exception as exc:
            self._send_html(f"<pre>{repr(exc)}</pre>", status=500)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def load_state(path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def append_state_snapshot(events: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    if not state:
        return events
    broker = state.get("broker", {}) if isinstance(state.get("broker"), dict) else {}
    risk = state.get("risk", {}) if isinstance(state.get("risk"), dict) else {}
    positions = broker.get("positions", {}) if isinstance(broker.get("positions"), dict) else {}
    last_prices = state.get("last_prices", {}) if isinstance(state.get("last_prices"), dict) else {}
    cash = float(broker.get("cash", 0.0) or 0.0)
    equity = cash + sum(
        float(position.get("qty", 0.0) or 0.0)
        * float(last_prices.get(market, position.get("avg_price", 0.0)) or 0.0)
        for market, position in positions.items()
        if isinstance(position, dict)
    )
    timestamp = str(state.get("last_run_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"))
    snapshot = {
        "timestamp": timestamp,
        "event": "state_snapshot",
        "payload": {
            "tick": state.get("tick"),
            "cash": cash,
            "equity": equity,
            "positions": positions,
            "last_prices": last_prices,
            "risk": risk,
        },
    }
    return [*events, snapshot]
