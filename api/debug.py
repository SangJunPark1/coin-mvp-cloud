from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler
from typing import Any

from coin_mvp.cloud_storage import get_cloud_storage, runtime_config
from coin_mvp.config import load_config
from coin_mvp.report import read_events, read_trades


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        secret = os.environ.get("CRON_SECRET", "")
        if secret:
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            if f"secret={secret}" not in query:
                self._send_json({"ok": False, "error": "unauthorized"}, status=401)
                return
        try:
            loaded_config = load_config(os.environ.get("COIN_MVP_CONFIG", "config.cloud.json"))
            storage = get_cloud_storage()
            config = runtime_config(loaded_config, storage)
            storage.hydrate(config)
            trades = read_trades(config.paths.trade_journal)
            events = read_events(config.paths.event_log)
            state = load_state(config.paths.state_file)
            self._send_json(
                {
                    "ok": True,
                    "storage": "remote" if storage.enabled else "local",
                    "state_tick": state.get("tick"),
                    "state_last_run_at": state.get("last_run_at"),
                    "state_positions": list(((state.get("broker") or {}).get("positions") or {}).keys()),
                    "trade_count": len(trades),
                    "last_trade": trade_to_dict(trades[-1]) if trades else None,
                    "event_count": len(events),
                    "last_event": event_summary(events[-1]) if events else None,
                }
            )
        except Exception as exc:
            self._send_json({"ok": False, "error": repr(exc)}, status=500)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def load_state(path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def trade_to_dict(trade) -> dict[str, Any]:
    return {
        "timestamp": trade.timestamp,
        "market": trade.market,
        "side": trade.side,
        "price": trade.price,
        "qty": trade.qty,
        "cash_after": trade.cash_after,
        "realized_pnl": trade.realized_pnl,
        "reason": trade.reason,
    }


def event_summary(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload", {}) if isinstance(event, dict) else {}
    return {
        "timestamp": event.get("timestamp"),
        "event": event.get("event"),
        "tick": payload.get("tick") if isinstance(payload, dict) else None,
        "market": payload.get("market") if isinstance(payload, dict) else None,
    }
