from __future__ import annotations

import os
from http.server import BaseHTTPRequestHandler

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
            html = render_report(read_trades(config.paths.trade_journal), read_events(config.paths.event_log))
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
