from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from coin_mvp.cloud_tick import run_cloud_ticks


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        secret = os.environ.get("CRON_SECRET", "")
        if secret:
            query = parse_qs(urlparse(self.path).query)
            provided = self.headers.get("x-cron-secret") or ""
            bearer = self.headers.get("authorization") or ""
            query_secret = query.get("secret", [""])[0]
            if provided != secret and bearer != f"Bearer {secret}" and query_secret != secret:
                self._send_json({"ok": False, "error": "unauthorized"}, status=401)
                return

        try:
            result = run_cloud_ticks(
                config_path=os.environ.get("COIN_MVP_CONFIG", "config.cloud.json"),
                top_markets=int(os.environ.get("TOP_MARKETS", "30")),
                request_delay=float(os.environ.get("REQUEST_DELAY", "0.35")),
                ticks=1,
                outputs=[Path(os.environ.get("REPORT_OUTPUT", "/tmp/coin_mvp/report.html"))],
                reset=False,
            )
            result["report_url"] = "/api/report"
            self._send_json(result)
        except Exception as exc:
            self._send_json({"ok": False, "error": repr(exc)}, status=500)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
