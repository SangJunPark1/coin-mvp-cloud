from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import AppConfig, PathConfig

TRADE_HEADER = "timestamp,market,side,price,qty,fee,cash_after,position_qty_after,realized_pnl,reason\n"


class CloudStorage:
    def hydrate(self, config: AppConfig) -> None:
        raise NotImplementedError

    def persist(self, config: AppConfig) -> None:
        raise NotImplementedError

    def reset(self, config: AppConfig) -> None:
        reset_local_files(config)

    @property
    def enabled(self) -> bool:
        return False


class NoopCloudStorage(CloudStorage):
    def hydrate(self, config: AppConfig) -> None:
        ensure_local_files(config)

    def persist(self, config: AppConfig) -> None:
        return


class UpstashRestStorage(CloudStorage):
    def __init__(self, url: str, token: str, prefix: str = "coin_mvp") -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.prefix = prefix.strip(":") or "coin_mvp"

    @property
    def enabled(self) -> bool:
        return True

    def hydrate(self, config: AppConfig) -> None:
        state_text = self._get("state")
        trades_text = self._get("trades") or TRADE_HEADER
        events_text = self._get("events") or ""
        write_text(config.paths.state_file, state_text or "{}")
        write_text(config.paths.trade_journal, trades_text)
        write_text(config.paths.event_log, events_text)

    def persist(self, config: AppConfig) -> None:
        self._set("state", read_text(config.paths.state_file, "{}"))
        self._set("trades", compact_trade_text(read_text(config.paths.trade_journal, TRADE_HEADER)))
        self._set("events", compact_event_text(read_text(config.paths.event_log, "")))

    def reset(self, config: AppConfig) -> None:
        reset_local_files(config)
        self._set("state", "{}")
        self._set("trades", TRADE_HEADER)
        self._set("events", "")

    def _key(self, name: str) -> str:
        return f"{self.prefix}:{name}"

    def _get(self, name: str) -> str | None:
        payload = self._command(["GET", self._key(name)])
        value = payload.get("result")
        return str(value) if value is not None else None

    def _set(self, name: str, value: str) -> None:
        self._command(["SET", self._key(name), value])

    def _command(self, command: list[str]) -> dict[str, Any]:
        body = json.dumps(command).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))


def get_cloud_storage() -> CloudStorage:
    provider = os.environ.get("COIN_MVP_STORAGE", "").strip().lower()
    url = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")
    if provider in {"upstash", "redis", "kv"} or (url and token):
        if not url or not token:
            raise RuntimeError("UPSTASH_REDIS_REST_URL/KV_REST_API_URL and REST token are required for cloud storage.")
        prefix = os.environ.get("COIN_MVP_STORAGE_PREFIX", "coin_mvp")
        return UpstashRestStorage(url, token, prefix)
    return NoopCloudStorage()


def runtime_config(config: AppConfig, storage: CloudStorage) -> AppConfig:
    if not storage.enabled:
        return config
    root = Path(os.environ.get("COIN_MVP_TMP_DIR", "/tmp/coin_mvp"))
    return replace(
        config,
        paths=PathConfig(
            trade_journal=root / "cloud_trades.csv",
            event_log=root / "cloud_events.jsonl",
            state_file=root / "cloud_state.json",
        ),
    )


def ensure_local_files(config: AppConfig) -> None:
    config.paths.trade_journal.parent.mkdir(parents=True, exist_ok=True)
    config.paths.event_log.parent.mkdir(parents=True, exist_ok=True)
    config.paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    if not config.paths.trade_journal.exists():
        config.paths.trade_journal.write_text(TRADE_HEADER, encoding="utf-8")
    if not config.paths.event_log.exists():
        config.paths.event_log.write_text("", encoding="utf-8")
    if not config.paths.state_file.exists():
        config.paths.state_file.write_text("{}", encoding="utf-8")


def reset_local_files(config: AppConfig) -> None:
    write_text(config.paths.trade_journal, TRADE_HEADER)
    write_text(config.paths.event_log, "")
    write_text(config.paths.state_file, "{}")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def read_text(path: Path, default: str) -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def compact_trade_text(value: str) -> str:
    max_rows = int(os.environ.get("COIN_MVP_MAX_TRADE_ROWS", "800"))
    lines = [line for line in value.splitlines() if line.strip()]
    if not lines:
        return TRADE_HEADER
    header = lines[0] if lines[0].startswith("timestamp,") else TRADE_HEADER.strip()
    rows = lines[1:] if lines[0].startswith("timestamp,") else lines
    kept = rows[-max_rows:] if max_rows > 0 else rows
    return "\n".join([header, *kept]) + "\n"


def compact_event_text(value: str) -> str:
    max_rows = int(os.environ.get("COIN_MVP_MAX_EVENT_ROWS", "1200"))
    lines = [line for line in value.splitlines() if line.strip()]
    kept = lines[-max_rows:] if max_rows > 0 else lines
    return "\n".join(kept) + ("\n" if kept else "")
