from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

REPORT_PARTIAL_TAKE_PROFIT_PCT = 1.0
REPORT_TARGET_UPSIDE_PCT = 3.0
REPORT_STOP_LOSS_PCT = 0.65
REPORT_TRAILING_STOP_PCT = 0.5


@dataclass(frozen=True)
class TradeRow:
    timestamp: str
    market: str
    side: str
    price: float
    qty: float
    fee: float
    cash_after: float
    position_qty_after: float
    realized_pnl: float
    reason: str


@dataclass(frozen=True)
class RoundTrip:
    entry: TradeRow
    exit: TradeRow

    @property
    def pnl(self) -> float:
        return self.exit.realized_pnl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the coin MVP HTML report.")
    parser.add_argument("--trades", default="data/trades.csv")
    parser.add_argument("--events", default="logs/events.jsonl")
    parser.add_argument("--output", default="reports/latest_report.html")
    args = parser.parse_args()

    trades = read_trades(Path(args.trades))
    events = read_events(Path(args.events))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(trades, events), encoding="utf-8")
    print(f"리포트 생성 완료: {output}")


def read_trades(path: Path) -> list[TradeRow]:
    if not path.exists():
        return []
    rows: list[TradeRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                TradeRow(
                    timestamp=row["timestamp"],
                    market=row["market"],
                    side=row["side"],
                    price=float(row["price"]),
                    qty=float(row["qty"]),
                    fee=float(row["fee"]),
                    cash_after=float(row["cash_after"]),
                    position_qty_after=float(row["position_qty_after"]),
                    realized_pnl=float(row["realized_pnl"]),
                    reason=row["reason"],
                )
            )
    return rows


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # A killed or overlapping serverless run can leave one partial
                # JSONL record. Keep the completed records usable for reports.
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def render_report(trades: list[TradeRow], events: list[dict[str, Any]]) -> str:
    return render_console_report(trades, events)


def render_compact_report(trades: list[TradeRow], events: list[dict[str, Any]]) -> str:
    trades = sorted(trades, key=lambda trade: parse_sort_time(trade.timestamp))
    events = sorted(events, key=lambda event: parse_sort_time(str(event.get("timestamp", ""))))
    metrics = calculate_metrics(trades)
    daily = calculate_daily_metrics(trades, events)
    last_event = events[-1] if events else {}
    last_payload = last_event.get("payload", {}) if isinstance(last_event, dict) else {}
    risk = last_payload.get("risk", {}) if isinstance(last_payload, dict) else {}
    portfolio = portfolio_summary(trades, events)
    open_positions = find_latest_positions(events)
    latest_context = find_latest_decision_context(events)
    refreshed_at = display_time(str(last_event.get("timestamp", ""))) if last_event else "아직 없음"
    market_mode = str(latest_context.get("market_mode", "unknown")) if latest_context else "unknown"
    session_label = str(latest_context.get("session_label", "-")) if latest_context else "-"
    change = float(portfolio["change_amount"])
    change_class = "pos" if change >= 0 else "neg"
    cash = float(portfolio.get("cash", portfolio["current_equity"]))
    halted = bool(risk.get("halted"))
    operation_state = "중지" if halted else "운영 중"
    operation_class = "warn" if halted else "ok"
    target_amount = float(portfolio["starting_cash"]) * 0.03
    target_progress = change / target_amount if target_amount else 0.0
    cards = [
        ("평가금액", krw(float(portfolio["current_equity"]))),
        ("손익", krw(change)),
        ("수익률", pct(float(portfolio["return_pct"]) / 100.0)),
        ("오늘 손익", krw(float(daily["change_amount"]))),
        ("오늘 수익률", pct(float(daily["return_pct"]) / 100.0)),
        ("오늘 목표", pct(float(daily["target_progress"]))),
        ("가용 현금", krw(cash)),
        ("완료 거래", str(metrics["exit_count"])),
        ("승률", pct(float(metrics["win_rate"]))),
        ("기대값", krw(float(metrics["expectancy"]))),
        ("오픈", f"{len(open_positions)}개"),
    ]
    state_rows = [
        ("상태", operation_state, operation_class),
        ("시장", market_mode, "warn" if market_mode in {"risk_off", "capital_protect"} else "ok" if market_mode == "risk_on" else ""),
        ("세션", session_label, ""),
        ("오늘 진입", str(risk.get("entries_today", 0)), ""),
        ("연속 손실", str(risk.get("consecutive_losses", 0)), "warn" if int(risk.get("consecutive_losses", 0) or 0) else ""),
        ("일 목표 진행률", pct(float(daily["target_progress"])), "pos" if float(daily["target_progress"]) >= 0 else "neg"),
    ]
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>코인 자동매매 리포트</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b0f16; --panel:#111722; --panel2:#171f2c; --line:#263244; --soft:#1c2635; --ink:#edf4ff; --muted:#8fa0ba; --green:#00d4a6; --red:#ff4d6d; --blue:#5b8cff; --amber:#f7b955; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:13px/1.45 "Segoe UI","Malgun Gothic",Arial,sans-serif; }}
    header {{ position:sticky; top:0; z-index:5; display:flex; align-items:center; justify-content:space-between; gap:16px; min-height:52px; padding:0 16px; background:#171d29; border-bottom:1px solid #313b4d; }}
    h1 {{ margin:0; font-size:16px; }}
    h2 {{ margin:0; padding:11px 12px; font-size:14px; border-bottom:1px solid var(--line); }}
    .brand {{ display:flex; align-items:center; gap:10px; min-width:0; }}
    .mark {{ width:22px; height:22px; border-radius:50%; background:#b42318; display:grid; place-items:center; font-weight:900; }}
    .top {{ display:flex; gap:16px; color:#c8d4e6; font-weight:800; white-space:nowrap; }}
    main {{ padding:8px; }}
    .hero {{ display:grid; grid-template-columns:minmax(260px,1.2fr) repeat(3,minmax(170px,.7fr)); gap:8px; margin-bottom:8px; }}
    .hero-card,.card,section {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; }}
    .hero-card {{ padding:16px; min-height:96px; }}
    .label {{ color:var(--muted); font-weight:700; font-size:12px; }}
    .amount {{ margin-top:8px; font-size:30px; line-height:1.1; font-weight:900; overflow-wrap:anywhere; }}
    .value {{ margin-top:7px; font-size:20px; font-weight:900; overflow-wrap:anywhere; }}
    .note {{ margin-top:6px; color:var(--muted); font-size:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin-bottom:8px; }}
    .card {{ padding:11px 12px; min-height:64px; }}
    .layout {{ display:grid; grid-template-columns:minmax(520px,1.3fr) minmax(320px,.7fr); gap:8px; align-items:start; }}
    .split {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:8px; margin-top:8px; }}
    section {{ overflow:hidden; }}
    .chart {{ padding:12px; }}
    .chart-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:8px; flex-wrap:wrap; }}
    .chart-title {{ font-weight:900; }}
    .muted {{ color:var(--muted); }}
    .state {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .state-item {{ padding:12px; border-right:1px solid var(--line); border-bottom:1px solid var(--line); }}
    .state-item:nth-child(2n) {{ border-right:0; }}
    .state-value {{ margin-top:6px; font-weight:900; overflow-wrap:anywhere; }}
    .table-wrap {{ overflow:auto; max-height:38vh; -webkit-overflow-scrolling:touch; }}
    table {{ width:100%; min-width:720px; border-collapse:separate; border-spacing:0; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid var(--soft); vertical-align:top; background:var(--panel); white-space:nowrap; }}
    th {{ position:sticky; top:0; z-index:2; color:var(--muted); background:var(--panel2); text-align:left; font-size:12px; }}
    td.text {{ white-space:normal; min-width:260px; }}
    td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .ops {{ max-height:310px; overflow:auto; font-family:Consolas,"Cascadia Mono",monospace; font-size:12px; background:#0e131c; }}
    .ops-line {{ padding:4px 10px; border-bottom:1px solid #172033; color:#c9d5e8; }}
    .time {{ color:var(--muted); }}
    .event {{ color:var(--green); font-weight:800; }}
    .pos,.ok {{ color:var(--green); font-weight:900; }}
    .neg {{ color:var(--red); font-weight:900; }}
    .warn {{ color:var(--amber); font-weight:900; }}
    .buy {{ color:var(--blue); font-weight:900; }}
    .sell {{ color:var(--red); font-weight:900; }}
    .empty {{ padding:16px; color:var(--muted); }}
    .equity-chart {{ display:block; width:100%; min-height:260px; max-height:390px; background:#0e131c; border:1px solid #222d3d; border-radius:4px; }}
    @media (max-width:1000px) {{ header {{ position:static; display:block; padding:12px; }} .top {{ margin-top:10px; flex-wrap:wrap; white-space:normal; }} .hero,.layout,.split {{ grid-template-columns:1fr; }} .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width:560px) {{ main {{ padding:6px; }} .grid,.state {{ grid-template-columns:1fr; }} .amount {{ font-size:26px; }} table {{ min-width:640px; }} .equity-chart {{ min-height:220px; }} }}
  </style>
</head>
<body>
  <header>
    <div class="brand"><span class="mark">↗</span><h1>코인 오토 트레이딩 시스템</h1></div>
    <div class="top"><span>보유 KRW {html.escape(krw(cash))}</span><span>누적 수익률 <span class="{change_class}">{html.escape(pct(float(portfolio["return_pct"]) / 100.0))}</span></span></div>
  </header>
  <main>
    <div class="hero">
      <div class="hero-card"><div class="label">현재 평가금액</div><div class="amount">{html.escape(krw(float(portfolio["current_equity"])))}</div><div class="note">마지막 갱신 {html.escape(refreshed_at)}</div></div>
      <div class="hero-card"><div class="label">시작 대비 손익</div><div class="value {change_class}">{html.escape(krw(change))}</div><div class="note">실현/미실현 합산</div></div>
      <div class="hero-card"><div class="label">현재 수익률</div><div class="value {change_class}">{html.escape(pct(float(portfolio["return_pct"]) / 100.0))}</div><div class="note">100만원 기준</div></div>
      <div class="hero-card"><div class="label">보유 상태</div><div class="value">{len(open_positions)}개 보유</div><div class="note">현금 {html.escape(krw(cash))}</div></div>
    </div>
    <div class="grid">{render_cards(cards)}</div>
    <div class="layout">
      <section class="chart"><div class="chart-head"><div><div class="chart-title">누적 수익률</div><div class="muted">핵심 성과만 표시합니다. 상세 리포트는 /api/report?full=1</div></div></div>{render_equity_chart(trades, events)}</section>
      <section><h2>운영 상태</h2><div class="state">{''.join(render_state_item(label, value, klass) for label, value, klass in state_rows)}</div></section>
    </div>
    <div class="split">
      <section><h2>오픈 포지션</h2>{render_open_positions(events)}</section>
      <section><h2>실시간 판단 로그</h2>{render_compact_ops_log(events)}</section>
    </div>
    <div class="split">
      <section><h2>최근 체결</h2>{render_compact_trade_table(trades)}</section>
      <section><h2>핵심 진단</h2>{render_compact_diagnosis(metrics, market_mode)}</section>
    </div>
  </main>
</body>
</html>"""


def render_console_report(trades: list[TradeRow], events: list[dict[str, Any]]) -> str:
    trades = sorted(trades, key=lambda trade: parse_sort_time(trade.timestamp))
    events = sorted(events, key=lambda event: parse_sort_time(str(event.get("timestamp", ""))))
    metrics = calculate_metrics(trades)
    pairs = pair_round_trips(trades)
    last_event = events[-1] if events else {}
    last_payload = last_event.get("payload", {}) if isinstance(last_event, dict) else {}
    risk = last_payload.get("risk", {}) if isinstance(last_payload, dict) else {}
    portfolio = portfolio_summary(trades, events)
    refreshed_at = display_time(str(last_event.get("timestamp", ""))) if last_event else "아직 없음"
    open_positions = find_latest_positions(events)
    latest_context = find_latest_decision_context(events)
    market_mode = str(latest_context.get("market_mode", "unknown")) if latest_context else "unknown"
    session_label = str(latest_context.get("session_label", "-")) if latest_context else "-"
    mode_class = "ok" if market_mode == "risk_on" else "warn" if market_mode in {"risk_off", "capital_protect"} else ""
    cash = portfolio.get("cash", portfolio["current_equity"])
    change = float(portfolio["change_amount"])
    return_class = "pos" if change >= 0 else "neg"
    halted = bool(risk.get("halted"))
    operation_state = "중지" if halted else "운영 중"
    operation_class = "warn" if halted else "ok"
    target_amount = float(portfolio["starting_cash"]) * 0.03
    target_progress = change / target_amount if target_amount else 0.0
    target_bar_width = max(0.0, min(100.0, target_progress * 100.0))
    cards = [
        ("현재 평가금액", krw(float(portfolio["current_equity"]))),
        ("누적 수익률", pct(float(portfolio["return_pct"]) / 100.0)),
        ("누적 수익금", krw(change)),
        ("승률", pct(float(metrics["win_rate"]))),
        ("완료 거래", str(metrics["exit_count"])),
        ("손익비", ratio(float(metrics["payoff_ratio"]))),
        ("기대값", krw(float(metrics["expectancy"]))),
        ("최대 낙폭", krw(float(metrics["max_drawdown"]))),
        ("오픈 포지션", f"{len(open_positions)}개"),
        ("가용 현금", krw(float(cash))),
        ("오늘 진입", str(risk.get("entries_today", 0))),
        ("연속 손실", str(risk.get("consecutive_losses", 0))),
    ]
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>코인 자동매매 운용 콘솔</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f16;
      --panel: #111722;
      --panel-2: #151c29;
      --panel-3: #0e131c;
      --ink: #edf4ff;
      --muted: #7f8da3;
      --line: #263244;
      --soft-line: #1c2635;
      --green: #00d4a6;
      --red: #ff4d6d;
      --blue: #5b8cff;
      --amber: #f7b955;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: "Segoe UI", "Malgun Gothic", Arial, sans-serif; font-size: 13px; }}
    main {{ margin: 0; padding: 0 0 36px; }}
    header {{ min-height: 52px; display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 0 14px; background: #171d29; border-bottom: 1px solid #313b4d; position: sticky; top: 0; z-index: 20; }}
    h1 {{ margin: 0; font-size: 15px; color: #f8fbff; font-weight: 750; }}
    h2 {{ font-size: 14px; margin: 0; padding: 12px 14px; border-bottom: 1px solid var(--line); color: #f8fbff; }}
    .brand {{ display: flex; align-items: center; gap: 10px; min-width: 0; flex-wrap: wrap; }}
    .brand-mark {{ width: 22px; height: 22px; border-radius: 50%; background: #b42318; display: inline-flex; align-items: center; justify-content: center; color: #fff; font-size: 12px; font-weight: 900; }}
    .nav-tabs {{ display: flex; align-items: center; gap: 28px; margin-left: 20px; height: 52px; overflow-x: auto; scrollbar-width: thin; }}
    .nav-tab {{ height: 52px; display: flex; align-items: center; color: #738198; border-bottom: 2px solid transparent; font-weight: 650; }}
    .nav-tab.active {{ color: #f4f8ff; border-bottom-color: #d8e3f5; }}
    .top-meta {{ display: flex; align-items: center; gap: 18px; color: #c8d4e6; font-weight: 700; white-space: nowrap; flex-wrap: wrap; justify-content: flex-end; }}
    .workspace {{ padding: 8px; }}
    .portfolio-hero {{ display: grid; grid-template-columns: minmax(280px, 1.25fr) repeat(3, minmax(170px, .75fr)); gap: 8px; margin-bottom: 8px; }}
    .portfolio-main, .portfolio-sub {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; min-height: 92px; }}
    .portfolio-title, .label, .state-name {{ color: var(--muted); font-size: 12px; font-weight: 650; }}
    .portfolio-title {{ margin-bottom: 9px; }}
    .portfolio-amount {{ font-size: 30px; line-height: 1.12; font-weight: 850; overflow-wrap: anywhere; }}
    .portfolio-value {{ font-size: 22px; line-height: 1.2; font-weight: 850; overflow-wrap: anywhere; }}
    .portfolio-note {{ margin-top: 8px; color: var(--muted); font-size: 12px; line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; margin-bottom: 8px; }}
    .card, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }}
    .card {{ padding: 11px 12px; min-height: 64px; }}
    .value {{ font-size: 18px; font-weight: 850; margin-top: 7px; overflow-wrap: anywhere; }}
    section {{ margin-top: 8px; overflow: hidden; }}
    .dashboard-grid {{ display: grid; grid-template-columns: minmax(520px, 1.35fr) minmax(340px, .65fr); gap: 8px; align-items: start; }}
    .split-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 8px; }}
    .table-wrap {{ width: 100%; overflow: auto; -webkit-overflow-scrolling: touch; max-height: 48vh; border-top: 1px solid var(--soft-line); }}
    table {{ width: 100%; min-width: 900px; border-collapse: separate; border-spacing: 0; table-layout: auto; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--soft-line); text-align: left; vertical-align: top; background: #111722; }}
    th {{ position: sticky; top: 0; z-index: 3; color: #91a0b8; font-size: 12px; background: #171f2c; white-space: nowrap; font-weight: 750; }}
    td {{ white-space: nowrap; }}
    td.text {{ white-space: normal; min-width: 260px; line-height: 1.45; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    tr:last-child td {{ border-bottom: 0; }}
    .buy, .blue {{ color: var(--blue); font-weight: 800; }}
    .sell, .neg {{ color: var(--red); font-weight: 800; }}
    .pos, .ok {{ color: var(--green); font-weight: 800; }}
    .warn {{ color: var(--amber); font-weight: 800; }}
    .state-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0; }}
    .state-item {{ padding: 13px 14px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
    .state-item:nth-child(2n) {{ border-right: 0; }}
    .state-value {{ margin-top: 6px; font-weight: 850; overflow-wrap: anywhere; }}
    .diagnosis {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0; }}
    .diagnosis div {{ padding: 13px 14px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); line-height: 1.55; color: #c8d4e6; }}
    .diagnosis div:nth-child(2n) {{ border-right: 0; }}
    .chart-panel {{ padding: 12px; }}
    .chart-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }}
    .chart-title {{ color: #eef5ff; font-weight: 850; }}
    .chart-tabs {{ display: inline-flex; gap: 4px; color: #738198; font-weight: 750; background: #0e131c; border: 1px solid #273448; border-radius: 6px; padding: 3px; max-width: 100%; overflow-x: auto; }}
    .chart-tabs button {{ appearance: none; border: 0; border-radius: 4px; background: transparent; color: #8ea0ba; cursor: pointer; font: inherit; min-height: 30px; padding: 5px 10px; white-space: nowrap; }}
    .chart-tabs button.active {{ background: #1d2a3d; color: var(--blue); }}
    .equity-chart {{ display: block; width: 100%; height: auto; min-height: 240px; max-height: 420px; background: #0e131c; border: 1px solid #222d3d; border-radius: 4px; }}
    .progress {{ height: 8px; background: #0b111a; border: 1px solid #243247; border-radius: 999px; overflow: hidden; margin-top: 10px; }}
    .progress-fill {{ height: 100%; background: linear-gradient(90deg, #00a884, #00d4a6); width: {target_bar_width:.2f}%; }}
    .ops-log {{ max-height: 444px; overflow: auto; font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px; line-height: 1.55; background: #0e131c; }}
    .ops-line {{ padding: 3px 10px; border-bottom: 1px solid #171f2c; color: #c9d5e8; }}
    .ops-line .time {{ color: #7f8da3; }}
    .ops-line .event {{ color: var(--green); }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 1000px) {{
      header {{ height: auto; padding: 12px; display: block; position: static; }}
      .top-meta {{ justify-content: flex-start; margin-top: 10px; white-space: normal; }}
      .nav-tabs {{ margin-left: 0; margin-top: 10px; height: 32px; }}
      .nav-tab {{ height: 32px; }}
      .portfolio-hero, .dashboard-grid, .split-grid {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      table {{ min-width: 760px; }}
    }}
    @media (max-width: 560px) {{
      .grid, .state-grid, .diagnosis {{ grid-template-columns: 1fr; }}
      .portfolio-amount {{ font-size: 26px; }}
      .workspace {{ padding: 6px; }}
      .portfolio-main, .portfolio-sub, .card {{ padding: 12px; }}
      .chart-panel {{ padding: 10px; }}
      .equity-chart {{ min-height: 210px; }}
      table {{ min-width: 680px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="brand">
      <span class="brand-mark">↗</span>
      <h1>코인 오토 트레이딩 시스템</h1>
      <nav class="nav-tabs">
        <span class="nav-tab">봇</span>
        <span class="nav-tab active">수익 리포트</span>
        <span class="nav-tab">주문 기록</span>
      </nav>
    </div>
    <div class="top-meta">
      <span>보유 KRW&nbsp; {html.escape(krw(float(cash)))}</span>
      <span>누적 수익률&nbsp; <span class="{return_class}">{html.escape(pct(float(portfolio["return_pct"]) / 100.0))}</span></span>
    </div>
  </header>
  <div class="workspace">
    {render_portfolio_hero(portfolio)}
    <div class="grid">{render_cards(cards)}</div>
    <div class="dashboard-grid">
      <section class="chart-panel">
        <div class="chart-head">
          <div>
            <div class="chart-title">누적 수익률</div>
            <div class="muted">마지막 갱신: {html.escape(refreshed_at)}</div>
          </div>
          <div class="chart-tabs" role="group" aria-label="차트 범위">
            <button type="button" class="active" data-chart-window="all">전체</button>
            <button type="button" data-chart-window="60">최근 60</button>
            <button type="button" data-chart-window="240">최근 240</button>
          </div>
        </div>
        {render_equity_chart(trades, events)}
      </section>
      <section>
        <h2>운영 상태</h2>
        <div class="state-grid">
          {render_state_item("상태", operation_state, operation_class)}
          {render_state_item("오픈 포지션", f"{len(open_positions)}개", "")}
          {render_state_item("시장 모드", market_mode, mode_class)}
          {render_state_item("시장 세션", session_label, "")}
          {render_state_item("오늘 진입", str(risk.get("entries_today", 0)), "")}
          {render_state_item("연속 손실", str(risk.get("consecutive_losses", 0)), "warn" if int(risk.get("consecutive_losses", 0) or 0) else "")}
          {render_state_item("3% 목표 진행률", pct(target_progress), return_class)}
          {render_state_item("목표 수익금", krw(target_amount), "")}
        </div>
        <div style="padding: 0 14px 14px;"><div class="progress"><div class="progress-fill"></div></div></div>
      </section>
    </div>
    <div class="dashboard-grid">
      <section><h2>오픈 포지션</h2>{render_open_positions(events)}</section>
      <section><h2>실시간 판단 로그</h2>{render_ops_log_current(events[-120:], trades[-80:])}</section>
    </div>
    <div class="split-grid">
      <section><h2>최근 체결</h2>{render_trade_table(trades[-80:])}</section>
      <section><h2>수익률 개선 진단</h2>{render_diagnosis(metrics)}</section>
    </div>
    <section><h2>필터 차단 분석</h2>{render_filter_block_table(events)}</section>
    <section><h2>통합 성과 지표</h2>{render_metric_table(metrics)}</section>
    <section><h2>매수 이유별 성과</h2>{render_group_table(group_by_entry_reason(pairs), "매수 이유")}</section>
    <section><h2>청산 이유별 성과</h2>{render_group_table(group_by_exit_reason(pairs), "청산 이유")}</section>
    <section><h2>시장 데이터</h2>{render_market_context_table(events)}</section>
    <section><h2>표본 진단</h2>{render_sample_law(metrics)}</section>
  </div>
</main>
{responsive_report_script()}
</body>
</html>
"""
    metrics = calculate_metrics(trades)
    pairs = pair_round_trips(trades)
    last_event = events[-1] if events else {}
    last_payload = last_event.get("payload", {}) if isinstance(last_event, dict) else {}
    risk = last_payload.get("risk", {}) if isinstance(last_payload, dict) else {}
    position = last_payload.get("position", {}) if isinstance(last_payload, dict) else {}
    cash = last_payload.get("cash") if isinstance(last_payload, dict) else None
    portfolio = portfolio_summary(trades, events)
    status = status_message(risk, position, float(metrics["total_realized"]))

    cards = [
        ("현재 평가금액", krw(portfolio["current_equity"])),
        ("시작 대비", krw(portfolio["change_amount"])),
        ("현재 수익률", pct(float(portfolio["return_pct"]) / 100.0)),
        ("전체 거래", str(len(trades))),
        ("완료 거래", str(metrics["exit_count"])),
        ("승률", pct(float(metrics["win_rate"]))),
        ("기대값", krw(float(metrics["expectancy"]))),
        ("손익비", ratio(float(metrics["payoff_ratio"]))),
        ("최대 낙폭", krw(float(metrics["max_drawdown"]))),
        ("실현 손익", krw(float(metrics["total_realized"]))),
        ("현재 현금", krw(cash) if isinstance(cash, (int, float)) else "-"),
    ]

    refreshed_at = display_time(str(last_event.get("timestamp", "아직 없음"))) if last_event else "아직 없음"
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>코인 페이퍼 트레이딩 리포트</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #101828;
      --muted: #667085;
      --line: #d0d5dd;
      --soft-line: #eaecf0;
      --green: #067647;
      --red: #b42318;
      --blue: #175cd3;
      --amber: #b54708;
      --header: #0b1220;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: "Segoe UI", "Malgun Gothic", Arial, sans-serif; font-size: 14px; }}
    main {{ max-width: 1380px; margin: 0 auto; padding: 26px 18px 52px; }}
    header {{ display: flex; align-items: end; justify-content: space-between; gap: 18px; margin-bottom: 16px; }}
    h1 {{ margin: 0; font-size: 28px; letter-spacing: 0; color: var(--header); }}
    h2 {{ font-size: 15px; margin: 0; padding: 13px 16px; border-bottom: 1px solid var(--line); color: var(--header); }}
    .subtitle, .muted {{ color: var(--muted); }}
    .notice {{ background: #f0f6ff; border: 1px solid #b2ccff; border-radius: 8px; padding: 13px 16px; margin-bottom: 16px; line-height: 1.55; }}
    .notice strong {{ color: var(--blue); }}
    .portfolio-hero {{ display: grid; grid-template-columns: minmax(260px, 1.4fr) repeat(3, minmax(160px, 1fr)); gap: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin-bottom: 16px; overflow: hidden; box-shadow: 0 1px 2px rgba(16,24,40,.05); }}
    .portfolio-main, .portfolio-sub {{ padding: 16px 18px; border-right: 1px solid var(--line); }}
    .portfolio-sub:last-child {{ border-right: 0; }}
    .portfolio-title {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .portfolio-amount {{ font-size: 34px; line-height: 1.12; font-weight: 760; letter-spacing: 0; overflow-wrap: anywhere; }}
    .portfolio-value {{ font-size: 22px; line-height: 1.2; font-weight: 760; overflow-wrap: anywhere; }}
    .portfolio-note {{ margin-top: 8px; color: var(--muted); font-size: 12px; line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
    .card {{ padding: 14px; min-height: 72px; }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{ font-size: 20px; font-weight: 760; margin-top: 8px; overflow-wrap: anywhere; }}
    section {{ margin-top: 18px; overflow: hidden; }}
    .table-shell {{ overflow: hidden; }}
    .table-wrap {{ width: 100%; overflow: auto; -webkit-overflow-scrolling: touch; max-height: 62vh; border-top: 1px solid var(--soft-line); }}
    table {{ width: 100%; min-width: 980px; border-collapse: separate; border-spacing: 0; table-layout: auto; }}
    th, td {{ padding: 9px 12px; border-bottom: 1px solid var(--soft-line); text-align: left; vertical-align: top; background: #fff; }}
    th {{ position: sticky; top: 0; z-index: 3; color: #475467; font-size: 12px; background: #f9fafb; white-space: nowrap; font-weight: 650; }}
    th:first-child, td:first-child {{ position: sticky; left: 0; z-index: 2; box-shadow: 1px 0 0 var(--soft-line); }}
    th:first-child {{ z-index: 4; }}
    td {{ white-space: nowrap; }}
    td.text {{ white-space: normal; min-width: 260px; line-height: 1.45; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    tr:last-child td {{ border-bottom: 0; }}
    .buy, .blue {{ color: var(--blue); font-weight: 700; }}
    .sell, .neg {{ color: var(--red); font-weight: 700; }}
    .pos, .ok {{ color: var(--green); font-weight: 700; }}
    .warn {{ color: var(--amber); font-weight: 700; }}
    .state-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0; }}
    .state-item {{ padding: 14px 16px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
    .state-item:nth-child(4n) {{ border-right: 0; }}
    .state-name {{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
    .state-value {{ font-weight: 700; overflow-wrap: anywhere; }}
    .diagnosis {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0; }}
    .diagnosis div {{ padding: 14px 16px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); line-height: 1.55; }}
    .diagnosis div:nth-child(2n) {{ border-right: 0; }}
    @media (max-width: 900px) {{
      main {{ padding: 20px 12px 40px; }}
      header {{ display: block; }}
      .portfolio-hero {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .portfolio-main {{ grid-column: 1 / -1; }}
      .grid, .state-grid, .diagnosis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      table {{ min-width: 900px; }}
    }}
    @media (max-width: 560px) {{
      h1 {{ font-size: 24px; }}
      .portfolio-hero {{ grid-template-columns: 1fr; }}
      .portfolio-main, .portfolio-sub {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .portfolio-amount {{ font-size: 29px; }}
      .grid, .state-grid, .diagnosis {{ grid-template-columns: 1fr; }}
      .table-wrap {{ max-height: 70vh; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>코인 페이퍼 트레이딩 리포트</h1>
      <div class="subtitle">수익률보다 먼저 기대값, 손익비, 표본 수, 낙폭을 함께 보는 화면입니다.</div>
    </div>
    <div class="subtitle">마지막 갱신: {html.escape(refreshed_at)}</div>
  </header>
  {render_portfolio_hero(portfolio)}
  <div class="notice"><strong>현재 모드: 모의거래</strong><br>이 화면은 수익을 보장하지 않습니다. 특히 완료 거래가 적을 때는 승률과 기대값이 크게 흔들리므로, 대수의 법칙 관점에서 표본 수와 신뢰구간을 먼저 보세요.</div>
  <div class="grid">{render_cards(cards)}</div>
  <section><h2>현재 상태</h2>{render_state_panel(risk, position, status)}</section>
  <section><h2>오픈 포지션</h2>{render_open_positions(events)}</section>
  <section><h2>수익률 개선 진단</h2>{render_diagnosis(metrics)}</section>
  <section><h2>대수의 법칙 기반 표본 진단</h2>{render_sample_law(metrics)}</section>
  <section><h2>리서치 반영 체크리스트</h2>{render_research_checklist()}</section>
  <section><h2>BTC 분석 프레임워크 적용 상태</h2>{render_btc_framework()}</section>
  <section><h2>자동 수집 시장 데이터</h2>{render_market_context_table(events)}</section>
  <section><h2>AI 의사결정자 로그</h2>{render_ai_decision_table(events)}</section>
  <section><h2>통합 성과 지표</h2>{render_metric_table(metrics)}</section>
  <section><h2>매수 이유별 성과</h2>{render_group_table(group_by_entry_reason(pairs), "매수 이유")}</section>
  <section><h2>청산 이유별 성과</h2>{render_group_table(group_by_exit_reason(pairs), "청산 이유")}</section>
  <section><h2>필터 차단 분석</h2>{render_filter_block_table(events)}</section>
  <section><h2>시간대별 성과</h2>{render_group_table(group_by_exit_hour(pairs), "시간대")}</section>
  <section><h2>최근 거래</h2>{render_trade_table(trades[-60:])}</section>
  <section><h2>최근 판단 로그</h2>{render_event_table(events[-40:])}</section>
</main>
</body>
</html>
"""


def calculate_metrics(trades: list[TradeRow]) -> dict[str, float | int]:
    exits = [trade for trade in trades if trade.side == "sell"]
    pnls = [trade.realized_pnl for trade in exits]
    wins = [trade.realized_pnl for trade in exits if trade.realized_pnl > 0]
    losses = [trade.realized_pnl for trade in exits if trade.realized_pnl < 0]
    total_realized = sum(pnls)
    exit_count = len(exits)
    win_rate = len(wins) / exit_count if exit_count else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    payoff_ratio = avg_win / abs(avg_loss) if avg_loss else 0.0
    expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss) if exit_count else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss else 0.0
    expectancy_se = standard_error(pnls)
    ci_low, ci_high = wilson_interval(win_rate, exit_count)
    return {
        "exit_count": exit_count,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": calculate_max_drawdown(pnls),
        "max_consecutive_losses": calculate_max_consecutive_losses(pnls),
        "expectancy_se": expectancy_se,
        "expectancy_ci_low": expectancy - 1.96 * expectancy_se if exit_count >= 2 else 0.0,
        "expectancy_ci_high": expectancy + 1.96 * expectancy_se if exit_count >= 2 else 0.0,
        "win_rate_ci_low": ci_low,
        "win_rate_ci_high": ci_high,
        "total_fee": sum(trade.fee for trade in trades),
        "total_realized": total_realized,
    }


def pair_round_trips(trades: list[TradeRow]) -> list[RoundTrip]:
    pairs: list[RoundTrip] = []
    open_entries: dict[str, TradeRow] = {}
    for trade in trades:
        if trade.side == "buy":
            open_entries[trade.market] = trade
        elif trade.side == "sell":
            open_entry = open_entries.pop(trade.market, None)
            if open_entry is not None:
                pairs.append(RoundTrip(open_entry, trade))
    return pairs


def calculate_max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return max_drawdown


def calculate_max_consecutive_losses(pnls: list[float]) -> int:
    current = 0
    worst = 0
    for pnl in pnls:
        if pnl < 0:
            current += 1
            worst = max(worst, current)
        elif pnl > 0:
            current = 0
    return worst


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def wilson_interval(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    denominator = 1.0 + (z * z / n)
    centre = (rate + (z * z / (2 * n))) / denominator
    margin = (z * math.sqrt((rate * (1 - rate) / n) + (z * z / (4 * n * n)))) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def render_cards(cards: list[tuple[str, str]]) -> str:
    return "\n".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )


def render_equity_chart(trades: list[TradeRow], events: list[dict[str, Any]]) -> str:
    starting_cash = find_starting_cash(events) or 1_000_000.0
    points = equity_curve_points(trades, events, starting_cash)
    width = 960
    height = 310
    pad_left = 54
    pad_right = 16
    pad_top = 18
    pad_bottom = 34
    if len(points) < 2:
        points = [(0, starting_cash), (1, starting_cash)]
    values = [value for _, value in points]
    low = min(values + [starting_cash])
    high = max(values + [starting_cash])
    if math.isclose(low, high):
        low -= starting_cash * 0.002
        high += starting_cash * 0.002
    span = high - low
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def xy(index: int, value: float) -> tuple[float, float]:
        x = pad_left + (index / max(1, len(points) - 1)) * plot_w
        y = pad_top + (high - value) / span * plot_h
        return x, y

    line_points = [xy(index, value) for index, (_, value) in enumerate(points)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in line_points)
    area = f"{pad_left},{height - pad_bottom} " + line + f" {width - pad_right},{height - pad_bottom}"
    grid_values = [low, low + span * 0.5, high]
    grid = []
    for value in grid_values:
        _, y = xy(0, value)
        grid.append(f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" stroke="#334155" stroke-dasharray="3 4"/>')
        grid.append(f'<text x="8" y="{y + 4:.1f}" fill="#7f8da3" font-size="12">{html.escape(short_krw(value))}</text>')
    latest = values[-1]
    latest_x, latest_y = line_points[-1]
    values_payload = html.escape(json.dumps(values), quote=True)
    return f"""
    <svg class="equity-chart" viewBox="0 0 {width} {height}" role="img" aria-label="누적 수익률 차트" data-values="{values_payload}" data-starting-cash="{starting_cash:.8f}">
      <defs>
        <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#00d4a6" stop-opacity="0.55"/>
          <stop offset="100%" stop-color="#00d4a6" stop-opacity="0.05"/>
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="{width}" height="{height}" fill="#0e131c"/>
      {''.join(grid)}
      <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#3b4658"/>
      <polyline points="{area}" fill="url(#equityFill)" stroke="none"/>
      <polyline points="{line}" fill="none" stroke="#00d4a6" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
      <circle cx="{latest_x:.1f}" cy="{latest_y:.1f}" r="4" fill="#00d4a6"/>
      <text x="{width - pad_right}" y="{height - 10}" fill="#7f8da3" font-size="12" text-anchor="end">현재 {html.escape(krw(latest))}</text>
    </svg>
    """


def responsive_report_script() -> str:
    return """
<script>
(() => {
  const svg = document.querySelector(".equity-chart");
  const buttons = Array.from(document.querySelectorAll("[data-chart-window]"));
  if (!svg || buttons.length === 0) return;

  const allValues = JSON.parse(svg.dataset.values || "[]").map(Number).filter(Number.isFinite);
  const startingCash = Number(svg.dataset.startingCash || 1000000);
  const ns = "http://www.w3.org/2000/svg";
  const width = 960;
  const height = 310;
  const padLeft = 54;
  const padRight = 16;
  const padTop = 18;
  const padBottom = 34;

  const fmtKrw = (value) => {
    const sign = value < 0 ? "-" : "";
    const abs = Math.abs(value);
    if (abs >= 100000000) return `${sign}${(abs / 100000000).toFixed(1)}억`;
    if (abs >= 10000) return `${sign}${Math.round(abs / 10000)}만`;
    return `${sign}${Math.round(abs)}`;
  };

  const node = (name, attrs = {}, text = "") => {
    const el = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
    if (text) el.textContent = text;
    return el;
  };

  const draw = (windowSize) => {
    let values = allValues.length ? allValues : [startingCash, startingCash];
    if (windowSize !== "all") {
      values = values.slice(-Math.max(2, Number(windowSize) || values.length));
      if (values.length === 1) values = [startingCash, values[0]];
    }
    if (values.length < 2) values = [startingCash, startingCash];

    let low = Math.min(...values, startingCash);
    let high = Math.max(...values, startingCash);
    if (Math.abs(high - low) < 1) {
      low -= startingCash * 0.002;
      high += startingCash * 0.002;
    }
    const span = high - low;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;
    const xy = (index, value) => {
      const x = padLeft + (index / Math.max(1, values.length - 1)) * plotW;
      const y = padTop + ((high - value) / span) * plotH;
      return [x, y];
    };

    const points = values.map((value, index) => xy(index, value));
    const line = points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    const area = `${padLeft},${height - padBottom} ${line} ${width - padRight},${height - padBottom}`;
    svg.replaceChildren();

    const defs = node("defs");
    const gradient = node("linearGradient", { id: "equityFill", x1: "0", y1: "0", x2: "0", y2: "1" });
    gradient.append(node("stop", { offset: "0%", "stop-color": "#00d4a6", "stop-opacity": "0.55" }));
    gradient.append(node("stop", { offset: "100%", "stop-color": "#00d4a6", "stop-opacity": "0.05" }));
    defs.append(gradient);
    svg.append(defs);
    svg.append(node("rect", { x: 0, y: 0, width, height, fill: "#0e131c" }));

    [low, low + span * 0.5, high].forEach((value) => {
      const [, y] = xy(0, value);
      svg.append(node("line", { x1: padLeft, y1: y.toFixed(1), x2: width - padRight, y2: y.toFixed(1), stroke: "#334155", "stroke-dasharray": "3 4" }));
      svg.append(node("text", { x: 8, y: (y + 4).toFixed(1), fill: "#7f8da3", "font-size": 12 }, fmtKrw(value)));
    });

    const [latestX, latestY] = points[points.length - 1];
    const latest = values[values.length - 1];
    svg.append(node("line", { x1: padLeft, y1: padTop, x2: padLeft, y2: height - padBottom, stroke: "#3b4658" }));
    svg.append(node("polyline", { points: area, fill: "url(#equityFill)", stroke: "none" }));
    svg.append(node("polyline", { points: line, fill: "none", stroke: "#00d4a6", "stroke-width": 3, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    svg.append(node("circle", { cx: latestX.toFixed(1), cy: latestY.toFixed(1), r: 4, fill: "#00d4a6" }));
    svg.append(node("text", { x: width - padRight, y: height - 10, fill: "#7f8da3", "font-size": 12, "text-anchor": "end" }, `현재 ${Math.round(latest).toLocaleString("ko-KR")} KRW`));
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((item) => item.classList.toggle("active", item === button));
      draw(button.dataset.chartWindow || "all");
    });
  });
})();
</script>
"""


def equity_curve_points(trades: list[TradeRow], events: list[dict[str, Any]], starting_cash: float) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = [(0, starting_cash)]
    for event in events:
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        equity = to_float(payload.get("equity"))
        if equity is not None:
            points.append((len(points), equity))
    if len(points) == 1:
        realized = 0.0
        for trade in trades:
            if trade.side == "sell":
                realized += trade.realized_pnl
                points.append((len(points), starting_cash + realized))
    return points[-160:]


def render_ops_log(events: list[dict[str, Any]]) -> str:
    if not events:
        return empty_block("아직 판단 로그가 없습니다.")
    lines = []
    for event in reversed(events):
        name = str(event.get("event", ""))
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        if name == "tick":
            signal = payload.get("signal", {})
            side = signal.get("side", "-") if isinstance(signal, dict) else "-"
            reason = signal.get("reason", "-") if isinstance(signal, dict) else "-"
            text = f'{payload.get("market", "-")} {side} · {korean_reason(str(reason))}'
        elif name in {"fill", "forced_exit"}:
            fill = payload.get("fill", {})
            text = f'{fill.get("market", "-")} {korean_side(str(fill.get("side", "-")))} 체결 · 손익 {krw(to_float(fill.get("realized_pnl")))}'
        elif name == "market_scan":
            reason = korean_reason(str(payload.get("reason", "")))
            blocked = payload.get("blocked_reasons", {})
            if isinstance(blocked, dict) and blocked:
                top_reason = max(blocked.items(), key=lambda item: int(item[1]))[0]
                reason = f"{reason} · 주 차단: {korean_reason(str(top_reason))}"
            text = f'스캔 {payload.get("markets_scanned", 0)}개 · 후보 {payload.get("candidates", 0)}개 · {reason}'
        elif name == "watch_error":
            text = f'오류 · {payload.get("error", "-")}'
        else:
            continue
        lines.append(
            f'<div class="ops-line"><span class="time">{html.escape(display_time(str(event.get("timestamp", ""))))}</span> '
            f'<span class="event">{html.escape(korean_event(name))}</span> {html.escape(text)}</div>'
        )
        if len(lines) >= 80:
            break
    return '<div class="ops-log">' + "".join(lines) + "</div>"


def render_ops_log_current(events: list[dict[str, Any]], trades: list[TradeRow]) -> str:
    if not events and not trades:
        return empty_block("아직 판단 로그가 없습니다.")
    entries: list[tuple[datetime, str, str, str]] = []
    for trade in trades:
        text = (
            f"{trade.market} {korean_side(trade.side)} 체결 · "
            f"손익 {krw(trade.realized_pnl)} · {korean_reason(trade.reason)}"
        )
        entries.append((parse_sort_time(trade.timestamp), trade.timestamp, "체결", text))
    for event in events:
        name = str(event.get("event", ""))
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        if name == "state_snapshot":
            positions = payload.get("positions", {})
            position_count = len(positions) if isinstance(positions, dict) else 0
            risk = payload.get("risk", {}) if isinstance(payload.get("risk"), dict) else {}
            state = "중지" if risk.get("halted") else "운영 중"
            text = (
                f"상태 동기화 · tick {payload.get('tick', '-')} · {state} · "
                f"오픈 포지션 {position_count}개 · 평가 {krw(to_float(payload.get('equity')))}"
            )
        elif name == "tick":
            signal = payload.get("signal", {})
            side = signal.get("side", "-") if isinstance(signal, dict) else "-"
            reason = signal.get("reason", "-") if isinstance(signal, dict) else "-"
            text = f'{payload.get("market", "-")} {side} · {korean_reason(str(reason))}'
        elif name in {"fill", "forced_exit"}:
            fill = payload.get("fill", {})
            text = f'{fill.get("market", "-")} {korean_side(str(fill.get("side", "-")))} 체결 · 손익 {krw(to_float(fill.get("realized_pnl")))}'
        elif name == "market_scan":
            text = f'스캔 {payload.get("markets_scanned", 0)}개 · 후보 {payload.get("candidates", 0)}개 · {korean_reason(str(payload.get("reason", "")))}'
        elif name == "watch_error":
            text = f'오류 · {payload.get("error", "-")}'
        else:
            continue
        entries.append((parse_sort_time(str(event.get("timestamp", ""))), str(event.get("timestamp", "")), korean_event(name), text))
    lines = []
    for _, timestamp, label, text in sorted(entries, key=lambda item: item[0], reverse=True):
        lines.append(
            f'<div class="ops-line"><span class="time">{html.escape(display_time(timestamp))}</span> '
            f'<span class="event">{html.escape(label)}</span> {html.escape(text)}</div>'
        )
        if len(lines) >= 80:
            break
    return '<div class="ops-log">' + "".join(lines) + "</div>"


def portfolio_summary(trades: list[TradeRow], events: list[dict[str, Any]]) -> dict[str, float | str]:
    starting_cash = find_starting_cash(events) or 1_000_000.0
    latest_equity = find_latest_equity(events)
    latest_prices = find_latest_prices(events)
    cash = find_latest_cash(trades, events)
    positions_snapshot = find_latest_positions_snapshot(events)
    open_positions = positions_snapshot if positions_snapshot is not None else {}

    if positions_snapshot is None:
        for trade in trades:
            if trade.side == "buy":
                open_positions[trade.market] = {
                    "qty": trade.position_qty_after,
                    "avg_price": trade.price,
                }
            elif trade.side == "sell":
                open_positions.pop(trade.market, None)

    if latest_equity is None:
        if cash is not None and open_positions:
            latest_equity = cash + sum(
                (to_float(position.get("qty")) or 0.0)
                * (latest_prices.get(market) or to_float(position.get("avg_price")) or 0.0)
                for market, position in open_positions.items()
            )
        elif cash is not None:
            latest_equity = cash
        else:
            latest_equity = starting_cash

    change_amount = latest_equity - starting_cash
    return_pct = (change_amount / starting_cash) * 100.0 if starting_cash else 0.0
    position_count = sum(1 for position in open_positions.values() if (to_float(position.get("qty")) or 0.0) > 0)
    position_status = f"{position_count}개 보유" if position_count else "미보유"
    return {
        "starting_cash": starting_cash,
        "current_equity": latest_equity,
        "change_amount": change_amount,
        "return_pct": return_pct,
        "cash": cash if cash is not None else latest_equity,
        "position_qty": sum((to_float(position.get("qty")) or 0.0) for position in open_positions.values()),
        "position_status": position_status,
        "position_count": position_count,
    }


def calculate_daily_metrics(trades: list[TradeRow], events: list[dict[str, Any]], target_pct: float = 3.0) -> dict[str, float]:
    latest_equity = find_latest_equity(events)
    daily_start = find_latest_daily_starting_equity(events)
    if latest_equity is None:
        latest_equity = (portfolio_summary(trades, events)["current_equity"] if events or trades else 1_000_000.0)  # type: ignore[index]
    if daily_start is None or daily_start <= 0:
        daily_start = find_starting_cash(events) or 1_000_000.0
    change = float(latest_equity) - float(daily_start)
    return_pct = (change / float(daily_start)) * 100.0 if daily_start else 0.0
    return {
        "starting_equity": float(daily_start),
        "current_equity": float(latest_equity),
        "change_amount": change,
        "return_pct": return_pct,
        "target_amount": float(daily_start) * (target_pct / 100.0),
        "target_progress": return_pct / target_pct if target_pct else 0.0,
    }


def find_latest_daily_starting_equity(events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        risk = payload.get("risk")
        if isinstance(risk, dict):
            value = to_float(risk.get("starting_equity"))
            if value is not None:
                return value
    return None


def render_portfolio_hero(portfolio: dict[str, float | str]) -> str:
    change = float(portfolio["change_amount"])
    change_class = "pos" if change > 0 else "neg" if change < 0 else ""
    position_status = str(portfolio["position_status"])
    return f"""
  <div class="portfolio-hero">
    <div class="portfolio-main">
      <div class="portfolio-title">현재 평가금액</div>
      <div class="portfolio-amount">{html.escape(krw(float(portfolio["current_equity"])))}</div>
      <div class="portfolio-note">시작금액 {html.escape(krw(float(portfolio["starting_cash"])))} 기준입니다.</div>
    </div>
    <div class="portfolio-sub">
      <div class="portfolio-title">시작 대비 손익</div>
      <div class="portfolio-value {change_class}">{html.escape(krw(change))}</div>
      <div class="portfolio-note">실현/미실현 평가를 합친 기준입니다.</div>
    </div>
    <div class="portfolio-sub">
      <div class="portfolio-title">현재 수익률</div>
      <div class="portfolio-value {change_class}">{html.escape(pct(float(portfolio["return_pct"]) / 100.0))}</div>
      <div class="portfolio-note">100만원 대비 변화율입니다.</div>
    </div>
    <div class="portfolio-sub">
      <div class="portfolio-title">보유 상태</div>
      <div class="portfolio-value">{html.escape(position_status)}</div>
      <div class="portfolio-note">현금 {html.escape(krw(float(portfolio["cash"])))}</div>
    </div>
  </div>
"""


def find_starting_cash(events: list[dict[str, Any]]) -> float | None:
    for event in events:
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if isinstance(payload, dict):
            value = to_float(payload.get("starting_cash"))
            if value is not None:
                return value
    return None


def find_latest_equity(events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if isinstance(payload, dict):
            value = to_float(payload.get("equity"))
            if value is not None:
                return value
    return None


def find_latest_price(events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if isinstance(payload, dict):
            value = to_float(payload.get("price"))
            if value is not None:
                return value
    return None


def find_latest_prices(events: list[dict[str, Any]]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        last_prices = payload.get("last_prices")
        if isinstance(last_prices, dict):
            for market, value in last_prices.items():
                price = to_float(value)
                if price is not None and str(market) not in prices:
                    prices[str(market)] = price
        market = payload.get("market")
        price = to_float(payload.get("price"))
        if market and price is not None and str(market) not in prices:
            prices[str(market)] = price
        fill = payload.get("fill")
        if isinstance(fill, dict):
            fill_market = fill.get("market")
            fill_price = to_float(fill.get("price"))
            if fill_market and fill_price is not None and str(fill_market) not in prices:
                prices[str(fill_market)] = fill_price
    return prices


def find_latest_positions(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return find_latest_positions_snapshot(events) or {}


def find_latest_decision_context(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        context = payload.get("decision_context")
        if isinstance(context, dict):
            return context
    return {}


def find_latest_positions_snapshot(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        positions = payload.get("positions")
        if isinstance(positions, dict):
            return {
                str(market): dict(position)
                for market, position in positions.items()
                if isinstance(position, dict) and (to_float(position.get("qty")) or 0.0) > 0
            }
    return None


def find_latest_cash(trades: list[TradeRow], events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if isinstance(payload, dict):
            value = to_float(payload.get("cash"))
            if value is not None:
                return value
            fill = payload.get("fill")
            if isinstance(fill, dict):
                value = to_float(fill.get("cash_after"))
                if value is not None:
                    return value
    if trades:
        return trades[-1].cash_after
    return None


def render_state_panel(risk: dict[str, Any], position: dict[str, Any], status: str) -> str:
    halted = bool(risk.get("halted"))
    position_qty = to_float(position.get("qty")) or 0.0
    items = [
        ("운영 상태", "중지" if halted else "진행 가능", "warn" if halted else "ok"),
        ("상태 요약", status, ""),
        ("오늘 신규 진입", str(risk.get("entries_today", 0)), ""),
        ("오늘 청산", str(risk.get("exits_today", 0)), ""),
        ("연속 손실", str(risk.get("consecutive_losses", 0)), ""),
        ("중지 사유", korean_reason(str(risk.get("halt_reason") or "없음")), "warn" if halted else ""),
        ("보유 수량", f"{position_qty:.8f}", ""),
        ("평균 단가", krw(to_float(position.get("avg_price"))), ""),
    ]
    return '<div class="state-grid">' + "".join(render_state_item(*item) for item in items) + "</div>"


def render_state_item(label: str, value: str, klass: str) -> str:
    value_class = f' class="state-value {klass}"' if klass else ' class="state-value"'
    return f'<div class="state-item"><div class="state-name">{html.escape(label)}</div><div{value_class}>{html.escape(value)}</div></div>'


def render_compact_ops_log(events: list[dict[str, Any]], limit: int = 12) -> str:
    lines: list[str] = []
    for event in reversed(events):
        name = str(event.get("event", ""))
        if name not in {"tick", "trade", "state_snapshot", "forced_exit", "watch_error"}:
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        timestamp = display_time(str(event.get("timestamp", "")))
        if name == "tick":
            signal = payload.get("signal", {})
            signal_reason = signal.get("reason", "") if isinstance(signal, dict) else ""
            market = str(payload.get("market", "-"))
            candidates = payload.get("candidate_count", 0)
            equity = payload.get("equity", "")
            text = f"{market} 후보 {candidates}개 · {short_reason(str(signal_reason), 96)}"
            if equity != "":
                text += f" · 평가 {krw(to_float(equity) or 0.0)}"
        elif name == "trade":
            fill = payload.get("fill", payload)
            if isinstance(fill, dict):
                side = "매수" if fill.get("side") == "buy" else "매도"
                text = f"{fill.get('market', '-')} {side} · 손익 {krw(to_float(fill.get('realized_pnl')) or 0.0)}"
            else:
                text = "체결 기록"
        elif name == "state_snapshot":
            risk = payload.get("risk", {})
            tick = payload.get("tick", "-")
            halted = bool(risk.get("halted")) if isinstance(risk, dict) else False
            text = f"상태 동기화 · tick {tick} · {'중지' if halted else '운영 중'}"
        else:
            text = short_reason(json.dumps(payload, ensure_ascii=False), 120)
        lines.append(
            f'<div class="ops-line"><span class="time">{html.escape(timestamp)}</span> '
            f'<span class="event">{html.escape(korean_event(name))}</span> {html.escape(text)}</div>'
        )
        if len(lines) >= limit:
            break
    if not lines:
        return '<div class="empty">아직 로그가 없습니다.</div>'
    return '<div class="ops">' + "".join(lines) + "</div>"


def render_compact_trade_table(trades: list[TradeRow], limit: int = 8) -> str:
    recent = sorted(trades, key=lambda trade: parse_sort_time(trade.timestamp), reverse=True)[:limit]
    if not recent:
        return '<div class="empty">아직 거래 기록이 없습니다.</div>'
    rows = []
    for trade in recent:
        side_class = "buy" if trade.side == "buy" else "sell"
        pnl_class = "pos" if trade.realized_pnl > 0 else "neg" if trade.realized_pnl < 0 else ""
        rows.append(
            (
                display_time(trade.timestamp),
                trade.market,
                f'<span class="{side_class}">{html.escape("매수" if trade.side == "buy" else "매도")}</span>',
                f"{trade.price:,.4f}",
                f'<span class="{pnl_class}">{html.escape(krw(trade.realized_pnl))}</span>',
                short_reason(trade.reason, 90),
            )
        )
    return render_simple_table(["시간", "마켓", "구분", "가격", "손익", "이유"], rows, raw_cols={2, 4}, text_cols={5}, num_cols={3, 4})


def render_compact_diagnosis(metrics: dict[str, float | int], market_mode: str) -> str:
    exit_count = int(metrics["exit_count"])
    expectancy = float(metrics["expectancy"])
    payoff = float(metrics["payoff_ratio"])
    win_rate = float(metrics["win_rate"])
    items = [
        ("표본", f"{exit_count}건", "warn" if exit_count < 30 else "ok"),
        ("기대값", krw(expectancy), "pos" if expectancy > 0 else "neg" if expectancy < 0 else ""),
        ("손익비", ratio(payoff), "pos" if payoff >= 1.2 else "warn" if payoff >= 0.9 else "neg"),
        ("승률", pct(win_rate), "pos" if win_rate >= 0.5 else "warn"),
        ("시장 모드", market_mode, "warn" if market_mode in {"risk_off", "capital_protect"} else "ok" if market_mode == "risk_on" else ""),
    ]
    rows = [(label, f'<span class="{klass}">{html.escape(value)}</span>' if klass else html.escape(value)) for label, value, klass in items]
    return render_simple_table(["항목", "값"], rows, raw_cols={1}, num_cols={1})


def short_reason(reason: str, max_chars: int) -> str:
    text = " ".join(str(reason).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def render_open_positions(events: list[dict[str, Any]]) -> str:
    positions = find_latest_positions(events)
    if not positions:
        return empty_block("현재 열려 있는 포지션이 없습니다.")
    latest_prices = find_latest_prices(events)
    rows = []
    for market, position in sorted(positions.items()):
        qty = to_float(position.get("qty")) or 0.0
        avg_price = to_float(position.get("avg_price")) or 0.0
        peak_price = to_float(position.get("peak_price")) or 0.0
        mark_price = latest_prices.get(market, avg_price)
        value = qty * mark_price
        pnl_pct = (mark_price / avg_price - 1.0) if avg_price > 0 else 0.0
        partial_take_price = avg_price * (1.0 + REPORT_PARTIAL_TAKE_PROFIT_PCT / 100.0)
        target_price = avg_price * (1.0 + REPORT_TARGET_UPSIDE_PCT / 100.0)
        stop_price = avg_price * (1.0 - REPORT_STOP_LOSS_PCT / 100.0)
        trailing_price = peak_price * (1.0 - REPORT_TRAILING_STOP_PCT / 100.0) if peak_price > 0 else 0.0
        cls = "pos" if pnl_pct > 0 else "neg" if pnl_pct < 0 else ""
        rows.append(
            (
                market,
                f"{qty:.8f}",
                f"{avg_price:,.4f}",
                f"{mark_price:,.4f}",
                krw(value),
                f'<span class="{cls}">{html.escape(pct(pnl_pct))}</span>',
                f"{peak_price:,.4f}",
                f"{partial_take_price:,.4f}",
                f"{target_price:,.4f}",
                f"{stop_price:,.4f}",
                f"{trailing_price:,.4f}",
            )
        )
    return render_simple_table(
        ["마켓", "수량", "평균 단가", "현재가", "평가금액", "평가수익률", "고점", "1차 익절가", "최종 목표가", "손절가", "트레일링가"],
        rows,
        raw_cols={5},
        num_cols={1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    )


def render_diagnosis(metrics: dict[str, float | int]) -> str:
    items = []
    if int(metrics["exit_count"]) < 30:
        items.append(("표본 부족", "완료 거래가 30건 미만입니다. 지금의 승률은 우연의 영향이 커서 전략 우열을 판단하기 어렵습니다.", "warn"))
    if float(metrics["expectancy"]) <= 0:
        items.append(("기대값 음수", "거래 1회당 평균 기대 손익이 0 이하입니다. 진입 조건을 더 엄격하게 하고 과열 구간 진입을 줄여야 합니다.", "neg"))
    else:
        items.append(("기대값 양수", "현재 기록만 보면 거래 1회당 기대 손익은 양수입니다. 다만 표본이 충분한지 함께 확인해야 합니다.", "ok"))
    if float(metrics["payoff_ratio"]) < 1:
        items.append(("손익비 약함", "평균 이익이 평균 손실보다 작습니다. 승률이 아주 높지 않다면 장기적으로 불리해질 수 있습니다.", "warn"))
    if int(metrics["max_consecutive_losses"]) >= 3:
        items.append(("연속 손실 주의", "연속 손실이 3회 이상 발생했습니다. 손실 후 재진입 대기 시간을 늘리는 편이 좋습니다.", "warn"))
    if float(metrics["profit_factor"]) and float(metrics["profit_factor"]) < 1.2:
        items.append(("수익 팩터 낮음", "총이익 대비 총손실 비율이 낮습니다. 후보 선별과 BTC 장세 필터가 더 중요합니다.", "warn"))
    return '<div class="diagnosis">' + "".join(
        f'<div><strong class="{klass}">{html.escape(title)}</strong><br>{html.escape(body)}</div>'
        for title, body, klass in items
    ) + "</div>"


def render_metric_table(metrics: dict[str, float | int]) -> str:
    rows = [
        ("완료 거래 수", str(metrics["exit_count"]), "승률과 기대값 계산에 사용한 청산 완료 거래 수입니다."),
        ("승리 / 패배", f'{metrics["win_count"]} / {metrics["loss_count"]}', "실현 손익이 양수/음수인 청산 거래 수입니다."),
        ("평균 이익", krw(float(metrics["avg_win"])), "이익 거래의 평균 실현 손익입니다."),
        ("평균 손실", krw(float(metrics["avg_loss"])), "손실 거래의 평균 실현 손익입니다."),
        ("손익비", ratio(float(metrics["payoff_ratio"])), "평균 이익 / 평균 손실 절댓값입니다."),
        ("기대값", krw(float(metrics["expectancy"])), "거래 1회당 평균적으로 기대하는 손익입니다."),
        ("기대값 95% 근사 범위", f'{krw(float(metrics["expectancy_ci_low"]))} ~ {krw(float(metrics["expectancy_ci_high"]))}', "표본 기반 근사 구간입니다. 표본이 적으면 넓게 흔들립니다."),
        ("승률 95% 신뢰구간", f'{pct(float(metrics["win_rate_ci_low"]))} ~ {pct(float(metrics["win_rate_ci_high"]))}', "대수의 법칙 관점에서 승률 추정의 불확실성을 보여줍니다."),
        ("수익 팩터", ratio(float(metrics["profit_factor"])), "총이익 / 총손실 절댓값입니다. 1보다 커야 합니다."),
        ("최대 낙폭", krw(float(metrics["max_drawdown"])), "실현 손익 누적 기준 최고점 대비 최악 하락폭입니다."),
        ("최대 연속 손실", str(metrics["max_consecutive_losses"]), "손실 청산이 연속으로 발생한 최댓값입니다."),
        ("총 수수료", krw(float(metrics["total_fee"])), "모든 모의 체결에 반영된 수수료입니다."),
    ]
    return render_simple_table(["지표", "값", "해석"], rows, text_cols={2}, num_cols={1})


def render_sample_law(metrics: dict[str, float | int]) -> str:
    n = int(metrics["exit_count"])
    if n < 30:
        level = "낮음"
        note = "아직 표본이 매우 적습니다. 방향만 참고하고 전략을 자주 바꾸지 않는 편이 좋습니다."
    elif n < 100:
        level = "중간"
        note = "초기 판단은 가능하지만 오차가 큽니다. 최소 100건 이상 완료 거래를 쌓으면 안정성이 좋아집니다."
    else:
        level = "높음"
        note = "표본이 어느 정도 쌓였습니다. 그래도 시장 국면이 바뀌면 분포도 달라질 수 있습니다."
    rows = [
        ("완료 표본 수", str(n), "대수의 법칙은 완료 거래가 많아질수록 평균과 승률 추정이 안정된다는 아이디어입니다."),
        ("신뢰 수준", level, note),
        ("승률 추정 범위", f'{pct(float(metrics["win_rate_ci_low"]))} ~ {pct(float(metrics["win_rate_ci_high"]))}', "표본이 늘수록 구간이 좁아지는지 확인하세요."),
        ("기대값 추정 범위", f'{krw(float(metrics["expectancy_ci_low"]))} ~ {krw(float(metrics["expectancy_ci_high"]))}', "구간 전체가 0보다 높아질 때 전략 신뢰도가 올라갑니다."),
        ("운영 원칙", "표본 확보 전 급격한 수정 금지", "30건 전에는 관찰, 30~100건은 완만한 조정, 100건 이후부터 본격 비교가 적절합니다."),
    ]
    return render_simple_table(["항목", "값", "해석"], rows, text_cols={2}, num_cols={1})


def render_research_checklist() -> str:
    rows = [
        ("단일 신호 의존 축소", "적용", "이동평균만 보지 않고 200EMA, 모멘텀, 거래량, RSI 과열, BTC 장세를 함께 확인합니다."),
        ("실행비용 반영", "적용", "수수료와 슬리피지를 모의 체결에 반영합니다. 실제 호가 깊이 기반 비용은 다음 단계입니다."),
        ("변동성 기반 포지션 조절", "적용", "최근 변동성이 목표보다 크면 같은 신호라도 진입 금액을 자동 축소합니다."),
        ("PIT/상폐 편향 통제", "부분 적용", "현재 업비트 활성 상위 마켓 스캔이라 완전한 PIT 유니버스는 아닙니다. 리포트에서 한계로 추적합니다."),
        ("표본 기반 검증", "적용", "완료 거래 수, 승률 신뢰구간, 기대값 신뢰구간으로 대수의 법칙 관점의 신뢰도를 봅니다."),
        ("손절 + 시간손절", "적용", "가격 손절과 보유시간 기준 시간손절을 함께 적용합니다."),
        ("Walk-forward 검증", "미적용", "현재는 실시간 paper 관찰 단계입니다. 충분한 로그가 쌓인 뒤 구간별 검증을 추가해야 합니다."),
    ]
    return render_simple_table(["항목", "상태", "현재 MVP 반영"], rows, text_cols={2})


def render_btc_framework() -> str:
    rows = [
        ("거시 경제", "수동 확인", "금리, CPI/PPI, 달러 인덱스 같은 외부 변수는 아직 자동 수집하지 않습니다."),
        ("온체인 지표", "수동 확인", "MVRV, SOPR, Puell Multiple 등은 MVP 범위 밖입니다. 추후 별도 데이터 API가 필요합니다."),
        ("전략 비교", "부분 적용", "현재는 이동평균 추세 전략에 거래량, 과열, BTC 장세 필터를 추가하는 방향으로 개선했습니다."),
        ("총매수 분석", "수동 확인", "ETF 자금 유입, 기관 리포트, 규제 뉴스는 자동 매매 근거가 아니라 관찰 필터로 다루는 편이 안전합니다."),
        ("리스크 분석", "적용", "일 손실 한도, 일 진입 횟수, 연속 손실 한도, 변동성 기반 진입금액 축소, 표본 기반 진단을 함께 봅니다."),
        ("3일 관찰 목적", "적용", "목표는 단기 수익이 아니라 후보 선별, 기대값, 손익비, 낙폭, 표본 수가 개선되는지 확인하는 것입니다."),
    ]
    return render_simple_table(["분석 영역", "현재 상태", "적용 내용"], rows, text_cols={2})


def render_group_table(groups: list[dict[str, Any]], first_header: str) -> str:
    if not groups:
        return empty_block("아직 분석할 완료 거래가 없습니다.")
    rows = []
    for group in groups:
        cls = "pos" if group["total_pnl"] > 0 else "neg" if group["total_pnl"] < 0 else ""
        rows.append(
            (
                korean_reason(str(group["name"])),
                str(group["count"]),
                pct(group["win_rate"]),
                krw(group["avg_pnl"]),
                f'<span class="{cls}">{html.escape(krw(group["total_pnl"]))}</span>',
            )
        )
    return render_simple_table([first_header, "거래 수", "승률", "평균 손익", "총 손익"], rows, raw_cols={4}, text_cols={0}, num_cols={1, 2, 3, 4})


def render_filter_block_table(events: list[dict[str, Any]]) -> str:
    groups = analyze_filter_blocks(events)
    if not groups:
        return empty_block("아직 필터 차단 로그가 충분하지 않습니다. 다음 관찰부터 스캔 차단 사유가 누적됩니다.")
    rows = []
    for group in groups:
        avg_change = group["avg_next_change_pct"]
        avg_text = pct(avg_change / 100.0) if isinstance(avg_change, (int, float)) else "표본 부족"
        rows.append(
            (
                korean_reason(str(group["reason"])),
                str(group["count"]),
                str(group["priced_samples"]),
                avg_text,
                group["note"],
            )
        )
    return render_simple_table(["차단 이유", "차단 횟수", "가격 추적 표본", "이후 평균 변화", "해석"], rows, text_cols={0, 4}, num_cols={1, 2, 3})


def render_ai_decision_table(events: list[dict[str, Any]]) -> str:
    rows = []
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        decision = payload.get("ai_decision")
        if not isinstance(decision, dict):
            continue
        context = payload.get("decision_context", {})
        context_reason = context.get("reason", "") if isinstance(context, dict) else ""
        notes = decision.get("risk_notes", [])
        if isinstance(notes, list):
            risk_notes = "; ".join(str(note) for note in notes) or "-"
        else:
            risk_notes = str(notes or "-")
        rows.append(
            (
                display_time(str(event.get("timestamp", ""))),
                str(payload.get("market", "-")),
                str(decision.get("action", "-")),
                str(decision.get("source", "-")),
                pct((to_float(decision.get("confidence")) or 0.0)),
                pct((to_float(decision.get("expected_upside_pct")) or 0.0) / 100.0),
                pct((to_float(decision.get("expected_downside_pct")) or 0.0) / 100.0),
                str(decision.get("thesis", "-")),
                str(decision.get("invalidation", "-")),
                risk_notes,
                str(context_reason),
            )
        )
        if len(rows) >= 30:
            break
    if not rows:
        return empty_block("AI decision logs are not available yet. New candidate reviews will appear here.")
    return render_simple_table(
        ["시간", "마켓", "판단", "출처", "신뢰도", "예상 상승", "예상 하락", "판단 근거", "무효 조건", "위험 메모", "시장 컨텍스트"],
        rows,
        text_cols={7, 8, 9, 10},
        num_cols={4, 5, 6},
    )


def render_market_context_table(events: list[dict[str, Any]]) -> str:
    rows = []
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        if str(event.get("event", "")) == "state_snapshot":
            positions = payload.get("positions", {})
            position_count = len(positions) if isinstance(positions, dict) else 0
            risk = payload.get("risk", {}) if isinstance(payload.get("risk"), dict) else {}
            context = payload.get("decision_context", {}) if isinstance(payload.get("decision_context"), dict) else {}
            risk_state = "중지" if risk.get("halted") else "운영 중"
            rows.append(
                (
                    display_time(str(event.get("timestamp", ""))),
                    risk_state,
                    str(context.get("market_mode", "-")),
                    str(context.get("session_label", "-")),
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    (
                        f"state tick {payload.get('tick', '-')}; "
                        f"cash {krw(to_float(payload.get('cash')))}; "
                        f"equity {krw(to_float(payload.get('equity')))}; "
                        f"open positions {position_count}"
                    ),
                )
            )
            if len(rows) >= 30:
                break
            continue
        context = payload.get("decision_context")
        if not isinstance(context, dict):
            continue
        rows.append(
            (
                display_time(str(event.get("timestamp", ""))),
                "허용" if context.get("allows_entries") else "차단",
                str(context.get("market_mode", "-")),
                str(context.get("session_label", "-")),
                pct((to_float(context.get("score_multiplier")) or 0.0) - 1.0),
                pct((to_float(context.get("btc_momentum_pct")) or 0.0) / 100.0),
                pct((to_float(context.get("btc_volatility_pct")) or 0.0) / 100.0),
                str(context.get("fear_greed_value") if context.get("fear_greed_value") is not None else "-"),
                pct((to_float(context.get("global_market_cap_change_pct")) or 0.0) / 100.0)
                if context.get("global_market_cap_change_pct") is not None
                else "-",
                pct((to_float(context.get("btc_dominance_pct")) or 0.0) / 100.0)
                if context.get("btc_dominance_pct") is not None
                else "-",
                pct((to_float(context.get("binance_btcusdt_change_pct")) or 0.0) / 100.0)
                if context.get("binance_btcusdt_change_pct") is not None
                else "-",
                str(context.get("reason", "-")),
            )
        )
        if len(rows) >= 30:
            break
    if not rows:
        return empty_block("자동 수집 시장 데이터가 아직 없습니다. 다음 후보 스캔부터 기록됩니다.")
    return render_simple_table(
        ["시간", "진입", "시장 모드", "시장 세션", "점수 조정", "BTC 모멘텀", "BTC 변동성", "공포탐욕", "글로벌 시총 24h", "BTC 점유율", "Binance BTC 24h", "요약"],
        rows,
        text_cols={2, 3, 11},
        num_cols={2, 3, 4, 6, 7, 8},
    )


def analyze_filter_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    observations: dict[str, list[tuple[int, float]]] = {}

    for index, event in enumerate(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue

        blocked_reasons = payload.get("blocked_reasons", {})
        if isinstance(blocked_reasons, dict):
            for reason, count in blocked_reasons.items():
                counts[str(reason)] = counts.get(str(reason), 0) + int(count)

        blocked_samples = payload.get("blocked_samples", [])
        if isinstance(blocked_samples, list):
            for sample in blocked_samples:
                if not isinstance(sample, dict):
                    continue
                market = str(sample.get("market", ""))
                reason = str(sample.get("reason", ""))
                price = to_float(sample.get("price"))
                if market and reason and price:
                    samples.append({"index": index, "market": market, "reason": reason, "price": price})
                    observations.setdefault(market, []).append((index, price))

        market = payload.get("market")
        price = to_float(payload.get("price"))
        if market and price:
            observations.setdefault(str(market), []).append((index, price))

        fill = payload.get("fill", {})
        if isinstance(fill, dict):
            fill_market = fill.get("market")
            fill_price = to_float(fill.get("price"))
            if fill_market and fill_price:
                observations.setdefault(str(fill_market), []).append((index, fill_price))

    changes_by_reason: dict[str, list[float]] = {}
    for sample in samples:
        later_price = next_later_price(observations.get(sample["market"], []), int(sample["index"]))
        if later_price is None:
            continue
        change_pct = (later_price / float(sample["price"]) - 1.0) * 100.0
        changes_by_reason.setdefault(str(sample["reason"]), []).append(change_pct)

    for reason in changes_by_reason:
        counts.setdefault(reason, 0)

    groups = []
    for reason, count in counts.items():
        changes = changes_by_reason.get(reason, [])
        if changes:
            avg_change = sum(changes) / len(changes)
            note = "차단 후 관측 가격 기준입니다. 양수면 놓친 상승, 음수면 회피한 하락 가능성을 뜻합니다."
        else:
            avg_change = None
            note = "이후 가격 표본이 아직 부족합니다."
        groups.append(
            {
                "reason": reason,
                "count": count,
                "priced_samples": len(changes),
                "avg_next_change_pct": avg_change,
                "note": note,
            }
        )
    return sorted(groups, key=lambda group: group["count"], reverse=True)


def next_later_price(observations: list[tuple[int, float]], current_index: int) -> float | None:
    for index, price in observations:
        if index > current_index:
            return price
    return None


def render_trade_table(trades: list[TradeRow]) -> str:
    if not trades:
        return empty_block("아직 거래 기록이 없습니다.")
    rows = []
    for trade in reversed(trades):
        pnl_class = "pos" if trade.realized_pnl > 0 else "neg" if trade.realized_pnl < 0 else ""
        side_class = "buy" if trade.side == "buy" else "sell"
        rows.append(
            (
                display_time(trade.timestamp),
                trade.market,
                f'<span class="{side_class}">{html.escape(korean_side(trade.side))}</span>',
                f"{trade.price:,.4f}",
                f"{trade.qty:.8f}",
                krw(trade.fee),
                f'<span class="{pnl_class}">{html.escape(krw(trade.realized_pnl))}</span>',
                korean_reason(trade.reason),
            )
        )
    return render_simple_table(["시간", "마켓", "구분", "가격", "수량", "수수료", "손익", "이유"], rows, raw_cols={2, 6}, text_cols={7}, num_cols={3, 4, 5, 6})


def render_event_table(events: list[dict[str, Any]]) -> str:
    if not events:
        return empty_block("아직 판단 로그가 없습니다.")
    rows = []
    for event in reversed(events):
        payload = event.get("payload", {})
        rows.append((display_time(str(event.get("timestamp", ""))), korean_event(str(event.get("event", ""))), summarize_event(str(event.get("event", "")), payload)))
    return render_simple_table(["시간", "이벤트", "요약"], rows, text_cols={2})


def render_simple_table(
    headers: list[str],
    rows: list[tuple[Any, ...]],
    raw_cols: set[int] | None = None,
    text_cols: set[int] | None = None,
    num_cols: set[int] | None = None,
) -> str:
    raw_cols = raw_cols or set()
    text_cols = text_cols or set()
    num_cols = num_cols or set()
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            classes = []
            if index in text_cols:
                classes.append("text")
            if index in num_cols:
                classes.append("num")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            if index in raw_cols:
                cells.append(f"<td{class_attr}>{value}</td>")
            else:
                cells.append(f"<td{class_attr}>{html.escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-shell"><div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div></div>'


def empty_block(message: str) -> str:
    return f'<div style="padding:16px;color:#667085;">{html.escape(message)}</div>'


def group_by_entry_reason(pairs: list[RoundTrip]) -> list[dict[str, Any]]:
    return group_pairs(pairs, lambda pair: pair.entry.reason)


def group_by_exit_reason(pairs: list[RoundTrip]) -> list[dict[str, Any]]:
    return group_pairs(pairs, lambda pair: pair.exit.reason.split(":")[0])


def group_by_exit_hour(pairs: list[RoundTrip]) -> list[dict[str, Any]]:
    return group_pairs(pairs, lambda pair: display_time(pair.exit.timestamp)[11:13] + "시")


def group_pairs(pairs: list[RoundTrip], key_func: Callable[[RoundTrip], str]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = {}
    for pair in pairs:
        buckets.setdefault(str(key_func(pair)), []).append(pair.pnl)
    groups = []
    for name, pnls in buckets.items():
        wins = [pnl for pnl in pnls if pnl > 0]
        groups.append(
            {
                "name": name,
                "count": len(pnls),
                "win_rate": len(wins) / len(pnls) if pnls else 0.0,
                "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
                "total_pnl": sum(pnls),
            }
        )
    return sorted(groups, key=lambda group: group["total_pnl"], reverse=True)


def summarize_event(name: str, payload: dict[str, Any]) -> str:
    if name == "tick":
        signal = payload.get("signal", {})
        approved = "승인" if payload.get("approved") else "대기/차단"
        market = payload.get("market")
        market_part = f"{market}, " if market else ""
        candidate_part = f", 후보 {payload.get('candidate_count')}개" if payload.get("candidate_count") is not None else ""
        return (
            f'{market_part}가격 {num(payload.get("price"))}, 평가금액 {num(payload.get("equity"))}, '
            f'신호 {korean_side(str(signal.get("side")))} ({korean_reason(str(signal.get("reason")))}) , '
            f'판정 {approved}, 리스크 사유: {korean_reason(str(payload.get("risk_reason")))}{candidate_part}'
        )
    if name == "market_scan":
        reason = korean_reason(str(payload.get("reason", "")))
        return f'스캔 마켓 {payload.get("markets_scanned", 0)}개, 후보 {payload.get("candidates", 0)}개, 사유: {reason}'
    if name in {"fill", "forced_exit"}:
        fill = payload.get("fill", {})
        return (
            f'{korean_side(str(fill.get("side")))} 체결, '
            f'수량 {fill.get("qty")}, 손익 {krw(to_float(fill.get("realized_pnl")))}, '
            f'이유: {korean_reason(str(fill.get("reason")))}'
        )
    if name == "bot_finished":
        return f'종료 현금 {krw(to_float(payload.get("cash")))}, 상태 {korean_risk(payload.get("risk", {}))}'
    return json.dumps(payload, ensure_ascii=False)[:260]


def status_message(risk: dict[str, Any], position: dict[str, Any], total_realized: float) -> str:
    if risk.get("halted"):
        return f'거래 중지: {korean_reason(str(risk.get("halt_reason") or "사유 없음"))}'
    if (to_float(position.get("qty")) or 0.0) > 0:
        return "포지션 보유 중입니다. 청산 조건을 계속 감시하고 있습니다."
    if total_realized > 0:
        return "실현 손익이 플러스입니다. 표본이 더 쌓여도 유지되는지 확인하세요."
    if total_realized < 0:
        return "실현 손익이 마이너스입니다. 손실 제한과 진입 조건을 더 보수적으로 봐야 합니다."
    return "아직 충분한 실현 손익이 없습니다. 신호를 관찰 중입니다."


def korean_side(side: str) -> str:
    return {"buy": "매수", "sell": "매도", "hold": "대기", "none": "없음"}.get(side.lower(), side)


def korean_event(name: str) -> str:
    return {
        "bot_started": "봇 시작",
        "watch_started": "관찰 시작",
        "market_scan": "마켓 스캔",
        "market_scan_error": "스캔 오류",
        "tick": "판단",
        "fill": "체결",
        "forced_exit": "강제 청산",
        "fill_skipped": "체결 생략",
        "bot_error": "오류",
        "watch_error": "관찰 오류",
        "bot_finished": "봇 종료",
        "watch_finished": "관찰 종료",
    }.get(name, name)


def korean_reason(reason: str) -> str:
    if not reason or reason == "None":
        return "없음"
    replacements = [
        ("not enough candles", "캔들 데이터 부족"),
        ("position open, no exit condition", "보유 중이나 청산 조건 없음"),
        ("uptrend filter passed", "상승 추세 조건 충족"),
        ("btc regime blocked", "BTC 장세 필터 차단"),
        ("long trend filter blocked", "장기 EMA 필터 차단"),
        ("price below EMA", "가격이 장기 EMA 아래"),
        ("overextended", "단기 과열"),
        ("weak recent momentum", "최근 모멘텀 약함"),
        ("thin volume", "거래량 부족"),
        ("trend break", "추세 이탈"),
        ("no entry condition", "진입 조건 없음"),
        ("hold signal", "대기 신호"),
        ("approved risk-reducing exit", "리스크 축소 청산 승인"),
        ("approved", "승인"),
        ("max daily entries reached", "하루 신규 진입 한도 도달"),
        ("max consecutive losses reached", "연속 손실 한도 도달"),
        ("position fraction exceeds risk limit", "포지션 비중 한도 초과"),
        ("daily profit target reached", "하루 수익 목표 도달"),
        ("daily loss limit reached", "하루 손실 한도 도달"),
        ("stop loss reached", "손절 조건 도달"),
        ("take profit reached", "익절 조건 도달"),
        ("time stop reached", "시간손절 조건 도달"),
        ("forced exit", "강제 청산"),
        ("selected from top-volume scan", "상위 거래대금 스캔에서 선택"),
    ]
    translated = reason
    for source, target in replacements:
        translated = translated.replace(source, target)
    return translated


def korean_risk(risk: dict[str, Any]) -> str:
    if not isinstance(risk, dict):
        return str(risk)
    halted = "중지" if risk.get("halted") else "진행 가능"
    return (
        f'{halted}, 신규 진입 {risk.get("entries_today", 0)}회, '
        f'청산 {risk.get("exits_today", 0)}회, '
        f'연속 손실 {risk.get("consecutive_losses", 0)}회'
    )


def krw(value: float | int | None) -> str:
    if value is None:
        return "-"
    sign = "-" if value < 0 else ""
    return f"{sign}{abs(value):,.0f} KRW"


def short_krw(value: float | int | None) -> str:
    if value is None:
        return "-"
    value = float(value)
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    if absolute >= 100_000_000:
        return f"{sign}{absolute / 100_000_000:.1f}억"
    if absolute >= 10_000:
        return f"{sign}{absolute / 10_000:.0f}만"
    return f"{sign}{absolute:.0f}"


def pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100.0:.1f}%"


def ratio(value: float | int | None) -> str:
    if value is None or float(value) == 0.0:
        return "-"
    return f"{float(value):.2f}x"


def num(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def short_time(value: str) -> str:
    if "T" in value:
        date, rest = value.split("T", 1)
        return f"{date} {rest[:8]}"
    return value[:19]


def display_time(value: str) -> str:
    parsed = parse_utc_time(value)
    if parsed is None:
        return short_time(value)
    kst = parsed.astimezone(timezone(timedelta(hours=9)))
    return kst.strftime("%Y-%m-%d %H:%M:%S KST")


def parse_sort_time(value: str) -> datetime:
    parsed = parse_utc_time(value)
    if parsed is not None:
        return parsed.astimezone(timezone.utc)
    text = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S KST"):
        try:
            parsed = datetime.strptime(text.replace(" KST", ""), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone(timedelta(hours=9))).astimezone(timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def parse_utc_time(value: str) -> datetime | None:
    if not value or "T" not in value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_console_report(trades: list[TradeRow], events: list[dict[str, Any]]) -> str:
    trades = sorted(trades, key=lambda trade: parse_sort_time(trade.timestamp))
    events = sorted(events, key=lambda event: parse_sort_time(str(event.get("timestamp", ""))))
    metrics = calculate_metrics(trades)
    portfolio = portfolio_summary(trades, events)
    open_positions = find_latest_positions(events)
    latest_context = find_latest_decision_context(events)
    last_event = events[-1] if events else {}
    last_payload = last_event.get("payload", {}) if isinstance(last_event, dict) else {}
    risk = last_payload.get("risk", {}) if isinstance(last_payload, dict) else {}
    refreshed_at = display_time(str(last_event.get("timestamp", ""))) if last_event else "아직 없음"
    cash = float(portfolio.get("cash", portfolio["current_equity"]))
    change = float(portfolio["change_amount"])
    return_class = "pos" if change >= 0 else "neg"
    halted = bool(risk.get("halted"))
    market_mode = str(latest_context.get("market_mode", "unknown")) if latest_context else "unknown"
    session_label = str(latest_context.get("session_label", "-")) if latest_context else "-"
    target_amount = float(portfolio["starting_cash"]) * 0.03
    target_progress = change / target_amount if target_amount else 0.0
    target_bar_width = max(0.0, min(100.0, target_progress * 100.0))
    cards = [
        ("현재 평가금액", krw(float(portfolio["current_equity"]))),
        ("시작 대비 손익", krw(change)),
        ("현재 수익률", pct(float(portfolio["return_pct"]) / 100.0)),
        ("가용 현금", krw(cash)),
        ("완료 거래", str(metrics["exit_count"])),
        ("승률", pct(float(metrics["win_rate"]))),
        ("기대값", krw(float(metrics["expectancy"]))),
        ("손익비", ratio(float(metrics["payoff_ratio"]))),
        ("최대 낙폭", krw(float(metrics["max_drawdown"]))),
        ("오픈 포지션", f"{len(open_positions)}개"),
        ("오늘 진입", str(risk.get("entries_today", 0))),
        ("연속 손실", str(risk.get("consecutive_losses", 0))),
    ]
    operation_state = "중지" if halted else "운영 중"
    operation_class = "warn" if halted else "ok"
    mode_class = "ok" if market_mode == "risk_on" else "warn" if market_mode in {"risk_off", "capital_protect"} else ""
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>코인 오토 트레이딩 시스템</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f16;
      --panel: #111722;
      --panel-2: #151c29;
      --ink: #edf4ff;
      --muted: #8ea0ba;
      --line: #263244;
      --soft-line: #1c2635;
      --green: #00d4a6;
      --red: #ff4d6d;
      --blue: #6b98ff;
      --amber: #f7b955;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: "Segoe UI", "Malgun Gothic", Arial, sans-serif; font-size: 13px; }}
    main {{ margin: 0; padding: 0 0 36px; }}
    header {{ min-height: 54px; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 0 14px; background: #171d29; border-bottom: 1px solid #313b4d; position: sticky; top: 0; z-index: 20; }}
    h1 {{ margin: 0; font-size: 15px; color: #f8fbff; font-weight: 750; }}
    h2 {{ font-size: 14px; margin: 0; padding: 12px 14px; border-bottom: 1px solid var(--line); color: #f8fbff; }}
    .brand {{ display: flex; align-items: center; gap: 10px; min-width: 0; flex-wrap: wrap; }}
    .brand-mark {{ width: 22px; height: 22px; border-radius: 50%; background: #b42318; display: inline-flex; align-items: center; justify-content: center; color: #fff; font-size: 12px; font-weight: 900; }}
    .nav-tabs {{ display: flex; align-items: center; gap: 20px; margin-left: 16px; height: 52px; overflow-x: auto; scrollbar-width: thin; }}
    .nav-tab {{ height: 52px; display: flex; align-items: center; color: #738198; border-bottom: 2px solid transparent; font-weight: 650; white-space: nowrap; }}
    .nav-tab.active {{ color: #f4f8ff; border-bottom-color: #d8e3f5; }}
    .top-meta {{ display: flex; align-items: center; gap: 16px; color: #c8d4e6; font-weight: 700; white-space: nowrap; flex-wrap: wrap; justify-content: flex-end; }}
    .workspace {{ padding: 8px; }}
    .portfolio-hero {{ display: grid; grid-template-columns: minmax(280px, 1.25fr) repeat(3, minmax(170px, .75fr)); gap: 8px; margin-bottom: 8px; }}
    .portfolio-main, .portfolio-sub, .card, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }}
    .portfolio-main, .portfolio-sub {{ padding: 14px 16px; min-height: 92px; }}
    .portfolio-title, .label, .state-name {{ color: var(--muted); font-size: 12px; font-weight: 650; }}
    .portfolio-title {{ margin-bottom: 9px; }}
    .portfolio-amount {{ font-size: 30px; line-height: 1.12; font-weight: 850; overflow-wrap: anywhere; }}
    .portfolio-value {{ font-size: 22px; line-height: 1.2; font-weight: 850; overflow-wrap: anywhere; }}
    .portfolio-note {{ margin-top: 8px; color: var(--muted); font-size: 12px; line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; margin-bottom: 8px; }}
    .card {{ padding: 11px 12px; min-height: 64px; }}
    .value {{ font-size: 18px; font-weight: 850; margin-top: 7px; overflow-wrap: anywhere; }}
    section {{ margin-top: 8px; overflow: hidden; }}
    .dashboard-grid {{ display: grid; grid-template-columns: minmax(520px, 1.35fr) minmax(340px, .65fr); gap: 8px; align-items: start; }}
    .split-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 8px; }}
    .table-shell {{ overflow: hidden; }}
    .table-wrap {{ width: 100%; overflow: auto; -webkit-overflow-scrolling: touch; max-height: 48vh; border-top: 1px solid var(--soft-line); }}
    table {{ width: 100%; min-width: 760px; border-collapse: separate; border-spacing: 0; table-layout: auto; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--soft-line); text-align: left; vertical-align: top; background: #111722; }}
    th {{ position: sticky; top: 0; z-index: 3; color: #91a0b8; font-size: 12px; background: #171f2c; white-space: nowrap; font-weight: 750; }}
    td {{ white-space: nowrap; }}
    td.text {{ white-space: normal; min-width: 260px; line-height: 1.45; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .buy, .blue {{ color: var(--blue); font-weight: 800; }}
    .sell, .neg {{ color: var(--red); font-weight: 800; }}
    .pos, .ok {{ color: var(--green); font-weight: 800; }}
    .warn {{ color: var(--amber); font-weight: 800; }}
    .state-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .state-item {{ padding: 13px 14px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
    .state-item:nth-child(2n) {{ border-right: 0; }}
    .state-value {{ margin-top: 6px; font-weight: 850; overflow-wrap: anywhere; }}
    .chart-panel {{ padding: 12px; }}
    .chart-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }}
    .chart-title {{ color: #eef5ff; font-weight: 850; }}
    .chart-tabs {{ display: inline-flex; gap: 4px; background: #0e131c; border: 1px solid #273448; border-radius: 6px; padding: 3px; max-width: 100%; overflow-x: auto; }}
    .chart-tabs button {{ appearance: none; border: 0; border-radius: 4px; background: transparent; color: #8ea0ba; cursor: pointer; font: inherit; min-height: 30px; padding: 5px 10px; white-space: nowrap; }}
    .chart-tabs button.active {{ background: #1d2a3d; color: var(--blue); }}
    .equity-chart {{ display: block; width: 100%; height: auto; min-height: 230px; max-height: 420px; background: #0e131c; border: 1px solid #222d3d; border-radius: 4px; }}
    .progress {{ height: 8px; background: #0b111a; border: 1px solid #243247; border-radius: 999px; overflow: hidden; margin-top: 10px; }}
    .progress-fill {{ height: 100%; background: linear-gradient(90deg, #00a884, #00d4a6); width: {target_bar_width:.2f}%; }}
    .ops-log {{ max-height: 444px; overflow: auto; font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px; line-height: 1.55; background: #0e131c; }}
    .ops-line {{ padding: 3px 10px; border-bottom: 1px solid #171f2c; color: #c9d5e8; }}
    .ops-line .time {{ color: #7f8da3; }}
    .ops-line .event {{ color: var(--green); }}
    .muted {{ color: var(--muted); }}
    .diagnosis {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .diagnosis div {{ padding: 13px 14px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); line-height: 1.55; color: #c8d4e6; }}
    .diagnosis div:nth-child(2n) {{ border-right: 0; }}
    @media (max-width: 1000px) {{
      header {{ height: auto; padding: 12px; display: block; position: static; }}
      .top-meta {{ justify-content: flex-start; margin-top: 10px; white-space: normal; }}
      .nav-tabs {{ margin-left: 0; margin-top: 10px; height: 32px; }}
      .nav-tab {{ height: 32px; }}
      .portfolio-hero, .dashboard-grid, .split-grid {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 560px) {{
      .workspace {{ padding: 6px; }}
      .grid, .state-grid, .diagnosis {{ grid-template-columns: 1fr; }}
      .portfolio-amount {{ font-size: 26px; }}
      .portfolio-main, .portfolio-sub, .card, .chart-panel {{ padding: 12px; }}
      .equity-chart {{ min-height: 210px; }}
      table {{ min-width: 680px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="brand">
      <span class="brand-mark">↗</span>
      <h1>코인 오토 트레이딩 시스템</h1>
      <nav class="nav-tabs">
        <span class="nav-tab">봇</span>
        <span class="nav-tab active">수익 리포트</span>
        <span class="nav-tab">주문 기록</span>
      </nav>
    </div>
    <div class="top-meta">
      <span>보유 KRW&nbsp; {html.escape(krw(cash))}</span>
      <span>누적 수익률&nbsp; <span class="{return_class}">{html.escape(pct(float(portfolio["return_pct"]) / 100.0))}</span></span>
    </div>
  </header>
  <div class="workspace">
    {render_portfolio_hero(portfolio)}
    <div class="grid">{render_cards(cards)}</div>
    <div class="dashboard-grid">
      <section class="chart-panel">
        <div class="chart-head">
          <div>
            <div class="chart-title">누적 수익률</div>
            <div class="muted">마지막 갱신: {html.escape(refreshed_at)}</div>
          </div>
          <div class="chart-tabs" role="group" aria-label="차트 범위">
            <button type="button" class="active" data-chart-window="all">전체</button>
            <button type="button" data-chart-window="60">최근 60</button>
            <button type="button" data-chart-window="240">최근 240</button>
          </div>
        </div>
        {render_equity_chart(trades, events)}
      </section>
      <section>
        <h2>운영 상태</h2>
        <div class="state-grid">
          {render_state_item("상태", operation_state, operation_class)}
          {render_state_item("오픈 포지션", f"{len(open_positions)}개", "")}
          {render_state_item("시장 모드", market_mode, mode_class)}
          {render_state_item("시장 세션", session_label, "")}
          {render_state_item("오늘 진입", str(risk.get("entries_today", 0)), "")}
          {render_state_item("연속 손실", str(risk.get("consecutive_losses", 0)), "warn" if int(risk.get("consecutive_losses", 0) or 0) else "")}
          {render_state_item("3% 목표 진행률", pct(target_progress), return_class)}
          {render_state_item("목표 수익금", krw(target_amount), "")}
        </div>
        <div style="padding:0 14px 14px;"><div class="progress"><div class="progress-fill"></div></div></div>
      </section>
    </div>
    <div class="dashboard-grid">
      <section><h2>오픈 포지션</h2>{render_open_positions(events)}</section>
      <section><h2>실시간 판단 로그</h2>{render_ops_log_current(events[-120:], trades[-80:])}</section>
    </div>
    <div class="split-grid">
      <section><h2>최근 체결</h2>{render_clean_trade_table(trades[-80:])}</section>
      <section><h2>수익률 개선 진단</h2>{render_diagnosis(metrics)}</section>
    </div>
    <section><h2>필터 차단 분석</h2>{render_filter_block_table(events)}</section>
    <section><h2>통합 성과 지표</h2>{render_clean_metric_table(metrics)}</section>
    <section><h2>시장 데이터</h2>{render_market_context_table(events)}</section>
  </div>
</main>
{responsive_report_script()}
</body>
</html>
"""


def render_clean_trade_table(trades: list[TradeRow]) -> str:
    if not trades:
        return empty_block("아직 거래 기록이 없습니다.")
    rows = []
    for trade in reversed(trades):
        side_class = "buy" if trade.side == "buy" else "sell"
        pnl_class = "pos" if trade.realized_pnl > 0 else "neg" if trade.realized_pnl < 0 else ""
        rows.append(
            (
                display_time(trade.timestamp),
                trade.market,
                f'<span class="{side_class}">{html.escape("매수" if trade.side == "buy" else "매도")}</span>',
                f"{trade.price:,.4f}",
                f"{trade.qty:.8f}",
                krw(trade.fee),
                f'<span class="{pnl_class}">{html.escape(krw(trade.realized_pnl))}</span>',
                trade.reason,
            )
        )
    return render_simple_table(["시간", "마켓", "구분", "가격", "수량", "수수료", "손익", "이유"], rows, raw_cols={2, 6}, text_cols={7}, num_cols={3, 4, 5, 6})


def render_clean_metric_table(metrics: dict[str, float | int]) -> str:
    rows = [
        ("완료 거래 수", metrics["exit_count"], "승률과 기대값 계산에 사용한 청산 완료 거래 수입니다."),
        ("승리 / 패배", f"{metrics['win_count']} / {metrics['loss_count']}", "실현 손익이 양수/음수인 청산 거래 수입니다."),
        ("평균 이익", krw(float(metrics["avg_win"])), "이익 거래의 평균 실현 손익입니다."),
        ("평균 손실", krw(float(metrics["avg_loss"])), "손실 거래의 평균 실현 손익입니다."),
        ("손익비", ratio(float(metrics["payoff_ratio"])), "평균 이익 / 평균 손실 절댓값입니다."),
        ("기대값", krw(float(metrics["expectancy"])), "거래 1회당 평균적으로 기대하는 손익입니다."),
        ("수익 팩터", ratio(float(metrics["profit_factor"])), "총이익 / 총손실 절댓값입니다. 1보다 커야 합니다."),
        ("최대 낙폭", krw(float(metrics["max_drawdown"])), "실현 손익 누적 기준 최고점 대비 최악 하락폭입니다."),
        ("최대 연속 손실", str(metrics["max_consecutive_losses"]), "손실 청산이 연속으로 발생한 최댓값입니다."),
    ]
    return render_simple_table(["지표", "값", "해석"], rows, text_cols={2}, num_cols={1})


if __name__ == "__main__":
    main()

