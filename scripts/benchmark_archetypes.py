from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coin_mvp.data import UpbitPublicDataSource
from coin_mvp.models import Candle
from coin_mvp.strategy import calculate_ema, calculate_rsi, control_limits, latest_volume_ratio


@dataclass
class SimResult:
    strategy: str
    pnl: float
    return_pct: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    payoff: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    equity: float


@dataclass
class PositionState:
    market: str
    qty: float
    avg_price: float
    peak_price: float
    held_ticks: int = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark common intraday crypto strategy archetypes.")
    parser.add_argument("--top-markets", type=int, default=8)
    parser.add_argument("--history-count", type=int, default=240)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--starting-cash", type=float, default=1_000_000.0)
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("reports") / f"archetype_benchmark_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    markets, candles_by_market = fetch_data(args)
    results = []
    for name in ("ma_momentum", "ucl_breakout", "lcl_reclaim", "hybrid_active"):
        results.append(run_strategy(name, markets, candles_by_market, args))
    results.sort(key=lambda result: score(result), reverse=True)
    write_outputs(out_dir, results, markets, args)
    for result in results:
        print(
            f"{result.strategy}: return={result.return_pct:.3f}% pnl={result.pnl:.0f} "
            f"trades={result.trades} win={result.win_rate * 100:.1f}% "
            f"pf={result.profit_factor:.2f} dd={result.max_drawdown:.0f}",
            flush=True,
        )
    print(f"\nBest archetype: {results[0].strategy}")
    print(f"Summary: {out_dir / 'archetype_summary.md'}")


def fetch_data(args: argparse.Namespace) -> tuple[list[str], dict[str, list[Candle]]]:
    source = UpbitPublicDataSource()
    markets = source.get_top_krw_markets(args.top_markets, min_trade_price_krw=300.0)
    candles_by_market = {}
    for market in markets:
        candles_by_market[market] = source.get_recent_candles(market, args.history_count, unit_minutes=1)
        if args.request_delay > 0:
            time.sleep(args.request_delay)
    return markets, candles_by_market


def run_strategy(
    name: str,
    markets: list[str],
    candles_by_market: dict[str, list[Candle]],
    args: argparse.Namespace,
) -> SimResult:
    cash = args.starting_cash
    position: PositionState | None = None
    realized: list[float] = []
    equity_curve: list[float] = [cash]
    start = 60
    end = min(len(candles) for candles in candles_by_market.values())
    for index in range(start, end):
        prices = {market: candles_by_market[market][index].close for market in markets}
        if position is not None:
            candles = candles_by_market[position.market][: index + 1]
            fill = maybe_exit(name, position, candles, args)
            position.held_ticks += 1
            position.peak_price = max(position.peak_price, candles[-1].high)
            if fill is not None:
                exit_price = fill * (1.0 - args.slippage_bps / 10000.0)
                proceeds = position.qty * exit_price
                fee = proceeds * args.fee_rate
                pnl = proceeds - fee - (position.qty * position.avg_price)
                cash += proceeds - fee
                realized.append(pnl)
                position = None
        if position is None:
            candidate = best_entry(name, markets, candles_by_market, index)
            if candidate is not None:
                market, entry_price = candidate
                buy_price = entry_price * (1.0 + args.slippage_bps / 10000.0)
                budget = cash * 0.94
                fee = budget * args.fee_rate
                qty = max(0.0, (budget - fee) / buy_price)
                if qty > 0:
                    cash -= budget
                    position = PositionState(market=market, qty=qty, avg_price=buy_price, peak_price=buy_price)
        equity = cash
        if position is not None:
            equity += position.qty * prices[position.market] * (1.0 - args.slippage_bps / 10000.0)
        equity_curve.append(equity)
    if position is not None:
        last_price = candles_by_market[position.market][end - 1].close * (1.0 - args.slippage_bps / 10000.0)
        proceeds = position.qty * last_price
        fee = proceeds * args.fee_rate
        pnl = proceeds - fee - (position.qty * position.avg_price)
        cash += proceeds - fee
        realized.append(pnl)
        equity_curve.append(cash)
    return summarize(name, realized, equity_curve, args.starting_cash)


def best_entry(
    name: str,
    markets: list[str],
    candles_by_market: dict[str, list[Candle]],
    index: int,
) -> tuple[str, float] | None:
    ranked: list[tuple[float, str, float]] = []
    for market in markets:
        candles = candles_by_market[market][: index + 1]
        score_value = entry_score(name, candles)
        if score_value is not None:
            ranked.append((score_value, market, candles[-1].close))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    _, market, price = ranked[0]
    return market, price


def entry_score(name: str, candles: list[Candle]) -> float | None:
    if len(candles) < 60:
        return None
    closes = [c.close for c in candles]
    latest = closes[-1]
    ema9 = calculate_ema(closes, 9) or latest
    ema21 = calculate_ema(closes, 21) or latest
    ema55 = calculate_ema(closes, 55) or latest
    rsi = calculate_rsi(closes, 14) or 50.0
    vol = latest_volume_ratio(candles, 10)
    momentum3 = pct(latest, closes[-4])
    momentum8 = pct(latest, closes[-9])
    latest_candle = candles[-1]
    close_pos = 1.0 if latest_candle.high <= latest_candle.low else (latest - latest_candle.low) / (latest_candle.high - latest_candle.low)
    bands = control_limits(closes, 20, 1.65)
    prior_bands = control_limits(closes[:-1], 20, 1.65)

    if name == "ma_momentum":
        if ema9 > ema21 > ema55 and 0.12 <= momentum3 <= 1.6 and vol >= 1.15 and 43 <= rsi <= 67 and close_pos >= 0.52:
            return momentum3 * 0.45 + momentum8 * 0.25 + min(vol, 4.0) * 0.12 + close_pos
        return None
    if name == "ucl_breakout":
        if not bands or not prior_bands:
            return None
        ucl, _, _ = bands
        prior_ucl, _, _ = prior_bands
        recent_high = max(c.high for c in candles[-21:-1])
        if latest > prior_ucl * 1.001 and latest >= recent_high * 0.998 and vol >= 1.45 and momentum3 >= 0.25 and close_pos >= 0.58 and rsi <= 70:
            return pct(latest, prior_ucl) + momentum3 + min(vol, 4.0) * 0.22 + close_pos
        return None
    if name == "lcl_reclaim":
        if not bands:
            return None
        _, _, lcl = bands
        touched = min(c.low for c in candles[-3:]) <= lcl * 1.006
        if touched and latest > lcl * 1.004 and momentum3 >= 0.08 and 30 <= rsi <= 55 and vol >= 1.15 and close_pos >= 0.50:
            return (55.0 - rsi) * 0.015 + momentum3 + min(vol, 3.0) * 0.16 + close_pos
        return None
    if name == "hybrid_active":
        trend = ema9 > ema21 and latest >= ema21 * 0.998 and momentum3 >= 0.10 and vol >= 1.05 and 38 <= rsi <= 68
        breakout = False
        reclaim = False
        if bands and prior_bands:
            prior_ucl = prior_bands[0]
            lcl = bands[2]
            breakout = latest > prior_ucl * 1.0005 and vol >= 1.25 and momentum3 >= 0.18 and close_pos >= 0.56
            reclaim = min(c.low for c in candles[-3:]) <= lcl * 1.006 and latest > lcl * 1.003 and momentum3 >= 0.06 and vol >= 1.05
        if trend or breakout or reclaim:
            return momentum3 * 0.42 + momentum8 * 0.18 + min(vol, 4.0) * 0.18 + close_pos + (0.25 if breakout else 0.0)
    return None


def maybe_exit(name: str, position: PositionState, candles: list[Candle], args: argparse.Namespace) -> float | None:
    latest = candles[-1]
    avg = position.avg_price
    peak = max(position.peak_price, latest.high)
    if latest.high >= avg * 1.006:
        return avg * 1.006
    if latest.low <= avg * 0.996:
        return avg * 0.996
    if peak >= avg * 1.004 and latest.low <= peak * 0.997:
        return peak * 0.997
    if position.held_ticks >= 10:
        pnl_pct = pct(latest.close, avg)
        if pnl_pct >= 0.12 or pnl_pct <= -0.18:
            return latest.close
    closes = [c.close for c in candles]
    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    if ema9 is not None and ema21 is not None and ema9 < ema21 and position.held_ticks >= 3:
        return latest.close
    return None


def summarize(name: str, realized: list[float], equity_curve: list[float], starting_cash: float) -> SimResult:
    wins = [value for value in realized if value > 0]
    losses = [value for value in realized if value < 0]
    pnl = sum(realized)
    win_rate = len(wins) / len(realized) if realized else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    payoff = avg_win / avg_loss if avg_loss else (999.0 if avg_win else 0.0)
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_win / gross_loss if gross_loss else (999.0 if gross_win else 0.0)
    expectancy = pnl / len(realized) if realized else 0.0
    max_dd = max_drawdown(equity_curve)
    return SimResult(
        strategy=name,
        pnl=pnl,
        return_pct=pnl / starting_cash * 100.0,
        trades=len(realized),
        wins=len(wins),
        losses=len(losses),
        win_rate=win_rate,
        payoff=payoff,
        profit_factor=profit_factor,
        expectancy=expectancy,
        max_drawdown=max_dd,
        equity=equity_curve[-1] if equity_curve else starting_cash,
    )


def score(result: SimResult) -> float:
    sample_penalty = 3.0 if result.trades < 4 else 0.0
    return result.return_pct * 12.0 + min(result.profit_factor, 5.0) * 1.2 + result.expectancy / 1200.0 + result.max_drawdown / 12000.0 - sample_penalty


def max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    worst = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def pct(new: float, old: float) -> float:
    if old <= 0:
        return 0.0
    return (new / old - 1.0) * 100.0


def write_outputs(out_dir: Path, results: list[SimResult], markets: list[str], args: argparse.Namespace) -> None:
    rows = [asdict(result) for result in results]
    (out_dir / "archetype_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Archetype Benchmark",
        "",
        f"- markets: {', '.join(markets)}",
        f"- history_count: {args.history_count}",
        f"- fee_rate: {args.fee_rate}",
        f"- slippage_bps: {args.slippage_bps}",
        "",
        "| rank | strategy | return % | pnl KRW | trades | win % | payoff | profit factor | expectancy | max DD |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, result in enumerate(results, start=1):
        lines.append(
            f"| {rank} | {result.strategy} | {result.return_pct:.3f} | {result.pnl:.0f} | "
            f"{result.trades} | {result.win_rate * 100:.1f} | {result.payoff:.2f} | "
            f"{result.profit_factor:.2f} | {result.expectancy:.0f} | {result.max_drawdown:.0f} |"
        )
    (out_dir / "archetype_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
