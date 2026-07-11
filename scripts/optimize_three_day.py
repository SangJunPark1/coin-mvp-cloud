from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

from coin_mvp.data import UpbitPublicDataSource
from coin_mvp.models import Candle
from coin_mvp.strategy import calculate_rsi


FEE_RATE = 0.0005
SLIPPAGE_BPS = 5


@dataclass(frozen=True)
class ModelParams:
    name: str
    take_profit_pct: float
    stop_loss_pct: float
    trailing_stop_pct: float
    partial_take_profit_pct: float
    partial_fraction: float
    entry_fraction: float
    min_score: float


@dataclass
class Position:
    market: str
    qty: float
    avg_price: float
    peak_price: float
    partial_taken: bool = False
    entry_reason: str = ""


@dataclass
class Result:
    params: ModelParams
    starting_cash: float
    ending_equity: float
    realized_pnl: float
    total_return_pct: float
    daily_compound_pct: float
    trade_count: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    max_drawdown_krw: float
    max_consecutive_losses: int
    open_position_count: int
    trades: list[dict]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize simple 3-day crypto models on recent Upbit data.")
    parser.add_argument("--top-markets", type=int, default=12)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--unit-minutes", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--starting-cash", type=float, default=1_000_000)
    parser.add_argument("--output", default="reports/three_day_model_optimization.json")
    args = parser.parse_args()

    data_source = UpbitPublicDataSource()
    markets = data_source.get_top_krw_markets(args.top_markets, min_trade_price_krw=300.0)
    replay_count = args.days * 24 * (60 // args.unit_minutes)
    count = replay_count + args.warmup + 5
    candles_by_market = {
        market: data_source.get_recent_candles(market, count, unit_minutes=args.unit_minutes)
        for market in markets
    }

    params = parameter_grid()
    results = [
        simulate_model(candles_by_market, item, args.starting_cash, replay_count, args.days)
        for item in params
    ]
    results.sort(key=lambda item: (item.ending_equity, item.profit_factor, -abs(item.max_drawdown_pct)), reverse=True)
    payload = {
        "markets": markets,
        "days": args.days,
        "unit_minutes": args.unit_minutes,
        "best": result_to_dict(results[0]),
        "top_10": [result_to_dict(item) for item in results[:10]],
        "all_count": len(results),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parameter_grid() -> list[ModelParams]:
    models = ["active_rank", "bb_breakout", "bb_reversal", "hybrid_rank"]
    values: list[ModelParams] = []
    for name in models:
        for tp in (1.2, 1.8, 2.4, 3.0):
            for sl in (0.45, 0.65, 0.85, 1.1):
                for trail in (0.35, 0.55, 0.8):
                    for partial in (0.7, 1.0, 1.4):
                        values.append(
                            ModelParams(
                                name=name,
                                take_profit_pct=tp,
                                stop_loss_pct=sl,
                                trailing_stop_pct=trail,
                                partial_take_profit_pct=partial,
                                partial_fraction=0.5,
                                entry_fraction=0.92,
                                min_score=0.0,
                            )
                        )
    return values


def simulate_model(
    candles_by_market: dict[str, list[Candle]],
    params: ModelParams,
    starting_cash: float,
    replay_count: int,
    days: int,
) -> Result:
    cash = starting_cash
    position: Position | None = None
    trades: list[dict] = []
    wins: list[float] = []
    losses: list[float] = []
    equity_curve: list[float] = []
    markets = list(candles_by_market)
    length = min(len(candles) for candles in candles_by_market.values())
    start = max(30, length - replay_count)

    for index in range(start, length):
        latest_prices = {market: candles_by_market[market][index].close for market in markets}
        equity = cash + (position.qty * latest_prices[position.market] if position else 0.0)
        equity_curve.append(equity)

        if position is not None:
            price = latest_prices[position.market]
            position.peak_price = max(position.peak_price, price)
            pnl_pct = (price / position.avg_price - 1.0) * 100.0
            peak_drawdown_pct = (price / position.peak_price - 1.0) * 100.0
            sell_fraction = 0.0
            reason = ""
            if pnl_pct <= -params.stop_loss_pct:
                sell_fraction = 1.0
                reason = "stop"
            elif not position.partial_taken and pnl_pct >= params.partial_take_profit_pct:
                sell_fraction = params.partial_fraction
                reason = "partial"
            elif pnl_pct >= params.take_profit_pct:
                sell_fraction = 1.0
                reason = "take_profit"
            elif peak_drawdown_pct <= -params.trailing_stop_pct and pnl_pct > 0.15:
                sell_fraction = 1.0
                reason = "trail_profit"
            elif peak_drawdown_pct <= -(params.trailing_stop_pct + 0.35) and pnl_pct < -0.05:
                sell_fraction = 1.0
                reason = "trail_defense"
            if sell_fraction > 0:
                qty = position.qty * sell_fraction
                proceeds = apply_sell_price(price) * qty
                fee = proceeds * FEE_RATE
                cash += proceeds - fee
                cost_basis = position.avg_price * qty
                pnl = proceeds - fee - cost_basis
                trades.append({"side": "sell", "market": position.market, "price": price, "pnl": pnl, "reason": reason})
                if pnl >= 0:
                    wins.append(pnl)
                else:
                    losses.append(pnl)
                position.qty -= qty
                if position.qty <= 1e-12:
                    position = None
                else:
                    position.partial_taken = True

        if position is None:
            pick = choose_market(candles_by_market, index, params.name)
            if pick is not None:
                market, score, reason = pick
                if score >= params.min_score:
                    buy_cash = min(cash, max(0.0, equity * params.entry_fraction))
                    if buy_cash >= 50_000:
                        price = apply_buy_price(latest_prices[market])
                        fee = buy_cash * FEE_RATE
                        qty = (buy_cash - fee) / price
                        cash -= buy_cash
                        position = Position(market=market, qty=qty, avg_price=price, peak_price=price, entry_reason=reason)
                        trades.append({"side": "buy", "market": market, "price": latest_prices[market], "score": score, "reason": reason})

    final_prices = {market: candles_by_market[market][-1].close for market in markets}
    ending_equity = cash + (position.qty * final_prices[position.market] if position else 0.0)
    realized = sum(float(trade.get("pnl", 0.0)) for trade in trades)
    total_return_pct = (ending_equity / starting_cash - 1.0) * 100.0
    daily_compound_pct = ((ending_equity / starting_cash) ** (1.0 / max(days, 1)) - 1.0) * 100.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    sell_count = len(wins) + len(losses)
    win_rate = len(wins) / sell_count * 100.0 if sell_count else 0.0
    max_dd_krw, max_dd_pct = max_drawdown(equity_curve or [starting_cash])
    return Result(
        params=params,
        starting_cash=starting_cash,
        ending_equity=ending_equity,
        realized_pnl=realized,
        total_return_pct=total_return_pct,
        daily_compound_pct=daily_compound_pct,
        trade_count=len(trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_krw=max_dd_krw,
        max_consecutive_losses=max_consecutive_losses(losses_from_trades(trades)),
        open_position_count=1 if position else 0,
        trades=trades[-20:],
    )


def choose_market(candles_by_market: dict[str, list[Candle]], index: int, model: str) -> tuple[str, float, str] | None:
    best: tuple[str, float, str] | None = None
    for market, candles in candles_by_market.items():
        if index < 30:
            continue
        window = candles[: index + 1]
        score, reason = model_score(window, model)
        if best is None or score > best[1]:
            best = (market, score, reason)
    return best


def model_score(candles: list[Candle], model: str) -> tuple[float, str]:
    closes = [c.close for c in candles]
    latest = candles[-1]
    prev = candles[-2]
    momentum1 = pct(latest.close, prev.close)
    momentum3 = pct(latest.close, closes[-4])
    momentum8 = pct(latest.close, closes[-9])
    volume_ratio = latest.volume / max(mean([c.volume for c in candles[-20:-1]]), 1e-9)
    rsi = calculate_rsi(closes, 14) or 50.0
    high = max(c.high for c in candles[-20:])
    low = min(c.low for c in candles[-20:])
    close_pos = 1.0 if high <= low else (latest.close - low) / (high - low)
    mid, upper, lower, width = bollinger(closes[-20:])

    active = (
        0.35 * clamp(momentum3 / 1.8, -1.0, 1.0)
        + 0.25 * clamp(momentum8 / 3.2, -1.0, 1.0)
        + 0.20 * clamp((volume_ratio - 1.0) / 2.5, -0.5, 1.0)
        + 0.15 * close_pos
        - 0.15 * max((rsi - 74.0) / 18.0, 0.0)
    )
    breakout = (
        0.45 * (1.0 if latest.close > upper and momentum1 > 0 else 0.0)
        + 0.25 * clamp(momentum3 / 2.0, -1.0, 1.0)
        + 0.20 * clamp((volume_ratio - 1.0) / 2.0, -0.5, 1.0)
        + 0.10 * clamp(width / 6.0, 0.0, 1.0)
        - 0.12 * max((rsi - 76.0) / 18.0, 0.0)
    )
    reversal = (
        0.40 * (1.0 if latest.low <= lower * 1.006 and latest.close > prev.close else 0.0)
        + 0.25 * clamp((45.0 - rsi) / 24.0, -0.5, 1.0)
        + 0.20 * clamp((mid / latest.close - 1.0) * 100.0 / 2.5, -0.5, 1.0)
        + 0.15 * clamp((volume_ratio - 0.8) / 2.2, -0.5, 1.0)
    )

    if model == "bb_breakout":
        return breakout, f"bb_breakout m3={momentum3:.2f} vol={volume_ratio:.2f} rsi={rsi:.1f}"
    if model == "bb_reversal":
        return reversal, f"bb_reversal m1={momentum1:.2f} rsi={rsi:.1f}"
    if model == "hybrid_rank":
        if breakout >= reversal and breakout >= active:
            return breakout + 0.03, f"hybrid breakout m3={momentum3:.2f} vol={volume_ratio:.2f} rsi={rsi:.1f}"
        if reversal >= active:
            return reversal + 0.02, f"hybrid reversal m1={momentum1:.2f} rsi={rsi:.1f}"
        return active, f"hybrid active m3={momentum3:.2f} vol={volume_ratio:.2f} rsi={rsi:.1f}"
    return active, f"active_rank m3={momentum3:.2f} m8={momentum8:.2f} vol={volume_ratio:.2f} rsi={rsi:.1f}"


def bollinger(values: list[float]) -> tuple[float, float, float, float]:
    mid = mean(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    width = ((upper / lower) - 1.0) * 100.0 if lower > 0 else 0.0
    return mid, upper, lower, width


def pct(a: float, b: float) -> float:
    return (a / b - 1.0) * 100.0 if b else 0.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def apply_buy_price(price: float) -> float:
    return price * (1.0 + SLIPPAGE_BPS / 10000.0)


def apply_sell_price(price: float) -> float:
    return price * (1.0 - SLIPPAGE_BPS / 10000.0)


def max_drawdown(values: list[float]) -> tuple[float, float]:
    peak = values[0]
    worst = 0.0
    worst_pct = 0.0
    for value in values:
        peak = max(peak, value)
        dd = value - peak
        dd_pct = (value / peak - 1.0) * 100.0 if peak else 0.0
        if dd < worst:
            worst = dd
            worst_pct = dd_pct
    return worst, worst_pct


def losses_from_trades(trades: list[dict]) -> list[float]:
    return [float(trade.get("pnl", 0.0)) for trade in trades if trade.get("side") == "sell"]


def max_consecutive_losses(values: list[float]) -> int:
    best = 0
    current = 0
    for value in values:
        if value < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def result_to_dict(result: Result) -> dict:
    data = asdict(result)
    data["params"] = asdict(result.params)
    return data


if __name__ == "__main__":
    main()
