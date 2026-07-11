from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coin_mvp.backtest import BacktestSummary, run_backtest


VariantPatch = dict[str, Any]


VARIANTS: dict[str, VariantPatch] = {
    "current_hybrid": {},
    "regime_ensemble_only": {
        "strategy": {
            "regime_ensemble_only": True,
            "min_expected_upside_pct": 0.95,
            "min_volume_ratio": 1.10,
            "partial_take_profit_pct": 0.50,
            "time_stop_min_pnl_pct": 0.18,
        },
        "risk": {
            "min_entries_per_day": 0,
            "min_candidate_score": 0.46,
        },
    },
    "cost_aware_momentum": {
        "strategy": {
            "regime_ensemble_only": False,
            "min_expected_upside_pct": 1.20,
            "min_net_edge_pct": 0.20,
            "min_volume_ratio": 1.25,
            "take_profit_pct": 1.10,
            "partial_take_profit_pct": 0.55,
            "partial_take_profit_fraction": 0.78,
            "trailing_stop_pct": 0.22,
            "time_stop_min_pnl_pct": 0.18,
            "range_rebound_min_expected_upside_pct": 1.25,
            "range_rebound_min_volume_ratio": 1.30,
        },
        "risk": {
            "min_entries_per_day": 0,
            "min_candidate_score": 0.50,
            "max_expected_downside_to_upside_ratio": 0.58,
        },
    },
    "active_cost_aware": {
        "strategy": {
            "regime_ensemble_only": False,
            "min_expected_upside_pct": 0.95,
            "min_net_edge_pct": 0.15,
            "min_volume_ratio": 1.12,
            "partial_take_profit_pct": 0.48,
            "partial_take_profit_fraction": 0.74,
            "trailing_stop_pct": 0.22,
            "time_stop_min_pnl_pct": 0.16,
            "range_rebound_min_expected_upside_pct": 1.05,
            "range_rebound_min_volume_ratio": 1.18,
        },
        "risk": {
            "min_entries_per_day": 2,
            "min_candidate_score": 0.45,
            "max_expected_downside_to_upside_ratio": 0.68,
        },
    },
    "participation_heavy": {
        "strategy": {
            "regime_ensemble_only": False,
            "min_expected_upside_pct": 0.75,
            "min_volume_ratio": 0.95,
            "partial_take_profit_pct": 0.42,
            "partial_take_profit_fraction": 0.72,
            "time_stop_min_pnl_pct": 0.14,
            "range_rebound_min_expected_upside_pct": 0.90,
            "range_rebound_min_volume_ratio": 1.00,
        },
        "risk": {
            "min_entries_per_day": 4,
            "min_candidate_score": 0.42,
            "max_expected_downside_to_upside_ratio": 0.78,
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare strategy variants on the same replay window.")
    parser.add_argument("--config", default="config.cloud.json")
    parser.add_argument("--source", choices=["sample", "upbit"], default="upbit")
    parser.add_argument("--top-markets", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument("--step-minutes", type=int, default=1)
    parser.add_argument("--history-count", type=int, default=240)
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("reports") / f"strategy_comparison_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for name, patch in VARIANTS.items():
        config = apply_patch_to_config(base_config, patch)
        config_path = out_dir / f"{name}.json"
        report_path = out_dir / f"{name}.html"
        summary_path = out_dir / f"{name}_summary.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = run_backtest(
            config_path=config_path,
            source=args.source,
            top_markets=args.top_markets,
            ticks=args.ticks,
            step_minutes=args.step_minutes,
            history_count=args.history_count,
            output=report_path,
            summary_output=summary_path,
            request_delay=args.request_delay,
        )
        row = summarize_variant(name, summary)
        rows.append(row)
        print(format_row(row), flush=True)

    rows.sort(key=variant_score, reverse=True)
    write_outputs(out_dir, rows, args)
    print(f"\nBest variant: {rows[0]['variant']}")
    print(f"Summary: {out_dir / 'comparison_summary.md'}")


def apply_patch_to_config(base: dict[str, Any], patch: VariantPatch) -> dict[str, Any]:
    config = deepcopy(base)
    deep_update(config, patch)
    return config


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def summarize_variant(name: str, summary: BacktestSummary) -> dict[str, Any]:
    data = summary.to_dict()
    return {
        "variant": name,
        "return_pct": data["return_pct"],
        "pnl": data["total_realized_pnl"],
        "entries": data["entry_count"],
        "exits": data["exit_count"],
        "candidate_ticks": data["candidate_ticks"],
        "total_candidates": data["total_candidates"],
        "win_rate": data["win_rate"],
        "payoff": data["payoff_ratio"],
        "profit_factor": data["profit_factor"],
        "expectancy": data["expectancy"],
        "max_drawdown": data["max_drawdown"],
        "max_consecutive_losses": data["max_consecutive_losses"],
        "open_positions": data["open_position_count"],
        "verdict": data["verdict"],
        "report_path": data["report_path"],
    }


def variant_score(row: dict[str, Any]) -> float:
    profit_factor = min(float(row["profit_factor"]), 5.0)
    expectancy = float(row["expectancy"])
    exits = int(row["exits"])
    trade_sample_bonus = min(exits, 20) * 0.08
    sample_penalty = 4.0 if exits < 4 else 0.0
    return (
        float(row["return_pct"]) * 12.0
        + profit_factor * 1.4
        + expectancy / 1200.0
        + trade_sample_bonus
        + float(row["max_drawdown"]) / 12000.0
        - int(row["max_consecutive_losses"]) * 0.45
        - sample_penalty
    )


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Strategy Comparison",
        "",
        f"- source: {args.source}",
        f"- top_markets: {args.top_markets}",
        f"- ticks: {args.ticks}",
        f"- history_count: {args.history_count}",
        "",
        "| rank | variant | return % | pnl KRW | entries | exits | win % | payoff | profit factor | expectancy | max DD | verdict |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row["variant"]),
                    f"{float(row['return_pct']):.3f}",
                    f"{float(row['pnl']):.0f}",
                    str(row["entries"]),
                    str(row["exits"]),
                    f"{float(row['win_rate']) * 100:.1f}",
                    f"{float(row['payoff']):.2f}",
                    f"{float(row['profit_factor']):.2f}",
                    f"{float(row['expectancy']):.0f}",
                    f"{float(row['max_drawdown']):.0f}",
                    str(row["verdict"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Best variant is ranked by return, trade sample, profit factor, expectancy, drawdown, and loss streak.")
    (out_dir / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_row(row: dict[str, Any]) -> str:
    return (
        f"{row['variant']}: return={float(row['return_pct']):.3f}% "
        f"pnl={float(row['pnl']):.0f} exits={row['exits']} "
        f"win={float(row['win_rate']) * 100:.1f}% "
        f"pf={float(row['profit_factor']):.2f} "
        f"dd={float(row['max_drawdown']):.0f} verdict={row['verdict']}"
    )


if __name__ == "__main__":
    main()
