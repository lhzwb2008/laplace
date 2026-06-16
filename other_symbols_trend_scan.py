#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
除不锈钢外其它品种的 Dual Thrust 扩展参数扫描。

目标：不同品种日内趋势规律不同，用更宽的参数网格（N/K/时段/检查间隔）
找出每个品种的最优组合，并比较「谁最适合趋势突破策略」。

不锈钢结果保留在 data/multi_symbol_dual_thrust_grid.csv，本脚本不重复跑 ss。

用法:
    python other_symbols_trend_scan.py
    python other_symbols_trend_scan.py --symbols rb au
    python other_symbols_trend_scan.py --quick   # 粗网格快速试跑
"""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import pandas as pd

from multi_symbol_dual_thrust import SYMBOL_SPECS, build_or_load_main_csv, DATA_DIR
from ss_dual_thrust_backtest import run_dual_thrust_backtest, base_config

ROOT_DIR = Path(__file__).resolve().parent
SS_GRID_CSV = DATA_DIR / "multi_symbol_dual_thrust_grid.csv"
OUT_GRID_CSV = DATA_DIR / "other_symbols_trend_scan_grid.csv"
OUT_RANK_CSV = DATA_DIR / "other_symbols_trend_ranking.csv"

OTHER_SYMBOLS = ["c", "jd", "au", "rb"]

# 扩展网格：覆盖「窄轨高频」到「宽轨低频」多种趋势捕捉方式
FULL_GRID = {
    "N": [1, 2, 3, 4, 5],
    "K": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5],
    "interval": [10, 15, 20, 30],
    "sessions": {
        "上午": ((9, 0), (11, 30)),
        "下午": ((13, 30), (15, 0)),
        "全日": ((9, 0), (15, 0)),
    },
}

QUICK_GRID = {
    "N": [1, 2, 3, 5],
    "K": [0.3, 0.5, 0.7, 1.0, 1.5],
    "interval": [15, 20],
    "sessions": {
        "上午": ((9, 0), (11, 30)),
        "下午": ((13, 30), (15, 0)),
        "全日": ((9, 0), (15, 0)),
    },
}


def _csv_path(prefix: str) -> Path:
    if prefix == "ss":
        return DATA_DIR / "ss_sina_raw_prevday_main_1m.csv"
    return DATA_DIR / f"{prefix}_sina_prevday_main_1m.csv"


def run_one(prefix: str, csv_path: Path, N: int, K: float, interval: int,
            sess_name: str, t0, t1) -> dict | None:
    spec = SYMBOL_SPECS[prefix]
    cfg = base_config(csv_path)
    cfg.update({
        "ticker": prefix,
        "dt_lookback": N,
        "dt_K1": K,
        "dt_K2": K,
        "check_interval_minutes": interval,
        "trading_start_time": t0,
        "trading_end_time": t1,
        "contract_multiplier": spec["mult"],
        "futures_ton_per_lot": spec["mult"],
        "futures_margin_rate": spec["margin"],
        "tick_size": spec["tick"],
        "slippage_ticks": spec["slip_ticks"],
        "futures_fee_per_lot": spec["fee"],
        "print_daily_trades": False,
    })
    try:
        daily_df, _, trades_df, m = run_dual_thrust_backtest(cfg)
    except Exception:
        return None

    irr = m["irr"]
    bh_irr = m["buy_hold_irr"]
    alpha = irr - bh_irr
    pf = m.get("profit_factor", 0.0)
    if pf == float("inf"):
        pf = 99.99

    return {
        "品种": spec["name"],
        "前缀": prefix,
        "时段": sess_name,
        "N": N,
        "K": K,
        "间隔min": interval,
        "总回报%": round(m["total_return"] * 100, 1),
        "年化%": round(irr * 100, 1),
        "BH年化%": round(bh_irr * 100, 1),
        "Alpha%": round(alpha * 100, 1),
        "夏普": round(m["sharpe_ratio"], 2),
        "最大回撤%": round(m["mdd"] * 100, 1),
        "交易数": m["total_trades"],
        "胜率%": round(m["hit_ratio"] * 100, 1),
        "利润因子": round(pf, 2),
        "_days": len(daily_df),
        "_positive": irr > 0,
        "_beats_bh": alpha > 0,
    }


def scan_symbol(prefix: str, csv_path: Path, grid: dict) -> list[dict]:
    spec = SYMBOL_SPECS[prefix]
    rows = []
    combos = list(product(
        grid["N"], grid["K"], grid["interval"], grid["sessions"].items()
    ))
    total = len(combos)
    print(f"[{spec['name']}] 开始扫描 {total} 组参数…")

    for i, (N, K, interval, (sess_name, (t0, t1))) in enumerate(combos, 1):
        row = run_one(prefix, csv_path, N, K, interval, sess_name, t0, t1)
        if row:
            rows.append(row)
        if i % 100 == 0 or i == total:
            print(f"  … {i}/{total} 完成")

    if rows:
        pos = sum(r["_positive"] for r in rows)
        beat = sum(r["_beats_bh"] for r in rows)
        print(f"[{spec['name']}] 完成 {len(rows)} 组 | 正收益 {pos}/{len(rows)} | 跑赢BH {beat}/{len(rows)}")
    return rows


def summarize_ranking(res: pd.DataFrame, ss_best: pd.DataFrame | None) -> pd.DataFrame:
    """按品种汇总趋势适配度，并给出各品种最优参数。"""
    rank_rows = []
    for prefix in res["前缀"].unique():
        sub = res[res["前缀"] == prefix]
        best_sharpe = sub.sort_values("夏普", ascending=False).iloc[0]
        best_alpha = sub.sort_values("Alpha%", ascending=False).iloc[0]
        best_irr = sub.sort_values("年化%", ascending=False).iloc[0]

        pos_rate = sub["_positive"].mean()
        beat_bh_rate = sub["_beats_bh"].mean()
        median_sharpe = sub["夏普"].median()
        top10_sharpe_mean = sub.sort_values("夏普", ascending=False).head(10)["夏普"].mean()

        # 趋势适配分：正参数占比 + Top夏普 + 相对BH超额
        trend_score = (
            pos_rate * 30
            + beat_bh_rate * 20
            + max(0, best_sharpe["夏普"]) * 25
            + max(0, best_alpha["Alpha%"]) * 0.5
            + top10_sharpe_mean * 10
        )

        rank_rows.append({
            "品种": best_sharpe["品种"],
            "前缀": prefix,
            "样本日": int(best_sharpe["_days"]),
            "BH年化%": best_sharpe["BH年化%"],
            "正收益参数占比%": round(pos_rate * 100, 1),
            "跑赢BH参数占比%": round(beat_bh_rate * 100, 1),
            "中位夏普": round(median_sharpe, 2),
            "Top10均夏普": round(top10_sharpe_mean, 2),
            "趋势适配分": round(trend_score, 1),
            "最优夏普": best_sharpe["夏普"],
            "最优参数": f"{best_sharpe['时段']} N{best_sharpe['N']} K{best_sharpe['K']} {best_sharpe['间隔min']}m",
            "最优年化%": best_sharpe["年化%"],
            "最优Alpha%": best_sharpe["Alpha%"],
            "最优回撤%": best_sharpe["最大回撤%"],
            "最优交易数": best_sharpe["交易数"],
            "最高Alpha参数": f"{best_alpha['时段']} N{best_alpha['N']} K{best_alpha['K']} {best_alpha['间隔min']}m",
            "最高Alpha%": best_alpha["Alpha%"],
            "最高年化参数": f"{best_irr['时段']} N{best_irr['N']} K{best_irr['K']} {best_irr['间隔min']}m",
            "最高年化%": best_irr["年化%"],
        })

    rank = pd.DataFrame(rank_rows).sort_values("趋势适配分", ascending=False).reset_index(drop=True)

    if ss_best is not None and len(ss_best) > 0:
        ss = ss_best.iloc[0]
        ss_row = {
            "品种": "不锈钢(保留)",
            "前缀": "ss",
            "样本日": "-",
            "BH年化%": ss.get("BH年化%", ss.get("年化%", 0)),
            "正收益参数占比%": "-",
            "跑赢BH参数占比%": "-",
            "中位夏普": "-",
            "Top10均夏普": "-",
            "趋势适配分": "-",
            "最优夏普": ss["夏普"],
            "最优参数": f"{ss['时段']} N{ss['N']} K{ss['K']}",
            "最优年化%": ss["年化%"],
            "最优Alpha%": "-",
            "最优回撤%": ss["最大回撤%"],
            "最优交易数": ss["交易数"],
            "最高Alpha参数": "-",
            "最高Alpha%": "-",
            "最高年化参数": "-",
            "最高年化%": ss["年化%"],
        }
        rank = pd.concat([pd.DataFrame([ss_row]), rank], ignore_index=True)

    return rank


def load_ss_best() -> pd.DataFrame | None:
    if not SS_GRID_CSV.is_file():
        return None
    ss = pd.read_csv(SS_GRID_CSV)
    ss = ss[ss["前缀"] == "ss"].sort_values("夏普", ascending=False).head(1)
    return ss


def main():
    ap = argparse.ArgumentParser(description="其它品种扩展趋势参数扫描（不含不锈钢）")
    ap.add_argument("--symbols", nargs="*", default=OTHER_SYMBOLS)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true", help="粗网格快速扫描")
    args = ap.parse_args()

    grid = QUICK_GRID if args.quick else FULL_GRID
    n_combos = len(grid["N"]) * len(grid["K"]) * len(grid["interval"]) * len(grid["sessions"])
    print(f"网格规模: N{len(grid['N'])} × K{len(grid['K'])} × 间隔{len(grid['interval'])} × 时段{len(grid['sessions'])} = {n_combos} 组/品种")

    all_rows = []
    for prefix in args.symbols:
        if prefix == "ss":
            print("跳过不锈钢（结果已保留在 multi_symbol_dual_thrust_grid.csv）")
            continue
        if prefix not in SYMBOL_SPECS:
            print(f"未知品种 {prefix}")
            continue
        print(f"\n{'#'*60}\n# {SYMBOL_SPECS[prefix]['name']} ({prefix})\n{'#'*60}")
        path = build_or_load_main_csv(prefix, _csv_path(prefix), rebuild=args.rebuild, sleep_s=args.sleep)
        if path is None:
            continue
        all_rows.extend(scan_symbol(prefix, path, grid))

    if not all_rows:
        print("无结果")
        return

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT_GRID_CSV, index=False)

    ss_best = load_ss_best()
    rank = summarize_ranking(res, ss_best)
    rank.to_csv(OUT_RANK_CSV, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", None)

    print(f"\n{'='*72}\n品种趋势适配排名（不锈钢仅作对照，不参与本次扫描）\n{'='*72}")
    cols = ["品种", "BH年化%", "正收益参数占比%", "跑赢BH参数占比%", "趋势适配分",
            "最优夏普", "最优参数", "最优年化%", "最优Alpha%", "最优回撤%"]
    print(rank[cols].to_string(index=False))

    print(f"\n{'='*72}\n各品种 Top5 参数（按夏普）\n{'='*72}")
    for name in res["品种"].unique():
        sub = res[res["品种"] == name].sort_values("夏普", ascending=False).head(5)
        print(f"\n[{name}] BH年化 {sub.iloc[0]['BH年化%']}%")
        print(sub[["时段", "N", "K", "间隔min", "年化%", "Alpha%", "夏普", "最大回撤%", "交易数", "胜率%", "利润因子"]].to_string(index=False))

    print(f"\n{'='*72}\n各品种 Top5 参数（按 Alpha，相对 Buy&Hold 超额）\n{'='*72}")
    for name in res["品种"].unique():
        sub = res[res["品种"] == name].sort_values("Alpha%", ascending=False).head(5)
        print(f"\n[{name}]")
        print(sub[["时段", "N", "K", "间隔min", "年化%", "Alpha%", "夏普", "最大回撤%", "交易数"]].to_string(index=False))

    top = rank[rank["前缀"] != "ss"].iloc[0] if len(rank[rank["前缀"] != "ss"]) else None
    if top is not None:
        print(f"\n{'='*72}")
        print(f"结论：除不锈钢外，趋势适配最佳为 【{top['品种']}】")
        print(f"  推荐参数: {top['最优参数']}")
        print(f"  年化 {top['最优年化%']}% | Alpha {top['最优Alpha%']}% | 夏普 {top['最优夏普']} | 回撤 {top['最优回撤%']}%")
        print(f"  该品种 {top['正收益参数占比%']}% 的参数组合为正收益，{top['跑赢BH参数占比%']}% 跑赢 Buy&Hold")
        print(f"{'='*72}")

    print(f"\n[明细] {OUT_GRID_CSV} ({len(res)} 组)")
    print(f"[排名] {OUT_RANK_CSV}")


if __name__ == "__main__":
    main()
