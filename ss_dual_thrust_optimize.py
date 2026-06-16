#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不锈钢 SS Dual Thrust 深度参数优化。

在 ss_dual_thrust_backtest 基础上扩展搜索：
  - N / K / 检查间隔 / 时段
  - leverage（仓位占用比例，1=保证金打满，0.5=只用一半资金算手数）
  - 固定手数 futures_fixed_lots（可选）

输出按「夏普优先、且交易数>=min_trades」排序的最优组合。

用法:
    python ss_dual_thrust_optimize.py
    python ss_dual_thrust_optimize.py --min-trades 8
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from ss_dual_thrust_backtest import run_dual_thrust_backtest, base_config, DATA_CSV

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_GRID = DATA_DIR / "ss_dual_thrust_optimize_grid.csv"
OUT_BEST = DATA_DIR / "ss_dual_thrust_optimize_best.csv"

# 扩展网格
GRID = {
    "N": [1, 2, 3, 4, 5, 7, 10],
    "K": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5],
    "interval": [10, 15, 20, 30],
    "leverage": [0.3, 0.5, 0.7, 1.0],
    "sessions": {
        "上午": ((9, 0), (11, 30)),
        "上午1": ((9, 0), (10, 15)),      # 09:00-10:15
        "上午2": ((10, 30), (11, 30)),   # 10:30-11:30
        "下午": ((13, 30), (15, 0)),
        "全日": ((9, 0), (15, 0)),
    },
}


def run_one(csv_path: Path, N, K, interval, leverage, sess_name, t0, t1) -> dict | None:
    cfg = base_config(csv_path)
    cfg.update({
        "dt_lookback": N, "dt_K1": K, "dt_K2": K,
        "check_interval_minutes": interval,
        "leverage": leverage,
        "trading_start_time": t0, "trading_end_time": t1,
        "futures_fixed_lots": None,
    })
    try:
        daily_df, _, _, m = run_dual_thrust_backtest(cfg)
    except Exception:
        return None
    pf = m.get("profit_factor", 0)
    if pf == float("inf"):
        pf = 99.99
    return {
        "时段": sess_name, "N": N, "K": K, "间隔min": interval,
        "杠杆": leverage,
        "总回报%": round(m["total_return"] * 100, 1),
        "年化%": round(m["irr"] * 100, 1),
        "夏普": round(m["sharpe_ratio"], 3),
        "最大回撤%": round(m["mdd"] * 100, 1),
        "交易数": m["total_trades"],
        "胜率%": round(m["hit_ratio"] * 100, 1),
        "利润因子": round(pf, 2),
        "BH年化%": round(m["buy_hold_irr"] * 100, 1),
        "_days": len(daily_df),
    }


def show_lot_example():
    """打印 10 万资金、保证金打满时的手数计算示例。"""
    capital = 100_000
    margin_rate = 0.10
    ton = 5
    for price, label in [(13000, "典型价13000"), (14000, "典型价14000"), (15000, "典型价15000")]:
        margin_per_lot = price * ton * margin_rate
        lots = int(capital / margin_per_lot)
        used_margin = lots * margin_per_lot
        notional = lots * price * ton
        print(f"  {label}: 每手保证金={margin_per_lot:,.0f}元 → 打满可开 {lots} 手 "
              f"(占用保证金 {used_margin:,.0f}, 名义价值 {notional:,.0f}, 约 {notional/capital:.1f}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DATA_CSV)
    ap.add_argument("--min-trades", type=int, default=8, help="稳健筛选最少交易笔数")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"数据不存在: {csv_path}，请先运行 ss_dual_thrust_backtest.py 或 ss_sina_recent_validation.py")

    print("=" * 70)
    print("仓位说明（当前默认 leverage=1 = 保证金打满）")
    print("=" * 70)
    print("公式: 手数 = floor(资金 × leverage / (开盘价 × 5吨/手 × 保证金率10%))")
    print("初始资金 10 万元示例:")
    show_lot_example()
    print("\nleverage=0.5 表示只用一半资金算手数（约 7~8 手）；leverage=0.3 约 4~5 手。")
    print("每日开盘按当日权益重新算手数，当日固定不变。\n")

    combos = list(product(
        GRID["N"], GRID["K"], GRID["interval"], GRID["leverage"], GRID["sessions"].items()
    ))
    print(f"开始扫描 {len(combos)} 组参数…")
    rows = []
    for i, (N, K, interval, lev, (sess, (t0, t1))) in enumerate(combos, 1):
        row = run_one(csv_path, N, K, interval, lev, sess, t0, t1)
        if row:
            rows.append(row)
        if i % 500 == 0 or i == len(combos):
            print(f"  … {i}/{len(combos)}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_GRID, index=False)

    robust = res[(res["交易数"] >= args.min_trades) & (res["夏普"] > 0)].copy()
    by_sharpe = robust.sort_values("夏普", ascending=False)
    by_calmar = robust.copy()
    by_calmar["calmar"] = by_calmar["年化%"] / by_calmar["最大回撤%"].clip(lower=1)
    by_calmar = by_calmar.sort_values("calmar", ascending=False)

    print(f"\n{'='*70}\n稳健 Top{args.top}（交易数>={args.min_trades}, 夏普>0, 按夏普）\n{'='*70}")
    cols = ["时段", "N", "K", "间隔min", "杠杆", "年化%", "夏普", "最大回撤%", "交易数", "胜率%", "利润因子"]
    print(by_sharpe.head(args.top)[cols].to_string(index=False))

    print(f"\n{'='*70}\n稳健 Top10（按 Calmar = 年化/回撤）\n{'='*70}")
    print(by_calmar.head(10)[cols + ["calmar"]].to_string(index=False))

    # 各杠杆档最优
    print(f"\n{'='*70}\n各杠杆档最优夏普（交易数>={args.min_trades}）\n{'='*70}")
    for lev in GRID["leverage"]:
        sub = robust[robust["杠杆"] == lev].sort_values("夏普", ascending=False).head(1)
        if len(sub):
            r = sub.iloc[0]
            print(f"  leverage={lev}: {r['时段']} N{r['N']} K{r['K']} {int(r['间隔min'])}m | "
                  f"年化{r['年化%']}% 夏普{r['夏普']} 回撤{r['最大回撤%']}% {int(r['交易数'])}笔")

    best = by_sharpe.head(5)
    best.to_csv(OUT_BEST, index=False)
    print(f"\n[明细] {OUT_GRID} ({len(res)} 组)")
    print(f"[Top5] {OUT_BEST}")


if __name__ == "__main__":
    main()
