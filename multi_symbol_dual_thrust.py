#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多品种 × 多参数 经典 Dual Thrust 回测。

在 ss_dual_thrust_backtest 的基础上，扩展到多品种（不锈钢/玉米/鸡蛋/黄金/螺纹钢），
并对每个品种跑一组参数网格（N × K × 时段），输出对比表。

数据：复用本项目新浪渠道 ak.futures_zh_minute_sina，按「上一交易日成交量最大可交易合约」选主力，
只保留日盘（09:00–10:15 / 10:30–11:30 / 13:30–15:00），各品种缓存独立 CSV。

说明：本策略的百分比收益对合约乘数近似不敏感（名义敞口=资金×杠杆/保证金率，与乘数无关），
真正影响结果的是「保证金率」与「滑点/价格 的比例（≈tick×滑点tick/价格）」，故按品种设置 tick/保证金/手续费。

用法：
    python multi_symbol_dual_thrust.py                 # 复用缓存(若无则拉取)，跑全部品种网格
    python multi_symbol_dual_thrust.py --rebuild       # 重新拉取所有品种数据
    python multi_symbol_dual_thrust.py --symbols ss rb # 只跑指定品种
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from futures_minute_sina import candidate_symbols_for_prefix, fetch_one
from ss_sina_recent_validation import is_day_session
from ss_dual_thrust_backtest import run_dual_thrust_backtest, base_config

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
INITIAL_CAPITAL = 100_000.0

# 各品种合约规格（mult=合约乘数, tick=最小变动价位[价格单位], fee=每手每边手续费[元],
# margin=保证金率）。tick/价格 决定滑点占比，是跨品种可比性的关键。
SYMBOL_SPECS = {
    "ss": {"name": "不锈钢", "mult": 5.0, "tick": 5.0, "slip_ticks": 0.5, "fee": 2.0, "margin": 0.10},
    "c":  {"name": "玉米",   "mult": 10.0, "tick": 1.0, "slip_ticks": 0.5, "fee": 1.2, "margin": 0.08},
    "jd": {"name": "鸡蛋",   "mult": 10.0, "tick": 1.0, "slip_ticks": 0.5, "fee": 3.0, "margin": 0.09},
    "au": {"name": "黄金",   "mult": 1000.0, "tick": 0.02, "slip_ticks": 0.5, "fee": 10.0, "margin": 0.10},
    "rb": {"name": "螺纹钢", "mult": 10.0, "tick": 1.0, "slip_ticks": 0.5, "fee": 3.0, "margin": 0.10},
}

# 参数网格
GRID_N = [1, 2, 3]
GRID_K = [0.3, 0.5, 0.7, 1.0]
GRID_SESSIONS = {
    "上午": ((9, 0), (11, 30)),
    "全日": ((9, 0), (15, 0)),
}


def build_or_load_main_csv(prefix: str, out_csv: Path, rebuild: bool, sleep_s: float) -> Path | None:
    """通用：按前缀拉所有合约 1min K → 过滤日盘 → 选上一交易日主力 → 写标准回测 CSV。"""
    if out_csv.is_file() and not rebuild:
        return out_csv

    chunks = []
    for symbol in candidate_symbols_for_prefix(prefix):
        df = fetch_one(symbol, "1")
        if df is None:
            continue
        chunks.append(df)
        time.sleep(sleep_s)
    if not chunks:
        print(f"[{prefix}] 未拉到任何合约 1min K，跳过")
        return None

    raw = pd.concat(chunks, ignore_index=True).sort_values(["datetime", "contract"]).reset_index(drop=True)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["date"] = raw["datetime"].dt.date

    day_raw = raw[raw["datetime"].map(is_day_session)].copy()
    if day_raw.empty:
        print(f"[{prefix}] 日盘过滤后为空，跳过")
        return None

    daily_volume = day_raw.groupby(["date", "contract"], as_index=False)["volume"].sum()
    dates = sorted(daily_volume["date"].unique())
    selected = []
    for i, current_date in enumerate(dates):
        ranking_date = dates[i - 1] if i > 0 else current_date
        ranked = (
            daily_volume[daily_volume["date"] == ranking_date]
            .sort_values("volume", ascending=False)["contract"].tolist()
        )
        available = set(daily_volume[daily_volume["date"] == current_date]["contract"])
        picked = next((c for c in ranked if c in available), None)
        if picked is None:
            picked = (
                daily_volume[daily_volume["date"] == current_date]
                .sort_values("volume", ascending=False)["contract"].iloc[0]
            )
        selected.append({"date": current_date, "main_contract": picked})

    main = day_raw.merge(pd.DataFrame(selected), on="date", how="left")
    main = main[main["contract"] == main["main_contract"]].copy()
    clean = pd.DataFrame({
        "DateTime": main["datetime"],
        "Date": main["date"].astype(str),
        "TradingDate": main["date"].astype(str),
        "Contract": main["contract"],
        "Open": pd.to_numeric(main["open"], errors="coerce"),
        "High": pd.to_numeric(main["high"], errors="coerce"),
        "Low": pd.to_numeric(main["low"], errors="coerce"),
        "Close": pd.to_numeric(main["close"], errors="coerce"),
        "Volume": pd.to_numeric(main["volume"], errors="coerce"),
        "OpenInterest": pd.to_numeric(main["hold"], errors="coerce"),
    }).dropna(subset=["DateTime", "Open", "High", "Low", "Close"])
    clean["Turnover"] = clean["Close"] * clean["Volume"]
    clean = clean.sort_values("DateTime").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_csv, index=False)
    print(f"[{prefix}] 已写入 {out_csv}：{len(clean)} 根，{clean['Date'].min()}~{clean['Date'].max()}，"
          f"{clean['Date'].nunique()} 日，{clean['Contract'].nunique()} 合约")
    return out_csv


def run_grid_for_symbol(prefix: str, csv_path: Path) -> list[dict]:
    spec = SYMBOL_SPECS[prefix]
    rows = []
    bh_irr = None
    for sess_name, (t0, t1) in GRID_SESSIONS.items():
        for N in GRID_N:
            for K in GRID_K:
                cfg = base_config(csv_path)
                cfg.update({
                    "ticker": prefix,
                    "dt_lookback": N, "dt_K1": K, "dt_K2": K,
                    "trading_start_time": t0, "trading_end_time": t1,
                    "contract_multiplier": spec["mult"], "futures_ton_per_lot": spec["mult"],
                    "futures_margin_rate": spec["margin"], "tick_size": spec["tick"],
                    "slippage_ticks": spec["slip_ticks"], "futures_fee_per_lot": spec["fee"],
                })
                try:
                    daily_df, _, trades_df, m = run_dual_thrust_backtest(cfg)
                except Exception as e:
                    print(f"[{prefix}] N={N} K={K} {sess_name} 失败: {e}")
                    continue
                bh_irr = m["buy_hold_irr"]
                rows.append({
                    "品种": spec["name"], "前缀": prefix, "时段": sess_name,
                    "N": N, "K": K,
                    "总回报%": round(m["total_return"] * 100, 1),
                    "年化%": round(m["irr"] * 100, 1),
                    "夏普": round(m["sharpe_ratio"], 2),
                    "最大回撤%": round(m["mdd"] * 100, 1),
                    "交易数": m["total_trades"],
                    "胜率%": round(m["hit_ratio"] * 100, 1),
                    "利润因子": round(m.get("profit_factor", 0), 2),
                    "_days": len(daily_df),
                })
    if rows:
        print(f"\n[{spec['name']} {prefix}] 网格完成 {len(rows)} 组 | 样本日 {rows[0]['_days']} | Buy&Hold 年化≈{bh_irr*100:.1f}%")
    return rows


def main():
    ap = argparse.ArgumentParser(description="多品种 × 多参数 Dual Thrust 回测")
    ap.add_argument("--rebuild", action="store_true", default=False)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--symbols", nargs="*", default=list(SYMBOL_SPECS.keys()),
                    help="要测试的品种前缀，默认全部: ss c jd au rb")
    args = ap.parse_args()

    all_rows = []
    for prefix in args.symbols:
        if prefix not in SYMBOL_SPECS:
            print(f"未知品种 {prefix}，跳过（可选: {list(SYMBOL_SPECS)}）")
            continue
        # 不锈钢复用既有缓存文件名，其它品种各自缓存
        if prefix == "ss":
            csv_path = DATA_DIR / "ss_sina_raw_prevday_main_1m.csv"
        else:
            csv_path = DATA_DIR / f"{prefix}_sina_prevday_main_1m.csv"
        print(f"\n{'#'*60}\n# {SYMBOL_SPECS[prefix]['name']} ({prefix})\n{'#'*60}")
        path = build_or_load_main_csv(prefix, csv_path, rebuild=args.rebuild, sleep_s=args.sleep)
        if path is None:
            continue
        all_rows.extend(run_grid_for_symbol(prefix, path))

    if not all_rows:
        print("没有任何回测结果")
        return

    res = pd.DataFrame(all_rows).drop(columns=["_days"])
    out_csv = DATA_DIR / "multi_symbol_dual_thrust_grid.csv"
    res.to_csv(out_csv, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)

    print(f"\n{'='*70}\n各品种最优参数（按夏普）\n{'='*70}")
    best = (res.sort_values("夏普", ascending=False)
            .groupby("品种", as_index=False).head(1)
            .sort_values("夏普", ascending=False))
    print(best.to_string(index=False))

    print(f"\n{'='*70}\n各品种 Top3（按夏普）\n{'='*70}")
    for name in res["品种"].unique():
        sub = res[res["品种"] == name].sort_values("夏普", ascending=False).head(3)
        print(f"\n[{name}]")
        print(sub[["时段", "N", "K", "年化%", "夏普", "最大回撤%", "交易数", "胜率%", "利润因子"]].to_string(index=False))

    print(f"\n[汇总] 全部 {len(res)} 组结果已写入 {out_csv}")


if __name__ == "__main__":
    main()
