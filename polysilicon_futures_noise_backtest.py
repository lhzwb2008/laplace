#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多晶硅期货（广期所 PS）× Quantra「噪声通道 + VWAP」策略回测（**仅真实分钟 K**）。

**怎么跑**：在项目根目录执行 `python polysilicon_futures_noise_backtest.py`，默认读取 `data/ps_real_minute_1m.csv`。
数据需先由 `fetch_polysilicon_real_minute.py` 拉取。引擎在 `noise_strategy_backtest.py`，勿直接运行该文件。

说明（必读）
------------
1. 策略逻辑来自本目录 noise_strategy_backtest.py（Quantra/backtest 同源）：日内 |ret| 按「时刻」做滚动得到 sigma，
   再用前收/开盘参考价构造上下轨，配合 VWAP 做突破与止损。

2. **数据**：仅支持真实分钟 CSV（列含 `DateTime, Open, High, Low, Close, Volume`，建议含 `Turnover`）。
   先运行 `fetch_polysilicon_real_minute.py` 生成 `data/ps_real_minute_1m.csv`（默认 disjoint 瀑布拼接或 single 单合约）。

3. 交易时段：9:00–10:15、10:30–11:30、13:30–15:00（与公开规则一致）；午饭休市无 K 线。

4. 手续费、滑点默认关闭（理想化）。

5. 默认参数在 `default_config` / `PS_NOISE_STRATEGY_PARAMS`：**lb=1、8 分钟检查、K≈1.26、关闭 VWAP、linreg5_r2≥0.45、9:30 起交易、日内止损关、追踪止盈开**（激活 1%、保护 70% 浮盈）。

6. 仓位（全仓、不加杠杆）  
   `leverage=1`，不传 `futures_fixed_lots` / `futures_ton_per_lot` 时：  
   **`position_size = floor(当日日初权益 / 当日开盘参考价)`**，再用 **`盈亏 = position_size × 价差`**（元/吨标价下即「按标价满仓」的极简名义）。  
   同一交易日内若多笔成交，手数仍按**日初冻结**权益计算，不会在开仓之间随浮盈即时调仓（引擎原设计如此）。

7. 若盈亏量纲异常：检查是否误用「吨×价」名义却未在 `simulate_day` 里按合约吨数调整（本脚本默认按标价单位手数逻辑）。
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from noise_strategy_backtest import run_backtest


def dump_trade_details(trades_df, out_csv: Path) -> None:
    """逐笔打印并落盘，便于核对开仓/平仓价与 pnl 是否自洽。"""
    if trades_df is None or len(trades_df) == 0:
        print("\n[逐笔交易] 无成交记录。")
        return

    df = trades_df.copy()
    df = df.sort_values(["Date", "entry_time"]).reset_index(drop=True)

    def pnl_from_prices(row):
        ps = float(row["position_size"])
        ep, xp = float(row["entry_price"]), float(row["exit_price"])
        if row["side"] == "Long":
            return ps * (xp - ep)
        return ps * (ep - xp)

    df["_pnl_check"] = df.apply(pnl_from_prices, axis=1)
    df["_diff"] = (df["pnl"] - df["_pnl_check"]).abs()
    bad = df[df["_diff"] > 1e-4]
    if len(bad) > 0:
        print(f"\n[警告] 有 {len(bad)} 笔 pnl 与「手数×价差」不一致（应检查手续费或字段）:")
        print(bad[["Date", "side", "pnl", "_pnl_check", "_diff"]].head(10))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    export_cols = [
        c
        for c in [
            "Date",
            "side",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "position_size",
            "pnl",
            "exit_reason",
            "transaction_fees",
        ]
        if c in df.columns
    ]
    out = df[export_cols].copy()
    out.insert(0, "trade_id", range(1, len(out) + 1))
    out.to_csv(out_csv, index=False)
    print(f"\n[逐笔交易] 已写入 CSV: {out_csv}（共 {len(df)} 笔）")
    print("=" * 100)
    print(
        f"{'#':>4} {'日期':12} {'方向':6} {'开仓时间':19} {'平仓时间':19} "
        f"{'开仓价':>12} {'平仓价':>12} {'手数':>8} {'盈亏':>14} {'平仓原因'}"
    )
    print("=" * 100)
    for i, row in df.iterrows():
        et = row["entry_time"]
        xt = row["exit_time"]
        et_s = et.strftime("%Y-%m-%d %H:%M") if pd.notna(et) else ""
        xt_s = xt.strftime("%Y-%m-%d %H:%M") if pd.notna(xt) else ""
        d = row.get("Date", "")
        if hasattr(d, "strftime"):
            d = str(d)[:10]
        print(
            f"{i+1:4d} {str(d)[:12]:12} {row['side']:6} {et_s:19} {xt_s:19} "
            f"{row['entry_price']:12.4f} {row['exit_price']:12.4f} {row['position_size']:8.0f} "
            f"{row['pnl']:14.2f} {row.get('exit_reason', '')}"
        )
    print("=" * 100)
    wins = (df["pnl"] > 0).sum()
    print(
        f"[汇总] 共 {len(df)} 笔 | 盈利 {wins} | 亏损/平 {(df['pnl'] <= 0).sum()} | "
        f"胜率 {100 * wins / len(df):.2f}%"
    )


def print_minute_data_diagnostics(csv_path: Path) -> None:
    """真实分钟数据量级体检。"""
    df = pd.read_csv(csv_path, parse_dates=["DateTime"])
    df = df.sort_values("DateTime")
    r = df["Close"].pct_change()
    abs_r = r.abs().dropna()
    z = (abs_r < 1e-12).mean()
    print(
        f"[数据体检] 分钟收益 |r|: 均值={abs_r.mean():.6f}, 中位数={abs_r.median():.6f}, "
        f"近似零收益占比={z*100:.1f}%"
    )
    print(
        f"[数据体检] 样本区间: {df['DateTime'].min()} ~ {df['DateTime'].max()}，共 {len(df)} 根分钟K"
    )


# 当前默认采用的一组策略参数
PS_NOISE_STRATEGY_PARAMS = {
    "lookback_days": 1,
    "check_interval_minutes": 8,
    "enable_transaction_fees": False,
    "slippage_per_share": 0.0,
    "transaction_fee_per_share": 0.0,
    "trading_start_time": (9, 30),
    "trading_end_time": (15, 0),
    "max_positions_per_day": 5,
    "print_daily_trades": False,
    "print_trade_details": False,
    "K1": 1.26,
    "K2": 1.26,
    "leverage": 1,
    "use_vwap": False,
    "enable_intraday_stop_loss": False,
    "enable_trailing_take_profit": True,
    "trailing_tp_activation_pct": 0.01,
    "trailing_tp_callback_pct": 0.7,
    "entry_trend_filter": {"metric": "linreg5_r2", "min": 0.45},
}


def default_config(csv_path: Path) -> dict:
    """回测窗口由 start_date / end_date 控制。"""
    return {
        "data_path": str(csv_path),
        "ticker": "PS_GFEX_real1m",
        "initial_capital": 2_000_000.0,
        "start_date": date(2025, 1, 1),
        "end_date": date(2025, 12, 31),
        **PS_NOISE_STRATEGY_PARAMS,
    }


def main():
    ap = argparse.ArgumentParser(description="多晶硅噪声通道策略回测（仅真实分钟数据）")
    ap.add_argument(
        "--minute-csv",
        type=Path,
        default=None,
        help="分钟数据 CSV（默认 data/ps_real_minute_1m.csv）",
    )
    args = ap.parse_args()

    data_dir = Path(__file__).resolve().parent / "data"
    csv_path = (args.minute_csv or (data_dir / "ps_real_minute_1m.csv")).resolve()
    if not csv_path.is_file():
        raise SystemExit(
            f"未找到真实分钟数据: {csv_path}\n"
            "请先运行: python fetch_polysilicon_real_minute.py\n"
            "或指定: python polysilicon_futures_noise_backtest.py --minute-csv /path/to/your.csv"
        )

    peek = pd.read_csv(csv_path, parse_dates=["DateTime"])
    bt_start = peek["DateTime"].dt.date.min()
    bt_end = peek["DateTime"].dt.date.max()
    print(f"[数据] 真实分钟 K: {csv_path}")
    print(f"      样本区间: {bt_start} ~ {bt_end}，共 {len(peek)} 根分钟K")

    cfg = default_config(csv_path)
    cfg["start_date"] = bt_start
    cfg["end_date"] = bt_end
    cfg["sigma_incomplete_day_drop"] = False

    print(
        "仓位规则: leverage=1 不加杠杆，每交易日 position_size=floor(日初权益/开盘价)，"
        "单笔盈亏=position_size×价差；手续费/滑点已关闭。"
    )
    print(f"数据文件: {csv_path}")
    print_minute_data_diagnostics(csv_path)
    daily_df, monthly, trades, metrics = run_backtest(cfg)

    trades_csv = Path(__file__).resolve().parent / "data" / "ps_backtest_trades_detail.csv"
    dump_trade_details(trades, trades_csv)

    print("\n--- 补充指标（metrics 字典）---")
    extras = [
        "profit_loss_ratio",
        "avg_daily_trades",
        "max_daily_trades",
        "max_daily_gain",
        "max_daily_loss",
        "max_single_gain",
        "max_single_loss",
        "exposure_time",
        "calmar_ratio",
        "mdd_eod_close_only",
        "max_drawdown_duration",
    ]
    for k in extras:
        if k not in metrics:
            continue
        v = metrics[k]
        if v is None:
            continue
        if isinstance(v, (float, np.floating)):
            print(f"  {k}: {float(v):.6g}")
        else:
            print(f"  {k}: {v}")
    if len(daily_df) > 0:
        print(f"  回测日度记录条数: {len(daily_df)} （首行日期 {daily_df.index[0].date()} → 末行 {daily_df.index[-1].date()}）")
    return daily_df, monthly, trades, metrics


if __name__ == "__main__":
    main()
