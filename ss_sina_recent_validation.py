#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不锈钢 SS — AkShare/Sina 原始 1min K 训练/验证。

说明：
- 数据源为 AkShare/Sina `futures_zh_minute_sina` 返回的真实 1min K，不使用 tick。
- 不做价格平移、不做复权、不用自行聚合；每根 K 保留接口原始 OHLC。
- 用上一交易日成交量最大的可交易合约作为当日主力，避免用当日成交量前视。
- 默认将样本按交易日期一分为二：前半用于拟合参数，后半只做验证。
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from futures_minute_sina import candidate_symbols_for_prefix, fetch_one
from noise_strategy_backtest import run_backtest


SS_MULTIPLIER = 5.0


SS_OPTIMIZED_PARAMS = {
    "lookback_days": 5,
    "check_interval_minutes": 20,
    "enable_transaction_fees": False,
    "slippage_per_share": 0.0,
    "transaction_fee_per_share": 0.0,
    "trading_sessions": (
        ((9, 0), (10, 15)),
        ((10, 30), (11, 30)),
        ((13, 30), (14, 59)),
    ),
    "trading_start_time": (9, 0),
    "trading_end_time": (14, 59),
    "max_positions_per_day": 1,
    "print_daily_trades": False,
    "print_trade_details": False,
    "K1": 2.6,
    "K2": 2.6,
    "leverage": 1,
    "futures_ton_per_lot": SS_MULTIPLIER,
    "contract_multiplier": SS_MULTIPLIER,
    "use_vwap": False,
    "enable_intraday_stop_loss": False,
    "enable_trailing_take_profit": True,
    "trailing_tp_activation_pct": 0.01,
    "trailing_tp_callback_pct": 0.7,
    "entry_trend_filter": None,
    "sigma_incomplete_day_drop": False,
    "random_plots": 0,
    "plot_days": [],
}


def dump_trade_details(trades_df: pd.DataFrame, out_csv: Path, multiplier: float = SS_MULTIPLIER) -> None:
    """导出逐笔交易，并按不锈钢 5 吨/手校验 PnL 口径。"""
    if trades_df is None or len(trades_df) == 0:
        print("\n[逐笔交易] 无成交记录。")
        return

    df = trades_df.copy().sort_values(["Date", "entry_time"]).reset_index(drop=True)

    def pnl_from_prices(row):
        size = float(row["position_size"]) * multiplier
        entry_price = float(row["entry_price"])
        exit_price = float(row["exit_price"])
        if row["side"] == "Long":
            return size * (exit_price - entry_price)
        return size * (entry_price - exit_price)

    df["_pnl_check"] = df.apply(pnl_from_prices, axis=1)
    df["_diff"] = (df["pnl"] - df["_pnl_check"]).abs()
    bad = df[df["_diff"] > 1e-4]
    if len(bad) > 0:
        print(f"\n[警告] 有 {len(bad)} 笔 pnl 与「手数×{multiplier:g}×价差」不一致。")
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
    wins = int((df["pnl"] > 0).sum())
    print(f"\n[逐笔交易] 已写入 {out_csv}，共 {len(df)} 笔，盈利 {wins} 笔，胜率 {wins / len(df):.2%}")


def build_or_load_raw_main_csv(out_csv: Path, rebuild: bool, sleep_s: float) -> Path:
    if out_csv.is_file() and not rebuild:
        return out_csv

    chunks = []
    for symbol in candidate_symbols_for_prefix("ss"):
        df = fetch_one(symbol, "1")
        if df is None:
            continue
        chunks.append(df)
        print(f"[拉取] {symbol}: {len(df)} 根，{df['datetime'].min()} ~ {df['datetime'].max()}")
        time.sleep(sleep_s)
    if not chunks:
        raise RuntimeError("未从 AkShare/Sina 拉到任何 SS 合约 1min K")

    raw = pd.concat(chunks, ignore_index=True).sort_values(["datetime", "contract"]).reset_index(drop=True)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["date"] = raw["datetime"].dt.date

    # 当前策略只交易日盘；先过滤日盘，再做主力选择，避免夜盘交易日归属争议。
    day_raw = raw[raw["datetime"].map(is_day_session)].copy()
    daily_volume = day_raw.groupby(["date", "contract"], as_index=False)["volume"].sum()
    dates = sorted(daily_volume["date"].unique())
    selected = []
    for i, current_date in enumerate(dates):
        ranking_date = dates[i - 1] if i > 0 else current_date
        ranked = (
            daily_volume[daily_volume["date"] == ranking_date]
            .sort_values("volume", ascending=False)["contract"]
            .tolist()
        )
        available = set(daily_volume[daily_volume["date"] == current_date]["contract"])
        picked = next((contract for contract in ranked if contract in available), None)
        if picked is None:
            picked = (
                daily_volume[daily_volume["date"] == current_date]
                .sort_values("volume", ascending=False)["contract"]
                .iloc[0]
            )
        selected.append({"date": current_date, "main_contract": picked})

    selected_df = pd.DataFrame(selected)
    main = day_raw.merge(selected_df, on="date", how="left")
    main = main[main["contract"] == main["main_contract"]].copy()
    clean = pd.DataFrame(
        {
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
        }
    ).dropna(subset=["DateTime", "Open", "High", "Low", "Close"])
    clean["Turnover"] = clean["Close"] * clean["Volume"]
    clean = clean.sort_values(["DateTime"]).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_csv, index=False)
    contracts = clean.groupby("Date")["Contract"].first()
    print(
        f"[数据] 已写入原始主力 1min: {out_csv}，共 {len(clean)} 根，"
        f"{clean['DateTime'].min()} ~ {clean['DateTime'].max()}，"
        f"{clean['Date'].nunique()} 个日盘日期，{contracts.nunique()} 个合约"
    )
    return out_csv


def is_day_session(ts: pd.Timestamp) -> bool:
    t = ts.time()
    return (
        pd.Timestamp("09:00").time() <= t <= pd.Timestamp("10:15").time()
        or pd.Timestamp("10:30").time() <= t <= pd.Timestamp("11:30").time()
        or pd.Timestamp("13:30").time() <= t <= pd.Timestamp("14:59").time()
    )


def eod_max_drawdown(daily_df: pd.DataFrame) -> float:
    cap = daily_df["capital"].astype(float)
    dd = cap / cap.cummax() - 1
    return float(-dd.min())


def run_window(csv_path: Path, name: str, start: date | None, end: date | None) -> dict:
    cfg = {
        "data_path": str(csv_path),
        "ticker": f"SS_SINA_raw_prevday_main_1m_{name}",
        "initial_capital": 2_000_000.0,
        "start_date": start,
        "end_date": end,
        **SS_OPTIMIZED_PARAMS,
    }
    daily_df, monthly, trades, metrics = run_backtest(cfg)
    ret = float(daily_df["capital"].iloc[-1] / cfg["initial_capital"] - 1)
    mdd = eod_max_drawdown(daily_df)
    sharpe = float(metrics.get("sharpe_ratio", np.nan))
    annual_return = float(metrics.get("irr", np.nan))
    engine_max_drawdown = float(metrics.get("mdd", np.nan))
    n_trades = 0 if trades is None else len(trades)
    win_rate = float((trades["pnl"] > 0).mean()) if n_trades else 0.0

    trades_csv = csv_path.parent / f"ss_sina_{name}_trades.csv"
    dump_trade_details(trades, trades_csv, multiplier=SS_MULTIPLIER)

    print(f"\n[{name}]")
    print(f"  日期: {daily_df.index.min().date()} ~ {daily_df.index.max().date()}，日数 {len(daily_df)}")
    print(
        f"  总收益: {ret:.2%} | 年化收益: {annual_return:.2%} | "
        f"最大回撤: {engine_max_drawdown:.2%} | EOD最大回撤: {mdd:.2%} | Sharpe: {sharpe:.3f}"
    )
    print(f"  交易: {n_trades} | 胜率: {win_rate:.2%} | 最终资金: {daily_df['capital'].iloc[-1]:,.2f}")
    return {
        "window": name,
        "start": daily_df.index.min().date(),
        "end": daily_df.index.max().date(),
        "days": len(daily_df),
        "total_return": ret,
        "annual_return": annual_return,
        "max_drawdown": engine_max_drawdown,
        "eod_max_drawdown": mdd,
        "sharpe": sharpe,
        "trades": n_trades,
        "win_rate": win_rate,
        "final_capital": float(daily_df["capital"].iloc[-1]),
    }


def split_dates(csv_path: Path) -> tuple[date, date, date, date]:
    peek = pd.read_csv(csv_path, parse_dates=["DateTime"])
    dates = sorted(pd.to_datetime(peek["Date"]).dt.date.unique())
    if len(dates) < 20:
        raise ValueError("样本交易日过少，无法做前半训练/后半验证")
    mid = len(dates) // 2
    return dates[0], dates[mid - 1], dates[mid], dates[-1]


def main():
    ap = argparse.ArgumentParser(description="拉取/复用新浪 SS 原始 1min K，并做前半训练/后半验证")
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "ss_sina_raw_prevday_main_1m.csv",
        help="缓存/输出的原始主力分钟 K CSV",
    )
    ap.add_argument("--rebuild", action="store_true", help="重新从新浪拉取并拼接")
    ap.add_argument("--sleep", type=float, default=0.05, help="AkShare 请求间隔")
    args = ap.parse_args()

    csv_path = build_or_load_raw_main_csv(args.csv.resolve(), rebuild=args.rebuild, sleep_s=args.sleep)
    train_start, train_end, valid_start, valid_end = split_dates(csv_path)

    print("[参数] lookback=5, interval=20min, K=2.6, day session, max 1 trade/day")
    print(f"[切分] train: {train_start} ~ {train_end}; valid: {valid_start} ~ {valid_end}")
    rows = [
        run_window(csv_path, "train", train_start, train_end),
        run_window(csv_path, "valid", valid_start, valid_end),
        run_window(csv_path, "full", train_start, valid_end),
    ]
    summary_csv = csv_path.parent / "ss_sina_raw_train_validate_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"[汇总] 已写入 {summary_csv}")


if __name__ == "__main__":
    main()
