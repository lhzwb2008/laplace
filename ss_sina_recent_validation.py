#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不锈钢 SS — AkShare/Sina 原始 1min K 全量回测。

说明：
- 数据源为 AkShare/Sina `futures_zh_minute_sina` 返回的真实 1min K，不使用 tick。
- 不做价格平移、不做复权、不用自行聚合；每根 K 保留接口原始 OHLC。
- 用上一交易日成交量最大的可交易合约作为当日主力，避免用当日成交量前视。
- 本脚本只负责用当前参数跑全量回测；训练/验证划分只用于线下调参，不放在主回测入口。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from futures_minute_sina import candidate_symbols_for_prefix, fetch_one
from noise_strategy_backtest import run_backtest


SS_MULTIPLIER = 5.0

# =============================================================================
# 用户可调配置
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent
DATA_CSV = ROOT_DIR / "data" / "ss_sina_raw_prevday_main_1m.csv"
SUMMARY_CSV = ROOT_DIR / "data" / "ss_sina_backtest_summary.csv"
TRADES_CSV = ROOT_DIR / "data" / "ss_sina_trades.csv"
INITIAL_CAPITAL = 100_000.0

# 数据配置
REBUILD_DATA = False
AKSHARE_SLEEP_SECONDS = 0.05

# 合约与交易成本：SS 不锈钢，5吨/手，最小变动价位 5元/吨。
CONTRACT_MULTIPLIER = SS_MULTIPLIER
FUTURES_MARGIN_RATE = 0.10
TICK_SIZE = 5.0
SLIPPAGE_TICKS = 0.5
FEE_PER_LOT_ROUND_TRIP_SIDE = 2.0

# 当前采用“训练集优先”的较均衡参数，而不是验证集最高参数。
LOOKBACK_DAYS = 5
CHECK_INTERVAL_MINUTES = 20
K1 = 2.2
K2 = 2.2
MAX_POSITIONS_PER_DAY = 1
TRADING_START_TIME = (10, 30)
TRADING_END_TIME = (11, 30)
TRADING_SESSIONS = (
    ((10, 30), (11, 30)),
)

# 风控/出场
ENABLE_TRAILING_TAKE_PROFIT = True
TRAILING_TP_ACTIVATION_PCT = 0.01
TRAILING_TP_CALLBACK_PCT = 0.7
ENABLE_INTRADAY_STOP_LOSS = False
USE_VWAP = False
ENTRY_TREND_FILTER = None

# 输出控制
PRINT_EACH_TRADE = True
WRITE_TRADE_CSV = True


BACKTEST_PARAMS = {
    "lookback_days": LOOKBACK_DAYS,
    "check_interval_minutes": CHECK_INTERVAL_MINUTES,
    "enable_transaction_fees": True,
    "tick_size": TICK_SIZE,
    "slippage_ticks": SLIPPAGE_TICKS,
    "futures_fee_per_lot": FEE_PER_LOT_ROUND_TRIP_SIDE,
    "transaction_fee_per_share": 0.0,
    "trading_sessions": TRADING_SESSIONS,
    "trading_start_time": TRADING_START_TIME,
    "trading_end_time": TRADING_END_TIME,
    "max_positions_per_day": MAX_POSITIONS_PER_DAY,
    "print_daily_trades": False,
    "print_trade_details": False,
    "K1": K1,
    "K2": K2,
    "leverage": 1,
    "futures_ton_per_lot": CONTRACT_MULTIPLIER,
    "contract_multiplier": CONTRACT_MULTIPLIER,
    "futures_margin_rate": FUTURES_MARGIN_RATE,
    "use_vwap": USE_VWAP,
    "enable_intraday_stop_loss": ENABLE_INTRADAY_STOP_LOSS,
    "enable_trailing_take_profit": ENABLE_TRAILING_TAKE_PROFIT,
    "trailing_tp_activation_pct": TRAILING_TP_ACTIVATION_PCT,
    "trailing_tp_callback_pct": TRAILING_TP_CALLBACK_PCT,
    "entry_trend_filter": ENTRY_TREND_FILTER,
    "prev_close_mode": "same_contract",
    "skip_contract_roll_days": True,
    "sigma_incomplete_day_drop": False,
    "random_plots": 0,
    "plot_days": [],
}


def dump_trade_details(trades_df: pd.DataFrame, out_csv: Path, multiplier: float = SS_MULTIPLIER) -> None:
    """导出并打印逐笔交易，展示买卖点、滑点、手续费和净盈亏。"""
    if trades_df is None or len(trades_df) == 0:
        print("\n[逐笔交易] 无成交记录。")
        return

    df = trades_df.copy().sort_values(["Date", "entry_time"]).reset_index(drop=True)
    slippage_price = TICK_SIZE * SLIPPAGE_TICKS

    def pnl_from_prices(row):
        size = float(row["position_size"]) * multiplier
        entry_price = float(row["entry_price"])
        exit_price = float(row["exit_price"])
        if row["side"] == "Long":
            gross = size * (exit_price - entry_price)
        else:
            gross = size * (entry_price - exit_price)
        return gross - float(row.get("transaction_fees", 0.0))

    df["_pnl_check"] = df.apply(pnl_from_prices, axis=1)
    df["_diff"] = (df["pnl"] - df["_pnl_check"]).abs()
    df["slippage_price_per_side"] = slippage_price
    df["slippage_cost"] = df["position_size"].astype(float) * multiplier * slippage_price * 2

    def raw_prices(row):
        entry = float(row["entry_price"])
        exit_ = float(row["exit_price"])
        if row["side"] == "Long":
            return pd.Series({"raw_entry_price": entry - slippage_price, "raw_exit_price": exit_ + slippage_price})
        return pd.Series({"raw_entry_price": entry + slippage_price, "raw_exit_price": exit_ - slippage_price})

    df[["raw_entry_price", "raw_exit_price"]] = df.apply(raw_prices, axis=1)
    bad = df[df["_diff"] > 1e-4]
    if len(bad) > 0:
        print(f"\n[警告] 有 {len(bad)} 笔 pnl 与「手数×{multiplier:g}×价差」不一致。")
        print(bad[["Date", "side", "pnl", "_pnl_check", "_diff"]].head(10))

    if PRINT_EACH_TRADE:
        print("\n[逐笔交易明细]")
        for i, row in df.iterrows():
            direction = "多" if row["side"] == "Long" else "空"
            print(
                f"{i + 1:03d} | {row['Date']} | {direction} | "
                f"{pd.Timestamp(row['entry_time']).strftime('%H:%M')} -> {pd.Timestamp(row['exit_time']).strftime('%H:%M')} | "
                f"信号价 {row['raw_entry_price']:.2f}->{row['raw_exit_price']:.2f} | "
                f"成交价 {row['entry_price']:.2f}->{row['exit_price']:.2f} | "
                f"手数 {int(row['position_size'])} | "
                f"滑点 {row['slippage_cost']:.2f} | 手续费 {row.get('transaction_fees', 0):.2f} | "
                f"净盈亏 {row['pnl']:.2f} | {row.get('exit_reason', '')}"
            )

    if not WRITE_TRADE_CSV:
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    export_cols = [
        c
        for c in [
            "Date",
            "side",
            "entry_time",
            "exit_time",
            "raw_entry_price",
            "raw_exit_price",
            "entry_price",
            "exit_price",
            "position_size",
            "slippage_price_per_side",
            "slippage_cost",
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


def run_full_backtest(csv_path: Path) -> dict:
    cfg = {
        "data_path": str(csv_path),
        "ticker": "SS_SINA_raw_prevday_main_1m",
        "initial_capital": INITIAL_CAPITAL,
        **BACKTEST_PARAMS,
    }
    daily_df, monthly, trades, metrics = run_backtest(cfg)
    ret = float(daily_df["capital"].iloc[-1] / cfg["initial_capital"] - 1)
    mdd = eod_max_drawdown(daily_df)
    sharpe = float(metrics.get("sharpe_ratio", np.nan))
    annual_return = float(metrics.get("irr", np.nan))
    engine_max_drawdown = float(metrics.get("mdd", np.nan))
    n_trades = 0 if trades is None else len(trades)
    win_rate = float((trades["pnl"] > 0).mean()) if n_trades else 0.0

    dump_trade_details(trades, TRADES_CSV, multiplier=SS_MULTIPLIER)

    print("\n[full]")
    print(f"  日期: {daily_df.index.min().date()} ~ {daily_df.index.max().date()}，日数 {len(daily_df)}")
    print(
        f"  总收益: {ret:.2%} | 年化收益: {annual_return:.2%} | "
        f"最大回撤: {engine_max_drawdown:.2%} | EOD最大回撤: {mdd:.2%} | Sharpe: {sharpe:.3f}"
    )
    print(f"  交易: {n_trades} | 胜率: {win_rate:.2%} | 最终资金: {daily_df['capital'].iloc[-1]:,.2f}")
    return {
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
        "trades_csv": str(TRADES_CSV),
    }


def main():
    ap = argparse.ArgumentParser(description="拉取/复用新浪 SS 原始 1min K，并用当前参数跑全量回测")
    ap.add_argument(
        "--csv",
        type=Path,
        default=DATA_CSV,
        help="缓存/输出的原始主力分钟 K CSV",
    )
    ap.add_argument("--rebuild", action="store_true", default=REBUILD_DATA, help="重新从新浪拉取并拼接")
    ap.add_argument("--sleep", type=float, default=AKSHARE_SLEEP_SECONDS, help="AkShare 请求间隔")
    args = ap.parse_args()

    csv_path = build_or_load_raw_main_csv(args.csv.resolve(), rebuild=args.rebuild, sleep_s=args.sleep)

    print(
        f"[成本] tick={TICK_SIZE}, 滑点={SLIPPAGE_TICKS} tick/边，"
        f"每手每边手续费={FEE_PER_LOT_ROUND_TRIP_SIDE}，保证金率={FUTURES_MARGIN_RATE:.0%}"
    )
    print(
        f"[参数] lookback={LOOKBACK_DAYS}, interval={CHECK_INTERVAL_MINUTES}min, "
        f"K1={K1}, K2={K2}, sessions={TRADING_SESSIONS}, max {MAX_POSITIONS_PER_DAY} trade/day"
    )
    row = run_full_backtest(csv_path)
    pd.DataFrame([row]).to_csv(SUMMARY_CSV, index=False)
    print(f"[汇总] 已写入 {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
