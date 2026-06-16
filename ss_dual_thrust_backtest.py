#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不锈钢 SS — 经典 Dual Thrust 日内突破策略回测。

参考 ../quantra/backtest_dual_thrust.py 的经典 Dual Thrust 实现，并做期货化适配：

经典公式（过去 N 个交易日）:
    Range = Max(HH - LC, HC - LL)
    Upper(买入线) = DayOpen + K1 × Range
    Lower(卖出线) = DayOpen - K2 × Range
    收盘价突破上轨做多、突破下轨做空，当日收盘强制平仓。

与 quantra 美股版的主要差异（移植时的调整点）:
1. 数据源复用本项目 AkShare/Sina 主力 1min K（ss_sina_recent_validation.build_or_load_raw_main_csv），
   而非 Longport QQQ。
2. 仓位以「手」为单位，按保证金计算: lots = floor(capital×leverage / (DayOpen×吨/手×保证金率))，
   与 noise_strategy_backtest 的期货口径一致。
3. 盈亏 = 手数 × 合约乘数(5吨/手) × 价差 − 手续费；手续费按「每手每边」计。
4. 滑点：开/平各以 50% 概率对价成交计，即每边期望 0.5 tick（slippage_ticks=0.5），
   往返合计期望 1 tick；开多 Close+滑点、平多 Close−滑点，开空 Close−滑点、平空 Close+滑点。
5. A 股日盘分两段且有午休（09:00–10:15 / 10:30–11:30 / 13:30–15:00），
   收盘平仓用当日最后一根 K 的 Close，开仓信号每 check_interval 分钟检查一次。
5. 换月日（当日主力 != 上一交易日主力）整日跳过，避免跨合约价差污染。

用法:
    python ss_dual_thrust_backtest.py                 # 默认全日盘
    python ss_dual_thrust_backtest.py --compare        # 跑多组参数对比
    python ss_dual_thrust_backtest.py --rebuild        # 重新从新浪拉数据
"""

from __future__ import annotations

import argparse
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd

from ss_sina_recent_validation import build_or_load_raw_main_csv

ROOT_DIR = Path(__file__).resolve().parent
DATA_CSV = ROOT_DIR / "data" / "ss_sina_raw_prevday_main_1m.csv"
SUMMARY_CSV = ROOT_DIR / "data" / "ss_dual_thrust_summary.csv"
TRADES_CSV = ROOT_DIR / "data" / "ss_dual_thrust_trades.csv"

# 合约与交易成本：SS 不锈钢，5 吨/手，最小变动价位 5 元/吨。
SS_MULTIPLIER = 5.0
TICK_SIZE = 5.0
SLIPPAGE_TICKS = 0.5
FEE_PER_LOT_PER_SIDE = 2.0
FUTURES_MARGIN_RATE = 0.10

INITIAL_CAPITAL = 100_000.0


def _entry_price_with_slippage(close: float, side: int, slippage: float) -> float:
    """side: 1=多, -1=空。对价期望：买贵/卖便宜各 slippage。"""
    return close + slippage if side > 0 else close - slippage


def _exit_price_with_slippage(close: float, side: int, slippage: float) -> float:
    """平仓：多头卖出/空头买回，均按对价方向减价。"""
    return close - slippage if side > 0 else close + slippage


def _build_allowed_times(trading_start, trading_end, check_interval):
    allowed = []
    h, m = trading_start
    eh, em = trading_end
    while h < eh or (h == eh and m <= em):
        allowed.append(f"{h:02d}:{m:02d}")
        m += check_interval
        if m >= 60:
            h += m // 60
            m %= 60
    return set(allowed)


def run_dual_thrust_backtest(config):
    """经典 Dual Thrust 期货回测，返回 (daily_df, monthly, trades_df, metrics)。"""
    data_path = config["data_path"]
    ticker = config.get("ticker", "SS")
    initial_capital = config.get("initial_capital", INITIAL_CAPITAL)
    leverage = config.get("leverage", 1)
    N = config.get("dt_lookback", 1)
    K1 = config.get("dt_K1", 0.5)
    K2 = config.get("dt_K2", 0.5)
    check_interval = config.get("check_interval_minutes", 20)
    trading_start = config.get("trading_start_time", (9, 0))
    trading_end = config.get("trading_end_time", (15, 0))
    max_positions_per_day = config.get("max_positions_per_day", 1)
    multiplier = config.get("contract_multiplier", SS_MULTIPLIER)
    ton_per_lot = config.get("futures_ton_per_lot", SS_MULTIPLIER)
    margin_rate = config.get("futures_margin_rate", FUTURES_MARGIN_RATE)
    tick_size = config.get("tick_size", TICK_SIZE)
    slippage_ticks = config.get("slippage_ticks", SLIPPAGE_TICKS)
    fee_per_lot = config.get("futures_fee_per_lot", FEE_PER_LOT_PER_SIDE)
    enable_fees = config.get("enable_transaction_fees", True)
    skip_roll = config.get("skip_contract_roll_days", True)
    start_date = config.get("start_date")
    end_date = config.get("end_date")
    print_daily = config.get("print_daily_trades", False)

    slippage = tick_size * slippage_ticks

    # ---------- 加载数据（已是日盘主力 1min K）----------
    price_df = pd.read_csv(data_path, parse_dates=["DateTime"])
    price_df.sort_values("DateTime", inplace=True)
    if "TradingDate" in price_df.columns:
        price_df["Date"] = pd.to_datetime(price_df["TradingDate"]).dt.date
    else:
        price_df["Date"] = price_df["DateTime"].dt.date
    price_df["Time"] = price_df["DateTime"].dt.strftime("%H:%M")

    # ---------- 聚合日线（仅用当日数据，shift 后只看过去 N 日，无前视）----------
    daily = (
        price_df.groupby("Date")
        .agg(
            DayOpen=("Open", "first"),
            DayHigh=("High", "max"),
            DayLow=("Low", "min"),
            DayClose=("Close", "last"),
            Contract=("Contract", "first") if "Contract" in price_df.columns else ("Open", "first"),
        )
        .reset_index()
        .sort_values("Date")
    )

    daily["HH"] = daily["DayHigh"].rolling(N).max().shift(1)
    daily["LC"] = daily["DayClose"].rolling(N).min().shift(1)
    daily["HC"] = daily["DayClose"].rolling(N).max().shift(1)
    daily["LL"] = daily["DayLow"].rolling(N).min().shift(1)
    daily["Range"] = np.maximum(daily["HH"] - daily["LC"], daily["HC"] - daily["LL"])
    daily["Upper"] = daily["DayOpen"] + K1 * daily["Range"]
    daily["Lower"] = daily["DayOpen"] - K2 * daily["Range"]

    # 换月日标记：当日主力合约与上一交易日不同
    if skip_roll and "Contract" in price_df.columns:
        daily["prev_contract"] = daily["Contract"].shift(1)
        daily["is_roll"] = (daily["prev_contract"].notna()) & (daily["Contract"] != daily["prev_contract"])
    else:
        daily["is_roll"] = False

    date_info = daily.set_index("Date").to_dict("index")

    allowed_times = _build_allowed_times(trading_start, trading_end, check_interval)

    unique_dates = sorted(price_df["Date"].unique())
    if start_date:
        unique_dates = [d for d in unique_dates if d >= start_date]
    if end_date:
        unique_dates = [d for d in unique_dates if d <= end_date]

    # ---------- 主循环 ----------
    capital = initial_capital
    all_trades = []
    daily_results = []
    total_fees = 0.0
    trading_days = set()
    non_trading_days = set()

    # 跨日精确 MDD（含日内极值）+ 单日真实最大回撤，与 noise 引擎口径一致
    capital_peak = initial_capital
    precise_mdd_pct = 0.0
    precise_peak_date = None
    precise_trough_date = None
    cur_peak_date = None
    max_intraday_mdd_pct = 0.0
    max_intraday_mdd_date = None

    for trade_date in unique_dates:
        info = date_info.get(trade_date)
        if info is None or np.isnan(info.get("Range", np.nan)) or info.get("is_roll", False):
            daily_results.append({"Date": trade_date, "capital": capital, "daily_return": 0})
            non_trading_days.add(trade_date)
            continue

        upper = info["Upper"]
        lower = info["Lower"]
        day_open = info["DayOpen"]

        day_df = price_df[price_df["Date"] == trade_date].sort_values("DateTime")
        if len(day_df) < 10:
            daily_results.append({"Date": trade_date, "capital": capital, "daily_return": 0})
            non_trading_days.add(trade_date)
            continue

        notional_per_lot = float(day_open) * float(ton_per_lot) * float(margin_rate)
        position_size = max(0, floor(capital * leverage / notional_per_lot)) if notional_per_lot > 0 else 0
        if position_size <= 0:
            daily_results.append({"Date": trade_date, "capital": capital, "daily_return": 0})
            non_trading_days.add(trade_date)
            continue

        position = 0
        entry_price = 0.0
        entry_time = None
        day_pnl = 0.0
        day_fees = 0.0
        day_trades = []
        positions_opened = 0

        day_start_capital = capital
        intraday_peak = day_start_capital
        intraday_max_dd = 0.0
        intraday_high = day_start_capital
        intraday_low = day_start_capital

        last_idx = len(day_df) - 1
        for bar_i, (_, row) in enumerate(day_df.iterrows()):
            t = row["Time"]
            price = row["Close"]
            high = row["High"]
            low = row["Low"]

            # 用本根 K 的 High/Low 更新日内资金极值
            if position == 1:
                best_un = position_size * multiplier * (high - entry_price)
                worst_un = position_size * multiplier * (low - entry_price)
            elif position == -1:
                best_un = position_size * multiplier * (entry_price - low)
                worst_un = position_size * multiplier * (entry_price - high)
            else:
                best_un = worst_un = 0.0
            cur_best = day_start_capital + day_pnl + best_un
            cur_worst = day_start_capital + day_pnl + worst_un
            intraday_peak = max(intraday_peak, cur_best)
            intraday_max_dd = max(intraday_max_dd, intraday_peak - cur_worst)
            intraday_high = max(intraday_high, cur_best)
            intraday_low = min(intraday_low, cur_worst)

            is_last_bar = bar_i == last_idx

            # 收盘强制平仓：Close ± 滑点（开平各 0.5 tick 期望，往返 1 tick）
            if is_last_bar and position != 0:
                exit_price = _exit_price_with_slippage(price, position, slippage)
                fees = (position_size * fee_per_lot * 2) if enable_fees else 0.0
                pnl = position_size * multiplier * (exit_price - entry_price) * (1 if position > 0 else -1) - fees
                day_pnl += pnl
                day_fees += fees
                day_trades.append({
                    "entry_time": entry_time,
                    "exit_time": row["DateTime"],
                    "side": "Long" if position > 0 else "Short",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "exit_reason": "Intraday Close",
                    "position_size": position_size,
                    "transaction_fees": fees,
                })
                position = 0
                continue

            # 开仓信号：仅在检查时点、非最后一根、空仓且未超过每日上限
            if (
                position == 0
                and not is_last_bar
                and t in allowed_times
                and positions_opened < max_positions_per_day
            ):
                if price > upper:
                    position = 1
                    entry_price = _entry_price_with_slippage(price, 1, slippage)
                    entry_time = row["DateTime"]
                    positions_opened += 1
                elif price < lower:
                    position = -1
                    entry_price = _entry_price_with_slippage(price, -1, slippage)
                    entry_time = row["DateTime"]
                    positions_opened += 1

        # 兜底：理论上最后一根已平，这里防御性处理
        if position != 0:
            last_row = day_df.iloc[-1]
            exit_price = _exit_price_with_slippage(last_row["Close"], position, slippage)
            fees = (position_size * fee_per_lot * 2) if enable_fees else 0.0
            pnl = position_size * multiplier * (exit_price - entry_price) * (1 if position > 0 else -1) - fees
            day_pnl += pnl
            day_fees += fees
            day_trades.append({
                "entry_time": entry_time,
                "exit_time": last_row["DateTime"],
                "side": "Long" if position > 0 else "Short",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "exit_reason": "Market Close",
                "position_size": position_size,
                "transaction_fees": fees,
            })

        # 跨日精确 MDD
        if cur_peak_date is None:
            cur_peak_date = trade_date
        if intraday_high > capital_peak:
            capital_peak = intraday_high
            cur_peak_date = trade_date
        cur_dd_pct = (capital_peak - intraday_low) / capital_peak if capital_peak > 0 else 0
        if cur_dd_pct > precise_mdd_pct:
            precise_mdd_pct = cur_dd_pct
            precise_peak_date = cur_peak_date
            precise_trough_date = trade_date

        intraday_mdd_pct = intraday_max_dd / day_start_capital if day_start_capital > 0 else 0
        if intraday_mdd_pct > max_intraday_mdd_pct:
            max_intraday_mdd_pct = intraday_mdd_pct
            max_intraday_mdd_date = trade_date

        capital_start = capital
        capital += day_pnl
        daily_return = day_pnl / capital_start if capital_start > 0 else 0
        total_fees += day_fees
        daily_results.append({"Date": trade_date, "capital": capital, "daily_return": daily_return})

        if day_trades:
            trading_days.add(trade_date)
            for tr in day_trades:
                tr["Date"] = trade_date
            all_trades.extend(day_trades)
            if print_daily:
                ds = pd.to_datetime(trade_date).strftime("%Y-%m-%d")
                parts = []
                for tr in day_trades:
                    d = "多" if tr["side"] == "Long" else "空"
                    et = pd.Timestamp(tr["entry_time"]).strftime("%H:%M")
                    xt = pd.Timestamp(tr["exit_time"]).strftime("%H:%M")
                    parts.append(f"{d}({et}->{xt}) {tr['pnl']:+.0f}")
                print(f"{ds} | 手数:{position_size} | 日盈亏:{day_pnl:+.0f} | {', '.join(parts)}")
        else:
            non_trading_days.add(trade_date)

    daily_df = pd.DataFrame(daily_results)
    if daily_df.empty:
        raise ValueError("没有有效的回测数据")
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    daily_df.set_index("Date", inplace=True)
    trades_df = pd.DataFrame(all_trades)

    # 买入持有基准（按日开->日收复利）
    bh = daily[["Date", "DayClose"]].copy()
    bh["Date"] = pd.to_datetime(bh["Date"])
    bh.set_index("Date", inplace=True)
    if start_date:
        bh = bh[bh.index >= pd.to_datetime(start_date)]
    if end_date:
        bh = bh[bh.index <= pd.to_datetime(end_date)]
    bh["daily_return"] = bh["DayClose"] / bh["DayClose"].shift(1) - 1
    bh["capital"] = initial_capital * (1 + bh["daily_return"]).cumprod().fillna(1)

    metrics = calculate_performance_metrics(daily_df, trades_df, initial_capital, bh)
    metrics["mdd_eod_close_only"] = metrics["mdd"]
    if precise_trough_date is not None:
        metrics["mdd"] = precise_mdd_pct
    metrics["max_single_day_intraday_mdd_pct"] = max_intraday_mdd_pct
    metrics["max_single_day_intraday_mdd_date"] = max_intraday_mdd_date
    metrics["total_fees"] = total_fees
    if len(trades_df) > 0:
        metrics["total_slippage"] = float((trades_df["position_size"] * multiplier * slippage * 2).sum())
    else:
        metrics["total_slippage"] = 0.0
    metrics["trading_days"] = len(trading_days)
    metrics["non_trading_days"] = len(non_trading_days)

    monthly = daily_df.resample("ME").first()[["capital"]].rename(columns={"capital": "month_start"})
    monthly["month_end"] = daily_df.resample("ME").last()["capital"]
    monthly["monthly_return"] = monthly["month_end"] / monthly["month_start"] - 1

    return daily_df, monthly, trades_df, metrics


def calculate_performance_metrics(daily_df, trades_df, initial_capital, buy_hold_df=None,
                                  risk_free_rate=0.02, trading_days_per_year=252):
    metrics = {}
    if len(daily_df) == 0:
        return {k: 0 for k in ["total_return", "irr", "volatility", "sharpe_ratio", "hit_ratio", "mdd",
                               "buy_hold_return", "buy_hold_irr", "buy_hold_volatility", "buy_hold_sharpe", "buy_hold_mdd"]}

    final_capital = daily_df["capital"].iloc[-1]
    metrics["total_return"] = final_capital / initial_capital - 1
    start_dt, end_dt = daily_df.index[0], daily_df.index[-1]
    years = (end_dt - start_dt).days / 365.25
    if years < 0.1:
        years = len(daily_df) / trading_days_per_year
    metrics["irr"] = (1 + metrics["total_return"]) ** (1 / years) - 1 if years > 0 else 0

    dr = daily_df["daily_return"]
    dr = dr[dr.between(dr.quantile(0.001), dr.quantile(0.999))]
    metrics["volatility"] = dr.std() * np.sqrt(trading_days_per_year)
    metrics["sharpe_ratio"] = (metrics["irr"] - risk_free_rate) / metrics["volatility"] if metrics["volatility"] > 0 else 0

    if len(trades_df) > 0:
        metrics["hit_ratio"] = (trades_df["pnl"] > 0).mean()
        metrics["total_trades"] = len(trades_df)
        wins = trades_df[trades_df["pnl"] > 0]["pnl"]
        losses = trades_df[trades_df["pnl"] <= 0]["pnl"]
        metrics["avg_win"] = wins.mean() if len(wins) else 0.0
        metrics["avg_loss"] = losses.mean() if len(losses) else 0.0
        metrics["profit_factor"] = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
    else:
        metrics["hit_ratio"] = 0
        metrics["total_trades"] = 0
        metrics["avg_win"] = metrics["avg_loss"] = 0.0
        metrics["profit_factor"] = 0.0

    daily_df = daily_df.copy()
    daily_df["peak"] = daily_df["capital"].cummax()
    daily_df["dd"] = (daily_df["capital"] - daily_df["peak"]) / daily_df["peak"]
    metrics["mdd"] = daily_df["dd"].min() * -1

    if buy_hold_df is not None and not buy_hold_df.empty and "capital" in buy_hold_df.columns:
        bh_final = buy_hold_df["capital"].iloc[-1]
        metrics["buy_hold_return"] = bh_final / initial_capital - 1
        metrics["buy_hold_irr"] = (1 + metrics["buy_hold_return"]) ** (1 / years) - 1 if years > 0 else 0
        bh_dr = buy_hold_df["daily_return"].dropna()
        bh_dr = bh_dr[bh_dr.between(bh_dr.quantile(0.001), bh_dr.quantile(0.999))]
        metrics["buy_hold_volatility"] = bh_dr.std() * np.sqrt(trading_days_per_year)
        metrics["buy_hold_sharpe"] = ((metrics["buy_hold_irr"] - risk_free_rate) / metrics["buy_hold_volatility"]
                                      if metrics["buy_hold_volatility"] > 0 else 0)
        bh2 = buy_hold_df.copy()
        bh2["peak"] = bh2["capital"].cummax()
        bh2["dd"] = (bh2["capital"] - bh2["peak"]) / bh2["peak"]
        metrics["buy_hold_mdd"] = bh2["dd"].min() * -1
    else:
        for k in ["buy_hold_return", "buy_hold_irr", "buy_hold_volatility", "buy_hold_sharpe", "buy_hold_mdd"]:
            metrics[k] = 0
    return metrics


def _print_report(cfg, daily_df, monthly, trades_df, metrics):
    name = (f"SS Dual Thrust N={cfg['dt_lookback']} K1={cfg['dt_K1']} K2={cfg['dt_K2']} "
            f"interval={cfg['check_interval_minutes']}m {cfg['trading_start_time']}~{cfg['trading_end_time']}")
    print(f"\n{'='*64}\n{name}\n{'='*64}")
    final_capital = daily_df["capital"].iloc[-1]
    print(f"区间: {daily_df.index.min().date()} ~ {daily_df.index.max().date()} | 交易日 {len(daily_df)}")
    print(f"初始资金: {cfg['initial_capital']:,.0f} | 最终资金: {final_capital:,.2f}")
    print(f"\n{'指标':<14} | {'策略':>14} | {'Buy&Hold':>14}")
    print("-" * 50)
    print(f"{'总回报率':<14} | {metrics['total_return']*100:>13.1f}% | {metrics['buy_hold_return']*100:>13.1f}%")
    print(f"{'年化收益':<14} | {metrics['irr']*100:>13.1f}% | {metrics['buy_hold_irr']*100:>13.1f}%")
    print(f"{'波动率':<14} | {metrics['volatility']*100:>13.1f}% | {metrics['buy_hold_volatility']*100:>13.1f}%")
    print(f"{'夏普':<14} | {metrics['sharpe_ratio']:>14.2f} | {metrics['buy_hold_sharpe']:>14.2f}")
    print(f"{'最大回撤':<14} | {metrics['mdd']*100:>13.1f}% | {metrics['buy_hold_mdd']*100:>13.1f}%")
    print(f"{'单日最大回撤':<14} | {metrics.get('max_single_day_intraday_mdd_pct',0)*100:>13.2f}% | {'-':>14}")
    long_n = int((trades_df["side"] == "Long").sum()) if len(trades_df) else 0
    short_n = int((trades_df["side"] == "Short").sum()) if len(trades_df) else 0
    print(f"\n交易统计:")
    print(f"  总交易: {metrics['total_trades']} (多:{long_n} 空:{short_n}) | 胜率: {metrics['hit_ratio']*100:.1f}%")
    print(f"  盈亏比: 均盈 {metrics.get('avg_win',0):,.0f} / 均亏 {metrics.get('avg_loss',0):,.0f} | 利润因子: {metrics.get('profit_factor',0):.2f}")
    print(f"  有交易日: {metrics['trading_days']} | 无交易日: {metrics['non_trading_days']}")
    print(f"  成本: 手续费 {metrics['total_fees']:,.0f} + 滑点 {metrics['total_slippage']:,.0f} = {metrics['total_fees']+metrics['total_slippage']:,.0f}")
    print(f"{'='*64}")


def base_config(csv_path):
    return {
        "data_path": str(csv_path),
        "ticker": "SS",
        "initial_capital": INITIAL_CAPITAL,
        "leverage": 1,
        # 统计稳健版：N3K0.5 仅 16 笔/3年，过拟合风险高；改为 N1K0.45 上午20m（~82笔，~35笔/年）
        # 若硬性要求 >=50笔/年，用 N1K0.40 上午10m（~119笔，夏普较低）
        "dt_lookback": 1,
        "dt_K1": 0.45,
        "dt_K2": 0.45,
        "check_interval_minutes": 20,
        "trading_start_time": (9, 0),
        "trading_end_time": (11, 30),
        "max_positions_per_day": 1,
        "contract_multiplier": SS_MULTIPLIER,
        "futures_ton_per_lot": SS_MULTIPLIER,
        "futures_margin_rate": FUTURES_MARGIN_RATE,
        "tick_size": TICK_SIZE,
        "slippage_ticks": SLIPPAGE_TICKS,
        "futures_fee_per_lot": FEE_PER_LOT_PER_SIDE,
        "enable_transaction_fees": True,
        "skip_contract_roll_days": True,
        "print_daily_trades": False,
    }


def main():
    ap = argparse.ArgumentParser(description="SS 经典 Dual Thrust 回测")
    ap.add_argument("--csv", type=Path, default=DATA_CSV)
    ap.add_argument("--rebuild", action="store_true", default=False)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--compare", action="store_true", help="跑多组参数对比")
    ap.add_argument("--print-trades", action="store_true")
    args = ap.parse_args()

    csv_path = build_or_load_raw_main_csv(args.csv.resolve(), rebuild=args.rebuild, sleep_s=args.sleep)

    if args.compare:
        variants = [
            {"name": "N1 K0.5 全日", "dt_lookback": 1, "dt_K1": 0.5, "dt_K2": 0.5,
             "trading_start_time": (9, 0), "trading_end_time": (15, 0)},
            {"name": "N1 K0.7 全日", "dt_lookback": 1, "dt_K1": 0.7, "dt_K2": 0.7,
             "trading_start_time": (9, 0), "trading_end_time": (15, 0)},
            {"name": "N2 K0.5 全日", "dt_lookback": 2, "dt_K1": 0.5, "dt_K2": 0.5,
             "trading_start_time": (9, 0), "trading_end_time": (15, 0)},
            {"name": "N3 K0.5 全日", "dt_lookback": 3, "dt_K1": 0.5, "dt_K2": 0.5,
             "trading_start_time": (9, 0), "trading_end_time": (15, 0)},
            {"name": "N1 K0.5 仅上午", "dt_lookback": 1, "dt_K1": 0.5, "dt_K2": 0.5,
             "trading_start_time": (9, 0), "trading_end_time": (11, 30)},
            {"name": "N1 K0.3 全日多次", "dt_lookback": 1, "dt_K1": 0.3, "dt_K2": 0.3,
             "trading_start_time": (9, 0), "trading_end_time": (15, 0), "max_positions_per_day": 3},
        ]
        rows = []
        for v in variants:
            cfg = base_config(csv_path)
            name = v.pop("name")
            cfg.update(v)
            _, _, trades_df, m = run_dual_thrust_backtest(cfg)
            rows.append({
                "变体": name,
                "总回报%": round(m["total_return"] * 100, 1),
                "年化%": round(m["irr"] * 100, 1),
                "夏普": round(m["sharpe_ratio"], 2),
                "最大回撤%": round(m["mdd"] * 100, 1),
                "交易数": m["total_trades"],
                "胜率%": round(m["hit_ratio"] * 100, 1),
                "利润因子": round(m.get("profit_factor", 0), 2),
            })
        cmp_df = pd.DataFrame(rows)
        print("\n参数对比 (Buy&Hold 年化≈{:.1f}%):".format(m["buy_hold_irr"] * 100))
        print(cmp_df.to_string(index=False))
        cmp_df.to_csv(ROOT_DIR / "data" / "ss_dual_thrust_compare.csv", index=False)
        print(f"\n[汇总] 已写入 {ROOT_DIR / 'data' / 'ss_dual_thrust_compare.csv'}")
        return

    cfg = base_config(csv_path)
    cfg["print_daily_trades"] = args.print_trades
    daily_df, monthly, trades_df, metrics = run_dual_thrust_backtest(cfg)
    _print_report(cfg, daily_df, monthly, trades_df, metrics)

    if len(trades_df) > 0:
        trades_df.to_csv(TRADES_CSV, index=False)
        print(f"[交易明细] 已写入 {TRADES_CSV}（{len(trades_df)} 笔）")
    pd.DataFrame([{
        "start": daily_df.index.min().date(),
        "end": daily_df.index.max().date(),
        "days": len(daily_df),
        "total_return": metrics["total_return"],
        "annual_return": metrics["irr"],
        "sharpe": metrics["sharpe_ratio"],
        "mdd": metrics["mdd"],
        "trades": metrics["total_trades"],
        "win_rate": metrics["hit_ratio"],
        "final_capital": float(daily_df["capital"].iloc[-1]),
    }]).to_csv(SUMMARY_CSV, index=False)
    print(f"[汇总] 已写入 {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
