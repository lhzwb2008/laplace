#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多晶硅期货（广期所 PS）× Quantra「噪声通道 + VWAP」策略的近似回测。

说明（必读）
------------
1. 策略逻辑来自本目录 noise_strategy_backtest.py（由 Quantra/backtest.py 复制）：日内 |ret| 按「时刻」做滚动得到 sigma，
   再用前收/开盘参考价构造上下轨，配合 VWAP 做突破与止损；与美股 QQQ 版一致。

2. 数据（优先真实分钟）  
   - **推荐**：先运行 `fetch_polysilicon_real_minute.py`，从新浪拉取多晶硅 1 分钟 K
     （默认 `--mode disjoint`：多合约按时间**瀑布拼接**，避免把不同合约按钟点交错排序造成伪跳价；
     或 `--mode single --contract PS2606` 仅单合约连续序列）。生成 `data/ps_real_minute_1m.csv`。  
     若该文件存在且未加 `--synthetic`，回测**默认使用真实分钟**。  
   - **备选**：加 `--synthetic` 时用 `ak.futures_zh_daily_sina` 日线再 **O→H→L→C 插成分钟**，
     仅作无分钟数据时的管线测试，**不是**真实盘口。

3. 交易时段：9:00–10:15、10:30–11:30、13:30–15:00（与公开规则一致）；午饭休市无 K 线。

4. 手续费、滑点默认关闭（理想化）。

5. 默认参数（真实分钟）在 `default_config`：当前为样本内优选的一组——**lb=1、8 分钟检查、K≈1.26、关闭 VWAP、linreg5_r2≥0.45、9:30 起交易、日内止损关、追踪止盈开**（与 QQQ 示例追踪参数一致：激活 1%、保护 70% 浮盈）。

6. 仓位（全仓、不加杠杆）  
   `leverage=1`，不传 `futures_fixed_lots` / `futures_ton_per_lot` 时，与美股脚本一致：  
   **`position_size = floor(当日日初权益 / 当日开盘参考价)`**，再用 **`盈亏 = position_size × 价差`**（元/吨标价下即「按标价满仓」的极简名义）。  
   同一交易日内若多笔成交，手数仍按**日初冻结**权益计算，不会在开仓之间随浮盈即时调仓（引擎原设计如此）。

7. **为何之前夏普会离谱**  
   - 若既用手数又用「吨×价」名义却不在 `simulate_day` 里乘合约吨数，**盈亏与名义量纲会错位**（现已统一改回「现价单位」逻辑）。  
   - 更主要：**由日线插值的合成分钟线**在分钟尺度上过于光滑、日与日形态相似，使 σ 通道与突破信号**严重失真**，夏普/胜率会**显著乐观**，不能当真；真实 1 分钟数据出来前应只看方向性结论。
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple

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
    """检查合成/分钟数据量级，避免把「失真数据」与真实绩效混淆。"""
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


# ---------- 广期所多晶硅：日盘切片（生成 HH:MM 列表） ----------


def _iter_ps_session_times() -> List[Tuple[int, int]]:
    """返回当日有序 (hour, minute) 列表（9:00–10:15、10:30–11:30、13:30–15:00）。"""
    out: List[Tuple[int, int]] = []
    segs = [
        (9 * 60 + 0, 10 * 60 + 15),
        (10 * 60 + 30, 11 * 60 + 30),
        (13 * 60 + 30, 15 * 60 + 0),
    ]
    for t0, t1 in segs:
        t = t0
        while t <= t1:
            out.append((t // 60, t % 60))
            t += 1
    return out


SESSION_MINUTES = len(_iter_ps_session_times())


def _minute_path_o_h_l_c(o: float, h: float, lo: float, c: float, n: int) -> np.ndarray:
    """分段线性连接 O→H→L→C，长度 n；必要时抬升/压低 H、L 以满足 OHLC 约束。"""
    o, h, lo, c = float(o), float(h), float(lo), float(c)
    hi = max(o, h, c, lo)
    lw = min(o, h, c, lo)
    if hi < h:
        hi = h
    if lw > lo:
        lw = lo
    t1 = max(n // 3, 1)
    t2 = max(2 * n // 3, t1 + 1)
    t2 = min(t2, n - 1)
    p = np.empty(n)
    p[0] = o
    for i in range(1, t1 + 1):
        p[i] = o + (hi - o) * (i / t1)
    for i in range(t1, t2 + 1):
        den = t2 - t1
        p[i] = hi + (lw - hi) * ((i - t1) / den) if den > 0 else lw
    for i in range(t2, n):
        den = (n - 1) - t2
        p[i] = lw + (c - lw) * ((i - t2) / den) if den > 0 else c
    p[-1] = c
    return p


def expand_daily_to_synthetic_minutes(day: pd.Series, day_vol: float) -> pd.DataFrame:
    """单行日线 -> 当日分钟 OHLCV。"""
    d = day["date"]
    if hasattr(d, "date"):
        d = d.date() if not isinstance(d, date) else d
    o, h, lo, c = day["open"], day["high"], day["low"], day["close"]
    times = _iter_ps_session_times()
    n = len(times)
    core = _minute_path_o_h_l_c(o, h, lo, c, n)
    # 每分钟 open = 上一分钟 close；高低略放宽，避免大量零振幅
    opens = np.empty(n)
    opens[0] = o
    opens[1:] = core[:-1]
    closes = core
    tick = 5.0  # 最小变动价位（元/吨），用于微调
    highs = np.maximum(opens, closes) + tick * 0.0
    lows = np.minimum(opens, closes) - tick * 0.0
    highs = np.maximum(highs, h - 1e-9)
    lows = np.minimum(lows, lo + 1e-9)
    vol_per = max(float(day_vol), 1.0) / n
    rows = []
    dt_base = datetime.combine(d if isinstance(d, date) else d.date(), datetime.min.time())
    for i, (hh, mm) in enumerate(times):
        ts = dt_base.replace(hour=hh, minute=mm, second=0)
        rows.append(
            {
                "datetime": ts,
                "Open": float(opens[i]),
                "High": float(highs[i]),
                "Low": float(lows[i]),
                "Close": float(closes[i]),
                "Volume": vol_per,
            }
        )
    return pd.DataFrame(rows)


def load_polysilicon_daily_sina(symbol: str = "PS2512") -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as e:
        raise SystemExit("请安装 akshare: pip install akshare") from e
    df = ak.futures_zh_daily_sina(symbol=symbol)
    df = df.rename(columns=str.lower)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def daily_to_minute_csv(
    daily: pd.DataFrame,
    out_path: Path,
    start: date,
    end: date,
    cache: bool = True,
) -> Path:
    if cache and out_path.exists():
        return out_path
    sub = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
    if sub.empty:
        raise ValueError(f"日线在 [{start}, {end}] 为空，请换合约或检查数据")
    parts = []
    for _, row in sub.iterrows():
        parts.append(expand_daily_to_synthetic_minutes(row, row["volume"]))
    minute = pd.concat(parts, ignore_index=True)
    minute["DateTime"] = pd.to_datetime(minute["datetime"])
    # 成交额近似：元/吨 × 吨，这里 volume 按「吨」计的拆分量，Turnover 供 VWAP
    minute["Turnover"] = minute["Close"] * minute["Volume"]
    minute = minute[["DateTime", "Open", "High", "Low", "Close", "Volume", "Turnover"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    minute.to_csv(out_path, index=False)
    return out_path


# 当前默认采用的一组策略参数（样本内：Sharpe≈1.2–1.3、回撤约 8% 量级；无日内止损，有追踪止盈）
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
    """回测窗口由 start_date / end_date 控制；分钟数据文件应含更早的暖启动区间。"""
    return {
        "data_path": str(csv_path),
        "ticker": "PS_GFEX_synth",
        "initial_capital": 2_000_000.0,
        "start_date": date(2025, 1, 1),
        "end_date": date(2025, 12, 31),
        **PS_NOISE_STRATEGY_PARAMS,
    }


def main():
    ap = argparse.ArgumentParser(description="多晶硅噪声通道策略回测")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="强制使用日线合成分钟（忽略 data/ps_real_minute_1m.csv）",
    )
    ap.add_argument(
        "--minute-csv",
        type=Path,
        default=None,
        help="指定分钟数据 CSV（默认 data/ps_real_minute_1m.csv 若存在）",
    )
    args = ap.parse_args()

    data_dir = Path(__file__).resolve().parent / "data"
    real_csv = data_dir / "ps_real_minute_1m.csv"
    synth_csv = data_dir / "ps_synth_minute_for_noise_backtest.csv"

    from_real_file: bool
    csv_path: Path
    if args.minute_csv is not None:
        csv_path = args.minute_csv
        if not csv_path.is_file():
            raise SystemExit(f"找不到分钟数据: {csv_path}")
        from_real_file = True
    elif not args.synthetic and real_csv.is_file():
        csv_path = real_csv
        from_real_file = True
    else:
        from_real_file = False
        warmup = date(2024, 11, 1)
        bt_end = date(2025, 12, 31)
        daily = load_polysilicon_daily_sina("PS2512")
        csv_path = synth_csv
        daily_to_minute_csv(daily, csv_path, warmup, bt_end, cache=True)

    peek = pd.read_csv(csv_path, parse_dates=["DateTime"])
    if from_real_file:
        bt_start = peek["DateTime"].dt.date.min()
        bt_end = peek["DateTime"].dt.date.max()
        print(f"[数据] 使用真实分钟 K: {csv_path}")
        print(f"      样本区间: {bt_start} ~ {bt_end}，共 {len(peek)} 根分钟K")
    else:
        bt_start = date(2025, 1, 1)
        bt_end = date(2025, 12, 31)
        print(
            "[数据] 使用日线合成分钟（--synthetic 或尚未生成 ps_real_minute_1m.csv）；"
            "真实分钟请先运行: python fetch_polysilicon_real_minute.py"
        )

    cfg = default_config(csv_path)
    cfg["start_date"] = bt_start
    cfg["end_date"] = bt_end
    cfg["ticker"] = "PS_GFEX_real1m" if from_real_file else "PS_GFEX_synth"
    # 真实分钟由多段合约拼接，日历不连续时 sigma 易「缺失率高」；勿按日整段剔除
    cfg["sigma_incomplete_day_drop"] = False if from_real_file else True

    if not from_real_file:
        print(
            "近似说明：分钟线由日线 O-H-L-C 插成，非真实盘口；绩效指标易不可信，仅作管线测试。"
        )
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

