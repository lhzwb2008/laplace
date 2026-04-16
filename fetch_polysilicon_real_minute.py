#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多晶硅（PS）期货 — 从新浪财经拉取真实分钟 K（AkShare: futures_zh_minute_sina）。

重要（必读）
------------
- 新浪单次约 **1023 根**，无法按起止日期任意拉历史。
- **禁止**把多合约结果按时间「全局排序后交错拼接」：不同合约绝对价差可达数千，
  相邻分钟会交替出现远月/近月报价，**不是**连续合约走势，会导致假突破与假巨亏。
- **默认**两种安全模式二选一：
  1) **single**：只拉一个合约（默认 PS2606），序列连续，约 1023 根。
  2) **disjoint**：多合约各拉一段，按「合约最早时间」排序后**瀑布拼接**——
     下一段只保留 **datetime 严格大于** 上一段已覆盖的最大时间，避免同日交错。

用法
----
  pip install akshare pandas
  python fetch_polysilicon_real_minute.py
  python fetch_polysilicon_real_minute.py --mode disjoint
  python fetch_polysilicon_real_minute.py --mode single --contract PS2606
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd

try:
    import akshare as ak
except ImportError as e:
    raise SystemExit("请先安装: pip install akshare pandas") from e


def candidate_ps_symbols() -> list[str]:
    out: List[str] = []
    for y in (24, 25, 26, 27):
        for m in range(1, 13):
            out.append(f"PS{y:02d}{m:02d}")
    return out


def fetch_one(symbol: str, period: str) -> pd.DataFrame | None:
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=str(period))
    except Exception:
        return None
    if df is None or len(df) < 30:
        return None
    df = df.copy()
    df["contract"] = symbol
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def merge_disjoint_segments(
    chunks: List[pd.DataFrame],
    sleep_s: float,
) -> pd.DataFrame:
    """
    按每段 min(datetime) 排序；依次只追加「时间戳晚于已覆盖最大时间」的行，
    避免不同合约在同一交易日内的分钟交错。
    """
    metas = []
    for df in chunks:
        metas.append((df["datetime"].min(), df["datetime"].max(), df))
    metas.sort(key=lambda x: x[0])

    parts: List[pd.DataFrame] = []
    max_seen = pd.Timestamp.min
    for t0, t1, df in metas:
        nxt = df[df["datetime"] > max_seen].copy()
        if len(nxt) == 0:
            continue
        parts.append(nxt)
        max_seen = nxt["datetime"].max()
        time.sleep(sleep_s)

    if not parts:
        raise RuntimeError("disjoint 合并后为空")

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values("datetime").reset_index(drop=True)
    print(
        f"disjoint 合并: {len(parts)} 段，共 {len(out)} 根K，"
        f"时间 {out['datetime'].min()} ~ {out['datetime'].max()}"
    )
    return out


def fetch_mode_single(contract: str, period: str, sleep_s: float) -> pd.DataFrame:
    df = fetch_one(contract, period)
    if df is None:
        raise RuntimeError(f"合约 {contract} 无分钟数据")
    time.sleep(sleep_s)
    print(f"single 模式: {contract}，共 {len(df)} 根")
    return df.sort_values("datetime").reset_index(drop=True)


def fetch_mode_disjoint(period: str, sleep_s: float) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for sym in candidate_ps_symbols():
        df = fetch_one(sym, period)
        if df is None:
            time.sleep(sleep_s)
            continue
        chunks.append(df)
        time.sleep(sleep_s)
    if not chunks:
        raise RuntimeError("未拉到任何合约分钟数据")
    return merge_disjoint_segments(chunks, sleep_s=0.0)


def validate_price_continuity(df: pd.DataFrame, max_abs_pct: float = 0.05) -> None:
    """打印分钟收益率极端值统计，便于发现错误拼接。"""
    col = "Close" if "Close" in df.columns else "close"
    c = pd.to_numeric(df[col], errors="coerce")
    r = c.pct_change().abs()
    bad = (r > max_abs_pct).sum()
    print(
        f"[校验] 分钟 |pct_change| 最大值 {r.max():.4%}，"
        f"超过 {max_abs_pct:.0%} 的根数: {bad} / {len(df)}"
    )
    if bad > len(df) * 0.02:
        print(
            "[警告] 极端分钟收益占比过高，若未使用 disjoint/single，"
            "可能是多合约交错导致，回测结果不可信。"
        )


def to_backtest_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(df["datetime"]),
            "Open": pd.to_numeric(df["open"], errors="coerce"),
            "High": pd.to_numeric(df["high"], errors="coerce"),
            "Low": pd.to_numeric(df["low"], errors="coerce"),
            "Close": pd.to_numeric(df["close"], errors="coerce"),
            "Volume": pd.to_numeric(df["volume"], errors="coerce"),
        }
    )
    out["Turnover"] = out["Close"] * out["Volume"]
    if "contract" in df.columns:
        out["contract"] = df["contract"].values
    return out.sort_values("DateTime").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["single", "disjoint"],
        default="disjoint",
        help="single=单合约；disjoint=多合约瀑布拼接（推荐拉长样本）",
    )
    ap.add_argument("--contract", default="PS2606", help="mode=single 时合约代码")
    ap.add_argument(
        "--period",
        default="1",
        choices=["1", "5", "15", "30", "60"],
        help="分钟周期",
    )
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--sleep", type=float, default=0.12)
    args = ap.parse_args()

    base = Path(__file__).resolve().parent / "data"
    base.mkdir(parents=True, exist_ok=True)
    suf = "1m" if args.period == "1" else f"{args.period}m"
    out_path = args.output or base / f"ps_real_minute_{suf}.csv"

    print(f"mode={args.mode} period={args.period} -> {out_path}")

    if args.mode == "single":
        raw = fetch_mode_single(args.contract, args.period, args.sleep)
    else:
        raw = fetch_mode_disjoint(args.period, args.sleep)

    clean = to_backtest_csv(raw)
    validate_price_continuity(clean)
    clean.to_csv(out_path, index=False)
    print(f"已写入 {len(clean)} 行，时间 {clean['DateTime'].min()} ~ {clean['DateTime'].max()}")


if __name__ == "__main__":
    main()
