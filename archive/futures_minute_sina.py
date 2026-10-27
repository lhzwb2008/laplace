#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新浪期货分钟 K（AkShare: futures_zh_minute_sina）多合约「瀑布拼接」工具。

拼接逻辑（重要）
----------------
1. **禁止**按时间全局交错多合约（会制造假跳价）。
2. 对每个合约拉约 1023 根最近分钟，按 **交割月顺序（合约代码 YYMM）** 排序各段，再瀑布拼接：
   每段只保留 **datetime 严格大于** 上一段已覆盖的最大时间。
3. **换月价位对齐**：拼接后按时间排序，在 **contract 变化** 处，用「前一根 Close − 新段首根 Open」
   对 **该根及之后** 所有 OHLC 做 **累加平移**，消除合约间 **绝对价差** 在换月当分钟的假收益（近似连续主力的「价差归零」处理，非交易所官方复权）。

单合约模式无对齐问题；disjoint 拉长样本时必须做第 3 步，否则 σ 与盈亏会被换月跳变严重污染。

供 `fetch_polysilicon_real_minute.py`、`noise_multi_symbol_scan.py` 复用。
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List

import pandas as pd

try:
    import akshare as ak
except ImportError as e:
    raise SystemExit("请先安装: pip install akshare pandas") from e


def contract_delivery_key(symbol: str) -> tuple[int, int]:
    """从合约代码解析交割 (年%100, 月)，用于排序：如 rb2505 -> (25, 5)。"""
    m = re.search(r"(\d{2})(\d{2})$", str(symbol).strip().lower())
    if not m:
        return (99, 99)
    return (int(m.group(1)), int(m.group(2)))


def apply_roll_additive_align(df: pd.DataFrame) -> pd.DataFrame:
    """
    换月处将新合约首根 Open 对齐到前一根 Close（累加平移后续 OHLC）。
    要求列：datetime, open, high, low, close, contract（小写）。
    """
    df = df.sort_values("datetime").reset_index(drop=True).copy()
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"apply_roll_additive_align: 缺少列 {c}")
    if "contract" not in df.columns:
        return df
    for i in range(1, len(df)):
        if df["contract"].iloc[i] != df["contract"].iloc[i - 1]:
            delta = float(df["close"].iloc[i - 1]) - float(df["open"].iloc[i])
            idx = ["open", "high", "low", "close"]
            df.iloc[i:, df.columns.get_indexer(idx)] = df.iloc[i:, df.columns.get_indexer(idx)] + delta
    return df


def candidate_symbols_for_prefix(prefix: str) -> list[str]:
    """如 rb -> rb2401 ... rb2712（小写，与多数品种新浪代码一致）。"""
    p = prefix.strip().lower()
    out: List[str] = []
    for y in (24, 25, 26, 27):
        for m in range(1, 13):
            out.append(f"{p}{y:02d}{m:02d}")
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


def merge_disjoint_segments(chunks: List[pd.DataFrame], sleep_s: float) -> pd.DataFrame:
    """
    按 **交割月** 排序合约段（非 min(datetime)，避免与换月顺序不一致），再瀑布拼接，
    最后 **换月价位对齐**。
    """
    if not chunks:
        raise RuntimeError("merge_disjoint_segments: chunks 为空")

    chunks = sorted(
        chunks,
        key=lambda g: contract_delivery_key(str(g["contract"].iloc[0])),
    )

    parts: List[pd.DataFrame] = []
    max_seen = pd.Timestamp.min
    for df in chunks:
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
    out = apply_roll_additive_align(out)
    return out


def fetch_disjoint_for_prefix(
    prefix: str,
    period: str = "1",
    sleep_s: float = 0.12,
    verbose: bool = True,
) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for sym in candidate_symbols_for_prefix(prefix):
        df = fetch_one(sym, period)
        if df is None:
            time.sleep(sleep_s)
            continue
        chunks.append(df)
        time.sleep(sleep_s)
    if not chunks:
        raise RuntimeError(f"品种前缀 {prefix!r} 未拉到任何合约分钟数据")
    raw = merge_disjoint_segments(chunks, sleep_s=0.0)
    if verbose:
        print(
            f"[{prefix}] disjoint: {len(chunks)} 合约段, {len(raw)} 根, "
            f"{raw['datetime'].min()} ~ {raw['datetime'].max()}（已换月价对齐）"
        )
    return raw


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


def save_disjoint_csv(prefix: str, out_path: Path, period: str = "1", sleep_s: float = 0.12) -> Path:
    raw = fetch_disjoint_for_prefix(prefix, period=period, sleep_s=sleep_s, verbose=True)
    clean = to_backtest_csv(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)
    print(f"已写入 {out_path} ({len(clean)} 行)")
    return out_path
