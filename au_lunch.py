#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""沪金午休跟随。

伦敦金在沪金午休（11:30–13:29）的涨跌，到 13:30 已经知道。
涨跌达到 5 个基点才做，方向与伦敦金相同，多空同一个门槛。
13:30 开盘进场，持有 60 分钟，14:30 平仓，不过夜。
合约用上一日成交量最大的那张。平今免费，每手另加进出各 1 跳。

20 万满仓：每天手数 = 权益 // （金价 × 1000 × 16%）。
亏损超过当日可用资金时，按强平把亏损封顶在可用资金。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
AU_DIR = ROOT / "data" / "cn_fut" / "5m" / "SHFE_au"
XAU_PKL = ROOT / "data" / "xauusd_1m_bj.pkl"
OUT_DIR = ROOT / "data" / "au_lunch"

GATE = 5e-4          # 5 bp
HOLD_HM = 14 * 60 + 25  # 5 分钟线起点，收盘时刻 14:30
ENTRY_HM = 13 * 60 + 30
MULT = 1000
MARGIN = 0.16
TICK = 0.02
FEE_YUAN = 20.01     # 开仓；平今为 0
CASH0 = 200_000.0


def trading_day(bj: pd.Series) -> pd.Series:
    """21:00 之后的夜盘算下一个工作日；周五夜盘算下周一。"""
    td = (bj + pd.Timedelta(hours=3)).dt.normalize()
    wd = td.dt.weekday.to_numpy()
    extra = np.where(wd == 5, 2, np.where(wd == 6, 1, 0))
    return (td + pd.to_timedelta(extra, unit="D")).dt.date


def load_au() -> pd.DataFrame:
    frames = []
    for f in sorted(AU_DIR.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["DateTime"])
        if df.empty:
            continue
        df["contract"] = f.stem
        frames.append(df)
    if not frames:
        raise FileNotFoundError(AU_DIR)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["contract", "DateTime"], keep="last")
    df = df[df["Close"] > 0].copy()
    df["hm"] = df["DateTime"].dt.hour * 60 + df["DateTime"].dt.minute
    night = (df["hm"] >= 21 * 60) | (df["hm"] < 2 * 60 + 30)
    day = (
        ((df["hm"] >= 9 * 60) & (df["hm"] < 10 * 60 + 15))
        | ((df["hm"] >= 10 * 60 + 30) & (df["hm"] < 11 * 60 + 30))
        | ((df["hm"] >= 13 * 60 + 30) & (df["hm"] < 15 * 60))
    )
    df = df.loc[night | day].copy()
    df["td"] = trading_day(df["DateTime"])
    return df.sort_values(["td", "contract", "DateTime"]).reset_index(drop=True)


def leader_bars(df: pd.DataFrame) -> pd.DataFrame:
    """当天可交易合约 = 上一日成交量最大的合约。"""
    vol = df.groupby(["td", "contract"], sort=False)["Volume"].sum()
    leader = vol.groupby(level=0).idxmax().map(lambda x: x[1]).sort_index().shift(1)
    use = df["td"].map(leader)
    out = df[df["contract"].to_numpy() == use.to_numpy()].copy()
    return out.sort_values(["td", "DateTime"]).reset_index(drop=True)


def load_xau() -> pd.Series:
    xau = pd.read_pickle(XAU_PKL)
    px = xau.set_index("BJ")["Close"].sort_index()
    return px[~px.index.duplicated(keep="last")]


def xau_at(px: pd.Series, ts: pd.Timestamp, tol: str = "5min") -> float:
    i = int(px.index.searchsorted(ts, side="right")) - 1
    if i < 0 or ts - px.index[i] > pd.Timedelta(tol):
        return np.nan
    return float(px.iloc[i])


def cost_frac(price: float) -> float:
    """开仓费 + 进出各 1 跳，占名义本金。平今为 0。"""
    return FEE_YUAN / (price * MULT) + 2 * TICK / price


def trades() -> pd.DataFrame:
    bars = leader_bars(load_au())
    px = load_xau()
    rows = []
    for td, g in bars.groupby("td", sort=True):
        if td < dt.date(2021, 1, 1):
            continue
        p0 = xau_at(px, pd.Timestamp(td.year, td.month, td.day, 11, 30))
        p1 = xau_at(px, pd.Timestamp(td.year, td.month, td.day, 13, 29))
        if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0):
            continue
        xret = p1 / p0 - 1.0
        if abs(xret) < GATE:
            continue
        eb = g[g["hm"] == ENTRY_HM]
        xb = g[g["hm"] == HOLD_HM]
        if eb.empty or xb.empty:
            continue
        entry = float(eb["Open"].iloc[0])
        exit_px = float(xb["Close"].iloc[0])
        if entry <= 0 or exit_px <= 0:
            continue
        side = 1 if xret > 0 else -1
        gross = side * (exit_px / entry - 1.0)
        rows.append({
            "td": td, "side": side, "xret": xret, "entry": entry, "exit": exit_px,
            "gross": gross, "net": gross - cost_frac(entry),
        })
    return pd.DataFrame(rows)


def full_margin(tr: pd.DataFrame, cash0: float = CASH0, years: int = 5) -> pd.DataFrame:
    """最近 years 年，每天按交易所保证金开满整数手。"""
    end = pd.Timestamp(tr["td"].max())
    cut = (end - pd.DateOffset(years=years)).date()
    tr = tr[tr["td"] >= cut].sort_values("td")
    eq = float(cash0)
    rows = []
    for r in tr.itertuples(index=False):
        m1 = float(r.entry) * MULT * MARGIN
        lots = int(eq // m1) if m1 > 0 else 0
        pnl = 0.0
        called = False
        if lots >= 1:
            pnl = float(r.net) * float(r.entry) * MULT * lots
            free = eq - m1 * lots
            if pnl < -free:
                called = True
                pnl = -free
            eq += pnl
        rows.append({
            "td": r.td, "side": int(r.side), "entry": float(r.entry), "exit": float(r.exit),
            "xret": float(r.xret), "lots": lots, "pnl": pnl, "equity": eq, "called": int(called),
        })
    return pd.DataFrame(rows)


def monthly(path: pd.DataFrame, cash0: float = CASH0) -> pd.DataFrame:
    p = path.copy()
    p["ym"] = pd.to_datetime(p["td"]).dt.strftime("%Y-%m")
    rows = []
    for ym, g in p.groupby("ym", sort=True):
        start = float(g["equity"].iloc[0] - g["pnl"].iloc[0])
        end = float(g["equity"].iloc[-1])
        curve = np.concatenate([[start], g["equity"].to_numpy()])
        peak = np.maximum.accumulate(curve)
        rows.append({
            "month": ym, "trades": len(g), "win": float((g["pnl"] > 0).mean()),
            "lots": int(g["lots"].median()), "pnl": float(g["pnl"].sum()),
            "start": start, "end": end, "ret": end / start - 1,
            "mdd": float((curve / peak - 1).min()), "liq": int(g["called"].sum()),
        })
    out = pd.DataFrame(rows)
    if abs(out["end"].iloc[-1] - path["equity"].iloc[-1]) > 1:
        raise RuntimeError("月末权益和成交明细不一致")
    if abs(out["start"].iloc[0] - cash0) > 1:
        raise RuntimeError("起始权益不是初始资金")
    return out


def main() -> None:
    path = full_margin(trades())
    mon = monthly(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.to_csv(OUT_DIR / "trades.csv", index=False)
    mon.to_csv(OUT_DIR / "monthly.csv", index=False)
    print(f"{path['td'].iloc[0]} -> {path['td'].iloc[-1]}  {len(path)} 笔")
    print(f"期末 {path['equity'].iloc[-1]:.0f}  强平 {int(path['called'].sum())}")
    print(mon.to_string(index=False))


if __name__ == "__main__":
    main()
