#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用天勤免费行情补沪金 5 分钟线。只连模拟盘，不下单。

账号从 butterfly/tianqin_test.py 的 TqAuth 读取。
已有文件会跳过。免费账户单次最多约 10000 根，覆盖合约到期前几个月。
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
from tqsdk import TqApi, TqAuth, TqSim

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "cn_fut" / "5m" / "SHFE_au"
AUTH_FILE = ROOT / "butterfly" / "tianqin_test.py"


def auth() -> TqAuth:
    src = AUTH_FILE.read_text()
    m = re.search(r'TqAuth\("([^"]+)",\s*"([^"]+)"\)', src)
    if not m:
        raise SystemExit(f"未在 {AUTH_FILE} 找到 TqAuth")
    return TqAuth(m.group(1), m.group(2))


def symbols(api: TqApi) -> list[str]:
    alive = set(api.query_quotes(ins_class="FUTURE", product_id="au", expired=False))
    dead = set(api.query_quotes(ins_class="FUTURE", product_id="au", expired=True))
    pat = re.compile(r"^SHFE\.au\d{4}$")
    picked = []
    for s in alive | dead:
        if not pat.match(s):
            continue
        ym = int(s[-4:])
        yyyymm = 200000 + ym
        if 202101 <= yyyymm <= 202712:
            picked.append(s)
    return sorted(picked)


def save(symbol: str, k: pd.DataFrame) -> None:
    df = k.dropna(subset=["close"])
    if len(df) > 1:
        df = df.iloc[:-1]
    out = pd.DataFrame({
        "DateTime": pd.to_datetime(df["datetime"].astype("int64"), unit="ns") + pd.Timedelta(hours=8),
        "Open": df["open"].to_numpy(dtype="float64"),
        "High": df["high"].to_numpy(dtype="float64"),
        "Low": df["low"].to_numpy(dtype="float64"),
        "Close": df["close"].to_numpy(dtype="float64"),
        "Volume": df["volume"].to_numpy(dtype="float64"),
        "OpenInterest": df["close_oi"].to_numpy(dtype="float64"),
    })
    dest = OUT / f"{symbol.split('.')[-1]}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"{symbol} {len(out)}  {out.DateTime.iloc[0]} -> {out.DateTime.iloc[-1]}", flush=True)


def main() -> None:
    api = TqApi(TqSim(), auth=auth())
    try:
        todo = symbols(api)
    finally:
        api.close()
    print(f"沪金合约 {len(todo)}", flush=True)
    api = TqApi(TqSim(), auth=auth())
    used = 0
    try:
        for i, symbol in enumerate(todo, 1):
            dest = OUT / f"{symbol.split('.')[-1]}.csv"
            if dest.exists() and dest.stat().st_size > 800:
                continue
            if used >= 12:
                api.close()
                api = TqApi(TqSim(), auth=auth())
                used = 0
            k = api.get_kline_serial(symbol, 300, data_length=10000)
            save(symbol, k)
            used += 1
            print(f"[{i}/{len(todo)}]", flush=True)
    finally:
        api.close()


if __name__ == "__main__":
    main()
