#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对多晶硅真实分钟数据做小规模网格搜索，优化夏普（stdout 静默）。"""
from __future__ import annotations

import contextlib
import io
import itertools
from pathlib import Path

import pandas as pd

from noise_strategy_backtest import run_backtest


def main():
    csv_path = Path(__file__).resolve().parent / "data" / "ps_real_minute_1m.csv"
    peek = pd.read_csv(csv_path, parse_dates=["DateTime"])
    bt_start = peek["DateTime"].dt.date.min()
    bt_end = peek["DateTime"].dt.date.max()

    def base() -> dict:
        return {
            "data_path": str(csv_path),
            "ticker": "PS_GFEX_real1m",
            "initial_capital": 2_000_000.0,
            "start_date": bt_start,
            "end_date": bt_end,
            "print_daily_trades": False,
            "print_trade_details": False,
            "enable_transaction_fees": False,
            "slippage_per_share": 0.0,
            "transaction_fee_per_share": 0.0,
            "trading_end_time": (15, 0),
            "leverage": 1,
            "sigma_incomplete_day_drop": False,
        }

    lookbacks = [1, 10, 20]
    checks = [10, 15]
    ks = [1.0, 1.25, 1.5]
    vwaps = [False, True]
    trends = [None, {"metric": "er5", "min": 0.10}]
    intradays = [(False, 0.04), (True, 0.045)]
    trails = [(True, 0.01, 0.7), (False, 0.01, 0.7)]
    starts = [(9, 15), (9, 30)]
    max_pos = [5, 8]

    results = []
    failed = 0
    for tup in itertools.product(
        lookbacks, checks, ks, vwaps, trends, intradays, trails, starts, max_pos
    ):
        lb, ci, k, vw, tr, intr, trail_pack, st, mx = tup
        en_intra, pct = intr
        en_tr, act, cb = trail_pack
        cfg = base()
        cfg.update(
            lookback_days=lb,
            check_interval_minutes=ci,
            K1=k,
            K2=k,
            use_vwap=vw,
            entry_trend_filter=tr,
            enable_intraday_stop_loss=en_intra,
            intraday_stop_loss_pct=pct,
            enable_trailing_take_profit=en_tr,
            trailing_tp_activation_pct=act,
            trailing_tp_callback_pct=cb,
            trading_start_time=st,
            max_positions_per_day=mx,
        )
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                daily_df, _m, _t, metrics = run_backtest(cfg)
        except Exception:
            failed += 1
            continue
        if len(daily_df) < 30 or metrics.get("total_trades", 0) < 10:
            continue
        sh = float(metrics.get("sharpe_ratio", -999))
        results.append(
            (
                sh,
                float(metrics.get("total_return", 0)),
                float(metrics.get("mdd", 0)),
                int(metrics["total_trades"]),
                cfg,
            )
        )

    results.sort(key=lambda x: x[0], reverse=True)
    print(f"网格 {len(list(itertools.product(lookbacks, checks, ks, vwaps, trends, intradays, trails, starts, max_pos)))} 组 | 有效 {len(results)} | 异常 {failed}")
    for row in results[:15]:
        sh, tot, mdd, ntr, cfg = row
        tr = cfg.get("entry_trend_filter")
        trs = "off" if tr is None else f"er5>={tr['min']}"
        print(
            f"Sharpe={sh:6.3f} ret={100*tot:6.2f}% mdd={100*mdd:5.1f}% n={ntr:3d} | "
            f"lb={cfg['lookback_days']} ci={cfg['check_interval_minutes']} K={cfg['K1']} "
            f"vwap={cfg['use_vwap']} {trs} intra={cfg['enable_intraday_stop_loss']} "
            f"trail={cfg['enable_trailing_take_profit']} start={cfg['trading_start_time']} mx={cfg['max_positions_per_day']}"
        )


if __name__ == "__main__":
    main()
