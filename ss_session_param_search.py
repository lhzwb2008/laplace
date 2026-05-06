#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SS 期货分时段参数搜索脚本。

这个文件只用于离线研究，不作为最终主回测入口。主回测仍由
ss_sina_recent_validation.py 执行。
"""

from __future__ import annotations

import ast
import contextlib
import io
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from noise_strategy_backtest import run_backtest
from ss_sina_recent_validation import (
    BACKTEST_PARAMS,
    DATA_CSV,
    INITIAL_CAPITAL,
    SUMMARY_CSV,
    TRADES_CSV,
    eod_max_drawdown,
)


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "data"
SESSION_DIAG_CSV = RESULTS_DIR / "ss_session_diagnostics.csv"
PARAM_SEARCH_CSV = RESULTS_DIR / "ss_session_param_search.csv"
VALIDATION_CSV = RESULTS_DIR / "ss_session_param_validation.csv"

BASE_CFG = {
    "data_path": str(DATA_CSV),
    "ticker": "SS_SINA_raw_prevday_main_1m",
    "initial_capital": INITIAL_CAPITAL,
    **BACKTEST_PARAMS,
    "print_daily_trades": False,
    "print_trade_details": False,
    "random_plots": 0,
    "plot_days": [],
}

SESSIONS: dict[str, tuple[tuple[tuple[int, int], tuple[int, int]], ...]] = {
    "morning_1": (((9, 0), (10, 15)),),
    "morning_2": (((10, 30), (11, 30)),),
    "afternoon": (((13, 30), (14, 59)),),
    "morning_all": (((9, 0), (10, 15)), ((10, 30), (11, 30))),
    "day_all": (((9, 0), (10, 15)), ((10, 30), (11, 30)), ((13, 30), (14, 59))),
}


def run_quiet(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    with contextlib.redirect_stdout(io.StringIO()):
        return run_backtest(cfg)


def summarize(
    label: str,
    cfg: dict[str, Any],
    start_date: Any = None,
    end_date: Any = None,
    split: str = "full",
) -> dict[str, Any]:
    run_cfg = dict(cfg)
    if start_date is not None:
        run_cfg["start_date"] = start_date
    if end_date is not None:
        run_cfg["end_date"] = end_date

    daily_df, _monthly, trades, metrics = run_quiet(run_cfg)
    if len(daily_df) == 0:
        raise RuntimeError(f"{label} {split} 没有回测日")

    n_trades = 0 if trades is None else len(trades)
    pnl_sum = float(trades["pnl"].sum()) if n_trades else 0.0
    avg_pnl = float(trades["pnl"].mean()) if n_trades else 0.0
    win_rate = float((trades["pnl"] > 0).mean()) if n_trades else 0.0
    final_capital = float(daily_df["capital"].iloc[-1])
    total_return = final_capital / float(run_cfg["initial_capital"]) - 1.0
    max_drawdown = float(metrics.get("mdd", np.nan))
    eod_mdd = eod_max_drawdown(daily_df)
    annual_return = float(metrics.get("irr", np.nan))
    sharpe = float(metrics.get("sharpe_ratio", np.nan))
    calmar = annual_return / max_drawdown if max_drawdown > 0 else math.inf

    return {
        "label": label,
        "split": split,
        "start": daily_df.index.min(),
        "end": daily_df.index.max(),
        "days": len(daily_df),
        "final_capital": final_capital,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "eod_max_drawdown": eod_mdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "trades": n_trades,
        "win_rate": win_rate,
        "pnl_sum": pnl_sum,
        "avg_pnl": avg_pnl,
    }


def session_cfg(name: str, sessions: tuple[tuple[tuple[int, int], tuple[int, int]], ...]) -> dict[str, Any]:
    start = sessions[0][0]
    end = sessions[-1][1]
    cfg = dict(BASE_CFG)
    cfg.update(
        {
            "trading_sessions": sessions,
            "trading_start_time": start,
            "trading_end_time": end,
        }
    )
    return cfg


def diagnose_sessions() -> pd.DataFrame:
    rows = []
    for name, sessions in SESSIONS.items():
        cfg = session_cfg(name, sessions)
        rows.append(summarize(name, cfg))
    out = pd.DataFrame(rows).sort_values(["sharpe", "calmar"], ascending=False)
    out.to_csv(SESSION_DIAG_CSV, index=False)
    return out


def candidate_configs() -> list[tuple[str, dict[str, Any]]]:
    lookbacks = [3, 5, 10]
    intervals = [10, 20]
    k_pairs = [(1.6, 1.6), (1.8, 1.8), (2.2, 2.2), (1.8, 2.2), (2.2, 1.8)]
    max_positions = [1, 2]
    use_vwaps = [False, True]
    trailing_sets = [
        (False, 0.01, 0.7),
        (True, 0.01, 0.7),
    ]
    trend_filters = [
        None,
        {"metric": "er5", "min": 0.25},
    ]

    configs = []
    search_sessions = ["morning_2", "morning_all", "day_all"]
    for session_name in search_sessions:
        sessions = SESSIONS[session_name]
        for lookback, interval, k_pair, max_pos, use_vwap, trailing, trend_filter in itertools.product(
            lookbacks,
            intervals,
            k_pairs,
            max_positions,
            use_vwaps,
            trailing_sets,
            trend_filters,
        ):
            k1, k2 = k_pair
            enable_tp, activation, callback = trailing
            cfg = session_cfg(session_name, sessions)
            cfg.update(
                {
                    "lookback_days": lookback,
                    "check_interval_minutes": interval,
                    "K1": k1,
                    "K2": k2,
                    "max_positions_per_day": max_pos,
                    "use_vwap": use_vwap,
                    "enable_trailing_take_profit": enable_tp,
                    "trailing_tp_activation_pct": activation,
                    "trailing_tp_callback_pct": callback,
                    "entry_trend_filter": trend_filter,
                }
            )
            label = (
                f"{session_name}|lb{lookback}|i{interval}|k{k1:g}/{k2:g}|"
                f"m{max_pos}|vwap{int(use_vwap)}|tp{int(enable_tp)}-{activation:g}-{callback:g}|"
                f"tf{trend_filter if trend_filter else 'none'}"
            )
            configs.append((label, cfg))
    return configs


def date_splits() -> tuple[Any, Any, Any, Any]:
    raw = pd.read_csv(DATA_CSV, usecols=["Date"])
    dates = pd.to_datetime(raw["Date"]).dt.date.drop_duplicates().sort_values().to_list()
    train_end = dates[int(len(dates) * 0.65)]
    valid_start = dates[int(len(dates) * 0.65) + 1]
    return dates[0], train_end, valid_start, dates[-1]


def run_grid(limit: int | None = None) -> pd.DataFrame:
    start, train_end, _valid_start, _end = date_splits()
    rows = []
    configs = candidate_configs()
    if limit is not None:
        configs = configs[:limit]
    for i, (label, cfg) in enumerate(configs, start=1):
        row = summarize(label, cfg, start_date=start, end_date=train_end, split="train")
        row.update(
            {
                "session": label.split("|", 1)[0],
                "lookback_days": cfg["lookback_days"],
                "check_interval_minutes": cfg["check_interval_minutes"],
                "K1": cfg["K1"],
                "K2": cfg["K2"],
                "max_positions_per_day": cfg["max_positions_per_day"],
                "use_vwap": cfg["use_vwap"],
                "enable_trailing_take_profit": cfg["enable_trailing_take_profit"],
                "trailing_tp_activation_pct": cfg["trailing_tp_activation_pct"],
                "trailing_tp_callback_pct": cfg["trailing_tp_callback_pct"],
                "entry_trend_filter": cfg["entry_trend_filter"],
            }
        )
        rows.append(row)
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(PARAM_SEARCH_CSV, index=False)
            print(f"[搜索] 已完成 {i}/{len(configs)}")

    out = pd.DataFrame(rows)
    out = out.sort_values(["calmar", "sharpe", "annual_return"], ascending=False)
    out.to_csv(PARAM_SEARCH_CSV, index=False)
    return out


def rebuild_cfg_from_row(row: pd.Series) -> dict[str, Any]:
    session_name = str(row["session"])
    cfg = session_cfg(session_name, SESSIONS[session_name])
    trend_filter = row.get("entry_trend_filter")
    if isinstance(trend_filter, str) and trend_filter not in ("", "None", "nan"):
        trend_filter = ast.literal_eval(trend_filter)
    else:
        trend_filter = None
    cfg.update(
        {
            "lookback_days": int(row["lookback_days"]),
            "check_interval_minutes": int(row["check_interval_minutes"]),
            "K1": float(row["K1"]),
            "K2": float(row["K2"]),
            "max_positions_per_day": int(row["max_positions_per_day"]),
            "use_vwap": bool(row["use_vwap"]),
            "enable_trailing_take_profit": bool(row["enable_trailing_take_profit"]),
            "trailing_tp_activation_pct": float(row["trailing_tp_activation_pct"]),
            "trailing_tp_callback_pct": float(row["trailing_tp_callback_pct"]),
            "entry_trend_filter": trend_filter,
        }
    )
    return cfg


def validate_top(search_df: pd.DataFrame, top_n: int = 80) -> pd.DataFrame:
    start, train_end, valid_start, end = date_splits()
    filtered = search_df[
        (search_df["trades"] >= 20)
        & (search_df["max_drawdown"] > 0)
    ].head(top_n)
    if filtered.empty:
        filtered = search_df.head(top_n)
    rows = []
    for _, row in filtered.iterrows():
        cfg = rebuild_cfg_from_row(row)
        label = str(row["label"])
        for split, split_start, split_end in (
            ("train", start, train_end),
            ("valid", valid_start, end),
            ("full", None, None),
        ):
            result = summarize(label, cfg, start_date=split_start, end_date=split_end, split=split)
            result.update(
                {
                    "session": row["session"],
                    "lookback_days": cfg["lookback_days"],
                    "check_interval_minutes": cfg["check_interval_minutes"],
                    "K1": cfg["K1"],
                    "K2": cfg["K2"],
                    "max_positions_per_day": cfg["max_positions_per_day"],
                    "use_vwap": cfg["use_vwap"],
                    "enable_trailing_take_profit": cfg["enable_trailing_take_profit"],
                    "trailing_tp_activation_pct": cfg["trailing_tp_activation_pct"],
                    "trailing_tp_callback_pct": cfg["trailing_tp_callback_pct"],
                    "entry_trend_filter": cfg["entry_trend_filter"],
                }
            )
            rows.append(result)
    out = pd.DataFrame(rows)
    out.to_csv(VALIDATION_CSV, index=False)
    return out


def choose_best(validation_df: pd.DataFrame) -> pd.Series:
    pivot = validation_df.pivot_table(
        index="label",
        columns="split",
        values=["annual_return", "max_drawdown", "eod_max_drawdown", "sharpe", "calmar", "trades"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{split}" for metric, split in pivot.columns]
    meta_cols = [
        "session",
        "lookback_days",
        "check_interval_minutes",
        "K1",
        "K2",
        "max_positions_per_day",
        "use_vwap",
        "enable_trailing_take_profit",
        "trailing_tp_activation_pct",
        "trailing_tp_callback_pct",
        "entry_trend_filter",
    ]
    meta = validation_df.drop_duplicates("label").set_index("label")[meta_cols]
    ranked = pivot.join(meta)
    ranked = ranked[
        (ranked["annual_return_valid"] > 0)
        & (ranked["sharpe_valid"] > 0)
        & (ranked["trades_full"] >= 80)
        & (ranked["annual_return_full"] >= ranked["max_drawdown_full"] * 0.75)
    ].copy()
    if ranked.empty:
        ranked = pivot.join(meta).copy()
    ranked["score"] = (
        ranked["sharpe_valid"].fillna(-9) * 2
        + ranked["calmar_full"].replace([np.inf, -np.inf], np.nan).fillna(0)
        + ranked["sharpe_full"].fillna(-9)
        - ranked["max_drawdown_full"].fillna(1)
    )
    ranked = ranked.sort_values("score", ascending=False)
    return ranked.iloc[0]


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    diag = diagnose_sessions()
    print("[诊断] 分时段当前参数表现:")
    print(diag[["label", "annual_return", "max_drawdown", "eod_max_drawdown", "sharpe", "calmar", "trades", "win_rate"]].to_string(index=False))
    print(f"[诊断] 已写入 {SESSION_DIAG_CSV}")

    search = run_grid()
    print(f"[搜索] 已写入 {PARAM_SEARCH_CSV}，共 {len(search)} 组")
    print(search[["label", "annual_return", "max_drawdown", "sharpe", "calmar", "trades"]].head(10).to_string(index=False))

    validation = validate_top(search, top_n=30)
    print(f"[验证] 已写入 {VALIDATION_CSV}")
    best = choose_best(validation)
    print("\n[最佳候选]")
    print(best.to_string())
    print(f"\n主回测汇总仍由 {SUMMARY_CSV} 输出，逐笔交易由 {TRADES_CSV} 输出。")


if __name__ == "__main__":
    main()
