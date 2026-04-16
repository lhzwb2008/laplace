#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多晶硅真实分钟数据 — 参数抽样 / 粗网格搜索，以夏普为主排序（stdout 静默）。

示例:
  ./.venv_run/bin/python optimize_polysilicon_sharpe.py --samples 400 --seed 42 --top 40 --also-sort-calmar
  ./.venv_run/bin/python optimize_polysilicon_sharpe.py --mode grid   # 全组合粗网格（约数千组，较慢）

说明:
  - `--mode random` 下会在「宽区间」内随机组合：lookback、检查间隔、K1/K2、VWAP、
    多种趋势门控（er5/range5/linreg/weekly_sn/dist_ma20/rsi/vol_ratio）、开盘推迟、
    日内止损与追踪止盈等。
  - 结果为**同一段样本内**排序，夏普极高时多为过拟合风险，后续请用样本外或滚动验证。

耗时: 约 2s/次量级，400 次约 12–18 分钟（视机器而定）。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from noise_strategy_backtest import run_backtest


def _trend_str(tr: Any) -> str:
    if tr is None:
        return "off"
    if isinstance(tr, list):
        return "AND(" + ",".join(_trend_str(x) for x in tr) + ")"
    if isinstance(tr, dict):
        if "min" in tr:
            return f"{tr['metric']}>={tr['min']}"
        if "max" in tr:
            return f"{tr['metric']}<={tr['max']}"
        if "min_abs" in tr:
            return f"|{tr['metric']}|>={tr['min_abs']}"
    return str(tr)


def _run_one(
    cfg: dict, min_trades: int = 8
) -> Tuple[Optional[dict], Optional[Tuple[float, float, float, int, float]]]:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            daily_df, _m, _t, metrics = run_backtest(cfg)
    except Exception:
        return None, None
    if len(daily_df) < 30:
        return None, None
    ntr = int(metrics.get("total_trades", 0))
    if ntr < min_trades:
        return None, None
    sh = float(metrics.get("sharpe_ratio", -999))
    tot = float(metrics.get("total_return", 0))
    mdd = float(metrics.get("mdd", 0))
    cr = metrics.get("calmar_ratio")
    if cr is None or (isinstance(cr, float) and cr != cr):  # nan
        cal = 0.0
    else:
        cal = float(cr)
    out_metrics = (sh, tot, mdd, ntr, cal)
    return None, out_metrics


def _base_cfg(csv_path: Path, bt_start, bt_end) -> dict:
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


def sample_random_trend(rng: random.Random) -> Optional[Dict[str, Any]]:
    """粗抽样：无门控 / 单条门控（metric 与阈值随机）。"""
    r = rng.random()
    if r < 0.18:
        return None
    metric = rng.choices(
        ["er5", "range5", "linreg5_r2", "weekly_sn", "dist_ma20_abs", "rsi5", "vol_ratio"],
        weights=[22, 18, 12, 12, 12, 12, 12],
        k=1,
    )[0]
    if metric == "er5":
        return {"metric": "er5", "min": rng.choice([0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20])}
    if metric == "range5":
        return {"metric": "range5", "max": rng.choice([0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08])}
    if metric == "linreg5_r2":
        return {"metric": "linreg5_r2", "min": rng.choice([0.15, 0.25, 0.35, 0.45, 0.55, 0.65])}
    if metric == "weekly_sn":
        return {"metric": "weekly_sn", "min": rng.choice([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5])}
    if metric == "dist_ma20_abs":
        return {"metric": "dist_ma20_abs", "min_abs": rng.choice([0.005, 0.01, 0.015, 0.02, 0.03, 0.04])}
    if metric == "rsi5":
        return {"metric": "rsi5", "min": rng.choice([35, 40, 45, 50])}  # 偏强
    if metric == "vol_ratio":
        return {"metric": "vol_ratio", "min": rng.choice([0.6, 0.8, 1.0, 1.2, 1.5])}
    return None


def build_cfg_from_sample(
    base: dict,
    rng: random.Random,
    *,
    lookbacks: List[int],
    checks: List[int],
    ks_lo: float,
    ks_hi: float,
) -> dict:
    lb = rng.choice(lookbacks)
    ci = rng.choice(checks)
    # K1/K2：多数对称，少数轻微不对称（期货远近月价差下轨宽可不同）
    if rng.random() < 0.82:
        k = round(rng.uniform(ks_lo, ks_hi), 3)
        k1 = k2 = k
    else:
        k1 = round(rng.uniform(ks_lo, ks_hi), 3)
        k2 = round(rng.uniform(ks_lo, ks_hi), 3)

    cfg = {**base}
    cfg.update(
        lookback_days=lb,
        check_interval_minutes=ci,
        K1=k1,
        K2=k2,
        use_vwap=rng.random() < 0.5,
        entry_trend_filter=sample_random_trend(rng),
        trading_start_time=rng.choice(
            [(9, 0), (9, 5), (9, 10), (9, 15), (9, 20), (9, 30)]
        ),
        max_positions_per_day=rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15]),
    )
    # 日内止损 / 追踪止盈：宽区间随机
    if rng.random() < 0.55:
        cfg["enable_intraday_stop_loss"] = False
        cfg["intraday_stop_loss_pct"] = 0.04
    else:
        cfg["enable_intraday_stop_loss"] = True
        cfg["intraday_stop_loss_pct"] = round(rng.uniform(0.015, 0.08), 4)

    if rng.random() < 0.55:
        cfg["enable_trailing_take_profit"] = False
        cfg["trailing_tp_activation_pct"] = 0.01
        cfg["trailing_tp_callback_pct"] = 0.7
    else:
        cfg["enable_trailing_take_profit"] = True
        cfg["trailing_tp_activation_pct"] = round(rng.uniform(0.003, 0.02), 5)
        cfg["trailing_tp_callback_pct"] = round(rng.uniform(0.35, 0.82), 3)

    return cfg


def run_random_search(
    base: dict,
    rng: random.Random,
    n_samples: int,
    lookbacks: List[int],
    checks: List[int],
    ks_lo: float,
    ks_hi: float,
    min_trades: int = 8,
) -> List[Tuple[float, float, float, float, int, dict]]:
    seen = set()
    results: List[Tuple[float, float, float, float, int, dict]] = []
    failed = 0
    skipped_dup = 0

    for _ in range(n_samples):
        cfg = build_cfg_from_sample(base, rng, lookbacks=lookbacks, checks=checks, ks_lo=ks_lo, ks_hi=ks_hi)
        key = (
            cfg["lookback_days"],
            cfg["check_interval_minutes"],
            cfg["K1"],
            cfg["K2"],
            cfg["use_vwap"],
            str(cfg.get("entry_trend_filter")),
            cfg["enable_intraday_stop_loss"],
            round(cfg["intraday_stop_loss_pct"], 4),
            cfg["enable_trailing_take_profit"],
            round(cfg["trailing_tp_activation_pct"], 5),
            round(cfg["trailing_tp_callback_pct"], 3),
            cfg["trading_start_time"],
            cfg["max_positions_per_day"],
        )
        if key in seen:
            skipped_dup += 1
            continue
        seen.add(key)

        _m, out = _run_one(cfg, min_trades=min_trades)
        if out is None:
            failed += 1
            continue
        sh, tot, mdd, ntr, cal = out
        results.append((sh, tot, mdd, cal, ntr, cfg))

    results.sort(key=lambda x: x[0], reverse=True)
    print(
        f"[随机抽样] 请求 {n_samples} 次 | 有效 {len(results)} | 重复跳过 {skipped_dup} | 失败/过滤 {failed}"
    )
    return results


def run_grid_coarse(base: dict, min_trades: int = 8) -> List[Tuple[float, float, float, float, int, dict]]:
    """粗网格：区间拉大但维度受限，避免爆炸。"""
    lookbacks = [5, 10, 20, 30, 40]
    checks = [3, 5, 10, 15, 30]
    ks = [0.8, 1.0, 1.2, 1.5, 1.8]
    vwaps = [True, False]
    trends: List[Optional[Dict[str, Any]]] = [
        None,
        {"metric": "er5", "min": 0.06},
        {"metric": "er5", "min": 0.12},
        {"metric": "range5", "max": 0.04},
        {"metric": "range5", "max": 0.06},
        {"metric": "linreg5_r2", "min": 0.35},
    ]
    intradays = [
        (False, 0.04),
        (True, 0.03),
        (True, 0.06),
    ]
    trails = [
        (False, 0.01, 0.7),
        (True, 0.008, 0.65),
        (True, 0.015, 0.55),
    ]
    starts = [(9, 0), (9, 15), (9, 30)]
    max_pos = [4, 6, 10]

    results = []
    failed = 0
    total = len(list(itertools.product(lookbacks, checks, ks, vwaps, trends, intradays, trails, starts, max_pos)))
    for tup in itertools.product(lookbacks, checks, ks, vwaps, trends, intradays, trails, starts, max_pos):
        lb, ci, k, vw, tr, intr, trail_pack, st, mx = tup
        en_intra, pct = intr
        en_tr, act, cb = trail_pack
        cfg = {**base}
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
        _m, out = _run_one(cfg, min_trades=min_trades)
        if out is None:
            failed += 1
            continue
        sh, tot, mdd, ntr, cal = out
        results.append((sh, tot, mdd, cal, ntr, cfg))

    results.sort(key=lambda x: x[0], reverse=True)
    print(f"[粗网格] 组合数 {total} | 有效 {len(results)} | 失败 {failed}")
    return results


def print_top(
    results: List[Tuple[float, float, float, float, int, dict]],
    top_n: int,
    also_calmar: bool,
) -> None:
    for i, row in enumerate(results[:top_n], 1):
        sh, tot, mdd, cal, ntr, cfg = row
        trs = _trend_str(cfg.get("entry_trend_filter"))
        line = (
            f"{i:2d}. Sharpe={sh:7.3f}  ret={100*tot:7.2f}%  mdd={100*mdd:6.1f}%  "
            f"ntr={ntr:4d}  "
            f"lb={cfg['lookback_days']:2d} ci={cfg['check_interval_minutes']:2d} "
            f"K1={cfg['K1']:.3f} K2={cfg['K2']:.3f} vwap={cfg['use_vwap']!s:5} "
            f"{trs[:40]:<40} "
            f"intra={cfg['enable_intraday_stop_loss']!s:5}({cfg['intraday_stop_loss_pct']:.3f}) "
            f"trail={cfg['enable_trailing_take_profit']!s:5} "
            f"st={cfg['trading_start_time']} mx={cfg['max_positions_per_day']}"
        )
        if also_calmar:
            if cal == float("inf") or cal > 1e200:
                line += "  calmar=inf"
            else:
                line += f"  calmar={cal:.4f}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="多晶硅噪声策略参数抽样 / 粗网格")
    ap.add_argument("--mode", choices=["random", "grid"], default="random")
    ap.add_argument("--samples", type=int, default=450, help="随机模式下的抽样次数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks-lo", type=float, default=0.55, help="随机模式下 K1/K2 下界")
    ap.add_argument("--ks-hi", type=float, default=2.2, help="随机模式下 K1/K2 上界")
    ap.add_argument("--top", type=int, default=30, help="打印前 N 名（按夏普）")
    ap.add_argument(
        "--also-sort-calmar",
        action="store_true",
        help="额外打印按 Calmar 排序的前 10（需 metrics 含 calmar）",
    )
    ap.add_argument("--min-trades", type=int, default=8, help="过滤：最少成交笔数")
    args = ap.parse_args()

    csv_path = Path(__file__).resolve().parent / "data" / "ps_real_minute_1m.csv"
    if not csv_path.is_file():
        raise SystemExit(f"缺少数据文件: {csv_path}")
    peek = pd.read_csv(csv_path, parse_dates=["DateTime"])
    bt_start = peek["DateTime"].dt.date.min()
    bt_end = peek["DateTime"].dt.date.max()
    base = _base_cfg(csv_path, bt_start, bt_end)

    rng = random.Random(args.seed)

    # 宽区间（lookback / check / K）
    lookbacks = [1, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60]
    checks = [1, 2, 3, 5, 8, 10, 15, 20, 30]

    if args.mode == "grid":
        results = run_grid_coarse(base, min_trades=args.min_trades)
    else:
        results = run_random_search(
            base,
            rng,
            n_samples=args.samples,
            lookbacks=lookbacks,
            checks=checks,
            ks_lo=args.ks_lo,
            ks_hi=args.ks_hi,
            min_trades=args.min_trades,
        )

    print(f"\n=== 按夏普比 Top {args.top} ===\n")
    print_top(results, args.top, also_calmar=True)

    if args.also_sort_calmar and results:
        def _cal_key(x: Tuple[float, float, float, float, int, dict]) -> float:
            c = x[3]
            if c != c or c == float("inf"):  # nan / inf
                return 1e300
            return c

        by_cal = sorted(results, key=_cal_key, reverse=True)
        print("\n=== 按 Calmar Top 10 ===\n")
        print_top(by_cal, 10, also_calmar=True)


if __name__ == "__main__":
    main()
