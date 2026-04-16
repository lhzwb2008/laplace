#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将「多晶硅当前默认」噪声策略参数（PS_NOISE_STRATEGY_PARAMS）套到其他期货品种，
拉新浪 1 分钟（disjoint 瀑布拼接）后静默回测，输出对比表。

用法:
  python noise_multi_symbol_scan.py
  python noise_multi_symbol_scan.py --symbols rb ss au jd ag
  python noise_multi_symbol_scan.py --skip-fetch   # 仅用 data/noise_scan/*.csv 已有文件

注意
----
- 瀑布拼接在 `futures_minute_sina.merge_disjoint_segments` 中：按**交割月**排序合约段，
  拼接后对**换月边界**做**价位累加对齐**（消除合约间绝对价差导致的假跳变）；仍非交易所官方连续主力。
- 交易时段仍沿用脚本内参数（默认 9:30–15:00 日盘检查点）；若分钟数据含**夜盘**，
  引擎仍会读入这些 K 线参与 sigma/VWAP，与真实「只做日盘」可能有偏差。
- 不同品种价位、波动、换月结构不同，**同一套参数不保证合理**，本脚本只做横向对比参考。
"""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import pandas as pd

from futures_minute_sina import fetch_disjoint_for_prefix, to_backtest_csv
from noise_strategy_backtest import run_backtest
from polysilicon_futures_noise_backtest import PS_NOISE_STRATEGY_PARAMS

# 默认扫描：螺纹钢、不锈钢、黄金、鸡蛋（新浪代码前缀小写）
DEFAULT_LABELS = {
    "rb": "螺纹钢",
    "ss": "不锈钢",
    "au": "黄金",
    "jd": "鸡蛋",
}


def build_config(csv_path: Path, ticker: str) -> dict:
    peek = pd.read_csv(csv_path, parse_dates=["DateTime"])
    bt_start = peek["DateTime"].dt.date.min()
    bt_end = peek["DateTime"].dt.date.max()
    return {
        "data_path": str(csv_path),
        "ticker": ticker,
        "initial_capital": 2_000_000.0,
        "start_date": bt_start,
        "end_date": bt_end,
        "sigma_incomplete_day_drop": False,
        **PS_NOISE_STRATEGY_PARAMS,
    }


def run_one_symbol(csv_path: Path, ticker: str) -> dict | None:
    cfg = build_config(csv_path, ticker)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            daily_df, _m, trades, metrics = run_backtest(cfg)
    except Exception as e:
        return {"error": str(e)}
    if len(daily_df) < 5:
        return {"error": "有效日度样本过少"}
    tr = int(metrics.get("total_trades", 0))
    return {
        "sharpe": float(metrics.get("sharpe_ratio", 0)),
        "total_return": float(metrics.get("total_return", 0)),
        "mdd": float(metrics.get("mdd", 0)),
        "irr": float(metrics.get("irr", 0)),
        "vol": float(metrics.get("volatility", 0)),
        "trades": tr,
        "days": len(daily_df),
        "bars": len(pd.read_csv(csv_path)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="多品种噪声策略同参数扫描")
    ap.add_argument(
        "--symbols",
        nargs="*",
        default=["rb", "ss", "au", "jd"],
        help="品种代码前缀（小写），如 rb ss au jd",
    )
    ap.add_argument("--skip-fetch", action="store_true", help="不拉取，只读 data/noise_scan/")
    ap.add_argument("--sleep", type=float, default=0.12, help="请求间隔秒")
    args = ap.parse_args()

    data_dir = Path(__file__).resolve().parent / "data" / "noise_scan"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for prefix in args.symbols:
        prefix = prefix.strip().lower()
        name = DEFAULT_LABELS.get(prefix, prefix.upper())
        csv_path = data_dir / f"{prefix}_real_minute_1m.csv"

        if args.skip_fetch:
            if not csv_path.is_file():
                print(f"[跳过] {prefix} 无缓存: {csv_path}")
                continue
            print(f"[缓存] 使用 {csv_path}")
        else:
            try:
                raw = fetch_disjoint_for_prefix(prefix, period="1", sleep_s=args.sleep, verbose=True)
                clean = to_backtest_csv(raw)
                clean.to_csv(csv_path, index=False)
            except Exception as e:
                rows.append(
                    {
                        "品种": name,
                        "代码": prefix,
                        "夏普": float("nan"),
                        "总回报%": float("nan"),
                        "最大回撤%": float("nan"),
                        "成交笔数": 0,
                        "分钟K": 0,
                        "备注": f"拉取失败: {e}",
                    }
                )
                continue

        ticker = f"{prefix.upper()}_noise_scan"
        res = run_one_symbol(csv_path, ticker)
        if res is None or "error" in res:
            msg = res.get("error", "unknown") if res else "unknown"
            rows.append(
                {
                    "品种": name,
                    "代码": prefix,
                    "夏普": float("nan"),
                    "总回报%": float("nan"),
                    "最大回撤%": float("nan"),
                    "成交笔数": 0,
                    "分钟K": 0,
                    "备注": f"回测失败: {msg}",
                }
            )
            continue

        rows.append(
            {
                "品种": name,
                "代码": prefix,
                "夏普": res["sharpe"],
                "总回报%": res["total_return"] * 100,
                "最大回撤%": res["mdd"] * 100,
                "成交笔数": res["trades"],
                "分钟K": res["bars"],
                "备注": "",
            }
        )

    if not rows:
        print("无结果")
        return

    out = pd.DataFrame(rows)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print("\n=== 同参数（PS_NOISE_STRATEGY_PARAMS）多品种对比 ===\n")
    print(out.to_string(index=False))
    out_path = data_dir / "noise_multi_symbol_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
