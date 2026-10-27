#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SS v2：上午全段收益版，交易更多、年化更高但 Sharpe 更低。"""

import ss_sina_recent_validation as ss


ss.SUMMARY_CSV = ss.ROOT_DIR / "data" / "ss_v2_morning_summary.csv"
ss.TRADES_CSV = ss.ROOT_DIR / "data" / "ss_v2_morning_trades.csv"

ss.LOOKBACK_DAYS = 5
ss.CHECK_INTERVAL_MINUTES = 20
ss.K1 = 2.2
ss.K2 = 2.2
ss.MAX_POSITIONS_PER_DAY = 1
ss.TRADING_START_TIME = (9, 0)
ss.TRADING_END_TIME = (11, 30)
ss.TRADING_SESSIONS = (
    ((9, 0), (10, 15)),
    ((10, 30), (11, 30)),
)

ss.ENABLE_TRAILING_TAKE_PROFIT = False
ss.TRAILING_TP_ACTIVATION_PCT = 0.01
ss.TRAILING_TP_CALLBACK_PCT = 0.7
ss.ENABLE_INTRADAY_STOP_LOSS = False
ss.USE_VWAP = False
ss.ENTRY_TREND_FILTER = None

ss.BACKTEST_PARAMS.update(
    {
        "lookback_days": ss.LOOKBACK_DAYS,
        "check_interval_minutes": ss.CHECK_INTERVAL_MINUTES,
        "trading_sessions": ss.TRADING_SESSIONS,
        "trading_start_time": ss.TRADING_START_TIME,
        "trading_end_time": ss.TRADING_END_TIME,
        "max_positions_per_day": ss.MAX_POSITIONS_PER_DAY,
        "K1": ss.K1,
        "K2": ss.K2,
        "use_vwap": ss.USE_VWAP,
        "enable_intraday_stop_loss": ss.ENABLE_INTRADAY_STOP_LOSS,
        "enable_trailing_take_profit": ss.ENABLE_TRAILING_TAKE_PROFIT,
        "trailing_tp_activation_pct": ss.TRAILING_TP_ACTIVATION_PCT,
        "trailing_tp_callback_pct": ss.TRAILING_TP_CALLBACK_PCT,
        "entry_trend_filter": ss.ENTRY_TREND_FILTER,
    }
)


if __name__ == "__main__":
    ss.main()
