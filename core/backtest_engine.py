"""Bar-by-bar backtest engine — returns results including equity curve."""
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import yfinance as yf
import streamlit as st

from core.indicators import Indicators

SYMBOL = "GC=F"


@st.cache_data(ttl=3600, show_spinner=False)
def run_backtest_period(days: int, timeframe: str = "1h") -> dict:
    """Run a single-period backtest. Cached for 1 hour."""
    period = f"{min(days + 10, 729)}d"
    interval = "1h" if days <= 360 else "1d"
    try:
        ticker = yf.Ticker(SYMBOL)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df[["open","high","low","close","volume"]].dropna()
    except Exception as e:
        return {"error": str(e), "total_trades": 0, "passed": False, "equity_curve": []}

    if len(df) < 100:
        return {"error": "Insufficient data", "total_trades": 0, "passed": False, "equity_curve": []}

    df_e = Indicators.enrich(df)
    if df_e.empty:
        return {"error": "Enrich failed", "total_trades": 0, "passed": False, "equity_curve": []}

    # ── Simple EMA confluence backtest ──────────────────────────────
    trades       = []
    initial_eq   = 10000.0
    equity       = initial_eq
    equity_curve = [initial_eq]
    open_trade   = None

    atr_sl = 1.5
    atr_tp2 = 2.5

    for i in range(50, len(df_e) - 1):
        bar  = df_e.iloc[i]
        nbar = df_e.iloc[i + 1]

        price = float(bar["close"])
        atr   = float(bar.get("atr", 10) or 10)
        ema9  = float(bar.get("ema9",  price) or price)
        ema21 = float(bar.get("ema21", price) or price)
        ema50 = float(bar.get("ema50", price) or price)
        rsi   = float(bar.get("rsi",   50)    or 50)
        adx   = float(bar.get("adx",   0)     or 0)

        # Check open trade exit
        if open_trade:
            exit_p = float(nbar["close"])
            if open_trade["dir"] == "long":
                if exit_p <= open_trade["sl"]:
                    pips = (exit_p - open_trade["entry"]) / 0.1
                    pnl  = pips * open_trade["lot"]
                    equity += pnl
                    trades.append({"pips": pips, "pnl": pnl, "status": "sl"})
                    equity_curve.append(equity)
                    open_trade = None
                elif exit_p >= open_trade["tp"]:
                    pips = (exit_p - open_trade["entry"]) / 0.1
                    pnl  = pips * open_trade["lot"]
                    equity += pnl
                    trades.append({"pips": pips, "pnl": pnl, "status": "tp"})
                    equity_curve.append(equity)
                    open_trade = None
            else:
                if exit_p >= open_trade["sl"]:
                    pips = (open_trade["entry"] - exit_p) / 0.1
                    pnl  = pips * open_trade["lot"]
                    equity += pnl
                    trades.append({"pips": pips, "pnl": pnl, "status": "sl"})
                    equity_curve.append(equity)
                    open_trade = None
                elif exit_p <= open_trade["tp"]:
                    pips = (open_trade["entry"] - exit_p) / 0.1
                    pnl  = pips * open_trade["lot"]
                    equity += pnl
                    trades.append({"pips": pips, "pnl": pnl, "status": "tp"})
                    equity_curve.append(equity)
                    open_trade = None
            if open_trade:
                continue

        # Entry signal
        direction = None
        if ema9 > ema21 > ema50 and 45 <= rsi <= 70 and adx > 20:
            direction = "long"
        elif ema9 < ema21 < ema50 and 30 <= rsi <= 55 and adx > 20:
            direction = "short"

        if direction and equity > 0:
            sl_pips = atr * atr_sl / 0.1
            lot     = round((equity * 0.015) / max(sl_pips, 1), 4)
            lot     = max(0.01, min(1.0, lot))
            if direction == "long":
                open_trade = {"entry": price, "dir": "long",
                              "sl": price - atr*atr_sl, "tp": price + atr*atr_tp2, "lot": lot}
            else:
                open_trade = {"entry": price, "dir": "short",
                              "sl": price + atr*atr_sl, "tp": price - atr*atr_tp2, "lot": lot}

    return _metrics(trades, initial_eq, equity, equity_curve, days)


def _metrics(trades, initial_eq, final_eq, equity_curve, days) -> dict:
    if not trades:
        return {"total_trades":0,"win_rate_pct":0,"profit_factor":0,
                "total_pips":0,"max_drawdown_pct":0,"total_return_pct":0,
                "equity_curve":equity_curve,"passed":False,"days":days}

    wins   = [t for t in trades if t["pips"] > 0]
    losses = [t for t in trades if t["pips"] <= 0]
    tp     = sum(t["pips"] for t in trades)
    gp     = sum(t["pnl"]  for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    pf     = round(gp / max(gl, 0.01), 2)
    wr     = round(len(wins) / len(trades) * 100, 1)
    ret    = round((final_eq - initial_eq) / initial_eq * 100, 2)

    peak   = initial_eq
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd

    passed = pf >= 1.3 and max_dd <= 15.0
    return {
        "total_trades":    len(trades),
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate_pct":    wr,
        "total_pips":      round(tp, 1),
        "profit_factor":   pf,
        "max_drawdown_pct": round(max_dd, 2),
        "total_return_pct": ret,
        "final_equity":    round(final_eq, 2),
        "equity_curve":    equity_curve,
        "passed":          passed,
        "days":            days,
    }
