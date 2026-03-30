"""Market regime classifier + session detector."""
from __future__ import annotations
import pandas as pd
from datetime import datetime, timezone
from enum import Enum


class Regime(str, Enum):
    TRENDING_BULL  = "trending_bull"
    TRENDING_BEAR  = "trending_bear"
    RANGING        = "ranging"
    HIGH_VOL       = "high_volatility"
    LOW_LIQ        = "low_liquidity"


class Session(str, Enum):
    ASIAN       = "asian"
    LONDON      = "london"
    NEW_YORK    = "new_york"
    OVERLAP     = "lon_ny_overlap"
    OFF_HOURS   = "off_hours"


REGIME_LABEL = {
    Regime.TRENDING_BULL: ("📈 Bull Trend",  "#22c55e"),
    Regime.TRENDING_BEAR: ("📉 Bear Trend",  "#ef4444"),
    Regime.RANGING:       ("↔ Ranging",      "#60a5fa"),
    Regime.HIGH_VOL:      ("⚡ High Vol",     "#f59e0b"),
    Regime.LOW_LIQ:       ("💤 Low Liq",     "#6b7280"),
}

SESSION_LABEL = {
    Session.ASIAN:     ("🌏 Asian",    "#8b5cf6"),
    Session.LONDON:    ("🇬🇧 London",   "#3b82f6"),
    Session.NEW_YORK:  ("🗽 New York",  "#f97316"),
    Session.OVERLAP:   ("🔥 Overlap",   "#ec4899"),
    Session.OFF_HOURS: ("😴 Off Hours", "#6b7280"),
}

# Which strategies are allowed per regime
REGIME_STRATEGIES = {
    Regime.TRENDING_BULL:  ["trend_continuation","ema_momentum","breakout_expansion","smc_concepts"],
    Regime.TRENDING_BEAR:  ["trend_continuation","ema_momentum","breakout_expansion","smc_concepts"],
    Regime.RANGING:        ["liquidity_sweep","smc_concepts","trend_continuation"],
    Regime.HIGH_VOL:       ["liquidity_sweep","breakout_expansion","smc_concepts"],
    Regime.LOW_LIQ:        [],
}

# Session confidence multipliers per strategy
SESSION_WEIGHTS = {
    Session.OVERLAP:   dict(liquidity_sweep=1.5,breakout_expansion=1.5,trend_continuation=1.4,ema_momentum=1.3,smc_concepts=1.4),
    Session.LONDON:    dict(liquidity_sweep=1.3,breakout_expansion=1.4,trend_continuation=1.2,ema_momentum=1.1,smc_concepts=1.3),
    Session.NEW_YORK:  dict(liquidity_sweep=1.2,breakout_expansion=1.1,trend_continuation=1.3,ema_momentum=1.4,smc_concepts=1.2),
    Session.ASIAN:     dict(liquidity_sweep=0.8,breakout_expansion=0.7,trend_continuation=0.9,ema_momentum=0.8,smc_concepts=0.9),
    Session.OFF_HOURS: dict(liquidity_sweep=0.6,breakout_expansion=0.5,trend_continuation=0.7,ema_momentum=0.6,smc_concepts=0.7),
}


def detect_regime(df: pd.DataFrame) -> Regime:
    if df is None or len(df) < 30:
        return Regime.RANGING
    last = df.iloc[-1]

    adx      = float(last.get("adx", 0) or 0)
    atr      = float(last.get("atr", 0) or 0)
    atr_avg  = float(last.get("atr_avg", atr) or atr)
    ema20    = float(last.get("ema20",  last["close"]) or last["close"])
    ema50    = float(last.get("ema50",  last["close"]) or last["close"])
    ema200   = float(last.get("ema200", last["close"]) or last["close"])
    close    = float(last["close"])
    vol      = float(last.get("volume", 1) or 1)
    vol_avg  = float(last.get("vol_avg", vol) or vol)

    hour = datetime.now(timezone.utc).hour
    if (hour >= 22 or hour < 2) and vol_avg > 0 and vol < vol_avg * 0.5:
        return Regime.LOW_LIQ
    if atr_avg > 0 and atr > atr_avg * 1.8:
        return Regime.HIGH_VOL
    if adx > 25:
        if close > ema20 > ema50:  return Regime.TRENDING_BULL
        if close < ema20 < ema50:  return Regime.TRENDING_BEAR
    if adx < 20:
        return Regime.RANGING
    return Regime.TRENDING_BULL if close > ema200 else Regime.TRENDING_BEAR


def detect_session() -> Session:
    h = datetime.now(timezone.utc).hour
    in_lon = 7  <= h < 16
    in_ny  = 13 <= h < 22
    in_as  = 0  <= h < 8
    if in_lon and in_ny: return Session.OVERLAP
    if in_lon:           return Session.LONDON
    if in_ny:            return Session.NEW_YORK
    if in_as:            return Session.ASIAN
    return Session.OFF_HOURS


def session_weight(session: Session, strategy: str) -> float:
    return SESSION_WEIGHTS.get(session, {}).get(strategy, 1.0)
