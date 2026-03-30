"""
All 5 trading strategies.
Each returns a Signal dict or None.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict
from core.smc_engine import SMCData, SMCEngine
from core.sr_engine import SRLevel
from core.regime import Regime, Session, session_weight, REGIME_STRATEGIES

_smc = SMCEngine()

MIN_CONFIDENCE = 65.0


@dataclass
class Signal:
    strategy: str
    direction: str          # BUY | SELL
    confidence: float
    entry: float
    sl: float
    tp1: float
    tp2: float
    atr: float
    timeframe: str
    session: str
    regime: str
    rr1: float = 0.0
    rr2: float = 0.0
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        risk = abs(self.entry - self.sl)
        if risk > 0:
            self.rr1 = round(abs(self.tp1 - self.entry) / risk, 2)
            self.rr2 = round(abs(self.tp2 - self.entry) / risk, 2)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "direction": self.direction,
            "confidence": round(self.confidence, 1),
            "entry": round(self.entry, 2), "sl": round(self.sl, 2),
            "tp1": round(self.tp1, 2), "tp2": round(self.tp2, 2),
            "atr": round(self.atr, 2), "timeframe": self.timeframe,
            "session": self.session, "regime": self.regime,
            "rr1": self.rr1, "rr2": self.rr2,
            "notes": self.notes, "timestamp": self.timestamp,
        }


# ── helpers ─────────────────────────────────────────────────────────

def _sl_tp(price: float, atr: float, direction: str,
           sl_mult=1.5, tp1_mult=1.0, tp2_mult=2.5):
    d = 1 if direction == "BUY" else -1
    return (price - d * atr * sl_mult,
            price + d * atr * tp1_mult,
            price + d * atr * tp2_mult)


# ── Strategy 1: EMA Momentum ────────────────────────────────────────

def strategy_ema_momentum(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                           price: float, atr: float,
                           regime: Regime, session: Session) -> Optional[Signal]:
    try:
        h = df_h1.iloc[-1]
        bull_h1 = h["ema9"] > h["ema21"] > h["ema50"]
        bear_h1 = h["ema9"] < h["ema21"] < h["ema50"]
        if not (bull_h1 or bear_h1):
            return None
        if h["atr"] < h.get("atr_avg", h["atr"]) * 0.7:
            return None

        if df_m15 is None or len(df_m15) < 3:
            return None
        m = df_m15.iloc[-1]
        mp = df_m15.iloc[-2]

        bull_x = mp["ema9"] < mp["ema21"] and m["ema9"] > m["ema21"]
        bear_x = mp["ema9"] > mp["ema21"] and m["ema9"] < m["ema21"]

        direction = conf = None
        if bull_h1 and bull_x and 45 <= m["rsi"] <= 70:
            direction, conf = "BUY",  60 + (m["rsi"] - 45) / 25 * 12
        elif bear_h1 and bear_x and 30 <= m["rsi"] <= 55:
            direction, conf = "SELL", 60 + (55 - m["rsi"]) / 25 * 12

        if direction is None:
            return None
        if h.get("adx", 0) > 25: conf += 10
        if h.get("adx", 0) > 35: conf +=  5
        sl, tp1, tp2 = _sl_tp(price, atr, direction)
        return Signal("ema_momentum", direction, min(95, conf), price, sl, tp1, tp2,
                      atr, "15m", session.value, regime.value,
                      notes="EMA 9/21 cross M15 + H1 trend aligned")
    except Exception:
        return None


# ── Strategy 2: Trend Continuation (OB / FVG pullback) ─────────────

def strategy_trend_continuation(df_h1: pd.DataFrame, price: float, atr: float,
                                  smc: SMCData, sr_levels: List[SRLevel],
                                  regime: Regime, session: Session) -> Optional[Signal]:
    try:
        if smc.bias == "neutral":
            return None
        direction = "BUY" if smc.bias == "bullish" else "SELL"
        conf  = 55.0
        notes = []

        ob  = _smc.in_ob(price, smc.order_blocks, "buy" if direction == "BUY" else "sell")
        fvg = _smc.in_fvg(price, smc.fvgs, "buy" if direction == "BUY" else "sell")

        if ob:  conf += 15; notes.append(f"OB {ob['low']:.1f}-{ob['high']:.1f}")
        if fvg: conf += 10; notes.append(f"FVG {fvg['bottom']:.1f}-{fvg['top']:.1f}")
        if not (ob or fvg):
            return None
        if smc.bos   and smc.bos_dir   == smc.bias: conf += 10; notes.append("BOS ✓")
        if smc.choch and smc.choch_dir == smc.bias: conf +=  8; notes.append("CHoCH ✓")

        sl, tp1, tp2 = _sl_tp(price, atr, direction)
        return Signal("trend_continuation", direction, min(95, conf), price, sl, tp1, tp2,
                      atr, "1h", session.value, regime.value, notes=" | ".join(notes))
    except Exception:
        return None


# ── Strategy 3: Liquidity Sweep ─────────────────────────────────────

def strategy_liquidity_sweep(df_h1: pd.DataFrame, price: float, atr: float,
                               smc: SMCData, sr_levels: List[SRLevel],
                               regime: Regime, session: Session) -> Optional[Signal]:
    try:
        if len(df_h1) < 5:
            return None
        last, prev = df_h1.iloc[-1], df_h1.iloc[-2]
        sweep_thr  = atr * 0.3
        direction  = None
        conf       = 58.0
        notes      = []

        # SMC liquidity levels
        for liq in smc.liquidity:
            lp = liq["price"]
            if liq["type"] == "sell_side" and prev["low"] < lp - sweep_thr and last["close"] > lp and last["close"] > last["open"]:
                direction = "BUY"; conf += 20; notes.append(f"Sell-side liq sweep {lp:.1f}"); break
            if liq["type"] == "buy_side"  and prev["high"] > lp + sweep_thr and last["close"] < lp and last["close"] < last["open"]:
                direction = "SELL"; conf += 20; notes.append(f"Buy-side liq sweep {lp:.1f}"); break

        # S/R sweep fallback
        if direction is None:
            for lvl in sr_levels[:5]:
                lp = lvl.price
                if lvl.kind == "support" and prev["low"] < lp * 0.999 and last["close"] > lp and last["close"] > last["open"]:
                    direction = "BUY"; conf += 15 + lvl.strength * 0.1; notes.append(f"S/R sweep {lp:.1f}"); break
                if lvl.kind == "resistance" and prev["high"] > lp * 1.001 and last["close"] < lp and last["close"] < last["open"]:
                    direction = "SELL"; conf += 15 + lvl.strength * 0.1; notes.append(f"S/R sweep {lp:.1f}"); break

        if direction is None:
            return None

        vol_avg = float(last.get("vol_avg", last.get("volume", 1)) or 1)
        if vol_avg > 0 and last.get("volume", 0) > vol_avg * 1.3:
            conf += 10; notes.append("vol spike ✓")

        if direction == "BUY":
            sl  = min(prev["low"], last["low"]) - atr * 0.3
            tp1 = price + atr * 1.0
            tp2 = price + atr * 2.5
        else:
            sl  = max(prev["high"], last["high"]) + atr * 0.3
            tp1 = price - atr * 1.0
            tp2 = price - atr * 2.5

        return Signal("liquidity_sweep", direction, min(95, conf), price, sl, tp1, tp2,
                      atr, "1h", session.value, regime.value, notes=" | ".join(notes))
    except Exception:
        return None


# ── Strategy 4: Breakout Expansion ──────────────────────────────────

def strategy_breakout(df_h1: pd.DataFrame, price: float, atr: float,
                       sr_levels: List[SRLevel],
                       regime: Regime, session: Session) -> Optional[Signal]:
    try:
        if len(df_h1) < 5:
            return None
        last, prev = df_h1.iloc[-1], df_h1.iloc[-2]
        vol_avg  = float(last.get("vol_avg", last.get("volume", 1)) or 1)
        retest   = atr * 0.3
        direction = None; conf = 55.0; notes = []

        for lvl in sr_levels[:8]:
            lp = lvl.price
            if lvl.kind == "resistance" and prev["close"] < lp and last["close"] > lp * 1.001:
                if vol_avg > 0 and last.get("volume", 0) >= vol_avg * 1.5:
                    direction = "BUY"; conf += lvl.strength * 0.2 + 12; notes.append(f"Break {lp:.1f} + vol"); break
                if last["close"] > lp and abs(last["low"] - lp) < retest:
                    direction = "BUY"; conf += lvl.strength * 0.2 + 8;  notes.append(f"Retest {lp:.1f}"); break
            if lvl.kind == "support" and prev["close"] > lp and last["close"] < lp * 0.999:
                if vol_avg > 0 and last.get("volume", 0) >= vol_avg * 1.5:
                    direction = "SELL"; conf += lvl.strength * 0.2 + 12; notes.append(f"Break {lp:.1f} + vol"); break
                if last["close"] < lp and abs(last["high"] - lp) < retest:
                    direction = "SELL"; conf += lvl.strength * 0.2 + 8;  notes.append(f"Retest {lp:.1f}"); break

        if direction is None:
            return None

        atr_avg = float(last.get("atr_avg", atr) or atr)
        if atr_avg > 0 and atr > atr_avg * 1.2:
            conf += 8; notes.append("ATR expanding ✓")

        sl, tp1, tp2 = _sl_tp(price, atr, direction)
        return Signal("breakout_expansion", direction, min(95, conf), price, sl, tp1, tp2,
                      atr, "1h", session.value, regime.value, notes=" | ".join(notes))
    except Exception:
        return None


# ── Strategy 5: SMC Full Confluence ─────────────────────────────────

def strategy_smc(df_h1: pd.DataFrame, price: float, atr: float,
                  smc_h1: SMCData, smc_m15: SMCData,
                  sr_levels: List[SRLevel],
                  regime: Regime, session: Session) -> Optional[Signal]:
    try:
        if not (smc_h1.choch or smc_h1.bos):
            return None
        if smc_h1.bias == "neutral":
            return None

        direction = "BUY" if smc_h1.bias == "bullish" else "SELL"
        d_key     = "buy" if direction == "BUY" else "sell"
        score     = 0; notes = []

        if smc_h1.choch:                                    score += 2; notes.append("CHoCH ✓")
        if smc_h1.bos:                                      score += 2; notes.append("BOS ✓")
        ob  = _smc.in_ob(price, smc_h1.order_blocks, d_key)
        fvg = _smc.in_fvg(price, smc_h1.fvgs, d_key)
        if ob:                                              score += 3; notes.append(f"OB {ob['low']:.1f}")
        if fvg:                                             score += 2; notes.append(f"FVG {fvg['bottom']:.1f}")
        liq_type = "sell_side" if direction == "BUY" else "buy_side"
        swept = any(abs(l["price"] - price) < atr for l in smc_h1.liquidity if l["type"] == liq_type)
        if swept:                                           score += 2; notes.append("Liq swept ✓")
        if smc_m15.bias == smc_h1.bias:                    score += 1; notes.append("M15 aligned ✓")

        if score < 4:
            return None

        conf = 50 + score * 5
        sl, tp1, tp2 = _sl_tp(price, atr, direction, tp2_mult=3.0)
        return Signal("smc_concepts", direction, min(95, conf), price, sl, tp1, tp2,
                      atr, "1h", session.value, regime.value, notes=" | ".join(notes))
    except Exception:
        return None


# ── Main evaluator ───────────────────────────────────────────────────

def evaluate_all(df_m1, df_m15, df_h1, df_h4,
                 smc_h1: SMCData, smc_m15: SMCData,
                 sr_levels: List[SRLevel],
                 regime: Regime, session: Session,
                 strategy_weights: dict) -> List[Signal]:
    if df_h1 is None or len(df_h1) < 50:
        return []

    price = float(df_h1["close"].iloc[-1])
    atr   = float(df_h1["atr"].iloc[-1])
    allowed = REGIME_STRATEGIES.get(regime, [])

    runners = [
        ("ema_momentum",      lambda: strategy_ema_momentum(df_h1, df_m15, price, atr, regime, session)),
        ("trend_continuation",lambda: strategy_trend_continuation(df_h1, price, atr, smc_h1, sr_levels, regime, session)),
        ("liquidity_sweep",   lambda: strategy_liquidity_sweep(df_h1, price, atr, smc_h1, sr_levels, regime, session)),
        ("breakout_expansion",lambda: strategy_breakout(df_h1, price, atr, sr_levels, regime, session)),
        ("smc_concepts",      lambda: strategy_smc(df_h1, price, atr, smc_h1, smc_m15, sr_levels, regime, session)),
    ]

    signals: List[Signal] = []
    for name, fn in runners:
        if name not in allowed:
            continue
        sig = fn()
        if sig is None:
            continue
        # Apply session weight
        w = session_weight(session, name)
        sig.confidence = min(100, sig.confidence * w)
        # Apply optimizer weight
        ow = strategy_weights.get(name, 1.0)
        sig.confidence = min(100, sig.confidence * ow)
        if sig.confidence >= MIN_CONFIDENCE:
            signals.append(sig)

    signals.sort(key=lambda x: x.confidence, reverse=True)
    return signals
