"""Smart Money Concepts — CHoCH, BOS, OB, FVG, Liquidity."""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class SMCData:
    bias: str = "neutral"           # bullish | bearish | neutral
    choch: bool = False
    choch_dir: str = ""
    bos: bool = False
    bos_dir: str = ""
    order_blocks: List[Dict] = field(default_factory=list)
    fvgs: List[Dict] = field(default_factory=list)
    liquidity: List[Dict] = field(default_factory=list)
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None


class SMCEngine:
    def analyze(self, df: pd.DataFrame) -> SMCData:
        if df is None or len(df) < 30:
            return SMCData()
        try:
            return self._run(df)
        except Exception:
            return SMCData()

    def _run(self, df: pd.DataFrame) -> SMCData:
        data = SMCData()
        sw_h = self._swings(df, "high")
        sw_l = self._swings(df, "low")

        if sw_h: data.last_swing_high = sw_h[-1]["price"]
        if sw_l: data.last_swing_low  = sw_l[-1]["price"]

        choch, bos = self._structure(df, sw_h, sw_l)
        data.choch, data.choch_dir = bool(choch), choch or ""
        data.bos,   data.bos_dir   = bool(bos),   bos   or ""

        if bos:   data.bias = bos
        elif choch: data.bias = choch
        else: data.bias = self._ema_bias(df)

        data.order_blocks = self._obs(df)
        data.fvgs         = self._fvgs(df)
        data.liquidity    = self._liquidity(sw_h, sw_l)
        return data

    # ── swings ──────────────────────────────────────────────────
    def _swings(self, df: pd.DataFrame, col: str, period: int = 10) -> List[Dict]:
        vals = df[col].values
        out  = []
        for i in range(period, len(vals) - period):
            w = vals[i - period: i + period + 1]
            if col == "high" and vals[i] == np.max(w):
                out.append({"price": vals[i], "idx": i, "ts": df.index[i]})
            elif col == "low" and vals[i] == np.min(w):
                out.append({"price": vals[i], "idx": i, "ts": df.index[i]})
        return out

    # ── structure breaks ────────────────────────────────────────
    def _structure(self, df, sw_h, sw_l):
        if len(sw_h) < 2 or len(sw_l) < 2:
            return None, None
        price  = float(df["close"].iloc[-1])
        lsh, psh = sw_h[-1]["price"], sw_h[-2]["price"]
        lsl, psl = sw_l[-1]["price"], sw_l[-2]["price"]
        choch = bos = None
        if price > lsh > psh:      bos   = "bullish"
        elif price < lsl < psl:    bos   = "bearish"
        elif price > lsh and lsl < psl: choch = "bullish"
        elif price < lsl and lsh > psh: choch = "bearish"
        return choch, bos

    # ── order blocks ────────────────────────────────────────────
    def _obs(self, df: pd.DataFrame) -> List[Dict]:
        obs = []
        d = df.tail(40)
        o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
        for i in range(2, len(d) - 3):
            impl = abs(c[i+2] - c[i]) / max(abs(c[i]), 1)
            if o[i] > c[i] and c[i+1] > o[i] and c[i+2] > c[i+1]:   # bullish ob
                obs.append({"type":"bullish_ob","high":h[i],"low":l[i],
                             "mid":(h[i]+l[i])/2,"strength":min(100,impl*1000),
                             "ts":str(d.index[i])})
            if c[i] > o[i] and c[i+1] < o[i] and c[i+2] < c[i+1]:   # bearish ob
                obs.append({"type":"bearish_ob","high":h[i],"low":l[i],
                             "mid":(h[i]+l[i])/2,"strength":min(100,impl*1000),
                             "ts":str(d.index[i])})
        obs.sort(key=lambda x: x["strength"], reverse=True)
        return obs[:6]

    # ── fair value gaps ─────────────────────────────────────────
    def _fvgs(self, df: pd.DataFrame) -> List[Dict]:
        fvgs = []
        d = df.tail(60)
        h, l = d["high"].values, d["low"].values
        min_gap = d["close"].mean() * 0.001
        for i in range(len(d) - 2):
            gap_b = l[i+2] - h[i]
            if gap_b >= min_gap:
                fvgs.append({"type":"bullish_fvg","top":l[i+2],"bottom":h[i],
                              "mid":(l[i+2]+h[i])/2,"pips":round(gap_b/0.1,1),"ts":str(d.index[i])})
            gap_s = h[i] - l[i+2]
            if gap_s >= min_gap:
                fvgs.append({"type":"bearish_fvg","top":h[i],"bottom":l[i+2],
                              "mid":(h[i]+l[i+2])/2,"pips":round(gap_s/0.1,1),"ts":str(d.index[i])})
        return fvgs[-8:]

    # ── liquidity ───────────────────────────────────────────────
    def _liquidity(self, sw_h: List[Dict], sw_l: List[Dict]) -> List[Dict]:
        liq = []
        tol = 0.5
        for i in range(len(sw_h) - 1):
            if abs(sw_h[i]["price"] - sw_h[i+1]["price"]) <= tol:
                liq.append({"type":"buy_side","price":(sw_h[i]["price"]+sw_h[i+1]["price"])/2})
        for i in range(len(sw_l) - 1):
            if abs(sw_l[i]["price"] - sw_l[i+1]["price"]) <= tol:
                liq.append({"type":"sell_side","price":(sw_l[i]["price"]+sw_l[i+1]["price"])/2})
        return liq[:8]

    # ── helpers ─────────────────────────────────────────────────
    def _ema_bias(self, df: pd.DataFrame) -> str:
        if "ema20" in df.columns and "ema200" in df.columns:
            last = df.iloc[-1]
            if last["ema20"] > last["ema200"]: return "bullish"
            if last["ema20"] < last["ema200"]: return "bearish"
        return "neutral"

    def in_ob(self, price: float, obs: List[Dict], direction: str) -> Optional[Dict]:
        t = "bullish_ob" if direction == "buy" else "bearish_ob"
        return next((o for o in obs if o["type"] == t and o["low"] <= price <= o["high"]), None)

    def in_fvg(self, price: float, fvgs: List[Dict], direction: str) -> Optional[Dict]:
        t = "bullish_fvg" if direction == "buy" else "bearish_fvg"
        return next((f for f in fvgs if f["type"] == t and f["bottom"] <= price <= f["top"]), None)
