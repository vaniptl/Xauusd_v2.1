"""Support & Resistance — price-respects-level detection for H1 + M15."""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from core.indicators import Indicators


@dataclass
class SRLevel:
    price: float
    kind: str           # "support" | "resistance"
    touches: int
    strength: float     # 0-100
    timeframe: str
    last_touch: datetime


def find_sr_levels(df: pd.DataFrame, timeframe: str,
                   min_touches: int = 3, tolerance_pct: float = 0.003,
                   lookback: int = 120) -> List[SRLevel]:
    """Cluster pivot highs/lows into S/R zones."""
    if df is None or len(df) < 30:
        return []
    df = df.tail(lookback).copy()
    tol = df["close"].mean() * tolerance_pct

    ph = Indicators.pivot_highs(df, 5).dropna()
    pl = Indicators.pivot_lows(df, 5).dropna()

    levels: List[SRLevel] = []
    n = len(df)

    for kind, pivots in [("resistance", ph), ("support", pl)]:
        if pivots.empty:
            continue
        clusters: list = []
        for ts, price in pivots.items():
            merged = False
            for cl in clusters:
                if abs(cl["price"] - price) <= tol:
                    cl["prices"].append(price)
                    cl["indices"].append(df.index.get_loc(ts) if ts in df.index else n - 1)
                    cl["last_ts"] = ts
                    merged = True
                    break
            if not merged:
                clusters.append({"price": price, "prices": [price],
                                  "indices": [df.index.get_loc(ts) if ts in df.index else n - 1],
                                  "last_ts": ts})

        for cl in clusters:
            if len(cl["prices"]) < min_touches:
                continue
            avg_price = float(np.mean(cl["prices"]))
            recency   = max(cl["indices"]) / n
            strength  = min(100.0, len(cl["prices"]) * 15 + recency * 25)
            levels.append(SRLevel(
                price=round(avg_price, 2),
                kind=kind,
                touches=len(cl["prices"]),
                strength=round(strength, 1),
                timeframe=timeframe,
                last_touch=cl["last_ts"],
            ))

    return sorted(levels, key=lambda x: x.strength, reverse=True)


def combine_sr(df_h1: pd.DataFrame, df_m15: pd.DataFrame) -> List[SRLevel]:
    """Merge H1 + M15 levels, deduplicate by proximity."""
    h1  = find_sr_levels(df_h1,  "1h",  min_touches=3)
    m15 = find_sr_levels(df_m15, "15m", min_touches=3)
    all_levels = h1 + m15
    if not all_levels:
        return []

    tol = df_h1["close"].mean() * 0.004
    merged: List[SRLevel] = []
    used = set()
    for i, a in enumerate(all_levels):
        if i in used:
            continue
        group = [a]
        for j, b in enumerate(all_levels):
            if j <= i or j in used:
                continue
            if abs(a.price - b.price) <= tol:
                group.append(b)
                used.add(j)
        used.add(i)
        best = max(group, key=lambda x: x.strength)
        best.touches = sum(g.touches for g in group)
        best.strength = min(100.0, best.strength + (len(group) - 1) * 8)
        merged.append(best)

    return sorted(merged, key=lambda x: x.strength, reverse=True)[:20]
