"""Walk-forward optimizer — adjusts strategy weights every 4h."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Dict, List
import streamlit as st


DEFAULT_WEIGHTS = {
    "liquidity_sweep": 1.0,
    "trend_continuation": 1.0,
    "breakout_expansion": 1.0,
    "ema_momentum": 1.0,
    "smc_concepts": 1.0,
}


def should_run(last_run: datetime | None, interval_hours: int = 4) -> bool:
    if last_run is None:
        return True
    return (datetime.now(timezone.utc) - last_run).total_seconds() >= interval_hours * 3600


def run_optimizer(trades: List[dict], weights: Dict[str, float]) -> Dict[str, float]:
    """Evaluate closed trade performance per strategy and adjust weights."""
    if not trades:
        return dict(weights)

    by_strategy: Dict[str, List[dict]] = {}
    for t in trades:
        if t.get("status") in ("open", "pending"):
            continue
        s = t.get("strategy", "unknown")
        by_strategy.setdefault(s, []).append(t)

    new_weights = dict(weights)
    for strat, strat_trades in by_strategy.items():
        if strat not in new_weights:
            continue
        if len(strat_trades) < 10:
            continue
        wins   = [t for t in strat_trades if (t.get("pnl_usd") or 0) > 0]
        losses = [t for t in strat_trades if (t.get("pnl_usd") or 0) < 0]
        wr     = len(wins) / len(strat_trades) * 100
        gp     = sum(t.get("pnl_usd", 0) for t in wins)
        gl     = abs(sum(t.get("pnl_usd", 0) for t in losses))
        pf     = gp / max(gl, 0.01)
        old_w  = new_weights[strat]

        if pf > 1.5 and wr > 55:   adj = +0.10
        elif pf > 1.2 and wr > 45: adj = +0.05
        elif pf < 0.8 or wr < 35:  adj = -0.15
        elif pf < 1.0 or wr < 40:  adj = -0.08
        else:                       adj =  0.00

        new_weights[strat] = round(max(0.3, min(1.8, old_w + adj)), 3)

    return new_weights
