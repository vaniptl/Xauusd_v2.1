"""Risk manager — position sizing, circuit breakers, daily pip target."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional, Tuple
from core.strategies import Signal


@dataclass
class RiskState:
    equity: float = 10000.0
    initial_equity: float = 10000.0
    daily_pnl_usd: float = 0.0
    daily_pips: float = 0.0
    open_trades: int = 0
    consecutive_losses: int = 0
    paused: bool = False
    pause_reason: str = ""
    daily_date: date = field(default_factory=date.today)
    pip_target: float = 200.0
    max_risk_pct: float = 1.5
    max_concurrent: int = 2
    daily_dd_limit: float = 8.0
    equity_floor_pct: float = 60.0
    max_consec_losses: int = 4


class RiskManager:
    def __init__(self, state: RiskState):
        self.s = state

    def can_trade(self) -> Tuple[bool, str]:
        self._day_reset()
        if self.s.paused:
            return False, f"⏸ Paused: {self.s.pause_reason}"
        if self.s.open_trades >= self.s.max_concurrent:
            return False, f"Max {self.s.max_concurrent} concurrent trades reached"
        if self.s.equity > 0:
            dd = (self.s.daily_pnl_usd / self.s.equity) * 100
            if dd <= -self.s.daily_dd_limit:
                self._pause(f"Daily drawdown -{self.s.daily_dd_limit}% hit")
                return False, self.s.pause_reason
        if self.s.daily_pips >= self.s.pip_target:
            return False, f"✅ Daily pip target {self.s.pip_target:.0f} reached"
        floor = self.s.initial_equity * (self.s.equity_floor_pct / 100)
        if self.s.equity < floor:
            self._pause(f"Equity below {self.s.equity_floor_pct:.0f}% floor")
            return False, self.s.pause_reason
        return True, "ok"

    def lot_size(self, signal: Signal) -> float:
        risk_amount = self.s.equity * (self.s.max_risk_pct / 100)
        sl_pips = abs(signal.entry - signal.sl) / 0.1
        if sl_pips <= 0:
            return 0.01
        lot = risk_amount / (sl_pips * 1.0)
        return max(0.01, min(5.0, round(lot, 4)))

    def on_open(self):
        self.s.open_trades += 1

    def on_close(self, pnl_pips: float, pnl_usd: float):
        self.s.open_trades = max(0, self.s.open_trades - 1)
        self.s.daily_pips    += pnl_pips
        self.s.daily_pnl_usd += pnl_usd
        self.s.equity        += pnl_usd
        if pnl_usd < 0:
            self.s.consecutive_losses += 1
            if self.s.consecutive_losses >= self.s.max_consec_losses:
                self._pause(f"{self.s.consecutive_losses} consecutive losses — review required")
        else:
            self.s.consecutive_losses = 0

    def resume(self):
        self.s.paused        = False
        self.s.pause_reason  = ""
        self.s.consecutive_losses = 0

    def _pause(self, reason: str):
        self.s.paused       = True
        self.s.pause_reason = reason

    def _day_reset(self):
        today = date.today()
        if today != self.s.daily_date:
            self.s.daily_pnl_usd = 0.0
            self.s.daily_pips    = 0.0
            self.s.daily_date    = today
            if "drawdown" in self.s.pause_reason or "consecutive" in self.s.pause_reason:
                self.resume()

    @property
    def daily_target_pct(self) -> float:
        return min(100.0, (self.s.daily_pips / max(self.s.pip_target, 1)) * 100)

    @property
    def equity_change_pct(self) -> float:
        return ((self.s.equity - self.s.initial_equity) / max(self.s.initial_equity, 1)) * 100
