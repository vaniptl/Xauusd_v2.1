"""
DataEngine — yfinance multi-timeframe fetcher.
Free, no API key, no rate limits enforced.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from typing import Dict, Optional
import streamlit as st


SYMBOL     = "GC=F"      # Gold Futures
DXY_SYMBOL = "DX-Y.NYB"
SP5_SYMBOL = "^GSPC"

YF_INTERVAL = {"1m":"1m","15m":"15m","1h":"1h","4h":"1h","1d":"1d"}
YF_PERIOD   = {"1m":"7d","15m":"60d","1h":"730d","4h":"730d","1d":"10y"}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Cached yfinance fetch — refreshes every 60 seconds."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "timestamp"
        for col in ["open","high","low","close","volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close"], inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()


class DataEngine:
    """Multi-timeframe data manager with 4h resampling."""

    def get(self, timeframe: str, limit: int = 300) -> pd.DataFrame:
        interval = YF_INTERVAL.get(timeframe, "1h")
        period   = YF_PERIOD.get(timeframe, "60d")
        df = fetch_candles(SYMBOL, interval, period)
        if df.empty:
            return df
        if timeframe == "4h":
            df = self._resample_4h(df)
        return df.tail(limit)

    def current_price(self) -> float:
        df = self.get("1m", 3)
        if not df.empty:
            return float(df["close"].iloc[-1])
        return 0.0

    def get_dxy(self, limit: int = 50) -> pd.DataFrame:
        df = fetch_candles(DXY_SYMBOL, "1h", "60d")
        return df.tail(limit) if not df.empty else pd.DataFrame()

    def get_sp500(self, limit: int = 50) -> pd.DataFrame:
        df = fetch_candles(SP5_SYMBOL, "1h", "60d")
        return df.tail(limit) if not df.empty else pd.DataFrame()

    def _resample_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = {c: ("first" if c == "open" else "max" if c == "high"
                    else "min" if c == "low" else "last" if c == "close" else "sum")
                for c in ["open","high","low","close","volume"] if c in df.columns}
        return df[list(cols.keys())].resample("4h").agg(cols).dropna()
