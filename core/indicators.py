"""
Pure numpy/pandas technical indicators.
NO pandas-ta — compatible with Python 3.12+ on Streamlit Cloud.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


class Indicators:
    """Stateless indicator calculations — all pure numpy/pandas."""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).mean()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, adjust=False).mean()

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        h, l, c = df["high"], df["low"], df["close"]
        up   = h.diff()
        down = -l.diff()
        plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        atr_s = Indicators.atr(df, period)
        plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(com=period-1, adjust=False).mean() / atr_s.replace(0, 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(com=period-1, adjust=False).mean() / atr_s.replace(0, 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
        adx_val  = dx.ewm(com=period - 1, adjust=False).mean()
        return pd.DataFrame({"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di}, index=df.index)

    @staticmethod
    def macd(close: pd.Series, fast=12, slow=26, signal=9):
        fast_ema   = Indicators.ema(close, fast)
        slow_ema   = Indicators.ema(close, slow)
        macd_line  = fast_ema - slow_ema
        signal_line = Indicators.ema(macd_line, signal)
        histogram   = macd_line - signal_line
        return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": histogram}, index=close.index)

    @staticmethod
    def bollinger(close: pd.Series, period=20, std_mult=2.0) -> pd.DataFrame:
        ma  = close.rolling(period).mean()
        sd  = close.rolling(period).std()
        return pd.DataFrame({"upper": ma + std_mult * sd, "mid": ma, "lower": ma - std_mult * sd}, index=close.index)

    @staticmethod
    def stochastic(df: pd.DataFrame, k=14, d=3) -> pd.DataFrame:
        low_min  = df["low"].rolling(k).min()
        high_max = df["high"].rolling(k).max()
        k_val = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-10)
        d_val = k_val.rolling(d).mean()
        return pd.DataFrame({"k": k_val, "d": d_val}, index=df.index)

    @staticmethod
    def pivot_highs(df: pd.DataFrame, window: int = 5) -> pd.Series:
        highs = df["high"]
        result = pd.Series(np.nan, index=df.index)
        for i in range(window, len(highs) - window):
            w = highs.iloc[i - window: i + window + 1]
            if highs.iloc[i] == w.max():
                result.iloc[i] = highs.iloc[i]
        return result

    @staticmethod
    def pivot_lows(df: pd.DataFrame, window: int = 5) -> pd.Series:
        lows = df["low"]
        result = pd.Series(np.nan, index=df.index)
        for i in range(window, len(lows) - window):
            w = lows.iloc[i - window: i + window + 1]
            if lows.iloc[i] == w.min():
                result.iloc[i] = lows.iloc[i]
        return result

    @staticmethod
    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        """Add all standard indicators to a DataFrame in one pass."""
        df = df.copy()
        c = df["close"]
        df["ema9"]   = Indicators.ema(c, 9)
        df["ema21"]  = Indicators.ema(c, 21)
        df["ema50"]  = Indicators.ema(c, 50)
        df["ema20"]  = Indicators.ema(c, 20)
        df["ema100"] = Indicators.ema(c, 100)
        df["ema200"] = Indicators.ema(c, 200)
        df["rsi"]    = Indicators.rsi(c, 14)
        df["atr"]    = Indicators.atr(df, 14)
        df["atr_avg"]= df["atr"].rolling(20).mean()
        df["vol_avg"]= df["volume"].rolling(20).mean() if "volume" in df.columns else 1
        adx_df = Indicators.adx(df, 14)
        df["adx"]       = adx_df["adx"]
        df["plus_di"]   = adx_df["plus_di"]
        df["minus_di"]  = adx_df["minus_di"]
        macd_df = Indicators.macd(c)
        df["macd"]      = macd_df["macd"]
        df["macd_sig"]  = macd_df["signal"]
        df["macd_hist"] = macd_df["hist"]
        bb = Indicators.bollinger(c)
        df["bb_upper"]  = bb["upper"]
        df["bb_mid"]    = bb["mid"]
        df["bb_lower"]  = bb["lower"]
        df.dropna(inplace=True)
        return df
