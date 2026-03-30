"""Macro correlation: DXY bias for gold signal alignment."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from typing import Tuple


@st.cache_data(ttl=900, show_spinner=False)
def _dxy_bias(dxy_df_json: str) -> str:
    try:
        import json
        data = json.loads(dxy_df_json)
        closes = data["close"]
        if len(closes) < 10:
            return "neutral"
        s = pd.Series(closes)
        fast = s.ewm(span=5).mean().iloc[-1]
        slow = s.ewm(span=20).mean().iloc[-1]
        if fast > slow * 1.001: return "bullish"
        if fast < slow * 0.999: return "bearish"
        return "neutral"
    except Exception:
        return "neutral"


def get_dxy_bias(dxy_df: pd.DataFrame) -> str:
    if dxy_df is None or dxy_df.empty:
        return "neutral"
    try:
        return _dxy_bias(dxy_df[["close"]].to_json())
    except Exception:
        return "neutral"


def dxy_alignment(bias: str, direction: str) -> Tuple[bool, float]:
    """Returns (aligned, confidence_delta).
    DXY bullish → gold should fall (SELL).
    DXY bearish → gold should rise (BUY).
    """
    if bias == "neutral":
        return True, 0.0
    if direction == "BUY":
        return (True, +5.0) if bias == "bearish" else (False, -10.0)
    if direction == "SELL":
        return (True, +5.0) if bias == "bullish" else (False, -10.0)
    return True, 0.0
