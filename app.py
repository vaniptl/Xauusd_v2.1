"""
XAUUSD Trading Bot — Streamlit Cloud Edition v1.1
Bloomberg-style dark terminal UI.

FIXES v1.1:
1. Persistence: RiskState saved to SQLite - survives page reload / sleep
2. Duplicate signals: dedup by strategy+direction+price_bucket per candle
3. Backtest tab: multi-period 1M/3M/6M/1Y with equity curves
4. S/R-anchored SL/TP: regime-aware, snapped to nearest S/R level
5. Open-trade gate: DB count check + 5-min cooldown prevent over-trading
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta, date
import os, sys, time

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="XAUUSD Trading Bot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.data_engine  import DataEngine
from core.indicators   import Indicators
from core.sr_engine    import combine_sr
from core.smc_engine   import SMCEngine, SMCData
from core.regime       import (detect_regime, detect_session,
                                REGIME_LABEL, SESSION_LABEL, Regime, Session)
from core.strategies   import evaluate_all
from core.risk         import RiskManager, RiskState
from core.macro        import get_dxy_bias, dxy_alignment
from core.optimizer    import should_run, run_optimizer, DEFAULT_WEIGHTS
from database.db       import (init_db, save_signal, save_trade, update_trade,
                                get_open_trades, get_trades, get_recent_signals,
                                get_metrics_cached, get_daily_history,
                                save_bot_state, load_bot_state,
                                get_open_trade_count, mark_signal_duplicate)

init_db()

# ── session state ──────────────────────────────────────────────────
def _init_state():
    defaults = {
        "bot_running":      False,
        "dry_run":          True,
        "equity":           10000.0,
        "pip_target":       200.0,
        "risk_pct":         1.5,
        "strategy_weights": dict(DEFAULT_WEIGHTS),
        "optimizer_last":   None,
        "last_signals":     [],
        "last_price":       0.0,
        "cycle_count":      0,
        "errors":           [],
        "risk_state":       None,
        "last_signal_key":  "",
        "last_trade_time":  None,
        "sr_levels_cache":  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.risk_state is None:
        saved = load_bot_state()
        if saved:
            rs = RiskState(
                equity=saved.get("equity", st.session_state.equity),
                initial_equity=saved.get("initial_equity", st.session_state.equity),
                pip_target=saved.get("pip_target", st.session_state.pip_target),
                max_risk_pct=saved.get("risk_pct", st.session_state.risk_pct),
                daily_pnl_usd=saved.get("daily_pnl_usd", 0.0),
                daily_pips=saved.get("daily_pips", 0.0),
                daily_date=date.fromisoformat(saved.get("daily_date", str(date.today()))),
                consecutive_losses=saved.get("consecutive_losses", 0),
                paused=saved.get("paused", False),
                pause_reason=saved.get("pause_reason", ""),
            )
            rs.open_trades = get_open_trade_count()
            st.session_state.risk_state      = rs
            st.session_state.bot_running     = saved.get("bot_running", False)
            st.session_state.dry_run         = saved.get("dry_run", True)
            st.session_state.equity          = rs.equity
            st.session_state.pip_target      = rs.pip_target
            st.session_state.risk_pct        = rs.max_risk_pct
            if saved.get("strategy_weights"):
                st.session_state.strategy_weights = saved["strategy_weights"]
            # Restore dedup guards so reloads don't retrigger same trade
            st.session_state.last_signal_key = saved.get("last_signal_key", "")
            saved_tt = saved.get("last_trade_time")
            if saved_tt:
                try:
                    st.session_state.last_trade_time = datetime.fromisoformat(saved_tt)
                except Exception:
                    st.session_state.last_trade_time = None
        else:
            st.session_state.risk_state = RiskState(
                equity=st.session_state.equity,
                initial_equity=st.session_state.equity,
                pip_target=st.session_state.pip_target,
                max_risk_pct=st.session_state.risk_pct,
            )

_init_state()
DATA    = DataEngine()
SMC_ENG = SMCEngine()


def _persist_state():
    rs = st.session_state.risk_state
    if rs is None:
        return
    last_tt = st.session_state.last_trade_time
    save_bot_state({
        "bot_running":        st.session_state.bot_running,
        "dry_run":            st.session_state.dry_run,
        "equity":             rs.equity,
        "initial_equity":     rs.initial_equity,
        "pip_target":         rs.pip_target,
        "risk_pct":           rs.max_risk_pct,
        "daily_pnl_usd":      rs.daily_pnl_usd,
        "daily_pips":         rs.daily_pips,
        "daily_date":         str(rs.daily_date),
        "consecutive_losses": rs.consecutive_losses,
        "paused":             rs.paused,
        "pause_reason":       rs.pause_reason,
        "strategy_weights":   st.session_state.strategy_weights,
        "last_signal_key":    st.session_state.last_signal_key,
        "last_trade_time":    last_tt.isoformat() if last_tt else None,
    })


def _signal_key(sig_dict: dict) -> str:
    entry = round(float(sig_dict.get("entry") or 0) / 2.0) * 2
    return f"{sig_dict.get('strategy')}_{sig_dict.get('direction')}_{entry}"


def _sr_anchored_sl_tp(price: float, atr: float, direction: str,
                        sr_levels: list, regime: Regime):
    regime_mults = {
        Regime.TRENDING_BULL: (1.2, 1.5, 3.0),
        Regime.TRENDING_BEAR: (1.2, 1.5, 3.0),
        Regime.RANGING:       (1.0, 1.0, 2.0),
        Regime.HIGH_VOL:      (2.0, 1.5, 3.5),
        Regime.LOW_LIQ:       (1.5, 1.2, 2.5),
    }
    sl_mult, tp1_mult, tp2_mult = regime_mults.get(regime, (1.5, 1.0, 2.5))

    if direction == "BUY":
        sl_atr  = price - atr * sl_mult
        tp1_atr = price + atr * tp1_mult
        tp2_atr = price + atr * tp2_mult
    else:
        sl_atr  = price + atr * sl_mult
        tp1_atr = price - atr * tp1_mult
        tp2_atr = price - atr * tp2_mult

    if not sr_levels:
        return round(sl_atr, 2), round(tp1_atr, 2), round(tp2_atr, 2)

    sl_cands, tp_cands = [], []
    for lvl in sr_levels:
        lp = lvl.price if hasattr(lvl, "price") else lvl.get("price", 0)
        if direction == "BUY":
            if lp < price - atr * 0.5: sl_cands.append(lp)
            if lp > price + atr * 0.8: tp_cands.append(lp)
        else:
            if lp > price + atr * 0.5: sl_cands.append(lp)
            if lp < price - atr * 0.8: tp_cands.append(lp)

    sl = sl_atr
    if sl_cands:
        best = max(sl_cands) if direction == "BUY" else min(sl_cands)
        sl = best - atr * 0.3 if direction == "BUY" else best + atr * 0.3
    if abs(price - sl) < atr * 0.5:
        sl = sl_atr

    risk = abs(price - sl)
    tp1 = tp1_atr
    if tp_cands:
        nearest = min(tp_cands) if direction == "BUY" else max(tp_cands)
        if abs(nearest - price) >= risk * 1.0:
            tp1 = nearest

    tp2 = tp2_atr
    remaining = [t for t in tp_cands if abs(t - price) > abs(tp1 - price)]
    if remaining:
        nxt = min(remaining) if direction == "BUY" else max(remaining)
        if abs(nxt - price) >= risk * 1.8:
            tp2 = nxt

    if abs(tp2 - price) < risk * 1.8:
        tp2 = price + risk * 2.5 if direction == "BUY" else price - risk * 2.5
    if abs(tp1 - price) < risk * 1.0:
        tp1 = price + risk * 1.2 if direction == "BUY" else price - risk * 1.2

    return round(sl, 2), round(tp1, 2), round(tp2, 2)


# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""<style>
:root{--gold:#B8860B;--gold2:#D4A017;--green:#00C851;--red:#FF4444;--bg:#0A0A0F;--bg2:#13131C;--bg3:#1A1A28;--border:#2A2A3D;--text:#E0E0E0;--text2:#9090A0;}
.stApp{background:var(--bg);}
.stApp>header{background:transparent!important;}
#MainMenu,footer,.stDeployButton{display:none!important;}
[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border);}
[data-testid="stMetric"]{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:8px!important;padding:.8rem!important;}
[data-testid="stMetricLabel"]{color:var(--text2)!important;font-size:11px!important;text-transform:uppercase;}
[data-testid="stMetricValue"]{color:var(--text)!important;font-size:20px!important;font-weight:700!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg2);border-bottom:1px solid var(--border);gap:0;}
.stTabs [data-baseweb="tab"]{color:var(--text2)!important;padding:.6rem 1.2rem;font-size:13px;border-radius:0!important;}
.stTabs [aria-selected="true"]{color:var(--gold)!important;border-bottom:2px solid var(--gold)!important;background:transparent!important;}
.stButton>button{background:var(--bg3)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:6px!important;font-size:13px!important;}
.stButton>button:hover{border-color:var(--gold)!important;color:var(--gold)!important;}
hr{border-color:var(--border)!important;}
.signal-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:1rem;margin-bottom:.75rem;}
.signal-buy{border-left:4px solid var(--green)!important;}
.signal-sell{border-left:4px solid var(--red)!important;}
.price-ticker{font-size:26px;font-weight:700;color:var(--gold);letter-spacing:2px;}
.pip-progress{background:var(--bg3);border-radius:4px;height:8px;margin:.3rem 0;overflow:hidden;}
.pip-fill{background:linear-gradient(90deg,var(--gold),var(--gold2));border-radius:4px;height:100%;}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;}
.dot-green{background:var(--green);box-shadow:0 0 6px var(--green);}
.dot-red{background:var(--red);box-shadow:0 0 6px var(--red);}
.dot-amber{background:#f59e0b;box-shadow:0 0 6px #f59e0b;}
.info-box{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:.7rem 1rem;margin:.4rem 0;font-size:12px;color:var(--text2);}
.warn-box{background:#1a0f00;border:1px solid #f59e0b55;border-radius:8px;padding:.7rem 1rem;margin:.4rem 0;font-size:12px;color:#f59e0b;}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:1rem 0 .5rem'>"
            "<div style='font-size:28px'>⚡</div>"
            "<div style='color:#B8860B;font-size:16px;font-weight:700;letter-spacing:2px'>XAUUSD BOT</div>"
            "<div style='color:#606070;font-size:11px'>24×7 Autonomous Trading</div>"
            "</div>",
            unsafe_allow_html=True
        )
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            lbl  = "▶ START" if not st.session_state.bot_running else "⏸ STOP"
            btyp = "primary"  if not st.session_state.bot_running else "secondary"
            if st.button(lbl, use_container_width=True, type=btyp):
                st.session_state.bot_running = not st.session_state.bot_running
                _persist_state()
                icon = "✅" if st.session_state.bot_running else "⏸"
                msg  = "Bot started — state persisted!" if st.session_state.bot_running else "Bot stopped"
                st.toast(msg, icon=icon)
        with col2:
            mode  = "PAPER" if st.session_state.dry_run else "LIVE"
            color = "#60a5fa" if st.session_state.dry_run else "#ef4444"
            st.markdown(
                f"<div style='text-align:center;padding:.4rem;background:#13131C;"
                f"border:1px solid #2A2A3D;border-radius:6px;font-size:12px;font-weight:700;color:{color}'>{mode}</div>",
                unsafe_allow_html=True
            )

        rs = st.session_state.risk_state
        if rs:
            rm     = RiskManager(rs)
            can, reason = rm.can_trade()
            dot    = "dot-green" if (st.session_state.bot_running and can) else "dot-amber"
            label  = "Running" if (st.session_state.bot_running and can) else reason[:32]
            st.markdown(
                f'<div style="margin:.4rem 0"><span class="status-dot {dot}"></span>'
                f'<span style="font-size:12px;color:#9090A0">{label}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="info-box">💾 State auto-saved to DB — survives page reload &amp; Streamlit sleep</div>',
            unsafe_allow_html=True
        )
        st.divider()

        st.markdown("**⚙️ Configuration**")
        new_dry = st.toggle("Paper Trading Mode", value=st.session_state.dry_run)
        if new_dry != st.session_state.dry_run:
            st.session_state.dry_run = new_dry
            _persist_state()

        eq = st.number_input("Account Equity ($)", value=float(st.session_state.equity),
                              min_value=100.0, step=100.0, format="%.2f")
        if eq != st.session_state.equity:
            st.session_state.equity = eq
            if rs: rs.equity = eq; rs.initial_equity = eq
            _persist_state()

        tgt = st.number_input("Daily Pip Target", value=float(st.session_state.pip_target),
                               min_value=20.0, step=10.0)
        if tgt != st.session_state.pip_target:
            st.session_state.pip_target = tgt
            if rs: rs.pip_target = tgt
            _persist_state()

        rsk = st.slider("Risk per Trade (%)", 0.5, 3.0, float(st.session_state.risk_pct), 0.1)
        if rsk != st.session_state.risk_pct:
            st.session_state.risk_pct = rsk
            if rs: rs.max_risk_pct = rsk
            _persist_state()

        st.divider()
        st.markdown("**📊 Strategy Weights**")
        for s_name, s_w in st.session_state.strategy_weights.items():
            label  = s_name.replace("_", " ").title()[:18]
            wcol   = "#22c55e" if s_w > 1.1 else "#ef4444" if s_w < 0.9 else "#B8860B"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:11px;color:#9090A0;margin:.15rem 0">'
                f'<span>{label}</span><span style="color:{wcol};font-weight:700">{s_w:.2f}×</span></div>',
                unsafe_allow_html=True
            )

        if st.button("⚙️ Run Optimizer Now", use_container_width=True):
            all_t = get_trades(limit=500)
            new_w = run_optimizer(all_t, st.session_state.strategy_weights)
            st.session_state.strategy_weights = new_w
            st.session_state.optimizer_last   = datetime.now(timezone.utc)
            _persist_state()
            st.toast("Optimizer complete!", icon="⚙️")

        if rs and rs.paused:
            st.divider()
            st.warning(f"⏸ {rs.pause_reason[:45]}")
            if st.button("▶ Resume Trading", use_container_width=True):
                RiskManager(rs).resume()
                _persist_state()
                st.toast("Resumed!", icon="✅")

        st.divider()
        st.markdown(
            f'<div style="font-size:10px;color:#404055;text-align:center">v1.1 · Cycle #{st.session_state.cycle_count}</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════
def render_header():
    price = DATA.current_price()
    if price > 0:
        st.session_state.last_price = price

    df_h1  = DATA.get("1h", 250)
    regime  = Regime.RANGING
    session = Session.OFF_HOURS
    if not df_h1.empty:
        df_h1e = Indicators.enrich(df_h1)
        if not df_h1e.empty:
            regime  = detect_regime(df_h1e)
            session = detect_session()

    r_label, r_color = REGIME_LABEL.get(regime,  ("Unknown", "#6b7280"))
    s_label, s_color = SESSION_LABEL.get(session, ("—",       "#6b7280"))

    c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])

    with c1:
        p_str = f"${price:,.2f}" if price > 0 else "Loading..."
        st.markdown(
            f'<div class="price-ticker">{p_str}</div>'
            f'<div style="color:#606070;font-size:11px">XAU/USD · Gold Spot</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div style="font-size:11px;color:#606070;text-transform:uppercase;margin-bottom:.3rem">Regime</div>'
            f'<span style="background:{r_color}22;color:{r_color};border:1px solid {r_color}44;'
            f'padding:.3rem .7rem;border-radius:20px;font-size:12px;font-weight:700">{r_label}</span>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div style="font-size:11px;color:#606070;text-transform:uppercase;margin-bottom:.3rem">Session</div>'
            f'<span style="background:{s_color}22;color:{s_color};border:1px solid {s_color}44;'
            f'padding:.3rem .7rem;border-radius:20px;font-size:12px;font-weight:700">{s_label}</span>',
            unsafe_allow_html=True
        )
    with c4:
        dxy_df  = DATA.get_dxy(30)
        dxy_b   = get_dxy_bias(dxy_df)
        dxy_col = "#ef4444" if dxy_b == "bullish" else "#22c55e" if dxy_b == "bearish" else "#6b7280"
        arrow   = "↑" if dxy_b == "bullish" else "↓" if dxy_b == "bearish" else "—"
        st.markdown(
            f'<div style="font-size:11px;color:#606070;text-transform:uppercase;margin-bottom:.3rem">DXY Macro</div>'
            f'<span style="color:{dxy_col};font-size:13px;font-weight:700">{arrow} DXY {dxy_b.title()}</span>',
            unsafe_allow_html=True
        )
    with c5:
        rs = st.session_state.risk_state
        if rs:
            pct   = min(100, (rs.daily_pips / max(rs.pip_target, 1)) * 100)
            tcol  = "#22c55e" if rs.daily_pips >= 0 else "#ef4444"
            st.markdown(
                f'<div style="font-size:11px;color:#606070;text-transform:uppercase;margin-bottom:.3rem">Daily Target</div>'
                f'<div class="pip-progress"><div class="pip-fill" style="width:{pct:.0f}%"></div></div>'
                f'<span style="color:{tcol};font-size:12px;font-weight:700">'
                f'{rs.daily_pips:+.1f} / {rs.pip_target:.0f} pips</span>',
                unsafe_allow_html=True
            )
    st.divider()


# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════
def tab_dashboard():
    rs = st.session_state.risk_state
    if rs is None:
        st.info("Configure settings in sidebar.")
        return

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    eq_chg = ((rs.equity - rs.initial_equity) / max(rs.initial_equity, 1)) * 100
    c1.metric("Equity",        f"${rs.equity:,.2f}",      f"{eq_chg:+.2f}%")
    c2.metric("Daily PnL",     f"${rs.daily_pnl_usd:+.2f}", f"{rs.daily_pips:+.1f} pips")
    c3.metric("Open Trades",   str(get_open_trade_count()), f"Max {rs.max_concurrent}")
    m = get_metrics_cached()
    c4.metric("Win Rate",      f"{m.get('win_rate',0):.1f}%",      f"{m.get('total',0)} trades")
    c5.metric("Profit Factor", f"{m.get('profit_factor',0):.2f}",  "Target >=1.3")
    c6.metric("Total Pips",    f"{(m.get('total_pips') or 0):+.1f}", f"Avg {(m.get('avg_pips') or 0):.1f}/trade")

    st.markdown("###")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("**📈 Daily Pip Performance**")
        daily = get_daily_history(30)
        if daily:
            df_d = pd.DataFrame(daily)
            df_d["pips"] = pd.to_numeric(df_d["pips"], errors="coerce").fillna(0)
            fig = go.Figure(go.Bar(
                x=df_d["dt"], y=df_d["pips"],
                marker_color=df_d["pips"].apply(lambda x: "#22c55e" if x >= 0 else "#ef4444"),
                hovertemplate="%{x}<br>%{y:+.1f} pips<extra></extra>"
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(l=0,r=0,t=10,b=0), font=dict(color="#9090A0",size=11),
                xaxis=dict(gridcolor="#1A1A28",color="#9090A0"),
                yaxis=dict(gridcolor="#1A1A28",color="#9090A0",zeroline=True,zerolinecolor="#2A2A3D"),
                showlegend=False,
            )
            st.plotly_chart(fig)
        else:
            st.info("No closed trades yet.")

    with col_r:
        st.markdown("**🎯 Strategy Performance**")
        by_s = m.get("by_strategy", [])
        if by_s:
            df_s = pd.DataFrame(by_s)
            df_s["label"] = df_s["strategy"].str.replace("_"," ").str.title().str[:14]
            fig2 = go.Figure(go.Bar(
                x=df_s["label"], y=df_s["pp"],
                marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in df_s["pp"]],
                hovertemplate="%{x}<br>%{y:+.1f} pips<extra></extra>",
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(l=0,r=0,t=10,b=0), font=dict(color="#9090A0",size=11),
                xaxis=dict(gridcolor="#1A1A28",color="#9090A0",tickangle=-20),
                yaxis=dict(gridcolor="#1A1A28",color="#9090A0"), showlegend=False,
            )
            st.plotly_chart(fig2)
        else:
            st.info("No trade data yet.")

    st.markdown("**🕯️ XAU/USD — H1 Chart with S/R Levels**")
    df_h1 = DATA.get("1h", 100)
    if not df_h1.empty:
        df_h1e = Indicators.enrich(df_h1)
        if not df_h1e.empty:
            _render_chart_sr(df_h1e.tail(80))


def _render_chart_sr(df: pd.DataFrame):
    if df.empty:
        return
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444", name="Price",
    ))
    for col, color, name in [("ema20","#60a5fa","EMA 20"),("ema50","#f59e0b","EMA 50"),("ema200","#a78bfa","EMA 200")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=color,width=1), name=name, opacity=0.8))

    for lvl in st.session_state.sr_levels_cache[:8]:
        lp   = lvl.price if hasattr(lvl,"price") else lvl.get("price",0)
        lk   = lvl.kind  if hasattr(lvl,"kind")  else lvl.get("kind","")
        lc   = "#22c55e" if lk == "support" else "#ef4444"
        fig.add_hline(y=lp, line_dash="dot", line_color=lc, line_width=1,
                      annotation_text=f"{'S' if lk=='support' else 'R'} {lp:.0f}",
                      annotation_font=dict(color=lc, size=10))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=0,r=0,t=10,b=0), font=dict(color="#9090A0",size=11),
        xaxis=dict(gridcolor="#1A1A28",color="#9090A0",rangeslider_visible=False),
        yaxis=dict(gridcolor="#1A1A28",color="#9090A0",side="right"),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#9090A0",size=10)),
    )
    st.plotly_chart(fig)


# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — Signals
# ══════════════════════════════════════════════════════════════════════
def tab_signals():
    st.markdown(
        '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:1rem">'
        '<div style="width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 8px #22c55e"></div>'
        '<span style="color:#9090A0;font-size:12px">Dedup active — same strategy+direction per candle = 1 signal max</span>'
        '</div>',
        unsafe_allow_html=True
    )
    live = st.session_state.last_signals
    if live:
        st.markdown(f"**🔴 Live Signals ({len(live)})**")
        _render_signal_cards(live, live=True)
        st.divider()
    st.markdown("**📡 Signal History**")
    recent = get_recent_signals(30)
    if not recent:
        st.markdown(
            '<div style="text-align:center;padding:3rem;color:#404055">'
            '<div style="font-size:40px">📡</div>'
            '<div>No signals yet. Start the bot.</div></div>',
            unsafe_allow_html=True
        )
        return
    _render_signal_cards(recent, live=False)


def _render_signal_cards(signals, live=False):
    cols = st.columns(2)
    for i, sig in enumerate(signals):
        with cols[i % 2]:
            direction = sig.get("direction","") or ""
            strategy  = (sig.get("strategy","") or "").replace("_"," ").upper()
            conf      = float(sig.get("confidence") or 0)
            entry     = float(sig.get("entry_price") or sig.get("entry") or 0)
            sl        = float(sig.get("sl") or 0)
            tp1       = float(sig.get("tp1") or sig.get("take_profit_1") or 0)
            tp2       = float(sig.get("tp2") or sig.get("take_profit_2") or 0)
            rr2       = float(sig.get("rr2") or sig.get("risk_reward_2") or 0)
            sess      = sig.get("session","") or ""
            regm      = (sig.get("regime","") or "").replace("_"," ")
            notes     = (sig.get("notes","") or "")[:80]
            ts_raw    = sig.get("timestamp","") or ""
            acted     = sig.get("acted_on",0)

            is_buy    = direction == "BUY"
            cc        = "#22c55e" if conf>=80 else "#B8860B" if conf>=65 else "#ef4444"
            bc        = "#22c55e66" if is_buy else "#ef444488"
            cls       = "signal-buy" if is_buy else "signal-sell"
            dbg       = "#0a2e1a"   if is_buy else "#2e0a0a"
            dc        = "#22c55e"   if is_buy else "#ef4444"
            de        = "▲"         if is_buy else "▼"
            ts        = ts_raw[:19].replace("T"," ") if ts_raw else ""

            td_html = (
                '<span style="background:#1a1400;color:#B8860B;border:1px solid #B8860B44;'
                'padding:.1rem .4rem;border-radius:4px;font-size:10px;font-weight:700;margin-left:6px">TRADED</span>'
            ) if acted else ""
            nt_html = (
                f'<div style="font-size:10px;color:#505060;margin-top:.4rem;border-top:1px solid #1A1A28;padding-top:.4rem">{notes}</div>'
            ) if notes else ""

            html = (
                f'<div class="signal-card {cls}" style="border:1px solid {bc}">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.6rem">'
                f'<div><div style="font-size:12px;font-weight:700;color:#E0E0E0;letter-spacing:.04em">{strategy}</div>'
                f'<span style="display:inline-block;background:{dbg};color:{dc};border-radius:4px;padding:.2rem .6rem;font-size:11px;font-weight:700;margin-top:4px">{de} {direction}</span>'
                f'{td_html}</div>'
                f'<div style="text-align:right"><div style="font-size:22px;font-weight:700;color:{cc}">{conf:.0f}%</div>'
                f'<div style="font-size:10px;color:#606070">confidence</div></div></div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem .8rem;margin:.5rem 0">'
                f'<div><div style="font-size:10px;color:#606070;text-transform:uppercase">Entry</div>'
                f'<div style="font-size:13px;font-weight:600;font-family:monospace">${entry:,.2f}</div></div>'
                f'<div><div style="font-size:10px;color:#606070;text-transform:uppercase">Stop Loss</div>'
                f'<div style="font-size:13px;font-weight:600;font-family:monospace;color:#ef4444">${sl:,.2f}</div></div>'
                f'<div><div style="font-size:10px;color:#606070;text-transform:uppercase">TP 1</div>'
                f'<div style="font-size:13px;font-weight:600;font-family:monospace;color:#22c55e">${tp1:,.2f}</div></div>'
                f'<div><div style="font-size:10px;color:#606070;text-transform:uppercase">TP 2</div>'
                f'<div style="font-size:13px;font-weight:600;font-family:monospace;color:#22c55e">${tp2:,.2f}</div></div>'
                f'</div>'
                f'<div style="display:flex;justify-content:space-between;border-top:1px solid #1A1A28;padding-top:.5rem;font-size:11px;color:#606070">'
                f'<span>{sess} · {regm}</span>'
                f'<span style="background:#1a1400;color:#B8860B;padding:.15rem .5rem;border-radius:4px;font-weight:700">1:{rr2:.1f} RR</span>'
                f'</div>{nt_html}'
                f'<div style="font-size:10px;color:#404050;margin-top:.3rem">{ts}</div>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — Trade History
# ══════════════════════════════════════════════════════════════════════
def tab_trades():
    open_trades = get_open_trades()
    cnt         = len(open_trades)
    dot_c       = "#22c55e" if cnt > 0 else "#606070"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.75rem">'
        f'<div style="width:10px;height:10px;border-radius:50%;background:{dot_c};box-shadow:0 0 6px {dot_c}"></div>'
        f'<strong>Open Trades ({cnt})</strong></div>',
        unsafe_allow_html=True
    )
    if open_trades:
        df_o = pd.DataFrame(open_trades)
        show = [c for c in ["strategy","direction","entry_price","sl","tp1","tp2","lot_size","confidence","open_time"] if c in df_o.columns]
        st.dataframe(
            df_o[show].rename(columns={"entry_price":"Entry","sl":"SL","tp1":"TP1","tp2":"TP2","lot_size":"Lot","confidence":"Conf","open_time":"Opened"}),
            use_container_width=True, hide_index=True
        )
        st.divider()

    c1,c2,c3 = st.columns(3)
    with c1: sf = st.selectbox("Strategy",["All","ema_momentum","trend_continuation","liquidity_sweep","breakout_expansion","smc_concepts"])
    with c2: df_from = st.date_input("From", value=None)
    with c3: df_to   = st.date_input("To",   value=None)

    trades = get_trades(limit=300, strategy=None if sf=="All" else sf,
                        date_from=str(df_from) if df_from else None,
                        date_to=str(df_to)   if df_to   else None)
    if not trades:
        st.info("No closed trades yet.")
        return

    df = pd.DataFrame(trades)
    def pip_c(v):
        try: return "color:#22c55e" if float(v)>0 else "color:#ef4444" if float(v)<0 else ""
        except: return ""

    cols = [c for c in ["open_time","strategy","direction","entry_price","exit_price","pnl_pips","pnl_usd","rr_actual","session","regime","status","duration_min"] if c in df.columns]
    df_s = df[cols].copy()
    df_s.columns = [c.replace("_"," ").title() for c in cols]
    st.markdown(f"**📋 Trade History ({len(df)})**")
    st.dataframe(df_s.style.map(pip_c, subset=[c for c in ["Pnl Pips","Pnl Usd"] if c in df_s.columns]),
                 use_container_width=True, hide_index=True, height=450)
    st.divider()
    r1,r2,r3,r4 = st.columns(4)
    wins = df[df["pnl_usd"]>0] if "pnl_usd" in df.columns else pd.DataFrame()
    r1.metric("Win Rate",     f"{len(wins)/max(len(df),1)*100:.1f}%")
    r2.metric("Total Pips",   f"{df['pnl_pips'].sum():+.1f}"         if "pnl_pips"    in df.columns else "—")
    r3.metric("Total PnL",    f"${df['pnl_usd'].sum():+.2f}"         if "pnl_usd"     in df.columns else "—")
    r4.metric("Avg Duration", f"{df['duration_min'].mean():.0f} min" if "duration_min" in df.columns else "—")


# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — Analytics
# ══════════════════════════════════════════════════════════════════════
def tab_analytics():
    m      = get_metrics_cached()
    trades = get_trades(limit=500)

    st.markdown("**🗺️ Win Rate Heatmap — Strategy × Session**")
    if trades:
        df = pd.DataFrame(trades)
        if "strategy" in df.columns and "session" in df.columns and "pnl_usd" in df.columns:
            df["win"] = (df["pnl_usd"].astype(float) > 0).astype(int)
            piv = df.pivot_table(values="win", index="strategy", columns="session",
                                 aggfunc=lambda x: round(x.mean()*100,1) if len(x)>0 else 0)
            if not piv.empty:
                fig = go.Figure(go.Heatmap(
                    z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
                    colorscale=[[0,"#2d0000"],[0.5,"#B8860B"],[1,"#004d00"]],
                    text=piv.values, texttemplate="%{text:.0f}%",
                    hovertemplate="%{y} x %{x}<br>WR: %{z:.1f}%<extra></extra>",
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  height=260, margin=dict(l=0,r=0,t=10,b=0),
                                  font=dict(color="#9090A0",size=11))
                st.plotly_chart(fig)

    cl, cr = st.columns(2)
    with cl:
        st.markdown("**📊 Equity Curve**")
        if trades:
            df2 = pd.DataFrame(trades).sort_values("open_time")
            df2["cum"]    = df2["pnl_usd"].astype(float).cumsum()
            df2["equity"] = st.session_state.equity + df2["cum"]
            f3 = go.Figure(go.Scatter(x=df2["open_time"], y=df2["equity"], mode="lines",
                                      line=dict(color="#B8860B",width=2),
                                      fill="tozeroy", fillcolor="rgba(184,134,11,0.1)"))
            f3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             height=260, margin=dict(l=0,r=0,t=10,b=0), font=dict(color="#9090A0",size=11),
                             xaxis=dict(gridcolor="#1A1A28",color="#9090A0"),
                             yaxis=dict(gridcolor="#1A1A28",color="#9090A0",side="right"), showlegend=False)
            st.plotly_chart(f3)
    with cr:
        st.markdown("**⚙️ Optimizer Weights**")
        df_w = pd.DataFrame([{
            "Strategy": k.replace("_"," ").title(),
            "Weight":   f"{v:.3f}x",
            "Status":   "Up Boosted" if v>1.1 else "Down Reduced" if v<0.9 else "Normal"
        } for k,v in st.session_state.strategy_weights.items()])
        st.dataframe(df_w, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
#  TAB 5 — Backtest
# ══════════════════════════════════════════════════════════════════════
def tab_backtest():
    st.markdown("**⚗️ Multi-Period Backtest**")
    st.markdown(
        '<div class="warn-box">Downloads 1-2 years of yfinance data. Takes 30-90 seconds per period. '
        'Use 1M first to validate, then run longer periods.</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([3,1])
    with c1:
        choices = st.multiselect("Periods", ["1M","3M","6M","1Y"], default=["1M","3M"])
    with c2:
        run = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    if not run:
        st.info("Select periods and click Run Backtest.")
        return

    period_days = {"1M":30,"3M":90,"6M":180,"1Y":365}

    try:
        from core.backtest_engine import run_backtest_period
    except ImportError:
        st.error("Backtest engine not found. Ensure core/backtest_engine.py exists.")
        return

    results = {}
    prog = st.progress(0, text="Starting...")
    for idx, ch in enumerate(choices):
        days = period_days.get(ch, 30)
        prog.progress(int(idx/len(choices)*100), text=f"Running {ch} ({days}d)...")
        results[ch] = run_backtest_period(days, "1h")
    prog.progress(100, text="Done!")

    st.divider()
    rows = []
    for lbl, r in results.items():
        rows.append({
            "Period":        lbl,
            "Trades":        r.get("total_trades", 0),
            "Win Rate":      f"{r.get('win_rate_pct',0):.1f}%",
            "Profit Factor": f"{r.get('profit_factor',0):.2f}",
            "Total Pips":    f"{r.get('total_pips',0):+.1f}",
            "Max DD":        f"{r.get('max_drawdown_pct',0):.1f}%",
            "Return":        f"{r.get('total_return_pct',0):+.2f}%",
            "Result":        "PASS" if r.get("passed") else "FAIL",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for lbl, r in results.items():
        ec = r.get("equity_curve",[])
        if ec and len(ec)>1:
            st.markdown(f"**Equity — {lbl}**")
            fig = go.Figure(go.Scatter(y=ec, mode="lines",
                                       line=dict(color="#B8860B" if r.get("passed") else "#ef4444", width=2),
                                       fill="tozeroy", fillcolor="rgba(184,134,11,0.08)"))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=180, margin=dict(l=0,r=0,t=10,b=0), font=dict(color="#9090A0",size=11),
                              xaxis=dict(gridcolor="#1A1A28",color="#9090A0",title="Trade #"),
                              yaxis=dict(gridcolor="#1A1A28",color="#9090A0",side="right"), showlegend=False)
            st.plotly_chart(fig)


# ══════════════════════════════════════════════════════════════════════
#  BOT CYCLE
# ══════════════════════════════════════════════════════════════════════
def run_bot_cycle():
    if not st.session_state.bot_running:
        return
    try:
        st.session_state.cycle_count += 1

        df_m1  = DATA.get("1m",  100)
        df_m15 = DATA.get("15m", 200)
        df_h1  = DATA.get("1h",  300)
        df_h4  = DATA.get("4h",  200)

        if df_h1 is None or df_h1.empty or len(df_h1) < 50:
            return

        df_h1e  = Indicators.enrich(df_h1)
        df_m15e = Indicators.enrich(df_m15) if not df_m15.empty and len(df_m15)>30 else None
        if df_h1e.empty:
            return

        price = float(df_h1e["close"].iloc[-1])
        atr   = float(df_h1e["atr"].iloc[-1])

        regime  = detect_regime(df_h1e)
        session = detect_session()
        smc_h1  = SMC_ENG.analyze(df_h1e)
        smc_m15 = SMC_ENG.analyze(df_m15e) if df_m15e is not None else SMCData()

        sr_levels = combine_sr(df_h1e, df_m15e if df_m15e is not None else df_h1e)
        st.session_state.sr_levels_cache = sr_levels

        # Monitor open trades FIRST
        rs = st.session_state.risk_state
        if rs:
            _monitor_open_trades(price, rs)

        # Generate signals
        signals = evaluate_all(
            df_m1, df_m15e, df_h1e, df_h4,
            smc_h1, smc_m15, sr_levels, regime, session,
            st.session_state.strategy_weights,
        )

        # Dedup — keep only signals with new key
        new_sigs = []
        for sig in signals:
            d   = sig.to_dict()
            key = _signal_key(d)
            if key != st.session_state.last_signal_key:
                d["is_duplicate"] = 0
                new_sigs.append((sig, d, key))
                save_signal(d, acted_on=False)
            else:
                d["is_duplicate"] = 1
                save_signal(d, acted_on=False)

        st.session_state.last_signals = [d for _, d, _ in new_sigs]

        # Trade execution: DB count gate + 15-minute cooldown
        open_count = get_open_trade_count()
        if rs and new_sigs and open_count < rs.max_concurrent:
            rm       = RiskManager(rs)
            can, _   = rm.can_trade()
            last_tt  = st.session_state.last_trade_time
            cooldown = (last_tt is None or
                        (datetime.now(timezone.utc) - last_tt).total_seconds() >= 900)  # 15 min

            if can and cooldown:
                top_sig, top_d, top_key = new_sigs[0]

                # DXY filter
                dxy_df   = DATA.get_dxy(30)
                dxy_b    = get_dxy_bias(dxy_df)
                _, cm    = dxy_alignment(dxy_b, top_sig.direction)
                top_sig.confidence = min(100, top_sig.confidence + cm)

                if top_sig.confidence >= 65:
                    sl, tp1, tp2 = _sr_anchored_sl_tp(price, atr, top_sig.direction, sr_levels, regime)
                    lot = rm.lot_size(top_sig)
                    rec = {
                        "strategy":    top_sig.strategy,
                        "direction":   top_sig.direction,
                        "entry_price": price,
                        "sl":          sl,
                        "tp1":         tp1,
                        "tp2":         tp2,
                        "lot_size":    lot,
                        "atr":         atr,
                        "confidence":  round(top_sig.confidence, 1),
                        "session":     top_sig.session,
                        "regime":      top_sig.regime,
                        "timeframe":   top_sig.timeframe,
                        "status":      "open",
                        "open_time":   datetime.now(timezone.utc).isoformat(),
                        "notes":       f"{top_sig.notes} | SL@{sl:.1f} S/R-anchored",
                    }
                    save_trade(rec)
                    rs.open_trades = get_open_trade_count()
                    st.session_state.last_trade_time = datetime.now(timezone.utc)
                    st.session_state.last_signal_key = top_key
                    save_signal(top_d, acted_on=True)

        # Optimizer every 4h
        if should_run(st.session_state.optimizer_last, interval_hours=4):
            all_t = get_trades(limit=500)
            st.session_state.strategy_weights = run_optimizer(all_t, st.session_state.strategy_weights)
            st.session_state.optimizer_last   = datetime.now(timezone.utc)

        _persist_state()

    except Exception as e:
        st.session_state.errors.append(f"{datetime.now().strftime('%H:%M')} {type(e).__name__}: {e}")


def _gold_pips(price_diff: float) -> float:
    """
    Gold pip calculation.
    XAU/USD is quoted in USD per troy ounce.
    1 pip for Gold = $0.10 movement in price.
    So if price moves from 2000.00 → 2001.00, that is 10 pips.
    price_diff should be positive for wins, negative for losses.
    """
    return price_diff / 0.10


def _gold_pnl_usd(pips: float, lot_size: float) -> float:
    """
    PnL in USD for Gold.
    Standard lot = 100 oz. Mini lot (0.1) = 10 oz. Micro lot (0.01) = 1 oz.
    Pip value per standard lot = $10 (since 1 pip = $0.10 × 100 oz).
    Pip value per micro lot (0.01) = $0.10.
    Formula: pnl = pips × lot_size × $10 (per standard lot pip value)
    """
    pip_value_per_lot = 10.0   # $10 per pip per 1.0 standard lot
    return round(pips * lot_size * pip_value_per_lot, 2)


def _monitor_open_trades(price: float, rs: RiskState):
    """
    Check all open trades against current live price.

    Guards in place:
    1. Minimum 300s age — prevents closing on the same cycle trade was opened
    2. Minimum movement guard — price must move at least 10% of ATR before checking
    3. TP1 partial close — books 50% profit and moves SL to breakeven
    4. TP2 full close — closes remaining 50%
    5. SL full close — closes 100% at a loss
    """
    now = datetime.now(timezone.utc)
    for t in get_open_trades():
        entry  = float(t.get("entry_price") or 0)
        sl_p   = float(t.get("sl") or 0)
        tp1_p  = float(t.get("tp1") or 0)
        tp2_p  = float(t.get("tp2") or 0)
        direct = t.get("direction", "")
        tid    = t.get("id")
        status = t.get("status", "open")

        if price <= 0 or entry <= 0 or not tid:
            continue

        # ── CRITICAL: minimum age check ───────────────────────────
        # Without this guard, price == entry causes 0-pip close
        # on the same rerun the trade was opened.
        opened_str = t.get("open_time", "")
        if opened_str:
            try:
                opened_dt = datetime.fromisoformat(opened_str)
                if opened_dt.tzinfo is None:
                    opened_dt = opened_dt.replace(tzinfo=timezone.utc)
                age_seconds = (now - opened_dt).total_seconds()
                if age_seconds < 300:   # 5 minutes = safe minimum age
                    continue
            except Exception:
                continue
        else:
            continue

        # ── Guard: skip if price has barely moved from entry ───────
        # Prevents 0-pip or near-zero-pip spurious closes
        min_move = float(t.get("atr") or 5) * 0.1   # 10% of ATR = minimum meaningful move
        if abs(price - entry) < min_move:
            continue

        # ── SL/TP checks ──────────────────────────────────────────
        if direct == "BUY":
            # BUY profit = price rose above entry (price - entry > 0)
            # BUY loss   = price fell below SL   (price - entry < 0)
            if price <= sl_p:
                pips = _gold_pips(price - entry)    # negative = loss
                _close_trade(t, price, "sl_hit", pips, rs)

            elif price >= tp2_p:
                pips = _gold_pips(price - entry)    # positive = full win
                _close_trade(t, price, "tp2_hit", pips, rs)

            elif price >= tp1_p and status not in ("tp1_hit", "breakeven"):
                # TP1 hit: book 50% profit, move SL to breakeven
                pips = _gold_pips(price - entry)
                pnl  = _gold_pnl_usd(pips, float(t.get("lot_size") or 0.01) * 0.5)
                update_trade(tid, {
                    "sl":     round(entry + 0.50, 2),   # SL above entry = can't lose
                    "status": "tp1_hit",
                    "notes":  (t.get("notes","") or "") + f" | TP1@{price:.2f}(+{pips:.1f}pips)",
                })
                RiskManager(rs).on_close(pips * 0.5, pnl)
                _persist_state()

        elif direct == "SELL":
            # SELL profit = entry > price (price fell). Loss = price > entry.
            # entry - price: positive when price fell (win), negative when price rose (loss)
            if price >= sl_p:
                pips = _gold_pips(entry - price)    # negative value = loss (price rose past SL)
                _close_trade(t, price, "sl_hit", pips, rs)

            elif price <= tp2_p:
                pips = _gold_pips(entry - price)    # positive value = full win
                _close_trade(t, price, "tp2_hit", pips, rs)

            elif price <= tp1_p and status not in ("tp1_hit", "breakeven"):
                pips = _gold_pips(entry - price)    # positive partial win
                pnl  = _gold_pnl_usd(pips, float(t.get("lot_size") or 0.01) * 0.5)
                update_trade(tid, {
                    "sl":     round(entry - 0.50, 2),   # SL to breakeven below entry
                    "status": "tp1_hit",
                    "notes":  (t.get("notes","") or "") + f" | TP1@{price:.2f}(+{pips:.1f}pips)",
                })
                RiskManager(rs).on_close(pips * 0.5, pnl)
                _persist_state()


def _close_trade(t: dict, price: float, status: str, pips: float, rs: RiskState):
    """Close a trade fully: update DB record and credit PnL to risk state."""
    lot     = float(t.get("lot_size") or 0.01)
    pnl     = _gold_pnl_usd(pips, lot)
    entry   = float(t.get("entry_price") or price)
    opened  = t.get("open_time", "")
    dur     = 0.0
    if opened:
        try:
            dt = datetime.fromisoformat(opened)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dur = (datetime.now(timezone.utc) - dt).total_seconds() / 60
        except Exception:
            pass
    sl_val = float(t.get("sl") or entry)
    rr     = abs(price - entry) / max(abs(entry - sl_val), 0.01)
    update_trade(t["id"], {
        "exit_price":    price,
        "status":        status,
        "pnl_pips":      round(pips, 1),
        "pnl_usd":       pnl,
        "close_time":    datetime.now(timezone.utc).isoformat(),
        "duration_min":  round(dur, 1),
        "rr_actual":     round(rr, 2),
    })
    RiskManager(rs).on_close(pips, pnl)
    rs.open_trades = get_open_trade_count()
    _persist_state()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    render_sidebar()
    render_header()

    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "📊 Dashboard","📡 Signals","📋 Trade History","🔬 Analytics","⚗️ Backtest"
    ])
    with tab1: tab_dashboard()
    with tab2: tab_signals()
    with tab3: tab_trades()
    with tab4: tab_analytics()
    with tab5: tab_backtest()

    if st.session_state.errors:
        with st.expander(f"⚠️ {len(st.session_state.errors)} errors", expanded=False):
            for e in st.session_state.errors[-5:]:
                st.code(e)

    run_bot_cycle()

    if st.session_state.bot_running:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
