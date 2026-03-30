"""
SQLite trade journal with WAL mode and backward-compatible migrations.
Works on Streamlit Cloud using local filesystem (/tmp for ephemeral, or SQLAlchemy).
"""
from __future__ import annotations
import sqlite3
import os
from datetime import datetime, date, timezone
from typing import List, Optional, Dict
import streamlit as st

DB_PATH = os.environ.get("DB_PATH", "trades.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    """Create tables and run migrations."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy    TEXT,
            direction   TEXT,
            entry_price REAL,
            exit_price  REAL,
            sl          REAL,
            tp1         REAL,
            tp2         REAL,
            lot_size    REAL,
            atr         REAL,
            confidence  REAL,
            session     TEXT,
            regime      TEXT,
            timeframe   TEXT,
            status      TEXT DEFAULT 'open',
            pnl_pips    REAL DEFAULT 0,
            pnl_usd     REAL DEFAULT 0,
            open_time   TEXT,
            close_time  TEXT,
            duration_min REAL DEFAULT 0,
            notes       TEXT,
            rr_actual   REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS signals_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy    TEXT,
            direction   TEXT,
            confidence  REAL,
            entry_price REAL,
            sl          REAL,
            tp1         REAL,
            tp2         REAL,
            atr         REAL,
            timeframe   TEXT,
            session     TEXT,
            regime      TEXT,
            rr1         REAL,
            rr2         REAL,
            notes       TEXT,
            timestamp   TEXT,
            acted_on    INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS optimizer_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_time    TEXT,
            strategy    TEXT,
            old_weight  REAL,
            new_weight  REAL,
            win_rate    REAL,
            profit_factor REAL,
            total_trades INTEGER,
            notes       TEXT
        );
    """)
    # migrations — safe to run repeatedly
    _migrate(conn)
    conn.commit()
    conn.close()


def _migrate(conn: sqlite3.Connection):
    """Add columns that may be missing in older DB versions."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    adds = {
        "rr_actual": "REAL DEFAULT 0",
        "duration_min": "REAL DEFAULT 0",
    }
    for col, typedef in adds.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typedef}")


# ── Write ────────────────────────────────────────────────────────────

def save_trade(t: dict) -> int:
    conn = _connect()
    try:
        cur = conn.execute("""
            INSERT INTO trades
            (strategy,direction,entry_price,exit_price,sl,tp1,tp2,
             lot_size,atr,confidence,session,regime,timeframe,status,
             pnl_pips,pnl_usd,open_time,close_time,duration_min,notes,rr_actual)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            t.get("strategy"), t.get("direction"), t.get("entry_price"),
            t.get("exit_price"), t.get("sl"), t.get("tp1"), t.get("tp2"),
            t.get("lot_size"), t.get("atr"), t.get("confidence"),
            t.get("session"), t.get("regime"), t.get("timeframe"),
            t.get("status", "open"), t.get("pnl_pips", 0), t.get("pnl_usd", 0),
            t.get("open_time"), t.get("close_time"), t.get("duration_min", 0),
            t.get("notes"), t.get("rr_actual", 0),
        ))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_trade(trade_id: int, updates: dict):
    if not updates:
        return
    cols = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [trade_id]
    conn = _connect()
    try:
        conn.execute(f"UPDATE trades SET {cols} WHERE id=?", vals)
        conn.commit()
    finally:
        conn.close()


def save_signal(s: dict, acted_on: bool = False):
    conn = _connect()
    try:
        conn.execute("""
            INSERT INTO signals_log
            (strategy,direction,confidence,entry_price,sl,tp1,tp2,
             atr,timeframe,session,regime,rr1,rr2,notes,timestamp,acted_on)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            s.get("strategy"), s.get("direction"), s.get("confidence"),
            s.get("entry"), s.get("sl"), s.get("tp1"), s.get("tp2"),
            s.get("atr"), s.get("timeframe"), s.get("session"),
            s.get("regime"), s.get("rr1"), s.get("rr2"),
            s.get("notes"), s.get("timestamp"), 1 if acted_on else 0,
        ))
        conn.commit()
    finally:
        conn.close()


def log_optimizer(run: dict):
    conn = _connect()
    try:
        conn.execute("""
            INSERT INTO optimizer_log
            (run_time,strategy,old_weight,new_weight,win_rate,profit_factor,total_trades,notes)
            VALUES (?,?,?,?,?,?,?,?)
        """, (run.get("run_time"), run.get("strategy"), run.get("old_weight"),
              run.get("new_weight"), run.get("win_rate"), run.get("profit_factor"),
              run.get("total_trades"), run.get("notes")))
        conn.commit()
    finally:
        conn.close()


# ── Read ─────────────────────────────────────────────────────────────

def get_open_trades() -> List[dict]:
    conn = _connect()
    try:
        cur = conn.execute("SELECT * FROM trades WHERE status IN ('open','breakeven') ORDER BY open_time DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


def get_trades(limit=200, strategy=None, date_from=None, date_to=None) -> List[dict]:
    conn = _connect()
    try:
        q = "SELECT * FROM trades WHERE status NOT IN ('open','pending') "
        p = []
        if strategy:  q += " AND strategy=?"; p.append(strategy)
        if date_from: q += " AND open_time>=?"; p.append(date_from)
        if date_to:   q += " AND open_time<=?"; p.append(date_to)
        q += f" ORDER BY open_time DESC LIMIT {int(limit)}"
        cur = conn.execute(q, p)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


def get_recent_signals(limit=30) -> List[dict]:
    conn = _connect()
    try:
        cur = conn.execute("SELECT * FROM signals_log ORDER BY timestamp DESC LIMIT ?", (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


@st.cache_data(ttl=30, show_spinner=False)
def get_metrics_cached() -> dict:
    return _calc_metrics()


def _calc_metrics() -> dict:
    conn = _connect()
    try:
        cur = conn.execute("""
            SELECT COUNT(*) total,
                   SUM(CASE WHEN pnl_usd>0 THEN 1 ELSE 0 END) wins,
                   SUM(CASE WHEN pnl_usd<0 THEN 1 ELSE 0 END) losses,
                   SUM(pnl_pips) total_pips, SUM(pnl_usd) total_pnl,
                   AVG(pnl_pips) avg_pips, MAX(pnl_pips) best_pips, MIN(pnl_pips) worst_pips,
                   AVG(rr_actual) avg_rr
            FROM trades WHERE status NOT IN ('open','pending','cancelled')
        """)
        row = dict(zip([d[0] for d in cur.description], cur.fetchone()))

        gp_gl = conn.execute("""
            SELECT SUM(CASE WHEN pnl_usd>0 THEN pnl_usd ELSE 0 END) gp,
                   SUM(CASE WHEN pnl_usd<0 THEN ABS(pnl_usd) ELSE 0 END) gl
            FROM trades WHERE status NOT IN ('open','pending','cancelled')
        """).fetchone()
        gp, gl = gp_gl[0] or 0, gp_gl[1] or 0.01
        row["profit_factor"] = round(gp / gl, 2)
        row["win_rate"] = round((row["wins"] or 0) / max(row["total"] or 1, 1) * 100, 1)

        # by strategy
        cur2 = conn.execute("""
            SELECT strategy,COUNT(*) t,SUM(CASE WHEN pnl_usd>0 THEN 1 ELSE 0 END) w,SUM(pnl_pips) pp
            FROM trades WHERE status NOT IN ('open','pending','cancelled') GROUP BY strategy
        """)
        row["by_strategy"] = [dict(zip([d[0] for d in cur2.description], r)) for r in cur2.fetchall()]

        # by session
        cur3 = conn.execute("""
            SELECT session,COUNT(*) t,SUM(CASE WHEN pnl_usd>0 THEN 1 ELSE 0 END) w,SUM(pnl_pips) pp
            FROM trades WHERE status NOT IN ('open','pending','cancelled') GROUP BY session
        """)
        row["by_session"] = [dict(zip([d[0] for d in cur3.description], r)) for r in cur3.fetchall()]

        # daily
        cur4 = conn.execute("""
            SELECT DATE(open_time) dt,COUNT(*) t,SUM(pnl_pips) pp,SUM(pnl_usd) pu
            FROM trades WHERE status NOT IN ('open','pending','cancelled')
            GROUP BY DATE(open_time) ORDER BY dt DESC LIMIT 30
        """)
        row["daily"] = [dict(zip([d[0] for d in cur4.description], r)) for r in cur4.fetchall()]

        return row
    finally:
        conn.close()


def get_daily_history(days=30) -> List[dict]:
    conn = _connect()
    try:
        cur = conn.execute("""
            SELECT DATE(open_time) dt,COUNT(*) trades,
                   SUM(pnl_pips) pips,SUM(pnl_usd) pnl
            FROM trades WHERE status NOT IN ('open','pending','cancelled')
              AND open_time >= datetime('now',?)
            GROUP BY DATE(open_time) ORDER BY dt DESC
        """, (f"-{days} days",))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


# ── Bot state persistence (survives page reload / Streamlit sleep) ───

def save_bot_state(state: dict):
    """Persist bot state dict as JSON in a key-value table."""
    import json
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_state (
                key TEXT PRIMARY KEY, value TEXT
            )""")
        conn.execute("INSERT OR REPLACE INTO bot_state(key,value) VALUES(?,?)",
                     ("main", json.dumps(state)))
        conn.commit()
    finally:
        conn.close()


def load_bot_state() -> dict:
    """Load persisted bot state. Returns {} if not found."""
    import json
    conn = _connect()
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS bot_state(key TEXT PRIMARY KEY, value TEXT)")
        row = conn.execute("SELECT value FROM bot_state WHERE key='main'").fetchone()
        return json.loads(row[0]) if row else {}
    except Exception:
        return {}
    finally:
        conn.close()


def get_open_trade_count() -> int:
    """Count open trades directly from DB — single source of truth."""
    conn = _connect()
    try:
        row = conn.execute("SELECT COUNT(*) FROM trades WHERE status IN ('open','breakeven')").fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def mark_signal_duplicate(key: str):
    """Mark the most recent signal with matching dedup key as duplicate."""
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_dedup (
                key TEXT PRIMARY KEY, last_seen TEXT
            )""")
        conn.execute("INSERT OR REPLACE INTO signal_dedup(key,last_seen) VALUES(?,?)",
                     (key, datetime.now(timezone.utc).isoformat()))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()
