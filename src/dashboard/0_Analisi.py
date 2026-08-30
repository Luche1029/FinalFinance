"""Dashboard di analisi — Modulo 1.

Avvio: dalla root del progetto,
    streamlit run src/dashboard/0_Analisi.py

Permette di scegliere un mercato (azioni/ETF, crypto, forex) e uno strumento
dalla watchlist configurata in config/settings.yaml, mostrando un grafico a
candele con indicatori tecnici sovrapposti.

Le pagine "Backtesting" e "Modelli predittivi" (Modulo 2) sono raggiungibili
dalla barra laterale: Streamlit le scopre automaticamente dalla cartella
src/dashboard/pages/.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permette di eseguire "streamlit run src/dashboard/app.py" dalla root del
# progetto senza dover installare il pacchetto: aggiunge la root al PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import load_settings
from src.dashboard.loader import MARKET_MODULES, load_data, lookback_selector, symbol_options_for
from src.indicators.technical import add_all_indicators, summarize_signals

st.set_page_config(page_title="Analisi Mercati", layout="wide")


def build_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name=symbol,
        )
    ])

    for col in df.columns:
        if col.startswith("sma_") or col.startswith("ema_"):
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))

    for col in df.columns:
        if col.startswith("bb_upper_") or col.startswith("bb_lower_"):
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col,
                line=dict(width=1, dash="dot"), opacity=0.5,
            ))

    fig.update_layout(
        title=f"{symbol} — prezzo e indicatori",
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def build_oscillator_chart(df: pd.DataFrame, rsi_col: str) -> go.Figure:
    fig = go.Figure()
    if rsi_col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[rsi_col], name="RSI"))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 100])
    return fig


def main() -> None:
    st.title("Dashboard di analisi mercati")
    st.caption(
        "Modulo 1 del progetto: solo visualizzazione e analisi tecnica. "
        "Nessun segnale operativo, nessuna esecuzione di ordini."
    )

    settings = load_settings()
    watchlist = settings["watchlist"]

    col_market, col_symbol, col_timeframe, col_lookback = st.columns(4)

    with col_market:
        market_label = st.selectbox("Mercato", list(MARKET_MODULES.keys()))
    market_key = MARKET_MODULES[market_label]

    symbol_options = symbol_options_for(market_key, watchlist)

    with col_symbol:
        symbol = st.selectbox("Strumento", symbol_options)

    with col_timeframe:
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15m"], index=0)

    with col_lookback:
        lookback = lookback_selector(market_key, key_prefix="analisi")

    try:
        with st.spinner(f"Carico dati per {symbol}..."):
            df = load_data(market_key, symbol, timeframe, lookback=lookback)
            df = add_all_indicators(df)
    except Exception as exc:  # noqa: BLE001 - mostriamo l'errore all'utente, non lo mascheriamo
        st.error(f"Errore nel caricamento dati: {exc}")
        st.stop()

    st.plotly_chart(build_candlestick_chart(df, symbol), use_container_width=True)

    rsi_period = settings["indicators"]["rsi_period"]
    st.subheader("RSI")
    st.plotly_chart(build_oscillator_chart(df, f"rsi_{rsi_period}"), use_container_width=True)

    st.subheader("Lettura sintetica")
    for line in summarize_signals(df).values():
        st.write(f"- {line}")

    with st.expander("Dati grezzi (ultime 20 righe)"):
        st.dataframe(df.tail(20))


if __name__ == "__main__":
    main()
