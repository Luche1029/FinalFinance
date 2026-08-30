"""Pagina Streamlit — Backtesting delle strategie su indicatori (Modulo 2).

Riusa lo stesso motore di backtest e le stesse strategie testate da riga di
comando in scripts/backtest_indicators.py: qui cambia solo l'interfaccia,
non la logica sottostante (già validata su AAPL e BTC/USDT con dati reali).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import plotly.graph_objects as go
import streamlit as st

from src.backtesting.engine import run_backtest
from src.backtesting.strategies import rsi_mean_reversion_signal, sma_crossover_signal
from src.config import load_settings
from src.dashboard.loader import MARKET_MODULES, load_data, lookback_selector, symbol_options_for
from src.indicators.technical import add_all_indicators

st.set_page_config(page_title="Backtesting", layout="wide")
st.title("Backtesting delle strategie")
st.caption(
    "Confronto tra strategie basate su indicatori e il benchmark buy & hold. "
    "Se una strategia non batte il benchmark è un risultato legittimo da "
    "riportare, non un errore: gran parte delle strategie semplici non lo fa."
)

settings = load_settings()
watchlist = settings["watchlist"]

col_market, col_symbol, col_strategy, col_lookback = st.columns(4)

with col_market:
    market_label = st.selectbox("Mercato", list(MARKET_MODULES.keys()))
market_key = MARKET_MODULES[market_label]

with col_symbol:
    symbol = st.selectbox("Strumento", symbol_options_for(market_key, watchlist))

with col_strategy:
    strategy_label = st.selectbox(
        "Strategia", ["SMA 20/50 crossover", "RSI 14 mean-reversion", "Confronta entrambe"]
    )

with col_lookback:
    lookback = lookback_selector(market_key, key_prefix="backtest")

col_capital, col_cost = st.columns(2)
with col_capital:
    initial_capital = st.number_input("Capitale iniziale", min_value=100, value=10_000, step=100)
with col_cost:
    transaction_cost_bps = st.slider(
        "Costo di transazione (bps per operazione)", min_value=0, max_value=50, value=5,
        help="1 bps = 0.01%. Applicato ad ogni cambio di posizione, per approssimare spread/commissioni.",
    )

run_clicked = st.button("Esegui backtest", type="primary")

if run_clicked:
    try:
        with st.spinner(f"Carico dati per {symbol}..."):
            df = load_data(market_key, symbol, "1d", lookback=lookback)
            df = add_all_indicators(df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Errore nel caricamento dati: {exc}")
        st.stop()

    strategies_to_run = {}
    if strategy_label in ("SMA 20/50 crossover", "Confronta entrambe"):
        strategies_to_run["SMA 20/50 crossover"] = sma_crossover_signal(df, "sma_20", "sma_50")
    if strategy_label in ("RSI 14 mean-reversion", "Confronta entrambe"):
        strategies_to_run["RSI 14 mean-reversion"] = rsi_mean_reversion_signal(df, "rsi_14")

    fig = go.Figure()
    benchmark_plotted = False

    for name, signal in strategies_to_run.items():
        df_strategy = df.copy()
        df_strategy["signal"] = signal
        result = run_backtest(
            df_strategy,
            signal_col="signal",
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
        )

        st.subheader(name)
        metrics_cols = st.columns(len(result.metrics))
        for col, (k, v) in zip(metrics_cols, result.metrics.items()):
            col.metric(k, v)

        fig.add_trace(go.Scatter(x=result.equity_curve.index, y=result.equity_curve.values, name=name))

        if not benchmark_plotted:
            fig.add_trace(go.Scatter(
                x=result.benchmark_equity_curve.index,
                y=result.benchmark_equity_curve.values,
                name="Buy & Hold (benchmark)",
                line=dict(dash="dash", color="gray"),
            ))
            benchmark_plotted = True

    fig.update_layout(
        title=f"Equity curve — {symbol}",
        yaxis_title=f"Capitale (partenza: {initial_capital:,.0f})",
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Imposta i parametri e premi \"Esegui backtest\".")
