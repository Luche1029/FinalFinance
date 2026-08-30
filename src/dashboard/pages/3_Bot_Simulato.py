"""Pagina Streamlit — Bot di trading simulato (Modulo 3, dummy).

ATTENZIONE: nessuna connessione a broker reali, nessun ordine reale. Questa
pagina mostra solo cosa avrebbe fatto, retrospettivamente, un bot che segue
il modello predittivo del Modulo 2 — riallenato periodicamente sui dati via
via disponibili, mai sul futuro. Riusa la stessa logica già testata da riga
di comando in scripts/simulate_paper_trading.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import plotly.graph_objects as go
import streamlit as st

from src.config import load_settings
from src.dashboard.loader import MARKET_MODULES, load_data, lookback_selector, symbol_options_for
from src.execution.model_signal import generate_walkforward_signal
from src.execution.paper_broker import run_bot_simulation
from src.indicators.technical import add_all_indicators
from src.models.features import build_feature_matrix

st.set_page_config(page_title="Bot Simulato", layout="wide")
st.title("Bot di trading simulato")
st.error(
    "**Simulazione, non trading reale.** Nessuna connessione a broker, nessun "
    "ordine viene davvero eseguito. Il bot si riallena periodicamente sui dati "
    "storici via via disponibili (mai sul futuro) e mostra cosa avrebbe fatto "
    "— è uno strumento didattico, non un consiglio di investimento.",
    icon="⚠️",
)

settings = load_settings()
watchlist = settings["watchlist"]

col_market, col_symbol, col_lookback = st.columns(3)
with col_market:
    market_label = st.selectbox("Mercato", list(MARKET_MODULES.keys()))
market_key = MARKET_MODULES[market_label]

with col_symbol:
    symbol = st.selectbox("Strumento", symbol_options_for(market_key, watchlist))

with col_lookback:
    lookback = lookback_selector(market_key, key_prefix="bot")

col_warmup, col_retrain, col_sizing, col_threshold = st.columns(4)
with col_warmup:
    min_train_size = st.number_input(
        "Periodi di warm-up", min_value=50, max_value=400, value=150,
        help="Quanti periodi di storia servono prima che il bot inizi a operare.",
    )
with col_retrain:
    retrain_every = st.number_input(
        "Riallena ogni N periodi", min_value=5, max_value=100, value=20,
    )
with col_sizing:
    sizing = st.selectbox(
        "Dimensionamento posizione", ["confidence", "binary"],
        help="'confidence': esposizione proporzionale a quanto il modello è convinto. "
             "'binary': posizione piena o nulla in base a una soglia.",
    )
with col_threshold:
    confidence_threshold = st.slider(
        "Soglia di confidenza", min_value=0.5, max_value=0.9, value=0.5, step=0.01,
        help="Sopra questa probabilità di rialzo il bot inizia ad aprire posizione.",
    )

col_capital, col_cost = st.columns(2)
with col_capital:
    initial_capital = st.number_input("Capitale virtuale iniziale", min_value=100, value=10_000, step=100)
with col_cost:
    transaction_cost_bps = st.slider("Costo di transazione (bps)", min_value=0, max_value=50, value=5)

run_clicked = st.button("Avvia simulazione", type="primary")

if run_clicked:
    try:
        with st.spinner(f"Carico dati per {symbol}..."):
            df = load_data(market_key, symbol, "1d", lookback=lookback)
            df = add_all_indicators(df)
        X, y_direction, _ = build_feature_matrix(df, horizon=1)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Errore nel caricamento dati o nel calcolo delle feature: {exc}")
        st.stop()

    if len(X) <= min_train_size:
        st.warning(
            f"Solo {len(X)} periodi disponibili, meno del warm-up richiesto "
            f"({min_train_size}): riduci il warm-up o scegli un periodo più lungo."
        )
        st.stop()

    with st.spinner("Simulo il bot (riallenamento periodico + esecuzione)..."):
        signal = generate_walkforward_signal(
            X, y_direction,
            min_train_size=int(min_train_size),
            retrain_every=int(retrain_every),
            sizing=sizing,
            confidence_threshold=float(confidence_threshold),
        )
        simulation = run_bot_simulation(
            df, signal,
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
        )

    st.subheader("Metriche della simulazione")
    metrics_cols = st.columns(len(simulation.result.metrics))
    for col, (k, v) in zip(metrics_cols, simulation.result.metrics.items()):
        col.metric(k, v)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=simulation.result.equity_curve.index, y=simulation.result.equity_curve.values,
        name="Bot simulato",
    ))
    fig.add_trace(go.Scatter(
        x=simulation.result.benchmark_equity_curve.index,
        y=simulation.result.benchmark_equity_curve.values,
        name="Buy & Hold (benchmark)", line=dict(dash="dash", color="gray"),
    ))
    fig.update_layout(
        title=f"Bot simulato vs Buy & Hold — {symbol}",
        yaxis_title=f"Capitale (partenza: {initial_capital:,.0f})",
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Log operazioni simulate ({len(simulation.trade_log)} operazioni)")
    if simulation.trade_log.empty:
        st.info("Nessuna operazione: il bot non ha mai raggiunto la soglia di confidenza impostata.")
    else:
        st.dataframe(simulation.trade_log, use_container_width=True)
else:
    st.info("Imposta i parametri e premi \"Avvia simulazione\".")
