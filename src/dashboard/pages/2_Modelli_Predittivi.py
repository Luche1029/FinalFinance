"""Pagina Streamlit — Modelli predittivi baseline (Modulo 2).

Riusa la stessa pipeline testata da riga di comando in
scripts/train_predictive_models.py: feature normalizzate, modelli
regolarizzati (Ridge / LogisticRegressionCV), validazione walk-forward,
confronto sempre con un baseline naive.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import streamlit as st

from src.config import load_settings
from src.dashboard.loader import MARKET_MODULES, load_data, lookback_selector, symbol_options_for
from src.indicators.technical import add_all_indicators
from src.models.baseline import (
    LogisticDirectionModelCV,
    NaiveMajorityClassifier,
    NaiveMeanReturnRegressor,
    NaivePersistenceClassifier,
    NaiveZeroReturnRegressor,
    RidgeReturnModel,
)
from src.models.evaluate import classification_metrics, regression_metrics, run_walk_forward_comparison
from src.models.features import build_feature_matrix

st.set_page_config(page_title="Modelli predittivi", layout="wide")
st.title("Modelli predittivi baseline")
st.caption(
    "Regressione logistica (direzione) e Ridge (rendimento atteso), entrambe "
    "regolarizzate e validate con walk-forward, confrontate con un baseline "
    "naive. Il numero che conta è il confronto modello-vs-naive, non il "
    "valore assoluto: se il modello non batte chiaramente il naive, non sta "
    "aggiungendo informazione utile — è una conclusione onesta, non un errore."
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
    lookback = lookback_selector(market_key, key_prefix="modelli")

col_horizon, col_splits = st.columns(2)
with col_horizon:
    horizon = st.number_input("Orizzonte di previsione (periodi)", min_value=1, max_value=20, value=1)
with col_splits:
    n_splits = st.number_input("Numero di fold walk-forward", min_value=3, max_value=10, value=5)

run_clicked = st.button("Allena e valuta i modelli", type="primary")

if run_clicked:
    try:
        with st.spinner(f"Carico dati per {symbol}..."):
            df = load_data(market_key, symbol, "1d", lookback=lookback)
            df = add_all_indicators(df)
        X, y_direction, y_return = build_feature_matrix(df, horizon=int(horizon))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Errore nel caricamento dati o nel calcolo delle feature: {exc}")
        st.stop()

    st.write(f"Osservazioni utilizzabili dopo pulizia: **{len(X)}** — feature: {', '.join(X.columns)}")

    if len(X) < 150:
        st.warning(
            "Poche osservazioni disponibili per una validazione walk-forward "
            "affidabile: i risultati vanno interpretati con molta cautela."
        )

    with st.spinner("Alleno e valuto i modelli sui fold walk-forward..."):
        class_report = run_walk_forward_comparison(
            X, y_direction,
            model_factory=LogisticDirectionModelCV,
            naive_factories={
                "naive_majority": NaiveMajorityClassifier,
                "naive_persistence": NaivePersistenceClassifier,
            },
            metrics_fn=classification_metrics,
            n_splits=int(n_splits),
        )
        reg_report = run_walk_forward_comparison(
            X, y_return,
            model_factory=RidgeReturnModel,
            naive_factories={
                "naive_zero": NaiveZeroReturnRegressor,
                "naive_mean": NaiveMeanReturnRegressor,
            },
            metrics_fn=regression_metrics,
            n_splits=int(n_splits),
        )

    st.subheader("Classificazione: direzione del prezzo")
    class_mean = class_report.mean_metrics()
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy modello", f"{class_mean.get('modello_accuracy', float('nan')):.3f}")
    col2.metric("Accuracy naive (majority)", f"{class_mean.get('naive_majority_accuracy', float('nan')):.3f}")
    col3.metric("Accuracy naive (persistence)", f"{class_mean.get('naive_persistence_accuracy', float('nan')):.3f}")
    with st.expander("Dettaglio per fold"):
        st.dataframe(class_report.as_dataframe().round(3))

    st.subheader("Regressione: rendimento atteso")
    reg_mean = reg_report.mean_metrics()
    col1, col2, col3 = st.columns(3)
    col1.metric("R² modello", f"{reg_mean.get('modello_r2', float('nan')):.3f}")
    col2.metric("R² naive (zero)", f"{reg_mean.get('naive_zero_r2', float('nan')):.3f}")
    col3.metric("R² naive (media)", f"{reg_mean.get('naive_mean_r2', float('nan')):.3f}")
    with st.expander("Dettaglio per fold"):
        st.dataframe(reg_report.as_dataframe().round(5))
else:
    st.info("Imposta i parametri e premi \"Allena e valuta i modelli\".")
