"""Caricamento dati condiviso tra tutte le pagine della dashboard Streamlit.

Estratto da app.py quando è stata aggiunta la navigazione multipagina
(Modulo 2): sia la pagina di analisi sia quelle di backtesting/modelli
predittivi devono scegliere mercato/strumento/timeframe e caricare lo stesso
tipo di DataFrame OHLCV, quindi la logica vive qui una volta sola.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data import crypto, equities, forex

MARKET_MODULES = {
    "Azioni / ETF": "equities",
    "Crypto": "crypto",
    "Forex": "forex",
}

# Opzioni di storico mostrate nei selettori delle pagine. Per le azioni/ETF
# sono stringhe "periodo" nel formato yfinance; per crypto/forex sono un
# numero di candele (entrambe le fonti lavorano per conteggio, non per data
# di inizio) — da qui il default diverso passato a load_data.
EQUITY_PERIOD_OPTIONS = ["3mo", "6mo", "1y", "2y", "5y", "max"]
EQUITY_DEFAULT_PERIOD = "2y"
CANDLE_COUNT_OPTIONS = [100, 200, 500, 1000, 2000]
CANDLE_COUNT_DEFAULT = 500


@st.cache_data(ttl=900, show_spinner=False)
def load_data(market_key: str, symbol: str, timeframe: str, lookback: str | int | None = None) -> pd.DataFrame:
    """Scarica (o legge dalla cache) i dati OHLCV per mercato/strumento/timeframe.

    Args:
        market_key: uno tra "equities", "crypto", "forex" (valori di
            MARKET_MODULES, non le etichette mostrate nell'interfaccia).
        symbol: simbolo nel formato atteso dalla fonte dati del mercato
            (es. "AAPL" per equities, "BTC/USDT" per crypto, "EUR/USD" per forex).
        timeframe: granularità ("1d", "1h", "15m", ...).
        lookback: quanto storico caricare. Per equities una stringa periodo
            yfinance (es. "6mo", "2y", "max" — vedi EQUITY_PERIOD_OPTIONS);
            per crypto/forex un numero di candele (vedi CANDLE_COUNT_OPTIONS).
            Se None, usa i default del progetto (2 anni per le equities, 500
            candele per crypto/forex) — utile per compatibilità con codice
            che non passa ancora questo parametro.
    """
    if market_key == "equities":
        period = lookback if lookback else EQUITY_DEFAULT_PERIOD
        return equities.get_ohlcv(symbol, period=period, interval=timeframe)
    if market_key == "crypto":
        limit = int(lookback) if lookback else CANDLE_COUNT_DEFAULT
        return crypto.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if market_key == "forex":
        outputsize = int(lookback) if lookback else CANDLE_COUNT_DEFAULT
        return forex.get_ohlcv(symbol, interval=timeframe, outputsize=outputsize)
    raise ValueError(market_key)


def default_lookback_for(market_key: str) -> str | int:
    """Valore di default per il selettore di storico, in base al mercato."""
    return EQUITY_DEFAULT_PERIOD if market_key == "equities" else CANDLE_COUNT_DEFAULT


def lookback_selector(market_key: str, key_prefix: str = "") -> str | int:
    """Widget di selezione storico, uguale su tutte le pagine.

    Azioni/ETF: periodo (yfinance). Crypto/forex: numero di candele (le due
    fonti dati lavorano per conteggio, non per intervallo di date). Il
    `key_prefix` evita collisioni tra widget con la stessa etichetta su
    pagine diverse di Streamlit.
    """
    if market_key == "equities":
        return st.selectbox(
            "Storico", EQUITY_PERIOD_OPTIONS,
            index=EQUITY_PERIOD_OPTIONS.index(EQUITY_DEFAULT_PERIOD),
            key=f"{key_prefix}_lookback",
        )
    return st.selectbox(
        "Numero di candele", CANDLE_COUNT_OPTIONS,
        index=CANDLE_COUNT_OPTIONS.index(CANDLE_COUNT_DEFAULT),
        key=f"{key_prefix}_lookback",
    )


def symbol_options_for(market_key: str, watchlist: dict) -> list[str]:
    """Ritorna la lista di simboli in watchlist per il mercato scelto."""
    return {
        "equities": watchlist["equities"] + watchlist["etfs"],
        "crypto": watchlist["crypto"],
        "forex": watchlist["forex"],
    }[market_key]
