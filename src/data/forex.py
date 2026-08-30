"""Dati di mercato forex.

A differenza di azioni e crypto, non esiste una fonte gratuita e affidabile
senza API key per il forex. Questo modulo usa Twelve Data (tier gratuito:
~800 richieste/giorno) e richiede una API key personale, gratuita su
https://twelvedata.com.

Impostare la chiave come variabile d'ambiente prima di avviare la dashboard:
    export TWELVEDATA_API_KEY="la-tua-chiave"

Finché la chiave non è configurata, get_ohlcv solleva un errore esplicito
invece di fallire in modo silenzioso: preferiamo che l'utente sappia subito
cosa manca piuttosto che vedere un grafico vuoto.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd
import requests

from src.data.cache import read_cached, read_cached_stale, write_cache
from src.data.equities import OHLCV_COLUMNS

API_URL = "https://api.twelvedata.com/time_series"

_INTERVAL_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h",
    "1d": "1day", "1wk": "1week",
}


def get_ohlcv(
    symbol: str,
    interval: str = "1d",
    outputsize: int = 365,
    use_cache: bool = True,
    cache_ttl_seconds: int = 3600,
) -> pd.DataFrame:
    """Scarica dati OHLCV forex (es. "EUR/USD") da Twelve Data.

    Richiede TWELVEDATA_API_KEY nell'ambiente. Vedi docstring del modulo.
    """
    api_key = os.environ.get("TWELVEDATA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TWELVEDATA_API_KEY non impostata. Registrati gratuitamente su "
            "twelvedata.com e imposta la variabile d'ambiente prima di "
            "richiedere dati forex."
        )

    cache_key = f"forex_{symbol}_{interval}_{outputsize}"
    if use_cache:
        cached = read_cached(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached is not None:
            return cached

    try:
        params = {
            "symbol": symbol,
            "interval": _INTERVAL_MAP.get(interval, interval),
            "outputsize": outputsize,
            "apikey": api_key,
            "format": "JSON",
        }
        response = requests.get(API_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()

        if "values" not in payload:
            raise ValueError(f"Risposta inattesa da Twelve Data per '{symbol}': {payload}")

        df = pd.DataFrame(payload["values"])
        df["date"] = pd.to_datetime(df["datetime"])
        df = df.set_index("date").sort_index()
        df = df.rename(columns=str.lower)[OHLCV_COLUMNS[:-1]].astype(float)
        df["volume"] = 0  # il forex spot non ha un volume centralizzato affidabile
    except Exception as exc:
        stale = read_cached_stale(cache_key) if use_cache else None
        if stale is not None:
            warnings.warn(
                f"Download live per '{symbol}' fallito ({exc}); uso dati in "
                "cache non aggiornati. Verifica la connessione di rete o la API key."
            )
            return stale
        raise ValueError(
            f"Impossibile ottenere dati per '{symbol}' e nessuna cache disponibile. "
            f"Errore originale: {exc}"
        ) from exc

    if use_cache:
        write_cache(cache_key, df)

    return df
