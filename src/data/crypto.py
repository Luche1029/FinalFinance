"""Dati di mercato crypto via ccxt (libreria unificata per decine di exchange).

Usiamo l'endpoint pubblico di mercato (nessuna API key richiesta) per i soli
dati storici OHLCV: sufficiente per analisi e backtesting.
"""
from __future__ import annotations

import warnings

import pandas as pd
import ccxt

from src.config import load_settings
from src.data.cache import read_cached, read_cached_stale, write_cache
from src.data.equities import OHLCV_COLUMNS

# Mappa timeframe del progetto -> formato richiesto da ccxt
_TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h",
    "1d": "1d", "1wk": "1w",
}


def _get_exchange():
    settings = load_settings()
    exchange_id = settings["exchanges"]["crypto_default"]
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def get_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 365,
    use_cache: bool = True,
    cache_ttl_seconds: int = 900,
) -> pd.DataFrame:
    """Scarica dati OHLCV per una coppia crypto (es. "BTC/USDT").

    Args:
        symbol: coppia in formato ccxt, es. "BTC/USDT", "ETH/USDT".
        timeframe: granularità ("1m", "5m", "15m", "1h", "1d", "1wk").
        limit: numero di candele da recuperare.
        use_cache: se True, riusa dati recenti dalla cache locale.
        cache_ttl_seconds: validità della cache (bassa di default: il crypto
            si muove 24/7 ed è meno sensato tenere dati vecchi a lungo).

    Returns:
        DataFrame indicizzato per data, colonne: open, high, low, close, volume.
    """
    cache_key = f"crypto_{symbol}_{timeframe}_{limit}"

    if use_cache:
        cached = read_cached(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached is not None:
            return cached

    try:
        exchange = _get_exchange()
        ccxt_timeframe = _TIMEFRAME_MAP.get(timeframe, timeframe)

        raw = exchange.fetch_ohlcv(symbol, timeframe=ccxt_timeframe, limit=limit)
        if not raw:
            raise ValueError(f"Nessun dato restituito per '{symbol}' su {exchange.id}.")

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("date")[OHLCV_COLUMNS]
    except Exception as exc:
        stale = read_cached_stale(cache_key) if use_cache else None
        if stale is not None:
            warnings.warn(
                f"Download live per '{symbol}' fallito ({exc}); uso dati in "
                "cache non aggiornati. Verifica la connessione di rete."
            )
            return stale
        raise ValueError(
            f"Nessun dato restituito per '{symbol}' e nessuna cache disponibile. "
            f"Verifica il simbolo e la connessione di rete. Errore originale: {exc}"
        ) from exc

    if use_cache:
        write_cache(cache_key, df)

    return df
