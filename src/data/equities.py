"""Dati di mercato per azioni ed ETF, via yfinance.

yfinance non richiede API key ed è sufficiente per uso personale/educativo.
Per uso intensivo o professionale andrebbe sostituito con un provider a pagamento
con SLA garantiti (yfinance dipende da endpoint non ufficiali di Yahoo Finance).
"""
from __future__ import annotations

import warnings

import pandas as pd
import yfinance as yf

from src.data.cache import read_cached, read_cached_stale, write_cache

# Colonne standard che tutti i moduli data.* devono restituire, indipendentemente
# dalla fonte (equities, crypto, forex), così indicators/ e dashboard/ non devono
# sapere da dove arrivano i dati.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def get_ohlcv(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    use_cache: bool = True,
    cache_ttl_seconds: int = 3600,
) -> pd.DataFrame:
    """Scarica dati OHLCV per un ticker azionario/ETF.

    Args:
        symbol: ticker Yahoo Finance (es. "AAPL", "ENEL.MI", "VWCE.DE").
        period: finestra storica ("1mo", "6mo", "1y", "5y", "max", ...).
        interval: granularità ("1m", "5m", "1h", "1d", "1wk", ...).
        use_cache: se True, riusa dati recenti dalla cache locale.
        cache_ttl_seconds: validità della cache in secondi.

    Returns:
        DataFrame indicizzato per data, colonne: open, high, low, close, volume.
    """
    cache_key = f"equity_{symbol}_{period}_{interval}"

    if use_cache:
        cached = read_cached(cache_key, max_age_seconds=cache_ttl_seconds)
        if cached is not None:
            return cached

    try:
        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise ValueError(f"Nessun dato restituito per '{symbol}'.")
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
            "Verifica il ticker (es. i titoli italiani richiedono il suffisso "
            f".MI) e la connessione di rete. Errore originale: {exc}"
        ) from exc

    # yfinance restituisce colonne con MultiIndex quando si passano più ticker;
    # con un singolo ticker normalizziamo comunque i nomi in minuscolo.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]

    df = raw[OHLCV_COLUMNS].copy()
    df.index.name = "date"

    if use_cache:
        write_cache(cache_key, df)

    return df


def get_fundamentals(symbol: str) -> dict:
    """Estrae alcuni indicatori fondamentali di base (P/E, market cap, settore)."""
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    return {
        "nome": info.get("longName"),
        "settore": info.get("sector"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "eps": info.get("trailingEps"),
        "dividend_yield": info.get("dividendYield"),
    }
