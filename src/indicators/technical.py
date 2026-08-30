"""Indicatori tecnici su un DataFrame OHLCV standard (colonne: open, high, low, close, volume).

Formule implementate direttamente con pandas invece di appoggiarsi a librerie
come pandas-ta: a fine 2024 pandas-ta ha avuto una rottura di compatibilità
(numpy.NaN rinominato) e le versioni più recenti richiedono Python 3.12+,
rendendola una dipendenza fragile per un progetto pensato per durare nel tempo.
Le formule di SMA/EMA/RSI/MACD/Bollinger sono standard e stabili: implementarle
qui evita di dipendere da un pacchetto terzo per calcoli comunque semplici.

I periodi di default sono letti da config/settings.yaml.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_settings


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    """RSI con smoothing di Wilder (lo standard usato dalla maggior parte delle piattaforme)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        f"macd_{fast}_{slow}_{signal}": macd_line,
        f"macd_signal_{fast}_{slow}_{signal}": signal_line,
        f"macd_hist_{fast}_{slow}_{signal}": histogram,
    })


def bollinger_bands(series: pd.Series, period: int, std_dev: float) -> pd.DataFrame:
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    return pd.DataFrame({
        f"bb_upper_{period}": mid + std_dev * std,
        f"bb_mid_{period}": mid,
        f"bb_lower_{period}": mid - std_dev * std,
    })


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge al DataFrame tutte le colonne indicatore configurate.

    Non modifica il DataFrame originale; ne ritorna una copia arricchita.
    """
    settings = load_settings()["indicators"]
    out = df.copy()
    close = out["close"]

    for period in settings["sma_periods"]:
        out[f"sma_{period}"] = sma(close, period)

    for period in settings["ema_periods"]:
        out[f"ema_{period}"] = ema(close, period)

    rsi_period = settings["rsi_period"]
    out[f"rsi_{rsi_period}"] = rsi(close, rsi_period)

    fast, slow, signal = settings["macd"]
    out = out.join(macd(close, fast, slow, signal))

    bb_period = settings["bollinger"]["period"]
    bb_std = settings["bollinger"]["std_dev"]
    out = out.join(bollinger_bands(close, bb_period, bb_std))

    return out


def summarize_signals(df_with_indicators: pd.DataFrame) -> dict:
    """Legge l'ultima riga del DataFrame arricchito e ritorna una lettura
    testuale sintetica (ipercomprato/ipervenduto, posizione vs. medie mobili).

    Questa è una lettura descrittiva, NON un segnale di trading: serve solo
    a facilitare la lettura del grafico nella dashboard.
    """
    settings = load_settings()["indicators"]
    last = df_with_indicators.iloc[-1]
    rsi_col = f"rsi_{settings['rsi_period']}"

    summary = {}

    rsi_value = last.get(rsi_col)
    if pd.notna(rsi_value):
        if rsi_value >= 70:
            summary["rsi"] = f"RSI a {rsi_value:.1f}: zona di ipercomprato"
        elif rsi_value <= 30:
            summary["rsi"] = f"RSI a {rsi_value:.1f}: zona di ipervenduto"
        else:
            summary["rsi"] = f"RSI a {rsi_value:.1f}: nessun estremo"

    sma_cols = [c for c in df_with_indicators.columns if c.startswith("sma_")]
    above = [c for c in sma_cols if pd.notna(last.get(c)) and last["close"] > last[c]]
    summary["trend"] = (
        f"Prezzo sopra {len(above)}/{len(sma_cols)} medie mobili monitorate"
        if sma_cols else "Nessuna media mobile configurata"
    )

    return summary
