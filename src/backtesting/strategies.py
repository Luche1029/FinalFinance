"""Strategie di esempio basate sugli indicatori tecnici del Modulo 1.

Servono soprattutto a validare il motore di backtesting end-to-end con
qualcosa di concreto, e come termine di paragone "semplice" quando poi
arriveranno i modelli predittivi: se un modello complesso non batte queste
strategie banali, non vale la complessità aggiunta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma_crossover_signal(df: pd.DataFrame, fast_col: str, slow_col: str) -> pd.Series:
    """Trend-following: long quando la media veloce è sopra quella lenta.

    Strategia "long-only, flat altrimenti": niente posizioni corte, coerente
    con chi opera principalmente su azioni/ETF (dove shortare richiede
    strumenti specifici, es. CFD, con rischi ulteriori discussi nella guida).
    """
    if fast_col not in df.columns or slow_col not in df.columns:
        raise ValueError(f"Colonne '{fast_col}' e/o '{slow_col}' non trovate.")

    signal = (df[fast_col] > df[slow_col]).astype(int)
    signal[df[fast_col].isna() | df[slow_col].isna()] = 0
    return signal.rename("signal")


def rsi_mean_reversion_signal(
    df: pd.DataFrame,
    rsi_col: str,
    oversold: float = 30,
    overbought: float = 70,
) -> pd.Series:
    """Mean-reversion: entra long in ipervenduto, esce in ipercomprato.

    A differenza dello SMA crossover, qui la posizione "tiene memoria": una
    volta entrati long su ipervenduto, si resta in posizione finché il prezzo
    non raggiunge la zona di ipercomprato (altrimenti si uscirebbe e
    rientrerebbe ad ogni piccola oscillazione attorno alla soglia).
    """
    if rsi_col not in df.columns:
        raise ValueError(f"Colonna '{rsi_col}' non trovata.")

    rsi = df[rsi_col]
    raw_signal = pd.Series(np.nan, index=df.index)
    raw_signal[rsi <= oversold] = 1
    raw_signal[rsi >= overbought] = 0

    signal = raw_signal.ffill().fillna(0)
    return signal.rename("signal")
