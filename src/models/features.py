"""Feature engineering per i modelli predittivi del Modulo 2.

Regola non negoziabile per evitare leakage: ogni feature alla riga t deve
essere calcolabile usando solo informazioni disponibili fino a t (compreso).
Il target, invece, guarda deliberatamente in avanti di `horizon` periodi:
è quello che vogliamo prevedere, non una feature.

Versione 2 — normalizzazione delle feature per ridurre overfitting.
La v1 usava sma_*/ema_*/bb_* così come restituiti da add_all_indicators():
valori in unità di prezzo (es. sma_200 di AAPL è "~185 dollari"). Sono feature
non stazionarie — la loro scala cresce/scende col prezzo dell'azione nel
tempo — e fortemente collineari tra loro (sma_20, sma_50, sma_200 ed ema_12,
ema_26 si muovono quasi insieme, essendo tutte medie dello stesso prezzo).
Il primo test su AAPL con quelle feature grezze (18 feature, 100-250 righe di
training per fold) ha dato un modello di regressione peggiore del baseline
naive: sintomo classico di overfitting su feature ridondanti.

Qui le trasformiamo in quantità relative e adimensionali (es. "il prezzo è il
3% sopra la sua media a 20 periodi" invece di "la media a 20 periodi vale
185.3"), che generalizzano meglio tra strumenti e nel tempo, e riduciamo il
numero di lag di rendimento per tagliare ridondanza.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_matrix(
    df_with_indicators: pd.DataFrame,
    horizon: int = 1,
    lag_periods: tuple[int, ...] = (1, 5, 10),
    volatility_window: int = 10,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Costruisce feature normalizzate (X) e due target (direzione, rendimento).

    Args:
        df_with_indicators: DataFrame OHLCV già arricchito da
            indicators.technical.add_all_indicators().
        horizon: numero di periodi futuri su cui calcolare il target
            (1 = rendimento/direzione del giorno successivo).
        lag_periods: rendimenti passati (in periodi) da includere come feature.
            Ridotti rispetto alla v1 (era 1,2,3,5,10) per limitare la
            ridondanza tra lag molto vicini tra loro.
        volatility_window: finestra per la volatilità storica rolling,
            usata come proxy del regime di rischio corrente.

    Returns:
        (X, y_direction, y_return): X è la matrice di feature (tutte
        adimensionali/relative, confrontabili tra strumenti diversi),
        y_direction è 1/0 (rialzo/ribasso nei prossimi `horizon` periodi),
        y_return è il rendimento percentuale nello stesso orizzonte. Le righe
        con NaN (warm-up degli indicatori o target non calcolabile a fine
        serie) sono già rimosse e gli indici restano allineati.
    """
    df = df_with_indicators.copy()
    close = df["close"]

    features: dict[str, pd.Series] = {}

    for lag in lag_periods:
        features[f"return_lag_{lag}"] = close.pct_change(lag)

    features[f"volatility_{volatility_window}"] = close.pct_change().rolling(volatility_window).std()

    # Medie mobili: distanza percentuale del prezzo dalla media, non il
    # livello assoluto della media. "close/sma - 1" è stazionaria e
    # confrontabile tra strumenti con prezzi molto diversi tra loro.
    for col in df.columns:
        if col.startswith("sma_") or col.startswith("ema_"):
            features[f"price_vs_{col}"] = close / df[col] - 1

    # RSI è già limitato in [0, 100]: lo riportiamo in [0, 1] solo per
    # coerenza di scala con le altre feature, il contenuto informativo non cambia.
    for col in df.columns:
        if col.startswith("rsi_"):
            features[col] = df[col] / 100.0

    # MACD: teniamo solo l'istogramma (differenza tra macd e la sua signal
    # line, la componente più informativa sul momentum) e lo normalizziamo
    # per il prezzo, altrimenti il suo valore assoluto dipende dalla scala
    # dei prezzi dello strumento.
    hist_cols = [c for c in df.columns if c.startswith("macd_hist_")]
    for col in hist_cols:
        features[f"{col}_norm"] = df[col] / close

    # Bollinger: la posizione del prezzo dentro la banda (0 = banda
    # inferiore, 1 = banda superiore) e l'ampiezza relativa della banda,
    # al posto dei tre livelli assoluti (upper/mid/lower) che sono
    # praticamente ridondanti col prezzo stesso.
    upper_cols = [c for c in df.columns if c.startswith("bb_upper_")]
    for upper_col in upper_cols:
        period = upper_col.replace("bb_upper_", "")
        lower_col = f"bb_lower_{period}"
        mid_col = f"bb_mid_{period}"
        if lower_col in df.columns and mid_col in df.columns:
            band_width = df[upper_col] - df[lower_col]
            features[f"bb_position_{period}"] = (close - df[lower_col]) / band_width
            features[f"bb_width_{period}"] = band_width / df[mid_col]

    X = pd.DataFrame(features, index=df.index)

    # Target: guarda avanti di `horizon` periodi rispetto alla riga corrente.
    future_return = close.shift(-horizon) / close - 1
    y_return = future_return.rename(f"target_return_{horizon}")
    y_direction = (future_return > 0).astype(int).rename(f"target_direction_{horizon}")

    combined = pd.concat([X, y_direction, y_return], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    X_clean = combined[X.columns]
    y_direction_clean = combined[y_direction.name]
    y_return_clean = combined[y_return.name]

    return X_clean, y_direction_clean, y_return_clean
