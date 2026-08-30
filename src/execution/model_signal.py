"""Generatore di segnali "simil-AI" con retraining periodico (Modulo 3, dummy).

Differenza rispetto alla valutazione walk-forward del Modulo 2
(models/evaluate.py): lì il modello viene allenato una volta per fold e
valutato su un blocco di test, per misurare quanto è affidabile in media.
Qui invece si simula un bot che "vive" nel tempo: parte da un modello
allenato sui primi `min_train_size` periodi, decide una posizione periodo per
periodo, e si riallena da zero ogni `retrain_every` periodi sommando i dati
nel frattempo diventati disponibili — come farebbe un sistema automatico
tenuto in esercizio, che periodicamente "impara" dai dati più recenti.

Vincolo non negoziabile identico al resto del progetto: per decidere la
posizione al periodo t si può allenare solo su dati fino a t-1 (mai su t,
il cui target dipende dal prezzo a t+1, non ancora "accaduto" dal punto di
vista della simulazione).
"""
from __future__ import annotations

import pandas as pd

from src.models.baseline import LogisticDirectionModelCV


def generate_walkforward_signal(
    X: pd.DataFrame,
    y_direction: pd.Series,
    min_train_size: int = 150,
    retrain_every: int = 20,
    sizing: str = "confidence",
    confidence_threshold: float = 0.5,
) -> pd.Series:
    """Simula un bot che decide una posizione periodo per periodo.

    Args:
        X, y_direction: feature e target di direzione, output di
            models.features.build_feature_matrix (horizon=1 consigliato,
            così il segnale al periodo t corrisponde esattamente al
            rendimento t -> t+1 che il motore di backtest si aspetta).
        min_train_size: quanti periodi iniziali servono prima che il bot
            inizi a operare (prima di allora resta flat: non ha ancora
            abbastanza storia per allenare un modello affidabile).
        retrain_every: ogni quanti periodi il modello viene riallenato da
            zero sui dati nel frattempo accumulati.
        sizing: "confidence" pesa la posizione in base a quanto il modello è
            convinto (0 se incerto, fino a 1 se molto sicuro di un rialzo);
            "binary" apre posizione piena (1) o nulla (0) in base a una sola
            soglia di probabilità, più simile a un interruttore acceso/spento.
        confidence_threshold: soglia di probabilità sopra cui aprire
            posizione (usata da entrambe le modalità: in "confidence" è il
            punto sotto cui l'esposizione è comunque zero).

    Returns:
        Serie di posizione nell'intervallo [0, 1] (long-only: niente short),
        indicizzata come X, pronta per essere passata a
        backtesting.engine.run_backtest come colonna "signal".
    """
    if sizing not in ("confidence", "binary"):
        raise ValueError("sizing deve essere 'confidence' o 'binary'")

    signal = pd.Series(0.0, index=X.index)
    model: LogisticDirectionModelCV | None = None

    for i in range(min_train_size, len(X)):
        if model is None or (i - min_train_size) % retrain_every == 0:
            model = LogisticDirectionModelCV()
            model.fit(X.iloc[:i], y_direction.iloc[:i])

        proba_up = model.predict_proba_up(X.iloc[[i]])[0]

        if sizing == "binary":
            signal.iloc[i] = 1.0 if proba_up >= confidence_threshold else 0.0
        else:
            # Esposizione proporzionale alla convinzione sopra la soglia:
            # proba_up == threshold -> 0, proba_up == 1.0 -> esposizione piena.
            if proba_up > confidence_threshold:
                headroom = 1.0 - confidence_threshold
                signal.iloc[i] = min(1.0, (proba_up - confidence_threshold) / headroom) if headroom > 0 else 1.0
            else:
                signal.iloc[i] = 0.0

    return signal.rename("signal")
