"""Validazione walk-forward per serie temporali.

Perché non usare la cross-validation standard (k-fold con shuffle): mischiare
dati passati e futuri tra train e test crea look-ahead bias — il modello
"impara dal futuro" e le metriche risultano artificialmente ottime, per poi
deludere in produzione. Il walk-forward impone che ogni fold di test segua
cronologicamente il proprio fold di train, mai il contrario.

Usiamo TimeSeriesSplit di scikit-learn (che rispetta già questo vincolo)
tramite un wrapper che restituisce split leggibili e riutilizzabili sia per
strategie su indicatori sia per modelli predittivi.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    min_train_size: int | None = None,
) -> Iterator[Fold]:
    """Genera fold train/test in ordine cronologico, senza sovrapposizioni.

    Args:
        n_samples: numero totale di osservazioni (righe) della serie.
        n_splits: numero di fold di test consecutivi da generare.
        min_train_size: se indicato, scarta i primi fold il cui train set è
            troppo corto per essere significativo (es. troppo pochi giorni
            per stimare un modello in modo affidabile).

    Yields:
        Fold con gli indici (posizionali, non le date) di train e test.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(np.arange(n_samples))):
        if min_train_size is not None and len(train_idx) < min_train_size:
            continue
        yield Fold(fold_id=fold_id, train_idx=train_idx, test_idx=test_idx)
