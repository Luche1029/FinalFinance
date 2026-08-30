"""Preparazione dati per il Mercato delle Pulci: feature standardizzate,
allineamento del calendario tra strumenti, split evoluzione/test.

La standardizzazione (media 0, varianza 1) è calcolata SOLO sulla finestra di
training e poi applicata anche al periodo di test: esattamente come per gli
scaler dei modelli del Modulo 2, calcolarla su tutto il dataset (training +
test insieme) sarebbe una forma sottile di leakage — il genoma "saprebbe"
qualcosa sulla distribuzione dei dati futuri (media e varianza) che non
dovrebbe ancora conoscere.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evolution.genome import select_core_genes
from src.models.features import build_feature_matrix


def build_flea_market_dataset(
    dfs_by_instrument: dict[str, pd.DataFrame],
    train_fraction: float = 0.7,
) -> dict:
    """Costruisce il dataset multi-strumento pronto per evoluzione + test.

    Args:
        dfs_by_instrument: {simbolo: DataFrame OHLCV con indicatori già
            calcolati (add_all_indicators)}.
        train_fraction: quota di date (in ordine cronologico) usata per la
            fase di evoluzione; il resto è il periodo fuori campione.

    Returns:
        dict con: features_by_instrument (standardizzate, gene_features come
        colonne), prices_by_instrument (close, stesso indice), gene_features,
        train_dates, test_dates, scaler (per eventuale riuso/ispezione).
    """
    raw_features = {}
    prices = {}
    for symbol, df in dfs_by_instrument.items():
        X, _, _ = build_feature_matrix(df, horizon=1)
        raw_features[symbol] = X
        prices[symbol] = df["close"]

    gene_features = select_core_genes(next(iter(raw_features.values())).columns.tolist())

    common_index = None
    for X in raw_features.values():
        common_index = X.index if common_index is None else common_index.intersection(X.index)
    common_index = common_index.sort_values()

    if len(common_index) < 100:
        raise ValueError(
            f"Solo {len(common_index)} date in comune tra gli strumenti scelti: "
            "troppo poco per una simulazione sensata. Scegli strumenti con più storia in comune."
        )

    split_point = int(len(common_index) * train_fraction)
    train_dates = common_index[:split_point]
    test_dates = common_index[split_point:]

    pooled_train = pd.concat(
        [raw_features[s].loc[raw_features[s].index.intersection(train_dates), gene_features] for s in raw_features],
        axis=0,
    )
    scaler = StandardScaler().fit(pooled_train.values)

    features_by_instrument = {}
    prices_by_instrument = {}
    for symbol, X in raw_features.items():
        idx = X.index.intersection(common_index)
        sub = X.loc[idx, gene_features]
        standardized = pd.DataFrame(scaler.transform(sub.values), index=idx, columns=gene_features)
        features_by_instrument[symbol] = standardized
        prices_by_instrument[symbol] = prices[symbol].reindex(idx)

    return {
        "features_by_instrument": features_by_instrument,
        "prices_by_instrument": prices_by_instrument,
        "gene_features": gene_features,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "scaler": scaler,
    }
