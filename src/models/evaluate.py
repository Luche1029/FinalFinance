"""Valutazione dei modelli predittivi con validazione walk-forward.

L'output centrale non è "il modello ha accuracy X", ma "il modello ha
accuracy X contro Y del baseline naive, su N fold walk-forward" — il confronto
è la parte che conta, molto più del numero assoluto.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.metrics import mean_squared_error

from src.backtesting.validation import walk_forward_splits


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }


@dataclass
class WalkForwardReport:
    fold_metrics: list = field(default_factory=list)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.fold_metrics)

    def mean_metrics(self) -> pd.Series:
        df = self.as_dataframe()
        numeric_cols = df.select_dtypes(include="number").columns
        return df[numeric_cols].mean()


def run_walk_forward_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    naive_factories: dict,
    metrics_fn,
    n_splits: int = 5,
    min_train_size: int = 100,
) -> WalkForwardReport:
    """Allena e valuta un modello e uno o più baseline naive sugli stessi fold.

    Args:
        X, y: feature e target allineati (output di features.build_feature_matrix).
        model_factory: funzione senza argomenti che ritorna un'istanza nuova
            del modello da valutare (serve una istanza nuova per ogni fold,
            altrimenti si riusano pesi appresi su fold precedenti).
        naive_factories: dict {nome: funzione_factory} per i baseline naive
            da valutare sugli stessi fold, per confronto diretto.
        metrics_fn: classification_metrics o regression_metrics.
        n_splits: numero di fold walk-forward.
        min_train_size: train set minimo per considerare valido un fold.

    Returns:
        WalkForwardReport con le metriche di ogni fold, per il modello e per
        ciascun baseline naive.
    """
    report = WalkForwardReport()

    for fold in walk_forward_splits(len(X), n_splits=n_splits, min_train_size=min_train_size):
        X_train, X_test = X.iloc[fold.train_idx], X.iloc[fold.test_idx]
        y_train, y_test = y.iloc[fold.train_idx], y.iloc[fold.test_idx]

        row = {"fold": fold.fold_id, "n_train": len(X_train), "n_test": len(X_test)}

        model = model_factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        for k, v in metrics_fn(y_test, pred).items():
            row[f"modello_{k}"] = v

        for name, factory in naive_factories.items():
            naive = factory()
            naive.fit(X_train, y_train)
            naive_pred = naive.predict(X_test)
            for k, v in metrics_fn(y_test, naive_pred).items():
                row[f"{name}_{k}"] = v

        report.fold_metrics.append(row)

    return report
