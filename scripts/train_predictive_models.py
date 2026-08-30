"""Addestra e valuta i modelli predittivi baseline (direzione + rendimento)
con validazione walk-forward, confrontandoli sempre con un baseline naive.

Uso:
    python scripts/train_predictive_models.py --market equities --symbol AAPL --period 2y

Il risultato da leggere con più attenzione è la riga finale (media sui fold):
se "modello_accuracy" non supera chiaramente "naive_majority_accuracy" (o
"naive_persistence_accuracy"), il modello non sta aggiungendo informazione
utile rispetto a un baseline banale — conclusione legittima quanto un
risultato positivo, non un fallimento del codice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import crypto, equities, forex
from src.indicators.technical import add_all_indicators
from src.models.baseline import (
    LogisticDirectionModelCV,
    NaiveMajorityClassifier,
    NaiveMeanReturnRegressor,
    NaivePersistenceClassifier,
    NaiveZeroReturnRegressor,
    RidgeReturnModel,
)
from src.models.evaluate import (
    classification_metrics,
    regression_metrics,
    run_walk_forward_comparison,
)
from src.models.features import build_feature_matrix


def load_data(market: str, symbol: str, period: str):
    if market == "equities":
        return equities.get_ohlcv(symbol, period=period, interval="1d")
    if market == "crypto":
        return crypto.get_ohlcv(symbol, timeframe="1d", limit=500)
    if market == "forex":
        return forex.get_ohlcv(symbol, interval="1d", outputsize=500)
    raise ValueError(f"Mercato non riconosciuto: {market}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="equities", choices=["equities", "crypto", "forex"])
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--period", default="2y", help="usato solo per equities")
    parser.add_argument("--horizon", type=int, default=1, help="periodi futuri da prevedere")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    df = load_data(args.market, args.symbol, args.period)
    df = add_all_indicators(df)
    X, y_direction, y_return = build_feature_matrix(df, horizon=args.horizon)

    print(f"\n=== Modelli predittivi — {args.symbol} ({args.market}) ===")
    print(f"Osservazioni utilizzabili dopo pulizia: {len(X)} (feature: {list(X.columns)})\n")

    print("--- Classificazione: direzione del prezzo (Logistic + regolarizzazione L2) ---")
    class_report = run_walk_forward_comparison(
        X, y_direction,
        model_factory=LogisticDirectionModelCV,
        naive_factories={
            "naive_majority": NaiveMajorityClassifier,
            "naive_persistence": NaivePersistenceClassifier,
        },
        metrics_fn=classification_metrics,
        n_splits=args.n_splits,
    )
    class_df = class_report.as_dataframe()
    print(class_df.round(3).to_string(index=False))
    print("\nMedia sui fold:")
    print(class_report.mean_metrics().round(3).to_string())

    print("\n--- Regressione: rendimento atteso (Ridge) ---")
    reg_report = run_walk_forward_comparison(
        X, y_return,
        model_factory=RidgeReturnModel,
        naive_factories={
            "naive_zero": NaiveZeroReturnRegressor,
            "naive_mean": NaiveMeanReturnRegressor,
        },
        metrics_fn=regression_metrics,
        n_splits=args.n_splits,
    )
    reg_df = reg_report.as_dataframe()
    print(reg_df.round(5).to_string(index=False))
    print("\nMedia sui fold:")
    print(reg_report.mean_metrics().round(5).to_string())


if __name__ == "__main__":
    main()
