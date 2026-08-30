"""Modelli baseline per direzione (classificazione) e rendimento (regressione).

Il punto centrale di questo modulo non è "avere un modello ML", ma avere un
metro di paragone onesto: i mercati sono in gran parte imprevedibili nel
breve periodo (ipotesi di efficienza dei mercati), quindi un modello va
giudicato rispetto a un baseline naive, non in assoluto. Se un modello
complesso non batte il naive in modo consistente su più fold walk-forward,
la conclusione corretta è "non sta aggiungendo valore", non "serve un
modello ancora più complesso".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class NaiveMajorityClassifier:
    """Predice sempre la classe più frequente osservata nel train set.

    Baseline minimo per la classificazione: se il mercato analizzato ha un
    leggero trend di fondo (es. un indice azionario che sale più spesso di
    quanto scenda), anche "predici sempre rialzo" può avere accuracy > 50%.
    Qualunque modello va confrontato anche con questo, non solo con un
    ipotetico 50% casuale.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveMajorityClassifier":
        self.majority_class_ = int(y.mode().iloc[0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_class_)


class NaivePersistenceClassifier:
    """Predice che la direzione futura sarà uguale all'ultimo rendimento osservato.

    Si appoggia alla feature 'return_lag_1' se presente; altrimenti degrada
    al comportamento del NaiveMajorityClassifier.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaivePersistenceClassifier":
        self.fallback_ = int(y.mode().iloc[0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "return_lag_1" not in X.columns:
            return np.full(len(X), self.fallback_)
        return (X["return_lag_1"] > 0).astype(int).to_numpy()


class NaiveZeroReturnRegressor:
    """Predice sempre rendimento futuro = 0 (ipotesi di random walk).

    È il baseline standard in finanza: se il mercato è efficiente, il miglior
    predittore del prezzo di domani è il prezzo di oggi, cioè rendimento
    atteso nullo. Qualunque modello di regressione va confrontato con questo
    prima di essere considerato utile.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveZeroReturnRegressor":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class NaiveMeanReturnRegressor:
    """Predice sempre il rendimento medio osservato nel train set."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveMeanReturnRegressor":
        self.mean_return_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_return_)


class LogisticDirectionModel:
    """Regressione logistica "semplice" per la direzione (nessuna ricerca di
    iperparametri). Lasciata per confronto con la versione regolarizzata
    sotto: con poche feature andava già alla pari col baseline naive, ma con
    più feature tende a overfittare rapidamente.
    """

    def __init__(self, **kwargs):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticDirectionModel":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class LinearReturnModel:
    """Regressione lineare "semplice" per il rendimento atteso (nessuna
    regolarizzazione). Lasciata per confronto: è il modello che nel primo
    test su AAPL aveva un R² fortemente negativo per overfitting.
    """

    def __init__(self, **kwargs):
        self.scaler = StandardScaler()
        self.model = LinearRegression(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearReturnModel":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class LogisticDirectionModelCV:
    """Regressione logistica con selezione automatica della forza di
    regolarizzazione L2 (parametro C), scelta tramite validazione interna
    walk-forward sui soli dati di training del fold corrente — nessuna fuga
    di informazione dal test set.

    Con poche osservazioni per fold (100-250, il nostro caso), la
    regolarizzazione conta quanto la scelta dell'algoritmo: senza di essa
    il modello tende a "imparare a memoria" le fluttuazioni del training set
    invece di pattern che si ripetono.
    """

    def __init__(self, cs: int = 10, inner_splits: int = 3):
        self.scaler = StandardScaler()
        self.model = LogisticRegressionCV(
            Cs=cs,
            cv=TimeSeriesSplit(n_splits=inner_splits),
            max_iter=2000,
            scoring="accuracy",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticDirectionModelCV":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilità stimata di direzione "su" (classe 1), non solo la
        predizione binaria. Usata dal Modulo 3 per pesare le posizioni in
        base a quanto il modello è "convinto", invece di un tutto-o-niente.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class RidgeReturnModel:
    """Regressione Ridge (lineare + penalità L2) per il rendimento atteso,
    con la forza di regolarizzazione (alpha) scelta automaticamente tramite
    validazione interna walk-forward sui soli dati di training del fold.

    È il fix diretto al problema di overfitting osservato con
    LinearReturnModel: la penalità L2 costringe i coefficienti a restare
    piccoli, riducendo la sensibilità del modello al rumore del training set
    quando le feature sono tante rispetto alle osservazioni.
    """

    def __init__(self, alphas=None, inner_splits: int = 3):
        self.scaler = StandardScaler()
        alphas = alphas if alphas is not None else np.logspace(-3, 3, 13)
        self.model = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=inner_splits))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeReturnModel":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
