"""Genoma delle pulci — Mercato delle Pulci (Modulo 4, sperimentale).

Ogni pulce possiede un genoma: un piccolo insieme di pesi che combinano
alcune feature già pronte (Modulo 2, src/models/features.py) in un punteggio
per uno strumento, più tre soglie comportamentali (quando entrare, quando
tagliare le perdite, quando incassare il profitto). È deliberatamente
piccolo (6 pesi + 3 soglie = 9 geni): con una popolazione che evolve
liberamente, un genoma grande troverebbe quasi sempre combinazioni che
sembrano vincenti nel passato per puro rumore, lo stesso problema di
overfitting già affrontato (e risolto) nel Modulo 2 con la riduzione delle
feature — qui il rischio è ancora più alto perché non c'è una singola
funzione di perdita da minimizzare ma una selezione naturale che premia
qualunque cosa abbia funzionato, anche per caso.

Le feature usate non sono hardcoded per nome esatto (dipendono dai periodi
configurati in config/settings.yaml) ma selezionate per "famiglia" tramite
prefisso, così il genoma resta valido anche cambiando la configurazione.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def select_core_genes(feature_columns: list[str]) -> list[str]:
    """Sceglie un sottoinsieme piccolo e diversificato di feature per il genoma.

    Una feature per famiglia (lag di rendimento più corto, volatilità, un
    solo price_vs_sma, rsi, macd normalizzato, posizione nelle Bollinger):
    evita di dare al genoma più copie quasi-ridondanti della stessa
    informazione (es. price_vs_sma_20 e price_vs_sma_50 sono molto correlate).
    """
    families = {
        "return_lag_": None,
        "volatility_": None,
        "price_vs_sma_": None,
        "rsi_": None,
        "macd_hist_": None,
        "bb_position_": None,
    }
    for col in sorted(feature_columns):
        for prefix in families:
            if col.startswith(prefix) and families[prefix] is None:
                families[prefix] = col

    selected = [col for col in families.values() if col is not None]
    if len(selected) < 3:
        raise ValueError(
            "Troppe poche feature riconosciute per costruire un genoma "
            f"sensato (trovate: {selected}). Colonne disponibili: {feature_columns}"
        )
    return selected


@dataclass
class FleaGenome:
    """Il "DNA" di una pulce: come valuta gli strumenti e come si comporta.

    Attributes:
        gene_features: nomi delle feature usate (stessi per tutte le pulci
            della simulazione, solo i pesi cambiano).
        weights: peso per ciascuna feature in gene_features, stesso ordine.
            Positivo = "quando questa feature è alta, mi piace di più lo
            strumento"; negativo = il contrario.
        entry_threshold: punteggio minimo (dopo standardizzazione delle
            feature) per decidere di attaccarsi a uno strumento. Usato anche
            come soglia di uscita: se il punteggio dello strumento a cui è
            attaccata scende sotto questa soglia, la pulce si stacca.
        stop_loss_pct: perdita percentuale dall'ingresso oltre la quale la
            pulce si stacca comunque, indipendentemente dal punteggio.
        take_profit_pct: guadagno percentuale dall'ingresso oltre il quale
            la pulce incassa ed esce.
    """

    gene_features: list[str]
    weights: np.ndarray
    entry_threshold: float
    stop_loss_pct: float
    take_profit_pct: float

    @classmethod
    def random_init(cls, gene_features: list[str], rng: random.Random) -> "FleaGenome":
        weights = np.array([rng.gauss(0, 1) for _ in gene_features])
        return cls(
            gene_features=gene_features,
            weights=weights,
            entry_threshold=rng.uniform(0.0, 1.0),
            stop_loss_pct=rng.uniform(0.02, 0.15),
            take_profit_pct=rng.uniform(0.02, 0.20),
        )

    def score(self, standardized_features: pd.Series) -> float:
        """Punteggio dello strumento secondo questa pulce (feature già standardizzate)."""
        values = standardized_features[self.gene_features].to_numpy(dtype=float)
        return float(np.dot(self.weights, values))

    def mutate(self, rng: random.Random, sigma: float = 0.15) -> "FleaGenome":
        """Crea una copia mutata del genoma (per un figlio alla riproduzione).

        Perturbazione gaussiana su ogni gene: piccole variazioni locali sono
        molto più frequenti di salti grandi, come nella mutazione biologica.
        Una piccola probabilità di mutazione "ampia" evita che la
        popolazione si blocchi tutta sullo stesso ottimo locale.
        """
        big_mutation = rng.random() < 0.1
        scale = sigma * (3.0 if big_mutation else 1.0)

        new_weights = self.weights + np.array([rng.gauss(0, scale) for _ in self.gene_features])
        new_entry = float(np.clip(self.entry_threshold + rng.gauss(0, scale), -2.0, 2.0))
        new_stop = float(np.clip(self.stop_loss_pct + rng.gauss(0, scale * 0.05), 0.01, 0.30))
        new_take = float(np.clip(self.take_profit_pct + rng.gauss(0, scale * 0.05), 0.01, 0.40))

        return FleaGenome(
            gene_features=self.gene_features,
            weights=new_weights,
            entry_threshold=new_entry,
            stop_loss_pct=new_stop,
            take_profit_pct=new_take,
        )

    def as_dict(self) -> dict:
        """Rappresentazione leggibile del genoma, per ispezione/logging."""
        d = {f"peso_{name}": round(w, 3) for name, w in zip(self.gene_features, self.weights)}
        d["soglia_ingresso"] = round(self.entry_threshold, 3)
        d["stop_loss_%"] = round(self.stop_loss_pct * 100, 1)
        d["take_profit_%"] = round(self.take_profit_pct * 100, 1)
        return d
