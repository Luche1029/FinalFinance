"""Motore di backtesting vettoriale per strategie basate su segnali di posizione.

Principio cardine per evitare il look-ahead bias: un segnale generato con le
informazioni disponibili alla chiusura del giorno t può essere eseguito solo
a partire dal rendimento t -> t+1, mai su quello che l'ha generato. Per questo
il segnale viene sempre shiftato di 1 periodo prima di essere moltiplicato per
il rendimento dell'asset (vedi `run_backtest`).

Non è un motore per esecuzione live: serve a valutare onestamente su dati
storici se una strategia (basata su indicatori o su un modello predittivo)
avrebbe avuto senso, prima di pensare a qualunque automazione.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    strategy_returns: pd.Series
    benchmark_equity_curve: pd.Series
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{k}: {v}" for k, v in self.metrics.items()]
        return "\n".join(lines)


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    initial_capital: float = 10_000.0,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """Esegue il backtest di una serie di segnali di posizione.

    Args:
        df: DataFrame con almeno le colonne [price_col, signal_col], indicizzato
            per data e ordinato cronologicamente.
        signal_col: colonna con la posizione desiderata in ogni istante
            (1 = long, 0 = flat, -1 = short). Il segnale al tempo t rappresenta
            una decisione presa "a bocce ferme" con i dati disponibili fino a t.
        price_col: colonna del prezzo di riferimento (tipicamente 'close').
        initial_capital: capitale iniziale nominale, solo per la equity curve.
        transaction_cost_bps: costo di transazione in basis points (1 bps =
            0.01%) applicato ogni volta che la posizione cambia, per
            approssimare spread/commissioni. Con 5 bps un turnover frequente
            penalizza visibilmente il rendimento, come nella realtà.

    Returns:
        BacktestResult con equity curve, rendimenti e metriche di sintesi.
    """
    if signal_col not in df.columns:
        raise ValueError(f"Colonna segnale '{signal_col}' non trovata nel DataFrame.")

    asset_return = df[price_col].pct_change()

    # Il cuore anti-look-ahead: il segnale calcolato a fine giornata t si applica
    # al rendimento che matura tra t e t+1, mai a quello tra t-1 e t.
    position = df[signal_col].shift(1).fillna(0)

    turnover = position.diff().abs().fillna(position.abs())
    transaction_cost = turnover * (transaction_cost_bps / 10_000.0)

    strategy_return = position * asset_return - transaction_cost
    strategy_return = strategy_return.fillna(0)

    equity_curve = initial_capital * (1 + strategy_return).cumprod()
    benchmark_equity_curve = initial_capital * (1 + asset_return.fillna(0)).cumprod()

    metrics = compute_metrics(strategy_return, position)

    return BacktestResult(
        equity_curve=equity_curve,
        strategy_returns=strategy_return,
        benchmark_equity_curve=benchmark_equity_curve,
        metrics=metrics,
    )


def compute_metrics(strategy_return: pd.Series, position: pd.Series) -> dict:
    """Metriche di sintesi standard del progetto (usate anche fuori da
    run_backtest, es. dal Modulo 4 per valutare i genomi sopravvissuti su un
    equity curve costruita diversamente).
    """
    n_periods = len(strategy_return)
    if n_periods == 0:
        return {}

    total_return = (1 + strategy_return).prod() - 1
    years = n_periods / TRADING_DAYS_PER_YEAR
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else float("nan")

    ann_vol = strategy_return.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (
        (strategy_return.mean() * TRADING_DAYS_PER_YEAR) / ann_vol
        if ann_vol > 0 else float("nan")
    )
    # Nota: Sharpe calcolato con tasso privo di rischio = 0, una semplificazione
    # comune ma da tenere a mente confrontando con benchmark che lo stimano diversamente.

    equity = (1 + strategy_return).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    in_position = position != 0
    win_rate = (
        (strategy_return[in_position] > 0).mean()
        if in_position.sum() > 0 else float("nan")
    )
    n_trades = int(position.diff().abs().gt(0).sum())

    return {
        "rendimento_totale_%": round(total_return * 100, 2),
        "CAGR_%": round(cagr * 100, 2),
        "volatilita_annualizzata_%": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_%": round(max_drawdown * 100, 2),
        "win_rate_%": round(win_rate * 100, 2) if pd.notna(win_rate) else None,
        "numero_operazioni": n_trades,
    }
