"""Paper broker simulato — Modulo 3 (dummy).

ATTENZIONE, da leggere prima di usare questo modulo: qui non c'è nessuna
connessione a un broker vero, nessuna API di trading, nessun ordine reale.
"Paper broker" è un nome convenzionale per un motore che simula l'esecuzione
di ordini su dati storici già noti, usando le stesse regole (costi di
transazione, niente look-ahead) del motore di backtesting del Modulo 2. Serve
a rispondere alla domanda "cosa avrebbe fatto un bot che si affida a questo
modello, operazione per operazione?", non a eseguire nulla di reale.

Prima di anche solo pensare a collegare un broker vero (Modulo 3 "non-dummy"),
la roadmap prevede mesi di questo tipo di simulazione con risultati
consistenti, poi un conto demo fornito da un broker reale (vedi guida
"Fiscalità e broker" e i capitoli su gestione del rischio) — mai capitale
reale come primo passo.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtesting.engine import BacktestResult, run_backtest


@dataclass
class TradingBotSimulation:
    result: BacktestResult
    trade_log: pd.DataFrame
    signal: pd.Series


def _build_trade_log(df: pd.DataFrame, signal_col: str, price_col: str) -> pd.DataFrame:
    """Ricostruisce un log leggibile delle operazioni simulate a partire dal segnale.

    Stessa logica di shift usata dal motore di backtest (la posizione decisa
    al tempo t si applica al rendimento t -> t+1): il log riporta quindi la
    posizione "in essere durante" ogni periodo, non quella decisa quel giorno.
    """
    position = df[signal_col].shift(1).fillna(0)
    price = df[price_col]

    rows = []
    prev_position = 0.0
    for date, pos in position.items():
        if abs(pos - prev_position) > 1e-9:
            if pos > prev_position:
                action = "BUY" if prev_position == 0 else "INCREASE"
            else:
                action = "SELL" if pos == 0 else "REDUCE"
            rows.append({
                "data": date,
                "azione": action,
                "prezzo": price.loc[date],
                "posizione_precedente": round(prev_position, 3),
                "posizione_nuova": round(pos, 3),
            })
        prev_position = pos

    return pd.DataFrame(rows)


def run_bot_simulation(
    df_with_indicators: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 10_000.0,
    transaction_cost_bps: float = 5.0,
) -> TradingBotSimulation:
    """Simula l'esecuzione di un bot che segue `signal`, periodo per periodo.

    Args:
        df_with_indicators: DataFrame OHLCV+indicatori originale (serve la
            colonna 'close' per i prezzi di esecuzione simulati).
        signal: posizione decisa dal modello ad ogni periodo (output di
            execution.model_signal.generate_walkforward_signal), indicizzata
            su un sottoinsieme di df_with_indicators (le righe sopravvissute
            al feature engineering del Modulo 2).
        initial_capital: capitale virtuale di partenza.
        transaction_cost_bps: costo di transazione simulato per operazione.

    Returns:
        TradingBotSimulation con il risultato del backtest (equity curve,
        metriche), il trade log leggibile e il segnale stesso.
    """
    df_aligned = df_with_indicators.loc[signal.index].copy()
    df_aligned["signal"] = signal

    result = run_backtest(
        df_aligned,
        signal_col="signal",
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
    )
    trade_log = _build_trade_log(df_aligned, signal_col="signal", price_col="close")

    return TradingBotSimulation(result=result, trade_log=trade_log, signal=signal)
