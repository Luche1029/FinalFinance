"""Valutazione fuori campione dei genomi sopravvissuti (Modulo 4, sperimentale).

Punto centrale, identico in spirito al Modulo 2: un algoritmo evolutivo su
dati storici finiti troverà quasi sempre genomi che sembrano vincenti nel
periodo su cui si è evoluto — è overfitting per costruzione, ancora più
insidioso che per un singolo modello statistico, perché la selezione
naturale premia esplicitamente qualunque cosa abbia funzionato, anche per
puro rumore. Per questo la popolazione evolve SOLO sulla finestra di
training (population.run_evolution); qui i genomi sopravvissuti vengono
congelati (nessuna ulteriore mutazione/riproduzione/morte) e testati su un
periodo successivo mai visto durante l'evoluzione, con la stessa identica
regola di ingresso/uscita ma senza possibilità di adattarsi nel frattempo.
"""
from __future__ import annotations

import pandas as pd

from src.backtesting.engine import compute_metrics
from src.evolution.genome import FleaGenome
from src.evolution.population import Flea, _check_exit


def simulate_single_genome(
    genome: FleaGenome,
    features_by_instrument: dict[str, pd.DataFrame],
    prices_by_instrument: dict[str, pd.Series],
    dates: pd.DatetimeIndex,
    initial_capital: float = 100.0,
    transaction_cost_bps: float = 5.0,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Fa "vivere" una singola pulce (genoma congelato) sull'arco di `dates`.

    A differenza di population.run_evolution, qui non c'è nascita, morte né
    mutazione: serve solo a rispondere "come si sarebbe comportato questo
    genoma, così com'è, su dati che non ha mai visto durante l'evoluzione?".

    Returns:
        (equity_curve, position_flag, trade_log): equity_curve è marcata a
        mercato ogni giorno (non solo alle chiusure di posizione, per avere
        un profilo di drawdown realistico); position_flag è 1 nei giorni in
        cui la pulce è investita, 0 altrimenti (serve a calcolare le metriche
        standard con backtesting.engine.compute_metrics); trade_log elenca
        gli ingressi/uscite.
    """
    capital = initial_capital
    instrument = None
    entry_price = None
    round_trip_cost = transaction_cost_bps / 10_000.0 * 2

    equity_values = []
    position_values = []
    trade_log = []

    for date in dates:
        if instrument is not None:
            if date not in prices_by_instrument[instrument].index:
                equity_values.append(capital)
                position_values.append(1)
                continue

            price = prices_by_instrument[instrument].loc[date]
            feat_row = features_by_instrument[instrument].loc[date]
            score_now = genome.score(feat_row)
            unrealized = price / entry_price - 1
            mark_to_market = capital * (1 + unrealized)

            dummy_flea = Flea(id="_", genome=genome, capital=capital, birth_capital=capital,
                               instrument=instrument, entry_price=entry_price)
            if _check_exit(dummy_flea, price, score_now):
                capital = mark_to_market * (1 - round_trip_cost)
                trade_log.append({
                    "data": date, "azione": "SELL", "strumento": instrument,
                    "prezzo": round(price, 4), "capitale": round(capital, 2),
                })
                instrument, entry_price = None, None
                equity_values.append(capital)
                position_values.append(0)
            else:
                equity_values.append(mark_to_market)
                position_values.append(1)
        else:
            best_symbol, best_score = None, float("-inf")
            for symbol, feat_df in features_by_instrument.items():
                if date not in feat_df.index:
                    continue
                score = genome.score(feat_df.loc[date])
                if score > best_score:
                    best_score, best_symbol = score, symbol

            if best_symbol is not None and best_score > genome.entry_threshold:
                instrument = best_symbol
                entry_price = prices_by_instrument[best_symbol].loc[date]
                trade_log.append({
                    "data": date, "azione": "BUY", "strumento": instrument,
                    "prezzo": round(entry_price, 4), "capitale": round(capital, 2),
                })
                position_values.append(1)
            else:
                position_values.append(0)
            equity_values.append(capital)

    equity_curve = pd.Series(equity_values, index=dates, name="equity")
    position_flag = pd.Series(position_values, index=dates, name="position")
    return equity_curve, position_flag, pd.DataFrame(trade_log)


def equal_weight_benchmark(
    prices_by_instrument: dict[str, pd.Series], dates: pd.DatetimeIndex, initial_capital: float = 100.0
) -> pd.Series:
    """Benchmark: capitale diviso in parti uguali tra tutti gli strumenti del
    mercato delle pulci e mai più toccato (buy & hold su un paniere), per un
    confronto onesto — la pulce può scegliere tra più strumenti, quindi il
    termine di paragone giusto non è un solo strumento ma l'intero paniere.
    """
    per_instrument_capital = initial_capital / len(prices_by_instrument)
    curves = []
    for symbol, prices in prices_by_instrument.items():
        aligned = prices.reindex(dates).ffill()
        normalized = per_instrument_capital * (aligned / aligned.iloc[0])
        curves.append(normalized)
    return pd.concat(curves, axis=1).sum(axis=1).rename("benchmark_equity")


def evaluate_survivors(
    survivors: list,
    features_by_instrument: dict[str, pd.DataFrame],
    prices_by_instrument: dict[str, pd.Series],
    out_of_sample_dates: pd.DatetimeIndex,
    top_k: int = 10,
    initial_capital: float = 100.0,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Valuta i migliori genomi sopravvissuti all'evoluzione su dati mai visti.

    Args:
        survivors: popolazione finale restituita da population.run_evolution.
        top_k: quante pulci valutare (le migliori per capitale accumulato
            durante l'evoluzione — valutarle tutte sarebbe costoso e in gran
            parte ridondante, molte si somigliano per parentela).

    Returns:
        DataFrame con una riga per genoma valutato: performance durante
        l'evoluzione (capitale finale, numero di figli — proxy di quanto è
        stato "premiato" dalla selezione naturale) e performance fuori
        campione (le metriche standard del progetto). Il confronto tra le
        due colonne è il punto più importante: un genoma che ha fatto bene
        in evoluzione ma male fuori campione è un caso da manuale di
        overfitting, non un'anomalia da correggere nel codice.
    """
    top_survivors = sorted(survivors, key=lambda f: f.capital, reverse=True)[:top_k]

    rows = []
    for rank, flea in enumerate(top_survivors, start=1):
        equity_curve, position_flag, trade_log = simulate_single_genome(
            flea.genome, features_by_instrument, prices_by_instrument, out_of_sample_dates,
            initial_capital=initial_capital, transaction_cost_bps=transaction_cost_bps,
        )
        daily_returns = equity_curve.pct_change().fillna(0)
        metrics = compute_metrics(daily_returns, position_flag)

        row = {
            "rank": rank,
            "id_pulce": flea.id,
            "generazione": flea.generation,
            "figli_durante_evoluzione": flea.n_children,
            "capitale_fine_evoluzione": round(flea.capital, 2),
        }
        row.update({f"oos_{k}": v for k, v in metrics.items()})
        row["oos_capitale_finale"] = round(equity_curve.iloc[-1], 2) if len(equity_curve) else None
        rows.append(row)

    return pd.DataFrame(rows)
