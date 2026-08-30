"""Mercato delle Pulci — meccanica di popolazione (Modulo 4, sperimentale).

Simula, giorno per giorno, una popolazione di "pulci" (agenti) che si
attaccano a uno strumento alla volta usando il proprio genoma (genome.py)
per decidere quando entrare e uscire. Il capitale di ogni pulce è un conto
virtuale indipendente (nessuna leva, nessuno short, un solo strumento alla
volta). Nessuna connessione a broker reali, nessun ordine reale — è una
ricerca di strategie per selezione naturale su dati storici, concettualmente
imparentata col Modulo 3 ma con un meccanismo di apprendimento diverso
(evoluzione di una popolazione invece di discesa del gradiente su un solo
modello).

Regola anti-look-ahead identica al resto del progetto: la decisione al
giorno t usa solo feature/prezzi noti a t; il capitale si aggiorna solo
quando una posizione viene chiusa (nessun "sguardo nel futuro" nemmeno per
calcolare quando una pulce muore o si riproduce).
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field

import pandas as pd

from src.evolution.genome import FleaGenome


@dataclass
class Flea:
    id: str
    genome: FleaGenome
    capital: float
    birth_capital: float          # riferimento per le soglie di morte/riproduzione
    parent_id: str | None = None
    generation: int = 0
    n_children: int = 0

    # stato di posizione corrente
    instrument: str | None = None
    entry_price: float | None = None
    entry_date: object | None = None


@dataclass
class SimulationHistory:
    population_over_time: pd.DataFrame   # data, n_pulci, capitale_totale, capitale_medio
    births: list                          # log delle nascite (data, id, parent_id, strumento genitore)
    deaths: list                          # log delle morti (data, id, capitale_finale)


def spawn_initial_population(
    n: int, gene_features: list[str], initial_capital: float, rng: random.Random
) -> list[Flea]:
    return [
        Flea(
            id=str(uuid.uuid4())[:8],
            genome=FleaGenome.random_init(gene_features, rng),
            capital=initial_capital,
            birth_capital=initial_capital,
            generation=0,
        )
        for _ in range(n)
    ]


def _check_exit(flea: Flea, price: float, score_now: float) -> bool:
    change = price / flea.entry_price - 1
    if change <= -flea.genome.stop_loss_pct:
        return True
    if change >= flea.genome.take_profit_pct:
        return True
    if score_now < flea.genome.entry_threshold:
        return True
    return False


def run_evolution(
    features_by_instrument: dict[str, pd.DataFrame],
    prices_by_instrument: dict[str, pd.Series],
    dates: pd.DatetimeIndex,
    gene_features: list[str],
    n_initial: int = 25,
    initial_capital: float = 100.0,
    death_frac: float = 0.5,
    reproduction_frac: float = 2.0,
    reproduction_split: float = 0.6,
    max_population: int = 200,
    mutation_sigma: float = 0.15,
    transaction_cost_bps: float = 5.0,
    seed: int = 42,
) -> tuple[list[Flea], SimulationHistory]:
    """Fa evolvere la popolazione di pulci sull'arco di `dates`.

    Args:
        features_by_instrument: {simbolo: DataFrame feature standardizzate},
            stesse colonne per tutti gli strumenti, indicizzate per data.
        prices_by_instrument: {simbolo: Series di prezzi close}, stesso indice.
        dates: date su cui far girare la simulazione, in ordine cronologico
            (tipicamente il sottoinsieme "di training" — vedi evaluate.py per
            il test fuori campione sui genomi sopravvissuti).
        gene_features: nomi delle feature usate dai genomi (da
            genome.select_core_genes).
        n_initial: dimensione della popolazione di partenza.
        initial_capital: capitale virtuale assegnato ad ogni pulce alla nascita.
        death_frac / reproduction_frac: soglie di morte e riproduzione, come
            frazione del capitale alla nascita della singola pulce (non del
            capitale iniziale globale: ogni pulce ha il proprio riferimento,
            così una pulce nata da una riproduzione precedente con meno
            capitale non è svantaggiata rispetto alle pulci originarie).
        reproduction_split: frazione di capitale che resta al genitore alla
            riproduzione (il resto va al figlio).
        max_population: tetto oltre il quale le nuove nascite vengono bloccate.
        mutation_sigma: ampiezza tipica della mutazione dei pesi alla nascita.
        transaction_cost_bps: costo di transazione simulato per operazione
            (round-trip, applicato all'uscita).
        seed: seme per la riproducibilità.

    Returns:
        (popolazione_finale, storia) — la popolazione sopravvissuta a fine
        periodo (da validare fuori campione) e uno storico per grafici/analisi.
    """
    rng = random.Random(seed)
    population = spawn_initial_population(n_initial, gene_features, initial_capital, rng)

    pop_history_rows = []
    births_log = []
    deaths_log = []
    round_trip_cost = transaction_cost_bps / 10_000.0 * 2

    for date in dates:
        births: list[Flea] = []
        deaths_ids: set[str] = set()

        for flea in population:
            if flea.instrument is not None:
                if date not in features_by_instrument[flea.instrument].index:
                    continue  # strumento senza dato in questa data (calendari non perfettamente allineati)

                price = prices_by_instrument[flea.instrument].loc[date]
                feat_row = features_by_instrument[flea.instrument].loc[date]
                score_now = flea.genome.score(feat_row)

                if _check_exit(flea, price, score_now):
                    realized_return = price / flea.entry_price - 1
                    flea.capital *= (1 + realized_return) * (1 - round_trip_cost)
                    flea.instrument = None
                    flea.entry_price = None
                    flea.entry_date = None

                    if flea.capital < death_frac * flea.birth_capital:
                        deaths_ids.add(flea.id)
                        deaths_log.append({"data": date, "id": flea.id, "capitale_finale": round(flea.capital, 2)})
                    elif (
                        flea.capital > reproduction_frac * flea.birth_capital
                        and len(population) + len(births) < max_population
                    ):
                        child_capital = flea.capital * (1 - reproduction_split)
                        flea.capital *= reproduction_split
                        flea.birth_capital = flea.capital  # nuovo riferimento dopo lo split
                        flea.n_children += 1

                        child = Flea(
                            id=str(uuid.uuid4())[:8],
                            genome=flea.genome.mutate(rng, sigma=mutation_sigma),
                            capital=child_capital,
                            birth_capital=child_capital,
                            parent_id=flea.id,
                            generation=flea.generation + 1,
                        )
                        births.append(child)
                        births_log.append({"data": date, "id": child.id, "parent_id": flea.id})
            else:
                best_symbol, best_score = None, float("-inf")
                for symbol, feat_df in features_by_instrument.items():
                    if date not in feat_df.index:
                        continue
                    score = flea.genome.score(feat_df.loc[date])
                    if score > best_score:
                        best_score, best_symbol = score, symbol

                if best_symbol is not None and best_score > flea.genome.entry_threshold:
                    flea.instrument = best_symbol
                    flea.entry_price = prices_by_instrument[best_symbol].loc[date]
                    flea.entry_date = date

        if deaths_ids:
            population = [f for f in population if f.id not in deaths_ids]
        population.extend(births)

        total_capital = sum(f.capital for f in population)
        pop_history_rows.append({
            "data": date,
            "n_pulci": len(population),
            "capitale_totale": round(total_capital, 2),
            "capitale_medio": round(total_capital / len(population), 2) if population else 0,
        })

        if not population:
            break  # estinzione: nessuna pulce sopravvissuta, la simulazione si ferma qui

    history = SimulationHistory(
        population_over_time=pd.DataFrame(pop_history_rows),
        births=births_log,
        deaths=deaths_log,
    )
    return population, history
