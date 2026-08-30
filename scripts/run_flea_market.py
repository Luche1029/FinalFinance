"""Mercato delle Pulci — algoritmo evolutivo di ricerca strategie (Modulo 4, sperimentale).

NESSUNA connessione a broker reali, nessun ordine reale. Una popolazione di
"pulci" (agenti) con capitale virtuale indipendente sceglie strumenti da
"aggredire" secondo un genoma di pesi che evolve per selezione naturale:
le pulci redditizie accumulano capitale e si riproducono (il figlio eredita
il genoma del genitore, mutato); quelle in perdita muoiono. La popolazione
evolve SOLO sulla prima parte dei dati storici; i migliori sopravvissuti
vengono poi "congelati" e testati su un periodo successivo mai visto,
per capire se hanno trovato pattern reali o solo rumore del passato.

Uso:
    python scripts/run_flea_market.py --market forex --symbols "EUR/USD,GBP/USD,USD/JPY"
    python scripts/run_flea_market.py --market crypto --symbols "BTC/USDT,ETH/USDT"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import crypto, equities, forex
from src.evolution.data_prep import build_flea_market_dataset
from src.evolution.evaluate import equal_weight_benchmark, evaluate_survivors
from src.evolution.population import run_evolution
from src.indicators.technical import add_all_indicators


def load_instrument(market: str, symbol: str, period: str) -> pd.DataFrame:
    if market == "equities":
        df = equities.get_ohlcv(symbol, period=period, interval="1d")
    elif market == "crypto":
        df = crypto.get_ohlcv(symbol, timeframe="1d", limit=500)
    elif market == "forex":
        df = forex.get_ohlcv(symbol, interval="1d", outputsize=500)
    else:
        raise ValueError(f"Mercato non riconosciuto: {market}")
    return add_all_indicators(df)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="forex", choices=["equities", "crypto", "forex"])
    parser.add_argument("--symbols", default="EUR/USD,GBP/USD,USD/JPY", help="lista separata da virgole")
    parser.add_argument("--period", default="2y", help="usato solo per equities")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--n-initial", type=int, default=25)
    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--death-frac", type=float, default=0.5)
    parser.add_argument("--reproduction-frac", type=float, default=2.0)
    parser.add_argument("--max-population", type=int, default=200)
    parser.add_argument("--mutation-sigma", type=float, default=0.15)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    print(
        "\n*** SIMULAZIONE — nessun ordine reale, nessuna connessione a broker. "
        "Ricerca di strategie per selezione naturale su dati storici. ***\n"
    )
    print(f"Mercato: {args.market} — strumenti: {symbols}\n")

    dfs = {symbol: load_instrument(args.market, symbol, args.period) for symbol in symbols}
    dataset = build_flea_market_dataset(dfs, train_fraction=args.train_fraction)

    print(f"Date totali in comune tra gli strumenti: {len(dataset['train_dates']) + len(dataset['test_dates'])}")
    print(f"  Periodo di evoluzione (training): {len(dataset['train_dates'])} periodi "
          f"({dataset['train_dates'][0].date()} -> {dataset['train_dates'][-1].date()})")
    print(f"  Periodo fuori campione (test):    {len(dataset['test_dates'])} periodi "
          f"({dataset['test_dates'][0].date()} -> {dataset['test_dates'][-1].date()})")
    print(f"  Geni (feature) usati dal genoma: {dataset['gene_features']}\n")

    print("--- Fase 1: evoluzione della popolazione (solo su dati di training) ---")
    survivors, history = run_evolution(
        features_by_instrument=dataset["features_by_instrument"],
        prices_by_instrument=dataset["prices_by_instrument"],
        dates=dataset["train_dates"],
        gene_features=dataset["gene_features"],
        n_initial=args.n_initial,
        initial_capital=args.initial_capital,
        death_frac=args.death_frac,
        reproduction_frac=args.reproduction_frac,
        max_population=args.max_population,
        mutation_sigma=args.mutation_sigma,
        transaction_cost_bps=args.cost_bps,
        seed=args.seed,
    )

    pop_history = history.population_over_time
    print(f"Popolazione finale: {len(survivors)} pulci sopravvissute "
          f"(partite da {args.n_initial}, {len(history.births)} nascite, {len(history.deaths)} morti totali)")
    if not pop_history.empty:
        print(f"Capitale totale della popolazione: {pop_history['capitale_totale'].iloc[-1]:.2f} "
              f"(partito da {args.n_initial * args.initial_capital:.2f})\n")

    if not survivors:
        print("Estinzione totale della popolazione durante l'evoluzione: nessun genoma da valutare.")
        return

    print(f"--- Fase 2: valutazione dei migliori {min(args.top_k, len(survivors))} sopravvissuti fuori campione ---")
    results = evaluate_survivors(
        survivors,
        features_by_instrument=dataset["features_by_instrument"],
        prices_by_instrument=dataset["prices_by_instrument"],
        out_of_sample_dates=dataset["test_dates"],
        top_k=args.top_k,
        initial_capital=args.initial_capital,
        transaction_cost_bps=args.cost_bps,
    )
    print(results.to_string(index=False))

    benchmark = equal_weight_benchmark(dataset["prices_by_instrument"], dataset["test_dates"], args.initial_capital)
    benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
    print(f"\nBenchmark (paniere equamente diviso, buy & hold, stesso periodo fuori campione): "
          f"{benchmark_return:+.2f}%")

    # Grafico: storia della popolazione durante l'evoluzione
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if not pop_history.empty:
        axes[0].plot(pop_history["data"], pop_history["n_pulci"])
        axes[0].set_title("Numero di pulci nel tempo (evoluzione)")
        axes[0].set_ylabel("N. pulci")

        axes[1].plot(pop_history["data"], pop_history["capitale_totale"])
        axes[1].set_title("Capitale totale della popolazione (evoluzione)")
        axes[1].set_ylabel("Capitale virtuale")
    fig.tight_layout()

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"mercato_pulci_{args.market}.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nGrafico della popolazione salvato in: {out_path}")


if __name__ == "__main__":
    main()
