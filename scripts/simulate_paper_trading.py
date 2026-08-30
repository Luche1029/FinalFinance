"""Simula un bot di trading automatico "dummy" (Modulo 3) su dati storici.

NESSUNA connessione a broker reali, nessun ordine reale: è una simulazione
retrospettiva che risponde alla domanda "cosa avrebbe fatto un bot basato su
questo modello?", riallenando periodicamente su dati via via disponibili
(mai sul futuro) e pesando le posizioni in base a quanto il modello è
convinto.

Uso:
    python scripts/simulate_paper_trading.py --market equities --symbol AAPL --period 2y
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import crypto, equities, forex
from src.execution.model_signal import generate_walkforward_signal
from src.execution.paper_broker import run_bot_simulation
from src.indicators.technical import add_all_indicators
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
    parser.add_argument("--min-train-size", type=int, default=150)
    parser.add_argument("--retrain-every", type=int, default=20)
    parser.add_argument("--sizing", choices=["confidence", "binary"], default="confidence")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    print(
        "\n*** SIMULAZIONE — nessun ordine reale, nessuna connessione a broker. "
        "Questo script mostra solo cosa avrebbe fatto un bot ipotetico su dati storici. ***\n"
    )

    df = load_data(args.market, args.symbol, args.period)
    df = add_all_indicators(df)
    X, y_direction, _ = build_feature_matrix(df, horizon=1)

    print(f"=== Bot simulato — {args.symbol} ({args.market}) ===")
    print(f"Periodi totali disponibili per la simulazione: {len(X)}")
    print(
        f"Il bot inizia a operare dopo {args.min_train_size} periodi di storia "
        f"(warm-up) e si riallena ogni {args.retrain_every} periodi.\n"
    )

    signal = generate_walkforward_signal(
        X, y_direction,
        min_train_size=args.min_train_size,
        retrain_every=args.retrain_every,
        sizing=args.sizing,
        confidence_threshold=args.confidence_threshold,
    )

    simulation = run_bot_simulation(
        df, signal,
        initial_capital=args.capital,
        transaction_cost_bps=args.cost_bps,
    )

    print("--- Metriche della simulazione ---")
    for k, v in simulation.result.metrics.items():
        print(f"  {k}: {v}")

    print(f"\n--- Log operazioni simulate ({len(simulation.trade_log)} operazioni) ---")
    if simulation.trade_log.empty:
        print("  Nessuna operazione: il bot non ha mai raggiunto la soglia di confidenza.")
    else:
        print(simulation.trade_log.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(simulation.result.equity_curve.index, simulation.result.equity_curve.values, label="Bot simulato")
    ax.plot(
        simulation.result.benchmark_equity_curve.index,
        simulation.result.benchmark_equity_curve.values,
        label="Buy & Hold (benchmark)", linestyle="--", color="gray",
    )
    ax.set_title(f"Bot simulato vs Buy & Hold — {args.symbol}")
    ax.set_ylabel(f"Capitale (partenza: {args.capital:,.0f})")
    ax.legend()
    fig.tight_layout()

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"bot_simulato_{args.symbol.replace('/', '-')}.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nGrafico salvato in: {out_path}")


if __name__ == "__main__":
    main()
