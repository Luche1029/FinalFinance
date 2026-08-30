"""Backtest delle strategie di esempio (SMA crossover, RSI mean-reversion).

Uso:
    python scripts/backtest_indicators.py --market equities --symbol AAPL --period 2y

Stampa le metriche di ciascuna strategia a confronto col benchmark
"buy & hold" (compra e tieni per tutto il periodo), e salva un grafico
dell'equity curve in outputs/backtest_<symbol>.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # nessun display grafico nell'ambiente di esecuzione
import matplotlib.pyplot as plt

from src.backtesting.engine import run_backtest
from src.backtesting.strategies import rsi_mean_reversion_signal, sma_crossover_signal
from src.data import crypto, equities, forex
from src.indicators.technical import add_all_indicators


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
    args = parser.parse_args()

    df = load_data(args.market, args.symbol, args.period)
    df = add_all_indicators(df)

    strategies = {
        "SMA 20/50 crossover": sma_crossover_signal(df, "sma_20", "sma_50"),
        "RSI 14 mean-reversion": rsi_mean_reversion_signal(df, "rsi_14"),
    }

    print(f"\n=== Backtest {args.symbol} ({args.market}) — {len(df)} periodi ===\n")

    fig, ax = plt.subplots(figsize=(10, 5))

    results = {}
    for name, signal in strategies.items():
        df_strategy = df.copy()
        df_strategy["signal"] = signal
        result = run_backtest(df_strategy, signal_col="signal")
        results[name] = result

        print(f"--- {name} ---")
        for k, v in result.metrics.items():
            print(f"  {k}: {v}")
        print()

        ax.plot(result.equity_curve.index, result.equity_curve.values, label=name)

    # Benchmark buy & hold, usando l'ultima result calcolata (stesso periodo per tutte)
    any_result = next(iter(results.values()))
    ax.plot(
        any_result.benchmark_equity_curve.index,
        any_result.benchmark_equity_curve.values,
        label="Buy & Hold (benchmark)",
        linestyle="--",
        color="gray",
    )

    ax.set_title(f"Equity curve — {args.symbol}")
    ax.set_ylabel("Capitale (partenza: 10.000)")
    ax.legend()
    fig.tight_layout()

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / f"backtest_{args.symbol.replace('/', '-')}.png"
    fig.savefig(out_path, dpi=120)
    print(f"Grafico salvato in: {out_path}")


if __name__ == "__main__":
    main()
