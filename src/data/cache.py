"""Cache locale generica per DataFrame OHLCV, indipendente dalla fonte dati.

Evita di richiamare le API esterne (spesso rate-limited) ogni volta che si apre
la dashboard. La cache è a file Parquet, con un time-to-live configurabile.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.config import cache_dir


def _cache_path(key: str) -> Path:
    safe_key = key.replace("/", "-").replace(" ", "_")
    return cache_dir() / f"{safe_key}.parquet"


def read_cached(key: str, max_age_seconds: int = 3600) -> pd.DataFrame | None:
    """Ritorna il DataFrame in cache se esiste ed è più recente di max_age_seconds."""
    path = _cache_path(key)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > max_age_seconds:
        return None
    return pd.read_parquet(path)


def read_cached_stale(key: str) -> pd.DataFrame | None:
    """Ritorna il DataFrame in cache indipendentemente dall'età.

    Da usare solo come fallback quando una chiamata API fallisce (rete assente,
    rate limit, outage temporaneo del provider): meglio mostrare dati non
    aggiornatissimi con un avviso esplicito, piuttosto che far crashare
    l'applicazione.
    """
    path = _cache_path(key)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_cache(key: str, df: pd.DataFrame) -> None:
    df.to_parquet(_cache_path(key))
