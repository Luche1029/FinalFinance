"""Caricamento centralizzato della configurazione da config/settings.yaml.

Ogni modulo che ha bisogno di parametri (simboli, periodi indicatori, path cache)
importa da qui invece di leggere il file YAML autonomamente.

Carica anche il file .env nella root del progetto (se presente), così le API
key (es. TWELVEDATA_API_KEY) possono essere salvate lì invece di doverle
reimpostare come variabili d'ambiente ad ogni sessione del terminale.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def load_settings() -> dict:
    """Legge settings.yaml una sola volta e la tiene in cache in memoria."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cache_dir() -> Path:
    """Ritorna il path assoluto della cartella di cache, creandola se manca."""
    settings = load_settings()
    path = PROJECT_ROOT / settings["cache"]["directory"]
    path.mkdir(parents=True, exist_ok=True)
    return path
