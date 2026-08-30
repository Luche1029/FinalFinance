"""Pagina Streamlit — Mercato delle Pulci (Modulo 4, sperimentale).

ATTENZIONE: nessuna connessione a broker reali, nessun ordine reale. Una
popolazione di agenti ("pulci") con capitale virtuale indipendente cerca
strategie per selezione naturale — genomi redditizi si riproducono (con
mutazione), quelli in perdita muoiono. La popolazione evolve SOLO sulla
prima parte dei dati storici; i migliori sopravvissuti vengono congelati e
testati su un periodo successivo mai visto, per capire se hanno trovato
pattern reali o solo rumore. Riusa la stessa logica già testata da riga di
comando in scripts/run_flea_market.py.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st

from src.config import load_settings
from src.dashboard.loader import MARKET_MODULES, load_data, lookback_selector, symbol_options_for
from src.evolution.data_prep import build_flea_market_dataset
from src.evolution.evaluate import equal_weight_benchmark, evaluate_survivors, simulate_single_genome
from src.evolution.population import run_evolution
from src.indicators.technical import add_all_indicators

st.set_page_config(page_title="Mercato delle Pulci", layout="wide")
st.title("Mercato delle Pulci")
st.error(
    "**Simulazione sperimentale, non trading reale.** Nessuna connessione a broker, "
    "nessun ordine viene davvero eseguito. Una popolazione di agenti con capitale "
    "virtuale indipendente cerca strategie per selezione naturale (algoritmo genetico), "
    "non è un consiglio di investimento.",
    icon="⚠️",
)
st.caption(
    "La popolazione evolve solo sulla prima parte dei dati; i migliori sopravvissuti "
    "vengono poi testati, congelati, su un periodo successivo mai visto. Il confronto "
    "tra performance in training e fuori campione è il numero più importante: se i "
    "migliori in training non sono anche i migliori fuori campione, è overfitting "
    "della selezione naturale, non un errore del codice."
)

settings = load_settings()
watchlist = settings["watchlist"]

col_market, col_symbols, col_lookback = st.columns([1, 2, 1])
with col_market:
    market_label = st.selectbox("Mercato", list(MARKET_MODULES.keys()))
market_key = MARKET_MODULES[market_label]

available_symbols = symbol_options_for(market_key, watchlist)
with col_symbols:
    symbols = st.multiselect(
        "Strumenti (almeno 2, stesso mercato/calendario)",
        available_symbols,
        default=available_symbols[: min(3, len(available_symbols))],
    )
with col_lookback:
    lookback = lookback_selector(market_key, key_prefix="pulci")

st.subheader("Parametri della popolazione")
col1, col2, col3, col4 = st.columns(4)
with col1:
    n_initial = st.number_input("Pulci iniziali", min_value=5, max_value=100, value=25)
    initial_capital = st.number_input("Capitale virtuale per pulce", min_value=10.0, value=100.0, step=10.0)
with col2:
    death_frac = st.slider(
        "Soglia di morte (frazione del capitale alla nascita)", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Sotto questa frazione del capitale che aveva alla nascita, la pulce muore.",
    )
    reproduction_frac = st.slider(
        "Soglia di riproduzione (frazione)", min_value=1.05, max_value=3.0, value=2.0, step=0.05,
        help="Sopra questa frazione del capitale alla nascita, la pulce si riproduce.",
    )
with col3:
    max_population = st.number_input("Tetto massimo popolazione", min_value=25, max_value=1000, value=200)
    mutation_sigma = st.slider("Ampiezza mutazione", min_value=0.01, max_value=0.5, value=0.15, step=0.01)
with col4:
    train_fraction = st.slider(
        "Quota dati per l'evoluzione (training)", min_value=0.5, max_value=0.9, value=0.7, step=0.05,
        help="Il resto è il periodo fuori campione su cui vengono testati i sopravvissuti.",
    )
    cost_bps = st.slider("Costo di transazione (bps)", min_value=0, max_value=50, value=5)

if "pulci_seed" not in st.session_state:
    st.session_state["pulci_seed"] = 42


def _randomize_seed() -> None:
    # Eseguita PRIMA che lo script si ridisegni: a questo punto il widget con
    # key="pulci_seed" non esiste ancora nel nuovo run, quindi modificarne il
    # valore in session_state è sicuro (farlo dopo l'istanziazione del widget,
    # come in un normale "if st.button(...):", solleva StreamlitAPIException).
    st.session_state["pulci_seed"] = random.randint(0, 999_999)


col5, col6, col7 = st.columns([2, 2, 1])
with col5:
    top_k = st.number_input("Quanti sopravvissuti valutare fuori campione", min_value=1, max_value=30, value=10)
with col6:
    seed = st.number_input(
        "Seed (riproducibilità)", min_value=0, max_value=999_999, key="pulci_seed",
        help="Stesso seed + stessi parametri = stessa identica simulazione. "
             "Cambialo (o usa il bottone) per vedere se i risultati sono stabili o dipendono dal caso.",
    )
with col7:
    st.write("")  # spaziatore per allineare il bottone al campo numerico accanto
    st.button("🎲 Randomizza", on_click=_randomize_seed)

run_clicked = st.button("Avvia evoluzione", type="primary")

if run_clicked:
    if len(symbols) < 2:
        st.warning("Seleziona almeno 2 strumenti: una singola pulce ha bisogno di poter scegliere tra più opzioni.")
        st.stop()

    try:
        with st.spinner(f"Carico dati per {', '.join(symbols)}..."):
            dfs = {}
            for symbol in symbols:
                df = load_data(market_key, symbol, "1d", lookback=lookback)
                dfs[symbol] = add_all_indicators(df)
        dataset = build_flea_market_dataset(dfs, train_fraction=train_fraction)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Errore nel caricamento dati o nella preparazione del dataset: {exc}")
        st.stop()

    train_dates, test_dates = dataset["train_dates"], dataset["test_dates"]
    st.write(
        f"Periodo di evoluzione: **{len(train_dates)}** periodi "
        f"({train_dates[0].date()} → {train_dates[-1].date()}) — "
        f"periodo fuori campione: **{len(test_dates)}** periodi "
        f"({test_dates[0].date()} → {test_dates[-1].date()})"
    )
    st.write(f"Geni (feature) usati dal genoma: {', '.join(dataset['gene_features'])}")

    with st.spinner("Evolvo la popolazione sui dati di training..."):
        survivors, history = run_evolution(
            features_by_instrument=dataset["features_by_instrument"],
            prices_by_instrument=dataset["prices_by_instrument"],
            dates=train_dates,
            gene_features=dataset["gene_features"],
            n_initial=int(n_initial),
            initial_capital=initial_capital,
            death_frac=death_frac,
            reproduction_frac=reproduction_frac,
            max_population=int(max_population),
            mutation_sigma=mutation_sigma,
            transaction_cost_bps=cost_bps,
            seed=int(seed),
        )

    pop_history = history.population_over_time
    st.subheader("Storia della popolazione durante l'evoluzione")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Popolazione finale", len(survivors), delta=len(survivors) - n_initial)
    col_b.metric("Nascite totali", len(history.births))
    col_c.metric("Morti totali", len(history.deaths))

    if len(history.births) == 0 and len(history.deaths) == 0:
        st.info(
            "Nessuna nascita né morte durante l'evoluzione: con queste soglie e su questo "
            "periodo, nessuna pulce ha mai guadagnato o perso abbastanza da attivare la "
            "selezione naturale. I risultati sotto sono quindi i migliori tra i genomi "
            "iniziali casuali, non il prodotto di una vera evoluzione — prova a stringere "
            "le soglie di morte/riproduzione o ad allungare il periodo."
        )

    if not pop_history.empty:
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Numero di pulci", "Capitale totale"))
        fig.add_trace(go.Scatter(x=pop_history["data"], y=pop_history["n_pulci"], name="N. pulci"), row=1, col=1)
        fig.add_trace(go.Scatter(x=pop_history["data"], y=pop_history["capitale_totale"], name="Capitale"), row=1, col=2)
        fig.update_layout(height=350, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if not survivors:
        st.warning("Estinzione totale della popolazione durante l'evoluzione: nessun genoma da valutare.")
        st.stop()

    with st.spinner("Valuto i migliori sopravvissuti fuori campione..."):
        results = evaluate_survivors(
            survivors,
            features_by_instrument=dataset["features_by_instrument"],
            prices_by_instrument=dataset["prices_by_instrument"],
            out_of_sample_dates=test_dates,
            top_k=int(top_k),
            initial_capital=initial_capital,
            transaction_cost_bps=cost_bps,
        )
        benchmark = equal_weight_benchmark(dataset["prices_by_instrument"], test_dates, initial_capital)

    benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100

    st.subheader("Valutazione fuori campione")
    st.metric("Benchmark (paniere equamente diviso, buy & hold)", f"{benchmark_return:+.2f}%")
    st.dataframe(results, use_container_width=True)

    top_survivors_sorted = sorted(survivors, key=lambda f: f.capital, reverse=True)
    if top_survivors_sorted:
        best_flea = top_survivors_sorted[0]
        st.subheader(f"Dettaglio della pulce migliore (id {best_flea.id})")
        st.json(best_flea.genome.as_dict())

        equity_curve, _, trade_log = simulate_single_genome(
            best_flea.genome, dataset["features_by_instrument"], dataset["prices_by_instrument"],
            test_dates, initial_capital=initial_capital, transaction_cost_bps=cost_bps,
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Pulce migliore (fuori campione)"))
        fig2.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, name="Benchmark (paniere)", line=dict(dash="dash", color="gray")))
        fig2.update_layout(
            title="Equity curve fuori campione — pulce migliore vs benchmark",
            height=450, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.write(f"Log operazioni fuori campione ({len(trade_log)} operazioni):")
        if trade_log.empty:
            st.info("Nessuna operazione nel periodo fuori campione.")
        else:
            st.dataframe(trade_log, use_container_width=True)
else:
    st.info("Imposta i parametri e premi \"Avvia evoluzione\".")
