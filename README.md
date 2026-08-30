# FinalFinance

Progetto a moduli progressivi: dashboard di analisi → backtesting e modelli
di previsione → trading automatico simulato → (sperimentale) ricerca di
strategie per selezione naturale. Tutti i moduli sono funzionanti, testati
con dati reali, e integrati nella stessa app web (Streamlit, cinque pagine
nella barra laterale). Il Modulo 3 è volutamente "dummy" e il Modulo 4 è
sperimentale. **Nessun modulo si connette a broker reali o esegue ordini
reali.**

## Struttura

```
FinalFinance/
├── config/
│   └── settings.yaml         # watchlist, timeframe, parametri indicatori — modifica qui
├── src/
│   ├── config.py              # loader di settings.yaml + .env
│   ├── data/
│   │   ├── equities.py         # dati azioni/ETF via yfinance (no API key)
│   │   ├── crypto.py           # dati crypto via ccxt (no API key)
│   │   ├── forex.py            # dati forex via Twelve Data (richiede API key gratuita)
│   │   └── cache.py             # cache locale Parquet, per non martellare le API
│   ├── indicators/
│   │   └── technical.py        # SMA, EMA, RSI, MACD, Bollinger — implementati a mano
│   ├── dashboard/
│   │   ├── 0_Analisi.py          # entry point Streamlit — pagina "Analisi" (Modulo 1)
│   │   ├── loader.py             # caricamento dati condiviso tra le pagine
│   │   └── pages/
│   │       ├── 1_Backtesting.py        # UI per il backtest delle strategie (Modulo 2)
│   │       ├── 2_Modelli_Predittivi.py # UI per training/valutazione modelli (Modulo 2)
│   │       ├── 3_Bot_Simulato.py       # UI per la simulazione del bot (Modulo 3, dummy)
│   │       └── 4_Mercato_Pulci.py      # UI per l'algoritmo evolutivo (Modulo 4, sperimentale)
│   ├── backtesting/            # Modulo 2
│   │   ├── engine.py             # motore di backtest vettoriale, anti look-ahead
│   │   ├── strategies.py         # strategie di esempio su indicatori (SMA crossover, RSI)
│   │   └── validation.py         # split walk-forward per serie temporali
│   ├── models/                 # Modulo 2
│   │   ├── features.py           # feature normalizzate/relative (no livelli di prezzo grezzi)
│   │   ├── baseline.py           # modelli naive + logistic/ridge regolarizzati (+ versioni semplici per confronto)
│   │   └── evaluate.py           # valutazione walk-forward, modello vs. naive
│   ├── execution/                # Modulo 3, dummy — NESSUNA connessione a broker reali
│   │   ├── model_signal.py         # retraining periodico + segnale pesato per confidenza
│   │   └── paper_broker.py         # simulazione esecuzione (riusa il motore di backtest) + trade log
│   └── evolution/                # Modulo 4, sperimentale — "Mercato delle Pulci"
│       ├── genome.py               # genoma evolvibile (pesi su feature + soglie comportamentali)
│       ├── population.py           # meccanica nascita/morte/riproduzione, evoluzione della popolazione
│       ├── data_prep.py            # standardizzazione feature + split evoluzione/test (no leakage)
│       └── evaluate.py             # congela i sopravvissuti e li testa fuori campione
├── scripts/
│   ├── backtest_indicators.py     # esegue e stampa il backtest delle strategie
│   ├── train_predictive_models.py # allena e valuta i modelli predittivi
│   ├── simulate_paper_trading.py  # simula il bot end-to-end e stampa il trade log
│   └── run_flea_market.py         # evolve la popolazione di pulci e valuta i sopravvissuti
├── reports/                     # output degli script (grafici) — ignorato da git
├── data_cache/                  # cache locale dati di mercato — ignorato da git
└── requirements.txt
```

## Avvio della dashboard (Modulo 1)

```bash
cd FinalFinance
python -m venv .venv && source .venv/bin/activate   # opzionale ma consigliato
pip install -r requirements.txt
python -m streamlit run src/dashboard/0_Analisi.py
```

Su Windows, se il comando `streamlit` da solo non viene riconosciuto (non è nel
PATH), usa sempre `python -m streamlit run ...` come sopra.

Si apre nel browser su `http://localhost:8501`, con cinque pagine nella
barra laterale:

- **Analisi** (pagina principale): grafico a candele, indicatori, lettura sintetica.
- **Backtesting**: scegli strategia (SMA crossover, RSI mean-reversion o
  entrambe a confronto), capitale iniziale e costo di transazione; mostra
  metriche ed equity curve interattiva vs. buy & hold.
- **Modelli predittivi**: allena i modelli baseline (direzione + rendimento)
  con validazione walk-forward e mostra il confronto con i baseline naive.
- **Bot Simulato**: simula (nessun ordine reale) un bot che segue il modello
  predittivo, riallenato periodicamente; mostra trade log ed equity curve.
- **Mercato delle Pulci**: fai evolvere (nessun ordine reale) una popolazione
  di agenti su più strumenti a scelta, con tutti i parametri (soglie di
  morte/riproduzione, popolazione iniziale, mutazione, split training/test)
  regolabili dall'interfaccia; mostra storia della popolazione, sopravvissuti
  valutati fuori campione e dettaglio del genoma migliore.

Tutte scelgono mercato e strumento dalla watchlist definita in `config/settings.yaml`, e
tutte hanno un selettore **Storico** per controllare quanti dati caricare:
periodo (3 mesi - max) per azioni/ETF, numero di candele (100-2000) per
crypto/forex — le due fonti dati lavorano per conteggio, non per intervallo
di date. Cambiare lo storico scarica dati nuovi (o li prende dalla cache se
già richiesti in precedenza con lo stesso periodo).

### API key per il forex

Per il forex serve una API key gratuita di Twelve Data (registrazione su
twelvedata.com, la chiave è visibile subito nella dashboard dell'account).

**Opzione consigliata — file `.env`** (non va reimpostata ad ogni sessione):
```bash
cp .env.example .env
# poi apri .env e incolla la tua chiave al posto di "la-tua-chiave-qui"
```

**Opzione alternativa — variabile d'ambiente della sessione:**
```bash
export TWELVEDATA_API_KEY="la-tua-chiave"        # macOS/Linux
$env:TWELVEDATA_API_KEY="la-tua-chiave"          # Windows PowerShell (solo sessione corrente)
setx TWELVEDATA_API_KEY "la-tua-chiave"          # Windows PowerShell (permanente, richiede riavvio terminale)
```

Nota: i simboli forex vanno scritti con lo slash, es. `EUR/USD` (non `EURUSD`) —
è il formato richiesto da Twelve Data.

## Backtesting e modelli predittivi (Modulo 2)

Disponibili sia dalla dashboard (pagine "Backtesting" e "Modelli predittivi",
vedi sopra) sia da riga di comando — gli script sono comodi per iterare
rapidamente o automatizzare, la dashboard per esplorare interattivamente:

```bash
# Backtest delle strategie di esempio (SMA crossover, RSI mean-reversion)
python scripts/backtest_indicators.py --market equities --symbol AAPL --period 2y

# Training e valutazione dei modelli predittivi (direzione + rendimento)
python scripts/train_predictive_models.py --market equities --symbol AAPL --period 2y
```

Entrambi accettano `--market equities|crypto|forex` e il simbolo relativo
(es. `--symbol BTC/USDT` con `--market crypto`).

### Come leggere i risultati

`backtest_indicators.py` stampa le metriche di ogni strategia (rendimento
totale, CAGR, volatilità, Sharpe ratio, max drawdown, win rate, numero di
operazioni) e salva un grafico dell'equity curve in `reports/`, con a fianco
il benchmark "buy & hold" (comprare e tenere per tutto il periodo). Se una
strategia non batte il benchmark, è un risultato legittimo da riportare, non
un errore: gran parte delle strategie semplici non batte sistematicamente il
buy & hold su singoli titoli.

`train_predictive_models.py` allena due modelli baseline — regressione
logistica per la direzione (su/giù), regressione Ridge per il rendimento
atteso, entrambi con selezione automatica della forza di regolarizzazione
via validazione interna sul solo training set — validati con **walk-forward**
(mai training su dati futuri rispetto al test) e li confronta con baseline
naive:

- `naive_majority` / `naive_persistence` per la classificazione,
- `naive_zero` (rendimento atteso sempre 0, ipotesi di random walk) e
  `naive_mean` per la regressione.

**Il numero che conta è il confronto, non il valore assoluto.** Se
`modello_accuracy` non supera chiaramente i baseline naive sulla maggior
parte dei fold, il modello non sta aggiungendo informazione utile — è una
conclusione onesta e attesa (i mercati sono in gran parte imprevedibili nel
breve periodo), non un bug del codice.

**Storia del debug (utile per capire le scelte fatte):** la prima versione
usava 18 feature, molte delle quali livelli di prezzo grezzi (sma_20, sma_50,
sma_200, ema_12, ema_26, le tre bande di Bollinger) — fortemente collineari
tra loro e non stazionarie, oltre a un normale LinearRegression senza
regolarizzazione. Su AAPL (2 anni di dati, 100-250 righe di training per
fold) il modello di regressione risultava nettamente **peggiore** del naive
(R² medio -3.81 contro -0.03 del baseline "predici sempre zero"): overfitting
classico. Il fix è stato duplice: (1) `features.py` ora calcola quantità
relative e adimensionali — es. "il prezzo è il 3% sopra la sua SMA a 20
periodi" invece del livello assoluto della SMA — riducendo le feature a 13 e
la loro collinearità; (2) `baseline.py` usa Ridge (regressione lineare con
penalità L2) al posto della regressione lineare semplice, con la forza della
penalità scelta automaticamente. Risultato dopo il fix: R² medio -0.06,
sostanzialmente in linea col naive (non lo batte, ma non lo tradisce più in
modo grossolano) — la conclusione onesta resta che questo modello semplice
non aggiunge valore predittivo su AAPL a orizzonte 1 giorno, ma almeno non è
più un artefatto di overfitting. Le versioni non regolarizzate restano nel
codice (`LinearReturnModel`, `LogisticDirectionModel`) come termine di
paragone per chi vuole vedere l'effetto della regolarizzazione con i propri
occhi.

## Bot di trading simulato (Modulo 3, dummy)

**Nessuna connessione a broker reali, nessun ordine reale.** Questo modulo
risponde alla domanda "cosa avrebbe fatto un bot basato su questo modello?",
non esegue nulla. È il passo logico prima di collegarsi a un vero conto demo
di un broker (paper trading "vero", con API reali) — quello resta un passo
successivo, deliberatamente non ancora fatto.

```bash
python scripts/simulate_paper_trading.py --market equities --symbol AAPL --period 2y
```

oppure dalla dashboard, pagina **Bot Simulato**.

### Come funziona la simulazione

A differenza della valutazione walk-forward del Modulo 2 (che allena una
volta per fold e misura l'accuratezza media), qui si simula un bot che
"vive" nel tempo:

1. Aspetta un periodo di **warm-up** (default 150 periodi) prima di iniziare
   a operare — non abbastanza storia per un modello affidabile prima di allora.
2. Si **riallena periodicamente** (default ogni 20 periodi) sommando ai dati
   di training tutto ciò che nel frattempo è diventato disponibile — mai sul
   futuro rispetto al momento della decisione.
3. Ad ogni periodo, decide una **posizione pesata per confidenza**: se il
   modello stima una probabilità di rialzo appena sopra la soglia impostata,
   apre una posizione piccola; se è molto convinto, una posizione più grande
   (fino al 100% del capitale virtuale, mai a leva, mai short).
4. L'esecuzione è simulata riusando lo stesso motore di backtest del Modulo 2
   (stessi costi di transazione, stessa regola anti-look-ahead), e produce un
   **log leggibile delle operazioni** (BUY/SELL/INCREASE/REDUCE, data, prezzo).

### Come leggere i risultati

Nel test su AAPL (dati reali, 2 anni), il bot con soglia di confidenza di
default (0.5) è risultato molto prudente: esposizione quasi sempre sotto il
30% del capitale, 114 operazioni, Sharpe ratio 1.23 ma rendimento assoluto
(+8.9%) molto inferiore al buy & hold nello stesso periodo (che nel frattempo
è salito parecchio) — a fronte però di un drawdown massimo di solo -3.8%
contro oscillazioni ben più ampie del benchmark. È un compromesso legittimo
tra rendimento e rischio, non un errore: un modello raramente "molto convinto"
tiene per lo più poca esposizione, sacrificando rendimento assoluto per
minor rischio. Alzando la soglia di confidenza il bot diventa più selettivo
(meno operazioni, posizioni più nette quando le apre); abbassandola (non
possibile sotto 0.5 nell'interfaccia, che corrisponde a "nessuna informazione")
diventerebbe via via più simile a un semplice buy & hold.

## Mercato delle Pulci (Modulo 4, sperimentale)

**Nessuna connessione a broker reali, nessun ordine reale.** Idea proposta
da Luca: invece di un solo modello allenato via gradiente (Modulo 2), una
**popolazione di agenti** ("pulci"), ciascuno con un capitale virtuale
indipendente e un proprio "genoma" — un piccolo insieme di pesi che combina
alcune feature tecniche in un punteggio per scegliere uno strumento. Le
pulci redditizie accumulano capitale e si riproducono (il figlio eredita il
genoma del genitore, mutato leggermente); quelle in perdita muoiono. È un
algoritmo evolutivo/genetico applicato alla ricerca di strategie, concettualmente
diverso dal Modulo 2 (che ottimizza un solo modello) e dal Modulo 3 (che
riallena periodicamente un unico modello): qui l'ottimizzazione avviene per
selezione naturale su una popolazione.

Disponibile sia da riga di comando sia dalla dashboard (pagina **Mercato
delle Pulci**, con tutti i parametri regolabili dall'interfaccia — soglie di
morte/riproduzione, popolazione iniziale, mutazione, split training/test):

```bash
python scripts/run_flea_market.py --market forex --symbols "EUR/USD,GBP/USD,USD/JPY"
python scripts/run_flea_market.py --market crypto --symbols "BTC/USDT,ETH/USDT"
```

### Come funziona

1. **Genoma**: 6 pesi (uno per famiglia di feature: rendimento recente,
   volatilità, posizione rispetto alla media mobile, RSI, MACD, posizione
   nelle Bollinger — scelti per essere poco ridondanti tra loro) più 3 soglie
   comportamentali (quando entrare, stop loss %, take profit %). Genoma
   deliberatamente piccolo: con una popolazione libera di evolvere, un
   genoma grande troverebbe quasi sempre combinazioni vincenti nel passato
   per puro rumore — lo stesso principio già visto (e risolto) nel Modulo 2,
   qui ancora più rilevante perché la selezione naturale premia esplicitamente
   qualunque cosa abbia funzionato, anche per caso.
2. **Vita di una pulce**: se non è investita, calcola il proprio punteggio su
   ogni strumento disponibile e si attacca al migliore (se sopra la soglia
   d'ingresso). Se è investita, resta finché non scatta uno stop loss, un
   take profit, o il punteggio dello strumento scende sotto la soglia
   d'ingresso (usata anche come soglia di uscita).
3. **Morte**: quando il capitale scende sotto una frazione (default 50%) del
   capitale che aveva alla nascita.
4. **Riproduzione**: quando il capitale supera una frazione (default 200%,
   cioè raddoppia) del capitale alla nascita — si divide in genitore
   (60% del capitale) e figlio (40%, genoma mutato).
5. **Split evoluzione/test, per evitare l'overfitting più insidioso di
   tutto il progetto**: la popolazione evolve **solo** sul primo 70% delle
   date storiche (in comune tra tutti gli strumenti scelti). I migliori
   sopravvissuti vengono poi "congelati" (nessuna ulteriore mutazione) e
   testati sul restante 30%, mai visto durante l'evoluzione — la
   standardizzazione delle feature è anch'essa calcolata solo sul periodo di
   training, per lo stesso motivo.

### Come leggere i risultati

Il confronto più importante è tra **capitale accumulato durante
l'evoluzione** (quanto la selezione naturale ha "premiato" quel genoma) e
**performance fuori campione** (come si comporta davvero su dati mai visti).
Un genoma che ha fatto bene in evoluzione ma male fuori campione non è
un'anomalia da correggere: è la dimostrazione empirica che l'evoluzione ha
trovato rumore, non un pattern reale — esattamente il rischio per cui questo
modulo è etichettato "sperimentale".

Nei test con dati reali e **soglie di default** (morte sotto 50%, riproduzione
sopra 200% del capitale alla nascita): su un paniere forex (EUR/USD, GBP/USD,
USD/JPY, ~200 giorni) e su un paniere crypto (BTC/USDT, ETH/USDT, stesso
periodo) **non è mai scattata né una morte né una riproduzione** — la
popolazione è rimasta invariata a 25 pulci in entrambi i casi. Con queste
soglie, in ~200 giorni e senza leva, nessuna combinazione casuale di pesi
guadagna o perde abbastanza da attivare la selezione naturale: i "risultati"
che si vedono sono quindi i migliori tra i genomi iniziali casuali, non il
prodotto di un'evoluzione reale — per osservare nascite/morti servono soglie
più strette (es. morte sotto 80%, riproduzione sopra 120%) o un periodo più
lungo, entrambi regolabili dall'interfaccia.

Con soglie più strette, su crypto (mercato calante nel periodo scelto) la
maggior parte delle pulci muore e i sopravvissuti, testati fuori campione,
perdono quasi tutti meno del benchmark buy & hold del paniere (circa -19%)
— ma **il rank in allenamento non predice il rank fuori campione**: la
pulce con più capitale accumulato in training può risultare tra le peggiori
fuori campione, e viceversa. È la controprova più diretta dell'overfitting
della selezione naturale di cui sopra, non un'anomalia da correggere.

### Limiti noti (onestamente, per chi vuole continuare a svilupparlo)

- Le soglie di morte/riproduzione sono frazioni fisse del capitale alla
  nascita: su mercati a bassa volatilità (es. forex) potrebbero non scattare
  mai nell'arco di una simulazione di durata tipica — vanno calibrate in base
  al mercato e alla durata scelti.
- Gli strumenti nello stesso mercato delle pulci richiedono calendari di date
  perfettamente sovrapponibili tra loro; mischiare mercati diversi (es. azioni
  + crypto, con orari di negoziazione incompatibili) non è ancora gestito.
- Il genoma non include position sizing frazionato sotto lo strumento scelto
  (una pulce investe sempre tutto il proprio capitale in un solo strumento):
  niente diversificazione a livello di singola pulce, solo a livello di
  popolazione.

## Personalizzare la watchlist

Modifica `config/settings.yaml` — non serve toccare il codice:

```yaml
watchlist:
  equities: [AAPL, MSFT, ENEL.MI]
  crypto: [BTC/USDT, ETH/USDT]
  forex: [EUR/USD]
```

## Nota sui test svolti in fase di sviluppo

Il codice è stato sviluppato in un ambiente con accesso di rete limitato
(sandbox con allowlist di domini), che blocca le chiamate dirette a Yahoo
Finance, Binance e Twelve Data. Dashboard, backtesting e modelli predittivi
sono comunque stati validati con dati reali già presenti in cache locale
(scaricati dalle tue sessioni precedenti) — solo il primo download di un
nuovo simbolo va verificato sul tuo computer, dove l'accesso a internet è
normale.

## Roadmap

- **Modulo 1**: dashboard di analisi tecnica, nessun segnale operativo. ✅
- **Modulo 2**: backtesting di strategie su indicatori + modelli predittivi
  baseline (classificazione e regressione), feature normalizzate,
  regolarizzazione (Ridge/LogisticCV), validazione walk-forward, integrato
  nella dashboard come pagine aggiuntive. ✅
  Prossimi miglioramenti possibili: più dati storici (5 anni invece di 2),
  feature selection ulteriore, modelli più complessi (solo se i baseline
  regolarizzati non bastano) su più strumenti.
- **Modulo 3 (dummy)**: simulazione retrospettiva di un bot basato sui
  modelli del Modulo 2, con retraining periodico e trade log — nessuna
  connessione a broker reali. ✅
- **Modulo 4 (sperimentale)**: "Mercato delle Pulci" — popolazione di agenti
  con genoma evolvibile per selezione naturale, split rigoroso
  evoluzione/test fuori campione, integrato nella dashboard con parametri
  regolabili. ✅
  Prossimi passi possibili: calibrazione automatica delle soglie di
  morte/riproduzione in base alla volatilità dello strumento, crossover tra
  genomi (non solo mutazione), supporto a mercati con calendari diversi
  nella stessa simulazione.
- **Passo non ancora fatto (deliberatamente)**: collegamento a un vero conto
  demo via API broker (es. Alpaca, OANDA demo, Binance testnet) per un paper
  trading "vero" con dati e latenza reali, poi — solo dopo risultati
  consistenti per settimane/mesi — eventuale esecuzione reale con capitale
  minimo.

## Avvertenza

Questo software è a scopo educativo/analitico. Non fornisce consigli di
investimento. Nessun modulo genera segnali operativi da eseguire
automaticamente né esegue ordini reali: il Modulo 3 è una simulazione
retrospettiva su dati storici, senza alcuna connessione a broker.
