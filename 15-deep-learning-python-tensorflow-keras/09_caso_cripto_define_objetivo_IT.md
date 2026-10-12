[← Precedente: [Reti Neurali Ricorrenti (RNN)](08_redes_recurrentes_rnn_IT.md) · [Indice](00_indice_IT.md) · Successivo: [Normalizza e costruisci sequenze temporali →](10_caso_cripto_normaliza_secuencias_IT.md)]

# 9. Caso pratico: definisci il tuo obiettivo con dati di criptovalute

## Perché ti importa

A partire da questo capitolo, costruirai un progetto completo da capo a coda: un modello che predice se il prezzo di una criptovaluta salirà o scenderà nei prossimi minuti, usando dati storici di prezzo e volume. Questo caso occupa le quattro lezioni finali del corso perché, a differenza dei capitoli precedenti, qui non impari una tecnica isolata: impari a encatenare tutte le tecniche del corso (preparazione dei dati, RNN, LSTM, regolarizzazione, TensorBoard) in una pipeline reale di serie temporali finanziarie.

Prima di scrivere una sola riga di rete neurale, devi rispondere a una domanda tutt'altro che banale: cos'è esattamente quello che vuoi prevedere?

## L'obiettivo del progetto

Lavorerai con dati storici di quattro criptovalute: Bitcoin (BTC), Litecoin (LTC), Bitcoin Cash (BCH) ed Ethereum (ETH). L'obiettivo è prevedere se il prezzo del Litecoin salirà o scenderà entro un orizzonte temporale futuro, usando come input una finestra di dati storici di tutte e quattro le criptovalute contemporaneamente. L'intuizione è che il comportamento di una criptovaluta non è indipendente dalle altre: includere il prezzo e il volume di più di una alla volta dà al modello più contesto rispetto a guardare solo Litecoin isolatamente.

Presta attenzione a un dettaglio importante: non stai per prevedere il prezzo esatto (quello sarebbe un problema di regressione). Stai per prevedere un'etichetta binaria molto più semplice: sale o scende. È un problema di classificazione, e è più facile da risolvere bene che tentare di indovinare un prezzo esatto.

## Ottieni i dati

Il materiale originale di questo corso usava file CSV scaricati manualmente. Affinché il tuo notebook sia riproducibile senza dipendere da un file esterno, scaricherai lo storico direttamente con la libreria `yfinance`:

```python
!pip install yfinance --quiet

import yfinance as yf
import pandas as pd

tickers = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]
datos_por_moneda = {}

for ticker in tickers:
    df = yf.download(ticker, period="2y", interval="1d")
    df = df[["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "volume"})
    datos_por_moneda[ticker] = df

datos_por_moneda["LTC-USD"].head()
```

Usiamo candele giornaliere (`interval="1d"`) invece di minuto per minuto, come nel materiale originale, perché lo storico intraday dettagliato non è disponibile gratuitamente nel lungo termine. La pipeline e i concetti sono identici, cambia solo l'unità di tempo: invece di prevedere "tra 3 minuti", prevedi "tra 3 giorni".

## Combina le quattro serie in un singolo dataset

Ogni criptovaluta ti dà due colonne (`close`, `volume`). Le combini tutte in un singolo DataFrame, indicizzato per data, e rinomini ogni colonna con il prefisso del suo ticker per non perdere di vista da dove proviene ogni dato:

```python
main_df = pd.DataFrame()

for ticker, df in datos_por_moneda.items():
    prefijo = ticker.split("-")[0]  # "BTC", "LTC", "BCH", "ETH"
    df = df.rename(columns={"close": f"{prefijo}_close", "volume": f"{prefijo}_volume"})
    main_df = df if main_df.empty else main_df.join(df, how="outer")

# Se qualche giorno manca un dato puntuale, riempi con l'ultimo valore noto
main_df = main_df.ffill()
main_df.dropna(inplace=True)

main_df.head()
```

`ffill()` (forward fill) propaga in avanti l'ultimo valore valido quando c'è un buco. È preferibile a riempire con zero, che introdurrebbe un dato falso e confonderrebbe il modello.

## Definisci la variabile obiettivo

Ora decidi l'orizzonte di previsione (quanti passi nel futuro guardi) e costruisci la colonna che il modello deve imparare a prevedere:

```python
RATIO_A_PREDECIR = "LTC"
PERIODO_FUTURO = 3  # quanti passi (giorni, in questo caso) nel futuro guardi

# Sposta il prezzo di chiusura 3 passi indietro: così, nella riga di "oggi",
# "future" contiene il prezzo di chiusura di "tra 3 giorni"
main_df["future"] = main_df[f"{RATIO_A_PREDECIR}_close"].shift(-PERIODO_FUTURO)

def clasifica(precio_actual, precio_futuro):
    # 1 = il prezzo salirà ("compra"), 0 = il prezzo scenderà o rimarrà uguale
    return 1 if float(precio_futuro) > float(precio_actual) else 0

main_df["target"] = list(map(clasifica, main_df[f"{RATIO_A_PREDECIR}_close"], main_df["future"]))

main_df[[f"{RATIO_A_PREDECIR}_close", "future", "target"]].head(10)
```

Verifica sempre manualmente, come nella cella precedente, che la colonna `target` abbia senso: per ogni riga, `target` deve essere 1 esattamente quando `future` è maggiore del prezzo di chiusura attuale.

## Riepilogo dei punti chiave

- L'obiettivo del progetto è classificazione binaria (sale/scende), non previsione esatta del prezzo
- Combinare dati di varie criptovalute correlate dà più contesto al modello rispetto a guardare una sola isolatamente
- `yfinance` ti permette di scaricare storici reali senza dipendere da file esterni che magari non hai
- `ffill()` riempie i buchi in modo più sensato che riempire con zero
- La colonna "future" si costruisce spostando il prezzo indietro nel tempo (`shift(-PERIODO_FUTURO)`), e la colonna "target" si calcola confrontando il prezzo attuale con quel prezzo futuro

## Mettilo in pratica

Esegui la pipeline completa di questo capitolo e, prima di passare al successivo, calcola quale percentuale di righe ha `target = 1` rispetto a `target = 0` con `main_df["target"].value_counts(normalize=True)`. **Annota il risultato**: se una delle due classi supera chiaramente il 60%, il tuo dataset è sbilanciato, e è esattamente il problema che risolverai nel capitolo 11.

---

[← Precedente: [Reti Neurali Ricorrenti (RNN)](08_redes_recurrentes_rnn_IT.md) · [Indice](00_indice_IT.md) · Successivo: [Normalizza e costruisci sequenze temporali →](10_caso_cripto_normaliza_secuencias_IT.md)]
