# 2. Regressione lineare applicata con scikit-learn

## Perché questo ti riguarda

Nel capitolo precedente hai visto cos'è l'apprendimento automatico o machine learning (ML) e come si organizza un progetto di questo tipo dall'inizio alla fine. In questo capitolo entri nella tecnica più usata per iniziare: la regressione.

La regressione ti serve a prevedere un numero a partire da altri numeri che già conosci. Con essa puoi stimare il prezzo di un prodotto, il consumo elettrico di uno stabilimento o la domanda di un servizio prima di prendere una decisione che costa denaro. Per questo è comune che un team tecnico costruisca prima un prototipo semplice di previsione per validare un'idea, e solo dopo investa tempo e risorse in una soluzione più complessa.

**Apri il tuo editor di codice e preparati a scrivere ogni blocco mentre avanzi.** Hai bisogno di Python installato insieme alle librerie NumPy, pandas, scikit-learn e Matplotlib. Se usi Anaconda o pip, installale così:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Cos'è la regressione

Immagina di voler calcolare il peso approssimativo di una persona a partire dalla sua altezza. Non hai una bilancia a portata di mano, ma sai che, in generale, più alta è una persona, più pesa. Se raccogli dati su altezza e peso di molte persone, puoi tracciare una linea che riassume quella tendenza. Quella linea ti permette, data un'altezza nuova, di calcolare un peso probabile.

È proprio quello che fa la regressione: cerca la relazione tra uno o più dati che già conosci (l'altezza) e un dato numerico che vuoi prevedere (il peso). Quando quella relazione è rappresentata da una linea retta, parliamo di regressione lineare.

Lo stesso ragionamento vale per stimare il prezzo di una casa a partire dalle sue dimensioni, il consumo energetico di un edificio in base alla sua superficie, o il prezzo di chiusura di un'azione in base al suo comportamento recente. Questo capitolo userà proprio quest'ultimo esempio, un prezzo che varia giorno per giorno, perché è quello usato dal materiale originale della certificazione EITC/AI/MLP (European IT Certification / Artificial Intelligence / Machine Learning with Python) per introdurre la regressione.

Non vedrai ancora la formula matematica che c'è dietro la linea di regressione né come si calcola il suo adattamento: questo arriva nel capitolo 4. Qui ti concentri sull'uso pratico della libreria scikit-learn per addestrare, valutare e usare un modello di regressione.

## Caratteristiche e etichette: separare quello che sai da quello che vuoi prevedere

In qualsiasi problema di regressione distingui due tipi di colonne:

- Le **caratteristiche** (features, o variabili predittive): i dati che già conosci e che usi come input
- L'**etichetta** (label, o variabile obiettivo): il numero che vuoi prevedere

Nel materiale originale di questa lezione, le caratteristiche e l'etichetta si ottenevano da una serie di prezzi azionari di Google scaricati in diretta tramite un'API (interfaccia di programmazione delle applicazioni) finanziaria. Quell'API oggi non è affidabile né gratuita, così in questo capitolo simuliamo una serie di prezzi con NumPy per poter praticare senza dipendere da servizi esterni. Il ragionamento e il codice di scikit-learn sono identici al materiale originale: solo l'origine del dato cambia.

Genera prima la serie simulata, con un seed fisso in modo che il risultato sia riproducibile:

```python
import numpy as np
import pandas as pd

np.random.seed(42)

n_dias = 500
retorno_diario = np.random.normal(loc=0.0005, scale=0.02, size=n_dias)
precio_cierre = 100 * (1 + retorno_diario).cumprod()

fechas = pd.date_range(start="2022-01-03", periods=n_dias, freq="B")

df = pd.DataFrame({"Close": precio_cierre}, index=fechas)
df.index.name = "Fecha"

ruido_apertura = np.random.normal(loc=0, scale=0.005, size=n_dias)
df["Open"] = df["Close"].shift(1).fillna(df["Close"].iloc[0]) * (1 + ruido_apertura)

ruido_rango = np.random.uniform(low=0.002, high=0.02, size=n_dias)
df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + ruido_rango)
df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - ruido_rango)

df["Volume"] = np.random.randint(1_000_000, 5_000_000, size=n_dias)

print(df.head())
```

Questo codice crea un cammino casuale (random walk): ogni prezzo di chiusura è il precedente moltiplicato per un piccolo rendimento diario casuale. A partire da quel prezzo simulato di chiusura, calcoliamo anche un prezzo di apertura, un massimo, un minimo e un volume scambiato, per imitare la forma di un dataset azionario reale.

Ora, il materiale originale avverte di qualcosa di importante: non tutte le colonne hanno lo stesso valore, e una regressione lineare semplice non scopre da sola le relazioni tra colonne. Se credi che la differenza tra il massimo e il minimo del giorno sia importante, devi calcolarla tu e aggiungerla come una caratteristica nuova. Fai lo stesso che fa il materiale originale con due colonne derivate:

```python
df["HL_percent"] = (df["High"] - df["Low"]) / df["Low"] * 100
df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100

columnas_features = ["Close", "HL_percent", "PCT_change", "Volume"]
print(df[columnas_features].head())
```

`HL_percent` cattura la volatilità del giorno (quanto il prezzo si è mosso tra il suo massimo e il suo minimo), e `PCT_change` cattura se il prezzo è salito o sceso lungo la sessione. Sono esempi di come trasformare colonne poco utili da sole in caratteristiche con significato.

Manca l'etichetta. Qui vogliamo prevedere il prezzo di chiusura, ma non quello di oggi, bensì quello di alcuni giorni nel futuro. Per ottenerlo, spostiamo la colonna `Close` verso l'alto un numero fisso di righe con il metodo `shift` di pandas:

```python
FORECAST_COL = "Close"
FORECAST_OUT = 30  # numero di giorni che vogliamo prevedere nel futuro

df["label"] = df[FORECAST_COL].shift(-FORECAST_OUT)

print(df[["Close", "label"]].head())
print(df[["Close", "label"]].tail())
```

Nota il risultato: in ogni riga, `label` contiene il prezzo di chiusura effettivo di 30 giorni dopo. Poiché non esistono dati 30 giorni dopo le ultime righe del dataset, quelle righe rimangono con valori nulli (`NaN`) in `label`. Questo è esattamente quello che accade nel materiale originale ed è intenzionale: quelle righe senza etichetta sono quelle che più tardi userai per prevedere il futuro, perché sono le uniche per le quali non conosci ancora la risposta.

## Addestrare e valutare il modello con scikit-learn

Con le caratteristiche e l'etichetta già definite, il passo successivo è dividere i dati in un insieme di addestramento e uno di test, addestrare il modello con il primo e verificarne le prestazioni con il secondo. Questa divisione è essenziale: se valuti il modello con gli stessi dati con cui lo hai addestrato, non saprai se davvero ha imparato un modello generale o se ha solo memorizzato i dati di addestramento.

```python
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = np.array(df[columnas_features])
X = preprocessing.scale(X)

X_lately = X[-FORECAST_OUT:]
X = X[:-FORECAST_OUT]

df.dropna(inplace=True)
y = np.array(df["label"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = LinearRegression(n_jobs=-1)
modelo.fit(X_train, y_train)

precision = modelo.score(X_test, y_test)
print("Precisione del modello (R²):", round(precision, 4))
```

Esamina cosa fa ogni riga:

- `preprocessing.scale(X)` riscala ogni caratteristica in modo che si trovino tutte in un intervallo simile; così nessuna colonna domina l'addestramento solo perché ha numeri più grandi
- `X_lately` conserva le righe più recenti, quelle che non hanno un'etichetta perché corrispondono al futuro che ancora vuoi prevedere
- `train_test_split` divide il resto delle righe in un 80% per l'addestramento e un 20% per il test, in modo casuale ma riproducibile grazie a `random_state`
- `LinearRegression()` crea il modello; il parametro `n_jobs=-1` gli dice di usare tutti i nuclei del tuo processore disponibili per addestrare più velocemente
- `modelo.fit(X_train, y_train)` è l'addestramento: il modello regola i suoi parametri interni in modo che le sue previsioni si avvicinino il più possibile a `y_train`
- `modelo.score(X_test, y_test)` misura quanto bene il modello prevede su dati che non ha visto durante l'addestramento

Quel valore finale è il coefficiente di determinazione, conosciuto come R². Si muove tra 0 e 1: un R² di 1 significa che il modello spiega perfettamente la variazione dell'etichetta a partire dalle caratteristiche, mentre un R² vicino a 0 significa che il modello non apporta molto più che indovinare il valore medio. Con i dati che hai appena simulato, è normale che tu ottenga un R² alto, perché l'etichetta è letteralmente lo stesso prezzo di chiusura spostato nel tempo e le caratteristiche mantengono una relazione molto diretta con esso.

Nel materiale originale, un modello di regressione lineare simile raggiunge un R² di 0,96 (96%) su questo tipo di dati, mentre un altro algoritmo, la regressione a vettori di supporto (SVR, dalle sue sigle in inglese), si ferma a 0,51 (51%) con i parametri di default. La lezione pratica è che scikit-learn facilita il confronto di algoritmi diversi con pochissime modifiche di codice, così provarne più di uno prima di deciderti è buona pratica.

Un'altra metrica usuale nella regressione, anche se non la calcoliamo qui, è l'errore quadratico medio (MSE, dalle sue sigle in inglese), che misura la media delle differenze al quadrato tra il valore previsto e quello reale. Più basso, meglio è. Vedrai l'MSE con più dettagli quando nel capitolo 4 entri in come si calcola realmente una regressione lineare.

## Prevedere il futuro: forecasting

A questo punto hai già un modello addestrato e valutato. Il passo finale è usarlo per prevedere davvero: applicarlo a `X_lately`, quel blocco di righe recenti che hai messo da parte dall'addestramento perché non avevano un'etichetta conosciuta.

```python
predicciones = modelo.predict(X_lately)

print("Previsioni per i prossimi", FORECAST_OUT, "giorni:")
print(predicciones)
```

`modelo.predict()` riceve un insieme di caratteristiche e restituisce un valore numerico per ogni riga: in questo caso, il prezzo di chiusura stimato per ognuno dei 30 giorni seguenti alla fine della serie simulata.

Per visualizzare il risultato insieme allo storico, aggiungi una colonna `Forecast` vuota e vai riempiendola riga per riga con date future, come fa il materiale originale:

```python
import datetime
import matplotlib.pyplot as plt

df["Forecast"] = np.nan

ultima_fecha = df.index[-1]
un_dia = datetime.timedelta(days=1)
siguiente_fecha = ultima_fecha + un_dia

for prediccion in predicciones:
    fila_vacia = [np.nan for _ in range(len(df.columns) - 1)]
    df.loc[siguiente_fecha] = fila_vacia + [prediccion]
    siguiente_fecha += un_dia

df["Close"].plot(label="Storico simulato")
df["Forecast"].plot(label="Previsione a 30 giorni")
plt.legend(loc="lower right")
plt.xlabel("Data")
plt.ylabel("Prezzo di chiusura simulato")
plt.title("Serie simulata e sua previsione con regressione lineare")
plt.show()
```

Il ciclo percorre ogni previsione e la aggiunge al dataframe in una riga nuova, con una data successiva all'ultima che avevi. Poiché in questo esempio simulato avanziamo di un giorno naturale ad ogni iterazione, anziché di un giorno bursatile come farebbe un dataset reale, qualche data potrebbe cadere nel fine settimana; per lo scopo didattico di questo capitolo non è un problema. Includere la data sull'asse orizzontale, anche se la data in sé non è una caratteristica del modello, è quello che ti permette di leggere il grafico con senso temporale: vedere dove finisce il conosciuto e dove inizia il previsto.

Con questo hai il ciclo completo: simulare dati, costruire caratteristiche e etichetta, addestrare, valutare e prevedere. Il capitolo 3 riprende questo stesso modello per salvarlo su disco con pickling e parla di scalatura e interpretazione con più dettaglio, e il capitolo 4 ti porta a programmare la regressione lineare da zero.

Un avvertimento sincero prima di chiudere questa sezione: un modello di regressione lineare come questo è un buon punto di partenza per imparare il flusso di lavoro, ma da solo non garantisce previsioni affidabili in un contesto reale, e tanto meno in un mercato finanziario reale. Quante più caratteristiche rilevanti incorpori e quanto meglio comprendi i dati di partenza, più utile sarà il modello. La semplicità di questo esempio è intenzionale: ti permette di concentrarti sul procedimento prima di complicarlo con dati reali.

## Riepilogo: cosa porti con te da questo capitolo

- La regressione predice un numero continuo a partire da altre variabili note, come stimare un peso a partire da un'altezza
- Le caratteristiche sono le colonne di input e l'etichetta è la colonna che vuoi prevedere
- Spostare una colonna con `shift` ti permette di convertire un valore presente in un'etichetta del futuro, con un orizzonte di previsione che tu scegli
- `train_test_split` divide i dati in addestramento e test in modo da poter misurare le prestazioni del modello su dati che non ha visto
- `LinearRegression().fit()` addestra il modello, e `.score()` restituisce l'R², un numero tra 0 e 1 che indica quanta variazione dell'etichetta il modello spiega
- `.predict()` applica il modello addestrato su dati nuovi, come il tratto finale senza etichetta che hai riservato al futuro

## Mettilo in pratica

Riprendi il codice di questo capitolo e sperimenta con due cambiamenti, uno dopo l'altro, annotando l'R² che ottieni in ogni caso:

**Cambia `FORECAST_OUT` da 30 a 5 e, poi, a 60, e confronta come varia il punteggio `.score()` del modello in ogni orizzonte.**

**Aggiungi o togli una caratteristica da `columnas_features` (per esempio, elimina `Volume` o aggiungi una nuova colonna da te calcolata) e addestra di nuovo il modello per vedere se l'R² migliora o peggiora.**

Non c'è un'unica risposta corretta: lo scopo è che tu verifichi con i tuoi occhi come l'orizzonte di previsione e la scelta delle caratteristiche influenzano la qualità di un modello di regressione.

Come sfida aggiuntiva, prova anche a cambiare il seed di `np.random.seed()` con un altro numero ed esegui di nuovo tutto il flusso. Vedrai che la serie simulata cambia completamente, ma il procedimento (caratteristiche, etichetta, addestramento, valutazione e previsione) rimane lo stesso. Proprio questo è l'idea centrale di questo capitolo: il flusso di lavoro con scikit-learn è lo stesso, sia che tu lavori con un prezzo azionario reale che con qualsiasi altra serie di dati numerici che vuoi prevedere.

**Sigle usate in questo capitolo:** ML (Machine Learning o apprendimento automatico), EITC/AI/MLP (European IT Certification / Artificial Intelligence / Machine Learning with Python), API (Application Programming Interface o interfaccia di programmazione delle applicazioni), R² (coefficiente di determinazione), SVR (Support Vector Regression o regressione a vettori di supporto), MSE (Mean Squared Error o errore quadratico medio).
