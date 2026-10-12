# 3. Salvare modelli, scalatura e interpretazione della regressione

## Perché questo ti riguarda

Nel capitolo precedente hai addestrato il tuo primo modello di regressione lineare con scikit-learn su una serie di prezzi simulata con NumPy. Ha funzionato, ma probabilmente hai notato qualcosa: ogni volta che lo vuoi usare, devi eseguire di nuovo il codice intero dall'inizio, incluso l'addestramento. Se il tuo insieme di dati è piccolo, questo è appena percettibile. Se un giorno lavori con migliaia o milioni di righe, riaddestare ogni volta che hai bisogno di una previsione può richiedere minuti o ore, un costo che non ha senso sostenere ancora e ancora.

Questo capitolo risolve tre problemi pratici che troverai in qualsiasi progetto reale di apprendimento automatico (ML, dall'inglese Machine Learning):

- Come salvare un modello già addestrato per riutilizzarlo senza ripetere l'addestramento
- Come preparare i dati quando le loro colonne hanno scale molto diverse tra loro
- Come leggere con criterio i numeri che scikit-learn ti fornisce dopo l'addestramento: la pendenza, il punto di incrocio e il punteggio di adattamento

Non programmerai nessuna formula matematica ancora; questo arriva nel capitolo quattro, quando costruirai la regressione lineare da zero. Qui ti concentri sullo sfruttare al massimo quello che la libreria già calcola per te.

## Salvare il tuo modello addestrato con pickle (con un'analogia quotidiana: salvare una partita di un videogioco per non iniziare da zero)

Pensa a un videogioco lungo. Se ogni volta che lo aprivi dovessi iniziare dal primo livello, la maggior parte delle persone lo abbandonerebbe prima di finirlo. Per questo esiste la funzione di salvataggio della partita: catturi lo stato esatto del gioco in un file e, più tardi, lo recuperi esattamente dove l'avevi lasciato. Un modello di ML addestrato funziona nello stesso modo. Addestrarlo è come progredire nei livelli: consuma tempo e risorse. Una volta che il modello "ha imparato" i suoi parametri dai dati, vuoi salvare quello stato per non ripetere il processo ogni volta che apri il tuo programma.

In Python, questo meccanismo si chiama pickling. Il modulo `pickle`, incluso nell'installazione standard del linguaggio, converte praticamente qualsiasi oggetto Python (un modello, una lista, un dizionario) in una sequenza di byte che puoi scrivere in un file. Poi, con l'operazione inversa, ricostituisci l'oggetto originale così com'era, senza ricalcolare nulla.

Riprendi la serie di prezzi simulata dal capitolo precedente, addestra un modello e salvalo:

```python
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Simuliamo di nuovo una serie di prezzi con un cammino casuale (random walk)
np.random.seed(42)
dias = np.arange(100).reshape(-1, 1)          # variabile indipendente: il giorno
pasos = np.random.normal(loc=0, scale=1, size=100)
precios = 100 + np.cumsum(pasos)              # variabile dipendente: il prezzo simulato

# Addestriamo il modello, come nel capitolo precedente
modelo = LinearRegression()
modelo.fit(dias, precios)

# Salviamo il modello addestrato in un file binario
with open("modelo_precios.pkl", "wb") as archivo:
    pickle.dump(modelo, archivo)

print("Modello salvato in modelo_precios.pkl")
```

Presta attenzione a due dettagli. Prima, apri il file in modalità `"wb"` (write binary, scrittura binaria), perché il contenuto non è testo leggibile, bensì byte. Secondo, per convenzione, i file generati con `pickle` portano l'estensione `.pkl`, anche se Python non ti obbliga a usarla.

Ora simula di aver chiuso il tuo programma, spento il computer e tornato un altro giorno. Non hai bisogno del codice di addestramento; solo del file salvato:

```python
import pickle

# Carichiamo il modello già addestrato, senza riaddestarlo
with open("modelo_precios.pkl", "rb") as archivo:
    modelo_cargado = pickle.load(archivo)

# Lo usiamo direttamente per prevedere il prezzo del giorno 150
prediccion = modelo_cargado.predict([[150]])
print(f"Prezzo stimato per il giorno 150: {prediccion[0]:.2f}")
```

Questa volta il file si apre in modalità `"rb"` (read binary, lettura binaria). Il modello caricato è funzionalmente identico all'originale: stessi coefficienti, stesse previsioni, zero minuti di addestramento.

Esiste un'alternativa moderna a `pickle` che molti progetti di ML preferiscono: la libreria `joblib`. La sua sintassi è quasi identica e funziona meglio con modelli che memorizzano grandi array numerici internamente, qualcosa di comune in scikit-learn:

```python
import joblib

# Salvare con joblib
joblib.dump(modelo, "modelo_precios.joblib")

# Caricare con joblib
modelo_cargado = joblib.load("modelo_precios.joblib")
```

Un avvertimento importante: un modello salvato non impara nulla di nuovo da solo. Se i dati del mondo reale cambiano con il tempo, il tuo modello salvato continua a riflettere modelli vecchi. Per questo è utile stabilire un calendario di riaddestramento, per esempio una volta al mese, anziché presumere che un modello pickled vada bene per sempre.

**Salva il tuo modello adesso** con il codice precedente e verifica che il file appaia nella tua cartella di lavoro.

## Perché a volte devi scalare i dati

Immagina di voler prevedere il prezzo di una casa a partire da due caratteristiche: la sua superficie in metri quadri (valori tra 40 e 300, approssimativamente) e la sua età in anni (valori tra 0 e 50). Queste due colonne vivono in scale completamente diverse. Per molti algoritmi, questa differenza di magnitudine non è solo un dettaglio estetico: può far sì che la caratteristica con numeri più grandi domini l'apprendimento, anche se non è la più rilevante per il problema.

La scalatura consiste nel trasformare ogni colonna affinché rimangono tutte in intervalli comparabili, senza alterare la relazione relativa tra i suoi valori. La tecnica più usata si chiama standardizzazione: sottrai la media di ogni colonna e dividi per la sua deviazione standard, in modo che il risultato abbia media zero e varianza uno. Scikit-learn la implementa nella classe `StandardScaler`.

Vediamo l'effetto in pratica, ampliando il nostro esempio con una seconda caratteristica simulata, il volume di operazioni di ogni giorno, che si muove su una scala molto più grande dello stesso giorno:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
dias = np.arange(100)
volumen = np.random.normal(loc=50000, scale=8000, size=100)  # scala grande
precios = 100 + np.cumsum(np.random.normal(0, 1, size=100))

# Combinamo entrambe le caratteristiche in un'unica matrice
X = np.column_stack([dias, volumen])
y = precios

# Prima di scalare: guarda la differenza di magnitudine tra le colonne
print("Senza scalare, prima riga:", X[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adattiamo il scalatore SOLO con i dati di addestramento
scalatore = StandardScaler()
X_train_scalato = scalatore.fit_transform(X_train)

# Applichiamo la STESSA trasformazione appresa ai dati di test
X_test_scalato = scalatore.transform(X_test)

print("Scalato, prima riga:", X_train_scalato[0])
```

Prima di scalare, la colonna del volume ha valori di decine di migliaia mentre quella del giorno appena raggiunge cento. Dopo la scalatura, entrambe le colonne rimangono attorno a zero, con una dispersione simile. Nota un dettaglio che marca la differenza tra un flusso corretto e uno con perdite di informazione: adatti (`fit_transform`) lo scalatore esclusivamente con i dati di addestramento, poi applichi (`transform`, senza `fit`) quella stessa trasformazione ai dati di test. Se adattassi lo scalatore con l'intero insieme, il tuo modello avrebbe accesso indiretto a informazioni dai dati di test prima della valutazione, e questo falsificherrebbe i tuoi risultati.

Per la regressione lineare semplice con una sola caratteristica, come quella del capitolo precedente, la scalatura non cambia troppo la qualità delle previsioni. Il suo vero valore appare con molte caratteristiche di scale diverse: migliora l'interpretazione dei coefficienti, perché rimangono in unità comparabili, e aiuta gli algoritmi di ottimizzazione come la discesa del gradiente a convergere più velocemente.

**Applica `StandardScaler` ai tuoi dati** prima di addestrare qualsiasi modello con più di una caratteristica.

## Cosa significano davvero i coefficienti, l'intercetta e l'R²

Quando addestri un modello di regressione lineare con scikit-learn, tre attributi concentrano tutta l'informazione di cui hai bisogno per interpretarlo: `coef_`, `intercept_` e il risultato di `score()`. Vale la pena capire cosa rappresenta ognuno oltre al numero sullo schermo.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

np.random.seed(42)
dias = np.arange(100).reshape(-1, 1)
precios = 100 + np.cumsum(np.random.normal(0, 1, size=100))

modelo = LinearRegression()
modelo.fit(dias, precios)

print("Pendenza (coef_):", modelo.coef_)
print("Intercetta:", modelo.intercept_)
print("R² sui medesimi dati:", modelo.score(dias, precios))
```

**Il coefficiente (`coef_`), o pendenza.** Indica quanto cambia la previsione per ogni unità aggiuntiva della variabile indipendente. Se `coef_` vale 0,8, ogni giorno aggiuntivo aggiunge, in media, 0,8 unità di prezzo secondo il modello appreso. Un coefficiente positivo segnala una relazione crescente; uno negativo, una relazione decrescente. Quando hai molte caratteristiche, ognuna riceve il suo coefficiente, e qui la scalatura della sezione precedente facilita il confronto di quale pesa più.

**L'intercetta (`intercept_`).** È il valore che il modello predice quando tutte le variabili indipendenti valgono zero. Nel nostro esempio, sarebbe il prezzo stimato per il giorno zero. A volte questo valore ha senso reale; altre volte è solo un punto matematico necessario per definire la retta, senza interpretazione pratica diretta (per esempio, se "età della casa uguale a zero" non appare mai nei tuoi dati reali). Interpretalo con cautela quando rimane fuori dall'intervallo dei tuoi dati originali.

**Il coefficiente di determinazione (R²), ottenuto con `score()`.** Misura quale proporzione della variabilità dei tuoi dati il modello spiega, su una scala che normalmente va da 0 a 1 (può diventare negativa se il modello predice peggio che usare semplicemente la media). Un R² di 0,95 significa che il modello spiega il 95% della variazione osservata nei tuoi dati: un adattamento molto buono. Un R² di 0,3 significa che il modello spiega solo il 30%: la retta cattura una tendenza, ma lascia fuori la maggior parte del comportamento dei dati, probabilmente perché mancano variabili rilevanti oppure perché la relazione non è lineare.

Un R² alto non garantisce automaticamente un buon modello per la produzione, né uno basso lo scarta sempre: dipende dal problema. Prevedere il comportamento umano o i mercati finanziari con un R² di 0,3 potrebbe essere un risultato accettabile, mentre un R² di 0,3 in un processo fisico controllato solitamente indica che manca qualcosa di importante nel tuo approccio.

## Riepilogo: cosa porti con te da questo capitolo

- Il modulo `pickle` serializza un modello addestrato in un file, così eviti di riaddestare ogni volta che ne hai bisogno
- `pickle.dump()` salva l'oggetto in modalità scrittura binaria (`"wb"`); `pickle.load()` lo recupera in modalità lettura binaria (`"rb"`)
- `joblib` è un'alternativa moderna a `pickle`, consigliata per modelli di scikit-learn con grandi array numerici
- `StandardScaler` trasforma le tue colonne a media zero e varianza uno, evitando che una caratteristica con scala maggiore domini l'apprendimento
- Adatta lo scalatore solo con i dati di addestramento e applicalo poi ai dati di test, mai al contrario
- `coef_`, `intercept_` e `score()` ti dicono, rispettivamente, quanto influisce ogni variabile, dove inizia la retta e quale percentuale della variazione il tuo modello spiega

## Mettilo in pratica

Esegui questo esercizio in due file separati, simulando due sessioni di lavoro diverse.

Nel primo file (`entrenar.py`), genera una serie di dati simulati con NumPy, addestra una `LinearRegression`, salva una previsione di riferimento e salva il modello con `pickle`:

```python
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

np.random.seed(7)
X = np.arange(50).reshape(-1, 1)
y = 3 * X.flatten() + np.random.normal(0, 2, size=50)

modelo = LinearRegression()
modelo.fit(X, y)

prediccion_referencia = modelo.predict([[25]])
print("Previsione prima di salvare:", prediccion_referencia)

with open("mi_modelo.pkl", "wb") as archivo:
    pickle.dump(modelo, archivo)
```

**Chiudi questo file e apri uno nuovo (`usar_modelo.py`)**, senza copiare il codice di addestramento, per simulare il lavoro di un altro giorno. Carica il modello e verifica che la previsione corrisponda esattamente a quella di riferimento:

```python
import pickle

with open("mi_modelo.pkl", "rb") as archivo:
    modelo_cargado = pickle.load(archivo)

prediccion_nueva = modelo_cargado.predict([[25]])
print("Previsione dopo il caricamento:", prediccion_nueva)
```

**Confronta entrambi i risultati** e conferma che sono identici. **Stampa anche** `modelo_cargado.coef_`, `modelo_cargado.intercept_` e `modelo_cargado.score(X, y)`, e scrivi in un commento cosa significa ogni valore secondo quello che hai imparato in questo capitolo. Se ti avanza tempo, **ripeti l'esercizio** sostituendo `pickle` con `joblib` e verifica che il risultato non cambi.

**Sigle usate in questo capitolo:** ML (Machine Learning), R² (coefficiente di determinazione).
