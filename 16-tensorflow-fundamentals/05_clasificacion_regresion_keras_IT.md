# 05. Classificazione, regressione e API di Keras

## Perché questo argomento è importante

Fino a ora hai addestrato modelli che classificano immagini: la rete guarda una foto e decide a quale categoria appartiene. Ma non tutti i problemi di Machine Learning (ML) chiedono una categoria. A volte hai bisogno di un numero: il prezzo di una casa, il consumo di un auto, la probabilità che un tumore sia maligno. Confondere questi due tipi di problema, o usare la funzione di perdita sbagliata per ognuno, è uno degli errori più comuni di chi inizia con TensorFlow.

Inoltre, non tutti i dati arrivano in forma di immagine. La maggior parte dei progetti reali in sanità, industria o amministrazione pubblica lavorano con tabelle: una riga per paziente, per pezzo o per pratica, e colonne con misure, categorie e date. Quel formato tabulare quasi sempre si salva come CSV (Comma-Separated Values, valori separati da virgole), così sapere come caricarlo e prepararlo bene è tanto importante quanto sapere come progettare la rete.

Infine, fino a questo punto hai costruito modelli solo con l'API (Application Programming Interface, interfaccia di programmazione di applicazioni) Sequential di Keras: una pila di strati, uno dietro l'altro. Quell'API ti porta lontano, ma non funziona per architetture con più di un input, più di un output o connessioni che saltano strati. Per questo esiste la Functional API di Keras, che in questa lezione imparerai a usare.

Questo capitolo chiude inoltre un fronte storico: chi consulta documentazione o codice vecchio di TensorFlow incontra una sintassi molto diversa da quella che hai usato fino a ora (sessioni, marcatori di posizione, grafi che devi eseguire a mano). Capire che cosa è cambiato tra TensorFlow 1.x e TensorFlow 2.0 ti evita confusioni e ti permette di leggere esempi di qualunque epoca senza bloccarti.

## Classificazione e regressione: due domande diverse

Pensa a due domande che puoi fare a un modello:

- "Questa immagine è una maglietta, un pantalone o uno stivale?" → **classificazione**
- "Quanti chilometri per litro fa questo auto?" → **regressione**

In classificazione, l'output è una categoria di un insieme chiuso di opzioni (due categorie, come benigno o maligno, o varie, come i dieci tipi di capo da Fashion MNIST). In regressione, l'output è un numero continuo che può prendere qualunque valore dentro un intervallo: 4,7, 32,1, 128,9.

Questa differenza non è solo concettuale: cambia l'ultimo strato della tua rete e, soprattutto, cambia la funzione di perdita che usi per addestrarla. La funzione di perdita misura quanto il modello si sbaglia ad ogni previsione, e l'ottimizzatore aggiusta i pesi della rete per ridurre quell'errore. Se misuri male l'errore, il modello impara a risolvere un problema diverso da quello che hai.

| Tipo di problema | Ultimo strato tipico | Funzione di perdita | Esempio |
|---|---|---|---|
| Classificazione binaria (due classi) | una neurone con attivazione `sigmoid` | `binary_crossentropy` | il tumore è maligno? |
| Classificazione multiclasse | *n* neuroni con attivazione `softmax` (o logits senza attivazione) | `sparse_categorical_crossentropy` | che capo è? |
| Regressione | una neurone senza attivazione (lineare) | `mean_squared_error` (MSE, errore quadratico medio) o `mean_absolute_error` (MAE, errore assoluto medio) | quanti chilometri per litro? |

Un'analogia utile: in classificazione, il modello distribuisce una "fiducia" tra varie caselle e vince la casella con più punti; in regressione, il modello disegna una linea (o una superficie, se ci sono più variabili) e prova a far cadere i tuoi dati il più vicino possibile ad essa. L'errore quadratico medio penalizza molto gli errori grandi perché li eleva al quadrato; l'errore assoluto medio li penalizza in modo proporzionale e per questo è meno sensibile agli outlier.

Un esempio minimo di ogni caso con Keras:

```python
# Classificazione multiclasse (per esempio, Fashion MNIST, dieci classi)
modello_classificazione.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# Regressione (per esempio, prevedere il consumo di un'auto)
modello_regressione.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mean_absolute_error')
```

Nota le metriche anche: in classificazione ha senso misurare `accuracy` (percentuale di risposte giuste); in regressione non esiste un "successo" esatto, così si usano metriche come il MAE o il MSE, che indicano di quanto di media la previsione si allontana dal valore reale.

## Caricare e preparare dati tabulari in CSV

Un file CSV è, essenzialmente, un foglio di calcolo in testo semplice: ogni riga è una riga e le virgole separano le colonne. TensorFlow non può addestrare direttamente su testo: prima devi caricare quel CSV in memoria e convertirlo in tensori numerici.

Per dataset piccoli che stanno in memoria, il percorso più semplice è pandas:

```python
import pandas as pd
import numpy as np
import tensorflow as tf

# Carica il CSV in un DataFrame di pandas
dati = pd.read_csv("mio_dataset.csv")

# Separa la colonna che vuoi prevedere (l'etichetta) dal resto delle colonne (le caratteristiche)
etichette = dati.pop("colonna_obiettivo")
caratteristiche = np.array(dati)

# Con un unico array numerico, un modello Sequential è sufficiente
modello = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
modello.compile(loss='mean_squared_error', optimizer='adam')
modello.fit(caratteristiche, etichette, epochs=10)
```

Questo flusso funziona bene quando tutte le colonne sono numeriche e hanno intervalli simili. Nella pratica, quasi nessun dataset reale è così:

- **Colonne con scale molto diverse.** Se una colonna misura l'età in anni (0 a 100) e un'altra misura i redditi in euro (0 a 100.000), la rete darà più peso a quella che ha i numeri più grandi senza che questo significhi che è più importante. La soluzione è la **normalizzazione**: riscalare ogni colonna affinché abbia una media di zero e una deviazione standard di uno. Keras offre lo strato `tf.keras.layers.Normalization` per questo: si "adatta" ai tuoi dati di addestramento (calcola media e varianza) e poi normalizza ogni esempio automaticamente, dentro il modello stesso.
- **Colonne categoriche.** Una colonna come "porto di imbarco" o "tipo di suolo" non è un numero, è un'etichetta di testo o un codice. Prima di arrivare alla rete devi convertirla in qualcosa di numerico, normalmente tramite *one-hot encoding*: una colonna binaria per ogni categoria possibile. Keras offre gli strati `tf.keras.layers.StringLookup` (converte stringhe di testo in indici) e `tf.keras.layers.CategoryEncoding` (converte quegli indici in vettori one-hot).
- **Valori mancanti o incoerenti.** Prima di addestrare conviene rivedere quanti valori mancano (`dati.isna().sum()` in pandas) e decidere se li elimini, li riempi con un valore di default o li tratti come una categoria ulteriore.

Quando il dataset mescola colonne numeriche e categoriche, integrare il preprocessing dentro il modello stesso (invece di farlo separatamente con un altro strumento) ha un vantaggio importante: se più tardi salvi il modello, il preprocessing si salva con esso, così chiunque lo riutilizzi applicherà esattamente le stesse trasformazioni senza dover ricordare come hai preparato i dati originali. Questo è precisamente quello che rende possibile la Functional API, come vedrai nella sezione successiva.

Per dataset grandi che non stanno in memoria, TensorFlow offre l'API `tf.data.Dataset`, con funzioni come `tf.data.experimental.make_csv_dataset` che leggono il CSV per lotti (*batches*) direttamente da disco, senza caricarlo tutto in una volta, e con metodi concatenabili come `.shuffle()` (mescolare), `.batch()` (raggruppare in lotti) e `.cache()` (salvare in memoria dopo la prima lettura per velocizzare i lotti successivi di addestramento).

## La Sequential API contro la Functional API

Fino a ora tutti i tuoi modelli si sono definiti così:

```python
modello = tf.keras.Sequential([
    strato_1,
    strato_2,
    strato_3
])
```

Questa è la **Sequential API**: una pila di strati dove ogni uno riceve l'output del precedente e nulla più. È come una catena di montaggio: il pezzo entra da un estremo, passa per ogni stazione in ordine, ed esce dall'altro estremo. È semplice, leggibile e sufficiente per la maggior parte dei modelli che hai costruito fino a ora.

Il problema appare non appena la tua architettura smette di essere una linea retta. Immagina un modello che ha bisogno di combinare due fonti di informazione diverse, per esempio, il testo di un'incidenza e le etichette che chi l'ha segnalata le ha messo, per decidere nello stesso momento la sua priorità e il dipartimento che deve affrontarla. Questo esige **due input e due output**, qualcosa che la Sequential API non può esprimere perché ammette solo un input, un output e un unico percorso tra i due.

Qui entra la **Functional API**. Invece di impilare gli strati dentro una lista, tratti ogni strato come una funzione che si applica su un tensore e restituisce un altro tensore. Questo ti permette di costruire qualunque architettura che si possa disegnare come un grafo aciclico diretto o DAG (Directed Acyclic Graph, grafo diretto senza cicli): strati che si diramano, si combinano o si condividono.

```python
import tensorflow as tf
from tensorflow.keras import layers

# Con la Functional API, definisci prima l'input come un oggetto esplicito...
input_numerico = tf.keras.Input(shape=(8,), name="dati_numerici")
input_categorico = tf.keras.Input(shape=(4,), name="dati_categorici")

# ...e concateni gli strati chiamandoli come funzioni sul tensore precedente
ramo_numerico = layers.Dense(32, activation='relu')(input_numerico)
ramo_categorico = layers.Dense(16, activation='relu')(input_categorico)

# Puoi combinare due rami che mai potrebbero stare insieme in una Sequential
combinato = layers.concatenate([ramo_numerico, ramo_categorico])
output = layers.Dense(1, activation='sigmoid')(combinato)

# Il modello si definisce indicando quali tensori sono gli input e quale l'output
modello = tf.keras.Model(inputs=[input_numerico, input_categorico], outputs=output)
```

Pensa alla differenza come a quella che c'è tra una catena di montaggio (Sequential) e una rete di tubi (Functional): in una rete di tubi puoi avere due serbatoi di input che confluiscono nello stesso deposito, o un tubo che si divide in due per alimentare due processi diversi. Con la Functional API puoi:

- Definire modelli con **più input** o **più output**
- **Combinare** rami diversi con strati come `concatenate` o `add`
- **Riutilizzare lo stesso strato** in più punti del modello (strati condivisi), utile per esempio quando due input di testo devono essere elaborati con lo stesso vocabolario imparato
- Creare **connessioni residuali**, dove l'output di uno strato si somma all'output di uno strato posteriore invece di seguire solo il percorso sequenziale

Una volta che il modello funzionale è definito, addestrarlo, valutarlo e salvarlo funziona esattamente come con un modello Sequential: `model.compile()`, `model.fit()`, `model.evaluate()`, `model.save()`. La Functional API non è un modo diverso di addestrare, è un modo diverso di descrivere l'architettura.

Quando usare ognuna? Se il tuo modello è una pila semplice di strati, la Sequential API è più corta e più leggibile. Non appena hai bisogno di più di un input, più di un output, strati condivisi o connessioni che non seguono l'ordine lineare, passa alla Functional API.

## Da TensorFlow 1.x a TensorFlow 2.0

Se mai apri un tutorial vecchio di TensorFlow, è facile che incontri codice come questo:

```python
# Stile TensorFlow 1.x
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)

with tf.Session() as sess:
    risultato = sess.run(c)
    print(risultato)
```

In TensorFlow 1.x, scrivere `c = tf.add(a, b)` non calcolava nulla: solo aggiungeva un nodo a un **grafo computazionale**, una specie di piano dell'operazione. Per ottenere un risultato reale dovevi aprire una `tf.Session` (sessione) e chiedere esplicitamente che eseguisse quel grafo con `sess.run()`. I dati di input si iniettavano al momento dell'esecuzione tramite `tf.placeholder` (marcatori di posizione) e un dizionario `feed_dict`. Questo modello di "definire prima, eseguire dopo" permetteva di ottimizzare il grafo prima di eseguirlo, ma rendeva il debug del codice scomodo: non potevi semplicemente stampare il valore di un tensore a metà del tuo programma, perché quel tensore ancora non aveva alcun valore.

TensorFlow 2.0, lanciato nel 2019, ha cambiato questo radicalmente con l'**esecuzione eager** (*eager execution*) come modalità di default: ogni operazione si calcola nel momento in cui la scrivi, come in un Python normale.

```python
# Lo stesso calcolo in TensorFlow 2.x
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)

print(c)  # tf.Tensor(5, shape=(), dtype=int32) -- ha già un valore, senza sessione
```

I cambiamenti più importanti che ha portato TensorFlow 2.0 sono questi:

- **Esecuzione eager di default.** Non serve più `tf.Session` né `tf.placeholder`. Puoi stampare, debuggare e ispezionare qualunque tensore al momento, come con qualunque altra variabile di Python.
- **Keras come API di alto livello ufficiale.** Prima convivevano varie forme di costruire modelli (strati di basso livello, `tf.estimator`, Keras...). Dalla versione 2.0, Keras (`tf.keras`) è il percorso consigliato per costruire, addestrare e distribuire modelli.
- **`tf.function` per recuperare la performance del grafo quando ti serve.** L'esecuzione eager è più comoda, ma un grafo compilato può eseguirsi più velocemente e distribuirsi in più ambienti. Il decoratore `@tf.function` converte una funzione di Python (inclusi i suoi cicli e i suoi condizionali, grazie a uno strumento interno chiamato AutoGraph) in un grafo ottimizzato, senza che tu debba tornare alla sintassi di sessioni. L'hai visto nella pratica nell'esempio di ciclo di addestramento personalizzato con `tf.GradientTape`, dove `train_step` e `test_step` si marcano con `@tf.function`.
- **API semplificata e più consistente.** TensorFlow 2.0 ha eliminato ridondanze tra moduli e unificato comportamenti, il che riduce la quantità di codice ripetitivo (*boilerplate*) necessario per compiti abituali.
- **Un unico formato di salvataggio (`SavedModel`).** Lo stesso modello salvato con `model.save()` può servirsi con TensorFlow Serving, eseguirsi nel browser con TensorFlow.js o convertirsi a TensorFlow Lite per dispositivi mobili, senza riaddestrare nulla.

Nella pratica, tutto il codice che hai scritto in questo corso è già scritto nello stile di TensorFlow 2.x. Conoscere lo stile vecchio ti serve soprattutto per due cose: capire documentazione o progetti legacy che ancora usano `tf.compat.v1`, e apprezzare perché il codice attuale è più corto e più facile da debuggare.

## Esempio di codice commentato: classificare tumori con dati tabulari

Applichiamo tutto quello che precede a un caso reale: il dataset di cancro al seno del Wisconsin, che raccoglie 30 misure numeriche estratte da immagini di biopsie (dimensione, texture, simmetria dei nuclei cellulari, tra gli altri) e un'etichetta che indica se il tumore è benigno o maligno. È un problema di **classificazione binaria** con **dati tabulari**, così combina i primi tre blocchi di questo capitolo.

```python
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Carica i dati. sklearn include questo dataset già pulito, senza valori
# mancanti, così ci risparmiamo quel passo di pulizia.
dati = load_breast_cancer()
X, y = dati.data, dati.target
# In questo dataset, 0 = maligno e 1 = benigno (target_names lo conferma)

# 2. Dividi in addestramento e test, come faresti con qualunque CSV
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# 3. Normalizza le caratteristiche numeriche dentro il modello stesso.
# Lo strato Normalization si "adatta" prima ai dati di addestramento...
normalizador = tf.keras.layers.Normalization(axis=-1)
normalizador.adapt(X_train)

# 4. Costruisci il modello. È un problema di classificazione binaria, così
# l'ultimo strato ha una sola neurone con attivazione sigmoid: l'output
# sarà un numero tra 0 e 1 che interpretiamo come una probabilità.
modello = tf.keras.Sequential([
    normalizador,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 5. Compila con la funzione di perdita giusta per la classificazione binaria
modello.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# 6. Addestra, riservando una parte dei dati di addestramento per validare
storico = modello.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    verbose=0)

# 7. Valuta sul set di test, che il modello non ha mai visto
perdita, precisione = modello.evaluate(X_test, y_test, verbose=0)
print(f"Precisione nel set di test: {precisione:.2%}")
```

Confronta questo esempio con il approccio originale del notebook `BreastCancer.ipynb`: lì si caricavano manualmente quattro file CSV già divisi e normalizzati fuori da TensorFlow, e il modello si compilava con `rmsprop` e `binary_crossentropy` senza strato di normalizzazione. Il risultato è equivalente, ma integrare la divisione dei dati e la normalizzazione dentro questo flusso rende il processo riproducibile ed evita il passo manuale di caricare file ogni volta che esegui il notebook.

## Riassunto dei punti chiave

- La classificazione prevede una categoria di un insieme chiuso di opzioni; la regressione prevede un numero continuo
- La funzione di perdita deve corrispondere al tipo di problema: `binary_crossentropy` per classificazione binaria, `sparse_categorical_crossentropy` per classificazione multiclasse, e `mean_squared_error` o `mean_absolute_error` per regressione
- I dati tabulari in CSV hanno bisogno di preprocessing prima di addestrare: separare caratteristiche da etichette, normalizzare colonne numeriche e convertire colonne categoriche in una rappresentazione numerica
- Gli strati di preprocessing di Keras (`Normalization`, `StringLookup`, `CategoryEncoding`) permettono di integrare quel preprocessing dentro il modello stesso, affinché viaggi con esso quando lo salvi
- La Sequential API serve per pile semplici di strati; la Functional API è necessaria per modelli con più input, più output, strati condivisi o connessioni non lineari
- TensorFlow 2.0 ha sostituito il modello di grafi e sessioni di TensorFlow 1.x con esecuzione eager di default, con Keras come API di alto livello ufficiale e `tf.function` disponibile per recuperare la performance di un grafo compilato quando serve

## Mettilo in pratica

Apri il notebook di pratica di questa lezione (`clase03_practica_clasificacion_regresion_keras.ipynb`) e completa l'esercizio di classificazione con il dataset di cancro al seno: ti manca decidere la funzione di perdita e l'attivazione dell'ultimo strato a partire da quello che hai letto in questo capitolo. Dopo, nel secondo esercizio, costruisci con la Functional API un modello con due input indipendenti che si combinano prima dell'output finale.

Prima di guardare la soluzione, fatti questa domanda: se invece di prevedere se un tumore è benigno o maligno dovessi prevedere la dimensione esatta di quel tumore in millimetri, che cosa cambieresti nel modello? Pensa all'ultimo strato, alla funzione di perdita e alla metrica che useresti per valutare il risultato.
