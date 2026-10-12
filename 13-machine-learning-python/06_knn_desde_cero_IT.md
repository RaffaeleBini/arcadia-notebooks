# 6. Programmare KNN da zero

## Perché questo ti riguarda

Nel capitolo precedente hai usato `KNeighborsClassifier` di scikit-learn per classificare i tumori dal dataset sul cancro al seno. Con poche righe hai ottenuto un modello funzionante, ma quella comodità nasconde il meccanismo reale. Ora aprirai la scatola nera.

In questo capitolo programmi il tuo classificatore dei k vicini più prossimi (KNN, dalle sue sigle in inglese K-Nearest Neighbors) usando solo Python e NumPy, senza scikit-learn. Lo scopo non è sostituire la libreria, che è molto più ottimizzata, bensì capire esattamente cosa accade dal momento in cui gli passi un punto nuovo fino a quando ricevi una previsione. Quella comprensione ti aiuta a debuggare modelli, scegliere meglio gli iperparametri e riconoscere quando un algoritmo non è lo strumento adatto per il tuo problema.

Alla fine, confronterai la tua implementazione con quella di scikit-learn sullo stesso dataset e verificherai che entrambe raggiungono risultati molto simili.

## L'algoritmo passo dopo passo

Pensa a come scegli un ristorante in una città che non conosci. Non chiedi a tutti gli abitanti cosa ne pensano di ogni locale: chiedi alle persone che hai più vicino, magari le cinque o dieci che vivono nel tuo stesso quartiere, e ti fidi dell'opinione della maggior parte di quel gruppo ridotto. Questo è esattamente quello che fa KNN con i dati.

Il processo si riduce a tre azioni concrete:

1. Calcolare la distanza tra il punto nuovo e ognuno dei punti dell'insieme di addestramento
2. Ordinare quelle distanze da piccola a grande e tenere le k più piccole, cioè gli k vicini più prossimi
3. Votare: la classe che si ripete più spesso tra quegli k vicini diventa la previsione per il punto nuovo

Non c'è una fase di addestramento che regoli i parametri interni come accade in altri algoritmi. KNN salva l'insieme di addestramento così com'è e fa tutto il lavoro al momento della previsione. Per questo si chiama un apprendista pigro (lazy learner): non impara un modello compatto, semplicemente confronta.

## Programmare il tuo classificatore KNN in Python

Inizia dalla parte più piccola: una funzione che calcoli la distanza euclidea tra due punti. NumPy ti permette di farlo in una sola riga grazie a `numpy.linalg.norm`, che calcola la lunghezza del vettore differenza tra i due punti.

Con quel pezzo puoi già costruire la funzione principale, `knn_predecir`, che riceve l'insieme di addestramento, il punto nuovo e il valore di k:

```python
import numpy as np
from collections import Counter


def distanza_euclidea(punto_a, punto_b):
    """Distanza in linea retta tra due punti."""
    punto_a = np.array(punto_a)
    punto_b = np.array(punto_b)
    return np.linalg.norm(punto_a - punto_b)


def knn_predecir(dati_addestramento, punto_nuovo, k):
    """
    dati_addestramento: lista di tuple (caratteristiche, etichetta)
    punto_nuovo: array con le caratteristiche del punto da classificare
    k: numero di vicini che partecipano alla votazione
    """
    if len(dati_addestramento) < k:
        raise ValueError("k non può superare il numero di punti di addestramento")

    distanze = []
    for caratteristiche, etichetta in dati_addestramento:
        distanza = distanza_euclidea(caratteristiche, punto_nuovo)
        distanze.append((distanza, etichetta))

    # Passo 2: ordinare e tenere gli k vicini più prossimi
    distanze.sort(key=lambda par: par[0])
    k_vicini = distanze[:k]

    # Passo 3: votare la classe maggioritaria
    voti = [etichetta for _, etichetta in k_vicini]
    classe_vincente = Counter(voti).most_common(1)[0][0]
    return classe_vincente
```

Presta attenzione a tre dettagli del codice:

- `dati_addestramento` è una lista di tuple che accoppia ogni vettore di caratteristiche con la sua etichetta, così la funzione conosce sia dove è ogni punto che a quale classe appartiene
- Il controllo iniziale evita un errore frequente: chiedere più vicini di quanti ne esistano nell'insieme di addestramento
- `Counter.most_common(1)` ti restituisce, in una sola chiamata, l'etichetta che appare con più frequenza tra i voti

Con questo hai un classificatore KNN completo. È più lento di quello di scikit-learn perché percorre la lista con un ciclo esplicito in Python anziché usare operazioni vettorizzate interne in C, ma fa esattamente la stessa cosa a livello concettuale.

## Confrontare la tua implementazione con quella di scikit-learn

È il momento della prova del fuoco: applicare `knn_predecir` sullo stesso dataset sul cancro al seno che hai usato nel capitolo cinque e verificare se la percentuale di successi si avvicina a quella di `KNeighborsClassifier`.

Poiché KNN si basa su distanze, e alcune caratteristiche di questo dataset hanno scale molto diverse tra loro, scala prima i dati con `StandardScaler`. Così nessuna caratteristica domina il calcolo della distanza solo per avere valori più grandi.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Caricare lo stesso dataset del capitolo precedente
dati = load_breast_cancer()
X, y = dati.data, dati.target

# Scalare le caratteristiche così tutte pesino uguale nella distanza
scalatore = StandardScaler()
X_scalato = scalatore.fit_transform(X)

# Dividere in addestramento e test
X_train, X_test, y_train, y_test = train_test_split(
    X_scalato, y, test_size=0.2, random_state=42
)

# Preparare i dati nel formato che si aspetta la tua funzione
dati_addestramento = list(zip(X_train, y_train))

k = 5

# Previsioni con il tuo KNN
previsioni_proprie = [
    knn_predecir(dati_addestramento, punto, k) for punto in X_test
]
precisione_propria = accuracy_score(y_test, previsioni_proprie)

# Previsioni con scikit-learn
modello_sklearn = KNeighborsClassifier(n_neighbors=k)
modello_sklearn.fit(X_train, y_train)
previsioni_sklearn = modello_sklearn.predict(X_test)
precisione_sklearn = accuracy_score(y_test, previsioni_sklearn)

print(f"Precisione del tuo KNN:      {precisione_propria:.3f}")
print(f"Precisione di scikit-learn:  {precisione_sklearn:.3f}")
```

Se esegui questo codice, vedrai che entrambe le precisioni si aggirano intorno al 95-97%, con differenze minime tra l'una e l'altra. Questo conferma qualcosa di importante: la tua implementazione non è un'approssimazione, è lo stesso algoritmo, solo senza le ottimizzazioni interne di scikit-learn. Le piccole differenze che potresti osservare di solito sono dovute a come ogni implementazione risolve i pareggi nella votazione, non a un errore di concetto.

**Esegui questo blocco con diversi valori di k, come 3, 7 o 11, e osserva come cambia la precisione di entrambe le versioni.** Noterai che né la tua funzione né quella di scikit-learn migliorano costantemente all'aumentare di k: da un certo punto in poi, un k troppo grande inizia a diluire l'informazione locale e la precisione si stabilizza o persino diminuisce.

## Vantaggi e limiti di KNN

Programmare l'algoritmo da zero rende molto più chiare le sue forze e debolezze rispetto all'usarlo come scatola chiusa.

**Costo computazionale.** Ogni previsione costringe a calcolare la distanza tra il punto nuovo e tutti i punti dell'insieme di addestramento. Se hai cento punti, sono cento calcoli di distanza; se ne hai un milione, sono un milione, e questo solo per classificare un unico punto nuovo. Poiché non esiste una fase di addestramento che riassuma l'informazione in un modello compatto, KNN non scala bene quando il volume di dati cresce molto, e ogni query diventa più lenta man mano che aggiungi più esempi.

**Sensibilità alla scala delle caratteristiche.** L'algoritmo presume che tutte le dimensioni contribuiscano allo stesso modo alla distanza. Se una caratteristica si misura in migliaia e un'altra in decimali, la prima domina il calcolo anche se non è più rilevante per il problema. Per questo hai scalato i dati prima di confrontare la tua funzione con scikit-learn: senza quel passaggio, il confronto sarebbe stato ingiusto e i risultati, peggiori di quanto previsto.

**Quando evitare KNN.** Conviene scartarlo, o almeno usarlo con cautela, in questi casi:

- Dataset molto grandi dove hai bisogno di previsioni veloci e costanti in produzione
- Spazi con molte dimensioni, dove il concetto di "vicinanza" perde significato perché quasi tutti i punti finiscono a distanze simili tra loro
- Classi molto squilibrate, dove una classe maggioritaria può dominare la votazione anche se il punto nuovo somiglia di più alla classe minoritaria
- Dati con molte caratteristiche irrilevanti, che aggiungono rumore alla distanza senza fornire informazione utile

In cambio, KNN rimane un'opzione ragionevole quando l'insieme di dati è piccolo o medio, le classi sono ragionevolmente equilibrate e vuoi un modello facile da spiegare a qualcuno senza formazione tecnica.

## Riepilogo: cosa porti con te da questo capitolo

- Programmare KNN da zero si riduce a tre passaggi: calcolare distanze, ordinarle e votare la classe maggioritaria tra gli k vicini più prossimi
- La funzione `knn_predecir` ha bisogno dell'insieme di addestramento con le sue etichette, il punto nuovo e il valore di k
- NumPy semplifica il calcolo della distanza euclidea con `numpy.linalg.norm`, evitando cicli manuali su ogni dimensione
- La tua implementazione e quella di scikit-learn raggiungono precisioni molto simili sullo stesso dataset, il che conferma che entrambe applicano lo stesso principio
- KNN è un apprendista pigro: non costruisce un modello compatto, confronta il punto nuovo con tutto l'insieme di addestramento ad ogni previsione
- Il costo computazionale e la sensibilità alla scala delle caratteristiche sono i due limiti che devi sorvegliare di più

## Mettilo in pratica

**Verifica con i tuoi dati quanto si rallenta la tua funzione rispetto a quella di scikit-learn man mano che cresce l'insieme di addestramento.** Usa il modulo `time` per misurare il tempo di entrambe le versioni con diverse dimensioni di dati:

```python
import time

tamanos = [50, 100, 200, 400]

for tamano in tamanos:
    sottinsieme = dati_addestramento[:tamano]

    inizio = time.perf_counter()
    for punto in X_test:
        knn_predecir(sottinsieme, punto, k)
    tempo_proprio = time.perf_counter() - inizio

    modello_sklearn.fit(X_train[:tamano], y_train[:tamano])
    inizio = time.perf_counter()
    modello_sklearn.predict(X_test)
    tempo_sklearn = time.perf_counter() - inizio

    print(
        f"Dimensione {tamano}: la tua funzione {tempo_proprio:.4f} s, "
        f"scikit-learn {tempo_sklearn:.4f} s"
    )
```

**Annota i risultati in una tabella e confronta la curva di crescita di ogni versione.** Vedrai che il tempo della tua funzione cresce molto più rapidamente di quello di scikit-learn man mano che aumenta la dimensione dell'insieme di addestramento. Questa differenza è la prova pratica di tutto quello che hai letto sul costo computazionale di KNN: la libreria usa strutture di dati e ottimizzazioni interne che la tua versione, scritta per imparare l'algoritmo, non include.

---

**Sigle usate in questo capitolo:** KNN (K-Nearest Neighbors, k vicini più prossimi)
