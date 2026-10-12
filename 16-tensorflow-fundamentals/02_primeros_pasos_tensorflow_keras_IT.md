# 02. Primi passi con TensorFlow e Keras

## Perché questo argomento è importante

Nel capitolo precedente hai visto che Machine Learning (ML) significa addestrare un modello a partire dai dati invece di programmare regole a mano. Adesso è il momento di mettere in pratica quell'idea nel codice. Prima di costruire la tua prima rete neurale devi conoscere la struttura di dati che TensorFlow (TF) usa per rappresentare praticamente tutto: il **tensore**. Una volta che capisci che cosa è un tensore, fare il salto a costruire e addestrare un modello con Keras è una questione di poche righe.

## Che cosa è un tensore

Un **tensore** è un array multidimensionale, simile a un `ndarray` della libreria NumPy, con due proprietà sempre definite: un tipo di dato (`dtype`, per esempio interi o decimali) e una forma (`shape`, cioè quanti elementi ha in ogni dimensione). Pensa ai tensori come una famiglia di contenitori di numeri che scala in base alla complessità del dato:

- Un numero singolo (uno scalare) è un tensore di dimensione zero
- Una lista di numeri (un vettore) è un tensore di una dimensione
- Una tabella di numeri (una matrice) è un tensore di due dimensioni
- Un'immagine a colori, o un lotto di diverse immagini, è un tensore di tre o più dimensioni

TensorFlow usa tensori come sua struttura centrale perché ha bisogno di un formato unico capace di rappresentare tanto un numero isolato quanto un'immagine completa o un lotto di migliaia di esse, e su cui possa applicare le stesse operazioni matematiche ottimizzate —somme, moltiplicazioni di matrici, riduzioni— in modo efficiente in parallelo, sia su CPU (Central Processing Unit) che su GPU (Graphics Processing Unit). A differenza di un array di NumPy, un tensore può risiedere in memoria di un acceleratore come la GPU e, salvo eccezioni, è immutabile: una volta creato, il suo contenuto non si modifica, si crea un nuovo tensore con il risultato di ogni operazione.

## Operazioni di base con i tensori

Puoi creare tensori e operare su di essi con una sintassi molto simile a quella di NumPy:

```python
import tensorflow as tf

# Operazioni matematiche di base su tensori
print(tf.math.add(1, 2))                  # somma
print(tf.math.add([1, 2], [3, 4]))        # somma di vettori, elemento per elemento
print(tf.math.square(5))                  # elevare al quadrato
print(tf.math.reduce_sum([1, 2, 3]))      # somma di tutti gli elementi

# TensorFlow sovraccarica gli operatori soliti di Python
print(tf.math.square(2) + tf.math.square(3))
```

Ogni tensore risultante espone la sua forma e il suo tipo di dato:

```python
x = tf.linalg.matmul([[1]], [[2, 3]])  # moltiplicazione di matrici
print(x)        # tf.Tensor([[2 3]], shape=(1, 2), dtype=int32)
print(x.shape)  # (1, 2)
print(x.dtype)  # <dtype: 'int32'>
```

E puoi convertire tra tensori e array di NumPy senza quasi fatica, grazie al fatto che TensorFlow e NumPy si intendono automaticamente in entrambe le direzioni:

```python
import numpy as np

ndarray = np.ones([3, 3])

# TensorFlow converte automaticamente un ndarray di NumPy in tensore
tensor = tf.math.multiply(ndarray, 42)

# NumPy converte anche automaticamente un tensore in ndarray
print(np.add(tensor, 1))

# Il metodo .numpy() converte un tensore in un array di NumPy in modo esplicito
print(tensor.numpy())
```

Non devi padroneggiare tutte le operazioni disponibili: ciò che conta è che interiorizzzi che, con qualunque cosa tu lavori in TensorFlow —numeri, immagini, testo convertito in numeri—, lo farai sempre su tensori.

## Costruire la tua prima rete neurale con Keras

Con questo puoi già costruire l'esempio di numeri del capitolo precedente (la relazione nascosta Y = 2X - 1) usando **Keras**, l'interfaccia di alto livello di TensorFlow per definire modelli. Stai per creare la rete neurale più semplice possibile: un solo strato, con una sola neurone.

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Uno strato, con una neurone (unit=1), che riceve un unico valore di input
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
```

`Sequential` impila gli strati dove ognuno ha un tensore di input e un tensore di output. Qui c'è solo uno strato `Dense` (denso, cioè completamente connesso) con una neurone: riceve un numero (X) e produce un altro numero (la previsione di Y).

### Compilare il modello: ottimizzatore e funzione di perdita

Prima di addestrare, devi **compilare** il modello indicando due elementi che hai già visto nel capitolo precedente:

```python
model.compile(optimizer='sgd', loss='mean_squared_error')
```

- `loss` (la funzione di perdita) misura quanto la previsione del modello si allontana dal valore reale. Qui usi l'errore quadratico medio (`mean_squared_error`), usuale nei problemi di regressione
- `optimizer` è l'algoritmo che, a partire da quell'errore, decide come aggiustare i parametri del modello per il tentativo successivo. Qui usi la discesa del gradiente stocastica (`sgd`, Stochastic Gradient Descent)

Non devi padroneggiare la matematica di queste funzioni per iniziare a usarle: Keras le incapsula per te. Con la pratica imparerai quale combinazione di perdita e ottimizzatore è appropriata per ogni tipo di problema.

### Fornire i dati di addestramento

```python
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
```

Nota che queste sono le stesse coppie X-Y del capitolo precedente, ora rappresentate come array di NumPy, il formato che Keras si aspetta per addestrare.

### Addestrare con fit

```python
model.fit(xs, ys, epochs=500)
```

Il metodo `fit` (adatta) esegue il ciclo completo che hai descritto nel capitolo precedente: il modello predice, calcola la perdita, l'ottimizzatore propone un aggiustamento e il processo si ripete per le 500 epoche indicate. Quando lo esegui in Colab vedrai come il valore della perdita scende epoca dopo epoca, segnale che il modello sta imparando.

### Predire con predict

```python
print(model.predict(np.array([10.0])))
```

Con il modello ormai addestrato, `predict` ti restituisce la sua stima di Y per un X nuovo che non ha mai visto durante l'addestramento. Se la relazione reale è Y = 2X - 1, per X = 10 ti aspetteresti Y = 19, ma di solito ottieni un valore molto vicino, come 18,9998, e non il numero esatto. Come già sai dal capitolo precedente, questo non è un errore: il modello lavora con probabilità apprese da soli sei esempi, non con certezze.

## Uno sguardo a un modello un po' più complesso

Lo stesso pattern —strati, `compile`, `fit`, `predict`— funziona altrettanto bene in modelli molto più grandi. Per esempio, un classificatore di cifre scritte a mano (il set di dati MNIST) aggiunge più strati e neuroni, ma segue lo stesso ciclo:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),   # appiattisce l'immagine 28x28 in un vettore
    tf.keras.layers.Dense(128, activation='relu'),   # strato denso di 128 neuroni
    tf.keras.layers.Dropout(0.2),                    # spegne il 20% dei neuroni a caso per evitare overfitting
    tf.keras.layers.Dense(10)                        # un output per ogni cifra (0-9)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
```

Non devi capire ogni nuovo strato adesso —`Flatten`, `Dropout`, la funzione di perdita `SparseCategoricalCrossentropy`— perché li vedrai in dettaglio nelle prossime lezioni dedicate alla visione artificiale. Resta con l'idea centrale: non importa se la tua rete ha una neurone o centinaia di migliaia, definisci sempre gli strati, compili con un ottimizzatore e una perdita, addestri con `fit` e predici con `predict`.

## Riassunto dei punti chiave

- Un tensore è un array multidimensionale con un tipo di dato e una forma definiti; è la struttura centrale di TensorFlow perché permette di rappresentare e operare in modo efficiente su qualunque tipo di dato numerico
- I tensori si convertono automaticamente (o esplicitamente, con `.numpy()`) verso e da array di NumPy
- In Keras, un modello `Sequential` si costruisce impilando gli strati; il più basilare è `Dense`, uno strato completamente connesso
- `compile` fissa la funzione di perdita (misura l'errore) e l'ottimizzatore (decide come correggerlo)
- `fit` esegue il ciclo di addestramento per il numero di epoche indicato; `predict` usa il modello ormai addestrato su dati nuovi
- Lo stesso pattern di strati + compile + fit + predict si applica da una rete di una sola neurone fino a modelli molto più complessi

## Mettilo in pratica

Apri un nuovo notebook in Google Colab e riproduci l'esempio della singola neurone, ma con una relazione diversa: Y = 3X + 1. Definisci tu gli array `xs` e `ys` con almeno sei coppie di valori che seguono quella formula, compila il modello come nell'esempio e addestralo con `fit`. Quando finisce, usa `predict` per stimare Y quando X è 10. Confronta il risultato con il valore esatto (31) e annota la differenza: è la stessa incertezza che hai visto nel capitolo precedente, ora verificata da te con codice vero.
