[⬅ Torna all'indice](00_indice_IT.md)

# Capitolo 1 — Fondamenti di IA, reti neurali e TensorFlow

## Perché ti importa questo

Ogni volta che il tuo telefono riconosce il tuo viso per sbloccarsi, un traduttore converte una frase all'istante, o una piattaforma di streaming indovina con la giusta raccomandazione, dietro c'è una rete neurale addestrata. L'Intelligenza Artificiale (IA) non è più fantascienza: è la tecnologia che decide cosa vedi, cosa compri e con chi parli. E il pezzo che rende possibile costruire quelle reti in modo pratico si chiama TensorFlow. In questo capitolo fai i tuoi due primi passi: capisci cosa sia una rete neurale e addestri la tua con solo poche righe di codice.

## Cos'è una rete neurale (con un'analogia)

Immagina una commissione di dieci persone che deve decidere se una foto mostra un gatto. Ognuna della commissione si concentra su un indizio diverso — le orecchie, i baffi, la forma degli occhi — e dà il suo voto con maggiore o minore convinzione. Alla fine, qualcuno somma quei voti ponderati e trae una conclusione. Una rete neurale artificiale funziona allo stesso modo: è formata da "neuroni" organizzati in strati, ognuno si concentra su un pattern diverso dai dati di input, e i suoi voti ponderati (chiamati "pesi") vengono combinati strato dopo strato fino a produrre una risposta finale.

L'apprendimento profondo, o Deep Learning (DL), è semplicemente usare reti con molti strati impilati: più strati ci sono, più pattern complessi la rete può imparare a riconoscere da sola, senza che tu debba programmare quelle regole manualmente.

TensorFlow è la scatola degli attrezzi — open source e creata da Google — che ti permette di definire, addestrare e usare queste reti senza dover scrivere tu l'algebra che sta dietro ogni "voto" della commissione.

## Installa TensorFlow sul tuo computer

Ti serve Python installato e, nel tuo terminale, eseguire:

```bash
pip install tensorflow
```

Questo è sufficiente per lavorare con questo manuale. Se il tuo computer ha una scheda grafica (Graphics Processing Unit, GPU) di NVIDIA e vuoi addestrare più velocemente, TensorFlow la rileva automaticamente a patto che tu abbia installati i driver CUDA corrispondenti; non è obbligatorio per seguire il corso.

Per verificare che tutto funzioni, apri Python ed esegui:

```python
import tensorflow as tf
print(tf.constant("¡Hola, TensorFlow!"))
```

Se vedi il messaggio stampato senza errori, sei pronto per addestrare la tua prima rete.

## La tua prima rete neurale con Keras

Keras è l'interfaccia di alto livello integrata in TensorFlow: ti lascia definire una rete come un elenco di strati, senza preoccuparti ancora dei dettagli interni. Andrai ad addestrare una rete che riconosce cifre scritte a mano, usando il dataset MNIST — settantamila immagini di numeri da zero a nove che sono già incluse in TensorFlow —.

```python
import tensorflow as tf

# Carichiamo il dataset MNIST: immagini di 28x28 pixel con la cifra corretta
(x_entreno, y_entreno), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizziamo i pixel a un intervallo da 0 a 1 affinché la rete impari meglio
x_entreno, x_test = x_entreno / 255.0, x_test / 255.0

# Definiamo la rete: appiattisce l'immagine e la passa attraverso due strati densi
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

modelo.fit(x_entreno, y_entreno, epochs=5)
modelo.evaluate(x_test, y_test)
```

In cinque epoch — cinque passate complete attraverso i dati di addestramento — questo modello di solito supera il 97% di accuratezza nel riconoscere cifre che non ha mai visto.

## Cosa succede dentro: tensori, pesi e gradienti

Keras nasconde molti dettagli apposta, ma comprenderli ti aiuta a risolvere i problemi in seguito. Un tensore è, semplicemente, un array multidimensionale: un numero solitario è un tensore di dimensione zero, un elenco è di dimensione uno, un'immagine in bianco e nero è di dimensione due, e un'immagine a colori è di dimensione tre. TensorFlow rappresenta così sia i tuoi dati di input che i pesi interni della rete.

Ogni strato denso di quelli che hai usato sopra fa, in fondo, un'operazione molto semplice: moltiplica l'input per una matrice di pesi, somma un bias, e applica una funzione di attivazione come ReLU (Rectified Linear Unit) che introduce non-linearità — senza di essa, impilare strati non servirebbe a nulla, perché diverse trasformazioni lineari consecutive rimangono comunque una trasformazione lineare —.

Per addestrare la rete, TensorFlow calcola l'errore tra la predizione e la risposta corretta (la "funzione di perdita") e usa un algoritmo chiamato discesa del gradiente per aggiustare i pesi nella direzione che riduce quell'errore, passo dopo passo, durante ogni epoch.

## Costruisci uno strato denso con le tue stesse mani

Andiamo a riprodurre, con i pezzi di basso livello di TensorFlow, esattamente quello che Keras faceva per te. Così capisci cosa si nasconde dietro `Dense`.

```python
import tensorflow as tf

# Stessi dati di prima, appiattiti a vettori di 784 valori (28 x 28)
(x_entreno, y_entreno), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_entreno = (x_entreno / 255.0).reshape(-1, 784).astype("float32")
y_entreno = tf.keras.utils.to_categorical(y_entreno, 10)

# Pesi e bias come variabili addestrabili, inizializzate casualmente
pesos = tf.Variable(tf.random.normal([784, 10], stddev=0.1))
sesgos = tf.Variable(tf.zeros([10]))

optimizador = tf.keras.optimizers.Adam(learning_rate=0.01)

def prediccion(x):
    return tf.nn.softmax(tf.matmul(x, pesos) + sesgos)

# Addestriamo per 3 epoch, in lotti di 128 immagini
lote = 128
for epoca in range(3):
    for i in range(0, len(x_entreno), lote):
        x_lote = x_entreno[i:i + lote]
        y_lote = y_entreno[i:i + lote]
        with tf.GradientTape() as cinta:
            y_predicha = prediccion(x_lote)
            perdida = tf.keras.losses.categorical_crossentropy(y_lote, y_predicha)
        gradientes = cinta.gradient(perdida, [pesos, sesgos])
        optimizador.apply_gradients(zip(gradientes, [pesos, sesgos]))
    print(f"Epoch {epoca + 1} completata")
```

`tf.GradientTape` è il "nastro" che registra le operazioni matematiche mentre si eseguono, per poter calcolare poi il gradiente — la direzione di aggiustamento — di ogni peso. È esattamente quello che Keras fa internamente ogni volta che chiami `fit`.

## Riepilogo dei punti chiave

- Una rete neurale combina "voti" ponderati di neuroni organizzati in strati per produrre una predizione.
- L'apprendimento profondo impila molti strati per riconoscere pattern complessi senza programmarli manualmente.
- TensorFlow ti dà due livelli di lavoro: Keras (alto livello, veloce da scrivere) e le operazioni di basso livello (tensori, variabili, `GradientTape`) che Keras usa dietro le quinte.
- L'addestramento consiste nell'aggiustare pesi e bias, epoch dopo epoch, per ridurre l'errore tra la predizione e la risposta corretta.

## Mettilo in pratica

Prendi il primo esempio di questo capitolo (la rete con Keras) e **cambia** il numero di neuroni dello strato nascosto da 128 a 32, e il numero di epoch da 5 a 10. **Esegui** di nuovo l'addestramento e **confronta** l'accuratezza finale. Migliora, peggiora o rimane uguale? Annota la tua ipotesi prima di eseguire il codice, e verifica se hai indovinato.

---

**Materiale sorgente di questo capitolo**: lezione 1 tema 1 ("Introduction to Deep Learning with Neural Networks and TensorFlow"); lezione 2 temi 1, 2, 3, 4, 9 e 10.

[⬅ Torna all'indice](00_indice_IT.md) · [Capitolo successivo ➡](02_sentiment_analysis_IT.md)
