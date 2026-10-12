# 04. Reti neurali convoluzionali

## Perché questo argomento è importante

Nel capitolo precedente hai addestrato una rete densa che classificava capi d'abbigliamento da Fashion MNIST con un'accuratezza accettabile, ma con una condizione molto ristretta: il capo doveva essere centrato, in scala di grigi e occupare tutta l'immagine di 28x28 pixel. Non appena gli dai una foto reale, con sfondo, con il capo spostato o ruotato, quel modello fallisce, perché ha imparato a memorizzare pattern di pixel specifici e non a identificare le caratteristiche visive (bordi, curve, texture) che rendono uno stivale uno stivale.

Le reti neurali convoluzionali, o CNN (Convolutional Neural Network), risolvono proprio quel problema. Invece di dare direttamente i pixel grezzi in input a uno strato denso, prima passano l'immagine per un processo di filtraggio che mette in evidenza le sue caratteristiche più rilevanti. Questo è ciò che permette alle CNN di funzionare bene con immagini di oggetti che non sono perfettamente centrati, e sono la base di praticamente qualunque sistema moderno di riconoscimento visivo, dal classificare foto alla guida autonoma.

## Che cosa è una convoluzione

Immagina che tu abbia una lente d'ingrandimento quadrata piccola, di tre pixel per tre pixel, e la fai scorrere per tutta l'immagine, posizione per posizione. In ogni posizione, la lente non guarda solo il pixel centrale: combina quel pixel con i suoi otto vicini immediati applicando una formula fissa, e il risultato diventa il nuovo valore di quel pixel in un'immagine trasformata. Questo è, in essenza, una **convoluzione**: un'operazione che percorre l'intera immagine applicando la stessa combinazione di vicini in ogni posizione.

La "formula fissa" che si applica in ogni posizione è il **filtro** (anche chiamato kernel), una piccola matrice di numeri. Ogni valore del filtro si moltiplica per il pixel dell'immagine che ha sotto, e poi si sommano tutti i risultati per ottenere il nuovo valore del pixel centrale. Per esempio, con un filtro di 3x3 come questo:

```
[-1, -2, -1]
[ 0,  0,  0]
[ 1,  2,  1]
```

Moltiplichi ognuno dei nove pixel vicini per il valore del filtro che gli corrisponde, sommi i nove risultati e quello è il nuovo valore del pixel centrale. Questo filtro specifico mette in evidenza i bordi orizzontali dell'immagine; se usi un filtro con gli stessi numeri ma orientato in colonne invece che in righe, metti in evidenza i bordi verticali. La scelta dei numeri del filtro determina quale tipo di caratteristica visiva si rende più visibile.

Pensa a ogni filtro come a un rilevatore specializzato in un tipo di pattern specifico: uno può attivarsi davanti a bordi verticali, uno davanti a curve, uno davanti a macchie di colore uniforme. Nessuno di loro "capisce" che sta guardando una scarpa da ginnastica: ognuno semplicemente risponde con più o meno intensità in base a quello che trova sotto, pixel per pixel.

## Come si imparano i filtri

Ecco il colpo di scena importante: in una CNN, tu non progetti a mano i numeri di ogni filtro. All'inizio dell'addestramento, i filtri si inizializzano con valori casuali, proprio come i pesi di uno strato denso. Durante l'addestramento, la rete aggiusta quei valori mediante retropropagazione (*backpropagation*), lo stesso meccanismo che aggiusta i pesi degli strati densi, cercando i filtri che meglio aiutano a distinguere una classe da un'altra.

Il risultato è che, dopo l'addestramento, alcuni filtri dei primi strati finiscono per rilevare bordi e texture semplici, mentre i filtri degli strati più profondi combinano quei segnali per riconoscere forme più complesse, come il contorno di una manica o la sagoma di una suola. Questo processo di estrarre caratteristiche sempre più astratte strato per strato si chiama **estrazione di caratteristiche** (*feature extraction*), ed è il motivo per cui una CNN non ha bisogno che tu le dica esplicitamente cosa cercare: lo impara dai dati, proprio come ha imparato i pesi di una rete densa.

In TensorFlow, uno strato convoluzionale si definisce con `Conv2D`:

```python
tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1))
```

Qui `64` è il numero di filtri distinti che lo strato impara in parallelo (ognuno specializzato in una caratteristica diversa), e `(3, 3)` è la dimensione di ogni filtro. Nota anche che la forma di input diventa `(28, 28, 1)`: l'ultimo numero è il canale di colore (1 per scala di grigi, 3 per immagini a colori con canali rosso, verde e blu).

## Che cosa offre il pooling

Dopo uno strato convoluzionale, è usuale aggiungere uno strato di **pooling** (raggruppamento), il cui compito è ridurre la dimensione dell'immagine mentre conserva l'essenziale di quello che i filtri hanno già rilevato.

La variante più comune è **max pooling**: l'immagine si percorre in blocchi (per esempio, di 2x2 pixel) e da ogni blocco si conserva solo il valore più alto. Il risultato è un'immagine con metà dell'altezza e metà della larghezza, ma che continua a conservare le zone dove i filtri hanno risposto con più forza.

Pensa al pooling come a riassumere un paragrafo lungo in una frase: perdi dettagli, ma conservi l'idea principale. Nel caso di un'immagine, il pooling ti permette di:

- Ridurre la quantità di calcoli che hanno bisogno gli strati successivi, perché lavorano con un'immagine più piccola
- Rendere la rete più robusta a piccoli spostamenti dell'oggetto nell'immagine, perché non importa più il pixel esatto dove è apparsa la caratteristica, solo che è apparsa in quella zona
- Permettere agli strati convoluzionali successivi di lavorare su una rappresentazione più compatta e concentrarsi su caratteristiche sempre più generali

In Keras, il pooling si aggiunge con `MaxPooling2D`:

```python
tf.keras.layers.MaxPooling2D(2, 2)
```

## Come si costruisce una CNN in Keras

Una CNN tipica combina blocchi di convoluzione e pooling, e termina con gli stessi strati densi che già conosci dal capitolo 3:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

Nota la struttura: prima due blocchi di convoluzione più pooling, che hanno il compito di estrarre e riassumere le caratteristiche visive, e poi un `Flatten` seguito dagli stessi strati densi del capitolo precedente, che si incaricano della classificazione finale a partire da quelle caratteristiche. È esattamente lo stesso pattern che descrive il protocollo ufficiale: filtrare prima di classificare, invece di classificare direttamente sui pixel grezzi.

Un dettaglio importante che cambia rispetto al capitolo 3: poiché `Conv2D` si aspetta immagini con un canale esplicito, devi dare a ogni immagine la forma `(28, 28, 1)` invece di `(28, 28)`. Con Fashion MNIST, ciò significa riformattare i dati prima di addestrare:

```python
training_images = training_images.reshape(60000, 28, 28, 1) / 255.0
test_images = test_images.reshape(10000, 28, 28, 1) / 255.0
```

## Confrontare la CNN con la rete densa

Il resto del processo (compilare, addestrare, valutare) è identico a quello del capitolo 3:

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
```

Se addestri entrambi i modelli con lo stesso numero di epoche su Fashion MNIST, normalmente vedrai che la CNN raggiunge un'accuratezza uguale o migliore della rete densa, sebbene impiega più tempo per epoca perché ha più calcoli da fare (ogni filtro si applica a tutta l'immagine). Il guadagno si nota ancora di più quando le immagini sono più grandi, più complesse o l'oggetto non è perfettamente centrato, che è esattamente lo scenario dove la rete densa del capitolo 3 iniziava a fallire.

## Costruire un classificatore di immagini completo

Con il modello già addestrato, un classificatore di immagini completo ha bisogno inoltre di un modo chiaro per interpretare e comunicare le sue previsioni. I passaggi soliti, dall'inizio alla fine, sono:

1. Caricare e preparare i dati (inclusa la normalizzazione e il reshape a 4 dimensioni che hai visto prima)
2. Definire l'architettura del modello (blocchi di convoluzione e pooling, più strati densi)
3. Compilare il modello con un ottimizzatore, una funzione di perdita e una metrica
4. Addestrare con `model.fit`, e validare con dati di test tramite `model.evaluate`
5. Generare previsioni con `model.predict` su immagini nuove
6. Interpretare l'output: poiché l'ultimo strato usa softmax, ogni previsione è una lista di probabilità, una per classe, e la classe prevista è quella che ha la probabilità più alta (con `np.argmax`)

Per esempio, per classificare una sola immagine nuova devi ricordare che Keras si aspetta sempre un lotto (*batch*) di immagini, anche se di una sola:

```python
import numpy as np

img = test_images[1]                    # una sola immagine, forma (28, 28, 1)
img_batch = np.expand_dims(img, 0)       # forma (1, 28, 28, 1): un lotto di un'immagine

prediction = model.predict(img_batch)
predicted_class = np.argmax(prediction[0])
print(f'Classe prevista: {predicted_class}, probabilità: {prediction[0][predicted_class]:.2%}')
```

Con questo hai un classificatore di immagini completo: dai dati grezzi fino a una previsione interpretabile su un'immagine che il modello non ha mai visto.

## Esempio di codice commentato

```python
import numpy as np
import tensorflow as tf

# Carichiamo e normalizziamo Fashion MNIST, come nel capitolo 3
fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Conv2D ha bisogno di un canale esplicito: passiamo da (28, 28) a (28, 28, 1)
training_images = training_images.reshape(60000, 28, 28, 1) / 255.0
test_images = test_images.reshape(10000, 28, 28, 1) / 255.0

# Architettura CNN: due blocchi di convoluzione + pooling, e poi classificazione densa
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Accuratezza CNN in test: {test_accuracy:.4f}')

# Previsione su una sola immagine nuova
img_batch = np.expand_dims(test_images[1], 0)
prediction = model.predict(img_batch)
predicted_class = np.argmax(prediction[0])
print(f'Classe prevista: {predicted_class} (reale: {test_labels[1]})')
```

## Riassunto dei punti chiave

- Una convoluzione fa scorrere un filtro (una piccola matrice di numeri) per tutta l'immagine, combinando ogni pixel con i suoi vicini per mettere in evidenza una caratteristica visiva specifica, come un bordo verticale od orizzontale
- I filtri non si progettano a mano: si inizializzano con valori casuali e si aggiustano durante l'addestramento mediante retropropagazione, proprio come i pesi di uno strato denso
- Il pooling (normalmente max pooling) riduce la dimensione dell'immagine tenendosi il valore più alto di ogni blocco di pixel, il che alleggerisce i calcoli e dà robustezza davanti a piccoli spostamenti dell'oggetto
- Una CNN tipica combina più blocchi di `Conv2D` più `MaxPooling2D` per estrarre caratteristiche, e termina con `Flatten` e strati `Dense` per classificare, come la rete del capitolo 3
- Un classificatore di immagini completo non finisce in `model.fit`: include preparare i dati con la forma corretta, addestrare, valutare con dati non visti e interpretare la previsione finale con `np.argmax` sull'output di softmax

## Mettilo in pratica

Apri il notebook di teoria di questa lezione, addestra la CNN di questo capitolo su Fashion MNIST per cinque epoche e annota la sua accuratezza nel set di test. Confrontala con l'accuratezza che hai ottenuto con la rete densa del capitolo 3 usando lo stesso numero di epoche. Poi, cambia il numero di filtri del primo strato `Conv2D` da 64 a 32 e addestra di nuovo. L'accuratezza cambia molto? E il tempo di addestramento per epoca? Scrivi le tue conclusioni: ti serviranno di base per gli esercizi del notebook di pratica, dove modificherai l'architettura della CNN per conto tuo.
