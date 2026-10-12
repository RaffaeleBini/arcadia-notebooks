[⬅ Torna all'indice](00_indice_IT.md)

# Capitolo 4 — Reti neurali convoluzionali: fondamenti

## Perché ti importa questo

Le reti dense che hai usato fino ad ora trattano ogni pixel come un dato indipendente, senza considerare che un pixel e i suoi vicini di solito fanno parte dello stesso tratto, bordo o oggetto. Le reti convoluzionali, o Convolutional Neural Network (CNN), sono progettate specificamente per sfruttare quella struttura spaziale. Sono alla base di quasi qualsiasi sistema di visione artificiale che usi oggi: riconoscimento facciale, automobili autonome, diagnosi per immagine o ricerca visiva.

## Come "vede" una CNN (con un'analogia)

Immagina che esamini una fotografia con una lente d'ingrandimento piccola, spostandola poco a poco per tutta l'immagine senza saltarti nessun pezzo. In ogni posizione, la lente d'ingrandimento ti permette di notare un dettaglio locale: un bordo, un angolo, una texture. Uno strato convoluzionale fa esattamente quello: fa scorrere un filtro (una lente d'ingrandimento) di piccolo tamaño — per esempio 3x3 pixel — per tutta l'immagine, e in ogni posizione calcola un singolo numero che riassume quello che "vede" quel filtro lì. Il risultato è una "mappa di caratteristiche" che segnala dove appare quel pattern specifico nell'immagine completa.

Dopo aver rilevato pattern locali, uno strato di "pooling" (raggruppamento) riassume quella informazione riducendo la dimensione della mappa: per esempio, prendendo solo il valore massimo di ogni blocco di 2x2. È come riassumere un paragrafo nella sua frase più importante: perdi il dettaglio, ma conservi l'essenziale e riduci il volume di informazione da elaborare.

## Costruisci la tua prima CNN con Keras

```python
import tensorflow as tf

(x_entreno, y_entreno), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Gli strati convoluzionali si aspettano un canale di colore esplicito, anche se è 1 (bianco e nero)
x_entreno = (x_entreno / 255.0).reshape(-1, 28, 28, 1)
x_test = (x_test / 255.0).reshape(-1, 28, 28, 1)

modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax"),
])

modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

modelo.fit(x_entreno, y_entreno, epochs=5, validation_split=0.1)
modelo.evaluate(x_test, y_test)
```

Nota il pattern che si ripete: convoluzione, pooling, convoluzione, pooling. Ogni blocco impara pattern più complessi del precedente — il primo strato rileva bordi e texture semplici; il secondo combina quei bordi in forme più elaborati —. Alla fine, `Flatten` converte la mappa di caratteristiche risultante in un vettore, e uno strato denso fa la classificazione finale, esattamente come nel capitolo 1.

## Riepilogo dei punti chiave

- Uno strato convoluzionale fa scorrere filtri piccoli per tutta l'immagine per rilevare pattern locali (bordi, texture, forme).
- Uno strato di pooling riduce la dimensione della mappa di caratteristiche conservando l'informazione più rilevante.
- Impilare diversi blocchi di convoluzione e pooling permette alla rete di imparare pattern sempre più complessi e astratti.
- Alla fine di una CNN quasi sempre trovi uno strato `Flatten` seguito da uno o più strati densi, proprio come nelle reti che già conosci.

## Mettilo in pratica

**Aggiungi** una terza coppia di strati `Conv2D` + `MaxPooling2D` alla rete precedente, con 128 filtri. **Addestra** il modello e **annota** se l'accuratezza migliora rispetto alla versione di due blocchi, e quanto tempo impiega ogni epoch sul tuo computer.

---

**Materiale sorgente di questo capitolo**: lezione 4 temi 1 e 2 ("Convolutional Neural Networks Basics", "Convolutional Neural Networks with TensorFlow"); `CNN_tensorFlow_Tutorial.ipynb`.

[⬅ Capitolo precedente](03_rnn_lstm_IT.md) · [Torna all'indice](00_indice_IT.md) · [Capitolo successivo ➡](05_tflearn_keras_IT.md)
