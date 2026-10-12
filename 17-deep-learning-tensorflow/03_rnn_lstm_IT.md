[⬅ Torna all'indice](00_indice_IT.md)

# Capitolo 3 — Reti neurali ricorrenti (RNN/LSTM)

## Perché ti importa questo

Quando leggi la parola "banca", il suo significato dipende da quello che hai letto prima: "mi sono seduto sulla panchina del parco" non è la stessa cosa di "sono andato in banca a prelevare denaro". Le reti che hai visto fino ad ora trattano ogni input in modo isolato, senza memoria di quello che è venuto prima. Per sequenze — testo, audio, serie temporali di borsa, sensori — ti serve un'architettura che ricordi il contesto. Quella è la Rete Neurale Ricorrente, o Recurrent Neural Network (RNN).

## Una rete con memoria (con un'analogia)

Immagina che leggi un libro e, alla fine di ogni pagina, scrivi un riassunto di una riga di tutto quello che hai letto fino a quel momento. Quando arrivi all'ultima pagina, non ricominci da zero: usi quel riassunto accumulato per capire il finale. Una RNN fa qualcosa di simile: elabora una sequenza passo dopo passo, e in ogni passo combina l'input attuale con un "riassunto" (lo stato nascosto) di tutto quello che ha visto prima.

Il problema di una RNN semplice è che quel riassunto tende a "dimenticare" quello che è successo molti passi fa, proprio come ti costerebbe ricordare un dettaglio del capitolo uno quando arrivi al capitolo venti. La soluzione si chiama Long Short-Term Memory (LSTM): una cella ricorrente con "gate" che decidono esplicitamente quale informazione mantenere, quale dimenticare e quale lasciare passare verso l'output. Pensa a questi gate come a un archivio con tre impiegati: uno decide quali nuovi documenti mantenere, un altro decide quali documenti vecchi buttare, e un terzo decide quali documenti mostrare proprio ora.

## Costruisci un LSTM con Keras

Andrai ad applicare un LSTM a un problema che già conosci — MNIST — ma con una piega: invece di dargli l'immagine completa tutta insieme, la darai riga per riga, come se fosse una sequenza di ventotto passi di ventotto valori ciascuno. Così eserciti il pattern di una sequenza senza avere bisogno di un dataset di testo nuovo.

```python
import tensorflow as tf

(x_entreno, y_entreno), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_entreno, x_test = x_entreno / 255.0, x_test / 255.0

modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(10, activation="softmax"),
])

modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

modelo.fit(x_entreno, y_entreno, epochs=3, validation_split=0.1)
modelo.evaluate(x_test, y_test)
```

Ogni immagine di 28x28 viene interpretata qui come una sequenza di 28 passi temporali, ognuno con un vettore di 28 valori: l'LSTM percorre riga per riga, aggiornando il suo stato interno, finché l'ultima riga produce il riassunto finale che alimenta lo strato di output.

## GRU: un'alternativa più leggera

Gated Recurrent Unit (GRU) è una variante dell'LSTM con meno gate interni — e quindi meno parametri da addestrare —, che nella pratica di solito ha prestazioni simili in molti compiti. Cambiare da uno all'altro è semplice come sostituire lo strato:

```python
modelo_gru = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(10, activation="softmax"),
])
```

## Riepilogo dei punti chiave

- Una RNN elabora sequenze mantenendo uno stato nascosto che riassume quello che ha visto fino ad ora.
- Una RNN semplice tende a dimenticare informazioni lontane nella sequenza; LSTM e GRU risolvono quel problema con meccanismi di gate.
- In Keras, passare da un'architettura densa a una ricorrente è semplice come sostituire lo strato iniziale `Dense`/`Flatten` con `LSTM` o `GRU`.
- Trattare un'immagine come sequenza di righe è un buon esercizio per capire il concetto di RNN prima di applicarlo a dati di testo reali o a serie temporali.

## Mettilo in pratica

**Sostituisci** lo strato `LSTM` del primo esempio con uno strato `GRU`, **addestra** entrambi i modelli con lo stesso numero di epoch, e **confronta** la loro accuratezza finale e il tempo di addestramento. Quale sceglieresti tu per un progetto con risorse limitate?

---

**Materiale sorgente di questo capitolo**: lezione 3 temi 1 e 2 ("Recurrent Neural Networks", "RNN Example in TensorFlow").

[⬅ Capitolo precedente](02_sentiment_analysis_IT.md) · [Torna all'indice](00_indice_IT.md) · [Capitolo successivo ➡](04_cnn_fundamentos_IT.md)
