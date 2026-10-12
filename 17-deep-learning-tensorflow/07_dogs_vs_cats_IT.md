[⬅ Torna all'indice](00_indice_IT.md)

# Capitolo 7 — Visione artificiale reale: cani vs. gatti

## Perché è importante

Fino a ora hai lavorato con MNIST: immagini piccole, in bianco e nero, già perfettamente ritagliate e normalizzate. Il mondo reale non è così. Le foto che caricano le persone hanno dimensioni diverse, colori, sfondi disordinati e soggetti mal centrati. In questo capitolo fai il salto a un dataset di fotografie reali —cani contro gatti, una delle sfide classiche di Kaggle— e apprendi il processo completo: preparare dati imperfetti, addestrare una CNN, e valutare i successi e gli errori.

## La sfida dei dati reali (con un'analogia)

Immagina di chiedere a cento persone diverse di inviarti una foto del loro animale domestico: riceverai foto verticali e orizzontali, di giorno e di notte, con l'animale al centro o in un angolo. Prima di poterle confrontare in modo equo, devi metterle tutte nello stesso "formato di scheda": stessa dimensione, stesso tipo di colore. Questo processo si chiama preprocessing, e nella visione artificiale è importante quanto l'architettura della rete stessa.

## Prepara il dataset

Il dataset "Dogs vs. Cats" di Kaggle organizza le immagini in cartelle secondo la loro classe, e ogni immagine ha una dimensione diversa. `image_dataset_from_directory` di Keras si occupa di leggere quella struttura di cartelle, ridimensionare ogni immagine alla stessa dimensione, e organizzarle in lotti pronti per l'addestramento.

```python
import tensorflow as tf

TAMANO_IMAGEN = (128, 128)
TAMANO_LOTE = 32

dataset_entreno = tf.keras.utils.image_dataset_from_directory(
    "dogs_vs_cats/train",
    image_size=TAMANO_IMAGEN,
    batch_size=TAMANO_LOTE,
    label_mode="binary",
)

# Normalizziamo i pixel a un intervallo da 0 a 1, come nei capitoli precedenti
normalizacion = tf.keras.layers.Rescaling(1.0 / 255)
dataset_entreno = dataset_entreno.map(lambda x, y: (normalizacion(x), y))
```

## Aumento dei dati: moltiplica i tuoi esempi senza nuove foto

Con poche migliaia di foto, una CNN grande può memorizzare il dataset di addestramento invece di imparare modelli generali —quello che si chiama sovraddattamento—. L'aumento dei dati ("data augmentation") genera variazioni casuali di ogni foto —ruotata, capovolta, con zoom leggermente diverso— in modo che la rete veda più varietà senza aver bisogno di nuove foto.

```python
aumento_datos = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

## Costruisci e addestra la CNN

```python
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    aumento_datos,
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
modelo.fit(dataset_entreno, epochs=10, validation_data=dataset_validacion)
```

Nota che l'output è di nuovo un singolo neurone con attivazione sigmoid, come nel classificatore di sentimenti: cane contro gatto è, ancora una volta, un problema di due classi.

## Interpreta i risultati

Un'accuratezza dell'85% suona bene, ma non ti dice che tipo di errori commette il modello. Visualizzare alcune previsioni —immagini reali insieme all'etichetta prevista— ti aiuta a rilevare modelli negli errori: magari confonde gatti a pelo lungo con cani piccoli, o fallisce più spesso con foto scure.

## Riepilogo dei punti chiave

- I dati reali arrivano con dimensioni e formati incoerenti; `image_dataset_from_directory` automatizza gran parte di quel preprocessing.
- L'aumento dei dati genera variazioni delle tue immagini per ridurre il sovraddattamento senza aver bisogno di più foto.
- L'architettura della CNN è la stessa idea del capitolo 4, solo con più blocchi di convoluzione per immagini più grandi e complesse.
- Visualizzare previsioni concrete rivela modelli di errore che una sola metrica di accuratezza non mostra.

## Mettilo in pratica

**Scegli** dieci immagini dall'insieme di validazione, **mostrali** con `matplotlib` insieme all'etichetta reale e a quella prevista dal tuo modello, e **identifica** almeno un caso in cui il modello sbagli. **Scrivi** un'ipotesi su perché credi che abbia fallito in quel caso specifico.

---

**Materiale sorgente di questo capitolo**: lezione 7 argomenti 1 a 4 ("Introduction and Preprocessing", "Building the Network", "Training the Network", "Using the Network").

[⬅ Capitolo precedente](06_reinforcement_learning_IT.md) · [Torna all'indice](00_indice_IT.md) · [Prossimo capitolo ➡](08_cnn_3d_imagen_medica_IT.md)
