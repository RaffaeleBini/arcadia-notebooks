[⬅ Torna all'indice](00_indice_IT.md)

# Capitolo 8 — CNN 3D e imaging medico: rilevamento del cancro ai polmoni

## Perché è importante

Una tomografia computerizzata, o Computed Tomography (CT), non è una sola foto: è una pila di decine o centinaia di sezioni trasversali del corpo, impilate come le pagine di un libro. Rilevare un nodulo polmonare sospetto significa guardare l'intera pila, non una sola sezione, perché un nodulo può essere visibile in diversi tagli consecutivi e quasi invisibile in uno solo. Questo capitolo ti insegna ad adattare una CNN per lavorare con volumi tridimensionali invece di immagini piatte, con un caso reale: la competizione di Kaggle sul rilevamento del cancro ai polmoni (Data Science Bowl 2017).

## Da 2D a 3D (con un'analogia)

Pensa alla differenza tra guardare una singola fotografia di una torta e tenere l'intera torta tra le mani, girandola per vederne la forma completa da tutti i lati. Un'immagine 2D ti lascia vedere solo una sezione; un volume 3D ti permette di apprezzare come la forma di qualcosa cambia mentre avanzi in profondità. Una CNN 3D fa esattamente questo: invece di far scorrere un filtro piatto (altezza x larghezza) su un'immagine, fa scorrere un filtro con volume (altezza x larghezza x profondità) su tutta la pila di sezioni, catturando così modelli che si estendono nelle tre dimensioni.

## La sfida dei dati medici reali

I dati di questo progetto arrivano in formato DICOM —lo standard per l'imaging medico— e presentano sfide che non hai visto nei capitoli precedenti:

- **Volumi di dimensioni diverse**: ogni paziente ha un numero diverso di sezioni, quindi devi ridimensionare anche nella dimensione di profondità, non solo in altezza e larghezza.
- **Volume di dati**: il dataset completo della competizione pesa decine di gigabyte, troppo per elaborarlo tutto in una macchina modesta senza ottimizzare il pipeline.
- **Rumore e variabilità**: scanner diversi, protocolli di acquisizione diversi, e pazienti con anatomie diverse.

Per questo, per praticare il pipeline completo senza dipendere da un download di gigabyte né da credenziali di Kaggle, nel notebook di questa lezione lavorerai con volumi sintetici generati con le stesse dimensioni che useresti con dati reali. La logica —ridimensionare, normalizzare, addestrare una CNN 3D— è identica; l'unica cosa che cambia è l'origine dei dati.

## Ridimensiona un volume 3D

```python
import tensorflow as tf

def redimensionar_volumen(volumen, tamano_destino=(64, 64, 32)):
    """Ridimensiona altezza, larghezza e profondità di un volume 3D."""
    volumen = tf.expand_dims(volumen, axis=-1)  # aggiunge il canale (bianco e nero)
    volumen = tf.image.resize(volumen, tamano_destino[:2])  # altezza e larghezza
    volumen = tf.transpose(volumen, perm=[2, 0, 1, 3])
    volumen = tf.image.resize(volumen, [tamano_destino[2], tamano_destino[0]])
    volumen = tf.transpose(volumen, perm=[1, 2, 0, 3])
    return volumen
```

Ridimensionare la profondità richiede un piccolo trucco: poiché `tf.image.resize` funziona solo sulle prime due dimensioni spaziali, trasponiamo il volume per mettere la profondità dove prima c'era l'altezza, ridimensioniamo, e traspo niamo di nuovo.

## Costruisci una CNN 3D

```python
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 32, 1)),
    tf.keras.layers.Conv3D(16, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling3D(pool_size=2),
    tf.keras.layers.Conv3D(32, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling3D(pool_size=2),
    tf.keras.layers.GlobalAveragePooling3D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

modelo.compile(optimizer="adam", loss="binary_crossentropy",
               metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
```

`Conv3D` e `MaxPooling3D` sono le versioni tridimensionali dei livelli che già conosci dal capitolo 4: l'unica differenza è che il filtro si muove anche in profondità. Aggiungiamo la metrica AUC (Area Under the Curve, area sotto la curva) perché è la metrica che usa la vera competizione di Kaggle per valutare gli invii: misura quanto bene il modello distingue tra casi positivi e negativi, al di là di una semplice percentuale di correttezza.

## Riepilogo dei punti chiave

- Una CT è un volume 3D formato da molte sezioni 2D impilate; una CNN 3D elabora quel volume completo invece di una sola sezione.
- I dati medici reali hanno bisogno di ridimensionare anche la dimensione di profondità, non solo altezza e larghezza.
- `Conv3D`, `MaxPooling3D` e `GlobalAveragePooling3D` sono le versioni tridimensionali dei livelli che già conoscevi.
- AUC è la metrica abituale nei problemi medici di rilevamento, perché valuta la capacità di distinguere positivi da negativi con più sfumatura della semplice accuratezza.

## Mettilo in pratica

**Cambia** la dimensione destino del volume da `(64, 64, 32)` a `(32, 32, 16)` nella funzione `redimensionar_volumen`, **adatta** il livello `Input` del modello in modo che corrisponda alla nuova dimensione, e **confronta** quanto si riduce il tempo di addestramento per epoca rispetto alla dimensione originale.

---

**Materiale sorgente di questo capitolo**: lezione 8 argomenti 1 a 6 ("Introduction", "Reading Files", "Visualizing", "Resizing Data", "Preprocessing Data", "Running the Network"); `first-pass-through-data-w-3d-convnet.ipynb`.

[⬅ Capitolo precedente](07_dogs_vs_cats_IT.md) · [Torna all'indice](00_indice_IT.md) · [Prossimo capitolo ➡](09_chatbot_IT.md)
