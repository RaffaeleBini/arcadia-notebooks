[← Precedente: [Prepara i tuoi dati](03_prepara_tus_datos_IT.md) · [Indice](00_indice_IT.md) · Prossimo: [TensorBoard: visualizza e depura i tuoi modelli →](05_tensorboard_visualiza_IT.md)]

# 4. Reti Neurali Convoluzionali (CNN)

## Perché è importante per te

Il modello del capitolo 2 (MNIST) funzionava con strati `Dense`, dove ogni pixel si connette con ogni neurone dello strato successivo. Questo funziona con immagini piccole e semplici, ma diventa inviabile con foto reali: un'immagine di 50x50 in scala di grigi ha già 2.500 valori di ingresso, e uno strato `Dense` connesso a tutti loro cresce in dimensione molto velocemente, senza sfruttare il fatto che un'immagine ha struttura spaziale (i pixel vicini sono legati l'uno all'altro). Le Reti Neurali Convoluzionali (CNN, Convolutional Neural Network) risolvono questo problema, e sono la base di quasi tutto quello che oggi chiamiamo "visione artificiale".

## La convoluzione, spiegata con un'analogia

Immagina di esaminare una fotografia enorme con una lente d'ingrandimento piccola, spostandola metodicamente da sinistra a destra e da cima a fondo. In ogni posizione, annoti quello che vedi: un bordo, una texture, un cambio di colore. Questo è esattamente quello che fa un filtro convoluzionale (chiamato anche kernel): è una piccola matrice (per esempio, di 3x3 pixel) che scorre su tutta l'immagine, calcolando in ogni posizione un prodotto scalare tra il filtro e la porzione di immagine che copre. Il risultato è una "mappa di feature" (feature map) che evidenzia dove appare il pattern che quel filtro rileva.

La grande differenza rispetto a uno strato `Dense` è che lo stesso filtro viene riutilizzato su tutta l'immagine: se impara a rilevare un bordo verticale, lo rileva ovunque nella foto, senza bisogno di un neurone diverso per ogni posizione.

## Pooling: comprimi senza perdere l'essenziale

Dopo aver applicato diversi filtri, ottieni diversi feature map, ciascuno quasi grande quanto l'immagine originale. Per ridurre quella quantità di informazione senza perdere l'essenziale, applichi il pooling. La variante più comune è il max pooling: divide la mappa in piccoli blocchi (per esempio, di 2x2) e tiene solo il valore più alto di ogni blocco.

Questo raggiunge due cose: riduce la dimensione dei dati (e quindi il costo computazionale) e dà alla rete una certa invarianza a piccoli spostamenti, cioè, se il gatto è un po' più a sinistra in una foto che in un'altra, il modello lo riconosce comunque allo stesso modo.

## L'architettura completa di una CNN

Una CNN tipica incatena questi blocchi diverse volte, e termina con strati densi per la decisione finale:

```
Ingresso (immagine) → [Conv2D → Attivazione → MaxPooling] × N → Flatten → Dense → Uscita
```

I primi strati convoluzionali imparano pattern semplici (bordi, texture). Gli strati più profondi combinano questi pattern semplici in concetti più astratti (un'orecchia, un occhio, una silhouette completa). Solo alla fine, quando hai già caratteristiche di alto livello, ha senso usare uno strato `Dense`.

## Costruisci il tuo classificatore di gatti e cani

Riprendi il dataset che hai preparato nel capitolo precedente:

```python
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Recupera i dati già elaborati
with open("X.pickle", "rb") as file:
    X = pickle.load(file)
with open("y.pickle", "rb") as file:
    y = pickle.load(file)

# Normalizza i pixel da [0, 255] a [0, 1]
X = X / 255.0

model = Sequential([
    # Primo blocco convoluzionale: 64 filtri di 3x3
    Conv2D(64, (3, 3), input_shape=X.shape[1:]),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Secondo blocco convoluzionale: approfondisce l'apprendimento dei pattern
    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Appiattisce le feature map in un vettore
    Flatten(),
    Dense(64),
    Activation('relu'),

    # Strato di uscita: un neurone con sigmoid per classificazione binaria (gatto/cane)
    Dense(1, activation='sigmoid'),
])
```

Nota `input_shape=X.shape[1:]`: dice al primo strato quale forma aspettarsi (50, 50, 1), senza includere il numero di immagini.

## Compila, addestra e valuta

```python
model.compile(
    loss='binary_crossentropy',   # appropriata per classificazione di due classi
    optimizer='adam',
    metrics=['accuracy'],
)

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)
```

`validation_split=0.1` riserva automaticamente il 10% dei tuoi dati come set di validazione: il modello non si addestra con loro, quindi ti danno una misura onesta di come si comporta con dati nuovi, in ogni epoch.

## Riepilogo dei punti chiave

- Uno strato `Conv2D` applica filtri scorrevoli che imparano a rilevare pattern visuali (bordi, texture, forme)
- Il pooling (normalmente max pooling) riduce la dimensione dei dati e fornisce invarianza a piccoli spostamenti
- Una CNN incatena blocchi di convoluzione e pooling, e termina con strati densi per la decisione finale
- I primi strati imparano pattern semplici, gli strati profondi combinano questi pattern in concetti astratti
- Per classificazione binaria, usa un neurone di uscita con attivazione sigmoid e `binary_crossentropy` come funzione di perdita

## Mettilo in pratica

Riprendi il classificatore di gatti e cani di questo capitolo e aggiungi un terzo blocco convoluzionale (`Conv2D` + `Activation` + `MaxPooling2D`) prima dello strato `Flatten`. Addestra il modello con `validation_split=0.1` per 10 epoch e confronta il `val_accuracy` finale con quello che hai ottenuto con solo due blocchi. **Annota la tua conclusione**: ha migliorato aggiungere profondità, o il modello ha iniziato a fare overfitting?

---

[← Precedente: [Prepara i tuoi dati](03_prepara_tus_datos_IT.md) · [Indice](00_indice_IT.md) · Prossimo: [TensorBoard: visualizza e depura i tuoi modelli →](05_tensorboard_visualiza_IT.md)]
