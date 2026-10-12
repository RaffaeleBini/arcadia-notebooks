[← Precedente: [Cos'è il Deep Learning?](01_que_es_deep_learning_IT.md) · [Indice](00_indice_IT.md) · Prossimo: [Prepara i tuoi dati →](03_prepara_tus_datos_IT.md)]

# 2. Il tuo primo modello con Keras: classificazione di cifre (MNIST)

## Perché è importante per te

Leggere solo teoria sulle reti neurali ti porta solo fino a un certo punto. Il modo reale per capire il Deep Learning è costruire un modello, vederlo addestrare e verificare se funziona. In questo capitolo costruisci il tuo primo classificatore di immagini: un modello che riconosce cifre scritte a mano da zero a nove. È l'"hello world" del Deep Learning, e per una buona ragione: è semplice da montare, si addestra in secondi e ti permette di toccare ogni pezzo del processo completo, dai dati grezzi alla previsione finale.

## Il dataset MNIST

MNIST è una collezione di 70.000 immagini in scala di grigi di cifre scritte a mano (da zero a nove), ognuna di 28x28 pixel. È incluso in Keras, quindi non devi scaricare nulla da solo.

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Carica i dati già divisi in addestramento e test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  # (60000, 28, 28): 60.000 immagini di 28x28 pixel
print(y_train.shape)  # (60000,): l'etichetta (0-9) di ogni immagine
```

## Normalizza i tuoi dati prima di addestrare

Ogni pixel di un'immagine MNIST è un numero intero tra zero e 255 (l'intensità del grigio). Le reti neurali si addestrano meglio quando i valori di ingresso sono in un intervallo piccolo, quindi li normalizzi dividendo per 255 per lasciarli tra zero e uno:

```python
# Normalizza: da [0, 255] a [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0
```

Questo passaggio sembra banale, ma dimenticarlo è uno degli errori più comuni quando si inizia: senza normalizzare, l'addestramento è più lento e instabile.

## Costruisci il modello

Userai l'API `Sequential` di Keras, che ti permette di impilare strati uno dopo l'altro:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
    # Appiattisce ogni immagine 28x28 in un vettore di 784 valori
    Flatten(input_shape=(28, 28)),
    # Primo strato nascosto: 128 neuroni con attivazione ReLU
    Dense(128, activation='relu'),
    # Secondo strato nascosto: altri 128 neuroni
    Dense(128, activation='relu'),
    # Strato di uscita: 10 neuroni (uno per cifra), softmax fornisce una probabilità per classe
    Dense(10, activation='softmax'),
])
```

`Flatten` non impara nulla: semplicemente riorganizza ogni immagine da una matrice 28x28 a una riga di 784 numeri, perché gli strati `Dense` si aspettano un vettore, non una matrice.

## Compila il modello: scegli come impara

Compilare un modello significa decidere tre cose: come misura i suoi errori (la funzione di perdita), come regola i pesi per ridurre quell'errore (l'ottimizzatore) e quale metrica vuoi monitorare durante l'addestramento:

```python
model.compile(
    optimizer='adam',                          # Adam: regola i pesi in modo efficiente
    loss='sparse_categorical_crossentropy',    # ideale quando le etichette sono numeri (0-9)
    metrics=['accuracy'],                      # segue la percentuale di successi
)
```

## Addestra e valuta

```python
# Addestra per 3 passaggi completi sui dati di addestramento (epochs)
model.fit(x_train, y_train, epochs=3)

# Valuta le prestazioni su dati che il modello non ha mai visto
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Perdita su test: {loss:.4f}")
print(f"Precisione su test: {accuracy:.4f}")
```

Con questa architettura così semplice, dovresti già vedere una precisione (accuracy) oltre il 96% nel set di test, in soli tre epoch (passaggi completi di addestramento sui dati).

## Salva, carica e predici

Un modello addestrato che vive solo in memoria del tuo notebook non ti serve a molto. Salvalo per riutilizzarlo dopo:

```python
# Salva il modello completo (architettura + pesi addestrati)
model.save('mi_primo_modello.keras')

# Carica il modello in un'altra sessione, senza dover riaddestrare
from tensorflow.keras.models import load_model
modello_caricato = load_model('mi_primo_modello.keras')

# Predici su un'immagine di test
import numpy as np

previsioni = modello_caricato.predict(x_test)
cifra_prevista = np.argmax(previsioni[0])  # la classe con probabilità più alta
print(f"Il modello prevede: {cifra_prevista}")
print(f"L'etichetta reale era: {y_test[0]}")
```

`predict` ti restituisce, per ogni immagine, un vettore con 10 probabilità (una per cifra). `argmax` ti dice quale di quelle 10 posizioni ha il valore più alto, cioè quale è la previsione finale del modello.

## Oltre l'accuracy: come valutare davvero un modello

L'accuracy (percentuale di successi) è la metrica più intuitiva, ma può ingannarti, soprattutto se le tue classi sono sbilanciate. Queste sono le metriche che incontrerai nel corso:

- **Precisione**: di tutto quello che il modello ha marcato come positivo, quanto era realmente positivo
- **Recall**: di tutto quello che era realmente positivo, quanto ha rilevato il modello
- **F1-score**: la media armonica tra precisione e recall, utile quando hai bisogno di un singolo numero che riassuma entrambe
- **Matrice di confusione**: una tabella che incrocia previsioni contro etichette reali, e ti mostra esattamente dove il modello si sbaglia (per esempio, se confonde il 4 con il 9)
- **Curva ROC e AUC (Area Sotto la Curva)**: misurano quanto bene il modello separa le classi a diverse soglie di decisione

## Riepilogo dei punti chiave

- MNIST è incluso in Keras: non devi scaricare nulla
- Normalizza sempre i tuoi dati di ingresso prima di addestrare
- `Sequential` ti permette di impilare strati; `Flatten` converte un'immagine in un vettore
- Compilare un modello significa fissare ottimizzatore, funzione di perdita e metriche
- `fit` addestra, `evaluate` misura le prestazioni su dati nuovi, `predict` genera previsioni
- L'accuracy non racconta tutta la storia: precisione, recall, F1-score e la matrice di confusione ti danno una visione più completa

## Mettilo in pratica

Riproduci il codice di questo capitolo in un notebook e, quando il modello è già addestrato, cambia il numero di epoch da 3 a 10. Annota come cambiano l'accuracy e la perdita (loss) nel set di test. **Verifica poi la matrice di confusione** con `sklearn.metrics.confusion_matrix(y_test, np.argmax(model.predict(x_test), axis=1))` e localizza quale coppia di cifre il tuo modello confonde di più.

---

[← Precedente: [Cos'è il Deep Learning?](01_que_es_deep_learning_IT.md) · [Indice](00_indice_IT.md) · Prossimo: [Prepara i tuoi dati →](03_prepara_tus_datos_IT.md)]
