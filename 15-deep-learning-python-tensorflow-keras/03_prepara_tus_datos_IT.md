[← Precedente: [Il tuo primo modello con Keras](02_primer_modelo_mnist_IT.md) · [Indice](00_indice_IT.md) · Prossimo: [Reti Neurali Convoluzionali →](04_redes_convolucionales_cnn_IT.md)]

# 3. Prepara i tuoi dati: immagini di gatti e cani

## Perché è importante per te

MNIST arriva perfettamente confezionato: immagini già della stessa dimensione, già in scala di grigi, già con le loro etichette pronte. Nel mondo reale quasi mai hai quella fortuna. Le tue foto arriveranno in dimensioni diverse, a colori, mescolate in cartelle, con nomi di file incoerenti. Prima che una rete neurale possa imparare dai tuoi dati, devi convertirli in un formato uniforme. Questo è il lavoro di questo capitolo, ed è un'abilità che userai in ogni progetto reale che farai.

## Il dataset: gatti contro cani

Lavorerai con un dataset reale di foto di gatti e cani (una versione ridotta del dataset originale di Microsoft, con migliaia di immagini per classe), organizzato in due cartelle: `Cat/` e `Dog/`. Il tuo obiettivo è convertire quelle cartelle di immagini in qualcosa che una rete neurale possa elaborare: array di numeri, tutti della stessa dimensione.

## Carica le immagini con OpenCV

OpenCV (`cv2`) è la libreria standard di Python per elaborare immagini. Ti permette di leggere un file di immagine e convertirlo direttamente in un array di NumPy:

```python
import os
import cv2
import matplotlib.pyplot as plt

directory_dati = "PetImages"
categorie = ["Dog", "Cat"]

for categoria in categorie:
    percorso = os.path.join(directory_dati, categoria)
    for nome_file in os.listdir(percorso)[:1]:  # solo la prima immagine, per ispezionare
        # IMREAD_GRAYSCALE converte l'immagine a scala di grigi al momento della lettura
        array_immagine = cv2.imread(os.path.join(percorso, nome_file), cv2.IMREAD_GRAYSCALE)
        plt.imshow(array_immagine, cmap='gray')
        plt.show()
        print(array_immagine.shape)  # ogni immagine ha dimensioni diverse
```

Noterai due cose: prima, ogni immagine ha dimensioni diverse; seconda, lavori in scala di grigi. Convertire a scala di grigi riduce l'informazione da tre canali di colore a uno solo, accelerando molto l'addestramento senza compromettere troppo la precisione in compiti come questo.

## Ridimensiona tutte le immagini alla stessa dimensione

Una rete neurale richiede che tutti gli ingressi abbiano esattamente la stessa forma. Ridimensiona ogni immagine a 50x50 pixel:

```python
IMG_SIZE = 50

array_ridimensionato = cv2.resize(array_immagine, (IMG_SIZE, IMG_SIZE))
plt.imshow(array_ridimensionato, cmap='gray')
plt.show()
```

Cinquanta per cinquanta è una dimensione volutamente piccola: mantiene il dataset leggero e l'addestramento veloce, al costo di perdere alcuni dettagli. È una decisione di compromesso comune all'inizio di un progetto: prototipa velocemente con immagini piccole, e aumenta la risoluzione solo se hai bisogno di più precisione.

## Costruisci il dataset completo

Ora ripeti il processo per ogni immagine di ogni categoria, e salvi la coppia (immagine, etichetta):

```python
dati_addestramento = []

def creare_dati_addestramento():
    for categoria in categorie:
        percorso = os.path.join(directory_dati, categoria)
        classe_num = categorie.index(categoria)  # Dog=0, Cat=1

        for nome_file in os.listdir(percorso):
            try:
                array_immagine = cv2.imread(os.path.join(percorso, nome_file), cv2.IMREAD_GRAYSCALE)
                array_ridimensionato = cv2.resize(array_immagine, (IMG_SIZE, IMG_SIZE))
                dati_addestramento.append([array_ridimensionato, classe_num])
            except Exception:
                # Alcuni file del dataset sono corrotti: li ignori e continui
                pass

creare_dati_addestramento()
print(len(dati_addestramento))
```

## Perché devi bilanciare e mescolare i tuoi dati

Se il tuo dataset ha 6.000 foto di cani e solo 500 di gatti, il tuo modello potrebbe "barare": impara a predire sempre "cane" e comunque indovina il 92,3% delle volte, senza aver imparato nulla di utile sui gatti. Per questo verifica sempre il bilancio delle classi prima di addestrare, e taglia la classe maggioritaria se necessario per equilibrare entrambe.

Inoltre, se non mescoli i dati, il modello vedrà prima 6.000 cani di fila e poi tutti i gatti: potrebbe imparare pattern legati all'ordine invece che al contenuto reale dell'immagine.

```python
import random

random.shuffle(dati_addestramento)

for campione in dati_addestramento[:10]:
    print(campione[1])  # dovresti vedere un mix di zeri e uni, non un blocco di ognuno
```

## Separa feature e etichette, e dai forma all'array finale

```python
import numpy as np

X = []  # feature: le immagini
y = []  # etichette: 0 (cane) o 1 (gatto)

for feature, etichetta in dati_addestramento:
    X.append(feature)
    y.append(etichetta)

# Keras si aspetta 4 dimensioni: (num_immagini, altezza, larghezza, canali)
# L'1 finale indica "un canale", perché lavori in scala di grigi
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)
```

## Salva il tuo lavoro con pickle

Elaborare migliaia di immagini impiega tempo. Non ripetere questo lavoro ogni volta che apri il notebook: salva il risultato con `pickle`, il modulo standard di Python per serializzare oggetti:

```python
import pickle

with open("X.pickle", "wb") as file_output:
    pickle.dump(X, file_output)

with open("y.pickle", "wb") as file_output:
    pickle.dump(y, file_output)

# La prossima volta, invece di rielaborare le immagini, semplicemente carica:
with open("X.pickle", "rb") as file_input:
    X = pickle.load(file_input)
```

## Riepilogo dei punti chiave

- I dati reali arrivano senza formato uniforme: devi standardizzarli prima di addestrare
- `cv2.imread` con `IMREAD_GRAYSCALE` carica un'immagine già convertita a scala di grigi
- `cv2.resize` forza tutte le immagini a condividere la stessa dimensione, requisito indispensabile per una rete neurale
- Verifica sempre il bilancio delle classi e mescola (shuffle) i tuoi dati prima di addestrare
- Keras si aspetta array di immagini con forma `(quantità, altezza, larghezza, canali)`
- `pickle` ti permette di salvare e ricaricare dati già elaborati senza ripetere il lavoro

## Mettilo in pratica

Scarica una versione piccola del dataset "Cats vs Dogs" (per esempio, con `tf.keras.utils.get_file` puntando al ZIP pubblico che Google usa nei suoi tutorial di TensorFlow) e riproduci la pipeline completa di questo capitolo: carica le immagini, ridimensionale a 50x50, verifica il bilancio delle classi, mescola il dataset e salvalo con `pickle`. **Verifica alla fine** che `X.shape` sia `(num_immagini, 50, 50, 1)` prima di passare al prossimo capitolo.

---

[← Precedente: [Il tuo primo modello con Keras](02_primer_modelo_mnist_IT.md) · [Indice](00_indice_IT.md) · Prossimo: [Reti Neurali Convoluzionali →](04_redes_convolucionales_cnn_IT.md)]
