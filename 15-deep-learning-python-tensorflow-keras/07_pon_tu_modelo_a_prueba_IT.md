[← Precedente: [Ottimizza i tuoi modelli](06_optimiza_hiperparametros_IT.md) · [Indice](00_indice_IT.md) · Successivo: [Reti Neurali Ricorrenti (RNN) →](08_redes_recurrentes_rnn_IT.md)]

# 7. Metti il tuo modello alla prova: previsioni su dati nuovi

## Perché ti importa

Un modello che ha dimostrato il suo valore solo sul set di validazione del suo dataset non ti dice la verità completa. La domanda che importa davvero è: funziona con dati che non ha mai visto, provenienti da un contesto diverso da quello dell'allenamento? Una foto che scatti tu stesso con il tuo cellulare, non un'immagine già preparata dal dataset. Questa è la prova del fuoco di qualsiasi modello, e chiude il blocco di CNN e TensorBoard di questo corso.

## Perché l'accuracy di validazione non basta

L'accuracy che vedi durante l'allenamento viene calcolato su immagini che provengono dallo stesso dataset, elaborate con la stessa pipeline, probabilmente catturate in condizioni simili. Una nuova immagine esterna (out-of-sample) potrebbe avere un'illuminazione diversa, un inquadratura diversa, uno sfondo diverso. Se il tuo modello ha imparato solo a riconoscere le particolarità del dataset di allenamento invece del concetto reale ("gatto" o "cane"), fallirà miseramente con immagini nuove. Questo è, essenzialmente, un modo più esigente di rilevare l'overfitting.

## Prepara una nuova immagine esattamente come hai preparato il dataset

La regola d'oro è questa: qualsiasi nuova immagine deve passare esattamente per lo stesso preprocessing che hai applicato ai tuoi dati di allenamento. Se hai allenato con immagini in scala di grigi di 50x50 pixel, la tua nuova immagine deve arrivare al modello in scala di grigi e di 50x50 pixel:

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 50
categorias = ["Dog", "Cat"]

def preparar_imagen(ruta_archivo):
    array_imagen = cv2.imread(ruta_archivo, cv2.IMREAD_GRAYSCALE)
    array_redimensionado = cv2.resize(array_imagen, (IMG_SIZE, IMG_SIZE))
    # Misma normalización que usaste al entrenar, y la misma forma: (1, 50, 50, 1)
    return array_redimensionado.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

modelo = load_model('mi_clasificador_gatos_perros.keras')

imagen_preparada = preparar_imagen("mi_foto_de_prueba.jpg")
prediccion = modelo.predict(imagen_preparada)

# Con salida sigmoid (una neurona), redondeas: menor de 0.5 es la clase 0, mayor o igual es la clase 1
clase_predicha = categorias[int(round(prediccion[0][0]))]
print(f"El modelo dice: {clase_predicha}")
```

## Perché la diversità del dataset di allenamento conta così tanto

Se il tuo dataset di allenamento contiene solo foto di gatti bianchi di razza persiana, il tuo modello imparerà a riconoscere quello, non il concetto generale di "gatto". Questo problema non è esclusivo delle foto di animali domestici: è la stessa ragione per cui un sistema di riconoscimento facciale allenato su un gruppo demografico limitato fallisce quando applicato a persone al di fuori di quel gruppo. Quanto più diverso e rappresentativo è il tuo dataset di allenamento, meglio il tuo modello generalizza a situazioni nuove.

## Riepilogo dei punti chiave

- L'accuracy di validazione non garantisce che il tuo modello funzioni con dati veramente nuovi ed esterni
- Qualsiasi dato nuovo deve passare esattamente per lo stesso preprocessing (dimensione, scala di colore, normalizzazione) dei tuoi dati di allenamento
- Testare con dati out-of-sample è un modo esigente di rilevare l'overfitting che il set di validazione interno non sempre rivela
- Un dataset di allenamento poco diverso limita la capacità di generalizzazione del modello, con conseguenze che vanno da errori banali a pregiudizi seri in applicazioni sensibili

## Mettilo in pratica

Scatta tre foto con il tuo cellulare (o scarica tre immagini di gatti o cani che non facciano parte del dataset originale) e passale attraverso la funzione `preparar_imagen` di questo capitolo insieme al tuo modello già allenato del capitolo 4. **Annota il risultato di ogni previsione** e, se il modello sbaglia in qualcuna, esamina l'immagine: ha un'illuminazione, un angolo o uno sfondo molto diverso da quelli delle foto del dataset di allenamento?

---

[← Precedente: [Ottimizza i tuoi modelli](06_optimiza_hiperparametros_IT.md) · [Indice](00_indice_IT.md) · Successivo: [Reti Neurali Ricorrenti (RNN) →](08_redes_recurrentes_rnn_IT.md)]
