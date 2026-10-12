[← Precedente: [Normalizza e costruisci sequenze temporali](10_caso_cripto_normaliza_secuencias_IT.md) · [Indice](00_indice_IT.md) · Successivo: [Casi d'uso reali del Deep Learning →](12_casos_de_uso_reales_IT.md)]

# 11. Bilancia i tuoi dati e allena il tuo modello LSTM finale

## Perché ti importa

Hai sequenze temporali normalizzate e separate correttamente in allenamento e validazione. Manca un ultimo aggiustamento prima di allenare: verificare se le tue classi (sale/scende) sono bilanciate, e se non lo sono, correggerlo. Dopo costruisci l'architettura LSTM finale, la alleni con i callback appropriati, e chiudi il progetto che hai iniziato nel capitolo 9.

## Bilancia le tue classi manualmente

Keras offre il parametro `class_weight` in `model.fit` per compensare classi sbilanciate, ma in pratica non sempre corregge bene il problema, soprattutto con dati sequenziali. Un bilanciamento manuale, più semplice e più prevedibile, funziona meglio: separi le sequenze per classe, tagli entrambi i gruppi alla lunghezza del più piccolo, e li mescoli di nuovo insieme.

```python
import random

def balancea_secuencias(datos_secuenciales):
    compras, ventas = [], []

    for secuencia, target in datos_secuenciales:
        if target == 0:
            ventas.append([secuencia, target])
        else:
            compras.append([secuencia, target])

    random.shuffle(compras)
    random.shuffle(ventas)

    limite = min(len(compras), len(ventas))
    compras = compras[:limite]
    ventas = ventas[:limite]

    datos_balanceados = compras + ventas
    random.shuffle(datos_balanceados)
    return datos_balanceados

secuencias_entrenamiento = balancea_secuencias(secuencias_entrenamiento)
secuencias_validacion = balancea_secuencias(secuencias_validacion)

print(f"Entrenamiento balanceato: {len(secuencias_entrenamiento)} sequenze")
print(f"Validazione balanceata: {len(secuencias_validacion)} sequenze")
```

**Mescola sempre l'insieme già combinato** (`random.shuffle(datos_balanceados)`), non solo ogni lista separatamente: se non lo fai, tutti gli "acquisti" rimangono insieme seguiti da tutte le "vendite", e il modello può imparare un pattern falso basato sull'ordine invece che sul contenuto reale di ogni sequenza.

## Separa in array di input (X) e obiettivo (y)

```python
import numpy as np

def separa_features_y_target(datos_secuenciales):
    X, y = [], []
    for secuencia, target in datos_secuenciales:
        X.append(secuencia)
        y.append(target)
    return np.array(X), np.array(y)

train_x, train_y = separa_features_y_target(secuencias_entrenamiento)
validation_x, validation_y = separa_features_y_target(secuencias_validacion)
```

## Costruisci l'architettura LSTM finale

Impili tre livelli LSTM, ciascuno seguito da `Dropout` (per evitare l'overfitting) e `BatchNormalization` (per stabilizzare l'allenamento):

```python
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

EPOCHS = 10
BATCH_SIZE = 64
NOMBRE_MODELO = f"LSTM-cripto-SEQ{SEQ_LEN}-PRED{PERIODO_FUTURO}-{int(time.time())}"

model = Sequential([
    LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),

    LSTM(128, return_sequences=True),
    Dropout(0.1),
    BatchNormalization(),

    LSTM(128),
    Dropout(0.2),
    BatchNormalization(),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(2, activation='softmax'),
])

optimizador = Adam(learning_rate=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizador,
    metrics=['accuracy'],
)
```

> **Nota sull'architettura originale**: il materiale del corso usava il livello `CuDNNLSTM`, una variante di LSTM accelerata da GPU che esisteva nelle versioni vecchie di TensorFlow/Keras. Quel livello è stato eliminato in TensorFlow 2.x: oggi usi `LSTM` direttamente, e Keras sfrutta l'accelerazione GPU automaticamente quando disponibile, senza che tu debba scegliere un livello diverso.

## Configura i callback di allenamento

```python
tensorboard_callback = TensorBoard(log_dir=f'logs/{NOMBRE_MODELO}')

checkpoint_callback = ModelCheckpoint(
    filepath="modelos/epoch-{epoch:02d}-val_acc-{val_accuracy:.3f}.keras",
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max',
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,  # ferma l'allenamento se val_loss non migliora in 3 epoch consecutivi
)
```

> **Nota su un bug del materiale originale**: nel notebook sorgente, la chiamata a `ModelCheckpoint` passava `monitor`, `verbose`, `save_best_only` e `mode` come argomenti di `str.format()` invece che come argomenti dello stesso `ModelCheckpoint`, così queste opzioni venivano ignorate silenziosamente e il checkpoint finiva per salvare tutti i modelli, non solo il migliore. Qui gli argomenti vanno direttamente a `ModelCheckpoint(...)`, dove hanno effetto. Inoltre, il nome della metrica di validazione è cambiato da `val_acc` (versioni vecchie di Keras) a `val_accuracy` (versione attuale).

## Allena e valuta il modello finale

```python
model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback],
)

loss, accuracy = model.evaluate(validation_x, validation_y)
print(f"Pérdida de validación: {loss:.4f}")
print(f"Precisión de validación: {accuracy:.4f}")
```

## Interpreta i tuoi risultati con onestà

Non aspettarti una precisione spettacolare: prevedere i movimenti dei prezzi a breve termine è, per design, un problema molto rumoroso. Il prezzo di una criptovaluta non dipende solo dal suo storico, dipende anche da notizie, sentimento del mercato e attività sui social media, informazioni che questo modello non vede in nessun momento. Diverse criptovalute si comportano diversamente: alcune più stabili, altre più volatili, e questo si riflette nella difficoltà di prevederle.

**Tratta questo modello come uno strumento educativo, non come un consiglio di investimento.** Se un giorno volessi avvicinarti a un caso d'uso reale di trading, avresti bisogno, come minimo, di incorporare più fonti di dati (sentimento del mercato, notizie, volume degli ordini), convalidare il modello con molto più rigore statistico, e capire che nessun modello elimina il rischio finanziario.

## Riepilogo dei punti chiave

- Bilancia manualmente le sequenze per classe quando `class_weight` non sia sufficiente, e mescola sempre l'insieme combinato, non solo ogni classe separatamente
- L'architettura finale combina LSTM impilate con `Dropout` e `BatchNormalization` per regolarizzare e stabilizzare l'allenamento
- `CuDNNLSTM` non esiste più in TensorFlow 2.x: usa `LSTM`, che sfrutta la GPU automaticamente
- Verifica sempre che gli argomenti dei tuoi callback (come `ModelCheckpoint`) vengano passati al callback e non si perdano in un `str.format()` scritto male
- `EarlyStopping` evita di continuare ad allenare quando non c'è già più miglioramento reale
- Interpreta i risultati di un modello finanziario con onestà: è uno strumento educativo, non una garanzia di redditività

## Mettilo in pratica

Allena il modello completo di questo capitolo con i dati che hai preparato nei capitoli 9 e 10. Quando finisce, apri TensorBoard e confronta la curva di `val_accuracy` di questo modello bilanciato con quella che otterresti se commenti la chiamata a `balancea_secuencias` e alleni con i dati sbilanciati originali. **Annota la differenza** e spiega, con tue proprie parole, perché il bilanciamento cambia il risultato.

---

[← Precedente: [Normalizza e costruisci sequenze temporali](10_caso_cripto_normaliza_secuencias_IT.md) · [Indice](00_indice_IT.md) · Successivo: [Casi d'uso reali del Deep Learning →](12_casos_de_uso_reales_IT.md)]
