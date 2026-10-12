[⬅ Torna all'indice](00_indice_IT.md)

# Capitolo 5 — Semplifica il tuo codice: TFLearn e Keras

## Perché ti importa questo

Nel capitolo 1 hai costruito una rete manualmente con `tf.Variable` e `GradientTape` per capire cosa succede dentro. Ma nel lavoro quotidiano, scrivere ogni strato da zero è lento e soggetto a errori: dimenticare un bias, usare male una funzione di attivazione, o sbagliare la forma di un tensore sono errori abituali e difficili da individuare. Per questo esistono le Application Programming Interface (API) di alto livello: strati di astrazione che ti lasciano definire architetture complete in poche righe, con le buone pratiche già incorporate.

## Due modi di semplificare: Keras e TFLearn

Già porti avanti diversi capitoli usando Keras, l'API di alto livello ufficiale e integrata in TensorFlow. Storicamente c'è stata un'altra libreria con lo stesso scopo, TFLearn — costruita anche su TensorFlow —, che era molto popolare in tutorial e corsi prima che Keras si integrasse ufficialmente. Oggi Keras è l'opzione consigliata e attivamente mantenuta, ma merita la pena che tu capisca l'idea generale: qualsiasi API di alto livello persegue lo stesso obiettivo, lasciarti scrivere meno codice ripetitivo e ridurre la superficie di errore.

Confronta questi due modi di esprimere la stessa rete:

```python
import tensorflow as tf

# Con Keras: strati incatenati in un elenco
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax"),
])
modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
```

Rispetto all'alternativa di basso livello del capitolo 1, con matrici di pesi e bias definite e aggiornate manualmente: la differenza nel numero di righe di codice — e di opportunità per sbagliare — è enorme.

## Callbacks: automatizza le buone pratiche

Un vantaggio pratico di lavorare con Keras è il suo sistema di "callbacks": funzioni che si eseguono automaticamente durante l'addestramento per risolvere compiti abituali, come arrestare l'addestramento se il modello smette di migliorare (`EarlyStopping`) o salvare automaticamente il miglior modello visto fino ad ora (`ModelCheckpoint`).

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("mejor_modelo.keras", save_best_only=True),
]

modelo.fit(x_entreno, y_entreno, epochs=20, validation_split=0.1, callbacks=callbacks)
```

Senza questa API di alto livello, dovrai scrivere tu stesso la logica per confrontare la metrica di ogni epoch con la precedente e decidere quando fermarsi o salvare il modello: esattamente il tipo di codice ripetitivo che una buona astrazione evita.

## Riepilogo dei punti chiave

- Scrivere una rete "a mano" con tensori e gradienti ti aiuta a capire i fondamenti, ma è lento e soggetto a errori per il lavoro quotidiano.
- Keras è l'API di alto livello ufficiale di TensorFlow; TFLearn è stata un'alternativa popolare in passato con lo stesso scopo.
- I callback di Keras (`EarlyStopping`, `ModelCheckpoint`, tra gli altri) automatizzano le buone pratiche che, diversamente, dovrai programmare manualmente.
- La regola generale: usa l'astrazione di alto livello per il tuo lavoro quotidiano, e ricorri al livello basso solo quando hai bisogno di un controllo molto specifico che l'astrazione non offre.

## Mettilo in pratica

**Aggiungi** i due callback — `EarlyStopping` e `ModelCheckpoint` — all'addestramento del classificatore di cifre del capitolo 1, **configura** `EarlyStopping` con `patience=2`, e **verifica** in quante epoch l'addestramento si ferma automaticamente invece di completare quelle che gli hai indicato.

---

**Materiale sorgente di questo capitolo**: lezione 5 tema 1 ("TensorFlow Deep Learning Library — TFLearn").

[⬅ Capitolo precedente](04_cnn_fundamentos_IT.md) · [Torna all'indice](00_indice_IT.md) · [Capitolo successivo ➡](06_reinforcement_learning_IT.md)
