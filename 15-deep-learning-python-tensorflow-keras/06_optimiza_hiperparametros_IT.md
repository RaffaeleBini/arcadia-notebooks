[← Precedente: [TensorBoard: visualizza e depura i tuoi modelli](05_tensorboard_visualiza_IT.md) · [Indice](00_indice_IT.md) · Prossimo: [Metti il tuo modello alla prova →](07_pon_tu_modelo_a_prueba_IT.md)]

# 6. Ottimizza i tuoi modelli: ricerca di iperparametri

## Perché è importante per te

Quanti strati convoluzionali ti servono? 64 filtri o 128? Uno strato denso alla fine, o nessuno? Non c'è una risposta universale: dipende dai tuoi dati. Gli iperparametri sono tutte quelle decisioni di design che tu fissi prima di addestrare (a differenza dei pesi, che la rete impara da sola). Questo capitolo ti insegna a confrontare varianti di un modello in modo sistematico, invece di provare a caso.

## Quali iperparametri puoi regolare

- **Architettura**: numero di strati convoluzionali, numero di strati densi, numero di filtri o neuroni per strato
- **Ottimizzatore**: Adam, SGD (Stochastic Gradient Descent), RMSprop, ognuno con comportamenti diversi
- **Learning rate**: quanto regola l'ottimizzatore i pesi ad ogni step. Troppo alto e l'addestramento diventa instabile; troppo basso e impiega un'eternità a convergere
- **Dimensione del kernel e stride**: la dimensione del filtro convoluzionale e di quanto si sposta ad ogni step
- **Decay**: di quanto si riduce il learning rate mentre procede l'addestramento

## Genera e confronta combinazioni di architetture

Invece di cambiare il codice a mano ogni volta, scrivi un ciclo che genera diverse combinazioni automaticamente, addestrando e registrando ognuna in TensorBoard con un nome descrittivo:

```python
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

strati_densi = [0, 1, 2]
dimensioni_strato = [32, 64, 128]
strati_conv = [1, 2, 3]

for num_strati_conv in strati_conv:
    for dimensione in dimensioni_strato:
        for num_strati_densi in strati_densi:
            nome_modello = f"{num_strati_conv}conv-{dimensione}nodi-{num_strati_densi}densi-{int(time.time())}"
            tensorboard_callback = TensorBoard(log_dir=f'logs/{nome_modello}')
            print(f"Addestramento: {nome_modello}")

            model = Sequential()
            model.add(Conv2D(dimensione, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for _ in range(num_strati_conv - 1):
                model.add(Conv2D(dimensione, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(num_strati_densi):
                model.add(Dense(dimensione))
                model.add(Activation('relu'))

            model.add(Dense(1, activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(
                X, y,
                batch_size=32,
                epochs=10,
                validation_split=0.1,
                callbacks=[tensorboard_callback],
            )
```

Con tre opzioni in ognuno dei tre iperparametri, questo ciclo addestra 27 modelli diversi, ognuno con il proprio log in TensorBoard. Apri TensorBoard e confronta le curve di `val_accuracy`: il nome descrittivo di ogni modello ti dice a colpo d'occhio quale combinazione di iperparametri ha dato il risultato migliore.

Dieci epoch per combinazione e 27 combinazioni sono molti minuti di addestramento. Se vuoi solo esplorare il concetto senza aspettare tanto, riduci l'elenco di valori (per esempio, prova solo con `strati_conv = [1, 3]`) prima di lanciare il ciclo completo.

## Tecniche oltre la ricerca manuale

Il ciclo precedente è una ricerca esaustiva ("Grid Search") fatta a mano. Quando il numero di combinazioni diventa troppo grande, esistono alternative più efficienti:

- **Grid Search**: prova tutte le combinazioni possibili (quello che hai appena fatto), garantisce copertura completa ma scala male
- **Random Search**: prova combinazioni casuali invece di tutte, normalmente trova buoni risultati con molto meno calcolo
- **Bayesian Optimization**: usa i risultati delle prove precedenti per decidere intelligentemente quale combinazione provare dopo
- **AutoML** (Automated Machine Learning): strumenti come AutoKeras automatizzano sia la ricerca dell'architettura che quella degli iperparametri

## Altre leve di ottimizzazione

Oltre all'architettura, queste tecniche migliorano l'addestramento:

- **Regolarizzazione**: `Dropout` (disattiva neuroni a caso durante l'addestramento per evitare che la rete dipenda troppo da alcuni pochi) e `BatchNormalization` (normalizza gli ingressi di ogni strato, accelera e stabilizza l'addestramento)
- **Early stopping**: interrompe l'addestramento automaticamente se la metrica di validazione smette di migliorare, per non sprecare tempo né sovrappesare
- **Hardware**: addestrare su GPU (Graphics Processing Unit) o TPU (Tensor Processing Unit) invece che su CPU riduce drasticamente il tempo di addestramento in reti grandi

## Riepilogo dei punti chiave

- Gli iperparametri sono le decisioni di design che fissi tu prima di addestrare, non quello che la rete impara
- Un ciclo che genera e addestra combinazioni, ognuna con il suo log di TensorBoard, ti permette di confrontare architetture in modo obiettivo
- Grid Search, Random Search e Bayesian Optimization sono modi sempre più efficienti di esplorare lo spazio degli iperparametri
- Dropout, BatchNormalization e early stopping sono tecniche di ottimizzazione che complementano (non sostituiscono) la ricerca dell'architettura

## Mettilo in pratica

Esegui il ciclo di questo capitolo con una versione ridotta degli elenchi di iperparametri (per esempio, `strati_conv = [1, 2]`, `dimensioni_strato = [32, 64]`, `strati_densi = [0, 1]`) per mantenere il tempo di addestramento gestibile. Apri TensorBoard, identifica la combinazione con il massimo `val_accuracy` e **ragiona per iscritto** perché pensi che quella combinazione abbia vinto contro le altre.

---

[← Precedente: [TensorBoard: visualizza e depura i tuoi modelli](05_tensorboard_visualiza_IT.md) · [Indice](00_indice_IT.md) · Prossimo: [Metti il tuo modello alla prova →](07_pon_tu_modelo_a_prueba_IT.md)]
