# 06. Buone pratiche di addestramento

## Perché questo argomento è importante

Già sai come costruire e addestrare modelli con Keras, l'API di alto livello di TensorFlow. Ma riuscire a far addestrare un modello non è lo stesso che ottenere un modello utile. Puoi addestrare una rete che indovini il 99% delle volte con i dati che già conosce e che fallisca miseramente non appena le mostri un esempio nuovo. Quel modello non serve a nulla in produzione: ha memorizzato l'esame invece di imparare la materia.

In questo capitolo imparerai a diagnosticare quel problema — e il suo opposto, un modello che non riesce nemmeno a imparare bene i dati di addestramento — e ad applicare le tecniche standard per correggerlo. Imparerai anche qualcosa altrettanto pratico: come salvare un modello già addestrato per non dover ripetere ore di addestramento ogni volta che ne hai bisogno, e come condividerlo con un'altra persona o distribuirlo in un altro ambiente.

Queste non sono tecniche facoltative né "avanzate": sono la differenza tra un esercizio di classe e un modello che puoi portare a un caso reale.

## Sovraddattamento e sottodattamento: due modi di fallire

Immagina due persone che si preparano per lo stesso esame a scelta multipla con una banca di 100 domande di pratica.

La prima ha imparato a memoria la risposta esatta di ognuna delle 100 domande, senza capire perché quella risposta sia corretta. Nel giorno dell'esame reale, con domande simili ma non identiche, fallisce quasi tutto: ha memorizzato la banca di domande invece di imparare la materia. Questo è il **sovraddattamento** (overfitting): il modello impara così bene i dati di addestramento — incluso il loro rumore e le loro particolarità — che perde la capacità di generalizzare a dati nuovi.

La seconda persona ha appena aperto il libro: non padroneggia nemmeno le domande di pratica. Nel giorno dell'esame fallisce sia le domande conosciute sia le nuove. Questo è il **sottodattamento** (underfitting): il modello è troppo semplice, o non è stato addestrato abbastanza, per catturare i pattern che esistono nei dati.

### Come riconoscerli con le curve di addestramento e validazione

Quando addestri un modello con Keras, in ogni epoch puoi registrare due valori di perdita (loss, l'errore che commette il modello): la perdita sui dati di addestramento e la perdita su un insieme di validazione che il modello non usa per aggiustare i suoi pesi. Confrontare quelle due curve è il modo più diretto di diagnosticare il problema:

- **Tutto va bene:** le due curve scendono insieme e si stabilizzano a un livello basso, con una piccola differenza tra loro
- **Sovraddattamento:** la perdita di addestramento continua a scendere, ma quella di validazione smette di scendere e inizia a salire. Da quel punto in poi il modello ha smesso di imparare pattern generali e ha iniziato a memorizzare il rumore dei dati di addestramento
- **Sottodattamento:** le due curve rimangono alte e scendono appena, anche dopo molti epoch (un epoch è un passaggio completo attraverso tutto l'insieme di addestramento). Il modello non ha capacità sufficiente, o non è stato addestrato abbastanza, per imparare la relazione tra i dati e il risultato che vuoi predire

Un esempio reale: se addestri tre modelli con capacità diversa (numero di strati e neuroni) sullo stesso problema, normalmente osservi che il modello più piccolo impiega più tempo a scendere nella sua perdita di addestramento (fatica a imparare: leggero sottodattamento), mentre il modello più grande scende nella sua perdita di addestramento molto velocemente ma la sua perdita di validazione inizia a salire subito (sovraddattamento). L'obiettivo è trovare il punto intermedio: la capacità minima necessaria per imparare bene il problema, né più né meno.

## Tecniche per correggere il sovraddattamento

Non esiste una formula unica per scegliere la dimensione perfetta di un modello: serve sperimentare. Ma sì esistono tecniche consolidate che riducono il sovraddattamento quando appare.

**Ottenere più dati di addestramento.** È la soluzione più efficace quando è fattibile: un modello addestrato con dati più completi generalizza meglio in modo naturale, perché ha meno margine per memorizzare casi concreti.

**Ridurre la capacità del modello.** Se la tua rete ha molti più neuroni e strati di quelli che il problema necessita, ha "memoria" in eccesso per memorizzare l'intero insieme di addestramento invece di imparare pattern generali. Inizia con un modello piccolo e aumenta la sua dimensione solo se la perdita di validazione migliora nel farlo.

**Regolarizzazione L2 (weight decay).** Aggiungi alla funzione di perdita una penalizzazione proporzionale al quadrato del valore dei pesi della rete. In pratica, questo costringe il modello a mantenere i suoi pesi piccoli, il che limita quanto "dettaglio" può memorizzare dai dati di addestramento. Quanto più grande è il parametro di regolarizzazione (chiamato lambda), più forte è quella penalizzazione. In Keras si applica strato per strato:

```python
from tensorflow.keras import layers, regularizers

layers.Dense(512, activation="relu",
             kernel_regularizer=regularizers.l2(0.001))
```

Esiste anche la regolarizzazione L1, che penalizza il valore assoluto dei pesi invece del loro quadrato e tende a portare alcuni pesi esattamente a zero (selezione di caratteristiche). L2 è la più abituale nelle reti neurali perché penalizza i pesi grandi senza forzare la dispersione.

**Dropout.** È una delle tecniche di regolarizzazione più usate nelle reti neurali. Durante l'addestramento, lo strato di dropout "spegne" casualmente una percentuale dei neuroni dello strato precedente a ogni passo (viene loro assegnato un valore di zero), costringendo la rete a non dipendere eccessivamente da nessun neurone concreto. Il tasso di dropout abituale è tra il 20% e il 50%. Nella fase di valutazione o predizione, il dropout viene disattivato e si usa la rete completa:

```python
layers.Dense(512, activation="relu"),
layers.Dropout(0.5),
```

**Early stopping (arresto anticipato).** Invece di fissare in anticipo un numero di epoch, controlli la perdita di validazione durante l'addestramento e fermi il processo non appena smette di migliorare durante un numero di epoch consecutive (il parametro `patience`). In questo modo eviti di continuare ad addestrare dopo il punto in cui il modello inizia a sovraddattarsi, e inoltre risparmi tempo di calcolo:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=5,
                            restore_best_weights=True)
```

**Aumento dei dati (data augmentation).** Quando non è possibile ottenere più dati reali, puoi generare variazioni artificiali dei dati che hai già. Nelle immagini, per esempio, applicare rotazioni, ritagli o capovolgimenti orizzontali alle foto di addestramento crea esempi nuovi e validi senza necessità di raccogliere nemmeno un'immagine in più. Questo riduce il sovraddattamento perché il modello vede più varietà e dipende meno da dettagli concreti di ogni immagine originale.

In pratica, nessuna di queste tecniche è mutuamente esclusiva: combinare dropout, regolarizzazione L2 e early stopping di solito dà risultati migliori di applicarne solo una.

### E se il problema è sottodattamento

Se il tuo modello non riesce nemmeno a imparare bene i dati di addestramento, la soluzione va nella direzione opposta: aumenta la capacità del modello (più strati o più neuroni), addestra durante più epoch, o verifica se le caratteristiche (features) che gli dai come input sono sufficienti per risolvere il problema.

## Salvare e caricare modelli

Addestrare un modello può richiedere minuti, ore o giorni. Senza salvarlo, perdi quel lavoro non appena chiudi il processo: non puoi riutilizzarlo, né condividerlo, né continuare ad addestrarlo più tardi da dove lo hai lasciato. TensorFlow offre diversi modi di salvare il lavoro, a seconda di quello che hai bisogno di conservare.

### Salvare solo i pesi: checkpoint

Un checkpoint salva solo i valori appresi dal modello (i suoi pesi), non la sua architettura né la configurazione dell'ottimizzatore. Per recuperarli hai bisogno di ricostruire un modello con esattamente la stessa architettura e poi caricare i pesi salvati in esso. Questo è utile, per esempio, per riprendere un addestramento lungo che è stato interrotto, o per riutilizzare i pesi appresi in un'architettura leggermente diversa.

Il modo più pratico di salvare checkpoint automaticamente durante l'addestramento è il callback `ModelCheckpoint`:

```python
import os
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = "training/cp.ckpt"

cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                               save_weights_only=True,
                               verbose=1)

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])
```

Per riutilizzare quei pesi in seguito, crei un modello con la stessa architettura (senza addestrare) e gli carichi i pesi salvati:

```python
model = create_model()          # stessa architettura, pesi non addestrati
model.load_weights(checkpoint_path)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Precisione del modello ripristinato: {acc:.2%}")
```

Puoi anche salvare e caricare i pesi manualmente, senza passare dall'addestramento, con `model.save_weights(percorso)` e `model.load_weights(percorso)`.

### Salvare il modello completo

Se vuoi conservare l'architettura, i pesi e lo stato dell'ottimizzatore in un unico file — per poterlo caricare e usare direttamente senza dover ridefinire il modello nel codice — usa `model.save()`. Il formato consigliato in TensorFlow 2.x è `.keras`:

```python
model.save("mi_modelo.keras")

# Più tardi, nello stesso script o in uno diverso:
modelo_cargado = tf.keras.models.load_model("mi_modelo.keras")
modelo_cargado.summary()
```

Il modello caricato con `load_model` è pronto per essere usato così com'è: puoi chiamare `evaluate()` o `predict()` senza ricompilarlo, perché il file include anche la configurazione dell'ottimizzatore e della funzione di perdita.

TensorFlow supporta altri due formati, che continui a trovare nei progetti esistenti:

| Formato | Estensione | Cosa salva | Quando usarlo |
|---|---|---|---|
| Keras nativo | `.keras` | Architettura, pesi e ottimizzatore | Formato consigliato per default in TF 2.x |
| SavedModel | directory, senza estensione | Architettura, pesi e il grafo di esecuzione completo | Quando hai bisogno di servire il modello con TensorFlow Serving o usarlo in TensorFlow Lite o TensorFlow.js |
| HDF5 | `.h5` | Architettura, pesi e ottimizzatore | Formato ereditato (legacy), utile per compatibilità con progetti vecchi |

HDF5 (Hierarchical Data Format, versione 5) è stato il formato standard nelle prime versioni di TensorFlow 2.x e rimane supportato, ma per progetti nuovi usa `.keras`.

## Riassunto dei punti chiave

- Il sovraddattamento (overfitting) avviene quando la perdita di addestramento continua a scendere mentre quella di validazione si ferma o sale: il modello memorizza invece di generalizzare
- Il sottodattamento (underfitting) avviene quando entrambe le perdite rimangono alte: il modello non ha capacità o addestramento sufficiente
- Confrontare le curve di addestramento e validazione in ogni epoch è il modo più diretto di diagnosticare quale dei due problemi hai
- Dropout, regolarizzazione L2, early stopping, aumento dei dati e ridurre la capacità del modello sono le tecniche standard contro il sovraddattamento, e combinarle di solito funziona meglio di applicarne solo una
- I checkpoint (`ModelCheckpoint`, `save_weights` / `load_weights`) salvano solo i pesi e richiedono di ricostruire la stessa architettura per riutilizzarli
- `model.save()` e `tf.keras.models.load_model()` salvano e recuperano il modello completo — architettura, pesi e ottimizzatore — in formato `.keras`, il consigliato in TensorFlow 2.x

## Mettilo in pratica

Prendi qualunque modello che hai addestrato in un capitolo precedente di questo corso (per esempio, il classificatore di immagini del capitolo 2 o il modello di classificazione del capitolo 5) e fai quanto segue:

1. Addestralo durante più epoch di quelle necessarie e disegna le curve della perdita di addestramento e di validazione. Identifica da quale epoch inizia a sovraddattarsi
2. Aggiungi uno strato `Dropout` e, se il modello lo permette, regolarizzazione L2 ad alcuni dei suoi strati densi. Addestra di nuovo e confronta le nuove curve con quelle del passo precedente: il sovraddattamento si ritarda o scompare?
3. Salva i pesi del miglior modello che ottieni con un `ModelCheckpoint` e, in una cella o script separato, ricostruisci l'architettura da zero e carica quei pesi con `load_weights`. Verifica che la precisione corrisponda a quella del modello originale

Se vuoi continuare a praticare in modo guidato, apri il notebook di pratica di questa classe: `clase04_practica_buenas_practicas_entrenamiento.ipynb`.
