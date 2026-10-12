# 07. Apprendimento strutturato e adversarial

## Perché questo argomento è importante

Fino ad ora hai addestrato modelli che trattano ogni esempio in modo isolato: le insegni un'immagine, una riga di un CSV o una frase, e il modello impara dalle sue caratteristiche (features) senza ulteriore contesto. Ma in molti problemi reali esiste un'informazione aggiuntiva che il modello non sta sfruttando. Se classifichi articoli scientifici, sai quale articolo cita quale. Se classifichi opinioni di film, puoi calcolare quali opinioni si assomigliano tra loro anche se non condividono una parola esatta. E se il tuo modello va in produzione, qualcuno potrebbe cercare di ingannarlo con input manipolati appositamente.

Neural Structured Learning (NSL, Apprendimento Strutturato Neurale) è un'estensione di TensorFlow (TF) progettata proprio per sfruttare quel tipo di informazione: relazioni tra esempi (grafi) e robustezza di fronte a manipolazioni (apprendimento adversarial). In questo capitolo imparerai ad aggiungere entrambi i segnali a un modello di Keras che già sai costruire, senza riprogettarlo da zero.

## Cosa apporta NSL rispetto all'addestramento standard

Nell'addestramento standard, la funzione di perdita (loss) misura solo l'errore del modello in ogni esempio separatamente: predizione rispetto all'etichetta reale. NSL aggiunge un secondo termine a quella perdita, un termine di regolarizzazione, che premia o penalizza il modello secondo come si comporta in relazione ad altri esempi.

Quel termine di regolarizzazione può provenire da due fonti:

- **Una struttura (grafo)**: il modello impara a dare predizioni simili per esempi che sono connessi in un grafo, sia perché la connessione esiste nei dati (grafo naturale) sia perché la crei tu a partire da una misura di somiglianza (grafo sintetizzato)
- **Una perturbazione adversarial**: il modello impara a dare predizioni simili per un esempio e una versione di quello stesso esempio leggermente manipolata per cercare di confonderlo

Nei due casi, la ricetta è la stessa ed è il grande vantaggio pratico di NSL: non hai bisogno di cambiare l'architettura del modello. Costruisci la tua rete con l'API (Application Programming Interface, Interfaccia di Programmazione di Applicazioni) di Keras che già conosci (sequenziale, funzionale o per sottoclassi) e poi la avvolgi con una classe del framework NSL (`GraphRegularization` o `AdversarialRegularization`). Quella classe aggiunge il termine di regolarizzazione alla perdita in automatico. Il risultato è un altro modello di Keras che compili, addestri e valuti esattamente come qualunque altro.

Questo importa specialmente quando hai pochi dati etichettati, una situazione abituale nei progetti reali, dove etichettare costa tempo e denaro. Sfruttando le relazioni tra esempi etichettati e non etichettati, NSL permette al modello di generalizzare meglio anche con una frazione piccola di etichette, e al contempo lo rende più robusto di fronte a dati rumorosi o manipolati.

## Addestramento con grafi naturali

Un **grafo naturale** è una rete di relazioni che già esiste nei tuoi dati, senza che tu debba inventarla. I nodi del grafo sono i tuoi esempi (documenti, persone utenti, immagini) e gli archi (edges) sono relazioni reali tra di loro: citazioni tra articoli scientifici, connessioni in una rete sociale, interazioni tra proteine.

Immagina che classifichi articoli scientifici per argomento, ma hai un'etichetta solo per una piccola parte di essi. Se l'articolo A cita l'articolo B, è ragionevole pensare che entrambi trattino un argomento simile, anche se A non ha etichetta. Quella è l'intuizione dietro la regolarizzazione di grafo: incoraggia il modello a dare predizioni simili per due nodi connessi da un arco, in modo che l'informazione fluisca dagli esempi etichettati verso i loro vicini senza etichetta.

L'esempio di riferimento di NSL per grafi naturali usa il dataset Cora, un grafo di citazioni tra 2.708 articoli di apprendimento automatico classificati in sette categorie. Il processo, in passi:

1. **Aumentare i dati con i vicini**: per ogni esempio, aggiungi le caratteristiche dei suoi vicini nel grafo ai dati di addestramento. NSL fa questo con la funzione `nsl.tools.pack_nbrs`, che combina il grafo di citazioni e le caratteristiche di ogni articolo (in Cora, una rappresentazione di quali parole del vocabolario contiene) in un unico dataset aumentato
2. **Definire un modello base**: una rete normale di Keras. Nell'esempio di Cora è un perceptron multistrato (MLP, Multi-Layer Perceptron, una rete densa di diversi strati completamente connessi), ma può essere qualunque architettura
3. **Avvolgere il modello con `nsl.keras.GraphRegularization`**, passandogli una configurazione (`nsl.configs.make_graph_reg_config`) che fissa, tra l'altro, quanti vicini usare e il peso relativo del termine di regolarizzazione di grafo rispetto alla perdita principale
4. **Compilare, addestrare e valutare** il modello avvolto esattamente come qualunque modello di Keras

Con questa tecnica, il modello di Cora migliora la sua precisione (accuracy) di circa 2-3 punti percentuali rispetto allo stesso modello senza regolarizzazione di grafo, semplicemente sfruttando la struttura di citazioni che già stava nei dati.

## Addestramento con grafi sintetizzati

Cosa succede se i tuoi dati non portano un grafo di serie? La maggior parte non lo porta: una collezione di opinioni di film, un insieme di immagini mediche o uno storico di vendite non vengono con "archi" espliciti. Per quei casi, NSL permette di **sintetizzare** un grafo a partire dalla somiglianza tra esempi.

L'idea è simile a organizzare una festa senza lista di invitati precedente: invece di usare relazioni che già esistono, decidi tu il criterio di prossimità (per esempio, "persone con hobby simili si siedono insieme") e costruisci le connessioni a partire da quel criterio. In NSL, il criterio di prossimità è una misura di somiglianza tra vettori.

Il processo, seguendo l'esempio di classificazione di sentimento sul dataset IMDb (recensioni di film di Internet Movie Database):

1. **Generare embeddings** di ogni esempio: una rappresentazione numerica densa che cattura il suo significato. Nell'esempio di IMDb si usa un modello preaddestrato di TensorFlow Hub (embeddings Swivel) per convertire il testo di ogni recensione in un vettore
2. **Costruire il grafo a partire dalla somiglianza tra embeddings**, normalmente con similarità coseno (cosine similarity): se due esempi superano una soglia di somiglianza, NSL aggiunge un arco tra di loro. Questo si fa con `nsl.tools.build_graph_from_config`
3. **Aumentare i dati con i vicini del grafo sintetizzato**, esattamente come nel caso di grafi naturali, con `nsl.tools.pack_nbrs`
4. **Definire il modello base, avvolgerlo con `GraphRegularization` e addestrare**, esattamente come con un grafo naturale

L'importante è che, una volta costruito il grafo (naturale o sintetizzato), il resto del processo di regolarizzazione di grafo è identico. La differenza è solo in come ottieni gli archi: da una relazione reale nel caso naturale, o da una misura di somiglianza che calcoli tu nel caso sintetizzato. Di fatto, nell'esempio di IMDb questa tecnica risulta specialmente utile quando ci sono poche etichette disponibili: con solo l'1% dei dati etichettati, un modello con regolarizzazione di grafo arriva a superare di circa 20 punti percentuali di precisione lo stesso modello senza regolarizzazione.

## Regolarizzazione adversarial: imparare a resistere ad attacchi

C'è uno scenario terzo: non hai nessun grafo, né naturale né facile da sintetizzare, ma ti preoccupa che qualcuno manipoli gli input del tuo modello appositamente per farlo fallire. Qui entra l'**apprendimento adversarial**.

Un **esempio adversarial** è una versione di un esempio originale con una perturbazione (un cambiamento) così piccola che un occhio umano la nota appena, ma calcolata specificamente per confondere il modello. La perturbazione non è rumore casuale: è calcolata seguendo la direzione che più aumenta l'errore del modello (la direzione inversa del gradiente), cioè il cambiamento più "efficace" possibile per provocare una predizione errata.

L'analogia più chiara è la vaccinazione: esponi il modello a una versione "indebolita" dell'attacco (una perturbazione piccola e controllata) durante l'addestramento, affinché impari a riconoscere e resistere quella famiglia di attacchi quando appaiano versioni reali più avanti.

Qui c'è la connessione con tutto quello che hai visto prima in questo capitolo: NSL tratta ogni esempio adversarial come un **vicino** del suo esempio originale, costruendo così una struttura al volo, esempio per esempio, anche quando non esiste un grafo esplicito. Connettendo l'esempio originale con il suo vicino adversarial, il modello impara a mantenere la stessa predizione per entrambi, con la stessa logica di "predizioni simili per nodi connessi" che hai visto nella regolarizzazione di grafo.

TensorFlow dispone inoltre di librerie specifiche per generare attacchi adversariali più elaborati, come cleverhans, che implementa metodi come il Fast Gradient Sign Method (FGSM, Metodo del Segno del Gradiente Veloce) o il Projected Gradient Descent (PGD, Discesa di Gradiente Proiettata), pensati per valutare a fondo la robustezza di un modello. NSL, d'altra parte, integra il suo proprio meccanismo di generazione di vicini adversariali direttamente nel flusso di addestramento, con un approccio più semplice da usare per regolarizzare un modello mentre lo addestri.

In pratica, il processo con NSL è:

1. **Definire un modello base** con Keras, come sempre
2. **Creare una configurazione adversarial** con `nsl.configs.make_adv_reg_config`, indicando il peso del termine adversarial nella perdita (`multiplier`) e la dimensione del passo della perturbazione (`adv_step_size`)
3. **Avvolgere il modello base con `nsl.keras.AdversarialRegularization`**
4. **Compilare, addestrare e valutare** come sempre

Il risultato, provato sul dataset MNIST (immagini di cifre scritte a mano da zero a nove), è convincente: un modello base ben addestrato raggiunge una precisione del 99% su dati puliti, ma quella precisione cade fino a vicino al 51% non appena gli si presentano immagini con perturbazioni adversariali. Lo stesso modello addestrato con regolarizzazione adversarial mantiene una precisione del 95% di fronte a quelle stesse immagini perturbate. La rete non è diventata più precisa nel caso normale, ma molto più difficile da ingannare.

## Esempio di codice commentato

L'esempio seguente, adattato dal tutorial ufficiale di NSL, addestra una rete neurale convoluzionale (CNN, Convolutional Neural Network, una rete con strati di convoluzione specializzati nel rilevare pattern visuali) su MNIST e aggiunge regolarizzazione adversarial con solo poche righe di codice extra.

```python
# Installa la libreria di Neural Structured Learning se non è ancora disponibile
!pip install --quiet neural-structured-learning

import neural_structured_learning as nsl  # convenzione abituale: abbreviare a "nsl"
import tensorflow as tf
import tensorflow_datasets as tfds

# 1. Carichiamo e normalizziamo MNIST esattamente come in qualunque problema di classificazione di immagini
datasets = tfds.load('mnist')
train_dataset, test_dataset = datasets['train'], datasets['test']

def normalize(features):
    features['image'] = tf.cast(features['image'], tf.float32) / 255.0
    return features

train_dataset = train_dataset.map(normalize).shuffle(10000).batch(32)
test_dataset = test_dataset.map(normalize).batch(32)

# 2. Definiamo un modello base normale, con la Functional API di Keras
def build_base_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name='image')
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

base_model = build_base_model()

# 3. Configuriamo la regolarizzazione adversarial:
#    - multiplier: peso del termine adversarial nella perdita totale
#    - adv_step_size: grandezza della perturbazione che si applica a ogni immagine
adv_config = nsl.configs.make_adv_reg_config(
    multiplier=0.2,
    adv_step_size=0.2,
    adv_grad_norm='infinity')

# 4. Avvolgiamo il modello base con AdversarialRegularization.
#    label_keys indica quale feature del dizionario contiene l'etichetta,
#    perché il modello adversarial ha bisogno dell'etichetta per calcolare la perturbazione.
adv_model = nsl.keras.AdversarialRegularization(
    base_model,
    label_keys=['label'],
    adv_config=adv_config)

# Il modello adversarial si aspetta dizionari {'image': ..., 'label': ...} invece di tuple
train_set_adv = train_dataset.map(lambda f: {'image': f['image'], 'label': f['label']})
test_set_adv = test_dataset.map(lambda f: {'image': f['image'], 'label': f['label']})

# 5. Compiliamo, addestriamo e valutiamo esattamente come un modello di Keras normale
adv_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
adv_model.fit(train_set_adv, epochs=5)
adv_model.evaluate(test_set_adv)
```

Nota che i cambiamenti rispetto a un addestramento standard sono minimi: una chiamata per creare la configurazione (`make_adv_reg_config`) e un'altra per avvolgere il modello (`AdversarialRegularization`). Tutto il resto — compilare, addestrare, valutare — è il flusso di Keras che già conosci.

## Riassunto dei punti chiave

- L'addestramento standard valuta ogni esempio separatamente; NSL aggiunge un termine di regolarizzazione alla perdita affinché il modello impari anche dalle relazioni tra esempi
- Un grafo naturale usa relazioni che già esistono nei dati (citazioni, connessioni sociali); un grafo sintetizzato si costruisce artificialmente a partire dalla somiglianza tra embeddings quando non esiste una relazione esplicita
- In entrambi i casi, il flusso è lo stesso: aumentare i dati con i vicini del grafo (`pack_nbrs`), definire un modello base di Keras e avvolgerlo con `nsl.keras.GraphRegularization`
- La regolarizzazione adversarial addestra il modello con esempi leggermente perturbati appositamente (vicini adversariali), usando `nsl.keras.AdversarialRegularization`, affinché impari a mantenere la stessa predizione nonostante quelle perturbazioni
- Nell'esempio di MNIST, un modello senza regolarizzazione adversarial cade dal 99% al 51% di precisione di fronte a immagini perturbate, mentre il modello regolarizzato si mantiene intorno al 95%
- Nessuna di queste tecniche richiede di riprogettare la tua rete: avvolgi il modello base già esistente con una classe di NSL e continui a compilare, addestrare e valutare come sempre

## Mettilo in pratica

Prima di toccare il codice, rifletti su questi tre scenari e decidi, per ognuno, se useresti un grafo naturale, un grafo sintetizzato o regolarizzazione adversarial:

1. Un ospedale vuole classificare cartelle cliniche per specialità. Le cartelle non citano altre cartelle, ma un modello di linguaggio può calcolare quanto il testo di due cartelle sia simile
2. Un social network vuole predire l'interesse di ogni persona utente in un argomento a partire dagli interessi dichiarati dalle persone che già segue
3. Un ente bancario vuole che il suo modello di rilevamento di frodi non fallisca se qualcuno prova a modificare leggermente i dati di una transazione per passare sotto il radar

Dopo, apri il notebook di pratica di questa classe e verifica il tuo ragionamento sullo scenario terzo aggiustando l'iperparametro `adv_multiplier` nell'esempio di MNIST: addestra il modello con un valore basso (per esempio, 0.05) e con uno alto (per esempio, 0.5), e confronta quanto cade la precisione su immagini perturbate in ogni caso.
