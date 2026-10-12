# 03. Visione artificiale e reti neurali

## Perché questo argomento è importante

Riconoscere un capo d'abbigliamento in una foto ti sembra banale: guardi l'immagine e
in un istante sai se è una scarpa da ginnastica, un cappotto o un pantalone. Ma prova
a spiegare quel processo con regole esplicite, del tipo "se c'è una suola piatta e una
zona alta che avvolge la caviglia, allora è uno stivale". Avrai bisogno di centinaia
di regole, e comunque falliranno nel momento in cui cambiano l'angolo, la luce o il
colore del capo.

Questo è esattamente il tipo di problema che la programmazione tradizionale risolve
male e l'apprendimento automatico o ML (Machine Learning) risolve bene: invece di
scrivere le regole a mano, insegni al computer migliaia di esempi già etichettati
("questo è uno stivale", "questa è una maglietta") e lasci che sia la stessa rete
neurale a scoprire quali pattern di pixel distinguono un capo da un altro. In questo
capitolo costruisci il tuo primo classificatore di immagini con una rete neurale densa
e, nel contempo, scopri perché quel primo approccio ha limiti chiari che nel capitolo 4
risolverai con le convoluzioni.

## La sfida del "vedere" per un computer

Per un computer, un'immagine non è una scarpa né una maglietta: è una griglia di numeri.
Un'immagine in scala di grigi di 28x28 pixel è, in realtà, una matrice di 28 righe per
28 colonne dove ogni cella contiene un valore tra 0 e 255 che indica l'intensità di quel
pixel (0 è nero, 255 è bianco). La rete neurale non "vede" mai un capo d'abbigliamento:
vede 784 numeri (28x28) e deve imparare quali combinazioni di quei numeri si trovano
solitamente in uno stivale e quali in una maglietta.

Questa è la chiave della visione artificiale con ML: trasformi un problema visivo in un
problema numerico, e lasci che il modello impari la relazione tra quei numeri e l'etichetta
corretta a partire da moltissimi esempi.

## Il dataset Fashion MNIST

Per esercitarti con questo primo classificatore usi Fashion MNIST, un set di
dati che sostituisce il classico MNIST di cifre scritte a mano con capi d'abbigliamento. Fashion MNIST contiene 70.000 immagini in scala di grigi,
divise in 60.000 immagini di addestramento e 10.000 immagini di test, tutte di 28x28 pixel e classificate in dieci categorie:

| Etichetta | Classe (inglese) | Classe (italiano) |
|---|---|---|
| 0 | T-shirt/top | Maglietta |
| 1 | Trouser | Pantalone |
| 2 | Pullover | Maglione |
| 3 | Dress | Vestito |
| 4 | Coat | Cappotto |
| 5 | Sandal | Sandalo |
| 6 | Shirt | Camicia |
| 7 | Sneaker | Scarpa da ginnastica |
| 8 | Bag | Borsa |
| 9 | Ankle boot | Stivale |

Nota che le etichette sono numeri (0 a 9) e non testo. Questo non è casuale: a un modello è più efficiente lavorare con i numeri che con le stringhe di testo, e inoltre evita di introdurre pregiudizi verso una lingua specifica. TensorFlow include Fashion MNIST direttamente nella sua API di set di dati, così caricarlo è facile come questo:

```python
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

# load_data() restituisce due tuple: dati di addestramento e dati di test
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

Perché due set? Perché devi verificare se il tuo modello ha imparato a generalizzare o se ha semplicemente memorizzato gli esempi di addestramento. Il set di test contiene immagini che il modello non vede durante l'addestramento, così valutarlo con quelle immagini ti dà una misura onesta di quanto bene funzionerebbe con dati nuovi.

## Normalizzazione dei dati

Ogni pixel di Fashion MNIST ha un valore tra 0 e 255. Le reti neurali si addestrano meglio e più velocemente quando tutti i valori di input sono in un intervallo piccolo e omogeneo, tipicamente tra 0 e 1. Questo processo si chiama **normalizzazione**, e in questo caso si riduce a dividere ogni valore di pixel per 255:

```python
training_images = training_images / 255.0
test_images = test_images / 255.0
```

È importante normalizzare il set di addestramento e quello di test esattamente allo stesso modo. Se salti questo passaggio, il modello può continuare ad addestrarsi, ma con risultati notevolmente peggiori: i gradienti si muovono in intervalli molto diversi e l'ottimizzatore impiega più tempo a trovare buoni valori per i parametri.

## Strati di una rete densa

Con i dati preparati, il passo successivo è definire l'architettura del modello. Una rete neurale densa per questo problema ha tre componenti:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

- **Sequential**: indica che gli strati si eseguono uno dopo l'altro, in sequenza, ognuno ricevendo l'output del precedente
- **Flatten**: converte la matrice di 28x28 in un unico vettore di 784 valori. Questo strato non impara nulla, solo riorganizza i dati: è il passo obbligatorio per poter collegare un'immagine 2D a uno strato denso, che si aspetta una lista piatta di numeri
- **Dense**: uno strato "densamente connesso", dove ogni neurone riceve tutti i valori dello strato precedente. Il primo strato denso ha 128 neuroni, ognuno con i suoi parametri che il modello aggiusterà durante l'addestramento
- L'ultimo strato **Dense** ha dieci neuroni, uno per ogni classe di Fashion MNIST

Una regola pratica da ricordare: il primo strato della tua rete deve corrispondere alla forma dei tuoi dati di input (per questo `Flatten` riceve `input_shape=(28, 28)`), e l'ultimo strato deve avere tanti neuroni quante sono le classi che vuoi distinguere. Se addestri con dieci classi ma configuri l'ultimo strato con un numero diverso di neuroni, l'addestramento fallirà.

## Funzioni di attivazione e softmax

Ogni strato denso ha bisogno di una **funzione di attivazione**, che decide quale segnale passa da quella neurone al prossimo strato. Senza funzioni di attivazione non lineari, impilare strati densi equivarrebbe a un'unica operazione lineare, non importa quanti strati aggiungi.

- **ReLU** (Rectified Linear Unit, o Unità Lineare Rettificata): la regola è molto semplice, se il valore di input è maggiore di zero lo lascia passare così com'è, e se è zero o negativo lo converte in zero. È l'attivazione più usata negli strati intermedi perché è economica da calcolare e funziona bene nella pratica
- **Softmax**: si usa nello strato di output quando il problema è di classificazione multiclasse (scegliere una tra varie categorie, come qui). Softmax prende un insieme di valori e li trasforma in probabilità che sommano a 1. Per esempio, un output come `[0.1, 0.1, 9.5, 0.1, ...]` diventa qualcosa di simile a `[0.001, 0.001, 0.99, 0.001, ...]`: la categoria con il valore più alto si prende quasi tutta la probabilità, così puoi leggere direttamente quale è la previsione del modello e con quanta fiducia la fa

Pensa a softmax come a un "e il vincitore è...": invece di darti dieci numeri separati che avresti dovuto confrontare tu stesso, ti fornisce direttamente una distribuzione di probabilità sulle dieci classi.

## Compilare e addestrare il modello

Prima di addestrare, compili il modello indicando tre componenti:

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
```

- L'**ottimizzatore** (`adam` in questo caso) è l'algoritmo che aggiusta i parametri del modello ad ogni passo di addestramento per ridurre l'errore
- La **funzione di perdita** (`sparse_categorical_crossentropy`) misura quanto lontane sono le previsioni del modello rispetto alle etichette corrette. È il segnale che l'ottimizzatore usa per sapere in quale direzione aggiustare i parametri. Si usa la variante "sparse" perché le etichette sono numeri interi (0 a 9) e non vettori codificati
- Le **metriche** (`accuracy`, o accuratezza) non influenzano l'addestramento, ti informano solo di quanto bene sta facendo il modello

`model.fit` è la chiamata che avvia l'addestramento: il modello percorre le immagini di addestramento durante cinque epoche (cinque passate complete per tutto il set di dati), aggiustando i suoi parametri in ognuna per migliorare l'accuratezza.

## Valutare il modello e i suoi limiti

Una volta addestrato, verifichi la performance con dati che il modello non ha visto:

```python
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Accuratezza nel set di test: {test_accuracy:.4f}')
```

È usuale osservare un'accuratezza un po' più bassa nel set di test che in quello di addestramento (per esempio, 91% in addestramento contro 88% in test). Quella differenza è un primo segno di overfitting, un argomento che riprenderai più avanti nel corso.

Questo primo modello funziona ragionevolmente bene con Fashion MNIST, ma ha un'importante limitazione: impara direttamente dai valori dei pixel grezzi, senza alcun concetto di "forma" o "bordo". Ciò implica che:

- Funziona bene solo se il capo è centrato nell'immagine
- Richiede che tutte le immagini abbiano esattamente la stessa dimensione (28x28) e siano in scala di grigi
- Si comporta male se l'immagine ha sfondo, più oggetti o il capo è spostato o ruotato

In altre parole, questa rete ha imparato a riconoscere pattern di pixel molto specifici, non a riconoscere le caratteristiche visive che rendono uno stivale uno stivale. Nel capitolo 4 risolvi proprio quella limitazione introducendo le convoluzioni, che insegnano al modello a concentrarsi prima sulle caratteristiche (bordi, curve, texture) prima di classificare.

## Esempio di codice commentato

```python
import tensorflow as tf

# Carichiamo il dataset Fashion MNIST direttamente da Keras
fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalizziamo i pixel da 0-255 a 0-1 affinché la rete si addestri meglio
training_images = training_images / 255.0
test_images = test_images / 255.0

# Definiamo una rete densa semplice
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),      # 28x28 -> vettore di 784
    tf.keras.layers.Dense(128, activation='relu'),       # strato nascosto
    tf.keras.layers.Dense(10, activation='softmax')      # 10 classi di output
])

# Compiliamo indicando ottimizzatore, funzione di perdita e metrica
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Addestriamo per 5 epoche
model.fit(training_images, training_labels, epochs=5)

# Valutiamo con dati che il modello non ha mai visto
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Accuratezza in test: {test_accuracy:.4f}')

# Facciamo previsioni sul set di test
predictions = model.predict(test_images)
print(predictions[0])          # probabilità di ognuna delle 10 classi
print(test_labels[0])          # etichetta reale di quell'immagine
```

## Riassunto dei punti chiave

- Un computer non "vede" immagini, elabora matrici di numeri: normalizzare quei numeri (dividere per 255) aiuta il modello ad addestrarsi meglio
- `Flatten` converte un'immagine 2D in un vettore 1D affinché possa entrare in strati densi, che collegano ogni neurone con tutte quelle dello strato precedente
- L'attivazione ReLU introduce non linearità negli strati nascosti; softmax converte l'output in probabilità quando il problema ha più classi
- Hai bisogno di un ottimizzatore per aggiustare i parametri e di una funzione di perdita per misurare l'errore; l'accuratezza è la metrica che ti dice quanto bene funziona il modello
- Una rete densa addestrata su pixel grezzi funziona bene solo con immagini centrate, della stessa dimensione e senza sfondo: quella limitazione è la porta d'ingresso alle reti neurali convoluzionali del capitolo 4

## Mettilo in pratica

Apri il notebook di teoria di questa lezione e, dopo aver addestrato il modello dell'esempio, cambia il numero di neuroni dello strato nascosto da 128 a 32 e addestra di nuovo. Annota l'accuratezza nel set di test e il tempo che impiega ogni epoca. L'accuratezza migliora, peggiora o rimane uguale? Corrisponde a ciò che ti aspettavi? Salva le tue conclusioni in una cella di testo: ti serviranno per confrontarle con la rete convoluzionale del prossimo capitolo.
