# 09. NLP: sentimento, RNN, LSTM e generazione di testo

## Perché questo argomento è importante

Nel capitolo precedente hai imparato a convertire testo in numeri: tokenizzazione e sequenziazione. Ma convertire parole in numeri è solo il primo passo. L'obiettivo reale è quasi sempre uno di questi due: **capire** cosa dice un testo (per esempio, se una recensione di film è positiva o negativa) o **generare** testo nuovo (per esempio, una strofa di poesia). Entrambi i compiti condividono una difficoltà che non avevi dovuto risolvere finora: l'ordine delle parole importa, e a volte la parola chiave per capire una frase è molto lontana da dove serve usarla.

In questo capitolo addestrai un classificatore di sentimento reale, riutilizzerai embeddings già addestrati da altri con TensorFlow Hub, e scoprirai perché una rete densa è insufficiente con dati sequenziali, cosa apportano le reti neurali ricorrenti (RNN, Recurrent Neural Network) e perché, in pratica, quasi sempre si usa una loro variante chiamata memoria a lungo-corto plazo (LSTM, Long Short-Term Memory). Chiuderai il capitolo generando testo nuovo con un modello LSTM addestrato con testi di canzoni.

## Addestrare un classificatore di sentimento con embeddings

Riprendi l'esempio del capitolo precedente: un insieme di titoli di notizie etichettati come sarcastici o non sarcastici (il dataset lo ha raccolto Rishabh Misra ed è disponibile pubblicamente). Già sai tokenizzare i titoli, convertirli in sequenze numeriche e riempirli con padding fino a una lunghezza comune. Il passo successivo è progettare una rete che impari, a partire da quelle sequenze e dai loro attributi, a distinguere il sarcasmo.

L'architettura più semplice che funziona bene per questo combina tre pezzi:

```python
import tensorflow as tf

vocab_size = 10000
embedding_dim = 16
max_length = 100

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

- Lo strato **Embedding** converte ogni parola (rappresentata come un numero intero) in un vettore denso di `embedding_dim` valori. All'inizio questi vettori sono casuali; durante l'addestramento, la rete li aggiusta affinché parole con un sentimento simile finiscano con vettori che puntano in direzioni simili
- **GlobalAveragePooling1D** fa la media dei vettori di tutte le parole di una frase in uno solo, di dimensione fissa, indipendentemente da quante parole aveva la frase originale
- I due strati **Dense** finali fanno il lavoro di classificazione propriamente detto, terminando in un neurone con attivazione sigmoide che produce un valore tra zero e uno (più vicino a uno, più sarcastico)

Addestri questo modello esattamente come qualunque altro con Keras: `model.fit(training_padded, training_labels, epochs=30, validation_data=(testing_padded, testing_labels))`. Con questa architettura così semplice è abituale raggiungere una precisione (accuracy) vicina al 99% sui dati di addestramento e dal 81% all'82% sui dati di prova (titoli che la rete non ha mai visto). Quella differenza è un segnale di sovraddattamento (overfitting) — lo hai trattato nel capitolo 6 — e un motivo in più per sorvegliare sempre le curve di addestramento rispetto a quelle di validazione.

## Cosa sono gli embeddings e perché catturano il sentimento

Un embedding è un modo di rappresentare una parola come un punto in uno spazio di diverse dimensioni, in modo che parole con significati o connotazioni simili stiano vicine le une alle altre. Immagina un piano con due assi: uno per "sentimento negativo → positivo" e uno per una sfumatura secondaria, come l'intensità. Quando addestri la rete con migliaia di titoli etichettati, lo strato Embedding va imparando quale direzione di quello spazio corrisponde a "sarcastico" e quale a "non sarcastico". Una parola che appare solo in titoli sarcastici finisce con un vettore che punta con forza verso quella direzione; una parola neutra sta vicino al centro.

Questo spiega qualcosa che a prima vista sembra magia: se chiedi al modello di valutare la frase "not bad, a bit meh" (niente di male, un po' tedioso), il risultato non dipende dal cercare una parola positiva o negativa isolata, ma dalla somma dei vettori di tutte le parole della frase. Il risultato finale è un punto nello spazio del sentimento — per esempio, leggermente positivo — anche se nessuna parola per sé sia chiaramente positiva.

## Riutilizzare embeddings preaddestrati con TensorFlow Hub

Addestrare i tuoi propri embeddings da zero funziona, ma ha un costo: hai bisogno di molti dati etichettati affinché la rete impari buone rappresentazioni. TensorFlow Hub è un repository di modelli e componenti già addestrati da Google e dalla comunità, pronti per riutilizzare nei tuoi propri progetti. Per compiti di testo, questo significa che puoi prendere un embedding addestrato su miliardi di parole e usarlo direttamente come primo strato della tua rete, senza dover tokenizzare né preprocessing il testo tu stesso.

L'esempio di riferimento è un classificatore di recensioni di film del dataset IMDB (Internet Movie Database), che contiene 50.000 recensioni già etichettate come positive o negative:

```python
import tensorflow as tf
import tensorflow_hub as hub

# Strato di embedding preaddestrato: riceve frasi di testo senza tokenizzazione
# e restituisce un vettore di 50 dimensioni per frase
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
                     epochs=10,
                     validation_data=validation_data.batch(512))
```

Nota tre dettagli importanti:

- Lo strato `hub_layer` riceve direttamente stringhe di testo (`dtype=tf.string`), senza passare per `Tokenizer` né per `pad_sequences`: il modello preaddestrato include già il suo proprio modo di tokenizzare e combinare parole in un vettore di dimensione fissa
- `trainable=True` permette all'embedding di continuare ad adattarsi (fine-tuning) durante il tuo addestramento; se lo metti in `False`, l'embedding rimane congelato e addestri solo gli strati che aggiungi sopra
- Il resto del modello — due strati Dense — è identico nello spirito al classificatore di sarcasmo che hai addestrato prima: una volta hai un vettore di dimensione fissa che rappresenta la frase, il resto è un problema di classificazione binaria normale

Questo approccio — usare un componente preaddestrato come primo strato — si chiama apprendimento per trasferimento (transfer learning) ed è una delle tecniche più redditizie in Machine Learning: ti fa risparmiare tempo di addestramento e dati, e anche con un'architettura molto semplice, questo modello raggiunge una precisione vicina all'87% sul dataset di prova.

## Cosa risolve una rete ricorrente rispetto a una rete densa

Per classificare il sentimento, fare la media dei vettori di tutte le parole (come fa `GlobalAveragePooling1D`) funziona ragionevolmente bene perché il sentimento generale di una frase non dipende troppo dall'ordine esatto delle parole. Ma ci sono compiti dove l'ordine è tutto: se vuoi generare la parola successiva di una frase, hai bisogno di sapere quali parole sono venute prima e in quale ordine. Una rete densa ordinaria non ha modo di ricordare quella sequenza: elabora ogni input in modo indipendente, senza memoria di quello che è venuto prima.

Qui entra la rete neurale ricorrente (RNN). Pensa alla successione di Fibonacci: ogni numero si calcola sommando i due precedenti, così ogni valore della serie porta "dentro di sé" una traccia di tutti i valori precedenti. Una RNN applica quella stessa idea al linguaggio: ogni neurone ricorrente riceve la parola attuale e, inoltre, un valore che viene dal neurone precedente (la sua "memoria" di quello che ha già visto). Quel valore si va aggiornando e passando avanti, parola per parola, in modo che la rete mantenga una traccia del contesto della frase mentre la elabora.

Il problema è che quella traccia si va diluendo con la distanza. Nella frase "I lived in Ireland, so I learned how to speak ___", la parola corretta è "Gaelic" (gaelico), ma l'indizio che lo determina — "Ireland" — è molto lontano da dove serve usarlo. Una RNN semplice tende a dare molto peso alle parole vicine e poco alle lontane, così nelle frasi lunghe perde proprio l'informazione che servirebbe per indovinare.

## Cosa apporta LSTM rispetto a una RNN semplice

Questo indebolimento del contesto con la distanza ha un nome tecnico: il problema del gradiente che scompare (vanishing gradient problem). Durante l'addestramento, la rete aggiusta i suoi pesi propagando l'errore all'indietro nel tempo (attraverso ogni passo della sequenza); in una RNN semplice, quell'errore si va riducendo esponenzialmente ad ogni passo che retrocede, così le parole all'inizio di una frase lunga influenzano poco l'aggiustamento finale. In pratica, questo significa che una RNN semplice impara bene le dipendenze corte e fallisce in quelle lunghe.

Le reti di memoria a lungo-corto plazo (LSTM) sono state progettate specificamente per risolvere questo problema. La differenza chiave è che una LSTM mantiene, oltre alla traccia abituale, uno **stato di cella** (cell state) che agisce come un nastro trasportatore di informazione: può percorrere molti passi della sequenza quasi senza degradarsi, portando con sé dati rilevanti da molto più indietro nella frase. Tre "porte" (gates) controllano cosa entra, cosa esce e cosa si dimentica di quel nastro:

- La **porta di input** (input gate) decide quanta informazione nuova si aggiunge allo stato di cella
- La **porta di dimenticanza** (forget gate) decide quanta informazione vecchia si scarta
- La **porta di output** (output gate) decide quale parte dello stato di cella si usa per produrre l'output di quel passo

Grazie a queste porte, una LSTM può imparare a "ricordare" che la frase ha menionato "Ireland" diverse parole fa e usarlo per predire correttamente "Gaelic", qualcosa che a una RNN semplice sfugge.

Due dettagli pratici che vedrai nel codice:

- Una LSTM può essere **bidirezionale**: oltre a leggere la frase da sinistra a destra, la legge anche da destra a sinistra, e combina entrambe le letture. Questo aiuta perché, a volte, una parola successiva fornisce contesto su una parola precedente. In TensorFlow si implementa avvolgendo lo strato LSTM in `tf.keras.layers.Bidirectional`
- Se impili diversi strati LSTM uno sopra l'altro, tutti tranne l'ultimo devono restituire la sequenza completa di output (non solo l'ultimo valore) affinché lo strato successivo abbia tutte le informazioni di cui ha bisogno. Questo si controlla con il parametro `return_sequences=True`

```python
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.models import Sequential

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_sequence_len),
    Bidirectional(LSTM(64)),
    Dense(num_classes, activation='softmax')
])
```

## Generare testo nuovo con un modello LSTM

Tutto quello che hai visto su LSTM si può riutilizzare, con un cambio di prospettiva, per generare testo nuovo invece di classificarlo. L'idea è addestrare una rete affinché, data una sequenza di parole, predica quale è la parola più probabile successiva. Una volta addestrata, puoi "seminare" il modello con una frase iniziale e lasciare che predica parola per parola, aggiungendo ogni predizione alla frase e ripetendo il processo: così generi testo nuovo, parola per parola, a partire dai pattern che ha imparato.

L'esempio di riferimento usa il testo di canzoni tradizionali irlandesi come corpus di addestramento. Il processo di preparazione dei dati ha una differenza importante rispetto al classificatore di sentimento:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
corpus = data.lower().split("\n")          # una riga della canzone per elemento
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Per ogni riga, generiamo tutti gli n-grammi possibili:
# "in un luogo" -> ["in", "un"], ["in", "un", "luogo"]
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i + 1])

max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Tutte le parole di ogni n-gramma tranne l'ultima sono l'input (X);
# l'ultima parola è l'etichetta che il modello deve imparare a predire (Y)
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
```

Qui hai due decisioni di progettazione che vale la pena capire:

- Un **n-gramma** è una sequenza di n parole consecutive. Quando spezzi ogni riga in tutti i suoi n-grammi possibili, converti un compito di "genera la frase completa" in molti compiti più piccoli di "data questa sequenza, predici la parola successiva", che è proprio quello che il modello ha bisogno di imparare
- Gli attributi si codificano con **one-hot encoding**: ogni parola del vocabolario si rappresenta come un vettore binario con un uno nella posizione di quella parola e zeri nel resto. Questo permette di trattare la predizione della parola successiva come una classificazione con tante classi quante parole ha il vocabolario, addestrata con la funzione di perdita `categorical_crossentropy`

Il modello in sé è una LSTM bidirezionale simile a quella che già conosci, terminata in uno strato denso con tanti neuroni quante parole ha il vocabolario e attivazione softmax (affinché l'output sia una distribuzione di probabilità sulla parola successiva):

```python
from tensorflow.keras.optimizers import Adam

model = tf.keras.Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len - 1),
    Bidirectional(LSTM(150)),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1)
```

Con questa architettura relativamente semplice, è abituale raggiungere una precisione dal 70% al 75% nel predire la parola successiva corretta; sufficiente per generare testo con una coerenza locale accettabile, anche se non perfetta. Per generare la poesia nuova, ripeti il ciclo di predizione tante volte quante parole vuoi aggiungere:

```python
seed_text = "I've got a bad feeling about this"
next_words = 20

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)
```

Questa stessa tecnica — tokenizzare, generare n-grammi, addestrare una LSTM a predire l'unità di testo successiva e poi generare iterativamente — funziona allo stesso modo se lavori carattere per carattere invece che parola per parola; cambia solo l'unità minima che tokenizzi. Scegliere parole o caratteri è una decisione di progettazione: per parole hai bisogno di meno passi per generare frasi lunghe, ma il vocabolario è più grande; per caratteri il vocabolario è molto più piccolo, ma hai bisogno di più passi e più dati affinché il modello impari ortografia e struttura.

## Riassunto dei punti chiave

- Un classificatore di sentimento semplice combina uno strato Embedding (impara vettori di parole), una media di quei vettori per frase e strati densi di classificazione
- Gli embeddings catturano sentimento e significato come direzioni in uno spazio vettoriale: parole simili finiscono con vettori simili
- TensorFlow Hub permette di riutilizzare embeddings già addestrati da terzi come primo strato del tuo modello, senza tokenizzare tu stesso il testo e con il beneficio dell'apprendimento per trasferimento
- Una rete densa non ha memoria di quello che ha visto prima; una RNN sì, perché ogni passo riceve il risultato del passo precedente, come ogni termine della successione di Fibonacci dipende dai precedenti
- Le RNN semplici soffrono del problema del gradiente che scompare: perdono la traccia delle parole lontane in frasi lunghe
- LSTM risolve questo con uno stato di cella e tre porte (input, dimenticanza, output) che decidono quale informazione conservare a lungo termine; può essere bidirezionale e si può impilare usando `return_sequences=True`
- Generare testo nuovo è, in fondo, un problema di classificazione: predire la parola (o il carattere) più probabile successiva, a partire dagli n-grammi di un corpus di addestramento e codificazione one-hot degli attributi

## Mettilo in pratica

Prendi il modello di classificazione di sarcasmo di questo capitolo (Embedding + GlobalAveragePooling1D + Dense) e, senza guardare ancora il notebook di pratica della classe, rispondi per scritto a queste tre domande prima di toccare il codice:

1. Se sostituissi lo strato `GlobalAveragePooling1D` con uno strato `LSTM(64)`, cosa ti aspetti che cambi nella capacità del modello di rilevare il sarcasmo? Tieni in conto che il sarcasmo spesso dipende dall'ordine e dal contrasto tra parti della frase, non solo dalle parole sciolte
2. Se addestrassi quel stesso modello con LSTM durante molti epoch e la precisione di addestramento salisse molto più velocemente di quella di validazione, quale problema di quelli che già conosci dai capitoli precedenti stai vedendo?
3. Per il generatore di poesia, quale differenza pratica ti aspetti tra seminare il modello con una frase molto simile al corpus di addestramento (testi di canzoni irlandesi) rispetto a seminarlo con una frase completamente estranea, come una notizia di tecnologia?

Dopo, apri il notebook di pratica della classe e verifica le tue risposte eseguendo il codice.
