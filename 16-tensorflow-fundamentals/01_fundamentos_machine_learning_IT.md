# 01. Fondamenti di Machine Learning

## Perché questo argomento è importante

Prima di scrivere una sola riga di codice con TensorFlow (TF), devi capire quale problema risolve. TensorFlow è una libreria di codice aperto sviluppata da Google che serve a costruire e addestrare modelli di Machine Learning (ML), cioè di Apprendimento Automatico. Se non hai chiaro cosa significa "imparare" per una macchina, nessun esempio di codice ti risulterà intuitivo. Questo capitolo ti fornisce questa base concettuale: la differenza tra programmare e addestrare, che cosa è un modello e perché serve uno strumento come TensorFlow per tutto questo.

## Programmazione tradizionale contro Machine Learning

Immagina che tu voglia che un computer riconosca se una foto mostra sasso, carta o forbice (il gioco "Rock, Paper, Scissors"). Per una persona è banale, ma programmarlo con regole esplicite è un problema enorme: ogni persona ha mani diverse, tonalità di pelle diversa e modi propri di fare il gesto delle forbici. Se dovessi descrivere con il codice, condizione per condizione, quale combinazione esatta di pixel rappresenta ogni gesto, avresti bisogno di migliaia di righe e comunque sbaglieresti con casi che non hai previsto.

Questa difficoltà rivela la differenza essenziale tra i due paradigmi:

- **Programmazione tradizionale**: tu scrivi le regole (il codice) e il computer applica quelle regole ai dati di input per produrre un risultato. Regole + dati → risultato
- **Machine Learning**: dai al computer i dati di input insieme ai risultati corretti (le risposte) e lasci che il sistema stesso deduca le regole. Dati + risultato → regole

In altre parole, ML rovescia il processo usuale di programmazione. Invece di anticipare tu tutti i casi possibili, addestri un modello con molti esempi —molte foto di sasso, carta e forbice già etichettate— e lasci che trovi i pattern che distinguono un gesto dall'altro. Questo è esattamente l'approccio che userai nelle lezioni successive per costruire un vero classificatore di immagini.

## Un esempio più semplice di quello delle mani

Riconoscere gesti con la mano è un problema complesso per iniziare. Per questo, prima di arrivarci, è bene esercitarsi con qualcosa di molto più semplice: un insieme di numeri con una relazione nascosta tra loro.

Guarda questa tabella di valori di X e Y:

| X | -1 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| Y | -3 | -1 | 1 | 3 | 5 | 7 |

Se osservi il pattern, puoi dedurre che la relazione è Y = 2X - 1. Un programmatore tradizionale scriverebbe direttamente quella formula nel codice. Nel Machine Learning, invece, consegneresti al computer questi coppie di numeri senza dirgli la formula, e il sistema —tramite un processo di prova e aggiustamento— la dedurebbe da solo. Questo è l'esempio che riprenderai con codice vero nel capitolo successivo.

## Che cosa è un modello e che cosa significa addestrarlo

Un **modello** è la rappresentazione matematica che il sistema costruisce per catturare la relazione tra i dati di input (feature o caratteristiche) e i dati di output (etichette o risultati attesi). All'inizio, un modello non sa niente: i suoi parametri interni di solito si inizializzano con valori casuali, quindi le sue prime previsioni sono praticamente indovinate. Per esempio, di fronte alla relazione reale Y = 2X - 1, il modello potrebbe inizialmente supporre qualcosa di così distante come Y = 5X + 5.

**Addestrare** un modello è il processo di aggiustare quei parametri interni, passo dopo passo, affinché le sue previsioni si avvicinino sempre di più ai risultati reali. Questo processo si basa su due elementi chiave che vedrai in dettaglio nel capitolo successivo:

- Una **funzione di perdita (loss)**, che misura numericamente quanto il modello si sbaglia ad ogni tentativo
- Un **ottimizzatore**, che usa quella misura di errore per proporre un aggiustamento migliore al giro successivo

Questo ciclo —predire, misurare l'errore, aggiustare— si ripete molte volte. Ogni passaggio completo sul set di dati di addestramento si chiama **epoca (epoch)**. Col tempo, il modello converge verso una formula che spiega i dati in modo ragionevole, anche se —e questo è importante— quasi mai in modo perfettamente esatto.

## Che cosa sono i dati di addestramento

I **dati di addestramento** sono l'insieme di esempi che usi perché il modello impari: nel caso dei numeri, sono le coppie di valori X e Y; nel caso di sasso-carta-forbice, sarebbero migliaia di foto già etichettate con il gesto che rappresentano. Quanti più esempi rappresentativi hai, più affidabile sarà il pattern che il modello può estrarre. Con solo sei coppie di numeri, come nell'esempio precedente, il modello può avvicinarsi molto alla formula reale, ma non ha modo di esserne assolutamente certo.

Questo spiega un comportamento che troverai costantemente nel Machine Learning: le previsioni di solito non sono esatte, ma approssimazioni con un grado di incertezza. Se addestri il modello dell'esempio precedente e gli chiedi di predire Y per X = 10, è probabile che ottengas qualcosa come 18,9998 invece di 19 esatto. Non è un errore del sistema: è la conseguenza logica di imparare da un campione limitato di dati e di lavorare con probabilità invece che con certezze assolute.

## Perché TensorFlow è utile per tutto questo

Potresti, in teoria, programmare tu stesso il ciclo di predire-misurare-aggiustare. Ma farlo a mano —calcolando derivate, gestendo matrici di parametri, distribuendo il calcolo tra CPU (Central Processing Unit) e GPU (Graphics Processing Unit)— è lento e soggetto a errori. TensorFlow ti offre:

- Una struttura dati comune, il **tensore**, per rappresentare qualunque tipo di dato numerico (numeri, liste, immagini, testo convertito a numeri)
- Operazioni matematiche già ottimizzate per essere eseguite in parallelo, sia su CPU che su GPU
- Un'interfaccia di alto livello, **Keras**, che ti permette di definire un modello impilando strati in poche righe, senza che tu scrivi il calcolo dei gradienti o il ciclo di addestramento

Grazie a questo, lo stesso pattern concettuale che hai appena visto con i numeri —dati + risultato → regole, aggiustate mediante perdita e ottimizzatore— si applica altrettanto bene a problemi molto più complessi: riconoscere cifre scritte a mano, classificare immagini mediche o analizzare il sentiment di un testo. Nel capitolo successivo scriverai il tuo primo codice con TensorFlow e Keras per mettere in pratica esattamente questo ciclo.

## Riassunto dei punti chiave

- La programmazione tradizionale parte da regole che tu definisci; il Machine Learning parte da dati e risultati, e deduce le regole
- Un modello è una rappresentazione matematica che relaziona input e output, e che inizia con parametri praticamente casuali
- Addestrare consiste nel ripetere il ciclo predire → misurare l'errore con una funzione di perdita → aggiustare con un ottimizzatore, epoca dopo epoca
- I dati di addestramento sono gli esempi con cui il modello impara; quanti più e più rappresentativi, meglio il modello generalizza
- Le previsioni di un modello sono approssimazioni probabili, non certezze esatte, perché si basano su un campione limitato di dati
- TensorFlow e Keras ti danno la struttura (tensori) e gli strumenti (strati, funzioni di perdita, ottimizzatori) per applicare questo ciclo senza programmarlo da zero

## Mettilo in pratica

Prima di aprire un notebook, prendi carta e penna. Scrivi questa tabella di valori:

| X | -2 | -1 | 0 | 1 | 2 |
|---|---|---|---|---|---|
| Y | -6 | -3 | 0 | 3 | 6 |

Deduci a mano la relazione tra X e Y e annota la formula che credi la spieghi. Poi, immagina di essere l'"ottimizzatore": la tua prima supposizione è Y = X + 1. Calcola quale errore (la differenza tra la tua previsione e il valore reale di Y) faresti per ogni X della tabella, e annota come cambieresti la tua formula per ridurre quell'errore al giro successivo. Quell'aggiustamento manuale, ripetuto molte volte e in modo automatico, è esattamente quello che fa TensorFlow quando addestra un modello.
