# 12. Casi d'uso reali e conclusione del corso

Hai raggiunto la fine del manuale. Nel corso di undici capitoli sei passato dalla teoria alla pratica e dalla pratica al codice da zero: regressione, K vicini più prossimi (KNN, dall'inglese K-Nearest Neighbors), macchine a vettori di supporto (SVM, dall'inglese Support Vector Machines), e agglomeramento con K-Means e Mean Shift. In questo capitolo chiudi il cerchio: vedi dove vive ogni algoritmo fuori dall'aula e decidi verso dove vuoi continuare.

## Perché questo ti riguarda

Imparare un algoritmo in astratto è utile, ma ha senso solo quando lo colleghi a un problema reale. L'apprendimento automatico (ML, dall'inglese Machine Learning) non è una collezione di formule: è una scatola di strumenti che già prende decisioni ogni giorno in banche, ospedali, auto e reti sociali. Se capisci quale algoritmo si adatta con quale tipo di domanda, smetti di vedere il ML come magia e cominci a vederlo come uno strumento più del tuo lavoro, altrettanto concreto quanto un foglio di calcolo o una banca dati.

Questa conclusione non aggiunge algoritmi nuovi. Ti aiuta a guardare indietro, riconoscere il modello dietro ogni tecnica e applicarlo mentalmente ai problemi che già conosci, siano del settore che siano.

## Quello che già sai fare: una rassegna rapida della tua scatola di strumenti

Prima di vedere esempi, esamina cosa risolve ogni algoritmo:

- **Regressione**: predice un valore numerico continuo a partire da altre variabili (quanto costerà qualcosa, quanto pioverà, quante unità si venderanno)
- **KNN**: classifica un caso nuovo guardando quali sono i suoi vicini più simili tra i dati già conosciuti
- **SVM**: traccia il confine che meglio separa due o più categorie, anche quando i dati non sono facilmente separabili
- **K-Means**: raggruppa dati senza etichettare in un numero di gruppi che decidi in anticipo
- **Mean Shift**: raggruppa dati senza etichettare senza che tu debba fissare quanti gruppi esistono; lo stesso algoritmo li scopre

Con la serie di prezzi simulata dei capitoli da due a quattro hai praticato la regressione. Con la diagnosi del cancro al seno dei capitoli da cinque a nove hai visto KNN e SVM in azione su un problema di classificazione reale. E con i dati di sopravvivenza del Titanic dei capitoli dieci e undici hai applicato K-Means e Mean Shift per scoprire gruppi di passeggeri senza usare alcuna etichetta. Ogni dataset del manuale è, in realtà, una versione semplificata di un problema che è già risolto così fuori dall'aula.

## Casi d'uso reali di regressione

La regressione è probabilmente la tecnica di ML più diffusa perché risponde a una domanda che appare in quasi qualsiasi affari: quanto andrà a passare?

- **Previsione di prezzi e domanda**: le aziende di tutti i settori usano la regressione per stimare il prezzo futuro di un prodotto o la quantità che venderanno, in modo simile a come hai stimato una serie di prezzi simulata nei capitoli da due a quattro. Questa stessa logica serve a prevedere il prezzo delle azioni nei mercati finanziari
- **Previsione meteorologica**: i servizi meteorologici combinano variabili come temperatura, pressione e umidità per prevedere quanto pioverà o quale temperatura farà; è uno dei classici esempi di regressione applicata alle serie temporali
- **Stima dei costi**: qualsiasi organizzazione che abbia bisogno di calcolare il costo previsto di un progetto, un trattamento o una campagna a partire da variabili conosciute può appoggiarsi a un modello di regressione addestrato con dati storici

## Casi d'uso reali di classificazione (KNN e SVM)

KNN e SVM rispondono a una domanda diversa: non quanto, bensì quale categoria. Entrambi gli algoritmi condividono applicazioni perché tutti e due separano dati in gruppi già conosciuti in anticipo.

- **Diagnosi medica**: lo stesso manuale usa la diagnosi del cancro al seno come esempio centrale nei capitoli da cinque a nove. È un'applicazione reale e molto diffusa: classificare un tumore come benigno o maligno a partire dalle misurazioni cellulari è esattamente il tipo di problema per il quale sono stati disegnati KNN e SVM
- **Rilevamento di email spam**: classificare un'email in ingresso come spam o non spam a partire dalle parole e dai modelli che contiene è una delle applicazioni più conosciute della classificazione automatica
- **Auto autonome**: distinguere se quello che c'è davanti al veicolo è un ostacolo senza importanza o un vero pericolo, come un pezzo di cartone sulla strada oppure una persona con una coperta, richiede una classificazione con una precisione molto superiore alla media; i sistemi di guida autonoma dipendono da decisioni di questo tipo
- **Riconoscimento della scrittura a mano**: le SVM si sono fatte conosciute negli anni novanta precisamente per superare le reti neurali dell'epoca nel riconoscimento dei caratteri manoscritti, un traguardo storico che ha spinto la loro adozione nell'industria

## Casi d'uso reali di agglomeramento (K-Means e Mean Shift)

L'agglomeramento risolve un problema diverso dai due precedenti: non hai etichette precedenti, solo dati, e vuoi scoprire quale struttura nascondono.

- **Segmentazione dei clienti**: le aziende raggruppano la loro clientela secondo le abitudini di acquisto o il comportamento per disegnare campagne o raccomandazioni diverse per ogni gruppo; è un'estensione diretta dell'esercizio di raccomandazione di prodotti che già abbiamo menzionato come applicazione tipica del ML
- **Raggruppamento di pazienti o di fattorie**: raggruppare i registri clinici con caratteristiche simili, o le fattorie agricole con condizioni di terreno e produzione simili, aiuta a disegnare protocolli o raccomandazioni adattate a ogni gruppo anziché applicare un'unica regola per tutti i casi; lo stesso principio che hai applicato al raggruppamento dei passeggeri del Titanic nei capitoli dieci e undici
- **Rilevamento di gruppi nelle reti sociali**: identificare comunità di utenti con interessi o interazioni simili dentro una rete sociale è un utilizzo frequente dell'agglomeramento, molto simile nella sua logica a Mean Shift quando non si sa in anticipo quante comunità esistono

## Verso dove andare dopo questo corso

Questo manuale copre cinque algoritmi classici, ma l'apprendimento automatico è un campo molto più ampio. Esistono altri rami, come le reti neurali e l'apprendimento profondo (deep learning), che risolvono problemi ancora più complessi: riconoscimento di immagini su larga scala, generazione di testo o traduzione automatica, tra gli altri. Queste tecniche rimangono fuori dal campo di questo corso, ma sono un passo naturale una volta domini i fondamenti che hai appena imparato. La potenza di calcolo attuale, accessibile da servizi nel cloud, rende questo prossimo passo oggi alla portata di qualsiasi persona con curiosità e perseveranza.

## Riepilogo: cosa porti con te da questo capitolo

- La regressione predice valori numerici: prezzi, domanda, previsione meteorologica o stima di costi
- KNN e SVM classificano i casi in categorie conosciute: diagnosi medica, spam, auto autonome o riconoscimento della scrittura
- K-Means e Mean Shift scoprono gruppi in dati senza etichettare: clienti, pazienti, fattorie o comunità
- Ogni dataset che hai usato nel manuale (prezzi simulati, cancro al seno, Titanic) è una versione d'apprendimento di un problema che è già risolto così nel mondo reale
- Scegliere l'algoritmo corretto dipende dalla domanda che vuoi rispondere, non dalla moda del momento
- Le reti neurali e l'apprendimento profondo sono il passo logico successivo, ma non l'unico: per primo padroneggia bene questi cinque fondamenti

## Mettilo in pratica

Pensa a un problema reale del tuo ambiente, sia lavorativo, accademico o personale. **Scegli** uno dei cinque algoritmi di questo manuale e **decidi** perché si adatta a quel problema. Per esempio, se lavori nel settore agroalimentare, potresti usare la regressione per prevedere il raccolto di una campagna a partire dai dati climatici; se lavori nella sanità, SVM o KNN per classificare i registri clinici; o se gestisci foreste, K-Means per segmentare le parcelle secondo le loro caratteristiche. **Scrivi** in due o tre frasi quali dati avresti bisogno e quale algoritmo useresti, e **salva** quell'idea: è il tuo primo progetto reale di apprendimento automatico.

**Sigle usate in questo capitolo:** ML (Machine Learning, apprendimento automatico), KNN (K-Nearest Neighbors, K vicini più prossimi), SVM (Support Vector Machines, macchine a vettori di supporto)
