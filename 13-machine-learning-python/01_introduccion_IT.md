# 1. Introduzione al machine learning con Python

## Perché questo ti riguarda

Immagina di voler insegnare a un computer a distinguere tra un'email normale e una di spam, oppure a prevedere il prezzo di una casa in base alle sue dimensioni e alla sua posizione. Non puoi scrivere una regola fissa per ogni caso possibile: ci sono troppe combinazioni ed eccezioni. Qui entra in gioco l'apprendimento automatico (ML, dall'inglese Machine Learning), un ramo dell'intelligenza artificiale (IA) che consente a un programma di imparare modelli a partire dai dati, anziché seguire istruzioni programmate passo dopo passo per ogni situazione.

Questo manuale ti accompagna da zero. Non devi sapere algebra lineare né aver programmato prima in progetti complessi. Ti serve solo curiosità e voglia di praticare. Nel corso di dodici capitoli costruirai, passo dopo passo, la tua competenza personale su come funziona il ML e come applicarlo con Python, uno dei linguaggi di programmazione più usati in questo settore grazie alla sua semplicità e alle sue librerie specializzate.

Imparare questo ti riguarda perché il ML è già parte di decisioni quotidiane: quale contenuto vedi in un'app, se una banca ti concede un prestito, come si diagnosticano le malattie da immagini mediche o come un'azienda agroalimentare rileva i parassiti nelle sue colture. Comprendere i fondamenti ti dà una base solida per partecipare a progetti di digitalizzazione, valutare gli strumenti di IA con criterio proprio e, se lo desideri, costruire i tuoi modelli.

## Cos'è il machine learning (con un'analogia quotidiana)

Nel 1959, Arthur Samuel ha definito il ML come il campo di studio che conferisce alle macchine la capacità di imparare senza programmarle esplicitamente per ogni compito. Questa idea si sviluppa da più di sessanta anni ed è ora alla portata del tuo computer personale.

Pensa a come un bambino o una bambina impara a riconoscere i cani. Nessuno gli consegna una lista di regole esatte ("quattro zampe, coda, pelo, abbaia"). Invece, vede molti esempi di cani e altri animali, e col tempo generalizza un modello che gli permette di riconoscere un cane che non aveva mai visto prima. Il ML funziona in modo simile: mostri al modello molti esempi (dati), e il modello regola i suoi parametri interni fino a riuscire a generalizzare un modello utile per prevedere o classificare nuovi casi.

Esistono tre grandi tipi di apprendimento:

- Apprendimento supervisionato: il modello impara a partire da esempi etichettati, cioè casi dove conosci già la risposta corretta. Per esempio, foto di frutti già classificate come "maturo" o "acerbo"
- Apprendimento non supervisionato: il modello cerca modelli o strutture in dati che non hanno etichette, come raggruppare clienti con abitudini di acquisto simili senza dire in anticipo quali gruppi esistono
- Apprendimento per rinforzo: il modello impara interagendo con un ambiente e ricevendo un segnale di ricompensa, regolando il suo comportamento per massimizzarla

Un errore frequente è pensare che il ML consista nel programmare regole molto dettagliate manualmente. È l'esatto opposto: il vantaggio del ML è che l'algoritmo stesso scopre quelle regole dai dati, senza che tu gliele detti una per una.

## Come funziona un progetto di machine learning in Python

Quasi ogni progetto di ML segue un flusso di lavoro simile, indipendentemente dall'algoritmo che scegli. Conoscere questo flusso sin dall'inizio ti aiuta a non perderti quando arriveremo ai capitoli più tecnici.

**Primo passo: preprocessare i dati.** I dati del mondo reale quasi mai arrivano puliti. Possono esserci valori mancanti, testo dove ti aspetti numeri o scale molto diverse tra colonne (per esempio, età in anni e stipendio in migliaia di euro). In questo passo pulisci, trasformi e normalizzi i dati così che l'algoritmo possa lavorare con essi correttamente.

**Secondo passo: scegliere l'algoritmo.** A seconda del tipo di problema che hai (prevedere un numero, classificare in categorie o raggruppare elementi simili), scegli un algoritmo appropriato. In questo corso lavorerai con molti di essi: regressione, k vicini più prossimi (KNN, dall'inglese K-Nearest Neighbors), macchine a vettori di supporto (SVM, dall'inglese Support Vector Machines) e algoritmi di agglomeramento.

**Terzo passo: addestrare il modello.** Dividi i tuoi dati in un insieme di addestramento e uno di test. Il modello impara solo con l'insieme di addestramento, come chi studia pratica con esercizi prima dell'esame.

**Quarto passo: valutare le prestazioni.** Usi l'insieme di test, che il modello non ha mai visto prima, per verificare quanto bene generalizza quello che ha imparato. Esistono metriche come l'accuratezza, la precisione, il richiamo (recall) o il punteggio F1 per misurare questo in modo oggettivo.

**Quinto passo: prevedere con dati nuovi.** Una volta che hai fiducia nel modello, lo usi per fare previsioni su dati che non ha mai visto: un cliente nuovo, un'immagine nuova, una misurazione nuova.

Per tutto questo flusso, Python ha librerie specializzate che ti risparmiano la necessità di scrivere codice da zero:

- **pandas**: organizza e manipola i dati in tabelle, in modo simile a un foglio di calcolo ma programmabile
- **NumPy**: gestisce calcoli numerici e strutture di dati matematiche in modo efficiente
- **scikit-learn**: offre implementazioni già costruite degli algoritmi di ML più comuni, insieme a strumenti per dividere i dati, addestrare modelli e valutarli
- **Matplotlib**: genera grafici e visualizzazioni per capire i tuoi dati e i tuoi risultati

Prima di scrivere una sola riga di codice di ML, devi installare queste librerie. Si installano con pip, il gestore di pacchetti di Python, dal tuo terminale:

```bash
# Installa le librerie principali che userai in tutto il corso
pip install pandas numpy scikit-learn matplotlib
```

Una volta installate, le importi all'inizio di qualsiasi script o notebook:

```python
# Importiamo le librerie principali del corso
import pandas as pd          # per caricare e organizzare dati in tabelle
import numpy as np           # per calcoli numerici
from sklearn.model_selection import train_test_split  # per dividere dati
import matplotlib.pyplot as plt  # per visualizzare risultati

# Controllo rapido: se non ci sono errori, tutto è pronto
print("Librerie caricate correttamente")
```

Questo piccolo blocco di codice non esegue ancora nessun calcolo di ML, ma è il punto di partenza di praticamente qualsiasi progetto che costruirai in questo manuale. Se lo esegui e vedi il messaggio stampato senza errori, il tuo ambiente è pronto per avanzare.

## La mappa di questo corso

Questo manuale organizza dodici capitoli in cinque blocchi tematici. Ogni algoritmo che studierai segue tre passaggi: prima la teoria (l'intuizione di come funziona), poi l'applicazione pratica con scikit-learn (usarlo già costruito su dati reali) e infine l'implementazione da zero (programmare l'algoritmo tu stesso, con la matematica inclusa, per comprendere a fondo il suo funzionamento interno). Questo approccio in tre passaggi è la colonna vertebrale di tutto il corso.

- **Blocco uno, introduzione (capitolo uno):** questo capitolo che stai leggendo, dove collocherai il ML all'interno dell'IA e conoscerai il flusso di lavoro tipico con Python
- **Blocco due, regressione (capitoli da due a quattro):** applichi la regressione lineare con scikit-learn, impari a salvare e scalare modelli e interpretare i loro risultati, e infine programmi la regressione lineare da zero
- **Blocco tre, classificazione con KNN (capitoli cinque e sei):** applichi l'algoritmo dei k vicini più prossimi con scikit-learn e poi lo implementi da zero
- **Blocco quattro, SVM (capitoli da sette a nove):** studi i fondamenti delle macchine a vettori di supporto, le programmi da zero e approfondisci concetti come i kernel e il margine blando
- **Blocco cinque, agglomeramento e conclusione (capitoli da dieci a dodici):** lavori con gli algoritmi di agglomeramento K-Means e Mean Shift, e chiudi il corso con casi d'uso reali che integrano tutto quello che hai imparato

Una nota importante: questo corso non copre le reti neurali né l'apprendimento profondo (deep learning). Questo è un altro ramo del ML, con il suo approccio e la sua complessità propri, e merita un corso dedicato. Qui ti concentrerai su algoritmi più semplici da comprendere matematicamente, che ti danno una base solida prima di fare quel salto in futuro se lo desideri.

Vale la pena anche un po' di contesto storico. Vladimir Vapnik ha introdotto la SVM nel 1963, ma l'algoritmo non si è popularizzato fino agli anni novanta, quando ha superato le reti neurali nelle attività di riconoscimento dei caratteri scritti a mano. Le SVM hanno dominato il campo del ML per anni, fino al recente rilancio delle reti neurali con il deep learning. Se senti di arrivare tardi in questo campo, non è così: la potenza di calcolo e l'accesso a strumenti che hai oggi superano di gran lunga quello che esisteva solo pochi decenni fa.

## Riepilogo: cosa porti con te da questo capitolo

- Il machine learning è un ramo dell'intelligenza artificiale che consente a un programma di imparare modelli dai dati, senza programmare ogni regola a mano
- Esistono tre tipi principali di apprendimento: supervisionato, non supervisionato e per rinforzo
- Un progetto di ML segue un flusso tipico: preprocessare i dati, scegliere l'algoritmo, addestrare, valutare e prevedere
- Python è un linguaggio ideale per il ML grazie a librerie come pandas, NumPy, scikit-learn e Matplotlib
- Ogni algoritmo di questo corso si spiega in tre passaggi: teoria, applicazione con scikit-learn e implementazione da zero
- Questo corso copre regressione, KNN, SVM e agglomeramento, ma non include reti neurali né deep learning

## Mettilo in pratica

**Prepara il tuo ambiente proprio adesso.** Prima di avanzare al capitolo due, verifica di avere Python 3 installato sul tuo computer e di poter installare e caricare le librerie che userai in tutto il corso.

1. Apri il tuo terminale ed esegui `python --version` per confermare che hai Python 3 installato
2. Installa le librerie con il comando `pip install pandas numpy scikit-learn matplotlib`
3. Crea un file chiamato `prueba_entorno.py` e copia il blocco di codice di importazione che hai visto in questo capitolo
4. Esegui il file con `python prueba_entorno.py`
5. **Verifica** che vedi il messaggio "Librerie caricate correttamente" senza alcun errore sullo schermo

Se completi questi cinque passaggi senza errori, il tuo ambiente è pronto per il blocco di regressione, che inizia nel prossimo capitolo.

**Sigle usate in questo capitolo:** ML (Machine Learning, apprendimento automatico), IA (intelligenza artificiale), KNN (K-Nearest Neighbors, k vicini più prossimi), SVM (Support Vector Machines, macchine a vettori di supporto)
