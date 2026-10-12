# 13. Applicazioni reali e progetto finale

## Perché questo argomento è importante

Hai percorso un cammino lungo: dal primo neurone fino a pipeline completi di produzione con TensorFlow Extended (TFX). Ma tutto quell'apprendimento può rimanere nella teoria se non lo connetti con problemi reali. Questo capitolo chiude il corso mostrandoti progetti concreti — molti sviluppati da studenti, piccoli team o persone autodidatte, non solo da grandi dipartimenti di ricerca — che usano esattamente le tecniche che già conosci: reti convoluzionali (CNN, Convolutional Neural Network) per classificare immagini, reti ricorrenti (RNN, Recurrent Neural Network) e memoria a lungo-corto plazo (LSTM, Long Short-Term Memory) per serie temporali, e TensorFlow Lite per portare un modello a un telefono mobile senza connessione.

Vedere questi casi ha un obiettivo molto concreto: che smetti di pensare "quale tecnica mi manca ancora da imparare?" e cominci a pensare "quale problema voglio risolvere con quello che già so?". Quella è esattamente la domanda che andrai a rispondere nel tuo progetto finale, la chiusura pratica di tutto il corso.

## Casi di uso reale di TensorFlow

I otto casi seguenti sono tratti dal materiale della certificazione EITC/AI/TFF (European IT Certificate — Certificato Europeo di Tecnologie dell'Informazione — in Intelligenza Artificiale, specialità TensorFlow Fundamentals). Ognuno risolve un problema diverso, in un settore diverso, con una combinazione di tecniche che hai visto già nelle classi precedenti.

### Air Cognizer: predire la qualità dell'aria con il mobile

A Delhi, l'inquinamento dell'aria è un problema di salute pubblica grave. Un gruppo di studenti di ingegneria ha sviluppato Air Cognizer, un'app che, a partire da una foto del cielo presa con il mobile, calcola l'Indice di Qualità dell'Aria (AQI, Air Quality Index) della zona. L'app combina tre modelli: una CNN che classifica se l'immagine contiene cielo, una LSTM che analizza pattern meteorologici storici come serie temporale, e un algoritmo di bosco casuale (RF, Random Forest) che fa la predizione finale di regressione sull'AQI a partire da tutte le variabili.

Quello che è interessante dal punto di vista tecnico è la distribuzione: i tre modelli si sono convertiti a TensorFlow Lite e sono rimasti ridotti a un binario di solo 10-20 KB, abbastanza leggero per funzionare in qualunque mobile senza connessione a internet né consumo rilevante di batteria. L'impatto è diretto: qualunque persona a Delhi può conoscere la qualità dell'aria del suo ambiente immediato e prendere decisioni (per esempio, evitare di fare esercizio all'aria aperta) senza dipendere da stazioni di misurazione ufficiali, che sono scarse.

### Tambua: ascoltare i polmoni per rilevare malattie respiratorie

Le malattie respiratorie — polmonite, asma, tubercolosi polmonare — causano più di 2,5 milioni di morti all'anno, molte per diagnosi tardiva o errata in zone con pochi risorse mediche, come l'Africa subsahariana. L'app Tambua attacca questo problema con un approccio insolito: converte il suono dei polmoni, catturato con uno stetoscopio digitale, in uno spettrogramma (una rappresentazione visuale delle frequenze del suono nel tempo) e addestra una CNN per riconoscere pattern associati a ogni malattia a partire da quell'immagine.

Il team ha addestrato il modello con dati di 621 pazienti reali, in collaborazione con cliniche e specialisti. Proprio come nel caso precedente, il modello si distribuisce con TensorFlow Lite per funzionare senza connessione: oggi 216 centri di salute, molti di loro cliniche rurali, usano l'app come supporto alla diagnosi. L'impatto non è sostituire il personale medico, ma dargli uno strumento veloce ed economico che riduce diagnosi sbagliate e evita prove non necessarie, come radiografie.

### Predire fenomeni meteorologici estremi con supercomputing

Ogni giorno si generano circa 100 terabyte di dati climatici provenienti da satelliti, sensori e modelli numerici. Ricercatori del NERSC (National Energy Research Scientific Computing Center, negli Stati Uniti) usano TensorFlow per addestrare modelli di apprendimento profondo che rilevano uragani, tempeste severe e altri fenomeni estremi a partire da immagini di satellite, combinando CNN per l'analisi spaziale di quelle immagini con RNN per catturare l'evoluzione temporale dei pattern meteorologici.

La particolarità di questo caso è la scala: addestrare questi modelli richiede computazione di alti rendimenti (HPC, High-Performance Computing) del livello del supercomputer Summit, capace di migliaia di miliardi di operazioni per secondo, e il team ha verificato che TensorFlow scala bene anche distribuendo l'addestramento tra migliaia di nodi di calcolo. L'impatto sono sistemi di allerta anticipata più precisi, che danno alle autorità più tempo per preparare la risposta di fronte a una catastrofe.

### Aiutare i paleografi a trascrivere manoscritti medievali

Trascrivere manoscritti medievali a mano è un lavoro lentissimo e molto specializzato: la lettera cambia da uno scriba all'altro, l'inchiostro è sbiadito e la lingua è arcaica. Un team di ricerca ha addestrato una CNN di classificazione multiclasse per riconoscere caratteri scritti a mano individuali, raggiungendo una precisione del 95% in media per carattere. La sfida non è stata solo tecnica ma di dati: poiché non esistono banche di immagini di manoscritti medievali come le hay di gatti o cani, il team ha costruito un'app web propria per raccogliere immagini mediante collaborazione cittadina (crowdsourcing), con studenti di istituto superiore contribuendo a etichettare il materiale.

L'impatto è diretto sulla ricerca storica: quello che prima esigeva team interi di paleografi lavorando durante anni, adesso si accelera enormemente, e un volume di conoscenza storica fino ad ora praticamente inaccessibile può consultarsi in molto meno tempo. È un buon esempio che l'apprendimento automatico (ML) non serve solo per affari o salute: anche accelera le discipline umanistiche.

### Airbnb: mettere ordine in centinaia di milioni di foto di alloggi

Non tutti i casi di uso di TensorFlow nascono da un'emergenza sociale: Airbnb aveva più di cinque milioni di alloggi in 81.000 città e centinaia di milioni di foto, molte male etichettate o ripetute (diverse foto dello stesso salotto e nessuna del bagno). Airbnb ha addestrato una CNN basata sull'architettura ResNet50 per classificare automaticamente ogni foto per tipo di stanza (camera, cucina, salotto, piscina...), usando la sua propria piattaforma di ML (Bighead) per gestire l'addestramento distribuito in GPU e la distribuzione con TensorFlow Serving.

Elaborare più di cinquecento milioni di immagini a mano avrebbe tardato mesi; con TensorFlow, il team ha avuto un modello funzionante in pochi giorni. L'impatto è puramente di prodotto: gli ospiti vedono prima le foto più rilevanti di ogni alloggio (salotto, camera) invece di foto poco attrattive del garage, il che migliora direttamente l'esperienza di ricerca e prenotazione. Questo caso dimostra che la classificazione di immagini con CNN — la stessa tecnica della classe 2 di questo corso — ha applicazione in qualunque negozio con grandi volumi di contenuto visuale.

### Rilevare il bruco cogollero nelle colture dell'Uganda

Il bruco cogollero (fall armyworm) è una piaga che può rovinare un raccolto intero e, senza controllo, aggravare la fame in zone agricole dell'Africa. Nazirini Siraji, una sviluppatrice di software ugandese, ha imparato TensorFlow in un Google Study Jam e, insieme al suo team, ha costruito un'app Android che usa una CNN per identificare l'infestazione a partire da foto delle piante prese con il mobile, suggerendo inoltre il trattamento adatto secondo il ciclo di vita della piaga.

L'impatto arriva direttamente all'agricoltore: rileva il problema prima che sia troppo tardi, senza dipendere dalla disponibilità di un esperto agrario, e riduce l'uso eccessivo di pesticidi al indicare il trattamento corretto nel momento corretto. È anche un buon ricordo che non serve un team grande né risorse di un'azienda grande per costruire con TensorFlow qualcosa con impatto reale: basta identificare bene il problema e applicare la tecnica adatta.

### Localizzare buche nelle strade di Los Angeles

Ispezionare manualmente tutta la rete stradale di una città così estesa come Los Angeles per localizzare buche è lento e costoso. Due studenti universitari, Alejandra Vasquez ed Ericson Hernandez, hanno attrezzato un auto con una fotocamera, hanno guidato per diverse strade e autostrade della città per raccogliere il loro proprio insieme di dati, e hanno addestrato con TensorFlow una CNN capace di identificare buche e crepe nel pavimento con alta precisione a partire da quelle immagini.

L'impatto è operativo: le squadre di manutenzione possono dedicare il loro tempo a ripararle invece che cercarle, e la città smette di dipendere unicamente dagli avvisi cittadini per sapere dove intervenire. È un esempio chiaro di come, con una fotocamera e voglia di raccogliere i tuoi propri dati, puoi applicare la classificazione di immagini a un problema di mobilità urbana che nessuno ti ha "preparato" in un dataset già esistente.

### Aiutare a salvare le api del mondo

Le api impollinano buona parte degli alimenti che consumiamo, ma la loro popolazione da anni è in declino per ragioni che non si capiscono completamente. Un progetto di conservazione ha installato fotocamere all'entrata degli alveari per contare quante api escono e quante ritornano, addestrando modelli di visione per computer con TensorFlow per analizzare quel video in automatico e rilevare discrepanze che puntino a un problema nell'ecosistema vicino. Lo stesso approccio si estende all'analisi di audio: le api si comunicano mediante ronzii che codificano informazione su dove c'è cibo, e l'apprendimento automatico può aiutare a interpretare quei pattern sonori.

I dati generati si condividono con persone esperte in biodiversità, che li usano per decidere, per esempio, quando tagliare l'erba di una zona o dove piantare alberi e fiori che favoricciano gli impollinatori. L'impatto non è che la tecnologia "salvi" le api da sola — la stessa documentazione del progetto insiste che l'intervento umano rimane imprescindibile — , ma che dà a chi prende decisioni sul territorio informazione che prima non aveva.

### E ci sono più casi da cui scegliere

Il materiale originale della certificazione include 16 casi d'uso completi, non solo questi otto. Se vuoi continuare a esplorare prima di definire il tuo progetto finale, nello stesso materiale troverai, tra altri, come TensorFlow aiuta il personale di Medici Senza Frontiere a prescrivere antibiotici a partire da foto di pietre di laboratorio, come si predicono inondazioni e grandi incendi forestali, come si rastreggiano asteroidi, e come un'app di ballo usa visione per computer per correggere la postura di chi la usa. Nessuna tecnica di quella lista esce da quello che già conosci: è sempre una combinazione di CNN, RNN/LSTM, modelli tabulari o distribuzione con TensorFlow Lite, applicata con criterio a un problema concreto.

## Progetta il tuo progetto finale: guida passo per passo

Il tuo progetto finale non necessita di risolvere un problema così ambizioso come i precedenti. Necessita di dimostrare che sai scegliere una tecnica adatta, applicarla con rigore e comunicare il risultato. Segui questi passi.

### Passo 1: scegli un problema e un dataset

Inizia dal problema, non dalla tecnica. Chiediti: cosa voglio predire o classificare, e perché mi importa? Può essere un problema serio (salute, ambiente, il tuo proprio settore professionale) o semplicemente qualcosa che ti risulti curioso. Hai due percorsi per il dataset:

- Riutilizzare uno di quelli che hai già usato nel corso (Fashion MNIST, recensioni di film di IMDB, il CSV di passeggeri del Titanic, il dataset di pinguini delle classi di TFX) per concentrarti sull'approfondire la tecnica invece che su cercare dati
- Cercare un dataset proprio in fonti aperte come Kaggle, TensorFlow Datasets, portali di dati aperti di amministrazioni pubbliche o dati che generi tu stesso (foto proprie, un CSV che costruisci a mano)

Se scegli dati propri, verifica prima di nulla che hai esempi sufficienti per categoria e che gli attributi sono affidabili: un dataset piccolo o male etichettato limita molto quello che puoi conseguire, per buona che sia la tecnica.

### Passo 2: scegli la tecnica del corso che andrai a applicare

Scegli una tecnica principale a seconda del tipo di dato di input, non il contrario:

- Dati tabulari (numeri e categorie in righe e colonne): rete densa, classificazione o regressione, come nella classe 3
- Immagini: rete convoluzionale (CNN), come nella classe 2
- Testo: tokenizzazione e sequenziazione seguita da una RNN o LSTM, come nelle classi 6 e 7
- Serie temporali: RNN o LSTM, uguale che nella predizione meteorologica che hai visto in questo capitolo

Combina quella tecnica principale con almeno una buona pratica della classe 4 (rilevare sovraddattamento con le curve di addestramento, applicare dropout o parada anticipata) per dimostrare che non solo addestri un modello, ma che valuti se è affidabile.

### Passo 3: struttura il consegnabile

Il tuo consegnabile ha due pezzi:

- **Un notebook** con il pipeline completo: carica e esplorazione del dataset, preprocessing, definizione del modello, addestramento, valutazione con metriche e curve di perdita/precisione, e salvataggio del modello addestrato. Se ha senso per il tuo caso, aggiungi una cella finale che converta il modello a TensorFlow Lite, come nella classe 8
- **Una memoria breve** (una o due pagine) che spieghi in linguaggio chiaro: quale problema risolvi, quale dataset hai usato e i suoi limiti, quale modello hai scelto e perché, quali risultati hai ottenuto e cosa faresti diverso con più tempo

### Passo 4: presenta e difendi il tuo progetto

Prepara una presentazione di cinque a dieci minuti per il resto del gruppo:

1. Poni il problema in una frase, senza gergo tecnico
2. Mostra il dataset e una o due cifre che lo descrivano (dimensione, numero di categorie)
3. Spiega l'architettura del modello con uno schema semplice, non con il codice completo
4. Mostra i risultati, compresa almeno una grafica (curva di addestramento, matrice di confusione, esempi di predizione)
5. Chiudi con i limiti del progetto e quali passi seguiresti se continuassi

Un termine orientativo ragionevole per tutto il progetto è da una a due settimane di lavoro autonomo dopo questa classe, con una sessione finale del corso dedicata alle presentazioni. I criteri che si andranno a valutare sono: che il problema sia ben posto e giustificato, che la tecnica scelta sia coerente con il tipo di dato, che valuti onestamente il rendimento del modello (compreso il sovraddattamento), che il notebook sia riproducibile da principio a fine, e che la presentazione si capisca senza bisogno di leggere il codice.

## Riassunto dei punti chiave

- I casi reali di TensorFlow che hai visto — qualità dell'aria, diagnosi respiratoria, meteorologia estrema, paleografia, classificazione di foto di alloggi, plagas agricole, buche urbane e conservazione di api — usano le stesse tecniche del corso: CNN, RNN/LSTM, modelli tabulari e TensorFlow Lite per distribuzione
- Nessuno di questi progetti è partito da risorse illimitate: molti li hanno sviluppati studenti o team piccoli con un problema chiaro e dati propri raccolti con sforzo
- Il materiale originale della certificazione include 16 casi di uso completi; gli otto di questo capitolo sono un campione rappresentativo per settori
- Il tuo progetto finale si costruisce in quattro passi: scegliere problema e dataset, scegliere la tecnica adatta al tipo di dato, strutturare notebook più memoria, e presentare difendendo le tue decisioni
- I criteri di valutazione premiano il rigore (valutare il sovraddattamento, un notebook riproducibile) tanto come il risultato finale

## Mettilo in pratica

Prima della prossima sessione, scrivi in una sola frase il problema che vuoi risolvere nel tuo progetto finale e, in un'altra frase, quale dataset andrai a usare (uno del corso o uno nuovo) e quale tecnica delle viste in classe encaja meglio. Condividile con il gruppo: è la prima versione della memoria che andrai a consegnare, e ricevere una prima reazione adesso ti fa risparmiare tempo più avanti. Dopo, apri il notebook di pratica di questa classe e inizia l'impalcatura del progetto: il modo migliore di perdere la paura del progetto finale è avere già una cella di codice che funziona.
