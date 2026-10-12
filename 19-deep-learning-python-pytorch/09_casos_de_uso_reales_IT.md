# 09. Casi d'uso reali

## Perché questo argomento è importante

Hai percorso il percorso completo: tensor, dati e `DataLoader`, una rete
densa addestrata su MNIST, una rete neurale convoluzionale (CNN,
Convolutional Neural Network) addestrata su CIFAR-10, computazione su GPU
(Graphics Processing Unit) e, per ultimo, come analizzare un modello già
addestrato. Tutto quell'apprendimento può rimanere nella teoria se non lo colleghi
con problemi reali. Questo capitolo chiude il corso mostrandoti, a partire da
le proprie menzioni del materiale ufficiale della certificazione EITC/AI/DLPP
(European IT Certificate — Certificato Europeo di Tecnologie dell'Informazione — in Intelligenza Artificiale, specialità Deep Learning with
Python and PyTorch), in quali domini si applica ogni pezzo di PyTorch che già
conosci.

Non imparerai nessuna sintassi nuova. Vedrai che la classificazione di
immagini della lezione 6, il `torch.device` della lezione 7 e la matrice di
confusione della lezione 8 non sono esercizi isolati: sono esattamente i
pezzi che sostengono sistemi di visione artificiale, modelli di linguaggio
e deployments in produzione che già esistono fuori da questo corso. L'obiettivo
è che smetti di pensare "quale tecnica mi manca da imparare?" e inizi a
pensare "quale problema voglio risolvere con quel che già so?". Quella domanda è
esattamente quella che risponderai nel tuo progetto finale.

## Visione artificiale: da una cifra a un'auto che guida da sola

Lo stesso materiale della certificazione presenta la visione artificiale
come uno dei primi domini di successo dell'apprendimento profondo (deep
learning), insieme all'elaborazione del linguaggio naturale (NLP, Natural Language
Processing) e al riconoscimento vocale. Il pezzo che lo rende possibile in
PyTorch è `torchvision`, la libreria che hai già usato nella lezione 2 per
caricare MNIST e nella lezione 6 per addestrare la tua CNN su CIFAR-10: include
dataset di riferimento come ImageNet, trasformazioni per preprocessare
immagini e modelli preaddestrati pronti per adattare a un problema nuovo.

L'architettura convoluzionale che hai costruito nella lezione 5 non è solo un
esercizio accademico: lo stesso materiale descrive la CNN come l'architettura
che ha rivoluzionato la classificazione e il riconoscimento di immagini a
colore, proprio perché ogni canale RGB (Red-Green-Blue — Rosso-Verde-Blu)
si rappresenta naturalmente come una dimensione più del tensor di input,
senza trucchi aggiunti. Quella stessa tecnica, scalata e combinata con dataset
molto più grandi, è la base di sistemi di riconoscimento di oggetti in
foto, di diagnosi per immagine medica e della percezione visiva di
veicoli con guida autonoma.

Proprio sulla guida autonoma, il materiale insiste su una sfumatura che
hai già affrontato nella lezione 8: addestrare un buon modello non basta, devi
valutarlo in condizioni diverse da quelle di addestramento —illuminazione diversa, clima diverso, tipo diverso di strada— per sapere se di
fatto generalizza o se solo ha memorizzato le condizioni concrete del suo
insieme di dati. È la stessa logica dell'overfitting che hai visto con la
matrice di confusione di MNIST, applicata a un contesto dove un errore di
classificazione ha conseguenze molto più serie.

## Elaborazione del linguaggio naturale: da torchtext ai grandi modelli di linguaggio

Il secondo grande dominio che appare costantemente nel materiale è l'
elaborazione del linguaggio naturale. `torchtext`, la terza libreria dell'
ecosistema di PyTorch insieme a `torchvision` e `torchaudio`, offre
strumenti per caricare e preprocessare testo e costruire modelli di
linguaggio, seguendo lo stesso pattern di dataset e modello che già conosci dalla
lezione 2 e la lezione 3.

Il materiale illustra la scala che può raggiungere questo dominio con un
esempio concreto: GPT-3 (Generative Pre-trained Transformer 3), sviluppato
da OpenAI, è un modello basato sull'architettura transformer con 175.000
miliardi di parametri, capace di risolvere un'ampia varietà di compiti di
elaborazione del linguaggio naturale con alta precisione. È la stessa idea di
"parametri che la rete impara" che hai calcolato a mano nella lezione 3 per la tua
prima rete densa, solo che moltiplicata per un fattore quasi inimmaginabile. Il
materiale segnala anche che PyTorch migliora continuamente la sua integrazione
con Hugging Face Transformers, la libreria più usata oggi per lavorare con
modelli di linguaggio preaddestrati, il che ti permette di partire da un modello già
addestrato invece di addestrarne uno da zero.

## Calcolo a gran scala su GPU: perché uscire dal tuo portatile importa

La lezione 7 ti ha insegnato a muovere tensor e modelli a una GPU con
`torch.device` e a misurare la differenza di tempo rispetto alla CPU (Central
Processing Unit). Il materiale è esplicito su perché questa capacità
importa oltre l'esercizio di lezione: per problemi reali, una GPU di alta gamma smette di essere un lusso e diventa un requisito, perché il
volume di dati e la dimensione dei modelli superano completamente quel che una
CPU può elaborare in un tempo ragionevole.

L'esempio di GPT-3 torna a essere utile qui: addestrare un modello con 175.000
miliardi di parametri richiede risorse computazionali considerevoli, una
gestione di memoria attenta —con tecniche come il parallelismo di modelli— e
tempi di addestramento che sono viabili solo distribuendo il lavoro tra
molte GPU o unità di elaborazione tensore (TPU, Tensor Processing
Unit) a volte, un approccio di computazione ad alte prestazioni (HPC,
High-Performance Computing). Il materiale ricorda anche una decisione
pratica che già conosci: se non disponi di una GPU locale, puoi noleggiarla
nel cloud e pagare solo per le ore che la usi, esattamente lo stesso
ragionamento di costo che applicheresti in qualsiasi progetto professionale.

Vale la pena ricordare, inoltre, un dato di contesto che lo stesso materiale
menziona sulla origine della ferramenta che hai usato in tutto il corso:
PyTorch è stato sviluppato dal laboratorio di ricerca in intelligenza
artificiale di Facebook (FAIR, Facebook AI Research), oggi integrato in Meta.
È un fatto ampiamente documentato che aziende come Tesla hanno parlato
pubblicamente dell'uso di PyTorch nei loro sistemi di visione artificiale
per guida autonoma, e che OpenAI ha adottato PyTorch come il suo framework
principale per lo sviluppo di modelli di deep learning. Non sono casi isolati di un ambiente accademico:
è la stessa libreria, con la stessa sintassi di tensor, `nn.Module` e ciclo di addestramento che già domini.

## Da un modello addestrato a un prodotto: valutazione, interpretabilità e deployment

La lezione 8 ti ha insegnato a distinguere overfitting da underfitting, a leggere una
matrice di confusione e a conservare i pesi di un modello con `torch.save`.
Il materiale amplia quella idea con tecniche di interpretabilità che risultano
imprescindibili quando un modello passa da un notebook a un'applicazione reale:
visualizzare quali filtri ha imparato una CNN, analizzare i meccanismi di attenzione
di un modello di linguaggio per capire su quali parole si sofferma, generare
mappe di rilevanza (saliency maps) che segnalano quale regione di un'immagine
ha influenzato più una previsione, o stimare l'incertezza di un modello con
tecniche come il Monte Carlo Dropout. Tutte rispondono alla stessa
domanda che ti sei fatto nella lezione 8 al ispezionare previsioni
errate: non basta sapere che il modello sbaglia, bisogna capire
perché.

Conservare i pesi di un modello con `torch.save`, come hai imparato nella
lezione 8, è solo il primo passo verso la produzione. Il materiale descrive
molti strumenti che danno il passo prossimo: `torch.quantization` riduce
la dimensione di un modello e accelera la sua inferenza, qualcosa specialmente utile per
dispositivi mobili e dell'internet delle cose (IoT, Internet of Things);
TorchScript permette di serializzare un modello per eseguirlo al di fuori di Python, in
un ambiente di produzione; il formato ONNX (Open Neural Network Exchange)
facilita esportare un modello di PyTorch perché funzioni con altri
strumenti; e TorchServe è il framework ufficiale di PyTorch per
servire modelli già addestrati attraverso una API (Application Programming
Interface) in produzione, con monitoraggio incluso. Nessuno di questi
strumenti sostituisce quello che già sai fare: tutti partono da un modello
addestrato e valutato esattamente come quello che hai costruito nei capitoli 3, 4,
6 e 8.

## Riassunto dei punti chiave

- La visione artificiale che hai affrontato con MNIST e CIFAR-10 (lezioni 2, 5
  e 6) è la stessa tecnica che sostiene sistemi di riconoscimento di
  immagini e di percezione nella guida autonoma, valutati in condizioni
  diverse da quelle di addestramento, esattamente come hai imparato nella lezione 8
- L'elaborazione del linguaggio naturale, con `torchtext` e modelli come GPT-3
  (175.000 miliardi di parametri) o l'integrazione con Hugging Face
  Transformers, applica lo stesso pattern di dataset, modello e addestramento dei
  capitoli 2, 3 e 4, solo che a una scala molto maggiore
- Il calcolo su GPU della lezione 7 smette di essere un dettaglio di prestazioni e
  diventa un requisito quando il modello o il dataset crescono; PyTorch,
  originato nel laboratorio di ricerca di Facebook (FAIR), è oggi
  uno strumento di uso documentato in aziende come Tesla e OpenAI
- Valutare un modello (lezione 8) non termina in la precisione e la matrice di
  confusione: tecniche di interpretabilità come le mappe di rilevanza, l'
  analisi di attenzione o la stima di incertezza sono imprescindibili
  prima di portare un modello in produzione
- Strumenti come `torch.quantization`, TorchScript, ONNX e TorchServe danno
  il passo finale da un modello conservato con `torch.save` fino a un prodotto
  reale, in un mobile, un dispositivo IoT o un servizio nel cloud

## E ora, il tuo progetto finale

Hai visto che ogni tecnica del corso ha un dominio di applicazione reale
dietro. Adesso ti tocca a te scegliere il tuo. Il tuo progetto finale non ha bisogno
di risolvere un problema così grande come GPT-3 né così critico come la guida
autonoma: ha bisogno di dimostrare che sai percorrere il pattern completo del corso
—tensor, dataset, modello, addestramento e valutazione— su un problema
che ti interessi davvero.

### Passo 1: scegli un problema e un dataset

Inizia dal problema, non dalla tecnica. Chiediti cosa vuoi prevedere
o classificare, e perché ti importa: può essere un problema del tuo settore
professionale, qualcosa legato alle catene di valore che lavora DIH
DATAlife (Agro-Mar Alimentazione, Forestale-Legno, Salute-Cura o
Biotecnologia) o semplicemente qualcosa che ti risulti curioso. Hai due
strade per il dataset:

- Riusare uno dei dataset che hai già usato nel corso (MNIST o CIFAR-10) per
  concentrarti su approfondire la tecnica invece che su cercare dati
- Cercare un dataset proprio in `torchvision.datasets`, in portali di dati
  aperti o costruito da te stesso (foto proprie, un CSV che elabori a mano)

Se scegli dati propri, verifica prima di tutto che hai sufficienti
esempi per categoria e che le etichette sono affidabili: un dataset piccolo o
male etichettato limita molto quel che puoi conseguire, non importa quanto sia
buona la tecnica.

### Passo 2: scegli la tecnica del corso che applicherai

Scegli la tecnica principale secondo il tipo di dato di input, non al contrario:

- Immagini: rete neurale convoluzionale (CNN), come nei capitoli 5 e 6
- Dati tabulari (numeri e categorie in righe e colonne): rete densa di
  classificazione o regressione, come nei capitoli 3 e 4
- Un volume di dati o un modello grande: addestra su GPU seguendo il
  pattern della lezione 7

Combina quella tecnica principale con almeno una buona pratica della lezione 8:
valuta l'overfitting con la perdita e la precisione dentro e fuori dal
campione, e calcola una matrice di confusione per sapere esattamente in cosa
sbaglia il tuo modello.

### Passo 3: struttura l'entregabile

Il tuo entregabile ha due pezzi:

- **Un notebook** con il pipeline completo: carica e esplorazione del
  dataset, definizione del modello con `nn.Module`, ciclo di addestramento,
  valutazione con metriche e matrice di confusione, e conservazione del modello
  addestrato con `torch.save`
- **Una memoria breve** (una o due pagine) che spieghi in linguaggio chiaro
  quale problema risolvi, quale dataset hai usato e le sue limitazioni, quale modello
  hai scelto e perché, quali risultati hai ottenuto e cosa faresti diverso con più
  tempo

### Passo 4: presenta e difendi il tuo progetto

Prepara una presentazione breve, di cinque a dieci minuti, per il resto del
gruppo:

1. Poni il problema in una frase, senza gergo tecnico
2. Mostra il dataset e una o due cifre che lo descrivano (numero di
   esempi, numero di categorie)
3. Spiega l'architettura del modello con uno schema semplice, non con il
   codice completo
4. Mostra i risultati, includendo almeno un grafico o la matrice di
   confusione
5. Chiudi con le limitazioni del progetto e quali passi seguiresti se
   continuassi

I criteri che si terranno in conto sono: che il problema sia ben posto
e giustificato, che la tecnica scelta sia coerente con il tipo di dato, che
valuti onestamente il rendimento del modello —incluso l'overfitting—,
che il notebook sia riproducibile da principio a fine, e che la presentazione
si capisca senza bisogno di leggere il codice.

Apri adesso il notebook di pratica di questa lezione:
`clase09_practica_casos_de_uso_reales.ipynb`. Lì trovi lo scheletro
iniziale del tuo progetto finale, pronto perché cominci a scrivere la prima
cella di codice.
