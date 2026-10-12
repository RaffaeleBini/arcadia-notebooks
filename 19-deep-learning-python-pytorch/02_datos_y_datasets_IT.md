# 02. Dati e dataset in PyTorch

## Perché questo argomento è importante

Nel capitolo precedente hai costruito la tua prima rete neurale minima con `nn.Module` e hai visto come PyTorch calcola i gradient automaticamente. Quella rete, però, non serve a nulla se non le arrivano dati con cui imparare. Nel Deep Learning (DL, apprendimento profondo), l'addestramento di un modello dipende completamente dalla qualità e dall'organizzazione dei dati che gli fornisci: indovinare l'architettura della rete è solo una parte del lavoro, e molte volte nemmeno la più lunga. Preparare, caricare e iterare sui dati di solito occupa la maggior parte del tempo reale di qualsiasi progetto di Intelligenza Artificiale (IA).

Prima di addestrare nulla nelle prossime lezioni, hai bisogno di sapere come PyTorch rappresenta un insieme di dati, in che modo lo serve al modello in porzioni gestibili e quali strumenti offre di fabbrica per non dover reinventare tutto questo ogni volta che inizi un progetto nuovo.

## I dati come punto di partenza

Pensa a una rete neurale come a una persona che impara un mestiere guardando esempi, non leggendo un manuale di istruzioni. Quanto più variati e meglio organizzati siano quegli esempi, meglio imparerà il mestiere. Se le mostri sempre gli stessi tre casi, o nello stesso ordine ogni volta, imparerà trucchi specifici di quelle situazioni invece di principi generali che le servano di fronte a dati nuovi.

Per questo, prima di scrivere una sola riga di architettura di rete, devi risolvere tre domande sui tuoi dati:

- Da dove vengono e in quale formato sono?
- Come li dividi in un insieme per addestrare e un altro per verificare se il modello generalizza?
- Come li consegni al modello durante l'addestramento, senza che diventi un collo di bottiglia?

PyTorch risponde a queste tre domande con due pezzi che funzionano insieme: `Dataset` e `DataLoader`.

## Che cos'è la classe Dataset

Un `Dataset` è, prima di tutto, un contratto: qualsiasi oggetto che lo rispetta sa rispondere a due domande molto concrete. La prima è quanti elementi ha in totale (`__len__`); la seconda è come si vede l'elemento numero *i* (`__getitem__`). Non importa se dietro c'è una lista di dieci numeri, centomila immagini su disco o un file audio: finché l'oggetto sa rispondere a quelle due domande, PyTorch può trattarlo come un dataset.

Questa semplicità è intenzionale. Puoi creare il tuo `Dataset` ereditando da `torch.utils.data.Dataset` e implementando quei due metodi con la tua propria logica di caricamento, o ricorrere a uno di quelli già pronti in `torchvision`, la libreria di PyTorch focalizzata sulla visione artificiale. In lezioni successive costruirai i tuoi dataset personalizzati; in questa lezione cominci con uno già fatto, per concentrarti sul meccanismo prima che sui dettagli dei tuoi dati.

## Cosa offre torchvision

`torchvision` è un modulo di PyTorch pensato soprattutto per compiti di visione artificiale. Il suo contributo principale per questa lezione è doppio:

- Offre **dataset di riferimento** già confezionati come classi `Dataset`, tra cui MNIST (cifre scritte a mano), CIFAR-10 (immagini a colori di dieci categorie) e ImageNet (milioni di immagini con migliaia di categorie). Usarli è quasi "barare": ti risparmia completamente il lavoro di cercare, scaricare e formattare i dati, qualcosa che in un progetto reale può richiederti settimane
- Offre **trasformazioni** già pronte all'uso, nel modulo `torchvision.transforms`, che convertono i dati grezzi nel formato che un modello si aspetta: convertire un'immagine in tensor, normalizzarla, ritagliarla o ruotarla, tra molte altre opzioni

Usare questi dataset di riferimento quando si inizia ad imparare ha senso, ma non rappresenta il caso solito: in un progetto reale quasi sempre lavorerai con dati tuoi dell'organizzazione o del tuo settore, e lì dovrai costruire il tuo `Dataset`. Tornerai su questo in lezioni successive.

## Che cos'è DataLoader e perché si addestra a batch

Avere i dati organizzati come un `Dataset` non basta: durante l'addestramento hai bisogno di servirli al modello in modo pratico. Qui entra in gioco `torch.utils.data.DataLoader`.

Immagina di dover distribuire parecchie migliaia di lettere per un evento. Non le metti tutte in una busta gigante né le consegni una per una: le raggruppi in pacchetti di dimensione gestibile. Un `DataLoader` fa esattamente questo con i tuoi dati: li raggruppa in **batch** (*batches*) di una dimensione che decidi tu (`batch_size`), e ti permette di iterare su di essi con un semplice ciclo `for`.

Addestrare per batch, invece che con l'intero dataset tutto insieme o con un esempio alla volta, ha vari vantaggi:

- Un dataset grande potrebbe non entrare interamente nella memoria del computer; un batch sì
- Il modello aggiorna i suoi parametri varie volte per ogni passo completo (*epoca*) attraverso il dataset, invece che una sola volta, il che di solito accelera l'apprendimento
- Una dimensione di batch usuale è tra otto e 64 esempi, spesso in potenze di due come 8, 16, 32 o 64, anche se il numero ottimale dipende dal tuo problema specifico

Inoltre, `DataLoader` accetta il parametro `shuffle`, che mescola l'ordine degli esempi prima di ogni epoca. Questo è importante perché un dataset ordinato (ad esempio, tutti gli "zeri" prima dei "uno") può far sì che il modello impari pattern legati a quella posizione invece che alle caratteristiche reali di ogni cifra. Mescolare i dati aiuta a imparare principi generali.

Infine, conviene che controlli se il tuo dataset è **equilibrato**: cioè, che ogni categoria abbia un numero simile di esempi. Se il 60% dei tuoi dati appartiene a una sola categoria, il modello può imparare la scorciatoia di predire sempre quella classe per minimizzare l'errore, invece di imparare a distinguere davvero tra le altre. Rilevare questo squilibrio è semplice quanto contare quanti esempi ci sono di ogni classe prima di iniziare ad addestrare.

## Esempio di codice commentato

L'esempio seguente carica il dataset MNIST con `torchvision`, lo converte in tensor con una trasformazione di base e lo avvolge in un `DataLoader` per iterare su un batch.

```python
import torch
from torchvision import datasets, transforms

# Trasformazione di base: converte ogni immagine (in formato PIL) in un tensor di PyTorch
trasformazione = transforms.ToTensor()

# Dataset di addestramento di MNIST: scarica i dati se non esistono in "./dati"
trainset = datasets.MNIST(
    root="./dati",
    train=True,
    download=True,
    transform=trasformazione,
)

# Dataset di test: stesso luogo, ma con la parte riservata per valutare il modello
testset = datasets.MNIST(
    root="./dati",
    train=False,
    download=True,
    transform=trasformazione,
)

# DataLoader: serve il dataset di addestramento in batch di 10 esempi, mescolati in ogni epoca
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)

# Il DataLoader di test non ha bisogno di mescolare i dati: lo userai solo per valutare
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

# Iteriamo un singolo batch per ispezionarlo
for dati in trainloader:
    X, y = dati  # X: immagini del batch: y: etichette (la cifra di ogni immagine)
    print("Forma del batch di immagini:", X.shape)  # torch.Size([10, 1, 28, 28])
    print("Forma del batch di etichette:", y.shape)  # torch.Size([10])
    break  # vogliamo solo vedere il primo batch, non scorrere l'intero dataset
```

Nota la forma del batch di immagini: `[10, 1, 28, 28]`. Questi quattro numeri sono, in ordine, la dimensione del batch (dieci immagini), il numero di canali di colore (uno, perché MNIST è in scala di grigi), e l'altezza e la larghezza di ogni immagine in pixel (28 per 28). Quando lavori con immagini a colori, come in CIFAR-10, vedrai tre canali invece di uno.

## Riassunto dei punti chiave

- I dati sono la base dell'addestramento: prepararli e organizzarli di solito richiede più tempo che definire l'architettura della rete
- Un `Dataset` è qualsiasi oggetto che sa quanti elementi ha (`__len__`) e come restituire l'elemento numero *i* (`__getitem__`)
- `torchvision` offre dataset di riferimento già pronti (MNIST, CIFAR-10, ImageNet) e trasformazioni (`torchvision.transforms`) per dare ai dati il formato che il modello si aspetta
- Un `DataLoader` serve i dati in batch (*batches*), permette di mescolarli con `shuffle` ed evita di dover caricare l'intero dataset in memoria tutto insieme
- Mescolare i dati (`shuffle=True`) evita che il modello impari pattern legati all'ordine in cui vede gli esempi
- Un dataset squilibrato, dove una classe ha molti più esempi delle altre, può far sì che il modello impari scorciatoie invece di pattern reali

## Mettilo in pratica

Apri un notebook nuovo in Google Colab e carica il dataset CIFAR-10 con `datasets.CIFAR10` (invece di MNIST), applicando la stessa trasformazione `transforms.ToTensor()`. Crea un `DataLoader` con una dimensione di batch di 16 e `shuffle=True`, itera il primo batch e confronta la forma del tensor di immagini con quella che hai ottenuto per MNIST in questo capitolo. Nota quanti canali di colore ha ogni immagine e perché pensi che sia diverso dal caso delle cifre scritte a mano.
