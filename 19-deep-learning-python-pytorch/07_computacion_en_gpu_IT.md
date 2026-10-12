# 07. Computazione su GPU con PyTorch

## Perché questo argomento è importante

Nei capitoli cinque e sei hai costruito e addestrato una rete neurale convoluzionale (Convolutional Neural Network, CNN) su CIFAR-10, un dataset di immagini a colori. Se hai eseguito quell'addestramento completo, l'avrai già notato: ogni epoca richiede molto più tempo che con la rete densa di MNIST dei capitoli precedenti. Le immagini hanno più pixel, i filtri convoluzionali implicano più operazioni e il modello ha più parametri da aggiustare.

Questo non è un problema del tuo codice: è un problema di hardware. Fino ad ora, tutto quello che hai addestrato è stato eseguito nell'unità centrale di elaborazione (Central Processing Unit, CPU) della macchina, che sia il tuo computer o la macchina virtuale che ti assegna Google Colab. La CPU è capace di fare quel lavoro, ma non è disegnata per farlo velocemente. In questo capitolo capirai perché esiste un tipo di hardware molto meglio preparato per addestrare reti neurali —l'unità di elaborazione grafica (Graphics Processing Unit, GPU)— e, soprattutto, imparerai il pattern di codice che PyTorch usa per approfittarne: verificare se ce n'è una disponibile, creare un `device` e muovervi sia il modello che i dati.

## CPU e GPU: due forme diverse di calcolare

Immagina una fabbrica che confeziona migliaia di scatole identiche. Puoi montarla in due forme molto diverse.

La prima è assumere un pugno di operai estremamente qualificati, capaci di risolvere qualsiasi imprevisto, cambiar compito in qualsiasi momento e prendere decisioni complesse al volo. Sono pochi —tra due e 64 in una CPU moderna— ma ognuno è molto potente e versatile. Così funziona una CPU: pochi nuclei, ognuno capace di eseguire istruzioni molto diverse tra loro, con logica complessa e accesso veloce a una memoria (RAM, Random Access Memory o Memoria di Accesso Casuale) generosa.

La seconda forma è montare una catena di montaggio con migliaia di operai molto più semplici, ognuno addestrato per ripetere un singolo gesto semplice su una scatola dopo l'altra, tutti a volte. Nessuno di loro potrebbe risolvere un imprevisto complesso per conto proprio, ma se il compito è "fai esattamente lo stesso con migliaia di scatole identiche", questa catena di montaggio lascia in ridicolo i pochi operai qualificati in termini di velocità totale. Così funziona una GPU: migliaia di nuclei più semplici, disegnati per eseguire la stessa operazione su molti dati a volte, con memoria propria di velocità alta (VRAM, Video RAM o Memoria Video).

Addestrare una rete neurale è, in fondo, moltiplicare matrici di numeri una volta e un'altra: migliaia di moltiplicazioni e somme identiche ripetute sui pixel di un batch (batch) di immagini o sui pesi di uno strato. È esattamente il tipo di compito che descrive la catena di montaggio, non l'officina di specialisti. Per questo una GPU può addestrare lo stesso modello in una frazione del tempo che le servirebbe una CPU: non è che la GPU sia "più potente" in un senso generale, è che la sua architettura è fatta a misura per questo tipo di calcolo parallelo.

## CUDA: il ponte tra PyTorch e la GPU

Affinché una libreria come PyTorch possa inviarle istruzioni a una GPU, ha bisogno di un linguaggio comune. Quel linguaggio è CUDA (Compute Unified Device Architecture), una piattaforma di calcolo parallelo e un'interfaccia di programmazione di applicazioni (Application Programming Interface, API) creata da NVIDIA che permette di usare le sue GPU per calcolo di propósito generale, non solo per grafica.

Quando installi PyTorch con supporto di GPU, il pacchetto include già tutto il necessario per parlare con CUDA: non devi scrivere codice CUDA tu stesso. Google Colab, inoltre, ti risparmia persino l'installazione dei driver e degli strumenti di CUDA sulla macchina: se attivi l'acceleratore di hardware GPU dal menù del'ambiente di esecuzione, tutto quello è già preparato. Il tuo lavoro si riduce a chiedere a PyTorch, con poche righe di codice, che usi quell'hardware.

## `torch.device` e `.to(device)`: dire a PyTorch dove calcolare

PyTorch rappresenta il "posto" dove vive un tensor o un modello con un oggetto `torch.device`. Per impostazione predefinita, qualsiasi tensor che crei viene immagazzinato nella memoria RAM della CPU. Affinché un calcolo si esegua sulla GPU, devi muovere esplicitamente quel tensor (o il modello intero) a essa, usando il metodo `.to(device)`.

La forma standard di creare quel `device` è verificare prima se c'è una GPU disponibile con `torch.cuda.is_available()`, che restituisce `True` o `False` a seconda se ci sia o no una GPU compatibile con CUDA sulla macchina:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
```

Questo pattern —una singola riga con un operatore condizionale— è il pezzo di codice più importante di tutto il capitolo. Con esso, il tuo codice funziona altrettanto bene in tre situazioni diverse: sul tuo computer senza GPU, in un Colab con l'acceleratore di hardware in modo GPU, o su un server con molte GPU. Non devi mai scrivere due versioni diverse del tuo codice di addestramento: scrivi una sola volta pensando a `device`, e `device` decide per te.

Una volta che hai il `device`, muovi a esso sia il modello che qualsiasi tensor di dati che interagisca con quel modello:

```python
modello = MiaRete().to(device)
input = input.to(device)
etichette = etichette.to(device)
```

C'è una regola che non ammette eccezioni: tutti i tensor che partecipano a una stessa operazione devono stare nello stesso `device`. Se provi a sommare o moltiplicare un tensor che sta sulla CPU con un altro che sta sulla GPU, PyTorch non fa la conversione per te; lancia un errore in tempo di esecuzione del tipo "expected device cpu but got device cuda". Questo non è una limitazione goffa di PyTorch: è una decisione di disegno deliberata. Muovere dati tra la memoria della CPU e la memoria della GPU ha un costo reale —viaggiano per un bus fisico più lento della memoria interna di qualsiasi dei due— così PyTorch preferisce avvertirti con un errore esplicito a che il tuo codice faccia trasferimenti di dati costosi senza che te ne accorga.

## Esempio di codice commentato: addestrare la stessa CNN sul device disponibile

Riprendi la CNN di CIFAR-10 dei capitoli cinque e sei e aggiungi il pattern di `device` al ciclo di addestramento. Il codice seguente verifica l'hardware disponibile, muove il modello una sola volta fuori dal ciclo, muove ogni batch di dati dentro il ciclo e misura quanto tarda un'epoca completa:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# 1. Il device è un interruttore: decide, con una sola riga, dove si calcolano tutti gli altri
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {dispositivo}")

# 2. La stessa architettura convoluzionale dei capitoli 5 e 6, semplificata,
#    pensata per CIFAR-10: immagini a colori di 32x32 pixel e dieci classi
class CNNCifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = x.view(x.size(0), -1)              # appiattisce la mappa di attivazioni
        x = F.relu(self.fc1(x))
        return self.fc2(x)                      # logit grezzi per le dieci classi

# 3. Muovi il modello al device UNA sola volta, prima di creare l'ottimizzatore
modello = CNNCifar().to(dispositivo)
funzione_perdita = nn.CrossEntropyLoss()
ottimizzatore = optim.Adam(modello.parameters(), lr=0.001)

# 4. Ciclo di addestramento di un'epoca: ogni batch viaggia al device prima del forward
inizio = time.time()
for imagini, etichette in caricatore_addestramento:      # caricatore_addestramento è un DataLoader (lezione 2)
    imagini = imagini.to(dispositivo)                       # il batch di immagini viaggia al device
    etichette = etichette.to(dispositivo)                      # le etichette anche, altrimenti CrossEntropyLoss fallirà

    ottimizzatore.zero_grad()
    salida = modello(imagini)
    perdita = funzione_perdita(salida, etichette)
    perdita.backward()
    ottimizzatore.step()

tempo_epoca = time.time() - inizio
print(f"Tempo di un'epoca su {dispositivo}: {tempo_epoca:.2f} secondi")
```

Nota due dettagli del codice. Primo, il modello si muove al `device` una sola volta, prima di entrare nel ciclo: non ha senso muoverlo in ogni iterazione, perché i suoi parametri restano già alloggiati nella memoria del `device` finché decidi diversamente. Secondo, ogni batch di dati —sia le immagini che le etichette— si muove al `device` dentro il ciclo, perché il `DataLoader` consegna batch nuovi in ogni iterazione e quei batch nascono sempre sulla CPU. Se commentassi la riga di `torch.device` e forzassi `dispositivo = torch.device("cpu")`, lo stesso codice continuerebbe a funzionare esattamente allo stesso modo, solo più lentamente: questa è la prova che il pattern è ben scritto. Con una GPU attivata, il blocco di codice non cambia una sola riga; cambia solo il tempo che impiega a eseguirsi.

## Buone pratiche: minimizza i trasferimenti, non l'hardware

Muovere dati tra la CPU e la GPU non è gratis, così conviene tenere presenti alcune raccomandazioni:

- Muovi il modello al `device` una sola volta, fuori dal ciclo di addestramento, non in ogni epoca né in ogni batch
- Muovi ogni batch di dati subito prima di usarlo, mai il dataset completo tutto insieme se non entra nella memoria della GPU: per questo lavori per batch invece che con l'intero insieme a volte
- Controlla con `torch.cuda.device_count()` quante GPU vede PyTorch se lavori su una macchina con molte, per decidere se hai bisogno di distribuire il modello tra loro
- Non presumere che la GPU sia imprescindibile in tutto il momento: durante l'addestramento, con batch grandi, la GPU marca una differenza enorme; durante la messa in produzione di un modello già addestrato, se il numero di consultazioni per minuto è basso, la CPU può essere perfettamente sufficiente e ti risparmia la complessità di gestire hardware specializzato

## Riassunto dei punti chiave

- Una CPU ha pochi nuclei molto versatili, ottimizzati per compiti sequenziali complessi; una GPU ha migliaia di nuclei più semplici, ottimizzati per ripetere la stessa operazione su molti dati in parallelo, che è esattamente quel che necessita l'addestramento di una rete neurale
- CUDA è la piattaforma di NVIDIA che permette a PyTorch di inviare istruzioni alla GPU; con PyTorch non scrivi codice CUDA direttamente, la libreria già lo gestisce per te
- `torch.cuda.is_available()` verifica se c'è una GPU compatibile disponibile; il pattern `torch.device("cuda" if torch.cuda.is_available() else "cpu")` costruisce un `device` che funziona allo stesso modo con o senza GPU
- `.to(device)` muove tensor e modelli al dispositivo scelto; tutti i tensor che partecipano alla stessa operazione devono stare nello stesso `device`, altrimenti PyTorch lancia un errore
- Muovi il modello al `device` una sola volta, fuori dal ciclo di addestramento, e ogni batch di dati dentro il ciclo, subito prima di usarlo
- La GPU accelera soprattutto l'addestramento con batch grandi; in produzione, con poca carga, la CPU può bastare

## Mettilo in pratica

Apri il notebook di teoria di questa lezione in Google Colab ed esegui la cella che crea il `device` con l'ambiente di esecuzione in modo "Nessuno" (solo CPU): conferma che stampa `cpu`. Dopo, cambia l'acceleratore di hardware a GPU da **Ambiente di esecuzione > Cambia tipo di ambiente di esecuzione**, torna a eseguire la stessa cella e verifica che ora stampa `cuda`. Per ultimo, senza guardare il codice di questo capitolo, scrivi di memoria le tre righe minime che ti servono per: verificare se c'è GPU disponibile, creare il `device` corrispondente e muovere un modello qualsiasi a quel `device`.
