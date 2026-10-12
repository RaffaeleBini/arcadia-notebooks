# 06. Addestrare una CNN

## Perché importa

Nel capitolo precedente hai costruito la tua prima rete neurale convoluzionale, o
CNN (Convolutional Neural Network): hai impilato strati `nn.Conv2d` e di pooling
per elaborare immagini di MNIST, e hai verificato come cambia la forma di un
tensor al passare attraverso ogni strato. Ma sei rimasto sulla porta di ingresso: non
hai mai addestrato quella rete, così non ha mai imparato nulla.

In questo capitolo chiudi quel cerchio. Riutilizzerai lo stesso ciclo di
addestramento che già conosci del capitolo 4 (perdita, ottimizzatore, forward,
backward) e lo applicherai all'architettura convoluzionale del capitolo 5.
Inoltre, sali di livello nella sfida: invece di MNIST, in bianco e nero, addestrai
la tua CNN su CIFAR-10, un dataset di fotografie a colori con
oggetti reali (aerei, auto, uccelli, gatti e altri otto tipi). È il
salto che separa un esercizio di laboratorio da un problema con la
complessità di un'immagine del mondo reale, ed è esattamente il salto che fanno
quasi tutti i progetti di visione artificiale quando passano da un
prototipo a un caso d'uso serio.

## Il ciclo di addestramento non cambia: solo le mani che muovi cambiano

Immagina che impari a lanciare freccette con gli occhi bendati. Ogni tentativo segue
sempre la stessa sequenza di quattro step: lanci la freccetta (*forward*),
qualcuno ti dice a che distanza sei caduto dal centro (la perdita o *loss*),
calcoli mentalmente verso che lato devi correggere il braccio
(*backward*, cioè la retropropagazione dell'errore) e aggiusti la tua postura
per il tentativo prossimo (`optimizer.step()`). Ripeti questa sequenza una e
un'altra volta, in serie chiamate epoche, finché non colpisci con frequenza.

Quel ciclo di quattro step è identico per una rete densa (quella che hai addestrato nel
capitolo 4) e per una CNN. Quello che cambia non è il ciclo, ma quali "mani"
muovi in ogni lancio: nella rete densa, ogni lancio aggiusta pesi
collegati direttamente a ogni pixel; nella CNN, ogni lancio aggiusta i
valori dei filtri convoluzionali, che si applicano allo stesso modo in
tutta l'immagine. Per questo puoi copiare letteralmente la struttura del ciclo di
addestramento del capitolo 4 e limitarti a cambiare il modello che le passi.

## Da bianco e nero a colore: cosa cambia con CIFAR-10

Pensa alla differenza tra leggere uno schizzo a matita e guardare una foto a
colore con gente e oggetti di sfondo. Lo schizzo è facile da interpretare perché
ha solo un tono di grigio e pochi dettagli; la foto ha molto più
informazione e molto più rumore visivo. Questo è, in essenza, il salto da MNIST a
CIFAR-10:

- MNIST sono cifre scritte a mano, in scala di grigi, centrate e su
  sfondo pulito: un tensor di input con forma `(batch, 1, 28, 28)`, dove l'
  1 è l'unico canale di colore
- CIFAR-10 sono fotografie a colore di sessantamila oggetti reali (aerei,
  auto, uccelli, gatti, cervi, cani, rane, cavalli, barche e
  camion), con sfondi variati e l'oggetto non sempre centrato: un tensor
  di input con forma `(batch, 3, 32, 32)`, dove il 3 corrisponde ai tre
  canali di colore rosso, verde e blu, o RGB (Red-Green-Blue)

Quel cambio da 1 a 3 canali di input è l'unico cambio strutturale
obbligatorio nell'architettura: il primo strato `nn.Conv2d` deve accettare
3 canali di input invece di 1. Il resto del pattern (vari blocchi di
convoluzione, attivazione ReLU e pooling, seguiti da strati completamente
connessi) rimane esattamente uguale a quello del capitolo 5.

C'è un dettaglio pratico che merita di essere ricordato dal materiale ufficiale:
PyTorch non include una funzione che "appiattisca" automaticamente la salida degli
strati convoluzionali prima di passarla al primo strato lineare, così
devi calcolare tu quella dimensione (o, come suggerisce il materiale, passare un
tensor di prova per la rete e osservare la forma che restituisce). Aggiungendo
`padding=1` in ogni convoluzione di 3x3, la larghezza e l'altezza dell'immagine sì
riducono solo negli strati di pooling, il che rende quel calcolo molto più
prevedibile: partendo da 32x32, tre blocchi di pooling di dimensione 2 la lasciano
in 4x4.

## Perché una CNN di solito supera una rete densa in immagini

Supponi di voler risolvere CIFAR-10 con la stessa rete densa del capitolo 4,
senza nessuno strato convoluzionale. Prima di nulla, dovresti appiattire ogni
immagine a un vettore di 3.072 valori (32 × 32 pixel × 3 canali). Se il
primo strato nascosto di quella rete densa avesse, ad esempio, 256 neuroni, solo
quello strato avrebbe bisogno di imparare 3.072 × 256 ≈ 786.000 pesi, uno per ogni
combinazione possibile di pixel di input e neurone di salida. E nonostante ciò, quella
rete non avrebbe alcuna nozione che un pixel sia "accanto" a un altro: per
lei, muovere un gatto tre pixel a destra nell'immagine produce un pattern
di input completamente diverso, che dovrebbe imparare separatamente.

Uno strato convoluzionale risolve quel problema in un altro modo: invece di un peso
per ogni combinazione di pixel e neurone, impara un numero piccolo di filtri
(ad esempio, 32 filtri di 3x3x3) e li riusa in ogni posizione dell'immagine. Quei stessi 32 filtri di 3x3 su 3 canali hanno bisogno solo di
32 × (3 × 3 × 3 + 1) ≈ 900 parametri, e applicandosi in ogni posizione dell'immagine,
la rete impara a riconoscere un carattere (un bordo, una trama, una macchia di
colore) non importa in quale zona dell'immagine appaia. A quella
proprietà la chiamiamo invarianza a la traslazione, ed è la ragione principale per cui,
addestrando lo stesso numero di epoche e sullo stesso dataset di
immagini, una CNN di solito raggiunge maggiore precisione che una rete densa
equivalente, con molti meno parametri da aggiustare e, quindi, meno
rischio di overfitting (*overfitting*).

## Esempio di codice commentato

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Prepariamo i dati: convertiamo le immagini a tensor e normalizziamo
#    ogni canale di colore al range approssimativo [-1, 1]
trasformazione = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./dati", train=True, download=True, transform=trasformazione
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./dati", train=False, download=True, transform=trasformazione
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 2. Definiamo l'architettura: stesso pattern del capitolo 5
#    (convoluzione + ReLU + pooling), ma con 3 canali di input
class ConvNetCIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 16x16 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 8x8 -> 8x8
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)  # 10 classi di CIFAR-10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


modello = ConvNetCIFAR()

# 3. Funzione di perdita e ottimizzatore: uguale al capitolo 4,
#    ma con CrossEntropyLoss perché sono 10 classi (classificazione multiclasse)
criterio = nn.CrossEntropyLoss()
ottimizzatore = optim.Adam(modello.parameters(), lr=0.001)

# 4. Ciclo di addestramento: forward, loss, backward, step, ripetuto per epoche
epoche = 5
for epoca in range(epoche):
    modello.train()
    perdita_accumulata = 0.0
    for imagini, etichette in train_loader:
        ottimizzatore.zero_grad()          # puliamo i gradient dal step precedente
        salide = modello(imagini)      # forward
        perdita = criterio(salide, etichette)
        perdita.backward()                # backward: calcola i gradient
        ottimizzatore.step()               # aggiorna i pesi del modello
        perdita_accumulata += perdita.item()

    perdita_media = perdita_accumulata / len(train_loader)
    print(f"Epoca {epoca + 1}/{epoche} - perdita media: {perdita_media:.4f}")

# 5. Valutazione finale sull'insieme di test (immagini mai viste)
modello.eval()
corrette = 0
totale = 0
with torch.no_grad():
    for imagini, etichette in test_loader:
        salide = modello(imagini)
        previsioni = torch.argmax(salide, dim=1)
        corrette += (previsioni == etichette).sum().item()
        totale += etichette.size(0)

print(f"Precisione nell'insieme di test: {corrette / totale:.2%}")
```

## Riassunto dei punti chiave

- Il ciclo di addestramento (forward, perdita, backward, `optimizer.step()`)
  è lo stesso per una rete densa e per una CNN: l'unico che cambia è il
  modello che le passi, non la struttura del ciclo
- CIFAR-10 introduce due cambiamenti rispetto a MNIST: le immagini hanno 3 canali
  di colore invece di uno, con forma `(batch, 3, 32, 32)`, e gli oggetti sono
  fotografie reali con sfondo e variabilità, non cifre pulite e centrate
- L'unico cambio strutturale obbligatorio nell'architettura è che il
  primo strato `nn.Conv2d` deve accettare 3 canali di input invece di 1
- PyTorch non appiattisce automaticamente la salida degli strati convoluzionali:
  devi calcolare la dimensione di input del primo strato `nn.Linear` a
  mano (o verificarlo passando un tensor di prova per la rete)
- Una CNN condivide gli stessi filtri in tutta l'immagine, così ha bisogno
  di molti meno parametri di una rete densa equivalente e di solito raggiunge
  maggiore precisione su immagini, grazie all'invarianza a la traslazione
- `nn.CrossEntropyLoss` è la funzione di perdita usuale per classificazione
  multiclasse, e combina internamente softmax con l'entropia incrociata

## Mettilo in pratica

Apri il notebook di teoria di questa lezione, addestra la CNN di questo capitolo
su CIFAR-10 durante cinque epoche e annota la precisione finale sull'insieme
di test. Calcola a mano (con carta e penna, o con un calcolatrice)
quanti parametri avrebbe solo il primo strato di una rete densa equivalente
che ricevesse i 3.072 input di un'immagine di CIFAR-10 e li collegasse a
uno strato nascosto di 256 neuroni. Confronta quella cifra con il numero di
parametri del primo strato `nn.Conv2d` della tua CNN (puoi ottenerlo con
`sum(p.numel() for p in modello.conv1.parameters())`). Conserva entrambe le
cifre: ti serviranno di riferimento nel notebook di pratica, dove completerai
tu stesso il ciclo di addestramento e il calcolo della precisione.
