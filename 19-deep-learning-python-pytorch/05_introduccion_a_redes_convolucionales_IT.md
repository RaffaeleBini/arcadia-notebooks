# 05. Introduzione alle reti neurali convoluzionali

## Perché importa

Nel capitolo precedente hai costruito e addestrato una rete densa con `nn.Linear`
per classificare le cifre di MNIST. Affinché quella rete potesse elaborarle,
hai dovuto appiattire ogni immagine di 28x28 pixel in un singolo vettore di 784
valori. Quella decisione ha un costo: appiattendo l'immagine, distruggi tutta
l'informazione su quale pixel è accanto a quale. La rete densa riceve una
lista di numeri senza alcuna nozione di "su", "giù" o "vicino", così
deve imparare da zero, a forza di esempi, che certi pixel
di solito appaiono insieme formando un tratto o una curva.

Le reti neurali convoluzionali, o CNN (Convolutional Neural Network),
risolvono esattamente questo problema. Invece di appiattire l'immagine, mantengono la sua
struttura di griglia e usano strati specializzati che sanno approfittare
della posizione relativa dei pixel. Il risultato sono modelli che imparano
più velocemente, con meno parametri e che generalizzano meglio a variazioni della
stessa immagine (una cifra leggermente spostata, ad esempio). Per questo le CNN
sono la base di praticamente qualsiasi sistema moderno di visione artificiale,
dal riconoscimento della scrittura alla guida autonoma.

In questa lezione non addestrai ancora la rete: lo farai nella lezione 6.
Qui ti concentri su capire e costruire l'architettura, e su verificare che
il forward pass produce le forme di tensor che ti aspetti in ogni step.

## Cosa risolve una convoluzione rispetto a uno strato denso

Pensa a come riconosci una cifra scritta a mano. Non memorizzi l'immagine
pixel per pixel: noti i tratti locali (una curva qui, una linea retta
lì) e combini quegli indizi per decidere se è un "3" o un "8". Uno strato
denso non può fare questo naturalmente, perché ognuno dei suoi neuroni
guarda **tutti** i pixel dell'immagine contemporaneamente, con un peso diverso
per ognuno. Se lo stesso tratto appare un po' più su o un po' più a destra
in un'altra immagine, il neurone deve imparare quel pattern un'altra volta, con pesi
nuovi, perché non riconosce che è "lo stesso tratto in un altro posto".

Uno strato convoluzionale funziona diversamente: invece di guardare l'intera
immagine tutto insieme, la percorre con una piccola finestra, applicando
**la stessa operazione** in ogni posizione. Questo ha due conseguenze
importanti:

- Lo strato rileva un pattern (ad esempio, un bordo) non importa in quale
  parte dell'immagine appaia, perché esamina l'intera immagine con la stessa
  finestra
- Lo strato ha bisogno di molti meno parametri rispetto a uno strato denso equivalente,
  perché la stessa finestra di pesi si riusa in tutte le posizioni, invece
  di avere un peso diverso per ogni pixel dell'immagine

A questa riusabilità degli stessi pesi su tutta l'immagine la chiamiamo
*weight sharing* (condivisione dei pesi), ed è la ragione principale per cui
le CNN scalano meglio a immagini grandi rispetto alle reti dense.

## Che cos'è un filtro (kernel)

La "finestra" che scorre sull'immagine si chiama **filtro**, o anche
**kernel**: una piccola matrice di numeri, normalmente di tre per tre
posizioni. In ogni posizione dell'immagine, il filtro moltiplica ognuno dei
suoi valori per il pixel che ha sotto, somma tutti i risultati e quel
numero diventa il valore di un pixel dell'immagine di salida (chiamata
**mappa di caratteristiche**, o *feature map*). Il filtro si sposta
per tutta l'immagine, generando un valore nuovo in ogni posizione.

Nessuno dei numeri del filtro è scelto a mano: si inizializzano a caso e
si aggiustano durante l'addestramento per retropropagazione
(*backpropagation*), esattamente come i pesi di uno strato `nn.Linear`.
Quando l'addestramento termina, ogni filtro finisce specializzato nel rilevare
un tipo specifico di caratteristica: i filtri dei primi strati di solito
rispondono a bordi, curve o angoli semplici, e i filtri di strati più
profondi combinano quei segnali per riconoscere forme più complesse, come l'anello di un "8" o il tratto chiuso di uno "0".

Uno strato convoluzionale non impara un singolo filtro, ma vari in parallelo,
ognuno cercando una caratteristica diversa. In PyTorch, uno strato convoluzionale si
definisce con `nn.Conv2d`:

```python
nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
```

- `in_channels`: numero di canali dell'immagine di input (uno per scala
  di grigi, come MNIST; tre per un'immagine a colori con canali rosso,
  verde e blu, RGB)
- `out_channels`: numero di filtri diversi che lo strato impara, cioè
  quante mappe di caratteristiche produce
- `kernel_size`: la dimensione del filtro (qui, 3x3)
- `padding`: quante righe e colonne di zeri si aggiungono intorno all'immagine
  prima di applicare il filtro; con `padding=1` e un filtro di 3x3, l'immagine
  di salida conserva lo stesso alto e largo di quella di input, invece
  di ridursi in ogni strato

## Che cos'è lo stride

Lo **stride** (passo) è la distanza, in pixel, che il filtro avanza ogni
volta che si sposta. Con `stride=1`, il filtro si muove di una posizione ogni
volta ed esamina praticamente tutte le posizioni possibili dell'immagine. Con
`stride=2`, il filtro salta di due posizioni, esamina la metà delle
posizioni e produce una mappa di caratteristiche più piccola.

Pensa allo stride come alla lunghezza del passo quando percorri l'immagine: con
passi corti (stride piccolo) esamini l'immagine con molti dettagli ma
impiega più tempo e generi una salida più grande; con passi lunghi (stride
grande) percorri l'immagine più velocemente, a spese di esaminarla con meno
dettaglio. Nelle CNN semplici, di solito si lascia `stride=1` negli strati
convoluzionali e si delega la riduzione di dimensione agli strati di pooling,
che è esattamente quello che vedrai dopo.

## Che cosa fa il pooling

Dopo uno o più strati convoluzionali, è normale aggiungere uno strato
di **pooling** (aggruppamento), il cui lavoro è ridurre la dimensione delle
mappe di caratteristiche conservando l'essenziale di quello che i filtri già
hanno rilevato.

La variante più comune è **max pooling**: l'immagine si percorre in blocchi
(ad esempio, di 2x2) e di ogni blocco si conserva solo il valore più alto.
Il risultato è una mappa di caratteristiche con metà dell'alto e metà
della larghezza, ma che conserva comunque le zone dove i filtri
hanno risposto con più intensità.

Pensa al pooling come a riassumere un paragrafo lungo in una frase: perdi
dettaglio, ma conservi l'idea principale. Applicato a un'immagine, il pooling
apporta due vantaggi:

- Riduce la quantità di calcoli che servono agli strati successivi, perché
  lavorano su una mappa di caratteristiche più piccola
- Fa che la rete sia più robusta a piccoli spostamenti del tratto nell'immagine,
  perché non importa il pixel esatto dove è apparso il carattere, solo
  che è apparso in quella zona

In PyTorch, il pooling si aggiunge con `nn.MaxPool2d`:

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

Qui `kernel_size=2` definisce la dimensione del blocco (2x2) e `stride=2` fa
che i blocchi non si sovrappongano tra loro, così che la mappa di caratteristiche
risultante misura esattamente metà dell'alto e della larghezza di quella di input.

## Esempio di codice commentato

L'esempio seguente definisce una CNN semplice per classificare le cifre di
MNIST (immagini in scala di grigi di 28x28 pixel) in dieci classi, una
per ogni cifra da zero a nove. L'architettura combina due blocchi di
convoluzione più pooling e termina con uno strato `nn.Linear`, seguendo il
stesso pattern che descrive il materiale ufficiale: estrarre caratteristiche con
convoluzioni prima di classificare con uno strato denso.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSemplice(nn.Module):
    """CNN semplice per classificare MNIST (1 canale, 28x28 pixel) in 10 classi."""

    def __init__(self):
        super().__init__()
        # Primo blocco: 1 canale di input (scala di grigi), 16 filtri
        # di 3x3, con padding=1 per non perdere dimensione ai bordi
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                                stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Secondo blocco: riceve i 16 mappe di caratteristiche dello strato
        # precedente e impara 32 filtri nuovi, ognuno combinando i caratteri
        # rilevati dal primo blocco
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                                stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Strato finale denso: classifica a partire dalle mappe di caratteristiche
        # già ridotte dai due strati di pooling (28 -> 14 -> 7)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)   # appiattissimo prima dello strato denso
        x = self.fc(x)
        return x


modello = CNNSemplice()
print(modello)

# Forward pass su un batch di esempio: 4 immagini di MNIST simulate
batch_esempio = torch.randn(4, 1, 28, 28)

x = batch_esempio
print('Input:                ', x.shape)
x = F.relu(modello.conv1(x)); print('Dopo conv1 + ReLU:       ', x.shape)
x = modello.pool1(x);         print('Dopo pool1:              ', x.shape)
x = F.relu(modello.conv2(x)); print('Dopo conv2 + ReLU:       ', x.shape)
x = modello.pool2(x);         print('Dopo pool2:              ', x.shape)
x = x.view(x.size(0), -1);   print('Dopo appiattire (flatten):  ', x.shape)
x = modello.fc(x);            print('Dopo fc (salida):        ', x.shape)
```

Nota due cose del codice. Primo, in PyTorch, la forma di un batch di
immagini segue sempre l'ordine `(batch, canali, alto, largo)`: per questo il
batch di esempio ha forma `(4, 1, 28, 28)`, quattro immagini di un canale di
28x28 pixel, invece di appiattirle come facevi con `nn.Linear`. Secondo,
il numero `32 * 7 * 7` dello strato `fc` non è arbitrario: esce da seguire
l'immagine step dopo step attraverso la rete (28 pixel di input, senza cambiamenti dal
padding in ogni convoluzione, ridotto a metà in ogni pooling: 28 → 14 →
7) e moltiplicare quella dimensione finale per i 32 filtri del secondo blocco
convoluzionale. Se cambi qualsiasi parametro dell'architettura, quel
numero cambia anche, e devi ricalcolarlo.

## Riassunto dei punti chiave

- Uno strato denso guarda tutti i pixel dell'immagine a volte e perde
  l'informazione su quale pixel è accanto a quale; uno strato convoluzionale
  percorre l'immagine con una piccola finestra che riusa gli stessi pesi
  in tutta l'immagine, così rileva un pattern non importa dove appaia
- Il filtro (o kernel) è una piccola matrice di numeri che si moltiplica
  con i pixel che ha sotto in ogni posizione; i suoi valori non si scelgono
  a mano, si imparano durante l'addestramento come i pesi
  di uno strato `nn.Linear`
- Lo stride controlla quanti pixel avanza il filtro in ogni step: uno
  stride più grande produce una salida più piccola e esamina l'immagine con meno
  dettaglio
- Il pooling (normalmente max pooling, con `nn.MaxPool2d`) riduce la dimensione
  delle mappe di caratteristiche tenendo il valore più alto di ogni
  blocco, il che alleggerisce i calcoli e dà robustezza di fronte a piccoli
  spostamenti
- Una CNN semplice in PyTorch combina blocchi di `nn.Conv2d` più pooling
  per estrarre caratteristiche, e termina con uno o più strati
  `nn.Linear` per classificare, definiti dentro una classe che eredita da
  `nn.Module`
- Prima di definire lo strato `nn.Linear` finale hai bisogno di calcolare a mano
  quante dimensioni ha il tensor una volta appiattito, seguendo come
  cambia la sua forma in ogni strato convoluzionale e di pooling

## Mettilo in pratica

Apri il notebook di teoria di questa lezione, esegui la CNN di questo capitolo e
osserva come cambia la forma del tensor dopo ogni strato. Poi,
cambia il numero di filtri di `conv1` da 16 a 8 e quello di `conv2` da 32 a 16,
senza toccare niente di più, ed esegui di nuovo il forward pass. Quale forma ha
ora l'input dello strato `fc`? Calcola il valore nuovo a mano prima
di eseguire il codice e verifica se coincide. Avrai bisogno di quel stesso
calcolo nel notebook di pratica, dove ti spetta aggiungere uno strato
convoluzionale più a una CNN già iniziata.
