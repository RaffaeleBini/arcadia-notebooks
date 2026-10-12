# 04. Addestrare il modello

## Perché questo argomento è importante

Nel capitolo precedente hai costruito una rete neurale con `nn.Module`: strati `Linear`, attivazioni `ReLU`, un input di 784 valori (i pixel di un'immagine di 28x28 di MNIST) e un output di dieci classi (le cifre da zero a nove). Se a questo punto le passi un'immagine quella rete, ottieni una previsione... ma una previsione inutile, perché i pesi degli strati sono numeri casuali. La rete non ha ancora visto un singolo esempio.

Addestrare è precisamente il processo che converte quei pesi casuali in pesi utili. È il capitolo dove la tua rete passa da non sapere nulla a riconoscere cifre scritte a mano con una precisione ragionevole, e dove per fine entra in gioco la parte di PyTorch che rende possibile tutto l'apprendimento profondo (Deep Learning, DL): il calcolo automatico dei gradient.

## Il ciclo di addestramento: forward, loss, backward, optimizer.step

Pensa all'addestramento di una rete neurale (Neural Network, NN) come all'aggiustamento della temperatura di un rubinetto con due manopole, una per l'acqua fredda e una per l'acqua calda, senza guardare i numeri: apri il rubinetto, tocchi l'acqua e noti quanto ti allontani dalla temperatura ideale. Se è troppo fredda, giri la manopola dell'acqua calda; se è troppo calda, la chiudi un po'. Ripeti il gesto una volta e un'altra, con aggiustamenti sempre più precisi, fino a avvicinarti alla temperatura che cerchi.

L'addestramento di una rete segue esattamente questa logica, ripetuta migliaia di volte e con molti più "manopole" (i pesi della rete). Ogni volta del ciclo ha quattro step:

1. **Forward**: passi un batch (batch) di dati attraverso la rete e ottieni una previsione. Questo l'hai già visto nel capitolo precedente
2. **Loss**: confronti quella previsione con l'etichetta reale e ottieni un numero che misura quanto si è sbagliata la rete
3. **Backward**: chiedi a PyTorch di calcolare, per ogni peso della rete, in quale direzione e con quale intensità dovrebbe cambiare affinché l'errore sia un po' minore la prossima volta
4. **Optimizer.step**: applichi quei cambiamenti e aggiorni i pesi

Torni a iniziare con il prossimo batch di dati, e così via. La perdita (loss) non scende a zero di colpo, scende poco a poco, batch a batch, proprio come la temperatura del rubinetto si avvicina all'ideale con vari tocchi della manopola invece che uno solo.

## Cosa misura una funzione di perdita

La funzione di perdita (loss function) è il pezzo che traduce "quanto si è sbagliata la rete" in un numero concreto che PyTorch può usare per correggerla. Quanto più la previsione si allontani dal valore reale, maggiore è la perdita; se la previsione fosse perfetta, la perdita sarebbe zero. Anche quando la rete indovina la classe, quasi sempre rimane un po' di errore, perché la previsione è una distribuzione di probabilità e non una certezza assoluta.

PyTorch offre varie funzioni di perdita a seconda del tipo di problema:

- Per regressione (prevedere un numero continuo), si usa l'errore quadratico medio (Mean Squared Error, MSE), che misura la media delle differenze al quadrato tra la previsione e il valore reale
- Per classificazione (prevedere una categoria tra molte, come in MNIST), si usa l'entropia incrociata (cross-entropy), disponibile in PyTorch come `nn.CrossEntropyLoss`

Poiché la tua rete di classificazione di MNIST restituisce dieci valori grezzi (uno per cifra, senza nessuna attivazione finale), `nn.CrossEntropyLoss` è la funzione giusta: internamente converte quei valori in una distribuzione di probabilità e confronta quella distribuzione con l'etichetta corretta.

## Cosa fa un ottimizzatore

Se la funzione di perdita ti dice quanto ti sei sbagliato, l'ottimizzatore è chi decide cosa fare con quella informazione: aggiusta i pesi della rete a partire dalla perdita e dai gradient, con l'obiettivo che la perdita scenda un po' di più nel giro successivo del ciclo.

Due ottimizzatori comuni in PyTorch, entrambi disponibili in `torch.optim`, sono:

- **SGD** (Stochastic Gradient Descent, discesa del gradiente stocastica): muove ogni peso nella direzione che più riduce la perdita, con una dimensione di passo fissa
- **Adam** (Adaptive Moment Estimation, stima adattativa di momenti): aggiusta quella dimensione di passo automaticamente per ogni peso, il che di solito si traduce in una convergenza più veloce e stabile con meno aggiustamento manuale

Quella "dimensione di passo" ha un nome proprio: il tasso di apprendimento (learning rate), quasi sempre rappresentato come una potenza negativa di dieci, come 1e-3 (che significa lo stesso che 0,001). Il tasso di apprendimento è uno degli iperparametri più importanti che andrai ad aggiustare nella tua carriera con deep learning:

- Se è troppo alto, l'ottimizzatore fa passi così grandi che passa per lungo il punto ottimale, e la perdita può oscillare senza convergere mai
- Se è troppo basso, l'ottimizzatore avanza così piano che l'addestramento richiede un sacco di tempo, o rimane bloccato prima di raggiungere un buon risultato

Non c'è un tasso di apprendimento universale: dipende dal problema, dai dati e dall'ottimizzatore che scegli. Un valore di partenza ragionevole con Adam di solito è 0,001; con la pratica svilupperai intuizione su quando rialzarlo o abbassarlo.

## Che cosa è un'epoca (epoch)

Un'**epoca** è una passata completa attraverso l'intero insieme di dati di addestramento. Addestrare una sola epoca quasi mai è sufficiente: la rete ha appena visto ogni esempio una volta, quindi ha bisogno di varie epoche per raffinare poco a poco i suoi pesi e imparare pattern generali invece di memorizzare esempi concreti. Per questo il ciclo di addestramento ha sempre due livelli: un ciclo esterno che percorre le epoche e un ciclo interno che percorre i batch (batches) di dati dentro ogni epoca.

## Esempio di codice: addestrare la rete di MNIST

Riprendi la rete che hai definito nel capitolo precedente (un input di 784 valori, due strati nascosti con attivazione `ReLU` e un output di dieci classi) e completala con funzione di perdita, ottimizzatore e ciclo di addestramento:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# La stessa architettura della classe precedente: tre strati Linear con ReLU in mezzo
class ReteeMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.strato1 = nn.Linear(28 * 28, 128)
        self.strato2 = nn.Linear(128, 64)
        self.strato3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)      # appiattisce ogni immagine 28x28 in un vettore di 784 valori
        x = F.relu(self.strato1(x))
        x = F.relu(self.strato2(x))
        x = self.strato3(x)               # logit grezzi: CrossEntropyLoss si occupa del resto
        return x

rete = ReteeMNIST()

# Funzione di perdita per classificazione con più classi
funzione_perdita = nn.CrossEntropyLoss()

# Ottimizzatore: aggiusta i pesi della rete (rete.parameters()) secondo il tasso di apprendimento indicato
ottimizzatore = optim.Adam(rete.parameters(), lr=0.001)

epoche = 5
for epoca in range(epoche):
    perdita_epoca = 0.0
    for imagini, etichette in caricatore_addestramento:  # caricatore_addestramento è un DataLoader (lezione 2)
        ottimizzatore.zero_grad()               # reinizializza i gradient accumulati dal passo precedente
        salida = rete(imagini)                 # forward: la rete genera una previsione
        perdita = funzione_perdita(salida, etichette)  # loss: quanto sbagliato la rete
        perdita.backward()                     # backward: calcola il gradient di ogni peso
        ottimizzatore.step()                     # optimizer.step: aggiorna i pesi
        perdita_epoca += perdita.item()

    print(f"Epoca {epoca + 1}/{epoche} - perdita media: {perdita_epoca / len(caricatore_addestramento):.4f}")
```

Nota che ogni uno dei quattro step del ciclo appare come una singola riga di codice: `zero_grad()`, `rete(imagini)`, `perdita.backward()` e `ottimizzatore.step()`. Tutto il resto è contabilità per poter stampare la perdita media di ogni epoca e verificare che scenda man mano che procede l'addestramento.

`ottimizzatore.zero_grad()` è un passo facile da dimenticare e molto facile da spiegare: PyTorch accumula i gradient per impostazione predefinita invece di sovrascriverli, quindi se non li reinizializzi prima di ogni `backward()`, stai sommando i gradient da batch precedenti e l'addestramento perde di senso.

## Valutare il modello: precisione su dati che non ha visto

Una perdita che scende è un buon segno, ma quello che davvero ti interessa è se il modello classifica bene immagini nuove, quelle che non hai usato per addestrare. Per quello usi l'insieme di test (test) e il gestore di contesto `torch.no_grad()`, che dice a PyTorch che non hai bisogno di calcolare i gradient perché non addestri in questo blocco, solo misuri:

```python
corretti = 0
totale = 0

with torch.no_grad():
    for imagini, etichette in caricatore_test:
        salida = rete(imagini)
        previsioni = torch.argmax(salida, dim=1)   # la classe con punteggio più alto per ogni immagine
        corretti += (previsioni == etichette).sum().item()
        totale += etichette.size(0)

precisione = corretti / totale
print(f"Precisione sull'insieme di test: {precisione:.1%}")
```

`torch.argmax` si prende l'indice del valore più alto dei dieci che restituisce la rete per ogni immagine, cioè la cifra che la rete considera più probabile. Confrontare quella previsione con l'etichetta reale, sommare i successi e dividere per il totale ti dà la precisione (accuracy): la proporzione di immagini che la rete ha classificato correttamente.

## Riassunto dei punti chiave

- Il ciclo di addestramento ha quattro step che si ripetono batch per batch: forward (previsione), loss (misurare l'errore), backward (calcolare i gradient) e optimizer.step (aggiornare i pesi)
- Una funzione di perdita come `nn.CrossEntropyLoss` misura quanto la previsione si allontana dall'etichetta reale; in classificazione, quanto più bassa, meglio è
- Un ottimizzatore come `SGD` o `Adam` decide come aggiustare i pesi a partire dall'errore e dai gradient; il tasso di apprendimento controlla la dimensione di questi aggiustamenti
- Un tasso di apprendimento troppo alto può fare sì che l'addestramento non converga; uno troppo basso lo rende inutilmente lento
- Un'epoca è una passata completa attraverso l'intero insieme di addestramento; servono varie epoche affinché la rete generalizzi invece di memorizzare
- `ottimizzatore.zero_grad()` prima di ogni `backward()` è essenziale perché PyTorch accumula i gradient per impostazione predefinita
- La precisione si misura su dati che il modello non ha visto durante l'addestramento, dentro un blocco `torch.no_grad()`

## Mettilo in pratica

Apri un notebook nuovo in Google Colab, carica MNIST con `torchvision` come hai fatto nella lezione 2 e riusa la rete della lezione 3. Addestrala durante tre epoche con `nn.CrossEntropyLoss` e `optim.SGD` (invece di Adam) usando un tasso di apprendimento di 0,01, stampando la perdita media di ogni epoca. Quando finisce, calcola la precisione sull'insieme di test e confrontala con quella che otterresti usando Adam con un tasso di apprendimento di 0,001: quale ottimizzatore raggiunge prima una perdita bassa con lo stesso numero di epoche?
