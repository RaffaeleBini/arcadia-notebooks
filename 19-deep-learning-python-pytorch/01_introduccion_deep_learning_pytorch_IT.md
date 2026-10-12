# 01. Introduzione a Deep Learning con Python e PyTorch

## Perché questo argomento è importante

Riconoscere un volto, capire una frase o prevedere se una macchina si guasterà sono compiti che gli esseri umani risolvono senza sforzo, ma che risultano quasi impossibili da programmare con regole esplicite: ci sono troppe varianti ed eccezioni per anticiparle tutte a mano. L'apprendimento profondo (DL, Deep Learning) risolve questo problema in un modo diverso: invece di programmare le regole, addestra un modello affinché le deduca dai dati.

Python e PyTorch sono, oggi, una delle combinazioni più utilizzate per mettere questa idea in pratica. Python offre una sintassi semplice e un ecosistema enorme di librerie; PyTorch, sviluppato dal laboratorio di ricerca in intelligenza artificiale (IA) di Facebook (FAIR, Facebook AI Research), offre gli strumenti necessari per costruire, addestrare e valutare reti neurali in modo flessibile. In questo capitolo farai i tuoi primi passi con entrambi: che cos'è l'apprendimento profondo, che cos'è una rete neurale, che cos'è un tensor e cosa fa autograd per te quando addestri un modello.

## Apprendimento profondo versus apprendimento automatico tradizionale

L'apprendimento automatico (ML, Machine Learning) cambia già la logica della programmazione tradizionale: invece di scrivere regole a mano, dai al sistema esempi e gli permetti di dedurre il pattern. L'apprendimento profondo è un sottocampo di ML che utilizza un tipo specifico di modello, la rete neurale, e in particolare reti con vari strati nascosti impilati ("profondi"), capaci di imparare rappresentazioni sempre più complesse dei dati: da pixel singoli a bordi, da bordi a forme, da forme a oggetti completi.

Pensa alla differenza con un'analogia culinaria. ML tradizionale è come seguire una ricetta che un cuoco esperto ha già perfezionato: tu, come programmatore, definisci a mano quali caratteristiche (feature) contano e in quale proporzione. L'apprendimento profondo è più simile a lasciare che un apprendista provi migliaia di combinazioni di ingredienti fino a trovare da solo la ricetta che funziona meglio, senza che nessuno gli spieghi in anticipo quali caratteristiche contano davvero. Questa capacità di imparare direttamente le caratteristiche, invece di averle già definite, è ciò che rende l'apprendimento profondo la tecnica dominante nella visione artificiale, nell'elaborazione del linguaggio naturale (NLP, Natural Language Processing) e nel riconoscimento vocale.

## Che cos'è una rete neurale

Una rete neurale (NN, Neural Network) è un modello matematico ispirato, in modo molto semplificato, a come si collegano i neuroni del cervello umano. È formata da nodi chiamati neuroni, organizzati in strati:

- Uno **strato di input**, che riceve i dati già convertiti in numeri (ad esempio, il valore di ogni pixel di un'immagine)
- Uno o più **strati nascosti**, chiamati così perché non controlli direttamente ciò che accade in essi
- Uno **strato di output**, che produce la previsione finale

Ogni neurone riceve più valori di input, ognuno moltiplicato per un peso, aggiunge un bias e passa il risultato attraverso una funzione di attivazione prima di inviarlo al prossimo strato. La funzione di attivazione introduce non linearità: senza di essa, non importa quanti strati impili, l'intera rete equivarrebbe a una semplice regressione lineare, incapace di imparare pattern complessi. Due funzioni di attivazione comuni sono ReLU (Rectified Linear Unit, unità lineare rettificata), che lascia passare i valori positivi e converte in zero i negativi, e sigmoid, che comprime qualsiasi valore a un intervallo tra zero e uno, utile per interpretare l'output come una probabilità.

Addestrare una rete neurale consiste nell'aggiustare i pesi e i bias per minimizzare la differenza tra ciò che predice e ciò che dovrebbe predire. Questo aggiustamento avviene in modo iterativo, ripetendo un ciclo di previsione, misurazione dell'errore e correzione, fino a migliaia di volte. Con strati di dimensione moderata, una rete può facilmente accumulare milioni di parametri: più ne ha, più pattern complessi può imparare, ma anche più rischio corre di memorizzare gli esempi di addestramento invece di generalizzare a dati nuovi, un problema noto come overfitting.

## Che cos'è un tensor

PyTorch ha bisogno di rappresentare assolutamente tutto —un numero singolo, un'immagine, un batch di migliaia di immagini— con una singola struttura di dati su cui applicare le stesse operazioni matematiche in modo efficiente. Quella struttura è il **tensor**: un array multidimensionale molto simile a un `ndarray` della libreria NumPy, con il vantaggio aggiunto di poter risiedere nella memoria di una GPU (Graphics Processing Unit) e beneficiare così della sua capacità di calcolo parallelo.

Puoi pensare ai tensor come a una famiglia di contenitori di numeri che scala con la complessità del dato:

- Un numero isolato è un tensor di dimensione zero (uno scalare)
- Una lista di numeri è un tensor di una dimensione (un vettore)
- Una tabella di numeri è un tensor di due dimensioni (una matrice)
- Un'immagine a colori, o un batch di più immagini, è un tensor di tre o più dimensioni

Non importa con cosa lavori in PyTorch —numeri singoli, immagini, testo già convertito in valori numerici—: lavorerai sempre con tensor.

## Che cos'è autograd

Addestrare una rete neurale richiede di calcolare, per ogni peso, quanto ha contribuito all'errore finale: è ciò che si conosce come il gradient. Farlo a mano, derivando ogni operazione della rete, sarebbe impraticabile non appena l'architettura cresce un po'. **Autograd** è il modulo di differenziazione automatica di PyTorch: registra, in un grafo, ogni operazione che applichi su un tensor contrassegnato con `requires_grad=True`, e quando chiami `.backward()` percorre quel grafo all'indietro e calcola automaticamente il gradient di ogni tensor coinvolto.

Grazie ad autograd, non hai bisogno di programmare la retropropagazione (backpropagation) dell'errore passo dopo passo: definisci le operazioni in avanti (il forward pass) e PyTorch si incarica di derivarle all'indietro per te. Questa caratteristica, insieme al fatto che PyTorch costruisce il grafo di calcolo al volo —a differenza di librerie con un grafo statico definito in anticipo—, è ciò che rende il debug e la sperimentazione con PyTorch così diretto.

## Esempio di codice commentato

### Tensor di base

```python
import torch

# Un tensor di una dimensione (vettore), a partire da una lista di Python
x = torch.tensor([5, 3])
y = torch.tensor([2, 1])

# PyTorch sovraccarica gli operatori usuali: questo moltiplica elemento per elemento
print(x * y)  # tensor([10, 3])

# torch.zeros crea un tensor pieno di zeri con la forma (shape) che indichi
z = torch.zeros((2, 5))
print(z.shape)  # torch.Size([2, 5])

# view riorganizza gli stessi dati in una forma diversa, senza copiarli
z_riordinato = z.view(1, 10)
print(z_riordinato.shape)  # torch.Size([1, 10])
```

Nota che `x * y` non somma né concatena i tensor: moltiplica ogni posizione con la posizione equivalente dell'altro tensor (5×2=10, 3×1=3). Questa operazione elemento per elemento è la base di quasi tutto il calcolo numerico nel deep learning.

### Il tuo primo gradient con autograd

```python
# requires_grad=True dice a PyTorch di registrare ogni operazione su x
x = torch.tensor([2.0, 3.0], requires_grad=True)

y = x + 2
z = y * y * 3

# backward() percorre il grafo all'indietro e calcola il gradient di z rispetto a x
z.backward(torch.tensor([1.0, 1.0]))

print(x.grad)  # tensor([24., 30.])
```

Non è necessario risolvere questa derivata a mano per fidarti del risultato: l'importante è capire che `.backward()` è l'istruzione che attiva tutto il calcolo dei gradient, e che `x.grad` è dove PyTorch li lascia disponibili per l'ottimizzatore.

### Una rete neurale minima con nn.Module e nn.Linear

```python
import torch
import torch.nn as nn

class RedMinima(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(3, 1): uno strato completamente connesso che riceve tre
        # valori di input e produce un singolo valore di output
        self.capa = nn.Linear(3, 1)

    def forward(self, x):
        return self.capa(x)

modelo = RedMinima()

# Un singolo esempio di input, con tre caratteristiche
entrada = torch.tensor([[1.0, 2.0, 3.0]])

# Il forward pass: PyTorch chiama forward() automaticamente quando usi modelo(entrada)
salida = modelo(entrada)
print(salida)  # un tensor con un singolo valore, la previsione del modello

# Poiché salida dipende da pesi con requires_grad=True (per impostazione predefinita in nn.Linear),
# puoi propagare l'errore all'indietro e ottenere il gradient di ogni peso
salida.backward()
print(modelo.capa.weight.grad)  # gradient di ognuno dei tre pesi
```

Ogni rete neurale in PyTorch segue questo stesso pattern: una classe che eredita da `nn.Module`, strati definiti in `__init__` (qui, un singolo `nn.Linear`) e la logica del forward pass nel metodo `forward`. PyTorch costruisce il grafo di calcolo mentre esegue `forward`, così basta chiamare `.backward()` sulla salida per ottenere i gradient di tutti i pesi coinvolti, esattamente come nell'esempio di autograd precedente.

## Riassunto dei punti chiave

- L'apprendimento profondo è un sottocampo di ML che utilizza reti neurali con vari strati per imparare rappresentazioni dei dati direttamente, senza che tu definisca a mano quali caratteristiche contano
- Una rete neurale è formata da neuroni organizzati in strati (input, nascosti e output); ogni connessione ha un peso, e una funzione di attivazione introduce la non linearità necessaria per imparare pattern complessi
- Un tensor è la struttura centrale di PyTorch: un array multidimensionale che rappresenta qualsiasi dato, da un numero singolo a un batch di immagini, e che può essere eseguito su CPU (Central Processing Unit) o su GPU
- Autograd registra le operazioni sui tensor con `requires_grad=True` e calcola i loro gradient automaticamente quando chiami `.backward()`, evitando che tu debba derivare manualmente la retropropagazione
- In PyTorch, una rete neurale si definisce come una classe che eredita da `nn.Module`, con i suoi strati (ad esempio, `nn.Linear`) dichiarati in `__init__` e la logica del forward pass nel metodo `forward`

## Mettilo in pratica

Apri un notebook nuovo in Google Colab e importa `torch`. Crea due tensor di una dimensione con tre valori ciascuno, uno con `torch.tensor` e valori di tua scelta e un altro con `torch.randn((3,))` (valori casuali). Moltiplicali elemento per elemento e stampa il risultato insieme al suo `shape`. Poi, definisci una classe `RedMinima` uguale a quella dell'esempio di questo capitolo, ma con `nn.Linear(3, 1)` sostituito da `nn.Linear(3, 5)`: verifica con `print(modelo.capa.weight.shape)` quanti pesi ha ora lo strato e spiega, in un commento, perché quel numero è cambiato quando hai cambiato la dimensione di output.
