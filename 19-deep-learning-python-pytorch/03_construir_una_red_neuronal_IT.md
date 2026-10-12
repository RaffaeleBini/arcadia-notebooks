# 03. Costruire una rete neurale con PyTorch

## Perché questo argomento è importante

Nel capitolo precedente hai imparato a rappresentare i tuoi dati con `Dataset` e a servirli in batch con `DataLoader`. Hai quindi la materia prima pronta. Ciò che manca è l'elemento che dà nome a tutto il campo: la rete neurale stessa, il pezzo che riceverà quei batch di dati e imparerà a trasformarli in previsioni.

Costruire una rete neurale in PyTorch non è magia né richiede algebra avanzata: è definire una classe di Python con due parti molto concrete. Una volta le capisci, puoi costruire dal classificatore più semplice fino ad architetture molto più sofisticate, perché il pattern di fondo non cambia.

## Da neuroni a architettura: l'anatomia di una rete

Pensa a una rete neurale come a una catena di montaggio organizzata in impianti o livelli. Ogni livello è uno **strato** (*layer*), e all'interno di ogni strato lavorano varie unità piccole chiamate **neuroni**. Ogni neurone fa qualcosa di molto semplice: riceve vari numeri di input, calcola una somma ponderata di quei numeri (ogni input moltiplicato per un **peso**, più un valore di aggiustamento chiamato **bias**) e consegna un singolo numero di output. Nessun neurone per conto proprio è intelligente; ciò che aggiunge valore è la combinazione di migliaia di queste operazioni semplici incatenate tra strati.

Una rete neurale completa organizza i suoi strati in tre tipi:

- **Strato di input**: riceve i dati grezzi, ad esempio i pixel di un'immagine già appiattiti in un vettore
- **Strati nascosti**: sono tra l'input e l'output, e sono quelli che estraggono pattern sempre più complessi a partire dai dati
- **Strato di output**: consegna il risultato finale, ad esempio un punteggio per ognuna delle classi che vuoi distinguere

A questa organizzazione completa —quanti strati ci sono, quanti neuroni ha ciascuno e come si collegano— la chiamiamo **architettura** della rete. Proprio come due edifici possono usare gli stessi mattoni e risultare completamente diversi a seconda di come li posizioni, due reti neurali possono usare gli stessi pezzi di PyTorch e comportarsi in modo molto diverso a seconda dell'architettura che scegli.

## `nn.Module`: il progetto della tua rete

In PyTorch, ogni rete neurale si definisce come una classe di Python che eredita da `nn.Module`. Pensa a `nn.Module` come a un progetto di costruzione: ti obbliga a organizzare la tua rete in due blocchi ben differenziati.

- Il metodo `__init__` è dove **dichiari** i pezzi della tua rete: quali strati esistono e quale dimensione hanno. È come fare l'inventario dei materiali prima di costruire
- Il metodo `forward` è dove **colleghi** quei pezzi: in quale ordine i dati passano attraverso ogni strato. È come il progetto di montaggio che dice quale pezzo viene dopo quale

All'interno di `__init__`, la prima riga è sempre `super().__init__()`. Questa riga chiama il costruttore di `nn.Module`, la classe da cui erediti, ed è essenziale: senza di essa, PyTorch non può registrare correttamente gli strati che definisci in seguito, e la tua rete neurale rimane male inizializzata.

Quando istanzi la tua classe (`modelo = MiaRete()`), PyTorch assegna automaticamente valori casuali iniziali ai pesi e ai bias di ogni strato. Questa inizializzazione casuale non è una negligenza: se tutti i neuroni iniziassero con lo stesso valore, imparerebbero esattamente la stessa cosa tra loro e la rete perderebbe tutta la sua capacità di imparare pattern variati. Iniziare in modo casuale "rompe la simmetria" tra i neuroni e dà a ognuno l'opportunità di specializzarsi in qualcosa di diverso durante l'addestramento (che vedrai nella lezione successiva).

## `nn.Linear`: strati completamente connessi

Il pezzo più basilare per costruire strati è `nn.Linear`. Uno strato `nn.Linear` è quello che si chiama uno strato **completamente connesso** (*fully connected*) o **denso**: ogni neurone dello strato riceve l'output di assolutamente tutti i neuroni dello strato precedente.

```python
import torch.nn as nn

# Uno strato completamente connesso: riceve 784 valori di input e produce 64 di output
strato = nn.Linear(in_features=784, out_features=64)
```

`in_features` è il numero di valori che arrivano allo strato (deve coincidere con l'output dello strato precedente, o con la dimensione dei tuoi dati se è il primo strato). `out_features` è il numero di neuroni che ha questo strato, cioè quanti valori produce. Internamente, `nn.Linear` conserva una matrice di pesi e un vettore di bias: sono precisamente quei numeri che la rete andrà aggiustando durante l'addestramento.

## Funzioni di attivazione: perché sono necessarie

Se concateni vari strati `nn.Linear` uno dietro l'altro senza nulla in mezzo, matematicamente il risultato è ancora equivalente a una singola trasformazione lineare, non importa quanti strati impili. Una rete così non potrebbe mai imparare pattern complessi come quelli che distinguono una cifra scritta a mano da un'altra.

Qui è dove entrano le **funzioni di attivazione**: un'operazione matematica che si applica all'output di uno strato prima di passarlo al prossimo, e introduce non linearità nel processo. Pensa a una funzione di attivazione come a una valvola che decide, neurone per neurone, quanta segnale lascia passare: senza di essa, l'informazione fluirebbe in modo troppo rigido e prevedibile; con essa, la rete può imparare relazioni molto più ricche tra i dati.

La funzione di attivazione più utilizzata negli strati nascosti è l'**unità lineare rettificata** (ReLU, *Rectified Linear Unit*): lascia passare senza cambiamenti i valori positivi e converte in zero qualsiasi valore negativo. È un'operazione molto economica da calcolare e, pur essendo così semplice, funziona bene nella stragrande maggioranza delle architetture.

Per lo strato di output di un problema di classificazione con più categorie si usa un'altra funzione diversa: `log_softmax`. Mentre ReLU si applica negli strati nascosti per dare flessibilità alla rete, `log_softmax` converte i valori finali in qualcosa di interpretabile come una distribuzione di probabilità, dove i dieci output (uno per ogni cifra, da zero a nove) sommano insieme la probabilità totale distribuita tra le dieci classi.

## Esempio di codice commentato

Definirai un classificatore per il dataset MNIST che hai già caricato nella lezione precedente: riceve un'immagine di cifra appiattita (784 valori, risultato di appiattire un'immagine di 28x28 pixel) e produce un punteggio per ognuno dei dieci cifre possibili (0-9).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificatoreMNIST(nn.Module):
    def __init__(self):
        # Chiamata obbligatoria al costruttore di nn.Module
        super().__init__()

        # Dichiariamo gli strati: ogni nn.Linear collega completamente
        # lo strato precedente con il prossimo
        self.fc1 = nn.Linear(784, 128)  # input -> primo strato nascosto
        self.fc2 = nn.Linear(128, 64)   # primo strato nascosto -> secondo strato nascosto
        self.fc3 = nn.Linear(64, 10)    # secondo strato nascosto -> output (10 cifre)

    def forward(self, x):
        # Applichiamo ReLU dopo ogni strato nascosto per introdurre non linearità
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Lo strato di output non ha ReLU: usiamo log_softmax per ottenere
        # una distribuzione di probabilità sulle 10 classi
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


# Istanziamo la rete: qui si inizializzano pesi e bias a caso
modelo = ClassificatoreMNIST()
print(modelo)

# Simuliamo un batch di 16 immagini MNIST già appiattite (16 x 784)
batch_esempio = torch.randn(16, 784)

# Il forward pass: i dati attraversano la rete e otteniamo una previsione
salida = modelo(batch_esempio)
print("Forma della salida:", salida.shape)  # torch.Size([16, 10])
```

Nota il `dim=1` di `log_softmax`: il batch completo è un tensor di due dimensioni (16 immagini x 10 classi), e vuoi che la distribuzione di probabilità si calcoli per riga (dimensione 1, le 10 classi di ogni immagine), non mescolando le 16 immagini del batch tra loro.

Anche se in questo capitolo non addestri ancora la rete (arriva nella lezione successiva), questo `forward` che hai appena eseguito è già metà del lavoro. Dietro le quinte, PyTorch usa uno strumento chiamato **autograd** per registrare, in modo automatico, ogni operazione matematica che accade durante il `forward`. Questo registro è esattamente quello che avrà bisogno l'addestramento per calcolare i gradient e aggiustare i pesi: non devi programmare tu quella parte, solo definire bene l'architettura e il `forward`.

Un vantaggio aggiuntivo di definire il `forward` come un metodo normale di Python è che puoi includervi logica condizionale (`if`, cicli, quel che serve), qualcosa che altre librerie di apprendimento profondo non sempre permettono con la stessa libertà. Per ora ti basta concatenare gli strati in ordine, ma è bene sapere che quella flessibilità è disponibile quando ne avrai bisogno.

## Riassunto dei punti chiave

- Una rete neurale si organizza in strati (input, nascosti e output) formati da neuroni che combinano pesi e bias
- Ogni rete in PyTorch si definisce come una classe che eredita da `nn.Module`, con `__init__` per dichiarare gli strati e `forward` per collegare il flusso dei dati tra loro
- `super().__init__()` è obbligatorio all'inizio di `__init__`: senza di esso, `nn.Module` non può registrare correttamente gli strati
- `nn.Linear(in_features, out_features)` crea uno strato completamente connesso, con pesi e bias che si inizializzano a caso
- Le funzioni di attivazione (come ReLU) introducono non linearità tra strati; senza di esse, concatenare strati `nn.Linear` non aggiunge più capacità di apprendimento
- Per classificazione in più classi, `log_softmax` nello strato di output converte i punteggi in una distribuzione di probabilità
- Il `forward` pass attiva già **autograd**, il meccanismo che PyTorch userà per calcolare i gradient automaticamente durante l'addestramento

## Mettilo in pratica

Apri un notebook nuovo in Google Colab e definisci la tua propria classe ereditando da `nn.Module`, diversa da quella dell'esempio: invece di 128 e 64 neuroni negli strati nascosti, usa 32 e 16. Ricorda di mantenere l'input a 784 (i pixel di un'immagine MNIST appiattita) e l'output a 10 (le cifre da 0 a 9). Istanzia la tua rete, stampala con `print()` per vederne la struttura, e fai un `forward` pass con un batch simulato di 8 immagini usando `torch.randn(8, 784)`. Verifica che la forma dell'output è quella che ti aspetti prima di continuare con la lezione successiva, dove per fine addestri questa architettura.
