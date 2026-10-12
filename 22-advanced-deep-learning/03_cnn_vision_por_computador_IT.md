# Capitolo 3. Reti neurali convoluzionali: come vede una macchina

*Sigle usate in questo capitolo: rete neurale convoluzionale (CNN), unità lineare rettificata (ReLU), unità di elaborazione grafica (GPU), rosso-verde-blu (RGB), discesa del gradiente stocastica (SGD).*

## Perché questo tema è importante

Nel capitolo precedente hai imparato cos'è un neurone, come si organizza un perceptron multistrato e come l'algoritmo di retropropagazione regola i pesi della rete per minimizzare l'errore. Tutto questo funziona molto bene quando i tuoi dati sono righe di una tabella: età, reddito, numero di stanze. Ma appena provi ad applicare la stessa ricetta a un'immagine, l'approccio comincia a fare acqua. Eppure le immagini sono ovunque: sono il tipo di dato alla base di alcune delle applicazioni di intelligenza artificiale (IA) con l'impatto sociale più forte dell'ultimo decennio.

Pensa a tre esempi molto diversi tra loro:

- Un'auto a guida autonoma che deve distinguere in millisecondi un pedone da un lampione
- Un sistema di diagnosi medica che analizza una radiografia del torace per individuare segnali precoci di polmonite
- Un'app sul tuo telefono che riconosce automaticamente quale prodotto agricolo stai fotografando in campagna

In tutti e tre i casi, l'input è lo stesso: un'immagine, cioè una griglia di pixel. E in tutti e tre i casi la sfida è la stessa: estrarre da quella griglia schemi visivi rilevanti (bordi, texture, forme, oggetti), in modo che la decisione finale sia affidabile, rapida e generalizzi a immagini mai viste.

Questo capitolo ti presenta lo strumento che ha risolto quella sfida e che, più di un decennio dopo, resta la base della maggior parte dei sistemi di visione artificiale: la rete neurale convoluzionale (CNN). Al termine, capirai perché una rete completamente connessa non è la soluzione adatta alle immagini, che cosa fa esattamente un filtro convoluzionale, come si controlla la dimensione delle sue uscite con il passo (*stride*) e il riempimento (*padding*), a cosa serve il *pooling* e come tutto questo si combina in un'architettura che impara, strato dopo strato, dai bordi semplici fino agli oggetti completi.

## Spiegazione concettuale con analogie

### Il problema di "appiattire" un'immagine

Immagina di avere una fotografia di nove per nove pixel in bianco e nero. Per darla in pasto a una rete completamente connessa, come quella del capitolo precedente, dovresti trasformarla in un vettore: prendi ogni riga di pixel e le incolli una dopo l'altra, formando un'unica lista di ottantuno numeri. Questa operazione si chiama appiattire (*flatten*) l'immagine.

Il problema è che, appiattendo, distruggi la struttura spaziale. Due pixel che nell'immagine originale erano attaccati l'uno all'altro possono finire molto distanti nel vettore, mentre due pixel lontani tra loro possono finire vicini. La rete completamente connessa non sa più quali numeri siano vicini nello spazio reale: per lei sono soltanto una lista di valori.

Inoltre, emerge un problema pratico. Se colleghi ogni pixel di ingresso con ogni neurone del primo strato nascosto, il numero di parametri che la rete deve apprendere esplode. Un'immagine a colori di 32 x 32 pixel con tre canali di colore (rosso, verde e blu, o RGB) ha 32 x 32 x 3 = 3.072 valori di ingresso. Se il primo strato nascosto ha solo 1.000 neuroni, ti servono già 3.072 x 1.000 = 3.072.000 parametri, solo in quello strato. Scala questo a fotografie reali di diversi megapixel e il modello diventa ingestibile: troppi parametri, troppa memoria, troppo rischio di sovradattamento.

### Il modello che cerca un pattern

Ora pensa a come riconosceresti un occhio dentro una fotografia se dovessi farlo a mano. Non guarderesti l'intera immagine in un colpo solo: scorreresti la foto con una specie di modello mentale di "questo è un occhio" e verificheresti, regione per regione, se quella regione gli assomiglia. Il modello è sempre lo stesso, lo fai scorrere su tutta l'immagine, e dove si adatta meglio ottieni una risposta forte.

Questa è, letteralmente, l'idea di un filtro (o *kernel*) convoluzionale. È una piccola matrice di numeri, per esempio di dimensione 3 x 3, che scorre sull'immagine. In ogni posizione, moltiplica i suoi valori per i pixel che ha sotto e somma il risultato. Se quella regione dell'immagine assomiglia allo schema codificato nel filtro (un bordo verticale, un angolo, una macchia di colore), la somma dà un valore alto. Se non gli assomiglia, dà un valore basso. Il risultato di percorrere tutta l'immagine con quel modello è una mappa di attivazioni (*feature map*): una nuova immagine che indica dove ha trovato lo schema.

La differenza rispetto a un modello tradizionale della visione artificiale classica è che qui i valori del modello non vengono progettati a mano da una persona: li impara la rete durante l'addestramento, esattamente come impara i pesi di uno strato lineare.

### Due proprietà del mondo visivo che la CNN sfrutta

Questo progetto non è arbitrario: sfrutta due proprietà che quasi tutte le immagini naturali soddisfano.

- **Località**. I pixel vicini nell'immagine sono di solito correlati tra loro (fanno parte dello stesso oggetto o della stessa texture), mentre pixel molto distanti, come l'angolo in alto a sinistra e quello in basso a destra, normalmente non hanno una relazione diretta
- **Invarianza alla traslazione**. Uno schema significativo può comparire in qualunque posizione dell'immagine. Un uccello resta un uccello sia che appaia nell'angolo in alto sia al centro della foto. Questa proprietà non è esclusiva delle immagini: nell'audio, un fonema può comparire in qualsiasi istante; nel testo, una parola può comparire in qualsiasi posizione della pagina

Uno strato convoluzionale è progettato esattamente per sfruttare queste due proprietà: collega ogni unità di uscita solo con una regione locale dell'ingresso (località) e riutilizza gli stessi pesi in tutte le posizioni dell'immagine (invarianza alla traslazione). Questo riutilizzo dei pesi si chiama **condivisione dei pesi** (*weight sharing*), ed è la chiave che spieghiamo in dettaglio qui sotto.

## Sviluppo teorico

### Dallo strato completamente connesso allo strato localmente connesso

Per capire da dove viene uno strato convoluzionale, è utile costruirlo in due passi partendo da ciò che già conosci.

**Primo passo: limitare la connettività.** In uno strato completamente connesso, ogni unità di uscita è collegata a tutti i pixel di ingresso. In uno strato localmente connesso, ogni unità di uscita si collega solo con una piccola regione dell'ingresso, per esempio una finestra di 3 x 3 pixel. Questa regione si chiama **campo recettivo** dell'unità. Dato che ogni unità di uscita corrisponde a una regione diversa dell'immagine, l'insieme delle uscite conserva la topologia spaziale dell'ingresso: continua ad avere la forma di una griglia, non di un vettore.

Questo cambiamento riduce già drasticamente il numero di connessioni (e quindi di parametri), perché ogni unità di uscita ha bisogno solo di nove pesi (più un bias) invece di uno per ogni pixel dell'immagine intera.

**Secondo passo: condividere i pesi.** Uno strato localmente connesso ha ancora un problema: ogni unità di uscita, pur guardando una regione piccola, ha il proprio insieme di pesi, diverso da quello delle sue vicine. Ciò significa che la rete dovrebbe imparare separatamente a rilevare "un bordo verticale" nell'angolo in alto a sinistra e "un bordo verticale" al centro dell'immagine, anche se si tratta esattamente dello stesso schema.

La soluzione è obbligare tutte le unità di uscita a usare lo stesso insieme di pesi. Cioè, lo stesso modello 3 x 3 scorre su tutta l'immagine, applicando sempre gli stessi numeri. Questa operazione di "modello che scorre, moltiplica e somma" è, formalmente, la **convoluzione**, e si rappresenta con il simbolo dell'asterisco (*).

Grazie alla condivisione dei pesi, la rete diventa **equivariante alla traslazione**: se sposti lo schema in ingresso, l'uscita si sposta esattamente nello stesso modo, invece di dover imparare lo schema di nuovo in ogni posizione.

### L'operazione di convoluzione discreta

Sia $I$ l'immagine di ingresso (una matrice bidimensionale, o tridimensionale se consideri anche l'asse dei canali di colore) e $K$ il filtro o *kernel* (una piccola matrice, tipicamente di 3 x 3 o 5 x 5). Il valore di uscita nella posizione $(i, j)$ della mappa di attivazioni si calcola come:

$$
(I * K)(i, j) = \sum_{m} \sum_{n} I(i + m,\; j + n) \cdot K(m, n)
$$

In parole: posizioni il filtro $K$ sopra l'immagine $I$, centrato (o allineato) nella posizione $(i, j)$, moltiplichi ciascun peso del filtro per il pixel dell'immagine che ha sotto, e sommi tutti questi prodotti per ottenere un unico numero. Poi sposti il filtro alla posizione successiva e ripeti. Il risultato di percorrere tutta l'immagine è la mappa di attivazioni.

Quando l'immagine ha più canali di ingresso (per esempio i tre canali RGB), il filtro ha anche un sottofiltro per ogni canale. I contributi dei tre sottofiltri si sommano per produrre un'unica mappa di attivazioni per filtro. Inoltre, in uno strato convoluzionale si apprendono più filtri contemporaneamente, ciascuno specializzato nel rilevare uno schema diverso (un bordo orizzontale, una macchia rossa, una texture granulosa); l'insieme delle mappe di attivazioni che producono si impila formando i **canali di uscita**. Per questo, sia l'ingresso sia l'uscita di uno strato convoluzionale sono in realtà tensori tridimensionali: altezza, larghezza e numero di canali.

Dopo la convoluzione si applica, come in uno strato lineare, una funzione di attivazione non lineare, quasi sempre la ReLU:

$$
\text{ReLU}(x) = \max(0, x)
$$

La ReLU lascia passare senza modifiche i valori positivi (attivazioni forti, cioè "qui c'è uno schema simile al filtro") e azzera quelli negativi. È economica da calcolare ed evita il problema di saturazione di altre funzioni come la sigmoide, il che facilita l'addestramento di reti più profonde.

### Dimensione dell'uscita: kernel, stride e padding

Un dettaglio importante che devi controllare è quanto si riduce (o non si riduce) la dimensione spaziale dell'immagine attraversando uno strato convoluzionale. Dipende da tre parametri: la dimensione del *kernel*, il passo (*stride*) e il riempimento (*padding*).

**Convoluzione valida (*valid*).** È la variante più semplice: calcoli l'uscita solo nelle posizioni in cui il filtro entra completamente nell'immagine, senza uscirne. Se hai un'immagine di ingresso di $N \times N$ pixel e un filtro di $F \times F$, il filtro ha solo $N - F + 1$ posizioni possibili per ogni dimensione. Per esempio, con un'immagine di 9 x 9 e un filtro di 3 x 3, ottieni una mappa di attivazioni di 7 x 7: l'uscita è sempre più piccola dell'ingresso.

**Riempimento (*padding*) e convoluzione "uguale" (*same*).** Se impili più convoluzioni valide di seguito, l'immagine si restringe strato dopo strato, cosa non sempre desiderabile. La soluzione è aggiungere una fascia di zeri attorno al bordo dell'immagine prima di applicare il filtro: questo è il *padding*. Quando aggiungi esattamente il riempimento necessario perché l'uscita abbia la stessa dimensione dell'ingresso, parli di convoluzione "uguale" (*same*). Per un filtro di 3 x 3 basta una riga di zeri di spessore uno attorno a tutta l'immagine. Questa variante funziona in modo naturale con filtri di dimensione dispari (3 x 3, 5 x 5), perché il riempimento può ripartirsi in modo simmetrico su entrambi i lati.

All'estremo opposto c'è la convoluzione "completa" (*full*), che aggiunge il massimo riempimento possibile: calcola un'uscita in qualunque posizione in cui il filtro tocchi almeno un pixel dell'immagine. Il risultato è un'uscita più grande dell'ingresso ($N + F - 1$), ma si usa poco nella pratica della classificazione di immagini.

La formula generale che combina kernel, *padding* ($P$) e *stride* ($S$) per calcolare la dimensione di uscita $O$ a partire da un ingresso di dimensione $N$ è:

$$
O = \left\lfloor \frac{N - F + 2P}{S} \right\rfloor + 1
$$

**Passo o *stride*.** Finora abbiamo supposto che il filtro scorra di un pixel alla volta. Lo *stride* controlla quel salto: con *stride* uguale a due, il filtro avanza di due pixel alla volta, calcolando solo la metà delle posizioni in ogni dimensione. Questo ha due effetti: riduce il costo computazionale (con *stride* due, il calcolo si riduce circa di un fattore quattro) e riduce la risoluzione della mappa di attivazioni, cosa utile per costruire una gerarchia di caratteristiche sempre più astratte e a minor risoluzione spaziale. Per esempio, una convoluzione valida con *stride* due su un'immagine di 9 x 9 con un filtro di 3 x 3 produce una mappa di uscita di 4 x 4.

Esiste anche la **convoluzione dilatata**, una variante in cui non si saltano posizioni dell'immagine, ma valori dello stesso filtro: tra i pesi del *kernel* si inseriscono degli spazi vuoti. Questo amplia il campo recettivo (la regione dell'immagine originale che "vede" ciascuna unità di uscita) senza aumentare il numero di parametri del filtro, utile quando vuoi catturare un contesto più ampio senza rendere il calcolo più costoso.

### Pooling: ridurre la risoluzione conservando l'essenziale

Il *pooling* è un'altra operazione classica per ridurre la dimensione spaziale delle mappe di attivazioni, complementare (o alternativa) alle convoluzioni con *stride*. A differenza della convoluzione, il *pooling* non ha parametri da apprendere: applica una funzione fissa (massimo o media) su piccole finestre dell'immagine.

La variante più diffusa è il **max pooling**: divide la mappa di attivazioni in finestre piccole, di solito di 2 x 2 e senza sovrapposizione, e conserva il valore massimo di ciascuna finestra:

$$
y(i, j) = \max_{(m, n) \,\in\, \text{finestra}(i, j)} x(m, n)
$$

Con una finestra di 2 x 2 e passo due, l'altezza e la larghezza della mappa di attivazioni si dimezzano, mentre il numero di canali resta invariato. Per esempio, una mappa di 32 x 32 x 32 diventa 16 x 16 x 32 dopo un *max pooling* di 2 x 2.

Il *max pooling* offre tre vantaggi pratici:

- Riduce il costo computazionale e la memoria necessari negli strati successivi
- Fornisce un certo margine di invarianza a piccole traslazioni: se lo schema si sposta di un pixel all'interno della finestra, il valore massimo selezionato cambia appena
- Riducendo la risoluzione, diminuisce anche il numero di parametri negli strati completamente connessi che di solito si trovano alla fine della rete, il che aiuta a controllare il sovradattamento

Esiste anche l'**average pooling** (calcola la media invece del massimo, producendo un effetto di sfumatura) e il **global pooling**, che riduce ogni mappa di attivazioni completa a un unico valore, molto usato subito prima dello strato di classificazione finale nelle architetture moderne.

### Come si costruisce una CNN strato per strato

Una CNN tipica non è altro che una sequenza di blocchi che combinano questi elementi, impilati uno dopo l'altro:

1. **Strati convoluzionali**, ciascuno seguito da una non linearità (ReLU), che estraggono mappe di attivazioni sempre più ricche
2. **Strati di *pooling*** (o convoluzioni con *stride*), intercalati ogni certo numero di strati convoluzionali, che riducono la risoluzione spaziale
3. Dopo diversi giri di convoluzione e *pooling*, le mappe di attivazione finali si **appiattiscono** in un vettore
4. Quel vettore alimenta uno o più **strati completamente connessi** (gli strati lineari che già conosci), che svolgono il ragionamento di alto livello
5. Un ultimo strato produce i punteggi di uscita, di solito passati attraverso una funzione *softmax* per ottenere una distribuzione di probabilità sulle classi

Questo schema (convoluzione, non linearità, *pooling*, ripetuto più volte, seguito da strati completamente connessi) comparve già in una delle prime CNN con successo pratico, LeNet-5, progettata alla fine degli anni novanta per riconoscere cifre scritte a mano, e resta, nella sostanza, lo scheletro di architetture molto più recenti.

L'addestramento di una CNN segue esattamente la stessa procedura che hai imparato con il perceptron multistrato: definisci una funzione di perdita (per la classificazione, quasi sempre l'entropia incrociata), calcoli le previsioni con un passaggio in avanti, propaghi l'errore all'indietro con la retropropagazione e aggiorni i pesi, inclusi i valori dei filtri convoluzionali, tramite discesa del gradiente stocastica (SGD) o una delle sue varianti. La differenza non sta nel modo in cui si addestra, ma nel modo in cui è progettata l'architettura che si addestra.

### Perché le CNN catturano gerarchie spaziali

Forse la proprietà più interessante delle CNN è che, impilando più strati convoluzionali, la rete costruisce in modo naturale una gerarchia di caratteristiche sempre più astratte.

Ogni strato convoluzionale agisce, in sostanza, come un classificatore lineare locale. Da solo, rileva schemi relativamente semplici. Ma quando impili un secondo strato convoluzionale sull'uscita del primo, ogni unità di quel secondo strato non guarda direttamente i pixel originali, bensì le mappe di attivazioni già elaborate dal primo strato. Poiché ogni unità del primo strato aveva un campo recettivo di, per esempio, 3 x 3 pixel, e il secondo strato guarda di nuovo una finestra di 3 x 3 su quelle uscite, il campo recettivo effettivo del secondo strato, misurato rispetto all'immagine originale, cresce fino a 5 x 5. Se continui a impilare strati di 3 x 3, ogni nuovo strato amplia il campo recettivo effettivo, permettendo alla rete di "vedere" regioni sempre più grandi dell'immagine senza dover usare filtri grandi fin dall'inizio.

Questa crescita del campo recettivo, combinata con la riduzione di risoluzione del *pooling*, produce la gerarchia caratteristica delle CNN che si osserva empiricamente quando si visualizza che cosa attiva ciascuno strato:

- I **primi strati** imparano rilevatori molto semplici e generici: bordi, variazioni di intensità, macchie di colore
- Gli **strati intermedi** combinano questi rilevatori semplici in schemi più elaborati: texture, angoli, contorni caratteristici
- Gli **strati profondi** combinano questi schemi intermedi in rappresentazioni di parti di oggetti e, infine, di oggetti completi: una ruota, un occhio, un intero volto

Questa composizione (schemi semplici che si combinano in schemi sempre più complessi) riflette proprio la natura composizionale delle immagini del mondo reale: un oggetto è formato da parti, e quelle parti sono formate da texture e bordi. Impilare strati convoluzionali con *pooling* significa, in sostanza, impilare livelli di astrazione.

Va sottolineato che le CNN, per progetto, sono robuste solo rispetto alle traslazioni (grazie alla condivisione dei pesi e al *pooling*). Non sono automaticamente invarianti a rotazioni, cambi di scala o di illuminazione. Per ottenere quella robustezza aggiuntiva, nella pratica si ricorre all'**aumento dei dati** (*data augmentation*): durante l'addestramento si generano versioni ruotate, scalate o ritagliate delle stesse immagini, in modo che la rete impari quell'invarianza dagli esempi, invece di averla incorporata nell'architettura.

## Esempio di codice commentato, in PyTorch

Il seguente esempio definisce una piccola CNN e la addestra su FashionMNIST, un insieme di immagini in scala di grigi di 28 x 28 pixel con dieci categorie di capi di abbigliamento, disponibile direttamente in `torchvision`. **Copia il codice in una cella di Google Colab ed eseguilo** per vedere la rete addestrarsi dall'inizio alla fine.

```python
# Instala/actualiza torchvision si hace falta (normalmente ya viene en Colab)
# !pip install torch torchvision --quiet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Usa GPU si Colab te la ha asignado, si no, usa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de entrenamiento: {device}")

# --- 1. Datos ---
# Normalizamos con la media y desviación típica habituales de FashionMNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

clases = [
    "camiseta", "pantalón", "jersey", "vestido", "abrigo",
    "sandalia", "camisa", "zapatilla", "bolso", "botín",
]

# --- 2. Definición de la CNN ---
class CNNSimple(nn.Module):
    def __init__(self, num_clases=10):
        super().__init__()
        # Bloque convolucional 1: 1 canal de entrada (escala de grises) -> 16 filtros
        # kernel 3x3, padding "same" (padding=1) para no reducir el tamaño aquí
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                                kernel_size=3, stride=1, padding=1)
        # Bloque convolucional 2: 16 -> 32 filtros
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                                kernel_size=3, stride=1, padding=1)
        # Max pooling 2x2: reduce el alto y el ancho a la mitad cada vez
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Tras dos pooling de 2x2 sobre una entrada de 28x28: 28 -> 14 -> 7
        # El último mapa de activaciones tiene forma (32 canales, 7, 7)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_clases)

        self.dropout = nn.Dropout(0.25)  # mitiga el sobreajuste

    def forward(self, x):
        # x llega con forma (lote, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # -> (lote, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (lote, 32, 7, 7)

        x = torch.flatten(x, start_dim=1)     # aplana todo menos el eje de lote
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                       # logits, sin softmax explícito
        return x

modelo = CNNSimple(num_clases=10).to(device)
print(modelo)

# --- 3. Función de pérdida y optimizador ---
# CrossEntropyLoss combina softmax + entropía cruzada en un solo paso
criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

# --- 4. Bucle de entrenamiento ---
def entrenar_una_epoca(modelo, loader, criterio, optimizador):
    modelo.train()
    perdida_total = 0.0
    for imagenes, etiquetas in loader:
        imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)

        optimizador.zero_grad()
        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * imagenes.size(0)
    return perdida_total / len(loader.dataset)

# --- 5. Evaluación ---
def evaluar(modelo, loader, criterio):
    modelo.eval()
    perdida_total = 0.0
    aciertos = 0
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            salidas = modelo(imagenes)
            perdida_total += criterio(salidas, etiquetas).item() * imagenes.size(0)
            predicciones = salidas.argmax(dim=1)
            aciertos += (predicciones == etiquetas).sum().item()
    perdida_media = perdida_total / len(loader.dataset)
    exactitud = aciertos / len(loader.dataset)
    return perdida_media, exactitud

# --- 6. Entrena durante unas pocas épocas ---
num_epocas = 5
for epoca in range(1, num_epocas + 1):
    perdida_train = entrenar_una_epoca(modelo, train_loader, criterio, optimizador)
    perdida_test, exactitud_test = evaluar(modelo, test_loader, criterio)
    print(f"Época {epoca}/{num_epocas} | "
          f"pérdida train: {perdida_train:.4f} | "
          f"pérdida test: {perdida_test:.4f} | "
          f"exactitud test: {exactitud_test:.2%}")

# --- 7. Prueba rápida sobre una imagen del conjunto de test ---
modelo.eval()
imagen_ejemplo, etiqueta_real = test_dataset[0]
with torch.no_grad():
    logits = modelo(imagen_ejemplo.unsqueeze(0).to(device))  # añade eje de lote
    prediccion = logits.argmax(dim=1).item()

print(f"Clase real: {clases[etiqueta_real]} | Clase predicha: {clases[prediccion]}")
```

Con cinque epoche e questa architettura così piccola, è normale raggiungere tra il 90 % e il 91 % di accuratezza sul set di prova in pochi minuti, anche sulla GPU gratuita di Colab. **Sperimenta cambiando il numero di filtri, aggiungendo un terzo blocco convoluzionale o provando `AvgPool2d` al posto di `MaxPool2d`** per verificare in prima persona come ciascuna scelta influisce sul rendimento finale.

## Riepilogo dei punti chiave

- Una rete completamente connessa non scala bene con le immagini, perché distrugge la struttura spaziale appiattendo l'ingresso e richiede un numero enorme di parametri
- Le immagini naturali soddisfano due proprietà chiave che le CNN sfruttano: la località (i pixel vicini sono correlati) e l'invarianza alla traslazione (uno schema può comparire in qualsiasi posizione)
- Un filtro o *kernel* convoluzionale è una piccola matrice di pesi apprendibili che scorre sull'immagine, funzionando come un modello che cerca uno schema specifico in ogni regione
- La condivisione dei pesi (usare lo stesso filtro su tutta l'immagine) riduce drasticamente il numero di parametri e rende la rete equivariante alla traslazione
- La convoluzione discreta si calcola come $(I * K)(i, j) = \sum_{m}\sum_{n} I(i+m, j+n) \cdot K(m, n)$, e la dimensione dell'uscita dipende da *kernel*, *stride* e *padding* secondo $O = \lfloor (N - F + 2P) / S \rfloor + 1$
- Il *padding* controlla se l'uscita si riduce (convoluzione valida), mantiene la dimensione (convoluzione "uguale") o cresce (convoluzione completa); lo *stride* controlla ogni quanti pixel si sposta il filtro
- Il *pooling*, tipicamente *max pooling* di 2 x 2, riduce la risoluzione spaziale delle mappe di attivazione senza aggiungere parametri, con un guadagno di efficienza computazionale e una certa invarianza a piccole traslazioni
- Una CNN si costruisce impilando blocchi di convoluzione, non linearità e *pooling*, per terminare con uno o più strati completamente connessi che producono la classificazione finale
- Impilando strati convoluzionali, il campo recettivo effettivo cresce, il che consente alla rete di costruire una gerarchia di caratteristiche: bordi e texture nei primi strati, forme e parti di oggetti negli strati intermedi, oggetti completi negli strati profondi
- Le CNN sono intrinsecamente robuste solo rispetto alle traslazioni; la robustezza rispetto a rotazioni, scale o cambi di illuminazione si ottiene con l'aumento dei dati durante l'addestramento

## Mettilo in pratica

**Dedica tra quindici e trenta minuti a questo esercizio prima di passare al capitolo successivo.**

1. **Copia il codice della sezione precedente in un notebook di Google Colab ed eseguilo così com'è**, per confermare che riproduci un'accuratezza simile (intorno al 90 %) sul set di prova di FashionMNIST
2. **Cambia il set di dati** da `datasets.FashionMNIST` a `datasets.MNIST` (cifre scritte a mano) senza modificare nient'altro dell'architettura, e verifica se l'accuratezza sale o scende. Rifletti: perché pensi che lo stesso modello si comporti in modo diverso sui due insiemi?
3. **Modifica la dimensione del *kernel*** dei due strati convoluzionali, passando da `kernel_size=3` a `kernel_size=5` (regola anche `padding=2` per mantenere una convoluzione "uguale"). Annota se cambia l'accuratezza finale e il tempo di addestramento per epoca
4. **Sostituisci uno strato `nn.MaxPool2d(2, 2)` con `nn.AvgPool2d(2, 2)`** e osserva se il risultato varia
5. **Aggiungi una stampa del numero totale di parametri del modello** con `sum(p.numel() for p in modelo.parameters())` e confrontalo mentalmente con i 3.072.000 parametri che abbiamo calcolato nella sezione teorica per un singolo strato completamente connesso su un'immagine piccola di 32 x 32 pixel a colori. Questo confronto è la migliore dimostrazione pratica del perché la condivisione dei pesi conta
