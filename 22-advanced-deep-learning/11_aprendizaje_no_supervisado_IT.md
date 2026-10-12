# Capitolo 11. Apprendimento non supervisionato e apprendimento di rappresentazioni

*Acronimi usati in questo capitolo: intelligenza artificiale (IA), apprendimento per rinforzo (RL), rete Q profonda (DQN), macchina di Boltzmann ristretta (RBM), analisi delle componenti principali (PCA), autoencoder variazionale (VAE), limite inferiore dell'evidenza (ELBO), divergenza di Kullback-Leibler (KL), rete generativa avversaria (GAN), rappresentazioni bidirezionali di encoder da transformer (BERT), codifica predittiva contrastiva (CPC), errore quadratico medio (MSE), autoencoder variazionale quantizzato vettorialmente (VQ-VAE), rete di query generativa (GQN).*

## Perché questo argomento è importante

Finora, in questo manuale, quasi tutto ciò che hai addestrato ha richiesto etichette: un'immagine e la sua classe, una sequenza e la sua traduzione. Questo approccio funziona bene, ma ha un costo di cui si parla raramente: qualcuno ha dovuto sedersi a etichettare ogni esempio. Classificare il milione di immagini di ImageNet ha richiesto migliaia di ore di lavoro umano, e in ambiti come la medicina l'etichettatura richiede anche personale esperto: una radiologa che esamina migliaia di radiografie per segnalare i segni di polmonite non è una risorsa a buon mercato.

Nel frattempo, il mondo produce una quantità quasi infinita di dati non etichettati: video, audio, testo, registrazioni di sensori. Se il tuo modello impara solo da ciò che è etichettato, sprechi la stragrande maggioranza di quell'informazione. La domanda che attraversa questo capitolo è semplice da formulare e difficile da rispondere: può un modello imparare qualcosa di utile osservando solo i dati, senza che nessuno gli dica che cos'è ciascuna cosa?

Questa è la terza grande famiglia del machine learning, accanto all'apprendimento supervisionato e all'apprendimento per rinforzo (RL), e si distingue dalle altre due per il tipo di feedback che riceve. Nell'apprendimento supervisionato, ogni ingresso arriva con la sua etichetta corretta. Nel RL, l'agente riceve una ricompensa scalare, quasi sempre solo alla fine di un compito. Nell'apprendimento non supervisionato non c'è un segnale di questo tipo: hai solo i dati di ingresso e devi decidere che cosa farne, il che solleva due domande: ci serve davvero, se l'apprendimento supervisionato end-to-end già funziona con dati e calcolo sufficienti? E se la risposta è sì, come sai se il tuo algoritmo ha fatto un buon lavoro, senza nessuna risposta corretta con cui confrontarlo?

La storia recente dell'intelligenza artificiale (IA) offre indizi sulla prima domanda. Prima del 2012, l'apprendimento di rappresentazioni era un lavoro manuale: ingegnerizzazione delle feature, metodi a kernel e, dal 2006, l'uso di macchine di Boltzmann ristrette (RBM) per pre-addestrare reti profonde in modo non supervisionato, una tecnica di Hinton e Salakhutdinov che migliorava la convergenza dei classificatori. Ma nel 2012 AlexNet ha vinto la sfida ImageNet con un margine enorme senza alcun pre-addestramento non supervisionato, appoggiandosi solo a più dati, reti più profonde e hardware migliore, il che ha portato a pensare che bastasse l'apprendimento supervisionato end-to-end.

Dieci anni dopo sono apparse crepe evidenti. L'efficienza sui dati resta un problema: gli algoritmi di RL profondo, come la rete Q profonda (DQN), richiedono ordini di grandezza più esperienza di una persona per imparare compiti semplici. Anche la robustezza viene meno: un rumore quasi impercettibile sull'immagine di un panda basta perché un classificatore all'avanguardia la etichetti come gibbone con più del 99 % di confidenza. E la generalizzazione è debole: agenti addestrati con decine di migliaia di varianti di un semplice videogioco rendono molto peggio sui livelli nuovi rispetto a quelli di addestramento.

Queste limitazioni sono l'argomento che Yann LeCun, Geoffrey Hinton e Yoshua Bengio, tutti e tre premi Turing, hanno sostenuto alla conferenza AAAI: prima di risolvere un compito concreto, conviene imparare a rappresentare il mondo, come sembra fare un bambino molto prima di imparare qualsiasi compito specifico. Con un modello del mondo dotato delle astrazioni adeguate, un cambiamento dell'ambiente si riduce a una o poche variabili, e ti adatti a esso con molti meno dati.

Questo capitolo percorre come si cerca di ottenerlo: prima con il clustering e la riduzione della dimensionalità, poi con gli autoencoder come esempio centrale di rappresentazione appresa per ricostruzione, e infine con il loro collegamento verso l'apprendimento autosupervisionato e contrastivo, la linea che oggi domina l'apprendimento di rappresentazioni moderno, da BERT fino ai modelli di visione addestrati senza alcuna etichetta.

## Spiegazione concettuale con analogie

### Comprimere per conservare l'essenziale

Immagina di dover descrivere una fotografia in un messaggio di testo molto breve. Non puoi inviare i pixel: devi riassumere, qualcosa come "una persona con giacca rossa, in piedi in un paesaggio innevato, sorridente". Con quel riassunto, l'altra persona si fa un'idea ragionevolmente fedele della foto, anche se non ricostruisce gli stessi pixel.

Un autoencoder fa qualcosa di simile, in modo automatico e numerico. È composto da due parti:

- Un **encoder**, che comprime l'immagine completa in un vettore molto più piccolo, la rappresentazione latente o *codice*, proprio come tu hai compresso la foto in una frase
- Un **decoder**, che prende quel vettore compresso e ricostruisce l'immagine originale a partire da esso, proprio come l'altra persona si immagina la foto a partire dalla tua frase

All'inizio dell'addestramento, il decoder produce immagini sfocate e poco fedeli, come un riassunto fatto male che perde dettagli importanti. Ma, regolando i pesi di entrambe le parti per minimizzare l'errore di ricostruzione, la rete impara quali informazioni vale la pena conservare nel riassunto e quali si possono scartare. Il vettore latente finisce per catturare i tratti più informativi dei dati, senza che nessuno gli abbia detto esplicitamente quali sono.

### Il modo di scrivere trentasette

Pensa a tre modi di scrivere il numero trentasette: la cifra araba "37", i numeri romani "XXXVII" e il binario "100101". Tutte e tre rappresentano la stessa quantità, ma ciascuna forma facilita operazioni diverse: la cifra araba mostra a colpo d'occhio la scomposizione in decine e unità, il binario la scomposizione in potenze di due, e i numeri romani quasi non facilitano nessuna operazione aritmetica.

Questa è l'idea centrale di "rappresentazione": il modo in cui rappresenti l'informazione è indipendente dall'informazione stessa, e scegliere bene quella forma semplifica il lavoro successivo. Imparare una buona rappresentazione di un'immagine, di un testo o dello stato di un videogioco non significa solo comprimere dati: significa trovare la forma che rende più semplici i compiti successivi (classificare, raggruppare, pianificare).

### Il mixer con una manopola per fattore

L'idea di rappresentazione "disgrovigliata" (*disentangled*), che compare più avanti, si capisce bene con l'analogia di un mixer audio con una manopola indipendente per il volume, un'altra per i bassi, un'altra per gli acuti: ruotare la manopola dei bassi cambia solo i bassi.

Immagina ora un pannello mal progettato in cui ruotare la manopola dei bassi alza anche il volume e distorce gli acuti: è "groviglio", non puoi toccare una proprietà senza trascinarne con sé i cambiamenti nelle altre. Una rappresentazione disgrovigliata è la versione ben progettata di quel pannello applicata ai dati: un insieme di variabili latenti in cui ciascuna controlla, in modo indipendente, un unico fattore generativo (posizione, dimensione, colore, forma), senza interferire con le altre.

## Sviluppo teorico

### Clustering e riduzione della dimensionalità: le due idee di base

Se hai solo osservazioni non etichettate, ci sono due strategie ragionevoli. La prima è raggruppare: decidere quali osservazioni si assomigliano di più tra loro e formare cluster. Il clustering è prezioso perché permette di risolvere in seguito problemi di classificazione con molti meno dati: se impari qualcosa su un robot all'interno di un cluster di robot con ruote, generalizzi quella conoscenza al resto del cluster.

La seconda è identificare un piccolo numero di assi che spieghino gran parte della variazione nei dati. Se le differenze tra un insieme di robot si spiegano con la loro altezza e il numero di ruote, rappresentare ogni robot come un punto in due dimensioni, invece che come un'immagine completa, risulta più interpretabile e facilita la classificazione successiva.

Il problema è che non esiste un unico modo "corretto" di raggruppare o ridurre le dimensioni: puoi raggruppare i robot per tipo di gamba, per numero di braccia o per altezza, e tutte e tre le opzioni sono valide. Questa ambiguità spiega perché valutare un algoritmo non supervisionato sia così difficile: non c'è una verità di riferimento con cui confrontarlo. Nella pratica si ricorre a indici di qualità del cluster, all'errore di ricostruzione o, in modo più pragmatico, alle prestazioni di un classificatore semplice addestrato sulla rappresentazione appresa con pochi dati etichettati: la valutazione semi-supervisionata, standard di fatto nella ricerca attuale, misurata spesso su ImageNet.

### Che cosa rende buona una rappresentazione

Le neuroscienze offrono un indizio: nel flusso visivo ventrale del cervello, l'informazione su un oggetto arriva completa alla retina, ma "aggrovigliata" nello spazio ad alta dimensionalità delle risposte neuronali iniziali. Secondo l'ipotesi del disgrovigliamento (*untangling*), l'elaborazione visiva successiva riformatta quell'informazione finché i confini decisionali diventano quasi lineari.

Un ragionamento simile emerge in un processo decisionale di Markov, il quadro abituale del RL: la rappresentazione dello stato determina la difficoltà del compito. Quando attraversi una strada, una rappresentazione che distingue "marciapiede senza auto" da "marciapiede con un'auto che si avvicina" aiuta; il colore esatto dell'auto non aiuta e può ostacolare la generalizzazione. Ma se il compito cambia (ora vuoi fermare un taxi), quel colore torna a essere rilevante: una buona rappresentazione scarta i dettagli irrilevanti per il compito attuale, ma mantiene disponibile l'informazione latente necessaria se il compito cambia.

A questo si aggiunge la composizionalità, presa dal linguaggio: il significato di un'espressione complessa si ottiene combinando i significati delle sue parti secondo certe regole, il che permette di costruire un numero enorme di concetti a partire da pezzi e regole finiti.

La fisica offre un terzo filo: la simmetria, una trasformazione che lascia invariata qualche proprietà rilevante del sistema (il teorema di Noether collega ogni simmetria continua con una quantità conservata). Trasportato nell'apprendimento di rappresentazioni, l'obiettivo è che la funzione di rappresentazione $f$ sia **equivariante** rispetto a un gruppo di trasformazioni $G$ di interesse (traslazione orizzontale, verticale, cambio di colore): applicare la trasformazione prima o dopo la mappatura deve dare lo stesso risultato.

$$
g \cdot f(w) = f(g \cdot w) \qquad \forall\, g \in G,\ w \in W
$$

Questo contrasta con l'**invarianza**, in cui la rappresentazione non cambia nonostante la trasformazione, utile quando la posizione di un oggetto è irrilevante per la sua classe, come sfrutta la condivisione dei pesi delle reti convoluzionali. Una rappresentazione equivariante ben costruita suddivide inoltre lo spazio in sottospazi indipendenti, uno per trasformazione, il che facilita l'attenzione (per esempio, con una maschera binaria) e il clustering con metriche di similarità semplici. Disgrovigliamento, equivarianza, composizionalità e supporto per attenzione e clustering formano il criterio con cui si giudica se una tecnica va nella direzione giusta.

### Autoencoder: l'esempio centrale

Un autoencoder è una rete addestrata a riprodurre il proprio ingresso, costretta a passare attraverso un collo di bottiglia che impedisce di copiare i dati in modo banale. Dato un vettore di ingresso $x$, l'encoder applica una trasformazione (pesi $W_e$, bias $b_e$, attivazione $f$) per ottenere la rappresentazione latente $z$:

$$
z = f(W_e x + b_e)
$$

Il decoder applica la trasformazione inversa (pesi $W_d$, bias $b_d$, attivazione $g$) per produrre la ricostruzione $\hat{x}$:

$$
\hat{x} = g(W_d z + b_d)
$$

L'addestramento minimizza l'errore di ricostruzione tra $x$ e $\hat{x}$. Per dati continui si usa l'errore quadratico medio (MSE):

$$
\mathcal{L}(x, \hat{x}) = \| x - \hat{x} \|^2
$$

e per pixel normalizzati tra zero e uno, l'entropia incrociata binaria. Poiché $z$ ha meno dimensioni di $x$, la rete non può imparare la funzione identità: deve scoprire quali combinazioni di caratteristiche bastano per ricostruire l'essenziale, scartando ridondanza e rumore. Esistono varianti: gli **autoencoder sparsi** (*sparse*) penalizzano in modo che solo poche unità del codice siano attive contemporaneamente; quelli di **rimozione del rumore** (*denoising*) ricostruiscono l'ingresso a partire da una versione corrotta, producendo rappresentazioni più robuste; e gli **autoencoder variazionali (VAE)**, che trattano lo spazio latente in modo probabilistico.

### Autoencoder variazionali e l'obiettivo ELBO

Un VAE assume che i dati $x$ siano generati a partire da variabili latenti $z$ campionate da un *prior* $p(z)$, quasi sempre una gaussiana standard, seguite da un processo di generazione $p_\theta(x \mid z)$. L'obiettivo è invertire quel processo: dato $x$, inferire la posteriore $p(z \mid x)$ su quali latenti lo abbiano probabilmente generato.

Calcolare esattamente quella posteriore è, in generale, intrattabile. La soluzione è approssimarla con una rete di inferenza (l'encoder) che produce $q_\phi(z \mid x)$, e addestrare massimizzando il limite inferiore dell'evidenza (ELBO):

$$
\log p_\theta(x) \geq \mathbb{E}_{z \sim q_\phi(z \mid x)}\big[\log p_\theta(x \mid z)\big] - D_{\mathrm{KL}}\big(q_\phi(z \mid x)\,\|\,p(z)\big)
$$

Il primo termine è, in sostanza, l'errore di ricostruzione negato. Il secondo, la divergenza di Kullback-Leibler (KL) tra la posteriore approssimata e il *prior*, agisce da regolarizzatore. Per questo il *prior* funziona come parametro di progetto: con una gaussiana a dimensioni indipendenti, il termine KL spinge la posteriore a mostrare anch'essa quell'indipendenza, un passo verso rappresentazioni disgrovigliate.

Da qui nasce il **beta-VAE**, che moltiplica il termine KL per un coefficiente $\beta > 1$:

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi(z \mid x)}\big[\log p_\theta(x \mid z)\big] - \beta \, D_{\mathrm{KL}}\big(q_\phi(z \mid x)\,\|\,p(z)\big)
$$

Aumentare $\beta$ non dà più un limite esatto della verosimiglianza, ma spinge verso rappresentazioni più disgrovigliate, al costo di un po' di fedeltà nella ricostruzione. Per verificarlo si usa il **percorso latente** (*latent traversal*): fissi tutte le variabili tranne una, la varii in modo continuo e osservi come cambia l'immagine generata. In un beta-VAE ben addestrato, ogni variabile di solito controlla un unico fattore interpretabile; in un VAE standard, una sola variabile muove più fattori contemporaneamente, la firma di una rappresentazione aggrovigliata. Nell'agente DARLA, questa proprietà ha permesso di trasferire politiche di RL dalla simulazione a un robot reale con molto meno degrado delle prestazioni.

### Reti generative avversarie come alternativa senza ricostruzione

Una rete generativa avversaria (GAN) modella la distribuzione dei dati senza limite di verosimiglianza né ricostruzione pixel per pixel. Contrappone due reti: un generatore $G$, che trasforma il rumore $z \sim p(z)$ in un campione sintetico $G(z)$, e un discriminatore $D$, che stima la probabilità che un campione sia reale. L'obiettivo è un gioco minimax, in cui il discriminatore massimizza l'espressione e il generatore la minimizza, ingannandolo sempre meglio:

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}(x)}\big[\log D(x)\big] + \mathbb{E}_{z \sim p(z)}\big[\log\big(1 - D(G(z))\big)\big]
$$

Il problema di una GAN standard è che non risolve l'inferenza: non ti dice la rappresentazione $z$ di un dato reale $x$. Varianti come BiGAN o BigBiGAN aggiungono un encoder al gioco avversario: il discriminatore confronta coppie dato-codice invece di campioni isolati, costringendo il modello a far coincidere le distribuzioni congiunte senza perdita di ricostruzione a livello di pixel. Il risultato cattura informazione di alto livello (quali oggetti ci sono, di che colore è un capo d'abbigliamento) invece di dettagli di basso livello, il che, al tempo, ha collocato questi modelli tra lo stato dell'arte per la classificazione semi-supervisionata.

### Apprendimento autosupervisionato: creare supervisione senza etichette

L'apprendimento autosupervisionato progetta compiti *pretesto* che non richiedono etichette umane ma obbligano il modello a imparare caratteristiche utili:

- **Predire la rotazione** applicata a un'immagine: indovinare l'angolo richiede di sapere che aspetto ha l'oggetto nel suo orientamento naturale
- **Colorazione**: convertire fotografie a colori in bianco e nero e addestrare il modello a recuperare il colore originale
- **Predizione di patch**: data un'immagine divisa in frammenti, predire la posizione relativa di uno rispetto a un altro
- **Ordinamento temporale**: dati fotogrammi di un video in disordine, ricostruire l'ordine corretto, catturando la coerenza temporale
- **Predizione di token mascherati**: BERT (rappresentazioni bidirezionali di encoder da transformer) nasconde parole a caso in una frase e addestra il modello a recuperarle, combinato con la predizione dell'ordine corretto tra coppie di frasi

Il filo comune: questi compiti sfruttano la struttura propria di ciascuna modalità per generare automaticamente coppie di ingresso e uscita corrette, senza che nessuno etichetti nulla.

### Apprendimento contrastivo: avvicinare il simile, allontanare il diverso

L'apprendimento contrastivo impara rappresentazioni senza un modello generativo completo, usando una funzione di perdita di classificazione costruita a partire da coppie: dato un dato di ancoraggio, un esempio positivo (una versione correlata) deve restare vicino a lui nello spazio di rappresentazione, e gli esempi negativi (dati non correlati), lontani.

Un esempio precoce è word2vec, che impara rappresentazioni di parole prevedendo quali parole compaiono nel loro contesto, con esempi positivi (parole vicine) e negativi (parole a caso). Il risultato cattura relazioni semantiche: addestrato separatamente in inglese e in spagnolo, bastano poche coppie di parole tradotte per imparare una trasformazione lineare che traduce il resto del vocabolario con buona fedeltà.

La codifica predittiva contrastiva (CPC) estende l'idea alle sequenze: prevede rappresentazioni future (positive) rispetto a rappresentazioni di altri punti (negative), massimizzando l'informazione mutua tra contesto e futuro. Nel benchmark semi-supervisionato di ImageNet, un modello con rappresentazioni CPC, rifinito con l'80 % di etichette in meno, supera uno addestrato direttamente sui pixel.

La formalizzazione standard di questa famiglia di perdite è l'InfoNCE, usata in SimCLR (un framework semplice per l'apprendimento contrastivo di rappresentazioni visive), che genera due versioni aumentate della stessa immagine (ritaglio, colore, sfocatura) e avvicina le loro rappresentazioni allontanando quelle del resto del lotto. Per una coppia positiva $(z_i, z_j)$ all'interno di un lotto di $2N$ viste:

$$
\mathcal{L}_{i,j} = -\log \frac{\exp\big(\mathrm{sim}(z_i, z_j) / \tau\big)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp\big(\mathrm{sim}(z_i, z_k) / \tau\big)}
$$

dove $\mathrm{sim}(\cdot,\cdot)$ è la similarità del coseno e $\tau$ la temperatura. Con un semplice classificatore lineare sopra queste rappresentazioni, SimCLR ha raggiunto risultati di riferimento su ImageNet, superando architetture comparabili addestrate con approcci generativi.

### Il filo che collega tutto

Generativo (autoencoder, VAE, GAN), autosupervisionato e contrastivo non sono alternative che si escludono a vicenda: sono modi complementari di imparare rappresentazioni senza etichette, e spesso si combinano. Monet usa l'attenzione insieme a un beta-VAE per segmentare e disgrovigliare oggetti di una scena senza supervisione; l'autoencoder variazionale quantizzato vettorialmente (VQ-VAE) sostituisce lo spazio latente continuo con variabili discrete indicizzate in una tabella di codici, ottenendo un'alta compressione con una ricostruzione ragionevole. E quando la rappresentazione appresa si combina con il RL, come in DARLA o nelle reti di query generativa (GQN), gli agenti imparano più in fretta, generalizzano meglio e trasferiscono conoscenza tra ambienti: i tre punti deboli con cui abbiamo aperto questo capitolo.

## Esempio di codice commentato, in PyTorch

Il seguente esempio costruisce un autoencoder semplice, con strati totalmente connessi (lineari), lo addestra sul dataset MNIST di cifre scritte a mano e confronta visivamente le immagini originali con le loro ricostruzioni. **Copia il codice in una cella di Google Colab ed eseguilo** per vederlo funzionare dall'inizio alla fine.

```python
# Instala/actualiza torchvision si hace falta (normalmente ya viene en Colab)
# !pip install torch torchvision --quiet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Usa GPU si Colab te la ha asignado, si no, usa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de entrenamiento: {device}")

# --- 1. Datos ---
# MNIST: dígitos de 28x28 en escala de grises, valores normalizados entre 0 y 1
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

DIM_ENTRADA = 28 * 28   # cada imagen se aplana a un vector de 784 valores
DIM_LATENTE = 32        # tamaño del cuello de botella: el "resumen" de cada imagen

# --- 2. Definición del autocodificador ---
class Autocodificador(nn.Module):
    def __init__(self, dim_entrada=DIM_ENTRADA, dim_latente=DIM_LATENTE):
        super().__init__()
        # Codificador: comprime progresivamente hasta el cuello de botella
        self.codificador = nn.Sequential(
            nn.Linear(dim_entrada, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dim_latente),
        )
        # Decodificador: expande de vuelta hasta el tamaño original
        # Sigmoid final porque los píxeles de MNIST están normalizados en [0, 1]
        self.decodificador = nn.Sequential(
            nn.Linear(dim_latente, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, dim_entrada),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.codificador(x)
        x_reconstruida = self.decodificador(z)
        return x_reconstruida, z

modelo = Autocodificador().to(device)
print(modelo)

# --- 3. Función de pérdida y optimizador ---
# El error de reconstrucción binario funciona bien con píxeles en [0, 1] y salida sigmoide
criterio = nn.BCELoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

# --- 4. Bucle de entrenamiento ---
def entrenar_una_epoca(modelo, loader, criterio, optimizador):
    modelo.train()
    perdida_total = 0.0
    for imagenes, _ in loader:  # las etiquetas no se usan: no hace falta supervisión
        imagenes = imagenes.view(imagenes.size(0), -1).to(device)  # aplana a vector

        optimizador.zero_grad()
        reconstruidas, _ = modelo(imagenes)
        perdida = criterio(reconstruidas, imagenes)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * imagenes.size(0)
    return perdida_total / len(loader.dataset)

num_epocas = 10
for epoca in range(1, num_epocas + 1):
    perdida_train = entrenar_una_epoca(modelo, train_loader, criterio, optimizador)
    print(f"Época {epoca}/{num_epocas} | pérdida de reconstrucción: {perdida_train:.4f}")

# --- 5. Comparación visual: originales frente a reconstrucciones ---
modelo.eval()
imagenes_prueba, _ = next(iter(test_loader))
imagenes_prueba = imagenes_prueba[:10]  # nos quedamos con diez ejemplos

with torch.no_grad():
    entrada_plana = imagenes_prueba.view(imagenes_prueba.size(0), -1).to(device)
    reconstruidas, _ = modelo(entrada_plana)
    reconstruidas = reconstruidas.view(-1, 28, 28).cpu()

fig, ejes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    ejes[0, i].imshow(imagenes_prueba[i].squeeze(), cmap="gray")
    ejes[0, i].axis("off")
    ejes[1, i].imshow(reconstruidas[i], cmap="gray")
    ejes[1, i].axis("off")

ejes[0, 0].set_title("Original", loc="left", fontsize=10)
ejes[1, 0].set_title("Reconstruida", loc="left", fontsize=10)
plt.tight_layout()
plt.show()
```

Dopo dieci epoche, le ricostruzioni dovrebbero somigliare chiaramente alle cifre originali, anche se un po' più sfocate e con bordi meno definiti: è la firma tipica di una perdita basata sulla ricostruzione, la stessa che vedevi nel capitolo teorico confrontando immagini originali e ricostruite di un VQ-VAE. **Sperimenta riducendo `DIM_LATENTE` a due o otto** e osserva come peggiora la fedeltà della ricostruzione man mano che il collo di bottiglia si restringe, una verifica diretta del fatto che il modello è costretto a dare priorità alle informazioni più rilevanti.

## Riepilogo dei punti chiave

- L'etichettatura manuale dei dati è costosa, lenta e non scala al volume di dati non etichettati disponibile nel mondo reale; l'apprendimento non supervisionato cerca di sfruttare quei dati senza bisogno di etichette
- L'apprendimento non supervisionato si distingue da quello supervisionato e dal RL per il tipo di feedback: non ci sono etichette esplicite né ricompensa, solo i dati di ingresso stessi
- Valutare un algoritmo non supervisionato è particolarmente difficile perché non esiste un'unica verità di riferimento; nella pratica si ricorre a indici di qualità del cluster, all'errore di ricostruzione o alle prestazioni di un classificatore addestrato con poche etichette sulla rappresentazione appresa
- Una buona rappresentazione, secondo indizi convergenti da neuroscienze, fisica e dalla stessa IA, dovrebbe essere disgrovigliata, supportare attenzione e clustering, incorporare informazione latente rilevante per possibili compiti futuri ed essere composizionale
- Un autoencoder impara comprimendo l'ingresso in un codice latente (encoder) e ricostruendolo a partire da quel codice (decoder), minimizzando l'errore di ricostruzione; il collo di bottiglia lo obbliga a dare priorità alle informazioni più rilevanti
- L'autoencoder variazionale (VAE) tratta lo spazio latente in modo probabilistico e si addestra massimizzando il limite ELBO, che combina un termine di ricostruzione e una divergenza KL che regolarizza la posteriore approssimata verso il *prior* scelto
- Aumentare il peso del termine KL (beta-VAE) favorisce rappresentazioni disgrovigliate, verificabili con i percorsi latenti, e migliora il trasferimento di politiche di RL dalla simulazione alla realtà
- Le GAN modellano i dati senza una perdita di ricostruzione esplicita, tramite un gioco minimax tra generatore e discriminatore; varianti come BiGAN o BigBiGAN aggiungono un encoder per poter inferire rappresentazioni a partire da dati reali
- L'apprendimento autosupervisionato crea compiti pretesto (rotazione, colorazione, ordine temporale, predizione di token mascherati come in BERT) che generano supervisione automaticamente dalla struttura stessa dei dati
- L'apprendimento contrastivo, formalizzato con la perdita InfoNCE e reso popolare da metodi come word2vec, CPC e SimCLR, avvicina le rappresentazioni di coppie positive e allontana quelle di coppie negative, senza bisogno di un modello generativo completo

## Mettilo in pratica

**Dedica tra quindici e trenta minuti a questo esercizio prima di passare al capitolo successivo.**

1. **Copia il codice della sezione precedente in un notebook di Google Colab ed eseguilo così com'è**, verificando che le ricostruzioni finali somiglino ragionevolmente alle cifre originali
2. **Riduci `DIM_LATENTE` da trentadue a due** e riaddestra il modello per le stesse dieci epoche. Confronta visivamente le ricostruzioni con quelle dell'esperimento originale: quale tipo di informazione credi che si perda per primo quando il collo di bottiglia è così stretto?
3. **Aggiungi una riga di codice dopo l'addestramento** che estragga il vettore latente `z` da duecento immagini di test (usando la seconda uscita di `modelo(entrada_plana)`) e lo rappresenti in due dimensioni con `sklearn.decomposition.PCA` o `matplotlib.pyplot.scatter`, colorando ogni punto in base alla sua etichetta reale (anche se il modello non l'ha mai vista). Osserva se le cifre uguali tendono a raggrupparsi nello spazio latente
4. **Sostituisci `nn.BCELoss()` con `nn.MSELoss()`** e riaddestra. Annota se cambia visibilmente la nitidezza delle ricostruzioni
5. **Rifletti per iscritto, in due o tre frasi**, su che cosa dovresti cambiare in questo autoencoder per avvicinarlo a un autoencoder variazionale: pensa a quale strato dovresti modificare perché l'encoder produca una media e una varianza invece di un unico vettore deterministico, e quale termine dovresti aggiungere alla funzione di perdita
