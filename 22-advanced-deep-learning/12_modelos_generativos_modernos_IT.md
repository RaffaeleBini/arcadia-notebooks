# Capitolo 12: Modelli generativi moderni a variabili latenti

## Perché questo argomento è importante

Nei capitoli precedenti hai conosciuto due modi di generare dati nuovi con una rete neurale: l'autoencoder, che impara a comprimere e ricostruire un ingresso, e la rete generativa avversaria (GAN), che contrappone un generatore a un discriminatore finché il primo produce campioni difficili da distinguere da quelli reali. Entrambe le idee funzionano, ma lasciano lacune importanti. L'autoencoder classico non ti dà alcun controllo su ciò che accade tra due punti del suo spazio latente: non sai quale immagine otterresti spostando un po' la rappresentazione interna di un dato. La GAN, dal canto suo, non ti offre un modo per calcolare la probabilità di un'osservazione né un modo diretto di codificare un dato esistente nel suo spazio latente.

Questo capitolo, corrispondente alla Lezione 11 del corso ("Modelli generativi avanzati"), colma entrambe le lacune con la famiglia dei modelli moderni a variabili latenti: l'autoencoder variazionale (VAE, dall'inglese *Variational Autoencoder*) e i flussi normalizzanti (*normalizing flows*). Questi modelli non si limitano a generare dati verosimili, ma strutturano lo spazio latente come una mappa continua e navigabile, dove spostarsi dolcemente da un punto all'altro produce variazioni morbide e coerenti del dato generato.

Questa proprietà ha conseguenze pratiche molto concrete. Nella progettazione di farmaci e nelle biotecnologie, uno spazio latente continuo permette di interpolare tra due molecole note ed esplorare varianti intermedie che conservano proprietà chimiche desiderabili, invece di saltare a punti arbitrari privi di senso strutturale. Nella diagnostica per immagini e in altre applicazioni sanitarie, permette di generare variazioni sintetiche di una stessa condizione per ampliare i dataset di addestramento. Nella creazione di contenuti, permette di interpolare in modo creativo tra due volti, due stili o due suoni e ottenere risultati intermedi plausibili, non rumore. E nel rilevamento di anomalie o nel controllo qualità, il modello stesso ti dà una misura di quanto un dato nuovo si allontani da ciò che ha imparato come normale. **Presta particolare attenzione alla differenza tra autoencoder e autoencoder variazionale**: è la chiave che apre tutto il resto di questo capitolo.

## Spiegazione concettuale con analogie

### Lo spazio latente come mappa: indirizzi fissi contro interi quartieri

Immagina di voler rappresentare ogni foto di un volto con un indirizzo postale su una mappa di città. Un autoencoder normale agisce come un impiegato che, per ogni foto, scrive un unico indirizzo esatto: la via e il numero dove "abita" quella foto sulla mappa. Il problema è che questo impiegato sistema gli indirizzi senza preoccuparsi del quartiere: due foto molto simili potrebbero finire in strade opposte della città, e lo spazio tra due indirizzi qualsiasi può essere un terreno vuoto, privo di qualsiasi significato. Se ti fermi su un punto della mappa dove non c'è nessun indirizzo registrato e chiedi al decoder di "leggere" quel punto, ottieni rumore, non un volto plausibile.

L'autoencoder variazionale cambia le regole del gioco: invece di assegnare un indirizzo esatto, assegna a ogni foto un intero quartiere, rappresentato come una distribuzione di probabilità (di solito una campana di Gauss) centrata su un punto della mappa, con un raggio di incertezza. Inoltre, l'addestramento spinge tutti quei quartieri a sovrapporsi in modo ordinato attorno al centro della mappa, seguendo una distribuzione di riferimento comune. Il risultato è una mappa senza terreni vuoti: qualsiasi punto tu scelga, che si trovi o no esattamente nel quartiere di una foto di addestramento, cade dentro qualche quartiere o al confine tra più quartieri, e il decoder sa interpretarlo come un volto ragionevole. Questa è la differenza essenziale tra i due modelli: un autoencoder normale apprende un punto fisso per ogni dato, mentre uno variazionale apprende una distribuzione di probabilità nello spazio latente. Questa proprietà è proprio ciò che ti permette di passeggiare da un quartiere all'altro, mescolando due volti in modo graduale, o di lanciare un dardo a caso sulla mappa intera e ottenere sempre un volto nuovo e coerente.

### Il trucco della riparametrizzazione: spostare il dado fuori dall'equazione

Addestrare una rete neurale richiede di poter calcolare i gradienti e propagarli all'indietro attraverso ogni operazione della rete. Il problema è che, dentro un autoencoder variazionale, c'è un passo di campionamento casuale: scegli un punto a caso dentro il quartiere di ogni foto. Il campionamento casuale, così com'è, blocca la propagazione del gradiente, perché non esiste una derivata ben definita di "un numero scelto a caso" rispetto ai parametri che definiscono quel caso.

La soluzione, nota come trucco della riparametrizzazione, è una separazione dei ruoli molto elegante. Invece di chiedere alla rete di "tirare il dado" direttamente, separi il processo in due parti: da un lato un generatore di rumore esterno e fisso, che non dipende da alcun parametro addestrabile, come una moneta che lanci in aria senza che la tua decisione influenzi il risultato; dall'altro una trasformazione deterministica e derivabile che prende quel rumore e i parametri appresi (il centro del quartiere e il suo raggio di incertezza) e produce il punto finale dello spazio latente. Poiché quella trasformazione è una semplice formula matematica derivabile, il gradiente può attraversarla senza problemi, e il caso resta isolato, fuori dal percorso lungo cui circola l'informazione di apprendimento.

### Flussi normalizzanti: origami che non perde mai informazione

La fonte complementare di questo capitolo, la lezione di Andriy Mnih per UCL x DeepMind, presenta un'altra famiglia di modelli moderni a variabili latenti: i flussi normalizzanti o modelli invertibili. Immagina un pezzo di plastilina a forma di sfera perfetta, che rappresenta una distribuzione semplice e nota. Un flusso normalizzante la deforma tramite una sequenza di pieghe e stiramenti, ciascuno reversibile, fino a dargli la forma irregolare dei tuoi dati reali, per esempio il contorno di tutte le possibili cifre scritte a mano. Poiché ogni piega è invertibile, in qualsiasi momento puoi disfare l'intera sequenza e recuperare esattamente la sfera originale: non si perde né si crea plastilina nel processo, si ridistribuisce soltanto il suo volume. Questa conservazione esatta è ciò che permette di calcolare la probabilità esatta di qualsiasi osservazione, qualcosa che né l'autoencoder variazionale né la GAN possono offrire con la stessa precisione.

## Sviluppo teorico

### Il quadro generale dei modelli a variabili latenti

Un modello a variabile latente definisce una distribuzione sulle osservazioni **x** introducendo una variabile latente **z**, tipicamente un vettore, che non si osserva direttamente. Il modello è completamente specificato da due componenti:

- La distribuzione a priori (*prior*) della variabile latente, p(z), normalmente una gaussiana standard
- La verosimiglianza p(x|z), la rete che collega la variabile latente con l'osservazione

Insieme definiscono la distribuzione congiunta:

p(x, z) = p(x|z) · p(z)

Dalla congiunta si derivano due distribuzioni fondamentali: la verosimiglianza marginale p(x), che indica quanto è probabile un'osservazione secondo il modello e che si massimizza durante l'addestramento, e la distribuzione a posteriori p(z|x), che rappresenta i valori latenti plausibili che potrebbero aver generato una particolare osservazione, cioè la spiegazione del dato in termini del modello. Generare un'osservazione nuova è semplice: prima campioni z dal prior, poi campioni x dalla verosimiglianza p(x|z). Il problema compare quando si inverte questo processo: calcolare la posteriore richiede di conoscere la marginale,

p(z|x) = p(x, z) / p(x), con p(x) = ∫ p(x|z) p(z) dz

Questo integrale, salvo casi speciali (miscele di gaussiane, modelli lineari-gaussiani, modelli invertibili), è intrattabile: implica integrare su uno spazio ad alta dimensione una funzione non lineare, e né il calcolo analitico né l'integrazione numerica sono praticabili. Per superare questo ostacolo esistono due strategie generali, esattamente le due famiglie che questo capitolo copre: progettare modelli in cui l'inferenza sia trattabile per costruzione (i flussi normalizzanti), oppure accettare che il modello sia intrattabile e ricorrere all'inferenza approssimata (l'inferenza variazionale, base dell'autoencoder variazionale).

### Modelli invertibili e la formula del cambio di variabile

Un flusso normalizzante genera osservazioni applicando una trasformazione f, invertibile e derivabile, a un campione del prior: x = f(z). Poiché f è invertibile, esiste una corrispondenza uno a uno tra ogni osservazione e la sua configurazione latente, e l'inferenza si risolve semplicemente invertendo la funzione: z = f⁻¹(x). Per addestrare per massima verosimiglianza, devi mettere in relazione la densità di z con la densità di x. Questa relazione è la formula del cambio di variabile:

log p(x) = log p(z) − log |det (∂f(z) / ∂z)|, con z = f⁻¹(x)

Il determinante dello Jacobiano di f agisce da fattore di correzione del volume: tiene conto di come la trasformazione stira o comprime lo spazio nell'intorno di ogni punto, in modo che la densità risultante continui a integrare a uno, come la densità originale. Perché l'addestramento sia efficiente, sia f sia la sua inversa, sia il determinante del suo Jacobiano — e i gradienti di entrambi — devono potersi calcolare rapidamente.

L'esempio più semplice di modello invertibile è l'analisi delle componenti indipendenti (ICA, *Independent Component Analysis*), che usa un prior fattorizzato e una trasformazione lineare tramite una matrice quadrata W: x = W·z. L'inferenza si riduce a invertire la matrice: z = W⁻¹·x. Un caso d'uso classico è il problema del cocktail party: separare le singole voci di diversi invitati a partire dalle registrazioni mescolate di più microfoni, cosa possibile grazie alla linearità dell'acustica, purché il prior non sia gaussiano.

I flussi normalizzanti generali si costruiscono concatenando trasformazioni invertibili semplici, dato che la composizione di funzioni invertibili è anch'essa invertibile. Il loro vantaggio è raro nel panorama dei modelli generativi: combinano alta capacità espressiva con verosimiglianza esatta e trattabile. La loro limitazione principale è strutturale: la dimensione dello spazio latente deve coincidere esattamente con quella dei dati, le variabili latenti devono essere continue, e per raggiungere una capacità espressiva sufficiente occorre concatenare molti strati, il che fa crescere rapidamente il numero di parametri e il consumo di memoria.

### Inferenza variazionale: trasformare l'inferenza in ottimizzazione

Quando il modello è intrattabile, esistono due grandi famiglie di inferenza approssimata. I metodi Monte Carlo a catene di Markov (MCMC, *Markov Chain Monte Carlo*) generano campioni della posteriore esatta lasciando evolvere una catena di Markov; sono generali ed esatti nel limite di tempo infinito, ma computazionalmente costosi e difficili da diagnosticare. L'inferenza variazionale, invece, approssima la vera posteriore p(z|x) con una distribuzione trattabile q_φ(z|x), detta posteriore variazionale, parametrizzata da φ. Puoi usare qualsiasi distribuzione per q, purché tu possa campionarne e calcolarne il logaritmo della probabilità e il gradiente di tale logaritmo rispetto a φ. La scelta predefinita è una distribuzione completamente fattorizzata, nota come approssimazione di campo medio.

Quale obiettivo ottimizzi se la verosimiglianza marginale log p(x) è intrattabile? La risposta è un limite inferiore. Per qualsiasi densità q(z) con supporto che copra quello di p(z|x), puoi scrivere:

log p(x) = log ∫ p(x, z) dz = log ∫ q(z) · [p(x, z) / q(z)] dz ≥ ∫ q(z) · log [p(x, z) / q(z)] dz

La disuguaglianza applica la disuguaglianza di Jensen: il logaritmo di un valore atteso è maggiore o uguale al valore atteso del logaritmo, perché il logaritmo è una funzione concava. Il termine a destra prende il nome di limite inferiore dell'evidenza (ELBO, *Evidence Lower Bound*), ed è valido per qualsiasi q e per qualsiasi valore dei parametri del modello. Usando concretamente la posteriore variazionale q_φ(z|x), l'ELBO si scrive:

ELBO(θ, φ; x) = E_{q_φ(z|x)} [log p_θ(x, z) − log q_φ(z|x)]

Esistono altri limiti variazionali, come il limite ponderato per importanza (IWAE), che generalizza l'ELBO usando più campioni e permette di regolare la fermezza del limite, ma l'ELBO è di gran lunga la più semplice e la più usata.

### La divergenza di Kullback-Leibler

Per capire che cosa misura esattamente l'ELBO ti serve la divergenza di Kullback-Leibler (KL), che quantifica quanto una distribuzione q differisce da un'altra distribuzione p:

KL(q(z) ‖ p(z)) = ∫ q(z) · log [q(z) / p(z)] dz = E_{q(z)} [log q(z) − log p(z)]

La divergenza KL gode di due proprietà importanti: è non negativa per qualsiasi coppia di distribuzioni q e p, e vale zero se e solo se q e p coincidono quasi ovunque. Ricorda anche che non è una metrica in senso stretto, perché non è simmetrica: in generale, KL(q‖p) è diversa da KL(p‖q).

### Scomposizione dell'ELBO e il divario variazionale

Riscrivendo la congiunta come p(x, z) = p(x) · p(z|x), l'ELBO si scompone in due termini:

ELBO(θ, φ; x) = log p_θ(x) − KL(q_φ(z|x) ‖ p_θ(z|x))

Il primo termine, la log-verosimiglianza marginale, non dipende dai parametri variazionali φ. Il secondo è la divergenza KL tra la posteriore variazionale e la vera posteriore, nota come divario variazionale (*variational gap*). Da questa scomposizione segue qualcosa di notevole: massimizzare l'ELBO rispetto a φ equivale esattamente a minimizzare il divario variazionale, cioè ad avvicinare la posteriore variazionale alla vera posteriore. Ed è possibile farlo anche se né la posteriore esatta né la sua divergenza KL rispetto a q sono calcolabili separatamente: l'ELBO, essendo una differenza di due quantità intrattabili, risulta trattabile. Quando aggiorni i parametri del modello θ per aumentare l'ELBO, quell'aumento può provenire da due fonti: un miglioramento reale della log-verosimiglianza marginale (esattamente ciò che persegue l'apprendimento per massima verosimiglianza) oppure una riduzione del divario variazionale ottenuta deformando il modello in modo che la sua vera posteriore somigli di più alla famiglia di q, il che a volte sacrifica capacità di adattamento ai dati. Per questo conviene usare la posteriore variazionale più espressiva che puoi permetterti dal punto di vista computazionale.

Un fenomeno associato, chiamato *pruning* variazionale o collasso della posteriore nel contesto degli autoencoder variazionali, si verifica quando il modello smette di usare alcune dimensioni latenti: per quelle dimensioni, la posteriore variazionale finisce per essere identica al prior. Può essere desiderabile, perché regola automaticamente la dimensionalità effettiva dello spazio latente, o indesiderabile, perché limita la capacità del modello di sfruttare tutta la capacità latente disponibile.

### Inferenza amortizzata e stima dei gradienti

Nell'inferenza variazionale classica, ogni osservazione richiede un proprio insieme di parametri variazionali, ottimizzati tramite un'ottimizzazione indipendente. Questo è molto inefficiente su larga scala. L'inferenza amortizzata risolve il problema addestrando una rete neurale, detta rete di inferenza o encoder, che prende l'osservazione x come ingresso e produce direttamente i parametri della sua posteriore variazionale, invece di ricalcolarli da zero per ogni dato. Questa idea, introdotta nel contesto delle macchine di Helmholtz negli anni novanta, è proprio ciò che ha reso popolare l'autoencoder variazionale: un unico insieme di pesi, condiviso tra tutte le osservazioni, sostituisce migliaia di ottimizzazioni indipendenti.

Addestrare con l'inferenza variazionale richiede di calcolare il gradiente dell'ELBO rispetto a θ e rispetto a φ. Il gradiente rispetto ai parametri del modello θ è semplice, perché la distribuzione su cui si prende il valore atteso, q_φ(z|x), non dipende da θ: basta portare il gradiente dentro il valore atteso, campionare da q e mediare i gradienti del logaritmo della congiunta. Il gradiente rispetto ai parametri variazionali φ è più delicato, perché ora è la stessa distribuzione su cui integri a dipendere dal parametro che vuoi derivare. Esistono due grandi famiglie di stimatori non distorti per questo caso: lo stimatore REINFORCE (o del rapporto di verosimiglianza), molto generale — funziona con variabili discrete e continue — ma con varianza alta; e lo stimatore di riparametrizzazione (o *pathwise*), meno generale — richiede variabili latenti continue e una funzione derivabile — ma con una varianza molto più bassa, il che in pratica lo rende l'opzione preferita ogni volta che è applicabile.

Il trucco della riparametrizzazione, già descritto con l'analogia del dado, esprime un campione di q_φ(z|x) come una trasformazione deterministica e derivabile g_φ di una variabile di rumore ε con distribuzione fissa, indipendente da φ:

z = g_φ(ε, x), con ε ~ p(ε)

Per una gaussiana univariata, la riparametrizzazione più comune è:

z = μ_φ(x) + σ_φ(x) · ε, con ε ~ N(0, 1)

Questa trasformazione è derivabile sia rispetto a μ sia rispetto a σ, soddisfacendo il requisito per propagare i gradienti. Molte distribuzioni della famiglia di localizzazione-scala, come la Laplace o la Cauchy, si riparametrizzano in modo analogo; altre, come la Gamma o la Dirichlet, non permettono di separare esplicitamente il caso dai parametri, anche se esistono varianti di riparametrizzazione implicita. Le variabili discrete, in generale, non ammettono questo trucco perché la trasformazione risultante non è derivabile. Framework come PyTorch implementano automaticamente la riparametrizzazione per le distribuzioni continue più comuni.

### L'autoencoder variazionale

Un autoencoder variazionale è, formalmente, un modello generativo a variabili latenti continue in cui sia la verosimiglianza (decoder) sia la posteriore variazionale (encoder) sono parametrizzate da reti neurali. Sia il prior sia la posteriore variazionale si modellano di solito come gaussiane completamente fattorizzate:

- Prior: p(z) = N(z; 0, I)
- Encoder / posteriore variazionale: q_φ(z|x) = N(z; μ_φ(x), diag(σ_φ(x)²))
- Decoder / verosimiglianza: per dati binari, una Bernoulli parametrizzata dalla rete; per dati a valori reali, una gaussiana la cui media e varianza diagonale vengono calcolate dalla rete

Il tipo concreto di rete — strati completamente connessi, convoluzionali, residuali — è un dettaglio di implementazione che non altera la struttura matematica del modello. L'essenziale è che l'encoder impari a produrre, per ogni x, i parametri μ_φ(x) e σ_φ(x) del suo quartiere nella mappa latente, e che il decoder impari a ricostruire x a partire da un punto campionato da quel quartiere.

L'obiettivo di addestramento, l'ELBO per il VAE, si scompone in due termini trattabili:

ELBO(θ, φ; x) = E_{q_φ(z|x)} [log p_θ(x|z)] − KL(q_φ(z|x) ‖ p(z))

Il primo termine misura quanto bene il modello ricostruisce l'osservazione a partire da un campione della posteriore variazionale; è noto come errore di ricostruzione negativo, e valori alti indicano una buona ricostruzione. Il secondo termine agisce da regolarizzatore: penalizza che la posteriore variazionale si allontani dal prior, evitando che lo spazio latente codifichi più informazione sull'osservazione del necessario, ed è proprio il meccanismo che mantiene la mappa latente continua e senza terreni vuoti. Quando sia il prior sia la posteriore sono gaussiane, questo termine ha forma chiusa e non richiede campionamento:

KL(q_φ(z|x) ‖ p(z)) = ½ Σ_j (σ_j² + μ_j² − 1 − log σ_j²)

dove la somma scorre su ciascuna delle dimensioni del vettore latente. Addestrare un VAE consiste quindi nel massimizzare la somma dei due termini (o, in modo equivalente, minimizzarne il negativo) con discesa del gradiente, usando la riparametrizzazione affinché l'errore si propaghi sia all'encoder sia al decoder.

Dalla sua introduzione nel 2014, il VAE si è evoluto fino a diventare più un framework che un modello singolo: qualsiasi modello a variabili latenti continue addestrato con inferenza variazionale amortizzata e riparametrizzazione rientra nella categoria dei VAE. Le estensioni più rilevanti documentate includono più strati latenti per rappresentazioni gerarchiche, variabili latenti non gaussiane, prior e posteriori più espressivi — per esempio tramite flussi normalizzanti —, architetture più potenti come le reti residuali, decoder autoregressivi che combinano i punti di forza delle due famiglie di modelli, schemi di inferenza variazionale iterativa con più aggiornamenti invece di uno solo, e tecniche di riduzione della varianza per accelerare l'addestramento.

## Esempio di codice commentato, in PyTorch

Il seguente esempio implementa un piccolo autoencoder variazionale e lo addestra sul dataset di cifre scritte a mano MNIST, disponibile direttamente tramite `torchvision`. **Copia ed esegui questo codice in Google Colab**, attivando un ambiente con GPU se disponibile, per vedere ricostruzioni e campioni generati in pochi minuti.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Datos: MNIST, imágenes de 28x28 en escala de grises, valores en [0, 1] ---
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

LATENT_DIM = 20  # dimensión del espacio latente z


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=LATENT_DIM):
        super().__init__()
        # Codificador: de la imagen aplanada a una representación oculta compartida
        self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
        # Dos cabezas separadas: una para la media, otra para el log de la varianza
        # (se predice el logaritmo de la varianza, no la varianza directamente,
        # para que la red pueda producir cualquier número real y aun así obtengamos
        # una varianza siempre positiva al exponenciar)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decodificador: del espacio latente de vuelta a los píxeles
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.encoder_hidden(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # Truco de reparametrización: z = mu + sigma * epsilon, con epsilon ~ N(0, I)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = F.relu(self.decoder_hidden(z))
        # Sigmoid porque el decodificador modela una Bernoulli por píxel (valores en [0, 1])
        return torch.sigmoid(self.decoder_out(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparameterize(mu, logvar)
        x_reconstruida = self.decode(z)
        return x_reconstruida, mu, logvar


def elbo_loss(x_reconstruida, x, mu, logvar):
    # Término de reconstrucción: log-verosimilitud de una Bernoulli, sumada sobre píxeles
    reconstruccion = F.binary_cross_entropy(
        x_reconstruida, x.view(x.size(0), -1), reduction="sum"
    )
    # KL entre q(z|x) = N(mu, sigma^2) y el prior p(z) = N(0, I), forma cerrada
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruccion + kl  # minimizar esto equivale a maximizar el ELBO


modelo = VAE().to(device)
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    modelo.train()
    perdida_total = 0.0
    for x, _ in train_loader:  # las etiquetas no se usan: el VAE es no supervisado
        x = x.to(device)
        optimizador.zero_grad()
        x_reconstruida, mu, logvar = modelo(x)
        perdida = elbo_loss(x_reconstruida, x, mu, logvar)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item()

    print(f"Época {epoch}/{EPOCHS} — pérdida media: {perdida_total / len(train_data):.2f}")

# --- Visualización: reconstrucciones frente a originales ---
modelo.eval()
with torch.no_grad():
    x_test, _ = next(iter(test_loader))
    x_test = x_test.to(device)
    x_reconstruida, _, _ = modelo(x_test)

n = 8
fig, ejes = plt.subplots(2, n, figsize=(n * 1.2, 3))
for i in range(n):
    ejes[0, i].imshow(x_test[i].cpu().squeeze(), cmap="gray")
    ejes[0, i].axis("off")
    ejes[1, i].imshow(x_reconstruida[i].cpu().view(28, 28), cmap="gray")
    ejes[1, i].axis("off")
ejes[0, 0].set_ylabel("Original")
ejes[1, 0].set_ylabel("Reconstruida")
plt.suptitle("Fila superior: dígitos originales — fila inferior: reconstrucciones del VAE")
plt.show()

# --- Muestras nuevas: dígitos generados desde puntos al azar del espacio latente ---
with torch.no_grad():
    z_aleatorio = torch.randn(n, LATENT_DIM).to(device)  # muestreo directo del prior N(0, I)
    muestras = modelo.decode(z_aleatorio).cpu()

fig, ejes = plt.subplots(1, n, figsize=(n * 1.2, 1.6))
for i in range(n):
    ejes[i].imshow(muestras[i].view(28, 28), cmap="gray")
    ejes[i].axis("off")
plt.suptitle("Dígitos generados desde puntos muestreados del espacio latente")
plt.show()
```

Con dieci epoche e un'architettura così piccola, non aspettarti cifre perfette, ma vedrai chiaramente due cose: le ricostruzioni somigliano ragionevolmente agli originali, e i campioni generati da punti completamente casuali dello spazio latente producono forme riconoscibili come cifre, anche se a volte ambigue tra due classi, proprio perché lo spazio latente è continuo e senza terreni vuoti.

## Riepilogo dei punti chiave

- Un autoencoder normale assegna a ogni dato un punto fisso dello spazio latente; un autoencoder variazionale gli assegna una distribuzione di probabilità, il che trasforma lo spazio latente in una mappa continua adatta a interpolare e generare
- Un modello a variabile latente è definito dal prior p(z), dalla verosimiglianza p(x|z) e dal loro prodotto, la congiunta p(x, z); l'inferenza consiste nel calcolare la posteriore p(z|x), che di solito è intrattabile a causa dell'integrale che definisce la marginale p(x)
- Esistono due strategie di fronte all'intrattabilità: modelli trattabili per costruzione, come i flussi normalizzanti, che usano trasformazioni invertibili e la formula del cambio di variabile per calcolare verosimiglianze esatte, oppure inferenza approssimata tramite inferenza variazionale
- L'inferenza variazionale approssima la vera posteriore con una posteriore variazionale q_φ(z|x) e trasforma il problema di inferenza in un problema di ottimizzazione
- Il limite inferiore dell'evidenza (ELBO) si ottiene applicando la disuguaglianza di Jensen ed è un limite inferiore trattabile della log-verosimiglianza marginale; si scompone in log p(x) meno la divergenza KL tra la posteriore variazionale e la vera posteriore
- La divergenza di Kullback-Leibler misura la differenza tra due distribuzioni, è sempre non negativa, si annulla solo quando le due distribuzioni coincidono, e non è simmetrica
- Il trucco della riparametrizzazione separa il caso (una variabile di rumore fissa) dai parametri addestrabili, permettendo di propagare i gradienti con bassa varianza attraverso il campionamento; in un VAE si traduce in z = μ + σ · ε
- L'obiettivo di addestramento del VAE combina un termine di ricostruzione e un termine di regolarizzazione KL rispetto al prior, con forma chiusa quando entrambe le distribuzioni sono gaussiane
- I flussi normalizzanti garantiscono verosimiglianza esatta e inferenza semplice, ma richiedono che la dimensione latente coincida con quella dei dati e di solito molti strati e parametri
- Il VAE è oggi un framework estensibile: ammette più strati latenti, posteriori e prior più espressivi, decoder autoregressivi e altri miglioramenti architetturali

## Mettilo in pratica

**Modifica il codice del VAE che hai appena addestrato** e rispondi a queste due domande con i tuoi esperimenti, in non più di trenta minuti:

1. Cambia `LATENT_DIM` da venti a due dimensioni e riaddestra il modello. Con solo due dimensioni latenti puoi disegnare direttamente lo spazio latente come un piano: passa l'intero insieme di test attraverso `modelo.encode(...)`, prendi le medie `mu` risultanti e fai un `plt.scatter` colorando ogni punto in base alla sua etichetta di cifra (l'etichetta che hai scartato nel ciclo di addestramento). **Osserva se le cifre simili — per esempio il quattro e il nove — finiscono vicine tra loro sulla mappa**
2. Con quello stesso modello a due dimensioni, scegli due cifre di test diverse, codificale entrambe per ottenere le loro medie μ₁ e μ₂, e genera dieci punti intermedi interpolando linearmente tra le due (`z_t = (1 - t) * mu1 + t * mu2`, con t tra zero e uno). Decodifica ciascun punto intermedio e mostrali in fila con `matplotlib`. **Verifica che la transizione tra le due cifre sia graduale, non un salto brusco**, e confronta mentalmente che cosa accadrebbe se provassi lo stesso esperimento con un autoencoder normale del capitolo precedente, dove non c'è alcuna garanzia che il percorso tra due punti attraversi regioni dotate di significato.
