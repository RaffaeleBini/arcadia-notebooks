# Reti generative avversarie (GAN): l'arte di imparare giocando

## Perché questo argomento è importante

Finora hai visto reti che classificano, prevedono o traducono. In questo capitolo cambi prospettiva: studierai reti che creano. Una rete generativa avversaria (GAN, dall'inglese *Generative Adversarial Network*) non risponde alla domanda "che cos'è questo?", ma a una molto più ambiziosa: "come genero qualcosa di nuovo che sembri reale?".

Da quando Ian Goodfellow e il suo team hanno presentato l'idea nel 2014, le GAN sono passate dal generare cifre sfocate in bianco e nero alla produzione di volti umani fotorealistici, scene complete o traduzioni tra stili di immagine. Questo salto spiega perché questo tema è così importante nella pratica:

- **Generazione di immagini e arte con intelligenza artificiale (IA)**. Strumenti di creazione visiva, design di prodotto o prototipazione usano varianti di GAN per produrre texture, volti sintetici o illustrazioni a partire da un semplice seme di rumore
- **Aumento di dati sintetici**. Quando addestri un modello con pochi esempi reali (immagini mediche, difetti di fabbricazione, segnali poco frequenti), una GAN ben addestrata ti permette di generare esempi aggiuntivi realistici per rafforzare il set di addestramento
- **Traduzione e modifica di immagini**. Trasformare uno schizzo in una fotografia, cambiare la stagione di un paesaggio o trasferire lo stile di un pittore a un'altra immagine sono applicazioni dirette di questa famiglia di modelli
- **Deepfake**. La stessa capacità che genera un volto sintetico convincente può sostituire il volto di una persona reale in un video senza il suo consenso

Quest'ultimo punto non è un dettaglio di poco conto. La stessa tecnologia che permette di aumentare dati medici scarsi permette anche di fabbricare prove audiovisive false. Più capisci come funziona il generatore e che cosa lo rende convincente, meglio sei preparato a riconoscerne i limiti, individuarne gli artefatti e valutare i rischi che comporta. Per questo motivo questo capitolo è anche l'anticamera del blocco dedicato all'IA responsabile: non puoi ragionare sui rischi di una tecnologia se prima non capisci come funziona dall'interno.

## Spiegazione concettuale con analogie

Immagina un falsario di banconote e la polizia che cerca di individuarlo. All'inizio il falsario produce banconote molto grossolane: carta sbagliata, inchiostro che sbava, proporzioni errate. La polizia le scopre senza sforzo. Ma il falsario impara da ogni rifiuto: sistema la carta, corregge il colore, migliora i dettagli. La polizia, a sua volta, impara: affina il proprio criterio, controlla le filigrane, confronta con più attenzione. Col tempo, entrambe le parti migliorano in parallelo, e le banconote false diventano sempre più difficili da distinguere da quelle autentiche.

Questa è esattamente la dinamica di una GAN. Il **generatore** svolge il ruolo del falsario: parte da rumore casuale e produce dati — immagini, nel caso tipico — che cerca di far passare per reali. Il **discriminatore** svolge il ruolo della polizia: riceve sia dati reali sia dati generati e deve decidere, per ogni campione, se proviene dal dataset vero o dalla fabbrica del generatore.

Nessuna delle due reti vede direttamente l'obiettivo finale dell'altra. Il generatore non osserva mai dati reali: riceve solo il segnale che gli dà il discriminatore su se le sue creazioni passano o no. Il discriminatore, da parte sua, funziona come un **insegnante** che non si limita a separare il reale dal falso, ma fornisce anche un gradiente — una direzione di miglioramento — che il generatore usa per correggere il suo prossimo lotto di "banconote". Più il poliziotto discrimina bene, più informazione riceve il falsario su dove sbaglia; più il falsario falsifica bene, più esigente deve diventare la polizia. L'addestramento si conclude, idealmente, quando la polizia non riesce a fare meglio che lanciare una moneta: a quel punto le banconote false e quelle reali sono statisticamente indistinguibili.

## Sviluppo teorico

### Modelli generativi espliciti e impliciti

Un modello generativo cerca di apprendere, a partire da un campione di dati, la distribuzione sottostante che li ha prodotti. Esistono due grandi famiglie. I **modelli espliciti** apprendono direttamente una distribuzione di probabilità sui dati: puoi chiedere loro la verosimiglianza di un nuovo punto e, inoltre, campionare da essi. Rientrano in questa categoria l'analisi delle componenti principali probabilistica, l'analisi fattoriale, le miscele gaussiane o modelli neurali come PixelCNN, PixelRNN o WaveNet, addestrati per massima verosimiglianza.

I **modelli impliciti**, invece, non modellano esplicitamente la densità di probabilità. Apprendono piuttosto un simulatore capace di generare nuovi campioni con le stesse proprietà statistiche dei dati originali, senza passare in nessun momento per una formula di probabilità. La GAN è l'esempio più rappresentativo di questa seconda famiglia.

### L'architettura e l'obiettivo minimax

Una GAN è formata da due reti neurali profonde addestrate simultaneamente:

- Il **generatore** $G$ prende un vettore di rumore latente $z$, campionato da una distribuzione a priori nota $p(z)$ — tipicamente una gaussiana multivariata — e lo trasforma tramite una rete deterministica in un campione generato $G(z)$. Il rumore in ingresso è ciò che apporta varietà: a differenza di un autoencoder variazionale, il generatore di una GAN non produce una distribuzione sull'uscita, quindi tutta la casualità dei dati deve entrare tramite $z$
- Il **discriminatore** $D$ riceve un campione $x$ — reale o generato — e restituisce un valore $D(x)$ interpretabile come la probabilità che quel campione provenga dal dataset reale e non dal generatore

L'obiettivo congiunto si formula come un problema minimax a somma zero:

$$\min_G \max_D \; V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

Leggilo a pezzi. Il primo termine premia il discriminatore quando assegna alta probabilità ai dati reali. Il secondo lo premia quando assegna bassa probabilità — cioè riconosce correttamente — ai dati generati. Il discriminatore vuole **massimizzare** questa somma: vuole classificare bene su entrambi i fronti. Il generatore, invece, vuole **minimizzarla**: vuole che $D(G(z))$ si avvicini a uno, in modo che il secondo termine diventi molto negativo e il discriminatore sbagli.

In pratica, i due obiettivi si separano in due funzioni di perdita che vengono ottimizzate in modo alternato. Il discriminatore massimizza:

$$\max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

e il generatore minimizza:

$$\min_G \; \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

Quel secondo termine ha un problema pratico: all'inizio dell'addestramento, quando il generatore è scarso, $D(G(z))$ è quasi zero e il gradiente di $\log(1-D(G(z)))$ è molto piatto, quindi spinge appena il generatore a migliorare. Per questo, in pratica si usa una variante **non saturante**, in cui il generatore massimizza $\log D(G(z))$ invece di minimizzare $\log(1-D(G(z)))$: l'obiettivo punta nella stessa direzione, ma fornisce gradienti molto più utili all'inizio dell'addestramento.

### Il ciclo di addestramento

Addestrare una GAN significa alternare passi di discesa (o salita) del gradiente stocastico tra le due reti:

1. Campiona un lotto di dati reali e un lotto di rumore; genera dati falsi con $G$
2. Aggiorna $D$ con pochi passi di gradiente per massimizzare la sua capacità di distinguere i reali dai generati
3. Campiona un nuovo lotto di rumore, genera dati con $G$ e aggiorna $G$ per minimizzare la probabilità che $D$ indovini
4. Ripeti il ciclo migliaia o milioni di volte

Non si addestra il discriminatore fino all'ottimalità a ogni passo: farlo sarebbe computazionalmente molto costoso e, inoltre, un discriminatore perfetto può smettere di fornire al generatore un gradiente utile. Per questo l'equilibrio pratico consiste nel dare solo pochi passi di gradiente al discriminatore prima di aggiornare il generatore.

Questo processo configura un **gioco a somma zero**: ciò che guadagna il generatore lo perde il discriminatore, e viceversa. L'obiettivo teorico è raggiungere un **equilibrio di Nash**, un punto in cui nessuna delle due reti può migliorare il proprio risultato cambiando unilateralmente la propria strategia mentre l'altra resta ferma. In quell'equilibrio, il discriminatore non riesce più a distinguere i dati reali da quelli generati meglio che a caso.

### GAN come minimizzazione di una divergenza

I modelli generativi, in generale, si possono intendere come la minimizzazione di una distanza o divergenza tra la distribuzione reale dei dati e la distribuzione prodotta dal modello. L'approccio più comune, la massima verosimiglianza, equivale a minimizzare la divergenza di Kullback-Leibler (KL):

$$D_{KL}(p \| q) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]$$

dove $p$ è la distribuzione reale dei dati e $q$ la distribuzione indotta dal modello.

Si può dimostrare che, se il discriminatore di una GAN raggiunge il suo valore ottimo, il generatore sta minimizzando in modo implicito la divergenza di Jensen-Shannon (JS) tra la distribuzione reale e quella generata:

$$D_{JS}(p \| q) = \tfrac{1}{2} D_{KL}(p \| m) + \tfrac{1}{2} D_{KL}(q \| m), \qquad m = \tfrac{p+q}{2}$$

Questa divergenza vale zero solo quando le due distribuzioni sono identiche, il che sostiene intuitivamente l'obiettivo dell'addestramento. Tuttavia, questo collegamento teorico si basa su un'ipotesi che raramente si verifica: che il discriminatore sia ottimale a ogni passo. In pratica, il discriminatore ha capacità limitata, viene addestrato con risorse computazionali finite e dispone solo di campioni, non dell'intera distribuzione reale.

### Perché l'addestramento di una GAN è così instabile

Qui arrivi al nucleo del problema che motiva buona parte dei progressi descritti più avanti. Le divergenze KL e JS condividono una debolezza: se la distribuzione del modello e quella reale non si sovrappongono — cosa molto frequente all'inizio dell'addestramento, quando il generatore produce dati assurdi — il rapporto di probabilità che compare nella loro definizione tende all'infinito. Il risultato è che la divergenza smette di dare un segnale di apprendimento utile: anche se il modello si avvicina progressivamente ai dati reali, la divergenza può rimanere infinita o restare invariata in modo apprezzabile, quindi il gradiente non informa sui progressi reali.

Questa fragilità teorica si traduce in diversi sintomi pratici che incontrerai più volte addestrando GAN:

- **Instabilità del gioco minimax**. Poiché le due reti si aggiornano simultaneamente perseguendo obiettivi opposti, il sistema può oscillare senza convergere, invece di stabilizzarsi sull'equilibrio di Nash atteso. Un discriminatore che migliora troppo in fretta lascia il generatore senza gradiente utile; uno che migliora troppo lentamente lascia passare dati di scarsa qualità
- **Collasso di modalità (*mode collapse*)**. Il generatore scopre che poche uscite concrete ingannano bene il discriminatore e smette di esplorare il resto della distribuzione reale. Il risultato è una varietà molto limitata di campioni, anche se ciascuno, preso singolarmente, sembra convincente. Individuarlo è difficile perché la funzione di perdita del generatore non sempre riflette direttamente questo problema
- **Assenza di una metrica di progresso affidabile**. Essendo un gioco tra due giocatori che migliorano contemporaneamente, il valore della perdita non decresce in modo monotono come in un problema di ottimizzazione classico, quindi non puoi leggere direttamente dalla curva di perdita se il modello sta migliorando

In pratica, inoltre, la GAN non realizza una minimizzazione esatta della divergenza JS, proprio perché il discriminatore non è una funzione arbitraria ma una rete neurale — o una funzione appartenente a una famiglia limitata, come uno spazio di Hilbert con nucleo riproducente. Questa limitazione, lungi dall'essere solo un difetto, risulta vantaggiosa: essendo un'approssimazione regolare, il discriminatore non può rappresentare rapporti infiniti, quindi continua a fornire gradiente anche quando le distribuzioni reale e generata non si sovrappongono ancora. Questo spiega, in parte, perché le GAN funzionano meglio nella pratica di quanto la teoria della divergenza JS lascerebbe supporre.

### Progressi che migliorano la stabilità

La fonte descrive diverse linee di lavoro, ciascuna delle quali attacca una causa diversa di instabilità.

**Reti generative avversarie profonde convoluzionali (DCGAN)**. Sostituiscono i perceptron multistrato (MLP) originali con reti convoluzionali sia nel generatore sia nel discriminatore, e aggiungono la normalizzazione per batch (*batch normalization*), che rende il processo di addestramento notevolmente più fluido. Inoltre, interpolare tra due vettori di rumore produce immagini intermedie semanticamente coerenti, segno che lo spazio latente appreso ha una struttura significativa.

**Rete generativa avversaria di Wasserstein (WGAN)**. Invece della perdita di entropia incrociata binaria basata su JS, usa la distanza di Wasserstein (chiamata anche distanza del movimento terra), che non dipende da un rapporto di probabilità ma da una differenza di aspettative, per cui continua a dare un segnale utile anche senza sovrapposizione dei supporti. La sua formulazione duale, ristretta alle funzioni 1-Lipschitz — quelle per cui $|f(x_1) - f(x_2)| \le |x_1 - x_2|$ —, è:

$$W(p, q) = \sup_{\|D\|_L \le 1} \; \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

L'obiettivo completo della WGAN è quindi:

$$\min_G \max_{D \in \text{1-Lipschitz}} \; \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

Qui il discriminatore — chiamato "critico" in questo contesto — deve restare 1-Lipschitz, imposizione che si ottiene tramite ritaglio dei pesi (*weight clipping*) o, in modo più efficace, tramite penalizzazione del gradiente.

**Altre divergenze**. La fonte menziona anche la discrepanza media massima (MMD, *maximum mean discrepancy*), che opera in uno spazio di Hilbert con nucleo riproducente (RKHS), e le f-divergenze, addestrabili tramite un limite inferiore variazionale simile a quello usato negli autoencoder variazionali. Tutte condividono la stessa struttura: minimizzazione rispetto al generatore, massimizzazione rispetto al discriminatore.

**Reti generative avversarie condizionali (cGAN)**. Aggiungono un'informazione ausiliaria $y$ — per esempio un'etichetta di classe — sia al generatore sia al discriminatore, il che permette di controllare che cosa si genera invece di lasciarlo al caso del rumore:

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x \mid y)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z \mid y) \mid y))]$$

**Normalizzazione spettrale (SNGAN)**. Limita a uno i valori singolari dei pesi del discriminatore in ogni strato, il che impedisce che questo risolva il suo obiettivo semplicemente aumentando la grandezza dei suoi pesi e lo obbliga a migliorare davvero i segnali di gradiente che trasmette al generatore.

**Reti di attenzione (AttnGAN, SAGAN)**. Incorporano meccanismi di autoattenzione per catturare dipendenze globali tra regioni distanti dell'immagine — per esempio, che la testa e la coda di un animale siano coerenti tra loro —, cosa che i livelli convoluzionali, limitati al loro campo recettivo locale, non ottengono da soli.

**Crescita progressiva (PGGAN) e StyleGAN**. Le GAN a crescita progressiva si addestrano prima a bassa risoluzione (per esempio, 4x4 pixel) e aggiungono gradualmente strati fino a raggiungere l'alta risoluzione, il che stabilizza l'addestramento in ogni fase. StyleGAN, su questa stessa base, separa lo spazio latente in un vettore globale — trasformato tramite una rete completamente connessa in uno spazio latente intermedio — e rumore spaziale iniettato in ogni strato, il che offre un controllo fine su attributi come l'espressione del viso o l'acconciatura.

**BigGAN**. Scala l'addestramento con lotti molto grandi (fino a 2048 campioni), architetture più profonde e tecniche combinate — funzione di perdita a bisturi (*hinge loss*), normalizzazione spettrale, autoattenzione, discriminatore per proiezione, regolarizzazione ortogonale — per generare campioni ad alta fedeltà condizionati per classe su dataset tanto diversi quanto ImageNet. Introduce inoltre il "trucco del troncamento": ridurre la deviazione standard del rumore in ingresso al momento dell'inferenza produce campioni più prototipici e di maggiore fedeltà, a costo di una minore varietà.

**Tecniche contro il collasso di modalità**. La fonte cita esplicitamente la discriminazione per minibatch (*minibatch discrimination*), le GAN srotolate (*unrolled GANs*) e la normalizzazione spettrale come tecniche orientate a diversificare i campioni generati e mitigare il collasso di modalità.

### Come si valuta una GAN

Essendo un modello implicito, una GAN non permette di calcolare verosimiglianze, quindi non puoi valutarla come valuteresti un modello esplicito. La fonte descrive due metriche principali. Il **punteggio Inception (IS)** usa un classificatore preaddestrato su ImageNet per confrontare, tramite divergenza KL, la distribuzione delle etichette dei dati generati con quella attesa: premia la diversità di classi e la confidenza di classificazione, anche se non rileva se, all'interno di una stessa classe, i campioni sono ripetitivi. La **distanza di Fréchet tra Inception (FID)** confronta invece la distribuzione delle caratteristiche interne del classificatore — non solo le etichette di uscita — tra dati reali e generati, il che cattura meglio sia la diversità tra classi sia quella all'interno di ciascuna classe; la sua variante corretta per campioni di piccole dimensioni è la distanza di Fréchet del nucleo (KID). Nessuna metrica, da sola, basta: la pratica abituale combina più metriche, insieme all'ispezione dei vicini più prossimi nello spazio delle caratteristiche per verificare che il modello generalizzi e non memorizzi.

## Esempio di codice commentato, in PyTorch

Il seguente esempio implementa una GAN minimale su dati sintetici bidimensionali: una miscela di otto gaussiane disposte in cerchio, il classico esempio giocattolo per osservare sia l'apprendimento sia il rischio di collasso di modalità. Il generatore e il discriminatore hanno solo un paio di strati lineari. Puoi eseguire il codice direttamente in Google Colab.

```python
# Instala/asegura las dependencias en Colab (normalmente ya están disponibles)
# !pip install torch matplotlib numpy -q

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Datos reales: mezcla de ocho gaussianas en círculo ---
def sample_real_data(n_samples, n_modes=8, radius=4.0, std=0.15):
    angles = 2 * np.pi * np.random.randint(0, n_modes, size=n_samples) / n_modes
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    points = centers + std * np.random.randn(n_samples, 2)
    return torch.tensor(points, dtype=torch.float32)

LATENT_DIM = 2

# --- Generador: ruido latente -> punto 2D ---
class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z):
        return self.net(z)

# --- Discriminador: punto 2D -> logit de "realidad" ---
class Discriminator(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),  # logit, sin sigmoide: se aplica dentro de la pérdida
        )

    def forward(self, x):
        return self.net(x)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

# BCEWithLogitsLoss combina sigmoide + entropía cruzada binaria de forma numéricamente estable
criterion = nn.BCEWithLogitsLoss()
opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

BATCH_SIZE = 256
N_STEPS = 6000

for step in range(1, N_STEPS + 1):
    # ---------- 1) Actualiza el discriminador ----------
    real_data = sample_real_data(BATCH_SIZE).to(device)
    z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
    fake_data = generator(z).detach()  # detach: el generador no se actualiza en este paso

    real_labels = torch.ones(BATCH_SIZE, 1, device=device)
    fake_labels = torch.zeros(BATCH_SIZE, 1, device=device)

    loss_d_real = criterion(discriminator(real_data), real_labels)
    loss_d_fake = criterion(discriminator(fake_data), fake_labels)
    loss_d = loss_d_real + loss_d_fake

    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()

    # ---------- 2) Actualiza el generador (pérdida no saturante) ----------
    z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
    fake_data = generator(z)
    # El generador "quiere" que D clasifique sus muestras como reales (etiqueta 1)
    loss_g = criterion(discriminator(fake_data), real_labels)

    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()

    if step % 1000 == 0:
        print(f"paso {step:5d} | pérdida D: {loss_d.item():.3f} | pérdida G: {loss_g.item():.3f}")

# --- Visualización: datos reales frente a muestras generadas antes y después del entrenamiento ---
with torch.no_grad():
    real_sample = sample_real_data(1000).numpy()
    z_vis = torch.randn(1000, LATENT_DIM, device=device)
    generated_sample = generator(z_vis).cpu().numpy()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(real_sample[:, 0], real_sample[:, 1], s=10, alpha=0.5, label="Datos reales")
ax.scatter(generated_sample[:, 0], generated_sample[:, 1], s=10, alpha=0.5, label="Muestras generadas")
ax.set_title("GAN mínima: datos reales frente a datos generados")
ax.legend()
ax.set_aspect("equal")
plt.show()
```

Quando lo esegui, osserva l'andamento delle due perdite stampate ogni mille passi: raramente decrescono in modo monotono, esattamente il comportamento instabile descritto dalla teoria. Nel grafico finale, verifica se il generatore copre tutte e otto le modalità della miscela o se, al contrario, concentra i suoi campioni solo su due o tre di esse: questo è il collasso di modalità in azione, e con un'architettura così semplice è frequente che compaia in qualcuna delle esecuzioni.

## Riepilogo dei punti chiave

- Una GAN è un modello generativo implicito composto da un generatore e un discriminatore che competono in un gioco minimax a somma zero
- L'obiettivo classico è $\min_G \max_D \; \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$, e in pratica il generatore si addestra con una variante non saturante per evitare gradienti piatti all'inizio
- Se il discriminatore fosse ottimale, il generatore minimizzerebbe la divergenza di Jensen-Shannon, ma in pratica il discriminatore non è mai ottimale, quindi la GAN non realizza una minimizzazione esatta di divergenza
- La mancanza di sovrapposizione tra la distribuzione reale e quella generata fa perdere utilità alle divergenze KL e JS come segnale di apprendimento, una delle radici dell'instabilità dell'addestramento
- Il collasso di modalità, l'instabilità dell'equilibrio del gioco e l'assenza di una metrica di progresso affidabile sono i tre problemi centrali dell'addestramento delle GAN
- La WGAN sostituisce la divergenza JS con la distanza di Wasserstein, calcolata su funzioni 1-Lipschitz, che offre un gradiente più stabile
- DCGAN, cGAN, SNGAN, SAGAN, PGGAN, StyleGAN, BigGAN e LOGAN sono progressi descritti nella fonte, ciascuno dei quali attacca una causa diversa di instabilità o di scarsa qualità dei campioni
- Valutare una GAN richiede di combinare più metriche — punteggio Inception, distanza di Fréchet tra Inception (FID), ispezione dei vicini più prossimi — perché nessuna, da sola, cattura contemporaneamente qualità e diversità
- La stessa capacità generativa che produce dati sintetici utili rende possibili anche i deepfake, il che collega direttamente questo tema all'etica e alla responsabilità nello sviluppo dell'IA

## Mettilo in pratica

**Dedica tra quindici e trenta minuti a sperimentare con il codice precedente.** Segui questi passi:

1. **Esegui** l'esempio così com'è in Google Colab e osserva il grafico finale
2. **Riduci** `N_STEPS` a 500 e riesegui: verifica se il generatore riesce a coprire solo una o due modalità della miscela, un collasso di modalità causato da un addestramento insufficiente
3. **Modifica** l'architettura del generatore in modo che abbia un solo strato nascosto invece di due, e confronta la qualità finale dei campioni
4. **Cambia** la perdita del generatore dalla variante non saturante (`criterion(discriminator(fake_data), real_labels)`) alla variante originale che minimizza `criterion(discriminator(fake_data), fake_labels)` con segno invertito, e osserva se l'addestramento diventa più lento o instabile all'inizio

**Annota** in due o tre frasi quale differenza hai osservato tra le varianti e perché pensi che accada, mettendola in relazione con quanto spiegato sui gradienti piatti e sul collasso di modalità nello sviluppo teorico.
