# Capitolo 12: Architetture di agenti: da A3C a IMPALA

## Perché questo argomento ti riguarda

Nel capitolo precedente hai visto come Rainbow combini diversi miglioramenti sulla Rete Q Profonda (DQN, dall'inglese Deep Q-Network) e come la prospettiva distribuzionale cambi il modo di stimare il valore di un'azione. Tutto questo risponde a una domanda: come far imparare meglio un singolo agente? Questo capitolo risponde a una domanda diversa e, per ambienti complessi a grande scala, altrettanto importante: come far imparare più velocemente molti agenti, insieme?

Pensa ai casi d'uso che vedrai nel capitolo successivo e ultimo del manuale: sistemi capaci di giocare a Go, scacchi o StarCraft a un livello sovrumano. Nessuno di essi è stato addestrato con un singolo agente su un'unica macchina per settimane. Sono stati addestrati con centinaia o migliaia di copie dell'agente che giocano in parallelo, accumulando esperienza a una velocità che nessuna macchina isolata potrebbe raggiungere. Il salto di qualità, a quel punto, non viene più solo da un algoritmo più intelligente, ma da un'architettura capace di distribuire il lavoro. Questo capitolo ti insegna quell'architettura: prima l'Actor-Critic di Vantaggio Asincrono (A3C, dall'inglese Asynchronous Advantage Actor-Critic), che sostituisce la repetition buffer con il parallelismo, e poi IMPALA, che porta quella stessa idea da un computer a un intero cluster.

## Il collo di bottiglia che la Rete Q Profonda lascia indietro

Ricorda i due problemi di correlazione che hanno obbligato la Rete Q Profonda a incorporare una repetition buffer e una rete target: i dati che genera un agente giocando in modo continuo sono molto correlati tra passi consecutivi, e una rete neurale — per la sua tendenza naturale a produrre funzioni fluide — assegna valori simili a stati simili, il che fa sì che l'obiettivo di addestramento persegua se stesso. La repetition buffer risolve entrambi i problemi salvando fino a un milione di fotogrammi e campionando lotti casuali, ma in cambio introduce una dipendenza rigida dall'apprendimento off-policy e lascia inutilizzata buona parte della capacità di una scheda grafica: le reti piccole che funzionano bene in questo schema non arrivano a saturare le sue prestazioni, così addestrare un solo gioco Atari poteva richiedere fino a una settimana su un'unità di elaborazione grafica (GPU).

La domanda che ha aperto la porta ad A3C è stata diretta: se il problema di fondo è la correlazione dei dati, serve davvero memorizzare esperienza passata per romperla, o basta generare esperienza diversa nel momento?

## A3C: sostituisce la memoria con il parallelismo

Immagina che, invece di una sola persona che gioca una partita dopo l'altra annotando ogni mossa su un quaderno per ripassarlo più tardi, metti 16 persone a giocare partite diverse contemporaneamente, ciascuna sulla propria scacchiera. Ogni tanto, ogni giocatrice consulta la strategia comune più recente, gioca alcuni turni con essa, calcola quale aggiustamento migliorerebbe quella strategia e lo sussurra a un quaderno condiviso, senza aspettare il turno né chiedere permesso alle altre. Poiché tutte giocano situazioni diverse nello stesso momento, nessun aggiustamento individuale trascina il quaderno comune verso un angolo distorto: la diversità simultanea tra giocatrici svolge lo stesso ruolo che prima svolgeva la diversità temporale della repetition buffer, solo senza salvare nulla.

Così funziona A3C. Ogni worker — un thread di processo con la propria copia dell'ambiente — ripete questo ciclo: prende i parametri più recenti della rete condivisa, gioca un numero fisso di passi con la politica attuale, calcola un gradiente e lo applica in modo asincrono a quella rete condivisa, senza bloccare gli altri worker né sincronizzarsi con essi. Non c'è repetition buffer né rete target: è la stessa concorrenza di worker che esplorano punti diversi dell'ambiente a stabilizzare l'apprendimento.

Il componente che impara dentro ogni worker è un Actor-Critic di Vantaggio (A2C, dall'inglese Advantage Actor-Critic), la stessa famiglia di algoritmi su cui hai lavorato nel capitolo 9: una rete che predice, contemporaneamente, una distribuzione di probabilità sulle azioni (l'actor) e una stima del valore dello stato (il critic). L'actor tronca il ritorno ai successivi n passi, usa il critic per completare il resto tramite bootstrap e sottrae il valore dello stato attuale come riferimento; a questa differenza si dà il nome di vantaggio, ed è ciò che moltiplica il gradiente della politica. C'è una ragione aggiuntiva per cui questo schema è più stabile che regolare valori Q isolati: l'output dell'actor è una distribuzione che somma a uno, quindi aumentare la probabilità di un'azione riduce automaticamente quella delle altre. È una forma di stabilizzazione numerica incorporata nel design stesso, qualcosa che un valore Q senza vincoli non ha.

Gli esperimenti di scalabilità con 16 core CPU e 16 worker contro una Rete Q Profonda con GPU, su sette giochi Atari, lasciano cifre concrete: l'actor-critic ha scalato in modo sublineare (tra 12 e 13 volte più veloce con 16 thread, contro le 16 volte ideali), mentre le varianti a un passo hanno scalato in modo superlineare (circa 20 volte più veloce), grazie all'effetto stabilizzante di avere diversi worker che esplorano contemporaneamente. In Breakout, un solo worker con 50 milioni di fotogrammi arrivava a circa 250 punti; gli stessi 50 milioni di fotogrammi distribuiti tra 16 worker arrivavano a circa 300 punti, per l'effetto di mediare su una porzione maggiore dello spazio degli stati. E nonostante le varianti a un passo scalassero meglio in velocità bruta, A3C restava, nel complesso, l'opzione più efficiente nei dati.

Il risultato pratico: A3C eguagliava o superava la Rete Q Profonda su Atari in un solo giorno, su una sola macchina, senza nessuna GPU; con quattro giorni di addestramento, migliorava ancora di più. Funziona sia con reti ricorrenti che con reti feed-forward, è stato uno dei primi metodi capaci di imparare a navigare labirinti tridimensionali casuali a partire dai pixel, e si adatta al controllo continuo sostituendo la distribuzione su azioni discrete con la media e la varianza di una distribuzione gaussiana. Ha anche limiti espliciti: può essere meno stabile della Rete Q Profonda, a volte necessita di regolare gli iperparametri gioco per gioco, sfrutta peggio la GPU e, soprattutto, resta legato ai thread di un'unica macchina. Quest'ultimo limite è esattamente quello che risolve IMPALA.

## Consigli pratici di ingegneria

Prima di scalare verso qualcosa come A3C, il materiale di riferimento stesso insiste su un principio di lavoro che conviene interiorizzare a tutti i costi: fai prima debug in piccolo.

- **Inizia con un problema giocattolo**: scegli un ambiente minimo che condivida le caratteristiche chiave del problema reale (assegnazione del credito a lungo termine, dimensione dello spazio di input) ma che tu possa risolvere in minuti sul tuo laptop. L'esempio raccomandato è "Catch": una palla cade dall'alto di una griglia piccola — per esempio, cinque colonne per dieci righe — e controlli una paletta larga due o tre pixel; ricevi più uno se la cattura e meno uno se fallisce. Puoi aumentare la difficoltà poco a poco (angoli, rimbalzi, una paletta più stretta) per trovare esattamente il punto dove il tuo algoritmo comincia a fallire
- **Visualizza tutto**: curve di addestramento, comportamento della politica, predizioni del critic. Una tendenza che non si vede in una tabella di numeri salta all'occhio in un grafico
- **Diffida del design delle ricompense**: anche se in teoria la ricompensa fa parte della definizione del problema, in pratica viene regolata costantemente (per esempio, tagliare la ricompensa tra meno uno e uno in Atari). Il rischio reale sta nel sovra-specificare ricompense intermedie: in un compito robotico di raggiungere, muovere e lasciare un oggetto, premiare troppo i passi intermedi può insegnare al robot a raggiungere e muovere l'oggetto, ma mai a lasciarlo, perché tenerlo gli risulta già la politica più stabile e meglio ricompensata
- **Tratta il fattore di sconto come un iperparametro in più**: anche se l'obiettivo finale non dipende da esso in teoria, scontare riduce la varianza delle tue stime di ritorno e può adattarsi meglio al bias induttivo del tuo optimizer
- **Regola la dimensione della rete con cura**: inizia piccolo perché addestra più velocemente, ma prova reti più grandi di tanto in tanto. Le prime versioni della Rete Q Profonda su Seaquest soffrivano con reti piccole, il che ha motivato la rete target stessa; con reti più grandi, la rete target risulta meno critica, perché una rete grande confonde meno stati simili tra loro
- **Non inizializzare la politica in modo deterministico**: dal primo passo, deve campionare azioni con una certa probabilità, non ripetere sempre la stessa
- **Resta al passo con l'apprendimento supervisionato**: RMSprop e Adam sono gli optimizer di riferimento nell'RL profondo; tecniche come il *dropout* meritano di essere provate, anche se la normalizzazione a lotti non è ancora molto diffusa in questo campo

## IMPALA: da un computer a un cluster, con V-trace

A3C risolve il problema dentro una sola macchina, ma la sua architettura non scala bene oltre questo: se semplicemente distribuisci diversi learner — ciascuno con la propria copia dei pesi — tra macchine diverse e chiedi loro di sincronizzare i gradienti contro un server centrale, appare un problema nuovo. Per quando il gradiente di una macchina arriva al server, altre macchine lo hanno già aggiornato, quindi quel gradiente arriva obsoleto e finisce per spingere la rete in una direzione che non corrisponde più allo stato attuale dei pesi.

La soluzione di IMPALA (dall'inglese Importance Weighted Actor-Learner Architecture) è cambiare cosa viene distribuito. Invece di distribuire learner, centralizza l'apprendimento in un'unica copia — di solito su GPU — e distribuisce solo la parte di agire: decine o centinaia di actor su CPU percorrono l'ambiente con la politica che hanno a disposizione e inviano la loro esperienza al learner centrale. Questo evita il problema del gradiente obsoleto, ma lascia un problema fratello: per quando l'esperienza di un actor arriva al learner, i pesi di questo possono già essere cambiati, quindi quell'esperienza è stata generata da una politica leggermente diversa dall'attuale.

Qui entra V-trace, il metodo di correzione off-policy che dà il nome all'architettura. Usa il campionamento per importanza per ripesare ogni aggiornamento in base alla differenza tra la politica di comportamento μ dell'actor che ha generato i dati e la politica obiettivo π del learner nel momento di usarli. Se entrambe coincidono, l'aggiornamento di V-trace si riduce esattamente all'aggiornamento di A3C; quanto più si separano, più peso perde quell'esperienza nell'addestramento. È un compromesso esplicito tra bias e varianza: V-trace introduce un po' di bias in cambio di ridurre la varianza e mantenere l'addestramento stabile, anche con dati leggermente obsoleti.

Le cifre di prestazione in ambienti tridimensionali mostrano perché valga la pena questo riprogettamento. A3C su una sola macchina, con 64 CPU e 32 thread, elabora tra 6.000 e 9.000 fotogrammi al secondo; A3C distribuito con 200 worker arriva a circa 50.000 fotogrammi al secondo; IMPALA raggiunge tra 200.000 e 250.000 fotogrammi al secondo.

Ma la cifra più interessante non è di velocità, ma di trasferimento tra compiti. Nell'insieme di 30 compiti cognitivi di DeepMind Lab (DMLab-30), addestrare una rete separatamente su ogni compito e addestrarla su tutti i 30 insieme sono esperimenti diversi. Con A3C e una rete profonda, la prestazione media si aggira sul 30% del livello umano. IMPALA con una rete piccola supera già questa cifra; IMPALA con una rete profonda addestrata gioco per gioco migliora ancora più; ma il miglior risultato di tutti è IMPALA con una rete residuale profonda addestrata sui 30 compiti simultaneamente con un'unica rete, al di sopra persino della versione addestrata gioco per gioco separatamente. Questo è il segnale di trasferimento positivo: imparare diversi compiti contemporaneamente, con l'architettura adeguata, fa sì che ogni compito individuale venga imparato meglio e con meno dati, non peggio. In tempo di addestramento, IMPALA con rete profonda raggiunge il livello di A3C in meno di un giorno; A3C necessita circa una settimana per arrivarci.

Nell'insieme di 57 giochi Atari, la distanza si ripete: A3C distribuito addestrato separatamente su ogni gioco arriva a un 54% di punteggio medio con rete piccola e a un 118% con rete profonda; IMPALA supera A3C in entrambe le dimensioni di rete, e con una rete residuale profonda addestrata separatamente si avvicina al 200% di punteggio medio. Il dato più sorprendente, tuttavia, è un altro: IMPALA addestrato sui 57 giochi contemporaneamente, con un'unica rete, raggiunge circa il 60% di punteggio medio — comparabile alla Rete Q Profonda originale addestrata gioco per gioco — nonostante la diversità visiva e di meccaniche tra 57 giochi Atari sia molto maggiore che tra i 30 compiti di DeepMind Lab.

Resta una domanda aperta: se IMPALA è più veloce e trasferisce meglio tra compiti, perché la Rete Q Profonda resta, per certi aspetti, più stabile di A3C? La risposta punta all'addestramento a lotti: campionare un lotto casuale dalla repetition buffer è, di per sé, un ingrediente stabilizzante importante, e persino piccole deviazioni rispetto alla politica attuale influenzano l'actor-critic. V-trace compensa buona parte di questo effetto, ma se l'esperienza arriva troppo obsoleta, i pesi di importanza si avvicinano a zero o vengono tagliati a uno, e quell'esperienza smette di apportare segnale di apprendimento utile.

## Codice: "Catch" come banco di prova per un actor-critic

Prima di continuare, una precisazione importante: il codice seguente **non** implementa A3C né IMPALA davvero. A3C necessita diversi processi che giocano in parallelo e aggiornano una rete condivisa in modo asincrono; IMPALA necessita, inoltre, actor e learner distribuiti su macchine diverse con correzione V-trace tra loro. Niente di tutto questo è realistico dentro un unico notebook di Colab. Quello che puoi fare — ed è esattamente quello che raccomanda il materiale di riferimento prima di fare il salto al parallelismo — è costruire il banco di prova "Catch" descritto nella sezione precedente e addestrarci sopra un actor-critic semplice a processo singolo, con la stessa idea vista nel capitolo 9. Serve per verificare che la logica dell'algoritmo funzioni prima di complicare l'architettura.

Il codice ha bisogno solo di `numpy`, `torch` e `matplotlib`, tutte e tre preinstallate in Colab.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. El entorno "Catch": una pelota cae, una pala debe atraparla
# ---------------------------------------------------------------------
# Cuadrícula pequeña tal como la describe el material de referencia: la
# pelota aparece en una columna aleatoria de la fila superior y baja una
# fila por paso. Controlas una pala de dos píxeles de ancho en la fila
# inferior, que puedes mover a la izquierda, a la derecha o dejar quieta.
# Cuando la pelota llega a la fila inferior, recibes +1 si la pala la
# cubre y -1 si no.

FILAS = 10
COLUMNAS = 5
ANCHO_PALA = 2
N_ACCIONES = 3  # 0 = izquierda, 1 = quieta, 2 = derecha


class Catch:
    """Implementación mínima del juego de juguete "Catch"."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.fila_pelota = 0
        self.columna_pelota = np.random.randint(COLUMNAS)
        # Punto de partida aleatorio de la pala, sin salirse del borde
        self.columna_pala = np.random.randint(COLUMNAS - ANCHO_PALA + 1)
        return self._estado()

    def _estado(self):
        """Codifica el estado como una imagen binaria aplanada: un uno
        donde está la pelota y unos donde ocupa la pala."""
        grid = np.zeros((FILAS, COLUMNAS), dtype=np.float32)
        grid[self.fila_pelota, self.columna_pelota] = 1.0
        grid[FILAS - 1, self.columna_pala:self.columna_pala + ANCHO_PALA] = 1.0
        return grid.flatten()

    def step(self, accion):
        # Mueve la pala; np.clip evita que se salga de la cuadrícula
        desplazamiento = {0: -1, 1: 0, 2: 1}[accion]
        self.columna_pala = int(
            np.clip(self.columna_pala + desplazamiento, 0, COLUMNAS - ANCHO_PALA)
        )
        # La pelota cae una fila en cada paso
        self.fila_pelota += 1
        terminado = self.fila_pelota == FILAS - 1
        if terminado:
            atrapada = (
                self.columna_pala
                <= self.columna_pelota
                < self.columna_pala + ANCHO_PALA
            )
            recompensa = 1.0 if atrapada else -1.0
        else:
            recompensa = 0.0
        return self._estado(), recompensa, terminado


# ---------------------------------------------------------------------
# 2. Red actor-crítico: una base compartida y dos cabezas de salida
# ---------------------------------------------------------------------
class ActorCritico(nn.Module):
    """Red pequeña con una base compartida: una cabeza produce la
    política (probabilidad de cada acción) y otra el valor del estado."""

    def __init__(self, dim_entrada, n_acciones, dim_oculta=128):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(dim_entrada, dim_oculta),
            nn.ReLU(),
        )
        self.cabeza_politica = nn.Linear(dim_oculta, n_acciones)
        self.cabeza_valor = nn.Linear(dim_oculta, 1)

    def forward(self, x):
        h = self.base(x)
        return self.cabeza_politica(h), self.cabeza_valor(h)


# ---------------------------------------------------------------------
# 3. Un episodio completo y la actualización del actor-crítico
# ---------------------------------------------------------------------
GAMMA = 0.99
TASA_APRENDIZAJE = 1e-3
N_EPISODIOS = 3000
COEF_VALOR = 0.5
COEF_ENTROPIA = 0.01  # fomenta la exploración: la política nunca debe
                       # arrancar (ni quedarse) determinista

torch.manual_seed(0)
np.random.seed(0)

entorno = Catch()
red = ActorCritico(dim_entrada=FILAS * COLUMNAS, n_acciones=N_ACCIONES)
optimizador = torch.optim.Adam(red.parameters(), lr=TASA_APRENDIZAJE)


def ejecutar_episodio():
    """Juega un episodio completo de Catch y devuelve, paso a paso, el
    log-probabilidad de la acción elegida, el valor estimado del
    estado, la recompensa obtenida y la entropía de la política."""
    estado = entorno.reset()
    log_probs, valores, recompensas, entropias = [], [], [], []
    terminado = False
    while not terminado:
        entrada = torch.from_numpy(estado)
        logits, valor = red(entrada)
        distribucion = torch.distributions.Categorical(logits=logits)
        accion = distribucion.sample()

        estado, recompensa, terminado = entorno.step(accion.item())

        log_probs.append(distribucion.log_prob(accion))
        entropias.append(distribucion.entropy())
        valores.append(valor.squeeze())
        recompensas.append(recompensa)

    return log_probs, valores, recompensas, entropias


def actualizar_red(log_probs, valores, recompensas, entropias):
    """Calcula el retorno descontado en cada paso, la ventaja frente al
    valor estimado por el crítico, y actualiza actor y crítico a la vez."""
    retornos = []
    G = 0.0
    for r in reversed(recompensas):
        G = r + GAMMA * G
        retornos.insert(0, G)
    retornos = torch.tensor(retornos, dtype=torch.float32)

    valores = torch.stack(valores)
    ventaja = retornos - valores.detach()  # sin gradiente hacia el crítico aquí

    perdida_actor = -(torch.stack(log_probs) * ventaja).mean()
    perdida_critico = F.mse_loss(valores, retornos)
    perdida_entropia = torch.stack(entropias).mean()

    perdida = (
        perdida_actor
        + COEF_VALOR * perdida_critico
        - COEF_ENTROPIA * perdida_entropia
    )

    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()


# ---------------------------------------------------------------------
# 4. Bucle de entrenamiento y seguimiento de la tasa de aciertos
# ---------------------------------------------------------------------
aciertos = []  # 1 si el episodio terminó en captura, 0 si terminó en fallo

for episodio in range(N_EPISODIOS):
    log_probs, valores, recompensas, entropias = ejecutar_episodio()
    actualizar_red(log_probs, valores, recompensas, entropias)
    aciertos.append(1 if recompensas[-1] > 0 else 0)

# Media móvil de 100 episodios: suaviza el ruido episodio a episodio y
# deja ver la tendencia real de aprendizaje
VENTANA = 100
tasa_movil = np.convolve(aciertos, np.ones(VENTANA) / VENTANA, mode="valid")

plt.figure(figsize=(8, 5))
plt.plot(tasa_movil)
plt.xlabel("Episodios")
plt.ylabel(f"Tasa de aciertos (media móvil de {VENTANA} episodios)")
plt.title('Actor-crítico simple aprendiendo a jugar "Catch"')
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.show()
```

Eseguendolo vedrai la percentuale di successi partire da vicino al caso — con tre azioni e due posizioni di paletta in una griglia di cinque colonne, il punto di partenza si aggira sul 40% — e salire in modo sostenuto fino ad avvicinarsi a catture quasi costanti mentre la rete impara ad allineare la paletta con la colonna della palla. Se la curva si blocca, prova ad aumentare `COEF_ENTROPIA` (la politica smette di esplorare troppo presto) o a diminuire `TASA_APRENDIZAJE` (gli aggiornamenti sono troppo brusci per un problema così piccolo).

## Punti chiave

- La Rete Q Profonda risolve la correlazione dei suoi dati con una repetition buffer e una rete target, ma questo la lega all'apprendimento off-policy e non sfrutta la GPU
- A3C sostituisce la memoria con il parallelismo: molti worker con la propria copia dell'ambiente aggiornano in modo asincrono una rete condivisa, senza bloccarsi a vicenda
- La diversità simultanea tra worker rompe la correlazione dei dati come prima faceva la diversità temporale della repetition buffer, senza bisogno di memorizzare nulla
- Il componente che impara dentro ogni worker è un actor-critic di vantaggio: una politica normalizzata (che somma a uno) più un critic che serve da riferimento
- IMPALA centralizza l'apprendimento in un'unica copia e distribuisce solo l'azione tra molti actor, e usa V-trace per correggere lo sfasamento tra la politica che ha generato i dati e la politica attuale del learner
- IMPALA non è solo più veloce: addestrare diversi compiti contemporaneamente con l'architettura adeguata produce trasferimento positivo, migliorando la prestazione in ogni compito individuale
- Il gioco giocattolo "Catch" è il banco di prova raccomandato per fare debug della logica di un agente prima di complicare l'architettura con il parallelismo
- Il design delle ricompense, il fattore di sconto, la dimensione della rete e l'inizializzazione non deterministica della politica sono leve pratiche altrettanto importanti quanto l'algoritmo scelto

## Mettilo in pratica

Prendi il codice di "Catch" e fai questo esperimento per conto tuo:

- **Cambia la difficoltà dell'ambiente**: riduci `ANCHO_PALA` a uno o amplia `COLUMNAS` a otto o dieci, senza toccare nient'altro. Riaddestra e confronta quanti episodi serve adesso all'agente per raggiungere la stessa percentuale di successi di prima
- **Osserva dove si rompe l'apprendimento**: continua ad aumentare `COLUMNAS` o a diminuire `ANCHO_PALA` finché non trovi il punto in cui la curva smette di salire chiaramente in 3.000 episodi

Annota in quale configurazione esatta il tuo agente comincia a fallire e quale cambiamento — più episodi, più entropia, una rete più grande — lo rimette a posto! Quell'esercizio di sondare i limiti di un problema piccolo è, parola per parola, il consiglio pratico centrale di questo capitolo.

---

Ora conosci i pezzi: come stabilizzare una rete con repetition buffer e rete target, come sostituire quella memoria con il parallelismo tramite A3C, e come scalare quel parallelismo a un intero cluster con IMPALA. Nel capitolo 13, l'ultimo del manuale, vedrai come queste idee si sono combinate — insieme a ricerca ad albero e autoapprendimento — per costruire sistemi come AlphaGo, AlphaZero e AlphaStar.
