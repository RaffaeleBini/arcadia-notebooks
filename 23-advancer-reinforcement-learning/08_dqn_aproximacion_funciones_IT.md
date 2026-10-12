# Capitolo 8. Approssimazione di funzioni e Deep Q-Networks (DQN)

Pensa a una scacchiera. Ogni disposizione possibile dei pezzi è uno stato diverso, e il numero totale di stati supera di gran lunga la quantità di atomi dell'universo osservabile. Oppure pensa a un videogioco: ogni fotogramma di pixel sullo schermo è, in pratica, uno stato nuovo. Nel capitolo precedente hai imparato il Q-learning tabulare, una tecnica di Apprendimento per Rinforzo (RL, dall'inglese *Reinforcement Learning*) che memorizza un valore esatto per ogni coppia stato-azione in una tabella. Questa tabella funziona benissimo nei problemi piccoli, ma si rompe non appena il numero di stati esplode: non sta in memoria e, soprattutto, il tuo agente quasi non torna mai a visitare esattamente lo stesso stato due volte. Senza una tabella che lo ricordi, non impara nulla di utile su uno stato che vede una sola volta.

In questo capitolo risolverai il problema con un'idea semplice: invece di memorizzare un valore per ogni stato, addestra una rete neurale che impari a generalizzare, cioè che stimi il valore di uno stato nuovo a partire da ciò che ha già imparato in stati simili. Questa idea è la porta d'ingresso all'Apprendimento per Rinforzo Profondo (Deep Reinforcement Learning), la famiglia di metodi che combina RL con reti neurali a più strati. Conoscerai la Deep Q-Network (DQN), l'algoritmo che nel 2015 ha imparato a giocare a decine di videogiochi Atari partendo direttamente dai pixel dello schermo, senza che nessuno gli spiegasse le regole di ciascun gioco.

## Dalle tabelle alle funzioni: l'idea dell'approssimazione di funzioni

In RL, un agente impara sempre una di queste tre cose: una politica (quale azione compiere in ogni stato), una funzione di valore (quanto vale ogni stato o ogni coppia stato-azione) oppure un modello (a quale stato ti porta ogni azione). Tutte e tre sono, matematicamente, funzioni. L'approssimazione di funzioni consiste nel rappresentare quella funzione con un insieme di parametri regolabili, invece che con una tabella esatta: `Q(s,a;θ) ≈ Q_vero(s,a)`, dove θ rappresenta questi parametri. Il tuo obiettivo è trovare i valori di θ che fanno avvicinare il più possibile questa approssimazione al valore reale.

Qui serve un breve chiarimento, nel caso in cui le reti neurali ti siano ancora un territorio nuovo: una rete neurale è, in sostanza, una funzione con parametri regolabili (i pesi), organizzata in strati che trasformano l'input passo dopo passo fino a produrre un output. Addestrarla significa modificare quei pesi, poco a poco e in modo automatico, finché la funzione che calcola somiglia sempre di più alla funzione che vuoi imitare; nel tuo caso, il valore di ogni azione possibile in ogni stato.

Perché questo cambiamento vale la pena? Per due motivi. Il primo è la generalizzazione: un agente che approssima Q con una funzione può stimare il valore di uno stato mai visitato, appoggiandosi a stati simili che invece conosce. Il secondo è che non devi più ottimizzare la politica alla perfezione a ogni passo; basta migliorare l'approssimazione in modo incrementale, ciclo dopo ciclo, perché anche la politica migliori di conseguenza.

Esistono vari tipi di approssimatori: combinazioni lineari di caratteristiche costruite a mano (utili e ben comprese, ma limitate), alberi decisionali, metodi dei vicini più prossimi e, l'opzione dominante oggi, le reti neurali. Il loro grande vantaggio è che possono imparare direttamente da segnali grezzi, come i pixel di uno schermo, senza che tu debba progettare a mano nessuna caratteristica. Questa idea non è nuova: negli anni ottanta Gerald Tesauro ha combinato l'apprendimento per differenza temporale (Temporal Difference, TD) con una rete neurale per addestrare un agente capace di giocare a backgammon, e Chris Watkins ha formulato il Q-learning nel 1989. L'intuizione di unire reti neurali e RL circola da decenni; ciò che è cambiato è la potenza di calcolo disponibile, che oggi permette di metterla in pratica su larga scala.

## Perché unire semplicemente Q-learning e una rete neurale è instabile

La versione ingenua è allettante: sostituisci la tabella del Q-learning con una rete neurale che riceve lo stato e restituisce un valore per ogni azione, scegli l'azione con epsilon-greedy, calcoli l'errore tra ciò che la rete ha previsto e ciò che "avrebbe dovuto" prevedere, e aggiusti i pesi con la discesa del gradiente. Sulla carta funziona. Nella pratica, addestrata così, la rete tende a oscillare senza convergere oppure diverge direttamente. Ci sono due motivi.

Il primo è che i dati generati da un agente di RL non sono indipendenti né identicamente distribuiti, un'ipotesi che quasi tutto l'addestramento delle reti neurali dà per scontata. I passi consecutivi di un episodio sono fortemente correlati tra loro: lo stato al passo cento assomiglia moltissimo allo stato al passo 101. È la differenza tra addestrare con foto casuali di migliaia di scene diverse e addestrare guardando, fotogramma dopo fotogramma, gli stessi secondi di un unico film: la rete si specializza in ciò che ha appena visto e dimentica ciò che ha visto prima.

Il secondo motivo è la non stazionarietà introdotta dal bootstrapping. Nel Q-learning, il target che la tua rete insegue viene calcolato usando la rete stessa: `target = r + γ · max_a' Q(s',a';θ)`. Ogni volta che regoli i pesi θ per avvicinarti a quel target, il target si sposta anche lui, perché dipende proprio da quei pesi. È come cercare di colpire un bersaglio che si sposta ogni volta che spari, e che per di più si sposta proprio perché hai sparato. A ciò si aggiunge che la politica che decide quali dati raccoglie l'agente cambia insieme alla rete, quindi i dati di addestramento cambiano mentre cambia la rete stessa che stai cercando di addestrare.

## I due ingredienti che stabilizzano DQN: replay delle esperienze e rete target

DQN (Mnih e collaboratori, 2015) risolve entrambi i problemi aggiungendo due ingredienti al Q-learning con reti neurali.

**Buffer di replay delle esperienze (experience replay).** Invece di addestrarsi sulla transizione che l'agente ha appena vissuto e scartarla, memorizza ogni transizione `(s, a, r, s')` in una memoria e, a ogni passo di addestramento, campiona a caso un piccolo lotto da quella memoria. Questo porta diversi vantaggi insieme: rompe la correlazione tra passi consecutivi (il lotto mescola esperienze di momenti molto diversi), sfrutta ogni transizione più volte invece che una sola, e rende più uniforme la distribuzione dei dati di addestramento perché combina esperienze raccolte con politiche leggermente diverse nel corso dell'addestramento. Il risultato assomiglia molto di più ai dati "ben educati" che una rete neurale addestrata con apprendimento supervisionato si aspetta.

**Rete target (target network).** Mantieni una seconda copia della rete, con parametri propri θ⁻, e usala solo per calcolare il target di addestramento. Questa rete target non viene aggiornata a ogni passo: viene sincronizzata con la rete principale ogni certo numero fisso di passi, copiandone i pesi. Nel frattempo resta congelata. Così, il target che la rete principale insegue smette di muoversi a ogni passo e diventa un riferimento stabile per un tratto di addestramento, il che riduce le oscillazioni e aiuta l'addestramento a convergere.

Con entrambi questi ingredienti, la funzione di perdita di DQN si scrive, in modo intuitivo, così:

`L(θ) = E [ (r + γ · max_a' Q(s',a';θ⁻) − Q(s,a;θ))² ]`

dove il valore atteso si calcola su mini-lotti campionati dal buffer di replay, θ sono i parametri della rete principale (quella che viene addestrata) e θ⁻ quelli della rete target (congelata tra una sincronizzazione e l'altra). Un dettaglio tecnico importante: nel calcolo del gradiente applichi uno "stop-gradient" sul termine del target, in modo che l'aggiornamento dei pesi scorra solo verso θ e mai verso θ⁻. Senza questo dettaglio, staresti spostando il target stesso per avvicinarlo alla tua previsione, invece del contrario.

Una sfumatura interessante da ricordare: la necessità della rete target non è assoluta. Più una rete ha capacità (più parametri, più neuroni), meno generalizza in modo aggressivo tra stati diversi, e questo di per sé riduce parte dell'instabilità. Ciononostante, la rete target resta una componente standard nella stragrande maggioranza delle implementazioni di DQN, insieme a un fattore di sconto tipico vicino a 0,99 e a un'esplorazione epsilon-greedy con decadimento.

DQN ha inoltre diversi miglioramenti successivi che per ora conoscerai solo di nome, perché li implementerai nel capitolo 11:

- **Double DQN** separa la scelta della migliore azione successiva (con la rete principale) dalla valutazione di quell'azione (con la rete target), per ridurre il bias di sovrastima tipico del Q-learning
- **Dueling DQN** divide la rete in due rami, uno che stima il valore dello stato e uno che stima il vantaggio (advantage) di ogni azione rispetto a quel valore, e li combina alla fine, il che migliora la robustezza della stima
- **Prioritized Experience Replay** campiona il buffer dando più peso alle transizioni con maggiore errore di previsione, invece di campionare tutte con la stessa probabilità, per imparare più in fretta da ciò che sorprende di più la rete

## Esempio in Python: DQN minimo su CartPole

Addestrerai un DQN su CartPole, l'ambiente classico in cui un carrello deve mantenere in equilibrio un pendolo muovendosi a sinistra o a destra. Lo stato è composto da quattro numeri (posizione e velocità del carrello, angolo e velocità angolare del pendolo) e le azioni possibili sono due. È un ambiente molto più semplice di Atari, ma contiene esattamente gli stessi ingredienti: rete Q, buffer di replay e rete target.

Copia ed esegui questo codice in Google Colab. L'unica dipendenza esterna di tutto il capitolo è `gymnasium`, la libreria che fornisce l'ambiente CartPole; PyTorch, `numpy` e `matplotlib` sono già preinstallati.

```python
# Instalamos gymnasium: es la ÚNICA dependencia externa de este capítulo.
# PyTorch, numpy y matplotlib ya vienen preinstalados en Google Colab.
!pip install gymnasium
```

```python
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym

# Semilla para que el experimento sea, en la medida de lo posible, reproducible
SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

# --- Hiperparámetros ---
# Ajustados para entrenar rápido en una CPU normal de Colab, priorizando
# claridad sobre rendimiento máximo.
NOMBRE_ENTORNO = "CartPole-v1"
NUM_EPISODIOS = 300                    # episodios de entrenamiento
GAMMA = 0.99                           # factor de descuento
TASA_APRENDIZAJE = 1e-3
TAMANO_BUFFER = 10000                  # capacidad del buffer de repetición
TAMANO_LOTE = 64                       # tamaño del mini-lote muestreado en cada paso
EPSILON_INICIAL = 1.0
EPSILON_FINAL = 0.05
EPISODIOS_DECAIMIENTO = 200            # episodios en los que epsilon baja de inicial a final
PASOS_ACTUALIZAR_RED_OBJETIVO = 500    # cada cuántos pasos se sincroniza la red objetivo
PASOS_MINIMOS_ANTES_ENTRENAR = 1000    # esperamos a tener algo de experiencia antes de entrenar


class RedQ(nn.Module):
    """Aproximador de funciones: recibe un estado y devuelve un valor
    estimado para cada acción posible. Es una red neuronal pequeña
    (un "perceptrón multicapa"), suficiente para un estado de solo
    cuatro números como el de CartPole."""

    def __init__(self, dim_estado, num_acciones):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Linear(dim_estado, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_acciones),
        )

    def forward(self, estado):
        return self.capas(estado)


# Cada transición guardada en el buffer es una tupla con nombre, para que
# el código sea más legible al construir los mini-lotes
Transicion = namedtuple(
    "Transicion", ["estado", "accion", "recompensa", "siguiente_estado", "terminado"]
)


class BufferRepeticion:
    """Buffer de repetición de experiencias: guarda transiciones pasadas y
    permite muestrear mini-lotes al azar en vez de entrenar solo con la
    transición que se acaba de vivir. Rompe la correlación entre pasos
    consecutivos y hace que los datos de entrenamiento se parezcan más a
    un conjunto i.i.d. (ver la explicación de este capítulo)."""

    def __init__(self, capacidad):
        # deque con tamaño máximo: al llenarse, descarta la transición más
        # antigua para dejar sitio a la nueva (comportamiento tipo FIFO)
        self.memoria = deque(maxlen=capacidad)

    def guardar(self, *args):
        self.memoria.append(Transicion(*args))

    def muestrear(self, tamano_lote):
        return random.sample(self.memoria, tamano_lote)

    def __len__(self):
        return len(self.memoria)


def elegir_accion(red_online, estado, epsilon, num_acciones):
    """Política epsilon-greedy sobre los valores Q que estima la red online."""
    if random.random() < epsilon:
        return random.randrange(num_acciones)
    with torch.no_grad():
        estado_tensor = torch.tensor(estado, dtype=torch.float32).unsqueeze(0)
        valores_q = red_online(estado_tensor)
        return int(torch.argmax(valores_q, dim=1).item())


def entrenar_paso(red_online, red_objetivo, buffer, optimizador):
    """Un paso de entrenamiento de DQN: muestrea un mini-lote del buffer y
    ajusta la red online para acercar su predicción al objetivo calculado
    con la red objetivo (target network)."""
    lote = buffer.muestrear(TAMANO_LOTE)
    estados = torch.tensor(np.array([t.estado for t in lote]), dtype=torch.float32)
    acciones = torch.tensor([t.accion for t in lote], dtype=torch.int64).unsqueeze(1)
    recompensas = torch.tensor([t.recompensa for t in lote], dtype=torch.float32)
    siguientes_estados = torch.tensor(
        np.array([t.siguiente_estado for t in lote]), dtype=torch.float32
    )
    terminados = torch.tensor([t.terminado for t in lote], dtype=torch.float32)

    # Q(s,a;θ): valor que la red online asigna a la acción que de verdad se tomó
    valores_q = red_online(estados).gather(1, acciones).squeeze(1)

    # Objetivo: r + γ · max_a' Q(s',a';θ⁻), calculado con la red objetivo.
    # torch.no_grad() implementa el "stop-gradient": el error se propaga
    # solo hacia la red online, nunca hacia la red objetivo.
    with torch.no_grad():
        max_q_siguiente = red_objetivo(siguientes_estados).max(dim=1)[0]
        objetivo = recompensas + GAMMA * max_q_siguiente * (1 - terminados)

    perdida = nn.functional.mse_loss(valores_q, objetivo)

    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()


# --- Preparamos entorno, redes, buffer y optimizador ---
entorno = gym.make(NOMBRE_ENTORNO)
dim_estado = entorno.observation_space.shape[0]
num_acciones = entorno.action_space.n

red_online = RedQ(dim_estado, num_acciones)     # red principal: la que se entrena (θ)
red_objetivo = RedQ(dim_estado, num_acciones)   # red objetivo: se congela entre sincronizaciones (θ⁻)
red_objetivo.load_state_dict(red_online.state_dict())  # al empezar, θ⁻ = θ
red_objetivo.eval()

optimizador = optim.Adam(red_online.parameters(), lr=TASA_APRENDIZAJE)
buffer = BufferRepeticion(TAMANO_BUFFER)

recompensas_por_episodio = []
pasos_totales = 0

# --- Bucle de entrenamiento ---
for episodio in range(NUM_EPISODIOS):
    estado, _ = entorno.reset(seed=SEMILLA + episodio)
    recompensa_episodio = 0.0
    terminado = False
    truncado = False

    # epsilon decae linealmente desde EPSILON_INICIAL hasta EPSILON_FINAL
    progreso = min(episodio / EPISODIOS_DECAIMIENTO, 1.0)
    epsilon = EPSILON_INICIAL + progreso * (EPSILON_FINAL - EPSILON_INICIAL)

    while not (terminado or truncado):
        accion = elegir_accion(red_online, estado, epsilon, num_acciones)
        siguiente_estado, recompensa, terminado, truncado, _ = entorno.step(accion)

        buffer.guardar(estado, accion, recompensa, siguiente_estado, terminado)
        estado = siguiente_estado
        recompensa_episodio += recompensa
        pasos_totales += 1

        # Solo entrenamos cuando el buffer ya acumuló suficiente experiencia
        if len(buffer) >= PASOS_MINIMOS_ANTES_ENTRENAR:
            entrenar_paso(red_online, red_objetivo, buffer, optimizador)

        # Sincronizamos la red objetivo cada cierto número de pasos, no en cada paso
        if pasos_totales % PASOS_ACTUALIZAR_RED_OBJETIVO == 0:
            red_objetivo.load_state_dict(red_online.state_dict())

    recompensas_por_episodio.append(recompensa_episodio)

    if (episodio + 1) % 20 == 0:
        media_reciente = np.mean(recompensas_por_episodio[-20:])
        print(
            f"Episodio {episodio + 1}/{NUM_EPISODIOS} | epsilon={epsilon:.2f} "
            f"| recompensa media (últimos 20)={media_reciente:.1f}"
        )

entorno.close()

# --- Graficamos la recompensa por episodio ---
plt.figure(figsize=(10, 5))
plt.plot(recompensas_por_episodio, alpha=0.4, label="Recompensa por episodio")
# Media móvil de 20 episodios, para ver la tendencia sin el ruido episodio a episodio
media_movil = np.convolve(recompensas_por_episodio, np.ones(20) / 20, mode="valid")
plt.plot(range(19, len(recompensas_por_episodio)), media_movil, label="Media móvil (20 episodios)")
plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.title("DQN en CartPole-v1: recompensa por episodio")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Eseguendolo, all'inizio vedrai che la ricompensa per episodio è bassa e irregolare: il pendolo cade quasi subito perché epsilon è ancora vicino a uno e la rete ha imparato poco o nulla. Man mano che gli episodi avanzano ed epsilon decresce, la media mobile dovrebbe salire in modo abbastanza chiaro, anche se con oscillazioni: questo misto di progresso e rumore episodio per episodio è del tutto normale in DQN, anche quando buffer di replay e rete target funzionano bene.

## Punti chiave

- I metodi tabulari falliscono quando il numero di stati cresce troppo: non stanno in memoria e l'agente raramente ripete esattamente lo stesso stato
- L'approssimazione di funzioni sostituisce la tabella con una funzione parametrizzata, `Q(s,a;θ)`, che può generalizzare a stati nuovi
- Una rete neurale è, in sostanza, una funzione con pesi regolabili che viene addestrata per somigliare sempre di più alla funzione che vuoi approssimare
- Unire in modo ingenuo Q-learning e rete neurale è instabile, perché i dati di RL non sono i.i.d. e il target di addestramento si muove insieme alla rete che lo insegue (bootstrapping)
- Il buffer di replay delle esperienze rompe la correlazione tra passi consecutivi e riutilizza ogni transizione più volte
- La rete target congela una copia dei parametri per un tratto di addestramento, così il target smette di muoversi a ogni passo e l'addestramento si stabilizza
- Double DQN, Dueling DQN e Prioritized Experience Replay sono miglioramenti di DQN che implementerai nel capitolo 11

## Mettilo in pratica

Prendi il codice di questo capitolo e modificalo a piacere:

- **Cambia `PASOS_ACTUALIZAR_RED_OBJETIVO`** (prova con 50, con 500 e con 5.000) e osserva cosa succede alla stabilità della curva di ricompensa: una rete target che si sincronizza troppo spesso somiglia sempre di più al Q-learning ingenuo e instabile che hai visto all'inizio del capitolo
- **Cambia `TAMANO_BUFFER`** (prova con 500 e con 50.000) e verifica se un buffer molto piccolo rende l'addestramento più erratico, perdendo la diversità delle esperienze passate
- **Elimina temporaneamente la rete target** (usa `red_online` al posto di `red_objetivo` dentro `entrenar_paso`) e confronta la curva risultante con la versione originale

Esegui il notebook, cambia gli iperparametri e annota con il tuo team quale combinazione si addestra in modo più stabile!

---

DQN impara i valori delle azioni e ricava da essi la politica, in modo indiretto, scegliendo sempre l'azione con Q più alto. Nel capitolo 9 conoscerai un'altra famiglia di metodi, i gradienti di politica e i metodi actor-critic, che imparano la politica in modo diretto, senza passare per una funzione di valore intermedia.
