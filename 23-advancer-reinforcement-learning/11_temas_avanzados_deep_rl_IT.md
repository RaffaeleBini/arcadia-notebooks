# Capitolo 11. Rainbow e l'apprendimento distribuzionale: quando più miglioramenti si sommano

Nel capitolo 8 hai conosciuto Deep Q-Network (DQN): un agente che combina il Q-learning con una rete neurale profonda, una rete target e la repetition buffer per giocare a videogiochi direttamente a partire dai pixel. È stata una pietra miliare, ma non un punto finale. Dalla sua pubblicazione, la comunità di ricerca in Reinforcement Learning (RL) ha proposto migliorie concrete a pezzi diversi di quello stesso algoritmo: come sceglie le azioni, cosa ricorda, come esplora, cosa predice. Ogni miglioria, presa separatamente, porta già qualcosa. La domanda a cui risponde questo capitolo è cosa succede quando le metti tutte insieme.

Nel capitolo precedente hai visto come un agente possa pianificare con un modello dell'ambiente invece di limitarsi a reagire. Qui torni sul terreno libero da modello, ma con uno sguardo diverso: invece di un algoritmo nuovo, vedrai un algoritmo che è la somma accurata di vari algoritmi già conosciuti. Si chiama Rainbow, ed è l'esempio di riferimento di questo capitolo, insieme a un'idea che attraversa diversi dei suoi pezzi: imparare la distribuzione completa dei risultati possibili, non solo la loro media.

## Rainbow: la somma di sei miglioramenti indipendenti

Il nome non è casuale. Proprio come un arcobaleno appare quando si combinano vari colori che, separatamente, esistono già, Rainbow DQN appare quando combini sei miglioramenti che, separatamente, erano già stati provati con successo sul DQN originale. Nessun componente è esclusivo di Rainbow; ciò che Rainbow apporta è la prova che integrarli tutti nello stesso agente funziona meglio di ciascuno di essi preso separatamente.

Ripassali uno per uno:

**Doppio Q-learning.** Lo hai già visto nel capitolo 7: DQN tende a sovrastimare il valore di alcune azioni perché usa la stessa rete per scegliere l'azione migliore e per valutarla. Il doppio Q-learning separa questi due compiti: una rete (la rete online) scegli quale azione sembra migliore, e un'altra (la rete target) calcola quanto vale quella scelta. Qui, dentro Rainbow, quella stessa idea si integra come un pezzo in più dell'insieme.

**Repetition buffer prioritizzata (PER, dall'inglese *Prioritized Experience Replay*).** Il DQN di base salva transizioni passate in un buffer e campiona lotti a caso per addestrare. PER cambia "a caso" in "con criterio": le transizioni in cui l'agente si è sbagliato di più (errore di differenza temporale maggiore) vengono ripetute più spesso, perché sono quelle che apportano più informazione nuova. È come ripassare con più intensità gli esercizi che hai sbagliato in un esame, invece di ripassarli tutti allo stesso modo.

**Architettura dueling.** Invece che la rete predica direttamente un valore Q per ogni azione, lo separa in due domande distinte: quanto vale questo stato in generale, indipendentemente da cosa farà in seguito? E, dentro quello stato, quanto è migliore ogni azione concreta rispetto alle altre? Alla prima domanda risponde una testa di valore V(s); alla seconda, una testa di vantaggio A(s,a). Combinandole, l'agente impara quanto vale "stare qui" separatamente da "fare questo in concreto", il che aiuta soprattutto negli stati in cui l'azione scelta quasi non cambia il risultato. Un esempio visivo che usa il materiale fonte: in un videogioco di corse, la testa di valore osserva la scena completa (sto andando bene nella gara?), mentre la testa di vantaggio si concentra su dettagli immediati, come l'auto che hai di fianco.

**Ritorni multi-passo.** Il DQN standard guarda un solo passo avanti prima di aggiornare le sue stime. Rainbow, invece, accumula la ricompensa di diversi passi consecutivi prima di fare bootstrap con la rete target. È un punto intermedio tra guardare solo il passo successivo e aspettare la fine completa dell'episodio (come fa Monte Carlo): vedi un po' più lontano prima di tirare conclusioni, il che di solito accelera l'apprendimento.

**Apprendimento distribuzionale.** Questa è, forse, l'idea concettualmente più diversa delle sei. Invece di imparare un unico numero per azione (il ritorno atteso), l'agente imparra la distribuzione completa dei ritorni possibili: non solo "quanto mi aspetto di guadagnare in media", ma "quale probabilità ha ciascun risultato possibile". L'algoritmo concreto usato da Rainbow si chiama C51: rappresenta quella distribuzione tramite 51 punti fissi (chiamati atomi) distribuiti in un intervallo conosciuto, impara una probabilità per ciascuno, e confronta la distribuzione predetta con la distribuzione target tramite la divergenza di Kullback-Leibler (KL) invece di un semplice errore quadratico. Imparare tutta la distribuzione, e non solo la media, dà all'agente una rappresentazione più ricca del mondo con la stessa quantità di dati, anche se conviene chiarire una cosa importante: quella distribuzione dei ritorni non è una misura dell'incertezza dell'agente, ma della variabilità reale dei risultati possibili.

**Reti rumorose (Noisy Nets).** Invece di esplorare con epsilon-greedy (una probabilità fissa di agire a caso), Rainbow aggiunge rumore apprendibile direttamente ai pesi e ai bias della rete. La rete stessa impara, tramite lo stesso optimizer con cui impara tutto il resto, quanto rumore "lasciare attivo": poco rumore per situazioni che già conosce bene, più rumore per situazioni rare dove ha ancora senso esplorare. L'esplorazione smette di essere un parametro esterno e diventa parte dell'apprendimento.

### Cosa ha rivelato lo studio di ablazione

Quando il team che ha proposto Rainbow ha provato cosa succedeva togliendo ogni componente uno per uno, il risultato non è stato simmetrico. L'apprendimento multi-passo, la repetition prioritizzata e l'apprendimento distribuzionale sono risultati quelli che contribuivano più alla prestazione finale. Il doppio Q-learning e l'architettura dueling, invece, hanno contribuito meno di quanto ci si aspettasse una volta combinati con il resto: una spiegazione proposta è che l'apprendimento distribuzionale già limita di per sé l'intervallo dei ritorni possibili, il che riduce buona parte del problema di sovrastima che il doppio Q-learning era stato progettato per risolvere. Quanto all'efficienza, la cifra che viene solitamente citata è notevole: Rainbow raggiunge, in circa sette milioni di fotogrammi di addestramento, lo stesso punteggio medio normalizzato che il DQN standard necessita circa 200 milioni di fotogrammi per ottenere. È un miglioramento di efficienza dei dati di circa 14 volte, con un tempo di calcolo simile a quello del DQN originale.

### Linee di ricerca attive che vedrai citate nella letteratura

Oltre a Rainbow, troverai altri nomi che circolano frequentemente nell'RL avanzato. Il materiale di riferimento di questo capitolo li menziona solo di passaggio, quindi trattali come una mappa di ciò che esiste, non come contenuto sviluppato qui:

- **RL multiagente**: diversi agenti che imparano contemporaneamente nello stesso ambiente, a volte cooperando e a volte competendo; concetti abituali includono l'addestramento centralizzato con esecuzione decentralizzata e la modellazione del comportamento di altri agenti
- **RL gerarchico**: invece di imparare solo azioni a un passo, l'agente imparra anche azioni "estese nel tempo" (chiamate opzioni), ciascuna con la propria politica interna e la propria condizione di terminazione
- **Meta-apprendimento, apprendimento per trasferimento e apprendimento per curriculum**: modi in cui un agente sfrutta ciò che ha imparato in un compito per imparare più velocemente un altro compito correlato, o di organizzare l'addestramento in un ordine di difficoltà crescente
- **Sicurezza nell'RL**: un'area che guadagna rilevanza man mano che questi agenti si avvicinano ad ambienti reali; l'apprendimento distribuzionale, per esempio, già apporta qualcosa di utile qui informando sulla probabilità di terminazioni premature o indesiderate, non solo sul ritorno atteso

Se qualcuno di questi nomi ti risulta interessante, annotalo come linea di studio propria: questo manuale non li sviluppa in profondità perché nemmeno il materiale di origine lo fa.

## Esempio in Python: aggiungendo un'architettura dueling al tuo DQN

Programmare le sei migliorie di Rainbow in un solo capitolo non sarebbe realistico. Invece, riprenderai il DQN minimo del capitolo 8 e gli aggiungerai una sola miglioria, ben delimitata: l'architettura dueling. Addestrerai due agenti su CartPole (un DQN standard e un Dueling DQN) con esattamente lo stesso ciclo di addestramento, e confronterai le loro curve di ricompensa. L'unica differenza tra i due sarà l'architettura interna della rete.

```python
!pip install gymnasium

import random
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Fijamos semillas para que el experimento sea reproducible
SEMILLA = 0
random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

Transicion = namedtuple(
    "Transicion", ["estado", "accion", "recompensa", "siguiente_estado", "terminado"]
)


class BufferReplay:
    """Memoria de repetición de experiencia (capítulo 8): guarda transiciones
    pasadas y permite muestrear lotes al azar, rompiendo la correlación
    temporal de los datos consecutivos de un episodio."""

    def __init__(self, capacidad=10000):
        self.memoria = deque(maxlen=capacidad)

    def guardar(self, *args):
        self.memoria.append(Transicion(*args))

    def muestrear(self, tamano_lote):
        return random.sample(self.memoria, tamano_lote)

    def __len__(self):
        return len(self.memoria)


class RedQEstandar(nn.Module):
    """DQN "de manual": una sola cabeza que produce directamente
    un valor Q por cada acción posible."""

    def __init__(self, n_entradas, n_acciones, n_ocultas=128):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_entradas, n_ocultas),
            nn.ReLU(),
            nn.Linear(n_ocultas, n_ocultas),
            nn.ReLU(),
            nn.Linear(n_ocultas, n_acciones),
        )

    def forward(self, x):
        return self.red(x)


class RedQDueling(nn.Module):
    """Arquitectura dueling: un tronco compartido alimenta dos cabezas.
    Una estima V(s) -- cuánto vale el estado, un único número --
    y otra estima A(s,a) -- cuánto mejor es cada acción frente a las demás.
    Se combinan restando la media de la ventaja, para que la suma
    quede en una escala estable."""

    def __init__(self, n_entradas, n_acciones, n_ocultas=128):
        super().__init__()
        self.tronco = nn.Sequential(
            nn.Linear(n_entradas, n_ocultas),
            nn.ReLU(),
        )
        self.cabeza_valor = nn.Sequential(
            nn.Linear(n_ocultas, n_ocultas),
            nn.ReLU(),
            nn.Linear(n_ocultas, 1),
        )
        self.cabeza_ventaja = nn.Sequential(
            nn.Linear(n_ocultas, n_ocultas),
            nn.ReLU(),
            nn.Linear(n_ocultas, n_acciones),
        )

    def forward(self, x):
        caracteristicas = self.tronco(x)
        valor = self.cabeza_valor(caracteristicas)
        ventaja = self.cabeza_ventaja(caracteristicas)
        # Q(s,a) = V(s) + (A(s,a) - media_a[A(s,a)])
        # Usamos dim=-1 en vez de dim=1: así funciona tanto si pasas un solo
        # estado (vector 1D, al elegir una acción) como un lote de estados (2D, al entrenar)
        return valor + (ventaja - ventaja.mean(dim=-1, keepdim=True))


def entrenar(clase_red, nombre, n_episodios=300, gamma=0.99, tamano_lote=64,
             tasa_aprendizaje=1e-3, actualizar_objetivo_cada=10):
    """Bucle de entrenamiento DQN estándar: red online, red objetivo,
    buffer de repetición y exploración epsilon-greedy decreciente.
    Cambiar `clase_red` entre RedQEstandar y RedQDueling es la única
    diferencia entre ambos experimentos -- el resto del algoritmo no cambia."""

    entorno = gym.make("CartPole-v1")
    n_entradas = entorno.observation_space.shape[0]
    n_acciones = entorno.action_space.n

    red_online = clase_red(n_entradas, n_acciones)
    red_objetivo = clase_red(n_entradas, n_acciones)
    red_objetivo.load_state_dict(red_online.state_dict())
    red_objetivo.eval()

    optimizador = optim.Adam(red_online.parameters(), lr=tasa_aprendizaje)
    buffer = BufferReplay(capacidad=10000)

    epsilon, epsilon_minimo, decaimiento = 1.0, 0.05, 0.97
    recompensas_por_episodio = []

    for episodio in range(n_episodios):
        estado, _ = entorno.reset(seed=SEMILLA + episodio)
        recompensa_total = 0.0
        terminado = False

        while not terminado:
            if random.random() < epsilon:
                accion = entorno.action_space.sample()  # exploración
            else:
                with torch.no_grad():
                    q_valores = red_online(torch.tensor(estado, dtype=torch.float32))
                    accion = int(torch.argmax(q_valores).item())  # explotación

            siguiente_estado, recompensa, fin, truncado, _ = entorno.step(accion)
            terminado = fin or truncado
            buffer.guardar(estado, accion, recompensa, siguiente_estado, fin)
            estado = siguiente_estado
            recompensa_total += recompensa

            # Solo entrenamos cuando el buffer tiene datos suficientes
            if len(buffer) >= tamano_lote:
                lote = buffer.muestrear(tamano_lote)
                estados = torch.tensor(np.array([t.estado for t in lote]), dtype=torch.float32)
                acciones = torch.tensor([t.accion for t in lote], dtype=torch.int64)
                recompensas = torch.tensor([t.recompensa for t in lote], dtype=torch.float32)
                siguientes = torch.tensor(np.array([t.siguiente_estado for t in lote]), dtype=torch.float32)
                terminados = torch.tensor([t.terminado for t in lote], dtype=torch.float32)

                q_actuales = red_online(estados).gather(1, acciones.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_siguientes = red_objetivo(siguientes).max(dim=1)[0]
                    objetivo = recompensas + gamma * q_siguientes * (1 - terminados)

                perdida = nn.functional.mse_loss(q_actuales, objetivo)
                optimizador.zero_grad()
                perdida.backward()
                optimizador.step()

        recompensas_por_episodio.append(recompensa_total)
        epsilon = max(epsilon_minimo, epsilon * decaimiento)

        if episodio % actualizar_objetivo_cada == 0:
            red_objetivo.load_state_dict(red_online.state_dict())

    entorno.close()
    print(f"{nombre}: recompensa media de los últimos 20 episodios = "
          f"{np.mean(recompensas_por_episodio[-20:]):.1f}")
    return recompensas_por_episodio


# --- Entrenamos ambos agentes con el mismo bucle ---
recompensas_dqn = entrenar(RedQEstandar, "DQN estándar")
recompensas_dueling = entrenar(RedQDueling, "Dueling DQN")


def media_movil(valores, ventana=10):
    return np.convolve(valores, np.ones(ventana) / ventana, mode="valid")


# --- Comparamos las curvas de recompensa ---
plt.figure(figsize=(10, 5))
plt.plot(media_movil(recompensas_dqn), label="DQN estándar")
plt.plot(media_movil(recompensas_dueling), label="Dueling DQN")
plt.xlabel("Episodio")
plt.ylabel("Recompensa (media móvil de 10 episodios)")
plt.title("CartPole-v1: DQN estándar vs. Dueling DQN")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Eseguendolo, osserva due cose: quanto tarda ogni curva ad avvicinarsi alla ricompensa massima di CartPole (500 passi) e quanta variabilità mostra ciascuna tra episodi. In un ambiente così semplice come CartPole, il vantaggio del Dueling DQN è di solito modesto (l'architettura dueling brilla più in ambienti con molti stati dove l'azione scelta quasi non cambia il risultato), ma l'esercizio ti lascia lo scheletro completo per aggiungere, più avanti, le altre cinque migliorie di Rainbow su questa stessa base.

## Punti chiave

- DQN, del capitolo 8, è il punto di partenza; Rainbow è la somma di sei miglioramenti indipendenti che, combinati, superano chiaramente il DQN preso separatamente
- Le sei migliorie sono: doppio Q-learning, repetition buffer prioritizzata, architettura dueling, ritorni multi-passo, apprendimento distribuzionale (C51) e reti rumorose
- Lo studio di ablazione mostra che multi-passo, repetition prioritizzata e apprendimento distribuzionale contribuiscono più del doppio Q-learning e del dueling una volta combinati
- Rainbow raggiunge in circa sette milioni di fotogrammi quello che DQN impiega circa 200 milioni a ottenere: circa 14 volte più efficienza dei dati
- L'apprendimento distribuzionale impara la distribuzione completa dei ritorni possibili, non solo la loro media, e usa la divergenza KL come funzione di perdita
- RL multiagente, RL gerarchico, meta-apprendimento/trasferimento/curriculum e sicurezza nell'RL sono linee di ricerca attive che vedrai citate nella letteratura, anche se questo capitolo non le sviluppa in profondità

## Mettilo in pratica

Scegli uno di questi due compiti e completalo:

- **Aggiungi ritorni multi-passo** al ciclo di `entrenar`: invece di salvare transizioni a un solo passo, accumula la ricompensa degli ultimi n passi (per esempio, n = 3) prima di calcolare l'obiettivo di bootstrap, e confronta la curva risultante con quella del Dueling DQN di questo capitolo
- **Investiga un altro componente di Rainbow** che non hai implementato (repetition prioritizzata, reti rumorose o apprendimento distribuzionale), e annota in un paragrafo cosa cambieresti nel codice di questo capitolo per incorporarlo e perché pensi che aiuterebbe in CartPole

Condividi il tuo risultato con il tuo team e **confronta** quale combinazione di migliorie ha dato la curva di apprendimento più stabile.

---

Nel capitolo 12 cambierai prospettiva: invece di migliorare un singolo agente che imparra da solo, vedrai architetture dove diverse unità di calcolo si addestrano in parallelo, come A3C e IMPALA.
