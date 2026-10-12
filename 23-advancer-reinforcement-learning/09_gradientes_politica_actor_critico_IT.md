# Capitolo 9. Gradienti di politica e metodi Actor-Critic

Nel capitolo precedente hai conosciuto la Deep Q-Network (DQN): un agente che impara, tramite una rete neurale, quanto vale ogni azione in ogni stato, e che poi agisce scegliendo sempre l'azione con il valore più alto. Questa strategia funziona molto bene, ma fa un giro largo: prima impara una funzione di valore completa e solo dopo, in modo indiretto, ricava una politica da essa.

In questo capitolo vedrai un'idea diversa: invece di imparare quanto vale ogni azione, impara direttamente la politica, cioè la regola che decide cosa fare in ogni stato. Pensa a un robot la cui strategia migliore è semplicemente "avanzare sempre". La sua politica ottimale è banale. Ma la funzione di valore che descrive quanto vale ogni combinazione di posizione e ostacoli può essere enormemente complessa. Se il tuo obiettivo reale è agire bene, perché imparare prima quel giro complicato invece di andare dritto alla politica? Questo principio, evitare di risolvere un problema più generale di quello che ti serve per risolvere il problema concreto che ti interessa, è attribuito allo statistico Vladimir Vapnik, ed è la motivazione centrale di questo capitolo.

Imparare direttamente la politica ha, inoltre, due vantaggi pratici che DQN non offre con la stessa naturalezza. Il primo si presenta negli spazi di azione continui o molto grandi, come l'angolo di un motore o la forza di un braccio robotico: DQN deve calcolare il massimo su tutte le azioni possibili, cosa impraticabile quando sono infinite. Un metodo basato sulla politica, invece, emette semplicemente l'azione (o una distribuzione di probabilità sulle azioni) senza bisogno di quel massimo. Il secondo vantaggio è la possibilità di imparare politiche genuinamente casuali, non solo deterministiche, cosa che capirai subito.

## Perché a volte conviene agire a caso

Immagina un labirinto con due corridoi che, visti dall'interno, sembrano esattamente uguali: stesse pareti a sinistra e a destra, stesso aspetto. Un agente che sa agire solo in modo deterministico può restare intrappolato, andando avanti e indietro tra un corridoio e l'altro in un ciclo infinito, perché dal suo punto di vista i due stati sono indistinguibili. Una politica stocastica, che in quei corridoi sceglie a caso tra sinistra e destra, prima o poi esce dal ciclo per pura probabilità e arriva a una zona in cui può distinguere dove si trova.

Lo stesso accade nei giochi con informazione nascosta. Nel sasso, carta, forbice, qualsiasi strategia deterministica è sfruttabile subito: il tuo avversario deve solo individuare lo schema. L'unica strategia ottimale è scegliere ogni opzione con la stessa probabilità. Nel poker succede qualcosa di simile: se il tuo modo di agire con una mano vincente è sempre lo stesso, un avversario attento lo impara e lo sfrutta; il bluff funziona solo se è probabilistico.

Una politica, quindi, non è più una regola fissa "stato → azione" (una funzione deterministica `a = μ(s)`), ma una distribuzione di probabilità sulle azioni dato lo stato: `π(a|s)`. Imparare questa distribuzione, invece di una regola fissa, è ciò che rende possibile rappresentare comportamenti come quelli del labirinto o del poker.

## Il trucco della funzione score

Qui emerge un problema tecnico. Vuoi regolare i parametri della tua politica, θ, per massimizzare la ricompensa attesa. Ma quella ricompensa attesa dipende da un processo di campionamento casuale: l'azione che compi viene estratta da una distribuzione di probabilità, e non puoi "derivare" direttamente attraverso un sorteggio.

La soluzione è nota come trucco della funzione score (o trucco REINFORCE, o trucco della log-derivata). Senza entrare nello sviluppo matematico completo, l'idea intuitiva è questa: invece di cercare di derivare il risultato di un campionamento, trasformi il problema in qualcosa che sai derivare, il logaritmo della probabilità dell'azione che hai scelto. Il risultato, dopo lo sviluppo, è sorprendentemente semplice da applicare: per regolare la politica, calcoli il gradiente del logaritmo della probabilità di ogni azione che hai compiuto, e lo moltiplichi per quanto è stata buona quell'azione (la sua ricompensa o il suo ritorno).

Detto in parole semplici: aumenta la probabilità delle azioni che hanno portato a una ricompensa alta, e diminuisci la probabilità di quelle che hanno portato a una ricompensa bassa o negativa. Quanto migliore è stato il risultato, tanto più forte è la spinta verso l'alto.

Un dettaglio importante di questa derivazione: quando scomponi la probabilità di una traiettoria completa, compaiono due tipi di termini, quelli che dipendono dalla dinamica dell'ambiente (quale stato segue quale) e quelli che dipendono solo dalla tua politica. Quando calcoli il gradiente rispetto ai parametri della politica, i termini della dinamica dell'ambiente scompaiono, perché non dipendono da θ. Questo significa qualcosa di molto pratico: i metodi di gradiente di politica non hanno bisogno di conoscere né di modellare come funziona l'ambiente; basta che possano agire in esso e osservare cosa succede.

## REINFORCE: l'algoritmo più semplice

Con questa idea hai già l'algoritmo REINFORCE, la versione più elementare del gradiente di politica:

1. Inizializza i parametri θ della politica, normalmente una rete neurale
2. Ripeti finché le prestazioni ti convincono:
   - Esegui un episodio completo seguendo la politica attuale e salva stati, azioni e ricompense
   - Per ogni passo dell'episodio, calcola il ritorno (la somma delle ricompense future, di solito scontate)
   - Calcola il gradiente: per ogni azione compiuta, moltiplica il gradiente del logaritmo della sua probabilità per il ritorno corrispondente
   - Aggiorna i parametri nella direzione di quel gradiente: `θ ← θ + α · gradiente`

È un algoritmo elegante e facile da programmare. Ma ha un problema serio: l'alta varianza. Ogni episodio dipende da decisioni casuali (quali azioni ha estratto il campionamento) e dalla casualità propria dell'ambiente, quindi il ritorno di un episodio può essere molto diverso da quello di un altro episodio quasi identico. Questo fa sì che gli aggiornamenti saltino da una parte all'altra e che l'addestramento sia lento e instabile.

## Ridurre la varianza: baseline e vantaggio

Qui entra in gioco un'idea molto elegante. Puoi sottrarre dal ritorno una baseline, `b(s)`, un valore che dipende solo dallo stato (non dall'azione scelta), senza introdurre alcun bias nell'apprendimento. Il motivo è che, sommando su tutte le azioni possibili, l'effetto di una baseline costante si annulla matematicamente: non cambia la direzione del gradiente, ma ne riduce solo la varianza.

La baseline più comune è la stessa funzione di valore di stato, `V(s)`: quanta ricompensa ti aspetti di ottenere da quello stato, seguendo la tua politica attuale. Sottraendola dal valore dell'azione `Q(s,a)`, ottieni la funzione di vantaggio (advantage):

`A(s,a) = Q(s,a) − V(s)`

Il vantaggio risponde a una domanda molto precisa: quanto è migliore (o peggiore) questa azione specifica rispetto a ciò che faresti in media in questo stato? Un vantaggio positivo spinge verso l'alto la probabilità di quell'azione; un vantaggio negativo la spinge verso il basso. È un segnale molto più pulito del ritorno grezzo, perché non trascina più la parte della ricompensa che dipende solo dallo stato in cui ti trovavi, ma soltanto la parte che dipende dalla decisione che hai preso.

In pratica, non sempre servono due reti separate per Q e V. Puoi stimare il vantaggio con l'errore di differenza temporale (TD, dall'inglese "temporal difference"): `δ = r + γV(s') − V(s)`. Questo errore confronta ciò che ti aspettavi di ottenere con ciò che hai realmente ottenuto dopo aver compiuto un passo, ed è, in pratica, un campione del vantaggio.

## Architettura Actor-Critic

Quest'ultima idea porta direttamente ai metodi Actor-Critic. Immagina uno chef (l'attore) che decide quale piatto cucinare, e un esperto di gastronomia (il critico) che assaggia ogni piatto e gli dice quanto è stato migliore o peggiore del previsto. Lo chef usa quella valutazione per regolare le sue decisioni future.

In termini tecnici, l'attore è la politica `π_θ`, che decide quale azione compiere, e il critico è una funzione di valore `V_ω`, che valuta quanto bene sta andando la prestazione dell'attore. Il critico si addestra con tecniche di differenza temporale, come la funzione di valore di DQN. L'attore si addestra con il gradiente di politica, usando l'errore TD del critico come segnale di vantaggio. Lo schema di base è questo:

```
Inicializa la red del actor (política) con parámetros θ
Inicializa la red del crítico (valor) con parámetros ω

para cada episodio:
    inicializa el estado s
    para cada paso del episodio:
        muestrea una acción a ~ π_θ(a|s)
        ejecuta a, observa la recompensa r y el siguiente estado s'
        δ = r + γ · V_ω(s') − V_ω(s)          # error TD, usado como ventaja
        ω ← ω + α_crítico · δ · ∇_ω V_ω(s)     # actualiza el crítico
        θ ← θ + α_actor · δ · ∇_θ log π_θ(a|s) # actualiza el actor
        s ← s'
```

Questo schema, in cui il critico impara a ogni passo (non serve aspettare la fine dell'episodio) e l'attore usa quel segnale per migliorare, si chiama Advantage Actor-Critic (A2C), e la sua variante che esegue più agenti in parallelo, ciascuno con una propria copia dell'ambiente, si chiama Asynchronous Advantage Actor-Critic (A3C). Incontrerai di nuovo A3C nel capitolo 12.

Esistono evoluzioni successive di queste idee che vedrai citate più avanti in questo manuale, senza entrare qui nel dettaglio tecnico: Trust Region Policy Optimization (TRPO) e Proximal Policy Optimization (PPO), che limitano quanto può cambiare la politica a ogni aggiornamento per evitare passi troppo bruschi, e Deep Deterministic Policy Gradient (DDPG), pensato per spazi di azione continui.

## Esempio in Python: REINFORCE su CartPole

Il codice seguente implementa REINFORCE da zero su CartPole, l'ambiente classico in cui un carrello deve mantenere in equilibrio un'asta muovendosi a sinistra o a destra. Puoi copiarlo ed eseguirlo direttamente in Google Colab: usa PyTorch (già preinstallato) e gymnasium (si installa nella prima cella). Non richiede GPU.

```python
# Celda 1: instala gymnasium (PyTorch ya viene preinstalado en Colab)
!pip install gymnasium
```

```python
# Celda 2: imports
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Fijamos semillas para que el experimento sea reproducible
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)


class RedPolitica(nn.Module):
    """Red neuronal pequeña que representa la política π_θ(a|s).
    Recibe el estado de CartPole (cuatro números: posición y velocidad del
    carro, ángulo y velocidad angular del palo) y devuelve una distribución
    de probabilidad sobre las dos acciones posibles (empujar el carro a la
    izquierda o a la derecha)."""

    def __init__(self, n_entradas, n_acciones, n_ocultas=128):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_entradas, n_ocultas),
            nn.ReLU(),
            nn.Linear(n_ocultas, n_acciones),
        )

    def forward(self, estado):
        logits = self.red(estado)
        return torch.softmax(logits, dim=-1)


def calcular_retornos(recompensas, gamma=0.99):
    """Calcula el retorno 'reward-to-go' en cada paso del episodio:
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    Se recorre la lista de atrás hacia adelante para reutilizar el cálculo
    del paso siguiente en cada paso anterior."""
    retornos = np.zeros(len(recompensas))
    G = 0.0
    for t in reversed(range(len(recompensas))):
        G = recompensas[t] + gamma * G
        retornos[t] = G
    return retornos


def ejecutar_episodio(env, politica):
    """Recolecta un episodio completo siguiendo la política actual.
    Devuelve la log-probabilidad de cada acción tomada (necesaria para el
    gradiente de política) y la recompensa obtenida en cada paso."""
    log_probs = []
    recompensas = []
    estado, _ = env.reset()
    terminado = False
    truncado = False

    while not (terminado or truncado):
        estado_tensor = torch.FloatTensor(estado).unsqueeze(0)
        probs = politica(estado_tensor)
        distribucion = torch.distributions.Categorical(probs)
        accion = distribucion.sample()

        log_probs.append(distribucion.log_prob(accion))
        estado, recompensa, terminado, truncado, _ = env.step(accion.item())
        recompensas.append(recompensa)

    return log_probs, recompensas


# --- Configuración del entrenamiento ---
env = gym.make("CartPole-v1")
n_entradas = env.observation_space.shape[0]
n_acciones = env.action_space.n

politica = RedPolitica(n_entradas, n_acciones)
optimizador = optim.Adam(politica.parameters(), lr=1e-2)

N_EPISODIOS = 500
GAMMA = 0.99

historial_recompensas = []

for episodio in range(N_EPISODIOS):
    log_probs, recompensas = ejecutar_episodio(env, politica)
    retornos = calcular_retornos(recompensas, GAMMA)
    retornos_tensor = torch.FloatTensor(retornos)

    # Pérdida del gradiente de política: -sum(log_prob(a_t|s_t) * G_t)
    # El signo negativo convierte el ASCENSO de gradiente que queremos
    # (maximizar la recompensa esperada) en el DESCENSO de gradiente que
    # optimiza PyTorch por defecto.
    perdida = torch.stack(
        [-log_prob * G for log_prob, G in zip(log_probs, retornos_tensor)]
    ).sum()

    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()

    historial_recompensas.append(sum(recompensas))

    if (episodio + 1) % 50 == 0:
        media_reciente = np.mean(historial_recompensas[-50:])
        print(f"Episodio {episodio + 1}: recompensa media (últimos 50) = {media_reciente:.1f}")

env.close()

# --- Gráfico de la recompensa por episodio ---
plt.figure(figsize=(10, 5))
plt.plot(historial_recompensas, alpha=0.4, label="Recompensa por episodio")

ventana = 20  # media móvil para ver la tendencia sin el ruido episodio a episodio
media_movil = np.convolve(historial_recompensas, np.ones(ventana) / ventana, mode="valid")
plt.plot(range(ventana - 1, len(historial_recompensas)), media_movil, label=f"Media móvil ({ventana} episodios)")

plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.title("REINFORCE en CartPole-v1")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Eseguendolo vedrai qualcosa di tipico di REINFORCE: la curva della ricompensa sale, ma con molti alti e bassi tra episodi vicini, proprio il problema della varianza che già conosci. In CartPole-v1 un episodio può durare fino a 500 passi, quindi una ricompensa media vicina a quel numero indica che l'asta resta in equilibrio quasi per tutto il tempo consentito.

## Punti chiave

- I metodi di gradiente di politica imparano direttamente la politica, senza passare per una funzione di valore completa, il che si adatta meglio a spazi di azione continui o molto grandi
- Una politica stocastica, `π(a|s)`, è necessaria in stati ambigui e nei giochi con informazione nascosta, dove una regola deterministica sarebbe sfruttabile
- Il trucco della funzione score trasforma un problema di derivazione attraverso un campionamento casuale in qualcosa di calcolabile: il gradiente del logaritmo della probabilità dell'azione, moltiplicato per quanto è stata buona
- REINFORCE applica direttamente questa idea, ma soffre di alta varianza perché dipende da ritorni completi di episodi rumorosi
- Sottrarre una baseline (di solito `V(s)`) riduce la varianza senza introdurre bias, e dà origine alla funzione di vantaggio `A(s,a) = Q(s,a) − V(s)`
- L'architettura Actor-Critic combina un attore (la politica) con un critico (una funzione di valore) che valuta ogni passo e guida l'aggiornamento dell'attore tramite l'errore TD
- A2C, A3C, PPO, TRPO e DDPG sono evoluzioni di queste idee che incontrerai nei prossimi capitoli, senza bisogno di padroneggiarne ancora il dettaglio tecnico

## Mettilo in pratica

Prendi il codice di questo capitolo e modificalo a piacere:

- **Aggiungi una baseline semplice**: dentro il ciclo di addestramento, calcola la media dei `retornos` di ogni episodio e sottraila a ciascun ritorno prima di calcolare la perdita
- **Confronta la stabilità**: esegui l'addestramento con e senza quella baseline, usando lo stesso seme, e osserva le due curve di ricompensa per episodio; fai attenzione soprattutto a quanto saltano i valori tra episodi consecutivi
- **Prova valori diversi di `GAMMA`** (per esempio, 0,9 contro 0,99) e annota come cambia la velocità di apprendimento

Esegui il notebook con e senza baseline, e condividi con il tuo team quale delle due versioni è arrivata prima a mantenere l'asta in equilibrio per tutti i 500 passi!

---

Finora l'agente ha imparato solo dall'esperienza diretta: ha agito, osservato cosa è successo e aggiustato il proprio comportamento, senza cercare di prevedere come funziona l'ambiente. Nel capitolo 10 vedrai cosa succede quando l'agente costruisce o usa un modello dell'ambiente per pianificare in anticipo, con tecniche come Dyna-Q e la ricerca ad albero di Monte Carlo (Monte Carlo Tree Search).
