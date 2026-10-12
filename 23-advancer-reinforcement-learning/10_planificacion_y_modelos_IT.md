# Capitolo 10. Pianificazione e modelli: quando l'agente impara come funziona il mondo

Immagina di dover andare dal tuo ufficio a una riunione in un'altra città. Puoi metterti al volante senza pensarci troppo e correggere la rotta ogni volta che sbagli strada, oppure puoi costruirti una mappa mentale della zona, immaginare due o tre percorsi possibili e scegliere quello che sembra più breve prima di partire. La seconda opzione è più lenta all'inizio (bisogna costruire quella mappa), ma una volta che ce l'hai ti permette di pianificare invece di improvvisare ogni svolta.

Finora, in questo manuale, il tuo agente è stato puramente reattivo. Nel capitolo precedente hai visto come i metodi di gradiente di politica e actor-critic regolano una politica direttamente a partire dall'esperienza, senza costruire mai una rappresentazione esplicita di come funziona l'ambiente. Questo è l'Apprendimento per Rinforzo (RL, dall'inglese *Reinforcement Learning*) senza modello (model-free): funziona molto bene, ma ogni miglioramento richiede nuova esperienza reale, che spesso è costosa, lenta o rischiosa da ottenere.

In questo capitolo farai un passo diverso. Vedrai cosa succede quando l'agente, oltre ad agire, impara un modello interno dell'ambiente: una sorta di simulatore proprio che gli permette di "immaginare" cosa accadrebbe se compisse questa o quella azione, senza doverla provare davvero. Questa capacità di immaginare i risultati prima di eseguirli è alla base di sistemi come AlphaZero, che conoscerai nel capitolo 12. Qui costruirai le fondamenta: cos'è un modello, come si combina con l'apprendimento diretto nella famiglia Dyna, e come la ricerca ad albero di Monte Carlo esplora futuri possibili senza doverli percorrere tutti.

## RL senza modello contro RL basato su modelli

Tutto ciò che hai visto fino a questo capitolo nove —bandit, Q-learning, gradienti di politica, actor-critic— appartiene alla famiglia dell'RL senza modello: l'agente impara direttamente una politica o una funzione di valore a partire dall'esperienza, senza rappresentare in modo esplicito come cambia l'ambiente da un passo all'altro.

L'RL basato su modelli aggiunge un elemento: un modello delle dinamiche dell'ambiente, composto da una funzione di transizione (quale stato viene dopo) e da una funzione di ricompensa (quale ricompensa si ottiene). Questo modello può essere già disponibile in partenza oppure, cosa più frequente, deve essere appreso dall'esperienza. È utile sapere che il termine "basato su modelli" è usato in modo un po' ambiguo in letteratura: alcune fonti lo riservano al solo caso in cui il modello viene appreso, altre lo applicano anche quando il modello è fornito direttamente. Persino il confine tra "agente" e "ambiente" può sfumare: un agente che ha accesso a un simulatore di Atari potrebbe trattare quel simulatore come il proprio modello interno.

Pensa a un modello come a un'approssimazione di un Processo Decisionale di Markov (MDP): non deve essere una replica esatta dell'ambiente, basta che sia abbastanza fedele da orientare le decisioni. Il testo di riferimento segnala tre tipi di modello che conviene distinguere, perché ciascuno fallisce in modo diverso:

- **Modello di valore atteso**: prevede lo stato successivo e la ricompensa attesi, in media. Il problema è che quella media potrebbe non corrispondere a nessuno stato reale. Immagina un robot in una griglia in cui muoversi "verso l'alto" lo sposta in realtà in diagonale, a sinistra o a destra, mai in linea retta. Un modello di valore atteso potrebbe prevedere che il robot finisca proprio in mezzo, in una casella che non esiste (una buca inventata nel muro)
- **Modello stocastico o generativo**: invece di prevedere una media, genera campioni di stati e ricompense validi, aggiungendo ogni volta un nuovo termine di rumore. Non ti restituisce mai uno stato impossibile, ma in cambio introduce più variabilità; se il modello è economico da consultare, puoi campionare molte volte per mediare quella variabilità
- **Modello di dinamica completa**: apprende l'intera distribuzione di probabilità sugli stati successivi, non solo un punto atteso o un singolo campione. Negli spazi continui questo richiede di lavorare con integrali anziché con somme, cosa che di solito si approssima con metodi come Markov Chain Monte Carlo (MCMC)

Questi modelli possono essere rappresentati in forme molto diverse: come una tabella (se il problema è piccolo), come una funzione lineare sulle caratteristiche dello stato, come una rete neurale profonda (per generalizzare in spazi grandi o continui) o anche in forma non parametrica, usando un semplice buffer di replay delle esperienze: "consultare il modello" diventa, in quel caso, estrarre a caso una transizione reale che hai già osservato.

## Dyna-Q: imparare agendo e imparare immaginando

La famiglia di algoritmi Dyna, proposta da Richard Sutton, combina due fonti di apprendimento che finora trattavi separatamente: l'esperienza reale e l'esperienza simulata da un modello appreso.

Dyna-Q, la sua versione tabulare, funziona in due fasi all'interno di ogni passo:

1. **Apprendimento diretto**: l'agente agisce nell'ambiente reale, osserva lo stato successivo e la ricompensa, applica un normale aggiornamento di Q-learning e, di passaggio, salva quella transizione nel suo modello interno
2. **Pianificazione simulata**: dopo ogni passo reale, l'agente ripete *n* volte la seguente operazione: sceglie a caso una combinazione di stato e azione già osservata in precedenza, chiede al modello cosa accadrebbe e applica un altro aggiornamento di Q-learning su quella transizione immaginata

Quel numero *n* si chiama passi di pianificazione. Con *n* uguale a zero, Dyna-Q si riduce esattamente al Q-learning puro: non c'è nulla da immaginare. Più passi di pianificazione aggiungi, più l'agente "ripassa" la propria esperienza passata prima di tornare ad agire, e meno episodi reali gli servono per convergere. In una griglia di riferimento, aumentare i passi di pianificazione da zero a cinque, e da cinque a 50, riduce in modo molto marcato il numero di passi che l'agente impiega per raggiungere l'obiettivo in ogni episodio, in cambio di più calcolo per ogni passo reale. Questo compromesso di solito vale la pena proprio quando raccogliere esperienza reale è lento o costoso.

C'è una sfumatura importante quando l'ambiente cambia. Se un percorso che prima era libero si blocca, Dyna-Q aggiorna il suo modello non appena urta il nuovo ostacolo e subito dopo rifà la pianificazione. Ma se compare una scorciatoia nuova che l'agente non ha mai provato, Dyna-Q può impiegare parecchio a scoprirla, perché nulla gli indica che quella scorciatoia esista finché l'esplorazione non ce lo porta per caso. Dyna-Q+ risolve in parte il problema aggiungendo un bonus di esplorazione che cresce più a lungo una combinazione di stato e azione non viene visitata: l'agente diventa ottimista riguardo alle zone che da tempo non controlla, supponendo che possano essere cambiate, e così trova le scorciatoie nuove più in fretta, anche se a volte paga il prezzo di compiere qualche azione subottimale in un ambiente che in realtà non è cambiato.

Va ricordato anche che un modello appreso può essere impreciso, per mancanza di esperienza o per i limiti stessi dell'approssimatore, e che pianificare su un modello così difettoso può portare l'agente a decisioni assurde come quella del robot che "allucina" una buca nel muro. Combinare la pianificazione basata su modelli con aggiornamenti senza modello —ricorrendo a questi ultimi quando il modello risulta poco affidabile— è un modo comune di ottenere il meglio dei due mondi: l'orientamento strutturato di un modello e la robustezza di imparare dalla realtà.

## MCTS: esplorare un albero di futuri possibili

La ricerca ad albero di Monte Carlo (Monte Carlo Tree Search, MCTS) porta l'idea della pianificazione un passo oltre: invece di simulare transizioni singole come fa Dyna-Q, costruisce e mantiene un albero di futuri possibili a partire dallo stato attuale, espandendosi solo nella parte che davvero esplora.

Ogni ricerca con MCTS si organizza in quattro fasi che si ripetono quante volte il budget di calcolo consente:

1. **Selezione**: partendo dalla radice dell'albero, scende attraverso i nodi già esistenti seguendo una politica dell'albero, fino a raggiungere un nodo che non è ancora stato espanso del tutto
2. **Espansione**: aggiunge all'albero un nuovo nodo figlio, corrispondente a un'azione che da lì non era ancora stata provata
3. **Simulazione (rollout)**: a partire da quel nuovo nodo, gioca una partita o una traiettoria completa seguendo una politica rapida, spesso casuale, fino a raggiungere uno stato terminale
4. **Retropropagazione**: propaga il risultato di quella simulazione verso l'alto, lungo tutto il percorso compiuto, aggiornando il valore e il contatore di visite di ogni nodo

Il valore di un nodo è, in sostanza, la media dei risultati di tutte le simulazioni che vi sono passate. E qui emerge una domanda che conosci già dal capitolo tre: nella fase di selezione, quale ramo dell'albero conviene esplorare? Upper Confidence Bound for Trees (UCT) risponde con la stessa logica che Upper Confidence Bound (UCB) applicava ai bandit a più braccia: aggiunge al valore stimato di ciascun nodo un bonus di esplorazione che cresce quando quel nodo è stato visitato poco, e si riduce man mano che viene visitato di più. Un nodo con poche visite riceve il beneficio del dubbio; un nodo molto visitato ha già una stima affidabile e non ha bisogno di tanto bonus extra. Come in UCB, una costante controlla quanto peso dai all'esplorare rispetto allo sfruttare.

MCTS con UCT è il motore di ricerca dietro AlphaGo, applicato a giochi come il Go o gli scacchi, dove il numero di configurazioni possibili della scacchiera è astronomico e non esiste modo di percorrere l'intero albero. È stato usato anche per pianificare sequenze di azioni nel controllo robotico. Due tecniche correlate, il Controllo Predittivo basato su Modello (MPC) e MuZero, compaiono nel programma ufficiale come approcci affini, ma restano fuori dalla portata di questo capitolo.

## Esempio in Python: Dyna-Q contro Q-learning puro

Il codice seguente costruisce una piccola griglia fatta a mano, con due muri che obbligano l'agente a girarvi intorno per raggiungere l'obiettivo. Confronta il Q-learning puro (zero passi di pianificazione) con due versioni di Dyna-Q (cinque e 50 passi di pianificazione) e disegna quanti passi ciascuna impiega per completare ogni episodio. Puoi copiarlo ed eseguirlo direttamente in Google Colab: ha bisogno solo di `numpy` per i calcoli e di `matplotlib` per il grafico, entrambi preinstallati.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# --- Definimos un mundo-cuadrícula pequeño, hecho a mano ---
ALTO, ANCHO = 6, 9                 # 6 filas, 9 columnas
INICIO = (5, 0)                    # esquina inferior izquierda
META = (0, 8)                      # esquina superior derecha

# Dos segmentos de muro que obligan a rodear en vez de ir en línea recta
PAREDES = {(1, 3), (2, 3), (3, 3), (0, 6), (1, 6), (2, 6)}

# Cuatro acciones posibles: arriba, abajo, izquierda, derecha
ACCIONES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACCIONES = len(ACCIONES)


def paso_entorno(estado, accion_idx):
    """Aplica una acción de forma determinista: si el movimiento choca contra
    un muro o el borde del mundo, el agente se queda donde estaba."""
    if estado == META:
        return estado, 0.0, True
    d_fila, d_col = ACCIONES[accion_idx]
    fila, col = estado
    nueva_fila = min(max(fila + d_fila, 0), ALTO - 1)
    nueva_col = min(max(col + d_col, 0), ANCHO - 1)
    nuevo_estado = (nueva_fila, nueva_col)
    if nuevo_estado in PAREDES:
        nuevo_estado = estado
    terminado = nuevo_estado == META
    recompensa = 0.0 if terminado else -1.0   # -1 por paso incentiva el camino corto
    return nuevo_estado, recompensa, terminado


def indice(estado):
    """Convierte una casilla (fila, columna) en un índice único de fila en la tabla Q."""
    return estado[0] * ANCHO + estado[1]


def elegir_accion(Q, estado, epsilon):
    """Política epsilon-greedy, la misma que usaste en el capítulo tres."""
    if np.random.random() < epsilon:
        return np.random.randint(N_ACCIONES)
    fila_q = Q[indice(estado)]
    mejores = np.flatnonzero(fila_q == np.max(fila_q))
    return np.random.choice(mejores)


def ejecutar_dyna_q(n_planning, n_episodios, alpha=0.1, gamma=0.95, epsilon=0.1, max_pasos=1000):
    """Entrena un agente Dyna-Q tabular. Con n_planning=0 es Q-learning puro:
    solo queda la actualización con experiencia real, sin repetir nada."""
    Q = np.zeros((ALTO * ANCHO, N_ACCIONES))
    modelo = {}                      # modelo aprendido: (estado, accion) -> (estado_siguiente, recompensa)
    pasos_por_episodio = np.zeros(n_episodios)

    for ep in range(n_episodios):
        estado = INICIO
        pasos = 0
        terminado = False

        while not terminado and pasos < max_pasos:
            # --- Fase 1: aprendizaje directo, con experiencia real ---
            accion = elegir_accion(Q, estado, epsilon)
            sig_estado, recompensa, terminado = paso_entorno(estado, accion)

            i_estado, i_sig = indice(estado), indice(sig_estado)
            objetivo = recompensa + gamma * np.max(Q[i_sig])
            Q[i_estado, accion] += alpha * (objetivo - Q[i_estado, accion])

            modelo[(estado, accion)] = (sig_estado, recompensa)

            # --- Fase 2: planificación simulada, con experiencia imaginada ---
            transiciones_vistas = list(modelo.items())
            for _ in range(n_planning):
                (s_sim, a_sim), (s_sig_sim, r_sim) = transiciones_vistas[
                    np.random.randint(len(transiciones_vistas))
                ]
                i_sim, i_sig_sim = indice(s_sim), indice(s_sig_sim)
                objetivo_sim = r_sim + gamma * np.max(Q[i_sig_sim])
                Q[i_sim, a_sim] += alpha * (objetivo_sim - Q[i_sim, a_sim])

            estado = sig_estado
            pasos += 1

        pasos_por_episodio[ep] = pasos

    return pasos_por_episodio


# --- Comparamos tres configuraciones: 0, 5 y 50 pasos de planificación ---
n_episodios = 50
n_repeticiones = 10       # promediamos varias corridas para suavizar el azar
configuraciones = [0, 5, 50]
resultados = {}

for n_planning in configuraciones:
    acumulado = np.zeros(n_episodios)
    for _ in range(n_repeticiones):
        acumulado += ejecutar_dyna_q(n_planning, n_episodios)
    resultados[n_planning] = acumulado / n_repeticiones

# --- Graficamos la comparación ---
plt.figure(figsize=(10, 5))
for n_planning in configuraciones:
    etiqueta = "Q-learning puro (0 pasos)" if n_planning == 0 else f"Dyna-Q ({n_planning} pasos de planificación)"
    plt.plot(resultados[n_planning], label=etiqueta)
plt.xlabel("Episodio")
plt.ylabel("Pasos hasta llegar a la meta")
plt.title("Dyna-Q vs. Q-learning puro en un mundo-cuadrícula con muros")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Eseguendolo vedrai tre curve discendenti, ma a ritmi molto diversi. Quella con zero passi di pianificazione (Q-learning puro) migliora lentamente, perché impara solo da ciò che vive in tempo reale. Le curve con cinque e, soprattutto, 50 passi di pianificazione scendono molto più in fretta: ogni passo reale viene sfruttato molte più volte, grazie agli aggiornamenti simulati sul modello che l'agente stesso ha costruito man mano.

## Punti chiave

- L'RL basato su modelli aggiunge una funzione di transizione e una funzione di ricompensa che l'agente usa per pianificare, invece di reagire solo all'esperienza diretta
- Esistono tre tipi di modello con modalità di fallimento diverse: il modello di valore atteso può prevedere stati che non esistono, il modello stocastico genera sempre stati validi in cambio di più varianza, e il modello di dinamica completa cattura l'intera distribuzione, a costo computazionale maggiore
- Dyna-Q combina, a ogni passo, un aggiornamento con esperienza reale e vari aggiornamenti con esperienza simulata dal modello appreso; con zero passi di pianificazione, si riduce al Q-learning puro
- Dyna-Q+ aggiunge un bonus di esplorazione che cresce col tempo dall'ultima visita, il che aiuta a scoprire scorciatoie nuove che Dyna-Q impiegherebbe molto a trovare
- MCTS esplora un albero di futuri possibili in quattro fasi (selezione, espansione, simulazione e retropropagazione), e UCT bilancia esplorazione e sfruttamento con la stessa idea di UCB nei bandit del capitolo tre
- Un modello appreso non è mai perfetto: combinarlo con aggiornamenti senza modello garantisce robustezza quando il modello sbaglia

## Mettilo in pratica

Prendi il codice di questo capitolo e sperimenta tu stesso:

- **Cambia il numero di passi di pianificazione** (prova con 1, 10 e 100) e osserva quanto si accelera, o non si accelera, la velocità di apprendimento rispetto al costo extra di calcolo per passo
- **Modifica il disegno dei muri** in `PAREDES` per creare un percorso più lungo o più stretto, e verifica se il vantaggio di Dyna-Q sul Q-learning puro si mantiene
- **Aggiungi un bonus di esplorazione in stile Dyna-Q+**: salva a quale passo è stata visitata per l'ultima volta ogni combinazione di stato e azione, e aggiungi un piccolo extra alla ricompensa simulata quando è passato molto tempo senza visitarla

**Esegui il notebook, cambia i parametri e condividi con il tuo team da quanti passi di pianificazione in poi il calcolo extra smette di valere la pena!**

---

Nel capitolo 11 tornerai all'RL profondo senza modello, ma con le tecniche più avanzate del settore: Rainbow DQN e l'RL distribuzionale.
