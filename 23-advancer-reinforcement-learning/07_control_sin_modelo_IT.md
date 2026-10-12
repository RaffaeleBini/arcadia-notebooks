# Capitolo 7. Controllo senza modello: SARSA e Q-learning

Nel capitolo precedente hai imparato a valutare una politica fissa: dato un agente che si comporta sempre allo stesso modo, hai calcolato quanta ricompensa si aspetta di ottenere da ciascuno stato, usando il Monte Carlo (MC) e la Differenza Temporale (TD, dall'inglese *Temporal Difference*). Questo si chiama **predizione**, ed è utile, ma lascia aperta una domanda: e se la politica di partenza non fosse buona? Stimare il valore di una strategia mediocre non la rende una strategia migliore.

Questo capitolo compie il passo successivo: il **controllo**. Non ti basta più misurare quanto vale una politica, vuoi che l'agente la migliori mentre impara, episodio dopo episodio, fino ad avvicinarsi alla politica ottimale. Per farlo conoscerai due algoritmi che sono il cuore di gran parte dell'apprendimento per rinforzo (RL, dall'inglese *Reinforcement Learning*) moderno: SARSA (dalle iniziali di *State-Action-Reward-State-Action*) e Q-learning. Entrambi apprendono senza modello, cioè senza conoscere in anticipo come funziona l'ambiente, e partono dalla stessa domanda ("quanto è buona questa azione in questo stato?"), ma rispondono in modo diverso, con conseguenze pratiche molto differenti.

## Dal prevedere al migliorare: Iterazione di Politica Generalizzata

Immagina di imparare a cucinare un piatto nuovo. Lo prepari, lo assaggi, correggi un po' la ricetta, lo prepari di nuovo, lo assaggi ancora e correggi di nuovo. Non aspetti di padroneggiare la ricetta perfetta sulla carta prima di cucinare per la prima volta: alterni tra "valutare come è venuto" e "migliorare la ricetta", più volte, fino ad avvicinarti a qualcosa di eccellente.

Questa alternanza è esattamente l'idea alla base dell'Iterazione di Politica Generalizzata (GPI, dall'inglese *Generalized Policy Iteration*), un principio che struttura quasi tutto il controllo in RL:

1. **Valutazione**: stima quanto vale ciascuna azione secondo la politica attuale
2. **Miglioramento**: modifica la politica in modo che privilegi le azioni con il miglior valore stimato
3. **Ripeti**: torna al passo uno con la politica migliorata

Il punto interessante della GPI è che non serve completare ogni passo alla perfezione perché funzioni. Non occorre valutare la politica fino all'ultimo dettaglio prima di migliorarla: puoi valutare un po', migliorare un po', valutare un po' di più, e questa versione "a metà" del processo converge comunque verso la politica ottimale nel caso tabulare (quando memorizzi un valore esplicito per ogni combinazione di stato e azione). SARSA e Q-learning sono, in fondo, due modi diversi di eseguire la GPI passo passo, aggiornando il valore dopo ogni singola azione invece di aspettare che termini l'intero episodio.

Una differenza importante rispetto al capitolo precedente: nel controllo non basta più stimare il valore dello stato V(s). Poiché non disponi di un modello dell'ambiente, ti serve sapere il valore di ogni **azione** in ogni stato, Q(s,a), perché così puoi migliorare la politica semplicemente scegliendo l'azione con Q più alto, senza dover simulare cosa succederebbe con ciascuna alternativa.

## Il dilemma dell'esplorazione, risolto con ε-greedy e GLIE

Se l'agente sceglie sempre solo l'azione che oggi sembra migliore, non scopre mai se un'altra azione, non ancora provata, sia in realtà superiore. Per questo il controllo ha bisogno di una strategia di esplorazione. La più diffusa è la politica **ε-greedy** (epsilon-greedy): con probabilità ε l'agente sceglie un'azione a caso (esplora), e con probabilità 1−ε sceglie l'azione con il valore Q più alto conosciuto finora (sfrutta). Un valore tipico di ε può diminuire nel tempo, ad esempio proporzionalmente a 1/k, dove k è il numero di episodi: all'inizio esplori molto perché sai poco, e col tempo sfrutti di più perché ti fidi delle tue stime.

Per garantire che questo processo converga davvero verso la politica ottimale, esiste una condizione formale chiamata GLIE (dall'inglese *Greedy in the Limit with Infinite Exploration*, "greedy al limite con esplorazione infinita"). Richiede due cose: che ogni combinazione di stato e azione venga visitata un numero infinito di volte (per non lasciare nessuna opzione sufficientemente inesplorata) e che la politica diventi completamente greedy man mano che il tempo passa (così, alla fine, smetti di esplorare e sfrutti ciò che hai imparato). Far decrescere ε in modo progressivo è il modo abituale per soddisfare questa condizione.

Prima di entrare in SARSA e Q-learning, vale la pena menzionare che esiste anche il **controllo Monte Carlo**: valuti Q(s,a) facendo la media dei ritorni completi degli episodi (come nel capitolo precedente) e migliori la politica rendendola ε-greedy rispetto a quei valori, ripetendo il ciclo. Funziona, ma come tutto ciò che si basa su MC, richiede episodi completi e tende ad avere più varianza. SARSA e Q-learning risolvono lo stesso problema aggiornando a ogni passo, senza aspettare la fine dell'episodio, grazie al bootstrapping della TD.

## SARSA: impara la politica che stai effettivamente seguendo

SARSA prende il nome dalla sequenza che usa in ogni aggiornamento: stato, azione, ricompensa, stato successivo, azione successiva (*State-Action-Reward-State-Action*). L'agente si trova in uno stato, sceglie un'azione con la sua politica ε-greedy, osserva la ricompensa e il nuovo stato e, questo è il passo chiave, **sceglie anche l'azione successiva con la stessa politica ε-greedy**, prima di aggiornare. L'aggiornamento usa il valore Q di quell'azione successiva, cioè quella che verrà davvero eseguita:

Q(S,A) si avvicina un po' a: ricompensa ricevuta + valore Q scontato nello stato successivo, per l'azione che la politica ε-greedy ha scelto davvero

Per questo SARSA è un algoritmo **on-policy** (dentro la politica): impara il valore della politica che sta effettivamente seguendo, compresa la sua stessa casualità esplorativa. Se la tua politica ε-greedy a volte commette "errori" esplorando, SARSA incorpora questi errori nella sua stima di valore, e questo la rende più prudente: impara a evitare le zone in cui un'esplorazione sfortunata potrebbe costare caro.

## Q-learning: impara la politica ottimale, anche se esplori in un altro modo

Q-learning cambia un solo dettaglio, ma con conseguenze grandi. Invece di usare il valore Q dell'azione che effettivamente compirai subito dopo, usa il valore **massimo** possibile tra tutte le azioni dello stato successivo, indipendentemente da quale sceglierai davvero:

Q(S,A) si avvicina un po' a: ricompensa ricevuta + valore scontato della migliore Q possibile nello stato successivo (il massimo su tutte le azioni)

Questo rende Q-learning un algoritmo **off-policy** (fuori politica): la politica di comportamento (con cui esplori, per esempio ε-greedy) può essere diversa dalla politica obiettivo che stai imparando (quella puramente greedy, l'ottimale). Q-learning impara il valore di "se da ora in poi agissi sempre in modo ottimale", anche mentre continua a esplorare in modo imperfetto. Questa separazione tra "come esploro" e "che cosa imparo" è proprio la definizione di apprendimento off-policy, e permette anche, per esempio, di imparare a partire da dati raccolti con un'altra politica o già registrati in precedenza.

## L'analogia della scogliera: prudente contro rischioso

Il modo più chiaro per percepire la differenza tra i due algoritmi è il classico esempio della **scogliera** (*Cliff Walking*). Immagina una griglia in cui l'agente parte dall'angolo in basso a sinistra e l'obiettivo si trova nell'angolo in basso a destra. Tra i due, lungo il bordo inferiore, c'è una scogliera: cadere giù comporta una forte penalità e ti riporta al punto di partenza. Ogni passo normale costa un po' (una piccola penalità), quindi conviene arrivare il prima possibile.

Il percorso più corto costeggia la scogliera, attaccato al bordo. È anche il più rischioso, perché un solo passo falso, per esempio un movimento esplorativo casuale dovuto a ε-greedy, ti fa cadere giù dal bordo.

- **Q-learning** impara che quel percorso corto e attaccato al bordo è l'ottimale, perché calcola il valore supponendo che da ora in poi agirà sempre in modo ottimale, senza considerare che la sua stessa esplorazione potrebbe farlo cadere. Il risultato finale, quando smette di esplorare, è il percorso più corto possibile
- **SARSA** impara un percorso più lontano dal bordo, più lungo ma più sicuro, perché la sua stima di valore tiene conto del fatto che, mentre esplora, a volte sbaglia, e un passo falso vicino alla scogliera costa molto caro

Durante l'addestramento stesso, mentre entrambi gli agenti continuano a esplorare con ε-greedy, SARSA tende ad accumulare più ricompensa per episodio perché cade meno volte dalla scogliera. Q-learning, invece, cade di più durante l'addestramento (perché la sua politica obiettivo ignora il rischio dell'esplorazione), ma alla fine impara la politica obiettivo più corta e ottimale. Nessuno dei due "vince" in astratto: ciascuno risponde a una domanda diversa su quale politica stai valutando.

## Esempio con codice: Cliff Walking con SARSA e Q-learning

L'esempio seguente implementa da zero l'ambiente della scogliera (senza librerie di ambienti esterne) e addestra un agente con ciascun algoritmo. Copialo in una cella di Google Colab; ti servono solo `numpy` e `matplotlib`, che sono già preinstallati.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- EL ENTORNO: CLIFF WALKING ----------------------------------------
# Cuadrícula de 4 filas x 12 columnas. El agente empieza abajo a la
# izquierda (fila 3, columna 0) y el objetivo está abajo a la derecha
# (fila 3, columna 11). Toda la fila inferior entre ambos, excepto el
# inicio y el objetivo, es el "acantilado".
#
# Acciones: 0=arriba, 1=abajo, 2=izquierda, 3=derecha

FILAS, COLUMNAS = 4, 12
INICIO = (3, 0)
META = (3, 11)
ACANTILADO = [(3, c) for c in range(1, 11)]
NUM_ACCIONES = 4


def paso(estado, accion):
    """Aplica una acción y devuelve (nuevo_estado, recompensa, terminado)."""
    fila, col = estado
    if accion == 0:
        fila = max(fila - 1, 0)
    elif accion == 1:
        fila = min(fila + 1, FILAS - 1)
    elif accion == 2:
        col = max(col - 1, 0)
    elif accion == 3:
        col = min(col + 1, COLUMNAS - 1)

    nuevo_estado = (fila, col)

    if nuevo_estado in ACANTILADO:
        # Caer por el acantilado: penalización fuerte y vuelta al inicio
        return INICIO, -100, False
    if nuevo_estado == META:
        return nuevo_estado, -1, True
    return nuevo_estado, -1, False


def politica_epsilon_greedy(Q, estado, epsilon):
    """Con probabilidad epsilon explora; si no, elige la mejor acción conocida."""
    if np.random.rand() < epsilon:
        return np.random.randint(NUM_ACCIONES)
    valores = Q[estado]
    return np.random.choice(np.flatnonzero(valores == valores.max()))


def entrenar_sarsa(num_episodios=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = np.zeros((FILAS, COLUMNAS, NUM_ACCIONES))
    recompensas_por_episodio = []

    for _ in range(num_episodios):
        estado = INICIO
        accion = politica_epsilon_greedy(Q, estado, epsilon)
        recompensa_total = 0
        terminado = False

        while not terminado:
            siguiente_estado, recompensa, terminado = paso(estado, accion)
            siguiente_accion = politica_epsilon_greedy(Q, siguiente_estado, epsilon)

            # Actualización SARSA: usa el Q de la acción que SÍ se va a tomar
            objetivo = recompensa + gamma * Q[siguiente_estado][siguiente_accion]
            Q[estado][accion] += alpha * (objetivo - Q[estado][accion])

            estado, accion = siguiente_estado, siguiente_accion
            recompensa_total += recompensa

        recompensas_por_episodio.append(recompensa_total)

    return Q, recompensas_por_episodio


def entrenar_q_learning(num_episodios=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = np.zeros((FILAS, COLUMNAS, NUM_ACCIONES))
    recompensas_por_episodio = []

    for _ in range(num_episodios):
        estado = INICIO
        recompensa_total = 0
        terminado = False

        while not terminado:
            accion = politica_epsilon_greedy(Q, estado, epsilon)
            siguiente_estado, recompensa, terminado = paso(estado, accion)

            # Actualización Q-learning: usa el MÁXIMO Q del estado siguiente,
            # sin importar qué acción se elija después en la práctica
            mejor_valor_siguiente = Q[siguiente_estado].max()
            objetivo = recompensa + gamma * mejor_valor_siguiente
            Q[estado][accion] += alpha * (objetivo - Q[estado][accion])

            estado = siguiente_estado
            recompensa_total += recompensa

        recompensas_por_episodio.append(recompensa_total)

    return Q, recompensas_por_episodio


def extraer_ruta_greedy(Q, max_pasos=50):
    """Sigue la política puramente codiciosa (sin exploración) desde el inicio."""
    estado = INICIO
    ruta = [estado]
    for _ in range(max_pasos):
        if estado == META:
            break
        accion = np.argmax(Q[estado])
        estado, _, terminado = paso(estado, accion)
        ruta.append(estado)
        if terminado:
            break
    return ruta


# --- ENTRENAMIENTO ------------------------------------------------------
np.random.seed(0)
Q_sarsa, recompensas_sarsa = entrenar_sarsa()
Q_qlearning, recompensas_qlearning = entrenar_q_learning()

# --- GRÁFICO 1: recompensa por episodio durante el entrenamiento -------
# Se suaviza con una media móvil porque episodio a episodio hay mucho ruido
def media_movil(valores, ventana=20):
    return np.convolve(valores, np.ones(ventana) / ventana, mode="valid")

plt.figure(figsize=(9, 5))
plt.plot(media_movil(recompensas_sarsa), label="SARSA (on-policy)")
plt.plot(media_movil(recompensas_qlearning), label="Q-learning (off-policy)")
plt.xlabel("Episodio")
plt.ylabel("Recompensa por episodio (media móvil)")
plt.title("Cliff Walking: recompensa durante el entrenamiento")
plt.legend()
plt.tight_layout()
plt.show()

# --- GRÁFICO 2: rutas finales aprendidas (política greedy) -------------
def dibujar_ruta(ax, ruta, titulo):
    cuadricula = np.zeros((FILAS, COLUMNAS))
    for (f, c) in ACANTILADO:
        cuadricula[f, c] = -1
    ax.imshow(cuadricula, cmap="Pastel1")
    filas_ruta = [p[0] for p in ruta]
    columnas_ruta = [p[1] for p in ruta]
    ax.plot(columnas_ruta, filas_ruta, marker="o", color="tab:blue")
    ax.scatter(*INICIO[::-1], color="green", s=100, label="Inicio", zorder=5)
    ax.scatter(*META[::-1], color="red", s=100, label="Meta", zorder=5)
    ax.set_title(titulo)
    ax.invert_yaxis()
    ax.legend(loc="upper center")

fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
dibujar_ruta(ejes[0], extraer_ruta_greedy(Q_sarsa), "Ruta final de SARSA (más prudente)")
dibujar_ruta(ejes[1], extraer_ruta_greedy(Q_qlearning), "Ruta final de Q-learning (más corta)")
plt.tight_layout()
plt.show()
```

Quando lo esegui, osserva il primo grafico: la curva di SARSA sta di solito sopra quella di Q-learning per gran parte dell'addestramento, perché cade meno volte dalla scogliera mentre esplora. Nel secondo grafico, guarda i due percorsi finali: quello di Q-learning di solito si attacca al bordo della scogliera (il più corto e ottimale), mentre quello di SARSA si allontana di qualche riga verso l'alto, sacrificando lunghezza in cambio di sicurezza di fronte all'esplorazione.

## Bias di sovrastima e Double Q-learning (cenno breve)

Q-learning ha un effetto collaterale sottile: poiché il suo aggiornamento usa il **massimo** tra diverse stime rumorose di Q, quel massimo tende a essere distorto verso l'alto, come quando scegli sempre "il candidato migliore" tra varie misurazioni imprecise: spesso finisci per prendere quella che esagera per caso, anche se il valore reale di tutte le opzioni è lo stesso. Questo fenomeno è noto come **bias di sovrastima** e può portare l'agente a sopravvalutare sistematicamente certe azioni.

**Double Q-learning** corregge questo problema mantenendo due tabelle di valori indipendenti: una decide quale azione sembra migliore, l'altra valuta quanto vale davvero quell'azione (scambiando i ruoli a ogni passo). Separando "chi sceglie" da "chi valuta", si elimina gran parte del bias, perché non è più lo stesso stimatore a scegliersi come vincitore e poi a congratularsi con sé stesso per questo.

Esiste anche una variante chiamata **SARSA atteso**, che invece di usare il valore Q dell'azione che effettivamente si compie subito dopo (come SARSA) o il massimo (come Q-learning), usa la media ponderata di Q su tutte le azioni possibili, secondo la probabilità che ciascuna ha nella politica. È un modo per rendere più morbido l'aggiornamento e ridurre la varianza del campionamento, e quando la politica obiettivo è completamente greedy, questa versione coincide esattamente con Q-learning.

## Riepilogo dei punti chiave

- Il controllo cerca di migliorare la politica, non solo di valutarla: alterna valutazione e miglioramento seguendo il principio dell'Iterazione di Politica Generalizzata (GPI)
- La politica ε-greedy bilancia esplorazione e sfruttamento; la condizione GLIE garantisce la convergenza se visiti abbastanza ogni cosa e rendi la politica greedy col tempo
- SARSA è on-policy: impara il valore della politica che sta davvero seguendo, esplorazione inclusa, il che la rende più prudente
- Q-learning è off-policy: impara il valore della politica ottimale anche se esplora in un altro modo, il che la rende più rischiosa durante l'addestramento ma ottimale alla fine
- L'esempio della scogliera mostra la differenza con chiarezza: SARSA accumula più ricompensa durante l'addestramento, Q-learning trova il percorso più corto
- L'operatore massimo di Q-learning introduce un bias di sovrastima; Double Q-learning lo corregge separando selezione e valutazione in due tabelle

## Mettilo in pratica

Prendi il codice precedente e **sperimenta con due modifiche**, una alla volta, per vedere come influiscono su entrambe le politiche:

1. **Cambia epsilon**: porta `epsilon` a 0.3 nelle due funzioni di addestramento e riesegui. Osserva se il divario tra la ricompensa di SARSA e quella di Q-learning durante l'addestramento si allarga o si riduce
2. **Cambia la penalità della scogliera**: nella funzione `paso`, sostituisci il −100 con, per esempio, −10 o −500, e confronta come cambiano i due percorsi finali. **Annota le tue osservazioni** e verifica se, con una penalità molto bassa, i due percorsi finiscono per assomigliarsi

Finora hai memorizzato tutti i valori Q in una tabella, una casella per ogni combinazione di stato e azione. Questa tabella funziona perfettamente su una griglia di dodici colonne, ma smette di essere praticabile non appena il numero di stati esplode, per esempio quando lo stato sono i pixel dello schermo di un videogioco. Nel capitolo otto sostituirai quella tabella con una rete neurale che impara ad approssimare Q(s,a): così nascono le reti Q profonde (DQN, dall'inglese *Deep Q-Networks*).
