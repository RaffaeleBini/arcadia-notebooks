# Capitolo 13. Casi d'uso reali: dai giochi classici ad AlphaStar

Hai passato dodici capitoli a costruire un edificio: hai iniziato con il dilemma di esplorazione ed exploitation nei bandit, hai continuato con i processi decisionali di Markov (PDM) e la programmazione dinamica, hai imparato a predire e controllare senza modello con Monte Carlo, Differenza Temporale (TD), SARSA e Q-learning, hai fatto il salto alle reti neurali con le reti Q profonde (DQN, dall'inglese *Deep Q-Network*) e i gradienti di politica, e hai chiuso con pianificazione, Ricerca ad Albero di Monte Carlo (MCTS, dall'inglese *Monte Carlo Tree Search*), Rainbow e i metodi actor-critic distribuiti come A3C e IMPALA. Tutto questo è, in fondo, teoria da lavagna: formule, pseudocodice, ambienti giocattolo.

Questo capitolo finale ti insegna che quella teoria non è rimasta sulla lavagna. Percorrerai una linea di sistemi reali, da Deep Blue ad AlphaStar, e in ciascuno riconoscerai pezzi che già conosci: una funzione di valore qui, una politica imparata per gradiente lì, una ricerca tipo MCTS più avanti. Non memorizzare nomi né date: osserva come le stesse idee semplici, combinate e scalate, sono bastate per battere i migliori giocatori umani del pianeta in giochi considerati da decenni — o secoli — baluardi dell'intelletto umano. È la chiusura perfetta del manuale, perché dimostra che ogni concetto che hai padroneggiato ha una propria versione funzionante dentro un sistema che ha fatto storia.

## Da Deep Blue al Monte Carlo Tree Search

Prima che qualsiasi rete neurale giocasse a Go, l'intelligenza artificiale (IA) batteva già campioni umani nei giochi da tavolo da trent'anni, con tecniche molto più vicine alla ricerca che all'apprendimento. I giochi funzionano come "moscerini della frutta" per la ricerca in IA: regole chiare, obiettivi misurabili, un punteggio che non mente, proprio come la genetica ha usato la Drosophila per scoprire principi poi generalizzati a organismi molto più complessi.

La prima pietra miliare fu **Chinook**, un programma di dama che raggiunse una partita praticamente perfetta combinando il taglio alfa-beta con l'analisi retrograda (risolvere i finali di partita all'indietro, da posizioni di vittoria già conosciute). Il suo rivale fu Marion Tinsley, probabilmente il miglior giocatore di dama della storia: perse solo cinque partite in tutta la sua carriera, e due di esse furono contro Chinook. Successivamente, Schaeffer e il suo team risolsero il gioco completamente.

Nel 1997 arrivò **Deep Blue**, di IBM, che vinse contro il campione mondiale di scacchi Garri Kasparov in uno scontro che divenne uno degli eventi più seguiti di internet del momento. Deep Blue non imparava nulla: valutava circa 8.000 caratteristiche codificate a mano (sicurezza del re, struttura dei pedoni, materiale) ed esaminava fino a 200 milioni di posizioni al secondo, guardando tra 16 e 40 mosse avanti. È, in sostanza, il minimax con taglio alfa-beta che hai studiato come fondamento teorico della ricerca nei giochi a somma zero: forza bruta guidata da una funzione di valutazione progettata da umani, senza alcuna traccia di reinforcement learning (RL, dall'inglese *Reinforcement Learning*).

Il contrappunto arrivò con **TD-Gammon**, un programma di backgammon che imparò una funzione di valore — la probabilità di vincere da una data posizione — addestrando una piccola rete neurale tramite TD, giocando unicamente contro se stesso, senza alcun dato umano. È la stessa idea che hai visto nel capitolo 6 (predizione senza modello con TD) e nel capitolo 8 (approssimare funzioni di valore con una rete neurale), applicata quasi un decennio prima che esistesse il termine "apprendimento profondo". Un dato curioso: il backgammon si gioca con i dadi, e quella casualità forniva già esplorazione naturale, così TD-Gammon non necessitò di una strategia esplicita di esplorazione come la ε-greedy del capitolo 7.

Poco dopo, **Maven** vinse contro il campione mondiale di Scrabble, Adam Logan, per 9 partite a 5, combinando una ricerca Monte Carlo troncata (simulare solo una decina di mosse avanti, non partite complete) con una funzione di valutazione imparata. Quella ricerca Monte Carlo troncata è l'anello che collega tutto quanto precede con il **MCTS**: invece di risolvere il gioco intero, simuli molte continuazioni possibili dalla posizione attuale e impari una politica locale, specifica di quella situazione. Il manuale ti ha già presentato il ciclo di selezione, espansione, simulazione e retropropagazione nel capitolo dedicato alla pianificazione; qui lo ritrovi come il ponte esatto verso AlphaGo.

## AlphaGo: la mossa che sorprese il mondo

Il Go si giocava da tremila anni e aveva circa 40 milioni di giocatori attivi quando DeepMind decise di affrontarlo. La sfida non era solo la sua scacchiera di 19 per 19: ogni posizione ammette circa 200 mosse legali in media (contro le 20-35 degli scacchi), e il numero di configurazioni possibili supera quello degli atomi nell'universo osservabile. La forza bruta di Deep Blue, qui, semplicemente non era un'opzione; il Go si gioca più per intuizione che per calcolo esaustivo, e AlphaGo doveva approssimare quell'intuizione.

L'architettura combinava due reti neurali convoluzionali: una **rete di politica**, che propone una distribuzione di probabilità sulle mosse legali, e una **rete di valore**, che stima un unico numero — la probabilità di vincere — da una data posizione. Riconosci entrambi i pezzi: la rete di valore è, in sostanza, la stessa funzione di valore del capitolo 4, imparata con una rete neurale come nel capitolo 8; la rete di politica è un gradiente di politica, come quelli del capitolo 9, addestrato prima per imitazione e poi per autoapprendimento (self-play). Le due reti si combinavano con MCTS tramite una variante chiamata PUCT (*Predictor + UCT*): la politica riduce l'ampiezza della ricerca (quali mosse vale la pena esplorare) e la rete di valore riduce la sua profondità (evita di giocare ogni ramo fino alla fine). Con questo, AlphaGo arrivava a guardare tra 50 e 60 mosse avanti.

L'addestramento ebbe due fasi: prima, apprendimento supervisionato a partire da partite di giocatori umani esperti, affinché la rete di politica imitasse mosse forti; poi, raffinamento per autoapprendimento, giocando milioni di partite contro versioni precedenti di se stessa, mentre la rete di valore imparava a predire il risultato finale di ogni posizione raggiunta. In circa 72 ore di addestramento, AlphaGo passò dal giocare come un principiante a un livello sovrumano.

I risultati parlano da sé: AlphaGo vinse contro il campione europeo Fan Hui per 5 partite a 0, e nel marzo 2016, a Seul, vinse contro il leggendario Lee Sedol — 9-dan, 18 titoli mondiali — per 4 partite a 1. La mossa più ricordata fu la **mossa 37** della seconda partita: un colpo di spalla sulla quinta linea così inusuale che AlphaGo stessa calcolò solo una probabilità su 10.000 che un umano la giocasse, e si rivelò una genialità strategica. Ma il sistema ebbe anche un limite: nella quarta partita, Lee Sedol trovò la **mossa 78**, a cui AlphaGo aveva assegnato appena lo 0,007% di probabilità, e vinse quella partita. Fu un ricordo utile, e onesto, che "sovrumano" non significa infallibile. Mesi dopo, AlphaGo Zero — addestrata da zero, senza alcun dato umano — vinse il novanta per cento delle partite contro AlphaGo Master, la versione che a sua volta aveva accumulato 60 vittorie e nessuna sconfitta contro i migliori professionisti del mondo.

## AlphaZero: imparare da zero, senza dati umani

AlphaZero semplifica e generalizza tutto quanto precede con un colpo solo: un'unica rete neurale, con output congiunto di politica e valore, e un unico algoritmo applicato senza modifiche a scacchi, shōgi (gli scacchi giapponesi) e Go, cambiando solo la codifica della scacchiera. Niente dati umani, niente aperture né finali precaricati: inizia con pesi casuali e le regole del gioco, e migliora esclusivamente giocando contro se stessa.

Il meccanismo che chiude il ciclo è puro GPI — l'Iterazione di Politica Generalizzata del capitolo 7, la stessa alternanza di valutare e migliorare che hai già visto con SARSA e Q-learning —, solo che qui MCTS svolge il ruolo di "operatore di miglioramento della politica": la ricerca produce una distribuzione di mosse migliore di quella suggerita dalla rete da sola, e la rete viene poi addestrata per imitare quella distribuzione migliorata. Formalmente:

π′(a|s) ∝ Q(s,a)^(1/τ)

dove π′ è la politica migliorata, Q(s,a) è il valore stimato da MCTS per ogni mossa e τ (tau) è un parametro di temperatura: valori alti esplorano di più, valori bassi concentrano la scelta sulla mossa più promettente. La rete si addestra con una perdita di entropia incrociata per la parte di politica e errore quadratico medio per la parte di valore, tutto in un'unica rete.

I risultati, con hardware specializzato (unità di elaborazione tensoriale, TPU), arrivarono con una velocità che sorprese persino i suoi creatori: dopo solo circa quattro ore di autoapprendimento, AlphaZero superava già Stockfish, il motore scacchistico più forte del momento; dopo circa due ore, superava Elmo, il miglior programma di shōgi; e dopo circa otto ore, superava la stessa versione di AlphaGo che aveva battuto Lee Sedol. Contro Stockfish — che valuta milioni di posizioni al secondo con regole di materiale fissate a mano (pedone=1, cavallo=alfiere=3, torre=5, donna=9) — AlphaZero sviluppò uno stile proprio: sacrificare materiale per attività dei pezzi e controllo a lungo termine. In una partita documentata con apertura Ruy Lopez, AlphaZero sacrificò la donna alla mossa 25 per smontare la struttura di Stockfish e finì vincendo per scacco matto; in un'altra, con apertura siciliana, sacrificò una torre per un pedone alla mossa 24 per assicurarsi una struttura di pedoni superiore. Nessun manuale di scacchi le ha insegnato a fare questo: lo ha scoperto giocando contro se stessa, nello stesso modo in cui il Q-learning del capitolo 7 impara una politica ottimale senza che nessuno gli dica in anticipo qual è.

## AlphaStar: la sfida di StarCraft II

Se il Go e gli scacchi sono giochi a informazione perfetta e turni discreti, StarCraft II è esattamente l'opposto: informazione parziale (la famosa "nebbia di guerra" nasconde buona parte della mappa), milioni di azioni possibili in ogni istante e decisioni da prendere in tempo reale, non a turni, combinando pianificazione strategica a lungo termine con microgestione di decine di unità contemporaneamente. DeepMind costruì **AlphaStar** per giocare la razza Protoss in questo ambiente, in collaborazione con Blizzard Entertainment.

L'addestramento combinò due fasi che già riconosci: prima, circa tre giorni di apprendimento per imitazione su partite umane reali, che portarono l'agente a un livello approssimativo di giocatore "platino"; poi, circa una settimana di autoapprendimento organizzato nella cosiddetta **Lega AlphaStar** — agenti principali, "exploiter" specializzati nel trovare le loro debolezze e agenti di prova, che competevano tra loro a un ritmo equivalente a centinaia di anni di partite umane. Quello schema di lega con molteplici agenti che giocano in parallelo è la stessa filosofia distribuita di A3C e IMPALA, del capitolo 12, portata a una scala molto maggiore; e l'algoritmo di RL che regolava la politica in ogni partita fu l'Ottimizzazione di Politica Prossimale (PPO, dall'inglese *Proximal Policy Optimization*), un metodo a gradiente di politica che limita quanto può cambiare la politica a ogni aggiornamento, per non destabilizzare l'apprendimento.

Architetturalmente, AlphaStar combinò reti convoluzionali per elaborare la mappa, memoria a lungo termine (LSTM, *Long Short-Term Memory*) per ricordare informazioni passate sotto osservazione parziale, e meccanismi di attenzione per dare priorità a quali unità contano in ogni momento. Perché il confronto con gli umani fosse equo, il team limitò il suo tempo di reazione a circa 350 millisecondi e le sue azioni al minuto (APM) a livelli comparabili a quelli di un professionista, in modo che una vittoria riflettesse decisioni migliori e non velocità di click sovrumana. Il risultato: una vittoria per 10 a 0 contro il professionista Grzegorz "MaNa" Komincz, dopo essersi misurato anche contro Liquid TLO. Osserva il pattern: esplorazione/exploitation a scala di lega (capitolo 3), autoapprendimento che genera i propri dati (come in AlphaZero), e gradienti di politica addestrati con PPO (capitolo 9) per digerire tutto questo, questa volta sotto informazione incompleta.

## Esempio con codice: mini-AlphaZero per Tris

L'esempio seguente implementa, in scala giocattolo, lo stesso pattern visto nelle tre sezioni precedenti: autoapprendimento, ricerca tipo MCTS e miglioramento iterativo della politica. Gioca a Tris invece che a Go, e usa una semplice funzione euristica più una tabella di valori invece di una rete neurale addestrata per davvero, così che possa essere eseguito in secondi su qualsiasi Google Colab, senza altra dipendenza che `numpy`. Il codice è commentato affinché tu veda, riga per riga, dove si inserisce ogni concetto dei capitoli precedenti.

```python
import math
import random

import numpy as np

# ============================================================
# AVISO DIDACTICO: esto es una version de juguete, a escala minima,
# del patron de AlphaZero (autojuego + busqueda tipo MCTS + mejora
# iterativa de la politica). NO es AlphaZero real: no hay red
# neuronal convolucional entrenada por descenso de gradiente, el
# "valor" se guarda en un simple diccionario en vez de una red, y el
# juego es Tres en Raya, no ajedrez ni Go. El objetivo es que veas el
# mismo engranaje -autojuego, busqueda, mejora de politica- funcionar
# de principio a fin en un juego trivial, no reproducir el sistema real.
# ============================================================

LINEAS_GANADORAS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def tablero_nuevo():
    return tuple([0] * 9)  # 0 = vacio, 1 = X, -1 = O


def movimientos_validos(tablero):
    return [i for i, v in enumerate(tablero) if v == 0]


def aplicar_movimiento(tablero, pos, jugador):
    nuevo = list(tablero)
    nuevo[pos] = jugador
    return tuple(nuevo)


def resultado(tablero):
    """Devuelve +1 si gana X, -1 si gana O, 0 si hay empate, None si sigue."""
    for a, b, c in LINEAS_GANADORAS:
        suma = tablero[a] + tablero[b] + tablero[c]
        if suma == 3:
            return 1
        if suma == -3:
            return -1
    if 0 not in tablero:
        return 0
    return None


def heuristica(tablero):
    """
    Sustituta minima de la red de valor de AlphaZero: para cada linea de
    tres casillas, si solo hay piezas de X (o ninguna) suma piezas propias
    al cuadrado; si solo hay piezas de O, resta lo mismo. Es una funcion
    fija, no aprendida: solo sirve como estimacion de partida cuando la
    tabla de valores aprendida todavia no ha visto una posicion. Devuelve
    un valor aproximado en [-1, 1] desde la perspectiva de X.
    """
    puntuacion = 0
    for linea in LINEAS_GANADORAS:
        valores = [tablero[i] for i in linea]
        piezas_x = valores.count(1)
        piezas_o = valores.count(-1)
        if piezas_o == 0:
            puntuacion += piezas_x ** 2
        if piezas_x == 0:
            puntuacion -= piezas_o ** 2
    return math.tanh(puntuacion / 6.0)


# Tabla de valores aprendida por autojuego: estado -> valor estimado
# desde la perspectiva de X (+1 = X gana, -1 = O gana, 0 = tablas).
# Este diccionario cumple, a pequenisima escala, el papel que en
# AlphaZero cumple la red de valor entrenada (capitulo 8: aproximacion
# de funciones con redes neuronales; capitulo 4: funciones de valor).
tabla_valores = {}


def valor_estimado(tablero):
    if tablero in tabla_valores:
        return tabla_valores[tablero]
    return heuristica(tablero)


class Nodo:
    __slots__ = ("tablero", "jugador", "hijos", "N", "W", "expandido")

    def __init__(self, tablero, jugador):
        self.tablero = tablero
        self.jugador = jugador  # jugador (+1 o -1) que debe mover aqui
        self.hijos = {}
        self.N = 0
        self.W = 0.0
        self.expandido = False

    def valor_medio(self):
        return self.W / self.N if self.N > 0 else 0.0


def ucb_score(padre, hijo, c=1.4):
    """
    Formula tipo UCB/PUCT (busqueda y planificacion): combina "cuanto de
    bueno parece" (Q, convertido a la perspectiva del jugador que mueve
    en el padre) con "cuanto se ha explorado" (U, que crece si el hijo se
    ha visitado poco). En AlphaZero, U tambien incorpora la probabilidad
    previa de la red de politica; aqui usamos exploracion uniforme para
    mantener el ejemplo simple.
    """
    q = hijo.valor_medio() * padre.jugador
    u = c * math.sqrt(math.log(padre.N + 1) / (hijo.N + 1))
    return q + u


def simular(nodo):
    """Un ciclo de seleccion, expansion, evaluacion y retropropagacion."""
    r = resultado(nodo.tablero)
    if r is not None:
        nodo.N += 1
        nodo.W += r
        return r

    if not nodo.expandido:
        for m in movimientos_validos(nodo.tablero):
            tablero_hijo = aplicar_movimiento(nodo.tablero, m, nodo.jugador)
            nodo.hijos[m] = Nodo(tablero_hijo, -nodo.jugador)
        nodo.expandido = True
        # En vez de un rollout aleatorio hasta el final (MCTS "clasico"),
        # evaluamos la posicion con la tabla de valores o la heuristica,
        # igual que AlphaZero sustituye el rollout por su red de valor.
        v = valor_estimado(nodo.tablero)
        nodo.N += 1
        nodo.W += v
        return v

    mejor_m = max(nodo.hijos, key=lambda m: ucb_score(nodo, nodo.hijos[m]))
    v = simular(nodo.hijos[mejor_m])
    nodo.N += 1
    nodo.W += v
    return v


def busqueda_mcts(tablero, jugador, num_simulaciones=60):
    """Devuelve {movimiento: visitas} tras num_simulaciones simulaciones."""
    raiz = Nodo(tablero, jugador)
    for _ in range(num_simulaciones):
        simular(raiz)
    return {m: hijo.N for m, hijo in raiz.hijos.items()}


def elegir_movimiento(visitas, temperatura):
    """
    Convierte las visitas de la MCTS en una politica y elige un movimiento.
    Con temperatura > 0 hay exploracion (util en autojuego, para generar
    partidas variadas); con temperatura = 0 se juega siempre el movimiento
    mas visitado (el modo que se usa al evaluar). Es la version practica
    de pi'(a|s) ~ Q(s,a)^(1/tau), la formula de mejora de politica que
    usa AlphaZero.
    """
    movimientos = list(visitas.keys())
    conteos = np.array([visitas[m] for m in movimientos], dtype=float)

    if temperatura == 0:
        return movimientos[int(np.argmax(conteos))]

    probabilidades = conteos ** (1.0 / temperatura)
    probabilidades /= probabilidades.sum()
    return np.random.choice(movimientos, p=probabilidades)


def jugar_partida_autojuego(num_simulaciones=60, temperatura=0.8):
    """
    Juega una partida completa de X contra O, ambos controlados por la
    misma busqueda MCTS (autojuego). Devuelve los tableros visitados y el
    resultado final, listos para actualizar la tabla de valores.
    """
    tablero = tablero_nuevo()
    jugador = 1
    historial = []

    while resultado(tablero) is None:
        visitas = busqueda_mcts(tablero, jugador, num_simulaciones)
        movimiento = elegir_movimiento(visitas, temperatura)
        historial.append(tablero)
        tablero = aplicar_movimiento(tablero, movimiento, jugador)
        jugador = -jugador

    return historial, resultado(tablero)


def actualizar_tabla_valores(historial, resultado_final, alpha=0.5):
    """
    Paso de mejora de politica: acerca el valor estimado de cada posicion
    visitada durante la partida hacia el resultado real, con una
    actualizacion tipo Monte Carlo / TD. Con el tiempo, la tabla deja de
    depender de la heuristica inicial y refleja lo que el propio
    autojuego fue aprendiendo.
    """
    for tablero in historial:
        actual = valor_estimado(tablero)
        tabla_valores[tablero] = actual + alpha * (resultado_final - actual)


def movimiento_aleatorio(tablero):
    return random.choice(movimientos_validos(tablero))


def jugar_vs_aleatorio(num_simulaciones=60):
    """
    Partida de evaluacion: el agente MCTS (con la tabla de valores actual,
    temperatura 0 = juega su mejor movimiento) contra un oponente que
    mueve al azar. El agente MCTS siempre es X.
    """
    tablero = tablero_nuevo()
    jugador = 1
    while resultado(tablero) is None:
        if jugador == 1:
            visitas = busqueda_mcts(tablero, jugador, num_simulaciones)
            movimiento = elegir_movimiento(visitas, temperatura=0)
        else:
            movimiento = movimiento_aleatorio(tablero)
        tablero = aplicar_movimiento(tablero, movimiento, jugador)
        jugador = -jugador
    return resultado(tablero)


# --- BUCLE PRINCIPAL: AUTOJUEGO + MEJORA, GENERACION TRAS GENERACION ----
np.random.seed(0)
random.seed(0)

GENERACIONES = 6
PARTIDAS_AUTOJUEGO_POR_GENERACION = 40
PARTIDAS_EVALUACION = 50

for generacion in range(1, GENERACIONES + 1):
    # 1) Autojuego: generar partidas y usarlas para mejorar la tabla de valores
    for _ in range(PARTIDAS_AUTOJUEGO_POR_GENERACION):
        historial, resultado_final = jugar_partida_autojuego()
        actualizar_tabla_valores(historial, resultado_final)

    # 2) Evaluacion: medir el nivel actual contra un jugador aleatorio
    victorias, empates, derrotas = 0, 0, 0
    for _ in range(PARTIDAS_EVALUACION):
        r = jugar_vs_aleatorio()
        if r == 1:
            victorias += 1
        elif r == 0:
            empates += 1
        else:
            derrotas += 1

    print(
        f"Generacion {generacion}: "
        f"{victorias} victorias, {empates} empates, {derrotas} derrotas "
        f"(sobre {PARTIDAS_EVALUACION} partidas contra un jugador aleatorio) "
        f"| tabla de valores con {len(tabla_valores)} posiciones aprendidas"
    )
```

Eseguendolo, vedrai una riga per generazione con il punteggio contro il giocatore casuale e la dimensione crescente della tabella di valori. In un'esecuzione di riferimento, l'agente non perse più nemmeno una partita a partire dalla prima generazione, e il numero di pareggi e vittorie variò leggermente secondo quanto esplorò ogni ciclo di autoapprendimento: il Tris è così piccolo che il pattern converge quasi immediatamente, cosa che non succede nemmeno lontanamente con gli scacchi o il Go, dove AlphaZero ha avuto bisogno di ore di calcolo massivo invece di secondi su un laptop.

## Riepilogo dei punti chiave

- Prima di AlphaGo, l'IA giocava già a livello sovrumano a dama, scacchi, backgammon e Scrabble, combinando ricerca (minimax, taglio alfa-beta) con funzioni di valore, a volte imparate e a volte progettate a mano
- Il MCTS è il ponte tra quella storia e AlphaGo: cerca tramite simulazioni invece di risolvere il gioco completo, e converge verso il valore minimax nei giochi a somma zero con informazione perfetta
- AlphaGo combinò una rete di politica, una rete di valore e MCTS, addestrate prima per imitazione di partite umane e poi per autoapprendimento, e vinse contro Lee Sedol 4 a 1
- AlphaZero eliminò completamente i dati umani: una sola rete, un solo algoritmo, applicato senza modifiche a scacchi, shōgi e Go, superando Stockfish, Elmo e la stessa AlphaGo in poche ore di autoapprendimento
- AlphaStar dimostrò che lo stesso pattern — imitazione più autoapprendimento, questa volta organizzato in una lega di agenti — funziona anche sotto informazione parziale e spazi di azione enormi, vincendo 10 a 0 contro un professionista di StarCraft II
- Ogni sistema è una ricombinazione di idee dei capitoli 2 a 12: funzioni di valore, politiche imparate per gradiente, controllo off-policy, approssimazione con reti neurali, e ricerca tipo MCTS guidata da quelle reti

## Mettilo in pratica

Il codice precedente è il tuo modello. Ora **sperimenta** con questi tre cambiamenti, uno alla volta:

1. **Riduci le simulazioni di MCTS**: abbassa `num_simulaciones` da 60 a 5 in `jugar_partida_autojuego` e in `jugar_vs_aleatorio`, e riesegui. **Osserva** quante generazioni servono adesso per smettere di perdere contro il giocatore casuale
2. **Rimuovi l'euristica**: cambia `heuristica(tablero)` affinché restituisca sempre `0.0`. **Confronta** quanto tempo serve all'agente per diventare competente quando parte da zero, senza alcun vantaggio iniziale, proprio come AlphaZero rispetto ad AlphaGo
3. **Adatta la scacchiera**: provi a portare lo stesso scheletro (autoapprendimento, MCTS, tabella di valori) a una versione piccola di Forza 4, per esempio su una griglia 4x4 con connessione di tre in linea. Dovrai **riscrivere** `LINEAS_GANADORAS`, `movimientos_validos` e la dimensione della scacchiera, ma il resto del pattern resta intatto

Questo manuale termina qui, ma il reinforcement learning no. Quello che hai percorso — bandit, PDM, programmazione dinamica, Monte Carlo e TD, SARSA e Q-learning, DQN, gradienti di politica, pianificazione, Rainbow, metodi distribuiti, e ora questi sistemi che hanno fatto storia — è una base solida per leggere gli articoli originali di DeepMind, sperimentare con librerie come Gymnasium o PettingZoo, o semplicemente continuare a giocare con le tue proprie varianti del codice di questo capitolo. Il campo continua a muoversi rapidamente, e il modo migliore per tenerne il ritmo è lo stesso che hai imparato in questo manuale: prova, osserva il risultato, aggiusta la politica e riprova.
