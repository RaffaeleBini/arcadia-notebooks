# Capitolo 5. Programmazione dinamica: iterazione di politica e di valore

Nel capitolo precedente hai conosciuto i Processi di Decisione di Markov (PDM): un modello formale con stati, azioni, probabilità di transizione, ricompense e un fattore di sconto. Hai anche imparato a calcolare Vπ, la funzione di valore di una politica fissa, tramite la valutazione iterativa della politica: applicare l'equazione di Bellman una volta dopo l'altra, scansione dopo scansione, finché i valori smettono di cambiare.

Questo è utile, ma incompleto. Sapere quanto vale seguire una politica concreta non ti dice se quella politica è buona. È come avere una mappa delle distanze dal tuo ufficio a ogni punto della città, ma continuare a usare sempre lo stesso percorso, senza verificare se esiste una scorciatoia migliore. In questo capitolo fai il passo che mancava: passare da "posso valutare una politica" a "posso trovare la migliore politica possibile". Gli strumenti che userai si chiamano, nel complesso, programmazione dinamica: una famiglia di algoritmi che calcolano politiche ottimali quando conosci il modello completo dell'ambiente, cioè quando conosci a memoria P (le probabilità di transizione) e R (le ricompense), senza bisogno di interagire con l'ambiente per tentativi ed errori.

## Dal valutare al migliorare

Immagina di aver già calcolato Vπ per una politica π qualsiasi, per esempio la politica casuale del capitolo precedente. Ora fermati in ogni casella del GridWorld e poniti una domanda diversa: "se, invece di seguire la politica attuale, prendo la migliore azione possibile in questo solo passo e poi torno a seguire π, ottengo un risultato migliore?". Confrontare quella scelta con ciò che faceva la politica originale, e restare sempre con l'azione migliore, si chiama miglioramento della politica: costruire una nuova politica π' che agisce in modo avido (greedy) rispetto a Vπ, cioè che in ogni stato sceglie l'azione che massimizza la ricompensa immediata più il valore scontato dello stato successivo.

Questo quasi mai peggiora le cose. Se in uno stato l'azione avida coincide con ciò che già facevi, non perdi nulla; se è diversa, migliori almeno un po'. Per questo π' è sempre uguale o migliore di π in tutti gli stati. È lo stesso principio che applichi quando riconsideri il tuo percorso abituale per il lavoro con le informazioni sul traffico di oggi: nel peggiore dei casi continui come prima; nel migliore trovi una scorciatoia reale.

## Iterazione di politica: valutare e migliorare, ancora e ancora

Qui compare l'idea centrale del capitolo. Se migliorare una politica una volta già aiuta, che cosa succede se ripeti l'intero ciclo? Valuti la politica attuale fino in fondo (calcoli la sua Vπ esatta con la valutazione iterativa), la migliori in modo avido, valuti la nuova politica, la migliori di nuovo, e così via. Questa procedura si chiama iterazione di politica (*policy iteration*).

Pensa a due frecce che si inseguono. La valutazione è una freccia verso l'alto: regola i valori finché combaciano perfettamente con la politica attuale. Il miglioramento è una freccia verso il basso: appena hai quei valori esatti, cambi la politica, quindi i valori che hai appena calcolato non descrivono più bene la nuova politica. Il processo ripete questo andirivieni (su, giù, su, giù) finché, in un ciclo, la politica non cambia più. A quel punto l'azione avida in ogni stato coincide esattamente con ciò che la politica già faceva, il che può significare una sola cosa: vale l'equazione di ottimalità di Bellman, e quindi la politica attuale è già la politica ottimale π*. Poiché il numero di politiche deterministiche possibili in un PDM finito è limitato e ogni ciclo produce una politica uguale o migliore della precedente, l'algoritmo è garantito nel raggiungere π* in un numero finito di cicli.

## Iterazione di valore: la scorciatoia diretta

L'iterazione di politica funziona, ma ha un punto debole: a ogni ciclo, valutare la politica a fondo (ripetere la valutazione iterativa finché converge del tutto) può costare molte scansioni, e gran parte di quella precisione viene buttata via non appena migliori comunque la politica. Qui nasce una domanda naturale: serve aspettare che la valutazione converga del tutto prima di migliorare?

La risposta è no, e questo porta direttamente all'iterazione di valore (*value iteration*): invece di completare la valutazione, la interrompi a una singola scansione e mescoli valutazione e miglioramento nello stesso passo. Più precisamente, in ogni scansione, ogni stato aggiorna il proprio valore prendendo direttamente il massimo tra le sue azioni (invece di mediare secondo una politica fissa), usando l'equazione di ottimalità di Bellman come regola di aggiornamento. Quando quei valori smettono di cambiare in modo apprezzabile, ricavi la politica ottimale con un ultimo sguardo a un passo: in ogni stato scegli l'azione che massimizza la ricompensa più il valore scontato dello stato successivo.

Puoi vederlo come un caso particolare all'interno della stessa famiglia dell'iterazione di politica: è un'iterazione di politica in cui la fase di valutazione è stata ridotta a una sola scansione. In pratica, di solito richiede più scansioni individuali di quanti cicli completi richieda l'iterazione di politica, ma ogni scansione è molto più economica, quindi il costo totale risulta minore in molti problemi.

## Sincronia, aggiornamenti sul posto e altre varianti

Tutto quanto detto finora presuppone una programmazione dinamica sincrona: in ogni scansione aggiorni tutti gli stati insieme, usando sempre i valori della scansione precedente. Esiste anche la programmazione dinamica asincrona, che aggiorna gli stati uno alla volta (o per sottoinsiemi), usando il valore più recente disponibile al momento, senza aspettare di completare un'intera scansione. Una variante comune sono gli aggiornamenti sul posto (*in-place*): invece di mantenere una copia "vecchia" e una "nuova" della funzione di valore, aggiorni direttamente lo stesso array, il che di solito accelera la convergenza. Altre varianti danno priorità agli stati da aggiornare per primi in base all'entità dell'errore di Bellman in ogni casella, oppure iniziano ad aggiornare vicino allo stato terminale e procedono all'indietro. Non approfondire qui: tieni solo presente che, nei problemi grandi, non serve scandire l'intera griglia in modo ordinato perché il metodo continui a convergere.

## Esempio in Python: iterazione di politica contro iterazione di valore

Il seguente codice riprende il GridWorld del capitolo precedente: una griglia 4×4 (16 stati), con un'unica casella terminale nell'angolo in basso a destra, ricompensa di −1 per ogni passo che non raggiunge la meta e 0 quando la si raggiunge. I movimenti sono deterministici: se l'agente prova a uscire dalla scacchiera, resta fermo. Su questo stesso GridWorld implementi l'iterazione di politica e l'iterazione di valore, e confronti quante scansioni richiede ciascuna per convergere. Puoi copiarlo ed eseguirlo direttamente in Google Colab: serve solo `numpy`.

```python
import numpy as np

# ---------------------------------------------------------------
# GridWorld: el mismo mundo-cuadrícula 4x4 del capítulo anterior,
# con una casilla terminal (la meta) y recompensa -1 por paso.
# ---------------------------------------------------------------

ACCIONES = ["arriba", "abajo", "izquierda", "derecha"]

DESPLAZAMIENTOS = {
    "arriba": (-1, 0),
    "abajo": (1, 0),
    "izquierda": (0, -1),
    "derecha": (0, 1),
}

FLECHAS = {"arriba": "^", "abajo": "v", "izquierda": "<", "derecha": ">"}


class GridWorld:
    """Cuadrícula n_filas x n_columnas con una casilla terminal (la meta).
    Estados = tuplas (fila, columna). Transiciones deterministas: moverse
    contra el borde del tablero deja al agente en la misma casilla."""

    def __init__(self, n_filas=4, n_columnas=4, estados_terminales=None, recompensa_paso=-1.0):
        self.n_filas = n_filas
        self.n_columnas = n_columnas
        self.recompensa_paso = recompensa_paso

        if estados_terminales is None:
            estados_terminales = [(n_filas - 1, n_columnas - 1)]  # esquina inferior derecha
        self.estados_terminales = set(estados_terminales)

        self.estados = [(f, c) for f in range(n_filas) for c in range(n_columnas)]
        self.acciones = ACCIONES

    def es_terminal(self, estado):
        return estado in self.estados_terminales

    def estado_siguiente(self, estado, accion):
        if self.es_terminal(estado):
            return estado
        df, dc = DESPLAZAMIENTOS[accion]
        fila, columna = estado
        nueva_fila = max(0, min(self.n_filas - 1, fila + df))
        nueva_columna = max(0, min(self.n_columnas - 1, columna + dc))
        return (nueva_fila, nueva_columna)

    def recompensa(self, estado, accion, estado_siguiente):
        if self.es_terminal(estado):
            return 0.0
        return self.recompensa_paso


def politica_aleatoria(mundo):
    """Política de partida: en cada estado, las cuatro acciones son
    igual de probables. Es el punto de arranque de iteración de políticas."""
    probabilidad_uniforme = 1.0 / len(mundo.acciones)
    return {
        estado: {accion: probabilidad_uniforme for accion in mundo.acciones}
        for estado in mundo.estados
    }


def evaluacion_iterativa_politica(mundo, politica, gamma=0.9, umbral=1e-4, max_iteraciones=1000):
    """Ya la viste en el capítulo anterior: aplica la ecuación de Bellman de
    expectativa hasta que Vpi deja de cambiar. La repetimos aquí para que el
    ejemplo sea autocontenido."""
    V = {estado: 0.0 for estado in mundo.estados}
    for iteracion in range(1, max_iteraciones + 1):
        delta_maximo = 0.0
        V_nuevo = dict(V)
        for estado in mundo.estados:
            if mundo.es_terminal(estado):
                V_nuevo[estado] = 0.0
                continue
            valor_esperado = 0.0
            for accion in mundo.acciones:
                prob_accion = politica[estado][accion]
                estado_sig = mundo.estado_siguiente(estado, accion)
                r = mundo.recompensa(estado, accion, estado_sig)
                valor_esperado += prob_accion * (r + gamma * V[estado_sig])
            V_nuevo[estado] = valor_esperado
            delta_maximo = max(delta_maximo, abs(V_nuevo[estado] - V[estado]))
        V = V_nuevo
        if delta_maximo < umbral:
            break
    return V, iteracion


def mejora_politica(mundo, V, gamma=0.9):
    """Construye una política determinista, codiciosa respecto a V: en cada
    estado, elige la acción que maximiza recompensa + gamma * V(siguiente)."""
    nueva_politica = {}
    for estado in mundo.estados:
        if mundo.es_terminal(estado):
            nueva_politica[estado] = {a: 0.0 for a in mundo.acciones}
            continue
        valores_q = {}
        for accion in mundo.acciones:
            estado_sig = mundo.estado_siguiente(estado, accion)
            r = mundo.recompensa(estado, accion, estado_sig)
            valores_q[accion] = r + gamma * V[estado_sig]
        mejor_accion = max(valores_q, key=valores_q.get)
        nueva_politica[estado] = {a: (1.0 if a == mejor_accion else 0.0) for a in mundo.acciones}
    return nueva_politica


def iteracion_politicas(mundo, gamma=0.9, umbral=1e-4):
    """Alterna evaluación completa y mejora hasta que la política deja de cambiar."""
    politica = politica_aleatoria(mundo)
    ciclos_evaluacion_mejora = 0
    barridos_totales = 0
    while True:
        ciclos_evaluacion_mejora += 1
        V, barridos = evaluacion_iterativa_politica(mundo, politica, gamma, umbral)
        barridos_totales += barridos
        nueva_politica = mejora_politica(mundo, V, gamma)
        if nueva_politica == politica:  # la política ya no cambia: es óptima
            break
        politica = nueva_politica
    return politica, V, ciclos_evaluacion_mejora, barridos_totales


def iteracion_valores(mundo, gamma=0.9, umbral=1e-4, max_iteraciones=1000):
    """Funde evaluación y mejora: cada barrido actualiza V con el máximo
    sobre acciones (ecuación de optimalidad de Bellman), no con un promedio
    según una política fija."""
    V = {estado: 0.0 for estado in mundo.estados}
    for iteracion in range(1, max_iteraciones + 1):
        delta_maximo = 0.0
        V_nuevo = dict(V)
        for estado in mundo.estados:
            if mundo.es_terminal(estado):
                V_nuevo[estado] = 0.0
                continue
            valores_q = []
            for accion in mundo.acciones:
                estado_sig = mundo.estado_siguiente(estado, accion)
                r = mundo.recompensa(estado, accion, estado_sig)
                valores_q.append(r + gamma * V[estado_sig])
            V_nuevo[estado] = max(valores_q)
            delta_maximo = max(delta_maximo, abs(V_nuevo[estado] - V[estado]))
        V = V_nuevo
        if delta_maximo < umbral:
            break
    politica_final = mejora_politica(mundo, V, gamma)  # una última mirada de un paso
    return politica_final, V, iteracion


def imprimir_politica(mundo, politica):
    for f in range(mundo.n_filas):
        fila_textos = []
        for c in range(mundo.n_columnas):
            estado = (f, c)
            if mundo.es_terminal(estado):
                fila_textos.append(" G ")
            else:
                mejor_accion = max(politica[estado], key=politica[estado].get)
                fila_textos.append(f" {FLECHAS[mejor_accion]} ")
        print("|".join(fila_textos))


if __name__ == "__main__":
    mundo = GridWorld(n_filas=4, n_columnas=4, recompensa_paso=-1.0)
    gamma = 0.9

    print("=== Iteración de políticas ===")
    politica_pi, V_pi, ciclos, barridos_pi = iteracion_politicas(mundo, gamma=gamma)
    print(f"Ciclos evaluación+mejora: {ciclos}")
    print(f"Barridos de evaluación acumulados: {barridos_pi}")
    imprimir_politica(mundo, politica_pi)

    print("\n=== Iteración de valores ===")
    politica_vi, V_vi, barridos_vi = iteracion_valores(mundo, gamma=gamma)
    print(f"Barridos hasta converger: {barridos_vi}")
    imprimir_politica(mundo, politica_vi)

    diferencia_maxima = max(abs(V_pi[s] - V_vi[s]) for s in mundo.estados)
    print(f"\nDiferencia máxima entre las dos V*: {diferencia_maxima:.6f}")
```

Con γ (gamma) = 0,9, entrambi i metodi arrivano esattamente alla stessa politica ottimale: da qualsiasi casella, la freccia punta sempre verso la meta lungo il percorso più breve (in basso e a destra, combinati a seconda della posizione). La differenza sta nello sforzo. L'iterazione di politica ha bisogno di soli tre cicli completi di valutazione e miglioramento, ma ogni valutazione viene eseguita fino in fondo, quindi in totale accumula 90 scansioni. L'iterazione di valore arriva allo stesso risultato in appena sette scansioni, perché non aspetta mai che la valutazione converga del tutto prima di migliorare. Questo schema è tipico: l'iterazione di politica di solito richiede meno cicli "esterni", ma ciascuno costa di più; l'iterazione di valore fa molte scansioni economiche invece di poche scansioni costose.

## Punti chiave

- Il miglioramento di politica costruisce una nuova politica agendo in modo avido (greedy) rispetto a Vπ; non peggiora mai la politica originale
- L'iterazione di politica alterna valutazione completa e miglioramento finché la politica smette di cambiare; a quel punto vale l'equazione di ottimalità di Bellman e la politica è ottimale
- L'iterazione di valore unisce valutazione e miglioramento in un'unica scansione per iterazione, usando il massimo sulle azioni invece di una media secondo una politica fissa
- Entrambi i metodi richiedono di conoscere il modello completo dell'ambiente (P e R); per questo appartengono alla programmazione dinamica, non all'apprendimento per tentativi ed errori
- La programmazione dinamica asincrona (aggiornamenti sul posto, priorità in base all'errore di Bellman, aggiornamento dallo stato terminale all'indietro) permette di estendere l'idea a problemi più grandi senza perdere la garanzia di convergenza

## Mettilo in pratica

Prendi il codice di questo capitolo e modificalo tu stesso o tu stessa:

- **Aggiungi ostacoli** al GridWorld: scegli due o tre caselle che l'agente non possa attraversare (trattale come se fossero un muro, in modo che tentare di entrarvi lo lasci al suo posto) e osserva come la politica ottimale le aggira
- **Cambia il fattore di sconto** γ (per esempio a 0,5 o a 0,99) e verifica se cambia il numero di scansioni che ogni metodo richiede per convergere
- **Aggiungi una seconda casella terminale** con una ricompensa diversa (per esempio, una meta secondaria con ricompensa minore) e verifica come si divide la griglia tra le due mete in base alla posizione di ciascuno stato

Esegui il notebook, modifica il GridWorld e condividi con il tuo team quale politica ottimale ha ottenuto ogni configurazione!

---

Finora, in tutto il blocco sulla programmazione dinamica, hai dato per scontato di conoscere il modello dell'ambiente: le probabilità di transizione P e la funzione di ricompensa R. Nel capitolo 6 lascerai da parte questa ipotesi e imparerai a valutare e migliorare le politiche senza conoscere il modello, solo a partire dall'esperienza: Monte Carlo e Differenze Temporali.
