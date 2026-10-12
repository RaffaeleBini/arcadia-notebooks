# Capitolo 4. Processi di Decisione di Markov: il linguaggio dell'RL

Nel capitolo precedente hai conosciuto i bandit, la tua prima porta d'accesso all'apprendimento per rinforzo (RL, dall'inglese *reinforcement learning*): un agente che sceglie tra diverse leve e deve solo bilanciare esplorazione e sfruttamento, perché ogni tiro è indipendente da quelli precedenti. Quel mondo è semplice perché non esiste uno "stato" che cambi: non importa cosa hai fatto prima, la macchina ti offre sempre le stesse opzioni.

La vita reale quasi mai funziona così. Quando guidi, irrighi una coltivazione o gestisci l'inventario di un magazzino, ogni decisione ti lascia in una situazione diversa che condiziona quali opzioni hanno senso dopo. Ti serve un linguaggio formale che catturi questa catena di causa ed effetto per ragionarci matematicamente: il Processo di Decisione di Markov (PDM), in inglese *Markov Decision Process* (MDP). Troverai entrambi i termini nel codice e nella letteratura.

Questo capitolo ti presenta il PDM pezzo per pezzo: i suoi elementi, la proprietà matematica che lo rende trattabile e come si derivano le equazioni che userai per il resto del manuale. Alla fine avrai in Python un mondo a griglia (GridWorld) funzionante che il capitolo 5 amplierà per trovare la migliore strategia possibile.

## Dalla slot machine alla mappa completa

Pensa a un robot che raccoglie lattine di alluminio per riciclarle e funziona a batteria. In ogni istante si trova in uno dei due **stati** possibili in base al livello di carica: alto o basso. A seconda dello stato in cui si trova, può scegliere tra diverse **azioni**: aspettare fermo, andare a cercare lattine o, se ha poca energia, tornare alla base per ricaricarsi.

Ecco già i primi ingredienti del PDM:

- **S** (states, stati): l'insieme delle situazioni possibili in cui può trovarsi l'agente — per il robot, {carica alta, carica bassa}
- **A** (actions, azioni): le opzioni disponibili in ogni stato — aspettare, cercare o ricaricarsi
- **P**: la funzione di transizione: la probabilità di finire in ciascuno stato successivo, a seconda di dove ti trovavi e dell'azione che hai scelto. Cercare con la batteria carica quasi sempre ti lascia nello stesso stato di carica, ma esiste una probabilità di restare senza energia e aver bisogno di un soccorso
- **R**: la funzione di ricompensa, il numero che ricevi dopo ogni transizione. Cercare lattine dà una ricompensa positiva; restare bloccato senza batteria, una forte penalizzazione
- **γ** (gamma): il fattore di sconto, un numero tra zero e uno che decide quanto ti importa del futuro

Un PDM completo è la tupla **(S, A, P, R, γ)**: stati, azioni, come si concatenano tra loro, quale ricompensa lascia ogni passo e quanto pesa il domani rispetto all'oggi.

### La proprietà di Markov: conta solo il presente

Il nome "Markov" non è decorativo. Il PDM assume la **proprietà di Markov**: il futuro dipende unicamente dallo stato attuale, non da come ci si è arrivati. Non importa se il robot ha passato due ore a cercare lattine o se si è appena ricaricato: l'unica cosa rilevante per decidere cosa fare ora è il suo livello di batteria in questo momento.

Questa ipotesi semplifica enormemente il problema: non devi conservare l'intero storico delle decisioni passate, basta uno stato ben progettato che riassuma ciò che conta. Per questo, quando progetti un sistema reale — per esempio lo stato di un veicolo autonomo, che potrebbe includere posizione, velocità e ostacoli vicini — il tuo compito è scegliere una rappresentazione che si avvicini il più possibile a questa proprietà.

### La politica: la tua strategia decisionale

Una **politica**, scritta come π (pi), è la strategia seguita dall'agente: una regola che, per ogni stato, indica quale azione intraprendere. Può essere:

- **Deterministica**: nello stato "batteria bassa", π sceglie sempre "ricaricarsi"
- **Stocastica**: nello stato "batteria alta", π sceglie "cercare" il 70% delle volte e "aspettare" nel 30% restante

Una politica casuale, che distribuisce la stessa probabilità tra tutte le azioni disponibili, è il punto di partenza più semplice per cominciare a ragionare su un PDM, ed è proprio quella che userai nell'esempio di codice.

### Quanto vale uno stato: la funzione di valore

Qui arriva la domanda centrale di questo capitolo: se segui una politica π, quanto "vale" trovarsi in uno stato concreto? La risposta non è solo ciò che guadagni in questo istante, ma la somma di tutte le ricompense future, scontate di γ. Questa somma si chiama **ritorno** (Gt):

```
Gt = Rt+1 + γ·Rt+2 + γ²·Rt+3 + ...
```

Il fattore γ fa due cose insieme: riflette il fatto che di solito preferisci ottenere qualcosa oggi piuttosto che la stessa ricompensa tra dieci passi, ed evita che la somma cresca senza limite nei compiti che potrebbero continuare indefinitamente, perché ogni ricompensa futura pesa un po' meno della precedente. Un γ vicino a zero produce un agente miope, che guarda solo il passo immediato; uno vicino a uno lo rende lungimirante, disposto a sacrificare una ricompensa immediata per un beneficio maggiore più avanti.

A partire dal ritorno si definiscono due **funzioni di valore**:

- **Vπ(s)**: il ritorno atteso se parti dallo stato s e segui la politica π da lì in avanti
- **Qπ(s,a)**: il ritorno atteso se, trovandoti nello stato s, esegui l'azione a e poi segui π

Le due sono collegate in modo diretto: Vπ(s) è la media di Qπ(s,a) su tutte le azioni, ponderata per la probabilità che la politica assegna a ciascuna.

### Le equazioni di Bellman: ricompensa ora, valore scontato dopo

Il ritorno ha una proprietà molto utile: si può scrivere in forma ricorsiva, perché Gt = Rt+1 + γ·Gt+1. In parole semplici, ciò che vale questo momento è la ricompensa di questo passo più il valore scontato di tutto ciò che viene dopo. Applicata alla funzione di valore, quest'idea dà la **equazione di Bellman di aspettativa**:

```
Vπ(s) = Σ_a π(a|s) · Σ_s' P(s'|s,a) · [R(s,a,s') + γ·Vπ(s')]
```

Non lasciare che la notazione ti spaventi: è solo una somma pesata. Per ogni azione possibile (secondo la tua politica) e per ogni possibile stato successivo (secondo la probabilità di transizione), sommi la ricompensa immediata e il suo valore scontato. La stessa idea, applicata a Q, dà la versione per coppie stato-azione:

```
Qπ(s,a) = Σ_s' P(s'|s,a) · [R(s,a,s') + γ·Σ_a' π(a'|s') Qπ(s',a')]
```

Queste due equazioni definiscono il valore di una politica già fissata: sono un sistema lineare, e per questo puoi risolverle — oppure approssimarle iterando, come farai nel codice di questo capitolo.

### La domanda che apre il capitolo 5: e la politica migliore?

Tutto quanto detto finora presuppone che tu abbia già una politica π e voglia solo sapere quanto vale. Ma la domanda che ti interessa davvero è un'altra: qual è la migliore politica possibile? Quella politica ottimale si chiama **π\***, e la sua funzione di valore **V\*(s)** soddisfa l'**equazione di ottimalità di Bellman**:

```
V*(s) = max_a Σ_s' P(s'|s,a) · [R(s,a,s') + γ·V*(s')]
```

La differenza rispetto all'equazione di aspettativa è sottile ma decisiva: invece di mediare sulle azioni secondo la tua politica, prendi il massimo. Non chiedi più "quanto vale questo stato se seguo la mia strategia attuale?", ma "quanto varrebbe se, da ora in poi, scegliessi sempre la migliore azione possibile?". Quel massimo rende il sistema non lineare — non si risolve più con l'algebra diretta — e richiede metodi iterativi, che è proprio ciò di cui tratta il prossimo capitolo: la programmazione dinamica.

Tieni a mente l'idea chiave: una volta che hai V* o Q*, ottieni direttamente la politica ottimale agendo in modo "avido" (greedy) — π\*(s) = argmax_a Q\*(s,a). La sfida, che risolverai nel capitolo 5, è come calcolare V* o Q* in primo luogo.

## Esempio di codice: un GridWorld e la sua funzione di valore

Costruirai un piccolo mondo a griglia di quattro righe per quattro colonne (4×4), con un'unica casella terminale (la meta) nell'angolo in basso a destra. L'agente può muoversi nelle quattro direzioni cardinali; se prova a uscire dalla scacchiera, resta dove si trova. Ogni passo costa −1 di ricompensa, tranne l'arrivo alla meta, dove l'episodio termina.

Definirai una politica casuale — sceglie ciascuna direzione con la stessa probabilità — e calcolerai la sua Vπ applicando l'equazione di Bellman di aspettativa più e più volte, finché i valori smettono di cambiare. Questo processo si chiama **valutazione iterativa della politica**: la stessa equazione che hai appena visto, trasformata in una regola di aggiornamento che ripeti scansione dopo scansione. Nota che la classe `GridWorld` e le funzioni sono scritte per essere riutilizzate — il capitolo 5 aggiungerà il miglioramento della politica sopra questa stessa base, senza riscriverla.

```python
import numpy as np

# ---------------------------------------------------------------
# GridWorld: un mundo-cuadrícula pequeño para practicar el PDM.
#
# Esta clase se diseña para que el capítulo 5 la reutilice tal cual
# y le añada iteración de políticas e iteración de valores encima.
# ---------------------------------------------------------------

ACCIONES = ["arriba", "abajo", "izquierda", "derecha"]

# Cada acción mueve al agente una casilla en una dirección (delta_fila, delta_columna)
DESPLAZAMIENTOS = {
    "arriba": (-1, 0),
    "abajo": (1, 0),
    "izquierda": (0, -1),
    "derecha": (0, 1),
}


class GridWorld:
    """
    Un mundo-cuadrícula de n_filas x n_columnas con una casilla terminal (la meta).

    - Los estados son tuplas (fila, columna).
    - Las acciones son las cuatro direcciones cardinales.
    - Las transiciones son deterministas: moverse contra un muro o el
      borde del tablero deja al agente en la misma casilla.
    - La recompensa es 'recompensa_paso' en cada movimiento (por defecto -1),
      salvo en la casilla terminal, donde el episodio ya ha terminado.
    """

    def __init__(self, n_filas=4, n_columnas=4, estados_terminales=None, recompensa_paso=-1.0):
        self.n_filas = n_filas
        self.n_columnas = n_columnas
        self.recompensa_paso = recompensa_paso

        # Por defecto, la meta es la esquina inferior derecha
        if estados_terminales is None:
            estados_terminales = [(n_filas - 1, n_columnas - 1)]
        self.estados_terminales = set(estados_terminales)

        # Lista de todos los estados (fila, columna)
        self.estados = [(f, c) for f in range(n_filas) for c in range(n_columnas)]
        self.acciones = ACCIONES

    def es_terminal(self, estado):
        return estado in self.estados_terminales

    def estado_siguiente(self, estado, accion):
        """Devuelve la casilla resultante de aplicar 'accion' desde 'estado'.
        Si el movimiento choca con el borde del tablero, el agente se queda quieto."""
        if self.es_terminal(estado):
            return estado

        df, dc = DESPLAZAMIENTOS[accion]
        fila, columna = estado
        nueva_fila = max(0, min(self.n_filas - 1, fila + df))
        nueva_columna = max(0, min(self.n_columnas - 1, columna + dc))
        return (nueva_fila, nueva_columna)

    def recompensa(self, estado, accion, estado_siguiente):
        """Recompensa inmediata de la transición. En este GridWorld es fija
        (-1 por paso), pero se recibe (s, a, s') completo para poder
        ampliar el ejemplo más adelante (por ejemplo, recompensas distintas por casilla)."""
        if self.es_terminal(estado):
            return 0.0
        return self.recompensa_paso


def politica_aleatoria(mundo):
    """Construye una política estocástica que, en cada estado, elige
    cualquiera de las cuatro acciones con la misma probabilidad (0.25).
    Se representa como un diccionario: politica[estado][accion] = probabilidad."""
    probabilidad_uniforme = 1.0 / len(mundo.acciones)
    return {
        estado: {accion: probabilidad_uniforme for accion in mundo.acciones}
        for estado in mundo.estados
    }


def evaluacion_iterativa_politica(mundo, politica, gamma=0.9, umbral=1e-4, max_iteraciones=1000):
    """
    Calcula Vpi(s) para una política fija aplicando repetidamente
    la ecuación de Bellman de expectación:

        Vpi(s) = suma_a pi(a|s) * suma_s' P(s'|s,a) * [R(s,a,s') + gamma * Vpi(s')]

    Como este GridWorld es determinista, P(s'|s,a) vale 1 para el único
    estado_siguiente que devuelve mundo.estado_siguiente(), y 0 para el resto,
    así que la suma sobre s' desaparece y queda un único término.

    Devuelve V como un diccionario {estado: valor} y el número de barridos usados.
    """
    V = {estado: 0.0 for estado in mundo.estados}

    for iteracion in range(1, max_iteraciones + 1):
        delta_maximo = 0.0
        V_nuevo = dict(V)  # barrido síncrono: usamos siempre los valores del barrido anterior

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


def imprimir_valores(mundo, V):
    """Muestra V como una cuadrícula legible, para inspeccionar el resultado."""
    for f in range(mundo.n_filas):
        fila_textos = []
        for c in range(mundo.n_columnas):
            fila_textos.append(f"{V[(f, c)]:6.2f}")
        print(" ".join(fila_textos))


if __name__ == "__main__":
    mundo = GridWorld(n_filas=4, n_columnas=4, recompensa_paso=-1.0)
    politica = politica_aleatoria(mundo)

    V, num_barridos = evaluacion_iterativa_politica(mundo, politica, gamma=0.9, umbral=1e-4)

    print(f"Convergió en {num_barridos} barridos.\n")
    print("Función de valor Vpi bajo la política aleatoria:")
    imprimir_valores(mundo, V)
```

Eseguendolo in Colab otterrai qualcosa di simile a questo (i valori esatti dipendono da γ e dalla soglia di convergenza che scegli):

```
Convergió en 76 barridos.

Función de valor Vpi bajo la política aleatoria:
 -9.36  -9.22  -8.97  -8.75
 -9.22  -8.97  -8.50  -7.97
 -8.97  -8.50  -7.42  -5.76
 -8.75  -7.97  -5.76   0.00
```

Osserva lo schema: le caselle più vicine alla meta (l'angolo in basso a destra, con 0.00) hanno un valore meno negativo, perché sotto una politica casuale servono in media meno passi per arrivarci. Quelle più lontane, come l'angolo in alto a sinistra, accumulano più passi di ricompensa −1 prima di terminare. Ed è esattamente ciò che misura Vπ: il costo atteso, scontato, di arrivare alla fine seguendo la tua politica attuale.

## Punti chiave

- Un PDM è definito dalla tupla (S, A, P, R, γ): stati, azioni, transizioni, ricompense e fattore di sconto
- La proprietà di Markov dice che il futuro dipende solo dallo stato presente, non dallo storico completo
- Una politica π può essere deterministica o stocastica, e assegna a ogni stato un'azione o una distribuzione di probabilità su di esse
- Vπ(s) misura il ritorno atteso partendo da uno stato e seguendo π; Qπ(s,a) fa lo stesso ma fissando anche la prima azione
- L'equazione di Bellman di aspettativa scompone il valore attuale in ricompensa immediata più ciò che vale, scontato, lo stato successivo
- L'equazione di ottimalità di Bellman sostituisce la media pesata con un massimo sulle azioni, e definisce V* e la politica ottimale π*
- La valutazione iterativa della politica applica ripetutamente l'equazione di Bellman di aspettativa finché Vπ converge

## Mettilo in pratica

Prendi il codice precedente e sperimentaci prima di passare al capitolo 5:

1. **Cambia il fattore di sconto.** Prova con γ = 0.5 e con γ = 0.99. Come cambiano i valori delle caselle lontane dalla meta? Che cosa succede al numero di scansioni necessarie per la convergenza?
2. **Sposta la meta.** Invece dell'angolo in basso a destra, colloca lo stato terminale al centro della scacchiera (passa `estados_terminales=[(1, 1)]` quando costruisci il `GridWorld`) e osserva come si riorganizza Vπ attorno alla nuova meta
3. **Aggiungi una seconda meta.** Passa una lista con due caselle terminali e verifica che l'agente ora "veda" entrambi i possibili percorsi di uscita

Esegui ogni variante e annota quale intuizione conferma su γ e su come il valore si propaga nella griglia!

---

Ora hai la mappa — il PDM completo — e sai calcolare quanto vale seguire una politica concreta. Nel capitolo 5 userai questo stesso `GridWorld` per fare il passo successivo. Invece di valutare soltanto una strategia fissa, la migliorerai passo dopo passo tramite iterazione di politica e iterazione di valore. Arriverai così alla politica ottimale π*.
