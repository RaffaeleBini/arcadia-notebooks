# Capitolo 6: Predizione senza modello — Monte Carlo e Differenze Temporali

## Perché ti interessa questo tema

Nel capitolo precedente hai visto la Programmazione Dinamica (DP): un insieme di metodi potenti, ma con una condizione molto esigente. Per applicarli devi conoscere in anticipo il modello completo dell'ambiente, cioè le probabilità di transizione tra gli stati e la funzione di ricompensa. In un problema con una scacchiera piccola questa esigenza è ragionevole. In un caso reale — ottimizzare l'irrigazione di una coltivazione, pianificare la manutenzione di una flotta da pesca o mettere a punto un protocollo di cura — quasi mai disponi di quel modello esatto. L'ambiente è troppo complesso, cambia nel tempo o semplicemente nessuno l'ha misurato con precisione.

Qui entra in gioco l'apprendimento **senza modello** (*model-free*): invece di calcolare il valore di ogni stato a partire da equazioni note, lo stimi a partire dall'esperienza, episodio dopo episodio, come farebbe chiunque impari un mestiere per tentativi e osservazione. Questo capitolo ti insegna le due famiglie di metodi che fanno questo: Monte Carlo (MC) e le Differenze Temporali (TD, dall'inglese *Temporal-Difference*). Entrambe risolvono lo stesso problema — la **predizione**: stimare la funzione di valore di una politica fissa, Vπ(s), senza conoscere il modello — ma ci arrivano per strade molto diverse, con conseguenze pratiche importanti che vedrai nel codice alla fine del capitolo.

## Che cosa significa "predizione" quando non hai il modello

Ricorda che Vπ(s) rappresenta il ritorno atteso se segui la politica π a partire dallo stato s. Con la DP calcolavi quel valore risolvendo equazioni sul modello. Senza modello, l'unica fonte di informazione che ti resta è l'esperienza reale: lasci che l'agente agisca secondo π, osservi quali ricompense riceve e con questi dati stimi Vπ(s). È la differenza tra calcolare la durata media di un tragitto con una mappa del traffico perfetta e stimarla cronometrando il tragitto molte volte.

## Monte Carlo: impara dall'episodio completo

Il metodo MC è il più diretto dei due: lascia che l'episodio finisca, calcola il ritorno totale ottenuto a partire da ogni stato visitato e usa quel ritorno come campione del valore reale. Ripeti l'operazione con molti episodi e fai la media. Per la legge dei grandi numeri, quella media converge al vero Vπ(s).

Immagina di guidare ogni pomeriggio dal lavoro a casa e di voler stimare quanto tempo impiega quel tragitto. L'approccio Monte Carlo consiste nell'aspettare di arrivare a casa, annotare il tempo totale e usarlo per aggiornare la tua stima dell'intero tragitto. Più giorni ripeti il percorso, migliore sarà la stima, ma devi arrivare a casa ogni volta per imparare qualcosa: se l'episodio non finisce, MC non ha nulla da mediare.

All'interno di MC esistono due varianti, a seconda di quante volte conti ogni stato all'interno dello stesso episodio:

- **Prima visita** (*first-visit*): usi il ritorno a partire dalla prima volta in cui compare uno stato nell'episodio, anche se quello stato si ripete più avanti
- **Ogni visita** (*every-visit*): usi il ritorno a partire da ogni occorrenza dello stato, non solo dalla prima

Entrambe convergono allo stesso valore con un numero sufficiente di episodi; la differenza sta nel modo in cui trattano gli stati che si ripetono all'interno di uno stesso episodio.

## Differenze Temporali: indovina a partire dalla tua stessa previsione

TD(0) risolve lo stesso problema con un'idea diversa: non aspettare la fine dell'episodio, aggiorna la tua stima dopo ogni passo. Per farlo combina la ricompensa che hai appena osservato con la tua stima attuale del valore dello stato successivo, invece di aspettare il ritorno reale completo. Questa tecnica si chiama **bootstrapping**: usi una stima per correggere un'altra stima.

Seguendo l'analogia del tragitto verso casa: invece di aspettare di arrivare per giudicare se il viaggio sta andando bene o male, aggiusti la tua previsione a ogni tratto. Uscendo dall'ufficio sotto la pioggia, rivedi al rialzo la stima della durata totale. Quando entri in autostrada senza code, la rivedi al ribasso. Non serve arrivare a casa per imparare qualcosa di utile a ogni tratto del percorso.

La regola di aggiornamento di TD(0) è questa:

V(S) ← V(S) + α · [R + γ·V(S') − V(S)]

Dove α è il tasso di apprendimento, R è la ricompensa osservata nel compiere il passo, γ è il fattore di sconto e V(S') è la tua stima attuale dello stato successivo (zero se S' è uno stato terminale). Il termine tra parentesi quadre si chiama **errore TD**: la differenza tra ciò che hai appena osservato più ciò che credi valga il resto del cammino, e ciò che credevi prima di compiere il passo.

Poiché TD non ha bisogno di aspettare la fine, funziona anche con episodi incompleti o con compiti che non terminano mai, cosa che MC non può fare.

## Distorsione e varianza: il grande compromesso

Questa differenza di meccanismo ha una conseguenza statistica diretta:

- MC calcola la media di ritorni completi e reali, quindi la sua stima è **non distorta** (*unbiased*): in media centra il valore vero. Il prezzo è una **varianza alta**, perché ogni ritorno dipende da tutta la sequenza di azioni e ricompense casuali che si sono verificate fino alla fine dell'episodio
- TD aggiorna con una sola ricompensa osservata più una stima ancora imperfetta del resto, quindi è **distorta** finché quella stima non converge. In cambio ha una **varianza bassa**, perché ogni aggiornamento dipende da un solo passo, non dall'intero episodio

Con infiniti episodi e un tasso di apprendimento che decresce in modo adeguato, entrambi i metodi convergono allo stesso Vπ(s). Con dati finiti — che è quasi sempre la situazione reale — possono dare risposte diverse, e di solito conviene accettare un po' di distorsione in cambio di una varianza molto minore. Per questo TD è così usato nella pratica: impara più in fretta e con stime più stabili, anche se impiega un po' a "togliersi" la distorsione iniziale.

## Il caso di studio: la passeggiata casuale

Per confrontare i due metodi in codice userai la **passeggiata casuale** (*random walk*), un ambiente minimo ma molto istruttivo: una catena di cinque stati non terminali, con uno stato terminale a ciascuna estremità. Dallo stato centrale, a ogni passo ti muovi a sinistra o a destra con la stessa probabilità. Se arrivi all'estremità destra ricevi una ricompensa di uno; se arrivi all'estremità sinistra, ricevi zero. È un ambiente così semplice che puoi calcolare a mano il valore reale di ogni stato, il che ti permette di misurare con precisione quanto rapidamente converge ciascun metodo.

## Codice: Monte Carlo vs. TD(0) nella passeggiata casuale

Il seguente codice è autocontenuto: serve solo `numpy` e `matplotlib`, e puoi eseguirlo così com'è in Google Colab.

```python
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Definición del entorno: paseo aleatorio con cinco estados internos
# ---------------------------------------------------------------------
# Estados 0 y 6 son terminales (izquierda y derecha).
# Estados 1-5 son A, B, C, D, E; el episodio siempre arranca en C (estado 3).
N_ESTADOS_INTERNOS = 5
ESTADO_INICIAL = 3          # C, el estado central de la cadena
TERMINAL_IZQUIERDA = 0
TERMINAL_DERECHA = N_ESTADOS_INTERNOS + 1

# Valor real de cada estado interno (calculable a mano en este entorno
# tan simple): V(estado i) = i / (N_ESTADOS_INTERNOS + 1)
VALORES_REALES = np.array(
    [i / (N_ESTADOS_INTERNOS + 1) for i in range(1, N_ESTADOS_INTERNOS + 1)]
)


def generar_episodio():
    """Simula un episodio de paseo aleatorio y devuelve la lista de
    (estado, recompensa) que atraviesa, terminando en el estado terminal."""
    estado = ESTADO_INICIAL
    trayectoria = []
    while True:
        # Con probabilidad 0.5 te mueves a la izquierda o a la derecha
        paso = -1 if np.random.rand() < 0.5 else 1
        siguiente_estado = estado + paso
        # Solo el extremo derecho da recompensa; el resto de pasos dan cero
        recompensa = 1.0 if siguiente_estado == TERMINAL_DERECHA else 0.0
        trayectoria.append((estado, recompensa))
        estado = siguiente_estado
        if estado in (TERMINAL_IZQUIERDA, TERMINAL_DERECHA):
            break
    return trayectoria


# ---------------------------------------------------------------------
# 2. Predicción por Monte Carlo (primera visita)
# ---------------------------------------------------------------------
def monte_carlo_prediccion(n_episodios, alpha, gamma=1.0):
    """Estima V para los estados internos 1..5 usando MC de primera visita
    con actualización incremental V(s) <- V(s) + alpha * (G - V(s))."""
    V = np.full(N_ESTADOS_INTERNOS, 0.5)  # inicialización neutra
    historial_rmse = []

    for _ in range(n_episodios):
        trayectoria = generar_episodio()

        # Calcula el retorno G_t desde cada paso hacia el final del episodio
        G = 0.0
        retornos_por_estado = {}
        for estado, recompensa in reversed(trayectoria):
            G = recompensa + gamma * G
            # Primera visita: si el estado se repite, nos quedamos con el
            # retorno calculado en la ocurrencia más temprana (la última
            # que procesamos al recorrer la trayectoria al revés)
            retornos_por_estado[estado] = G

        for estado, G in retornos_por_estado.items():
            if estado not in (TERMINAL_IZQUIERDA, TERMINAL_DERECHA):
                idx = estado - 1  # los estados internos van de 1 a 5
                V[idx] += alpha * (G - V[idx])

        rmse = np.sqrt(np.mean((V - VALORES_REALES) ** 2))
        historial_rmse.append(rmse)

    return V, historial_rmse


# ---------------------------------------------------------------------
# 3. Predicción por TD(0)
# ---------------------------------------------------------------------
def td0_prediccion(n_episodios, alpha, gamma=1.0):
    """Estima V para los estados internos 1..5 con TD(0): actualiza en
    cada paso, sin esperar a que el episodio termine."""
    V = np.full(N_ESTADOS_INTERNOS, 0.5)
    historial_rmse = []

    for _ in range(n_episodios):
        estado = ESTADO_INICIAL
        while estado not in (TERMINAL_IZQUIERDA, TERMINAL_DERECHA):
            paso = -1 if np.random.rand() < 0.5 else 1
            siguiente_estado = estado + paso
            recompensa = 1.0 if siguiente_estado == TERMINAL_DERECHA else 0.0

            # Valor del siguiente estado: cero si es terminal, si no, la
            # estimación actual (esto es el "bootstrapping")
            if siguiente_estado in (TERMINAL_IZQUIERDA, TERMINAL_DERECHA):
                v_siguiente = 0.0
            else:
                v_siguiente = V[siguiente_estado - 1]

            idx = estado - 1
            error_td = recompensa + gamma * v_siguiente - V[idx]
            V[idx] += alpha * error_td

            estado = siguiente_estado

        rmse = np.sqrt(np.mean((V - VALORES_REALES) ** 2))
        historial_rmse.append(rmse)

    return V, historial_rmse


# ---------------------------------------------------------------------
# 4. Comparación: velocidad de convergencia de MC vs TD(0)
# ---------------------------------------------------------------------
np.random.seed(0)
N_EPISODIOS = 150
ALPHA = 0.1

_, rmse_mc = monte_carlo_prediccion(N_EPISODIOS, ALPHA)
_, rmse_td = td0_prediccion(N_EPISODIOS, ALPHA)

plt.figure(figsize=(8, 5))
plt.plot(rmse_mc, label="Monte Carlo (primera visita)")
plt.plot(rmse_td, label="TD(0)")
plt.xlabel("Episodios")
plt.ylabel("RMSE frente al valor real")
plt.title("Convergencia de MC vs TD(0) en el paseo aleatorio (α = 0.1)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Eseguendolo vedrai, in genere, che TD(0) riduce l'errore più in fretta di MC durante i primi episodi, grazie alla sua varianza più bassa, anche se entrambi finiscono per avvicinarsi allo stesso risultato all'aumentare degli episodi. È la teoria di distorsione e varianza che hai visto prima, resa in forma di grafico.

## Punti chiave

- La predizione senza modello significa stimare Vπ(s) solo a partire dall'esperienza, senza conoscere le probabilità di transizione né la funzione di ricompensa
- Monte Carlo calcola la media di ritorni completi di episodi; non fa bootstrapping e richiede che l'episodio termini
- La prima visita (*first-visit*) usa solo la prima occorrenza di uno stato per episodio; ogni visita (*every-visit*) usa tutte le occorrenze
- TD(0) aggiorna dopo ogni passo usando la ricompensa osservata più la tua stima dello stato successivo: questo è il bootstrapping
- MC è non distorto ma ha varianza alta; TD è distorto all'inizio ma ha varianza bassa, e in entrambi i casi converge con sufficiente esperienza
- La passeggiata casuale è un ambiente minimo perfetto per misurare in modo quantitativo questa differenza, con l'errore RMSE rispetto al valore reale

## Mettilo in pratica

Prendi il codice precedente e fai questi due esperimenti da solo:

- **Cambia il tasso di apprendimento**: prova con α = 0.01, α = 0.05 e α = 0.15 per entrambi i metodi. Osserva come un α troppo alto acceleri l'apprendimento iniziale ma lasci più rumore residuo, mentre un α troppo basso converge in modo più regolare ma molto più lentamente
- **Aumenta il numero di episodi** a 500 o 1.000 e verifica che, con esperienza sufficiente, la curva di TD(0) e quella di Monte Carlo finiscano per avvicinarsi allo stesso errore minimo

Esegui entrambi gli esperimenti e annota in quale punto esatto le due curve si incrociano! Questa osservazione è il modo migliore per interiorizzare il bilanciamento tra distorsione e varianza prima di proseguire.

---

Ora sai stimare il valore di una politica senza conoscere il modello. Nel capitolo 7 farai il passo successivo: usare la stessa idea delle Differenze Temporali non solo per valutare la politica, ma per migliorarla, con due algoritmi di controllo classici — SARSA e Q-learning — sull'ambiente di Cliff Walking.
