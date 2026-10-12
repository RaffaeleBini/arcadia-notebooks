# Capitolo 2. Introduzione all'apprendimento per rinforzo

Immagina di insegnare a un robot a muoversi in un magazzino, a un algoritmo a decidere quando comprare e vendere azioni, o a un programma a giocare a scacchi meglio di qualsiasi essere umano. In tutti e tre i casi, nessuno consegna al sistema una lista di "mosse corrette" per ogni situazione. Il sistema prova, osserva cosa succede e corregge il proprio comportamento in base al risultato. Questa forma di apprendimento per tentativi ed errori ha un nome preciso: apprendimento per rinforzo (RL, dall'inglese *Reinforcement Learning*), uno dei rami più promettenti dell'intelligenza artificiale (IA).

L'RL sta già alla base di risultati molto concreti. È la tecnica che ha permesso ad AlphaGo di sconfiggere campioni umani di Go, quella che addestra robot a manipolare oggetti senza danneggiarli e quella che mette a punto strategie di negoziazione automatica nei mercati finanziari. È anche alla base delle auto a guida autonoma e di molti sistemi di raccomandazione. Se lavori nella digitalizzazione delle aziende, prima o poi ti imbatterai in un caso d'uso in cui l'RL offre qualcosa che altri approcci dell'IA non riescono a ottenere: la capacità di apprendere una strategia completa a partire dalla propria esperienza, senza che nessuno debba dire al sistema quale sia la risposta corretta in ogni momento.

In questo capitolo costruirai la base concettuale dell'RL da zero. Non serve esperienza pregressa di machine learning né di algebra lineare: solo curiosità e voglia di sperimentare. Alla fine scriverai ed eseguirai il tuo primo codice in Python per vedere questi concetti in azione.

## Che cos'è l'apprendimento per rinforzo?

Pensa a come addestri un animale domestico. Non gli consegni un manuale con l'elenco dei movimenti corretti per sedersi: gli dai un premio quando fa ciò che vuoi e speri che, poco a poco, associ l'azione alla ricompensa. Così funziona l'RL. Il sistema, che chiamiamo **agente**, interagisce con un ambiente, prova azioni e riceve un segnale che gli indica se quell'azione è stata buona o cattiva. Col tempo, impara a scegliere le azioni che massimizzano quel segnale.

Questo modo di apprendere è molto diverso dall'apprendimento supervisionato, un altro approccio diffuso nell'IA. Nell'apprendimento supervisionato addestri un modello con un insieme di dati in cui ogni esempio ha già la sua risposta corretta (per esempio, migliaia di foto di gatti e cani già etichettate). Nell'RL non esiste un insieme di risposte corrette: l'agente impara unicamente attraverso l'interazione e per tentativi ed errori. Nessuno gli dice "questa era l'azione giusta"; l'agente deve scoprirlo da solo, provando e osservando le conseguenze.

## I tre protagonisti: agente, ambiente e ricompensa

Ogni problema di RL può essere descritto con tre elementi e un ciclo che si ripete continuamente:

- **Agente**: chi prende le decisioni. Può essere un robot, un programma di trading o un personaggio di un videogioco
- **Ambiente**: tutto ciò con cui l'agente interagisce. Può essere un magazzino fisico, un mercato finanziario o una scacchiera
- **Ricompensa**: un numero che l'ambiente restituisce all'agente dopo ogni azione, e che indica se quell'azione è stata vantaggiosa o dannosa

Il ciclo funziona così: l'agente osserva lo **stato** attuale dell'ambiente, sceglie un'**azione**, l'ambiente passa a un nuovo stato e fornisce una **ricompensa**. L'agente ripete questo ciclo più volte e usa l'esperienza accumulata per migliorare il proprio modo di decidere. L'intera sequenza di stati, azioni e ricompense che vive un agente si chiama **episodio** o **traiettoria**.

Un esempio del programma illustra bene l'importanza di progettare bene la ricompensa: un robot che riceve una ricompensa positiva quando si avvicina a un obiettivo e una negativa quando urta contro un ostacolo. Se progetti male quella ricompensa, l'agente può trovare "scorciatoie" indesiderate: imparerà esattamente ciò per cui gli stai pagando, non necessariamente ciò che volevi insegnargli. Questa sfumatura è fondamentale quando progetti soluzioni di RL per un caso reale: la ricompensa che definisci plasma completamente il comportamento finale del sistema.

Conviene anche distinguere tra ricompensa immediata e ricompensa a lungo termine. A volte è opportuno rinunciare a un guadagno immediato in cambio di un beneficio maggiore più avanti, proprio come nella vita reale. Per rappresentare questa idea, l'RL usa un **fattore di sconto** (indicato con la lettera greca gamma, γ), un valore compreso tra zero e uno che indica quanto peso dai alle ricompense future rispetto a quelle immediate. Un valore di γ vicino a uno dà al futuro quasi la stessa importanza del presente; un valore vicino a zero fa sì che l'agente pensi solo al breve termine.

## La politica: come decide l'agente

La strategia che l'agente segue per scegliere le azioni si chiama **politica**, e di solito si indica con la lettera greca pi (π). Esistono due tipi:

- **Politica deterministica**: per ogni stato, sceglie sempre la stessa azione. È come memorizzare "a questo incrocio, gira sempre a sinistra"
- **Politica stocastica**: per ogni stato, assegna una probabilità a ciascuna azione possibile, invece di scegliere sempre la stessa. È come dire "a questo incrocio giro a sinistra otto volte su dieci, e a destra le altre due"

Pensa a un labirinto. Una politica deterministica che gira sempre a sinistra a ogni incrocio può funzionare bene in un labirinto fisso, ma si blocca se compare un ostacolo imprevisto. Una politica stocastica, variando ogni tanto la propria scelta, ha più possibilità di scoprire scorciatoie nuove o di adattarsi quando l'ambiente cambia. Questa capacità di "provare cose diverse ogni tanto" si collega direttamente a un dilemma centrale dell'RL che vedrai nel prossimo capitolo: l'equilibrio tra esplorazione e sfruttamento.

Lo scopo ultimo dell'apprendimento è trovare la **politica ottimale** (π\*): la strategia che massimizza la ricompensa cumulativa attesa a lungo termine. Questa è la meta finale di qualsiasi algoritmo di RL, anche se il percorso per arrivarci cambia a seconda dell'algoritmo che usi.

## Il formalismo alla base dell'ambiente: il PDM

Per poter programmare e analizzare un problema di RL, l'ambiente viene descritto con un modello matematico chiamato Processo di Decisione di Markov (PDM). Non preoccuparti della profondità matematica adesso: la vedrai con calma nel capitolo quattro. Per ora basta che tu conosca le sue quattro componenti:

- **S**: l'insieme di tutti gli stati possibili dell'ambiente
- **A**: l'insieme di tutte le azioni che l'agente può eseguire
- **P**: la funzione di transizione, che indica la probabilità di passare da uno stato a un altro eseguendo una determinata azione
- **R**: la funzione di ricompensa, che indica quale ricompensa riceve l'agente eseguendo un'azione in un dato stato

Un'idea intuitiva importante alla base del PDM è la cosiddetta "proprietà di Markov": per decidere cosa succede dopo, conta solo lo stato attuale, non tutta la storia degli stati precedenti. È come giocare una partita a scacchi: per decidere la prossima mossa, l'unica cosa che conta è la posizione attuale della scacchiera, non la sequenza esatta di mosse che ci ha portato lì.

## Una prima idea di Q-learning

Per decidere quale azione intraprendere, è molto utile sapere "quanta ricompensa mi aspetto di ottenere se faccio questo". Tale stima si chiama **valore di azione** o **valore Q**, e uno degli algoritmi più noti per impararlo è il **Q-learning**. L'idea è semplice da esprimere in modo intuitivo: l'agente tiene una tabella con un valore stimato per ogni combinazione di stato e azione, e ogni volta che prova un'azione e osserva la ricompensa reale, corregge un po' quella stima per avvicinarla a ciò che è realmente accaduto. Con ripetizioni sufficienti, quei valori stimati finiscono per riflettere con buona fedeltà quanta ricompensa porta ciascuna azione in ciascuno stato, e l'agente può usarli per scegliere sempre l'opzione migliore disponibile.

Il Q-learning è un algoritmo "senza modello" (model-free): non ha bisogno di conoscere in anticipo come funziona l'ambiente, impara solo dall'esperienza diretta. Esiste anche l'approccio opposto, "basato su modello" (model-based), in cui l'agente costruisce prima una rappresentazione di come funziona l'ambiente e poi pianifica usando quella rappresentazione, un po' come quando un giocatore di scacchi o di Go simula mentalmente diverse mosse future prima di muovere un pezzo. Tornerai su questa distinzione con più dettagli nei capitoli successivi.

## Il dilemma esplorazione/sfruttamento, in una frase

Per apprendere bene, l'agente deve provare azioni nuove (**esplorare**) per scoprire se nascondono una ricompensa migliore di quella che già conosce, ma deve anche sfruttare ciò che sa già (**sfruttare**) per non sprecare occasioni. Troppa esplorazione fa perdere tempo all'agente a provare opzioni mediocri; troppo sfruttamento può bloccarlo su una soluzione accettabile senza scoprirne un'altra molto migliore. Pensa a un aspirapolvere robotico: se pulisce solo le zone che sa già essere sporche, non scopre altre zone sporche che non conosce; se esplora solo zone nuove, perde tempo senza pulire ciò che sa che richiede attenzione. Questo dilemma è così importante che gli dedichiamo per intero il capitolo tre, quindi per ora basta che tu lo tenga presente.

## Manos a la obra: il tuo primo ambiente in Python

Prima di imparare qualsiasi algoritmo, conviene vedere i concetti di agente, ambiente, politica e ricompensa funzionare in codice reale. Il seguente esempio crea un ambiente minimo (un corridoio di cinque celle) e un agente che si muove in modo completamente casuale. Per ora l'agente non impara nulla: l'obiettivo è che tu veda il ciclo agente-ambiente in azione, passo dopo passo.

**Esegui questa cella in Google Colab** (serve solo `numpy`, che è già preinstallato):

```python
import numpy as np

# --- EL ENTORNO -------------------------------------------------------
# Un pasillo de 5 celdas, numeradas de 0 a 4.
# El agente empieza siempre en la celda 0 (izquierda) y el objetivo
# está en la celda 4 (derecha). Puede moverse a la izquierda (0)
# o a la derecha (1).
#
#   [0] [1] [2] [3] [4=META]

class PasilloEntorno:
    def __init__(self, longitud=5):
        self.longitud = longitud
        self.meta = longitud - 1
        self.estado = 0  # posición inicial del agente

    def reset(self):
        """Reinicia el entorno y devuelve el estado inicial."""
        self.estado = 0
        return self.estado

    def step(self, accion):
        """
        Ejecuta una acción en el entorno.
        accion: 0 = mover a la izquierda, 1 = mover a la derecha
        Devuelve: (nuevo_estado, recompensa, terminado)
        """
        if accion == 1:
            self.estado = min(self.estado + 1, self.longitud - 1)
        elif accion == 0:
            self.estado = max(self.estado - 1, 0)

        terminado = self.estado == self.meta

        # Diseño de la recompensa: cada paso "cuesta" un poco (-1),
        # para animar al agente a llegar a la meta cuanto antes.
        # Al llegar a la meta, recompensa extra de +10.
        if terminado:
            recompensa = 10
        else:
            recompensa = -1

        return self.estado, recompensa, terminado


# --- LA POLÍTICA (AGENTE ALEATORIO) ------------------------------------
# Una política estocástica muy simple: en cada estado, elige
# la acción al azar, con la misma probabilidad para cada una.
# No usa ninguna información del estado todavía; en capítulos
# siguientes sustituirás esta política por una que sí aprende.

def politica_aleatoria(estado, num_acciones=2):
    return np.random.randint(num_acciones)


# --- EL BUCLE AGENTE-ENTORNO --------------------------------------------
# Aquí se ve el ciclo central del RL: observar estado, elegir acción,
# recibir nuevo estado y recompensa, y repetir hasta el final del episodio.

def ejecutar_episodio(entorno, max_pasos=20, semilla=None):
    if semilla is not None:
        np.random.seed(semilla)

    estado = entorno.reset()
    recompensa_acumulada = 0
    nombres_acciones = {0: "izquierda", 1: "derecha"}

    print(f"Estado inicial: {estado}")

    for paso in range(1, max_pasos + 1):
        accion = politica_aleatoria(estado)
        nuevo_estado, recompensa, terminado = entorno.step(accion)
        recompensa_acumulada += recompensa

        print(
            f"Paso {paso}: estado={estado} -> "
            f"acción={nombres_acciones[accion]} -> "
            f"nuevo_estado={nuevo_estado}, recompensa={recompensa}"
        )

        estado = nuevo_estado

        if terminado:
            print(f"\n¡Meta alcanzada en el paso {paso}!")
            break

    print(f"Recompensa acumulada del episodio: {recompensa_acumulada}")
    return recompensa_acumulada


# Ejecuta un episodio completo con una semilla fija, para que el
# resultado sea reproducible cada vez que ejecutes esta celda.
ejecutar_episodio(PasilloEntorno(), max_pasos=20, semilla=42)
```

Al eseguire questa cella vedrai stampata la sequenza completa di stati, azioni e ricompense di un episodio. Poiché la politica è casuale, l'agente a volte avanza, a volte indietreggia, e può impiegare parecchi passi per arrivare alla meta, o addirittura non arrivarci entro il limite di passi. È esattamente ciò che ci si aspetta: senza alcun tipo di apprendimento, l'agente non ha alcun motivo per preferire "destra" a "sinistra".

**Cambia questo valore e osserva cosa succede**: sostituisci `semilla=42` con un altro numero ed esegui di nuovo la cella. Vedrai una traiettoria diversa, perché cambia la sequenza di azioni casuali. **Prova anche** a portare `max_pasos` a 50 oppure a ridurre la lunghezza del corridoio a 3 celle, e confronta quanti passi impiega l'agente ad arrivare alla meta in ciascun caso.

## Riepilogo dei punti chiave

- L'apprendimento per rinforzo (RL) insegna a un agente a decidere per tentativi ed errori, senza un insieme di risposte corrette come nell'apprendimento supervisionato
- Il ciclo centrale dell'RL collega tre elementi: agente, ambiente e ricompensa, in un ciclo di stato → azione → nuovo stato → ricompensa
- La politica (π) è la strategia che l'agente usa per scegliere le azioni; può essere deterministica o stocastica
- Il Processo di Decisione di Markov (PDM) formalizza l'ambiente con quattro elementi: stati (S), azioni (A), transizioni (P) e ricompense (R)
- Il fattore di sconto (γ) regola quanto peso dai alle ricompense future rispetto a quelle immediate
- Il Q-learning stima, per ogni combinazione di stato e azione, quanta ricompensa ci si può aspettare, e corregge quella stima con l'esperienza
- La progettazione della ricompensa è fondamentale: una ricompensa mal pensata può insegnare all'agente il comportamento sbagliato
- Esplorare e sfruttare sono due necessità in tensione costante; trovare l'equilibrio giusto è una delle grandi sfide dell'RL

## Mettilo in pratica

Prendi il codice precedente e modificalo per verificare da solo come cambia il comportamento dell'agente:

1. **Aggiungi una terza azione**, per esempio "restare fermi" (azione 2), e adatta `politica_aleatoria` e `step` per gestirla
2. **Cambia il progetto della ricompensa**: invece di -1 per passo, prova con 0 per passo e osserva se l'agente ha ancora qualche incentivo a raggiungere la meta in fretta
3. **Esegui dieci episodi di seguito** (usa un ciclo `for` attorno a `ejecutar_episodio`) senza fissare la semente, e annota quanti passi impiega l'agente casuale ad arrivare alla meta in ciascuno. Questa variabilità è proprio ciò che un algoritmo di apprendimento, come il Q-learning, è progettato per ridurre col tempo

Conserva i tuoi risultati: li riutilizzerai come punto di confronto quando, nei prossimi capitoli, sostituirai la politica casuale con una che impara davvero.

---

Nel capitolo tre approfondiremo il dilemma esplorazione/sfruttamento che hai solo intravisto qui, prendendo come punto di partenza il classico problema dei bandit multi-braccio (*multi-armed bandits*).
