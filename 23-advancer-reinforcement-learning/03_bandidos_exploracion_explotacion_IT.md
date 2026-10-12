# Capitolo 3. Il dilemma dei bandit: esplorazione e sfruttamento

Immagina di trasferirti in una città nuova. Hai un ristorante che ti piace a cinque minuti da casa: sai esattamente cosa mangerai e sai che ti piacerà. Ma a due isolati c'è un locale che non hai mai provato. Torni all'opzione sicura o rischi di scoprire qualcosa di meglio?

Questa domanda, così quotidiana, è lo stesso problema che affronta un'équipe medica quando sceglie tra una terapia consolidata e una sperimentale, o una piattaforma digitale quando decide quale pubblicità mostrarti. In tutti e tre i casi esiste una tensione tra due impulsi: sfruttare ciò che sai già funzionare ed esplorare per capire se c'è qualcosa di meglio. Nell'apprendimento per rinforzo (RL, dall'inglese *Reinforcement Learning*) questa tensione si chiama dilemma tra esplorazione e sfruttamento, ed è uno degli aspetti più importanti dell'intero campo.

Nel capitolo precedente hai conosciuto i mattoni fondamentali dell'RL: un agente che agisce, un ambiente che risponde, una politica che decide cosa fare e una ricompensa che misura se quella decisione è stata buona. In questo capitolo vedrai come un agente decide, passo dopo passo, tra ripetere ciò che conosce e provare qualcosa di nuovo. Per farlo in modo semplice, lavorerai con la versione più semplice possibile di un problema decisionale: il bandit multi-braccio.

## Il problema del bandit multi-braccio

In un casinò, una slot machine è nota in inglese come "one-armed bandit" (bandito a un braccio), perché ha una sola leva e ti "deruba" il denaro poco a poco. Ora immagina una fila di diverse slot machine, una accanto all'altra. Ognuna paga premi diversi: alcune pagano molto e spesso; altre quasi mai pagano qualcosa. Il problema è che tu non sai quale sia quale. L'unico modo per scoprirlo è tirare le leve e osservare cosa succede. Questo scenario completo si chiama bandit multi-braccio (*multi-armed bandit*) e dà il nome a questa famiglia di problemi.

Ogni leva rappresenta un'azione possibile. Ogni azione ha un valore reale: la ricompensa media che otterresti se la scegliessi infinite volte. Quel valore esiste, ma tu non lo conosci all'inizio; puoi soltanto stimarlo a partire da ciò che osservi ogni volta che tiri una leva. Inoltre, in questa versione semplificata del problema, ogni macchina distribuisce sempre i premi secondo la stessa distribuzione fissa: non cambia nel tempo né reagisce alle tue decisioni precedenti. Per questo si dice che è un problema "senza stato": l'unica cosa che cambia mentre giochi non è l'ambiente, ma la tua conoscenza di esso.

Un esempio classico aiuta a capire perché questo sia più difficile di quanto sembri. Pensa a un topo da laboratorio che può scegliere tra due leve, una su sfondo nero e una su sfondo bianco. Un giorno la leva nera dà formaggio; un altro giorno dà una scossa. Se il topo si guidasse solo dall'ultimo risultato, o da una sola esperienza negativa, potrebbe abbandonare per sempre una leva che, in media, è la scelta migliore. Lo stesso accade a qualsiasi agente che decide in modo troppo affrettato: una cattiva serie iniziale può convincerlo a evitare la migliore azione possibile per il resto del gioco.

## Stimare il valore di ogni azione

Se non conosci il valore reale di una leva, la cosa ragionevole è stimarlo a partire da ciò che hai osservato. Il modo più diretto è calcolare la media delle ricompense ricevute ogni volta che hai scelto quell'azione. Questa stima si chiama valore di azione, e viene aggiornata ogni volta che riprovi la stessa leva.

Non serve memorizzare l'intero storico delle ricompense per calcolare quella media. Puoi aggiornarla in modo incrementale: ogni volta che osservi una nuova ricompensa, correggi la tua stima precedente un po' verso quel nuovo valore, con un peso che diminuisce man mano che hai provato quell'azione più volte. All'inizio ogni nuova osservazione cambia molto la tua stima; col tempo, ogni osservazione aggiuntiva pesa di meno e la tua stima si stabilizza vicino al valore reale.

Questa stessa idea ammette una variante utile: invece di dare lo stesso peso a tutte le osservazioni passate, puoi usare un tasso di passo fisso (un "learning rate" costante) che dia più peso alle ricompense recenti. Questo è particolarmente utile quando l'ambiente cambia nel tempo, cioè quando non è più stazionario: una macchina che oggi paga bene domani potrebbe pagare peggio, e preferisci che la tua stima reagisca rapidamente a quel cambiamento invece di diluirlo tra centinaia di osservazioni antiche.

## Epsilon-greedy: esplorare un po', sfruttare molto

La strategia più semplice per risolvere il dilemma si chiama epsilon-greedy (ε-greedy). Funziona così: definisci un numero piccolo, épsilon (ε), compreso tra zero e uno. In ogni decisione, con probabilità ε scegli un'azione a caso (esplori); con probabilità 1−ε scegli l'azione che finora sembra migliore secondo le tue stime (sfrutti). Un dettaglio curioso: poiché la scelta casuale può anche coincidere con la migliore azione conosciuta, la probabilità reale di sfruttare è leggermente maggiore di 1−ε.

Torniamo all'esempio del ristorante. Se il tuo ε è 0,1, allora il 90% delle volte vai al tuo posto di fiducia e il 10% delle volte provi qualcosa di diverso, scelto a caso tra tutte le opzioni disponibili. Quel 10% è ciò che ti permette di scoprire, prima o poi, se esiste un ristorante migliore del tuo.

Un ε fisso garantisce che tu continui a esplorare per sempre, quindi col tempo finirai per provare tutte le azioni abbastanza volte da conoscerne il valore reale con buona precisione. Ma ha un costo: poiché non smetti mai di scegliere azioni a caso, continui a commettere la stessa percentuale di errori anche dopo aver imparato qual è la scelta migliore. Per questo un miglioramento comune è far diminuire ε nel tempo: esplorare molto all'inizio, quando sai poco, e sfruttare sempre di più man mano che la tua conoscenza migliora. Questa strategia, con le sue varianti, è stata usata per esempio nella Deep Q-Network (DQN), uno dei sistemi che ha imparato a giocare ai videogiochi Atari direttamente a partire dai pixel dello schermo.

## Upper Confidence Bound: l'ottimismo di fronte all'incertezza

Epsilon-greedy ha un limite: quando esplora, sceglie a caso tra tutte le azioni, senza distinguere tra una leva che conosci già bene e un'altra che hai appena provato. Upper Confidence Bound (UCB) risolve questo problema in modo più intelligente, applicando un'idea che di solito si riassume come "l'ottimismo di fronte all'incertezza".

La logica di UCB è la seguente: invece di guardare solo la tua stima attuale del valore di ogni azione, aggiungi a quella stima un "bonus di incertezza". Questo bonus è grande per le azioni che hai provato poche volte (perché non sai bene quanto valgano, quindi dai loro il beneficio del dubbio) e piccolo per quelle che hai provato molte volte (perché ormai hai una stima abbastanza affidabile). A ogni passo scegli l'azione la cui somma tra valore stimato e bonus è più alta.

All'inizio del processo, poiché non hai ancora provato nessuna leva, il bonus delle azioni non provate è enorme, quindi UCB ti obbliga a provare ogni azione almeno una volta prima di iniziare a confrontarle. Man mano che giochi, il bonus di ciascuna leva si riduce con il crescere dei tentativi, seguendo approssimativamente il ritmo della radice quadrata del numero di volte in cui l'hai provata. In questo modo l'algoritmo passa, in modo naturale e senza che tu debba regolare nulla a mano, da una fase con molta esplorazione a una fase concentrata sullo sfruttamento di ciò che sa già.

Esistono varianti di UCB per diversi scenari: UCB1-Tuned adatta il bonus in base a quanto variano le ricompense osservate; UCB Scontato dà più peso alle osservazioni recenti, utile in ambienti non stazionari; UCB Contestuale aggiunge informazioni supplementari sulla situazione (per esempio, le caratteristiche dell'utente in un sistema di raccomandazione) per personalizzare la decisione. Nel capitolo 9 vedrai con più dettaglio la teoria che spiega perché UCB funziona così bene e come si confronta matematicamente con le altre strategie.

## Thompson Sampling: un'alternativa bayesiana

Esiste una terza famiglia di strategie, di impostazione bayesiana, nota come Thompson Sampling (chiamato anche campionamento di Thompson o bandit bayesiano). Invece di conservare una sola stima per azione, mantiene una distribuzione di probabilità completa sul valore di ciascuna azione: una specie di "credenza" che riflette sia la tua migliore stima sia la tua incertezza su di essa.

L'idea, a ogni passo, è semplice: per ogni azione, ottieni un valore campione dalla sua distribuzione di credenza attuale e scegli l'azione con il campione più alto. Poi osservi la ricompensa reale e aggiorni la credenza di quell'azione. Quando un'azione ha una distribuzione molto incerta (perché l'hai provata poco), i suoi campioni variano molto e a volte escono valori alti, quindi ogni tanto la scegli: questa è esplorazione. Quando un'azione ha già una distribuzione molto consolidata intorno a un valore alto, i suoi campioni sono quasi sempre alti e coerenti: questa è sfruttamento. Tutto avviene in modo naturale, senza bisogno di un parametro come ε né di una formula esplicita di bonus.

Per ricompense binarie (successo o fallimento), Thompson Sampling usa di solito la distribuzione Beta, che si aggiorna in modo molto semplice: parti da una distribuzione neutra che non favorisce nessuna opzione e, ogni volta che osservi un successo, incrementi un parametro; ogni volta che osservi un fallimento, incrementi l'altro. Non serve alcun calcolo complicato.

## Esempio in Python: epsilon-greedy contro UCB

Il seguente codice simula un bandit a più bracci e confronta, in modo visivo, come si comportano epsilon-greedy e UCB lungo molti passi. Puoi copiarlo ed eseguirlo direttamente in Google Colab: serve solo `numpy` e `matplotlib`, che sono già preinstallati.

```python
import numpy as np
import matplotlib.pyplot as plt

# Fijamos una semilla para que el experimento sea reproducible
np.random.seed(42)

# --- Configuración del bandido multi-brazo ---
n_brazos = 10          # número de "palancas" o acciones disponibles
n_pasos = 1000          # cuántas veces va a jugar cada estrategia
n_repeticiones = 200    # repetimos el experimento varias veces para suavizar el azar

# El valor verdadero de cada brazo se genera al azar (el agente NO lo conoce)
# Estos valores representan la recompensa media real de cada acción
valores_reales = np.random.normal(loc=0.0, scale=1.0, size=n_brazos)


def tirar_de_la_palanca(brazo):
    """Simula el resultado de elegir un brazo: devuelve una recompensa con ruido
    alrededor del valor verdadero de ese brazo (como en una tragaperras real)."""
    return np.random.normal(loc=valores_reales[brazo], scale=1.0)


def ejecutar_epsilon_greedy(epsilon, n_pasos):
    """Estrategia epsilon-greedy: explora al azar con probabilidad epsilon,
    y el resto de las veces elige la acción con mejor valor estimado."""
    valores_estimados = np.zeros(n_brazos)   # Q_t(a): estimación actual de cada brazo
    conteo_intentos = np.zeros(n_brazos)     # cuántas veces se ha probado cada brazo
    recompensas = np.zeros(n_pasos)

    for t in range(n_pasos):
        if np.random.random() < epsilon:
            # Exploración: elige un brazo al azar
            brazo = np.random.randint(n_brazos)
        else:
            # Explotación: elige el brazo con mejor valor estimado hasta ahora
            brazo = np.argmax(valores_estimados)

        recompensa = tirar_de_la_palanca(brazo)
        conteo_intentos[brazo] += 1

        # Actualización incremental de la media muestral (ver sección "Estimando
        # el valor de cada acción"): Q_{n+1} = Q_n + (1/n) * (R_n - Q_n)
        paso = 1.0 / conteo_intentos[brazo]
        valores_estimados[brazo] += paso * (recompensa - valores_estimados[brazo])

        recompensas[t] = recompensa

    return recompensas


def ejecutar_ucb(c, n_pasos):
    """Estrategia UCB: combina el valor estimado con un bono de incertidumbre
    que favorece a los brazos poco probados."""
    valores_estimados = np.zeros(n_brazos)
    conteo_intentos = np.zeros(n_brazos)
    recompensas = np.zeros(n_pasos)

    for t in range(1, n_pasos + 1):
        # Si algún brazo aún no se ha probado, se elige primero (bono "infinito")
        brazos_sin_probar = np.where(conteo_intentos == 0)[0]
        if len(brazos_sin_probar) > 0:
            brazo = brazos_sin_probar[0]
        else:
            # UCB: valor estimado + bono de exploración que crece con log(t)
            # y se reduce con la raíz cuadrada del número de intentos del brazo
            bono = c * np.sqrt(np.log(t) / conteo_intentos)
            brazo = np.argmax(valores_estimados + bono)

        recompensa = tirar_de_la_palanca(brazo)
        conteo_intentos[brazo] += 1
        paso = 1.0 / conteo_intentos[brazo]
        valores_estimados[brazo] += paso * (recompensa - valores_estimados[brazo])

        recompensas[t - 1] = recompensa

    return recompensas


# --- Ejecutamos ambas estrategias varias veces y promediamos ---
recompensas_epsilon = np.zeros(n_pasos)
recompensas_ucb = np.zeros(n_pasos)

for _ in range(n_repeticiones):
    # Re-generamos los valores reales en cada repetición para que el promedio
    # no dependa de una única configuración de brazos
    valores_reales = np.random.normal(loc=0.0, scale=1.0, size=n_brazos)
    recompensas_epsilon += ejecutar_epsilon_greedy(epsilon=0.1, n_pasos=n_pasos)
    recompensas_ucb += ejecutar_ucb(c=2.0, n_pasos=n_pasos)

recompensas_epsilon /= n_repeticiones
recompensas_ucb /= n_repeticiones

# Convertimos la recompensa por paso en recompensa media acumulada
recompensa_media_acumulada_epsilon = np.cumsum(recompensas_epsilon) / np.arange(1, n_pasos + 1)
recompensa_media_acumulada_ucb = np.cumsum(recompensas_ucb) / np.arange(1, n_pasos + 1)

# --- Graficamos la comparación ---
plt.figure(figsize=(10, 5))
plt.plot(recompensa_media_acumulada_epsilon, label="epsilon-greedy (ε = 0.1)")
plt.plot(recompensa_media_acumulada_ucb, label="UCB (c = 2.0)")
plt.xlabel("Paso de tiempo")
plt.ylabel("Recompensa media acumulada")
plt.title("Comparación epsilon-greedy vs. UCB en un bandido de 10 brazos")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Eseguendolo vedrai che entrambe le curve migliorano nel tempo, ma UCB tende a superare epsilon-greedy: poiché distribuisce la sua esplorazione in modo più intelligente (provando per primo ciò che conosce meno), impiega meno tempo a trovare il braccio ottimale e spreca meno tentativi su bracci chiaramente cattivi.

## Punti chiave

- Il dilemma tra esplorazione e sfruttamento compare in ogni decisione presa in condizioni di incertezza, dentro e fuori dall'RL
- Il bandit multi-braccio è la versione più semplice di questo problema: più azioni, ciascuna con un valore reale sconosciuto e fisso nel tempo
- Il valore di un'azione si stima con la media delle ricompense osservate, e può essere aggiornato in modo incrementale senza memorizzare tutto lo storico
- Epsilon-greedy esplora a caso con probabilità ε e sfrutta il resto del tempo; un ε decrescente di solito funziona meglio di uno fisso
- UCB dirige l'esplorazione verso le azioni più incerte, sommando un bonus che si riduce man mano che un'azione viene provata di più
- Thompson Sampling usa distribuzioni di probabilità complete e bilancia esplorazione e sfruttamento in modo naturale, senza parametri espliciti come ε

## Mettilo in pratica

Prendi il codice di questo capitolo e modificalo tu stesso o tu stessa:

- **Cambia il valore di ε** in `ejecutar_epsilon_greedy` (prova con 0,01, 0,1 e 0,5) e osserva come influisce sulla curva della recompensa media cumulativa
- **Cambia il numero di bracci** `n_brazos` (per esempio, a 5 o a 20) e verifica se UCB mantiene il suo vantaggio su epsilon-greedy
- **Prova diversi valori di `c`** in UCB e annota a partire da quale valore la strategia inizia a esplorare troppo

Esegui il notebook, cambia i parametri e condividi con il tuo team quale combinazione ha trovato per prima il braccio migliore!

---

Nel capitolo 4 lascerai il mondo "senza stato" dei bandit ed entrerai nei Processi di Decisione di Markov, dove le azioni dell'agente cambiano davvero lo stato dell'ambiente.
