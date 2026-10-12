# Capitolo 5. Ottimizzazione I: come impara una rete

## Perché questo tema è importante

Ormai sai addestrare una rete semplice in PyTorch: definisci un modello, calcoli una perdita e chiami `backward()` per ottenere i gradienti. Ma che cosa succede esattamente dopo? Come trasforma la rete quei gradienti in un modello che funziona? La risposta è l'ottimizzazione, il motore che sta dietro a tutto l'apprendimento profondo (*deep learning*, DL).

Ogni volta che addestri una rete neurale, risolvi un problema di ottimizzazione: regolare milioni di parametri affinché una funzione di perdita sia il più piccola possibile. Se questo processo fallisce, non importa quanto sia ingegnosa la tua architettura né quanto siano puliti i tuoi dati: la rete non impara. E se il processo funziona male —troppo lento, o instabile—, perdi ore o giorni di calcolo senza ottenere nulla di utile in cambio.

Questo capitolo è la prima metà della Lezione 4 del corso, "Ottimizzazione per il *machine learning*" (apprendimento automatico). Qui ti concentri sull'algoritmo più basilare e più importante di tutti: la discesa del gradiente (*gradient descent*, GD). Capirai perché il tasso di apprendimento (*learning rate*) è l'iperparametro che dà più grattacapi a chiunque addestri modelli, e perché la forma della superficie di perdita —la sua curvatura— determina se l'addestramento sarà un compito semplice o un incubo. Nel prossimo capitolo vedrai come il momento (*momentum*) e i metodi del secondo ordine risolvono alcuni dei problemi che oggi imparerai a diagnosticare.

**Presta attenzione**: capire bene questo capitolo ti dà la base per diagnosticare perché un addestramento diverge, perché si blocca o perché avanza a passo di tartaruga, cosa che ti capiterà sicuramente nella tua carriera di professionista dell'intelligenza artificiale (IA).

## Spiegazione concettuale con analogie

Immagina di trovarti sulla cima di una montagna, avvolto da una nebbia fitta che ti permette di vedere solo il terreno sotto i tuoi piedi. Il tuo obiettivo è raggiungere la valle, il punto più basso possibile. Non hai una mappa, quindi puoi usare una sola strategia: in ogni punto in cui ti trovi, senti con i piedi verso dove il terreno scende più ripidamente e fai un passo in quella direzione. Ripeti questo passo dopo passo, più volte, finché ti avvicini al fondo della valle.

Questa è, in sostanza, l'idea della discesa del gradiente. La "montagna" è la funzione obiettivo o funzione di perdita: una superficie che misura quanto sbaglia il tuo modello per ogni combinazione di parametri. La "posizione" in cui ti trovi sono i valori attuali dei parametri della rete (i pesi). E il "senso della pendenza" è il gradiente, un vettore che punta nella direzione di maggior crescita della funzione. Poiché vuoi scendere, non salire, ti muovi nella direzione opposta: il gradiente negativo.

Ora, in questa analogia c'è una decisione cruciale: la dimensione del passo. È esattamente il tasso di apprendimento.

- Se fai **passi troppo grandi**, rischi di scavalcare la valle e atterrare sul versante opposto, magari anche più in alto di dove eri partito. Se ripeti questo più volte, invece di avvicinarti al fondo, ti allontani sempre di più: il processo diverge.
- Se fai **passi troppo piccoli**, avanzi con assoluta sicurezza, ma ci metti un'eternità ad arrivare in fondo. L'addestramento converge, ma così lentamente da risultare impraticabile.
- Se il passo è quello giusto, avanzi in modo costante e ragionevolmente rapido verso il minimo.

Ora aggiungi un'ulteriore complicazione all'analogia: immagina che la valle non abbia la forma di una conca simmetrica, ma quella di un canyon stretto e allungato, con pareti molto ripide ai lati e un fondo quasi piatto che scende dolcemente in una sola direzione. Se cammini con passi grandi, rimbalzerai da una parete all'altra del canyon senza avanzare quasi per nulla lungo il fondo: oscilli da un lato all'altro invece di progredire verso l'uscita. Se riduci il passo per evitare il rimbalzo, avanzerai con sicurezza lungo le pareti, ma ti muoverai con esasperante lentezza lungo il fondo piatto. Non esiste un unico tasso di apprendimento che risolva bene entrambe le direzioni insieme.

Questa forma di "canyon stretto" non è un capriccio matematico: è esattamente ciò che succede nella maggior parte delle superfici di perdita reali di una rete neurale. Alcune direzioni dello spazio dei parametri hanno molta curvatura (pareti ripide) e altre quasi nessuna (fondo piatto). A quel rapporto tra la curvatura massima e quella minima si dà il nome di numero di condizionamento, ed è la ragione matematica per cui la discesa del gradiente di base, pur funzionando, può essere così lenta e delicata da regolare.

## Sviluppo teorico (fedele alla fonte, con profondità reale)

### La funzione obiettivo

Nell'apprendimento automatico, ottimizzare significa regolare i parametri di un modello, che rappresentiamo come un vettore $\theta \in \mathbb{R}^d$, per minimizzare una funzione obiettivo (o obiettivo di perdita) $h(\theta)$. Questa funzione misura gli errori che commette il modello: nella classificazione può essere l'errore di previsione; nell'apprendimento per rinforzo, la ricompensa negativa.

L'obiettivo di addestramento tipico di una rete neurale si esprime come una somma (o una media) sugli $N$ esempi del set di addestramento:

$$h(\theta) = \frac{1}{N}\sum_{i=1}^{N} L\big(f(x_i,\theta),\, y_i\big)$$

dove $f$ è la rete neurale, $x_i$ sono gli input, $\theta$ i parametri, $y_i$ i valori veri e $L$ la funzione di perdita che confronta la previsione $f(x_i,\theta)$ con il valore reale $y_i$.

### La regola di aggiornamento della discesa del gradiente

La discesa del gradiente aggiorna i parametri spostandoli nella direzione del gradiente negativo della funzione obiettivo. La sua regola di aggiornamento è:

$$\theta_{k+1} = \theta_k - \alpha \, \nabla h(\theta_k)$$

Qui, $\theta_k$ sono i parametri al passo $k$, $\alpha$ è il tasso di apprendimento (chiamato anche dimensione del passo) e $\nabla h(\theta_k)$ è il gradiente della funzione obiettivo rispetto ai parametri, valutato in $\theta_k$. Il gradiente indica la direzione di massima crescita della funzione; sottraendolo, ogni passo riduce (o tenta di ridurre) il valore dell'obiettivo.

### Perché il gradiente è la direzione di discesa più ripida

L'intuizione alla base di questa regola si appoggia all'approssimazione di Taylor del primo ordine. Se $h$ è sufficientemente regolare, il valore della funzione vicino al punto attuale $\theta$ si può approssimare con uno sviluppo lineare:

$$h(\theta + d) \approx h(\theta) + \nabla h(\theta)^{\top} d$$

dove $d$ è un piccolo spostamento nello spazio dei parametri. Questa approssimazione lineare è fedele al comportamento reale di $h$ solo quando $d$ è piccolo; più grande è lo spostamento, meno affidabile è.

La discesa del gradiente minimizza questa approssimazione lineare all'interno di una regione limitata, per esempio una sfera di raggio $\alpha$ attorno a $\theta$. La soluzione di quel sottoproblema, cioè la direzione che minimizza $\nabla h(\theta)^{\top} d$ con il vincolo $\|d\| \le \alpha$, è precisamente $d = -\alpha \, \nabla h(\theta) / \|\nabla h(\theta)\|$: muoversi in direzione opposta al gradiente. Da qui il nome "discesa più ripida" (*steepest descent*): all'interno della regione di fiducia in cui l'approssimazione lineare è valida, nessun altro passo riduce la funzione più di così.

Questo ragionamento spiega anche perché il tasso di apprendimento non può essere arbitrariamente grande: più grande è il passo, più ti allontani dalla zona in cui l'approssimazione lineare è corretta, e il passo può finire per aumentare il valore reale di $h$ invece di ridurlo, anche se l'approssimazione lineare prevedeva una discesa.

### Regolarità, costante di Lipschitz e convessità forte

Per poter dire qualcosa di rigoroso su quando e come converge la discesa del gradiente, la teoria ha bisogno di alcune ipotesi sulla "forma" di $h$.

La prima ipotesi abituale è che $h$ abbia derivate continue di tipo lipschitziano, cioè che $h$ sia "regolarmente lipschitziana". Formalmente, esiste una costante $L$ tale che:

$$\|\nabla h(\theta_1) - \nabla h(\theta_2)\| \le L \, \|\theta_1 - \theta_2\| \quad \text{per ogni } \theta_1, \theta_2$$

Questa condizione dice che il gradiente non può cambiare in modo troppo brusco quando ti sposti un poco nello spazio dei parametri. In termini geometrici, $L$ è un limite superiore della curvatura di $h$: nessuna direzione della superficie di perdita si curva più di quanto consenta $L$. Per questo a $L$ si dà anche il nome di costante di Lipschitz.

La seconda ipotesi, più forte e non sempre necessaria, è la convessità forte, con parametro $\mu$. Questa condizione impone un limite inferiore alla curvatura: la superficie non può essere "troppo piatta" in nessuna direzione rilevante. Combinate, $L$ e $\mu$ limitano la curvatura di $h$ rispettivamente dall'alto e dal basso.

Nel caso più abituale dell'addestramento di reti neurali, la funzione obiettivo in generale non è convessa, ma vicino a un minimo locale la sua curvatura può essere limitata dal basso tramite un termine quadratico con parametro di curvatura $\mu$, il che permette di applicare ragionamenti simili, anche se in forma locale.

### Il tasso di apprendimento massimo e la stabilità

Con la costante di Lipschitz $L$ come limite superiore della curvatura, si può derivare il tasso di apprendimento massimo che garantisce che la discesa del gradiente non diverga:

$$\alpha \le \frac{1}{L}$$

Puoi capire questo limite ricordando l'analogia del canyon: $L$ misura quanto è "ripida" la pendenza più accentuata di tutta la superficie. Se il tuo passo è maggiore di $1/L$, in quella direzione di massima curvatura finirai per saltare da una parete all'altra, e il processo può oscillare con forza crescente fino a divergere. Un passo di dimensione $1/L$ o inferiore garantisce che, anche nella direzione più curva, ogni aggiornamento riduca (o per lo meno non aumenti) il valore dell'obiettivo.

Questo risultato permette anche di reinterpretare la discesa del gradiente stessa: è matematicamente equivalente a minimizzare un'approssimazione del secondo ordine di $h$ in cui la matrice di curvatura reale (la matrice Hessiana) è sostituita da $L$ moltiplicato per la matrice identità. Cioè, la discesa del gradiente assume, in modo conservativo, che la curvatura sia uguale in tutte le direzioni, quando in realtà quasi mai è così. Questa è la radice matematica del problema del "canyon stretto": l'algoritmo tratta il fondo piatto con la stessa cautela delle pareti ripide, perché non distingue le direzioni ad alta e a bassa curvatura. Nel prossimo capitolo vedrai come il momento e i metodi del secondo ordine correggono proprio questo limite.

### Il numero di condizionamento e il limite di convergenza

La difficoltà reale di ottimizzare una funzione non dipende solo da $L$, ma dal rapporto tra la curvatura massima e quella minima. Questo rapporto si chiama numero di condizionamento, $\kappa$:

$$\kappa = \frac{L}{\mu}$$

cioè il quoziente tra il più grande e il più piccolo autovalore della matrice Hessiana (o, in modo equivalente, tra la curvatura massima e quella minima della funzione). Il numero di condizionamento offre una caratterizzazione globale di quanto sia "difficile" un problema di ottimizzazione: più grande è $\kappa$, più la superficie di perdita assomiglia a un canyon stretto e allungato, e più lentamente converge la discesa del gradiente.

Sotto le ipotesi di regolarità lipschitziana e convessità forte, con il tasso di apprendimento adeguato, si può dimostrare che l'errore, misurato come differenza tra il valore attuale e il valore ottimo, decresce in modo approssimativamente esponenziale con il numero di iterazioni $k$:

$$h(\theta_k) - h(\theta^\ast) \;\le\; \left(1 - \frac{1}{\kappa}\right)^{k} \big[h(\theta_0) - h(\theta^\ast)\big]$$

dove $\theta^\ast$ è il punto ottimo. Da questo limite si deduce quante iterazioni $k$ servono per raggiungere un errore minore di $\varepsilon$:

$$k = O\!\left(\kappa \log \frac{1}{\varepsilon}\right)$$

In altre parole, il numero di iterazioni necessarie cresce in modo proporzionale al numero di condizionamento. Un problema con $\kappa$ piccolo (superficie di perdita "arrotondata", simile a una conca) converge rapidamente; un problema con $\kappa$ grande (canyon molto stretto) può richiedere moltissime più iterazioni per raggiungere la stessa precisione.

### I limiti di questa teoria

Questi limiti sono preziosi per costruire intuizione, ma non conviene prenderli alla lettera nella pratica, per diverse ragioni:

- Sono troppo pessimistici, perché devono coprire il caso peggiore possibile, non quello tipico
- Alcune ipotesi, come la convessità, sono troppo forti: la maggior parte delle funzioni obiettivo di una rete neurale non sono convesse
- Altre ipotesi sono troppo deboli: i problemi reali di solito hanno una struttura aggiuntiva che questi limiti non catturano
- Si basano su misure "grossolane" del problema, come il numero di condizionamento globale, che potrebbe non riflettere bene la difficoltà reale
- Si concentrano sul comportamento asintotico (quando $k$ è molto grande), ma nella pratica quasi mai si addestra fino alla convergenza completa; ciò che accade nelle prime iterazioni conta quanto o più

Per questo, la scelta dell'ottimizzatore deve essere guidata soprattutto dall'esperienza pratica, sostenuta da queste intuizioni teoriche, e non il contrario. **Tieni a mente questa idea**: la teoria ti aiuta a capire perché qualcosa può fallire, ma è la validazione empirica a decidere quale ottimizzatore usare in ciascun problema.

## Esempio di codice commentato

Il seguente esempio implementa la discesa del gradiente a mano, senza alcun framework di apprendimento profondo, su una funzione di perdita quadratica a due parametri. Scegli la curvatura di ciascuna direzione per simulare un "canyon stretto" e confronti tre tassi di apprendimento: uno adeguato, uno troppo alto (che diverge) e uno troppo basso (che converge molto lentamente). Funziona direttamente su Google Colab; ti servono solo NumPy e Matplotlib, che sono già preinstallati.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definimos la función de pérdida cuadrática ---
# h(theta) = 0.5 * (a * theta_1^2 + b * theta_2^2)
# 'a' y 'b' son las curvaturas en cada dirección: si son muy distintas,
# la superficie se parece al "cañón estrecho" de la analogía del capítulo.
a, b = 1.0, 20.0  # número de condición kappa = b / a = 20

def loss(theta):
    return 0.5 * (a * theta[0] ** 2 + b * theta[1] ** 2)

def gradient(theta):
    # Derivadas parciales de la cuadrática respecto a cada parámetro
    return np.array([a * theta[0], b * theta[1]])

# --- 2. Implementamos el descenso de gradiente a mano ---
def gradient_descent(theta_init, alpha, num_steps):
    theta = np.array(theta_init, dtype=float)
    trayectoria = [theta.copy()]
    for _ in range(num_steps):
        theta = theta - alpha * gradient(theta)  # regla de actualización de GD
        trayectoria.append(theta.copy())
        # Si los valores se disparan, detenemos para no desbordar el tipo float
        if np.any(np.abs(theta) > 1e6):
            break
    return np.array(trayectoria)

# --- 3. Punto de partida y tasas de aprendizaje a comparar ---
theta_inicial = [4.0, 4.0]
num_pasos = 40

# La tasa máxima teórica de estabilidad es 1 / L, con L = max(a, b) = b
L = max(a, b)
alpha_estable_max = 1.0 / L

tasas = {
    "Buena (alpha = 0.04)": 0.04,
    "Demasiado alta (alpha = 0.11)": 0.11,   # supera 1/L y hace diverger el proceso
    "Demasiado baja (alpha = 0.005)": 0.005,  # converge, pero muy lentamente
}

resultados = {
    nombre: gradient_descent(theta_inicial, alpha, num_pasos)
    for nombre, alpha in tasas.items()
}

# --- 4. Visualizamos la superficie de pérdida y las trayectorias ---
theta1_rango = np.linspace(-6, 6, 200)
theta2_rango = np.linspace(-6, 6, 200)
T1, T2 = np.meshgrid(theta1_rango, theta2_rango)
Z = 0.5 * (a * T1 ** 2 + b * T2 ** 2)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel izquierdo: curvas de nivel con las tres trayectorias
ax = axes[0]
ax.contour(T1, T2, Z, levels=25, cmap="viridis", alpha=0.6)
for nombre, trayectoria in resultados.items():
    ax.plot(trayectoria[:, 0], trayectoria[:, 1], marker="o", markersize=3, label=nombre)
ax.plot(0, 0, "r*", markersize=15, label="Mínimo")
ax.set_xlabel("theta_1 (dirección de baja curvatura)")
ax.set_ylabel("theta_2 (dirección de alta curvatura)")
ax.set_title("Trayectorias del descenso de gradiente")
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.legend(fontsize=8)

# Panel derecho: evolución de la pérdida en escala logarítmica
ax = axes[1]
for nombre, trayectoria in resultados.items():
    valores_perdida = [loss(theta) for theta in trayectoria]
    ax.plot(valores_perdida, marker="o", markersize=3, label=nombre)
ax.set_yscale("log")
ax.set_xlabel("Iteración")
ax.set_ylabel("Pérdida h(theta) [escala log]")
ax.set_title("Convergencia según la tasa de aprendizaje")
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

print(f"Numero de condicion (kappa) = {b / a:.1f}")
print(f"Tasa de aprendizaje maxima estable (1/L) = {alpha_estable_max:.4f}")
```

Eseguendo questo codice, osserva tre comportamenti molto diversi nel pannello di destra:

- La curva con tasso buono scende in modo costante e dolce
- La curva con tasso troppo alto oscilla e cresce senza controllo: nel pannello di sinistra vedrai come la traiettoria rimbalza tra i due lati del "canyon" nella direzione di alta curvatura
- La curva con tasso troppo basso converge sì, ma ha bisogno di moltissime più iterazioni per avvicinarsi al minimo, e avanza particolarmente piano lungo la direzione piatta

**Modifica il valore di `b`** nella tua copia del notebook e osserva come, aumentando il numero di condizionamento, diventi sempre più difficile trovare un tasso di apprendimento che funzioni bene in entrambe le direzioni contemporaneamente.

## Riepilogo dei punti chiave

- Ottimizzare una rete neurale significa minimizzare una funzione obiettivo (o di perdita) regolando i suoi parametri passo dopo passo
- La discesa del gradiente aggiorna i parametri secondo $\theta_{k+1} = \theta_k - \alpha \nabla h(\theta_k)$: si muove in direzione opposta al gradiente, cioè la direzione di maggior crescita della perdita
- Questa regola deriva dalla minimizzazione dell'approssimazione di Taylor del primo ordine della funzione obiettivo entro una piccola regione attorno al punto attuale
- Il tasso di apprendimento $\alpha$ controlla la dimensione del passo: se è troppo alto, il processo oscilla e diverge; se è troppo basso, converge, ma molto lentamente
- La costante di Lipschitz $L$ limita la curvatura massima della funzione; il tasso di apprendimento massimo che garantisce la stabilità è $\alpha \le 1/L$
- Il numero di condizionamento $\kappa = L/\mu$ misura il rapporto tra la curvatura massima e quella minima; più è grande, più "stretto" è il canyon della superficie di perdita e più lenta è la convergenza
- Il numero di iterazioni necessarie per raggiungere un errore $\varepsilon$ cresce in modo proporzionale a $\kappa$: $k = O(\kappa \log(1/\varepsilon))$
- La discesa del gradiente è matematicamente equivalente a minimizzare un'approssimazione del secondo ordine con curvatura uniforme uguale a $L$ in tutte le direzioni, il che spiega perché fatica così tanto a navigare superfici con curvature molto diverse a seconda della direzione
- I limiti teorici di convergenza sono utili come guida intuitiva, ma di solito sono troppo pessimistici rispetto al comportamento reale nei problemi di apprendimento profondo

## Mettilo in pratica

Dedica tra quindici e trenta minuti a questa mini-attività, usando come base il codice della sezione precedente.

1. Apri un nuovo notebook su Google Colab e copia l'esempio di codice di questo capitolo
2. **Cambia** il valore di `b` a dieci, cinquanta e cento, mantenendo `a = 1.0`, ed esegui il notebook con ciascun valore
3. **Annota**, per ogni numero di condizionamento, qual è il tasso di apprendimento più grande che resta stabile (puoi calcolarlo con la formula $\alpha \le 1/L$ e verificarlo provando valori vicini)
4. **Osserva** quante iterazioni impiega il tasso "buono" per far scendere la perdita sotto 0,01 in ciascun caso, e verifica se questo numero cresce approssimativamente in modo proporzionale al numero di condizionamento, come prevede la teoria
5. **Condividi** con i tuoi compagni di corso, se possibile, i tuoi grafici insieme a una frase che spieghi con parole tue perché un numero di condizionamento alto rende più difficile scegliere un buon tasso di apprendimento

Se finisci prima del tempo, prova a sostituire la funzione quadratica con una semplice funzione non convessa, per esempio la funzione di Rosenbrock, e osserva come cambia il comportamento della discesa del gradiente quando la curvatura varia da un punto all'altro invece di essere costante.
