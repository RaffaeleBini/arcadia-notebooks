# Capitolo 4 — Supremazia quantistica e i limiti della simulazione classica

## Perché è importante

Nel 2019 Google ha annunciato che il suo processore Sycamore aveva risolto in 200 secondi un calcolo che, secondo le sue stime, avrebbe richiesto migliaia di anni al supercomputer classico più potente del mondo. Questo traguardo è stato chiamato **supremazia quantistica**: il momento in cui un computer quantistico risolve un compito che nessun computer classico può risolvere in un tempo ragionevole. Capire perché quel calcolo è così difficile da simulare in modo classico (e come si dimostra che il risultato quantistico è corretto) ti dà l'intuizione chiave di tutto questo corso: la potenza del calcolo quantistico cresce esponenzialmente con il numero di qubit, mentre il costo di simularlo su un computer classico cresce altrettanto rapidamente.

## Spiegazione concettuale

### Perché simulare un circuito quantistico costa così tanto

Un sistema di n qubit in sovrapposizione richiede 2^n numeri per essere descritto completamente. Con 53 qubit (quelli di Sycamore), si tratta di più di 9.000 miliardi di numeri: non entrano più nella memoria di nessun computer classico esistente. Per questo, più qubit ha il tuo circuito, più velocemente cresce il costo di simularlo in modo classico, mentre il computer quantistico esegue lo stesso circuito senza questo costo aggiuntivo: letteralmente, "è" quel sistema quantistico.

### Campionamento di circuiti casuali: il compito risolto da Sycamore

Il compito concreto che dimostra la supremazia non è "risolvere un problema utile", ma qualcosa di più sottile: generare campioni della distribuzione di probabilità in uscita da un circuito quantistico casuale e profondo. Un computer quantistico genera questi campioni in modo naturale, eseguendo il circuito più volte. Un computer classico, invece, deve calcolare quella distribuzione di probabilità moltiplicando matrici enormi, e questo calcolo diventa impraticabile oltre un certo numero di qubit e di profondità del circuito.

### Come si verifica che il risultato quantistico è corretto: XEB

Se nessuno può calcolare classicamente la distribuzione di probabilità completa, come fai a sapere che i campioni del computer quantistico sono corretti? La risposta è il **benchmarking di entropia incrociata** (XEB): invece di calcolare l'intera distribuzione, calcoli la probabilità teorica di ogni singolo campione che hai osservato (un'operazione molto più economica che calcolare tutte le probabilità possibili) e confronti quella probabilità con ciò che ti aspetteresti da un risultato puramente casuale. Quanto più i campioni osservati somigliano alla distribuzione teorica del circuito (e non al rumore casuale), tanta più fiducia hai che l'hardware stia davvero funzionando come un computer quantistico.

La forma che assume quella distribuzione teorica in uscita —molte probabilità basse e poche alte, come macchie di luce laser sparpagliate— è nota come **distribuzione di Porter-Thomas**, ed è la "impronta digitale" che si cerca confrontando i risultati reali con la teoria. Una tecnica correlata, il **benchmarking della purezza delle macchie** (speckle purity benchmarking), aiuta a separare quanto di quell'impronta è segnale quantistico reale e quanto è rumore dell'hardware.

### Quantificare la fiducia: significatività statistica

Affermare "abbiamo dimostrato la supremazia quantistica" è un'affermazione forte, quindi va supportata con statistica rigorosa. Si usano strumenti come il **test di Kolmogorov-Smirnov** (KS), che verifica se due distribuzioni di dati sono statisticamente indistinguibili, e tecniche di **bootstrapping** (ricampionamento ripetuto degli stessi dati) per stimare quanto varierebbero le tue conclusioni se ripetessi l'esperimento. Il risultato finale si esprime di solito in "livelli sigma": quante più deviazioni standard di margine hai rispetto all'ipotesi nulla (cioè che il risultato sia rumore casuale), tanto più solida è la conclusione.

### L'era NISQ: potenti, ma imperfetti

Sycamore, Bristlecone (il suo predecessore) e tutti i processori quantistici attuali appartengono all'era **NISQ** (quantistico a scala intermedia e rumoroso, ricorda il capitolo 2): hanno decine o al massimo qualche centinaio di qubit, e il rumore limita la profondità dei circuiti che si possono eseguire in modo affidabile. La supremazia quantistica dimostra che, anche in quest'era imperfetta, esistono già compiti che superano il calcolo classico. La sfida dei prossimi capitoli è diversa: non dimostrare la supremazia con circuiti casuali, ma usare quegli stessi computer rumorosi per risolvere problemi utili di machine learning, chimica e ottimizzazione.

## Esempio di codice commentato

Un modo pratico per sentire il "costo esponenziale" è simulare l'evoluzione nel tempo di un sistema quantistico. Calcolare esattamente quell'evoluzione richiede di moltiplicare matrici che crescono come 2^n, quindi in pratica si approssima dividendo l'evoluzione in passi piccoli: questo si chiama **trotterizzazione**. Ecco un esempio con un modello di quattro qubit (una catena di spin con campo trasverso, un sistema tipico della fisica dei materiali):

```python
import cirq
import tensorflow_quantum as tfq
import numpy as np
import matplotlib.pyplot as plt

def cadena_ising_campo_transverso(n, campos):
    """Hamiltoniano de una cadena de n qubits con interacción Z-Z y campo X."""
    qubits = [cirq.GridQubit(0, i) for i in range(n)]
    hamiltoniano = cirq.PauliSum()
    for i in range(n - 1):
        hamiltoniano -= cirq.Z(qubits[i]) * cirq.Z(qubits[i + 1])
    for i in range(n):
        hamiltoniano -= campos[i] * cirq.X(qubits[i])
    return hamiltoniano

def paso_trotter(hamiltoniano, t):
    """Aproxima exp(-i * H * t) aplicando cada término del Hamiltoniano por separado."""
    return cirq.Circuit([tfq.util.exponential([t * termino]) for termino in hamiltoniano])

def trotter_lie(hamiltoniano, t, repeticiones):
    """Cuantas más repeticiones, más fino el paso y más preciso el resultado."""
    circuito = cirq.Circuit()
    for _ in range(repeticiones):
        circuito += paso_trotter(hamiltoniano, t / repeticiones)
    return circuito

n = 4
campos = [1.0] * n
hamiltoniano = cadena_ising_campo_transverso(n, campos)
qubits = [cirq.GridQubit(0, i) for i in range(n)]
magnetizacion = sum(cirq.Z(q) for q in qubits)

tiempos = np.linspace(0.01, 2, 20)

# Con solo 1 repetición, el paso de trotter es "grueso" y menos preciso
circuitos_1_paso = [trotter_lie(hamiltoniano, float(t), 1) for t in tiempos]
# Con 10 repeticiones, el paso es más fino y se acerca más a la evolución real
circuitos_10_pasos = [trotter_lie(hamiltoniano, float(t), 10) for t in tiempos]

capa_expectativa = tfq.layers.Expectation()
exp_1_paso = capa_expectativa(circuitos_1_paso, symbol_names=[], symbol_values=[[] for _ in tiempos], operators=magnetizacion).numpy()
exp_10_pasos = capa_expectativa(circuitos_10_pasos, symbol_names=[], symbol_values=[[] for _ in tiempos], operators=magnetizacion).numpy()

plt.plot(tiempos, exp_1_paso, label="1 paso de Trotter")
plt.plot(tiempos, exp_10_pasos, label="10 pasos de Trotter")
plt.xlabel("Tiempo")
plt.ylabel("Magnetización")
plt.legend()
plt.show()
```

Più passi di Trotter usi, più ti avvicini all'evoluzione esatta del sistema, ma servono anche più porte quantistiche in sequenza, e ricorda il capitolo 2: più porte significa più tempo esposto alla decoerenza. Questo equilibrio tra precisione e profondità del circuito è esattamente lo stesso dilemma che affronta ogni computer quantistico NISQ reale.

## Riepilogo dei punti chiave

- La supremazia quantistica è stata dimostrata risolvendo un compito di campionamento di circuiti casuali, impraticabile da simulare in modo classico per la crescita esponenziale (2^n) degli stati possibili
- Il benchmarking di entropia incrociata (XEB) permette di verificare che i campioni quantistici osservati corrispondano alla distribuzione teorica del circuito (distribuzione di Porter-Thomas) e non al rumore
- La fiducia statistica nel risultato si quantifica con test come Kolmogorov-Smirnov, bootstrapping e livelli sigma
- Viviamo nell'era NISQ: computer quantistici reali ma rumorosi, con pochi qubit e circuiti poco profondi
- La trotterizzazione approssima l'evoluzione di un sistema quantistico dividendola in passi piccoli; più passi significano più precisione, ma anche più profondità del circuito e più esposizione al rumore

## Mettilo in pratica

Riprendi l'esempio di codice e aggiungi una terza curva con 3 ripetizioni di Trotter (`trotter_lie(hamiltoniano, float(t), 3)`), a metà strada tra la versione a 1 passo e quella a 10. Disegna le tre curve insieme e osserva come la curva a 3 passi si discosta meno da quella a 10 passi rispetto a quella a 1 passo. Annota, in una frase, quale relazione osservi tra il numero di passi di Trotter e la somiglianza con la curva di riferimento (10 passi).
