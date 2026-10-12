# Capitolo 11 — VQE avanzato: Rotosolve e confronto tra ottimizzatori

## Perché è importante

Nel capitolo precedente hai ottimizzato VQE con Adam (basato sul gradiente) e con Nelder-Mead (scatola nera generica). Nessuno dei due è progettato specificamente per circuiti quantistici parametrizzati con rotazioni. **Rotosolve** invece lo è: sfrutta una proprietà matematica esclusiva delle porte di rotazione quantistiche per trovare, a ogni passo, il valore ottimo esatto di un parametro, senza dover fare piccoli passi di gradiente né tentare per prova ed errore. Capire Rotosolve ti dà intuizione su come progettare ottimizzatori su misura per problemi quantistici, invece di applicare sempre gli stessi strumenti classici.

## Spiegazione concettuale

### Perché un parametro di rotazione si comporta come un'onda

Quando fissi tutti i parametri del tuo ansatz tranne uno, e lasci variare quel parametro (l'angolo di una porta `rx`, `ry` o `rz`), l'energia risultante `E(θ)` non è una curva qualsiasi: ha sempre la forma `E(θ) = A·cos(θ) + B·sin(θ) + C`, una semplice onda sinusoidale. È una conseguenza diretta del modo in cui le porte di rotazione trasformano lo stato quantistico. Puoi verificarlo tu stesso nell'esempio del capitolo precedente: la curva di energia che hai disegnato variando l'angolo da 0 a 2π aveva esattamente quella forma ondulata.

### Rotosolve: risolvere l'onda invece di approssimarla

Se sai che `E(θ)` è un'onda con quella forma esatta, non devi procedere con piccoli passi di gradiente per trovarne il minimo: ti basta valutare l'energia in tre punti precisi (per esempio, in un angolo `φ` qualsiasi, e in `φ + π/2` e `φ - π/2`) per ricavare matematicamente i coefficienti A, B e C, e calcolare direttamente l'angolo che minimizza quell'onda. Rotosolve ripete questo processo parametro per parametro, ciclo dopo ciclo: in ogni ciclo, ciascun parametro viene impostato sul suo valore ottimo esatto supponendo che gli altri restino fissi, e dopo più cicli l'insieme completo dei parametri converge.

Il vantaggio rispetto a un metodo basato sul gradiente è che ogni aggiornamento di un parametro è un salto diretto all'ottimo (per quel parametro, con gli altri fissi), non un piccolo passo nella direzione di massima pendenza. Lo svantaggio è che devi valutare il circuito più volte per ogni parametro e per ogni ciclo, quindi il costo per iterazione è più alto.

### Confronto tra tre ottimizzatori a parità di condizioni

Oltre a Rotosolve, questo capitolo confronta due ottimizzatori classici di scatola nera che già conosci o che conoscerai:

- **Nelder-Mead**: esplora lo spazio dei parametri con un "simplesso" (una figura geometrica che si deforma) che si contrae verso la regione di energia più bassa; è robusto ma può richiedere molte valutazioni in problemi con molti parametri
- **SPSA** (approssimazione stocastica a perturbazione simultanea): a ogni passo perturba **tutti** i parametri insieme con una direzione casuale, valuta l'energia in quella direzione e stima il gradiente completo con solo due valutazioni del circuito, qualunque sia il numero di parametri. È particolarmente utile quando valutare il circuito è costoso (per esempio, su hardware quantistico reale) e hai molti parametri da ottimizzare

Confrontare i tre sullo stesso problema ti permette di vedere che non esiste un ottimizzatore universalmente migliore: dipende da quanti parametri hai, da quanto costa valutare il circuito e da quanto rumore c'è in quella valutazione.

## Esempio di codice commentato

Per prima cosa, l'implementazione di Rotosolve (una versione ridotta di quella usata nel resto del capitolo, con meno iterazioni per poter essere eseguita in pochi secondi):

```python
import numpy as np
import random

def rotosolve(funcion_energia, num_parametros, iteraciones=30):
    """Optimiza cada parámetro de forma analítica, ciclo a ciclo."""
    theta = np.random.uniform(-np.pi, np.pi, num_parametros)
    for _ in range(iteraciones):
        for d in range(num_parametros):
            phi = random.uniform(-np.pi, np.pi)
            theta_prueba = np.copy(theta)

            theta_prueba[d] = phi
            energia_phi = funcion_energia(theta_prueba)
            theta_prueba[d] = phi + np.pi / 2
            energia_mas = funcion_energia(theta_prueba)
            theta_prueba[d] = phi - np.pi / 2
            energia_menos = funcion_energia(theta_prueba)

            # Fórmula cerrada que resuelve el mínimo exacto de la onda A*cos + B*sin + C
            theta[d] = phi - np.pi / 2 - np.arctan2(
                2 * energia_phi - energia_mas - energia_menos, energia_mas - energia_menos
            )
    return {"fun": funcion_energia(theta), "params": theta}
```

Ora confrontiamo Rotosolve, Nelder-Mead e SPSA su un Hamiltoniano casuale di due qubit (simulato direttamente con Cirq, senza passare da TFQ, per poter controllare la misura base per base):

```python
import cirq
from scipy.optimize import minimize
from noisyopt import minimizeSPSA

def capa_ansatz(circuito, qubits, parametros):
    for qubit in qubits:
        circuito.append([cirq.rx(parametros[0]).on(qubit)])
        circuito.append([cirq.rz(parametros[1]).on(qubit)])
    for i in range(len(qubits) - 1):
        circuito.append([cirq.CNOT(qubits[i], qubits[i + 1])])
    return circuito

def rotar_a_base_de_medida(circuito, qubits, terminos_pauli):
    for i, qubit in enumerate(qubits):
        if terminos_pauli[i] == "x":
            circuito.append(cirq.ry(-np.pi / 2).on(qubit))
        elif terminos_pauli[i] == "y":
            circuito.append(cirq.rx(np.pi / 2).on(qubit))
    return circuito

def valor_esperado(circuito, qubits_medidos, repeticiones=1000):
    if len(qubits_medidos) == 0:
        return 1
    circuito.append(cirq.measure(*qubits_medidos, key="resultado"))
    simulador = cirq.Simulator()
    resultados = simulador.run(circuito, repetitions=repeticiones)
    medidas = resultados.measurements["resultado"]
    return np.mean([1 if sum(fila) % 2 == 0 else -1 for fila in medidas])

# Hamiltoniano de ejemplo: dos términos de Pauli sobre 2 qubits, con pesos fijos
terminos_hamiltoniano = [["x", "z"], ["z", "y"]]
pesos_hamiltoniano = [0.6, 0.3]
num_qubits, num_capas = 2, 2

def energia(parametros):
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    total = 0
    for termino, peso in zip(terminos_hamiltoniano, pesos_hamiltoniano):
        circuito = cirq.Circuit()
        for _ in range(num_capas):
            circuito = capa_ansatz(circuito, qubits, parametros)
        circuito = rotar_a_base_de_medida(circuito, qubits, termino)
        qubits_a_medir = [qubits[i] for i in range(len(termino)) if termino[i] != "i"]
        total += peso * valor_esperado(circuito, qubits_a_medir)
    return total

num_parametros = num_qubits * num_capas

resultado_nm = minimize(energia, np.random.uniform(0, 2 * np.pi, num_parametros), method="Nelder-Mead")
resultado_spsa = minimizeSPSA(energia, bounds=[[0, 2 * np.pi]] * num_parametros,
                               x0=np.random.uniform(0, 2 * np.pi, num_parametros), niter=100, paired=False)
resultado_roto = rotosolve(energia, num_parametros)

print("Nelder-Mead:", resultado_nm["fun"])
print("SPSA:", resultado_spsa["fun"])
print("Rotosolve:", resultado_roto["fun"])
```

Eseguendolo più volte noterai che i tre ottimizzatori di solito convergono a energie simili, ma con un numero di valutazioni del circuito molto diverso: Rotosolve di solito richiede meno cicli completi per stabilizzarsi, in cambio di valutare il circuito più volte per ogni parametro in ogni ciclo.

## Riepilogo dei punti chiave

- Il valore di aspettazione dell'energia, variando un solo parametro di rotazione e lasciando fissi gli altri, ha sempre la forma di un'onda `A·cos(θ) + B·sin(θ) + C`
- Rotosolve sfrutta questa proprietà per calcolare l'angolo ottimo esatto di ogni parametro, valutando il circuito in soli tre punti per parametro
- SPSA stima il gradiente completo con solo due valutazioni del circuito, qualunque sia il numero di parametri, utile quando valutare il circuito è costoso
- Non esiste un ottimizzatore universalmente migliore: la scelta dipende dal numero di parametri, dal costo di valutare il circuito e dal rumore presente in quella valutazione

## Mettilo in pratica

Aggiungi un terzo termine all'Hamiltoniano di esempio: `["y", "x"]` con peso `0.4` (ricorda di aggiornare sia `terminos_hamiltoniano` sia `pesos_hamiltoniano`). Esegui di nuovo i tre ottimizzatori e annota, per ciascuno, l'energia finale ottenuta. Verifica se Rotosolve resta competitivo rispetto a Nelder-Mead e SPSA ora che l'Hamiltoniano ha un termine in più.
