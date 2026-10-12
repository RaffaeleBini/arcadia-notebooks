# Capitolo 9 — QAOA e ottimizzazione combinatoria

## Perché è importante

Molti problemi del mondo reale sono problemi di ottimizzazione combinatoria: distribuire percorsi di consegna, raggruppare clienti, progettare reti di telecomunicazioni. Uno dei problemi di riferimento in questo campo è **Max-Cut**: dato un grafo (nodi collegati da archi), dividere i nodi in due gruppi in modo che il numero di archi "tagliati" (con un nodo in ciascun gruppo) sia il massimo possibile. Il **Quantum Approximate Optimization Algorithm** (QAOA, Algoritmo Quantistico Approssimato di Ottimizzazione) è una delle strategie più note per affrontare questo tipo di problemi con un computer quantistico NISQ, e in questo capitolo lo applichi esattamente a Max-Cut.

## Spiegazione concettuale

### Tradurre un problema di ottimizzazione in un Hamiltoniano

L'idea centrale di QAOA è tradurre il tuo problema di ottimizzazione in un **Hamiltoniano di costo**: un operatore quantistico il cui valore di aspettazione, misurato su uno stato concreto, coincide con "quanti archi taglia" quella divisione dei nodi. Per Max-Cut, questo Hamiltoniano si costruisce sommando un termine `Z_i · Z_j` per ogni arco tra i nodi i e j: il termine vale -1 se i due nodi sono nello stesso gruppo, e +1 se si trovano in gruppi diversi. Più alto è il valore di aspettazione dell'Hamiltoniano, migliore è il taglio.

### Il circuito QAOA: alternare costo e mescolamento

Il circuito parte mettendo tutti i qubit in sovrapposizione con porte Hadamard (rappresentando tutte le divisioni possibili contemporaneamente) e poi alterna due blocchi, ciascuno con il proprio parametro addestrabile:

- L'**Hamiltoniano di costo**, parametrizzato da un angolo che di solito chiamiamo γ (gamma): "spinge" lo stato verso le divisioni che tagliano più archi
- L'**Hamiltoniano di mescolamento**, parametrizzato da un angolo che di solito chiamiamo β (beta): mescola lo stato in modo che il circuito possa esplorare altre divisioni oltre a quella che sembra già buona, evitando di restare bloccato su una soluzione mediocre

Quante più volte ripeti quel ciclo costo-mescolamento (parametro `p` nella letteratura su QAOA), tanto più il circuito può avvicinarsi alla soluzione ottimale, al prezzo di un circuito più profondo (e ricorda il capitolo 2: più profondità, più esposizione alla decoerenza).

### Dalla teoria a Cirq: `tfq.util.exponential`

Cirq e TFQ non ti obbligano a "scomporre a mano" ogni Hamiltoniano in porte quantistiche elementari: `tfq.util.exponential(operadores, coeficientes)` costruisce direttamente il circuito che applica `exp(-i * coefficiente * operatore)`, che è esattamente l'operazione necessaria per applicare un Hamiltoniano per un tempo controllato dal parametro addestrabile.

### Leggere la risposta: campionare il circuito addestrato

Dopo aver addestrato i parametri γ e β, misuri il circuito molte volte (con `tfq.layers.Sample`). Ogni misura ti dà una possibile divisione dei nodi in due gruppi; la divisione che compare più spesso (o quella con il taglio maggiore tra i campioni) è la tua soluzione approssimata al problema Max-Cut.

## Esempio di codice commentato

Usiamo un grafo piccolo (un ciclo di 6 nodi) per rendere rapido l'addestramento in una sessione di classe.

```python
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

grafo = nx.cycle_graph(6)  # 6 nodos en círculo: 0-1-2-3-4-5-0

qubits = [cirq.GridQubit(0, i) for i in range(grafo.number_of_nodes())]

estado_inicial = cirq.Circuit()
for qubit in qubits:
    estado_inicial.append(cirq.H(qubit))  # superposición de todas las divisiones posibles

hamiltoniano_mezcla = sum(cirq.PauliString(cirq.X(qubits[i])) for i in range(len(qubits)))

hamiltoniano_coste = grafo.number_of_edges() / 2
for nodo_a, nodo_b in grafo.edges():
    hamiltoniano_coste += cirq.PauliString(0.5 * cirq.Z(qubits[nodo_a]) * cirq.Z(qubits[nodo_b]))

p = 1  # número de repeticiones del ciclo coste-mezcla
gamma, beta = sympy.symbols("gamma beta")
circuito_qaoa = cirq.Circuit()
circuito_qaoa += tfq.util.exponential(operators=[hamiltoniano_coste, hamiltoniano_mezcla],
                                       coefficients=[gamma, beta])

entradas = tfq.convert_to_tensor([estado_inicial])
capa_entrada = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
salida = tfq.layers.PQC(circuito_qaoa, hamiltoniano_coste)(capa_entrada)
modelo_qaoa = tf.keras.models.Model(inputs=capa_entrada, outputs=salida)
modelo_qaoa.compile(loss=tf.keras.losses.MAE, optimizer=tf.keras.optimizers.Adam())

# Entrenamos para que el valor esperado del corte se acerque al máximo teórico (etiqueta ficticia 0)
historial = modelo_qaoa.fit(entradas, np.array([0]), epochs=200)

plt.plot(historial.history["loss"])
plt.title("QAOA — pérdida durante el entrenamiento")
plt.xlabel("Iteración")
plt.ylabel("Pérdida")
plt.show()

# Muestreamos el circuito entrenado para obtener divisiones concretas
parametros_entrenados = modelo_qaoa.trainable_variables
circuito_con_datos = tfq.layers.AddCircuit()(entradas, append=circuito_qaoa)
muestras = tfq.layers.Sample()(circuito_con_datos, symbol_names=[gamma, beta],
                                symbol_values=parametros_entrenados, repetitions=1000)

cortes_obtenidos = []
for bits in muestras.values:
    grupo_uno = [i for i, bit in enumerate(bits) if bit == 1]
    cortes_obtenidos.append(nx.algorithms.cuts.cut_size(grafo, grupo_uno))

print("Corte medio obtenido:", np.mean(cortes_obtenidos))
print("Mejor corte posible (fuerza bruta, grafo pequeño):",
      max(nx.algorithms.cuts.cut_size(grafo, list(combo))
          for r in range(grafo.number_of_nodes() + 1)
          for combo in __import__("itertools").combinations(grafo.nodes(), r)))
```

Con un grafo così piccolo puoi calcolare il taglio massimo reale per forza bruta e confrontarlo: vedrai che il taglio medio ottenuto da QAOA si avvicina parecchio a quel massimo, anche se quasi mai lo raggiunge in modo perfetto con una sola ripetizione (`p=1`).

## Riepilogo dei punti chiave

- QAOA traduce un problema di ottimizzazione combinatoria (Max-Cut) in un Hamiltoniano di costo, il cui valore di aspettazione misura la qualità di una soluzione
- Il circuito alterna un blocco di costo (parametrizzato da γ) e un blocco di mescolamento (parametrizzato da β), ripetuti `p` volte
- `tfq.util.exponential` costruisce direttamente il circuito che applica un Hamiltoniano per un tempo dato da un parametro addestrabile
- Dopo l'addestramento, il circuito viene campionato molte volte per ottenere soluzioni concrete al problema
- Aumentare `p` (più ripetizioni del ciclo costo-mescolamento) di solito migliora la soluzione, al prezzo di un circuito più profondo

## Mettilo in pratica

Cambia `p = 1` in `p = 2` nell'esempio di codice: ti serviranno due simboli di costo e due di mescolamento (per esempio, `sympy.symbols("gamma1 gamma2 beta1 beta2")`) e due chiamate a `tfq.util.exponential` concatenate con `+=`, una per ciascuna ripetizione del ciclo. Addestra di nuovo e confronta il taglio medio ottenuto con quello di `p=1`. Annota se la soluzione migliora, e collega il risultato all'idea che più ripetizioni avvicinano la soluzione all'ottimo.
