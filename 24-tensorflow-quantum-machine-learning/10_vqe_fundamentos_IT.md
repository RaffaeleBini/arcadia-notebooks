# Capitolo 10 — VQE: fondamenti del calcolo delle energie quantistiche

## Perché è importante

Uno degli usi più promettenti del calcolo quantistico è la simulazione di sistemi fisici e molecole, un compito che un computer classico svolge molto male perché gli stati quantistici crescono in modo esponenziale (ricorda il capitolo 4). Il **Variational Quantum Eigensolver** (VQE, Risolutore Variazionale Quantistico di Autovalori) è l'algoritmo di riferimento per calcolare l'energia dello stato fondamentale (il livello energetico più basso possibile) di un sistema quantistico, ed è alla base di applicazioni reali in chimica quantistica, come il calcolo dell'energia di una molecola di idrogeno.

## Spiegazione concettuale

### Un Hamiltoniano è una "ricetta" dell'energia

In fisica quantistica, l'**Hamiltoniano** di un sistema è l'operatore che descrive la sua energia totale. Qualsiasi Hamiltoniano si può scomporre come somma pesata di prodotti tensoriali delle matrici di Pauli (I, X, Y, Z che già conosci dai capitoli precedenti). Per esempio, un Hamiltoniano di un solo qubit potrebbe scriversi come `H = a·I + b·X + c·Y + d·Z`, dove a, b, c, d sono numeri reali che dipendono dal sistema fisico concreto che vuoi descrivere. L'obiettivo di VQE è trovare il valore minimo possibile del valore di aspettazione di quell'Hamiltoniano: questa è, per definizione, l'energia dello stato fondamentale.

### Il principio variazionale: indovinare e migliorare

VQE si basa sul **principio variazionale**: per qualunque stato quantistico tu provi, il valore di aspettazione dell'Hamiltoniano su quello stato non è mai inferiore all'energia reale dello stato fondamentale. Ciò significa che puoi trattare la ricerca dell'energia minima come un problema di ottimizzazione: prepari un circuito con parametri regolabili (l'"ansatz", la tua migliore congettura su come dovrebbe apparire lo stato fondamentale), misuri il valore di aspettazione dell'Hamiltoniano e regoli i parametri in modo che quel valore scenda il più possibile. Quanto più in basso riesci a portarlo, tanto più sei vicino all'energia reale.

### Misurare un Hamiltoniano con più termini

Un Hamiltoniano di più qubit, come `H = a·(Z₀+Z₁) + b·(Z₂+Z₃)`, si misura termine per termine: per ogni termine dell'Hamiltoniano misuri il circuito nella base corrispondente (ruotando prima di misurare se il termine include X o Y, dato che puoi misurare direttamente solo nella base Z), moltiplichi per il suo coefficiente e sommi tutti i risultati. TFQ lo fa in modo trasparente: passi la lista dei termini come "operatore di lettura" di un layer `PQC`, e si occupa lui delle misure e della somma internamente.

### Ottimizzatori classici: gradiente contro scatola nera

Puoi ottimizzare i parametri dell'ansatz in due modi:

- **Basato sul gradiente**: usando `tf.GradientTape` e un ottimizzatore come Adam, come nei capitoli precedenti, purché tu possa calcolare il gradiente del circuito (con `ParameterShift`, per esempio)
- **Ottimizzazione a scatola nera**: trattando il circuito come una funzione che puoi solo valutare (senza gradiente), e usando algoritmi come `scipy.optimize.minimize` con il metodo Nelder-Mead, che prova diverse combinazioni di parametri e converge senza bisogno di derivate

Entrambi gli approcci sono validi; nel prossimo capitolo vedrai anche l'algoritmo Rotosolve, progettato specificamente per i circuiti quantistici parametrizzati.

## Esempio di codice commentato

Per prima cosa, un Hamiltoniano semplice di un solo qubit con un ansatz minimo, ottimizzato con Nelder-Mead (senza bisogno di calcolare gradienti):

```python
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

qubit = cirq.GridQubit(0, 0)
angulo = sympy.symbols("angulo")

# Ansatz: una única rotación Y entrenable
circuito_vqe = cirq.Circuit(cirq.ry(angulo).on(qubit))

# Hamiltoniano H = 1*I + 1*Z  (un ejemplo sencillo con energía fundamental conocida: -1 + 1 = 0... )
# aquí usamos H = 1*I + 2*Z, cuya energía fundamental teórica es 1 - 2 = -1
hamiltoniano = 1 + 2 * cirq.Z(qubit)

entrada_vacia = tfq.convert_to_tensor([cirq.Circuit()])
capa_entrada = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
inicializador = tf.keras.initializers.RandomUniform(0, 2 * np.pi)
salida = tfq.layers.PQC(circuito_vqe, hamiltoniano,
                         differentiator=tfq.differentiators.CentralDifference(),
                         initializer=inicializador)(capa_entrada)
modelo_vqe = tf.keras.models.Model(inputs=capa_entrada, outputs=salida)

def energia(parametros):
    modelo_vqe.set_weights(np.array([parametros]))
    return modelo_vqe(entrada_vacia).numpy()[0][0]

resultado = minimize(energia, np.random.uniform(0, 2 * np.pi, 1), method="Nelder-Mead")
print("Energía mínima encontrada:", resultado["fun"])
print("Ángulo óptimo:", resultado["x"])

# Visualizamos el paisaje de energía completo, barriendo el ángulo entre 0 y 2*pi
angulos = np.linspace(0, 2 * np.pi, 100)
energias = [energia([a]) for a in angulos]
plt.plot(angulos, energias)
plt.xlabel("Ángulo del ansatz")
plt.ylabel("Energía")
plt.title("VQE — paisaje de energía de un único parámetro")
plt.show()
```

Ora, un Hamiltoniano di quattro qubit con due termini indipendenti, addestrato con il gradiente:

```python
def construir_ansatz(qubits, parametros):
    circuito = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuito.append(cirq.rx(parametros[0]).on(qubit))
        circuito.append(cirq.ry(parametros[1]).on(qubit))
    circuito.append(cirq.H(qubits[1]))
    circuito.append(cirq.CNOT(qubits[0], qubits[1]))
    circuito.append(cirq.CNOT(qubits[2], qubits[3]))
    return circuito

qubits = [cirq.GridQubit(0, i) for i in range(4)]
parametros = sympy.symbols("theta0:2")
circuito_vqe_multi = construir_ansatz(qubits, parametros)

# H = 1*(Z0 + Z1) + 1*(Z2 + Z3)
hamiltoniano_multi = (cirq.Z(qubits[0]) + cirq.Z(qubits[1])) + (cirq.Z(qubits[2]) + cirq.Z(qubits[3]))

entrada_vacia = tfq.convert_to_tensor([cirq.Circuit()])
capa_entrada = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
salida = tfq.layers.PQC(circuito_vqe_multi, hamiltoniano_multi)(capa_entrada)
modelo_multi = tf.keras.models.Model(inputs=capa_entrada, outputs=salida)

optimizador = tf.keras.optimizers.Adam(learning_rate=0.1)
historial_energia = []
for paso in range(100):
    with tf.GradientTape() as cinta:
        energia_actual = modelo_multi(entrada_vacia)
    gradientes = cinta.gradient(energia_actual, modelo_multi.trainable_variables)
    optimizador.apply_gradients(zip(gradientes, modelo_multi.trainable_variables))
    historial_energia.append(energia_actual.numpy()[0][0])

plt.plot(historial_energia)
plt.xlabel("Iteración")
plt.ylabel("Energía")
plt.title("VQE multi-qubit — convergencia por gradiente")
plt.show()
```

## Riepilogo dei punti chiave

- Qualsiasi Hamiltoniano si scompone in una somma di prodotti di matrici di Pauli (I, X, Y, Z); VQE ne cerca il minimo del valore di aspettazione
- Il principio variazionale garantisce che, abbassando quel valore il più possibile, ci si avvicina all'energia reale dello stato fondamentale
- Un Hamiltoniano con più termini si misura termine per termine, e TFQ somma automaticamente i risultati quando glieli passi come lista al `PQC`
- Puoi ottimizzare con il gradiente (Adam + `ParameterShift`/`CentralDifference`) oppure con ottimizzatori a scatola nera come Nelder-Mead, senza bisogno di gradienti

## Mettilo in pratica

Nel primo esempio (Hamiltoniano di un solo qubit), cambia l'Hamiltoniano da `1 + 2 * cirq.Z(qubit)` a `1 + 2 * cirq.X(qubit)`. Prima di eseguire qualsiasi cosa, rifletti: l'ansatz attuale ha solo una rotazione `ry`, che non può mai portare il qubit su un autostato di X. Aggiungi una porta `cirq.rz` (con un secondo parametro addestrabile) all'ansatz in modo che il circuito abbia abbastanza libertà di ruotare verso qualsiasi asse, e ottimizza di nuovo con Nelder-Mead. Verifica che ora converga davvero all'energia minima teorica (-1).
