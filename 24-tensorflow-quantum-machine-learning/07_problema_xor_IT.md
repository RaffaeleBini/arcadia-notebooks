# Capitolo 7 — Risolvere lo XOR con il machine learning quantistico

## Perché è importante

Il problema XOR (o esclusivo) è celebre nel machine learning classico: un singolo percettrone (il neurone più semplice) non può risolverlo, perché i suoi quattro punti non si possono separare con un'unica retta. Serve una rete con almeno uno strato nascosto per risolverlo. In questo capitolo verificherai che un circuito quantistico con soli due qubit, grazie all'entanglement, riesce a separare lo XOR senza bisogno di "strati nascosti" classici. È la dimostrazione più chiara che l'entanglement non è solo una curiosità teorica: è una risorsa computazionale reale.

## Spiegazione concettuale

### Perché lo XOR non è linearmente separabile

Lo XOR ha quattro combinazioni di ingresso (0-0, 1-0, 0-1, 1-1) con uscite (-1, +1, +1, -1). Se disegni questi quattro punti su un piano, vedrai che non esiste nessuna retta che lasci tutti i "+1" da un lato e tutti i "-1" dall'altro: le due classi sono intrecciate geometricamente, in diagonale. Ti serve una funzione non lineare (una curva, non una retta) per separarle.

### Come l'entanglement fornisce quella "curva"

Il circuito di questo capitolo inizia con una porta `CNOT` tra i due qubit di ingresso, che li entangla prima di applicare qualsiasi rotazione addestrabile. Questo entanglement fa sì che il risultato della misura del circuito non dipenda più da ciascun qubit separatamente, ma dalla relazione tra i due, esattamente la proprietà non lineare necessaria per separare lo XOR. Senza quel `CNOT` iniziale, il circuito si comporterebbe come due classificatori indipendenti e fallirebbe sullo XOR, proprio come accade a un percettrone classico.

### Raggruppamento quantistico (pooling): comprimere due qubit nella risposta di uno

Dopo le rotazioni addestrabili, il circuito applica un'operazione di **raggruppamento quantistico** (quantum pooling, la stessa idea del "pooling" di una rete convoluzionale classica, ma per i qubit): combina l'informazione dei due qubit in uno solo tramite una sequenza di rotazioni e un `CNOT` finale, in modo che basti misurare un unico qubit (il "ricevitore") per ottenere la previsione. È la stessa tecnica che ricomparirà nel capitolo sull'apprendimento per rinforzo quantistico per ridurre il numero di qubit da misurare alla fine del circuito.

### Due modi di codificare lo stesso problema

Vedrai due varianti dello stesso circuito nel codice di questo capitolo:

- Codifica per **base computazionale**: ogni bit di ingresso (0 o 1) viene tradotto direttamente in `cirq.I` (non fare nulla) o `cirq.X` (ribaltare il qubit)
- Codifica per **potenza frazionaria**: ogni ingresso viene tradotto in `cirq.X(qubit)**ingresso`, il che permette anche di rappresentare valori continui tra 0 e 1 (non solo 0 e 1 esatti), ed è quella che ti consente di disegnare il confine di decisione completo, non solo i quattro punti discreti dello XOR

## Esempio di codice commentato

```python
import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def codificar_dato(qubits, circuito, dato):
    """X^i = Rx(pi * i): permite codificar valores continuos entre 0 y 1."""
    circuito.append([cirq.X(qubits[0]) ** dato[0]])
    circuito.append([cirq.X(qubits[1]) ** dato[1]])
    return circuito

def generar_datos_xor(qubits):
    datos = [[0, 0], [1, 0], [0, 1], [1, 1]]
    etiquetas = [-1, 1, 1, -1]
    circuitos = [codificar_dato(qubits, cirq.Circuit(), d) for d in datos]
    return tfq.convert_to_tensor(circuitos), np.array(etiquetas)

def rotacion_de_un_qubit(qubit, simbolos):
    return cirq.Circuit(
        cirq.X(qubit) ** simbolos[0],
        cirq.Y(qubit) ** simbolos[1],
        cirq.Z(qubit) ** simbolos[2],
    )

def agrupamiento_dos_qubits(fuente, receptor, simbolos):
    """Combina la información de dos qubits en la respuesta de uno solo (el receptor)."""
    circuito = cirq.Circuit()
    selector_receptor = rotacion_de_un_qubit(receptor, simbolos[0:3])
    selector_fuente = rotacion_de_un_qubit(fuente, simbolos[3:6])
    circuito.append(selector_receptor)
    circuito.append(selector_fuente)
    circuito.append(cirq.CNOT(control=fuente, target=receptor))
    circuito.append(selector_receptor ** -1)
    return circuito

def construir_circuito(qubits):
    x1, y1, z1, x2, y2, z2 = sympy.symbols("x1 y1 z1 x2 y2 z2")
    agrupamiento = sympy.symbols("agrupamiento0:6")
    circuito = cirq.Circuit()
    circuito.append(cirq.CNOT(qubits[0], qubits[1]))  # entrelazamiento clave para el XOR
    circuito.append(cirq.rx(x1).on(qubits[0]))
    circuito.append(cirq.ry(y1).on(qubits[0]))
    circuito.append(cirq.rz(z1).on(qubits[0]))
    circuito.append(cirq.rx(x2).on(qubits[1]))
    circuito.append(cirq.ry(y2).on(qubits[1]))
    circuito.append(cirq.rz(z2).on(qubits[1]))
    circuito += agrupamiento_dos_qubits(qubits[0], qubits[1], agrupamiento)
    return circuito

def precision_hinge(etiqueta_real, prediccion):
    etiqueta_real = tf.squeeze(etiqueta_real) > 0.0
    prediccion = tf.squeeze(prediccion) > 0.0
    return tf.reduce_mean(tf.cast(etiqueta_real == prediccion, tf.float32))

qubits = [cirq.GridQubit(0, i) for i in range(2)]
entrenamiento, etiquetas = generar_datos_xor(qubits)

operador_lectura = [cirq.Z(qubits[1])]
entradas = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
capa_cuantica = tfq.layers.PQC(
    construir_circuito(qubits), operador_lectura, repetitions=32,
    differentiator=tfq.differentiators.ParameterShift(),
    initializer=tf.keras.initializers.Zeros,
)(entradas)
modelo = tf.keras.models.Model(inputs=entradas, outputs=capa_cuantica)
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
               loss=tf.keras.losses.hinge, metrics=[precision_hinge])

historial = modelo.fit(entrenamiento, etiquetas, epochs=512, batch_size=4,
                        validation_data=(entrenamiento, etiquetas))

# Frontera de decisión: barremos todo el plano [-0.1, 1.1] x [-0.1, 1.1]
mapa_colores = ListedColormap(["midnightblue", "maroon"])
N = 150
x1_malla, x2_malla = np.meshgrid(np.linspace(-0.1, 1.1, N), np.linspace(-0.1, 1.1, N))
puntos = np.c_[x1_malla.ravel(), x2_malla.ravel()]
predicciones = [modelo(tfq.convert_to_tensor([codificar_dato(qubits, cirq.Circuit(), p)])).numpy().flatten()[0]
                for p in puntos]
predicciones = np.asarray(predicciones).reshape(x1_malla.shape)

plt.pcolormesh(x1_malla, x2_malla, predicciones, cmap=mapa_colores)
plt.scatter([0, 1], [0, 1], color="blue", label="-1")
plt.scatter([0, 1], [1, 0], color="red", label="1")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```

Nella mappa dei colori finale vedrai due regioni diagonali, una per ciascuna classe: esattamente il confine di decisione curvo che un classificatore lineare classico non riesce a disegnare.

## Riepilogo dei punti chiave

- Lo XOR non è linearmente separabile: serve un confine di decisione curvo, non una retta
- Un `CNOT` iniziale entangla i due qubit di ingresso e fornisce la non linearità necessaria per risolvere lo XOR
- Il raggruppamento quantistico (pooling) combina l'informazione di due qubit nella misura di uno solo
- Codificare i dati con potenze frazionarie (`cirq.X(qubit)**dato`) permette di rappresentare valori continui e di disegnare il confine di decisione completo, non solo i quattro punti discreti

## Mettilo in pratica

Elimina la riga `circuito.append(cirq.CNOT(qubits[0], qubits[1]))` all'inizio di `construir_circuito` (lascia il resto invariato) e riaddestra il modello da zero con le stesse 512 epoche. Disegna di nuovo il confine di decisione e confrontalo con l'originale. Annota quale differenza osservi nell'accuratezza finale e collega il risultato alla spiegazione del perché l'entanglement è indispensabile per risolvere lo XOR.
