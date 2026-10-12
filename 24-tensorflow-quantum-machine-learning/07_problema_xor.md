# Capítulo 7 — Resolviendo el XOR con machine learning cuántico

## Por qué importa

El problema XOR (o-exclusivo) es célebre en el mundo del machine learning clásico: un único perceptrón (la neurona más simple) no puede resolverlo, porque sus cuatro puntos no se pueden separar con una única línea recta. Hace falta una red con al menos una capa oculta para resolverlo. En este capítulo vas a comprobar que un circuito cuántico con solo dos qubits, gracias al entrelazamiento, sí puede separar el XOR sin necesitar "capas ocultas" clásicas. Es la demostración más clara de que el entrelazamiento no es solo una curiosidad teórica: es un recurso computacional real.

## Explicación conceptual

### Por qué el XOR no es linealmente separable

El XOR tiene cuatro combinaciones de entrada (0-0, 1-0, 0-1, 1-1) con salidas (-1, +1, +1, -1). Si dibujas esos cuatro puntos en un plano, verás que no existe ninguna línea recta que deje todos los "+1" a un lado y todos los "-1" al otro: las dos clases están entrelazadas geométricamente, en diagonal. Necesitas una función no lineal (una curva, no una recta) para separarlas.

### Cómo el entrelazamiento aporta esa "curva"

El circuito de este capítulo empieza con una puerta `CNOT` entre los dos qubits de entrada, que los entrelaza antes de aplicar ninguna rotación entrenable. Ese entrelazamiento hace que el resultado de medir el circuito ya no dependa de cada qubit por separado, sino de la relación entre ambos, exactamente la propiedad no lineal que necesitas para separar el XOR. Sin ese `CNOT` inicial, el circuito se comportaría como dos clasificadores independientes y fallaría en el XOR, igual que le pasa a un perceptrón clásico.

### Agrupamiento cuántico (pooling): comprimir dos qubits en la respuesta de uno

Tras las rotaciones entrenables, el circuito aplica una operación de **agrupamiento cuántico** (quantum pooling, la misma idea que el "pooling" de una red convolucional clásica, pero para qubits): combina la información de los dos qubits en uno solo mediante una secuencia de rotaciones y un `CNOT` final, de forma que solo necesitas medir un único qubit (el "receptor") para obtener la predicción. Esta es la misma técnica que reaparecerá en el capítulo de aprendizaje por refuerzo cuántico para reducir el número de qubits que hay que medir al final del circuito.

### Dos formas de codificar el mismo problema

Vas a ver dos variantes del mismo circuito en el código de este capítulo:

- Codificación por **base computacional**: cada bit de entrada (0 o 1) se traduce directamente a `cirq.I` (no hacer nada) o `cirq.X` (voltear el qubit)
- Codificación por **potencia fraccionaria**: cada entrada se traduce a `cirq.X(qubit)**entrada`, lo que permite además representar valores continuos entre 0 y 1 (no solo 0 y 1 exactos), y es la que te permite dibujar la frontera de decisión completa, no solo los cuatro puntos discretos del XOR

## Ejemplo de código comentado

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

En el mapa de colores final verás dos regiones diagonales, una para cada clase: exactamente la frontera de decisión curva que un clasificador lineal clásico no puede dibujar.

## Resumen de puntos clave

- El XOR no es linealmente separable: se necesita una frontera de decisión curva, no una recta
- Un `CNOT` inicial entrelaza los dos qubits de entrada y aporta la no linealidad necesaria para resolver el XOR
- El agrupamiento cuántico (pooling) combina la información de dos qubits en la medida de uno solo
- Codificar los datos con potencias fraccionarias (`cirq.X(qubit)**dato`) permite representar valores continuos y dibujar la frontera de decisión completa, no solo los cuatro puntos discretos

## Ponlo en práctica

Elimina la línea `circuito.append(cirq.CNOT(qubits[0], qubits[1]))` del principio de `construir_circuito` (deja el resto igual) y vuelve a entrenar el modelo desde cero con las mismas 512 épocas. Dibuja de nuevo la frontera de decisión y compárala con la original. Anota qué diferencia observas en la precisión final y relaciona el resultado con la explicación de por qué el entrelazamiento es imprescindible para resolver el XOR.
