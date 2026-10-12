# Capítulo 9 — QAOA y optimización combinatoria

## Por qué importa

Muchos problemas del mundo real son de optimización combinatoria: repartir rutas de reparto, agrupar clientes, diseñar redes de telecomunicaciones. Uno de los problemas de referencia en este campo es **Max-Cut**: dado un grafo (nodos conectados por aristas), dividir los nodos en dos grupos de forma que el número de aristas que quedan "cortadas" (con un nodo en cada grupo) sea el máximo posible. El **Algoritmo Cuántico Aproximado de Optimización** (QAOA) es una de las estrategias más conocidas para abordar este tipo de problemas con un ordenador cuántico NISQ, y en este capítulo lo aplicas exactamente a Max-Cut.

## Explicación conceptual

### Traducir un problema de optimización a un Hamiltoniano

La idea central de QAOA es traducir tu problema de optimización a un **Hamiltoniano de coste**: un operador cuántico cuyo valor esperado, al medirlo sobre un estado concreto, coincide con "cuántas aristas corta" esa división de nodos. Para Max-Cut, ese Hamiltoniano se construye sumando un término `Z_i · Z_j` por cada arista entre los nodos i y j: ese término vale -1 si los dos nodos están en el mismo grupo, y +1 si están en grupos distintos. Cuanto mayor sea el valor esperado del Hamiltoniano, mejor es el corte.

### El circuito QAOA: alternar coste y mezcla

El circuito arranca poniendo todos los qubits en superposición con puertas Hadamard (representando todas las divisiones posibles a la vez) y después alterna dos bloques, cada uno con su propio parámetro entrenable:

- El **Hamiltoniano de coste**, parametrizado por un ángulo que solemos llamar γ (gamma): "empuja" el estado hacia las divisiones que cortan más aristas
- El **Hamiltoniano de mezcla**, parametrizado por un ángulo que solemos llamar β (beta): mezcla el estado para que el circuito pueda explorar otras divisiones además de la que ya parece buena, evitando quedarse atascado en una solución mediocre

Cuantas más veces repitas ese ciclo coste-mezcla (parámetro `p` en la literatura de QAOA), más cerca puede llegar el circuito de la solución óptima, a costa de un circuito más profundo (y recuerda el capítulo 2: más profundidad, más exposición a la decoherencia).

### De la teoría a Cirq: `tfq.util.exponential`

Cirq y TFQ no te obligan a "descomponer a mano" cada Hamiltoniano en puertas cuánticas elementales: `tfq.util.exponential(operadores, coeficientes)` construye directamente el circuito que aplica `exp(-i * coeficiente * operador)`, que es justo la operación que necesitas para aplicar un Hamiltoniano durante un tiempo controlado por el parámetro entrenable.

### Leer la respuesta: muestrear el circuito entrenado

Tras entrenar los parámetros γ y β, mides el circuito muchas veces (con `tfq.layers.Sample`). Cada medida te da una posible división de los nodos en dos grupos; la división que más veces aparece (o la de mayor corte entre las muestras) es tu solución aproximada al problema Max-Cut.

## Ejemplo de código comentado

Usamos un grafo pequeño (un ciclo de 6 nodos) para que el entrenamiento sea rápido en una sesión de clase.

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

Con un grafo tan pequeño puedes calcular el corte máximo real por fuerza bruta y comparar: verás que el corte medio que da QAOA se acerca bastante a ese máximo, aunque casi nunca lo alcanza de forma perfecta con solo `p=1` repetición.

## Resumen de puntos clave

- QAOA traduce un problema de optimización combinatoria (Max-Cut) a un Hamiltoniano de coste, cuyo valor esperado mide la calidad de una solución
- El circuito alterna un bloque de coste (parametrizado por γ) y un bloque de mezcla (parametrizado por β), repetidos `p` veces
- `tfq.util.exponential` construye directamente el circuito que aplica un Hamiltoniano durante un tiempo dado por un parámetro entrenable
- Tras entrenar, se muestrea el circuito muchas veces para obtener soluciones concretas al problema
- Aumentar `p` (más repeticiones del ciclo coste-mezcla) suele mejorar la solución, a costa de un circuito más profundo

## Ponlo en práctica

Cambia `p = 1` por `p = 2` en el ejemplo de código: necesitarás dos símbolos de coste y dos de mezcla (por ejemplo, `sympy.symbols("gamma1 gamma2 beta1 beta2")`) y dos llamadas a `tfq.util.exponential` encadenadas con `+=`, una por cada repetición del ciclo. Entrena de nuevo y compara el corte medio obtenido con el de `p=1`. Anota si la solución mejora, y relaciona el resultado con la idea de que más repeticiones acercan la solución al óptimo.
