# Capítulo 10 — VQE: fundamentos del cálculo de energías cuánticas

## Por qué importa

Uno de los usos más prometedores de la computación cuántica es la simulación de sistemas físicos y moléculas, algo que un ordenador clásico hace muy mal porque los estados cuánticos crecen exponencialmente (recuerda el capítulo 4). El **Solucionador Variacional Cuántico de Autovalores** (VQE) es el algoritmo de referencia para calcular la energía del estado fundamental (el nivel de energía más bajo posible) de un sistema cuántico, y es la base de aplicaciones reales en química cuántica, como calcular la energía de una molécula de hidrógeno.

## Explicación conceptual

### Un Hamiltoniano es una "receta" de energía

En física cuántica, el **Hamiltoniano** de un sistema es el operador que describe su energía total. Cualquier Hamiltoniano se puede descomponer como una suma ponderada de productos tensoriales de las matrices de Pauli (I, X, Y, Z que ya conoces de los capítulos anteriores). Por ejemplo, un Hamiltoniano de un único qubit podría escribirse como `H = a·I + b·X + c·Y + d·Z`, donde a, b, c, d son números reales que dependen del sistema físico concreto que quieras describir. El objetivo de VQE es encontrar el valor mínimo posible del valor esperado de ese Hamiltoniano: esa es, por definición, la energía del estado fundamental.

### El principio variacional: adivinar y mejorar

VQE se apoya en el **principio variacional**: para cualquier estado cuántico que pruebes, el valor esperado del Hamiltoniano sobre ese estado nunca es menor que la energía real del estado fundamental. Eso significa que puedes tratar la búsqueda de la energía mínima como un problema de optimización: preparas un circuito con parámetros ajustables (el "ansatz", tu mejor conjetura de cómo debería verse el estado fundamental), mides el valor esperado del Hamiltoniano, y ajustas los parámetros para que ese valor baje todo lo posible. Cuanto más bajo consigas llevarlo, más cerca estás de la energía real.

### Midiendo un Hamiltoniano con varios términos

Un Hamiltoniano de varios qubits, como `H = a·(Z₀+Z₁) + b·(Z₂+Z₃)`, se mide término a término: para cada término del Hamiltoniano, mides el circuito en la base correspondiente (rotando antes de medir si el término incluye X o Y, ya que solo puedes medir directamente en la base Z), multiplicas por su coeficiente y sumas todos los resultados. TFQ hace esto de forma transparente: le pasas la lista de términos como "operador de lectura" de una capa `PQC`, y se encarga de las mediciones y la suma internamente.

### Optimizadores clásicos: gradiente frente a caja negra

Puedes optimizar los parámetros del ansatz de dos formas:

- **Basado en gradiente**: usando `tf.GradientTape` y un optimizador como Adam, igual que en los capítulos anteriores, siempre que puedas calcular el gradiente del circuito (con `ParameterShift`, por ejemplo)
- **Optimización de caja negra**: tratando el circuito como una función que solo puedes evaluar (sin gradiente), y usando algoritmos como `scipy.optimize.minimize` con el método Nelder-Mead, que prueba distintas combinaciones de parámetros y converge sin necesitar derivadas

Ambos enfoques son válidos; en el próximo capítulo verás además el algoritmo Rotosolve, diseñado específicamente para circuitos cuánticos parametrizados.

## Ejemplo de código comentado

Primero, un Hamiltoniano sencillo de un único qubit con un ansatz mínimo, optimizado con Nelder-Mead (sin necesidad de calcular gradientes):

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

Ahora, un Hamiltoniano de cuatro qubits con dos términos independientes, entrenado por gradiente:

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

## Resumen de puntos clave

- Cualquier Hamiltoniano se descompone en una suma de productos de matrices de Pauli (I, X, Y, Z); VQE busca minimizar su valor esperado
- El principio variacional garantiza que ese mínimo, al bajarlo todo lo posible, se aproxima a la energía real del estado fundamental
- Un Hamiltoniano con varios términos se mide término a término, y TFQ suma los resultados automáticamente al pasarlos como lista al `PQC`
- Puedes optimizar por gradiente (Adam + `ParameterShift`/`CentralDifference`) o con optimizadores de caja negra como Nelder-Mead, sin necesitar gradientes

## Ponlo en práctica

En el primer ejemplo (Hamiltoniano de un único qubit), cambia el Hamiltoniano de `1 + 2 * cirq.Z(qubit)` a `1 + 2 * cirq.X(qubit)`. Antes de ejecutar nada, piensa: el ansatz actual solo tiene una rotación `ry`, que nunca puede llevar el qubit a un autoestado de X. Añade una puerta `cirq.rz` (con un segundo parámetro entrenable) al ansatz para que el circuito tenga suficiente libertad de girar hacia cualquier eje, y vuelve a optimizar con Nelder-Mead. Comprueba que ahora sí converge a la energía mínima teórica (-1).
