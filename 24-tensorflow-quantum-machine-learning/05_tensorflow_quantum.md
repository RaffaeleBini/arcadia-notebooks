# Capítulo 5 — TensorFlow Quantum y el problema de las mesetas estériles

## Por qué importa

Ya sabes programar circuitos cuánticos con Cirq. Ahora necesitas la pieza que le falta: cómo entrenar esos circuitos con las mismas herramientas de machine learning que usarías para entrenar una red neuronal clásica (descenso de gradiente, funciones de pérdida, optimizadores como Adam). Esa pieza es **TensorFlow Quantum** (TFQ), la plataforma que conecta Cirq con Keras. Pero antes de lanzarte a entrenar, tienes que conocer un obstáculo que aparece justo cuando los circuitos cuánticos crecen: el problema de las **mesetas estériles** (barren plateaus), que puede hacer que el entrenamiento simplemente no aprenda nada.

## Explicación conceptual

### TFQ: el puente entre Cirq y Keras

TFQ convierte un circuito de Cirq en un tensor de TensorFlow (con `tfq.convert_to_tensor`) para que pueda fluir por una red de Keras como cualquier otro dato. A partir de ahí, dispones de capas especiales:

- `tfq.layers.PQC` (circuito cuántico parametrizado): convierte un circuito con símbolos en una capa de Keras entrenable. Tú le dices qué circuito usar y qué "operador de lectura" medir (por ejemplo, `cirq.Z(qubit)`), y la capa se encarga de ajustar los ángulos del circuito durante el entrenamiento, igual que una capa densa ajusta sus pesos
- `tfq.layers.Expectation` y `tfq.layers.SampledExpectation`: calculan el valor esperado de una medida sobre un circuito. La versión "Sampled" simula el ruido de un hardware real, repitiendo el circuito un número finito de veces (`repetitions`) en vez de calcular la probabilidad exacta
- `tfq.differentiators.ParameterShift`: el método que usa TFQ para calcular gradientes de un circuito cuántico. A diferencia de una red clásica, en un circuito cuántico no puedes "derivar" una puerta cuántica con las reglas habituales de cálculo; el desplazamiento de parámetro (parameter shift) rodea ese problema evaluando el circuito con el parámetro ligeramente desplazado hacia arriba y hacia abajo, y combinando ambos resultados

Con estas piezas, entrenar un circuito cuántico se parece mucho a entrenar una red clásica: defines un modelo de Keras, compilas con una función de pérdida y un optimizador, y llamas a `.fit(...)`.

### El problema de las mesetas estériles

Cuando aumentas el número de qubits o la profundidad de un circuito cuántico parametrizado inicializado al azar, ocurre algo contraintuitivo: el gradiente (la señal que le dice al optimizador hacia dónde moverse) se vuelve, en promedio, prácticamente cero en casi todo el espacio de parámetros. A esta situación se la llama **meseta estéril**: el paisaje de la función de pérdida es tan plano que el optimizador no sabe hacia dónde avanzar, como intentar bajar una montaña en una niebla espesa sobre un terreno completamente llano.

Esto no es un fallo de tu código: es una propiedad matemática de los circuitos cuánticos aleatorios y profundos. Cuantos más qubits añades, más rápido decae la varianza del gradiente, y ese decaimiento es exponencial. Es la razón por la que, en este curso, vas a trabajar con circuitos deliberadamente poco profundos y con pocos qubits: cuantos menos qubits y menos capas, menos grave es el problema.

### Aprendizaje por capas (layer-wise learning): una salida a la meseta

Una estrategia para mitigar las mesetas estériles es el **aprendizaje por capas**: en vez de inicializar todo el circuito al azar y entrenar todos los parámetros a la vez, entrenas primero una capa poco profunda, la "congelas" (o la dejas cerca de sus valores entrenados) y añades una nueva capa poco a poco. De esta forma, el circuito nunca pasa por una fase de profundidad alta con parámetros completamente aleatorios, que es justo la combinación que dispara la meseta estéril.

También ayuda elegir con cuidado qué mides: en el ejemplo de código de este capítulo comprobarás que medir la suma de varios operadores Z (uno por qubit) sufre menos mesetas estériles que medir un único operador que combine varios qubits a la vez (por ejemplo, Z⊗Z).

## Ejemplo de código comentado

Este ejemplo reproduce el experimento clásico de las mesetas estériles: genera circuitos aleatorios de profundidad creciente y mide cómo se hunde la varianza del gradiente. Usamos pocos qubits y pocas repeticiones para que se ejecute en pocos minutos, pero la tendencia ya es visible.

```python
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt

def generar_circuito_aleatorio(qubits, profundidad, parametro):
    """Un circuito con una puerta aleatoria distinta en cada qubit y en cada capa,
    entrelazado con puertas CZ. Solo el primer parámetro es simbólico (entrenable);
    el resto son ángulos aleatorios fijos, tal y como exige el experimento original."""
    circuito = cirq.Circuit()
    for qubit in qubits:
        circuito += cirq.ry(np.pi / 4.0)(qubit)

    for capa in range(profundidad):
        for i, qubit in enumerate(qubits):
            es_el_parametro_entrenable = (i == 0 and capa == 0)
            angulo = parametro if es_el_parametro_entrenable else np.random.uniform(0, 2 * np.pi)
            eleccion = np.random.uniform()
            if eleccion > 2 / 3:
                circuito += cirq.rz(angulo)(qubit)
            elif eleccion > 1 / 3:
                circuito += cirq.ry(angulo)(qubit)
            else:
                circuito += cirq.rx(angulo)(qubit)
        for origen, destino in zip(qubits, qubits[1:]):
            circuito += cirq.CZ(origen, destino)
    return circuito

def varianza_del_gradiente(num_qubits, profundidad, num_circuitos, repeticiones):
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    simbolo = sympy.symbols("parametro")
    circuitos = [generar_circuito_aleatorio(qubits, profundidad, simbolo) for _ in range(num_circuitos)]

    operador_lectura = [sum(cirq.Z(q) for q in qubits)]  # suma de Z, uno por qubit
    diferenciador = tfq.differentiators.ParameterShift()
    capa_expectativa = tfq.layers.SampledExpectation(differentiator=diferenciador)

    circuitos_tensor = tfq.convert_to_tensor(circuitos)
    valores = tf.convert_to_tensor(np.random.uniform(0, 2 * np.pi, (num_circuitos, 1)).astype(np.float32))

    with tf.GradientTape() as cinta:
        cinta.watch(valores)
        salida = capa_expectativa(circuitos_tensor, operators=operador_lectura,
                                   repetitions=repeticiones, symbol_names=[simbolo], symbol_values=valores)

    gradientes = cinta.gradient(salida, valores)
    return tf.math.reduce_std(gradientes, axis=0).numpy()[0]

# Reducido respecto al experimento original para caber en una sesión de clase
num_circuitos = 20
repeticiones = 500
tamanos_qubits = [2, 4, 6, 8]
profundidad_fija = 20

varianzas = [varianza_del_gradiente(n, profundidad_fija, num_circuitos, repeticiones) for n in tamanos_qubits]

plt.plot(tamanos_qubits, varianzas, marker="o")
plt.xlabel("Número de qubits")
plt.ylabel("Varianza del gradiente")
plt.title("La meseta estéril: la varianza cae al añadir qubits")
plt.show()
```

Al ejecutarlo verás que la varianza del gradiente cae claramente según aumentas el número de qubits, incluso con esta versión reducida del experimento. Con más qubits y más repeticiones (como en el experimento original, publicado por el propio equipo de TensorFlow Quantum) la caída es todavía más pronunciada.

## Resumen de puntos clave

- TFQ conecta Cirq con Keras: `tfq.layers.PQC` convierte un circuito con símbolos en una capa entrenable, y `tfq.differentiators.ParameterShift` calcula sus gradientes
- `tfq.layers.Expectation` calcula el valor esperado exacto; `tfq.layers.SampledExpectation` lo aproxima simulando un número finito de repeticiones, como en hardware real
- Las mesetas estériles son una caída exponencial de la varianza del gradiente al aumentar qubits o profundidad en circuitos inicializados al azar, y hacen que el optimizador no sepa hacia dónde avanzar
- El aprendizaje por capas (entrenar poco a poco, capa a capa, en vez de todo el circuito a la vez desde el principio) es una estrategia para mitigar este problema
- Por eso, en este curso, los circuitos que vas a entrenar son deliberadamente pequeños y poco profundos

## Ponlo en práctica

En el ejemplo de código, cambia el operador de lectura de `sum(cirq.Z(q) for q in qubits)` a un único operador `cirq.Z(qubits[0]) * cirq.Z(qubits[1])` (el producto de Z sobre solo dos qubits). Vuelve a ejecutar el barrido de tamaños de qubits y compara la nueva curva de varianza con la original. Anota si la caída es más rápida, más lenta o igual, y relaciona lo que observas con la recomendación de "elegir con cuidado qué mides" del apartado conceptual.
