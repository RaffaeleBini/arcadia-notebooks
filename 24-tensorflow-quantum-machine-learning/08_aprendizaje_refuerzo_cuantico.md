# Capítulo 8 — Aprendizaje por refuerzo cuántico

## Por qué importa

Hasta ahora has entrenado circuitos cuánticos con datos ya etiquetados (clasificación supervisada). El **aprendizaje por refuerzo** (RL) es distinto: un agente interactúa con un entorno, recibe una recompensa por cada acción y aprende, por prueba y error, la estrategia que maximiza la recompensa acumulada a largo plazo. En este capítulo sustituyes la red neuronal clásica que normalmente aprende esa estrategia por un circuito cuántico variacional, dando lugar al **aprendizaje por refuerzo cuántico** (QRL). Es el primer capítulo del curso en el que el circuito cuántico no clasifica datos estáticos, sino que controla la toma de decisiones de un agente en tiempo real.

## Explicación conceptual

### Q-learning y la ecuación de Bellman, en una frase

En Q-learning, el agente aprende una función Q(estado, acción) que estima la recompensa total esperada si toma esa acción en ese estado y luego actúa de forma óptima. La **ecuación de Bellman** dice que el valor Q de hoy debe ser coherente con la recompensa inmediata más el mejor valor Q posible en el siguiente estado (descontado por un factor `gamma` que da menos peso a las recompensas lejanas). Entrenar el agente consiste en ajustar Q para que esa coherencia se cumpla cada vez mejor.

### El circuito cuántico como aproximador de la función Q

Un **circuito variacional cuántico** (QVC) sustituye a la red neuronal clásica que normalmente representa Q: el estado del entorno (por ejemplo, la posición y velocidad de un carro en el problema CartPole) se codifica como ángulos de rotación de los qubits de entrada, el circuito entrenable procesa esa información, y la medida final de un subconjunto de qubits te da el valor Q estimado para cada acción posible.

### Política epsilon-greedy: explorar frente a explotar

Si el agente siempre eligiera la acción que cree mejor, nunca descubriría estrategias mejores que desconoce. La política **epsilon-greedy** resuelve ese dilema: con probabilidad `epsilon` elige una acción al azar (explorar) y con probabilidad `1 - epsilon` elige la acción que su función Q considera mejor (explotar). `epsilon` empieza alto (mucha exploración) y decae poco a poco a medida que el agente gana experiencia.

### Memoria de repetición (replay buffer)

En vez de aprender únicamente de la última interacción, el agente guarda sus experiencias pasadas (estado, acción, recompensa, siguiente estado) en una **memoria de repetición**, y entrena en cada paso con un lote aleatorio de esa memoria. Esto rompe la correlación entre experiencias consecutivas y hace el entrenamiento mucho más estable; sin ella, el agente tiende a sufrir **olvido catastrófico**: aprende a resolver una situación reciente y, al hacerlo, "olvida" cómo resolver situaciones que dominaba antes.

### Agrupamiento cuántico para reducir dimensiones

Igual que en el capítulo del XOR, aquí se reutiliza el **agrupamiento cuántico** (quantum pooling): el circuito parte de más qubits de los que necesitas medir al final, y una capa de agrupamiento comprime esa información en un número menor de qubits de salida (uno por cada acción posible del agente).

### Más allá de este capítulo: QRL con visión artificial

El mismo principio (sustituir la red que aproxima Q por un circuito cuántico) se puede aplicar a entornos mucho más complejos, como videojuegos de Atari, donde el agente recibe píxeles en vez de un estado numérico simple, y hace falta combinar el circuito cuántico con una red convolucional clásica que primero comprima la imagen. Ese caso queda fuera del alcance práctico de este curso por la complejidad añadida (una CNN grande, mucha más memoria y tiempo de entrenamiento), pero merece la pena que sepas que la misma idea se escala en esa dirección.

## Ejemplo de código comentado

Este ejemplo entrena un agente QDQN (red cuántica profunda Q) para resolver CartPole, el entorno clásico donde hay que mantener un palo en equilibrio moviendo un carro a izquierda o derecha.

```python
import gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy

class AgenteQDQN:
    def __init__(self, num_acciones, num_variables_estado):
        self.num_acciones = num_acciones
        self.qubits = [cirq.GridQubit(0, i) for i in range(4)]
        self.red_q = self._construir_red()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.memoria = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decaimiento = 0.9
        self.epsilon_minimo = 0.01

    def _rotacion_de_un_qubit(self, qubit, simbolos):
        return cirq.Circuit(cirq.X(qubit) ** simbolos[0], cirq.Y(qubit) ** simbolos[1], cirq.Z(qubit) ** simbolos[2])

    def _agrupamiento(self, fuente, receptor, simbolos):
        circuito = cirq.Circuit()
        selector_receptor = self._rotacion_de_un_qubit(receptor, simbolos[0:3])
        selector_fuente = self._rotacion_de_un_qubit(fuente, simbolos[3:6])
        circuito.append(selector_receptor)
        circuito.append(selector_fuente)
        circuito.append(cirq.CNOT(control=fuente, target=receptor))
        circuito.append(selector_receptor ** -1)
        return circuito

    def _capa(self, pesos, qubits):
        capa = cirq.Circuit()
        for i in range(len(qubits) - 1):
            capa.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        capa.append([cirq.Moment([cirq.rx(pesos[j]).on(qubits[j]) for j in range(4)])])
        capa.append([cirq.Moment([cirq.ry(pesos[j + 4]).on(qubits[j]) for j in range(4)])])
        capa.append([cirq.Moment([cirq.rz(pesos[j + 8]).on(qubits[j]) for j in range(4)])])
        return capa

    def _construir_circuito(self):
        simbolos = sympy.symbols("q0:48")  # 3 capas x 12 + 2 agrupamientos x 6 = 48
        circuito = cirq.Circuit()
        circuito += self._capa(simbolos[:12], self.qubits)
        circuito += self._capa(simbolos[12:24], self.qubits)
        circuito += self._capa(simbolos[24:36], self.qubits)
        circuito += self._agrupamiento(self.qubits[0], self.qubits[2], simbolos[36:42])
        circuito += self._agrupamiento(self.qubits[1], self.qubits[3], simbolos[42:])
        return circuito

    def _construir_red(self):
        operadores_lectura = [cirq.Z(self.qubits[i]) for i in range(2, 4)]  # 2 acciones posibles
        entradas = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        capa_cuantica = tfq.layers.PQC(
            self._construir_circuito(), operadores_lectura,
            differentiator=tfq.differentiators.ParameterShift(),
            initializer=tf.keras.initializers.Zeros,
        )(entradas)
        return tf.keras.Model(inputs=entradas, outputs=capa_cuantica)

    def _codificar_estado(self, estado_clasico):
        circuito = cirq.Circuit()
        for i, valor in enumerate(estado_clasico):
            angulo = 0 if valor < 0 else 1
            circuito.append(cirq.rx(np.pi * angulo).on(self.qubits[i]))
            circuito.append(cirq.rz(np.pi * angulo).on(self.qubits[i]))
        return circuito

    def elegir_accion(self, estado):
        if random.random() < self.epsilon:
            return np.random.choice(self.num_acciones)
        entrada = tfq.convert_to_tensor([self._codificar_estado(estado)])
        return np.argmax(self.red_q.predict(entrada, verbose=0))

    def recordar(self, estado, accion, recompensa, siguiente_estado, terminado):
        self.memoria.append((estado, accion, recompensa, siguiente_estado, terminado))

    def entrenar(self, tamano_lote=32):
        if len(self.memoria) < tamano_lote:
            return
        lote = random.sample(self.memoria, tamano_lote)
        estados = tfq.convert_to_tensor([self._codificar_estado(e[0]) for e in lote])
        siguientes = tfq.convert_to_tensor([self._codificar_estado(e[3]) for e in lote])
        acciones = [[i, lote[i][1]] for i in range(tamano_lote)]
        recompensas = tf.convert_to_tensor([[e[2]] for e in lote], dtype=tf.float32)
        terminados = tf.convert_to_tensor([[float(e[4])] for e in lote], dtype=tf.float32)

        with tf.GradientTape() as cinta:
            q_siguiente = tf.reduce_max(self.red_q(siguientes), axis=1, keepdims=True)
            objetivo = recompensas + (1 - terminados) * self.gamma * q_siguiente
            q_actual = tf.gather_nd(self.red_q(estados, training=True), acciones)
            error = tf.reduce_mean(tf.square(objetivo - tf.reshape(q_actual, [tamano_lote, 1])))

        gradientes = cinta.gradient(error, self.red_q.trainable_variables)
        self.opt.apply_gradients(zip(gradientes, self.red_q.trainable_variables))
        if self.epsilon > self.epsilon_minimo:
            self.epsilon *= self.epsilon_decaimiento

# Bucle de entrenamiento reducido (menos episodios que el experimento original, para caber en clase)
entorno = gym.make("CartPole-v1")
agente = AgenteQDQN(entorno.action_space.n, entorno.observation_space.shape[0])
recompensas_por_episodio = []

for episodio in range(50):
    estado = entorno.reset()
    recompensa_total = 0
    terminado = False
    while not terminado:
        accion = agente.elegir_accion(estado)
        siguiente_estado, recompensa, terminado, _ = entorno.step(accion)
        recompensa_total += recompensa
        agente.recordar(estado, accion, recompensa, siguiente_estado, terminado)
        estado = siguiente_estado
        if terminado:
            agente.entrenar()
    recompensas_por_episodio.append(recompensa_total)
    print(f"Episodio {episodio}: recompensa {recompensa_total}, epsilon {agente.epsilon:.2f}")
```

Con solo 50 episodios (frente a los cientos que suele necesitar un entrenamiento completo) ya deberías notar que la recompensa por episodio empieza a subir: el agente aguanta el palo en equilibrio más tiempo según avanza el entrenamiento.

## Resumen de puntos clave

- Q-learning aprende una función Q(estado, acción) que respeta la ecuación de Bellman; un circuito variacional cuántico puede sustituir a la red neuronal clásica que normalmente representa esa función
- La política epsilon-greedy equilibra exploración (acciones al azar) y explotación (la mejor acción conocida), con `epsilon` decayendo con el tiempo
- La memoria de repetición estabiliza el entrenamiento y evita el olvido catastrófico
- El agrupamiento cuántico reduce el número de qubits que hace falta medir al final del circuito
- El mismo enfoque se puede escalar a entornos con entrada visual (Atari), combinando el circuito cuántico con una red convolucional clásica, aunque eso queda fuera del alcance práctico de este curso

## Ponlo en práctica

En la función `elegir_accion`, sustituye la política epsilon-greedy por una política totalmente aleatoria (elimina la condición `if random.random() < self.epsilon` y haz que siempre devuelva `np.random.choice(self.num_acciones)`). Entrena de nuevo al agente durante 50 episodios y compara la curva de recompensas con la del ejemplo original. Explica en una frase por qué la exploración pura, sin explotación, no permite que el agente mejore con el tiempo.
