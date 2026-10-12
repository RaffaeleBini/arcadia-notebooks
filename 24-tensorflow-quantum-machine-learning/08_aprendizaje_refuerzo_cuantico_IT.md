# Capitolo 8 — Apprendimento per rinforzo quantistico

## Perché è importante

Finora hai addestrato circuiti quantistici con dati già etichettati (classificazione supervisionata). L'**apprendimento per rinforzo** (RL) è diverso: un agente interagisce con un ambiente, riceve una ricompensa per ogni azione e impara, per tentativi ed errori, la strategia che massimizza la ricompensa accumulata nel lungo periodo. In questo capitolo sostituisci la rete neurale classica che normalmente impara quella strategia con un circuito quantistico variazionale, dando vita all'**apprendimento per rinforzo quantistico** (QRL). È il primo capitolo del corso in cui il circuito quantistico non classifica dati statici, ma controlla il processo decisionale di un agente in tempo reale.

## Spiegazione concettuale

### Q-learning e l'equazione di Bellman, in breve

Nel Q-learning, l'agente impara una funzione Q(stato, azione) che stima la ricompensa totale attesa se compie quell'azione in quello stato e agisce in modo ottimale da lì in poi. L'**equazione di Bellman** dice che il valore Q di oggi deve essere coerente con la ricompensa immediata più il miglior valore Q possibile nello stato successivo (scontato di un fattore `gamma` che dà meno peso alle ricompense lontane). Addestrare l'agente significa regolare Q in modo che questa coerenza sia rispettata sempre meglio.

### Il circuito quantistico come approssimatore della funzione Q

Un **circuito variazionale quantistico** (QVC) sostituisce la rete neurale classica che normalmente rappresenta Q: lo stato dell'ambiente (per esempio, la posizione e la velocità di un carrello nel problema CartPole) viene codificato come angoli di rotazione dei qubit di ingresso, il circuito addestrabile elabora questa informazione, e la misura finale di un sottoinsieme di qubit ti dà il valore Q stimato per ciascuna azione possibile.

### Politica epsilon-greedy: esplorare contro sfruttare

Se l'agente scegliesse sempre l'azione che ritiene migliore, non scoprirebbe mai strategie migliori di quelle che già conosce. La politica **epsilon-greedy** risolve questo dilemma: con probabilità `epsilon` sceglie un'azione a caso (esplorare) e con probabilità `1 - epsilon` sceglie l'azione che la sua funzione Q considera migliore (sfruttare). `epsilon` parte da un valore alto (molta esplorazione) e diminuisce gradualmente man mano che l'agente acquisisce esperienza.

### Memoria di replay (replay buffer)

Invece di imparare solo dall'ultima interazione, l'agente conserva le sue esperienze passate (stato, azione, ricompensa, stato successivo) in una **memoria di replay**, e ad ogni passo si addestra su un lotto casuale di quella memoria. Questo spezza la correlazione tra esperienze consecutive e rende l'addestramento molto più stabile; senza di essa, l'agente tende a soffrire di **oblio catastrofico**: impara a risolvere una situazione recente e, nel farlo, "dimentica" come risolvere situazioni che padroneggiava prima.

### Raggruppamento quantistico per ridurre le dimensioni

Come nel capitolo sullo XOR, qui si riutilizza il **raggruppamento quantistico** (quantum pooling): il circuito parte da più qubit di quanti ne serviranno da misurare alla fine, e uno strato di raggruppamento comprime quell'informazione in un numero minore di qubit di uscita (uno per ogni azione possibile dell'agente).

### Oltre questo capitolo: QRL con visione artificiale

Lo stesso principio (sostituire la rete che approssima Q con un circuito quantistico) si può applicare ad ambienti molto più complessi, come i videogiochi Atari, dove l'agente riceve pixel invece di uno stato numerico semplice, e occorre combinare il circuito quantistico con una rete convoluzionale classica che prima comprima l'immagine. Questo caso esula dalla portata pratica di questo corso per la complessità aggiuntiva (una CNN grande, molta più memoria e tempo di addestramento), ma vale la pena sapere che la stessa idea si estende in quella direzione.

## Esempio di codice commentato

Questo esempio addestra un agente QDQN (rete quantistica profonda Q) per risolvere CartPole, l'ambiente classico in cui bisogna mantenere in equilibrio un'asta muovendo un carrello a sinistra o a destra.

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

Con solo 50 episodi (contro le centinaia che di solito servono per un addestramento completo) dovresti già notare che la ricompensa per episodio inizia a salire: l'agente tiene l'asta in equilibrio più a lungo man mano che l'addestramento procede.

## Riepilogo dei punti chiave

- Il Q-learning impara una funzione Q(stato, azione) che rispetta l'equazione di Bellman; un circuito variazionale quantistico può sostituire la rete neurale classica che normalmente rappresenta quella funzione
- La politica epsilon-greedy bilancia esplorazione (azioni casuali) e sfruttamento (la migliore azione nota), con `epsilon` che decresce nel tempo
- La memoria di replay stabilizza l'addestramento ed evita l'oblio catastrofico
- Il raggruppamento quantistico riduce il numero di qubit da misurare alla fine del circuito
- Lo stesso approccio si può estendere ad ambienti con input visivo (Atari), combinando il circuito quantistico con una rete convoluzionale classica, anche se ciò esula dalla portata pratica di questo corso

## Mettilo in pratica

Nella funzione `elegir_accion`, sostituisci la politica epsilon-greedy con una politica completamente casuale (elimina la condizione `if random.random() < self.epsilon` e fai in modo che restituisca sempre `np.random.choice(self.num_acciones)`). Addestra di nuovo l'agente per 50 episodi e confronta la curva delle ricompense con quella dell'esempio originale. Spiega in una frase perché l'esplorazione pura, senza sfruttamento, non permette all'agente di migliorare nel tempo.
