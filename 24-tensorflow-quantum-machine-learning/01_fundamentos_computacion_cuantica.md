# Capítulo 1 — Qué es la computación cuántica

## Por qué importa

Imagina que buscas la salida de un laberinto. Un ordenador clásico prueba un camino, se choca con una pared, retrocede y prueba el siguiente: uno detrás de otro. Un ordenador cuántico, en cambio, explora muchos caminos a la vez gracias a un fenómeno llamado superposición. Esa diferencia no es un simple truco de velocidad: abre la puerta a resolver problemas de química, optimización y aprendizaje automático que a un ordenador clásico le llevarían miles de años.

Google AI Quantum, el equipo detrás de procesadores cuánticos como Sycamore, lleva años invirtiendo en esta tecnología precisamente porque promete resolver ese tipo de problemas. Y TensorFlow Quantum, la herramienta que usas en este curso, nace de unir ese hardware cuántico con el machine learning que ya conoces (o que vas a conocer) del mundo clásico. Antes de escribir una sola línea de código, necesitas entender tres ideas: el qubit, la superposición y el entrelazamiento.

## Explicación conceptual

### El bit clásico frente al qubit

Un bit clásico es como un interruptor: está en 0 o está en 1, nunca en ambos a la vez. Un qubit (bit cuántico) es como una moneda que gira en el aire: mientras gira, no es ni cara ni cruz, es una combinación de las dos. Solo cuando la moneda cae —cuando "mides" el qubit— se decide si el resultado es 0 o 1.

Esa combinación se llama **superposición**. Matemáticamente, un qubit se describe como una mezcla ponderada de los estados 0 y 1, y esos pesos determinan la probabilidad de obtener cada resultado al medir. Cuantos más qubits combinas en superposición, más combinaciones puedes representar a la vez: con solo 10 qubits ya representas 1.024 combinaciones simultáneas, algo que a un ordenador clásico le costaría 1.024 pasos independientes.

### Puertas cuánticas: cómo manipulas un qubit

Así como un ordenador clásico usa puertas lógicas (AND, OR, NOT) para transformar bits, un ordenador cuántico usa **puertas cuánticas** para transformar qubits. La diferencia clave es que las puertas cuánticas son reversibles: siempre puedes deshacer la operación. Las puertas más habituales son:

- **Puerta Hadamard (H)**: coloca un qubit en superposición perfecta (50% de probabilidad de 0, 50% de probabilidad de 1)
- **Puertas X, Y, Z**: rotan el estado del qubit sobre los tres ejes de una esfera imaginaria (la esfera de Bloch)
- **Puerta CNOT**: actúa sobre dos qubits a la vez y es la base del entrelazamiento

### Entrelazamiento: cuando dos qubits dejan de ser independientes

Cuando entrelazas dos qubits, el estado de uno queda ligado al estado del otro, sin importar la distancia física entre ellos. Si mides el primer qubit y obtienes un 0, automáticamente sabes qué valor va a salir en el segundo, aunque no lo hayas medido todavía. Einstein lo llamó "acción fantasmal a distancia" porque le resultaba perturbador, pero hoy es una herramienta de trabajo diaria en los laboratorios de computación cuántica. El entrelazamiento es el ingrediente que permite que los qubits colaboren entre sí para resolver un problema, en lugar de comportarse como bits independientes.

### Medir: el paso que lo cambia todo

Mientras no mides un qubit, vive en superposición. En el instante en que lo mides, ese qubit "colapsa" a un valor concreto (0 o 1) y pierde toda la información de la superposición anterior. Este comportamiento no tiene equivalente en la computación clásica y es la razón por la que programar en cuántica requiere pensar de forma distinta: diseñas un circuito para que, al medir al final, la probabilidad de obtener la respuesta correcta sea alta.

## Ejemplo de código comentado

Vas a usar Cirq, la librería de Google para programar circuitos cuánticos (la usarás en detalle en el próximo capítulo). Aquí tienes un primer vistazo: un qubit que pones en superposición con una puerta Hadamard y después mides.

```python
import cirq

# Un qubit físico situado en la posición (0, 0) de una rejilla
qubit = cirq.GridQubit(0, 0)

# Un circuito es la secuencia de operaciones que aplicas al qubit
circuito = cirq.Circuit()

# La puerta Hadamard pone el qubit en superposición: 50% 0, 50% 1
circuito.append(cirq.H(qubit))

# Medimos el qubit y guardamos el resultado con la clave "resultado"
circuito.append(cirq.measure(qubit, key="resultado"))

print(circuito)

# Simulamos el circuito 1000 veces para ver la distribución de resultados
simulador = cirq.Simulator()
resultados = simulador.run(circuito, repetitions=1000)

print(resultados.histogram(key="resultado"))
# Debería salir aproximadamente 500 veces "0" y 500 veces "1"
```

Fíjate en algo importante: cada vez que ejecutas el circuito completo (superposición + medida), obtienes un único 0 o un único 1, nunca los dos a la vez. Por eso repites el experimento 1.000 veces: así compruebas que, en conjunto, la moneda cayó cara la mitad de las veces y cruz la otra mitad, tal y como predice la superposición.

## Resumen de puntos clave

- Un qubit puede estar en superposición: una combinación de 0 y 1 al mismo tiempo, hasta que lo mides
- Las puertas cuánticas transforman qubits de forma reversible; la puerta Hadamard crea superposición y la puerta CNOT ayuda a crear entrelazamiento
- El entrelazamiento liga el estado de varios qubits entre sí, incluso a distancia
- Medir un qubit colapsa su superposición a un único valor concreto (0 o 1)
- Estos tres fenómenos (superposición, puertas y entrelazamiento) son la base de todo lo que construyes en TensorFlow Quantum

## Ponlo en práctica

Modifica el ejemplo de código: en vez de un único qubit con una puerta Hadamard, crea un circuito con dos qubits, aplica una puerta Hadamard al primero y después una puerta `cirq.CNOT(qubit1, qubit2)` para entrelazarlos. Mide ambos qubits y ejecuta el circuito 1.000 veces. Observa el histograma de resultados: comprueba que solo aparecen las combinaciones "00" y "11", nunca "01" ni "10". Eso es el entrelazamiento en acción: los dos qubits siempre coinciden.
