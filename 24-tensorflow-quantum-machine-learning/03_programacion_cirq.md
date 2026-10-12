# Capítulo 3 — Tu primer programa cuántico con Cirq

## Por qué importa

Ya conoces la teoría: qubits, superposición, entrelazamiento y puertas cuánticas. Ahora toca programarlos de verdad. Cirq es la librería de Python que Google publicó para diseñar, simular y ejecutar circuitos cuánticos, y es la base sobre la que se construye TensorFlow Quantum. Todo lo que hagas a partir de este capítulo (clasificadores cuánticos, QAOA, VQE, aprendizaje por refuerzo cuántico) se apoya en los mismos bloques que vas a aprender aquí: qubits, puertas y circuitos.

## Explicación conceptual

### Los qubits viven en una rejilla

En Cirq, un qubit no es solo una idea abstracta: tiene una posición. La clase `cirq.GridQubit(fila, columna)` representa un qubit físico situado en una rejilla, imitando cómo están distribuidos los qubits en un chip real como Sycamore. Piensa en ello como las casillas de un tablero de ajedrez: cada pieza (qubit) ocupa una casilla concreta.

### Un circuito es una receta de pasos

Un `cirq.Circuit()` es la secuencia ordenada de puertas que aplicas a tus qubits, igual que una receta de cocina es una secuencia ordenada de pasos. Añades puertas con `circuito.append(...)`, y Cirq las organiza automáticamente en "momentos": un momento es un grupo de puertas que se ejecutan a la vez, sobre qubits distintos, para aprovechar el paralelismo del hardware.

### El catálogo de puertas que más vas a usar

- `cirq.H(qubit)`: Hadamard, crea superposición
- `cirq.X(qubit)`, `cirq.Y(qubit)`, `cirq.Z(qubit)`: rotaciones de 180 grados sobre cada eje (equivalentes cuánticos de "voltear" el qubit)
- `cirq.rx(angulo)`, `cirq.ry(angulo)`, `cirq.rz(angulo)`: rotaciones de un ángulo cualquiera sobre cada eje; estas son las que usarás para "entrenar" un circuito, porque el ángulo puede ser un parámetro ajustable
- `cirq.CNOT(control, objetivo)` y `cirq.CZ(control, objetivo)`: puertas de dos qubits que crean entrelazamiento
- Potencias fraccionarias, como `cirq.X(qubit)**0.5`: aplican solo "una parte" de la rotación. `cirq.X(qubit)**i` equivale a una rotación de `pi * i` radianes, una forma compacta de codificar un número real como ángulo de rotación (la usarás para convertir datos clásicos en estados cuánticos)

### Parámetros simbólicos con sympy

Cirq se apoya en la librería `sympy` para definir puertas con un ángulo que todavía no conoces, un símbolo en vez de un número fijo. Esto es exactamente lo que necesitas para que un circuito cuántico sea "entrenable": defines la puerta con un símbolo (por ejemplo, `theta`), y más adelante un optimizador va probando distintos valores de `theta` hasta encontrar el que mejor resuelve tu problema. Esta idea de "parámetro simbólico que se ajusta durante el entrenamiento" es el equivalente cuántico de un peso en una red neuronal clásica.

### Simular y medir

Sin acceso a un ordenador cuántico real, puedes simular el circuito en tu propio ordenador con `cirq.Simulator()`. El método `.run(circuito, repetitions=N)` ejecuta el circuito N veces (porque cada ejecución individual solo te da un resultado, recuerda el capítulo 1) y te devuelve la estadística de resultados.

## Ejemplo de código comentado

```python
import cirq
import sympy

# Dos qubits en una rejilla de una sola fila
q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)

# Un símbolo: un ángulo que todavía no tiene valor numérico
theta = sympy.Symbol("theta")

circuito = cirq.Circuit()

# Ponemos el primer qubit en superposición
circuito.append(cirq.H(q0))

# Entrelazamos los dos qubits
circuito.append(cirq.CNOT(q0, q1))

# Rotamos el segundo qubit un ángulo "theta" sobre el eje Y (parámetro entrenable)
circuito.append(cirq.ry(theta).on(q1))

# Medimos los dos qubits
circuito.append(cirq.measure(q0, q1, key="resultado"))

print(circuito)

# Para simular, sustituimos el símbolo "theta" por un valor concreto
simulador = cirq.Simulator()
resolver = cirq.ParamResolver({"theta": 0.5})
resultado = simulador.run(circuito, resolver, repetitions=500)

print(resultado.histogram(key="resultado"))
```

Fíjate en el `cirq.ParamResolver`: es el mecanismo que sustituye el símbolo `theta` por un número concreto justo antes de simular. Este patrón (circuito con símbolos + resolver que asigna valores) es exactamente el que usa TensorFlow Quantum por debajo cuando entrena un modelo, solo que en vez de asignar el valor a mano, un optimizador lo va ajustando automáticamente.

## Resumen de puntos clave

- `cirq.GridQubit` sitúa un qubit en una posición concreta de una rejilla, como en un chip real
- `cirq.Circuit()` y `.append(...)` construyen la secuencia de puertas del circuito
- Las puertas de rotación (`rx`, `ry`, `rz`) aceptan símbolos de `sympy` como ángulo, lo que convierte el circuito en "entrenable"
- Las potencias fraccionarias de una puerta (`cirq.X(q)**i`) codifican un número real como rotación
- `cirq.Simulator()` ejecuta el circuito; `cirq.ParamResolver` sustituye los símbolos por valores concretos antes de simular

## Ponlo en práctica

Construye un circuito de tres qubits que: ponga el primer qubit en superposición con una puerta Hadamard, lo entrelace con el segundo mediante un `CNOT`, entrelace el segundo con el tercero mediante otro `CNOT`, y finalmente mida los tres qubits. Ejecuta el circuito 1.000 veces y comprueba en el histograma que solo aparecen las combinaciones "000" y "111". Este circuito de tres qubits entrelazados en cadena se conoce como estado GHZ, y es uno de los ejemplos de entrelazamiento múltiple más citados en computación cuántica.
