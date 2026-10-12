# Capítulo 11 — VQE avanzado: Rotosolve y comparativa de optimizadores

## Por qué importa

En el capítulo anterior optimizaste VQE con Adam (basado en gradiente) y con Nelder-Mead (caja negra genérica). Ninguno de los dos está diseñado específicamente para circuitos cuánticos parametrizados con rotaciones. **Rotosolve** sí lo está: aprovecha una propiedad matemática exclusiva de las puertas de rotación cuántica para encontrar, en cada paso, el valor óptimo exacto de un parámetro, sin necesidad de dar pasos pequeños de gradiente ni de adivinar por prueba y error. Entender Rotosolve te da intuición sobre cómo diseñar optimizadores a medida para problemas cuánticos, en vez de aplicar siempre las mismas herramientas clásicas.

## Explicación conceptual

### Por qué un parámetro de rotación se comporta como una onda

Cuando fijas todos los parámetros de tu ansatz salvo uno, y dejas que ese parámetro (el ángulo de una puerta `rx`, `ry` o `rz`) varíe, la energía resultante `E(θ)` no es una curva cualquiera: siempre tiene la forma `E(θ) = A·cos(θ) + B·sin(θ) + C`, una onda sinusoidal simple. Esto es consecuencia directa de cómo las puertas de rotación transforman el estado cuántico. Puedes comprobarlo tú mismo en el ejemplo del capítulo anterior: la curva de energía que dibujaste al barrer el ángulo de 0 a 2π tenía justo esa forma ondulada.

### Rotosolve: resolver la onda en vez de aproximarla

Si sabes que `E(θ)` es una onda con esa forma exacta, no necesitas ir dando pasitos de gradiente para encontrar su mínimo: te basta con evaluar la energía en tres puntos concretos (por ejemplo, en un ángulo `φ` cualquiera, y en `φ + π/2` y `φ - π/2`) para despejar matemáticamente los coeficientes A, B y C, y calcular directamente el ángulo que minimiza esa onda. Rotosolve repite este proceso parámetro a parámetro, ciclo tras ciclo: en cada ciclo, cada parámetro se ajusta a su valor óptimo exacto suponiendo que el resto de parámetros se mantienen fijos, y con varios ciclos el conjunto completo de parámetros converge.

La ventaja frente a un método basado en gradiente es que cada actualización de un parámetro es un salto directo al óptimo (para ese parámetro, con los demás fijos), no un paso pequeño en la dirección de mayor pendiente. La desventaja es que necesitas evaluar el circuito varias veces por cada parámetro y por cada ciclo, así que el coste por iteración es mayor.

### Comparando tres optimizadores en igualdad de condiciones

Además de Rotosolve, este capítulo compara dos optimizadores clásicos de caja negra que ya conoces o vas a conocer:

- **Nelder-Mead**: explora el espacio de parámetros con un "simplex" (una figura geométrica que se deforma) que se contrae hacia la región de menor energía; es robusto pero puede necesitar muchas evaluaciones en problemas con muchos parámetros
- **SPSA** (aproximación estocástica por perturbación simultánea): en cada paso, perturba **todos** los parámetros a la vez con una dirección aleatoria, evalúa la energía en esa dirección y estima el gradiente completo con solo dos evaluaciones del circuito, sea cual sea el número de parámetros. Es especialmente útil cuando evaluar el circuito es caro (por ejemplo, en hardware cuántico real) y tienes muchos parámetros que optimizar

Comparar los tres sobre el mismo problema te permite ver que no existe un optimizador universalmente mejor: depende de cuántos parámetros tengas, de cuánto cueste evaluar el circuito y de cuánto ruido haya en esa evaluación.

## Ejemplo de código comentado

Primero, la implementación de Rotosolve (una versión reducida de la usada en el resto del capítulo, con menos iteraciones para que se ejecute en pocos segundos):

```python
import numpy as np
import random

def rotosolve(funcion_energia, num_parametros, iteraciones=30):
    """Optimiza cada parámetro de forma analítica, ciclo a ciclo."""
    theta = np.random.uniform(-np.pi, np.pi, num_parametros)
    for _ in range(iteraciones):
        for d in range(num_parametros):
            phi = random.uniform(-np.pi, np.pi)
            theta_prueba = np.copy(theta)

            theta_prueba[d] = phi
            energia_phi = funcion_energia(theta_prueba)
            theta_prueba[d] = phi + np.pi / 2
            energia_mas = funcion_energia(theta_prueba)
            theta_prueba[d] = phi - np.pi / 2
            energia_menos = funcion_energia(theta_prueba)

            # Fórmula cerrada que resuelve el mínimo exacto de la onda A*cos + B*sin + C
            theta[d] = phi - np.pi / 2 - np.arctan2(
                2 * energia_phi - energia_mas - energia_menos, energia_mas - energia_menos
            )
    return {"fun": funcion_energia(theta), "params": theta}
```

Ahora, comparamos Rotosolve, Nelder-Mead y SPSA sobre un Hamiltoniano aleatorio de dos qubits (simulado directamente con Cirq, sin pasar por TFQ, para poder controlar la medición base a base):

```python
import cirq
from scipy.optimize import minimize
from noisyopt import minimizeSPSA

def capa_ansatz(circuito, qubits, parametros):
    for qubit in qubits:
        circuito.append([cirq.rx(parametros[0]).on(qubit)])
        circuito.append([cirq.rz(parametros[1]).on(qubit)])
    for i in range(len(qubits) - 1):
        circuito.append([cirq.CNOT(qubits[i], qubits[i + 1])])
    return circuito

def rotar_a_base_de_medida(circuito, qubits, terminos_pauli):
    for i, qubit in enumerate(qubits):
        if terminos_pauli[i] == "x":
            circuito.append(cirq.ry(-np.pi / 2).on(qubit))
        elif terminos_pauli[i] == "y":
            circuito.append(cirq.rx(np.pi / 2).on(qubit))
    return circuito

def valor_esperado(circuito, qubits_medidos, repeticiones=1000):
    if len(qubits_medidos) == 0:
        return 1
    circuito.append(cirq.measure(*qubits_medidos, key="resultado"))
    simulador = cirq.Simulator()
    resultados = simulador.run(circuito, repetitions=repeticiones)
    medidas = resultados.measurements["resultado"]
    return np.mean([1 if sum(fila) % 2 == 0 else -1 for fila in medidas])

# Hamiltoniano de ejemplo: dos términos de Pauli sobre 2 qubits, con pesos fijos
terminos_hamiltoniano = [["x", "z"], ["z", "y"]]
pesos_hamiltoniano = [0.6, 0.3]
num_qubits, num_capas = 2, 2

def energia(parametros):
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    total = 0
    for termino, peso in zip(terminos_hamiltoniano, pesos_hamiltoniano):
        circuito = cirq.Circuit()
        for _ in range(num_capas):
            circuito = capa_ansatz(circuito, qubits, parametros)
        circuito = rotar_a_base_de_medida(circuito, qubits, termino)
        qubits_a_medir = [qubits[i] for i in range(len(termino)) if termino[i] != "i"]
        total += peso * valor_esperado(circuito, qubits_a_medir)
    return total

num_parametros = num_qubits * num_capas

resultado_nm = minimize(energia, np.random.uniform(0, 2 * np.pi, num_parametros), method="Nelder-Mead")
resultado_spsa = minimizeSPSA(energia, bounds=[[0, 2 * np.pi]] * num_parametros,
                               x0=np.random.uniform(0, 2 * np.pi, num_parametros), niter=100, paired=False)
resultado_roto = rotosolve(energia, num_parametros)

print("Nelder-Mead:", resultado_nm["fun"])
print("SPSA:", resultado_spsa["fun"])
print("Rotosolve:", resultado_roto["fun"])
```

Al ejecutarlo varias veces, comprobarás que los tres optimizadores suelen converger a energías parecidas, pero con un número de evaluaciones del circuito muy distinto: Rotosolve suele necesitar menos ciclos completos para estabilizarse, a cambio de evaluar el circuito varias veces por cada parámetro en cada ciclo.

## Resumen de puntos clave

- El valor esperado de la energía, al variar un único parámetro de rotación y dejar el resto fijos, siempre tiene la forma de una onda `A·cos(θ) + B·sin(θ) + C`
- Rotosolve aprovecha esa propiedad para calcular el ángulo óptimo exacto de cada parámetro, evaluando el circuito en solo tres puntos por parámetro
- SPSA estima el gradiente completo con solo dos evaluaciones del circuito, sea cual sea el número de parámetros, útil cuando evaluar el circuito es caro
- No hay un optimizador universalmente mejor: la elección depende del número de parámetros, del coste de evaluar el circuito y del ruido presente en esa evaluación

## Ponlo en práctica

Añade un tercer término al Hamiltoniano de ejemplo: `["y", "x"]` con peso `0.4` (recuerda actualizar tanto `terminos_hamiltoniano` como `pesos_hamiltoniano`). Vuelve a ejecutar los tres optimizadores y anota, para cada uno, la energía final obtenida. Comprueba si Rotosolve sigue siendo competitivo frente a Nelder-Mead y SPSA ahora que el Hamiltoniano tiene un término más.
