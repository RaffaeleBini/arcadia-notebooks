# Capítulo 12 — Casos de uso reales

## Por qué importa

Has recorrido el curso completo: desde qubits y puertas hasta VQE avanzado. Toca dar un paso atrás y responder a la pregunta que probablemente te hiciste en el capítulo 1: ¿para qué sirve todo esto en la vida real? Este capítulo de cierre recoge cuatro aplicaciones concretas de lo que has aprendido, todas construidas sobre el mismo material que has trabajado durante el curso, para que termines con una visión clara de hacia dónde puedes llevar estos conocimientos.

## Explicación conceptual

### Química cuántica: calcular la energía de una molécula real

El ejemplo más citado de aplicación de VQE (capítulos 10 y 11) es calcular la **curva de disociación** de una molécula de hidrógeno (H₂): la energía del sistema en función de la distancia entre los dos átomos. Ese cálculo es exactamente el mismo problema que resolviste con Hamiltonianos sencillos, pero ahora el Hamiltoniano se obtiene de la física real de la molécula (mediante la transformación de Jordan-Wigner, que convierte un problema de electrones en un problema de qubits) en vez de inventarlo tú. Empresas farmacéuticas y de materiales siguen de cerca este tipo de cálculos porque, a largo plazo, prometen diseñar nuevos fármacos y materiales simulando su comportamiento cuántico real, algo que un ordenador clásico solo puede aproximar para moléculas muy pequeñas.

### Optimización combinatoria: de Max-Cut a problemas logísticos

El problema Max-Cut que resolviste con QAOA (capítulo 9) es un modelo simplificado de problemas mucho más aplicados: repartir tareas entre servidores para minimizar la comunicación entre ellos, agrupar clientes por similitud, o diseñar redes que minimicen interferencias. La estructura matemática (dividir un conjunto de elementos conectados entre sí en grupos que optimicen alguna métrica) se repite en logística, telecomunicaciones y diseño de redes eléctricas.

### Control y toma de decisiones: de CartPole a la robótica

El agente que aprendió a equilibrar un palo en CartPole (capítulo 8) usa el mismo principio que necesitarías para controlar un robot, gestionar una cartera de inversión o decidir cuándo cargar una batería: un agente que observa un estado, elige una acción y aprende de la recompensa que recibe. Sustituir la red neuronal clásica por un circuito cuántico es todavía un campo de investigación activo, pero ya demuestra que el aprendizaje por refuerzo cuántico puede aprender políticas de control competitivas en problemas de tamaño reducido.

### Validar el hardware: la supremacía cuántica como control de calidad

Por último, recuerda el capítulo 4: el experimento de supremacía cuántica de Sycamore no resolvía un problema útil por sí mismo, pero demostraba que el hardware cuántico se comporta como se espera de él. Ese mismo tipo de verificación (comparar el comportamiento real de un chip con lo que predice la teoría) es el primer paso obligatorio antes de confiar en cualquiera de las aplicaciones anteriores: de poco sirve un circuito de VQE o QAOA perfectamente diseñado si el hardware que lo ejecuta no es fiable.

## Ejemplo de código comentado (extensión opcional)

El siguiente ejemplo calcula la curva de disociación del H₂ con VQE, usando la librería `openfermion` para obtener el Hamiltoniano molecular real. Instalar `openfermion` y `openfermionpyscf` en Colab tarda varios minutos (compilan la librería de química cuántica `pyscf`), así que este bloque queda como **extensión opcional**: no hace falta ejecutarlo para completar el curso, pero te lo enseñamos comentado para que veas cómo VQE pasa de un ejemplo de clase a un caso de uso real.

```python
# Instalación (varios minutos): !pip install openfermion openfermionpyscf
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
import openfermion as of
from openfermionpyscf import generate_molecular_hamiltonian
from scipy.sparse import linalg

def capa_ansatz(circuito, qubits, parametros):
    for i, qubit in enumerate(qubits):
        circuito += cirq.ry(parametros[3 * i]).on(qubit)
        circuito += cirq.rz(parametros[3 * i + 1]).on(qubit)
        circuito += cirq.ry(parametros[3 * i + 2]).on(qubit)
    for i in range(len(qubits) - 1):
        circuito += cirq.CNOT(qubits[i], qubits[i + 1])
    return circuito

def construir_vqe(qubits, capas, hamiltoniano):
    num_parametros = capas * 3 * len(qubits)
    parametros = sympy.symbols(f"vqe0:{num_parametros}")
    circuito = cirq.Circuit()
    for _ in range(capas):
        circuito = capa_ansatz(circuito, qubits, parametros)
    entradas = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    salida = tfq.layers.PQC(circuito, hamiltoniano, differentiator=tfq.differentiators.Adjoint())(entradas)
    return tf.keras.models.Model(inputs=entradas, outputs=salida)

distancias = [0.3, 0.6, 0.9, 1.2, 1.5, 2.0]  # distancia entre los dos átomos de hidrógeno, en angstroms
energias_reales, energias_vqe = [], []

for distancia in distancias:
    geometria = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, distancia))]
    hamiltoniano_molecular = generate_molecular_hamiltonian(geometria, "sto-3g", multiplicity=1, charge=0)
    num_qubits = of.count_qubits(hamiltoniano_molecular)
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]

    operador_jw = of.transforms.jordan_wigner(hamiltoniano_molecular)
    matriz_dispersa = of.get_sparse_operator(operador_jw)
    autovalor_real, _ = linalg.eigsh(matriz_dispersa, k=1, which="SA")
    hamiltoniano_cirq = of.transforms.qubit_operator_to_pauli_sum(operador_jw, qubits=qubits)

    modelo = construir_vqe(qubits, capas=2, hamiltoniano=hamiltoniano_cirq)
    entrada_vacia = tfq.convert_to_tensor([cirq.Circuit()])
    optimizador = tf.keras.optimizers.Adam(learning_rate=0.1)

    energia_actual, anterior = None, np.inf
    for _ in range(200):
        with tf.GradientTape() as cinta:
            energia_actual = modelo(entrada_vacia)
        gradientes = cinta.gradient(energia_actual, modelo.trainable_variables)
        optimizador.apply_gradients(zip(gradientes, modelo.trainable_variables))
        energia_actual = energia_actual.numpy()[0][0]
        if abs(energia_actual - anterior) < 1e-5:
            break
        anterior = energia_actual

    energias_vqe.append(energia_actual)
    energias_reales.append(autovalor_real[0])

plt.plot(distancias, energias_vqe, "o-", label="VQE (TensorFlow Quantum)")
plt.plot(distancias, energias_reales, "--", label="Energía exacta")
plt.xlabel("Distancia entre átomos (angstroms)")
plt.ylabel("Energía (Hartree)")
plt.title("Curva de disociación del H₂ con VQE")
plt.legend()
plt.show()
```

La curva resultante tiene forma de "U": la energía baja al principio (los átomos se atraen), toca un mínimo (la distancia de enlace real de la molécula) y vuelve a subir al separarlos demasiado (se rompe el enlace). Comprobar que la curva de VQE se superpone a la energía exacta es la validación final de que el algoritmo funciona sobre un sistema físico real, no solo sobre ejemplos de juguete.

## Resumen de puntos clave

- VQE se aplica en química cuántica real calculando curvas de disociación molecular, con el Hamiltoniano generado a partir de la geometría real de una molécula
- QAOA sobre Max-Cut modela problemas de optimización logística, de redes y de reparto de recursos
- El aprendizaje por refuerzo cuántico, entrenado en entornos de control como CartPole, apunta hacia aplicaciones de robótica y toma de decisiones secuenciales
- La verificación de hardware mediante supremacía cuántica es el paso previo indispensable para confiar en cualquiera de las aplicaciones anteriores
- Todas estas aplicaciones parten de las mismas piezas que aprendiste en el curso: qubits, circuitos parametrizados, Hamiltonianos y optimizadores clásicos o cuánticos

## Ponlo en práctica

Elige una de las cuatro aplicaciones de este capítulo (química cuántica, optimización combinatoria, aprendizaje por refuerzo o validación de hardware) y busca un artículo reciente (menos de dos años) que la aplique a un problema fuera de los ejemplos de este curso. Escribe un resumen de tres a cinco frases explicando: qué problema resuelve, qué algoritmo cuántico de los que has aprendido usa (o una variante), y qué limitación menciona el propio artículo sobre el estado actual de la tecnología.
