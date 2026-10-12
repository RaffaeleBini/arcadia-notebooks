# Capítulo 4 — Supremacía cuántica y los límites de la simulación clásica

## Por qué importa

En 2019, Google anunció que su procesador Sycamore había resuelto en 200 segundos un cálculo que, según sus estimaciones, le habría llevado miles de años al superordenador clásico más potente del mundo. A ese hito se le llamó **supremacía cuántica**: el momento en el que un ordenador cuántico resuelve una tarea que ningún ordenador clásico puede resolver en un tiempo razonable. Entender por qué ese cálculo es tan difícil de simular clásicamente (y cómo se demuestra que el resultado cuántico es correcto) te da la intuición clave de todo este curso: la potencia de la computación cuántica crece exponencialmente con el número de qubits, mientras que el coste de simularla en un ordenador clásico crece igual de rápido.

## Explicación conceptual

### Por qué simular un circuito cuántico es tan caro

Un sistema de n qubits en superposición necesita 2^n números para describirse por completo. Con 53 qubits (los que tiene Sycamore), eso son más de 9.000 billones de números: ya no caben en la memoria de ningún ordenador clásico existente. Por eso, cuantos más qubits tiene tu circuito, más rápido se dispara el coste de simularlo clásicamente, mientras que el propio ordenador cuántico ejecuta ese mismo circuito sin ese coste añadido: literalmente, "es" ese sistema cuántico.

### Muestreo de circuitos aleatorios: la tarea que Sycamore resolvió

La tarea concreta que demuestra supremacía no es "resolver un problema útil", sino algo más sutil: generar muestras de la distribución de probabilidad de salida de un circuito cuántico aleatorio y profundo. Un ordenador cuántico genera esas muestras de forma natural, ejecutando el circuito una y otra vez. Un ordenador clásico, en cambio, tiene que calcular esa distribución de probabilidad multiplicando matrices gigantes, y ese cálculo es el que se vuelve inviable a partir de cierto número de qubits y de profundidad del circuito.

### Cómo se comprueba que el resultado cuántico es correcto: XEB

Si nadie puede calcular clásicamente la distribución de probabilidad completa, ¿cómo sabes que las muestras del ordenador cuántico son correctas? La respuesta es el **benchmarking de entropía cruzada** (XEB): en vez de calcular la distribución completa, calculas la probabilidad teórica de cada muestra concreta que observaste (algo mucho más barato que calcular todas las probabilidades posibles) y comparas esa probabilidad con lo que esperarías de un resultado puramente aleatorio. Cuanto más se parezcan las muestras observadas a la distribución teórica del circuito (y no al ruido aleatorio), más confianza tienes en que el hardware está funcionando de verdad como un ordenador cuántico.

La forma que toma esa distribución teórica de salida —muchas probabilidades bajas y pocas altas, como manchas de luz láser dispersa— se conoce como **distribución de Porter-Thomas**, y es la "huella dactilar" que se busca al comparar resultados reales con la teoría. Una técnica relacionada, el **benchmarking de pureza de las manchas** (speckle purity benchmarking), ayuda a separar cuánto de esa huella es señal cuántica real y cuánto es ruido del hardware.

### Cuantificando la confianza: significancia estadística

Afirmar "hemos demostrado supremacía cuántica" es una afirmación fuerte, así que hay que respaldarla con estadística rigurosa. Se usan herramientas como la **prueba de Kolmogorov-Smirnov** (KS), que compara si dos distribuciones de datos son estadísticamente indistinguibles, y técnicas de **bootstrapping** (remuestreo repetido de los mismos datos) para estimar cuánto varían tus conclusiones si repitieras el experimento. El resultado final se suele expresar en "niveles sigma": cuantas más desviaciones estándar de margen tengas respecto a la hipótesis nula (que el resultado es ruido aleatorio), más sólida es la conclusión.

### La era NISQ: potentes, pero imperfectos

Sycamore, Bristlecone (su predecesor) y todos los procesadores cuánticos actuales pertenecen a la era **NISQ** (cuántico de escala intermedia y ruidoso, recuerda el capítulo 2): tienen decenas o pocos cientos de qubits, y ese ruido limita la profundidad de los circuitos que se pueden ejecutar de forma fiable. La supremacía cuántica demuestra que, incluso en esta era imperfecta, ya hay tareas que superan a la computación clásica. El reto de los próximos capítulos es distinto: no demostrar supremacía con circuitos aleatorios, sino usar esos mismos ordenadores ruidosos para resolver problemas útiles de machine learning, química y optimización.

## Ejemplo de código comentado

Una forma práctica de sentir "el coste exponencial" es simular la evolución en el tiempo de un sistema cuántico. Calcular esa evolución exactamente requiere multiplicar matrices que crecen como 2^n, así que en la práctica se aproxima dividiendo la evolución en pasos pequeños: esto se llama **trotterización**. Aquí tienes un ejemplo con un modelo de cuatro qubits (una cadena de espines con campo transverso, un sistema típico de física de materiales):

```python
import cirq
import tensorflow_quantum as tfq
import numpy as np
import matplotlib.pyplot as plt

def cadena_ising_campo_transverso(n, campos):
    """Hamiltoniano de una cadena de n qubits con interacción Z-Z y campo X."""
    qubits = [cirq.GridQubit(0, i) for i in range(n)]
    hamiltoniano = cirq.PauliSum()
    for i in range(n - 1):
        hamiltoniano -= cirq.Z(qubits[i]) * cirq.Z(qubits[i + 1])
    for i in range(n):
        hamiltoniano -= campos[i] * cirq.X(qubits[i])
    return hamiltoniano

def paso_trotter(hamiltoniano, t):
    """Aproxima exp(-i * H * t) aplicando cada término del Hamiltoniano por separado."""
    return cirq.Circuit([tfq.util.exponential([t * termino]) for termino in hamiltoniano])

def trotter_lie(hamiltoniano, t, repeticiones):
    """Cuantas más repeticiones, más fino el paso y más preciso el resultado."""
    circuito = cirq.Circuit()
    for _ in range(repeticiones):
        circuito += paso_trotter(hamiltoniano, t / repeticiones)
    return circuito

n = 4
campos = [1.0] * n
hamiltoniano = cadena_ising_campo_transverso(n, campos)
qubits = [cirq.GridQubit(0, i) for i in range(n)]
magnetizacion = sum(cirq.Z(q) for q in qubits)

tiempos = np.linspace(0.01, 2, 20)

# Con solo 1 repetición, el paso de trotter es "grueso" y menos preciso
circuitos_1_paso = [trotter_lie(hamiltoniano, float(t), 1) for t in tiempos]
# Con 10 repeticiones, el paso es más fino y se acerca más a la evolución real
circuitos_10_pasos = [trotter_lie(hamiltoniano, float(t), 10) for t in tiempos]

capa_expectativa = tfq.layers.Expectation()
exp_1_paso = capa_expectativa(circuitos_1_paso, symbol_names=[], symbol_values=[[] for _ in tiempos], operators=magnetizacion).numpy()
exp_10_pasos = capa_expectativa(circuitos_10_pasos, symbol_names=[], symbol_values=[[] for _ in tiempos], operators=magnetizacion).numpy()

plt.plot(tiempos, exp_1_paso, label="1 paso de Trotter")
plt.plot(tiempos, exp_10_pasos, label="10 pasos de Trotter")
plt.xlabel("Tiempo")
plt.ylabel("Magnetización")
plt.legend()
plt.show()
```

Cuantos más pasos de Trotter usas, más te acercas a la evolución exacta del sistema, pero también más puertas cuánticas necesitas encadenar, y recuerda el capítulo 2: más puertas significa más tiempo expuesto a la decoherencia. Este equilibrio entre precisión y profundidad del circuito es exactamente el mismo dilema al que se enfrenta cualquier ordenador cuántico NISQ real.

## Resumen de puntos clave

- La supremacía cuántica se demostró resolviendo una tarea de muestreo de circuitos aleatorios, inviable de simular clásicamente por el crecimiento exponencial (2^n) de los estados posibles
- El benchmarking de entropía cruzada (XEB) permite verificar que las muestras cuánticas observadas corresponden a la distribución teórica del circuito (distribución de Porter-Thomas) y no a ruido
- La confianza estadística en el resultado se cuantifica con pruebas como Kolmogorov-Smirnov, bootstrapping y niveles sigma
- Vivimos en la era NISQ: ordenadores cuánticos reales pero ruidosos, con pocos qubits y circuitos poco profundos
- La trotterización aproxima la evolución de un sistema cuántico dividiéndola en pasos pequeños; más pasos = más precisión, pero también más profundidad de circuito y más exposición al ruido

## Ponlo en práctica

Retoma el ejemplo de código y añade una tercera curva con 3 repeticiones de Trotter (`trotter_lie(hamiltoniano, float(t), 3)`), a medio camino entre la versión de 1 paso y la de 10. Dibuja las tres curvas juntas y observa cómo la curva de 3 pasos se aleja menos de la de 10 pasos que la de 1 paso. Anota, en una frase, qué relación observas entre el número de pasos de Trotter y el parecido con la curva de referencia (10 pasos).
