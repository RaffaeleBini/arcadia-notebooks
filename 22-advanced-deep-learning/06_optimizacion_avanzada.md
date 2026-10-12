# Capítulo 6: Optimización avanzada y regularización

## Por qué importa este tema

En el capítulo anterior aprendiste el descenso de gradiente: en cada paso, calculas el gradiente de la función de pérdida y mueves los parámetros en la dirección opuesta, con un tamaño de paso que marca la tasa de aprendizaje. Esa idea funciona, pero tiene un punto débil que descubre cualquiera que entrene una red neuronal real: en muchos problemas, el descenso de gradiente puro converge con una lentitud desesperante, o directamente oscila sin asentarse en el mínimo.

Este capítulo, correspondiente a la Clase 5 del curso ("Optimización II: técnicas avanzadas y regularización"), responde a tres preguntas encadenadas. Primero, por qué el descenso de gradiente básico falla justo en los paisajes de pérdida más habituales en deep learning. Segundo, cómo el momento, una técnica que acumula gradientes pasados, corrige buena parte de ese problema con un coste computacional mínimo. Tercero, qué gana un optimizador cuando además usa información de curvatura, la llamada matriz Hessiana, y por qué esa ganancia tiene un precio computacional tan alto que en redes neuronales reales solo es viable mediante aproximaciones inteligentes.

Entender estas ideas te importa más allá de la teoría. Cuando en DIH DATAlife acompañamos a equipos técnicos de sectores como Salud-Cuidados o Agro-Mar Alimentación en proyectos de digitalización con componentes de inteligencia artificial (IA), una de las preguntas más habituales es por qué un modelo tarda tantas horas en entrenarse o por qué la pérdida se estanca. Saber que la causa suele ser un paisaje de pérdida mal condicionado, y saber qué optimizador corresponde a cada situación, es la diferencia entre perder días ajustando manualmente la tasa de aprendizaje y elegir en minutos la herramienta correcta.

## Explicación conceptual con analogías

### El valle estrecho: por qué el gradiente puro se atasca

Imagina que caminas de noche por un valle muy alargado y estrecho, como el cauce seco de un río entre dos laderas empinadas. Tu única información es la pendiente que notas bajo tus pies en cada punto donde te paras: esa pendiente es el gradiente. Como las laderas son mucho más pronunciadas que el propio cauce del valle, cada paso que das te empuja con fuerza hacia una ladera o hacia la otra, y solo un poco hacia delante, que es la dirección en la que realmente quieres avanzar para llegar a la salida del valle.

El resultado es un zigzag: rebotas de una ladera a la otra, avanzando muy poco en cada paso hacia tu destino real. Si acortas el paso (bajas la tasa de aprendizaje) para no salirte del valle, avanzas de forma segura pero exasperantemente lenta. Si alargas el paso para avanzar más rápido, te sales del valle y oscilas todavía más, o incluso diverges. Este es exactamente el comportamiento del descenso de gradiente en un valle estrecho, uno de los ejemplos clásicos para visualizar por qué el método básico falla en muchos paisajes de pérdida reales.

### El momento: la bola que rueda y acumula inercia

Ahora cambia de estrategia. En lugar de caminar mirando solo la pendiente bajo tus pies en cada instante, imagina que sueltas una bola pesada en ese mismo valle y dejas que ruede por efecto de la gravedad. La bola no reacciona de forma inmediata a cada irregularidad del terreno: tiene inercia, es decir, conserva parte de la velocidad que ya llevaba.

Cuando la bola rueda por las paredes del valle, las fuerzas que la empujan hacia una ladera y hacia la otra se acumulan en direcciones opuestas y tienden a cancelarse: el "tirón" hacia la izquierda de un tramo se compensa con el "tirón" hacia la derecha del tramo siguiente. En cambio, la fuerza que la empuja a lo largo del cauce del valle, hacia la salida, apunta siempre en la misma dirección en cada paso, así que esa componente se refuerza una y otra vez. El resultado es que la bola termina rodando rápido a lo largo del valle y apenas rebota entre las paredes.

Ese es, con precisión, el principio del momento en optimización: en lugar de moverte solo según el gradiente actual, acumulas una media de los gradientes recientes (la "velocidad" de la bola). Las componentes del gradiente que cambian de signo de un paso a otro, como las de las paredes del valle, se cancelan al promediarse. Las componentes que apuntan siempre en la misma dirección, como la del fondo del valle, se refuerzan. El momento actúa como un filtro de paso bajo: suaviza las oscilaciones de alta frecuencia y deja pasar la tendencia de fondo, consistente a lo largo de muchas iteraciones.

### El segundo orden: no solo hacia dónde bajar, sino cuánto se curva el terreno

El momento mejora mucho la situación, pero sigue usando solo el gradiente, es decir, solo la pendiente local. Imagina ahora que, en lugar de caminar a ciegas, dispones de un mapa topográfico completo de la zona que pisas: no solo sabes hacia dónde baja el terreno, sino también cómo de pronunciada es la curvatura en cada dirección. Con ese mapa puedes calcular directamente, de un solo vistazo, cuál es el mejor punto de destino dentro de esa región, en lugar de dar pasos pequeños y prudentes uno detrás de otro.

Esa información de curvatura es justo lo que aporta la matriz Hessiana, y es la base de los métodos de segundo orden, como el método de Newton. El precio de ese mapa tan detallado es que calcularlo y "leerlo" (invertirlo matemáticamente) es mucho más costoso que simplemente notar la pendiente bajo tus pies. Por eso, como verás más adelante, en redes neuronales con millones de parámetros hace falta recurrir a versiones simplificadas de ese mapa.

## Desarrollo teórico

### Momento clásico

El descenso de gradiente estocástico (SGD) actualiza los parámetros con la regla que ya conoces:

θ_{t+1} = θ_t − η ∇_θ L(θ_t)

donde θ representa los parámetros del modelo, η es la tasa de aprendizaje y ∇_θ L(θ_t) es el gradiente de la función de pérdida L en la iteración t.

El momento clásico introduce una variable adicional, la velocidad v, que acumula una media móvil con decaimiento exponencial de los gradientes pasados:

v_t = β v_{t−1} + η ∇_θ L(θ_{t−1})
θ_t = θ_{t−1} − v_t

El coeficiente de momento β toma valores típicos entre 0,9 y 0,99, y controla cuánto "recuerda" la velocidad de los gradientes anteriores. Fíjate en la mecánica: si el gradiente mantiene un signo consistente iteración tras iteración (como el "empuje hacia delante" del fondo del valle), su contribución a v se suma de forma constructiva y la velocidad crece. Si el gradiente cambia de signo de una iteración a la siguiente (como el rebote entre las dos laderas), las contribuciones se cancelan parcialmente y la velocidad en esa dirección se mantiene pequeña. El efecto neto es una trayectoria de optimización mucho más suave, con menos oscilaciones y un avance más rápido en las direcciones de pendiente consistente.

Una variante relevante es el gradiente acelerado de Nesterov (NAG), que en lugar de calcular el gradiente en la posición actual, lo calcula en la posición "anticipada" a la que llevaría el momento acumulado. Esta pequeña corrección, un vistazo hacia adelante antes de dar el paso, produce actualizaciones algo más informadas y, en la práctica, una convergencia todavía más estable en presencia de curvatura pronunciada o gradientes ruidosos.

### Por qué el gradiente puro es tan sensible al mal condicionamiento

La razón matemática de fondo detrás del valle estrecho tiene un nombre preciso: el número de condición del problema, definido como el cociente entre la curvatura máxima y la curvatura mínima del paisaje de pérdida en torno al punto actual. Cuando ese número de condición es alto (una dirección se curva mucho más que otra, exactamente como las paredes del valle frente a su cauce), el número de pasos que necesita cualquier método de primer orden para converger crece con ese cociente. El descenso de gradiente, al no distinguir direcciones, se ve obligado a usar una tasa de aprendizaje lo bastante pequeña como para no divergir en la dirección de curvatura máxima, y esa misma tasa resulta insuficiente para avanzar con rapidez en la dirección de curvatura mínima. Es, en esencia, la causa raíz de la lentitud que viste en el capítulo anterior y que el momento mitiga sin eliminar del todo.

### Métodos de segundo orden: la matriz Hessiana

Los métodos de primer orden usan únicamente el gradiente, la primera derivada de la pérdida. Los métodos de segundo orden incorporan además la curvatura, representada por la matriz Hessiana, H, que reúne todas las derivadas parciales de segundo orden de la función de pérdida respecto a los parámetros:

H_{ij} = ∂²L(θ) / ∂θ_i ∂θ_j

Para un modelo con n parámetros, H es una matriz cuadrada de tamaño n × n. Cada elemento H_ij te dice cómo cambia la pendiente respecto al parámetro θ_i cuando mueves el parámetro θ_j: es, literalmente, el mapa de curvatura del que hablábamos en la analogía.

El método de Newton se obtiene de aproximar la función de pérdida, en el entorno del punto actual, por una función cuadrática (una parábola en varias dimensiones) mediante su desarrollo de Taylor de segundo orden, y de minimizar exactamente esa aproximación local. El resultado es la siguiente regla de actualización:

θ_{t+1} = θ_t − η H⁻¹ ∇_θ L(θ_t)

donde H⁻¹ es la matriz inversa de la Hessiana. Fíjate en la diferencia clave respecto al descenso de gradiente: el gradiente ya no se usa "en crudo", sino corregido por la inversa de la curvatura. En una dirección muy curvada, H⁻¹ reduce el paso, porque una pequeña variación del parámetro ya provoca un gran cambio en la pérdida. En una dirección poco curvada, H⁻¹ amplía el paso, porque ahí puedes permitirte avanzar mucho más antes de notar el efecto. En el ejemplo del valle estrecho, el método de Newton "sabe" de antemano que las laderas son mucho más pronunciadas que el cauce, y ajusta el paso en consecuencia en una sola actualización, en lugar de necesitar cientos de pasos pequeños y zigzagueantes.

Esta es la razón de fondo por la que los métodos de segundo orden, cuando son aplicables, convergen en muchos menos pasos que el descenso de gradiente: eliminan (o reducen drásticamente) la dependencia del número de condición que penaliza a los métodos de primer orden.

### El coste computacional: por qué el segundo orden es caro

El propio Newton tiene un problema evidente: calcular y, sobre todo, invertir la matriz Hessiana es costoso, y ese coste crece muy deprisa con el número de parámetros n:

- Almacenar la Hessiana completa cuesta O(n²), porque es una matriz densa de n × n
- Invertirla mediante los algoritmos estándar de álgebra lineal cuesta O(n³)

Para hacerte una idea de la magnitud: una red neuronal modesta puede tener varios millones de parámetros. Elevar ese número al cubo da una cifra de operaciones completamente inviable, incluso para el hardware más potente disponible hoy. Compáralo con el coste de una actualización de descenso de gradiente puro, que es O(n): lineal en el número de parámetros, y por tanto perfectamente escalable a redes con millones o miles de millones de pesos. Esta brecha de coste computacional es la razón por la que, en la práctica, casi ninguna red neuronal profunda de tamaño realista se entrena con el método de Newton "puro".

Existen métodos cuasi-Newton, como el algoritmo BFGS (por las iniciales de sus autores, Broyden, Fletcher, Goldfarb y Shanno), que evitan calcular la Hessiana exacta y en su lugar construyen una aproximación de su inversa de forma iterativa, usando solo información de gradientes en pasos sucesivos. Reducen el coste respecto a Newton puro, pero siguen resultando caros para las escalas de parámetros habituales en deep learning.

### Aproximaciones diagonales por bloques y de Kronecker

Para que la información de curvatura sea utilizable en redes neuronales grandes, sin pagar el coste O(n³) de la inversión completa, se recurre a aproximaciones que simplifican la estructura de la Hessiana (o de matrices de curvatura relacionadas, como la matriz de información de Fisher o la matriz Gauss-Newton generalizada, que en la práctica se comportan mejor que la Hessiana exacta para entrenar redes neuronales).

**Aproximación diagonal.** La simplificación más drástica consiste en quedarte solo con la diagonal de la matriz de curvatura y poner a cero todos los elementos fuera de ella. Esto reduce el coste de almacenamiento y de "inversión" a O(n), porque invertir una matriz diagonal equivale simplemente a invertir cada uno de sus elementos por separado. La contrapartida es que pierdes toda la información sobre cómo interactúan entre sí los distintos parámetros, que es precisamente la información que vive fuera de la diagonal.

**Aproximación diagonal por bloques.** Un punto intermedio consiste en dividir la matriz de curvatura en bloques más pequeños a lo largo de la diagonal (por ejemplo, un bloque por cada capa de la red) y tratar cada bloque como si fuera independiente del resto, ignorando de nuevo las interacciones entre bloques distintos. Si divides una Hessiana de tamaño n × n en k bloques de tamaño (n/k) × (n/k), el coste de invertir cada bloque es O((n/k)³), y el coste total de invertir los k bloques es:

k · O((n/k)³) = O(n³ / k²)

Es decir, el coste total se reduce en un factor de k² respecto a invertir la matriz completa, a cambio de perder la información de interacción entre parámetros de bloques distintos. Cuantos más bloques uses (bloques más pequeños), mayor es el ahorro computacional, pero también mayor la pérdida de precisión en la curvatura capturada. Esta aproximación funciona razonablemente bien en redes donde la interacción entre capas distintas es relativamente débil, y peor en arquitecturas donde los parámetros están muy correlacionados, como suele ocurrir en redes convolucionales.

**Aproximación de Kronecker.** Una idea más elaborada, conocida en la literatura por el nombre K-FAC (curvatura aproximada factorizada de Kronecker), explota la estructura particular que adopta la matriz de curvatura en cada capa de una red neuronal. En lugar de aproximar el bloque de curvatura de una capa como una matriz densa cualquiera, lo aproxima como un producto de Kronecker de dos matrices mucho más pequeñas, A y B:

H_capa ≈ A ⊗ B

El producto de Kronecker tiene una propiedad algebraica muy conveniente: la inversa de un producto de Kronecker es el producto de Kronecker de las inversas,

(A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹

Esto significa que, en lugar de invertir una matriz grande, basta con invertir dos matrices mucho más pequeñas por separado, lo que reduce drásticamente el coste computacional y hace factible aplicar información de curvatura incluso en redes de gran tamaño. La contrapartida es que esta aproximación solo funciona bien cuando la Hessiana real de la capa efectivamente se parece a un producto de Kronecker, algo que depende de la arquitectura concreta, y que su implementación es notablemente más compleja que la de las aproximaciones diagonales.

En síntesis, tanto la aproximación diagonal por bloques como la de Kronecker persiguen el mismo objetivo: conservar la parte de la información de curvatura que más aporta a la calidad de la actualización, sacrificando la parte menos relevante, para que el uso de segundo orden sea computacionalmente viable en redes neuronales con muchos parámetros.

## Ejemplo de código comentado, en PyTorch

El siguiente ejemplo entrena el mismo modelo pequeño con tres configuraciones distintas de optimizador, sobre un problema sintético con curvatura mal condicionada (parecido al valle estrecho de la analogía), y compara sus curvas de pérdida. **Copia y ejecuta este código en Google Colab** para ver con tus propios ojos la diferencia entre SGD puro, SGD con momento y Adam.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Fijamos la semilla para que el experimento sea reproducible
torch.manual_seed(0)

# --- Datos sintéticos con curvatura mal condicionada ---
# Generamos un problema de regresión donde una característica tiene una
# escala mucho mayor que la otra: esto crea, de forma artificial, un
# paisaje de pérdida en forma de "valle estrecho", como en la analogía.
n_muestras = 500
X = torch.randn(n_muestras, 2)
X[:, 0] *= 20.0  # una dirección con mucha más varianza que la otra

pesos_reales = torch.tensor([[3.0], [-2.0]])
y = X @ pesos_reales + 0.1 * torch.randn(n_muestras, 1)


def crear_modelo():
    """Modelo lineal simple: una sola capa, sin sesgo, para poder
    comparar los optimizadores sobre el mismo punto de partida."""
    torch.manual_seed(42)  # mismos pesos iniciales en las tres pruebas
    return nn.Linear(2, 1, bias=False)


def entrenar(optimizador_fn, n_pasos=200):
    modelo = crear_modelo()
    perdida_fn = nn.MSELoss()
    optimizador = optimizador_fn(modelo.parameters())

    historial_perdida = []
    for _ in range(n_pasos):
        optimizador.zero_grad()
        pred = modelo(X)
        perdida = perdida_fn(pred, y)
        perdida.backward()
        optimizador.step()
        historial_perdida.append(perdida.item())

    return historial_perdida


# Usamos la misma tasa de aprendizaje base en los tres casos para que
# la comparación sea justa: lo que cambia es solo el optimizador.
tasa_aprendizaje = 0.001

resultados = {
    "SGD (sin momento)": entrenar(
        lambda params: torch.optim.SGD(params, lr=tasa_aprendizaje)
    ),
    "SGD con momento (0.9)": entrenar(
        lambda params: torch.optim.SGD(
            params, lr=tasa_aprendizaje, momentum=0.9
        )
    ),
    "Adam": entrenar(
        lambda params: torch.optim.Adam(params, lr=tasa_aprendizaje)
    ),
}

# --- Gráfico comparativo ---
plt.figure(figsize=(8, 5))
for nombre, perdidas in resultados.items():
    plt.plot(perdidas, label=nombre)

plt.yscale("log")  # la escala logarítmica facilita comparar velocidades de caída
plt.xlabel("Iteración")
plt.ylabel("Pérdida (MSE, escala log)")
plt.title("Comparación de optimizadores en un valle estrecho")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

Si ejecutas este código, verás que la curva de SGD sin momento desciende de forma notablemente más lenta que las otras dos, precisamente por el mal condicionamiento que introducimos al escalar una de las características. La curva de SGD con momento cae con más rapidez, porque suaviza el zigzag entre direcciones. Adam suele converger todavía más rápido en las primeras iteraciones, porque además de acumular momento adapta la tasa de aprendizaje por parámetro, algo que en el apartado siguiente verás con más detalle.

## Optimizadores adaptativos y regularización: buenas prácticas actuales

Este apartado no forma parte del contenido central de la fuente oficial de este capítulo, que se centra en el momento y en los métodos de segundo orden. Lo añadimos como cierre práctico, siguiendo las buenas prácticas estándar de la disciplina, para que sepas que estas herramientas existen y en qué situación conviene recurrir a ellas.

**Optimizadores adaptativos.** RMSprop y Adam, que ya viste en las fórmulas de la sección teórica, ajustan la tasa de aprendizaje de cada parámetro de forma individual, dividiendo por una media móvil de la magnitud de sus gradientes recientes. En la práctica, esto los hace mucho más robustos que el SGD puro frente a paisajes de pérdida con curvaturas muy distintas entre parámetros, sin necesitar la costosa información de la Hessiana. Hoy, Adam es el punto de partida por defecto en la inmensa mayoría de proyectos de deep learning, y RMSprop sigue siendo habitual en tareas como el aprendizaje por refuerzo.

**Dropout.** Durante el entrenamiento, el dropout desactiva aleatoriamente un porcentaje de neuronas en cada paso (un valor habitual es entre el 20 % y el 50 %). Esto obliga a la red a no depender en exceso de neuronas concretas y mejora su capacidad de generalizar a datos nuevos, reduciendo el sobreajuste (overfitting).

**Weight decay (decaimiento de pesos).** Esta técnica añade una pequeña penalización proporcional a la magnitud de los pesos dentro de la función de pérdida, lo que empuja al modelo hacia soluciones con pesos más pequeños y, por tanto, más simples. En PyTorch se activa con el parámetro `weight_decay` de casi cualquier optimizador, incluidos SGD y Adam.

Ninguna de estas técnicas sustituye a lo que has aprendido en este capítulo sobre momento y curvatura: resuelven un problema distinto, el de la capacidad de generalización del modelo, no el de la velocidad de convergencia del optimizador. Pero **combínalas** con un buen optimizador siempre que entrenes un modelo real: la elección del optimizador y las técnicas de regularización trabajan en conjunto, no de forma aislada.

## Resumen de puntos clave

- El descenso de gradiente puro es lento u oscilante en paisajes de pérdida mal condicionados, como los valles estrechos, porque no distingue direcciones con distinta curvatura
- El número de condición (cociente entre curvatura máxima y mínima) determina cuántos pasos necesita un método de primer orden para converger
- El momento acumula una media móvil con decaimiento exponencial de los gradientes pasados, cancelando las oscilaciones y reforzando el avance en direcciones consistentes, como una bola que rueda y acumula inercia
- Nesterov (NAG) mejora el momento clásico calculando el gradiente en la posición anticipada, no en la posición actual
- Los métodos de segundo orden, como Newton, usan la matriz Hessiana (las segundas derivadas de la pérdida) para incorporar información de curvatura y calcular actualizaciones mucho más informadas que el gradiente por sí solo
- El coste de la Hessiana es O(n²) en almacenamiento y O(n³) en inversión, frente al O(n) del descenso de gradiente, lo que hace inviable a Newton puro en redes con muchos parámetros
- Las aproximaciones diagonales por bloques y de Kronecker (K-FAC) reducen ese coste asumiendo una estructura simplificada de la matriz de curvatura, a cambio de perder parte de la información de interacción entre parámetros
- Adam y RMSprop son optimizadores adaptativos de uso estándar hoy, mientras que dropout y weight decay son técnicas de regularización habituales para mejorar la generalización del modelo

## Ponlo en práctica

Dedica entre quince y treinta minutos a esta mini-tarea:

1. **Copia** el código de la sección de ejemplo en un cuaderno de Google Colab
2. **Modifica** el factor de escala de la primera característica (`X[:, 0] *= 20.0`) probando los valores 2, 20 y 100, y vuelve a generar el gráfico comparativo en cada caso
3. **Observa** cómo cambia la distancia entre las tres curvas a medida que el problema está peor condicionado: ¿qué optimizador aguanta mejor cuando el valle se vuelve más estrecho?
4. **Añade** una cuarta curva usando `torch.optim.RMSprop` y comprueba si su comportamiento se parece más al de SGD con momento o al de Adam
5. **Anota** en dos o tres frases qué optimizador elegirías por defecto para un proyecto nuevo y por qué, teniendo en cuenta lo que has observado
