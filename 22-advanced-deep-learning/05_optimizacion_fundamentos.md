# Capítulo 5. Optimización I: cómo aprende una red

## Por qué importa este tema

Ya sabes entrenar una red simple en PyTorch: defines un modelo, calculas una pérdida y llamas a `backward()` para obtener gradientes. Pero, ¿qué ocurre exactamente después? ¿Cómo convierte la red esos gradientes en un modelo que funciona? La respuesta es la optimización, el motor que está detrás de todo el aprendizaje profundo (*deep learning*, DL).

Cada vez que entrenas una red neuronal, resuelves un problema de optimización: ajustar millones de parámetros para que una función de pérdida sea lo más pequeña posible. Si ese proceso falla, no importa lo ingeniosa que sea tu arquitectura ni lo limpios que estén tus datos: la red no aprende. Y si el proceso funciona mal —demasiado lento, o inestable—, pierdes horas o días de cómputo sin obtener nada útil a cambio.

Este capítulo es la primera mitad de la Clase 4 del curso, "Optimización para *machine learning*" (aprendizaje automático). Aquí te centras en el algoritmo más básico y más importante de todos: el descenso de gradiente (*gradient descent*, GD). Comprenderás por qué la tasa de aprendizaje (*learning rate*) es el hiperparámetro que más quebraderos de cabeza da a cualquier persona que entrena modelos, y por qué la forma de la superficie de pérdida —su curvatura— determina si el entrenamiento será una tarea sencilla o una pesadilla. En el próximo capítulo verás cómo el momento (*momentum*) y los métodos de segundo orden resuelven algunos de los problemas que hoy vas a diagnosticar.

**Presta atención**: entender bien este capítulo te da la base para diagnosticar por qué un entrenamiento diverge, por qué se estanca o por qué avanza a paso de tortuga, algo que te ocurrirá con total seguridad en tu carrera como profesional de la inteligencia artificial (IA).

## Explicación conceptual con analogías

Imagina que estás en la cima de una montaña, envuelto en una niebla espesa que solo te deja ver el terreno a tus pies. Tu objetivo es llegar al valle, al punto más bajo posible. No tienes un mapa, así que solo puedes usar una estrategia: en cada punto donde te encuentras, sientes con los pies hacia dónde baja más pronunciadamente el terreno y das un paso en esa dirección. Repites esto una y otra vez, paso a paso, hasta acercarte al fondo del valle.

Esta es, en esencia, la idea del descenso de gradiente. La "montaña" es la función objetivo o función de pérdida: una superficie que mide cuánto se equivoca tu modelo para cada combinación de parámetros. La "posición" en la que te encuentras son los valores actuales de los parámetros de la red (los pesos). Y el "sentido de la pendiente" es el gradiente, un vector que apunta en la dirección de mayor crecimiento de la función. Como quieres bajar, no subir, te mueves en la dirección contraria: el gradiente negativo.

Ahora bien, hay una decisión crucial en esta analogía: el tamaño del paso. Esa es exactamente la tasa de aprendizaje.

- Si das **pasos demasiado grandes**, corres el riesgo de saltar por encima del valle y aterrizar en la ladera opuesta, incluso más arriba de donde empezaste. Repite esto varias veces y, en lugar de acercarte al fondo, te alejas cada vez más: el proceso diverge.
- Si das **pasos demasiado pequeños**, avanzas con total seguridad, pero tardas una eternidad en llegar al fondo. El entrenamiento converge, pero tan lento que puede resultar impracticable.
- Si el paso es el adecuado, avanzas de forma constante y razonablemente rápida hacia el mínimo.

Ahora añade una complicación más a la analogía: imagina que el valle no tiene forma de cuenco simétrico, sino de cañón estrecho y alargado, con paredes muy empinadas a los lados y un fondo casi plano que desciende suavemente en una dirección. Si caminas con pasos grandes, rebotarás de pared a pared del cañón sin avanzar apenas por el fondo: oscilas de un lado a otro en lugar de progresar hacia la salida. Si reduces el paso para evitar el rebote, avanzarás con seguridad por las paredes, pero te moverás con exasperante lentitud a lo largo del fondo plano. No existe una tasa de aprendizaje única que resuelva bien las dos direcciones a la vez.

Esta forma de "cañón estrecho" no es un capricho matemático: es exactamente lo que ocurre en la mayoría de las superficies de pérdida reales de una red neuronal. Algunas direcciones del espacio de parámetros tienen mucha curvatura (paredes empinadas) y otras casi ninguna (fondo plano). A esa relación entre la curvatura máxima y la mínima se le llama número de condición, y es la razón matemática de por qué el descenso de gradiente básico, aunque funciona, puede ser tan lento y delicado de ajustar.

## Desarrollo teórico (fiel a la fuente, con profundidad real)

### La función objetivo

En aprendizaje automático, optimizar significa ajustar los parámetros de un modelo, que representamos como un vector $\theta \in \mathbb{R}^d$, para minimizar una función objetivo (u objetivo de pérdida) $h(\theta)$. Esta función mide los errores que comete el modelo: en clasificación puede ser el error de predicción; en aprendizaje por refuerzo, la recompensa negativa.

El objetivo de entrenamiento típico de una red neuronal se expresa como una suma (o media) sobre los $N$ ejemplos del conjunto de entrenamiento:

$$h(\theta) = \frac{1}{N}\sum_{i=1}^{N} L\big(f(x_i,\theta),\, y_i\big)$$

donde $f$ es la red neuronal, $x_i$ son las entradas, $\theta$ los parámetros, $y_i$ los valores verdaderos y $L$ la función de pérdida que compara la predicción $f(x_i,\theta)$ con el valor real $y_i$.

### La regla de actualización del descenso de gradiente

El descenso de gradiente actualiza los parámetros moviéndolos en la dirección del gradiente negativo de la función objetivo. Su regla de actualización es:

$$\theta_{k+1} = \theta_k - \alpha \, \nabla h(\theta_k)$$

Aquí, $\theta_k$ son los parámetros en el paso $k$, $\alpha$ es la tasa de aprendizaje (también llamada tamaño de paso) y $\nabla h(\theta_k)$ es el gradiente de la función objetivo respecto a los parámetros, evaluado en $\theta_k$. El gradiente marca la dirección de máximo crecimiento de la función; al restarlo, cada paso reduce (o intenta reducir) el valor del objetivo.

### Por qué el gradiente es la dirección de descenso más pronunciado

La intuición detrás de esta regla se apoya en la aproximación de Taylor de primer orden. Si $h$ es suficientemente suave, el valor de la función cerca del punto actual $\theta$ se puede aproximar mediante una expansión lineal:

$$h(\theta + d) \approx h(\theta) + \nabla h(\theta)^{\top} d$$

donde $d$ es un pequeño desplazamiento en el espacio de parámetros. Esta aproximación lineal es fiel al comportamiento real de $h$ solo cuando $d$ es pequeño; cuanto mayor sea el desplazamiento, menos fiable resulta.

El descenso de gradiente minimiza esta aproximación lineal dentro de una región restringida, por ejemplo, una esfera de radio $\alpha$ alrededor de $\theta$. La solución de ese subproblema, dirección que minimiza $\nabla h(\theta)^{\top} d$ sujeta a $\|d\| \le \alpha$, es precisamente $d = -\alpha \, \nabla h(\theta) / \|\nabla h(\theta)\|$: moverse en dirección contraria al gradiente. De ahí el nombre "descenso más pronunciado" (*steepest descent*): dentro de la región de confianza donde la aproximación lineal es válida, ningún otro paso reduce más la función.

Este razonamiento explica también por qué la tasa de aprendizaje no puede ser arbitrariamente grande: cuanto mayor sea el paso, más te alejas de la zona donde la aproximación lineal es correcta, y el paso puede acabar aumentando el valor real de $h$ en lugar de reducirlo, aunque la aproximación lineal predijera una bajada.

### Suavidad, constante de Lipschitz y convexidad fuerte

Para poder decir algo riguroso sobre cuándo y cómo converge el descenso de gradiente, la teoría necesita algunas hipótesis sobre la "forma" de $h$.

La primera hipótesis habitual es que $h$ tiene derivadas continuas de tipo Lipschitz, es decir, que $h$ es "Lipschitz suave". Formalmente, existe una constante $L$ tal que:

$$\|\nabla h(\theta_1) - \nabla h(\theta_2)\| \le L \, \|\theta_1 - \theta_2\| \quad \text{para todo } \theta_1, \theta_2$$

Esta condición dice que el gradiente no puede cambiar de forma demasiado brusca cuando te desplazas un poco en el espacio de parámetros. En términos geométricos, $L$ es una cota superior de la curvatura de $h$: ninguna dirección de la superficie de pérdida se curva más de lo que permite $L$. Por eso a $L$ se le llama también constante de Lipschitz.

La segunda hipótesis, más fuerte y no siempre necesaria, es la convexidad fuerte, con parámetro $\mu$. Esta condición impone una cota inferior a la curvatura: la superficie no puede ser "demasiado plana" en ninguna dirección relevante. Combinadas, $L$ y $\mu$ acotan la curvatura de $h$ por arriba y por abajo respectivamente.

En el caso más habitual del entrenamiento de redes neuronales, la función objetivo no es convexa en general, pero cerca de un mínimo local su curvatura puede acotarse por abajo mediante un término cuadrático con parámetro de curvatura $\mu$, lo que permite aplicar razonamientos similares aunque de forma local.

### La tasa de aprendizaje máxima y la estabilidad

Con la constante de Lipschitz $L$ como cota superior de la curvatura, se puede derivar la tasa de aprendizaje máxima que garantiza que el descenso de gradiente no diverja:

$$\alpha \le \frac{1}{L}$$

Puedes entender este límite recordando la analogía del cañón: $L$ mide cuán "empinada" es la ladera más pronunciada de toda la superficie. Si tu paso es mayor que $1/L$, en esa dirección de máxima curvatura acabarás saltando de una pared a la opuesta, y el proceso puede oscilar cada vez con más fuerza hasta diverger. Un paso de tamaño $1/L$ o menor garantiza que, incluso en la dirección más curvada, cada actualización reduzca (o como mínimo no aumente) el valor del objetivo.

Este resultado también permite reinterpretar el propio descenso de gradiente: es matemáticamente equivalente a minimizar una aproximación de segundo orden de $h$ en la que la matriz de curvatura real (la matriz Hessiana) se sustituye por $L$ multiplicada por la matriz identidad. Es decir, el descenso de gradiente asume, de forma conservadora, que la curvatura es igual de alta en todas las direcciones, cuando en realidad casi nunca es así. Esta es la raíz matemática del problema del "cañón estrecho": el algoritmo trata el fondo plano con la misma cautela que las paredes empinadas, porque no distingue direcciones de alta y baja curvatura. En el próximo capítulo verás cómo el momento y los métodos de segundo orden corrigen justamente esta limitación.

### El número de condición y la cota de convergencia

La dificultad real de optimizar una función no depende solo de $L$, sino de la relación entre la curvatura máxima y la curvatura mínima. Esa relación se llama número de condición, $\kappa$:

$$\kappa = \frac{L}{\mu}$$

es decir, el cociente entre el mayor y el menor autovalor de la matriz Hessiana (o, de forma equivalente, entre la curvatura máxima y la mínima de la función). El número de condición ofrece una caracterización global de lo "difícil" que es un problema de optimización: cuanto mayor sea $\kappa$, más se parece la superficie de pérdida a un cañón estrecho y alargado, y más despacio converge el descenso de gradiente.

Bajo las hipótesis de suavidad Lipschitz y convexidad fuerte, con la tasa de aprendizaje adecuada, se puede demostrar que el error, medido como la diferencia entre el valor actual y el valor óptimo, decrece de forma aproximadamente exponencial con el número de iteraciones $k$:

$$h(\theta_k) - h(\theta^\ast) \;\le\; \left(1 - \frac{1}{\kappa}\right)^{k} \big[h(\theta_0) - h(\theta^\ast)\big]$$

donde $\theta^\ast$ es el punto óptimo. A partir de esta cota se deriva cuántas iteraciones $k$ hacen falta para alcanzar un error menor que $\varepsilon$:

$$k = O\!\left(\kappa \log \frac{1}{\varepsilon}\right)$$

Es decir, el número de iteraciones necesarias crece de forma proporcional al número de condición. Un problema con $\kappa$ pequeño (superficie de pérdida "redondeada", parecida a un cuenco) converge rápido; un problema con $\kappa$ grande (cañón muy estrecho) puede necesitar muchísimas más iteraciones para alcanzar la misma precisión.

### Los límites de esta teoría

Estas cotas son valiosas para construir intuición, pero conviene no tomarlas al pie de la letra en la práctica, por varias razones:

- Son demasiado pesimistas, porque tienen que cubrir el peor caso posible, no el caso típico
- Algunas hipótesis, como la convexidad, son demasiado fuertes: la mayoría de las funciones objetivo de una red neuronal no son convexas
- Otras hipótesis son demasiado débiles: los problemas reales suelen tener estructura adicional que estas cotas no capturan
- Se apoyan en medidas "toscas" del problema, como el número de condición global, que puede no reflejar bien la dificultad real
- Se centran en el comportamiento asintótico (cuando $k$ es muy grande), pero en la práctica casi nunca entrenas hasta la convergencia completa; lo que ocurre en las primeras iteraciones importa tanto o más

Por eso, la elección del optimizador debe guiarse sobre todo por la experiencia práctica, apoyada por estas intuiciones teóricas, y no al revés. **Guarda esta idea**: la teoría te ayuda a entender por qué algo puede fallar, pero la validación empírica es la que decide qué optimizador usar en cada problema.

## Ejemplo de código comentado

El siguiente ejemplo implementa el descenso de gradiente a mano, sin ningún framework de aprendizaje profundo, sobre una función de pérdida cuadrática de dos parámetros. Eliges la curvatura de cada dirección para simular un "cañón estrecho" y comparas tres tasas de aprendizaje: una adecuada, una demasiado alta (que diverge) y una demasiado baja (que converge muy despacio). Funciona directamente en Google Colab; solo necesitas NumPy y Matplotlib, que ya vienen preinstalados.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definimos la función de pérdida cuadrática ---
# h(theta) = 0.5 * (a * theta_1^2 + b * theta_2^2)
# 'a' y 'b' son las curvaturas en cada dirección: si son muy distintas,
# la superficie se parece al "cañón estrecho" de la analogía del capítulo.
a, b = 1.0, 20.0  # número de condición kappa = b / a = 20

def loss(theta):
    return 0.5 * (a * theta[0] ** 2 + b * theta[1] ** 2)

def gradient(theta):
    # Derivadas parciales de la cuadrática respecto a cada parámetro
    return np.array([a * theta[0], b * theta[1]])

# --- 2. Implementamos el descenso de gradiente a mano ---
def gradient_descent(theta_init, alpha, num_steps):
    theta = np.array(theta_init, dtype=float)
    trayectoria = [theta.copy()]
    for _ in range(num_steps):
        theta = theta - alpha * gradient(theta)  # regla de actualización de GD
        trayectoria.append(theta.copy())
        # Si los valores se disparan, detenemos para no desbordar el tipo float
        if np.any(np.abs(theta) > 1e6):
            break
    return np.array(trayectoria)

# --- 3. Punto de partida y tasas de aprendizaje a comparar ---
theta_inicial = [4.0, 4.0]
num_pasos = 40

# La tasa máxima teórica de estabilidad es 1 / L, con L = max(a, b) = b
L = max(a, b)
alpha_estable_max = 1.0 / L

tasas = {
    "Buena (alpha = 0.04)": 0.04,
    "Demasiado alta (alpha = 0.11)": 0.11,   # supera 1/L y hace diverger el proceso
    "Demasiado baja (alpha = 0.005)": 0.005,  # converge, pero muy lentamente
}

resultados = {
    nombre: gradient_descent(theta_inicial, alpha, num_pasos)
    for nombre, alpha in tasas.items()
}

# --- 4. Visualizamos la superficie de pérdida y las trayectorias ---
theta1_rango = np.linspace(-6, 6, 200)
theta2_rango = np.linspace(-6, 6, 200)
T1, T2 = np.meshgrid(theta1_rango, theta2_rango)
Z = 0.5 * (a * T1 ** 2 + b * T2 ** 2)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel izquierdo: curvas de nivel con las tres trayectorias
ax = axes[0]
ax.contour(T1, T2, Z, levels=25, cmap="viridis", alpha=0.6)
for nombre, trayectoria in resultados.items():
    ax.plot(trayectoria[:, 0], trayectoria[:, 1], marker="o", markersize=3, label=nombre)
ax.plot(0, 0, "r*", markersize=15, label="Mínimo")
ax.set_xlabel("theta_1 (dirección de baja curvatura)")
ax.set_ylabel("theta_2 (dirección de alta curvatura)")
ax.set_title("Trayectorias del descenso de gradiente")
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.legend(fontsize=8)

# Panel derecho: evolución de la pérdida en escala logarítmica
ax = axes[1]
for nombre, trayectoria in resultados.items():
    valores_perdida = [loss(theta) for theta in trayectoria]
    ax.plot(valores_perdida, marker="o", markersize=3, label=nombre)
ax.set_yscale("log")
ax.set_xlabel("Iteración")
ax.set_ylabel("Pérdida h(theta) [escala log]")
ax.set_title("Convergencia según la tasa de aprendizaje")
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

print(f"Numero de condicion (kappa) = {b / a:.1f}")
print(f"Tasa de aprendizaje maxima estable (1/L) = {alpha_estable_max:.4f}")
```

Al ejecutar este código, observa tres comportamientos muy distintos en el panel derecho:

- La curva con tasa buena baja de forma constante y suave
- La curva con tasa demasiado alta oscila y crece sin control: en el panel izquierdo verás cómo la trayectoria rebota entre los dos lados del "cañón" en la dirección de alta curvatura
- La curva con tasa demasiado baja sí converge, pero necesita muchísimas más iteraciones para acercarse al mínimo, y avanza especialmente despacio a lo largo de la dirección plana

**Modifica el valor de `b`** en tu copia del cuaderno y observa cómo, al aumentar el número de condición, se hace cada vez más difícil encontrar una tasa de aprendizaje que funcione bien en las dos direcciones a la vez.

## Resumen de puntos clave

- Optimizar una red neuronal significa minimizar una función objetivo (o de pérdida) ajustando sus parámetros paso a paso
- El descenso de gradiente actualiza los parámetros según $\theta_{k+1} = \theta_k - \alpha \nabla h(\theta_k)$: se mueve en dirección contraria al gradiente, la dirección de mayor crecimiento de la pérdida
- Esta regla surge de minimizar la aproximación de Taylor de primer orden de la función objetivo dentro de una región pequeña alrededor del punto actual
- La tasa de aprendizaje $\alpha$ controla el tamaño del paso: si es demasiado alta, el proceso oscila y diverge; si es demasiado baja, converge, pero muy despacio
- La constante de Lipschitz $L$ acota la curvatura máxima de la función; la tasa de aprendizaje máxima que garantiza estabilidad es $\alpha \le 1/L$
- El número de condición $\kappa = L/\mu$ mide la relación entre la curvatura máxima y la mínima; cuanto mayor sea, más "estrecho" es el cañón de la superficie de pérdida y más lenta es la convergencia
- El número de iteraciones necesarias para alcanzar un error $\varepsilon$ crece de forma proporcional a $\kappa$: $k = O(\kappa \log(1/\varepsilon))$
- El descenso de gradiente es matemáticamente equivalente a minimizar una aproximación de segundo orden con curvatura uniforme igual a $L$ en todas las direcciones, lo que explica por qué le cuesta tanto navegar superficies con curvaturas muy distintas según la dirección
- Las cotas teóricas de convergencia son útiles como guía intuitiva, pero suelen ser demasiado pesimistas respecto al comportamiento real en problemas de aprendizaje profundo

## Ponlo en práctica

Dedica entre quince y treinta minutos a esta mini-tarea, usando como base el código de la sección anterior.

1. Abre un cuaderno nuevo en Google Colab y copia el ejemplo de código de este capítulo
2. **Cambia** el valor de `b` a diez, cincuenta y cien, manteniendo `a = 1.0`, y ejecuta el cuaderno con cada valor
3. **Anota**, para cada número de condición, qué tasa de aprendizaje es la mayor que sigue siendo estable (puedes calcularla con la fórmula $\alpha \le 1/L$ y comprobarla probando valores cercanos)
4. **Observa** cuántas iteraciones necesita la tasa "buena" para bajar la pérdida por debajo de 0,01 en cada caso, y comprueba si esa cantidad crece aproximadamente de forma proporcional al número de condición, tal como predice la teoría
5. **Comparte** con tus compañeros de curso, si es posible, tus gráficas junto con una frase que explique con tus propias palabras por qué un número de condición alto hace más difícil elegir una buena tasa de aprendizaje

Si terminas antes de tiempo, prueba a sustituir la función cuadrática por una función no convexa sencilla, por ejemplo la función de Rosenbrock, y observa cómo cambia el comportamiento del descenso de gradiente cuando la curvatura varía de un punto a otro en lugar de ser constante.
