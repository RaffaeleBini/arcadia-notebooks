# Capítulo 3. El dilema de los bandidos: exploración y explotación

Imagina que te mudas a una ciudad nueva. Tienes un restaurante que te gusta a cinco minutos de casa: sabes exactamente qué vas a comer y sabes que te va a gustar. Pero, a dos calles, hay un local que nunca has probado. ¿Vuelves a tu opción segura o te arriesgas a descubrir algo mejor?

Esa pregunta, tan cotidiana, es el mismo problema que enfrenta un equipo médico al elegir entre un tratamiento consolidado y uno experimental, o una plataforma digital al decidir qué anuncio mostrarte. En los tres casos existe una tensión entre dos impulsos: explotar lo que ya sabes que funciona y explorar para averiguar si hay algo mejor. En Aprendizaje por Refuerzo (RL, por sus siglas en inglés), a esta tensión se le llama el dilema entre exploración y explotación, y es uno de los aspectos más importantes de todo el campo.

En el capítulo anterior conociste las piezas básicas de RL: un agente que actúa, un entorno que responde, una política que decide qué hacer y una recompensa que mide si esa decisión fue buena. En este capítulo, vas a ver cómo un agente decide, paso a paso, entre repetir lo conocido y probar algo nuevo. Para hacerlo de forma sencilla, vas a trabajar con la versión más simple posible de un problema de decisión: el bandido multi-brazo.

## El problema del bandido multi-brazo

En un casino, una máquina tragaperras se conoce, en inglés, como "one-armed bandit" (bandido de un brazo), porque tiene una sola palanca y "te roba" el dinero poco a poco. Ahora imagina una fila de varias de estas máquinas, una junto a otra. Cada máquina paga premios distintos: unas pagan mucho, con frecuencia; otras casi nunca pagan nada. El problema es que tú no sabes cuál es cuál. Tu única forma de averiguarlo es tirar de las palancas y observar qué ocurre. Ese escenario completo se llama bandido multi-brazo (multi-armed bandit), y da nombre a esta familia de problemas.

Cada palanca representa una acción posible. Cada acción tiene un valor verdadero: la recompensa media que obtendrías si la eligieras infinitas veces. Ese valor existe, pero tú no lo conoces de entrada; solo puedes estimarlo a partir de lo que observas cada vez que tiras de una palanca. Además, en esta versión simplificada del problema, cada máquina siempre reparte premios según la misma distribución fija: no cambia con el tiempo ni reacciona a tus decisiones anteriores. Por eso se dice que es un problema "sin estado": lo único que cambia mientras juegas no es el entorno, sino tu propio conocimiento sobre él.

Un ejemplo clásico ayuda a entender por qué esto es más difícil de lo que parece. Piensa en una rata de laboratorio que puede escoger entre dos palancas, una sobre fondo negro y otra sobre fondo blanco. Un día, la palanca negra da queso; otro día, da una descarga. Si la rata se guiara solo por el último resultado, o por una sola mala experiencia, podría abandonar para siempre una palanca que, en promedio, es la mejor opción. Lo mismo le pasa a cualquier agente que decide de forma demasiado precipitada: una mala racha inicial puede convencerlo de evitar la mejor acción posible durante el resto del juego.

## Estimando el valor de cada acción

Si no conoces el valor real de una palanca, lo razonable es estimarlo a partir de lo que has observado. La forma más directa es calcular la media de las recompensas que has recibido cada vez que elegiste esa acción. A esa estimación se le llama valor de acción, y se actualiza cada vez que vuelves a probar la misma palanca.

No hace falta guardar todo el historial de recompensas para calcular esa media. Puedes actualizarla de forma incremental: cada vez que observas una recompensa nueva, ajustas tu estimación anterior un poco hacia ese nuevo valor, con un peso que disminuye cuanto más veces hayas probado esa acción. Al principio, cada nueva observación cambia mucho tu estimación; con el tiempo, cada observación adicional pesa menos y tu estimación se estabiliza cerca del valor real.

Esta misma idea admite una variante útil: en vez de dar el mismo peso a todas las observaciones pasadas, puedes usar una tasa de paso fija (un "learning rate" constante) que dé más peso a las recompensas recientes. Esto es especialmente útil cuando el entorno cambia con el tiempo, es decir, cuando ya no es estacionario: una máquina que hoy paga bien puede pagar peor mañana, y prefieres que tu estimación reaccione rápido a ese cambio en vez de diluirlo entre cientos de observaciones antiguas.

## Epsilon-greedy: explorar un poco, explotar mucho

La estrategia más sencilla para resolver el dilema se llama epsilon-greedy (ε-greedy). Funciona así: define un número pequeño, épsilon (ε), entre cero y uno. En cada decisión, con probabilidad ε eliges una acción al azar (exploras); con probabilidad 1−ε, eliges la acción que hasta ahora parece mejor según tus estimaciones (explotas). Un detalle curioso: como la elección aleatoria también puede coincidir con la mejor acción conocida, la probabilidad real de explotar es ligeramente mayor que 1−ε.

Volvamos al ejemplo del restaurante. Si tu ε es 0,1, entonces el 90% de las veces vas a tu sitio de confianza y el 10% de las veces pruebas algo distinto, elegido al azar entre todas las opciones disponibles. Ese 10% es lo que te permite descubrir, tarde o temprano, si existe un restaurante mejor que el tuyo.

Un ε fijo garantiza que sigas explorando para siempre, así que con el tiempo acabarás probando todas las acciones las veces suficientes para conocer su valor real con bastante precisión. Pero tiene un coste: como nunca dejas de elegir acciones al azar, seguirás cometiendo el mismo porcentaje de errores incluso después de haber aprendido cuál es la mejor opción. Por eso, una mejora habitual es hacer que ε decrezca con el tiempo: explorar mucho al principio, cuando sabes poco, y explotar cada vez más a medida que tu conocimiento mejora. Esta estrategia, con sus variantes, es la que se usó, por ejemplo, en Deep Q-Network (DQN), uno de los sistemas que aprendió a jugar videojuegos de Atari directamente a partir de los píxeles de la pantalla.

## Upper Confidence Bound: el optimismo ante la incertidumbre

Epsilon-greedy tiene una limitación: cuando explora, elige al azar entre todas las acciones, sin distinguir entre una palanca que ya conoces bien y otra que apenas has probado. Upper Confidence Bound (UCB) resuelve esto de una forma más inteligente, aplicando una idea que suele resumirse como "el optimismo ante la incertidumbre".

La lógica de UCB es la siguiente: en vez de mirar solo tu estimación actual del valor de cada acción, súmale un "bono de incertidumbre". Ese bono es grande para las acciones que has probado pocas veces (porque no sabes bien cuánto valen, así que les das el beneficio de la duda) y pequeño para las que has probado muchas veces (porque ya tienes una estimación bastante fiable). En cada paso, eliges la acción cuya suma de valor estimado más bono sea la más alta.

Al principio del proceso, como no has probado ninguna palanca, el bono de las acciones sin probar es enorme, así que UCB te obliga a probar cada acción al menos una vez antes de empezar a comparar. A medida que juegas, el bono de cada palanca se va reduciendo conforme la pruebas más, siguiendo aproximadamente el ritmo de la raíz cuadrada del número de veces que la has intentado. Esto hace que el algoritmo pase, de forma natural y sin que tengas que ajustar nada a mano, de una fase con mucha exploración a una fase centrada en explotar lo que ya sabe.

Existen variantes de UCB para distintos escenarios: UCB1-Tuned ajusta el bono según cuánto varían las recompensas observadas; UCB Descontado da más peso a las observaciones recientes, útil en entornos no estacionarios; UCB Contextual añade información adicional sobre la situación (por ejemplo, características del usuario en un sistema de recomendación) para personalizar la decisión. En el capítulo 9 verás con más detalle la teoría que explica por qué UCB funciona tan bien y cómo se compara matemáticamente con otras estrategias.

## Thompson Sampling: una alternativa bayesiana

Existe una tercera familia de estrategias, de corte bayesiano, que se conoce como Thompson Sampling (también llamado muestreo de Thompson o bandido bayesiano). En vez de guardar una sola estimación por acción, mantiene una distribución de probabilidad completa sobre el valor de cada acción: una especie de "creencia" que refleja tanto tu mejor estimación como tu incertidumbre sobre ella.

La idea, en cada paso, es simple: para cada acción, obtén un valor de muestra a partir de su distribución de creencia actual y elige la acción con la muestra más alta. Después, observa la recompensa real y actualiza la creencia de esa acción. Cuando una acción tiene una distribución muy incierta (porque la has probado poco), sus muestras varían mucho y a veces salen valores altos, así que de vez en cuando la eliges: eso es exploración. Cuando una acción ya tiene una distribución muy consolidada en torno a un valor alto, sus muestras son casi siempre altas y consistentes: eso es explotación. Todo ocurre de forma natural, sin necesitar un parámetro como ε ni una fórmula de bono explícita.

Para recompensas binarias (éxito o fracaso), Thompson Sampling suele usar la distribución Beta, que se actualiza de una manera muy sencilla: empiezas con una distribución neutra que no favorece ninguna opción y, cada vez que observas un éxito, ajustas un parámetro; cada vez que observas un fracaso, ajustas el otro. No hace falta ningún cálculo complicado.

## Ejemplo en Python: epsilon-greedy contra UCB

El siguiente código simula un bandido de varios brazos y compara, de forma visual, cómo se comporta epsilon-greedy frente a UCB a lo largo de muchos pasos. Puedes copiarlo y ejecutarlo directamente en Google Colab: solo necesita `numpy` y `matplotlib`, que ya vienen preinstalados.

```python
import numpy as np
import matplotlib.pyplot as plt

# Fijamos una semilla para que el experimento sea reproducible
np.random.seed(42)

# --- Configuración del bandido multi-brazo ---
n_brazos = 10          # número de "palancas" o acciones disponibles
n_pasos = 1000          # cuántas veces va a jugar cada estrategia
n_repeticiones = 200    # repetimos el experimento varias veces para suavizar el azar

# El valor verdadero de cada brazo se genera al azar (el agente NO lo conoce)
# Estos valores representan la recompensa media real de cada acción
valores_reales = np.random.normal(loc=0.0, scale=1.0, size=n_brazos)


def tirar_de_la_palanca(brazo):
    """Simula el resultado de elegir un brazo: devuelve una recompensa con ruido
    alrededor del valor verdadero de ese brazo (como en una tragaperras real)."""
    return np.random.normal(loc=valores_reales[brazo], scale=1.0)


def ejecutar_epsilon_greedy(epsilon, n_pasos):
    """Estrategia epsilon-greedy: explora al azar con probabilidad epsilon,
    y el resto de las veces elige la acción con mejor valor estimado."""
    valores_estimados = np.zeros(n_brazos)   # Q_t(a): estimación actual de cada brazo
    conteo_intentos = np.zeros(n_brazos)     # cuántas veces se ha probado cada brazo
    recompensas = np.zeros(n_pasos)

    for t in range(n_pasos):
        if np.random.random() < epsilon:
            # Exploración: elige un brazo al azar
            brazo = np.random.randint(n_brazos)
        else:
            # Explotación: elige el brazo con mejor valor estimado hasta ahora
            brazo = np.argmax(valores_estimados)

        recompensa = tirar_de_la_palanca(brazo)
        conteo_intentos[brazo] += 1

        # Actualización incremental de la media muestral (ver sección "Estimando
        # el valor de cada acción"): Q_{n+1} = Q_n + (1/n) * (R_n - Q_n)
        paso = 1.0 / conteo_intentos[brazo]
        valores_estimados[brazo] += paso * (recompensa - valores_estimados[brazo])

        recompensas[t] = recompensa

    return recompensas


def ejecutar_ucb(c, n_pasos):
    """Estrategia UCB: combina el valor estimado con un bono de incertidumbre
    que favorece a los brazos poco probados."""
    valores_estimados = np.zeros(n_brazos)
    conteo_intentos = np.zeros(n_brazos)
    recompensas = np.zeros(n_pasos)

    for t in range(1, n_pasos + 1):
        # Si algún brazo aún no se ha probado, se elige primero (bono "infinito")
        brazos_sin_probar = np.where(conteo_intentos == 0)[0]
        if len(brazos_sin_probar) > 0:
            brazo = brazos_sin_probar[0]
        else:
            # UCB: valor estimado + bono de exploración que crece con log(t)
            # y se reduce con la raíz cuadrada del número de intentos del brazo
            bono = c * np.sqrt(np.log(t) / conteo_intentos)
            brazo = np.argmax(valores_estimados + bono)

        recompensa = tirar_de_la_palanca(brazo)
        conteo_intentos[brazo] += 1
        paso = 1.0 / conteo_intentos[brazo]
        valores_estimados[brazo] += paso * (recompensa - valores_estimados[brazo])

        recompensas[t - 1] = recompensa

    return recompensas


# --- Ejecutamos ambas estrategias varias veces y promediamos ---
recompensas_epsilon = np.zeros(n_pasos)
recompensas_ucb = np.zeros(n_pasos)

for _ in range(n_repeticiones):
    # Re-generamos los valores reales en cada repetición para que el promedio
    # no dependa de una única configuración de brazos
    valores_reales = np.random.normal(loc=0.0, scale=1.0, size=n_brazos)
    recompensas_epsilon += ejecutar_epsilon_greedy(epsilon=0.1, n_pasos=n_pasos)
    recompensas_ucb += ejecutar_ucb(c=2.0, n_pasos=n_pasos)

recompensas_epsilon /= n_repeticiones
recompensas_ucb /= n_repeticiones

# Convertimos la recompensa por paso en recompensa media acumulada
recompensa_media_acumulada_epsilon = np.cumsum(recompensas_epsilon) / np.arange(1, n_pasos + 1)
recompensa_media_acumulada_ucb = np.cumsum(recompensas_ucb) / np.arange(1, n_pasos + 1)

# --- Graficamos la comparación ---
plt.figure(figsize=(10, 5))
plt.plot(recompensa_media_acumulada_epsilon, label="epsilon-greedy (ε = 0.1)")
plt.plot(recompensa_media_acumulada_ucb, label="UCB (c = 2.0)")
plt.xlabel("Paso de tiempo")
plt.ylabel("Recompensa media acumulada")
plt.title("Comparación epsilon-greedy vs. UCB en un bandido de 10 brazos")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Al ejecutarlo, verás que ambas curvas mejoran con el tiempo, pero UCB suele adelantarse a epsilon-greedy: como reparte su exploración de forma más inteligente (probando primero lo que menos conoce), tarda menos en encontrar el brazo óptimo y desperdicia menos intentos en brazos claramente malos.

## Puntos clave

- El dilema entre exploración y explotación aparece en cualquier decisión bajo incertidumbre, dentro y fuera del RL
- El bandido multi-brazo es la versión más simple de este problema: varias acciones, cada una con un valor real desconocido y fijo en el tiempo
- El valor de una acción se estima con la media de las recompensas observadas, y se puede actualizar de forma incremental sin guardar todo el historial
- Epsilon-greedy explora al azar con probabilidad ε y explota el resto del tiempo; un ε decreciente suele funcionar mejor que uno fijo
- UCB dirige la exploración hacia las acciones más inciertas, sumando un bono que se reduce a medida que una acción se prueba más
- Thompson Sampling usa distribuciones de probabilidad completas y equilibra exploración y explotación de forma natural, sin parámetros explícitos como ε

## Ponlo en práctica

Toma el código de este capítulo y modifícalo tú misma o tú mismo:

- **Cambia el valor de ε** en `ejecutar_epsilon_greedy` (prueba con 0,01, 0,1 y 0,5) y observa cómo afecta a la curva de recompensa media acumulada
- **Cambia el número de brazos** `n_brazos` (por ejemplo, a 5 o a 20) y comprueba si UCB mantiene su ventaja sobre epsilon-greedy
- **Prueba distintos valores de `c`** en UCB y anota a partir de qué valor la estrategia empieza a explorar demasiado

¡Ejecuta el notebook, cambia los parámetros y comparte con tu equipo qué combinación encontró antes el mejor brazo!

---

En el capítulo 4 vas a dejar atrás el mundo "sin estado" de los bandidos para entrar en los Procesos de Decisión de Markov, donde las acciones del agente sí cambian el estado del entorno.
