# Capítulo 7. Control sin modelo: SARSA y Q-learning

En el capítulo anterior aprendiste a evaluar una política fija: dado un agente que siempre actúa igual, calculaste cuánta recompensa espera obtener desde cada estado, con Monte Carlo (MC) y con Diferencia Temporal (TD, por sus siglas en inglés *Temporal Difference*). Eso se llama **predicción**, y es útil, pero deja una pregunta abierta: ¿y si la política de partida no es buena? Estimar el valor de una estrategia mediocre no la convierte en una estrategia mejor.

Este capítulo da el siguiente paso: el **control**. Ya no te conformas con medir cuánto vale una política, quieres que el agente la mejore mientras aprende, episodio tras episodio, hasta acercarse a la política óptima. Para lograrlo vas a conocer dos algoritmos que son el corazón de gran parte del aprendizaje por refuerzo (RL, por sus siglas en inglés *Reinforcement Learning*) moderno: SARSA (siglas de *State-Action-Reward-State-Action*) y Q-learning. Ambos aprenden sin modelo, es decir, sin conocer de antemano cómo funciona el entorno, y ambos parten de la misma pregunta ("¿qué tan buena es esta acción en este estado?"), pero responden de forma distinta, con consecuencias prácticas muy diferentes.

## De predecir a mejorar: Iteración de Política Generalizada

Imagina que aprendes a cocinar un plato nuevo. Lo preparas, lo pruebas, ajustas la receta un poco, vuelves a prepararlo, lo pruebas otra vez y ajustas de nuevo. No esperas a dominar la receta perfecta en la teoría antes de cocinar por primera vez: alternas entre "evaluar cómo quedó" y "mejorar la receta", una y otra vez, hasta acercarte a algo excelente.

Esa alternancia es exactamente la idea detrás de la Iteración de Política Generalizada (GPI, por sus siglas en inglés *Generalized Policy Iteration*), un principio que estructura casi todo el control en RL:

1. **Evaluación**: estima cuánto vale cada acción bajo la política actual
2. **Mejora**: ajusta la política para que favorezca las acciones con mejor valor estimado
3. **Repite**: vuelve al paso uno con la política mejorada

Lo interesante de GPI es que no necesitas completar cada paso a la perfección para que funcione. No hace falta evaluar la política hasta el último detalle antes de mejorarla: puedes evaluar un poco, mejorar un poco, evaluar un poco más, y esa versión "a medias" del proceso converge igualmente hacia la política óptima en el caso tabular (cuando guardas un valor explícito por cada combinación de estado y acción). SARSA y Q-learning son, en el fondo, dos formas distintas de ejecutar GPI paso a paso, actualizando el valor después de cada acción individual en vez de esperar a que termine el episodio completo.

Una diferencia importante frente al capítulo anterior: en control ya no basta con estimar el valor de estado V(s). Como no tienes un modelo del entorno, necesitas saber el valor de cada **acción** en cada estado, Q(s,a), porque así puedes mejorar la política simplemente eligiendo la acción con mayor Q, sin necesidad de simular qué pasaría con cada alternativa.

## El dilema de explorar, resuelto con ε-greedy y GLIE

Si el agente solo elige siempre la acción que hoy parece mejor, nunca descubre si otra acción, todavía no probada, es en realidad superior. Por eso el control necesita una estrategia de exploración. La más habitual es la política **ε-greedy** (epsilon-greedy): con probabilidad ε el agente elige una acción al azar (explora), y con probabilidad 1−ε elige la acción con mayor valor Q conocido hasta ahora (explota). Un valor típico de ε podría ir decayendo con el tiempo, por ejemplo proporcional a 1/k, donde k es el número de episodios: al principio exploras mucho porque sabes poco, y con el tiempo explotas más porque ya confías en tus estimaciones.

Para garantizar que este proceso converja de verdad hacia la política óptima, existe una condición formal llamada GLIE (siglas en inglés de *Greedy in the Limit with Infinite Exploration*, "codiciosa en el límite con exploración infinita"). Exige dos cosas: que cada combinación de estado y acción se visite un número infinito de veces (para no dejar ninguna opción sin probar lo suficiente) y que la política se vuelva completamente codiciosa a medida que pasa el tiempo (para que, al final, dejes de explorar y aproveches lo aprendido). Decaer ε de forma progresiva es la manera habitual de cumplir esta condición.

Antes de entrar en SARSA y Q-learning, vale la pena mencionar que también existe el **control por Monte Carlo**: evalúas Q(s,a) promediando retornos completos de episodios (como en el capítulo anterior) y mejoras la política haciéndola ε-greedy respecto a esos valores, repitiendo el ciclo. Funciona, pero como todo lo que se basa en MC, necesita episodios completos y tiende a tener más varianza. SARSA y Q-learning resuelven el mismo problema actualizando en cada paso, sin esperar al final del episodio, gracias al bootstrapping de TD.

## SARSA: aprende la política que realmente sigues

SARSA toma su nombre de la secuencia que usa en cada actualización: estado, acción, recompensa, estado siguiente, acción siguiente (*State-Action-Reward-State-Action*). El agente está en un estado, elige una acción con su política ε-greedy, observa la recompensa y el nuevo estado, y entonces —este es el paso clave— **elige también la siguiente acción con esa misma política ε-greedy**, antes de actualizar. La actualización usa el valor Q de esa acción siguiente, la que realmente se va a ejecutar:

Q(S,A) se acerca un poco a: recompensa recibida + valor descontado de Q en el estado siguiente, para la acción que la política ε-greedy eligió de verdad

Por eso SARSA es un algoritmo **on-policy** (dentro de política): aprende el valor de la política que efectivamente está siguiendo, incluida su propia aleatoriedad exploratoria. Si tu política ε-greedy a veces comete "errores" al explorar, SARSA incorpora esos errores en su estimación de valor, y eso lo hace más prudente: aprende a evitar zonas donde una exploración desafortunada podría salir cara.

## Q-learning: aprende la política óptima, aunque explores de otra forma

Q-learning cambia un solo detalle, pero con consecuencias grandes. En vez de usar el valor Q de la acción que realmente vas a tomar a continuación, usa el valor **máximo** posible entre todas las acciones del estado siguiente, sin importar cuál elijas después de verdad:

Q(S,A) se acerca un poco a: recompensa recibida + valor descontado del mejor Q posible en el estado siguiente (el máximo sobre todas las acciones)

Esto convierte a Q-learning en un algoritmo **off-policy** (fuera de política): la política de comportamiento (con la que exploras, por ejemplo ε-greedy) puede ser distinta de la política objetivo que aprendes (la puramente codiciosa, la óptima). Q-learning aprende el valor de "si a partir de ahora actuara siempre de forma óptima", incluso mientras sigue explorando de forma imperfecta. Esta separación entre "cómo exploro" y "qué aprendo" es justo la definición de aprendizaje off-policy, y también permite, por ejemplo, aprender a partir de datos recogidos con otra política o incluso ya registrados de antemano.

## La analogía del acantilado: prudente frente a arriesgado

La forma más clara de sentir la diferencia entre ambos algoritmos es el ejemplo clásico del **acantilado** (*Cliff Walking*). Imagina una cuadrícula donde el agente empieza en la esquina inferior izquierda y el objetivo está en la esquina inferior derecha. Entre ambos, a lo largo del borde inferior, hay un acantilado: caer en él da una penalización fuerte y te devuelve al punto de partida. Cada paso normal cuesta un poco (una pequeña penalización), así que conviene llegar cuanto antes.

El camino más corto bordea el acantilado, pegado al borde. Es también el más arriesgado, porque un solo paso en falso —por ejemplo, un movimiento exploratorio aleatorio de ε-greedy— te tira por el borde.

- **Q-learning** aprende que ese camino corto y pegado al borde es el óptimo, porque calcula el valor asumiendo que a partir de ahora actuará siempre de forma óptima, sin considerar que su propia exploración podría hacerle caer. El resultado final, una vez dejas de explorar, es la ruta más corta posible
- **SARSA** aprende una ruta más alejada del borde, más larga pero más segura, porque su estimación de valor sí tiene en cuenta que, mientras explora, a veces se equivoca, y un paso en falso cerca del acantilado sale muy caro

Durante el propio entrenamiento, mientras ambos agentes siguen explorando con ε-greedy, SARSA suele acumular más recompensa por episodio porque cae menos veces por el acantilado. Q-learning, en cambio, cae más durante el entrenamiento (porque su política objetivo ignora el riesgo de la exploración), pero termina aprendiendo la política objetivo más corta y óptima. Ninguno de los dos "gana" en abstracto: cada uno responde a una pregunta distinta sobre qué política estás evaluando.

## Ejemplo con código: Cliff Walking con SARSA y Q-learning

El siguiente ejemplo implementa el entorno del acantilado desde cero (sin librerías de entornos externas) y entrena un agente con cada algoritmo. Cópialo en una celda de Google Colab; solo necesitas `numpy` y `matplotlib`, que ya vienen preinstalados.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- EL ENTORNO: CLIFF WALKING ----------------------------------------
# Cuadrícula de 4 filas x 12 columnas. El agente empieza abajo a la
# izquierda (fila 3, columna 0) y el objetivo está abajo a la derecha
# (fila 3, columna 11). Toda la fila inferior entre ambos, excepto el
# inicio y el objetivo, es el "acantilado".
#
# Acciones: 0=arriba, 1=abajo, 2=izquierda, 3=derecha

FILAS, COLUMNAS = 4, 12
INICIO = (3, 0)
META = (3, 11)
ACANTILADO = [(3, c) for c in range(1, 11)]
NUM_ACCIONES = 4


def paso(estado, accion):
    """Aplica una acción y devuelve (nuevo_estado, recompensa, terminado)."""
    fila, col = estado
    if accion == 0:
        fila = max(fila - 1, 0)
    elif accion == 1:
        fila = min(fila + 1, FILAS - 1)
    elif accion == 2:
        col = max(col - 1, 0)
    elif accion == 3:
        col = min(col + 1, COLUMNAS - 1)

    nuevo_estado = (fila, col)

    if nuevo_estado in ACANTILADO:
        # Caer por el acantilado: penalización fuerte y vuelta al inicio
        return INICIO, -100, False
    if nuevo_estado == META:
        return nuevo_estado, -1, True
    return nuevo_estado, -1, False


def politica_epsilon_greedy(Q, estado, epsilon):
    """Con probabilidad epsilon explora; si no, elige la mejor acción conocida."""
    if np.random.rand() < epsilon:
        return np.random.randint(NUM_ACCIONES)
    valores = Q[estado]
    return np.random.choice(np.flatnonzero(valores == valores.max()))


def entrenar_sarsa(num_episodios=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = np.zeros((FILAS, COLUMNAS, NUM_ACCIONES))
    recompensas_por_episodio = []

    for _ in range(num_episodios):
        estado = INICIO
        accion = politica_epsilon_greedy(Q, estado, epsilon)
        recompensa_total = 0
        terminado = False

        while not terminado:
            siguiente_estado, recompensa, terminado = paso(estado, accion)
            siguiente_accion = politica_epsilon_greedy(Q, siguiente_estado, epsilon)

            # Actualización SARSA: usa el Q de la acción que SÍ se va a tomar
            objetivo = recompensa + gamma * Q[siguiente_estado][siguiente_accion]
            Q[estado][accion] += alpha * (objetivo - Q[estado][accion])

            estado, accion = siguiente_estado, siguiente_accion
            recompensa_total += recompensa

        recompensas_por_episodio.append(recompensa_total)

    return Q, recompensas_por_episodio


def entrenar_q_learning(num_episodios=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = np.zeros((FILAS, COLUMNAS, NUM_ACCIONES))
    recompensas_por_episodio = []

    for _ in range(num_episodios):
        estado = INICIO
        recompensa_total = 0
        terminado = False

        while not terminado:
            accion = politica_epsilon_greedy(Q, estado, epsilon)
            siguiente_estado, recompensa, terminado = paso(estado, accion)

            # Actualización Q-learning: usa el MÁXIMO Q del estado siguiente,
            # sin importar qué acción se elija después en la práctica
            mejor_valor_siguiente = Q[siguiente_estado].max()
            objetivo = recompensa + gamma * mejor_valor_siguiente
            Q[estado][accion] += alpha * (objetivo - Q[estado][accion])

            estado = siguiente_estado
            recompensa_total += recompensa

        recompensas_por_episodio.append(recompensa_total)

    return Q, recompensas_por_episodio


def extraer_ruta_greedy(Q, max_pasos=50):
    """Sigue la política puramente codiciosa (sin exploración) desde el inicio."""
    estado = INICIO
    ruta = [estado]
    for _ in range(max_pasos):
        if estado == META:
            break
        accion = np.argmax(Q[estado])
        estado, _, terminado = paso(estado, accion)
        ruta.append(estado)
        if terminado:
            break
    return ruta


# --- ENTRENAMIENTO ------------------------------------------------------
np.random.seed(0)
Q_sarsa, recompensas_sarsa = entrenar_sarsa()
Q_qlearning, recompensas_qlearning = entrenar_q_learning()

# --- GRÁFICO 1: recompensa por episodio durante el entrenamiento -------
# Se suaviza con una media móvil porque episodio a episodio hay mucho ruido
def media_movil(valores, ventana=20):
    return np.convolve(valores, np.ones(ventana) / ventana, mode="valid")

plt.figure(figsize=(9, 5))
plt.plot(media_movil(recompensas_sarsa), label="SARSA (on-policy)")
plt.plot(media_movil(recompensas_qlearning), label="Q-learning (off-policy)")
plt.xlabel("Episodio")
plt.ylabel("Recompensa por episodio (media móvil)")
plt.title("Cliff Walking: recompensa durante el entrenamiento")
plt.legend()
plt.tight_layout()
plt.show()

# --- GRÁFICO 2: rutas finales aprendidas (política greedy) -------------
def dibujar_ruta(ax, ruta, titulo):
    cuadricula = np.zeros((FILAS, COLUMNAS))
    for (f, c) in ACANTILADO:
        cuadricula[f, c] = -1
    ax.imshow(cuadricula, cmap="Pastel1")
    filas_ruta = [p[0] for p in ruta]
    columnas_ruta = [p[1] for p in ruta]
    ax.plot(columnas_ruta, filas_ruta, marker="o", color="tab:blue")
    ax.scatter(*INICIO[::-1], color="green", s=100, label="Inicio", zorder=5)
    ax.scatter(*META[::-1], color="red", s=100, label="Meta", zorder=5)
    ax.set_title(titulo)
    ax.invert_yaxis()
    ax.legend(loc="upper center")

fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
dibujar_ruta(ejes[0], extraer_ruta_greedy(Q_sarsa), "Ruta final de SARSA (más prudente)")
dibujar_ruta(ejes[1], extraer_ruta_greedy(Q_qlearning), "Ruta final de Q-learning (más corta)")
plt.tight_layout()
plt.show()
```

Al ejecutarlo, observa el primer gráfico: la curva de SARSA suele estar por encima de la de Q-learning durante buena parte del entrenamiento, porque cae menos veces por el acantilado mientras explora. En el segundo gráfico, fíjate en las dos rutas finales: la de Q-learning suele pegarse al borde del acantilado (la más corta y óptima), mientras que la de SARSA se aleja unas filas hacia arriba, sacrificando longitud a cambio de seguridad frente a la exploración.

## Sesgo de sobreestimación y Double Q-learning (mención breve)

Q-learning tiene un efecto secundario sutil: como su actualización usa el **máximo** entre varias estimaciones ruidosas de Q, ese máximo tiende a estar sesgado hacia arriba, igual que si eliges siempre "el mejor candidato" entre varias mediciones imprecisas, sueles quedarte con una que exagera por azar, aunque el valor real de todas las opciones sea el mismo. Esto se conoce como **sesgo de sobreestimación**, y puede llevar al agente a sobrevalorar sistemáticamente ciertas acciones.

**Double Q-learning** corrige este problema manteniendo dos tablas de valores independientes: una decide qué acción parece mejor, y la otra evalúa cuánto vale realmente esa acción (alternando los papeles en cada paso). Al separar "quién elige" de "quién evalúa", se elimina buena parte del sesgo, porque ya no es el mismo estimador el que se elige a sí mismo como ganador y luego se felicita por ello.

También existe una variante llamada **SARSA esperado**, que en vez de usar el valor Q de la acción que realmente se toma a continuación (como SARSA) o el máximo (como Q-learning), usa el promedio ponderado de Q sobre todas las acciones posibles, según la probabilidad que cada una tiene bajo la política. Es una forma de suavizar la actualización y reducir la varianza de sampling, y cuando la política objetivo es completamente codiciosa, esta versión coincide exactamente con Q-learning.

## Resumen de puntos clave

- El control busca mejorar la política, no solo evaluarla: alterna evaluación y mejora siguiendo el principio de Iteración de Política Generalizada (GPI)
- La política ε-greedy equilibra exploración y explotación; la condición GLIE garantiza convergencia si visitas todo lo suficiente y vuelves la política codiciosa con el tiempo
- SARSA es on-policy: aprende el valor de la política que realmente sigue, incluida su exploración, lo que la hace más prudente
- Q-learning es off-policy: aprende el valor de la política óptima aunque explore de otra forma, lo que la hace más arriesgada durante el entrenamiento pero óptima al final
- El ejemplo del acantilado muestra la diferencia con claridad: SARSA acumula más recompensa mientras entrena, Q-learning encuentra la ruta más corta
- El operador máximo de Q-learning introduce sesgo de sobreestimación; Double Q-learning lo corrige separando selección y evaluación en dos tablas

## Ponlo en práctica

Toma el código anterior y **experimenta con dos cambios**, uno cada vez, para ver cómo afectan a ambas políticas:

1. **Cambia epsilon**: sube `epsilon` a 0.3 en las dos funciones de entrenamiento y vuelve a ejecutar. Observa si la brecha entre la recompensa de SARSA y la de Q-learning durante el entrenamiento se agranda o se reduce
2. **Cambia la penalización del acantilado**: en la función `paso`, sustituye el −100 por, por ejemplo, −10 o −500, y compara cómo cambian las dos rutas finales. **Anota tus observaciones** y comprueba si con una penalización muy baja las dos rutas terminan pareciéndose

Hasta ahora guardaste todos los valores Q en una tabla, una casilla por cada combinación de estado y acción. Esa tabla funciona perfectamente en una cuadrícula de doce columnas, pero deja de ser viable en cuanto el número de estados se dispara, por ejemplo, cuando el estado son los píxeles de una pantalla de videojuego. En el capítulo ocho vas a sustituir esa tabla por una red neuronal que aprende a aproximar Q(s,a): así nacen las redes Q profundas (DQN, por sus siglas en inglés *Deep Q-Networks*).
