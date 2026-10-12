# Capítulo 10. Planificación y modelos: cuando el agente aprende cómo funciona el mundo

Imagina que quieres ir de tu oficina a una reunión en otra ciudad. Puedes lanzarte a conducir sin más y corregir el rumbo cada vez que te equivocas de calle, o puedes abrir un mapa mental de la zona, imaginar dos o tres rutas posibles y elegir la que parece más corta antes de arrancar el coche. La segunda opción es más lenta al principio (hay que construir ese mapa), pero, una vez que lo tienes, te permite planificar en vez de improvisar cada giro.

Hasta ahora, en este manual, tu agente ha sido puramente reactivo. En el capítulo anterior viste cómo los métodos de gradiente de política y actor-crítico ajustan una política directamente a partir de la experiencia, sin construir jamás una representación explícita de cómo funciona el entorno. Eso es Aprendizaje por Refuerzo (RL, por sus siglas en inglés) sin modelo (model-free): funciona muy bien, pero cada mejora exige nueva experiencia real, que a menudo es cara, lenta o arriesgada de conseguir.

En este capítulo vas a dar un paso distinto. Vas a ver qué ocurre cuando el agente, además de actuar, aprende un modelo interno del entorno: una especie de simulador propio que le permite "imaginar" qué pasaría si tomara tal o cual acción, sin tener que probarla de verdad. Esa capacidad de imaginar resultados antes de ejecutarlos es la base de sistemas como AlphaZero, que conocerás en el capítulo 12. Aquí vas a construir los cimientos: qué es un modelo, cómo se combina con el aprendizaje directo en la familia Dyna, y cómo la búsqueda de Monte Carlo en árboles explora futuros posibles sin necesidad de recorrerlos todos.

## RL sin modelo frente a RL basado en modelo

Todo lo que has visto hasta el capítulo nueve —bandidos, Q-learning, gradientes de política, actor-crítico— pertenece a la familia de RL sin modelo: el agente aprende directamente una política o una función de valor a partir de la experiencia, sin representar de forma explícita cómo cambia el entorno de un paso a otro.

El RL basado en modelo añade una pieza: un modelo de las dinámicas del entorno, formado por una función de transición (qué estado viene después) y una función de recompensa (qué recompensa se obtiene). Ese modelo puede estar ya disponible de antemano o, lo más habitual, tener que aprenderse a partir de la experiencia. Conviene que sepas que el propio término "basado en modelo" se usa de forma un poco ambigua en la literatura: algunas fuentes lo reservan solo para el caso en que el modelo se aprende, y otras lo aplican también cuando el modelo se proporciona directamente. Incluso la frontera entre "agente" y "entorno" puede desdibujarse: un agente con acceso a un simulador de Atari podría tratar ese simulador como su propio modelo interno.

Piensa en un modelo como una aproximación de un Proceso de Decisión de Markov (MDP): no necesita ser una réplica exacta del entorno, solo lo bastante fiel como para orientar decisiones. El texto oficial señala tres tipos de modelo que conviene distinguir, porque cada uno falla de una manera distinta:

- **Modelo de esperanza**: predice el estado siguiente y la recompensa esperados, en promedio. El problema es que ese promedio puede no corresponder a ningún estado real. Imagina un robot en una cuadrícula donde moverse "hacia arriba" en realidad lo desplaza en diagonal, a la izquierda o a la derecha, nunca en línea recta. Un modelo de esperanza podría predecir que el robot termina justo en medio, en una casilla que ni siquiera existe (un hueco inventado en la pared)
- **Modelo estocástico o generativo**: en vez de predecir un promedio, genera muestras de estados y recompensas válidos, añadiendo cada vez un término de ruido nuevo. Nunca te devuelve un estado imposible, aunque a cambio introduce más variabilidad; si el modelo es barato de consultar, puedes muestrear muchas veces para promediar esa variabilidad
- **Modelo de dinámica completa**: aprende la distribución de probabilidad entera sobre los estados siguientes, no solo un punto esperado ni una muestra aislada. En espacios continuos, esto exige trabajar con integrales en vez de sumas, algo que suele aproximarse con métodos como Markov Chain Monte Carlo (MCMC)

Estos modelos pueden representarse de formas muy distintas: como una tabla (si el problema es pequeño), como una función lineal sobre características del estado, como una red neuronal profunda (para generalizar en espacios grandes o continuos) o incluso de forma no paramétrica, usando un simple buffer de repetición de experiencias: "consultar el modelo" pasa a ser, en ese caso, sacar al azar una transición real que ya observaste.

## Dyna-Q: aprender actuando y aprender imaginando

La familia de algoritmos Dyna, propuesta por Richard Sutton, combina dos fuentes de aprendizaje que hasta ahora tratabas por separado: la experiencia real y la experiencia simulada por un modelo aprendido.

Dyna-Q, su versión tabular, funciona en dos fases dentro de cada paso:

1. **Aprendizaje directo**: el agente actúa en el entorno real, observa el estado siguiente y la recompensa, aplica una actualización de Q-learning normal y, de paso, guarda esa transición en su modelo interno
2. **Planificación simulada**: después de cada paso real, el agente repite *n* veces lo siguiente: elige al azar una combinación de estado y acción que ya haya observado antes, le pregunta al modelo qué ocurriría, y aplica otra actualización de Q-learning sobre esa transición imaginada

Ese número *n* se llama pasos de planificación. Con *n* igual a cero, Dyna-Q se reduce exactamente a Q-learning puro: no hay nada que imaginar. Cuantos más pasos de planificación añades, más "repasa" el agente su propia experiencia pasada antes de volver a actuar, y menos episodios reales necesita para converger. En un mundo-cuadrícula de referencia, aumentar los pasos de planificación de cero a cinco, y de cinco a 50, reduce de forma muy marcada el número de pasos que el agente tarda en llegar a la meta en cada episodio, a cambio de más cómputo por paso real. Ese intercambio suele merecer la pena precisamente cuando recolectar experiencia real es lento o costoso.

Hay un matiz importante cuando el entorno cambia. Si un camino que antes estaba libre se bloquea, Dyna-Q actualiza su modelo en cuanto choca contra el nuevo obstáculo y replanifica enseguida. Pero si aparece un atajo nuevo que el agente nunca ha probado, Dyna-Q puede tardar en descubrirlo, porque nada le indica que ese atajo existe hasta que la exploración lo lleva allí por casualidad. Dyna-Q+ resuelve parcialmente esto añadiendo un bono de exploración que crece cuanto más tiempo lleva sin visitarse una combinación de estado y acción: el agente se vuelve optimista respecto a las zonas que hace tiempo que no revisa, asumiendo que podrían haber cambiado, y así encuentra atajos nuevos más rápido, aunque a veces pague el precio de tomar alguna acción subóptima en un entorno que en realidad no ha cambiado.

Conviene recordar también que un modelo aprendido puede ser inexacto, por falta de experiencia o por limitaciones del propio aproximador, y que planificar sobre un modelo así de defectuoso puede llevar al agente a decisiones tan absurdas como la del robot que "alucina" un hueco en la pared. Combinar planificación basada en modelo con actualizaciones libres de modelo —cayendo en estas últimas cuando el modelo resulta poco fiable— es una forma habitual de obtener lo mejor de ambos mundos: la orientación estructurada de un modelo y la robustez de aprender de la realidad.

## MCTS: explorar un árbol de futuros posibles

La búsqueda de Monte Carlo en árboles (Monte Carlo Tree Search, MCTS) lleva la idea de planificación un paso más allá: en vez de simular transiciones sueltas como hace Dyna-Q, construye y mantiene un árbol de posibles futuros a partir del estado actual, creciendo solo por la parte que realmente explora.

Cada búsqueda con MCTS se organiza en cuatro fases que se repiten tantas veces como el presupuesto de cómputo lo permita:

1. **Selección**: partiendo de la raíz del árbol, recorre los nodos ya existentes siguiendo una política de árbol, hasta llegar a un nodo que todavía no se ha expandido del todo
2. **Expansión**: añade al árbol un nuevo nodo hijo, correspondiente a una acción que aún no se había probado desde ahí
3. **Simulación (rollout)**: desde ese nodo nuevo, juega una partida o trayectoria completa siguiendo una política rápida, a menudo aleatoria, hasta llegar a un estado terminal
4. **Retropropagación**: propaga el resultado de esa simulación hacia arriba, por todo el camino recorrido, actualizando el valor y el contador de visitas de cada nodo

El valor de un nodo es, en el fondo, el promedio de los resultados de todas las simulaciones que han pasado por él. Y aquí aparece una pregunta que ya conoces del capítulo tres: en la fase de selección, ¿qué rama del árbol conviene explorar? Upper Confidence Bound for Trees (UCT) responde con la misma lógica que Upper Confidence Bound (UCB) aplicaba a los bandidos multi-brazo: suma al valor estimado de cada nodo un bono de exploración que crece cuando ese nodo se ha visitado poco, y se reduce a medida que se visita más. Un nodo con pocas visitas recibe el beneficio de la duda; un nodo muy visitado ya tiene una estimación fiable y no necesita tanto bono extra. Igual que en UCB, una constante controla cuánto peso das a explorar frente a explotar.

MCTS con UCT es el motor de búsqueda detrás de AlphaGo, aplicado sobre juegos como el Go o el ajedrez, donde el número de configuraciones posibles del tablero es astronómico y no existe forma de recorrer el árbol completo. También se ha usado para planificar secuencias de acciones en control robótico. Dos técnicas relacionadas, el Control Predictivo por Modelo (MPC) y MuZero, aparecen mencionadas en el temario oficial como enfoques emparentados, pero quedan fuera del alcance de este capítulo.

## Ejemplo en Python: Dyna-Q contra Q-learning puro

El siguiente código construye un mundo-cuadrícula pequeño, hecho a mano, con dos muros que obligan al agente a rodearlos para llegar a la meta. Compara Q-learning puro (cero pasos de planificación) con dos versiones de Dyna-Q (cinco y 50 pasos de planificación) y dibuja cuántos pasos necesita cada una para completar cada episodio. Puedes copiarlo y ejecutarlo directamente en Google Colab: solo necesita `numpy` para el cálculo y `matplotlib` para la gráfica, ambos preinstalados.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# --- Definimos un mundo-cuadrícula pequeño, hecho a mano ---
ALTO, ANCHO = 6, 9                 # 6 filas, 9 columnas
INICIO = (5, 0)                    # esquina inferior izquierda
META = (0, 8)                      # esquina superior derecha

# Dos segmentos de muro que obligan a rodear en vez de ir en línea recta
PAREDES = {(1, 3), (2, 3), (3, 3), (0, 6), (1, 6), (2, 6)}

# Cuatro acciones posibles: arriba, abajo, izquierda, derecha
ACCIONES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACCIONES = len(ACCIONES)


def paso_entorno(estado, accion_idx):
    """Aplica una acción de forma determinista: si el movimiento choca contra
    un muro o el borde del mundo, el agente se queda donde estaba."""
    if estado == META:
        return estado, 0.0, True
    d_fila, d_col = ACCIONES[accion_idx]
    fila, col = estado
    nueva_fila = min(max(fila + d_fila, 0), ALTO - 1)
    nueva_col = min(max(col + d_col, 0), ANCHO - 1)
    nuevo_estado = (nueva_fila, nueva_col)
    if nuevo_estado in PAREDES:
        nuevo_estado = estado
    terminado = nuevo_estado == META
    recompensa = 0.0 if terminado else -1.0   # -1 por paso incentiva el camino corto
    return nuevo_estado, recompensa, terminado


def indice(estado):
    """Convierte una casilla (fila, columna) en un índice único de fila en la tabla Q."""
    return estado[0] * ANCHO + estado[1]


def elegir_accion(Q, estado, epsilon):
    """Política epsilon-greedy, la misma que usaste en el capítulo tres."""
    if np.random.random() < epsilon:
        return np.random.randint(N_ACCIONES)
    fila_q = Q[indice(estado)]
    mejores = np.flatnonzero(fila_q == np.max(fila_q))
    return np.random.choice(mejores)


def ejecutar_dyna_q(n_planning, n_episodios, alpha=0.1, gamma=0.95, epsilon=0.1, max_pasos=1000):
    """Entrena un agente Dyna-Q tabular. Con n_planning=0 es Q-learning puro:
    solo queda la actualización con experiencia real, sin repetir nada."""
    Q = np.zeros((ALTO * ANCHO, N_ACCIONES))
    modelo = {}                      # modelo aprendido: (estado, accion) -> (estado_siguiente, recompensa)
    pasos_por_episodio = np.zeros(n_episodios)

    for ep in range(n_episodios):
        estado = INICIO
        pasos = 0
        terminado = False

        while not terminado and pasos < max_pasos:
            # --- Fase 1: aprendizaje directo, con experiencia real ---
            accion = elegir_accion(Q, estado, epsilon)
            sig_estado, recompensa, terminado = paso_entorno(estado, accion)

            i_estado, i_sig = indice(estado), indice(sig_estado)
            objetivo = recompensa + gamma * np.max(Q[i_sig])
            Q[i_estado, accion] += alpha * (objetivo - Q[i_estado, accion])

            modelo[(estado, accion)] = (sig_estado, recompensa)

            # --- Fase 2: planificación simulada, con experiencia imaginada ---
            transiciones_vistas = list(modelo.items())
            for _ in range(n_planning):
                (s_sim, a_sim), (s_sig_sim, r_sim) = transiciones_vistas[
                    np.random.randint(len(transiciones_vistas))
                ]
                i_sim, i_sig_sim = indice(s_sim), indice(s_sig_sim)
                objetivo_sim = r_sim + gamma * np.max(Q[i_sig_sim])
                Q[i_sim, a_sim] += alpha * (objetivo_sim - Q[i_sim, a_sim])

            estado = sig_estado
            pasos += 1

        pasos_por_episodio[ep] = pasos

    return pasos_por_episodio


# --- Comparamos tres configuraciones: 0, 5 y 50 pasos de planificación ---
n_episodios = 50
n_repeticiones = 10       # promediamos varias corridas para suavizar el azar
configuraciones = [0, 5, 50]
resultados = {}

for n_planning in configuraciones:
    acumulado = np.zeros(n_episodios)
    for _ in range(n_repeticiones):
        acumulado += ejecutar_dyna_q(n_planning, n_episodios)
    resultados[n_planning] = acumulado / n_repeticiones

# --- Graficamos la comparación ---
plt.figure(figsize=(10, 5))
for n_planning in configuraciones:
    etiqueta = "Q-learning puro (0 pasos)" if n_planning == 0 else f"Dyna-Q ({n_planning} pasos de planificación)"
    plt.plot(resultados[n_planning], label=etiqueta)
plt.xlabel("Episodio")
plt.ylabel("Pasos hasta llegar a la meta")
plt.title("Dyna-Q vs. Q-learning puro en un mundo-cuadrícula con muros")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Al ejecutarlo vas a ver tres curvas descendentes, pero a ritmos muy distintos. La de cero pasos de planificación (Q-learning puro) mejora despacio, porque solo aprende de lo que va viviendo en tiempo real. Las curvas de cinco y, sobre todo, 50 pasos de planificación caen mucho más rápido: cada paso real se aprovecha muchas veces más, gracias a las actualizaciones simuladas sobre el modelo que el propio agente ha ido construyendo.

## Puntos clave

- El RL basado en modelo añade una función de transición y una función de recompensa que el agente usa para planificar, en vez de reaccionar solo a la experiencia directa
- Existen tres tipos de modelo con distintos modos de fallo: el modelo de esperanza puede predecir estados que no existen, el modelo estocástico genera siempre estados válidos a cambio de más varianza, y el modelo de dinámica completa captura toda la distribución, a mayor coste computacional
- Dyna-Q combina, en cada paso, una actualización con experiencia real y varias actualizaciones con experiencia simulada por el modelo aprendido; con cero pasos de planificación, se reduce a Q-learning puro
- Dyna-Q+ añade un bono de exploración que crece con el tiempo desde la última visita, lo que ayuda a descubrir atajos nuevos que Dyna-Q tardaría en encontrar
- MCTS explora un árbol de futuros posibles en cuatro fases (selección, expansión, simulación y retropropagación), y UCT equilibra exploración y explotación con la misma idea que UCB en los bandidos del capítulo tres
- Un modelo aprendido nunca es perfecto: combinarlo con actualizaciones libres de modelo aporta robustez cuando el modelo falla

## Ponlo en práctica

Toma el código de este capítulo y experimenta tú misma o tú mismo:

- **Cambia el número de pasos de planificación** (prueba con 1, 10 y 100) y observa cómo se acelera, o no, la velocidad de aprendizaje frente al coste extra de cómputo por paso
- **Modifica el diseño de los muros** en `PAREDES` para crear un camino más largo o más estrecho, y comprueba si la ventaja de Dyna-Q sobre Q-learning puro se mantiene
- **Añade un bono de exploración al estilo Dyna-Q+**: guarda en qué paso se visitó cada combinación de estado y acción por última vez, y suma un pequeño extra a la recompensa simulada cuando lleve mucho tiempo sin visitarse

**¡Ejecuta el notebook, cambia los parámetros y comparte con tu equipo a partir de cuántos pasos de planificación deja de merecer la pena el cómputo extra!**

---

En el capítulo 11 vas a volver al RL profundo sin modelo, pero con las técnicas más avanzadas del campo: Rainbow DQN y el RL distribucional.
