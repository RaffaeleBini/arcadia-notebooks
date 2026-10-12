# Capítulo 13. Casos de uso reales: de los juegos clásicos a AlphaStar

Llevas doce capítulos construyendo un edificio: empezaste con el dilema de explorar y explotar en los bandidos, seguiste con los procesos de decisión de Markov (PDM) y la programación dinámica, aprendiste a predecir y controlar sin modelo con Monte Carlo, Diferencia Temporal (TD), SARSA y Q-learning, diste el salto a las redes neuronales con las redes Q profundas (DQN, por sus siglas en inglés *Deep Q-Network*) y los gradientes de política, y cerraste con planificación, Búsqueda en Árbol de Monte Carlo (MCTS, por sus siglas en inglés *Monte Carlo Tree Search*), Rainbow y los métodos actor-crítico distribuidos como A3C e IMPALA. Todo eso es, en el fondo, teoría de pizarra: fórmulas, pseudocódigo, entornos de juguete.

Este capítulo final te enseña que esa teoría no se quedó en la pizarra. Vas a recorrer una línea de sistemas reales, de Deep Blue a AlphaStar, y en cada uno vas a reconocer piezas que ya conoces: una función de valor aquí, una política aprendida por gradiente allá, una búsqueda tipo MCTS más adelante. No memorices nombres ni fechas: fíjate en cómo las mismas ideas simples, combinadas y escaladas, bastaron para vencer a los mejores jugadores humanos del planeta en juegos que llevaban décadas —o siglos— considerándose baluartes del intelecto humano. Es el cierre perfecto del manual, porque demuestra que cada concepto que dominaste tiene una versión de sí mismo funcionando dentro de un sistema que hizo historia.

## De Deep Blue al Monte Carlo Tree Search

Antes de que ninguna red neuronal jugara al Go, la inteligencia artificial (IA) ya llevaba treinta años derrotando campeones humanos en juegos de mesa, con técnicas mucho más cercanas a la búsqueda que al aprendizaje. Los juegos funcionan como "moscas de la fruta" para la investigación en IA: reglas claras, objetivos medibles, un marcador que no miente, igual que la genética usó la Drosophila para descubrir principios que luego se generalizaron a organismos mucho más complejos.

El primer hito fue **Chinook**, un programa de damas que alcanzó una partida prácticamente perfecta combinando poda alfa-beta con análisis retrógrado (resolver los finales de partida hacia atrás, desde posiciones de victoria ya conocidas). Su rival fue Marion Tinsley, probablemente el mejor jugador de damas de la historia: perdió solo cinco partidas en toda su carrera, y dos de ellas fueron contra Chinook. Después, Schaeffer y su equipo resolvieron el juego por completo.

En 1997 llegó **Deep Blue**, de IBM, que venció al campeón mundial de ajedrez Garri Kaspárov en un enfrentamiento que se convirtió en uno de los eventos más seguidos de internet en su momento. Deep Blue no aprendía nada: evaluaba unos 8.000 rasgos codificados a mano (seguridad del rey, estructura de peones, material) y examinaba hasta 200 millones de posiciones por segundo, mirando entre 16 y 40 jugadas por delante. Es, en esencia, el minimax con poda alfa-beta que estudiaste como fundamento teórico de la búsqueda en juegos de suma cero: fuerza bruta guiada por una función de evaluación diseñada por humanos, sin rastro de aprendizaje por refuerzo (RL, por sus siglas en inglés *Reinforcement Learning*).

El contrapunto llegó con **TD-Gammon**, un programa de backgammon que aprendió una función de valor —la probabilidad de ganar desde una posición dada— entrenando una red neuronal pequeña mediante TD, únicamente jugando contra sí misma, sin ningún dato humano. Es la misma idea que viste en el capítulo 6 (predicción sin modelo con TD) y en el capítulo 8 (aproximar funciones de valor con una red neuronal), aplicada casi una década antes de que existiera el término "aprendizaje profundo". Un dato curioso: al backgammon se juega con dados, y esa aleatoriedad ya proporcionaba exploración natural, así que TD-Gammon no necesitó una estrategia explícita de exploración como la ε-greedy del capítulo 7.

Poco después, **Maven** venció al campeón mundial de Scrabble, Adam Logan, por 9 partidas a 5, combinando una búsqueda Monte Carlo truncada (simular solo unos diez movimientos por delante, no partidas completas) con una función de evaluación aprendida. Esa búsqueda Monte Carlo truncada es el eslabón que conecta todo lo anterior con el **MCTS**: en vez de resolver el juego entero, simulas muchas continuaciones posibles desde la posición actual y aprendes una política local, específica de esa situación. El manual ya te presentó el ciclo de selección, expansión, simulación y retropropagación en el capítulo dedicado a planificación; aquí lo vuelves a encontrar como el puente exacto hacia AlphaGo.

## AlphaGo: la jugada que sorprendió al mundo

El Go llevaba tres mil años jugándose y tenía alrededor de 40 millones de jugadores activos cuando DeepMind decidió abordarlo. El reto no era solo su tablero de 19 por 19: cada posición admite unas 200 jugadas legales de media (frente a las 20-35 del ajedrez), y el número de configuraciones posibles supera al de átomos en el universo observable. La fuerza bruta de Deep Blue, aquí, simplemente no era una opción; el Go se juega más por intuición que por cálculo exhaustivo, y AlphaGo necesitaba aproximar esa intuición.

La arquitectura combinaba dos redes neuronales convolucionales: una **red de política**, que propone una distribución de probabilidad sobre las jugadas legales, y una **red de valor**, que estima un único número —la probabilidad de ganar— desde una posición dada. Reconoce ambas piezas: la red de valor es, en esencia, la misma función de valor del capítulo 4, aprendida con una red neuronal como en el capítulo 8; la red de política es un gradiente de política, como los del capítulo 9, entrenado primero por imitación y después por autojuego. Las dos redes se combinaban con MCTS mediante una variante llamada PUCT (*Predictor + UCT*): la política reduce la anchura de la búsqueda (qué jugadas merece la pena explorar) y la red de valor reduce su profundidad (evita jugar cada rama hasta el final). Con eso, AlphaGo llegaba a mirar entre 50 y 60 jugadas por delante.

El entrenamiento tuvo dos fases: primero, aprendizaje supervisado a partir de partidas de jugadores humanos expertos, para que la red de política imitara jugadas fuertes; después, refinamiento por autojuego, jugando millones de partidas contra versiones anteriores de sí misma, mientras la red de valor aprendía a predecir el resultado final de cada posición alcanzada. En unas 72 horas de entrenamiento, AlphaGo pasó de jugar como un principiante a un nivel sobrehumano.

Los resultados hablan solos: AlphaGo venció al campeón europeo Fan Hui por 5 partidas a 0, y en marzo de 2016, en Seúl, venció al legendario Lee Sedol —9-dan, 18 títulos mundiales— por 4 partidas a 1. La jugada más recordada fue la **jugada 37** de la segunda partida: un golpe de hombro en la quinta línea tan inusual que la propia AlphaGo calculó solo una probabilidad entre 10.000 de que un humano la jugara, y terminó siendo una genialidad estratégica. Pero el sistema también tuvo un límite: en la partida cuatro, Lee Sedol encontró la **jugada 78**, a la que AlphaGo había asignado apenas un 0,007% de probabilidad, y ganó esa partida. Fue un recordatorio útil, y honesto, de que "sobrehumano" no significa infalible. Meses después, AlphaGo Zero —entrenada desde cero, sin ningún dato humano— venció al noventa por ciento de las partidas a AlphaGo Master, la versión que a su vez había acumulado 60 victorias y ninguna derrota contra los mejores profesionales del mundo.

## AlphaZero: aprender desde cero, sin datos humanos

AlphaZero simplifica y generaliza todo lo anterior de un plumazo: una única red neuronal, con salida conjunta de política y valor, y un único algoritmo aplicado sin cambios a ajedrez, shōgi (el ajedrez japonés) y Go, cambiando solo la codificación del tablero. Nada de datos humanos, nada de aperturas ni finales precargados: empieza con pesos aleatorios y las reglas del juego, y mejora exclusivamente jugando contra sí misma.

El mecanismo que cierra el bucle es puro GPI —la Iteración de Política Generalizada del capítulo 7, la misma alternancia de evaluar y mejorar que ya viste con SARSA y Q-learning—, solo que aquí MCTS hace el papel de "operador de mejora de política": la búsqueda produce una distribución de jugadas mejor que la que sugiere la red por sí sola, y la red se entrena después para imitar esa distribución mejorada. Formalmente:

π′(a|s) ∝ Q(s,a)^(1/τ)

donde π′ es la política mejorada, Q(s,a) es el valor estimado por MCTS para cada jugada y τ (tau) es un parámetro de temperatura: valores altos exploran más, valores bajos concentran la elección en la jugada más prometedora. La red se entrena con una pérdida de entropía cruzada para la parte de política y error cuadrático medio para la parte de valor, todo en una sola red.

Los resultados, con hardware especializado (unidades de procesamiento tensorial, TPU), llegaron con una velocidad que sorprendió incluso a sus creadores: tras solo unas cuatro horas de autojuego, AlphaZero ya superaba a Stockfish, el motor de ajedrez más fuerte del momento; tras unas dos horas, superaba a Elmo, el mejor programa de shōgi; y tras unas ocho horas, superaba a la propia versión de AlphaGo que había vencido a Lee Sedol. Frente a Stockfish —que evalúa millones de posiciones por segundo con reglas de material fijadas a mano (peón=1, caballo=alfil=3, torre=5, dama=9)— AlphaZero desarrolló un estilo propio: sacrificar material por actividad de las piezas y control a largo plazo. En una partida documentada con apertura Ruy López, AlphaZero sacrificó la dama en la jugada 25 para desmontar la estructura de Stockfish y acabó ganando por jaque mate; en otra, con apertura siciliana, sacrificó una torre por un peón en la jugada 24 para asegurar una estructura de peones superior. Ningún manual de ajedrez le enseñó a hacer eso: lo descubrió jugando contra sí misma, del mismo modo en que Q-learning del capítulo 7 aprende una política óptima sin que nadie le diga de antemano cuál es.

## AlphaStar: el reto de StarCraft II

Si el Go y el ajedrez son juegos de información perfecta y turnos discretos, StarCraft II es justo lo contrario: información parcial (la famosa "niebla de guerra" oculta buena parte del mapa), millones de acciones posibles en cada instante y decisiones que hay que tomar en tiempo real, no por turnos, combinando planificación estratégica a largo plazo con microgestión de decenas de unidades a la vez. DeepMind construyó **AlphaStar** para jugar la raza Protoss en este entorno, en colaboración con Blizzard Entertainment.

El entrenamiento combinó dos fases que ya reconoces: primero, unos tres días de aprendizaje por imitación sobre partidas humanas reales, que llevaron al agente hasta un nivel aproximado de jugador "platino"; después, cerca de una semana de autojuego organizado en la llamada **Liga AlphaStar** —agentes principales, "explotadores" especializados en encontrar sus debilidades y agentes de prueba, compitiendo entre sí a un ritmo equivalente a cientos de años de partidas humanas—. Ese esquema de liga con múltiples agentes jugando en paralelo es la misma filosofía distribuida de A3C e IMPALA, del capítulo 12, llevada a una escala mucho mayor; y el algoritmo de RL que ajustaba la política en cada partida fue la Optimización de Política Próxima (PPO, por sus siglas en inglés *Proximal Policy Optimization*), un método de gradiente de política que limita cuánto puede cambiar la política en cada actualización, para no desestabilizar el aprendizaje.

Arquitectónicamente, AlphaStar combinó redes convolucionales para procesar el mapa, memoria de largo plazo (LSTM, *Long Short-Term Memory*) para recordar información pasada bajo observación parcial, y mecanismos de atención para priorizar qué unidades importan en cada momento. Para que la comparación con humanos fuera justa, el equipo limitó su tiempo de reacción a unos 350 milisegundos y sus acciones por minuto (APM) a niveles comparables a los de un profesional, de modo que una victoria reflejara mejores decisiones y no velocidad sobrehumana de clic. El resultado: una victoria por 10 a 0 contra el profesional Grzegorz "MaNa" Komincz, tras haberse medido también contra Liquid TLO. Fíjate en el patrón: exploración/explotación a escala de liga (capítulo 3), autojuego que genera sus propios datos (como en AlphaZero), y gradientes de política entrenados con PPO (capítulo 9) para digerir todo eso, esta vez bajo información incompleta.

## Ejemplo con código: mini-AlphaZero para Tres en Raya

El siguiente ejemplo implementa, a escala de juguete, el mismo patrón que viste en las tres secciones anteriores: autojuego, búsqueda tipo MCTS y mejora iterativa de la política. Juega Tres en Raya en vez de Go, y usa una función heurística sencilla más una tabla de valores en lugar de una red neuronal entrenada de verdad, para que se pueda ejecutar en segundos en cualquier Google Colab, sin más dependencia que `numpy`. El código está comentado para que veas, línea a línea, dónde encaja cada concepto de los capítulos anteriores.

```python
import math
import random

import numpy as np

# ============================================================
# AVISO DIDACTICO: esto es una version de juguete, a escala minima,
# del patron de AlphaZero (autojuego + busqueda tipo MCTS + mejora
# iterativa de la politica). NO es AlphaZero real: no hay red
# neuronal convolucional entrenada por descenso de gradiente, el
# "valor" se guarda en un simple diccionario en vez de una red, y el
# juego es Tres en Raya, no ajedrez ni Go. El objetivo es que veas el
# mismo engranaje -autojuego, busqueda, mejora de politica- funcionar
# de principio a fin en un juego trivial, no reproducir el sistema real.
# ============================================================

LINEAS_GANADORAS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def tablero_nuevo():
    return tuple([0] * 9)  # 0 = vacio, 1 = X, -1 = O


def movimientos_validos(tablero):
    return [i for i, v in enumerate(tablero) if v == 0]


def aplicar_movimiento(tablero, pos, jugador):
    nuevo = list(tablero)
    nuevo[pos] = jugador
    return tuple(nuevo)


def resultado(tablero):
    """Devuelve +1 si gana X, -1 si gana O, 0 si hay empate, None si sigue."""
    for a, b, c in LINEAS_GANADORAS:
        suma = tablero[a] + tablero[b] + tablero[c]
        if suma == 3:
            return 1
        if suma == -3:
            return -1
    if 0 not in tablero:
        return 0
    return None


def heuristica(tablero):
    """
    Sustituta minima de la red de valor de AlphaZero: para cada linea de
    tres casillas, si solo hay piezas de X (o ninguna) suma piezas propias
    al cuadrado; si solo hay piezas de O, resta lo mismo. Es una funcion
    fija, no aprendida: solo sirve como estimacion de partida cuando la
    tabla de valores aprendida todavia no ha visto una posicion. Devuelve
    un valor aproximado en [-1, 1] desde la perspectiva de X.
    """
    puntuacion = 0
    for linea in LINEAS_GANADORAS:
        valores = [tablero[i] for i in linea]
        piezas_x = valores.count(1)
        piezas_o = valores.count(-1)
        if piezas_o == 0:
            puntuacion += piezas_x ** 2
        if piezas_x == 0:
            puntuacion -= piezas_o ** 2
    return math.tanh(puntuacion / 6.0)


# Tabla de valores aprendida por autojuego: estado -> valor estimado
# desde la perspectiva de X (+1 = X gana, -1 = O gana, 0 = tablas).
# Este diccionario cumple, a pequenisima escala, el papel que en
# AlphaZero cumple la red de valor entrenada (capitulo 8: aproximacion
# de funciones con redes neuronales; capitulo 4: funciones de valor).
tabla_valores = {}


def valor_estimado(tablero):
    if tablero in tabla_valores:
        return tabla_valores[tablero]
    return heuristica(tablero)


class Nodo:
    __slots__ = ("tablero", "jugador", "hijos", "N", "W", "expandido")

    def __init__(self, tablero, jugador):
        self.tablero = tablero
        self.jugador = jugador  # jugador (+1 o -1) que debe mover aqui
        self.hijos = {}
        self.N = 0
        self.W = 0.0
        self.expandido = False

    def valor_medio(self):
        return self.W / self.N if self.N > 0 else 0.0


def ucb_score(padre, hijo, c=1.4):
    """
    Formula tipo UCB/PUCT (busqueda y planificacion): combina "cuanto de
    bueno parece" (Q, convertido a la perspectiva del jugador que mueve
    en el padre) con "cuanto se ha explorado" (U, que crece si el hijo se
    ha visitado poco). En AlphaZero, U tambien incorpora la probabilidad
    previa de la red de politica; aqui usamos exploracion uniforme para
    mantener el ejemplo simple.
    """
    q = hijo.valor_medio() * padre.jugador
    u = c * math.sqrt(math.log(padre.N + 1) / (hijo.N + 1))
    return q + u


def simular(nodo):
    """Un ciclo de seleccion, expansion, evaluacion y retropropagacion."""
    r = resultado(nodo.tablero)
    if r is not None:
        nodo.N += 1
        nodo.W += r
        return r

    if not nodo.expandido:
        for m in movimientos_validos(nodo.tablero):
            tablero_hijo = aplicar_movimiento(nodo.tablero, m, nodo.jugador)
            nodo.hijos[m] = Nodo(tablero_hijo, -nodo.jugador)
        nodo.expandido = True
        # En vez de un rollout aleatorio hasta el final (MCTS "clasico"),
        # evaluamos la posicion con la tabla de valores o la heuristica,
        # igual que AlphaZero sustituye el rollout por su red de valor.
        v = valor_estimado(nodo.tablero)
        nodo.N += 1
        nodo.W += v
        return v

    mejor_m = max(nodo.hijos, key=lambda m: ucb_score(nodo, nodo.hijos[m]))
    v = simular(nodo.hijos[mejor_m])
    nodo.N += 1
    nodo.W += v
    return v


def busqueda_mcts(tablero, jugador, num_simulaciones=60):
    """Devuelve {movimiento: visitas} tras num_simulaciones simulaciones."""
    raiz = Nodo(tablero, jugador)
    for _ in range(num_simulaciones):
        simular(raiz)
    return {m: hijo.N for m, hijo in raiz.hijos.items()}


def elegir_movimiento(visitas, temperatura):
    """
    Convierte las visitas de la MCTS en una politica y elige un movimiento.
    Con temperatura > 0 hay exploracion (util en autojuego, para generar
    partidas variadas); con temperatura = 0 se juega siempre el movimiento
    mas visitado (el modo que se usa al evaluar). Es la version practica
    de pi'(a|s) ~ Q(s,a)^(1/tau), la formula de mejora de politica que
    usa AlphaZero.
    """
    movimientos = list(visitas.keys())
    conteos = np.array([visitas[m] for m in movimientos], dtype=float)

    if temperatura == 0:
        return movimientos[int(np.argmax(conteos))]

    probabilidades = conteos ** (1.0 / temperatura)
    probabilidades /= probabilidades.sum()
    return np.random.choice(movimientos, p=probabilidades)


def jugar_partida_autojuego(num_simulaciones=60, temperatura=0.8):
    """
    Juega una partida completa de X contra O, ambos controlados por la
    misma busqueda MCTS (autojuego). Devuelve los tableros visitados y el
    resultado final, listos para actualizar la tabla de valores.
    """
    tablero = tablero_nuevo()
    jugador = 1
    historial = []

    while resultado(tablero) is None:
        visitas = busqueda_mcts(tablero, jugador, num_simulaciones)
        movimiento = elegir_movimiento(visitas, temperatura)
        historial.append(tablero)
        tablero = aplicar_movimiento(tablero, movimiento, jugador)
        jugador = -jugador

    return historial, resultado(tablero)


def actualizar_tabla_valores(historial, resultado_final, alpha=0.5):
    """
    Paso de mejora de politica: acerca el valor estimado de cada posicion
    visitada durante la partida hacia el resultado real, con una
    actualizacion tipo Monte Carlo / TD. Con el tiempo, la tabla deja de
    depender de la heuristica inicial y refleja lo que el propio
    autojuego fue aprendiendo.
    """
    for tablero in historial:
        actual = valor_estimado(tablero)
        tabla_valores[tablero] = actual + alpha * (resultado_final - actual)


def movimiento_aleatorio(tablero):
    return random.choice(movimientos_validos(tablero))


def jugar_vs_aleatorio(num_simulaciones=60):
    """
    Partida de evaluacion: el agente MCTS (con la tabla de valores actual,
    temperatura 0 = juega su mejor movimiento) contra un oponente que
    mueve al azar. El agente MCTS siempre es X.
    """
    tablero = tablero_nuevo()
    jugador = 1
    while resultado(tablero) is None:
        if jugador == 1:
            visitas = busqueda_mcts(tablero, jugador, num_simulaciones)
            movimiento = elegir_movimiento(visitas, temperatura=0)
        else:
            movimiento = movimiento_aleatorio(tablero)
        tablero = aplicar_movimiento(tablero, movimiento, jugador)
        jugador = -jugador
    return resultado(tablero)


# --- BUCLE PRINCIPAL: AUTOJUEGO + MEJORA, GENERACION TRAS GENERACION ----
np.random.seed(0)
random.seed(0)

GENERACIONES = 6
PARTIDAS_AUTOJUEGO_POR_GENERACION = 40
PARTIDAS_EVALUACION = 50

for generacion in range(1, GENERACIONES + 1):
    # 1) Autojuego: generar partidas y usarlas para mejorar la tabla de valores
    for _ in range(PARTIDAS_AUTOJUEGO_POR_GENERACION):
        historial, resultado_final = jugar_partida_autojuego()
        actualizar_tabla_valores(historial, resultado_final)

    # 2) Evaluacion: medir el nivel actual contra un jugador aleatorio
    victorias, empates, derrotas = 0, 0, 0
    for _ in range(PARTIDAS_EVALUACION):
        r = jugar_vs_aleatorio()
        if r == 1:
            victorias += 1
        elif r == 0:
            empates += 1
        else:
            derrotas += 1

    print(
        f"Generacion {generacion}: "
        f"{victorias} victorias, {empates} empates, {derrotas} derrotas "
        f"(sobre {PARTIDAS_EVALUACION} partidas contra un jugador aleatorio) "
        f"| tabla de valores con {len(tabla_valores)} posiciones aprendidas"
    )
```

Al ejecutarlo, verás una línea por generación con el marcador contra el jugador aleatorio y el tamaño creciente de la tabla de valores. En una ejecución de referencia, el agente ya no perdió ni una sola partida desde la primera generación, y el número de empates y victorias fue variando ligeramente según cuánto exploró cada tanda de autojuego: el Tres en Raya es tan pequeño que el patrón converge casi de inmediato, algo que ni de lejos ocurre con el ajedrez o el Go, donde AlphaZero necesitó horas de cómputo masivo en vez de segundos en un portátil.

## Resumen de puntos clave

- Antes de AlphaGo, la IA ya jugaba a nivel sobrehumano en damas, ajedrez, backgammon y Scrabble, combinando búsqueda (minimax, poda alfa-beta) con funciones de valor, a veces aprendidas y a veces diseñadas a mano
- El MCTS es el puente entre esa historia y AlphaGo: busca mediante simulaciones en vez de resolver el juego completo, y converge hacia el valor minimax en juegos de suma cero con información perfecta
- AlphaGo combinó una red de política, una red de valor y MCTS, entrenadas primero por imitación de partidas humanas y después por autojuego, y venció a Lee Sedol 4 a 1
- AlphaZero eliminó los datos humanos por completo: una sola red, un solo algoritmo, aplicado sin cambios a ajedrez, shōgi y Go, superando a Stockfish, Elmo y a la propia AlphaGo en pocas horas de autojuego
- AlphaStar demostró que el mismo patrón —imitación más autojuego, esta vez organizado en una liga de agentes— también funciona bajo información parcial y espacios de acción enormes, venciendo 10 a 0 a un profesional de StarCraft II
- Cada sistema es una recombinación de ideas de los capítulos 2 a 12: funciones de valor, políticas aprendidas por gradiente, control off-policy, aproximación con redes neuronales, y búsqueda tipo MCTS guiada por esas redes

## Ponlo en práctica

El código anterior es tu plantilla. Ahora **experimenta** con estos tres cambios, uno cada vez:

1. **Reduce las simulaciones de MCTS**: baja `num_simulaciones` de 60 a 5 en `jugar_partida_autojuego` y en `jugar_vs_aleatorio`, y vuelve a ejecutar. **Observa** cuántas generaciones hacen falta ahora para dejar de perder contra el jugador aleatorio
2. **Quita la heurística**: cambia `heuristica(tablero)` para que siempre devuelva `0.0`. **Compara** cuánto tarda el agente en volverse competente cuando parte de cero, sin ninguna ventaja inicial, igual que AlphaZero frente a AlphaGo
3. **Adapta el tablero**: intenta llevar este mismo esqueleto (autojuego, MCTS, tabla de valores) a una versión pequeña de Conecta 4, por ejemplo en un tablero de 4x4 con conexión de tres en línea. Tendrás que **reescribir** `LINEAS_GANADORAS`, `movimientos_validos` y el tamaño del tablero, pero el resto del patrón se mantiene intacto

Este manual termina aquí, pero el aprendizaje por refuerzo no. Lo que has recorrido —bandidos, PDM, programación dinámica, Monte Carlo y TD, SARSA y Q-learning, DQN, gradientes de política, planificación, Rainbow, métodos distribuidos, y ahora estos sistemas que hicieron historia— es una base sólida para leer los artículos originales de DeepMind, experimentar con librerías como Gymnasium o PettingZoo, o simplemente seguir jugando con tus propias variantes del código de este capítulo. El campo sigue moviéndose rápido, y la mejor forma de seguirle el ritmo es la misma que aprendiste en este manual: prueba, observa el resultado, ajusta la política y vuelve a intentarlo.
