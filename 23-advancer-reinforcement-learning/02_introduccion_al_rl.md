# Capítulo 2. Introducción al aprendizaje por refuerzo

Imagina que enseñas a un robot a moverse por un almacén, a un algoritmo a decidir cuándo comprar y vender acciones, o a un programa a jugar al ajedrez mejor que cualquier humano. En los tres casos, nadie le entrega al sistema una lista de "movimientos correctos" para cada situación. El sistema prueba, observa qué ocurre y ajusta su comportamiento según el resultado. Esta forma de aprender por prueba y error tiene nombre propio: aprendizaje por refuerzo (RL, por sus siglas en inglés *Reinforcement Learning*), una de las ramas más prometedoras de la inteligencia artificial (IA).

El RL ya está detrás de logros muy concretos. Es la técnica que permitió a AlphaGo derrotar a campeones humanos de Go, la que entrena robots para manipular objetos sin dañarlos y la que ajusta estrategias de negociación automática en mercados financieros. También es la base de los coches autónomos y de muchos sistemas de recomendación. Si trabajas en digitalización de empresas, tarde o temprano te vas a topar con un caso de uso donde el RL aporta algo que otros enfoques de IA no consiguen: la capacidad de aprender una estrategia completa a partir de la propia experiencia, sin que nadie le diga al sistema cuál es la respuesta correcta en cada momento.

En este capítulo vas a construir la base conceptual del RL desde cero. No necesitas experiencia previa en machine learning ni en álgebra lineal: solo curiosidad y ganas de experimentar. Al final, escribirás y ejecutarás tu primer código en Python para ver estos conceptos en acción.

## ¿Qué es el aprendizaje por refuerzo?

Piensa en cómo entrenas a una mascota. No le entregas un manual con la lista de movimientos correctos para sentarse: le das una golosina cuando hace lo que quieres y esperas que, poco a poco, asocie la acción con la recompensa. Así funciona el RL. El sistema —al que llamamos **agente**— interactúa con un entorno, prueba acciones y recibe una señal que le indica si esa acción fue buena o mala. Con el tiempo, aprende a elegir las acciones que maximizan esa señal.

Esta forma de aprender es muy distinta del aprendizaje supervisado, otro enfoque habitual en IA. En el aprendizaje supervisado, entrenas un modelo con un conjunto de datos donde cada ejemplo ya tiene su respuesta correcta (por ejemplo, miles de fotos de gatos y perros ya etiquetadas). En el RL no existe ese conjunto de respuestas correctas: el agente aprende únicamente a través de la interacción y del ensayo y error. Nadie le dice "esta era la acción correcta"; el agente tiene que descubrirlo por sí mismo, probando y observando las consecuencias.

## Los tres protagonistas: agente, entorno y recompensa

Todo problema de RL se puede describir con tres elementos y un bucle que se repite una y otra vez:

- **Agente**: quien toma las decisiones. Puede ser un robot, un programa de trading o un personaje de videojuego
- **Entorno**: todo aquello con lo que el agente interactúa. Puede ser un almacén físico, un mercado financiero o un tablero de juego
- **Recompensa**: un número que el entorno devuelve al agente después de cada acción, e indica si esa acción fue beneficiosa o perjudicial

El bucle funciona así: el agente observa el **estado** actual del entorno, elige una **acción**, el entorno cambia a un nuevo estado y entrega una **recompensa**. El agente repite este ciclo una y otra vez, y usa la experiencia acumulada para mejorar su forma de decidir. A la secuencia completa de estados, acciones y recompensas que vive un agente se le llama **episodio** o **trayectoria**.

Un ejemplo del temario ilustra bien la importancia de diseñar bien la recompensa: un robot que recibe una recompensa positiva por acercarse a un objetivo y una negativa por chocar con obstáculos. Si diseñas mal esa recompensa, el agente puede encontrar "atajos" no deseados: aprenderá exactamente lo que le pagas por aprender, no necesariamente lo que tú querías enseñarle. Este matiz es clave cuando diseñes soluciones de RL para un caso real: la recompensa que definas moldea por completo el comportamiento final del sistema.

También conviene distinguir entre recompensa inmediata y recompensa a largo plazo. A veces conviene sacrificar una ganancia inmediata a cambio de un beneficio mayor más adelante, igual que en la vida real. Para reflejar esta idea, el RL usa un **factor de descuento** (representado con la letra griega gamma, γ), un valor entre cero y uno que indica cuánto peso das a las recompensas futuras frente a las inmediatas. Un valor de γ cercano a uno le da casi la misma importancia al futuro que al presente; un valor cercano a cero hace que el agente solo piense en el corto plazo.

## La política: cómo decide el agente

La estrategia que sigue el agente para elegir acciones se llama **política**, y se representa habitualmente con la letra griega pi (π). Existen dos tipos:

- **Política determinista**: para cada estado, siempre elige la misma acción. Es como memorizar "en este cruce, gira siempre a la izquierda"
- **Política estocástica**: para cada estado, asigna una probabilidad a cada acción posible, en vez de elegir siempre la misma. Es como decir "en este cruce, giro a la izquierda ocho de cada diez veces, y a la derecha las otras dos"

Piensa en un laberinto. Una política determinista que siempre gira a la izquierda en cada cruce puede funcionar bien en un laberinto fijo, pero se queda atascada si aparece un obstáculo inesperado. Una política estocástica, al variar ocasionalmente su elección, tiene más posibilidades de descubrir atajos nuevos o de adaptarse cuando el entorno cambia. Esta capacidad de "probar cosas distintas de vez en cuando" conecta directamente con un dilema central del RL que verás en el próximo capítulo: el equilibrio entre exploración y explotación.

El objetivo último del aprendizaje es encontrar la **política óptima** (π\*): la estrategia que maximiza la recompensa acumulada esperada a largo plazo. Ese es el destino final de cualquier algoritmo de RL, aunque el camino para llegar hasta ahí varíe según el algoritmo que uses.

## El formalismo detrás del entorno: el PDM

Para poder programar y analizar un problema de RL, el entorno se describe con un modelo matemático llamado Proceso de Decisión de Markov (PDM). No te preocupes por la profundidad matemática ahora, eso lo verás con calma en el capítulo cuatro. De momento, basta con que conozcas sus cuatro piezas:

- **S**: el conjunto de todos los estados posibles del entorno
- **A**: el conjunto de todas las acciones que el agente puede ejecutar
- **P**: la función de transición, que indica la probabilidad de pasar de un estado a otro al ejecutar una acción determinada
- **R**: la función de recompensa, que indica qué recompensa recibe el agente al ejecutar una acción en un estado dado

Una idea intuitiva importante detrás del PDM es la llamada "propiedad de Markov": para decidir qué pasa a continuación, solo importa el estado actual, no toda la historia de estados anteriores. Es como jugar una partida de ajedrez: para decidir tu próxima jugada, lo único que importa es la posición actual del tablero, no la secuencia exacta de movimientos que llevaron hasta ahí.

## Una primera idea de Q-learning

Para decidir qué acción tomar, ayuda mucho saber "cuánta recompensa espero conseguir si hago esto". A esa estimación se le llama **valor de acción** o **valor Q**, y uno de los algoritmos más conocidos para aprenderlo es **Q-learning**. La idea es sencilla de expresar de forma intuitiva: el agente mantiene una tabla con un valor estimado para cada combinación de estado y acción, y cada vez que prueba una acción y observa la recompensa real, ajusta un poco esa estimación para acercarla a lo que realmente ocurrió. Con suficientes repeticiones, esos valores estimados terminan reflejando con bastante fidelidad cuánta recompensa aporta cada acción en cada estado, y el agente puede usarlos para elegir siempre la mejor opción disponible.

Q-learning es un algoritmo "sin modelo" (model-free): no necesita conocer de antemano cómo funciona el entorno, aprende únicamente de la experiencia directa. Existe también el enfoque contrario, "basado en modelo" (model-based), donde el agente primero construye una representación de cómo funciona el entorno y luego planifica usando esa representación, un poco como cuando un jugador de ajedrez o Go simula mentalmente varias jugadas futuras antes de mover una ficha. Volverás sobre esta distinción con más detalle en capítulos posteriores.

## El dilema exploración/explotación, en una frase

Para aprender bien, el agente necesita probar acciones nuevas (**explorar**) para descubrir si esconden una recompensa mejor de la que ya conoce, pero también necesita aprovechar lo que ya sabe (**explotar**) para no desperdiciar oportunidades. Demasiada exploración hace que el agente pierda tiempo probando opciones mediocres; demasiada explotación puede dejarlo atascado en una solución aceptable sin descubrir otra mucho mejor. Piensa en una aspiradora robótica: si solo limpia las zonas que ya sabe que están sucias, nunca descubre otras zonas sucias que no conoce; si solo explora zonas nuevas, pierde tiempo sin limpiar lo que ya sabe que necesita atención. Este dilema es tan importante que le dedicamos el capítulo tres completo, así que aquí basta con que lo tengas presente.

## Manos a la obra: tu primer entorno en Python

Antes de aprender ningún algoritmo, vale la pena que veas los conceptos de agente, entorno, política y recompensa funcionando en código real. El siguiente ejemplo crea un entorno mínimo (un pasillo de cinco celdas) y un agente que se mueve de forma completamente aleatoria. Todavía no aprende nada: el objetivo es que veas el bucle agente-entorno en acción, paso a paso.

**Ejecuta esta celda en Google Colab** (solo necesitas `numpy`, que ya viene preinstalado):

```python
import numpy as np

# --- EL ENTORNO -------------------------------------------------------
# Un pasillo de 5 celdas, numeradas de 0 a 4.
# El agente empieza siempre en la celda 0 (izquierda) y el objetivo
# está en la celda 4 (derecha). Puede moverse a la izquierda (0)
# o a la derecha (1).
#
#   [0] [1] [2] [3] [4=META]

class PasilloEntorno:
    def __init__(self, longitud=5):
        self.longitud = longitud
        self.meta = longitud - 1
        self.estado = 0  # posición inicial del agente

    def reset(self):
        """Reinicia el entorno y devuelve el estado inicial."""
        self.estado = 0
        return self.estado

    def step(self, accion):
        """
        Ejecuta una acción en el entorno.
        accion: 0 = mover a la izquierda, 1 = mover a la derecha
        Devuelve: (nuevo_estado, recompensa, terminado)
        """
        if accion == 1:
            self.estado = min(self.estado + 1, self.longitud - 1)
        elif accion == 0:
            self.estado = max(self.estado - 1, 0)

        terminado = self.estado == self.meta

        # Diseño de la recompensa: cada paso "cuesta" un poco (-1),
        # para animar al agente a llegar a la meta cuanto antes.
        # Al llegar a la meta, recompensa extra de +10.
        if terminado:
            recompensa = 10
        else:
            recompensa = -1

        return self.estado, recompensa, terminado


# --- LA POLÍTICA (AGENTE ALEATORIO) ------------------------------------
# Una política estocástica muy simple: en cada estado, elige
# la acción al azar, con la misma probabilidad para cada una.
# No usa ninguna información del estado todavía; en capítulos
# siguientes sustituirás esta política por una que sí aprende.

def politica_aleatoria(estado, num_acciones=2):
    return np.random.randint(num_acciones)


# --- EL BUCLE AGENTE-ENTORNO --------------------------------------------
# Aquí se ve el ciclo central del RL: observar estado, elegir acción,
# recibir nuevo estado y recompensa, y repetir hasta el final del episodio.

def ejecutar_episodio(entorno, max_pasos=20, semilla=None):
    if semilla is not None:
        np.random.seed(semilla)

    estado = entorno.reset()
    recompensa_acumulada = 0
    nombres_acciones = {0: "izquierda", 1: "derecha"}

    print(f"Estado inicial: {estado}")

    for paso in range(1, max_pasos + 1):
        accion = politica_aleatoria(estado)
        nuevo_estado, recompensa, terminado = entorno.step(accion)
        recompensa_acumulada += recompensa

        print(
            f"Paso {paso}: estado={estado} -> "
            f"acción={nombres_acciones[accion]} -> "
            f"nuevo_estado={nuevo_estado}, recompensa={recompensa}"
        )

        estado = nuevo_estado

        if terminado:
            print(f"\n¡Meta alcanzada en el paso {paso}!")
            break

    print(f"Recompensa acumulada del episodio: {recompensa_acumulada}")
    return recompensa_acumulada


# Ejecuta un episodio completo con una semilla fija, para que el
# resultado sea reproducible cada vez que ejecutes esta celda.
ejecutar_episodio(PasilloEntorno(), max_pasos=20, semilla=42)
```

Al ejecutar esta celda, verás impresa la secuencia completa de estados, acciones y recompensas de un episodio. Como la política es aleatoria, el agente a veces avanza, a veces retrocede y puede tardar bastantes pasos en llegar a la meta, o incluso no llegar dentro del límite de pasos. Eso es exactamente lo esperable: sin ningún tipo de aprendizaje, el agente no tiene ninguna razón para preferir "derecha" sobre "izquierda".

**Cambia este valor y observa qué pasa**: modifica `semilla=42` por otro número y vuelve a ejecutar la celda. Verás una trayectoria distinta, porque cambia la secuencia de acciones aleatorias. **Prueba también** a subir `max_pasos` a 50 o a reducir la longitud del pasillo a 3 celdas, y compara cuántos pasos tarda el agente en llegar a la meta en cada caso.

## Resumen de puntos clave

- El aprendizaje por refuerzo (RL) enseña a un agente a decidir mediante prueba y error, sin un conjunto de respuestas correctas como en el aprendizaje supervisado
- El bucle central del RL conecta tres piezas: agente, entorno y recompensa, en un ciclo de estado → acción → nuevo estado → recompensa
- La política (π) es la estrategia que usa el agente para elegir acciones; puede ser determinista o estocástica
- El Proceso de Decisión de Markov (PDM) formaliza el entorno mediante cuatro elementos: estados (S), acciones (A), transiciones (P) y recompensas (R)
- El factor de descuento (γ) regula cuánto peso das a las recompensas futuras frente a las inmediatas
- Q-learning estima, para cada combinación de estado y acción, cuánta recompensa cabe esperar, y ajusta esa estimación con la experiencia
- El diseño de la recompensa importa muchísimo: una recompensa mal pensada puede enseñar al agente el comportamiento equivocado
- Explorar y explotar son dos necesidades en tensión constante; encontrar el equilibrio adecuado es uno de los grandes retos del RL

## Ponlo en práctica

Toma el código anterior y modifícalo para comprobar por ti mismo cómo cambia el comportamiento del agente:

1. **Añade una tercera acción**, por ejemplo "quedarse quieto" (acción 2), y ajusta `politica_aleatoria` y `step` para soportarla
2. **Cambia el diseño de la recompensa**: en vez de -1 por paso, prueba con 0 por paso y observa si el agente sigue teniendo algún incentivo para llegar rápido a la meta
3. **Ejecuta diez episodios seguidos** (usa un bucle `for` alrededor de `ejecutar_episodio`) sin fijar la semilla, y anota cuántos pasos tarda el agente aleatorio en llegar a la meta en cada uno. Esa variabilidad es justo lo que un algoritmo de aprendizaje, como Q-learning, está diseñado para reducir con el tiempo

Guarda tus resultados: los volverás a usar como punto de comparación cuando, en próximos capítulos, sustituyas la política aleatoria por una que realmente aprende.

---

En el capítulo tres profundizamos en el dilema exploración/explotación que has visto de pasada aquí, con el problema clásico de los bandidos multi-brazo (*multi-armed bandits*) como punto de partida.
