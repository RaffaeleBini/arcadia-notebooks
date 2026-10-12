# Capítulo 4. Procesos de Decisión de Markov: el lenguaje del RL

En el capítulo anterior conociste los bandidos, tu primera puerta de entrada al aprendizaje por refuerzo (RL, del inglés reinforcement learning): un agente que elige entre varias palancas y solo debe equilibrar exploración y explotación, porque cada tirada es independiente de las anteriores. Ese mundo es simple porque no existe un "estado" que cambie: da igual lo que hicieras antes, la máquina siempre te ofrece las mismas opciones.

La vida real casi nunca funciona así. Cuando conduces, riegas un cultivo o gestionas el inventario de un almacén, cada decisión te deja en una situación distinta que condiciona qué opciones tienen sentido después. Necesitas un lenguaje formal que capture esa cadena de causa y efecto para razonar matemáticamente sobre ella: el Proceso de Decisión de Markov (PDM), conocido en inglés como Markov Decision Process (MDP) —encontrarás ambos términos en el código y en la literatura—.

Este capítulo te presenta el PDM pieza por pieza: sus elementos, la propiedad matemática que lo hace manejable y cómo se derivan las ecuaciones que usarás durante el resto del manual. Al final tendrás en Python un mundo-cuadrícula (GridWorld) funcional que el capítulo 5 ampliará para encontrar la mejor estrategia posible.

## De la máquina tragaperras al mapa completo

Piensa en un robot que recoge latas de aluminio para reciclar y funciona con batería. En cada instante, se encuentra en uno de dos **estados** posibles según su nivel de carga: alto o bajo. Según el estado en el que esté, puede elegir entre varias **acciones**: esperar sin moverse, salir a buscar latas o, si va escaso de energía, volver a la base a recargar.

Aquí ya tienes los primeros ingredientes del PDM:

- **S** (states, estados): el conjunto de situaciones posibles en las que puede estar el agente —para el robot, {carga alta, carga baja}—
- **A** (actions, acciones): las opciones disponibles en cada estado —esperar, buscar o recargar—
- **P**: la función de transición: probabilidad de terminar en cada estado siguiente, según dónde estabas y qué acción tomaste. Buscar con la batería alta casi siempre te deja igual de cargado, pero existe una probabilidad de quedarte sin energía y necesitar un rescate
- **R**: la función de recompensa, el número que recibes tras cada transición. Buscar latas da una recompensa positiva; quedarte tirado sin batería, una penalización fuerte
- **γ** (gamma): el factor de descuento, un número entre cero y uno que decide cuánto te importa el futuro

Un PDM completo es la tupla **(S, A, P, R, γ)**: estados, acciones, cómo se encadenan unas con otras, qué recompensa deja cada paso y cuánto pesa el mañana frente al hoy.

### La propiedad de Markov: solo importa el presente

El nombre "Markov" no es decorativo. El PDM asume la **propiedad de Markov**: el futuro depende únicamente del estado actual, no de cómo llegaste hasta él. Da igual si el robot lleva dos horas buscando latas o si acaba de recargar: lo único relevante para decidir qué hacer ahora es su nivel de batería en este momento.

Esta suposición simplifica enormemente el problema: no necesitas guardar todo el historial de decisiones pasadas, basta con un estado bien diseñado que resuma lo importante. Por eso, al diseñar un sistema real —por ejemplo, el estado de un vehículo autónomo, que podría incluir posición, velocidad y obstáculos cercanos—, tu tarea es elegir una representación que se acerque lo máximo posible a esta propiedad.

### La política: tu estrategia de decisión

Una **política**, escrita como π (pi), es la estrategia que sigue el agente: una regla que, para cada estado, indica qué acción tomar. Puede ser:

- **Determinista**: en el estado "batería baja", π siempre elige "recargar"
- **Estocástica**: en el estado "batería alta", π elige "buscar" el 70% de las veces y "esperar" el 30% restante

Una política aleatoria, que reparte la misma probabilidad entre todas las acciones disponibles, es el punto de partida más simple posible para empezar a razonar sobre un PDM, y es justo la que usarás en el ejemplo de código.

### Cuánto vale un estado: la función de valor

Aquí llega la pregunta central de este capítulo: si sigues una política π, ¿cuánto "vale" estar en un estado concreto? La respuesta no es solo lo que ganas en este instante, sino la suma de todas las recompensas futuras, descontadas por γ. A esa suma se le llama **retorno** (Gt):

```
Gt = Rt+1 + γ·Rt+2 + γ²·Rt+3 + ...
```

El factor γ hace dos cosas a la vez: refleja que sueles preferir ganar algo hoy antes que la misma recompensa dentro de diez pasos, y evita que la suma crezca sin límite en tareas que podrían continuar indefinidamente, ya que cada recompensa futura pesa un poco menos que la anterior. Un γ cercano a cero produce un agente miope, que solo mira el paso inmediato; uno cercano a uno lo vuelve previsor, dispuesto a sacrificar recompensa inmediata por un beneficio mayor más adelante.

A partir del retorno se definen dos **funciones de valor**:

- **Vπ(s)**: el retorno esperado si empiezas en el estado s y sigues la política π a partir de ahí
- **Qπ(s,a)**: el retorno esperado si, estando en el estado s, tomas la acción a y después sigues π

Ambas están relacionadas de forma directa: Vπ(s) es el promedio de Qπ(s,a) sobre todas las acciones, ponderado por la probabilidad que la política asigna a cada una.

### Las ecuaciones de Bellman: recompensa ahora, valor descontado después

El retorno tiene una propiedad muy útil: se puede escribir de forma recursiva, porque Gt = Rt+1 + γ·Gt+1. En palabras, lo que vale este momento es la recompensa de este paso más el valor descontado de todo lo que viene después. Esa idea, aplicada a la función de valor, es la **ecuación de Bellman de expectación**:

```
Vπ(s) = Σ_a π(a|s) · Σ_s' P(s'|s,a) · [R(s,a,s') + γ·Vπ(s')]
```

No dejes que la notación te asuste: es solo una suma ponderada. Para cada acción posible (según tu política) y cada posible estado siguiente (según la probabilidad de transición), sumas la recompensa inmediata más su valor descontado. La misma idea, aplicada a Q, da la versión para pares estado-acción:

```
Qπ(s,a) = Σ_s' P(s'|s,a) · [R(s,a,s') + γ·Σ_a' π(a'|s') Qπ(s',a')]
```

Estas dos ecuaciones definen el valor de una política ya fija: son un sistema lineal, y por eso puedes resolverlas —o aproximarlas iterando, como harás en el código de este capítulo—.

### La pregunta que abre el capítulo 5: ¿y la mejor política?

Todo lo anterior asume que ya tienes una política π y solo quieres saber cuánto vale. Pero la pregunta que de verdad te interesa es otra: ¿cuál es la mejor política posible? A esa política óptima se le llama **π\***, y su función de valor **V\*(s)** cumple la **ecuación de optimalidad de Bellman**:

```
V*(s) = max_a Σ_s' P(s'|s,a) · [R(s,a,s') + γ·V*(s')]
```

La diferencia con la ecuación de expectación es sutil pero decisiva: en lugar de promediar sobre las acciones según tu política, te quedas con el máximo. Ya no preguntas "¿cuánto vale este estado si sigo mi estrategia actual?", sino "¿cuánto valdría si, a partir de ahora, eligiera siempre la mejor acción posible?". Ese máximo convierte el sistema en no lineal —ya no se resuelve con álgebra directa— y necesita métodos iterativos, que es justo de lo que trata el próximo capítulo: programación dinámica.

Quédate con la idea clave: una vez tienes V* o Q*, obtienes la política óptima directamente, actuando de manera voraz —π\*(s) = argmax_a Q\*(s,a)—. El reto, que resolverás en el capítulo 5, es cómo calcular V* o Q* en primer lugar.

## Ejemplo de código: un GridWorld y su función de valor

Vas a construir un mundo-cuadrícula pequeño, de cuatro filas por cuatro columnas (4×4), con una única casilla terminal (la meta) en la esquina inferior derecha. El agente puede moverse en las cuatro direcciones cardinales; si intenta salir del tablero, se queda donde está. Cada paso cuesta −1 de recompensa, salvo llegar a la meta, donde el episodio termina.

Vas a definir una política aleatoria —elige cada dirección con la misma probabilidad— y calcularás su Vπ aplicando la ecuación de Bellman de expectación una y otra vez, hasta que los valores dejen de cambiar. Este proceso se llama **evaluación iterativa de políticas**: la misma ecuación que acabas de ver, convertida en una regla de actualización que repites barrido tras barrido. Fíjate en que la clase `GridWorld` y las funciones están escritas para ser reutilizadas —el capítulo 5 añadirá mejora de políticas encima de esta misma base, sin reescribirla—.

```python
import numpy as np

# ---------------------------------------------------------------
# GridWorld: un mundo-cuadrícula pequeño para practicar el PDM.
#
# Esta clase se diseña para que el capítulo 5 la reutilice tal cual
# y le añada iteración de políticas e iteración de valores encima.
# ---------------------------------------------------------------

ACCIONES = ["arriba", "abajo", "izquierda", "derecha"]

# Cada acción mueve al agente una casilla en una dirección (delta_fila, delta_columna)
DESPLAZAMIENTOS = {
    "arriba": (-1, 0),
    "abajo": (1, 0),
    "izquierda": (0, -1),
    "derecha": (0, 1),
}


class GridWorld:
    """
    Un mundo-cuadrícula de n_filas x n_columnas con una casilla terminal (la meta).

    - Los estados son tuplas (fila, columna).
    - Las acciones son las cuatro direcciones cardinales.
    - Las transiciones son deterministas: moverse contra un muro o el
      borde del tablero deja al agente en la misma casilla.
    - La recompensa es 'recompensa_paso' en cada movimiento (por defecto -1),
      salvo en la casilla terminal, donde el episodio ya ha terminado.
    """

    def __init__(self, n_filas=4, n_columnas=4, estados_terminales=None, recompensa_paso=-1.0):
        self.n_filas = n_filas
        self.n_columnas = n_columnas
        self.recompensa_paso = recompensa_paso

        # Por defecto, la meta es la esquina inferior derecha
        if estados_terminales is None:
            estados_terminales = [(n_filas - 1, n_columnas - 1)]
        self.estados_terminales = set(estados_terminales)

        # Lista de todos los estados (fila, columna)
        self.estados = [(f, c) for f in range(n_filas) for c in range(n_columnas)]
        self.acciones = ACCIONES

    def es_terminal(self, estado):
        return estado in self.estados_terminales

    def estado_siguiente(self, estado, accion):
        """Devuelve la casilla resultante de aplicar 'accion' desde 'estado'.
        Si el movimiento choca con el borde del tablero, el agente se queda quieto."""
        if self.es_terminal(estado):
            return estado

        df, dc = DESPLAZAMIENTOS[accion]
        fila, columna = estado
        nueva_fila = max(0, min(self.n_filas - 1, fila + df))
        nueva_columna = max(0, min(self.n_columnas - 1, columna + dc))
        return (nueva_fila, nueva_columna)

    def recompensa(self, estado, accion, estado_siguiente):
        """Recompensa inmediata de la transición. En este GridWorld es fija
        (-1 por paso), pero se recibe (s, a, s') completo para poder
        ampliar el ejemplo más adelante (por ejemplo, recompensas distintas por casilla)."""
        if self.es_terminal(estado):
            return 0.0
        return self.recompensa_paso


def politica_aleatoria(mundo):
    """Construye una política estocástica que, en cada estado, elige
    cualquiera de las cuatro acciones con la misma probabilidad (0.25).
    Se representa como un diccionario: politica[estado][accion] = probabilidad."""
    probabilidad_uniforme = 1.0 / len(mundo.acciones)
    return {
        estado: {accion: probabilidad_uniforme for accion in mundo.acciones}
        for estado in mundo.estados
    }


def evaluacion_iterativa_politica(mundo, politica, gamma=0.9, umbral=1e-4, max_iteraciones=1000):
    """
    Calcula Vpi(s) para una política fija aplicando repetidamente
    la ecuación de Bellman de expectación:

        Vpi(s) = suma_a pi(a|s) * suma_s' P(s'|s,a) * [R(s,a,s') + gamma * Vpi(s')]

    Como este GridWorld es determinista, P(s'|s,a) vale 1 para el único
    estado_siguiente que devuelve mundo.estado_siguiente(), y 0 para el resto,
    así que la suma sobre s' desaparece y queda un único término.

    Devuelve V como un diccionario {estado: valor} y el número de barridos usados.
    """
    V = {estado: 0.0 for estado in mundo.estados}

    for iteracion in range(1, max_iteraciones + 1):
        delta_maximo = 0.0
        V_nuevo = dict(V)  # barrido síncrono: usamos siempre los valores del barrido anterior

        for estado in mundo.estados:
            if mundo.es_terminal(estado):
                V_nuevo[estado] = 0.0
                continue

            valor_esperado = 0.0
            for accion in mundo.acciones:
                prob_accion = politica[estado][accion]
                estado_sig = mundo.estado_siguiente(estado, accion)
                r = mundo.recompensa(estado, accion, estado_sig)
                valor_esperado += prob_accion * (r + gamma * V[estado_sig])

            V_nuevo[estado] = valor_esperado
            delta_maximo = max(delta_maximo, abs(V_nuevo[estado] - V[estado]))

        V = V_nuevo
        if delta_maximo < umbral:
            break

    return V, iteracion


def imprimir_valores(mundo, V):
    """Muestra V como una cuadrícula legible, para inspeccionar el resultado."""
    for f in range(mundo.n_filas):
        fila_textos = []
        for c in range(mundo.n_columnas):
            fila_textos.append(f"{V[(f, c)]:6.2f}")
        print(" ".join(fila_textos))


if __name__ == "__main__":
    mundo = GridWorld(n_filas=4, n_columnas=4, recompensa_paso=-1.0)
    politica = politica_aleatoria(mundo)

    V, num_barridos = evaluacion_iterativa_politica(mundo, politica, gamma=0.9, umbral=1e-4)

    print(f"Convergió en {num_barridos} barridos.\n")
    print("Función de valor Vpi bajo la política aleatoria:")
    imprimir_valores(mundo, V)
```

Al ejecutarlo en Colab obtienes algo parecido a esto (los valores exactos dependen de γ y del umbral de convergencia que elijas):

```
Convergió en 76 barridos.

Función de valor Vpi bajo la política aleatoria:
 -9.36  -9.22  -8.97  -8.75
 -9.22  -8.97  -8.50  -7.97
 -8.97  -8.50  -7.42  -5.76
 -8.75  -7.97  -5.76   0.00
```

Observa el patrón: las casillas más cercanas a la meta (la esquina inferior derecha, en 0.00) tienen un valor menos negativo, porque bajo una política aleatoria hacen falta en promedio menos pasos para llegar. Las más alejadas, como la esquina superior izquierda, arrastran más pasos de recompensa −1 antes de terminar. Eso es exactamente lo que mide Vπ: el coste esperado, descontado, de llegar hasta el final siguiendo tu política actual.

## Puntos clave

- Un PDM se define por la tupla (S, A, P, R, γ): estados, acciones, transiciones, recompensas y factor de descuento
- La propiedad de Markov dice que el futuro solo depende del estado presente, no del historial completo
- Una política π puede ser determinista o estocástica, y asigna a cada estado una acción o una distribución de probabilidad sobre ellas
- Vπ(s) mide el retorno esperado desde un estado siguiendo π; Qπ(s,a) hace lo mismo pero fijando también la primera acción
- La ecuación de Bellman de expectación descompone tu valor actual en recompensa inmediata más lo que vale, descontado, el estado siguiente
- La ecuación de optimalidad de Bellman sustituye el promedio ponderado por un máximo sobre acciones, y define V* y la política óptima π*
- La evaluación iterativa de políticas aplica la ecuación de Bellman de expectación repetidamente hasta que Vπ converge

## Ponlo en práctica

Toma el código anterior y experimenta con él antes de seguir al capítulo 5:

1. **Cambia el factor de descuento.** Prueba con γ = 0.5 y con γ = 0.99. ¿Cómo cambian los valores de las casillas alejadas de la meta? ¿Qué le pasa al número de barridos que tarda en converger
2. **Mueve la meta.** En lugar de la esquina inferior derecha, coloca el estado terminal en el centro del tablero (pasa `estados_terminales=[(1, 1)]` al construir el `GridWorld`) y observa cómo se reorganiza Vπ alrededor de la nueva meta
3. **Añade una segunda meta.** Pasa una lista con dos casillas terminales y comprueba que el agente ahora "ve" ambos caminos posibles de escape

¡Ejecuta cada variante y anota qué intuición confirma sobre γ y sobre la forma en que se propaga el valor por la cuadrícula!

---

Ya tienes el mapa —el PDM completo— y sabes calcular cuánto vale seguir una política concreta. En el capítulo 5 vas a usar este mismo `GridWorld` para dar el siguiente paso. En lugar de solo evaluar una estrategia fija, la mejorarás paso a paso mediante iteración de políticas e iteración de valores. Así llegarás a la política óptima π*.
