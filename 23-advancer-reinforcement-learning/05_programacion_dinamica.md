# Capítulo 5. Programación dinámica: iteración de políticas y de valores

En el capítulo anterior conociste los Procesos de Decisión de Markov (PDM): un modelo formal con estados, acciones, probabilidades de transición, recompensas y un factor de descuento. También aprendiste a calcular Vπ, la función de valor de una política fija, mediante evaluación iterativa de políticas: aplicar la ecuación de Bellman una y otra vez, barrido tras barrido, hasta que los valores dejan de cambiar.

Eso es útil, pero incompleto. Saber cuánto vale seguir una política concreta no te dice si esa política es buena. Es como tener un mapa de distancias desde tu oficina hasta cada punto de la ciudad, pero seguir usando siempre la misma ruta, sin comprobar si existe un atajo mejor. En este capítulo das el paso que faltaba: pasar de "puedo evaluar una política" a "puedo encontrar la mejor política posible". Las herramientas que vas a usar se llaman, en conjunto, programación dinámica: una familia de algoritmos que calculan políticas óptimas cuando conoces el modelo completo del entorno, es decir, cuando conoces P (las probabilidades de transición) y R (las recompensas) de memoria, sin necesidad de interactuar con él por ensayo y error.

## De evaluar a mejorar

Imagina que ya calculaste Vπ para una política π cualquiera, por ejemplo la política aleatoria del capítulo anterior. Ahora, párate en cada casilla del GridWorld y hazte una pregunta distinta: "si en vez de seguir la política actual, tomo la mejor acción posible en este único paso, y luego vuelvo a seguir π, ¿mejoro mi resultado?". Comparar esa opción con lo que hacía la política original y quedarte siempre con la mejor acción se llama mejora de políticas: construir una nueva política π' que actúa de forma codiciosa (greedy) respecto a Vπ, es decir, que en cada estado elige la acción que maximiza recompensa inmediata más valor descontado del siguiente estado.

Esto casi nunca empeora las cosas. Si en un estado la acción codiciosa coincide con lo que ya hacías, no pierdes nada; si es distinta, mejoras al menos un poco. Por eso π' es siempre igual o mejor que π en todos los estados. Es el mismo principio que aplicas cuando revisas tu ruta habitual al trabajo con la información de tráfico de hoy: en el peor de los casos, sigues igual que antes; en el mejor, encuentras un atajo real.

## Iteración de políticas: evaluar y mejorar, una y otra vez

Aquí aparece la idea central del capítulo. Si mejorar una política una vez ya ayuda, ¿qué pasa si repites el ciclo completo? Evalúas la política actual hasta el final (calculas su Vπ exacta con evaluación iterativa), la mejoras de forma codiciosa, evalúas la política nueva, la vuelves a mejorar, y así sucesivamente. A este procedimiento se le llama iteración de políticas (policy iteration).

Piensa en dos flechas que se persiguen. La evaluación es una flecha hacia arriba: ajusta los valores hasta que encajen a la perfección con la política actual. La mejora es una flecha hacia abajo: en cuanto tienes esos valores exactos, cambias la política, así que los valores que acabas de calcular ya no describen bien a la nueva política. El proceso repite ese vaivén (arriba, abajo, arriba, abajo) hasta que en un ciclo la política ya no cambia. En ese momento, la acción codiciosa en cada estado coincide exactamente con lo que la política ya hacía, lo cual solo puede significar una cosa: se cumple la ecuación de optimalidad de Bellman, y por tanto la política actual ya es la política óptima π*. Como el número de políticas deterministas posibles en un PDM finito es limitado y cada ciclo produce una política igual o mejor que la anterior, el algoritmo tiene garantizado llegar a π* en un número finito de ciclos.

## Iteración de valores: el atajo directo

Iteración de políticas funciona, pero tiene un punto flojo: en cada ciclo, evaluar la política a fondo (repetir la evaluación iterativa hasta que converja del todo) puede costar muchos barridos, y buena parte de esa precisión se va a tirar en cuanto mejores la política de todas formas. Aquí surge una pregunta natural: ¿hace falta esperar a que la evaluación converja del todo antes de mejorar?

La respuesta es no, y lleva directamente a iteración de valores (value iteration): en vez de completar la evaluación, la truncas a un único barrido y mezclas evaluación y mejora en el mismo paso. Concretamente, en cada barrido, cada estado actualiza su valor tomando directamente el máximo entre sus acciones (en lugar de promediar según una política fija), usando la ecuación de optimalidad de Bellman como regla de actualización. Cuando esos valores dejan de cambiar de forma apreciable, derivas la política óptima con una última mirada de un paso: en cada estado, eliges la acción que maximiza recompensa más valor descontado del siguiente estado.

Puedes verlo como un caso particular dentro de la misma familia que la iteración de políticas: es una iteración de políticas donde la fase de evaluación se ha recortado a un solo barrido. En la práctica, suele necesitar más barridos individuales que ciclos completos necesita la iteración de políticas, pero cada barrido es mucho más barato, así que el coste total termina siendo menor en muchos problemas.

## Sincronía, actualizaciones en el sitio y otras variantes

Todo lo anterior asume programación dinámica síncrona: en cada barrido, actualizas todos los estados a la vez, usando siempre los valores del barrido anterior. Existe también la programación dinámica asíncrona, que actualiza los estados de uno en uno (o por subconjuntos), usando el valor más reciente disponible en el momento, sin esperar a completar un barrido entero. Una variante habitual son las actualizaciones en el sitio (in-place): en vez de mantener una copia "antigua" y una "nueva" de la función de valor, se actualiza directamente el mismo array, lo que suele acelerar la convergencia. Otras variantes priorizan qué estado actualizar primero según el tamaño del error de Bellman en cada casilla, o empiezan a actualizar cerca del estado terminal y avanzan hacia atrás. No profundices aquí: solo quédate con la idea de que, en problemas grandes, no hace falta barrer la cuadrícula entera de forma ordenada para que el método siga convergiendo.

## Ejemplo en Python: iteración de políticas contra iteración de valores

El siguiente código retoma el GridWorld del capítulo anterior: una cuadrícula de 4×4 (16 estados), con una única casilla terminal en la esquina inferior derecha, recompensa de −1 por cada paso que no llega a la meta y 0 al alcanzarla. Los movimientos son deterministas: si el agente intenta salir del tablero, se queda quieto. Sobre ese mismo GridWorld, implementa iteración de políticas e iteración de valores, y compara cuántos barridos necesita cada una para converger. Puedes copiarlo y ejecutarlo directamente en Google Colab: solo necesita `numpy`.

```python
import numpy as np

# ---------------------------------------------------------------
# GridWorld: el mismo mundo-cuadrícula 4x4 del capítulo anterior,
# con una casilla terminal (la meta) y recompensa -1 por paso.
# ---------------------------------------------------------------

ACCIONES = ["arriba", "abajo", "izquierda", "derecha"]

DESPLAZAMIENTOS = {
    "arriba": (-1, 0),
    "abajo": (1, 0),
    "izquierda": (0, -1),
    "derecha": (0, 1),
}

FLECHAS = {"arriba": "^", "abajo": "v", "izquierda": "<", "derecha": ">"}


class GridWorld:
    """Cuadrícula n_filas x n_columnas con una casilla terminal (la meta).
    Estados = tuplas (fila, columna). Transiciones deterministas: moverse
    contra el borde del tablero deja al agente en la misma casilla."""

    def __init__(self, n_filas=4, n_columnas=4, estados_terminales=None, recompensa_paso=-1.0):
        self.n_filas = n_filas
        self.n_columnas = n_columnas
        self.recompensa_paso = recompensa_paso

        if estados_terminales is None:
            estados_terminales = [(n_filas - 1, n_columnas - 1)]  # esquina inferior derecha
        self.estados_terminales = set(estados_terminales)

        self.estados = [(f, c) for f in range(n_filas) for c in range(n_columnas)]
        self.acciones = ACCIONES

    def es_terminal(self, estado):
        return estado in self.estados_terminales

    def estado_siguiente(self, estado, accion):
        if self.es_terminal(estado):
            return estado
        df, dc = DESPLAZAMIENTOS[accion]
        fila, columna = estado
        nueva_fila = max(0, min(self.n_filas - 1, fila + df))
        nueva_columna = max(0, min(self.n_columnas - 1, columna + dc))
        return (nueva_fila, nueva_columna)

    def recompensa(self, estado, accion, estado_siguiente):
        if self.es_terminal(estado):
            return 0.0
        return self.recompensa_paso


def politica_aleatoria(mundo):
    """Política de partida: en cada estado, las cuatro acciones son
    igual de probables. Es el punto de arranque de iteración de políticas."""
    probabilidad_uniforme = 1.0 / len(mundo.acciones)
    return {
        estado: {accion: probabilidad_uniforme for accion in mundo.acciones}
        for estado in mundo.estados
    }


def evaluacion_iterativa_politica(mundo, politica, gamma=0.9, umbral=1e-4, max_iteraciones=1000):
    """Ya la viste en el capítulo anterior: aplica la ecuación de Bellman de
    expectativa hasta que Vpi deja de cambiar. La repetimos aquí para que el
    ejemplo sea autocontenido."""
    V = {estado: 0.0 for estado in mundo.estados}
    for iteracion in range(1, max_iteraciones + 1):
        delta_maximo = 0.0
        V_nuevo = dict(V)
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


def mejora_politica(mundo, V, gamma=0.9):
    """Construye una política determinista, codiciosa respecto a V: en cada
    estado, elige la acción que maximiza recompensa + gamma * V(siguiente)."""
    nueva_politica = {}
    for estado in mundo.estados:
        if mundo.es_terminal(estado):
            nueva_politica[estado] = {a: 0.0 for a in mundo.acciones}
            continue
        valores_q = {}
        for accion in mundo.acciones:
            estado_sig = mundo.estado_siguiente(estado, accion)
            r = mundo.recompensa(estado, accion, estado_sig)
            valores_q[accion] = r + gamma * V[estado_sig]
        mejor_accion = max(valores_q, key=valores_q.get)
        nueva_politica[estado] = {a: (1.0 if a == mejor_accion else 0.0) for a in mundo.acciones}
    return nueva_politica


def iteracion_politicas(mundo, gamma=0.9, umbral=1e-4):
    """Alterna evaluación completa y mejora hasta que la política deja de cambiar."""
    politica = politica_aleatoria(mundo)
    ciclos_evaluacion_mejora = 0
    barridos_totales = 0
    while True:
        ciclos_evaluacion_mejora += 1
        V, barridos = evaluacion_iterativa_politica(mundo, politica, gamma, umbral)
        barridos_totales += barridos
        nueva_politica = mejora_politica(mundo, V, gamma)
        if nueva_politica == politica:  # la política ya no cambia: es óptima
            break
        politica = nueva_politica
    return politica, V, ciclos_evaluacion_mejora, barridos_totales


def iteracion_valores(mundo, gamma=0.9, umbral=1e-4, max_iteraciones=1000):
    """Funde evaluación y mejora: cada barrido actualiza V con el máximo
    sobre acciones (ecuación de optimalidad de Bellman), no con un promedio
    según una política fija."""
    V = {estado: 0.0 for estado in mundo.estados}
    for iteracion in range(1, max_iteraciones + 1):
        delta_maximo = 0.0
        V_nuevo = dict(V)
        for estado in mundo.estados:
            if mundo.es_terminal(estado):
                V_nuevo[estado] = 0.0
                continue
            valores_q = []
            for accion in mundo.acciones:
                estado_sig = mundo.estado_siguiente(estado, accion)
                r = mundo.recompensa(estado, accion, estado_sig)
                valores_q.append(r + gamma * V[estado_sig])
            V_nuevo[estado] = max(valores_q)
            delta_maximo = max(delta_maximo, abs(V_nuevo[estado] - V[estado]))
        V = V_nuevo
        if delta_maximo < umbral:
            break
    politica_final = mejora_politica(mundo, V, gamma)  # una última mirada de un paso
    return politica_final, V, iteracion


def imprimir_politica(mundo, politica):
    for f in range(mundo.n_filas):
        fila_textos = []
        for c in range(mundo.n_columnas):
            estado = (f, c)
            if mundo.es_terminal(estado):
                fila_textos.append(" G ")
            else:
                mejor_accion = max(politica[estado], key=politica[estado].get)
                fila_textos.append(f" {FLECHAS[mejor_accion]} ")
        print("|".join(fila_textos))


if __name__ == "__main__":
    mundo = GridWorld(n_filas=4, n_columnas=4, recompensa_paso=-1.0)
    gamma = 0.9

    print("=== Iteración de políticas ===")
    politica_pi, V_pi, ciclos, barridos_pi = iteracion_politicas(mundo, gamma=gamma)
    print(f"Ciclos evaluación+mejora: {ciclos}")
    print(f"Barridos de evaluación acumulados: {barridos_pi}")
    imprimir_politica(mundo, politica_pi)

    print("\n=== Iteración de valores ===")
    politica_vi, V_vi, barridos_vi = iteracion_valores(mundo, gamma=gamma)
    print(f"Barridos hasta converger: {barridos_vi}")
    imprimir_politica(mundo, politica_vi)

    diferencia_maxima = max(abs(V_pi[s] - V_vi[s]) for s in mundo.estados)
    print(f"\nDiferencia máxima entre las dos V*: {diferencia_maxima:.6f}")
```

Con γ (gamma) = 0,9, ambos métodos llegan exactamente a la misma política óptima: desde cualquier casilla, la flecha apunta siempre hacia la meta por el camino más corto (abajo y a la derecha, combinados según la posición). La diferencia está en el esfuerzo. Iteración de políticas necesita solo tres ciclos completos de evaluación y mejora, pero cada evaluación se ejecuta hasta el final, así que en total acumula 90 barridos. Iteración de valores llega al mismo resultado en apenas siete barridos, porque nunca espera a que la evaluación converja del todo antes de mejorar. Este patrón es habitual: iteración de políticas suele necesitar menos ciclos "externos", pero cada uno cuesta más; iteración de valores hace muchos barridos baratos en vez de pocos barridos caros.

## Puntos clave

- La mejora de políticas construye una política nueva actuando de forma codiciosa (greedy) respecto a Vπ; nunca empeora la política original
- Iteración de políticas alterna evaluación completa y mejora hasta que la política deja de cambiar; en ese punto se cumple la ecuación de optimalidad de Bellman y la política es óptima
- Iteración de valores funde evaluación y mejora en un único barrido por iteración, usando el máximo sobre acciones en vez de un promedio según una política fija
- Ambos métodos requieren conocer el modelo completo del entorno (P y R); por eso pertenecen a la programación dinámica, no al aprendizaje por ensayo y error
- La programación dinámica asíncrona (actualizaciones en el sitio, priorización por error de Bellman, actualización desde el estado terminal hacia atrás) permite escalar la idea a problemas más grandes sin perder la garantía de convergencia

## Ponlo en práctica

Toma el código de este capítulo y modifícalo tú misma o tú mismo:

- **Añade obstáculos** al GridWorld: elige dos o tres casillas que el agente no pueda atravesar (trátalas como si fueran un muro, de modo que intentar entrar en ellas lo deje en su sitio) y observa cómo la política óptima las rodea
- **Cambia el factor de descuento** γ (por ejemplo a 0,5 o a 0,99) y comprueba si cambia el número de barridos que necesita cada método para converger
- **Añade una segunda casilla terminal** con recompensa distinta (por ejemplo, una meta secundaria con recompensa menor) y comprueba cómo se reparte la cuadrícula entre las dos metas según la posición de cada estado

¡Ejecuta el notebook, cambia el GridWorld y comparte con tu equipo qué política óptima obtuvo cada configuración!

---

Hasta ahora, en todo el bloque de programación dinámica, asumiste que conocías el modelo del entorno: las probabilidades de transición P y la función de recompensa R. En el capítulo 6 vas a dejar esa suposición atrás y vas a aprender a evaluar y mejorar políticas sin conocer el modelo, solo a partir de la experiencia: Monte Carlo y Diferencias Temporales.
