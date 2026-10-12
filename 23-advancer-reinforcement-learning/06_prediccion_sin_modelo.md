# Capítulo 6: Predicción sin modelo — Monte Carlo y Diferencias Temporales

## Por qué te importa este tema

En el capítulo anterior viste la Programación Dinámica (DP): un conjunto de métodos potentes, pero con una condición muy exigente. Para aplicarlos necesitas conocer de antemano el modelo completo del entorno, es decir, las probabilidades de transición entre estados y la función de recompensa. En un problema de tablero pequeño esa exigencia es razonable. En un caso real —optimizar el riego de un cultivo, priorizar el mantenimiento de una flota pesquera, o ajustar un protocolo de cuidados— casi nunca dispones de ese modelo exacto. El entorno es demasiado complejo, cambia con el tiempo o simplemente nadie lo ha medido con precisión.

Aquí entra el aprendizaje **sin modelo** (model-free): en vez de calcular el valor de cada estado a partir de ecuaciones conocidas, lo estimas a partir de la experiencia, episodio tras episodio, como haría cualquier persona que aprende un oficio por prueba y observación. Este capítulo te enseña las dos familias de métodos que hacen eso: Monte Carlo (MC) y las Diferencias Temporales (TD, del inglés Temporal-Difference). Ambas resuelven el mismo problema —la **predicción**: estimar la función de valor de una política fija, Vπ(s), sin conocer el modelo— pero llegan ahí por caminos muy distintos, con consecuencias prácticas importantes que vas a ver en código al final del capítulo.

## Qué significa "predicción" cuando no tienes el modelo

Recuerda que Vπ(s) representa el retorno esperado si sigues la política π a partir del estado s. Con DP calculabas ese valor resolviendo ecuaciones sobre el modelo. Sin modelo, la única fuente de información que te queda es la experiencia real: dejas que el agente actúe según π, observas qué recompensas recibe y con esos datos estimas Vπ(s). Es la diferencia entre calcular la duración media de un trayecto con un mapa de tráfico perfecto y estimarla cronometrando el trayecto muchas veces.

## Monte Carlo: aprende del episodio completo

El método MC es el más directo de los dos: deja que el episodio termine, calcula el retorno total obtenido desde cada estado visitado y usa ese retorno como muestra del valor real. Repite esto con muchos episodios y promedia. Por la ley de los grandes números, ese promedio converge al verdadero Vπ(s).

Imagina que conduces cada tarde desde el trabajo hasta casa y quieres estimar cuánto tarda ese trayecto. El enfoque Monte Carlo es esperar a llegar a casa, anotar el tiempo total y usarlo para actualizar tu estimación del trayecto completo. Cuantos más días repitas el trayecto, mejor será tu estimación, pero necesitas llegar a casa cada vez para aprender algo: si el episodio no termina, MC no tiene nada que promediar.

Dentro de MC existen dos variantes, según cuántas veces cuentes cada estado dentro de un mismo episodio:

- **Primera visita** (first-visit): solo usas el retorno a partir de la primera vez que aparece un estado en el episodio, aunque ese estado se repita más adelante
- **Cada visita** (every-visit): usas el retorno a partir de cada ocurrencia del estado, no solo la primera

Ambas convergen al mismo valor con suficientes episodios; la diferencia está en cómo tratan los estados que se repiten dentro de un mismo episodio.

## Diferencias Temporales: adivina a partir de tu propia adivinanza

TD(0) resuelve el mismo problema con una idea distinta: no esperes al final del episodio, actualiza tu estimación después de cada paso. Para eso combina la recompensa que acabas de observar con tu propia estimación actual del valor del siguiente estado, en lugar de esperar el retorno real completo. A esta técnica se le llama **bootstrapping**: usas una estimación para corregir otra estimación.

Siguiendo la analogía del trayecto a casa: en vez de esperar a llegar para juzgar si el viaje va bien o mal, ajustas tu previsión en cada tramo. Al salir de la oficina bajo la lluvia, revisas al alza tu estimación de duración total. Al incorporarte a la autopista sin atascos, la revisas a la baja. No necesitas llegar a casa para aprender algo útil en cada tramo del camino.

La regla de actualización de TD(0) es esta:

V(S) ← V(S) + α · [R + γ·V(S') − V(S)]

Donde α es la tasa de aprendizaje, R es la recompensa observada al dar el paso, γ es el factor de descuento y V(S') es tu estimación actual del siguiente estado (cero si S' es un estado terminal). El término entre corchetes se llama **error TD**: la diferencia entre lo que acabas de observar más lo que crees que vale el resto del camino, y lo que creías antes de dar el paso.

Como TD no necesita esperar al final, funciona incluso con episodios incompletos o tareas que no terminan nunca, algo que MC no puede hacer.

## Sesgo y varianza: la gran disyuntiva

Esta diferencia de mecánica tiene una consecuencia estadística directa:

- MC promedia retornos completos y reales, así que su estimación es **no sesgada**: en promedio acierta el valor verdadero. El precio es una **varianza alta**, porque cada retorno depende de toda la secuencia de acciones y recompensas aleatorias que ocurrieron hasta el final del episodio
- TD actualiza con una sola recompensa observada más una estimación todavía imperfecta del resto, así que es **sesgado** mientras esa estimación no haya convergido. A cambio tiene **varianza baja**, porque cada actualización depende solo de un paso, no de todo el episodio

Con infinitos episodios y una tasa de aprendizaje que decae adecuadamente, ambos métodos convergen al mismo Vπ(s). Con datos finitos —que es la situación real casi siempre— pueden dar respuestas distintas, y suele compensar aceptar algo de sesgo a cambio de mucha menos varianza. Por eso TD se usa tanto en la práctica: aprende más rápido y con estimaciones más estables, aunque tarde un poco en "quitarse" el sesgo inicial.

## El caso de estudio: el paseo aleatorio

Para comparar ambos métodos en código vas a usar el **paseo aleatorio** (random walk), un entorno mínimo pero muy ilustrativo: una cadena de cinco estados no terminales, con un estado terminal a cada extremo. Desde el estado central, en cada paso te mueves a la izquierda o a la derecha con la misma probabilidad. Si llegas al extremo derecho recibes una recompensa de uno; si llegas al extremo izquierdo, recibes cero. Es un entorno tan simple que puedes calcular el valor real de cada estado a mano, lo que te permite medir con precisión qué tan rápido converge cada método.

## Código: Monte Carlo vs. TD(0) en el paseo aleatorio

El siguiente código es autocontenido: solo necesita `numpy` y `matplotlib`, y puedes ejecutarlo tal cual en Google Colab.

```python
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Definición del entorno: paseo aleatorio con cinco estados internos
# ---------------------------------------------------------------------
# Estados 0 y 6 son terminales (izquierda y derecha).
# Estados 1-5 son A, B, C, D, E; el episodio siempre arranca en C (estado 3).
N_ESTADOS_INTERNOS = 5
ESTADO_INICIAL = 3          # C, el estado central de la cadena
TERMINAL_IZQUIERDA = 0
TERMINAL_DERECHA = N_ESTADOS_INTERNOS + 1

# Valor real de cada estado interno (calculable a mano en este entorno
# tan simple): V(estado i) = i / (N_ESTADOS_INTERNOS + 1)
VALORES_REALES = np.array(
    [i / (N_ESTADOS_INTERNOS + 1) for i in range(1, N_ESTADOS_INTERNOS + 1)]
)


def generar_episodio():
    """Simula un episodio de paseo aleatorio y devuelve la lista de
    (estado, recompensa) que atraviesa, terminando en el estado terminal."""
    estado = ESTADO_INICIAL
    trayectoria = []
    while True:
        # Con probabilidad 0.5 te mueves a la izquierda o a la derecha
        paso = -1 if np.random.rand() < 0.5 else 1
        siguiente_estado = estado + paso
        # Solo el extremo derecho da recompensa; el resto de pasos dan cero
        recompensa = 1.0 if siguiente_estado == TERMINAL_DERECHA else 0.0
        trayectoria.append((estado, recompensa))
        estado = siguiente_estado
        if estado in (TERMINAL_IZQUIERDA, TERMINAL_DERECHA):
            break
    return trayectoria


# ---------------------------------------------------------------------
# 2. Predicción por Monte Carlo (primera visita)
# ---------------------------------------------------------------------
def monte_carlo_prediccion(n_episodios, alpha, gamma=1.0):
    """Estima V para los estados internos 1..5 usando MC de primera visita
    con actualización incremental V(s) <- V(s) + alpha * (G - V(s))."""
    V = np.full(N_ESTADOS_INTERNOS, 0.5)  # inicialización neutra
    historial_rmse = []

    for _ in range(n_episodios):
        trayectoria = generar_episodio()

        # Calcula el retorno G_t desde cada paso hacia el final del episodio
        G = 0.0
        retornos_por_estado = {}
        for estado, recompensa in reversed(trayectoria):
            G = recompensa + gamma * G
            # Primera visita: si el estado se repite, nos quedamos con el
            # retorno calculado en la ocurrencia más temprana (la última
            # que procesamos al recorrer la trayectoria al revés)
            retornos_por_estado[estado] = G

        for estado, G in retornos_por_estado.items():
            if estado not in (TERMINAL_IZQUIERDA, TERMINAL_DERECHA):
                idx = estado - 1  # los estados internos van de 1 a 5
                V[idx] += alpha * (G - V[idx])

        rmse = np.sqrt(np.mean((V - VALORES_REALES) ** 2))
        historial_rmse.append(rmse)

    return V, historial_rmse


# ---------------------------------------------------------------------
# 3. Predicción por TD(0)
# ---------------------------------------------------------------------
def td0_prediccion(n_episodios, alpha, gamma=1.0):
    """Estima V para los estados internos 1..5 con TD(0): actualiza en
    cada paso, sin esperar a que el episodio termine."""
    V = np.full(N_ESTADOS_INTERNOS, 0.5)
    historial_rmse = []

    for _ in range(n_episodios):
        estado = ESTADO_INICIAL
        while estado not in (TERMINAL_IZQUIERDA, TERMINAL_DERECHA):
            paso = -1 if np.random.rand() < 0.5 else 1
            siguiente_estado = estado + paso
            recompensa = 1.0 if siguiente_estado == TERMINAL_DERECHA else 0.0

            # Valor del siguiente estado: cero si es terminal, si no, la
            # estimación actual (esto es el "bootstrapping")
            if siguiente_estado in (TERMINAL_IZQUIERDA, TERMINAL_DERECHA):
                v_siguiente = 0.0
            else:
                v_siguiente = V[siguiente_estado - 1]

            idx = estado - 1
            error_td = recompensa + gamma * v_siguiente - V[idx]
            V[idx] += alpha * error_td

            estado = siguiente_estado

        rmse = np.sqrt(np.mean((V - VALORES_REALES) ** 2))
        historial_rmse.append(rmse)

    return V, historial_rmse


# ---------------------------------------------------------------------
# 4. Comparación: velocidad de convergencia de MC vs TD(0)
# ---------------------------------------------------------------------
np.random.seed(0)
N_EPISODIOS = 150
ALPHA = 0.1

_, rmse_mc = monte_carlo_prediccion(N_EPISODIOS, ALPHA)
_, rmse_td = td0_prediccion(N_EPISODIOS, ALPHA)

plt.figure(figsize=(8, 5))
plt.plot(rmse_mc, label="Monte Carlo (primera visita)")
plt.plot(rmse_td, label="TD(0)")
plt.xlabel("Episodios")
plt.ylabel("RMSE frente al valor real")
plt.title("Convergencia de MC vs TD(0) en el paseo aleatorio (α = 0.1)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Al ejecutarlo verás típicamente que TD(0) reduce el error más rápido durante los primeros episodios que MC, gracias a su varianza más baja, aunque ambos acaban acercándose al mismo resultado a medida que aumentan los episodios. Es la teoría de sesgo y varianza que viste antes, hecha gráfica.

## Puntos clave

- Predicción sin modelo significa estimar Vπ(s) solo a partir de experiencia, sin conocer las probabilidades de transición ni la función de recompensa
- Monte Carlo promedia retornos completos de episodios; no hace bootstrapping y necesita que el episodio termine
- Primera visita (first-visit) usa solo la primera ocurrencia de un estado por episodio; cada visita (every-visit) usa todas
- TD(0) actualiza tras cada paso usando la recompensa observada más tu propia estimación del siguiente estado: esto es bootstrapping
- MC es no sesgado pero tiene varianza alta; TD es sesgado al principio pero tiene varianza baja, y ambos convergen con suficiente experiencia
- El paseo aleatorio es un entorno mínimo perfecto para medir esta diferencia de forma cuantitativa, con el error RMSE frente al valor real

## Ponlo en práctica

Coge el código anterior y haz estos dos experimentos por tu cuenta:

- **Cambia la tasa de aprendizaje**: prueba con α = 0.01, α = 0.05 y α = 0.15 en ambos métodos. Observa cómo un α demasiado alto acelera el aprendizaje inicial pero deja más ruido residual, mientras que un α demasiado bajo converge de forma más suave pero mucho más lenta
- **Aumenta el número de episodios** a 500 o 1.000 y comprueba que, con suficiente experiencia, la curva de TD(0) y la de Monte Carlo terminan acercándose al mismo error mínimo

¡Ejecuta ambos experimentos y anota en qué punto exacto las dos curvas se cruzan! Esa observación es la mejor forma de interiorizar el balance entre sesgo y varianza antes de seguir avanzando.

---

Ya sabes estimar el valor de una política sin conocer el modelo. En el capítulo 7 das el siguiente paso: usar esa misma idea de Diferencias Temporales para mejorar la política, no solo evaluarla, con dos algoritmos de control clásicos —SARSA y Q-learning— sobre el entorno de Cliff Walking.
