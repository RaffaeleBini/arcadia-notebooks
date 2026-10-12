# Capítulo 12: Arquitecturas de agentes: de A3C a IMPALA

## Por qué te importa este tema

En el capítulo anterior viste cómo Rainbow combina varias mejoras sobre la Red Q Profunda (DQN, del inglés Deep Q-Network) y cómo la perspectiva distribucional cambia la forma de estimar el valor de una acción. Todo eso responde a una pregunta: ¿cómo hacer que un solo agente aprenda mejor? Este capítulo responde a una pregunta distinta y, para entornos complejos a gran escala, igual de importante: ¿cómo hacer que muchos agentes aprendan más rápido, juntos?

Piensa en los casos de uso que vas a ver en el capítulo siguiente y último del manual: sistemas capaces de jugar Go, ajedrez o StarCraft a un nivel sobrehumano. Ninguno de ellos se entrenó con un único agente en una única máquina durante semanas. Se entrenaron con cientos o miles de copias del agente jugando en paralelo, acumulando experiencia a una velocidad que ninguna máquina aislada podría igualar. El salto de calidad, en ese punto, ya no viene solo de un algoritmo más inteligente, sino de una arquitectura capaz de repartir el trabajo. Este capítulo te enseña esa arquitectura: primero el Actor-Crítico de Ventaja Asíncrono (A3C, del inglés Asynchronous Advantage Actor-Critic), que sustituye la memoria de repetición por el paralelismo, y después IMPALA, que lleva esa misma idea de un ordenador a un clúster completo.

## El cuello de botella que deja atrás la Red Q Profunda

Recuerda los dos problemas de correlación que obligaron a la Red Q Profunda a incorporar una memoria de repetición y una red objetivo: los datos que genera un agente jugando de forma continua están muy correlacionados entre pasos consecutivos, y una red neuronal —por su tendencia natural a producir funciones suaves— asigna valores parecidos a estados parecidos, lo que hace que el objetivo de entrenamiento se persiga a sí mismo. La memoria de repetición resuelve ambos problemas guardando hasta un millón de fotogramas y muestreando lotes aleatorios, pero a cambio introduce una dependencia estricta del aprendizaje fuera de política y deja sin aprovechar buena parte de la capacidad de una tarjeta gráfica: las redes pequeñas que funcionan bien en este esquema no llegan a saturar su rendimiento, así que entrenar un solo juego de Atari podía llevar hasta una semana en una unidad de procesamiento gráfico (GPU).

La pregunta que abrió la puerta a A3C fue directa: si el problema de fondo es la correlación de los datos, ¿hace falta almacenar experiencia pasada para romperla, o basta con generar experiencia diversa en el momento?

## A3C: sustituye la memoria por el paralelismo

Imagina que, en vez de una sola persona jugando una partida tras otra y anotando cada jugada en un cuaderno para repasarlo más tarde, pones a 16 personas jugando partidas distintas al mismo tiempo, cada una en su propio tablero. Cada cierto tiempo, cada jugadora consulta la estrategia común más reciente, juega unos cuantos turnos con ella, calcula qué ajuste haría mejor esa estrategia y lo susurra a un cuaderno compartido, sin esperar turno ni pedir permiso a las demás. Como todas juegan situaciones distintas a la vez, ningún ajuste individual arrastra al cuaderno común hacia un rincón sesgado: la diversidad simultánea entre jugadoras cumple el mismo papel que antes cumplía la diversidad temporal de la memoria de repetición, solo que sin guardar nada.

Así funciona A3C. Cada trabajador —un hilo de proceso con su propia copia del entorno— repite este ciclo: toma los parámetros más recientes de la red compartida, juega un número fijo de pasos con la política actual, calcula un gradiente y lo aplica de forma asíncrona a esa red compartida, sin bloquear a los demás trabajadores ni sincronizarse con ellos. No hay memoria de repetición ni red objetivo: la propia concurrencia de trabajadores explorando puntos distintos del entorno estabiliza el aprendizaje.

El componente que aprende dentro de cada trabajador es un Actor-Crítico de Ventaja (A2C, del inglés Advantage Actor-Critic), la misma familia de algoritmos que trabajaste en el capítulo 9: una red que predice, a la vez, una distribución de probabilidad sobre acciones (el actor) y una estimación del valor del estado (el crítico). El actor trunca el retorno a los n pasos siguientes, usa el crítico para completar el resto por bootstrapping y resta el valor del estado actual como referencia; a esa diferencia se le llama ventaja, y es lo que multiplica al gradiente de la política. Hay una razón adicional por la que este esquema es más estable que ajustar valores Q sueltos: la salida del actor es una distribución que suma uno, así que aumentar la probabilidad de una acción reduce automáticamente la de las demás. Es una forma de estabilización numérica que viene incorporada en el propio diseño, algo que un valor Q sin restricciones no tiene.

Los experimentos de escalado con 16 núcleos de CPU y 16 trabajadores frente a una Red Q Profunda con GPU, en siete juegos de Atari, dejan cifras concretas: el actor-crítico escaló de forma sublineal (entre 12 y 13 veces más rápido con 16 hilos, frente a las 16 veces ideales), mientras que las variantes de un solo paso escalaron de forma superlineal (unas 20 veces más rápido), gracias al efecto estabilizador de tener varios trabajadores explorando a la vez. En Breakout, un solo trabajador con 50 millones de fotogramas llegaba a unos 250 puntos; los mismos 50 millones de fotogramas repartidos entre 16 trabajadores llegaban a unos 300 puntos, por el efecto de promediar sobre una porción mayor del espacio de estados. Y pese a que las variantes de un paso escalaban mejor en velocidad bruta, A3C seguía siendo, en conjunto, la opción más eficiente en datos.

El resultado práctico: A3C igualaba o superaba a la Red Q Profunda en Atari en un solo día, en una sola máquina, sin ninguna GPU; con cuatro días de entrenamiento, mejoraba todavía más. Funciona tanto con redes recurrentes como con redes hacia delante, fue uno de los primeros métodos capaces de aprender a navegar laberintos tridimensionales aleatorios a partir de píxeles, y se adapta a control continuo sustituyendo la distribución sobre acciones discretas por la media y la varianza de una distribución gaussiana. Tiene también límites explícitos: puede ser menos estable que la Red Q Profunda, a veces necesita ajustar hiperparámetros juego a juego, aprovecha peor la GPU y, sobre todo, sigue atado a los hilos de una única máquina. Ese último límite es justo el que resuelve IMPALA.

## Consejos prácticos de ingeniería

Antes de escalar a algo como A3C, el propio material de referencia insiste en un principio de trabajo que conviene interiorizar cueste lo que cueste: depura primero en pequeño.

- **Empieza con un problema de juguete**: elige un entorno mínimo que comparta las características clave del problema real (asignación de crédito a largo plazo, tamaño del espacio de entrada) pero que puedas resolver en minutos en tu portátil. El ejemplo recomendado es "Catch": una pelota cae desde arriba de una cuadrícula pequeña —por ejemplo, cinco columnas por diez filas— y controlas una pala de dos o tres píxeles de ancho; recibes más uno si la atrapas y menos uno si falla. Puedes subir la dificultad poco a poco (ángulos, rebotes, una pala más estrecha) para encontrar exactamente el punto donde tu algoritmo empieza a fallar
- **Visualízalo todo**: curvas de entrenamiento, comportamiento de la política, predicciones del crítico. Una tendencia que no se ve en una tabla de números salta a la vista en una gráfica
- **Desconfía del diseño de recompensas**: aunque en teoría la recompensa forma parte de la definición del problema, en la práctica se ajusta constantemente (por ejemplo, recortar la recompensa entre menos uno y uno en Atari). El riesgo real está en sobreespecificar recompensas intermedias: en una tarea robótica de alcanzar, mover y soltar un objeto, premiar demasiado los pasos intermedios puede enseñar al robot a alcanzar y mover el objeto, pero nunca a soltarlo, porque sostenerlo ya le resulta la política más estable y mejor recompensada
- **Trata el factor de descuento como un hiperparámetro más**: aunque el objetivo final no dependa de él en teoría, descontar reduce la varianza de tus estimaciones de retorno y puede encajar mejor con el sesgo inductivo de tu optimizador
- **Ajusta el tamaño de red con cuidado**: empieza pequeño porque entrena más rápido, pero prueba redes más grandes de vez en cuando. Las primeras versiones de la Red Q Profunda en Seaquest sufrían con redes pequeñas, lo que motivó la propia red objetivo; con redes más grandes, la red objetivo resulta menos crítica, porque una red grande confunde menos estados parecidos entre sí
- **No inicialices la política de forma determinista**: desde el primer paso, debe muestrear acciones con cierta probabilidad, no repetir siempre la misma
- **Mantente al día con el aprendizaje supervisado**: RMSprop y Adam son los optimizadores de referencia en RL profundo; técnicas como el *dropout* merecen probarse, aunque la normalización por lotes todavía no está muy extendida en este campo

## IMPALA: de un ordenador a un clúster, con V-trace

A3C resuelve el problema dentro de una sola máquina, pero su arquitectura no crece bien más allá de eso: si simplemente repartes varios aprendices —cada uno con su propia copia de los pesos— entre distintas máquinas y les pides que sincronicen gradientes contra un servidor central, aparece un problema nuevo. Para cuando el gradiente de una máquina llega al servidor, otras máquinas ya lo han actualizado, así que ese gradiente llega desactualizado y termina empujando la red en una dirección que ya no corresponde al estado actual de los pesos.

La solución de IMPALA (del inglés Importance Weighted Actor-Learner Architecture) es cambiar qué se distribuye. En vez de repartir aprendices, centraliza el aprendizaje en una sola copia —normalmente sobre GPU— y distribuye solo la parte de actuar: decenas o cientos de actores en CPU recorren el entorno con la política que tienen a mano y envían su experiencia al aprendiz central. Esto evita el problema del gradiente obsoleto, pero deja un problema hermano: para cuando la experiencia de un actor llega al aprendiz, los pesos de este ya pueden haber cambiado, así que esa experiencia fue generada por una política ligeramente distinta de la actual.

Aquí entra V-trace, el método de corrección fuera de política que da nombre a la arquitectura. Usa muestreo por importancia para reponderar cada actualización según la diferencia entre la política de comportamiento μ del actor que generó los datos y la política objetivo π del aprendiz en el momento de usarlos. Si ambas coinciden, la actualización de V-trace se reduce exactamente a la actualización de A3C; cuanto más se separan, más peso pierde esa experiencia en el entrenamiento. Es un compromiso explícito entre sesgo y varianza: V-trace introduce algo de sesgo a cambio de reducir la varianza y mantener el entrenamiento estable, incluso con datos ligeramente desactualizados.

Las cifras de rendimiento en entornos tridimensionales muestran por qué merece la pena este rediseño. A3C en una sola máquina, con 64 CPU y 32 hilos, procesa entre 6.000 y 9.000 fotogramas por segundo; A3C distribuido con 200 trabajadores llega hasta unos 50.000 fotogramas por segundo; IMPALA alcanza entre 200.000 y 250.000 fotogramas por segundo.

Pero la cifra más interesante no es de velocidad, sino de transferencia entre tareas. En el conjunto de 30 tareas cognitivas de DeepMind Lab (DMLab-30), entrenar una red por separado en cada tarea y entrenarla en las 30 a la vez son experimentos distintos. Con A3C y una red profunda, el rendimiento medio ronda el 30% del nivel humano. IMPALA con una red pequeña ya supera esa cifra; IMPALA con una red profunda entrenada juego a juego mejora todavía más; pero el mejor resultado de todos es IMPALA con una red residual profunda entrenada en las 30 tareas simultáneamente con una única red, por encima incluso de la versión entrenada juego a juego por separado. Esa es la señal de transferencia positiva: aprender varias tareas a la vez, con la arquitectura adecuada, hace que cada tarea individual se aprenda mejor y con menos datos, no peor. En tiempo de entrenamiento, IMPALA con red profunda alcanza el nivel de A3C en menos de un día; A3C necesita alrededor de una semana para llegar ahí.

En el conjunto de 57 juegos de Atari, la distancia se repite: A3C distribuido entrenado por separado en cada juego llega a un 54% de puntuación media con red pequeña y a un 118% con red profunda; IMPALA supera a A3C en ambos tamaños de red, y con una red residual profunda entrenada por separado se acerca al 200% de puntuación media. El dato más llamativo, sin embargo, es otro: IMPALA entrenado en los 57 juegos a la vez, con una sola red, alcanza en torno al 60% de puntuación media —comparable a la Red Q Profunda original entrenada juego por juego— pese a que la diversidad visual y de mecánicas entre 57 juegos de Atari es mucho mayor que entre las 30 tareas de DeepMind Lab.

Queda una pregunta abierta: si IMPALA es más rápido y transfiere mejor entre tareas, ¿por qué la Red Q Profunda sigue siendo, en algunos aspectos, más estable que A3C? La respuesta apunta al entrenamiento por lotes: muestrear un lote aleatorio de la memoria de repetición es, en sí mismo, un ingrediente estabilizador importante, y hasta pequeñas desviaciones respecto a la política actual afectan al actor-crítico. V-trace compensa buena parte de ese efecto, pero si la experiencia llega demasiado desactualizada, los pesos de importancia se acercan a cero o quedan recortados en uno, y esa experiencia deja de aportar señal de aprendizaje útil.

## Código: "Catch" como banco de pruebas para un actor-crítico

Antes de seguir, una aclaración importante: el código siguiente **no** implementa A3C ni IMPALA de verdad. A3C necesita varios procesos jugando en paralelo y actualizando una red compartida de forma asíncrona; IMPALA necesita, además, actores y aprendiz repartidos en máquinas distintas con corrección V-trace entre ellos. Nada de eso es realista dentro de un único notebook de Colab. Lo que sí puedes hacer —y es justo lo que recomienda el material de referencia antes de dar el salto al paralelismo— es construir el banco de pruebas "Catch" descrito en la sección anterior y entrenar sobre él un actor-crítico simple de un solo proceso, con la misma idea que viste en el capítulo 9. Sirve para comprobar que la lógica del algoritmo funciona antes de complicar la arquitectura.

El código solo necesita `numpy`, `torch` y `matplotlib`, las tres preinstaladas en Colab.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. El entorno "Catch": una pelota cae, una pala debe atraparla
# ---------------------------------------------------------------------
# Cuadrícula pequeña tal como la describe el material de referencia: la
# pelota aparece en una columna aleatoria de la fila superior y baja una
# fila por paso. Controlas una pala de dos píxeles de ancho en la fila
# inferior, que puedes mover a la izquierda, a la derecha o dejar quieta.
# Cuando la pelota llega a la fila inferior, recibes +1 si la pala la
# cubre y -1 si no.

FILAS = 10
COLUMNAS = 5
ANCHO_PALA = 2
N_ACCIONES = 3  # 0 = izquierda, 1 = quieta, 2 = derecha


class Catch:
    """Implementación mínima del juego de juguete "Catch"."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.fila_pelota = 0
        self.columna_pelota = np.random.randint(COLUMNAS)
        # Punto de partida aleatorio de la pala, sin salirse del borde
        self.columna_pala = np.random.randint(COLUMNAS - ANCHO_PALA + 1)
        return self._estado()

    def _estado(self):
        """Codifica el estado como una imagen binaria aplanada: un uno
        donde está la pelota y unos donde ocupa la pala."""
        grid = np.zeros((FILAS, COLUMNAS), dtype=np.float32)
        grid[self.fila_pelota, self.columna_pelota] = 1.0
        grid[FILAS - 1, self.columna_pala:self.columna_pala + ANCHO_PALA] = 1.0
        return grid.flatten()

    def step(self, accion):
        # Mueve la pala; np.clip evita que se salga de la cuadrícula
        desplazamiento = {0: -1, 1: 0, 2: 1}[accion]
        self.columna_pala = int(
            np.clip(self.columna_pala + desplazamiento, 0, COLUMNAS - ANCHO_PALA)
        )
        # La pelota cae una fila en cada paso
        self.fila_pelota += 1
        terminado = self.fila_pelota == FILAS - 1
        if terminado:
            atrapada = (
                self.columna_pala
                <= self.columna_pelota
                < self.columna_pala + ANCHO_PALA
            )
            recompensa = 1.0 if atrapada else -1.0
        else:
            recompensa = 0.0
        return self._estado(), recompensa, terminado


# ---------------------------------------------------------------------
# 2. Red actor-crítico: una base compartida y dos cabezas de salida
# ---------------------------------------------------------------------
class ActorCritico(nn.Module):
    """Red pequeña con una base compartida: una cabeza produce la
    política (probabilidad de cada acción) y otra el valor del estado."""

    def __init__(self, dim_entrada, n_acciones, dim_oculta=128):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(dim_entrada, dim_oculta),
            nn.ReLU(),
        )
        self.cabeza_politica = nn.Linear(dim_oculta, n_acciones)
        self.cabeza_valor = nn.Linear(dim_oculta, 1)

    def forward(self, x):
        h = self.base(x)
        return self.cabeza_politica(h), self.cabeza_valor(h)


# ---------------------------------------------------------------------
# 3. Un episodio completo y la actualización del actor-crítico
# ---------------------------------------------------------------------
GAMMA = 0.99
TASA_APRENDIZAJE = 1e-3
N_EPISODIOS = 3000
COEF_VALOR = 0.5
COEF_ENTROPIA = 0.01  # fomenta la exploración: la política nunca debe
                       # arrancar (ni quedarse) determinista

torch.manual_seed(0)
np.random.seed(0)

entorno = Catch()
red = ActorCritico(dim_entrada=FILAS * COLUMNAS, n_acciones=N_ACCIONES)
optimizador = torch.optim.Adam(red.parameters(), lr=TASA_APRENDIZAJE)


def ejecutar_episodio():
    """Juega un episodio completo de Catch y devuelve, paso a paso, el
    log-probabilidad de la acción elegida, el valor estimado del
    estado, la recompensa obtenida y la entropía de la política."""
    estado = entorno.reset()
    log_probs, valores, recompensas, entropias = [], [], [], []
    terminado = False
    while not terminado:
        entrada = torch.from_numpy(estado)
        logits, valor = red(entrada)
        distribucion = torch.distributions.Categorical(logits=logits)
        accion = distribucion.sample()

        estado, recompensa, terminado = entorno.step(accion.item())

        log_probs.append(distribucion.log_prob(accion))
        entropias.append(distribucion.entropy())
        valores.append(valor.squeeze())
        recompensas.append(recompensa)

    return log_probs, valores, recompensas, entropias


def actualizar_red(log_probs, valores, recompensas, entropias):
    """Calcula el retorno descontado en cada paso, la ventaja frente al
    valor estimado por el crítico, y actualiza actor y crítico a la vez."""
    retornos = []
    G = 0.0
    for r in reversed(recompensas):
        G = r + GAMMA * G
        retornos.insert(0, G)
    retornos = torch.tensor(retornos, dtype=torch.float32)

    valores = torch.stack(valores)
    ventaja = retornos - valores.detach()  # sin gradiente hacia el crítico aquí

    perdida_actor = -(torch.stack(log_probs) * ventaja).mean()
    perdida_critico = F.mse_loss(valores, retornos)
    perdida_entropia = torch.stack(entropias).mean()

    perdida = (
        perdida_actor
        + COEF_VALOR * perdida_critico
        - COEF_ENTROPIA * perdida_entropia
    )

    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()


# ---------------------------------------------------------------------
# 4. Bucle de entrenamiento y seguimiento de la tasa de aciertos
# ---------------------------------------------------------------------
aciertos = []  # 1 si el episodio terminó en captura, 0 si terminó en fallo

for episodio in range(N_EPISODIOS):
    log_probs, valores, recompensas, entropias = ejecutar_episodio()
    actualizar_red(log_probs, valores, recompensas, entropias)
    aciertos.append(1 if recompensas[-1] > 0 else 0)

# Media móvil de 100 episodios: suaviza el ruido episodio a episodio y
# deja ver la tendencia real de aprendizaje
VENTANA = 100
tasa_movil = np.convolve(aciertos, np.ones(VENTANA) / VENTANA, mode="valid")

plt.figure(figsize=(8, 5))
plt.plot(tasa_movil)
plt.xlabel("Episodios")
plt.ylabel(f"Tasa de aciertos (media móvil de {VENTANA} episodios)")
plt.title('Actor-crítico simple aprendiendo a jugar "Catch"')
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.show()
```

Al ejecutarlo verás la tasa de aciertos partir de cerca del azar —con tres acciones y dos posiciones de pala en una cuadrícula de cinco columnas, el punto de partida ronda el 40%— y subir de forma sostenida hasta acercarse a capturas casi constantes conforme la red aprende a alinear la pala con la columna de la pelota. Si la curva se estanca, prueba a subir `COEF_ENTROPIA` (la política deja de explorar demasiado pronto) o a bajar `TASA_APRENDIZAJE` (las actualizaciones son demasiado bruscas para un problema tan pequeño).

## Puntos clave

- La Red Q Profunda resuelve la correlación de sus datos con una memoria de repetición y una red objetivo, pero eso la ata al aprendizaje fuera de política y desaprovecha la GPU
- A3C sustituye la memoria por el paralelismo: muchos trabajadores con su propia copia del entorno actualizan de forma asíncrona una red compartida, sin bloquearse entre sí
- La diversidad simultánea entre trabajadores rompe la correlación de los datos igual que antes lo hacía la diversidad temporal de la memoria de repetición, sin necesidad de almacenar nada
- El componente que aprende dentro de cada trabajador es un actor-crítico de ventaja: una política normalizada (que suma uno) más un crítico que sirve de referencia
- IMPALA centraliza el aprendizaje en una sola copia y distribuye solo la actuación entre muchos actores, y usa V-trace para corregir el desfase entre la política que generó los datos y la política actual del aprendiz
- IMPALA no solo es más rápido: entrenar varias tareas a la vez con la arquitectura adecuada produce transferencia positiva, mejorando el rendimiento en cada tarea individual
- El juego de juguete "Catch" es el banco de pruebas recomendado para depurar la lógica de un agente antes de complicar la arquitectura con paralelismo
- El diseño de recompensas, el factor de descuento, el tamaño de red y la inicialización no determinista de la política son palancas prácticas tan importantes como el algoritmo elegido

## Ponlo en práctica

Coge el código de "Catch" y haz este experimento por tu cuenta:

- **Cambia la dificultad del entorno**: reduce `ANCHO_PALA` a uno o amplía `COLUMNAS` a ocho o diez, sin tocar nada más. Vuelve a entrenar y compara cuántos episodios necesita el agente ahora para alcanzar la misma tasa de aciertos que antes
- **Observa dónde se rompe el aprendizaje**: sigue subiendo `COLUMNAS` o bajando `ANCHO_PALA` hasta encontrar el punto en el que la curva deja de subir de forma clara en 3.000 episodios

¡Anota en qué configuración exacta empieza a fallar tu agente y qué cambio —más episodios, más entropía, una red más grande— lo vuelve a arreglar! Ese ejercicio de tantear los límites de un problema pequeño es, palabra por palabra, el consejo práctico central de este capítulo.

---

Ya conoces las piezas: cómo estabilizar una red con memoria de repetición y red objetivo, cómo sustituir esa memoria por paralelismo con A3C, y cómo escalar ese paralelismo a un clúster completo con IMPALA. En el capítulo 13, el último del manual, verás cómo estas ideas se combinaron —junto con búsqueda en árbol y autojuego— para construir sistemas como AlphaGo, AlphaZero y AlphaStar.
