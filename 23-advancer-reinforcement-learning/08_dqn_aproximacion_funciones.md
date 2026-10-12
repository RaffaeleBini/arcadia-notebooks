# Capítulo 8. Aproximación de funciones y Deep Q-Networks (DQN)

Piensa en un tablero de ajedrez. Cada disposición posible de las piezas es un estado distinto, y el número total de estados supera con creces la cantidad de átomos del universo observable. O piensa en un videojuego: cada fotograma de píxeles en pantalla es, en la práctica, un estado nuevo. En el capítulo anterior aprendiste Q-learning tabular, una técnica de Aprendizaje por Refuerzo (RL, por sus siglas en inglés) que guarda un valor exacto para cada par estado-acción en una tabla. Esa tabla funciona de maravilla en problemas pequeños, pero se rompe en cuanto el número de estados se dispara: no cabe en memoria y, sobre todo, tu agente casi nunca vuelve a visitar exactamente el mismo estado dos veces. Sin una tabla que lo recuerde, no aprende nada útil sobre un estado que ve una sola vez.

En este capítulo vas a resolver ese problema con una idea sencilla: en lugar de guardar un valor por cada estado, entrena una red neuronal que aprenda a generalizar, es decir, que estime el valor de un estado nuevo a partir de lo que ya aprendió en estados parecidos. Esta idea es la puerta de entrada al Aprendizaje por Refuerzo Profundo (Deep Reinforcement Learning), la familia de métodos que combina RL con redes neuronales de varias capas. Vas a conocer Deep Q-Network (DQN), el algoritmo que en 2015 aprendió a jugar decenas de videojuegos de Atari directamente a partir de los píxeles de la pantalla, sin que nadie le explicara las reglas de cada juego.

## De tablas a funciones: la idea de la aproximación de funciones

En RL, un agente aprende siempre alguna de estas tres cosas: una política (qué acción tomar en cada estado), una función de valor (cuánto vale cada estado o cada par estado-acción) o un modelo (a qué estado te lleva cada acción). Las tres son, matemáticamente, funciones. La aproximación de funciones consiste en representar esa función con un conjunto de parámetros ajustables, en vez de con una tabla exacta: `Q(s,a;θ) ≈ Q_verdadero(s,a)`, donde θ representa esos parámetros. Tu objetivo es encontrar los valores de θ que hacen que esa aproximación se acerque lo máximo posible al valor real.

Aquí conviene una aclaración breve, por si las redes neuronales todavía te resultan un territorio nuevo: una red neuronal es, en el fondo, una función con parámetros ajustables (los pesos), organizada en capas que transforman la entrada paso a paso hasta producir una salida. Entrenarla consiste en mover esos pesos, poco a poco y de forma automática, hasta que la función que calcula se parezca cada vez más a la función que quieres imitar; en tu caso, el valor de cada acción posible en cada estado.

¿Por qué merece la pena este cambio? Por dos razones. La primera es la generalización: un agente que aproxima Q con una función puede estimar el valor de un estado que nunca visitó, apoyándose en estados similares que sí conoce. La segunda es que ya no necesitas optimizar la política a la perfección en cada paso; basta con ir mejorando la aproximación de forma incremental, ciclo tras ciclo, para que la política mejore con ella.

Existen varios tipos de aproximadores: combinaciones lineales de características construidas a mano (útiles y bien entendidas, pero limitadas), árboles de decisión, métodos de vecinos más próximos y, la opción dominante hoy, las redes neuronales. Su gran ventaja es que pueden aprender directamente de señales en bruto —como los píxeles de una pantalla— sin que tengas que diseñar a mano ninguna característica. Esta idea no es nueva: en los años ochenta, Gerald Tesauro combinó aprendizaje por diferencia temporal (Temporal Difference, TD) con una red neuronal para entrenar un agente capaz de jugar al backgammon, y Chris Watkins formuló Q-learning en 1989. La intuición de unir redes neuronales con RL lleva décadas circulando; lo que ha cambiado es la capacidad de cómputo disponible, que ahora permite ponerla en práctica a gran escala.

## Por qué unir Q-learning y una red neuronal, sin más, es inestable

La versión ingenua es tentadora: sustituye la tabla de Q-learning por una red neuronal que reciba el estado y devuelva un valor por cada acción, elige la acción con épsilon-greedy, calcula el error entre lo que predijo la red y lo que "debería" haber predicho, y ajusta los pesos con descenso de gradiente. Sobre el papel funciona. En la práctica, entrenada así, la red suele oscilar sin converger o directamente diverge. Hay dos motivos.

El primero es que los datos que genera un agente de RL no son independientes ni están idénticamente distribuidos, algo que casi todo el entrenamiento de redes neuronales da por hecho. Los pasos consecutivos de un episodio están muy correlacionados entre sí: el estado en el paso cien se parece muchísimo al estado en el paso 101. Es la diferencia entre entrenar con fotos aleatorias de miles de escenas distintas y entrenar viendo, fotograma a fotograma, los mismos segundos de una única película: la red se especializa en lo que acaba de ver y olvida lo que vio antes.

El segundo motivo es la no estacionariedad que introduce el bootstrapping. En Q-learning, el objetivo (target) que persigue tu red se calcula usando la propia red: `objetivo = r + γ · max_a' Q(s',a';θ)`. Cada vez que ajustas los pesos θ para acercarte a ese objetivo, el objetivo también se mueve, porque depende de esos mismos pesos. Es como intentar dar en una diana que se desplaza cada vez que disparas, y que además se desplaza justamente porque disparaste. A eso súmale que la política que decide qué datos recoge el agente también cambia con la red, así que los datos de entrenamiento van cambiando a la vez que la propia red que intentas entrenar.

## Las dos piezas que estabilizan DQN: repetición de experiencias y red objetivo

DQN (Mnih y colaboradores, 2015) resuelve ambos problemas añadiendo dos ingredientes al Q-learning con redes neuronales.

**Buffer de repetición de experiencias (experience replay).** En vez de entrenar con la transición que el agente acaba de vivir y descartarla, guarda cada transición `(s, a, r, s')` en una memoria y, en cada paso de entrenamiento, muestrea al azar un pequeño lote de esa memoria. Esto aporta varias ventajas a la vez: rompe la correlación entre pasos consecutivos (el lote mezcla experiencias de momentos muy distintos), aprovecha cada transición varias veces en vez de una sola, y suaviza la distribución de los datos de entrenamiento porque combina experiencia recogida bajo políticas ligeramente distintas a lo largo del entrenamiento. El resultado se parece mucho más a los datos "bien portados" que espera una red neuronal entrenada con aprendizaje supervisado.

**Red objetivo (target network).** Mantén una segunda copia de la red, con sus propios parámetros θ⁻, y utilízala solo para calcular el objetivo de entrenamiento. Esta red objetivo no se actualiza en cada paso: se sincroniza con la red principal cada cierto número fijo de pasos, copiando sus pesos. Mientras tanto, permanece congelada. Así, el objetivo que persigue tu red principal deja de moverse en cada paso y se convierte en una referencia estable durante un tramo de entrenamiento, lo que reduce la oscilación y ayuda a que el entrenamiento converja.

Con ambas piezas, la función de pérdida de DQN queda así, de forma intuitiva:

`L(θ) = E [ (r + γ · max_a' Q(s',a';θ⁻) − Q(s,a;θ))² ]`

donde la esperanza se calcula sobre mini-lotes muestreados del buffer de repetición, θ son los parámetros de la red principal (la que se entrena) y θ⁻ los de la red objetivo (congelada entre sincronizaciones). Un detalle técnico importante: al calcular el gradiente, aplicas un "stop-gradient" sobre el término del objetivo, de modo que el ajuste de pesos solo fluye hacia θ, nunca hacia θ⁻. Sin ese detalle, estarías moviendo el propio objetivo para que se acerque a tu predicción, en vez de al revés.

Un matiz interesante que vale la pena recordar: la necesidad de la red objetivo no es absoluta. Cuanto mayor es la capacidad de una red (más parámetros, más neuronas), menos generaliza de forma agresiva entre estados distintos, y eso por sí mismo reduce parte de la inestabilidad. Aun así, la red objetivo sigue siendo un componente estándar en la inmensa mayoría de implementaciones de DQN, junto con un factor de descuento típico cercano a 0,99 y una exploración épsilon-greedy con decaimiento.

DQN tiene además varias mejoras posteriores que solo vas a conocer de nombre por ahora, porque las vas a implementar en el capítulo 11:

- **Double DQN** separa la elección de la mejor acción siguiente (con la red principal) de la evaluación de esa acción (con la red objetivo), para reducir el sesgo de sobreestimación típico de Q-learning
- **Dueling DQN** divide la red en dos ramas, una que estima el valor del estado y otra que estima la ventaja de cada acción sobre ese valor, y las combina al final, lo que mejora la robustez de la estimación
- **Prioritized Experience Replay** muestrea el buffer dando más peso a las transiciones con mayor error de predicción, en vez de muestrear todas con la misma probabilidad, para aprender más rápido de lo que más sorprende a la red

## Ejemplo en Python: DQN mínimo sobre CartPole

Vas a entrenar un DQN sobre CartPole, el entorno clásico en el que un carrito debe mantener un péndulo en equilibrio moviéndose a izquierda o derecha. El estado son cuatro números (posición y velocidad del carrito, ángulo y velocidad angular del péndulo) y hay dos acciones posibles. Es un entorno mucho más simple que Atari, pero contiene exactamente los mismos ingredientes: red Q, buffer de repetición y red objetivo.

Copia y ejecuta este código en Google Colab. La única dependencia externa de todo el capítulo es `gymnasium`, la librería que provee el entorno CartPole; PyTorch, `numpy` y `matplotlib` ya vienen preinstalados.

```python
# Instalamos gymnasium: es la ÚNICA dependencia externa de este capítulo.
# PyTorch, numpy y matplotlib ya vienen preinstalados en Google Colab.
!pip install gymnasium
```

```python
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym

# Semilla para que el experimento sea, en la medida de lo posible, reproducible
SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

# --- Hiperparámetros ---
# Ajustados para entrenar rápido en una CPU normal de Colab, priorizando
# claridad sobre rendimiento máximo.
NOMBRE_ENTORNO = "CartPole-v1"
NUM_EPISODIOS = 300                    # episodios de entrenamiento
GAMMA = 0.99                           # factor de descuento
TASA_APRENDIZAJE = 1e-3
TAMANO_BUFFER = 10000                  # capacidad del buffer de repetición
TAMANO_LOTE = 64                       # tamaño del mini-lote muestreado en cada paso
EPSILON_INICIAL = 1.0
EPSILON_FINAL = 0.05
EPISODIOS_DECAIMIENTO = 200            # episodios en los que epsilon baja de inicial a final
PASOS_ACTUALIZAR_RED_OBJETIVO = 500    # cada cuántos pasos se sincroniza la red objetivo
PASOS_MINIMOS_ANTES_ENTRENAR = 1000    # esperamos a tener algo de experiencia antes de entrenar


class RedQ(nn.Module):
    """Aproximador de funciones: recibe un estado y devuelve un valor
    estimado para cada acción posible. Es una red neuronal pequeña
    (un "perceptrón multicapa"), suficiente para un estado de solo
    cuatro números como el de CartPole."""

    def __init__(self, dim_estado, num_acciones):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Linear(dim_estado, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_acciones),
        )

    def forward(self, estado):
        return self.capas(estado)


# Cada transición guardada en el buffer es una tupla con nombre, para que
# el código sea más legible al construir los mini-lotes
Transicion = namedtuple(
    "Transicion", ["estado", "accion", "recompensa", "siguiente_estado", "terminado"]
)


class BufferRepeticion:
    """Buffer de repetición de experiencias: guarda transiciones pasadas y
    permite muestrear mini-lotes al azar en vez de entrenar solo con la
    transición que se acaba de vivir. Rompe la correlación entre pasos
    consecutivos y hace que los datos de entrenamiento se parezcan más a
    un conjunto i.i.d. (ver la explicación de este capítulo)."""

    def __init__(self, capacidad):
        # deque con tamaño máximo: al llenarse, descarta la transición más
        # antigua para dejar sitio a la nueva (comportamiento tipo FIFO)
        self.memoria = deque(maxlen=capacidad)

    def guardar(self, *args):
        self.memoria.append(Transicion(*args))

    def muestrear(self, tamano_lote):
        return random.sample(self.memoria, tamano_lote)

    def __len__(self):
        return len(self.memoria)


def elegir_accion(red_online, estado, epsilon, num_acciones):
    """Política epsilon-greedy sobre los valores Q que estima la red online."""
    if random.random() < epsilon:
        return random.randrange(num_acciones)
    with torch.no_grad():
        estado_tensor = torch.tensor(estado, dtype=torch.float32).unsqueeze(0)
        valores_q = red_online(estado_tensor)
        return int(torch.argmax(valores_q, dim=1).item())


def entrenar_paso(red_online, red_objetivo, buffer, optimizador):
    """Un paso de entrenamiento de DQN: muestrea un mini-lote del buffer y
    ajusta la red online para acercar su predicción al objetivo calculado
    con la red objetivo (target network)."""
    lote = buffer.muestrear(TAMANO_LOTE)
    estados = torch.tensor(np.array([t.estado for t in lote]), dtype=torch.float32)
    acciones = torch.tensor([t.accion for t in lote], dtype=torch.int64).unsqueeze(1)
    recompensas = torch.tensor([t.recompensa for t in lote], dtype=torch.float32)
    siguientes_estados = torch.tensor(
        np.array([t.siguiente_estado for t in lote]), dtype=torch.float32
    )
    terminados = torch.tensor([t.terminado for t in lote], dtype=torch.float32)

    # Q(s,a;θ): valor que la red online asigna a la acción que de verdad se tomó
    valores_q = red_online(estados).gather(1, acciones).squeeze(1)

    # Objetivo: r + γ · max_a' Q(s',a';θ⁻), calculado con la red objetivo.
    # torch.no_grad() implementa el "stop-gradient": el error se propaga
    # solo hacia la red online, nunca hacia la red objetivo.
    with torch.no_grad():
        max_q_siguiente = red_objetivo(siguientes_estados).max(dim=1)[0]
        objetivo = recompensas + GAMMA * max_q_siguiente * (1 - terminados)

    perdida = nn.functional.mse_loss(valores_q, objetivo)

    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()


# --- Preparamos entorno, redes, buffer y optimizador ---
entorno = gym.make(NOMBRE_ENTORNO)
dim_estado = entorno.observation_space.shape[0]
num_acciones = entorno.action_space.n

red_online = RedQ(dim_estado, num_acciones)     # red principal: la que se entrena (θ)
red_objetivo = RedQ(dim_estado, num_acciones)   # red objetivo: se congela entre sincronizaciones (θ⁻)
red_objetivo.load_state_dict(red_online.state_dict())  # al empezar, θ⁻ = θ
red_objetivo.eval()

optimizador = optim.Adam(red_online.parameters(), lr=TASA_APRENDIZAJE)
buffer = BufferRepeticion(TAMANO_BUFFER)

recompensas_por_episodio = []
pasos_totales = 0

# --- Bucle de entrenamiento ---
for episodio in range(NUM_EPISODIOS):
    estado, _ = entorno.reset(seed=SEMILLA + episodio)
    recompensa_episodio = 0.0
    terminado = False
    truncado = False

    # epsilon decae linealmente desde EPSILON_INICIAL hasta EPSILON_FINAL
    progreso = min(episodio / EPISODIOS_DECAIMIENTO, 1.0)
    epsilon = EPSILON_INICIAL + progreso * (EPSILON_FINAL - EPSILON_INICIAL)

    while not (terminado or truncado):
        accion = elegir_accion(red_online, estado, epsilon, num_acciones)
        siguiente_estado, recompensa, terminado, truncado, _ = entorno.step(accion)

        buffer.guardar(estado, accion, recompensa, siguiente_estado, terminado)
        estado = siguiente_estado
        recompensa_episodio += recompensa
        pasos_totales += 1

        # Solo entrenamos cuando el buffer ya acumuló suficiente experiencia
        if len(buffer) >= PASOS_MINIMOS_ANTES_ENTRENAR:
            entrenar_paso(red_online, red_objetivo, buffer, optimizador)

        # Sincronizamos la red objetivo cada cierto número de pasos, no en cada paso
        if pasos_totales % PASOS_ACTUALIZAR_RED_OBJETIVO == 0:
            red_objetivo.load_state_dict(red_online.state_dict())

    recompensas_por_episodio.append(recompensa_episodio)

    if (episodio + 1) % 20 == 0:
        media_reciente = np.mean(recompensas_por_episodio[-20:])
        print(
            f"Episodio {episodio + 1}/{NUM_EPISODIOS} | epsilon={epsilon:.2f} "
            f"| recompensa media (últimos 20)={media_reciente:.1f}"
        )

entorno.close()

# --- Graficamos la recompensa por episodio ---
plt.figure(figsize=(10, 5))
plt.plot(recompensas_por_episodio, alpha=0.4, label="Recompensa por episodio")
# Media móvil de 20 episodios, para ver la tendencia sin el ruido episodio a episodio
media_movil = np.convolve(recompensas_por_episodio, np.ones(20) / 20, mode="valid")
plt.plot(range(19, len(recompensas_por_episodio)), media_movil, label="Media móvil (20 episodios)")
plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.title("DQN en CartPole-v1: recompensa por episodio")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Al ejecutarlo vas a ver que, al principio, la recompensa por episodio es baja e irregular: el péndulo se cae casi de inmediato porque epsilon todavía está cerca de uno y la red apenas ha aprendido nada. A medida que avanzan los episodios y epsilon decae, la media móvil debería subir de forma bastante clara, aunque con oscilaciones: esa mezcla de progreso y ruido episodio a episodio es completamente normal en DQN, incluso con buffer de repetición y red objetivo funcionando bien.

## Puntos clave

- Los métodos tabulares fallan cuando el número de estados crece demasiado: no caben en memoria y el agente rara vez repite exactamente el mismo estado
- La aproximación de funciones sustituye la tabla por una función parametrizada, `Q(s,a;θ)`, que puede generalizar a estados nuevos
- Una red neuronal es, en esencia, una función con pesos ajustables que se entrena para parecerse cada vez más a la función que quieres aproximar
- Unir Q-learning con una red neuronal de forma ingenua es inestable, porque los datos de RL no son i.i.d. y el objetivo de entrenamiento se mueve a la vez que la red que lo persigue (bootstrapping)
- El buffer de repetición de experiencias rompe la correlación entre pasos consecutivos y reutiliza cada transición varias veces
- La red objetivo congela una copia de los parámetros durante un tramo de entrenamiento, así el objetivo deja de moverse en cada paso y el entrenamiento se estabiliza
- Double DQN, Dueling DQN y Prioritized Experience Replay son mejoras sobre DQN que vas a implementar en el capítulo 11

## Ponlo en práctica

Toma el código de este capítulo y modifícalo tú misma o tú mismo:

- **Cambia `PASOS_ACTUALIZAR_RED_OBJETIVO`** (prueba con 50, con 500 y con 5.000) y observa qué pasa con la estabilidad de la curva de recompensa: una red objetivo que se sincroniza demasiado rápido se parece cada vez más al Q-learning ingenuo e inestable que viste al principio del capítulo
- **Cambia `TAMANO_BUFFER`** (prueba con 500 y con 50.000) y comprueba si un buffer muy pequeño hace que el entrenamiento se vuelva más errático, al perder la diversidad de experiencias pasadas
- **Elimina temporalmente la red objetivo** (usa `red_online` en vez de `red_objetivo` dentro de `entrenar_paso`) y compara la curva resultante con la versión original

¡Ejecuta el notebook, cambia los hiperparámetros y anota con tu equipo qué combinación entrena de forma más estable!

---

DQN aprende valores de acción y deriva la política de ellos, de forma indirecta, escogiendo siempre la acción con mayor Q. En el capítulo 9 vas a conocer otra familia de métodos, los gradientes de política y los métodos actor-crítico, que aprenden la política de forma directa, sin pasar por una función de valor intermedia.
