# Capítulo 9. Gradientes de política y métodos Actor-Crítico

En el capítulo anterior conociste Deep Q-Network (DQN): un agente que aprende cuánto vale cada acción en cada estado, mediante una red neuronal, y que después actúa eligiendo siempre la acción con mayor valor. Esa estrategia funciona muy bien, pero da un rodeo: primero aprende una función de valor completa y solo después, de forma indirecta, deriva una política a partir de ella.

En este capítulo vas a ver una idea distinta: en lugar de aprender cuánto vale cada acción, aprende directamente la política, es decir, la regla que decide qué hacer en cada estado. Piensa en un robot cuya mejor estrategia es, simplemente, "avanzar siempre". Su política óptima es trivial. Pero la función de valor que describe cuánto vale cada combinación de posición y obstáculos puede ser enormemente compleja. Si tu objetivo real es actuar bien, ¿por qué aprender primero ese rodeo complicado en vez de ir directo a la política? Este principio, evitar resolver un problema más general del que necesitas para resolver el problema concreto que te interesa, se atribuye al estadístico Vladimir Vapnik, y es la motivación central de este capítulo.

Aprender la política directamente tiene, además, dos ventajas prácticas que DQN no ofrece con la misma naturalidad. La primera aparece en espacios de acción continuos o muy grandes, como el ángulo de un motor o la fuerza de un brazo robótico: DQN necesita calcular el máximo sobre todas las acciones posibles, algo inviable cuando hay infinitas. Un método basado en política, en cambio, simplemente emite la acción (o una distribución sobre acciones) sin necesitar ese máximo. La segunda ventaja es la posibilidad de aprender políticas genuinamente aleatorias, no solo deterministas, algo que vas a entender enseguida.

## Por qué a veces conviene actuar al azar

Imagina un laberinto con dos pasillos que, vistos desde dentro, parecen exactamente iguales: mismas paredes a izquierda y derecha, mismo aspecto. Un agente que solo sabe actuar de forma determinista puede quedar atrapado, yendo de un pasillo al otro en un bucle infinito, porque desde su punto de vista ambos estados son indistinguibles. Una política estocástica, que en esos pasillos elige izquierda o derecha al azar, tarde o temprano escapa del bucle por pura probabilidad y llega a una zona donde sí puede distinguir dónde está.

Lo mismo pasa en juegos con información oculta. En piedra, papel o tijera, cualquier estrategia determinista es explotable de inmediato: tu rival solo necesita detectar el patrón. La única estrategia óptima es elegir cada opción con la misma probabilidad. En el póquer ocurre algo parecido: si tu forma de actuar con una mano ganadora es siempre la misma, un rival atento la aprende y la explota; el farol solo funciona si es probabilístico.

Una política, entonces, ya no es una regla fija "estado → acción" (una función determinista `a = μ(s)`), sino una distribución de probabilidad sobre las acciones dado el estado: `π(a|s)`. Aprender esta distribución, en vez de una regla fija, es lo que hace posible representar comportamientos como los del laberinto o el póquer.

## El truco de la función score

Aquí surge un problema técnico. Quieres ajustar los parámetros de tu política, θ, para maximizar la recompensa esperada. Pero esa recompensa esperada depende de un proceso de muestreo aleatorio: la acción que tomas se saca de una distribución de probabilidad, y no puedes "derivar" directamente a través de un sorteo.

La solución se conoce como truco de la función score (o truco REINFORCE, o truco log-derivada). Sin entrar en el desarrollo matemático completo, la idea intuitiva es esta: en vez de intentar derivar el resultado de un muestreo, conviertes el problema en algo que sí sabes derivar, el logaritmo de la probabilidad de la acción que elegiste. El resultado, tras el desarrollo, es sorprendentemente simple de aplicar: para ajustar la política, calculas el gradiente del logaritmo de la probabilidad de cada acción que tomaste, y lo multiplicas por lo buena que resultó esa acción (su recompensa o retorno).

Dicho de forma llana: sube la probabilidad de las acciones que llevaron a una recompensa alta, y baja la probabilidad de las que llevaron a una recompensa baja o negativa. Cuanto mejor haya sido el resultado, más fuerte es el empujón hacia arriba.

Un detalle importante de esta derivación: cuando descompones la probabilidad de una trayectoria completa, aparecen dos tipos de términos, los que dependen de la dinámica del entorno (qué estado sigue a cuál) y los que dependen solo de tu política. Al calcular el gradiente respecto a los parámetros de la política, los términos de la dinámica del entorno desaparecen, porque no dependen de θ. Esto significa algo muy práctico: los métodos de gradiente de política no necesitan conocer ni modelar cómo funciona el entorno; les basta con poder actuar en él y observar qué pasa.

## REINFORCE: el algoritmo más simple

Con esa idea ya tienes el algoritmo REINFORCE, la versión más básica de gradiente de política:

1. Inicializa los parámetros θ de la política, normalmente una red neuronal
2. Repite hasta que el rendimiento te convenza:
   - Ejecuta un episodio completo siguiendo la política actual y guarda estados, acciones y recompensas
   - Para cada paso del episodio, calcula el retorno (la suma de recompensas futuras, normalmente con descuento)
   - Calcula el gradiente: para cada acción tomada, multiplica el gradiente del logaritmo de su probabilidad por el retorno correspondiente
   - Actualiza los parámetros en la dirección de ese gradiente: `θ ← θ + α · gradiente`

Es un algoritmo elegante y fácil de programar. Pero tiene un problema serio: alta varianza. Cada episodio depende de decisiones aleatorias (qué acciones sacó el muestreo) y de la aleatoriedad propia del entorno, así que el retorno de un episodio puede ser muy distinto del retorno de otro episodio casi idéntico. Eso hace que las actualizaciones salten de un lado a otro y el entrenamiento sea lento e inestable.

## Reducir la varianza: línea base y ventaja

Aquí entra una idea muy elegante. Puedes restar del retorno una línea base, `b(s)`, un valor que depende solo del estado (no de la acción elegida), sin introducir ningún sesgo en el aprendizaje. La razón es que, al sumar sobre todas las acciones posibles, el efecto de una línea base constante se cancela matemáticamente: no cambia la dirección del gradiente, solo reduce su varianza.

La línea base más habitual es la propia función de valor de estado, `V(s)`: cuánta recompensa esperas obtener desde ese estado, siguiendo tu política actual. Al restarla del valor de acción `Q(s,a)`, obtienes la función de ventaja:

`A(s,a) = Q(s,a) − V(s)`

La ventaja responde a una pregunta muy concreta: ¿cuánto mejor (o peor) es esta acción concreta comparada con lo que harías en promedio en este estado? Una ventaja positiva empuja la probabilidad de esa acción hacia arriba; una ventaja negativa la empuja hacia abajo. Es una señal mucho más limpia que el retorno bruto, porque ya no arrastra la parte de la recompensa que depende solo del estado en el que estabas, sino solo la parte que depende de la decisión que tomaste.

En la práctica, no siempre necesitas dos redes separadas para Q y V. Puedes estimar la ventaja con el error de diferencia temporal (TD, por las siglas de "temporal difference"): `δ = r + γV(s') − V(s)`. Este error compara lo que esperabas obtener con lo que realmente obtuviste tras dar un paso, y es, en la práctica, una muestra de la ventaja.

## Arquitectura Actor-Crítico

Esa última idea lleva directamente a los métodos Actor-Crítico. Imagina un chef (el actor) que decide qué plato cocinar, y una persona experta en gastronomía (el crítico) que prueba cada plato y le dice cuánto mejor o peor fue que lo esperado. El chef usa esa valoración para ajustar sus próximas decisiones.

En términos técnicos, el actor es la política `π_θ`, que decide qué acción tomar, y el crítico es una función de valor `V_ω`, que evalúa qué tan buena está resultando la actuación del actor. El crítico se entrena con técnicas de diferencia temporal, igual que la función de valor de DQN. El actor se entrena con el gradiente de política, usando el error TD del crítico como señal de ventaja. El esquema básico es este:

```
Inicializa la red del actor (política) con parámetros θ
Inicializa la red del crítico (valor) con parámetros ω

para cada episodio:
    inicializa el estado s
    para cada paso del episodio:
        muestrea una acción a ~ π_θ(a|s)
        ejecuta a, observa la recompensa r y el siguiente estado s'
        δ = r + γ · V_ω(s') − V_ω(s)          # error TD, usado como ventaja
        ω ← ω + α_crítico · δ · ∇_ω V_ω(s)     # actualiza el crítico
        θ ← θ + α_actor · δ · ∇_θ log π_θ(a|s) # actualiza el actor
        s ← s'
```

Este patrón, en el que el crítico aprende con cada paso (no hace falta esperar a que termine el episodio) y el actor usa esa señal para mejorar, se conoce como Advantage Actor-Critic (A2C), y su variante que ejecuta varios agentes en paralelo, cada uno con su propia copia del entorno, se llama Asynchronous Advantage Actor-Critic (A3C). Vas a volver a encontrarte con A3C en el capítulo 12.

Existen evoluciones posteriores de estas ideas que verás mencionadas más adelante en este manual, sin entrar en su detalle técnico aquí: Trust Region Policy Optimization (TRPO) y Proximal Policy Optimization (PPO), que limitan cuánto puede cambiar la política en cada actualización para evitar pasos demasiado bruscos, y Deep Deterministic Policy Gradient (DDPG), pensado para espacios de acción continuos.

## Ejemplo en Python: REINFORCE en CartPole

El siguiente código implementa REINFORCE desde cero sobre CartPole, el entorno clásico en el que un carro debe mantener un palo en equilibrio moviéndose a izquierda o derecha. Puedes copiarlo y ejecutarlo directamente en Google Colab: usa PyTorch (ya preinstalado) y gymnasium (se instala en la primera celda). No necesita GPU.

```python
# Celda 1: instala gymnasium (PyTorch ya viene preinstalado en Colab)
!pip install gymnasium
```

```python
# Celda 2: imports
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Fijamos semillas para que el experimento sea reproducible
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)


class RedPolitica(nn.Module):
    """Red neuronal pequeña que representa la política π_θ(a|s).
    Recibe el estado de CartPole (cuatro números: posición y velocidad del
    carro, ángulo y velocidad angular del palo) y devuelve una distribución
    de probabilidad sobre las dos acciones posibles (empujar el carro a la
    izquierda o a la derecha)."""

    def __init__(self, n_entradas, n_acciones, n_ocultas=128):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_entradas, n_ocultas),
            nn.ReLU(),
            nn.Linear(n_ocultas, n_acciones),
        )

    def forward(self, estado):
        logits = self.red(estado)
        return torch.softmax(logits, dim=-1)


def calcular_retornos(recompensas, gamma=0.99):
    """Calcula el retorno 'reward-to-go' en cada paso del episodio:
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    Se recorre la lista de atrás hacia adelante para reutilizar el cálculo
    del paso siguiente en cada paso anterior."""
    retornos = np.zeros(len(recompensas))
    G = 0.0
    for t in reversed(range(len(recompensas))):
        G = recompensas[t] + gamma * G
        retornos[t] = G
    return retornos


def ejecutar_episodio(env, politica):
    """Recolecta un episodio completo siguiendo la política actual.
    Devuelve la log-probabilidad de cada acción tomada (necesaria para el
    gradiente de política) y la recompensa obtenida en cada paso."""
    log_probs = []
    recompensas = []
    estado, _ = env.reset()
    terminado = False
    truncado = False

    while not (terminado or truncado):
        estado_tensor = torch.FloatTensor(estado).unsqueeze(0)
        probs = politica(estado_tensor)
        distribucion = torch.distributions.Categorical(probs)
        accion = distribucion.sample()

        log_probs.append(distribucion.log_prob(accion))
        estado, recompensa, terminado, truncado, _ = env.step(accion.item())
        recompensas.append(recompensa)

    return log_probs, recompensas


# --- Configuración del entrenamiento ---
env = gym.make("CartPole-v1")
n_entradas = env.observation_space.shape[0]
n_acciones = env.action_space.n

politica = RedPolitica(n_entradas, n_acciones)
optimizador = optim.Adam(politica.parameters(), lr=1e-2)

N_EPISODIOS = 500
GAMMA = 0.99

historial_recompensas = []

for episodio in range(N_EPISODIOS):
    log_probs, recompensas = ejecutar_episodio(env, politica)
    retornos = calcular_retornos(recompensas, GAMMA)
    retornos_tensor = torch.FloatTensor(retornos)

    # Pérdida del gradiente de política: -sum(log_prob(a_t|s_t) * G_t)
    # El signo negativo convierte el ASCENSO de gradiente que queremos
    # (maximizar la recompensa esperada) en el DESCENSO de gradiente que
    # optimiza PyTorch por defecto.
    perdida = torch.stack(
        [-log_prob * G for log_prob, G in zip(log_probs, retornos_tensor)]
    ).sum()

    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()

    historial_recompensas.append(sum(recompensas))

    if (episodio + 1) % 50 == 0:
        media_reciente = np.mean(historial_recompensas[-50:])
        print(f"Episodio {episodio + 1}: recompensa media (últimos 50) = {media_reciente:.1f}")

env.close()

# --- Gráfico de la recompensa por episodio ---
plt.figure(figsize=(10, 5))
plt.plot(historial_recompensas, alpha=0.4, label="Recompensa por episodio")

ventana = 20  # media móvil para ver la tendencia sin el ruido episodio a episodio
media_movil = np.convolve(historial_recompensas, np.ones(ventana) / ventana, mode="valid")
plt.plot(range(ventana - 1, len(historial_recompensas)), media_movil, label=f"Media móvil ({ventana} episodios)")

plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.title("REINFORCE en CartPole-v1")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Al ejecutarlo vas a ver algo típico de REINFORCE: la curva de recompensa sube, pero con muchos altibajos entre episodios cercanos, justo el problema de varianza que ya conoces. En CartPole-v1, un episodio puede durar hasta 500 pasos, así que una recompensa media cercana a ese número indica que el palo se mantiene en equilibrio casi todo el tiempo permitido.

## Puntos clave

- Los métodos de gradiente de política aprenden directamente la política, sin pasar por una función de valor completa, lo que encaja mejor con espacios de acción continuos o muy grandes
- Una política estocástica, `π(a|s)`, es necesaria en estados ambiguos y en juegos con información oculta, donde una regla determinista sería explotable
- El truco de la función score convierte un problema de derivar a través de un muestreo aleatorio en algo calculable: el gradiente del logaritmo de la probabilidad de la acción, multiplicado por lo buena que resultó
- REINFORCE aplica esa idea de forma directa, pero sufre de varianza alta porque depende de retornos completos de episodios ruidosos
- Restar una línea base (habitualmente `V(s)`) reduce la varianza sin introducir sesgo, y da lugar a la función de ventaja `A(s,a) = Q(s,a) − V(s)`
- La arquitectura Actor-Crítico combina un actor (la política) con un crítico (una función de valor) que evalúa cada paso y guía la actualización del actor con el error TD
- A2C, A3C, PPO, TRPO y DDPG son evoluciones de estas ideas que irás encontrando en próximos capítulos, sin necesidad de dominar su detalle técnico todavía

## Ponlo en práctica

Toma el código de este capítulo y modifícalo tú misma o tú mismo:

- **Añade una línea base simple**: dentro del bucle de entrenamiento, calcula la media de `retornos` en cada episodio y réstasela a cada retorno antes de calcular la pérdida
- **Compara la estabilidad**: ejecuta el entrenamiento con y sin esa línea base, usando la misma semilla, y observa las dos curvas de recompensa por episodio; fíjate especialmente en cuánto saltan los valores entre episodios consecutivos
- **Prueba distintos valores de `GAMMA`** (por ejemplo, 0,9 frente a 0,99) y anota cómo cambia la velocidad de aprendizaje

¡Ejecuta el notebook con y sin línea base, y comparte con tu equipo cuál de las dos versiones llegó antes a mantener el palo en equilibrio durante los 500 pasos!

---

Hasta ahora, el agente aprendió únicamente a partir de la experiencia directa: actuó, observó qué pasó y ajustó su comportamiento, sin intentar predecir cómo funciona el entorno. En el capítulo 10 vas a ver qué pasa cuando el agente sí construye o usa un modelo del entorno para planificar por adelantado, con técnicas como Dyna-Q y la búsqueda en árbol de Monte Carlo (Monte Carlo Tree Search).
