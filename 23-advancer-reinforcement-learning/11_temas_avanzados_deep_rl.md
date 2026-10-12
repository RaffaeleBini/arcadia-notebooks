# Capítulo 11. Rainbow y el aprendizaje distribucional: cuando varias mejoras se suman

En el capítulo 8 conociste Deep Q-Network (DQN): un agente que combina aprendizaje-Q con una red neuronal profunda, una red objetivo y repetición de experiencia para jugar videojuegos directamente a partir de píxeles. Fue un hito, pero no fue un punto final. Desde su publicación, la comunidad investigadora en Aprendizaje por Refuerzo (RL) ha ido proponiendo mejoras concretas a distintas piezas de ese mismo algoritmo: cómo elige acciones, qué recuerda, cómo explora, qué predice. Cada mejora, por separado, ya aporta algo. La pregunta que responde este capítulo es qué pasa cuando las juntas todas.

En el capítulo anterior viste cómo un agente puede planificar con un modelo del entorno en vez de limitarse a reaccionar. Aquí vuelves al terreno libre de modelo, pero con una mirada distinta: en lugar de un algoritmo nuevo, vas a ver un algoritmo que es la suma cuidadosa de varios algoritmos ya conocidos. Se llama Rainbow, y es el ejemplo de referencia de este capítulo, junto con una idea que atraviesa varias de sus piezas: aprender la distribución completa de resultados posibles, no solo su promedio.

## Rainbow: la suma de seis mejoras independientes

El nombre no es casual. Igual que un arcoíris aparece cuando se combinan varios colores que, por separado, ya existen, Rainbow DQN aparece cuando combinas seis mejoras que, por separado, ya se habían probado con éxito sobre el DQN original. Ningún componente es exclusivo de Rainbow; lo que aporta Rainbow es la evidencia de que integrarlos todos en el mismo agente funciona mejor que cualquiera de ellos por separado.

Repásalos uno a uno:

**Doble aprendizaje-Q.** Ya lo viste en el capítulo 7: DQN tiende a sobreestimar el valor de algunas acciones porque usa la misma red para elegir la mejor acción y para evaluarla. Doble aprendizaje-Q separa esas dos tareas: una red (la red online) elige qué acción parece mejor, y otra (la red objetivo) calcula cuánto vale esa elección. Aquí, dentro de Rainbow, esa misma idea se integra como una pieza más del conjunto.

**Repetición de experiencia priorizada (PER, por sus siglas en inglés).** El DQN básico guarda transiciones pasadas en un buffer y muestrea lotes al azar para entrenar. PER cambia el "al azar" por "con criterio": las transiciones donde el agente se equivocó más (mayor error de diferencia temporal) se repiten con más frecuencia, porque son las que más información nueva aportan. Es como repasar con más intensidad los ejercicios que fallaste en un examen, en lugar de repasar todos por igual.

**Arquitectura dueling.** En vez de que la red prediga directamente un valor Q por cada acción, lo separa en dos preguntas distintas: ¿cuánto vale este estado en general, independientemente de lo que haga a continuación? Y, dentro de ese estado, ¿cuánto mejor es cada acción concreta frente a las demás? La primera pregunta la responde una cabeza de valor V(s); la segunda, una cabeza de ventaja A(s,a). Al combinarlas, el agente aprende cuánto vale "estar aquí" por separado de "hacer esto en concreto", lo que ayuda especialmente en estados donde la acción elegida casi no importa. Un ejemplo visual que usa el material fuente: en un videojuego de carreras, la cabeza de valor atiende a la escena completa (¿voy bien en la carrera?), mientras que la cabeza de ventaja se fija en detalles inmediatos, como el coche que tienes al lado.

**Retornos multi-paso.** El DQN estándar mira un solo paso hacia delante antes de actualizar sus estimaciones. Rainbow, en cambio, acumula la recompensa de varios pasos seguidos antes de bootstrapear con la red objetivo. Es un punto intermedio entre mirar solo el siguiente paso y esperar al final completo del episodio (como hace Monte Carlo): ves un poco más lejos antes de sacar conclusiones, lo que suele acelerar el aprendizaje.

**Aprendizaje distribucional.** Esta es, quizá, la idea conceptualmente más distinta de las seis. En vez de aprender un único número por acción (el retorno esperado), el agente aprende la distribución completa de retornos posibles: no solo "cuánto espero ganar en promedio", sino "qué probabilidad tiene cada posible resultado". El algoritmo concreto que usa Rainbow se llama C51: representa esa distribución mediante 51 puntos fijos (llamados átomos) repartidos en un rango conocido, aprende una probabilidad para cada uno, y compara la distribución predicha con la distribución objetivo mediante la divergencia de Kullback-Leibler (KL) en vez de un simple error cuadrático. Aprender toda la distribución, y no solo la media, le da al agente una representación más rica del mundo con la misma cantidad de datos, aunque conviene aclarar algo importante: esa distribución de retornos no es una medida de incertidumbre del agente, sino de la variabilidad real de los resultados posibles.

**Redes ruidosas (Noisy Nets).** En lugar de explorar con epsilon-greedy (una probabilidad fija de actuar al azar), Rainbow añade ruido aprendible directamente a los pesos y los sesgos de la red. La propia red aprende, mediante el mismo optimizador con el que aprende todo lo demás, cuánto ruido "dejar activo": poco ruido para situaciones que ya conoce bien, más ruido para situaciones raras donde todavía tiene sentido explorar. La exploración deja de ser un parámetro externo y pasa a formar parte del aprendizaje.

### Lo que reveló el estudio de ablación

Cuando el equipo que propuso Rainbow probó qué pasaba al quitar cada componente uno por uno, el resultado no fue simétrico. El aprendizaje multi-paso, la repetición priorizada y el aprendizaje distribucional resultaron ser los que más aportaban al rendimiento final. El doble aprendizaje-Q y la arquitectura dueling, en cambio, aportaron menos de lo esperado una vez combinados con el resto: una explicación propuesta es que el aprendizaje distribucional ya limita por sí mismo el rango de retornos posibles, lo que reduce buena parte del problema de sobreestimación que el doble aprendizaje-Q había sido diseñado para resolver. En cuanto a eficiencia, la cifra que se suele citar es contundente: Rainbow logra, en unos siete millones de fotogramas de entrenamiento, la misma puntuación media normalizada que el DQN estándar necesita cerca de 200 millones de fotogramas para conseguir. Es una mejora de eficiencia de datos de aproximadamente 14 veces, con un tiempo de cómputo similar al del DQN original.

### Líneas de investigación activas que verás nombradas en la literatura

Más allá de Rainbow, encontrarás otros nombres que circulan con frecuencia en RL avanzado. El material de referencia de este capítulo solo los menciona de pasada, así que trátalos como mapa de lo que existe, no como contenido desarrollado aquí:

- **RL multiagente**: varios agentes que aprenden a la vez en el mismo entorno, a veces cooperando y a veces compitiendo; conceptos habituales incluyen el entrenamiento centralizado con ejecución descentralizada y el modelado del comportamiento de otros agentes
- **RL jerárquico**: en vez de aprender solo acciones de un paso, el agente aprende también acciones "extendidas en el tiempo" (llamadas opciones), cada una con su propia política interna y su propia condición de terminación
- **Meta-aprendizaje, aprendizaje por transferencia y aprendizaje por currículo**: formas de que un agente aproveche lo aprendido en una tarea para aprender más rápido otra tarea relacionada, o de organizar el entrenamiento en un orden de dificultad creciente
- **Seguridad en RL**: un área que gana relevancia a medida que estos agentes se acercan a entornos reales; el aprendizaje distribucional, por ejemplo, ya aporta algo útil aquí al informar sobre la probabilidad de terminaciones tempranas o indeseadas, no solo sobre el retorno esperado

Si alguno de estos nombres te resulta interesante, apúntalo como línea propia de estudio: este manual no los desarrolla en profundidad porque el material de origen tampoco lo hace.

## Ejemplo en Python: añadiendo una arquitectura dueling a tu DQN

Programar las seis mejoras de Rainbow en un solo capítulo no sería realista. En cambio, vas a retomar el DQN mínimo del capítulo 8 y añadirle una sola mejora, bien delimitada: la arquitectura dueling. Vas a entrenar dos agentes sobre CartPole (un DQN estándar y un Dueling DQN) con exactamente el mismo bucle de entrenamiento, y vas a comparar sus curvas de recompensa. La única diferencia entre ambos va a ser la arquitectura interna de la red.

```python
!pip install gymnasium

import random
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Fijamos semillas para que el experimento sea reproducible
SEMILLA = 0
random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

Transicion = namedtuple(
    "Transicion", ["estado", "accion", "recompensa", "siguiente_estado", "terminado"]
)


class BufferReplay:
    """Memoria de repetición de experiencia (capítulo 8): guarda transiciones
    pasadas y permite muestrear lotes al azar, rompiendo la correlación
    temporal de los datos consecutivos de un episodio."""

    def __init__(self, capacidad=10000):
        self.memoria = deque(maxlen=capacidad)

    def guardar(self, *args):
        self.memoria.append(Transicion(*args))

    def muestrear(self, tamano_lote):
        return random.sample(self.memoria, tamano_lote)

    def __len__(self):
        return len(self.memoria)


class RedQEstandar(nn.Module):
    """DQN "de manual": una sola cabeza que produce directamente
    un valor Q por cada acción posible."""

    def __init__(self, n_entradas, n_acciones, n_ocultas=128):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_entradas, n_ocultas),
            nn.ReLU(),
            nn.Linear(n_ocultas, n_ocultas),
            nn.ReLU(),
            nn.Linear(n_ocultas, n_acciones),
        )

    def forward(self, x):
        return self.red(x)


class RedQDueling(nn.Module):
    """Arquitectura dueling: un tronco compartido alimenta dos cabezas.
    Una estima V(s) -- cuánto vale el estado, un único número --
    y otra estima A(s,a) -- cuánto mejor es cada acción frente a las demás.
    Se combinan restando la media de la ventaja, para que la suma
    quede en una escala estable."""

    def __init__(self, n_entradas, n_acciones, n_ocultas=128):
        super().__init__()
        self.tronco = nn.Sequential(
            nn.Linear(n_entradas, n_ocultas),
            nn.ReLU(),
        )
        self.cabeza_valor = nn.Sequential(
            nn.Linear(n_ocultas, n_ocultas),
            nn.ReLU(),
            nn.Linear(n_ocultas, 1),
        )
        self.cabeza_ventaja = nn.Sequential(
            nn.Linear(n_ocultas, n_ocultas),
            nn.ReLU(),
            nn.Linear(n_ocultas, n_acciones),
        )

    def forward(self, x):
        caracteristicas = self.tronco(x)
        valor = self.cabeza_valor(caracteristicas)
        ventaja = self.cabeza_ventaja(caracteristicas)
        # Q(s,a) = V(s) + (A(s,a) - media_a[A(s,a)])
        # Usamos dim=-1 en vez de dim=1: así funciona tanto si pasas un solo
        # estado (vector 1D, al elegir una acción) como un lote de estados (2D, al entrenar)
        return valor + (ventaja - ventaja.mean(dim=-1, keepdim=True))


def entrenar(clase_red, nombre, n_episodios=300, gamma=0.99, tamano_lote=64,
             tasa_aprendizaje=1e-3, actualizar_objetivo_cada=10):
    """Bucle de entrenamiento DQN estándar: red online, red objetivo,
    buffer de repetición y exploración epsilon-greedy decreciente.
    Cambiar `clase_red` entre RedQEstandar y RedQDueling es la única
    diferencia entre ambos experimentos -- el resto del algoritmo no cambia."""

    entorno = gym.make("CartPole-v1")
    n_entradas = entorno.observation_space.shape[0]
    n_acciones = entorno.action_space.n

    red_online = clase_red(n_entradas, n_acciones)
    red_objetivo = clase_red(n_entradas, n_acciones)
    red_objetivo.load_state_dict(red_online.state_dict())
    red_objetivo.eval()

    optimizador = optim.Adam(red_online.parameters(), lr=tasa_aprendizaje)
    buffer = BufferReplay(capacidad=10000)

    epsilon, epsilon_minimo, decaimiento = 1.0, 0.05, 0.97
    recompensas_por_episodio = []

    for episodio in range(n_episodios):
        estado, _ = entorno.reset(seed=SEMILLA + episodio)
        recompensa_total = 0.0
        terminado = False

        while not terminado:
            if random.random() < epsilon:
                accion = entorno.action_space.sample()  # exploración
            else:
                with torch.no_grad():
                    q_valores = red_online(torch.tensor(estado, dtype=torch.float32))
                    accion = int(torch.argmax(q_valores).item())  # explotación

            siguiente_estado, recompensa, fin, truncado, _ = entorno.step(accion)
            terminado = fin or truncado
            buffer.guardar(estado, accion, recompensa, siguiente_estado, fin)
            estado = siguiente_estado
            recompensa_total += recompensa

            # Solo entrenamos cuando el buffer tiene datos suficientes
            if len(buffer) >= tamano_lote:
                lote = buffer.muestrear(tamano_lote)
                estados = torch.tensor(np.array([t.estado for t in lote]), dtype=torch.float32)
                acciones = torch.tensor([t.accion for t in lote], dtype=torch.int64)
                recompensas = torch.tensor([t.recompensa for t in lote], dtype=torch.float32)
                siguientes = torch.tensor(np.array([t.siguiente_estado for t in lote]), dtype=torch.float32)
                terminados = torch.tensor([t.terminado for t in lote], dtype=torch.float32)

                q_actuales = red_online(estados).gather(1, acciones.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_siguientes = red_objetivo(siguientes).max(dim=1)[0]
                    objetivo = recompensas + gamma * q_siguientes * (1 - terminados)

                perdida = nn.functional.mse_loss(q_actuales, objetivo)
                optimizador.zero_grad()
                perdida.backward()
                optimizador.step()

        recompensas_por_episodio.append(recompensa_total)
        epsilon = max(epsilon_minimo, epsilon * decaimiento)

        if episodio % actualizar_objetivo_cada == 0:
            red_objetivo.load_state_dict(red_online.state_dict())

    entorno.close()
    print(f"{nombre}: recompensa media de los últimos 20 episodios = "
          f"{np.mean(recompensas_por_episodio[-20:]):.1f}")
    return recompensas_por_episodio


# --- Entrenamos ambos agentes con el mismo bucle ---
recompensas_dqn = entrenar(RedQEstandar, "DQN estándar")
recompensas_dueling = entrenar(RedQDueling, "Dueling DQN")


def media_movil(valores, ventana=10):
    return np.convolve(valores, np.ones(ventana) / ventana, mode="valid")


# --- Comparamos las curvas de recompensa ---
plt.figure(figsize=(10, 5))
plt.plot(media_movil(recompensas_dqn), label="DQN estándar")
plt.plot(media_movil(recompensas_dueling), label="Dueling DQN")
plt.xlabel("Episodio")
plt.ylabel("Recompensa (media móvil de 10 episodios)")
plt.title("CartPole-v1: DQN estándar vs. Dueling DQN")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

Al ejecutarlo, fíjate en dos cosas: cuánto tarda cada curva en acercarse a la recompensa máxima de CartPole (500 pasos) y cuánta variabilidad muestra cada una entre episodios. En un entorno tan simple como CartPole, la ventaja del Dueling DQN suele ser modesta (la arquitectura dueling brilla más en entornos con muchos estados donde la acción elegida casi no cambia el resultado), pero el ejercicio te deja el esqueleto completo para añadir, más adelante, las otras cinco mejoras de Rainbow sobre esta misma base.

## Puntos clave

- DQN, del capítulo 8, es el punto de partida; Rainbow es la suma de seis mejoras independientes que, combinadas, superan claramente a DQN por separado
- Las seis mejoras son: doble aprendizaje-Q, repetición de experiencia priorizada, arquitectura dueling, retornos multi-paso, aprendizaje distribucional (C51) y redes ruidosas
- El estudio de ablación muestra que multi-paso, repetición priorizada y aprendizaje distribucional aportan más que doble aprendizaje-Q y dueling una vez combinados
- Rainbow logra en unos siete millones de fotogramas lo que DQN tarda cerca de 200 millones en conseguir: aproximadamente 14 veces más eficiencia de datos
- El aprendizaje distribucional aprende la distribución completa de retornos posibles, no solo su media, y usa divergencia KL como función de pérdida
- RL multiagente, RL jerárquico, meta-aprendizaje/transferencia/currículo y seguridad en RL son líneas de investigación activas que verás nombradas en la literatura, aunque este capítulo no las desarrolla en profundidad

## Ponlo en práctica

Elige una de estas dos tareas y complétala:

- **Añade retornos multi-paso** al bucle de `entrenar`: en vez de guardar transiciones de un solo paso, acumula la recompensa de los últimos n pasos (por ejemplo, n = 3) antes de calcular el objetivo de bootstrapeo, y compara la curva resultante con la del Dueling DQN de este capítulo
- **Investiga otro componente de Rainbow** que no hayas implementado (repetición priorizada, redes ruidosas o aprendizaje distribucional), y anota en un párrafo qué cambiarías en el código de este capítulo para incorporarlo y por qué crees que ayudaría en CartPole

Comparte tu resultado con tu equipo y **compara** qué combinación de mejoras dio la curva de aprendizaje más estable.

---

En el capítulo 12 vas a cambiar de perspectiva: en vez de mejorar un único agente que aprende solo, vas a ver arquitecturas donde varias unidades de cómputo entrenan en paralelo, como A3C e IMPALA.
