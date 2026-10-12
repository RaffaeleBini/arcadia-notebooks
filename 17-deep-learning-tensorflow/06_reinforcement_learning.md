[⬅ Volver al índice](00_indice.md)

# Capítulo 6 — Aprendizaje por refuerzo: enseña a una IA a jugar

## Por qué te importa esto

Hasta ahora tus modelos aprendían de ejemplos ya etiquetados: "esta imagen es un
gato", "esta reseña es negativa". Pero, ¿cómo entrenas a una IA cuando no existe
una respuesta correcta de antemano, solo la posibilidad de intentarlo y ver qué
pasa? Ese es el terreno del aprendizaje por refuerzo, o Reinforcement Learning
(RL): la técnica detrás de sistemas que aprenden a jugar videojuegos mejor que
cualquier humano, a controlar robots o a optimizar decisiones en tiempo real.

## Cómo aprende un agente por prueba y error (con una analogía)

Piensa en cómo aprendiste a montar en bicicleta: nadie te dio una lista de reglas
exactas de cuánto girar el manillar en cada instante. Probaste, te caíste, y
ajustaste tu comportamiento según lo que funcionaba (mantenerte en pie) y lo que no
(caerte). Con el tiempo, desarrollaste una intuición —una "política"— de qué hacer
en cada situación para maximizar el tiempo que te mantienes en equilibrio.

En RL, un "agente" interactúa con un "entorno": en cada paso observa un estado,
elige una acción, y recibe una recompensa que le indica si esa acción fue buena o
mala. El objetivo del agente es aprender una política que maximice la recompensa
acumulada a lo largo del tiempo, no solo la recompensa inmediata.

## El entorno CartPole

OpenAI Gym ofrece entornos estandarizados para practicar RL sin tener que
programar tú la simulación. El más clásico es CartPole: un carrito sobre un raíl
con un palo en equilibrio encima, que el agente debe mantener vertical moviendo el
carrito a izquierda o derecha. El entorno termina el episodio si el palo se
inclina demasiado o si el carrito se sale de los límites; el objetivo es aguantar
el mayor número de pasos posible.

```python
import gym

entorno = gym.make("CartPole-v1")
observacion, info = entorno.reset()

print("Forma de la observación:", observacion.shape)
print("Acciones posibles:", entorno.action_space.n)
```

Cada observación tiene cuatro valores: posición y velocidad del carrito, y ángulo
y velocidad angular del palo. Solo hay dos acciones posibles: empujar a la
izquierda o a la derecha.

## Un agente que aprende: Deep Q-Network

Un Deep Q-Network (DQN) usa una red neuronal para estimar, para cada estado y cada
acción posible, cuánta recompensa futura espera obtener el agente si elige esa
acción. Con esa estimación, el agente elige siempre la acción con mayor valor
esperado.

```python
import tensorflow as tf
import numpy as np

modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(24, activation="relu"),
    tf.keras.layers.Dense(24, activation="relu"),
    tf.keras.layers.Dense(2, activation="linear"),
])
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")

def elegir_accion(observacion, epsilon):
    # Con probabilidad epsilon, explora con una acción al azar
    if np.random.random() < epsilon:
        return np.random.randint(2)
    # Si no, explota el conocimiento actual: la acción con mayor valor estimado
    valores_q = modelo.predict(observacion.reshape(1, -1), verbose=0)
    return int(np.argmax(valores_q[0]))
```

El parámetro `epsilon` controla el equilibrio entre "explorar" (probar acciones al
azar para descubrir estrategias nuevas) y "explotar" (usar lo ya aprendido). Al
principio del entrenamiento conviene un `epsilon` alto —el agente no sabe nada
todavía—, y se reduce poco a poco a medida que el agente gana experiencia.

## Resumen de puntos clave

- En aprendizaje por refuerzo no hay respuestas correctas dadas de antemano: el
  agente aprende interactuando con un entorno y recibiendo recompensas.
- OpenAI Gym / Gymnasium ofrece entornos estandarizados, como CartPole, para
  practicar sin programar tú la simulación.
- Un DQN usa una red neuronal para estimar el valor esperado de cada acción en
  cada estado.
- El equilibrio entre exploración y explotación, controlado por `epsilon`, es
  clave para que el agente descubra buenas estrategias sin quedarse atascado en
  las primeras que encuentra.

## Ponlo en práctica

**Modifica** la función `elegir_accion` para que `epsilon` empiece en 1.0 y se
reduzca multiplicándolo por 0.99 después de cada episodio, hasta un mínimo de 0.05.
**Ejecuta** unos cuantos episodios de prueba con acciones aleatorias y **anota**
cuántos pasos aguanta el palo en pie de media, como referencia antes de entrenar el
agente de verdad.

---

**Material fuente de este capítulo**: lección 6 temas 1 a 4 ("Introduction",
"Training Data", "Training Model", "Testing Network") sobre el entorno CartPole.

[⬅ Capítulo anterior](05_tflearn_keras.md) · [Volver al índice](00_indice.md) · [Siguiente capítulo ➡](07_dogs_vs_cats.md)
