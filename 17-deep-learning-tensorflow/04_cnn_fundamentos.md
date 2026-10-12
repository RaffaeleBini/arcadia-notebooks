[⬅ Volver al índice](00_indice.md)

# Capítulo 4 — Redes neuronales convolucionales: fundamentos

## Por qué te importa esto

Las redes densas que has usado hasta ahora tratan cada píxel como un dato
independiente, sin tener en cuenta que un píxel y sus vecinos suelen formar parte
del mismo trazo, borde u objeto. Las redes convolucionales, o Convolutional Neural
Network (CNN), están diseñadas específicamente para explotar esa estructura
espacial. Son la base de casi cualquier sistema de visión por computador que uses
hoy: reconocimiento facial, coches autónomos, diagnóstico por imagen o búsqueda
visual.

## Cómo "ve" una CNN (con una analogía)

Imagina que examinas una fotografía con una lupa pequeña, desplazándola poco a poco
por toda la imagen sin saltarte ningún trozo. En cada posición, la lupa te permite
fijarte en un detalle local: un borde, una esquina, una textura. Una capa
convolucional hace exactamente eso: desliza un filtro (una lupa) de tamaño
pequeño —por ejemplo 3x3 píxeles— por toda la imagen, y en cada posición calcula un
único número que resume lo que "ve" ese filtro ahí. El resultado es un "mapa de
características" que señala dónde aparece ese patrón concreto en la imagen
completa.

Después de detectar patrones locales, una capa de "pooling" (agrupación) resume
esa información reduciendo el tamaño del mapa: por ejemplo, quedándose solo con el
valor máximo de cada bloque de 2x2. Es como resumir un párrafo en su frase más
importante: pierdes detalle, pero conservas lo esencial y reduces el volumen de
información a procesar.

## Construye tu primera CNN con Keras

```python
import tensorflow as tf

(x_entreno, y_entreno), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Las capas convolucionales esperan un canal de color explícito, aunque sea 1 (blanco y negro)
x_entreno = (x_entreno / 255.0).reshape(-1, 28, 28, 1)
x_test = (x_test / 255.0).reshape(-1, 28, 28, 1)

modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax"),
])

modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

modelo.fit(x_entreno, y_entreno, epochs=5, validation_split=0.1)
modelo.evaluate(x_test, y_test)
```

Fíjate en el patrón que se repite: convolución, pooling, convolución, pooling.
Cada bloque aprende patrones más complejos que el anterior —la primera capa
detecta bordes y texturas simples; la segunda combina esos bordes en formas más
elaboradas—. Al final, `Flatten` convierte el mapa de características resultante
en un vector, y una capa densa hace la clasificación final, exactamente igual que
en el capítulo 1.

## Resumen de puntos clave

- Una capa convolucional desliza filtros pequeños por toda la imagen para detectar
  patrones locales (bordes, texturas, formas).
- Una capa de pooling reduce el tamaño del mapa de características conservando la
  información más relevante.
- Apilar varios bloques de convolución y pooling permite que la red aprenda
  patrones cada vez más complejos y abstractos.
- Al final de una CNN casi siempre encuentras una capa `Flatten` seguida de una o
  varias capas densas, igual que en las redes que ya conoces.

## Ponlo en práctica

**Añade** una tercera pareja de capas `Conv2D` + `MaxPooling2D` a la red anterior,
con 128 filtros. **Entrena** el modelo y **anota** si la exactitud mejora respecto
a la versión de dos bloques, y cuánto tarda cada época en tu equipo.

---

**Material fuente de este capítulo**: lección 4 temas 1 y 2 ("Convolutional Neural
Networks Basics", "Convolutional Neural Networks with TensorFlow");
`CNN_tensorFlow_Tutorial.ipynb`.

[⬅ Capítulo anterior](03_rnn_lstm.md) · [Volver al índice](00_indice.md) · [Siguiente capítulo ➡](05_tflearn_keras.md)
