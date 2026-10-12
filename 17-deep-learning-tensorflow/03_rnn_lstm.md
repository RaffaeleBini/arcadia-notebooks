[⬅ Volver al índice](00_indice.md)

# Capítulo 3 — Redes neuronales recurrentes (RNN/LSTM)

## Por qué te importa esto

Cuando lees la palabra "banco", su significado depende de lo que has leído antes:
"me senté en el banco del parque" no es lo mismo que "fui al banco a sacar dinero".
Las redes que has visto hasta ahora tratan cada entrada de forma aislada, sin
memoria de lo anterior. Para secuencias —texto, audio, series temporales de bolsa,
sensores— necesitas una arquitectura que recuerde el contexto. Esa es la Red
Neuronal Recurrente, o Recurrent Neural Network (RNN).

## Una red con memoria (con una analogía)

Imagina que lees un libro y, al terminar cada página, escribes un resumen de una
línea de todo lo que llevas leído hasta ese punto. Al llegar a la última página, no
partes de cero: usas ese resumen acumulado para entender el desenlace. Una RNN
hace algo parecido: procesa una secuencia paso a paso, y en cada paso combina la
entrada actual con un "resumen" (el estado oculto) de todo lo que ha visto antes.

El problema de una RNN simple es que ese resumen tiende a "olvidar" lo que pasó
hace muchos pasos, igual que te costaría recordar un detalle del capítulo uno al
llegar al capítulo veinte. La solución se llama Long Short-Term Memory (LSTM):
una célula recurrente con "puertas" que deciden explícitamente qué información
guardar, qué olvidar y qué dejar pasar hacia la salida. Piensa en esas puertas como
un archivador con tres empleados: uno decide qué papeles nuevos guardar, otro
decide qué papeles antiguos tirar, y un tercero decide qué papeles enseñar ahora
mismo.

## Construye una LSTM con Keras

Vas a aplicar una LSTM a un problema que ya conoces —MNIST— pero con un giro:
en lugar de darle la imagen completa de golpe, se la vas a dar fila a fila, como si
fuera una secuencia de veintiocho pasos de veintiocho valores cada uno. Así
practicas el patrón de una secuencia sin necesitar un dataset de texto nuevo.

```python
import tensorflow as tf

(x_entreno, y_entreno), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_entreno, x_test = x_entreno / 255.0, x_test / 255.0

modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(10, activation="softmax"),
])

modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

modelo.fit(x_entreno, y_entreno, epochs=3, validation_split=0.1)
modelo.evaluate(x_test, y_test)
```

Cada imagen de 28x28 se interpreta aquí como una secuencia de 28 pasos de tiempo,
cada uno con un vector de 28 valores: la LSTM recorre fila a fila, actualizando su
estado interno, hasta que la última fila produce el resumen final que alimenta la
capa de salida.

## GRU: una alternativa más ligera

Gated Recurrent Unit (GRU) es una variante de la LSTM con menos puertas internas
—y por tanto menos parámetros que entrenar—, que en la práctica suele rendir de
forma parecida en muchas tareas. Cambiar de una a otra es tan sencillo como
sustituir la capa:

```python
modelo_gru = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(10, activation="softmax"),
])
```

## Resumen de puntos clave

- Una RNN procesa secuencias manteniendo un estado oculto que resume lo visto
  hasta el momento.
- Una RNN simple tiende a olvidar información lejana en la secuencia; LSTM y GRU
  resuelven ese problema con mecanismos de puertas.
- En Keras, cambiar de una arquitectura densa a una recurrente es tan sencillo
  como sustituir la capa `Dense`/`Flatten` inicial por `LSTM` o `GRU`.
- Tratar una imagen como secuencia de filas es un buen ejercicio para entender el
  concepto de RNN antes de aplicarlo a datos de texto o series temporales reales.

## Ponlo en práctica

**Sustituye** la capa `LSTM` del primer ejemplo por una capa `GRU`, **entrena**
ambos modelos con el mismo número de épocas, y **compara** su exactitud final y su
tiempo de entrenamiento. ¿Cuál eligirías tú para un proyecto con recursos
limitados?

---

**Material fuente de este capítulo**: lección 3 temas 1 y 2 ("Recurrent Neural
Networks", "RNN Example in TensorFlow").

[⬅ Capítulo anterior](02_sentiment_analysis.md) · [Volver al índice](00_indice.md) · [Siguiente capítulo ➡](04_cnn_fundamentos.md)
