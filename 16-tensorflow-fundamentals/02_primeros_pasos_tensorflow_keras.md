# 02. Primeros pasos con TensorFlow y Keras

## Por qué importa este tema

En el capítulo anterior viste que Machine Learning (ML) significa entrenar un modelo a partir de datos en lugar de programar reglas a mano. Ahora toca aterrizar esa idea en código. Antes de construir tu primera red neuronal necesitas conocer la estructura de datos que TensorFlow (TF) usa para representarlo absolutamente todo: el **tensor**. Una vez entiendas qué es un tensor, dar el salto a construir y entrenar un modelo con Keras es cuestión de pocas líneas.

## Qué es un tensor

Un **tensor** es un array multidimensional, similar a un `ndarray` de la biblioteca NumPy, con dos propiedades siempre definidas: un tipo de dato (`dtype`, por ejemplo enteros o decimales) y una forma (`shape`, es decir, cuántos elementos tiene en cada dimensión). Piensa en los tensores como una familia de contenedores de números que escala según la complejidad del dato:

- Un número suelto (un escalar) es un tensor de dimensión cero
- Una lista de números (un vector) es un tensor de una dimensión
- Una tabla de números (una matriz) es un tensor de dos dimensiones
- Una imagen en color, o un lote de varias imágenes, es un tensor de tres o más dimensiones

TensorFlow usa tensores como su estructura central porque necesita un formato único capaz de representar tanto un número aislado como una imagen completa o un lote de miles de ellas, y sobre el que pueda aplicar las mismas operaciones matemáticas optimizadas —sumas, multiplicaciones de matrices, reducciones— de forma eficiente en paralelo, tanto en CPU (Central Processing Unit) como en GPU (Graphics Processing Unit). A diferencia de un array de NumPy, un tensor puede residir en memoria de un acelerador como la GPU y, salvo excepciones, es inmutable: una vez creado, no se modifica su contenido, se crea un tensor nuevo con el resultado de cada operación.

## Operaciones básicas con tensores

Puedes crear tensores y operar con ellos con una sintaxis muy parecida a la de NumPy:

```python
import tensorflow as tf

# Operaciones matemáticas básicas sobre tensores
print(tf.math.add(1, 2))                  # suma
print(tf.math.add([1, 2], [3, 4]))        # suma de vectores, elemento a elemento
print(tf.math.square(5))                  # elevar al cuadrado
print(tf.math.reduce_sum([1, 2, 3]))      # suma de todos los elementos

# TensorFlow sobrecarga los operadores habituales de Python
print(tf.math.square(2) + tf.math.square(3))
```

Cada tensor resultante expone su forma y su tipo de dato:

```python
x = tf.linalg.matmul([[1]], [[2, 3]])  # multiplicación de matrices
print(x)        # tf.Tensor([[2 3]], shape=(1, 2), dtype=int32)
print(x.shape)  # (1, 2)
print(x.dtype)  # <dtype: 'int32'>
```

Y puedes convertir entre tensores y arrays de NumPy sin apenas esfuerzo, gracias a que TensorFlow y NumPy se entienden de forma automática en ambas direcciones:

```python
import numpy as np

ndarray = np.ones([3, 3])

# TensorFlow convierte automáticamente un ndarray de NumPy en tensor
tensor = tf.math.multiply(ndarray, 42)

# NumPy también convierte automáticamente un tensor en ndarray
print(np.add(tensor, 1))

# El método .numpy() convierte un tensor en un array de NumPy de forma explícita
print(tensor.numpy())
```

No necesitas dominar todas las operaciones disponibles: lo importante es que interiorices que, trabajes con lo que trabajes en TensorFlow —números, imágenes, texto convertido a números—, siempre lo harás sobre tensores.

## Construir tu primera red neuronal con Keras

Con esto ya puedes construir el ejemplo de números del capítulo anterior (la relación oculta Y = 2X - 1) usando **Keras**, la interfaz de alto nivel de TensorFlow para definir modelos. Vas a crear la red neuronal más simple posible: una sola capa, con una sola neurona.

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Una capa, con una neurona (unit=1), que recibe un único valor de entrada
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
```

`Sequential` apila capas donde cada una tiene un tensor de entrada y un tensor de salida. Aquí solo hay una capa `Dense` (densa, es decir, totalmente conectada) con una neurona: recibe un número (X) y produce otro número (la predicción de Y).

### Compilar el modelo: optimizador y función de pérdida

Antes de entrenar, tienes que **compilar** el modelo indicando dos piezas que ya viste en el capítulo anterior:

```python
model.compile(optimizer='sgd', loss='mean_squared_error')
```

- `loss` (la función de pérdida) mide cuánto se aleja la predicción del modelo del valor real. Aquí usas el error cuadrático medio (`mean_squared_error`), habitual en problemas de regresión
- `optimizer` es el algoritmo que, a partir de ese error, decide cómo ajustar los parámetros del modelo para el siguiente intento. Aquí usas descenso de gradiente estocástico (`sgd`, Stochastic Gradient Descent)

No necesitas dominar las matemáticas de estas funciones para empezar a usarlas: Keras las encapsula para ti. Con la práctica irás aprendiendo qué combinación de pérdida y optimizador conviene a cada tipo de problema.

### Aportar los datos de entrenamiento

```python
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
```

Fíjate en que estos son los mismos pares X-Y del capítulo anterior, ahora representados como arrays de NumPy, el formato que Keras espera para entrenar.

### Entrenar con fit

```python
model.fit(xs, ys, epochs=500)
```

El método `fit` (ajustar) ejecuta el ciclo completo que describiste en el capítulo anterior: el modelo predice, calcula la pérdida, el optimizador propone un ajuste y se repite el proceso durante las 500 épocas indicadas. Al ejecutarlo en Colab verás cómo el valor de la pérdida baja época a época, señal de que el modelo va aprendiendo.

### Predecir con predict

```python
print(model.predict(np.array([10.0])))
```

Con el modelo ya entrenado, `predict` te devuelve su estimación de Y para un X nuevo que nunca vio durante el entrenamiento. Si la relación real es Y = 2X - 1, para X = 10 esperarías Y = 19, pero lo habitual es obtener un valor muy cercano, como 18,9998, y no el número exacto. Como ya sabes por el capítulo anterior, esto no es un fallo: el modelo trabaja con probabilidades aprendidas de solo seis ejemplos, no con certezas.

## Un vistazo a un modelo algo más complejo

El mismo patrón —capas, `compile`, `fit`, `predict`— funciona igual de bien en modelos mucho más grandes. Por ejemplo, un clasificador de dígitos escritos a mano (el conjunto de datos MNIST) añade más capas y neuronas, pero sigue el mismo ciclo:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),   # aplana la imagen 28x28 en un vector
    tf.keras.layers.Dense(128, activation='relu'),   # capa densa de 128 neuronas
    tf.keras.layers.Dropout(0.2),                    # apaga el 20% de las neuronas al azar para evitar sobreajuste
    tf.keras.layers.Dense(10)                        # una salida por cada dígito (0-9)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
```

No necesitas entender cada capa nueva ahora mismo —`Flatten`, `Dropout`, la función de pérdida `SparseCategoricalCrossentropy`— porque las verás en detalle en las próximas clases dedicadas a visión por computador. Quédate con la idea central: da igual si tu red tiene una neurona o cientos de miles, siempre defines capas, compilas con un optimizador y una pérdida, entrenas con `fit` y predices con `predict`.

## Resumen de puntos clave

- Un tensor es un array multidimensional con un tipo de dato y una forma definidos; es la estructura central de TensorFlow porque permite representar y operar de forma eficiente sobre cualquier tipo de dato numérico
- Los tensores se convierten de forma automática (o explícita, con `.numpy()`) hacia y desde arrays de NumPy
- En Keras, un modelo `Sequential` se construye apilando capas; la más básica es `Dense`, una capa totalmente conectada
- `compile` fija la función de pérdida (mide el error) y el optimizador (decide cómo corregirlo)
- `fit` ejecuta el ciclo de entrenamiento durante el número de épocas indicado; `predict` usa el modelo ya entrenado sobre datos nuevos
- El mismo patrón de capas + compile + fit + predict se aplica desde una red de una sola neurona hasta modelos mucho más complejos

## Ponlo en práctica

Abre un notebook nuevo en Google Colab y reproduce el ejemplo de la neurona única, pero con una relación distinta: Y = 3X + 1. Define tú los arrays `xs` e `ys` con al menos seis pares de valores que cumplan esa fórmula, compila el modelo igual que en el ejemplo y entrénalo con `fit`. Cuando termine, usa `predict` para estimar Y cuando X vale 10. Compara el resultado con el valor exacto (31) y anota la diferencia: es la misma incertidumbre que viste en el capítulo anterior, ahora comprobada por ti con código real.
