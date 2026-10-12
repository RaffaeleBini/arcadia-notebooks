# 03. Visión por computador y redes neuronales

## Por qué importa

Reconocer una prenda de ropa en una foto te parece trivial: miras la imagen y
en un instante sabes si es una zapatilla, un abrigo o un pantalón. Pero
intenta explicar ese proceso con reglas explícitas, del tipo "si hay una
suela plana y una zona alta que envuelve el tobillo, entonces es una bota".
Vas a necesitar cientos de reglas, y aun así fallarán en cuanto cambie el
ángulo, la luz o el color de la prenda.

Este es exactamente el tipo de problema que la programación tradicional
resuelve mal y el aprendizaje automático o ML (Machine Learning) resuelve
bien: en lugar de escribir las reglas a mano, le enseñas al ordenador miles
de ejemplos ya etiquetados ("esto es una bota", "esto es una camiseta") y
dejas que sea la propia red neuronal la que descubra qué patrones de píxeles
distinguen una prenda de otra. En este capítulo construyes tu primer
clasificador de imágenes con una red neuronal densa y, de paso, descubres por
qué ese primer enfoque tiene límites claros que en el capítulo 4 resolverás
con convoluciones.

## El reto de "ver" para un ordenador

Para un ordenador, una imagen no es una zapatilla ni una camisa: es una
cuadrícula de números. Una imagen en escala de grises de 28x28 píxeles es,
en realidad, una matriz de 28 filas por 28 columnas donde cada celda contiene
un valor entre 0 y 255 que indica la intensidad de ese píxel (0 es negro, 255
es blanco). La red neuronal nunca "ve" una prenda de ropa: ve 784 números
(28x28) y tiene que aprender qué combinaciones de esos números suelen
aparecer en una bota y cuáles en una camiseta.

Esa es la clave de la visión por computador con ML: conviertes un problema
visual en un problema numérico, y dejas que el modelo aprenda la relación
entre esos números y la etiqueta correcta a partir de muchísimos ejemplos.

## El dataset Fashion MNIST

Para practicar este primer clasificador usas Fashion MNIST, un conjunto de
datos que sustituye al clásico MNIST de dígitos escritos a mano por prendas
de ropa. Fashion MNIST contiene 70.000 imágenes en escala de grises,
repartidas en 60.000 imágenes de entrenamiento y 10.000 imágenes de prueba,
todas ellas de 28x28 píxeles y clasificadas en diez categorías:

| Etiqueta | Clase (inglés) | Clase (español) |
|---|---|---|
| 0 | T-shirt/top | Camiseta |
| 1 | Trouser | Pantalón |
| 2 | Pullover | Jersey |
| 3 | Dress | Vestido |
| 4 | Coat | Abrigo |
| 5 | Sandal | Sandalia |
| 6 | Shirt | Camisa |
| 7 | Sneaker | Zapatilla |
| 8 | Bag | Bolso |
| 9 | Ankle boot | Botín |

Fíjate en que las etiquetas son números (0 a 9) y no texto. Esto no es
casualidad: a un modelo le resulta más eficiente trabajar con números que
con cadenas de texto, y además evita introducir sesgos hacia un idioma
concreto. TensorFlow incluye Fashion MNIST directamente en su API de
conjuntos de datos, así que cargarlo es tan sencillo como esto:

```python
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

# load_data() devuelve dos tuplas: datos de entrenamiento y datos de prueba
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

¿Por qué dos conjuntos? Porque necesitas comprobar si tu modelo aprendió a
generalizar o si simplemente memorizó los ejemplos de entrenamiento. El
conjunto de prueba contiene imágenes que el modelo no ve durante el
entrenamiento, así que evaluarlo con esas imágenes te da una medida honesta
de qué tan bien funcionaría con datos nuevos.

## Normalización de datos

Cada píxel de Fashion MNIST tiene un valor entre 0 y 255. Las redes
neuronales entrenan mejor y más rápido cuando todos los valores de entrada
están en un rango pequeño y homogéneo, típicamente entre 0 y 1. A este
proceso se le llama **normalización**, y en este caso se reduce a dividir
cada valor de píxel por 255:

```python
training_images = training_images / 255.0
test_images = test_images / 255.0
```

Es importante normalizar el conjunto de entrenamiento y el de prueba
exactamente de la misma forma. Si te saltas este paso, el modelo puede seguir
entrenando, pero con resultados notablemente peores: los gradientes se
mueven en rangos muy distintos y el optimizador tarda más en encontrar buenos
valores para los parámetros.

## Capas de una red densa

Con los datos preparados, el siguiente paso es definir la arquitectura del
modelo. Una red neuronal densa para este problema tiene tres piezas:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

- **Sequential**: indica que las capas se ejecutan una detrás de otra, en
  secuencia, cada una recibiendo la salida de la anterior
- **Flatten**: convierte la matriz de 28x28 en un único vector de 784
  valores. Esta capa no aprende nada, solo reorganiza los datos: es el paso
  obligatorio para poder conectar una imagen 2D a una capa densa, que
  espera una lista plana de números
- **Dense**: una capa "densamente conectada", donde cada neurona recibe
  todos los valores de la capa anterior. La primera capa densa tiene 128
  neuronas, cada una con sus propios parámetros que el modelo ajustará
  durante el entrenamiento
- La última capa **Dense** tiene diez neuronas, una por cada clase de
  Fashion MNIST

Una regla práctica que conviene recordar: la primera capa de tu red debe
coincidir con la forma de tus datos de entrada (por eso `Flatten` recibe
`input_shape=(28, 28)`), y la última capa debe tener tantas neuronas como
clases quieras distinguir. Si entrenas con diez clases pero configuras la
última capa con un número distinto de neuronas, el entrenamiento fallará.

## Funciones de activación y softmax

Cada capa densa necesita una **función de activación**, que decide qué señal
pasa esa neurona a la siguiente capa. Sin funciones de activación no
lineales, apilar capas densas equivaldría a una única operación lineal, por
muchas capas que añadas.

- **ReLU** (Rectified Linear Unit, o Unidad Lineal Rectificada): la regla es
  muy simple, si el valor de entrada es mayor que cero lo deja pasar tal
  cual, y si es cero o negativo lo convierte en cero. Es la activación más
  usada en capas intermedias porque es barata de calcular y funciona bien
  en la práctica
- **Softmax**: se usa en la capa de salida cuando el problema es de
  clasificación multiclase (elegir una entre varias categorías, como aquí).
  Softmax toma un conjunto de valores y los transforma en probabilidades que
  suman 1. Por ejemplo, una salida como `[0.1, 0.1, 9.5, 0.1, ...]` se
  convierte en algo parecido a `[0.001, 0.001, 0.99, 0.001, ...]`: la
  categoría con el valor más alto se queda con casi toda la probabilidad,
  así que puedes leer directamente cuál es la predicción del modelo y con
  cuánta confianza la hace

Piensa en softmax como un "y el ganador es...": en vez de darte diez números
sueltos que tendrías que comparar tú mismo, te entrega directamente una
distribución de probabilidad sobre las diez clases.

## Compilar y entrenar el modelo

Antes de entrenar, compilas el modelo indicando tres piezas:

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
```

- El **optimizador** (`adam` en este caso) es el algoritmo que ajusta los
  parámetros del modelo en cada paso de entrenamiento para reducir el error
- La **función de pérdida** (`sparse_categorical_crossentropy`) mide qué tan
  lejos están las predicciones del modelo respecto a las etiquetas
  correctas. Es la señal que el optimizador usa para saber en qué dirección
  ajustar los parámetros. Se usa la variante "sparse" porque las etiquetas
  son números enteros (0 a 9) y no vectores codificados
- Las **métricas** (`accuracy`, o exactitud) no afectan al entrenamiento,
  solo te informan de qué tan bien lo está haciendo el modelo

`model.fit` es la llamada que dispara el entrenamiento: el modelo recorre
las imágenes de entrenamiento durante cinco épocas (cinco pasadas completas
por todo el conjunto de datos), ajustando sus parámetros en cada una para
mejorar la exactitud.

## Evaluar el modelo y sus límites

Una vez entrenado, compruebas el rendimiento con datos que el modelo no ha
visto:

```python
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Exactitud en el conjunto de prueba: {test_accuracy:.4f}')
```

Es habitual observar una exactitud algo más baja en el conjunto de prueba
que en el de entrenamiento (por ejemplo, 91% en entrenamiento frente a 88%
en prueba). Esa diferencia es una primera señal de sobreajuste
(*overfitting*), un tema que retomarás más adelante en el curso.

Este primer modelo funciona razonablemente bien con Fashion MNIST, pero
arrastra una limitación importante: aprende directamente de los valores de
los píxeles en bruto, sin ningún concepto de "forma" o "borde". Esto implica
que:

- Solo funciona bien si la prenda está centrada en la imagen
- Necesita que todas las imágenes tengan exactamente el mismo tamaño (28x28)
  y estén en escala de grises
- Se comporta mal si la imagen tiene fondo, varios objetos o la prenda está
  desplazada o girada

En otras palabras, esta red aprendió a reconocer patrones de píxeles muy
concretos, no a reconocer las características visuales que hacen que una
bota sea una bota. En el capítulo 4 resuelves justo esa limitación
introduciendo las convoluciones, que enseñan al modelo a fijarse primero en
los rasgos (bordes, curvas, texturas) antes de clasificar.

## Ejemplo de código comentado

```python
import tensorflow as tf

# Cargamos el dataset Fashion MNIST directamente desde Keras
fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalizamos los píxeles de 0-255 a 0-1 para que la red entrene mejor
training_images = training_images / 255.0
test_images = test_images / 255.0

# Definimos una red densa sencilla
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),      # 28x28 -> vector de 784
    tf.keras.layers.Dense(128, activation='relu'),       # capa oculta
    tf.keras.layers.Dense(10, activation='softmax')      # 10 clases de salida
])

# Compilamos indicando optimizador, función de pérdida y métrica
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamos durante 5 épocas
model.fit(training_images, training_labels, epochs=5)

# Evaluamos con datos que el modelo nunca vio
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Exactitud en prueba: {test_accuracy:.4f}')

# Hacemos predicciones sobre el conjunto de prueba
predictions = model.predict(test_images)
print(predictions[0])          # probabilidad de cada una de las 10 clases
print(test_labels[0])          # etiqueta real de esa imagen
```

## Resumen de puntos clave

- Un ordenador no "ve" imágenes, procesa matrices de números: normalizar
  esos números (dividir entre 255) ayuda al modelo a entrenar mejor
- `Flatten` convierte una imagen 2D en un vector 1D para que pueda entrar en
  capas densas, que conectan cada neurona con todas las de la capa anterior
- La activación ReLU introduce no linealidad en las capas ocultas; softmax
  convierte la salida en probabilidades cuando el problema tiene varias
  clases
- Necesitas un optimizador para ajustar los parámetros y una función de
  pérdida para medir el error; la exactitud es la métrica que te dice qué
  tan bien funciona el modelo
- Una red densa entrenada sobre píxeles en bruto solo funciona bien con
  imágenes centradas, del mismo tamaño y sin fondo: esa limitación es la
  puerta de entrada a las redes neuronales convolucionales del capítulo 4

## Ponlo en práctica

Abre el notebook de teoría de esta clase y, después de entrenar el modelo
del ejemplo, cambia el número de neuronas de la capa oculta de 128 a 32 y
vuelve a entrenar. Anota la exactitud en el conjunto de prueba y el tiempo
que tarda cada época. ¿Mejora, empeora o se mantiene igual la exactitud?
¿Coincide con lo que esperabas? Guarda tus conclusiones en una celda de
texto: te van a servir para comparar con la red convolucional del próximo
capítulo.
