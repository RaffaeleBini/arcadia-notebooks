# 04. Redes neuronales convolucionales

## Por qué importa

En el capítulo anterior entrenaste una red densa que clasificaba prendas de
Fashion MNIST con una exactitud aceptable, pero con una condición muy
estricta: la prenda tenía que estar centrada, en escala de grises y ocupando
toda la imagen de 28x28 píxeles. En cuanto le pases una foto real, con fondo,
con la prenda desplazada o rotada, ese modelo falla, porque aprendió a
memorizar patrones de píxeles concretos y no a identificar los rasgos
visuales (bordes, curvas, texturas) que hacen que una bota sea una bota.

Las redes neuronales convolucionales, o CNN (Convolutional Neural Network),
resuelven justo ese problema. En lugar de alimentar directamente los píxeles
en bruto a una capa densa, primero pasan la imagen por un proceso de
filtrado que resalta sus características más relevantes. Esto es lo que
permite que las CNN funcionen bien con imágenes de objetos que no están
perfectamente centrados, y son la base de prácticamente cualquier sistema
moderno de reconocimiento visual, desde clasificar fotos hasta la conducción
autónoma.

## Qué es una convolución

Imagina que tienes una lupa cuadrada pequeña, de tres píxeles por tres
píxeles, y la vas deslizando por toda la imagen, posición a posición. En cada
posición, la lupa no solo mira el píxel central: combina ese píxel con sus
ocho vecinos inmediatos aplicando una fórmula fija, y el resultado se
convierte en el nuevo valor de ese píxel en una imagen transformada. Eso es,
en esencia, una **convolución**: una operación que recorre la imagen entera
aplicando la misma combinación de vecinos en cada posición.

La "fórmula fija" que se aplica en cada posición es el **filtro** (también
llamado kernel), una pequeña matriz de números. Cada valor del filtro
multiplica al píxel de la imagen que tiene debajo, y luego se suman todos los
resultados para obtener el nuevo valor del píxel central. Por ejemplo, con un
filtro de 3x3 como este:

```
[-1, -2, -1]
[ 0,  0,  0]
[ 1,  2,  1]
```

Multiplicas cada uno de los nueve píxeles vecinos por el valor del filtro que
le corresponde, sumas los nueve resultados y ese es el nuevo valor del
píxel central. Este filtro en concreto resalta los bordes horizontales de la
imagen; si usas un filtro con los mismos números pero orientado en columnas
en lugar de filas, resaltas los bordes verticales. La elección de los
números del filtro determina qué tipo de rasgo visual se hace más visible.

Piensa en cada filtro como un detector especializado en un tipo de patrón
concreto: uno puede activarse ante bordes verticales, otro ante curvas, otro
ante manchas de color uniforme. Ninguno de ellos "entiende" que está mirando
una zapatilla: cada uno simplemente responde con más o menos intensidad
según lo que encuentra debajo, píxel a píxel.

## Cómo se aprenden los filtros

Aquí está el giro importante: en una CNN, tú no diseñas a mano los números
de cada filtro. Al principio del entrenamiento, los filtros se inicializan
con valores aleatorios, igual que los pesos de una capa densa. Durante el
entrenamiento, la red ajusta esos valores mediante retropropagación
(*backpropagation*), el mismo mecanismo que ajusta los pesos de las capas
densas, buscando los filtros que mejor ayudan a distinguir una clase de
otra.

El resultado es que, tras el entrenamiento, algunos filtros de las primeras
capas terminan detectando bordes y texturas simples, mientras que los
filtros de capas más profundas combinan esas señales para reconocer formas
más complejas, como el contorno de una manga o la silueta de una suela. Este
proceso de ir extrayendo rasgos cada vez más abstractos capa a capa se llama
**extracción de características** (*feature extraction*), y es la razón por
la que una CNN no necesita que le digas explícitamente qué buscar: lo
aprende de los datos, igual que aprendió los pesos de una red densa.

En TensorFlow, una capa convolucional se define con `Conv2D`:

```python
tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1))
```

Aquí `64` es el número de filtros distintos que la capa aprende en
paralelo (cada uno especializado en un rasgo distinto), y `(3, 3)` es el
tamaño de cada filtro. Fíjate también en que la forma de entrada pasa a ser
`(28, 28, 1)`: el último número es el canal de color (1 para escala de
grises, 3 para imágenes en color con canales rojo, verde y azul).

## Qué aporta el pooling

Después de una capa convolucional, es habitual añadir una capa de
**pooling** (agrupamiento), cuyo trabajo es reducir el tamaño de la imagen
mientras conserva lo esencial de lo que los filtros ya detectaron.

La variante más común es **max pooling**: la imagen se recorre en bloques
(por ejemplo, de 2x2 píxeles) y de cada bloque solo se conserva el valor más
alto. El resultado es una imagen con la mitad de alto y la mitad de ancho,
pero que sigue conservando las zonas donde los filtros respondieron con más
fuerza.

Piensa en el pooling como resumir un párrafo largo en una frase: pierdes
detalle, pero conservas la idea principal. En el caso de una imagen, el
pooling te permite:

- Reducir la cantidad de cálculos que necesitan las siguientes capas, porque
  trabajan con una imagen más pequeña
- Hacer que la red sea más robusta a pequeños desplazamientos del objeto en
  la imagen, porque ya no importa el píxel exacto donde apareció el rasgo,
  solo que apareció en esa zona
- Permitir que capas convolucionales posteriores trabajen sobre una
  representación más compacta y se centren en rasgos cada vez más generales

En Keras, el pooling se añade con `MaxPooling2D`:

```python
tf.keras.layers.MaxPooling2D(2, 2)
```

## Cómo se construye una CNN en Keras

Una CNN típica combina bloques de convolución y pooling, y termina con las
mismas capas densas que ya conoces del capítulo 3:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

Fíjate en la estructura: primero dos bloques de convolución más pooling, que
se encargan de extraer y resumir características visuales, y luego un
`Flatten` seguido de las mismas capas densas del capítulo anterior, que se
encargan de la clasificación final a partir de esas características. Es
exactamente el mismo patrón que describe la capacitación oficial: filtrar
antes de clasificar, en lugar de clasificar directamente sobre los píxeles
en bruto.

Un detalle importante que cambia respecto al capítulo 3: como `Conv2D`
espera imágenes con un canal explícito, necesitas darle a cada imagen la
forma `(28, 28, 1)` en lugar de `(28, 28)`. Con Fashion MNIST, eso significa
reformatear los datos antes de entrenar:

```python
training_images = training_images.reshape(60000, 28, 28, 1) / 255.0
test_images = test_images.reshape(10000, 28, 28, 1) / 255.0
```

## Comparar la CNN con la red densa

El resto del proceso (compilar, entrenar, evaluar) es idéntico al del
capítulo 3:

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
```

Si entrenas ambos modelos con el mismo número de épocas sobre Fashion MNIST,
normalmente verás que la CNN alcanza una exactitud igual o mejor que la red
densa, aunque tarda más por época porque tiene más cálculos que hacer (cada
filtro se aplica sobre toda la imagen). La ganancia se nota todavía más
cuando las imágenes son más grandes, más complejas o el objeto no está
perfectamente centrado, que es justo el escenario donde la red densa del
capítulo 3 empezaba a fallar.

## Construir un clasificador de imágenes completo

Con el modelo ya entrenado, un clasificador de imágenes completo necesita
además una forma clara de interpretar y comunicar sus predicciones. Los
pasos habituales, de principio a fin, son:

1. Cargar y preparar los datos (incluida la normalización y el reshape a
   4 dimensiones que viste antes)
2. Definir la arquitectura del modelo (bloques de convolución y pooling,
   más capas densas)
3. Compilar el modelo con un optimizador, una función de pérdida y una
   métrica
4. Entrenar con `model.fit`, y validar con datos de prueba mediante
   `model.evaluate`
5. Generar predicciones con `model.predict` sobre imágenes nuevas
6. Interpretar la salida: como la última capa usa softmax, cada predicción
   es una lista de probabilidades, una por clase, y la clase predicha es la
   que tiene la probabilidad más alta (con `np.argmax`)

Por ejemplo, para clasificar una única imagen nueva necesitas recordar que
Keras espera siempre un lote (*batch*) de imágenes, aunque sea de una sola:

```python
import numpy as np

img = test_images[1]                    # una sola imagen, forma (28, 28, 1)
img_batch = np.expand_dims(img, 0)       # forma (1, 28, 28, 1): un lote de una imagen

prediction = model.predict(img_batch)
predicted_class = np.argmax(prediction[0])
print(f'Clase predicha: {predicted_class}, probabilidad: {prediction[0][predicted_class]:.2%}')
```

Con esto tienes un clasificador de imágenes completo: desde los datos en
bruto hasta una predicción interpretable sobre una imagen que el modelo
nunca vio.

## Ejemplo de código comentado

```python
import numpy as np
import tensorflow as tf

# Cargamos y normalizamos Fashion MNIST, igual que en el capítulo 3
fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Conv2D necesita un canal explícito: pasamos de (28, 28) a (28, 28, 1)
training_images = training_images.reshape(60000, 28, 28, 1) / 255.0
test_images = test_images.reshape(10000, 28, 28, 1) / 255.0

# Arquitectura CNN: dos bloques de convolución + pooling, y luego clasificación densa
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Exactitud CNN en prueba: {test_accuracy:.4f}')

# Predicción sobre una única imagen nueva
img_batch = np.expand_dims(test_images[1], 0)
prediction = model.predict(img_batch)
predicted_class = np.argmax(prediction[0])
print(f'Clase predicha: {predicted_class} (real: {test_labels[1]})')
```

## Resumen de puntos clave

- Una convolución desliza un filtro (una pequeña matriz de números) por toda
  la imagen, combinando cada píxel con sus vecinos para resaltar un rasgo
  visual concreto, como un borde vertical u horizontal
- Los filtros no se diseñan a mano: se inicializan con valores aleatorios y
  se ajustan durante el entrenamiento mediante retropropagación, igual que
  los pesos de una capa densa
- El pooling (normalmente max pooling) reduce el tamaño de la imagen
  quedándose con el valor más alto de cada bloque de píxeles, lo que aligera
  los cálculos y da robustez frente a pequeños desplazamientos del objeto
- Una CNN típica combina varios bloques de `Conv2D` más `MaxPooling2D` para
  extraer características, y termina con `Flatten` y capas `Dense` para
  clasificar, igual que la red del capítulo 3
- Un clasificador de imágenes completo no termina en `model.fit`: incluye
  preparar los datos con la forma correcta, entrenar, evaluar con datos no
  vistos e interpretar la predicción final con `np.argmax` sobre la salida
  de softmax

## Ponlo en práctica

Abre el notebook de teoría de esta clase, entrena la CNN de este capítulo
sobre Fashion MNIST durante cinco épocas y anota su exactitud en el conjunto
de prueba. Compárala con la exactitud que obtuviste con la red densa del
capítulo 3 usando el mismo número de épocas. Después, cambia el número de
filtros de la primera capa `Conv2D` de 64 a 32 y vuelve a entrenar. ¿Cambia
mucho la exactitud? ¿Y el tiempo de entrenamiento por época? Escribe tus
conclusiones: te van a servir de base para los ejercicios del notebook de
práctica, donde vas a modificar la arquitectura de la CNN por tu cuenta.
