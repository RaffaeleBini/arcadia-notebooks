[⬅ Volver al índice](00_indice.md)

# Capítulo 1 — Fundamentos de IA, redes neuronales y TensorFlow

## Por qué te importa esto

Cada vez que tu móvil reconoce tu cara para desbloquearse, un traductor convierte
una frase al instante, o una plataforma de streaming acierta con su recomendación,
hay una red neuronal entrenada detrás. La Inteligencia Artificial (IA) ya no es
ciencia ficción: es la tecnología que decide qué ves, qué compras y con quién
hablas. Y la pieza que hace posible construir esas redes de forma práctica se llama
TensorFlow. En este capítulo das tus dos primeros pasos: entiendes qué es una red
neuronal y entrenas la tuya con solo unas líneas de código.

## Qué es una red neuronal (con una analogía)

Imagina un comité de diez personas que debe decidir si una foto muestra un gato.
Cada persona del comité se fija en una pista distinta —las orejas, los bigotes, la
forma de los ojos— y da su voto con más o menos convicción. Al final, alguien suma
esos votos ponderados y saca una conclusión. Una red neuronal artificial funciona
igual: está formada por "neuronas" organizadas en capas, cada una se fija en un
patrón distinto de los datos de entrada, y sus votos ponderados (llamados "pesos")
se combinan capa tras capa hasta producir una respuesta final.

El aprendizaje profundo, o Deep Learning (DL), es simplemente usar redes con muchas
capas apiladas: cuantas más capas, más patrones complejos puede aprender a
reconocer la red por sí misma, sin que tú tengas que programar esas reglas a mano.

TensorFlow es la caja de herramientas —de código abierto y creada por Google— que
te permite definir, entrenar y usar estas redes sin tener que escribir tú el
álgebra que hay detrás de cada "voto" del comité.

## Instala TensorFlow en tu equipo

Necesitas Python instalado y, en tu terminal, ejecutar:

```bash
pip install tensorflow
```

Con eso basta para trabajar con este manual. Si tu ordenador tiene una tarjeta
gráfica (Graphics Processing Unit, GPU) de NVIDIA y quieres entrenar más rápido,
TensorFlow la detecta automáticamente siempre que tengas instalados los
controladores CUDA correspondientes; no es obligatorio para seguir el curso.

Para comprobar que todo funciona, abre Python y ejecuta:

```python
import tensorflow as tf
print(tf.constant("¡Hola, TensorFlow!"))
```

Si ves el mensaje impreso sin errores, ya puedes entrenar tu primera red.

## Tu primera red neuronal con Keras

Keras es la interfaz de alto nivel integrada en TensorFlow: te deja definir una red
como una lista de capas, sin preocuparte todavía de los detalles internos. Vas a
entrenar una red que reconoce dígitos escritos a mano, usando el dataset MNIST
—setenta mil imágenes de números del cero al nueve que ya vienen incluidas en
TensorFlow—.

```python
import tensorflow as tf

# Cargamos el dataset MNIST: imágenes de 28x28 píxeles con su dígito correcto
(x_entreno, y_entreno), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizamos los píxeles a un rango de 0 a 1 para que la red aprenda mejor
x_entreno, x_test = x_entreno / 255.0, x_test / 255.0

# Definimos la red: aplana la imagen y la pasa por dos capas densas
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

modelo.fit(x_entreno, y_entreno, epochs=5)
modelo.evaluate(x_test, y_test)
```

En cinco épocas —cinco pasadas completas por los datos de entrenamiento— este
modelo suele superar el 97% de exactitud reconociendo dígitos que nunca ha visto.

## Qué ocurre por dentro: tensores, pesos y gradientes

Keras esconde muchos detalles a propósito, pero entenderlos te ayuda a depurar
problemas más adelante. Un tensor es, sencillamente, un array multidimensional: un
número suelto es un tensor de dimensión cero, una lista es de dimensión uno, una
imagen en blanco y negro es de dimensión dos, y una imagen en color es de
dimensión tres. TensorFlow representa así tanto tus datos de entrada como los
pesos internos de la red.

Cada capa densa de las que usaste arriba hace, en el fondo, una operación muy
simple: multiplica la entrada por una matriz de pesos, suma un sesgo, y aplica una
función de activación como ReLU (Rectified Linear Unit) que introduce
no-linealidad —sin ella, apilar capas no serviría de nada, porque varias
transformaciones lineales seguidas siguen siendo una transformación lineal—.

Para entrenar la red, TensorFlow calcula el error entre la predicción y la
respuesta correcta (la "función de pérdida") y usa un algoritmo llamado descenso de
gradiente para ajustar los pesos en la dirección que reduce ese error, paso a
paso, durante cada época.

## Construye una capa densa con tus propias manos

Vamos a reproducir, con las piezas de bajo nivel de TensorFlow, exactamente lo que
Keras hacía por ti. Así entiendes qué se esconde detrás de `Dense`.

```python
import tensorflow as tf

# Mismos datos que antes, aplanados a vectores de 784 valores (28 x 28)
(x_entreno, y_entreno), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_entreno = (x_entreno / 255.0).reshape(-1, 784).astype("float32")
y_entreno = tf.keras.utils.to_categorical(y_entreno, 10)

# Pesos y sesgos como variables entrenables, inicializados al azar
pesos = tf.Variable(tf.random.normal([784, 10], stddev=0.1))
sesgos = tf.Variable(tf.zeros([10]))

optimizador = tf.keras.optimizers.Adam(learning_rate=0.01)

def prediccion(x):
    return tf.nn.softmax(tf.matmul(x, pesos) + sesgos)

# Entrenamos durante 3 épocas, en lotes de 128 imágenes
lote = 128
for epoca in range(3):
    for i in range(0, len(x_entreno), lote):
        x_lote = x_entreno[i:i + lote]
        y_lote = y_entreno[i:i + lote]
        with tf.GradientTape() as cinta:
            y_predicha = prediccion(x_lote)
            perdida = tf.keras.losses.categorical_crossentropy(y_lote, y_predicha)
        gradientes = cinta.gradient(perdida, [pesos, sesgos])
        optimizador.apply_gradients(zip(gradientes, [pesos, sesgos]))
    print(f"Época {epoca + 1} completada")
```

`tf.GradientTape` es la "cinta" que graba las operaciones matemáticas mientras se
ejecutan, para poder calcular después el gradiente —la dirección de ajuste— de
cada peso. Es exactamente lo que Keras hace internamente cada vez que llamas a
`fit`.

## Resumen de puntos clave

- Una red neuronal combina "votos" ponderados de neuronas organizadas en capas
  para producir una predicción.
- El aprendizaje profundo apila muchas capas para reconocer patrones complejos sin
  programarlos a mano.
- TensorFlow te da dos niveles de trabajo: Keras (alto nivel, rápido de escribir) y
  las operaciones de bajo nivel (tensores, variables, `GradientTape`) que Keras usa
  por debajo.
- Entrenar consiste en ajustar pesos y sesgos, época tras época, para reducir el
  error entre la predicción y la respuesta correcta.

## Ponlo en práctica

Toma el primer ejemplo de este capítulo (la red con Keras) y **cambia** el número
de neuronas de la capa oculta de 128 a 32, y el número de épocas de 5 a 10.
**Ejecuta** el entrenamiento de nuevo y **compara** la exactitud final. ¿Mejora,
empeora o se queda igual? Anota tu hipótesis antes de ejecutar el código, y
comprueba si acertaste.

---

**Material fuente de este capítulo**: lección 1 tema 1 ("Introduction to Deep
Learning with Neural Networks and TensorFlow"); lección 2 temas 1, 2, 3, 4, 9 y 10.

[⬅ Volver al índice](00_indice.md) · [Siguiente capítulo ➡](02_sentiment_analysis.md)
