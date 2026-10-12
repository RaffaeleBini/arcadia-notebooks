[⬅ Volver al índice](00_indice.md)

# Capítulo 7 — Visión por computador real: perros vs. gatos

## Por qué te importa esto

Hasta ahora has trabajado con MNIST: imágenes pequeñas, en blanco y negro, ya
perfectamente recortadas y normalizadas. El mundo real no viene así. Las fotos que
sube la gente tienen tamaños distintos, colores, fondos desordenados y sujetos mal
centrados. En este capítulo das el salto a un dataset de fotografías reales
—perros contra gatos, uno de los retos clásicos de Kaggle— y aprendes el proceso
completo: preparar datos imperfectos, entrenar una CNN, y evaluar sus aciertos y
sus errores.

## El reto de los datos reales (con una analogía)

Imagina que le pides a cien personas distintas que te envíen una foto de su
mascota: recibirás fotos verticales y horizontales, de día y de noche, con la
mascota en el centro o en una esquina. Antes de poder compararlas de forma justa,
necesitas ponerlas todas en el mismo "formato de ficha": mismo tamaño, mismo tipo
de color. Ese proceso se llama preprocesamiento, y en visión por computador es
tan importante como la propia arquitectura de la red.

## Prepara el dataset

El dataset "Dogs vs. Cats" de Kaggle organiza las imágenes en carpetas según su
clase, y cada imagen tiene un tamaño distinto. `image_dataset_from_directory` de
Keras se encarga de leer esa estructura de carpetas, redimensionar cada imagen al
mismo tamaño, y organizarlas en lotes listos para entrenar.

```python
import tensorflow as tf

TAMANO_IMAGEN = (128, 128)
TAMANO_LOTE = 32

dataset_entreno = tf.keras.utils.image_dataset_from_directory(
    "dogs_vs_cats/train",
    image_size=TAMANO_IMAGEN,
    batch_size=TAMANO_LOTE,
    label_mode="binary",
)

# Normalizamos los píxeles a un rango de 0 a 1, igual que en capítulos anteriores
normalizacion = tf.keras.layers.Rescaling(1.0 / 255)
dataset_entreno = dataset_entreno.map(lambda x, y: (normalizacion(x), y))
```

## Aumento de datos: multiplica tus ejemplos sin fotos nuevas

Con pocos miles de fotos, una CNN grande puede memorizar el dataset de
entrenamiento en lugar de aprender patrones generales —lo que se llama
sobreajuste—. El aumento de datos ("data augmentation") genera variaciones
aleatorias de cada foto —girada, volteada, con el zoom ligeramente distinto— para
que la red vea más variedad sin necesitar fotos nuevas.

```python
aumento_datos = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

## Construye y entrena la CNN

```python
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    aumento_datos,
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
modelo.fit(dataset_entreno, epochs=10, validation_data=dataset_validacion)
```

Fíjate en que la salida vuelve a ser una sola neurona con activación sigmoide,
igual que en el clasificador de sentimientos: perro contra gato es, de nuevo, un
problema de dos clases.

## Interpreta los resultados

Una exactitud del 85% suena bien, pero no te dice qué tipo de errores comete el
modelo. Visualizar algunas predicciones —imágenes reales junto a la etiqueta
predicha— te ayuda a detectar patrones en los fallos: quizás confunde gatos de
pelo largo con perros pequeños, o falla más con fotos oscuras.

## Resumen de puntos clave

- Los datos reales llegan con tamaños y formatos inconsistentes;
  `image_dataset_from_directory` automatiza gran parte de ese preprocesamiento.
- El aumento de datos genera variaciones de tus imágenes para reducir el
  sobreajuste sin necesitar más fotos.
- La arquitectura de la CNN es la misma idea del capítulo 4, solo con más bloques
  de convolución para imágenes más grandes y complejas.
- Visualizar predicciones concretas revela patrones de error que una sola métrica
  de exactitud no muestra.

## Ponlo en práctica

**Escoge** diez imágenes del conjunto de validación, **muéstralas** con
`matplotlib` junto a la etiqueta real y la predicha por tu modelo, y **identifica**
al menos un caso en el que el modelo se equivoque. **Escribe** una hipótesis de
por qué crees que falló en ese caso concreto.

---

**Material fuente de este capítulo**: lección 7 temas 1 a 4 ("Introduction and
Preprocessing", "Building the Network", "Training the Network", "Using the
Network").

[⬅ Capítulo anterior](06_reinforcement_learning.md) · [Volver al índice](00_indice.md) · [Siguiente capítulo ➡](08_cnn_3d_imagen_medica.md)
