[⬅ Volver al índice](00_indice.md)

# Capítulo 8 — CNN 3D e imagen médica: detección de cáncer de pulmón

## Por qué te importa esto

Una tomografía computarizada, o Computed Tomography (CT), no es una sola foto: es
una pila de decenas o cientos de cortes transversales del cuerpo, apilados como las
páginas de un libro. Detectar un nódulo pulmonar sospechoso significa mirar esa
pila entera, no un corte suelto, porque un nódulo puede ser visible en varios
cortes consecutivos y casi invisible en uno solo. Este capítulo te enseña a
adaptar una CNN para trabajar con volúmenes tridimensionales en lugar de imágenes
planas, con un caso real: el concurso de Kaggle sobre detección de cáncer de
pulmón (Data Science Bowl 2017).

## De 2D a 3D (con una analogía)

Piensa en la diferencia entre mirar una única fotografía de una tarta y sostener la
tarta entera en tus manos, girándola para ver su forma completa por todos los
lados. Una imagen 2D solo te deja ver un corte; un volumen 3D te deja apreciar cómo
cambia la forma de algo a medida que avanzas en profundidad. Una CNN 3D hace
exactamente eso: en lugar de deslizar un filtro plano (alto x ancho) sobre una
imagen, desliza un filtro con volumen (alto x ancho x profundidad) sobre toda la
pila de cortes, capturando así patrones que se extienden en las tres dimensiones.

## El reto de los datos médicos reales

Los datos de este proyecto llegan en formato DICOM —el estándar de imagen médica—
y presentan retos que no viste en capítulos anteriores:

- **Volúmenes de tamaño distinto**: cada paciente tiene un número diferente de
  cortes, así que hay que redimensionar también en la dimensión de profundidad, no
  solo en alto y ancho.
- **Volumen de datos**: el dataset completo de la competición pesa decenas de
  gigabytes, demasiado para procesarlo entero en una máquina modesta sin
  optimizar el pipeline.
- **Ruido y variabilidad**: distintos escáneres, distintos protocolos de captura,
  y pacientes con anatomías distintas.

Por eso, para practicar el pipeline completo sin depender de una descarga de
gigabytes ni de credenciales de Kaggle, en el notebook de esta clase vas a trabajar
con volúmenes sintéticos generados con las mismas dimensiones que usarías con datos
reales. La lógica —redimensionar, normalizar, entrenar una CNN 3D— es idéntica; lo
único que cambia es el origen de los datos.

## Redimensiona un volumen 3D

```python
import tensorflow as tf

def redimensionar_volumen(volumen, tamano_destino=(64, 64, 32)):
    """Redimensiona alto, ancho y profundidad de un volumen 3D."""
    volumen = tf.expand_dims(volumen, axis=-1)  # añade el canal (blanco y negro)
    volumen = tf.image.resize(volumen, tamano_destino[:2])  # alto y ancho
    volumen = tf.transpose(volumen, perm=[2, 0, 1, 3])
    volumen = tf.image.resize(volumen, [tamano_destino[2], tamano_destino[0]])
    volumen = tf.transpose(volumen, perm=[1, 2, 0, 3])
    return volumen
```

Redimensionar la profundidad requiere un pequeño truco: como `tf.image.resize`
solo trabaja sobre las dos primeras dimensiones espaciales, transponemos el
volumen para poner la profundidad donde antes estaba el alto, redimensionamos, y
transponemos de vuelta.

## Construye una CNN 3D

```python
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 32, 1)),
    tf.keras.layers.Conv3D(16, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling3D(pool_size=2),
    tf.keras.layers.Conv3D(32, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling3D(pool_size=2),
    tf.keras.layers.GlobalAveragePooling3D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

modelo.compile(optimizer="adam", loss="binary_crossentropy",
               metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
```

`Conv3D` y `MaxPooling3D` son las versiones tridimensionales de las capas que ya
conoces del capítulo 4: la única diferencia es que el filtro se mueve también en
profundidad. Añadimos la métrica AUC (Area Under the Curve, área bajo la curva)
porque es la métrica que usa la competición real de Kaggle para evaluar los
envíos: mide qué tan bien distingue el modelo entre casos positivos y negativos,
más allá de un simple porcentaje de aciertos.

## Resumen de puntos clave

- Una CT es un volumen 3D formado por muchos cortes 2D apilados; una CNN 3D
  procesa ese volumen completo en lugar de un corte suelto.
- Los datos médicos reales necesitan redimensionar también la dimensión de
  profundidad, no solo alto y ancho.
- `Conv3D`, `MaxPooling3D` y `GlobalAveragePooling3D` son las versiones
  tridimensionales de las capas que ya conocías.
- AUC es la métrica habitual en problemas médicos de detección, porque evalúa la
  capacidad de distinguir positivos de negativos con más matiz que la exactitud.

## Ponlo en práctica

**Cambia** el tamaño destino del volumen de `(64, 64, 32)` a `(32, 32, 16)` en la
función `redimensionar_volumen`, **ajusta** la capa `Input` del modelo para que
coincida con el nuevo tamaño, y **compara** cuánto se reduce el tiempo de
entrenamiento por época frente al tamaño original.

---

**Material fuente de este capítulo**: lección 8 temas 1 a 6 ("Introduction",
"Reading Files", "Visualizing", "Resizing Data", "Preprocessing Data", "Running the
Network"); `first-pass-through-data-w-3d-convnet.ipynb`.

[⬅ Capítulo anterior](07_dogs_vs_cats.md) · [Volver al índice](00_indice.md) · [Siguiente capítulo ➡](09_chatbot.md)
