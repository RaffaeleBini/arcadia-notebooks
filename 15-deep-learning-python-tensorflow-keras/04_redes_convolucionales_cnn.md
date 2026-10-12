[← Anterior: [Prepara tus propios datos](03_prepara_tus_datos.md) · [Índice](00_indice.md) · Siguiente: [TensorBoard: visualiza y depura tus modelos →](05_tensorboard_visualiza.md)]

# 4. Redes Neuronales Convolucionales (CNN)

## Por qué te importa esto

El modelo del capítulo 2 (MNIST) funcionaba con capas `Dense`, donde cada píxel se conecta con cada neurona de la siguiente capa. Eso funciona con imágenes pequeñas y simples, pero se vuelve inviable con fotos reales: una imagen de 50x50 en escala de grises ya tiene 2.500 valores de entrada, y una capa `Dense` conectada a todos ellos crece de tamaño muy rápido, sin aprovechar que una imagen tiene estructura espacial (los píxeles cercanos están relacionados entre sí). Las Redes Neuronales Convolucionales (CNN, Convolutional Neural Network) resuelven este problema, y son la base de casi todo lo que hoy llamamos "visión por computador".

## La convolución, explicada con una analogía

Imagina que examinas una fotografía enorme con una lupa pequeña, desplazándola metódicamente de izquierda a derecha y de arriba abajo. En cada posición, anotas lo que ves: un borde, una textura, un cambio de color. Eso es exactamente lo que hace un filtro convolucional (también llamado kernel): es una pequeña matriz (por ejemplo, de 3x3 píxeles) que se desliza sobre toda la imagen, calculando en cada posición un producto punto entre el filtro y la porción de imagen que cubre. El resultado es un "mapa de características" (feature map) que resalta dónde aparece el patrón que ese filtro detecta.

La gran diferencia con una capa `Dense` es que el mismo filtro se reutiliza en toda la imagen: si aprende a detectar un borde vertical, lo detecta esté donde esté en la foto, sin necesitar una neurona distinta para cada posición.

## Pooling: comprime sin perder lo importante

Después de aplicar varios filtros, obtienes varios mapas de características, cada uno casi tan grande como la imagen original. Para reducir esa cantidad de información sin perder lo esencial, aplicas pooling. La variante más común es el max pooling: divide el mapa en pequeños bloques (por ejemplo, de 2x2) y se queda solo con el valor más alto de cada bloque.

Esto logra dos cosas: reduce el tamaño de los datos (y por tanto el coste de cómputo) y da a la red cierta invariancia a pequeños desplazamientos, es decir, si el gato está un poco más a la izquierda en una foto que en otra, el modelo sigue reconociéndolo igual.

## La arquitectura completa de una CNN

Una CNN típica encadena estos bloques varias veces, y termina con capas densas para la decisión final:

```
Entrada (imagen) → [Conv2D → Activación → MaxPooling] × N → Flatten → Dense → Salida
```

Las primeras capas convolucionales aprenden patrones simples (bordes, texturas). Las capas más profundas combinan esos patrones simples en conceptos más abstractos (una oreja, un ojo, una silueta completa). Solo al final, cuando ya tienes características de alto nivel, tiene sentido usar una capa `Dense`.

## Construye tu clasificador de gatos y perros

Retoma el dataset que preparaste en el capítulo anterior:

```python
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Recupera los datos ya procesados
with open("X.pickle", "rb") as archivo:
    X = pickle.load(archivo)
with open("y.pickle", "rb") as archivo:
    y = pickle.load(archivo)

# Normaliza los píxeles de [0, 255] a [0, 1]
X = X / 255.0

model = Sequential([
    # Primer bloque convolucional: 64 filtros de 3x3
    Conv2D(64, (3, 3), input_shape=X.shape[1:]),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Segundo bloque convolucional: profundiza el aprendizaje de patrones
    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Aplana los mapas de características en un vector
    Flatten(),
    Dense(64),
    Activation('relu'),

    # Capa de salida: una neurona con sigmoid para clasificación binaria (gato/perro)
    Dense(1, activation='sigmoid'),
])
```

Fíjate en `input_shape=X.shape[1:]`: le dice a la primera capa qué forma esperar (50, 50, 1), sin incluir el número de imágenes.

## Compila, entrena y evalúa

```python
model.compile(
    loss='binary_crossentropy',   # apropiada para clasificación de dos clases
    optimizer='adam',
    metrics=['accuracy'],
)

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)
```

`validation_split=0.1` reserva automáticamente el 10% de tus datos como conjunto de validación: el modelo no entrena con ellos, así que te dan una medida honesta de cómo se comporta con datos nuevos, en cada epoch.

## Resumen de puntos clave

- Una capa `Conv2D` aplica filtros deslizantes que aprenden a detectar patrones visuales (bordes, texturas, formas)
- El pooling (normalmente max pooling) reduce el tamaño de los datos y aporta invariancia a pequeños desplazamientos
- Una CNN encadena bloques de convolución y pooling, y termina con capas densas para la decisión final
- Las primeras capas aprenden patrones simples, las capas profundas combinan esos patrones en conceptos abstractos
- Para clasificación binaria, usa una neurona de salida con activación sigmoid y `binary_crossentropy` como función de pérdida

## Ponlo en práctica

Retoma el clasificador de gatos y perros de este capítulo y añade un tercer bloque convolucional (`Conv2D` + `Activation` + `MaxPooling2D`) antes de la capa `Flatten`. Entrena el modelo con `validation_split=0.1` durante 10 epochs y compara el `val_accuracy` final contra el que obtuviste con solo dos bloques. **Anota tu conclusión**: ¿mejoró añadir profundidad, o el modelo empezó a sobreajustar (overfitting)?

---

[← Anterior: [Prepara tus propios datos](03_prepara_tus_datos.md) · [Índice](00_indice.md) · Siguiente: [TensorBoard: visualiza y depura tus modelos →](05_tensorboard_visualiza.md)]
