[← Anterior: [Tu primer modelo con Keras](02_primer_modelo_mnist.md) · [Índice](00_indice.md) · Siguiente: [Redes Neuronales Convolucionales →](04_redes_convolucionales_cnn.md)]

# 3. Prepara tus propios datos: imágenes de gatos y perros

## Por qué te importa esto

MNIST viene perfectamente empaquetado: imágenes ya del mismo tamaño, ya en escala de grises, ya con sus etiquetas listas. En el mundo real casi nunca tienes esa suerte. Tus fotos vendrán en tamaños distintos, en color, mezcladas en carpetas, con nombres de archivo inconsistentes. Antes de que una red neuronal pueda aprender de tus datos, tienes que convertirlos a un formato uniforme. Ese es el trabajo de este capítulo, y es una habilidad que vas a usar en cada proyecto real que hagas.

## El dataset: gatos contra perros

Vas a trabajar con un dataset real de fotos de gatos y perros (una versión reducida del dataset original de Microsoft, con miles de imágenes por clase), organizado en dos carpetas: `Cat/` y `Dog/`. Tu objetivo es convertir esas carpetas de imágenes en algo que una red neuronal pueda procesar: arrays de números, todos del mismo tamaño.

## Carga las imágenes con OpenCV

OpenCV (`cv2`) es la librería estándar de Python para procesar imágenes. Te permite leer un archivo de imagen y convertirlo directamente en un array de NumPy:

```python
import os
import cv2
import matplotlib.pyplot as plt

directorio_datos = "PetImages"
categorias = ["Dog", "Cat"]

for categoria in categorias:
    ruta = os.path.join(directorio_datos, categoria)
    for nombre_archivo in os.listdir(ruta)[:1]:  # solo la primera imagen, para inspeccionar
        # IMREAD_GRAYSCALE convierte la imagen a escala de grises al leerla
        array_imagen = cv2.imread(os.path.join(ruta, nombre_archivo), cv2.IMREAD_GRAYSCALE)
        plt.imshow(array_imagen, cmap='gray')
        plt.show()
        print(array_imagen.shape)  # cada imagen tiene un tamaño distinto
```

Vas a notar dos cosas: primero, cada imagen tiene dimensiones distintas; segundo, trabajas en escala de grises. Convertir a escala de grises reduce la información de tres canales de color a uno solo, lo que acelera mucho el entrenamiento sin perjudicar demasiado la precisión en tareas como esta.

## Redimensiona todas las imágenes al mismo tamaño

Una red neuronal necesita que todas las entradas tengan exactamente la misma forma. Redimensiona cada imagen a 50x50 píxeles:

```python
IMG_SIZE = 50

array_redimensionado = cv2.resize(array_imagen, (IMG_SIZE, IMG_SIZE))
plt.imshow(array_redimensionado, cmap='gray')
plt.show()
```

Cincuenta por cincuenta es un tamaño pequeño a propósito: mantiene el dataset ligero y el entrenamiento rápido, a costa de perder algo de detalle. Es una decisión de compromiso habitual al empezar un proyecto: prototipa rápido con imágenes pequeñas, y solo sube la resolución si necesitas más precisión.

## Construye el dataset completo

Ahora repites el proceso para cada imagen de cada categoría, y guardas el par (imagen, etiqueta):

```python
datos_entrenamiento = []

def crear_datos_entrenamiento():
    for categoria in categorias:
        ruta = os.path.join(directorio_datos, categoria)
        clase_num = categorias.index(categoria)  # Dog=0, Cat=1

        for nombre_archivo in os.listdir(ruta):
            try:
                array_imagen = cv2.imread(os.path.join(ruta, nombre_archivo), cv2.IMREAD_GRAYSCALE)
                array_redimensionado = cv2.resize(array_imagen, (IMG_SIZE, IMG_SIZE))
                datos_entrenamiento.append([array_redimensionado, clase_num])
            except Exception:
                # Algunos archivos del dataset están corruptos: los ignoras y sigues
                pass

crear_datos_entrenamiento()
print(len(datos_entrenamiento))
```

## Por qué debes balancear y mezclar tus datos

Si tu dataset tiene 6.000 fotos de perros y solo 500 de gatos, tu modelo puede "hacer trampa": aprende a predecir siempre "perro" y aun así acierta el 92,3% de las veces, sin haber aprendido nada útil sobre gatos. Por eso comprueba siempre el balance de clases antes de entrenar, y recorta la clase mayoritaria si hace falta para igualar ambas.

Además, si no mezclas los datos, el modelo verá primero 6.000 perros seguidos y después todos los gatos: puede aprender patrones relacionados con el orden en vez de con el contenido real de la imagen.

```python
import random

random.shuffle(datos_entrenamiento)

for muestra in datos_entrenamiento[:10]:
    print(muestra[1])  # deberías ver una mezcla de ceros y unos, no un bloque de cada
```

## Separa features y etiquetas, y da forma al array final

```python
import numpy as np

X = []  # features: las imágenes
y = []  # labels: 0 (perro) o 1 (gato)

for features, label in datos_entrenamiento:
    X.append(features)
    y.append(label)

# Keras espera 4 dimensiones: (num_imágenes, alto, ancho, canales)
# El 1 final indica "un canal", porque trabajas en escala de grises
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)
```

## Guarda tu trabajo con pickle

Procesar miles de imágenes lleva tiempo. No repitas ese trabajo cada vez que abras la notebook: guarda el resultado con `pickle`, el módulo estándar de Python para serializar objetos:

```python
import pickle

with open("X.pickle", "wb") as archivo_salida:
    pickle.dump(X, archivo_salida)

with open("y.pickle", "wb") as archivo_salida:
    pickle.dump(y, archivo_salida)

# La próxima vez, en lugar de reprocesar las imágenes, simplemente cargas:
with open("X.pickle", "rb") as archivo_entrada:
    X = pickle.load(archivo_entrada)
```

## Resumen de puntos clave

- Los datos reales llegan sin formato uniforme: tú tienes que estandarizarlos antes de entrenar
- `cv2.imread` con `IMREAD_GRAYSCALE` carga una imagen ya convertida a escala de grises
- `cv2.resize` fuerza a todas las imágenes a compartir el mismo tamaño, requisito indispensable para una red neuronal
- Comprueba siempre el balance de clases y mezcla (shuffle) tus datos antes de entrenar
- Keras espera arrays de imágenes con forma `(cantidad, alto, ancho, canales)`
- `pickle` te permite guardar y recargar datos ya procesados sin repetir el trabajo

## Ponlo en práctica

Descarga una versión pequeña del dataset "Cats vs Dogs" (por ejemplo, con `tf.keras.utils.get_file` apuntando al ZIP público que usa Google en sus tutoriales de TensorFlow) y reproduce el pipeline completo de este capítulo: carga las imágenes, redimensiónalas a 50x50, comprueba el balance de clases, mezcla el dataset y guárdalo con `pickle`. **Verifica al final** que `X.shape` sea `(num_imágenes, 50, 50, 1)` antes de seguir al próximo capítulo.

---

[← Anterior: [Tu primer modelo con Keras](02_primer_modelo_mnist.md) · [Índice](00_indice.md) · Siguiente: [Redes Neuronales Convolucionales →](04_redes_convolucionales_cnn.md)]
