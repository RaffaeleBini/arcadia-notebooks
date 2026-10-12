[← Anterior: [¿Qué es el Deep Learning?](01_que_es_deep_learning.md) · [Índice](00_indice.md) · Siguiente: [Prepara tus propios datos →](03_prepara_tus_datos.md)]

# 2. Tu primer modelo con Keras: clasificación de dígitos (MNIST)

## Por qué te importa esto

Leer teoría sobre redes neuronales solo te lleva hasta cierto punto. La forma real de entender el Deep Learning es construir un modelo, verlo entrenar y comprobar si funciona. En este capítulo construyes tu primer clasificador de imágenes: un modelo que reconoce dígitos manuscritos del cero al nueve. Es el "hola mundo" del Deep Learning, y por una buena razón: es simple de montar, entrena en segundos y te deja tocar cada pieza del proceso completo, desde los datos en crudo hasta la predicción final.

## El dataset MNIST

MNIST es una colección de 70.000 imágenes en escala de grises de dígitos manuscritos (del cero al nueve), cada una de 28x28 píxeles. Viene incluido en Keras, así que no necesitas descargar nada por tu cuenta.

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Carga los datos ya divididos en entrenamiento y prueba
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  # (60000, 28, 28): 60.000 imágenes de 28x28 píxeles
print(y_train.shape)  # (60000,): la etiqueta (0-9) de cada imagen
```

## Normaliza tus datos antes de entrenar

Cada píxel de una imagen MNIST es un número entero entre cero y 255 (la intensidad de gris). Las redes neuronales entrenan mejor cuando los valores de entrada están en un rango pequeño, así que los normalizas dividiendo entre 255 para dejarlos entre cero y uno:

```python
# Normaliza: de [0, 255] a [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0
```

Este paso parece trivial, pero olvidarlo es uno de los errores más comunes al empezar: sin normalizar, el entrenamiento es más lento e inestable.

## Construye el modelo

Vas a usar la API `Sequential` de Keras, que te deja apilar capas una tras otra:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
    # Aplana cada imagen 28x28 en un vector de 784 valores
    Flatten(input_shape=(28, 28)),
    # Primera capa oculta: 128 neuronas con activación ReLU
    Dense(128, activation='relu'),
    # Segunda capa oculta: otras 128 neuronas
    Dense(128, activation='relu'),
    # Capa de salida: 10 neuronas (una por dígito), softmax da una probabilidad por clase
    Dense(10, activation='softmax'),
])
```

`Flatten` no aprende nada: solo reorganiza cada imagen de una matriz 28x28 en una fila de 784 números, porque las capas `Dense` esperan un vector, no una matriz.

## Compila el modelo: elige cómo aprende

Compilar un modelo significa decidir tres cosas: cómo mide sus errores (la función de pérdida), cómo ajusta sus pesos para reducir ese error (el optimizador) y qué métrica quieres vigilar mientras entrena:

```python
model.compile(
    optimizer='adam',                          # Adam: ajusta los pesos de forma eficiente
    loss='sparse_categorical_crossentropy',    # ideal cuando las etiquetas son números (0-9)
    metrics=['accuracy'],                      # sigue el porcentaje de aciertos
)
```

## Entrena y evalúa

```python
# Entrena durante 3 pasadas completas sobre los datos de entrenamiento (epochs)
model.fit(x_train, y_train, epochs=3)

# Evalúa el rendimiento sobre datos que el modelo nunca ha visto
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Pérdida en test: {loss:.4f}")
print(f"Precisión en test: {accuracy:.4f}")
```

Con esta arquitectura tan simple, ya deberías ver una precisión (accuracy) por encima del 96% en el conjunto de prueba, en solo tres epochs (pasadas completas de entrenamiento sobre los datos).

## Guarda, carga y predice

Un modelo entrenado que solo vive en la memoria de tu notebook no te sirve de mucho. Guárdalo para reutilizarlo después:

```python
# Guarda el modelo completo (arquitectura + pesos entrenados)
model.save('mi_primer_modelo.keras')

# Carga el modelo en otra sesión, sin tener que reentrenar
from tensorflow.keras.models import load_model
modelo_cargado = load_model('mi_primer_modelo.keras')

# Predice sobre una imagen de test
import numpy as np

predicciones = modelo_cargado.predict(x_test)
digito_predicho = np.argmax(predicciones[0])  # la clase con mayor probabilidad
print(f"El modelo predice: {digito_predicho}")
print(f"La etiqueta real era: {y_test[0]}")
```

`predict` te devuelve, para cada imagen, un vector con 10 probabilidades (una por dígito). `argmax` te dice cuál de esas 10 posiciones tiene el valor más alto, es decir, cuál es la predicción final del modelo.

## Más allá del accuracy: cómo evaluar de verdad un modelo

El accuracy (porcentaje de aciertos) es la métrica más intuitiva, pero puede engañarte, sobre todo si tus clases están desbalanceadas. Estas son las métricas que vas a encontrar a lo largo del curso:

- **Precision**: de todo lo que el modelo marcó como positivo, cuánto era realmente positivo
- **Recall**: de todo lo que era realmente positivo, cuánto detectó el modelo
- **F1-score**: la media armónica entre precision y recall, útil cuando necesitas un único número que resuma ambas
- **Matriz de confusión**: una tabla que cruza predicciones contra etiquetas reales, y te muestra exactamente dónde se equivoca el modelo (por ejemplo, si confunde el 4 con el 9)
- **Curva ROC y AUC (Area Under the Curve)**: miden qué tan bien separa el modelo las clases a distintos umbrales de decisión

## Resumen de puntos clave

- MNIST viene incluido en Keras: no necesitas descargar nada
- Normaliza siempre tus datos de entrada antes de entrenar
- `Sequential` te permite apilar capas; `Flatten` convierte una imagen en un vector
- Compilar un modelo significa fijar optimizador, función de pérdida y métricas
- `fit` entrena, `evaluate` mide el rendimiento sobre datos nuevos, `predict` genera predicciones
- El accuracy no cuenta toda la historia: precision, recall, F1-score y la matriz de confusión te dan una visión más completa

## Ponlo en práctica

Reproduce el código de este capítulo en una notebook y, cuando el modelo ya esté entrenado, cambia el número de epochs de 3 a 10. Anota cómo cambian el accuracy y la pérdida (loss) en el conjunto de test. **Comprueba después la matriz de confusión** con `sklearn.metrics.confusion_matrix(y_test, np.argmax(model.predict(x_test), axis=1))` y localiza qué par de dígitos confunde más tu modelo.

---

[← Anterior: [¿Qué es el Deep Learning?](01_que_es_deep_learning.md) · [Índice](00_indice.md) · Siguiente: [Prepara tus propios datos →](03_prepara_tus_datos.md)]
