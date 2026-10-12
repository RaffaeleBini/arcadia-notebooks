[⬅ Volver al índice](00_indice.md)

# Capítulo 5 — Simplifica tu código: TFLearn y Keras

## Por qué te importa esto

En el capítulo 1 construiste una red a mano con `tf.Variable` y `GradientTape`
para entender qué ocurre por dentro. Pero en el día a día, escribir cada capa
desde cero es lento y propenso a errores: olvidar un sesgo, usar mal una función de
activación, o equivocarte en la forma de un tensor son fallos habituales y
difíciles de detectar. Por eso existen las Application Programming Interface (API)
de alto nivel: capas de abstracción que te dejan definir arquitecturas completas en
pocas líneas, con buenas prácticas ya incorporadas.

## Dos formas de simplificar: Keras y TFLearn

Ya llevas varios capítulos usando Keras, la API de alto nivel oficial e integrada
en TensorFlow. Históricamente existió otra librería con el mismo propósito, TFLearn
—construida también sobre TensorFlow—, que fue muy popular en tutoriales y cursos
anteriores a que Keras se integrase de forma oficial. Hoy Keras es la opción
recomendada y mantenida activamente, pero merece la pena que entiendas la idea
general: cualquier API de alto nivel persigue el mismo objetivo, dejarte escribir
menos código repetitivo y reducir la superficie de error.

Compara estas dos formas de expresar la misma red:

```python
import tensorflow as tf

# Con Keras: capas encadenadas en una lista
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax"),
])
modelo.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
```

Frente a la alternativa de bajo nivel del capítulo 1, con matrices de pesos y
sesgos definidas y actualizadas a mano: la diferencia de líneas de código —y de
oportunidades para equivocarse— es enorme.

## Callbacks: automatiza buenas prácticas

Una ventaja práctica de trabajar con Keras es su sistema de "callbacks": funciones
que se ejecutan automáticamente durante el entrenamiento para resolver tareas
habituales, como detener el entrenamiento si el modelo deja de mejorar
(`EarlyStopping`) o guardar automáticamente el mejor modelo visto hasta el momento
(`ModelCheckpoint`).

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("mejor_modelo.keras", save_best_only=True),
]

modelo.fit(x_entreno, y_entreno, epochs=20, validation_split=0.1, callbacks=callbacks)
```

Sin esta API de alto nivel, tendrías que escribir tú mismo la lógica para comparar
la métrica de cada época con la anterior y decidir cuándo parar o guardar el
modelo: exactamente el tipo de código repetitivo que una buena abstracción evita.

## Resumen de puntos clave

- Escribir una red "a pelo" con tensores y gradientes te ayuda a entender los
  fundamentos, pero es lento y propenso a errores para el trabajo diario.
- Keras es la API de alto nivel oficial de TensorFlow; TFLearn fue una alternativa
  popular en el pasado con el mismo propósito.
- Los callbacks de Keras (`EarlyStopping`, `ModelCheckpoint`, entre otros)
  automatizan buenas prácticas que, de otro modo, tendrías que programar a mano.
- La regla general: usa la abstracción de alto nivel para tu trabajo habitual, y
  recurre al nivel bajo solo cuando necesites un control muy específico que la
  abstracción no ofrezca.

## Ponlo en práctica

**Añade** los dos callbacks —`EarlyStopping` y `ModelCheckpoint`— al entrenamiento
del clasificador de dígitos del capítulo 1, **configura** `EarlyStopping` con
`patience=2`, y **comprueba** en cuántas épocas se detiene el entrenamiento
automáticamente en lugar de completar las que le indiques.

---

**Material fuente de este capítulo**: lección 5 tema 1 ("TensorFlow Deep Learning
Library — TFLearn").

[⬅ Capítulo anterior](04_cnn_fundamentos.md) · [Volver al índice](00_indice.md) · [Siguiente capítulo ➡](06_reinforcement_learning.md)
