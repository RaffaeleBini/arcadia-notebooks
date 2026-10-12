# Capítulo 6 — Tu primera red neuronal cuántica: clasificador binario

## Por qué importa

Ha llegado el momento de entrenar tu primer modelo de machine learning cuántico de principio a fin. Vas a construir un clasificador binario: un modelo que aprende a distinguir entre dos categorías, el equivalente cuántico del "hola mundo" del machine learning clásico (piensa en distinguir spam de no-spam, o gatos de perros, pero aquí con un único qubit). Este ejemplo es deliberadamente sencillo porque quiere enseñarte el patrón completo (datos → circuito → entrenamiento → predicción) que vas a reutilizar en cada capítulo restante del curso.

## Explicación conceptual

### El problema: separar dos grupos de ángulos

Vas a generar datos sintéticos: números aleatorios cercanos a 0 o cercanos a 1, que usarás como ángulo de rotación de un único qubit (recuerda la potencia fraccionaria `cirq.X(qubit)**angulo` del capítulo 3). La etiqueta de cada dato es +1 si el ángulo está cerca de 0, y -1 si está cerca de 1. El circuito tiene que aprender a distinguir ambos casos midiendo el qubit resultante.

### El circuito: tres rotaciones entrenables

El circuito de este capítulo es el más simple posible: tres rotaciones (`rx`, `ry`, `rz`) sobre un único qubit, cada una con su propio símbolo entrenable. Al medir en la base X (con el operador `cirq.X(qubit)` como lectura), el valor esperado te da un número entre -1 y 1: ese número es la "predicción" del modelo, comparable directamente con las etiquetas +1/-1 de tus datos.

### La función de pérdida hinge y su métrica de precisión

Como las etiquetas son +1 y -1 (no 0 y 1, como en un clasificador binario clásico típico), usamos la **pérdida hinge** (`tf.keras.losses.hinge`), pensada exactamente para ese formato de etiquetas. Como Keras no trae de fábrica una métrica de precisión compatible con hinge, se define una función propia, `hinge_accuracy`, que compara el signo de la predicción con el signo de la etiqueta real.

### El patrón de entrenamiento que vas a repetir todo el curso

1. Generas los datos y los conviertes a tensor con `tfq.convert_to_tensor`
2. Defines el circuito parametrizado y lo envuelves en una capa `tfq.layers.PQC`
3. Compilas el modelo de Keras con una función de pérdida, un optimizador (Adam) y una métrica
4. Entrenas con `.fit(...)`, exactamente igual que con cualquier red neuronal clásica

## Ejemplo de código comentado

```python
import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import random
import numpy as np
import matplotlib.pyplot as plt

def generar_datos(n_entrenamiento, n_prueba):
    """Genera ángulos cercanos a 0 (etiqueta +1) o cercanos a 1 (etiqueta -1)."""
    qubit = cirq.GridQubit(0, 0)

    def lote(n):
        circuitos, etiquetas = [], []
        for _ in range(n):
            angulo = random.uniform(0, 0.1) if random.random() < 0.5 else random.uniform(0.9, 1)
            circuito = cirq.Circuit()
            circuito.append(cirq.X(qubit) ** angulo)
            circuitos.append(circuito)
            etiquetas.append(1 if angulo < 0.5 else -1)
        return tfq.convert_to_tensor(circuitos), np.array(etiquetas)

    entrenamiento, etiquetas_entrenamiento = lote(n_entrenamiento)
    prueba, etiquetas_prueba = lote(n_prueba)
    return entrenamiento, etiquetas_entrenamiento, prueba, etiquetas_prueba

def construir_circuito(qubit):
    x, y, z = sympy.symbols("rotacion_x rotacion_y rotacion_z")
    circuito = cirq.Circuit()
    circuito.append(cirq.rx(x).on(qubit))
    circuito.append(cirq.ry(y).on(qubit))
    circuito.append(cirq.rz(z).on(qubit))
    return circuito

def precision_hinge(etiqueta_real, prediccion):
    etiqueta_real = tf.squeeze(etiqueta_real) > 0.0
    prediccion = tf.squeeze(prediccion) > 0.0
    return tf.reduce_mean(tf.cast(etiqueta_real == prediccion, tf.float32))

entrenamiento, etiquetas_entrenamiento, prueba, etiquetas_prueba = generar_datos(1000, 100)

qubit = cirq.GridQubit(0, 0)
operador_lectura = [cirq.X(qubit)]
entradas = tf.keras.Input(shape=(), dtype=tf.dtypes.string)

capa_cuantica = tfq.layers.PQC(
    construir_circuito(qubit),
    operador_lectura,
    repetitions=32,
    differentiator=tfq.differentiators.ParameterShift(),
    initializer=tf.keras.initializers.Zeros,
)(entradas)

modelo = tf.keras.models.Model(inputs=entradas, outputs=capa_cuantica)
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.hinge,
    metrics=[precision_hinge],
)

historial = modelo.fit(entrenamiento, etiquetas_entrenamiento, epochs=64, batch_size=32,
                        validation_data=(prueba, etiquetas_prueba))

plt.plot(historial.history["precision_hinge"], label="Entrenamiento")
plt.plot(historial.history["val_precision_hinge"], label="Validación")
plt.legend()
plt.title("Precisión del clasificador binario cuántico")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.show()
```

Tras las 64 épocas de entrenamiento, deberías ver la precisión de validación acercarse al 100%: el circuito de un único qubit es más que suficiente para separar dos grupos tan sencillos.

## Resumen de puntos clave

- Un clasificador binario cuántico convierte cada dato en un circuito, lo pasa por un circuito entrenable y mide el resultado como predicción
- Con etiquetas +1/-1, se usa la pérdida hinge y una métrica de precisión hecha a medida (`hinge_accuracy`)
- El patrón datos → circuito parametrizado → `PQC` → `compile` → `fit` se repite en todos los modelos de este curso
- `repetitions=32` en la capa `PQC` simula que el hardware mide el circuito 32 veces por cada predicción, en vez de calcular el valor exacto

## Ponlo en práctica

Cambia el operador de lectura de `cirq.X(qubit)` a `cirq.Z(qubit)` y vuelve a entrenar el modelo desde cero. Compara la curva de precisión de validación con la del ejemplo original. Si el modelo no aprende igual de bien, piensa por qué: recuerda que los datos se codificaron con una rotación en el eje X (`cirq.X(qubit)**angulo`), así que la base de medida más informativa no tiene por qué ser la misma que la base de codificación.
