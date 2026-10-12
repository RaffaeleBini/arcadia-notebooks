[← Anterior: [Pon tu modelo a prueba](07_pon_tu_modelo_a_prueba.md) · [Índice](00_indice.md) · Siguiente: [Caso práctico: define tu objetivo con datos de criptomonedas →](09_caso_cripto_define_objetivo.md)]

# 8. Redes Neuronales Recurrentes (RNN): memoria para datos secuenciales

## Por qué te importa esto

Una CNN es excelente reconociendo patrones espaciales en una imagen fija, pero no tiene ningún concepto de "orden en el tiempo". ¿Qué pasa cuando lo que quieres predecir depende de una secuencia, como el precio de una acción a lo largo del tiempo, las palabras de una frase o el audio de una conversación? Ahí entran las Redes Neuronales Recurrentes (RNN, Recurrent Neural Network): redes diseñadas específicamente para datos donde el orden importa y donde el pasado influye en el presente. Este capítulo abre el bloque final del curso, donde aplicas RNN a un caso real: predecir el movimiento del precio de una criptomoneda.

## Qué hace "recurrente" a una RNN

Una capa `Dense` o `Conv2D` procesa cada entrada de forma aislada, sin memoria de lo que vino antes. Una RNN, en cambio, mantiene un "estado oculto" que se actualiza en cada paso de la secuencia y se pasa al siguiente paso. Es como leer una novela: al llegar a la página 50, tu comprensión de la trama no depende solo de esa página, depende de todo lo que leíste antes. Ese "recuerdo acumulado" es exactamente lo que el estado oculto de una RNN representa.

## El problema del vanishing gradient

Las RNN más simples (a veces llamadas "Elman networks") tienen un problema práctico serio: cuando la secuencia es larga, la información de los primeros pasos se va diluyendo hasta desaparecer antes de llegar a los últimos pasos. Esto se conoce como el problema del vanishing gradient (gradiente que se desvanece), y en la práctica significa que una RNN simple "olvida" rápido, siendo incapaz de aprender dependencias a largo plazo.

## LSTM: memoria a corto y largo plazo

LSTM (Long Short-Term Memory) resuelve este problema con una arquitectura de puertas (gates) que controlan qué información se guarda, qué se olvida y qué se deja pasar a la salida en cada paso:

- **Puerta de olvido (forget gate)**: decide qué parte del estado anterior descartar
- **Puerta de entrada (input gate)**: decide qué información nueva incorporar
- **Puerta de salida (output gate)**: decide qué parte del estado actual exponer como salida

Gracias a estas puertas, una LSTM puede mantener información relevante durante muchos pasos de la secuencia, y descartar la que ya no importa. Es la arquitectura recurrente más usada en la práctica, y la que vas a usar en el caso de las criptomonedas.

## GRU: una alternativa más ligera

GRU (Gated Recurrent Unit) simplifica la idea de LSTM combinando algunas de sus puertas en solo dos (puerta de actualización y puerta de reinicio). Con menos parámetros que entrenar, GRU suele ser más rápida y a veces funciona igual de bien que LSTM, aunque LSTM sigue siendo la opción más robusta cuando las dependencias a largo plazo son críticas.

## RNN unidireccionales y bidireccionales

Una RNN unidireccional procesa la secuencia en un solo sentido (normalmente de principio a fin). Una RNN bidireccional la procesa en ambos sentidos a la vez y combina la información: útil cuando el contexto completo (pasado y futuro dentro de la secuencia) está disponible de antemano, como al analizar un texto ya escrito completo. Para predecir el futuro a partir del pasado (como en el caso de las criptomonedas), usas la variante unidireccional: en el momento de predecir, el "futuro" de la secuencia todavía no existe.

## Un primer ejemplo con la capa LSTM de Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    # 128 unidades LSTM. return_sequences=True porque apilas otra capa recurrente encima
    LSTM(128, input_shape=(timesteps, num_caracteristicas), return_sequences=True),
    Dropout(0.2),
    # Última capa recurrente: return_sequences=False (por defecto) porque ya solo necesitas el resultado final
    LSTM(128),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

`return_sequences=True` le dice a la capa que devuelva su salida en cada paso de tiempo, no solo en el último, porque la siguiente capa LSTM necesita una secuencia completa como entrada.

## MNIST como secuencia: un ejemplo didáctico

Puedes practicar RNN incluso sin datos temporales "de verdad": tratando cada imagen de MNIST como una secuencia de 28 filas, cada una con 28 valores. Es un ejercicio artificial, pero te deja comparar directamente el rendimiento de una RNN contra la red densa del capítulo 2 sobre el mismo problema:

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train ya tiene forma (60000, 28, 28): 60.000 secuencias de 28 pasos, con 28 valores cada uno

model = Sequential([
    LSTM(128, input_shape=(28, 28), return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
```

## Resumen de puntos clave

- Una RNN mantiene un estado oculto que se actualiza en cada paso de la secuencia, dándole "memoria" del pasado
- El vanishing gradient hace que las RNN simples olviden rápido en secuencias largas
- LSTM usa puertas de olvido, entrada y salida para mantener información relevante durante más tiempo
- GRU simplifica LSTM con menos puertas, a costa de algo de capacidad de modelado
- `return_sequences=True` es necesario cuando apilas una capa recurrente sobre otra
- Para predecir el futuro a partir del pasado, usa RNN unidireccionales

## Ponlo en práctica

Reproduce el ejemplo de MNIST como secuencia de este capítulo y entrena el modelo LSTM durante 3 epochs. Compara el `val_accuracy` final contra el que obtuviste con la red densa del capítulo 2 con el mismo número de epochs. **Escribe una hipótesis**: ¿por qué crees que una arquitectura pensada para secuencias temporales obtiene ese resultado en un problema que, en realidad, no tiene una estructura temporal real?

---

[← Anterior: [Pon tu modelo a prueba](07_pon_tu_modelo_a_prueba.md) · [Índice](00_indice.md) · Siguiente: [Caso práctico: define tu objetivo con datos de criptomonedas →](09_caso_cripto_define_objetivo.md)]
