[← Anterior: [TensorBoard: visualiza y depura tus modelos](05_tensorboard_visualiza.md) · [Índice](00_indice.md) · Siguiente: [Pon tu modelo a prueba →](07_pon_tu_modelo_a_prueba.md)]

# 6. Optimiza tus modelos: búsqueda de hiperparámetros

## Por qué te importa esto

¿Cuántas capas convolucionales necesitas? ¿64 filtros o 128? ¿Una capa densa al final, o ninguna? No hay una respuesta universal: depende de tus datos. Los hiperparámetros son todas esas decisiones de diseño que tú fijas antes de entrenar (a diferencia de los pesos, que la red aprende sola). Este capítulo te enseña a comparar variantes de un modelo de forma sistemática, en vez de ir probando a ciegas.

## Qué hiperparámetros puedes ajustar

- **Arquitectura**: número de capas convolucionales, número de capas densas, número de filtros o neuronas por capa
- **Optimizador**: Adam, SGD (Stochastic Gradient Descent), RMSprop, cada uno con comportamientos distintos
- **Learning rate**: cuánto ajusta el optimizador los pesos en cada paso. Demasiado alto y el entrenamiento se vuelve inestable; demasiado bajo y tarda una eternidad en converger
- **Tamaño del kernel y stride**: el tamaño del filtro convolucional y cuánto se desplaza en cada paso
- **Decay**: cuánto se reduce el learning rate a medida que avanza el entrenamiento

## Genera y compara combinaciones de arquitecturas

En vez de cambiar el código a mano cada vez, escribe un bucle que genere varias combinaciones automáticamente, entrenando y registrando cada una en TensorBoard con un nombre descriptivo:

```python
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

capas_densas = [0, 1, 2]
tamanos_capa = [32, 64, 128]
capas_conv = [1, 2, 3]

for num_capas_conv in capas_conv:
    for tamano in tamanos_capa:
        for num_capas_densas in capas_densas:
            nombre_modelo = f"{num_capas_conv}conv-{tamano}nodos-{num_capas_densas}densas-{int(time.time())}"
            tensorboard_callback = TensorBoard(log_dir=f'logs/{nombre_modelo}')
            print(f"Entrenando: {nombre_modelo}")

            model = Sequential()
            model.add(Conv2D(tamano, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for _ in range(num_capas_conv - 1):
                model.add(Conv2D(tamano, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(num_capas_densas):
                model.add(Dense(tamano))
                model.add(Activation('relu'))

            model.add(Dense(1, activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(
                X, y,
                batch_size=32,
                epochs=10,
                validation_split=0.1,
                callbacks=[tensorboard_callback],
            )
```

Con tres opciones en cada uno de los tres hiperparámetros, este bucle entrena 27 modelos distintos, cada uno con su propio log en TensorBoard. Abre TensorBoard y compara las curvas de `val_accuracy`: el nombre descriptivo de cada modelo te dice de un vistazo qué combinación de hiperparámetros dio el mejor resultado.

Diez epochs por combinación y 27 combinaciones son muchos minutos de entrenamiento. Si solo quieres explorar el concepto sin esperar tanto, reduce la lista de valores (por ejemplo, prueba solo con `capas_conv = [1, 3]`) antes de lanzar el bucle completo.

## Técnicas más allá de la búsqueda manual

El bucle anterior es una búsqueda exhaustiva ("Grid Search") hecha a mano. Cuando el número de combinaciones crece demasiado, existen alternativas más eficientes:

- **Grid Search**: prueba todas las combinaciones posibles (lo que acabas de hacer), garantiza cobertura completa pero escala mal
- **Random Search**: prueba combinaciones aleatorias en vez de todas, suele encontrar buenos resultados con mucho menos cómputo
- **Bayesian Optimization**: usa los resultados de las pruebas anteriores para decidir de forma inteligente qué combinación probar a continuación
- **AutoML** (Automated Machine Learning): herramientas como AutoKeras automatizan tanto la búsqueda de arquitectura como la de hiperparámetros

## Otras palancas de optimización

Además de la arquitectura, estas técnicas mejoran el entrenamiento:

- **Regularización**: `Dropout` (apaga neuronas al azar durante el entrenamiento para evitar que la red dependa demasiado de unas pocas) y `BatchNormalization` (normaliza las entradas de cada capa, acelera y estabiliza el entrenamiento)
- **Early stopping**: detiene el entrenamiento automáticamente si la métrica de validación deja de mejorar, para no malgastar tiempo ni sobreajustar
- **Hardware**: entrenar en GPU (Graphics Processing Unit) o TPU (Tensor Processing Unit) en vez de CPU reduce drásticamente el tiempo de entrenamiento en redes grandes

## Resumen de puntos clave

- Los hiperparámetros son las decisiones de diseño que fijas tú antes de entrenar, no lo que la red aprende
- Un bucle que genera y entrena combinaciones, cada una con su log de TensorBoard, te permite comparar arquitecturas de forma objetiva
- Grid Search, Random Search y Bayesian Optimization son formas cada vez más eficientes de explorar el espacio de hiperparámetros
- Dropout, BatchNormalization y early stopping son técnicas de optimización que complementan (no sustituyen) la búsqueda de arquitectura

## Ponlo en práctica

Ejecuta el bucle de este capítulo con una versión reducida de las listas de hiperparámetros (por ejemplo, `capas_conv = [1, 2]`, `tamanos_capa = [32, 64]`, `capas_densas = [0, 1]`) para mantener el tiempo de entrenamiento manejable. Abre TensorBoard, identifica la combinación con mayor `val_accuracy` y **razona por escrito** por qué crees que esa combinación ganó a las demás.

---

[← Anterior: [TensorBoard: visualiza y depura tus modelos](05_tensorboard_visualiza.md) · [Índice](00_indice.md) · Siguiente: [Pon tu modelo a prueba →](07_pon_tu_modelo_a_prueba.md)]
