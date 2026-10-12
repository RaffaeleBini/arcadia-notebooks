[← Anterior: [Redes Neuronales Convolucionales](04_redes_convolucionales_cnn.md) · [Índice](00_indice.md) · Siguiente: [Optimiza tus modelos →](06_optimiza_hiperparametros.md)]

# 5. TensorBoard: visualiza y depura tus modelos

## Por qué te importa esto

Hasta ahora has visto los resultados de tu entrenamiento como números sueltos que Keras imprime en la consola: una línea de loss y accuracy por cada epoch. Eso funciona con 10 epochs, pero se vuelve inmanejable en cuanto empiezas a comparar variantes de un modelo o a entrenar durante más tiempo. TensorBoard es la herramienta de visualización que viene con TensorFlow, y te deja ver la evolución de tu entrenamiento en gráficos interactivos, en tiempo real, directamente en tu navegador.

## Qué puedes ver en TensorBoard

TensorBoard organiza la información en varias pestañas:

- **Scalars**: gráficos de loss y accuracy (de entrenamiento y de validación) a lo largo de los epochs. Es la pestaña que más vas a usar
- **Graphs**: una representación visual de la arquitectura de tu modelo, capa por capa
- **Histograms** y **Distributions**: cómo evolucionan los pesos y las activaciones de cada capa durante el entrenamiento
- **Embeddings / Projector**: proyecta datos de alta dimensión en un espacio de 3D, útil para explorar cómo tu modelo agrupa ejemplos similares

## Añade el callback de TensorBoard a tu entrenamiento

Un callback es una función que Keras ejecuta automáticamente en momentos concretos del entrenamiento (al final de cada epoch, por ejemplo). El callback `TensorBoard` guarda los datos de entrenamiento en un directorio de logs que después TensorBoard lee:

```python
import time
from tensorflow.keras.callbacks import TensorBoard

# Un nombre único por modelo, con timestamp, evita que dos entrenamientos
# se pisen entre sí en la carpeta de logs
nombre_modelo = f"cnn-gatos-perros-{int(time.time())}"
tensorboard_callback = TensorBoard(log_dir=f'logs/{nombre_modelo}')

model.fit(
    X, y,
    batch_size=32,
    epochs=10,
    validation_split=0.1,
    callbacks=[tensorboard_callback],
)
```

**Nombra siempre tus modelos con un identificador único** (aquí, un timestamp). Si reutilizas el mismo nombre en varios entrenamientos, TensorBoard mezcla los datos de todos ellos en la misma gráfica y pierdes la capacidad de comparar.

## Lanza TensorBoard

Desde una terminal, en el directorio donde está la carpeta `logs/`:

```bash
tensorboard --logdir=logs
```

Este comando levanta un servidor local. Abre `http://localhost:6006/` en tu navegador y verás las gráficas actualizarse a medida que el modelo entrena. Si trabajas en Google Colab, usa en su lugar:

```python
%load_ext tensorboard
%tensorboard --logdir logs
```

## Usa la validation loss para detectar overfitting

La gráfica más importante de la pestaña Scalars es la comparación entre `loss` (entrenamiento) y `val_loss` (validación). Un modelo sano muestra ambas curvas bajando juntas. Si `loss` sigue bajando pero `val_loss` empieza a subir, tu modelo está memorizando los datos de entrenamiento en vez de aprender patrones generales: eso es overfitting, y es la señal más clara de que necesitas más datos, regularización (como `Dropout`), o un modelo más simple.

## Resumen de puntos clave

- TensorBoard visualiza en tiempo real la evolución de tu entrenamiento
- El callback `TensorBoard(log_dir=...)` es lo único que necesitas añadir a `model.fit` para activarlo
- Da a cada modelo un nombre único (por ejemplo, con un timestamp) para no mezclar logs de entrenamientos distintos
- La pestaña Scalars, comparando `loss` contra `val_loss`, es tu primera herramienta para detectar overfitting
- En Colab, usa la magic `%tensorboard --logdir logs` en vez del comando de terminal

## Ponlo en práctica

Entrena de nuevo el clasificador de gatos y perros del capítulo anterior, esta vez con el callback de TensorBoard activado y un nombre de modelo único. Lanza TensorBoard y localiza el punto exacto (qué epoch) en el que `val_loss` deja de bajar. **Captura una imagen de esa gráfica** y anota, en una celda de texto, en qué epoch empezarías a preocuparte por overfitting si este fuera un proyecto real.

---

[← Anterior: [Redes Neuronales Convolucionales](04_redes_convolucionales_cnn.md) · [Índice](00_indice.md) · Siguiente: [Optimiza tus modelos →](06_optimiza_hiperparametros.md)]
