[← Anterior: [Normaliza y construye secuencias temporales](10_caso_cripto_normaliza_secuencias.md) · [Índice](00_indice.md) · Siguiente: [Casos de uso reales de Deep Learning →](12_casos_de_uso_reales.md)]

# 11. Balancea tus datos y entrena tu modelo LSTM final

## Por qué te importa esto

Tienes secuencias temporales normalizadas y separadas correctamente en entrenamiento y validación. Falta un último ajuste antes de entrenar: comprobar si tus clases (sube/baja) están balanceadas, y si no lo están, corregirlo. Después construyes la arquitectura LSTM final, la entrenas con los callbacks adecuados, y cierras el proyecto que empezaste en el capítulo 9.

## Balancea tus clases manualmente

Keras ofrece el parámetro `class_weight` en `model.fit` para compensar clases desbalanceadas, pero en la práctica no siempre corrige bien el problema, sobre todo en datos secuenciales. Un balanceo manual, más simple y más predecible, funciona mejor: separas las secuencias por clase, recortas ambos grupos a la longitud del más pequeño, y los vuelves a mezclar juntos.

```python
import random

def balancea_secuencias(datos_secuenciales):
    compras, ventas = [], []

    for secuencia, target in datos_secuenciales:
        if target == 0:
            ventas.append([secuencia, target])
        else:
            compras.append([secuencia, target])

    random.shuffle(compras)
    random.shuffle(ventas)

    limite = min(len(compras), len(ventas))
    compras = compras[:limite]
    ventas = ventas[:limite]

    datos_balanceados = compras + ventas
    random.shuffle(datos_balanceados)
    return datos_balanceados

secuencias_entrenamiento = balancea_secuencias(secuencias_entrenamiento)
secuencias_validacion = balancea_secuencias(secuencias_validacion)

print(f"Entrenamiento balanceado: {len(secuencias_entrenamiento)} secuencias")
print(f"Validación balanceada: {len(secuencias_validacion)} secuencias")
```

**Mezcla siempre el conjunto ya combinado** (`random.shuffle(datos_balanceados)`), no solo cada lista por separado: si no lo haces, todas las "compras" quedan juntas seguidas de todas las "ventas", y el modelo puede aprender un patrón falso basado en el orden en vez de en el contenido real de cada secuencia.

## Separa en arrays de entrada (X) y objetivo (y)

```python
import numpy as np

def separa_features_y_target(datos_secuenciales):
    X, y = [], []
    for secuencia, target in datos_secuenciales:
        X.append(secuencia)
        y.append(target)
    return np.array(X), np.array(y)

train_x, train_y = separa_features_y_target(secuencias_entrenamiento)
validation_x, validation_y = separa_features_y_target(secuencias_validacion)
```

## Construye la arquitectura LSTM final

Apilas tres capas LSTM, cada una seguida de `Dropout` (para evitar overfitting) y `BatchNormalization` (para estabilizar el entrenamiento):

```python
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

EPOCHS = 10
BATCH_SIZE = 64
NOMBRE_MODELO = f"LSTM-cripto-SEQ{SEQ_LEN}-PRED{PERIODO_FUTURO}-{int(time.time())}"

model = Sequential([
    LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),

    LSTM(128, return_sequences=True),
    Dropout(0.1),
    BatchNormalization(),

    LSTM(128),
    Dropout(0.2),
    BatchNormalization(),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(2, activation='softmax'),
])

optimizador = Adam(learning_rate=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizador,
    metrics=['accuracy'],
)
```

> **Nota sobre la arquitectura original**: el material del curso usaba la capa `CuDNNLSTM`, una variante de LSTM acelerada por GPU que existía en versiones antiguas de TensorFlow/Keras. Esa capa se eliminó en TensorFlow 2.x: hoy usas `LSTM` directamente, y Keras aprovecha la aceleración de GPU de forma automática cuando hay una disponible, sin que tengas que elegir una capa distinta.

## Configura los callbacks de entrenamiento

```python
tensorboard_callback = TensorBoard(log_dir=f'logs/{NOMBRE_MODELO}')

checkpoint_callback = ModelCheckpoint(
    filepath="modelos/epoch-{epoch:02d}-val_acc-{val_accuracy:.3f}.keras",
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max',
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,  # detiene el entrenamiento si val_loss no mejora en 3 epochs consecutivos
)
```

> **Nota sobre un bug del material original**: en el notebook fuente, la llamada a `ModelCheckpoint` pasaba `monitor`, `verbose`, `save_best_only` y `mode` como argumentos de `str.format()` en vez de como argumentos del propio `ModelCheckpoint`, así que esas opciones se ignoraban silenciosamente y el checkpoint terminaba guardando todos los modelos, no solo el mejor. Aquí los argumentos van directamente a `ModelCheckpoint(...)`, donde sí tienen efecto. Además, el nombre de la métrica de validación cambió de `val_acc` (versiones antiguas de Keras) a `val_accuracy` (versión actual).

## Entrena y evalúa el modelo final

```python
model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback],
)

loss, accuracy = model.evaluate(validation_x, validation_y)
print(f"Pérdida de validación: {loss:.4f}")
print(f"Precisión de validación: {accuracy:.4f}")
```

## Interpreta tus resultados con honestidad

No esperes una precisión espectacular: predecir movimientos de precio a corto plazo es, por diseño, un problema muy ruidoso. El precio de una criptomoneda no depende solo de su propio histórico, también de noticias, sentimiento de mercado y actividad en redes sociales, información que este modelo no ve en ningún momento. Distintas criptomonedas se comportan de forma distinta: unas más estables, otras más volátiles, y eso se refleja en la dificultad de predecirlas.

**Trata este modelo como una herramienta educativa, no como un consejo de inversión.** Si algún día quisieras acercarte a un caso de uso real de trading, necesitarías, como mínimo, incorporar más fuentes de datos (sentimiento de mercado, noticias, volumen de órdenes), validar el modelo con mucho más rigor estadístico, y entender que ningún modelo elimina el riesgo financiero.

## Resumen de puntos clave

- Balancea manualmente las secuencias por clase cuando `class_weight` no sea suficiente, y mezcla siempre el conjunto combinado, no solo cada clase por separado
- La arquitectura final combina LSTM apiladas con `Dropout` y `BatchNormalization` para regularizar y estabilizar el entrenamiento
- `CuDNNLSTM` ya no existe en TensorFlow 2.x: usa `LSTM`, que aprovecha la GPU automáticamente
- Verifica siempre que los argumentos de tus callbacks (como `ModelCheckpoint`) se pasen al callback y no se pierdan en un `str.format()` mal escrito
- `EarlyStopping` evita seguir entrenando cuando ya no hay mejora real
- Interpreta los resultados de un modelo financiero con honestidad: es una herramienta educativa, no una garantía de rentabilidad

## Ponlo en práctica

Entrena el modelo completo de este capítulo con los datos que preparaste en los capítulos 9 y 10. Cuando termine, abre TensorBoard y compara la curva de `val_accuracy` de este modelo balanceado contra la que obtendrías si comentas la llamada a `balancea_secuencias` y entrenas con los datos desbalanceados originales. **Anota la diferencia** y explica, con tus propias palabras, por qué el balanceo cambia el resultado.

---

[← Anterior: [Normaliza y construye secuencias temporales](10_caso_cripto_normaliza_secuencias.md) · [Índice](00_indice.md) · Siguiente: [Casos de uso reales de Deep Learning →](12_casos_de_uso_reales.md)]
