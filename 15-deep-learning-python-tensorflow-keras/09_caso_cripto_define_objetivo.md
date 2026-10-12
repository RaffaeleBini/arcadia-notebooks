[← Anterior: [Redes Neuronales Recurrentes (RNN)](08_redes_recurrentes_rnn.md) · [Índice](00_indice.md) · Siguiente: [Normaliza y construye secuencias temporales →](10_caso_cripto_normaliza_secuencias.md)]

# 9. Caso práctico: define tu objetivo con datos de criptomonedas

## Por qué te importa esto

A partir de este capítulo, vas a construir un proyecto completo de principio a fin: un modelo que predice si el precio de una criptomoneda subirá o bajará en los próximos minutos, usando datos históricos de precio y volumen. Este caso ocupa las cuatro clases finales del curso porque, a diferencia de los capítulos anteriores, aquí no aprendes una técnica aislada: aprendes a encadenar todas las técnicas del curso (preparación de datos, RNN, LSTM, regularización, TensorBoard) en un pipeline real de series temporales financieras.

Antes de escribir una sola línea de red neuronal, tienes que responder una pregunta nada trivial: ¿qué es exactamente lo que quieres predecir?

## El objetivo del proyecto

Vas a trabajar con datos históricos de cuatro criptomonedas: Bitcoin (BTC), Litecoin (LTC), Bitcoin Cash (BCH) y Ethereum (ETH). El objetivo es predecir si el precio de Litecoin subirá o bajará dentro de un horizonte de tiempo futuro, usando como entrada una ventana de datos históricos de las cuatro criptomonedas a la vez. La intuición es que el comportamiento de una criptomoneda no es independiente de las demás: incluir el precio y volumen de varias a la vez le da al modelo más contexto que mirar solo Litecoin de forma aislada.

Fíjate en un detalle importante: no vas a predecir el precio exacto (eso sería un problema de regresión). Vas a predecir una etiqueta binaria mucho más simple: sube o baja. Es un problema de clasificación, y es más fácil de resolver bien que intentar adivinar un precio exacto.

## Consigue los datos

El material original de este curso usaba archivos CSV descargados manualmente. Para que tu notebook sea reproducible sin depender de un archivo externo, vas a descargar el histórico directamente con la librería `yfinance`:

```python
!pip install yfinance --quiet

import yfinance as yf
import pandas as pd

tickers = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]
datos_por_moneda = {}

for ticker in tickers:
    df = yf.download(ticker, period="2y", interval="1d")
    df = df[["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "volume"})
    datos_por_moneda[ticker] = df

datos_por_moneda["LTC-USD"].head()
```

Usamos velas diarias (`interval="1d"`) en vez de minuto a minuto, como en el material original, porque el histórico intradía detallado no está disponible de forma gratuita a largo plazo. El pipeline y los conceptos son idénticos, solo cambia la unidad de tiempo: en vez de predecir "dentro de 3 minutos", predices "dentro de 3 días".

## Combina las cuatro series en un único dataset

Cada criptomoneda te da dos columnas (`close`, `volume`). Las combinas todas en un único DataFrame, indexado por fecha, y renombras cada columna con el prefijo de su ticker para no perder de vista de dónde viene cada dato:

```python
main_df = pd.DataFrame()

for ticker, df in datos_por_moneda.items():
    prefijo = ticker.split("-")[0]  # "BTC", "LTC", "BCH", "ETH"
    df = df.rename(columns={"close": f"{prefijo}_close", "volume": f"{prefijo}_volume"})
    main_df = df if main_df.empty else main_df.join(df, how="outer")

# Si algún día falta un dato puntual, rellena con el último valor conocido
main_df = main_df.ffill()
main_df.dropna(inplace=True)

main_df.head()
```

`ffill()` (forward fill) propaga hacia delante el último valor válido cuando hay un hueco. Es preferible a rellenar con cero, que introduciría un dato falso y confundiría al modelo.

## Define la variable objetivo

Ahora decides el horizonte de predicción (cuántos pasos hacia el futuro miras) y construyes la columna que el modelo tiene que aprender a predecir:

```python
RATIO_A_PREDECIR = "LTC"
PERIODO_FUTURO = 3  # cuántos pasos (días, en este caso) hacia el futuro miras

# Desplaza el precio de cierre 3 pasos hacia atrás: así, en la fila de "hoy",
# "future" contiene el precio de cierre de "dentro de 3 días"
main_df["future"] = main_df[f"{RATIO_A_PREDECIR}_close"].shift(-PERIODO_FUTURO)

def clasifica(precio_actual, precio_futuro):
    # 1 = el precio subirá ("compra"), 0 = el precio bajará o se mantendrá igual
    return 1 if float(precio_futuro) > float(precio_actual) else 0

main_df["target"] = list(map(clasifica, main_df[f"{RATIO_A_PREDECIR}_close"], main_df["future"]))

main_df[[f"{RATIO_A_PREDECIR}_close", "future", "target"]].head(10)
```

Comprueba siempre a mano, como en la celda anterior, que la columna `target` tiene sentido: para cada fila, `target` debe ser 1 exactamente cuando `future` es mayor que el precio de cierre actual.

## Resumen de puntos clave

- El objetivo del proyecto es clasificación binaria (sube/baja), no predicción exacta del precio
- Combinar datos de varias criptomonedas correlacionadas le da más contexto al modelo que mirar una sola de forma aislada
- `yfinance` te permite descargar históricos reales sin depender de archivos externos que quizá no tengas
- `ffill()` rellena huecos de forma más sensata que rellenar con cero
- La columna "future" se construye desplazando el precio hacia atrás en el tiempo (`shift(-PERIODO_FUTURO)`), y la columna "target" se calcula comparando el precio actual contra ese precio futuro

## Ponlo en práctica

Ejecuta el pipeline completo de este capítulo y, antes de seguir al próximo, calcula qué porcentaje de filas tiene `target = 1` frente a `target = 0` con `main_df["target"].value_counts(normalize=True)`. **Anota el resultado**: si una de las dos clases supera claramente el 60%, tu dataset está desbalanceado, y es exactamente el problema que vas a resolver en el capítulo 11.

---

[← Anterior: [Redes Neuronales Recurrentes (RNN)](08_redes_recurrentes_rnn.md) · [Índice](00_indice.md) · Siguiente: [Normaliza y construye secuencias temporales →](10_caso_cripto_normaliza_secuencias.md)]
