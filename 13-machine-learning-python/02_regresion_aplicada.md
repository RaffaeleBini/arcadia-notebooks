# 2. Regresión lineal aplicada con scikit-learn

## Por qué te importa esto

En el capítulo anterior viste qué es el aprendizaje automático o machine learning (ML) y cómo se organiza un proyecto de este tipo de principio a fin. En este capítulo entras en la técnica más usada para empezar: la regresión.

La regresión te sirve para predecir un número a partir de otros números que ya conoces. Con ella puedes estimar el precio de un producto, el consumo eléctrico de una nave o la demanda de un servicio antes de tomar una decisión que cuesta dinero. Por eso es habitual que un equipo técnico construya primero un prototipo sencillo de predicción para validar una idea, y solo después invierta tiempo y recursos en una solución más compleja.

**Abre tu editor de código y prepárate para escribir cada bloque a medida que avanzas.** Necesitas Python instalado junto con las librerías NumPy, pandas, scikit-learn y Matplotlib. Si trabajas con Anaconda o con pip, instálalas así:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Qué es la regresión

Imagina que quieres calcular el peso aproximado de una persona a partir de su altura. No tienes una báscula a mano, pero sí sabes que, en general, cuanto más alta es una persona, más pesa. Si reúnes datos de altura y peso de muchas personas, puedes trazar una línea que resuma esa tendencia. Esa línea te permite, dada una altura nueva, calcular un peso probable.

Eso es justo lo que hace la regresión: busca la relación entre uno o varios datos que ya conoces (la altura) y un dato numérico que quieres predecir (el peso). Cuando esa relación se representa con una línea recta, hablamos de regresión lineal.

El mismo razonamiento sirve para estimar el precio de una vivienda a partir de su tamaño, el gasto energético de un edificio a partir de su superficie, o el precio de cierre de una acción a partir de su comportamiento reciente. Este capítulo usará precisamente ese último ejemplo, el de un precio que varía día a día, porque es el que utiliza el material original de la certificación EITC/AI/MLP (European IT Certification / Artificial Intelligence / Machine Learning with Python) para introducir la regresión.

No vas a ver todavía la fórmula matemática que hay detrás de la línea de regresión ni cómo se calcula su ajuste: eso llega en el capítulo 4. Aquí te centras en el uso práctico de la librería scikit-learn para entrenar, evaluar y usar un modelo de regresión.

## Features y labels: separar lo que sabes de lo que quieres predecir

En cualquier problema de regresión distingues dos tipos de columnas:

- Las **features** (características o variables predictoras): los datos que ya conoces y que usas como entrada
- El **label** (etiqueta o variable objetivo): el número que quieres predecir

En el material original de esta lección, las features y el label se obtenían de una serie de precios bursátiles de Google descargada en directo con una API (interfaz de programación de aplicaciones) financiera. Esa API ya no es fiable ni gratuita hoy en día, así que en este capítulo simulamos una serie de precios con NumPy para poder practicar sin depender de servicios externos. El razonamiento y el código de scikit-learn son idénticos a los del material original: solo cambia el origen del dato.

Genera primero la serie simulada, con una semilla fija para que el resultado sea reproducible:

```python
import numpy as np
import pandas as pd

np.random.seed(42)

n_dias = 500
retorno_diario = np.random.normal(loc=0.0005, scale=0.02, size=n_dias)
precio_cierre = 100 * (1 + retorno_diario).cumprod()

fechas = pd.date_range(start="2022-01-03", periods=n_dias, freq="B")

df = pd.DataFrame({"Close": precio_cierre}, index=fechas)
df.index.name = "Fecha"

ruido_apertura = np.random.normal(loc=0, scale=0.005, size=n_dias)
df["Open"] = df["Close"].shift(1).fillna(df["Close"].iloc[0]) * (1 + ruido_apertura)

ruido_rango = np.random.uniform(low=0.002, high=0.02, size=n_dias)
df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + ruido_rango)
df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - ruido_rango)

df["Volume"] = np.random.randint(1_000_000, 5_000_000, size=n_dias)

print(df.head())
```

Este código crea un paseo aleatorio (random walk): cada precio de cierre es el anterior multiplicado por un pequeño retorno diario aleatorio. A partir de ese cierre simulado, calculamos también un precio de apertura, un máximo, un mínimo y un volumen negociado, para imitar la forma de un dataset bursátil real.

Ahora bien, el material original advierte de algo importante: no todas las columnas aportan lo mismo, y una regresión lineal simple no descubre por sí sola relaciones entre columnas. Si crees que la diferencia entre el máximo y el mínimo del día importa, tienes que calcularla tú y añadirla como una feature nueva. Haz lo mismo que hace el material original con dos columnas derivadas:

```python
df["HL_percent"] = (df["High"] - df["Low"]) / df["Low"] * 100
df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100

columnas_features = ["Close", "HL_percent", "PCT_change", "Volume"]
print(df[columnas_features].head())
```

`HL_percent` recoge la volatilidad del día (cuánto se movió el precio entre su máximo y su mínimo), y `PCT_change` recoge si el precio subió o bajó a lo largo de la sesión. Son ejemplos de cómo transformar columnas poco útiles por separado en features con significado.

Falta el label. Aquí queremos predecir el precio de cierre, pero no el de hoy, sino el de unos días en el futuro. Para conseguirlo, desplazamos la columna `Close` hacia arriba un número fijo de filas con el método `shift` de pandas:

```python
FORECAST_COL = "Close"
FORECAST_OUT = 30  # número de días que queremos predecir hacia el futuro

df["label"] = df[FORECAST_COL].shift(-FORECAST_OUT)

print(df[["Close", "label"]].head())
print(df[["Close", "label"]].tail())
```

Fíjate en el resultado: en cada fila, `label` contiene el precio de cierre real de 30 días después. Como no existen datos 30 días después de las últimas filas del dataset, esas filas quedan con valores nulos (`NaN`) en `label`. Esto es exactamente lo que ocurre en el material original y es intencionado: esas filas sin label son las que más adelante usarás para predecir el futuro, porque son las únicas para las que todavía no conoces la respuesta.

## Entrenar y evaluar el modelo con scikit-learn

Con las features y el label ya definidos, el siguiente paso es separar los datos en un conjunto de entrenamiento y un conjunto de prueba, entrenar el modelo con el primero y comprobar su rendimiento con el segundo. Esta separación es clave: si evalúas el modelo con los mismos datos con los que lo entrenaste, no sabrás si de verdad aprendió un patrón general o si solo memorizó los datos de entrenamiento.

```python
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = np.array(df[columnas_features])
X = preprocessing.scale(X)

X_lately = X[-FORECAST_OUT:]
X = X[:-FORECAST_OUT]

df.dropna(inplace=True)
y = np.array(df["label"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = LinearRegression(n_jobs=-1)
modelo.fit(X_train, y_train)

precision = modelo.score(X_test, y_test)
print("Precisión del modelo (R²):", round(precision, 4))
```

Repasa qué hace cada línea:

- `preprocessing.scale(X)` reescala cada feature para que todas queden en un rango parecido; así ninguna columna domina el entrenamiento solo por tener números más grandes
- `X_lately` guarda las filas más recientes, las que no tienen label porque corresponden al futuro que aún quieres predecir
- `train_test_split` reparte el resto de filas en un 80% para entrenar y un 20% para probar, de forma aleatoria pero reproducible gracias a `random_state`
- `LinearRegression()` crea el modelo; el parámetro `n_jobs=-1` le indica que use todos los núcleos de tu procesador disponibles para entrenar más rápido
- `modelo.fit(X_train, y_train)` es el entrenamiento: el modelo ajusta sus parámetros internos para que sus predicciones se acerquen lo máximo posible a `y_train`
- `modelo.score(X_test, y_test)` mide qué tan bien predice el modelo sobre datos que no vio durante el entrenamiento

Ese último valor es el coeficiente de determinación, conocido como R². Se mueve entre 0 y 1: un R² de 1 significa que el modelo explica perfectamente la variación del label a partir de las features, mientras que un R² cercano a 0 significa que el modelo no aporta mucho más que adivinar el valor medio. Con los datos que acabas de simular, es normal que obtengas un R² alto, porque el label es literalmente el mismo precio de cierre desplazado en el tiempo y las features guardan una relación muy directa con él.

En el material original, un modelo de regresión lineal parecido alcanza un R² de 0.96 (96%) sobre este tipo de datos, mientras que otro algoritmo, la regresión de vectores de soporte (SVR, por sus siglas en inglés), se queda en 0.51 (51%) con los parámetros por defecto. La lección práctica es que scikit-learn facilita comparar algoritmos distintos con muy pocos cambios de código, así que probar más de uno antes de decidirte es buena práctica.

Otra métrica habitual en regresión, aunque no la calculamos aquí, es el error cuadrático medio (MSE, por sus siglas en inglés), que mide el promedio de las diferencias al cuadrado entre el valor predicho y el real. Cuanto más bajo, mejor. Verás el MSE con más detalle cuando en el capítulo 4 entres en cómo se calcula realmente una regresión lineal.

## Predecir el futuro: forecasting

Llegados aquí ya tienes un modelo entrenado y evaluado. El paso final es usarlo para predecir de verdad: aplicarlo a `X_lately`, ese bloque de filas recientes que dejaste fuera del entrenamiento porque no tenían label conocido.

```python
predicciones = modelo.predict(X_lately)

print("Predicciones para los próximos", FORECAST_OUT, "días:")
print(predicciones)
```

`modelo.predict()` recibe un conjunto de features y devuelve un valor numérico por cada fila: en este caso, el precio de cierre estimado para cada uno de los 30 días siguientes al final de la serie simulada.

Para visualizar el resultado junto al histórico, añade una columna `Forecast` vacía y ve rellenándola fila a fila con fechas futuras, tal como hace el material original:

```python
import datetime
import matplotlib.pyplot as plt

df["Forecast"] = np.nan

ultima_fecha = df.index[-1]
un_dia = datetime.timedelta(days=1)
siguiente_fecha = ultima_fecha + un_dia

for prediccion in predicciones:
    fila_vacia = [np.nan for _ in range(len(df.columns) - 1)]
    df.loc[siguiente_fecha] = fila_vacia + [prediccion]
    siguiente_fecha += un_dia

df["Close"].plot(label="Histórico simulado")
df["Forecast"].plot(label="Predicción a 30 días")
plt.legend(loc="lower right")
plt.xlabel("Fecha")
plt.ylabel("Precio de cierre simulado")
plt.title("Serie simulada y su predicción con regresión lineal")
plt.show()
```

El bucle recorre cada predicción y la añade al dataframe en una fila nueva, con una fecha posterior a la última que ya tenías. Como en este ejemplo simulado avanzamos un día natural en cada iteración, en vez de un día bursátil como haría un dataset real, alguna fecha puede caer en fin de semana; para el objetivo didáctico de este capítulo no supone ningún problema. Incluir la fecha en el eje horizontal, aunque la fecha en sí no sea una feature del modelo, es lo que te permite leer el gráfico con sentido temporal: ver dónde termina lo conocido y dónde empieza lo predicho.

Con esto tienes el ciclo completo: simular datos, construir features y label, entrenar, evaluar y predecir. El capítulo 3 retoma este mismo modelo para guardarlo en disco con pickling y hablar de escalado e interpretación con más detalle, y el capítulo 4 te lleva a programar la regresión lineal desde cero.

Una advertencia honesta antes de cerrar esta sección: un modelo de regresión lineal como este es un buen punto de partida para aprender el flujo de trabajo, pero por sí solo no garantiza predicciones fiables en un contexto real, y mucho menos en un mercado financiero real. Cuantas más features relevantes incorpores y cuanto mejor entiendas los datos de partida, más útil será el modelo. La simplicidad de este ejemplo es intencionada: te permite centrarte en el procedimiento antes de complicarlo con datos reales.

## Resumen: lo que te llevas de este capítulo

- La regresión predice un número continuo a partir de otras variables conocidas, como estimar un peso a partir de una altura
- Las features son las columnas de entrada y el label es la columna que quieres predecir
- Desplazar una columna con `shift` te permite convertir un valor presente en un label del futuro, con un horizonte de predicción que tú eliges
- `train_test_split` separa los datos en entrenamiento y prueba para poder medir el rendimiento del modelo sobre datos que no ha visto
- `LinearRegression().fit()` entrena el modelo, y `.score()` devuelve el R², un número entre 0 y 1 que indica cuánta variación del label explica el modelo
- `.predict()` aplica el modelo entrenado sobre datos nuevos, como el tramo final sin label que reservaste para el futuro

## Ponlo en práctica

Retoma el código de este capítulo y experimenta con dos cambios, uno detrás de otro, anotando el R² que obtienes en cada caso:

**Cambia `FORECAST_OUT` de 30 a 5 y, después, a 60, y compara cómo varía la puntuación `.score()` del modelo en cada horizonte.**

**Añade o quita una feature de `columnas_features` (por ejemplo, elimina `Volume` o añade una nueva columna calculada por ti) y vuelve a entrenar el modelo para ver si el R² mejora o empeora.**

No hay una única respuesta correcta: el objetivo es que compruebes con tus propios ojos cómo el horizonte de predicción y la elección de features afectan a la calidad de un modelo de regresión.

Como reto adicional, prueba también a cambiar la semilla de `np.random.seed()` por otro número y vuelve a ejecutar todo el flujo. Verás que la serie simulada cambia por completo, pero el procedimiento (features, label, entrenamiento, evaluación y predicción) sigue siendo idéntico. Esa es precisamente la idea central de este capítulo: el flujo de trabajo con scikit-learn es el mismo, tanto si trabajas con un precio bursátil real como con cualquier otra serie de datos numéricos que quieras predecir.

**Siglas usadas en este capítulo:** ML (Machine Learning o aprendizaje automático), EITC/AI/MLP (European IT Certification / Artificial Intelligence / Machine Learning with Python), API (Application Programming Interface o interfaz de programación de aplicaciones), R² (coeficiente de determinación), SVR (Support Vector Regression o regresión de vectores de soporte), MSE (Mean Squared Error o error cuadrático medio).
