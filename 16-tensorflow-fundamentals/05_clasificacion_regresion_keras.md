# 05. Clasificación, regresión y APIs de Keras

## Por qué importa este tema

Hasta ahora has entrenado modelos que clasifican imágenes: la red mira una
foto y decide a qué categoría pertenece. Pero no todos los problemas de
Machine Learning (ML) piden una categoría. A veces necesitas un número: el
precio de una vivienda, el consumo de un coche, la probabilidad de que un
tumor sea maligno. Confundir estos dos tipos de problema, o usar la función
de pérdida equivocada para cada uno, es uno de los errores más comunes de
quien empieza con TensorFlow.

Tampoco todos los datos llegan en forma de imagen. La mayoría de proyectos
reales en salud, industria o administración pública trabajan con tablas: una
fila por paciente, por pieza o por expediente, y columnas con medidas,
categorías y fechas. Ese formato tabular casi siempre se guarda como CSV
(Comma-Separated Values, valores separados por comas), así que saber cargarlo
y prepararlo bien es tan importante como saber diseñar la red.

Por último, hasta este punto solo has construido modelos con la API
(Application Programming Interface, interfaz de programación de
aplicaciones) Sequential de Keras: una pila de capas, una detrás de otra.
Esa API te lleva lejos, pero no sirve para arquitecturas con más de una
entrada, más de una salida o conexiones que se saltan capas. Para eso existe
la Functional API de Keras, que en esta clase aprenderás a usar.

Este capítulo cierra además un flanco histórico: quien consulta
documentación o código antiguo de TensorFlow se encuentra con una sintaxis
muy distinta a la que has usado hasta ahora (sesiones, marcadores de
posición, grafos que hay que ejecutar a mano). Entender qué cambió entre
TensorFlow 1.x y TensorFlow 2.0 te evita confusiones y te permite leer
ejemplos de cualquier época sin bloquearte.

## Clasificación y regresión: dos preguntas distintas

Piensa en dos preguntas que le puedes hacer a un modelo:

- "¿Esta imagen es una camiseta, un pantalón o una bota?" → **clasificación**
- "¿Cuántos kilómetros por litro hace este coche?" → **regresión**

En clasificación, la salida es una categoría de un conjunto cerrado de
opciones (dos categorías, como benigno o maligno, o varias, como los diez
tipos de prenda de Fashion MNIST). En regresión, la salida es un número
continuo que puede tomar cualquier valor dentro de un rango: 4,7, 32,1,
128,9.

Esta diferencia no es solo conceptual: cambia la última capa de tu red y,
sobre todo, cambia la función de pérdida que usas para entrenarla. La
función de pérdida mide cuánto se equivoca el modelo en cada predicción, y
el optimizador ajusta los pesos de la red para reducir ese error. Si mides
mal el error, el modelo aprende a resolver un problema distinto del que
tienes.

| Tipo de problema | Última capa típica | Función de pérdida | Ejemplo |
|---|---|---|---|
| Clasificación binaria (dos clases) | una neurona con activación `sigmoid` | `binary_crossentropy` | ¿el tumor es maligno? |
| Clasificación multiclase | *n* neuronas con activación `softmax` (o logits sin activación) | `sparse_categorical_crossentropy` | ¿qué prenda es? |
| Regresión | una neurona sin activación (lineal) | `mean_squared_error` (MSE, error cuadrático medio) o `mean_absolute_error` (MAE, error absoluto medio) | ¿cuántos kilómetros por litro? |

Una analogía útil: en clasificación, el modelo reparte una "confianza" entre
varias casillas y gana la casilla con más puntuación; en regresión, el
modelo dibuja una línea (o una superficie, si hay varias variables) e
intenta que tus datos caigan lo más cerca posible de ella. El error cuadrático
medio penaliza mucho los errores grandes porque los eleva al cuadrado; el
error absoluto medio los penaliza de forma proporcional y por eso es menos
sensible a valores atípicos.

Un ejemplo mínimo de cada caso con Keras:

```python
# Clasificación multiclase (por ejemplo, Fashion MNIST, diez clases)
modelo_clasificacion.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# Regresión (por ejemplo, predecir el consumo de un coche)
modelo_regresion.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mean_absolute_error')
```

Fíjate en las métricas también: en clasificación tiene sentido medir
`accuracy` (porcentaje de aciertos); en regresión no existe un "acierto"
exacto, así que se usan métricas como el MAE o el MSE, que indican cuánto se
aleja de media la predicción del valor real.

## Cargar y preparar datos tabulares en CSV

Un archivo CSV es, en esencia, una hoja de cálculo en texto plano: cada línea
es una fila y las comas separan las columnas. TensorFlow no puede entrenar
directamente sobre texto: primero necesitas cargar ese CSV en memoria y
convertirlo en tensores numéricos.

Para datasets pequeños que caben en memoria, el camino más simple es pandas:

```python
import pandas as pd
import numpy as np
import tensorflow as tf

# Carga el CSV en un DataFrame de pandas
datos = pd.read_csv("mi_dataset.csv")

# Separa la columna que quieres predecir (la etiqueta) del resto de columnas (las características)
etiquetas = datos.pop("columna_objetivo")
caracteristicas = np.array(datos)

# Con un único array numérico, un modelo Sequential es suficiente
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
modelo.compile(loss='mean_squared_error', optimizer='adam')
modelo.fit(caracteristicas, etiquetas, epochs=10)
```

Este flujo funciona bien cuando todas las columnas son numéricas y tienen
rangos parecidos. En la práctica, casi ningún dataset real es así:

- **Columnas con escalas muy distintas.** Si una columna mide edad en años
  (0 a 100) y otra mide ingresos en euros (0 a 100.000), la red dará más peso
  a la que tiene números más grandes sin que eso signifique que es más
  importante. La solución es la **normalización**: reescalar cada columna
  para que tenga una media de cero y una desviación típica de uno. Keras
  ofrece la capa `tf.keras.layers.Normalization` para esto: se "adapta" a
  tus datos de entrenamiento (calcula media y varianza) y luego normaliza
  cada ejemplo automáticamente, dentro del propio modelo.
- **Columnas categóricas.** Una columna como "puerto de embarque" o "tipo de
  suelo" no es un número, es una etiqueta de texto o un código. Antes de
  llegar a la red necesitas convertirla en algo numérico, normalmente
  mediante *one-hot encoding*: una columna binaria por cada categoría
  posible. Keras ofrece las capas `tf.keras.layers.StringLookup` (convierte
  cadenas de texto en índices) y `tf.keras.layers.CategoryEncoding`
  (convierte esos índices en vectores one-hot).
- **Valores ausentes o inconsistentes.** Antes de entrenar conviene revisar
  cuántos valores faltan (`datos.isna().sum()` en pandas) y decidir si los
  eliminas, los completas con un valor por defecto o los tratas como una
  categoría más.

Cuando el dataset mezcla columnas numéricas y categóricas, integrar el
preprocesamiento dentro del propio modelo (en vez de hacerlo aparte con
otra herramienta) tiene una ventaja importante: si más tarde guardas el
modelo, el preprocesamiento se guarda con él, así que cualquiera que lo
reutilice aplicará exactamente las mismas transformaciones sin tener que
recordar cómo preparaste los datos originalmente. Esto es precisamente lo
que hace posible la Functional API, como verás en el siguiente apartado.

Para datasets grandes que no caben en memoria, TensorFlow ofrece la API
`tf.data.Dataset`, con funciones como
`tf.data.experimental.make_csv_dataset` que leen el CSV por lotes
(*batches*) directamente desde disco, sin cargarlo entero de una vez, y con
métodos encadenables como `.shuffle()` (mezclar), `.batch()` (agrupar en
lotes) y `.cache()` (guardar en memoria tras la primera lectura para
acelerar las siguientes épocas de entrenamiento).

## La Sequential API frente a la Functional API

Hasta ahora todos tus modelos se han definido así:

```python
modelo = tf.keras.Sequential([
    capa_1,
    capa_2,
    capa_3
])
```

Esto es la **Sequential API**: una pila de capas donde cada una recibe la
salida de la anterior y nada más. Es como una cadena de montaje: la pieza
entra por un extremo, pasa por cada estación en orden, y sale por el otro
extremo. Es simple, legible y suficiente para la mayoría de los modelos que
has construido hasta ahora.

El problema aparece en cuanto tu arquitectura deja de ser una línea recta.
Imagina un modelo que necesita combinar dos fuentes de información
distintas, por ejemplo, el texto de una incidencia y las etiquetas que le
puso quien la reportó, para decidir a la vez su prioridad y el departamento
que debe atenderla. Esto exige **dos entradas y dos salidas**, algo que la
Sequential API no puede expresar porque solo admite una entrada, una
salida y un único camino entre ambas.

Aquí entra la **Functional API**. En vez de apilar capas dentro de una lista,
tratas cada capa como una función que se aplica sobre un tensor y devuelve
otro tensor. Esto te permite construir cualquier arquitectura que se pueda
dibujar como un grafo acíclico dirigido o DAG (Directed Acyclic Graph, grafo
dirigido sin ciclos): capas que se ramifican, se combinan o se comparten.

```python
import tensorflow as tf
from tensorflow.keras import layers

# Con la Functional API, defines primero la entrada como un objeto explícito...
entrada_numerica = tf.keras.Input(shape=(8,), name="datos_numericos")
entrada_categorica = tf.keras.Input(shape=(4,), name="datos_categoricos")

# ...y encadenas capas llamándolas como funciones sobre el tensor anterior
rama_numerica = layers.Dense(32, activation='relu')(entrada_numerica)
rama_categorica = layers.Dense(16, activation='relu')(entrada_categorica)

# Puedes combinar dos ramas que nunca podrían convivir en una Sequential
combinado = layers.concatenate([rama_numerica, rama_categorica])
salida = layers.Dense(1, activation='sigmoid')(combinado)

# El modelo se define indicando qué tensores son las entradas y cuáles la salida
modelo = tf.keras.Model(inputs=[entrada_numerica, entrada_categorica], outputs=salida)
```

Piensa en la diferencia como la que hay entre una cadena de montaje (Sequential)
y una red de tuberías (Functional): en una red de tuberías puedes tener dos
tanques de entrada que confluyen en un mismo depósito, o una tubería que se
divide en dos para alimentar dos procesos distintos. Con la Functional API
puedes:

- Definir modelos con **varias entradas** o **varias salidas**
- **Combinar** ramas distintas con capas como `concatenate` o `add`
- **Reutilizar la misma capa** en varios puntos del modelo (capas
  compartidas), útil por ejemplo cuando dos entradas de texto deben
  procesarse con el mismo vocabulario aprendido
- Crear **conexiones residuales**, donde la salida de una capa se suma a la
  salida de una capa posterior en vez de seguir solo el camino secuencial

Una vez que el modelo funcional está definido, entrenarlo, evaluarlo y
guardarlo funciona exactamente igual que con un modelo Sequential:
`model.compile()`, `model.fit()`, `model.evaluate()`, `model.save()`. La
Functional API no es una forma distinta de entrenar, es una forma distinta
de describir la arquitectura.

¿Cuándo usar cada una? Si tu modelo es una pila simple de capas, la
Sequential API es más corta y más legible. En cuanto necesites más de una
entrada, más de una salida, capas compartidas o conexiones que no siguen el
orden lineal, cambia a la Functional API.

## De TensorFlow 1.x a TensorFlow 2.0

Si alguna vez abres un tutorial antiguo de TensorFlow, es fácil que te
encuentres con código como este:

```python
# Estilo TensorFlow 1.x
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)

with tf.Session() as sess:
    resultado = sess.run(c)
    print(resultado)
```

En TensorFlow 1.x, escribir `c = tf.add(a, b)` no calculaba nada: solo
añadía un nodo a un **grafo computacional**, una especie de plano de la
operación. Para obtener un resultado real había que abrir una `tf.Session`
(sesión) y pedirle explícitamente que ejecutara ese grafo con `sess.run()`.
Los datos de entrada se inyectaban en tiempo de ejecución mediante
`tf.placeholder` (marcadores de posición) y un diccionario `feed_dict`. Este
modelo de "definir primero, ejecutar después" permitía optimizar el grafo
antes de ejecutarlo, pero hacía que depurar el código fuera incómodo: no
podías simplemente imprimir el valor de un tensor a mitad de tu programa,
porque ese tensor todavía no tenía ningún valor.

TensorFlow 2.0, lanzado en 2019, cambió esto de raíz con la **ejecución
eager** (*eager execution*) como modo por defecto: cada operación se calcula
en el momento en que la escribes, igual que en Python normal.

```python
# El mismo cálculo en TensorFlow 2.x
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)

print(c)  # tf.Tensor(5, shape=(), dtype=int32) -- ya tiene un valor, sin sesión
```

Los cambios más importantes que trajo TensorFlow 2.0 son estos:

- **Ejecución eager por defecto.** Ya no hace falta `tf.Session` ni
  `tf.placeholder`. Puedes imprimir, depurar e inspeccionar cualquier tensor
  en el momento, como con cualquier otra variable de Python.
- **Keras como API de alto nivel oficial.** Antes convivían varias formas de
  construir modelos (capas de bajo nivel, `tf.estimator`, Keras...). Desde
  la versión 2.0, Keras (`tf.keras`) es la vía recomendada para construir,
  entrenar y desplegar modelos.
- **`tf.function` para recuperar el rendimiento del grafo cuando lo
  necesitas.** La ejecución eager es más cómoda, pero un grafo compilado
  puede ejecutarse más rápido y desplegarse en más entornos. El decorador
  `@tf.function` convierte una función de Python (incluidos sus bucles y
  condicionales, gracias a una herramienta interna llamada AutoGraph) en un
  grafo optimizado, sin que tengas que volver a la sintaxis de sesiones. Lo
  viste en la práctica en el ejemplo de bucle de entrenamiento personalizado
  con `tf.GradientTape`, donde `train_step` y `test_step` se marcan con
  `@tf.function`.
- **API simplificada y más consistente.** TensorFlow 2.0 eliminó
  redundancias entre módulos y unificó comportamientos, lo que reduce la
  cantidad de código repetitivo (*boilerplate*) necesario para tareas
  habituales.
- **Un único formato de guardado (`SavedModel`).** El mismo modelo guardado
  con `model.save()` puede servirse con TensorFlow Serving, ejecutarse en el
  navegador con TensorFlow.js o convertirse a TensorFlow Lite para
  dispositivos móviles, sin volver a entrenar nada.

En la práctica, todo el código que has escrito en este curso ya está escrito
en el estilo de TensorFlow 2.x. Conocer el estilo antiguo te sirve sobre
todo para dos cosas: entender documentación o proyectos heredados que
todavía usan `tf.compat.v1`, y apreciar por qué el código actual es más
corto y más fácil de depurar.

## Ejemplo de código comentado: clasificar tumores con datos tabulares

Vamos a aplicar todo lo anterior a un caso real: el dataset de cáncer de
mama de Wisconsin, que recoge 30 medidas numéricas extraídas de imágenes de
biopsias (tamaño, textura, simetría de los núcleos celulares, entre otras) y
una etiqueta que indica si el tumor es benigno o maligno. Es un problema de
**clasificación binaria** con **datos tabulares**, así que combina los tres
primeros bloques de este capítulo.

```python
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Cargar los datos. sklearn incluye este dataset ya limpio, sin valores
# ausentes, así que nos ahorramos ese paso de limpieza.
datos = load_breast_cancer()
X, y = datos.data, datos.target
# En este dataset, 0 = maligno y 1 = benigno (target_names lo confirma)

# 2. Dividir en entrenamiento y prueba, igual que harías con cualquier CSV
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# 3. Normalizar las características numéricas dentro del propio modelo.
# La capa Normalization se "adapta" primero a los datos de entrenamiento...
normalizador = tf.keras.layers.Normalization(axis=-1)
normalizador.adapt(X_train)

# 4. Construir el modelo. Es un problema de clasificación binaria, así que
# la última capa tiene una única neurona con activación sigmoid: la salida
# será un número entre 0 y 1 que interpretamos como una probabilidad.
modelo = tf.keras.Sequential([
    normalizador,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 5. Compilar con la función de pérdida correcta para clasificación binaria
modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# 6. Entrenar, reservando una parte de los datos de entrenamiento para validar
historial = modelo.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    verbose=0)

# 7. Evaluar sobre el conjunto de prueba, que el modelo no ha visto nunca
perdida, precision = modelo.evaluate(X_test, y_test, verbose=0)
print(f"Precisión en el conjunto de prueba: {precision:.2%}")
```

Compara este ejemplo con el enfoque original del notebook `BreastCancer.ipynb`:
allí se subían manualmente cuatro archivos CSV ya divididos y normalizados
fuera de TensorFlow, y el modelo se compilaba con `rmsprop` y
`binary_crossentropy` sin capa de normalización. El resultado es equivalente,
pero integrar la división de datos y la normalización dentro de este flujo
hace el proceso reproducible y evita el paso manual de subir archivos cada
vez que ejecutas el notebook.

## Resumen de puntos clave

- La clasificación predice una categoría de un conjunto cerrado de opciones
  la regresión predice un número continuo
- La función de pérdida debe encajar con el tipo de problema:
  `binary_crossentropy` para clasificación binaria,
  `sparse_categorical_crossentropy` para clasificación multiclase, y
  `mean_squared_error` o `mean_absolute_error` para regresión
- Los datos tabulares en CSV necesitan preprocesamiento antes de entrenar:
  separar características de etiquetas, normalizar columnas numéricas y
  convertir columnas categóricas a una representación numérica
- Las capas de preprocesamiento de Keras (`Normalization`, `StringLookup`,
  `CategoryEncoding`) permiten integrar ese preprocesamiento dentro del
  propio modelo, para que viaje con él cuando lo guardas
- La Sequential API sirve para pilas simples de capas la Functional API es
  necesaria para modelos con varias entradas, varias salidas, capas
  compartidas o conexiones no lineales
- TensorFlow 2.0 sustituyó el modelo de grafos y sesiones de TensorFlow 1.x
  por ejecución eager por defecto, con Keras como API de alto nivel oficial
  y `tf.function` disponible para recuperar el rendimiento de un grafo
  compilado cuando hace falta

## Ponlo en práctica

Abre el notebook de práctica de esta clase
(`clase03_practica_clasificacion_regresion_keras.ipynb`) y completa el
ejercicio de clasificación con el dataset de cáncer de mama: te falta
decidir la función de pérdida y la activación de la última capa a partir de
lo que has leído en este capítulo. Después, en el segundo ejercicio,
construye con la Functional API un modelo con dos entradas independientes
que se combinan antes de la salida final.

Antes de mirar la solución, hazte esta pregunta: si en vez de predecir si un
tumor es benigno o maligno tuvieras que predecir el tamaño exacto de ese
tumor en milímetros, ¿qué cambiarías en el modelo? Piensa en la última capa,
en la función de pérdida y en la métrica que usarías para evaluar el
resultado.
