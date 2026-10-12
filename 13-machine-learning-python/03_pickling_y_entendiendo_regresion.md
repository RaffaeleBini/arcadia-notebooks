# 3. Guardar modelos, escalado e interpretación de la regresión

## Por qué te importa esto

En el capítulo anterior entrenaste tu primer modelo de regresión lineal con scikit-learn sobre una serie de precios simulada con NumPy. Funcionó, pero seguramente notaste algo: cada vez que quieres usarlo, tienes que volver a ejecutar todo el script desde cero, incluido el entrenamiento. Si tu conjunto de datos es pequeño, esto apenas se nota. Si algún día trabajas con miles o millones de filas, reentrenar cada vez que necesitas una predicción puede tardar minutos u horas, un coste que no tiene sentido asumir una y otra vez.

Este capítulo resuelve tres problemas prácticos que vas a encontrar en cualquier proyecto real de aprendizaje automático (ML, del inglés Machine Learning):

- Cómo guardar un modelo ya entrenado para reutilizarlo sin repetir el entrenamiento
- Cómo preparar los datos cuando sus columnas tienen escalas muy distintas entre sí
- Cómo leer con criterio los números que scikit-learn ya te entrega tras entrenar: la pendiente, el punto de corte y la puntuación de ajuste

No vas a programar ninguna fórmula matemática todavía; eso llega en el capítulo cuatro, cuando construyas la regresión lineal desde cero. Aquí te centras en aprovechar al máximo lo que la librería ya calcula por ti.

## Guardar tu modelo entrenado con pickle (con una analogía cotidiana, ej. guardar una partida de un videojuego para no empezar de cero)

Piensa en un videojuego largo. Si cada vez que lo abrieras tuvieras que empezar desde el primer nivel, la mayoría de las personas abandonaría antes de terminarlo. Por eso existe la función de guardar partida: capturas el estado exacto del juego en un archivo y, más tarde, lo recuperas justo donde lo dejaste. Un modelo de ML entrenado funciona igual. Entrenarlo es como avanzar niveles: consume tiempo y recursos. Una vez que el modelo "aprendió" sus parámetros a partir de los datos, quieres guardar ese estado para no repetir el proceso cada vez que abras tu programa.

En Python, este mecanismo se llama pickling. El módulo `pickle`, incluido en la instalación estándar del lenguaje, convierte casi cualquier objeto de Python (un modelo, una lista, un diccionario) en una secuencia de bytes que puedes escribir en un archivo. Después, con la operación inversa, reconstruyes el objeto original tal como estaba, sin volver a calcular nada.

Vamos a retomar la serie de precios simulada del capítulo anterior, entrenar un modelo y guardarlo:

```python
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Simulamos de nuevo una serie de precios con un camino aleatorio (random walk)
np.random.seed(42)
dias = np.arange(100).reshape(-1, 1)          # variable independiente: el día
pasos = np.random.normal(loc=0, scale=1, size=100)
precios = 100 + np.cumsum(pasos)              # variable dependiente: el precio simulado

# Entrenamos el modelo, igual que en el capítulo anterior
modelo = LinearRegression()
modelo.fit(dias, precios)

# Guardamos el modelo entrenado en un archivo binario
with open("modelo_precios.pkl", "wb") as archivo:
    pickle.dump(modelo, archivo)

print("Modelo guardado en modelo_precios.pkl")
```

Fíjate en dos detalles. Primero, abres el archivo en modo `"wb"` (write binary, escritura binaria), porque el contenido no es texto legible, sino bytes. Segundo, por convención, los archivos generados con `pickle` llevan la extensión `.pkl`, aunque Python no te obliga a usarla.

Ahora simula que cierras tu programa, apagas el ordenador y vuelves otro día. No necesitas el código de entrenamiento; solo el archivo guardado:

```python
import pickle

# Cargamos el modelo ya entrenado, sin volver a entrenarlo
with open("modelo_precios.pkl", "rb") as archivo:
    modelo_cargado = pickle.load(archivo)

# Lo usamos directamente para predecir el precio del día 150
prediccion = modelo_cargado.predict([[150]])
print(f"Precio estimado para el día 150: {prediccion[0]:.2f}")
```

Esta vez el archivo se abre en modo `"rb"` (read binary, lectura binaria). El modelo cargado es funcionalmente idéntico al original: mismos coeficientes, mismas predicciones, cero minutos de entrenamiento.

Existe una alternativa moderna a `pickle` que muchos proyectos de ML prefieren: la librería `joblib`. Su sintaxis es casi idéntica y funciona mejor con modelos que almacenan grandes arreglos numéricos internamente, algo habitual en scikit-learn:

```python
import joblib

# Guardar con joblib
joblib.dump(modelo, "modelo_precios.joblib")

# Cargar con joblib
modelo_cargado = joblib.load("modelo_precios.joblib")
```

Una advertencia importante: un modelo guardado no aprende nada nuevo por sí solo. Si los datos del mundo real cambian con el tiempo, tu modelo guardado sigue reflejando patrones antiguos. Por eso conviene establecer un calendario de reentrenamiento, por ejemplo una vez al mes, en vez de asumir que un modelo pickled sirve para siempre.

**Guarda tu propio modelo ahora mismo** con el código anterior y comprueba que el archivo aparece en tu carpeta de trabajo.

## Por qué a veces hay que escalar los datos

Imagina que quieres predecir el precio de una vivienda a partir de dos características: su superficie en metros cuadrados (valores entre 40 y 300, aproximadamente) y su antigüedad en años (valores entre 0 y 50). Estas dos columnas viven en escalas completamente distintas. Para muchos algoritmos, esa diferencia de magnitud no es un simple detalle estético: puede hacer que la característica con números más grandes domine el aprendizaje, aunque no sea la más relevante para el problema.

El escalado consiste en transformar cada columna para que todas queden en rangos comparables, sin alterar la relación relativa entre sus valores. La técnica más usada se llama estandarización: resta la media de cada columna y divide entre su desviación estándar, de modo que el resultado tiene media cero y varianza uno. Scikit-learn la implementa en la clase `StandardScaler`.

Veamos el efecto en la práctica, ampliando nuestro ejemplo con una segunda característica simulada, el volumen de operaciones de cada día, que se mueve en una escala mucho mayor que el propio día:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
dias = np.arange(100)
volumen = np.random.normal(loc=50000, scale=8000, size=100)  # escala grande
precios = 100 + np.cumsum(np.random.normal(0, 1, size=100))

# Combinamos ambas características en una sola matriz
X = np.column_stack([dias, volumen])
y = precios

# Antes de escalar: mira la diferencia de magnitud entre columnas
print("Sin escalar, primera fila:", X[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustamos el escalador SOLO con los datos de entrenamiento
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)

# Aplicamos la MISMA transformación aprendida a los datos de prueba
X_test_escalado = escalador.transform(X_test)

print("Escalado, primera fila:", X_train_escalado[0])
```

Antes de escalar, la columna del volumen tiene valores de decenas de miles mientras que la del día apenas llega a un centenar. Después de escalar, ambas columnas quedan alrededor de cero, con una dispersión similar. Fíjate en un detalle que marca la diferencia entre un flujo correcto y uno con fugas de información: ajustas (`fit_transform`) el escalador únicamente con los datos de entrenamiento, y luego aplicas (`transform`, sin `fit`) esa misma transformación a los datos de prueba. Si ajustaras el escalador con todo el conjunto, tu modelo tendría acceso indirecto a información de los datos de prueba antes de la evaluación, y eso falsearía tus resultados.

Para la regresión lineal simple con una sola característica, como la del capítulo anterior, el escalado no cambia demasiado la calidad de las predicciones. Su verdadero valor aparece con varias características de escalas distintas: mejora la interpretación de los coeficientes, porque quedan en unidades comparables, y ayuda a que algoritmos de optimización como el descenso de gradiente converjan más rápido.

**Aplica `StandardScaler` a tus propios datos** antes de entrenar cualquier modelo con más de una característica.

## Qué significan de verdad los coeficientes, el intercepto y el R²

Cuando entrenas un modelo de regresión lineal con scikit-learn, tres atributos concentran toda la información que necesitas para interpretarlo: `coef_`, `intercept_` y el resultado de `score()`. Vale la pena entender qué representa cada uno más allá del número en pantalla.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

np.random.seed(42)
dias = np.arange(100).reshape(-1, 1)
precios = 100 + np.cumsum(np.random.normal(0, 1, size=100))

modelo = LinearRegression()
modelo.fit(dias, precios)

print("Pendiente (coef_):", modelo.coef_)
print("Intercepto:", modelo.intercept_)
print("R² sobre los mismos datos:", modelo.score(dias, precios))
```

**El coeficiente (`coef_`), o pendiente.** Indica cuánto cambia la predicción por cada unidad adicional de la variable independiente. Si `coef_` vale 0,8, cada día adicional añade, en promedio, 0,8 unidades de precio según el patrón aprendido. Un coeficiente positivo señala una relación creciente; uno negativo, una relación decreciente. Cuando tienes varias características, cada una recibe su propio coeficiente, y ahí es donde el escalado de la sección anterior facilita comparar cuál pesa más.

**El intercepto (`intercept_`).** Es el valor que predice el modelo cuando todas las variables independientes valen cero. En nuestro ejemplo, sería el precio estimado para el día cero. A veces este valor tiene sentido real; otras veces es solo un punto matemático necesario para definir la recta, sin interpretación práctica directa (por ejemplo, si "antigüedad de la vivienda igual a cero" no aparece nunca en tus datos reales). Interprétalo con cautela cuando quede fuera del rango de tus datos originales.

**El coeficiente de determinación (R²), obtenido con `score()`.** Mide qué proporción de la variabilidad de tus datos explica el modelo, en una escala que normalmente va de 0 a 1 (puede volverse negativo si el modelo predice peor que simplemente usar el promedio). Un R² de 0,95 significa que el modelo explica el 95% de la variación observada en tus datos: un ajuste muy bueno. Un R² de 0,3 significa que el modelo solo explica el 30%: la recta capta algo de la tendencia, pero deja fuera la mayor parte del comportamiento de los datos, probablemente porque faltan variables relevantes o porque la relación no es lineal.

Un R² alto no garantiza automáticamente un buen modelo para producción, ni uno bajo lo descarta siempre: depende del problema. Predecir el comportamiento humano o los mercados financieros con un R² de 0,3 puede ser un resultado aceptable, mientras que un R² de 0,3 en un proceso físico controlado suele indicar que falta algo importante en tu planteamiento.

## Resumen: lo que te llevas de este capítulo

- El módulo `pickle` serializa un modelo entrenado en un archivo, así evitas reentrenar cada vez que lo necesitas
- `pickle.dump()` guarda el objeto en modo escritura binaria (`"wb"`); `pickle.load()` lo recupera en modo lectura binaria (`"rb"`)
- `joblib` es una alternativa moderna a `pickle`, recomendable para modelos de scikit-learn con arreglos numéricos grandes
- `StandardScaler` transforma tus columnas a media cero y varianza uno, evitando que una característica con escala mayor domine el aprendizaje
- Ajusta el escalador solo con los datos de entrenamiento y aplícalo después a los datos de prueba, nunca al revés
- `coef_`, `intercept_` y `score()` te dicen, respectivamente, cuánto influye cada variable, dónde arranca la recta y qué porcentaje de la variación explica tu modelo

## Ponlo en práctica

Realiza este ejercicio en dos archivos separados, simulando dos sesiones de trabajo distintas.

En el primer archivo (`entrenar.py`), genera una serie de datos simulados con NumPy, entrena una `LinearRegression`, guarda una predicción de referencia y guarda el modelo con `pickle`:

```python
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

np.random.seed(7)
X = np.arange(50).reshape(-1, 1)
y = 3 * X.flatten() + np.random.normal(0, 2, size=50)

modelo = LinearRegression()
modelo.fit(X, y)

prediccion_referencia = modelo.predict([[25]])
print("Predicción antes de guardar:", prediccion_referencia)

with open("mi_modelo.pkl", "wb") as archivo:
    pickle.dump(modelo, archivo)
```

**Cierra este archivo y abre uno nuevo (`usar_modelo.py`)**, sin copiar el código de entrenamiento, para simular que trabajas otro día. Carga el modelo y comprueba que la predicción coincide exactamente con la de referencia:

```python
import pickle

with open("mi_modelo.pkl", "rb") as archivo:
    modelo_cargado = pickle.load(archivo)

prediccion_nueva = modelo_cargado.predict([[25]])
print("Predicción tras cargar:", prediccion_nueva)
```

**Compara ambos resultados** y confirma que son idénticos. **Imprime también** `modelo_cargado.coef_`, `modelo_cargado.intercept_` y `modelo_cargado.score(X, y)`, y escribe en un comentario qué significa cada valor según lo que aprendiste en este capítulo. Si te sobra tiempo, **repite el ejercicio** sustituyendo `pickle` por `joblib` y comprueba que el resultado no cambia.

**Siglas usadas en este capítulo:** ML (Machine Learning), R² (coeficiente de determinación).
