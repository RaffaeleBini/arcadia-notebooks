# 4. Programar la regresión lineal desde cero

## Por qué te importa esto

En los dos capítulos anteriores usaste scikit-learn para entrenar una regresión lineal en un par de líneas de código: `LinearRegression()`, `.fit()`, `.score()` y listo. Esa librería resuelve todo el trabajo matemático por ti, y eso está muy bien para trabajar rápido. Pero también convierte el algoritmo en una caja negra: le pasas datos, recibes un modelo, y no ves qué ocurre por dentro.

En este capítulo abres esa caja. Vas a programar la regresión lineal simple con Python y NumPy puro, sin scikit-learn, para entender exactamente de dónde salen la pendiente, el intercepto y el R cuadrado que hasta ahora solo mirabas como resultados. Esto no es un ejercicio académico sin propósito: cuando entiendes la fórmula, sabes qué significa realmente cada número que devuelve un modelo, detectas cuándo un resultado no tiene sentido y ganas la confianza necesaria para explicar tus decisiones a otras personas. Esa comprensión es la base de cualquier proyecto serio de Inteligencia Artificial (IA), incluidos los que combinan varias técnicas de aprendizaje automático.

Trabajarás con un único dataset de ejemplo, muy pequeño, y lo llevarás de principio a fin: primero calculas la recta a mano, luego mides su calidad y, al final, comprobarás que el algoritmo se comporta como esperas frente a datos distintos.

## La fórmula de la mejor recta

Imagina que alguien te entrega una hoja con una nube de puntos dibujados a lápiz y te pide que traces, a ojo, la línea recta que mejor los atraviesa. Tu mano, de forma intuitiva, busca un equilibrio: ni pasar demasiado cerca de unos puntos dejando otros muy lejos, ni inclinar la recta hacia un lado solo porque ahí hay más puntos agrupados. Buscas la línea que, en conjunto, queda más cerca de todos los puntos a la vez.

La regresión lineal automatiza exactamente ese gesto. En vez de una mano que traza a ojo, usa una fórmula matemática que calcula la única recta que minimiza la distancia total al cuadrado entre los puntos reales y la propia recta. A este método se le llama mínimos cuadrados, y la recta resultante se conoce como la recta de mejor ajuste.

Recuerda la ecuación de una recta, la misma que ya viste en la educación secundaria:

```
y = mx + b
```

Aquí, `m` es la pendiente (cuánto sube o baja la recta por cada unidad que avanza `x`) y `b` es el intercepto o punto donde la recta cruza el eje vertical, cuando `x` vale cero. Para trazar la recta de mejor ajuste solo necesitas calcular estos dos números, `m` y `b`, a partir de tus datos.

**La fórmula de la pendiente (m)**, adaptada de mínimos cuadrados a partir de medias, es esta:

```
        (media(X) · media(Y)) − media(X · Y)
m  =    ─────────────────────────────────────
             media(X)² − media(X²)
```

Dicho en palabras sencillas: multiplicas las dos medias entre sí, le restas la media de multiplicar cada `X` por su `Y` correspondiente, y ese resultado lo divides entre la diferencia que hay entre el cuadrado de la media de `X` y la media de los cuadrados de `X`. Son solo medias, multiplicaciones, restas y una división, nada de álgebra avanzada.

**La fórmula del intercepto (b)** es todavía más corta, y solo la puedes calcular una vez que ya tienes `m`:

```
b = media(Y) − m · media(X)
```

Es decir: la media de `Y`, menos la pendiente multiplicada por la media de `X`. Con `m` y `b` en la mano, ya tienes la ecuación completa de tu recta de mejor ajuste y puedes usarla para predecir un valor de `y` para cualquier `x` que quieras.

## Programar la pendiente y la recta en Python puro

Vas a trabajar con un dataset diminuto, de solo seis puntos, que te acompañará durante todo el capítulo:

```python
import numpy as np

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)
```

Fíjate en el `dtype=np.float64`: le indicas a NumPy que trabaje con números decimales de alta precisión, para que las divisiones de la fórmula no pierdan exactitud.

Ahora traduce la fórmula de la pendiente y del intercepto a una única función. NumPy te permite escribir la fórmula casi tal cual la leíste arriba, sin bucles, porque `xs` e `ys` son arreglos y las operaciones se aplican a todos sus elementos a la vez:

```python
def best_fit_slope_and_intercept(xs, ys):
    m = (
        ((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) /
        ((np.mean(xs) ** 2) - np.mean(xs ** 2))
    )
    b = np.mean(ys) - m * np.mean(xs)
    return m, b
```

Un par de detalles importantes de la fórmula, para que no te encalles si la escribes tú mismo:

- Usa el operador `**` (no `^`) para elevar al cuadrado en Python; `^` significa otra cosa y da un error
- Coloca los paréntesis exactamente donde están arriba: si divides antes de restar, PEMDAS (paréntesis, exponentes, multiplicación y división, suma y resta) deja de cumplirse y el resultado sale mal
- `np.mean(xs) ** 2` (el cuadrado de la media) y `np.mean(xs ** 2)` (la media de los cuadrados) son dos cosas distintas: no las confundas

Ahora aplica la función a tu dataset:

```python
m, b = best_fit_slope_and_intercept(xs, ys)
print(m, b)
# 0.42857142857142855 4.0
```

Con `m` y `b` calculados, ya puedes construir la recta de mejor ajuste completa, punto por punto, aplicando `y = mx + b` a cada valor de `xs`:

```python
regression_line = [(m * x) + b for x in xs]
print(regression_line)
# [4.4286, 4.8571, 5.2857, 5.7143, 6.1429, 6.5714]
```

Y, como cualquier modelo entrenado, puedes usar esa misma ecuación para predecir un valor de `y` que no está en tu dataset original. Por ejemplo, ¿qué valor de `y` esperarías si `x` valiera ocho?

```python
prediccion_x8 = (m * 8) + b
print(prediccion_x8)
# 7.428571428571429
```

Eso es todo lo que hace `LinearRegression().fit()` por dentro para una sola variable: calcula estas dos medias, resuelve la fórmula y guarda `m` y `b` para usarlos después en cada predicción.

## Medir qué tan buena es tu recta: el R cuadrado

Tener una recta no basta: necesitas saber si esa recta representa bien tus datos o si es un ajuste mediocre. Para eso existe el R cuadrado, también llamado coeficiente de determinación, la misma métrica que viste con `.score()` en el capítulo anterior. Ahora la vas a construir tú.

El punto de partida es el error al cuadrado. Para cada punto de tu dataset, el error es la distancia entre el valor real de `y` y el valor que predice tu recta para ese mismo `x`. Elevas esa distancia al cuadrado por dos motivos: te aseguras de que los errores por encima y por debajo de la recta sumen en la misma dirección, en vez de cancelarse entre sí, y penalizas con más fuerza los errores grandes que los pequeños. Sumando el error al cuadrado de todos los puntos obtienes el error cuadrático total de la recta.

Para saber si ese error es alto o bajo, necesitas un punto de comparación. El punto de comparación que usa el R cuadrado es la línea más simple posible: una recta horizontal en la media de `Y`. Calculas también el error al cuadrado de esa recta plana y comparas ambos errores con esta fórmula:

```
R² = 1 − (error al cuadrado de tu recta / error al cuadrado de la media de Y)
```

Si tu recta predice igual de mal que la simple media de `Y`, el cociente vale uno y el R cuadrado sale cero: tu modelo no aporta nada. Si tu recta predice perfectamente, el error al cuadrado de la recta es cero, el cociente vale cero y el R cuadrado sale uno: ajuste perfecto. Cuanto más cerca de uno, mejor explica tu recta la variación de los datos.

Prográmalo en dos funciones pequeñas. Primero, el error al cuadrado, que recibe los valores reales y los valores de una línea (puede ser tu recta o la línea de la media) y devuelve la suma de las diferencias al cuadrado:

```python
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)
```

Después, el coeficiente de determinación, que compara el error de tu recta con el error de la línea de la media:

```python
def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [np.mean(ys_orig) for _ in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)
```

Aplícalo a la recta que ya calculaste sobre `xs` e `ys`:

```python
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)
# 0.5844155844155845
```

Obtienes un R cuadrado de 0.58 aproximadamente. Interpretado en palabras: tu recta explica el 58% de la variación de los valores de `y`; el 42% restante queda sin explicar por el modelo. No es un ajuste perfecto, pero con solo seis puntos y una relación bastante ruidosa, es un resultado razonable.

## Poner a prueba tu algoritmo

Ya tienes las dos piezas del algoritmo: `best_fit_slope_and_intercept` y `coefficient_of_determination`. Antes de confiar en ellas, compruébalas contra datos donde ya sabes qué resultado esperar, igual que harías con una prueba unitaria en programación. Para eso necesitas generar datasets sintéticos donde tú controlas dos cosas: cuánta variación (ruido) tienen los puntos y si existe o no una correlación entre `X` e `Y`.

Esta función crea un dataset a medida: `hm` es cuántos puntos quieres, `variance` controla cuánto se dispersan los valores de `Y` alrededor de la tendencia, `step` es cuánto sube o baja `Y` en promedio en cada paso, y `correlation` puede ser `'pos'` (correlación positiva), `'neg'` (correlación negativa) o `False` (sin correlación):

```python
import random

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation == 'pos':
            val += step
        elif correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
```

Ahora usa esta función para generar tres escenarios distintos y comprueba, en cada uno, cómo responde tu R cuadrado:

```python
# Escenario 1: correlación positiva clara, poco ruido
xs1, ys1 = create_dataset(40, variance=10, step=2, correlation='pos')
m1, b1 = best_fit_slope_and_intercept(xs1, ys1)
linea1 = [(m1 * x) + b1 for x in xs1]
r2_1 = coefficient_of_determination(ys1, linea1)

# Escenario 2: correlación positiva, pero con mucho ruido
xs2, ys2 = create_dataset(40, variance=80, step=2, correlation='pos')
m2, b2 = best_fit_slope_and_intercept(xs2, ys2)
linea2 = [(m2 * x) + b2 for x in xs2]
r2_2 = coefficient_of_determination(ys2, linea2)

# Escenario 3: sin ninguna correlación
xs3, ys3 = create_dataset(40, variance=10, step=2, correlation=False)
m3, b3 = best_fit_slope_and_intercept(xs3, ys3)
linea3 = [(m3 * x) + b3 for x in xs3]
r2_3 = coefficient_of_determination(ys3, linea3)

print("Correlación fuerte, poco ruido:", r2_1)
print("Correlación fuerte, mucho ruido:", r2_2)
print("Sin correlación:", r2_3)
```

Como `create_dataset` usa números aleatorios, tus valores exactos cambiarán cada vez que ejecutes el código, pero el patrón se mantiene y es justo lo que confirma que tu algoritmo funciona bien:

- El escenario 1 debería dar un R cuadrado alto, cercano a uno: poco ruido y una tendencia clara son fáciles de ajustar con una recta
- El escenario 2 debería dar un R cuadrado más bajo que el 1, aunque siga habiendo correlación: más ruido significa más distancia entre los puntos y la recta
- El escenario 3 debería dar un R cuadrado cercano a cero: sin relación real entre `X` e `Y`, tu recta no explica casi nada de la variación

**Ejecuta el bloque completo varias veces** y observa cómo cambian los tres valores de R cuadrado en cada ejecución, manteniendo siempre el mismo orden entre ellos. Si alguna vez ves lo contrario, por ejemplo que el dataset sin correlación obtiene sistemáticamente un R cuadrado más alto que el de correlación fuerte, tienes un indicio claro de que hay un error en tu código, no en la teoría.

## Resumen: lo que te llevas de este capítulo

- La recta de mejor ajuste, `y = mx + b`, sale de dos fórmulas basadas únicamente en medias, multiplicaciones, restas y una división
- La pendiente `m` se calcula primero; el intercepto `b` depende de `m` y de las medias de `X` e `Y`
- NumPy te permite escribir estas fórmulas casi como las lees en papel, sin bucles manuales, porque opera sobre arreglos completos
- El R cuadrado compara el error al cuadrado de tu recta con el error al cuadrado de la simple media de `Y`, y va de cero (el modelo no aporta nada) a uno (ajuste perfecto)
- Generar datasets sintéticos con distinta variación y correlación es una forma sencilla de comprobar que tu algoritmo se comporta como la teoría predice
- Ahora que conoces la caja negra por dentro, todo lo que hace scikit-learn con `.fit()` y `.score()` deja de ser magia y pasa a ser aritmética que puedes explicar

## Ponlo en práctica

**Abre tu editor y copia el código de las dos primeras secciones de este capítulo.** Tu tarea es sencilla y acotada: cambia los valores de la lista `ys` del dataset original, por ejemplo a `[3, 4, 4, 6, 7, 8]`, y deja `xs` como está.

**Vuelve a ejecutar todo el flujo completo** (`best_fit_slope_and_intercept`, la construcción de `regression_line` y `coefficient_of_determination`) y anota los tres resultados nuevos: la pendiente `m`, el intercepto `b` y el R cuadrado.

Después, responde estas tres preguntas con tus propias palabras:

- ¿La pendiente subió o bajó respecto al valor original de 0.43? ¿Qué te dice eso sobre la nueva tendencia de tus datos?
- ¿El R cuadrado mejoró o empeoró respecto al 0.58 original? ¿Los nuevos puntos están más o menos alineados entre sí?
- Si ahora predices el valor de `y` para `x` igual a ocho con tu nueva recta, ¿el resultado te parece razonable dado el resto de puntos?

**Repite el ejercicio** con un tercer conjunto de valores de `ys`, esta vez diseñado a propósito para que la relación con `xs` sea prácticamente plana (por ejemplo, `[5, 5, 5, 5, 5, 5]`), y comprueba que el R cuadrado se desploma. Ese es el mismo comportamiento que ya comprobaste con `create_dataset`, ahora visto sobre un dataset que has construido tú mismo a mano.

---

**Siglas usadas en este capítulo:** IA (Inteligencia Artificial)
