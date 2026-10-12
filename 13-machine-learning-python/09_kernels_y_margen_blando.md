# 9. Kernels y margen blando en SVM

## Por qué te importa esto

En el capítulo siete descubriste la intuición de la máquina de vectores de soporte (SVM): un algoritmo que busca la línea, o "calle", que separa dos clases dejando el mayor margen posible a cada lado. En el capítulo ocho programaste esa idea desde cero, con un margen "duro" que exige que todos los puntos queden bien clasificados, sin excepciones.

Ese planteamiento tiene dos límites evidentes. Primero, muchos datasets reales no se pueden separar con una línea recta, por muy bien que coloques el hiperplano. Segundo, incluso cuando casi se pueden separar, exigir una separación perfecta te deja a merced de un solo punto ruidoso o mal medido, que puede torcer todo el resultado.

Este capítulo, el último dedicado a SVM, resuelve ambos problemas. Primero vas a conocer los kernels, la herramienta que permite separar datos con formas complicadas sin cambiar ni una línea del algoritmo base. Después vas a flexibilizar la exigencia de separación perfecta con el margen blando, controlado por un parámetro llamado `C`. Cierras el capítulo ajustando ambas piezas —kernel y `C`— con la clase `SVC`, la misma herramienta de scikit-learn que ya usaste antes.

Piensa en un problema de clasificación con dos características, por ejemplo la temperatura y la humedad de una muestra, y dos clases mezcladas de forma circular: una categoría en el centro y la otra formando un anillo a su alrededor. Ninguna recta separa esas dos clases, por muy bien que la coloques, así que el SVM del capítulo ocho fracasaría de raíz ante este dataset. Y aunque encontraras un problema donde las clases casi se pudieran separar con una recta, seguiría existiendo el riesgo de que un solo dato atípico —una medición errónea, un caso raro— impidiera al margen duro encontrar cualquier solución. Los kernels resuelven el primer problema; el margen blando, el segundo. Juntos convierten la SVM, todavía muy rígida al final del capítulo ocho, en un algoritmo que puedes aplicar con confianza a datos reales.

## El truco del kernel: separar lo que no se puede separar con una línea recta

Imagina que tienes un puñado de canicas rojas formando un círculo, rodeado por un anillo azul. Mirando la mesa desde arriba, en dos dimensiones, no existe ninguna línea recta capaz de separar el círculo rojo del anillo azul: cualquier trazo que hagas deja canicas de los dos colores a cada lado. Pero si levantas esa misma disposición y la miras desde un ángulo distinto, por ejemplo empujando las canicas rojas del centro un poco hacia arriba, en tres dimensiones, de repente sí puedes deslizar una lámina plana entre ambos grupos. El problema seguía siendo el mismo: solo cambiaste el punto de vista.

Esto es exactamente lo que hacen los kernels en SVM. Un kernel es una función que transforma tus datos, añadiendo dimensiones nuevas, hasta un espacio donde sí existe una separación lineal. La diferencia con "añadir columnas a mano" —por ejemplo, calcular `x3 = x1 * x2` e incorporarlo como dato adicional— es que hacerlo explícitamente se vuelve carísimo en cuanto el dataset crece en características o quieres subir a dimensiones muy altas. El kernel evita ese coste: calcula el resultado que obtendrías en esa dimensión superior sin llegar a construirla nunca. A esto se le llama el "truco del kernel".

¿Por qué funciona este atajo? Porque, si revisas las fórmulas del capítulo ocho, verás que el algoritmo de SVM solo necesita el producto punto entre pares de vectores, nunca sus coordenadas por separado. A esta operación también se la llama producto interno, o inner product en inglés: da igual el nombre, siempre toma dos vectores y devuelve un único número que mide su similitud. Un kernel es precisamente una función que calcula ese producto punto tal como quedaría en el espacio transformado, sin viajar hasta allí. No necesitas dominar la demostración matemática completa para aprovecharlo: te basta con saber que existe esta puerta trasera y que scikit-learn ya la implementa por ti.

Un ejemplo ayuda a fijar la idea. Supón que tienes dos características, `x1` y `x2`, y quieres pasar a un espacio de segundo orden que incluya `x1²`, `x2²` y `x1 * x2`. Esto es justo lo que evita el kernel polinómico. Construir esas tres columnas nuevas para cada dato exige tiempo de cálculo. Multiplicar dos vectores ya transformados de esa manera, en cambio, equivale exactamente a aplicar la fórmula `K(x, x') = (1 + x · x')^p` sobre los valores originales, sin construir nada explícitamente. El resultado es matemáticamente idéntico, pero mucho más barato de calcular. Con dos o tres características, el ahorro de tiempo es pequeño. Con cientos o miles, como ocurre al analizar imágenes, usar el kernel en lugar de construir el espacio transformado a mano es la diferencia entre un cálculo instantáneo y otro impracticable.

Los kernels más usados son:

- **Lineal**: no transforma nada, es la SVM que ya conoces de los capítulos siete y ocho
- **Polinómico** (`poly`): añade combinaciones de potencias de las características originales, útil cuando la frontera entre clases tiene forma de curva
- **RBF** (Radial Basis Function o función de base radial): mide la similitud entre puntos según su distancia, y es capaz de dibujar fronteras muy flexibles; es el kernel por defecto en scikit-learn
- **Sigmoide**: inspirado en las redes neuronales, útil en algunos problemas de clasificación binaria

Compruébalo con un dataset que, por construcción, no admite ninguna línea recta separadora: dos anillos concéntricos, generados con `make_circles` de scikit-learn.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

# Generas un dataset con dos anillos concéntricos: imposible de
# separar con una línea recta
X, y = make_circles(n_samples=200, factor=0.4, noise=0.08, random_state=42)

# Un kernel lineal intenta trazar una recta y falla
svm_lineal = SVC(kernel='linear')
svm_lineal.fit(X, y)
print("Precisión con kernel lineal:", svm_lineal.score(X, y))

# Un kernel RBF "levanta" los datos a otra dimensión y sí los separa
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X, y)
print("Precisión con kernel RBF:", svm_rbf.score(X, y))
```

Ejecuta este código y compara las dos precisiones. El kernel lineal se queda cerca del 50%, lo mismo que acertar al azar entre dos clases, porque ninguna recta separa un círculo interior de un anillo exterior. El kernel RBF, en cambio, se acerca al 100%, porque encuentra una frontera curva que rodea el círculo interior.

Puedes visualizar esa frontera curva directamente:

```python
import numpy as np

# Creamos una rejilla de puntos que cubre todo el gráfico
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                      np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))
Z = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("SVM con kernel RBF sobre dos anillos concéntricos")
plt.show()
```

Al ejecutarlo verás una frontera curva, casi circular, que separa el anillo interior del exterior; algo que el kernel lineal nunca podría dibujar. Si quieres comprobarlo con otro tipo de datos no lineales, prueba `make_moons` en lugar de `make_circles` y repite el experimento con `kernel='poly'`: verás que también encuentra una frontera curva, aunque con una forma distinta a la del RBF.

## El margen blando: qué hacer cuando los datos no son perfectamente separables

Piensa en quien corrige exámenes con una regla de aprobado muy estricta: cualquier persona que se quede a un solo punto de la nota de corte, suspende, sin excepciones. Esa regla es sencilla, pero es frágil: un único examen mal puntuado por error, o un mal día de quien lo presenta, tuerce toda la clasificación de aprobados y suspensos. Quien corrige de forma más flexible, en cambio, acepta que algún caso límite entre en la categoría "aprobado" aunque no cumpla la regla al cien por cien, porque sabe que exigir una separación perfecta puede ser peor que aceptar unas pocas excepciones razonables.

El margen duro que programaste en el capítulo ocho es como esa primera regla: exige que la restricción `yi * (w · xi + b) >= 1` se cumpla para todos los puntos, sin excepción. Si un solo punto la incumple, el algoritmo ni siquiera encuentra una solución. El margen blando introduce una variable de holgura (o "slack") por cada punto del dataset, que mide cuánto se le permite violar la restricción. La condición pasa a ser:

```
yi * (w . xi + b) >= 1 - slack_i,  con slack_i >= 0
```

Cuando `slack_i` vale cero, ese punto cumple la condición estricta de siempre. Cuanto mayor es `slack_i`, más se le permite invadir el margen o incluso caer del lado equivocado del hiperplano. El objetivo de la optimización ya no es solo minimizar la magnitud de `w` (maximizar el margen), sino también reducir la suma de todas las holguras, ponderada por un nuevo parámetro: `C`.

`C` es exactamente el dial que decide cuán estricta es esa corrección. Un valor de `C` alto penaliza mucho las violaciones y hace que el algoritmo se comporte casi como un margen duro: prioriza clasificar bien cada punto de entrenamiento, aunque eso implique una calle estrecha y un hiperplano muy sensible a casos concretos, con más riesgo de sobreajuste. Con un valor de `C` bajo, el algoritmo tolera más violaciones a cambio de un margen ancho y una frontera más suave, que generaliza mejor a datos nuevos pero puede clasificar mal algún punto de entrenamiento por el camino.

No existe un valor de `C` universalmente correcto: depende de cuánto ruido tenga tu dataset y de si prefieres arriesgarte al sobreajuste o al infraajuste. La forma habitual de decidirlo es probar varios valores y comparar el resultado sobre datos de prueba, algo que harás en la próxima sección.

Existe además una señal indirecta muy práctica para saber si tu elección de `C` (o de kernel) va por buen camino: el número de vectores de soporte que termina usando el modelo entrenado. Si ese número se acerca al total de muestras de entrenamiento, es una pista de que el modelo está memorizando casos concretos en lugar de aprender un patrón general, síntoma típico de sobreajuste. Como referencia orientativa, si los vectores de soporte superan el 10% de las muestras totales, conviene sospechar de sobreajuste y probar con un `C` menor o con otro kernel. En scikit-learn, puedes consultar ese número tras entrenar con el atributo `support_vectors_` de tu modelo `SVC`.

## Resolver la optimización con CVXOPT

En el capítulo ocho resolviste la optimización de la SVM a mano, con un método sencillo de prueba y descarte de valores de `w`. Ese método funciona para aprender, pero no es el que se usa en la práctica: el reto real de entrenar una SVM con margen blando y kernels es un problema de programación cuadrática (QP, del inglés Quadratic Programming), y existen bibliotecas especializadas para resolverlo de forma mucho más eficiente.

Una de esas bibliotecas es CVXOPT (Convex Optimization, optimización convexa), pensada en Python para resolver problemas convexos, entre ellos los de programación cuadrática. No hace falta que implementes tú mismo el SVM con kernel completo: ya viste la idea de optimización en el capítulo anterior y aquí basta con entender qué papel cumple CVXOPT y cómo se usa a un nivel básico.

CVXOPT no viene preinstalada en Google Colab, así que en el notebook de esta clase la instalas con un simple comando al principio, antes de importarla:

```python
!pip install cvxopt
```

El corazón de CVXOPT para esta tarea es la función `cvxopt.solvers.qp()`, que resuelve un problema con esta forma general:

```
minimizar   (1/2) * x^T * P * x + q^T * x
sujeto a    G * x <= h
            A * x = b
```

Aquí `x` es el vector de incógnitas que quieres encontrar (en el caso de la SVM, los multiplicadores de Lagrange asociados a cada punto), `P` y `q` describen la función que minimizas, y `G`, `h`, `A`, `b` definen las restricciones, incluida la que limita cada multiplicador según el parámetro `C` del margen blando. Un ejemplo mínimo, solo para ver la mecánica de la función, resuelve un problema cuadrático sencillo con dos variables:

```python
import numpy as np
from cvxopt import matrix, solvers

# Problema de ejemplo: minimizar 2x1^2 + x2^2 + x1*x2,
# sujeto a x1 >= 0, x2 >= 0, x1 + x2 >= 1
P = matrix(np.array([[4.0, 1.0], [1.0, 2.0]]))
q = matrix(np.array([0.0, 0.0]))
G = matrix(np.array([[-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]]))
h = matrix(np.array([0.0, 0.0, -1.0]))

solvers.options['show_progress'] = False
solucion = solvers.qp(P, q, G, h)
print("Solución x:", np.array(solucion['x']).flatten())
```

Al ejecutarlo, `solucion['x']` te da los valores óptimos de `x1` y `x2`. En el caso de una SVM con kernel, `P` se construye a partir de la matriz de kernels (el resultado de aplicar `K(xi, xj)` a cada par de puntos del dataset), y las filas de `G` y `h` incorporan tanto la condición `alpha_i >= 0` del margen duro como el límite superior `alpha_i <= C` que introduce el margen blando. Una vez resuelto el problema, los multiplicadores distintos de cero señalan cuáles son los vectores de soporte, exactamente el mismo concepto que ya conoces del capítulo siete.

De hecho, si programaras tu propia clase SVM apoyada en CVXOPT, como continuación natural del capítulo ocho, el parámetro `C` aparecería en el método `__init__` de esa clase: pasar `C=None` activaría un margen duro idéntico al que ya programaste, mientras que asignarle cualquier número activaría el margen blando descrito en la sección anterior, limitando el rango que puede tomar cada multiplicador de Lagrange en la matriz `G` del problema de programación cuadrática. El método `fit` de esa clase prepararía las matrices `P`, `q`, `G`, `h`, `A` y `b`, llamaría a `solvers.qp()` y extraería del resultado los multiplicadores de Lagrange, los vectores de soporte, el sesgo y la función de proyección necesaria para clasificar puntos nuevos.

En la práctica, cuando entrenas una SVM con `SVC` de scikit-learn, no ves nada de esto: la biblioteca resuelve este problema de programación cuadrática por dentro, con solucionadores todavía más especializados que CVXOPT, como los que implementa la biblioteca LIBSVM sobre la que se apoya scikit-learn. Conocer esta pieza te sirve para entender qué ocurre realmente al llamar a `fit()`, aunque nunca necesites escribir una línea de CVXOPT en un proyecto real.

## Ajustar los parámetros de SVM en scikit-learn

Con la teoría de kernels y margen blando ya cubierta, toca ponerla en práctica con los parámetros de `SVC`: `kernel`, `C` y `gamma`. Recuerda el dataset de cáncer de mama que ya usaste en los capítulos cinco y siete para clasificar tumores como benignos o malignos.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

datos = load_breast_cancer()
X = datos.data
y = datos.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Un primer acercamiento, antes de automatizar la búsqueda, es un bucle simple que recorre varios valores de `C` y de `kernel` y anota la precisión de cada combinación:

```python
kernels = ['linear', 'rbf', 'poly']
valores_de_c = [0.1, 1, 10, 100]

for kernel in kernels:
    for c in valores_de_c:
        modelo = SVC(kernel=kernel, C=c)
        modelo.fit(X_train, y_train)
        precision = modelo.score(X_test, y_test)
        print(f"kernel={kernel:6s}  C={c:<6}  precisión={precision:.4f}")
```

Fíjate en que, además de `kernel` y `C`, `SVC` acepta el parámetro `gamma`, que solo tiene efecto con los kernels `rbf`, `poly` y `sigmoid`. `gamma` controla cuánta influencia tiene cada punto de entrenamiento sobre la frontera de decisión. Con un valor bajo, esa influencia llega lejos y la frontera de decisión queda suave. Con un valor alto, en cambio, la influencia se concentra cerca de cada punto y la frontera se ajusta mucho al detalle de los datos, con más riesgo de sobreajuste. Por defecto, scikit-learn calcula un valor razonable de `gamma` a partir del número de características (la opción `'scale'`), así que no siempre necesitas tocarlo a mano; solo conviene explorarlo cuando ya has fijado un kernel no lineal y quieres afinar el resultado.

Al recorrer la salida del bucle anterior, vas a notar un patrón habitual en este dataset. Con `kernel='linear'`, la precisión apenas cambia entre distintos valores de `C`, porque los datos ya son casi separables con una recta. Con `kernel='rbf'` o `kernel='poly'`, en cambio, el resultado sí varía de forma notable según `C`, porque la frontera curva tiene mucho más margen para ajustarse —o sobreajustarse— a los datos de entrenamiento. Esa diferencia de comportamiento es justo la que capturaste en la teoría de las secciones anteriores: cuantas más dimensiones y más libertad tiene la frontera, más importa vigilar el equilibrio entre margen y violaciones.

Cuando quieres explorar varias combinaciones de parámetros de forma más ordenada, sin anidar bucles a mano, `GridSearchCV` hace el trabajo por ti, probando cada combinación con validación cruzada y quedándose con la mejor:

```python
from sklearn.model_selection import GridSearchCV

rejilla_parametros = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.001]
}

busqueda = GridSearchCV(SVC(), rejilla_parametros, cv=5)
busqueda.fit(X_train, y_train)

print("Mejores parámetros:", busqueda.best_params_)
print("Mejor precisión en validación cruzada:", busqueda.best_score_)
print("Precisión sobre el conjunto de prueba:", busqueda.score(X_test, y_test))
```

`GridSearchCV` entrena una SVM por cada combinación de la rejilla, evaluándola con cinco particiones de validación cruzada (`cv=5`), y guarda en `best_params_` la que obtiene mejor rendimiento medio. Es la forma habitual de decidir, con datos y no a ojo, qué kernel y qué valor de `C` conviene a un problema concreto.

Una vez que `GridSearchCV` te da su mejor combinación, aplica la comprobación de sobreajuste que viste en la sección del margen blando, ahora sobre el modelo ganador:

```python
mejor_modelo = busqueda.best_estimator_
n_vectores_soporte = len(mejor_modelo.support_vectors_)
n_muestras = len(X_train)

print("Vectores de soporte:", n_vectores_soporte, "de", n_muestras)
print("Porcentaje:", round(100 * n_vectores_soporte / n_muestras, 1), "%")
```

Si ese porcentaje se dispara por encima del 10%, ya sabes qué hacer: vuelve a la rejilla, añade valores de `C` más bajos y comprueba si la precisión sobre el conjunto de prueba se mantiene mientras el modelo se vuelve más sencillo.

Por último, ten en cuenta que `SVC` es, en su origen, un clasificador binario: separa solo dos clases a la vez. El dataset de cáncer de mama tiene exactamente dos clases (benigno y maligno), así que no lo notas. Pero si trabajas con tres o más categorías, `SVC` recurre por dentro a la estrategia uno-contra-uno (OvO, del inglés One versus One): entrena un clasificador binario por cada par de clases y combina sus votos. Puedes cambiarlo a uno-contra-el-resto (OvR, del inglés One versus Rest) con el parámetro `decision_function_shape='ovr'` si tu problema tiene muchas categorías y prefieres entrenar menos clasificadores.

## Resumen: lo que te llevas de este capítulo

- Un kernel transforma los datos a un espacio de más dimensiones donde sí existe una separación lineal, sin necesidad de construirlas de forma explícita; a esto se le llama el truco del kernel
- Los kernels más comunes en `SVC` son `linear`, `poly`, `rbf` (el que usa scikit-learn por defecto) y `sigmoid`, cada uno adecuado para distintas formas de frontera
- El margen blando permite que algunos puntos violen esa condición o queden mal clasificados, mediante variables de holgura, en lugar de exigir una separación perfecta como el margen duro del capítulo ocho
- El parámetro `C` controla ese equilibrio: cuanto más alto, más penaliza las violaciones y más se estrecha la calle de separación; cuanto más bajo, más las tolera y más ancha la deja
- CVXOPT es la biblioteca que resuelve el problema de programación cuadrática subyacente al entrenamiento de una SVM con margen blando y kernels; en Google Colab se instala aparte con `!pip install cvxopt`
- En `SVC`, ajustas kernel, `C` y `gamma` con un bucle simple o con `GridSearchCV`, comparando la precisión sobre datos de prueba para elegir la mejor combinación

## Ponlo en práctica

Retoma el dataset de dos anillos concéntricos de la sección sobre kernels.

**Genera** el dataset con `make_moons(n_samples=200, noise=0.15, random_state=42)` en lugar de `make_circles`. **Entrena** dos modelos `SVC` distintos sobre esos mismos datos: uno con `kernel='poly'` y otro con `kernel='rbf'`, dejando el resto de parámetros por defecto. **Calcula** la precisión de cada uno con `score()` y **dibuja** la frontera de decisión de ambos, reutilizando el código de la rejilla (`meshgrid`) de este capítulo. **Compara** las dos fronteras a simple vista: fíjate en si una se ajusta más a la forma de "medialuna" del dataset que la otra, y anota qué kernel te convence más para este caso concreto antes de seguir al siguiente tema del curso.

**Siglas usadas en este capítulo:** SVM (Support Vector Machine, máquina de vectores de soporte), RBF (Radial Basis Function, función de base radial), QP (Quadratic Programming, programación cuadrática), CVXOPT (Convex Optimization, optimización convexa)
