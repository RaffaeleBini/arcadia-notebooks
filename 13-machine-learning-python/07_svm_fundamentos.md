# 7. Fundamentos de las máquinas de vectores de soporte (SVM)

## Por qué te importa esto

En los capítulos cinco y seis descubriste la clasificación con K vecinos más cercanos (KNN, del inglés K-Nearest Neighbors): un algoritmo intuitivo que decide la categoría de un dato nuevo mirando a sus vecinos. Funciona bien, es fácil de explicar y, precisamente por eso, suele ser el primer modelo que se prueba en un proyecto de clasificación.

Pero KNN tiene límites. Necesita guardar todos los datos de entrenamiento, se vuelve lento con conjuntos grandes y es sensible al ruido cuando el valor de K no encaja bien con el problema. En este capítulo empiezas un bloque nuevo dedicado a otra familia de algoritmos de clasificación: las máquinas de vectores de soporte (SVM, del inglés Support Vector Machine).

Una SVM ataca el mismo problema —separar datos en categorías— desde un ángulo distinto: en lugar de votar entre vecinos, busca la frontera que mejor separa las clases y la mantiene lo más alejada posible de los datos de cada lado. Esa idea, aparentemente simple, la convierte en una de las herramientas más robustas y precisas de la inteligencia artificial (IA) aplicada, con aplicaciones en diagnóstico médico, reconocimiento de imágenes, bioinformática y clasificación de texto.

Este capítulo se centra en la intuición geométrica de la SVM: qué es un vector, qué es un hiperplano separador y por qué maximizar el margen entre clases es la clave del algoritmo. También la aplicarás con scikit-learn sobre el mismo dataset médico del capítulo cinco, para que compares directamente sus resultados con los de KNN. En el capítulo ocho programarás una SVM desde cero, y en el capítulo nueve profundizarás en los kernels y el margen blando con la biblioteca de optimización CVXOPT. Aquí, guarda la calculadora de fórmulas complejas en el cajón: toca entender el dibujo antes que la ecuación completa.

Este algoritmo se usa hoy en campos tan distintos como el reconocimiento de imágenes, la clasificación de textos, la bioinformática y las finanzas. Le sienta especialmente bien a los problemas con muchas características por muestra —como los treinta valores numéricos del dataset médico que retomarás en este capítulo—, precisamente el terreno donde otros algoritmos empiezan a perder precisión o velocidad.

## Qué es un vector

Antes de hablar de SVM, necesitas dominar su ingrediente básico: el vector. Un vector es una magnitud que tiene, a la vez, longitud y dirección. Se suele representar como una flecha: el punto donde nace indica el origen, y la punta indica hacia dónde apunta y cuánto se extiende.

Piensa en una instrucción de senderismo: "camina tres pasos al norte". Esa frase describe un vector completo. La dirección es "norte" y la longitud es "tres pasos". Si en cambio caminas cinco pasos al este, obtienes otro vector distinto, con otra dirección y otra longitud. Dos vectores solo son iguales cuando coinciden en ambas cosas a la vez.

En matemáticas, un vector se representa como una lista de números, uno por cada dimensión del espacio en el que trabajas. En un plano de dos dimensiones, un vector `a` con valores 3 y 4 se escribe como (3, 4): avanza tres unidades en un eje y cuatro en el otro. En una SVM, cada dato de tu conjunto —cada tumor, cada correo, cada imagen— se convierte en un vector: cada característica medida ocupa una dimensión, y el conjunto de esos valores define un punto en un espacio con tantas dimensiones como características tengas.

No todos los vectores representan datos. Más adelante conocerás el vector normal, que no describe ninguna muestra concreta, sino la orientación del propio hiperplano separador: apunta en la dirección perpendicular a la frontera, como una brújula clavada en mitad de la calle que marca hacia dónde mira. Con ese matiz en mente, ya puedes distinguir los dos papeles que cumplen los vectores dentro de una SVM: unos representan datos, y otro define la frontera que los separa.

### La magnitud de un vector

La magnitud de un vector es su longitud, y se calcula igual que la distancia euclídea que ya usaste con KNN: elevas al cuadrado cada componente, sumas los cuadrados y sacas la raíz cuadrada del resultado. Para el vector (3, 4), la magnitud es la raíz cuadrada de (3² + 4²), es decir, la raíz cuadrada de 25, que da 5.

```python
from math import sqrt

vector_a = (3, 4)

magnitud = sqrt(vector_a[0] ** 2 + vector_a[1] ** 2)

print("Magnitud del vector a:", magnitud)
# Magnitud del vector a: 5.0
```

Fíjate en que este cálculo es una aplicación directa del teorema de Pitágoras: el vector actúa como la hipotenusa de un triángulo rectángulo cuyos catetos son sus componentes.

### El producto punto entre dos vectores

Además de tener magnitud propia, dos vectores pueden combinarse mediante el producto punto (también llamado producto escalar): multiplicas cada componente de un vector por la componente correspondiente del otro y sumas los resultados. Si el vector `a` vale (1, 3) y el vector `b` vale (2, 4), el producto punto es (1×2) + (3×4), que da 10.

```python
vector_a = (1, 3)
vector_b = (2, 4)

producto_punto = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]

print("Producto punto entre a y b:", producto_punto)
# Producto punto entre a y b: 10
```

**Guarda esta operación en mente**, porque el producto punto es la pieza que permite a una SVM decidir en qué lado de la frontera cae un dato nuevo, como verás en la siguiente sección.

## La idea de SVM: separar clases con la calle más ancha posible

Imagina un mapa con dos barrios que quieres separar mediante una calle recta. Podrías trazar esa calle de muchas formas distintas, siempre que quede entre ambos barrios. Pero hay una opción mejor que las demás: la calle más ancha posible, la que deja el mayor espacio libre a ambos lados antes de tocar la primera vivienda de cada barrio.

Esa es exactamente la lógica de una SVM. La línea central de la calle es el hiperplano separador: la frontera que divide el espacio en dos regiones, una por clase. En un plano de dos dimensiones, el hiperplano es una línea recta; en tres dimensiones, es un plano; y en dimensiones superiores —como las treinta características del dataset médico que usarás en este capítulo—, sigue llamándose hiperplano aunque ya no puedas dibujarlo.

Las viviendas más próximas a los bordes de la calle, una a cada lado, son las que definen su anchura. En una SVM, esos datos reciben el nombre de vectores de soporte: son los puntos de entrenamiento más cercanos a la frontera, y son los únicos que realmente importan para calcular dónde va el hiperplano. El resto de las viviendas, más alejadas hacia el interior de cada barrio, podrían moverse sin afectar el trazado de la calle.

El ancho completo de la calle, medido de acera a acera, es el margen. El objetivo de una SVM es encontrar el hiperplano que maximiza ese margen. Cuanto más ancha sea la calle, más espacio de seguridad tienes antes de equivocarte al clasificar un dato nuevo, y mejor generaliza el modelo ante datos que no ha visto durante el entrenamiento. Un margen estrecho, en cambio, deja el modelo pegado a los datos de entrenamiento y lo hace más frágil ante el ruido.

### Por qué te conviene un margen amplio

Piensa de nuevo en la calle entre los dos barrios. Si la trazas muy pegada a las viviendas de un lado, cualquier vivienda nueva que se construya un poco más adentro del otro barrio quedará mal clasificada, porque la frontera no dejó margen de maniobra. Si en cambio la calle queda bien centrada y ancha, tienes espacio de sobra para admitir pequeñas variaciones sin que la frontera falle.

Con datos reales pasa lo mismo: un margen amplio tolera mejor el ruido y los valores atípicos, porque la frontera no depende de un puñado de casos extremos. Esa tolerancia es la razón por la que una SVM entrenada con el criterio del margen máximo suele generalizar mejor ante datos nuevos que una frontera trazada sin ese criterio, ya sea a mano o mediante un algoritmo distinto.

### Ventajas frente a otros clasificadores

Antes de pasar a la parte de optimización, conviene que sepas por qué esta familia de algoritmos goza de tan buena reputación. Entre sus puntos fuertes destacan estos:

- Rinde bien en espacios con muchas características, incluso cuando el número de características supera al número de muestras
- El propio criterio del margen actúa como una forma de control contra el sobreajuste, ya que no se deja arrastrar por el ruido de los datos de entrenamiento
- Solo necesita guardar los vectores de soporte para clasificar datos nuevos, así que es más eficiente en memoria que un algoritmo que deba recordar todo el conjunto de entrenamiento, como hace KNN

Una vez trazada la calle, clasificar un dato nuevo es sencillo: solo necesitas mirar en qué acera cae. Matemáticamente, el hiperplano se describe con la ecuación:

```
w · x + b = 0
```

Aquí, `x` es el vector que representa el dato, `w` es el vector normal al hiperplano (perpendicular a la calle, marcando su orientación) y `b` es el término de sesgo, que desplaza la calle hacia un lado u otro sin cambiar su ángulo. Para clasificar un punto nuevo, calculas el producto punto entre `w` y `x`, le sumas `b`, y observas el signo del resultado: si es positivo o cero, el punto cae en un lado de la calle; si es negativo, cae en el otro.

Veámoslo con un ejemplo sencillo e inventado, solo para fijar la idea, con un vector `w` igual a (1, 1) y un sesgo `b` igual a -4:

```python
def clasificar(punto, w, b):
    producto = punto[0] * w[0] + punto[1] * w[1] + b
    if producto >= 0:
        return "positiva", producto
    return "negativa", producto

w = (1, 1)
b = -4

for punto in [(3, 3), (1, 1)]:
    clase, valor = clasificar(punto, w, b)
    print(f"Punto {punto} -> valor {valor} -> clase {clase}")

# Punto (3, 3) -> valor 2 -> clase positiva
# Punto (1, 1) -> valor -2 -> clase negativa
```

El punto (3, 3) da un resultado positivo y cae en un lado de la frontera; el punto (1, 1) da un resultado negativo y cae en el otro. Así funciona, en esencia, la predicción de cualquier SVM una vez entrenada: el trabajo pesado ocurre antes, cuando el algoritmo encuentra los valores óptimos de `w` y `b`.

## Por qué encontrar ese hiperplano es un problema de optimización

Trazar la calle más ancha suena sencillo cuando la dibujas a mano sobre un mapa con pocos puntos, pero encontrarla de forma exacta, con decenas de características y miles de datos, exige matemáticas. Una SVM convierte esa búsqueda en un problema de optimización: encontrar los valores de `w` y `b` que maximizan el margen sin dejar ningún dato de entrenamiento mal clasificado.

Aquí aparece un resultado útil: el ancho del margen es inversamente proporcional a la magnitud del vector `w`. Cuanto menor sea esa magnitud, más ancha es la calle. Maximizar el margen equivale, entonces, a minimizar la magnitud de `w`, siempre respetando la condición de que cada dato de entrenamiento quede correctamente clasificado y fuera del margen.

Este tipo de problema —minimizar una función sujeta a restricciones— se conoce como programación cuadrática, y pertenece a una categoría especial llamada optimización convexa. La palabra "convexa" describe la forma del terreno de búsqueda: imagina un cuenco. Si sueltas una bola en cualquier punto de su superficie, siempre rueda hacia el mismo fondo, sin quedarse atascada en huecos intermedios. Eso es justo lo que ocurre con el problema de la SVM: por su naturaleza convexa, existe un único mínimo global, así que cualquier algoritmo que "ruede cuenco abajo" con suficiente cuidado termina encontrando la mejor solución posible, sin el riesgo de quedarse atrapado en una solución mediocre.

Resolver este problema con precisión implica herramientas como los multiplicadores de Lagrange y técnicas de optimización cuadrática, que quedan fuera del alcance de este capítulo. En el capítulo ocho programarás tú mismo el algoritmo de entrenamiento paso a paso, y en el capítulo nueve usarás la biblioteca CVXOPT para resolver el problema de forma eficiente, además de introducir el margen blando —que permite algunas clasificaciones erróneas mediante un parámetro llamado `C`— y los kernels, que permiten separar datos que no son separables con una línea recta. Por ahora, quédate con la idea central: entrenar una SVM consiste en resolver un problema de optimización convexa que garantiza encontrar la mejor calle posible, no una calle cualquiera.

En la práctica, calcular la solución exacta con miles de datos sería demasiado lento si se resolviera de forma directa, así que scikit-learn recurre por dentro a algoritmos especializados, como la optimización mínima secuencial, que dividen el problema grande en muchos problemas pequeños y los resuelven uno a uno hasta converger. Tú no necesitas programar esa parte para usar `SVC`: la biblioteca ya la incorpora, y solo la reconstruirás con fines didácticos en el capítulo ocho.

## Aplicar SVM con scikit-learn

Aunque la matemática completa espera al capítulo nueve, ya puedes entrenar y evaluar una SVM hoy mismo, gracias a la clase `SVC` (Support Vector Classification) de scikit-learn. Vas a repetir el mismo experimento del capítulo cinco —clasificar tumores como benignos o malignos— para comparar directamente el resultado de SVM con el que obtuviste entonces con KNN.

### Paso uno: cargar los datos y dividirlos

Reutilizas el dataset médico ya incluido en scikit-learn, con la misma división entre entrenamiento y prueba que usaste con KNN, para que la comparación sea justa:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

datos = load_breast_cancer()
X = datos.data
y = datos.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Muestras de entrenamiento:", len(X_train))
print("Muestras de prueba:", len(X_test))
```

Usar el mismo `random_state` que en el capítulo cinco asegura que ambos algoritmos ven exactamente la misma partición de datos, así que cualquier diferencia en la precisión final proviene del algoritmo, no del reparto aleatorio.

### Paso dos: crear, entrenar y evaluar la SVM

La clase `SVC` funciona con la misma lógica de tres pasos que ya conoces: creas el clasificador, lo entrenas con `fit()` y lo evalúas con `score()`. Empiezas con un kernel lineal, la opción más simple, que busca directamente una calle recta sin transformar el espacio de características:

```python
from sklearn.svm import SVC

# Creas la SVM con un kernel lineal (la calle más simple: una línea recta)
clasificador_svm = SVC(kernel="linear")

# Entrenas el modelo: aquí ocurre la optimización convexa que buscaste
clasificador_svm.fit(X_train, y_train)

# Evalúas la precisión sobre los datos de prueba
precision_svm = clasificador_svm.score(X_test, y_test)
print("Precisión de la SVM:", precision_svm)
```

Ejecuta este código y compara el número que obtienes con la precisión que anotaste en el capítulo cinco para `KNeighborsClassifier` sobre el mismo dataset. En muchos problemas tabulares como este, la SVM iguala o incluso supera el resultado de KNN, aunque el valor exacto depende de la partición de datos, del kernel elegido y de otros parámetros que explorarás en el capítulo nueve. Lo importante ahora no es memorizar una cifra, sino comprobar con tus propios datos que dos algoritmos distintos, entrenados sobre la misma información, pueden llegar a decisiones parecidas por caminos completamente diferentes.

Ten presente, además, que la SVM comparte una particularidad con KNN: ambos algoritmos calculan distancias y productos punto entre vectores, así que ambos son sensibles a la escala de las características. Si una columna se mide en miles y otra en unidades, la de números más grandes domina el resultado. Por simplicidad, este ejemplo entrena directamente sobre los datos originales, pero en un proyecto real conviene aplicar el mismo tipo de escalado que ya conoces, por ejemplo con `StandardScaler`, antes de entrenar la SVM.

### Paso tres: predecir un caso nuevo

Igual que con KNN, `predict()` te permite clasificar una muestra concreta:

```python
muestra_nueva = X_test[0].reshape(1, -1)

prediccion = clasificador_svm.predict(muestra_nueva)
clase_predicha = datos.target_names[prediccion[0]]

print("Clase predicha por la SVM:", clase_predicha)
print("Clase real:", datos.target_names[y_test[0]])
```

Puedes reutilizar `X_test[0]`, la misma muestra que probaste en el capítulo cinco, y comparar directamente si SVM y KNN coinciden en su predicción para ese caso concreto.

Si los dos modelos coinciden, ganas confianza en el diagnóstico; si difieren, tienes una señal clara de que ese caso concreto merece una revisión adicional antes de tomar cualquier decisión automática, algo especialmente relevante cuando el resultado alimenta un proceso clínico real.

## Resumen: lo que te llevas de este capítulo

- Un vector es una magnitud con dirección y longitud, como la instrucción "camina tres pasos al norte"; en SVM, cada dato de entrenamiento se representa como un vector en un espacio de características
- La magnitud de un vector se calcula con la misma fórmula que la distancia euclídea, y el producto punto combina dos vectores en un único número
- Una SVM busca el hiperplano —la "calle"— que separa las clases con el margen más ancho posible; los datos más cercanos a los bordes de esa calle son los vectores de soporte
- Encontrar ese hiperplano óptimo es un problema de optimización convexa: minimizar la magnitud del vector `w` sujeto a clasificar bien todos los datos, un problema con un único mínimo global
- Con scikit-learn, entrenas una SVM en pocas líneas: `SVC`, `fit()` para entrenar, `predict()` para predecir y `score()` para medir la precisión, exactamente igual que hiciste con `KNeighborsClassifier`
- La derivación matemática completa del entrenamiento —multiplicadores de Lagrange, margen blando y kernels— llega en los capítulos ocho y nueve

## Ponlo en práctica

Retoma el código de la SVM sobre `load_breast_cancer()` y observa cómo cambia la precisión al modificar el parámetro `C` de `SVC`. Todavía no vas a estudiar en profundidad qué representa `C` matemáticamente —eso corresponde al capítulo nueve, cuando trabajes el margen blando—, así que por ahora limítate a observar el efecto sobre el resultado.

**Escribe un bucle** que entrene una SVM distinta para varios valores de `C` y que imprima la precisión obtenida en cada caso:

```python
valores_de_c = [0.01, 0.1, 1, 10, 100]

for c in valores_de_c:
    clasificador_c = SVC(kernel="linear", C=c)
    clasificador_c.fit(X_train, y_train)
    precision_c = clasificador_c.score(X_test, y_test)
    print(f"C = {c} -> precisión = {precision_c:.4f}")
```

**Observa** si la precisión cambia mucho entre los valores extremos de `C` o si se mantiene estable, y **anota** tus resultados junto con los que ya guardaste para KNN en el capítulo cinco. **Guarda** también tus conclusiones sobre el kernel lineal, porque en el capítulo nueve las usarás como punto de partida para entender qué aporta exactamente el parámetro `C` y qué cambia cuando introduces un kernel no lineal.

Antes de cerrar el capítulo, **responde** por escrito a estas dos preguntas, sin buscar todavía la explicación formal: ¿un valor de `C` muy pequeño ensancha o estrecha la calle que separa las clases? ¿Y qué relación intuitiva encuentras entre ese comportamiento y la idea del margen amplio que trabajaste en la sección sobre el hiperplano? No necesitas acertar con precisión matemática; basta con que anotes tu intuición actual, porque volverás sobre estas mismas preguntas al llegar al margen blando en el capítulo nueve.

---

**Siglas usadas en este capítulo:** IA (Inteligencia Artificial), KNN (K vecinos más cercanos, del inglés K-Nearest Neighbors), SVM (Máquinas de Vectores de Soporte, del inglés Support Vector Machine).
