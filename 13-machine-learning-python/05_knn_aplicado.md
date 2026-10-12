# 5. Clasificación con K vecinos más cercanos (KNN)

## Por qué te importa esto

Hasta ahora has trabajado con regresión: predices un número, como el precio de una vivienda o la nota de un examen. Pero muchas preguntas del mundo real no piden un número, piden una categoría. ¿Este tumor es benigno o maligno? ¿Este correo es spam o no lo es? ¿Este cliente abandonará el servicio o seguirá con nosotros?

Ese tipo de pregunta pertenece a la clasificación, la otra gran familia de problemas de la inteligencia artificial (IA) aplicada. En este capítulo empiezas ese bloque con el algoritmo más intuitivo de todos: K vecinos más cercanos (KNN, del inglés K-Nearest Neighbors). Vas a entender su lógica, la matemática que lo sostiene y cómo aplicarlo en pocas líneas con scikit-learn sobre un caso real de diagnóstico médico.

En el sector salud, en el agroalimentario o en cualquier ámbito donde clasifiques muestras, productos o pacientes en categorías, este algoritmo es a menudo el primer modelo que se prueba, precisamente porque es fácil de explicar a quien no programa. Esa cualidad —poder justificar una decisión automática con una frase sencilla— tiene mucho valor cuando el proyecto necesita generar confianza en personas que no son técnicas.

## Qué es clasificar y cómo funciona KNN

Clasificar consiste en asignar una categoría a un dato nuevo a partir de las categorías que ya conoces. Nada más. Si separas la ropa en montones de "verano" e "invierno", estás clasificando. Si un banco decide si aprueba o rechaza un préstamo, también clasifica.

Imagina un plano con puntos de dos colores: unos marcados con más y otros con menos. Visualmente, distingues dos grupos. Ahora aparece un punto nuevo, sin marca, y quieres decidir a qué grupo pertenece. La intuición más natural es mirar qué hay alrededor de ese punto: si está rodeado de "más", probablemente sea un "más"; si está rodeado de "menos", probablemente sea un "menos".

Esa intuición es exactamente KNN. El nombre lo describe: mira los K vecinos más cercanos al punto nuevo y deja que voten. Si la mayoría de esos vecinos pertenece a una clase, el punto nuevo se clasifica en esa clase.

Piénsalo con el dicho popular: "dime con quién andas y te diré quién eres". KNN aplica esa misma lógica al pie de la letra: para saber qué es un dato nuevo, mira quiénes son sus vecinos más próximos y clasifícalo como la mayoría de ellos. Es también como mudarte a un vecindario nuevo: si la mayoría de las personas de tu calle trabaja en el sector tecnológico, alguien que no te conozca podría suponer, con bastante acierto, que tú también trabajas en ese sector.

Los pasos del algoritmo son estos:

- Guarda todos los datos ya etiquetados que tienes disponibles (el conjunto de entrenamiento)
- Cuando llega un dato nuevo, calcula su distancia a cada uno de esos datos conocidos
- Selecciona los K datos más cercanos, es decir, los K vecinos
- Cuenta cuántos vecinos pertenecen a cada clase
- Asigna al dato nuevo la clase con más votos

El parámetro K —el número de vecinos que consultas— lo eliges tú antes de entrenar el modelo. Un K pequeño, como uno, hace que el modelo reaccione mucho a cada dato individual: es flexible, pero también más sensible al ruido y a valores atípicos. Un K grande suaviza la decisión y la hace más estable, pero puede diluir patrones locales importantes. Además, conviene elegir un número impar de vecinos cuando trabajas con dos clases, para evitar empates en la votación: con K=4 podrías tener dos votos contra dos, y no sabrías qué decidir; con K=3 o K=5, siempre hay una clase mayoritaria.

Ten en cuenta también estas particularidades del algoritmo:

- KNN no "aprende" una fórmula durante el entrenamiento, simplemente memoriza los datos, por lo que se conoce como un algoritmo perezoso (lazy learning)
- Funciona muy bien con pocos datos y pocas dimensiones, pero se vuelve más lento a medida que el conjunto de datos crece, porque debe comparar el punto nuevo con todos los puntos guardados
- Es sensible a la escala de las características: si una columna mide en miles y otra en unidades, la que tiene números más grandes domina el cálculo de distancia, así que conviene normalizar los datos antes de entrenar

## La distancia euclídea: cómo mide KNN quién es "cercano"

Para decidir quién es vecino de quién, KNN necesita una forma de medir la distancia entre dos puntos. La métrica más habitual es la distancia euclídea, que lleva el nombre de Euclides, el matemático griego considerado el padre de la geometría. Es, sencillamente, la distancia en línea recta entre dos puntos, la misma que medirías con una regla sobre un papel.

En un plano de dos dimensiones, con un punto A de coordenadas (x1, y1) y un punto B de coordenadas (x2, y2), la fórmula es esta:

```
distancia(A, B) = raíz cuadrada de [ (x2 - x1)^2 + (y2 - y1)^2 ]
```

Es, en el fondo, el teorema de Pitágoras: restas las coordenadas, elevas al cuadrado cada diferencia, sumas los cuadrados y sacas la raíz cuadrada del resultado.

Veamos un ejemplo con números. Si el punto A está en (1, 2) y el punto B está en (4, 6):

```
distancia = raíz cuadrada de [ (4 - 1)^2 + (6 - 2)^2 ]
distancia = raíz cuadrada de [ 9 + 16 ]
distancia = raíz cuadrada de 25
distancia = 5
```

La misma idea se generaliza a cualquier número de dimensiones. Si tus datos tienen n características en lugar de dos, la fórmula simplemente suma más términos, uno por cada característica:

```
distancia = raíz cuadrada de [ (a1 - b1)^2 + (a2 - b2)^2 + ... + (an - bn)^2 ]
```

No necesitas dibujar un plano de treinta dimensiones para entenderlo: la lógica es la misma que en dos dimensiones, solo que comparas más coordenadas a la vez. Esto es justo lo que ocurre en el dataset médico que vas a usar en este capítulo: cada tumor se describe con varias características numéricas, y KNN calcula la distancia euclídea considerando todas ellas a la vez para decidir qué tumores conocidos son "vecinos" del tumor nuevo.

Así se vería en Python, solo como referencia matemática, sin usarlo todavía dentro de un clasificador:

```python
from math import sqrt

punto_a = (1, 2)
punto_b = (4, 6)

distancia = sqrt((punto_a[0] - punto_b[0]) ** 2 + (punto_a[1] - punto_b[1]) ** 2)

print("Distancia euclídea entre A y B:", distancia)
# Distancia euclídea entre A y B: 5.0
```

**Guarda este cálculo en mente**, porque en el próximo capítulo programarás tú mismo, desde cero, tanto la distancia euclídea como el algoritmo KNN completo, sin depender de scikit-learn. Por ahora, céntrate en usar la versión ya construida para entender qué resultados produce y cómo se comporta.

## Aplicar KNN con scikit-learn

Vas a aplicar KNN sobre un caso real: clasificar tumores de mama como benignos o malignos a partir de características numéricas medidas en el tejido, como el radio, la textura o la simetría de las células. La fuente original de este ejercicio descarga un archivo de datos desde el repositorio de la Universidad de California en Irvine (UCI). Aquí usamos el dataset ya incluido en scikit-learn en vez de descargarlo, para no depender de conexión a internet: se trata del mismo tipo de datos de diagnóstico de cáncer de mama, listo para usar con la función `load_breast_cancer()`.

### Paso uno: cargar y explorar los datos

```python
from sklearn.datasets import load_breast_cancer

datos = load_breast_cancer()

# datos.data contiene las características (mediciones del tumor)
# datos.target contiene la etiqueta: 0 (maligno) o 1 (benigno)
print("Forma de las características:", datos.data.shape)
print("Nombres de las clases:", datos.target_names)
print("Primera fila de características:", datos.data[0])
print("Etiqueta de la primera fila:", datos.target[0])
```

Este dataset trae 569 muestras y 30 características por muestra, ya limpias y sin valores ausentes, lo que te permite centrarte en el algoritmo sin perder tiempo en depurar datos.

### Paso dos: separar entrenamiento y prueba

Igual que en la regresión lineal, necesitas dividir los datos en un conjunto de entrenamiento, que el modelo usa para aprender, y un conjunto de prueba, que reservas para comprobar si el modelo generaliza bien ante datos que no ha visto:

```python
from sklearn.model_selection import train_test_split

X = datos.data
y = datos.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Muestras de entrenamiento:", len(X_train))
print("Muestras de prueba:", len(X_test))
```

Con `test_size=0.2` reservas el 20% de las muestras para la prueba y el 80% restante para el entrenamiento. El parámetro `random_state` fija la aleatoriedad de la división, de modo que, si vuelves a ejecutar el código, obtienes siempre el mismo reparto.

### Paso tres: crear, entrenar y evaluar el clasificador

Ahora entra en juego `KNeighborsClassifier`, la clase de scikit-learn que implementa KNN:

```python
from sklearn.neighbors import KNeighborsClassifier

# Creas el clasificador indicando cuántos vecinos consultará (K = 5)
clasificador = KNeighborsClassifier(n_neighbors=5)

# Lo entrenas con el método fit(), pasándole características y etiquetas
clasificador.fit(X_train, y_train)

# Calculas la precisión del modelo sobre los datos de prueba
precision = clasificador.score(X_test, y_test)
print("Precisión del modelo:", precision)
```

El método `fit()` guarda internamente los datos de entrenamiento, ya que KNN no calcula ninguna fórmula compleja durante el entrenamiento, solo memoriza. El método `score()` compara las predicciones del modelo sobre `X_test` con las etiquetas reales `y_test` y devuelve el porcentaje de aciertos. Con este dataset y K=5, es habitual obtener una precisión superior al 90%, un resultado que demuestra que un algoritmo tan sencillo puede resultar sorprendentemente eficaz en un caso médico real.

### Paso cuatro: predecir un caso nuevo

Si quieres predecir la clase de una muestra concreta, usa el método `predict()`:

```python
# Tomamos una muestra cualquiera del conjunto de prueba como ejemplo
muestra_nueva = X_test[0].reshape(1, -1)

prediccion = clasificador.predict(muestra_nueva)
clase_predicha = datos.target_names[prediccion[0]]

print("Clase predicha:", clase_predicha)
print("Clase real:", datos.target_names[y_test[0]])
```

Aquí usas `reshape(1, -1)` porque scikit-learn espera siempre una matriz de dos dimensiones como entrada, donde cada fila es una muestra y cada columna una característica, aunque solo quieras predecir una muestra a la vez.

### Cómo elegir el número de vecinos K

No existe un valor de K universalmente correcto: depende del dataset y del problema. Como referencia general:

- Un K pequeño hace el modelo más flexible, pero más sensible al ruido, con riesgo de sobreajuste
- Un K grande suaviza la decisión y reduce el ruido, pero puede perder matices, con riesgo de infraajuste
- Con dos clases, prefiere valores impares de K para evitar empates en la votación
- Con datasets pequeños, valores de K bajos (por ejemplo, tres o cinco) suelen funcionar bien
- La forma más fiable de elegir K es probar varios valores y comparar la precisión resultante sobre el conjunto de prueba, tal como harás en la sección de práctica

## Resumen: lo que te llevas de este capítulo

- Clasificar significa asignar una categoría a un dato nuevo a partir de datos ya etiquetados, a diferencia de la regresión, que predice un número
- KNN clasifica un dato nuevo consultando a sus K vecinos más cercanos y asignándole la clase mayoritaria entre ellos, igual que un vecindario influye en cómo te perciben desde fuera
- La distancia euclídea mide qué tan cerca están dos puntos, calculando la raíz cuadrada de la suma de las diferencias al cuadrado entre sus coordenadas, y se generaliza sin problema a decenas de características
- Con scikit-learn, aplicas KNN en pocas líneas: `KNeighborsClassifier`, `fit()` para entrenar, `predict()` para predecir y `score()` para medir la precisión
- El valor de K es una decisión tuya: valores impares evitan empates, y probar varios valores te ayuda a encontrar el que mejor equilibra flexibilidad y estabilidad
- En el próximo capítulo programarás KNN y la distancia euclídea desde cero, sin scikit-learn, para entender exactamente qué ocurre "por dentro"

## Ponlo en práctica

Retoma el código del clasificador sobre `load_breast_cancer()` y comprueba con tus propios ojos cómo afecta K al resultado.

**Escribe un bucle** que entrene un `KNeighborsClassifier` distinto para cada valor de K entre uno y quince, y que imprima la precisión obtenida en cada caso:

```python
for k in range(1, 16):
    clasificador_k = KNeighborsClassifier(n_neighbors=k)
    clasificador_k.fit(X_train, y_train)
    precision_k = clasificador_k.score(X_test, y_test)
    print(f"K = {k} -> precisión = {precision_k:.4f}")
```

**Observa** en qué valor de K se estabiliza o empeora la precisión, y **anota** tus conclusiones antes de continuar: ¿hay algún K que destaque claramente sobre los demás? ¿Qué ocurre con K=1 frente a valores más altos? Guarda tus respuestas, porque te servirán de referencia cuando en el próximo capítulo compares tu propia implementación de KNN con la de scikit-learn.

---

**Siglas usadas en este capítulo:** IA (Inteligencia Artificial), KNN (K vecinos más cercanos, del inglés K-Nearest Neighbors), UCI (Universidad de California en Irvine).
