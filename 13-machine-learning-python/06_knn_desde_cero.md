# 6. Programar KNN desde cero

## Por qué te importa esto

En el capítulo anterior usaste `KNeighborsClassifier` de scikit-learn para clasificar tumores del conjunto de datos de cáncer de mama. Con pocas líneas obtuviste un modelo funcional, pero esa comodidad esconde el mecanismo real. Ahora vas a abrir la caja negra.

En este capítulo programas tu propio clasificador de los k vecinos más cercanos (KNN, por sus siglas en inglés *K-Nearest Neighbors*) usando solo Python y NumPy, sin scikit-learn. El objetivo no es sustituir la librería, que está mucho más optimizada, sino entender exactamente qué ocurre entre que le pasas un punto nuevo y recibes una predicción. Esa comprensión te ayuda a depurar modelos, elegir mejor los hiperparámetros y reconocer cuándo un algoritmo no es la herramienta adecuada para tu problema.

Al final, comparas tu implementación con la de scikit-learn sobre el mismo conjunto de datos y comprobarás que ambas llegan a resultados muy parecidos.

## El algoritmo paso a paso

Piensa en cómo eliges un restaurante en una ciudad que no conoces. No preguntas a todos los habitantes qué les parece cada local: preguntas a las personas que tienes más cerca, quizás las cinco o diez que viven en tu mismo barrio, y te fías de lo que opina la mayoría de ese grupo reducido. Eso es exactamente lo que hace KNN con los datos.

El proceso se reduce a tres acciones concretas:

1. Calcular la distancia entre el punto nuevo y cada uno de los puntos del conjunto de entrenamiento
2. Ordenar esas distancias de menor a mayor y quedarte con las k más pequeñas, es decir, con los k vecinos más próximos
3. Votar: la clase que más se repite entre esos k vecinos se convierte en la predicción para el punto nuevo

No hay una fase de entrenamiento que ajuste parámetros internos como ocurre en otros algoritmos. KNN guarda el conjunto de entrenamiento tal cual y hace todo el trabajo en el momento de predecir. Por eso se le llama un aprendiz perezoso (*lazy learner*): no aprende un modelo compacto, simplemente compara.

## Programar tu propio clasificador KNN en Python

Empieza por la pieza más pequeña: una función que calcule la distancia euclídea entre dos puntos. NumPy te permite hacerlo en una sola línea gracias a `numpy.linalg.norm`, que calcula la longitud del vector diferencia entre ambos puntos.

Con esa pieza ya puedes construir la función principal, `knn_predecir`, que recibe el conjunto de entrenamiento, el punto nuevo y el valor de k:

```python
import numpy as np
from collections import Counter


def distancia_euclidea(punto_a, punto_b):
    """Distancia en línea recta entre dos puntos."""
    punto_a = np.array(punto_a)
    punto_b = np.array(punto_b)
    return np.linalg.norm(punto_a - punto_b)


def knn_predecir(datos_entrenamiento, punto_nuevo, k):
    """
    datos_entrenamiento: lista de tuplas (caracteristicas, etiqueta)
    punto_nuevo: array con las caracteristicas del punto a clasificar
    k: numero de vecinos que participan en la votacion
    """
    if len(datos_entrenamiento) < k:
        raise ValueError("k no puede superar el número de puntos de entrenamiento")

    distancias = []
    for caracteristicas, etiqueta in datos_entrenamiento:
        distancia = distancia_euclidea(caracteristicas, punto_nuevo)
        distancias.append((distancia, etiqueta))

    # Paso 2: ordenar y quedarte con los k vecinos más cercanos
    distancias.sort(key=lambda par: par[0])
    k_vecinos = distancias[:k]

    # Paso 3: votar la clase mayoritaria
    votos = [etiqueta for _, etiqueta in k_vecinos]
    clase_ganadora = Counter(votos).most_common(1)[0][0]
    return clase_ganadora
```

Fíjate en tres detalles del código:

- `datos_entrenamiento` es una lista de tuplas que empareja cada vector de características con su etiqueta, así la función conoce a la vez dónde está cada punto y a qué clase pertenece
- La comprobación inicial evita un error habitual: pedir más vecinos de los que existen en el conjunto de entrenamiento
- `Counter.most_common(1)` te devuelve, en una sola llamada, la etiqueta que aparece con más frecuencia entre los votos

Con esto ya tienes un clasificador KNN completo. Es más lento que el de scikit-learn porque recorre la lista con un bucle explícito en Python en lugar de usar operaciones vectorizadas internas en C, pero hace exactamente lo mismo a nivel conceptual.

## Comparar tu implementación con la de scikit-learn

Toca la prueba de fuego: aplicar `knn_predecir` sobre el mismo conjunto de datos de cáncer de mama que usaste en el capítulo cinco y comprobar si el porcentaje de aciertos se acerca al de `KNeighborsClassifier`.

Como KNN se basa en distancias, y algunas características de este conjunto de datos tienen escalas muy distintas entre sí, escalas primero los datos con `StandardScaler`. Así ninguna característica domina el cálculo de distancia solo por tener valores más grandes.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Cargar el mismo dataset que en el capítulo anterior
datos = load_breast_cancer()
X, y = datos.data, datos.target

# Escalar las características para que todas pesen lo mismo en la distancia
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_escalado, y, test_size=0.2, random_state=42
)

# Preparar los datos en el formato que espera tu función
datos_entrenamiento = list(zip(X_train, y_train))

k = 5

# Predicciones con tu propio KNN
predicciones_propias = [
    knn_predecir(datos_entrenamiento, punto, k) for punto in X_test
]
precision_propia = accuracy_score(y_test, predicciones_propias)

# Predicciones con scikit-learn
modelo_sklearn = KNeighborsClassifier(n_neighbors=k)
modelo_sklearn.fit(X_train, y_train)
predicciones_sklearn = modelo_sklearn.predict(X_test)
precision_sklearn = accuracy_score(y_test, predicciones_sklearn)

print(f"Precisión de tu KNN:        {precision_propia:.3f}")
print(f"Precisión de scikit-learn:  {precision_sklearn:.3f}")
```

Si ejecutas este código, verás que ambas precisiones rondan el 95-97 %, con diferencias mínimas entre una y otra. Eso confirma algo importante: tu implementación no es una simplificación aproximada, es el mismo algoritmo, solo que sin las optimizaciones internas de scikit-learn. Las pequeñas diferencias que puedas observar suelen deberse a cómo cada implementación resuelve los empates en la votación, no a un error de concepto.

**Ejecuta este bloque con distintos valores de k, como 3, 7 o 11, y observa cómo cambia la precisión de ambas versiones.** Notarás que ni tu función ni la de scikit-learn mejoran de forma constante al aumentar k: a partir de cierto punto, un k demasiado grande empieza a diluir la información local y la precisión se estanca o incluso baja.

## Ventajas y límites de KNN

Programar el algoritmo desde cero deja mucho más claros sus puntos fuertes y débiles que usarlo como caja cerrada.

**Coste computacional.** Cada predicción obliga a calcular la distancia entre el punto nuevo y todos los puntos del conjunto de entrenamiento. Si tienes cien puntos, son cien cálculos de distancia; si tienes un millón, son un millón, y eso solo para clasificar un único punto nuevo. Como no existe una fase de entrenamiento que resuma la información en un modelo compacto, KNN no escala bien cuando el volumen de datos crece mucho, y cada consulta se vuelve más lenta a medida que añades más ejemplos.

**Sensibilidad a la escala de las características.** El algoritmo asume que todas las dimensiones aportan por igual a la distancia. Si una característica se mide en miles y otra en decimales, la primera domina el cálculo aunque no sea más relevante para el problema. Por eso escalaste los datos antes de comparar tu función con scikit-learn: sin ese paso, la comparación habría sido injusta y los resultados, peores de lo esperado.

**Cuándo evitar KNN.** Conviene descartarlo, o al menos usarlo con cautela, en estos casos:

- Conjuntos de datos muy grandes donde necesitas predicciones rápidas y constantes en producción
- Espacios con muchas dimensiones, donde el concepto de "cercanía" pierde significado porque casi todos los puntos acaban a distancias parecidas entre sí
- Clases muy desequilibradas, donde una clase mayoritaria puede dominar la votación aunque el punto nuevo se parezca más a la clase minoritaria
- Datos con muchas características irrelevantes, que añaden ruido a la distancia sin aportar información útil

A cambio, KNN sigue siendo una opción razonable cuando el conjunto de datos es pequeño o mediano, las clases están razonablemente equilibradas y quieres un modelo fácil de explicar a alguien sin formación técnica.

## Resumen: lo que te llevas de este capítulo

- Programar KNN desde cero se reduce a tres pasos: calcular distancias, ordenarlas y votar la clase mayoritaria entre los k vecinos más próximos
- La función `knn_predecir` necesita el conjunto de entrenamiento con sus etiquetas, el punto nuevo y el valor de k
- NumPy simplifica el cálculo de la distancia euclídea con `numpy.linalg.norm`, evitando bucles manuales sobre cada dimensión
- Tu implementación y la de scikit-learn logran precisiones muy similares sobre el mismo conjunto de datos, lo que confirma que ambas aplican el mismo principio
- KNN es un aprendiz perezoso: no construye un modelo compacto, compara el punto nuevo contra todo el conjunto de entrenamiento en cada predicción
- El coste computacional y la sensibilidad a la escala de las características son las dos limitaciones que más debes vigilar

## Ponlo en práctica

**Comprueba con tus propios datos cuánto se ralentiza tu función frente a la de scikit-learn a medida que crece el conjunto de entrenamiento.** Usa el módulo `time` para medir el tiempo de ambas versiones con distintos tamaños de datos:

```python
import time

tamanos = [50, 100, 200, 400]

for tamano in tamanos:
    subconjunto = datos_entrenamiento[:tamano]

    inicio = time.perf_counter()
    for punto in X_test:
        knn_predecir(subconjunto, punto, k)
    tiempo_propio = time.perf_counter() - inicio

    modelo_sklearn.fit(X_train[:tamano], y_train[:tamano])
    inicio = time.perf_counter()
    modelo_sklearn.predict(X_test)
    tiempo_sklearn = time.perf_counter() - inicio

    print(
        f"Tamaño {tamano}: tu función {tiempo_propio:.4f} s, "
        f"scikit-learn {tiempo_sklearn:.4f} s"
    )
```

**Anota los resultados en una tabla y compara la curva de crecimiento de cada versión.** Verás que el tiempo de tu función crece de forma mucho más pronunciada que el de scikit-learn a medida que aumenta el tamaño del conjunto de entrenamiento. Esa diferencia es la prueba práctica de todo lo que has leído sobre el coste computacional de KNN: la librería usa estructuras de datos y optimizaciones internas que tu versión, escrita para aprender el algoritmo, no incluye.

---

**Siglas usadas en este capítulo:** KNN (K-Nearest Neighbors, k vecinos más cercanos)
