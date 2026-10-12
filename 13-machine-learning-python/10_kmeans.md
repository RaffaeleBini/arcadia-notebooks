# 10. Clustering con K-Means

## Por qué te importa esto

Hasta ahora, en todos los capítulos anteriores, cada modelo aprendía a partir de ejemplos con una respuesta correcta conocida. En regresión le dabas el precio real, en los k vecinos más cercanos (KNN) y en la máquina de vectores de soporte (SVM) le dabas la clase real. A ese enfoque se le llama aprendizaje supervisado: alguien ya etiquetó los datos y el algoritmo solo tiene que aprender la relación entre las características y esa etiqueta.

Este capítulo abre un bloque distinto: el aprendizaje no supervisado. Aquí nadie te da la respuesta correcta. Le entregas los datos al algoritmo sin ninguna etiqueta y le pides que encuentre por sí mismo grupos de elementos parecidos entre sí. Vas a conocer K-Means, el algoritmo de clustering más usado en la práctica, y lo vas a aplicar sobre el conjunto de datos del Titanic para comprobar algo sorprendente: sin decirle nunca quién sobrevivió, el algoritmo agrupa a los pasajeros de una forma que se parece bastante a la supervivencia real.

Este tipo de técnica tiene aplicaciones directas en proyectos reales de digitalización: segmentar clientes sin categorías previas, agrupar explotaciones agrarias por patrones de consumo, o detectar perfiles de pacientes con comportamientos similares en un conjunto de datos de salud. Entender cómo funciona K-Means por dentro te permite usarlo con criterio, no como una caja mágica.

## Aprendizaje no supervisado: agrupar sin etiquetas

Imagina que vuelcas sobre la mesa una caja de botones mezclados: de colores distintos, tamaños distintos, algunos con cuatro agujeros y otros con dos. Nadie te ha dicho de antemano cuántos grupos existen ni qué característica debes usar para separarlos. Aun así, tú, de forma natural, empiezas a acercar los botones parecidos entre sí: aquí los rojos, allí los grandes, más allá los de dos agujeros. No conocías las categorías antes de empezar; las descubriste mirando los propios botones.

Eso es exactamente lo que hace un algoritmo de aprendizaje no supervisado. No recibe ninguna etiqueta que le diga "este botón es del grupo A" y "este otro es del grupo B". Solo recibe las características de cada elemento (color, tamaño, número de agujeros) y tiene que decidir, basándose en el parecido entre esas características, qué elementos van juntos.

El clustering es la familia de técnicas que resuelve este problema. Existen dos formas principales:

- Clustering plano: tú decides de antemano cuántos grupos quieres, por ejemplo dos o tres, y el algoritmo reparte los datos en ese número exacto de grupos
- Clustering jerárquico: el propio algoritmo decide cuántos grupos existen y cómo se relacionan entre ellos, sin que tú fijes ese número

K-Means pertenece al primer tipo: tú eliges cuántos grupos, llamados clústeres, quieres encontrar, y el algoritmo busca los mejores centros posibles para esa cantidad. La letra k del nombre representa precisamente ese número de clústeres que tú decides antes de empezar.

La intuición central de K-Means es sencilla: coloca k puntos centrales, llamados centroides, e intenta que cada dato quede lo más cerca posible de su centroide asignado. El algoritmo mueve esos centroides paso a paso hasta que encajan bien con la forma real de los datos. Más adelante en este capítulo verás el mecanismo exacto, paso a paso.

## Preparar datos con columnas de texto: pasar de categorías a números

K-Means, como casi todos los algoritmos de aprendizaje automático, solo entiende números. Calcula distancias entre puntos, y una distancia solo tiene sentido entre valores numéricos. El problema es que muchos conjuntos de datos reales incluyen columnas de texto: el sexo de una persona, el puerto donde embarcó o el color de un producto.

Vas a trabajar con el conjunto de datos del Titanic, que recoge información de los pasajeros del barco: clase del billete, sexo, edad, tarifa pagada y si sobrevivieron o no, entre otras columnas. Usamos una copia pública estable del conjunto de datos del Titanic en vez de la fuente original, que ya no está disponible, y la cargamos directamente desde una URL con pandas:

```python
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df.head())
print(df.dtypes)
```

Al revisar `df.dtypes` verás que columnas como `Pclass`, `Age`, `Fare` o `Survived` ya son numéricas, pero otras, como `Sex` o `Embarked`, son de tipo texto (`object`). Esas columnas de texto son las que necesitas convertir antes de aplicar K-Means.

La técnica más directa se llama codificación por etiquetas (*label encoding*): a cada valor distinto de una columna de texto le asignas un número entero único. Por ejemplo, "female" pasa a ser 0 y "male" pasa a ser 1. Puedes construir tú mismo esta conversión con un diccionario, sin depender de ninguna librería adicional:

```python
def convertir_columnas_texto_a_numero(df):
    df = df.copy()
    columnas = df.columns.values

    for columna in columnas:
        texto_a_numero = {}

        def convertir_valor(valor):
            return texto_a_numero[valor]

        # Solo tocamos columnas que no son ya numéricas
        if df[columna].dtype != "int64" and df[columna].dtype != "float64":
            valores_columna = df[columna].fillna(0).tolist()
            valores_unicos = set(valores_columna)

            contador = 0
            for valor_unico in valores_unicos:
                if valor_unico not in texto_a_numero:
                    texto_a_numero[valor_unico] = contador
                    contador += 1

            df[columna] = list(map(convertir_valor, df[columna].fillna(0)))

    # Las columnas que ya eran numéricas (como Age) también pueden tener
    # huecos: los rellenamos aquí para que K-Means no falle con un ValueError
    df = df.fillna(0)

    return df


# Nos quedamos con columnas relevantes y descartamos las que no aportan
# información útil para agrupar pasajeros (nombre, identificador de billete...)
df_modelo = df.drop(
    ["PassengerId", "Name", "Ticket", "Cabin"], axis=1
)
df_modelo = convertir_columnas_texto_a_numero(df_modelo)

print(df_modelo.head())
print(df_modelo.dtypes)
```

Con este proceso, `Sex` pasa a contener 0 y 1, y `Embarked` pasa a contener un número entero por cada puerto de embarque. Fíjate en que también rellenamos los valores ausentes con `fillna(0)` antes de la conversión: la columna `Age`, por ejemplo, tiene huecos en varios pasajeros, y K-Means no puede trabajar con valores vacíos. Rellenar con cero es una solución simple para este ejercicio; en un proyecto real valorarías rellenar con la media o la mediana de la columna en vez de con cero.

**Ejecuta `df_modelo.dtypes` después de la conversión y comprueba que todas las columnas son ya `int64` o `float64`.** Ese es el requisito mínimo para poder aplicar K-Means.

## Aplicar K-Means con scikit-learn sobre el Titanic

Con los datos ya convertidos a números, puedes aplicar `KMeans`, la clase de agrupamiento que scikit-learn ofrece en `sklearn.cluster`. La pregunta que quieres responder es: si le pides al algoritmo que separe a los pasajeros en dos grupos sin decirle nunca quién sobrevivió, ¿esos dos grupos coincidirán con la supervivencia real?

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Separamos la columna objetivo: la usaremos solo para comparar, nunca
# para entrenar. Dársela al algoritmo sería hacer trampa.
y = df_modelo["Survived"]
X = df_modelo.drop(["Survived"], axis=1)

# K-Means calcula distancias, así que escalamos las características para
# que ninguna domine el resultado solo por tener valores más grandes
# (la tarifa, por ejemplo, llega a superar los 500, mientras que el sexo
# solo vale 0 o 1)
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# Creamos el modelo con dos clústeres: buscamos si el algoritmo separa
# a los pasajeros en dos grupos por su cuenta
modelo = KMeans(n_clusters=2, n_init=10, random_state=42)
modelo.fit(X_escalado)

etiquetas = modelo.labels_

# Comparamos las etiquetas encontradas por el algoritmo con la
# supervivencia real. Como K-Means numera sus grupos de forma arbitraria
# (el grupo 0 no significa "no sobrevivió" por definición), probamos
# las dos correspondencias posibles y nos quedamos con la mejor
aciertos_directos = sum(etiquetas == y) / len(y)
aciertos_invertidos = sum(etiquetas != y) / len(y)
precision = max(aciertos_directos, aciertos_invertidos)

print(f"Coincidencia entre los clústeres y la supervivencia real: {precision:.2%}")
```

Si ejecutas este código verás que la coincidencia ronda el 60-70%, un resultado nada trivial teniendo en cuenta que el algoritmo nunca vio la columna `Survived`. Esto confirma algo importante sobre el clustering: aunque es una técnica no supervisada, en muchos conjuntos de datos reales los grupos que encuentra por similitud de características coinciden, al menos en parte, con categorías que sí importan en el mundo real, como la supervivencia. Columnas como `Sex` o `Pclass` influyen mucho en ese parecido, porque separan de forma bastante clara a los pasajeros en subgrupos.

Este resultado también recuerda algo que ya viste con la SVM: en la práctica es habitual usar clustering como paso previo a un problema supervisado. Primero agrupas los datos sin etiquetas para explorar patrones, y después, si te interesa, usas esos grupos como si fueran etiquetas para entrenar un clasificador.

## Programar K-Means desde cero: el algoritmo de Lloyd

El procedimiento que sigue `KMeans` de scikit-learn por dentro se llama algoritmo de Lloyd, y es mucho más sencillo de lo que su nombre sugiere. Consta de cuatro pasos que se repiten en bucle:

1. Inicialización: eliges k puntos del conjunto de datos como centroides de partida
2. Asignación: cada punto del conjunto de datos se asigna al centroide más cercano
3. Actualización: cada centroide se recalcula como el promedio de todos los puntos que tiene asignados
4. Repetición: vuelves a los pasos 2 y 3 hasta que los centroides dejan de moverse de forma significativa

Vas a programar este algoritmo desde cero, sin usar `KMeans` de scikit-learn, sobre un conjunto de datos pequeño y sencillo de seis puntos en dos dimensiones. Así ves con total claridad cómo se mueven los centroides en cada vuelta.

```python
import numpy as np


class KMeansDesdeCero:
    def __init__(self, k=2, tolerancia=0.001, max_iteraciones=300):
        self.k = k
        self.tolerancia = tolerancia
        self.max_iteraciones = max_iteraciones

    def ajustar(self, datos):
        # Paso 1: inicialización. Tomamos los primeros k puntos del
        # conjunto de datos como centroides de partida
        self.centroides = {}
        for i in range(self.k):
            self.centroides[i] = datos[i]

        for iteracion in range(self.max_iteraciones):
            self.clasificaciones = {i: [] for i in range(self.k)}

            # Paso 2: asignación. Cada punto se asigna al centroide
            # más cercano, medido con la distancia euclídea
            for punto in datos:
                distancias = [
                    np.linalg.norm(punto - self.centroides[centroide])
                    for centroide in self.centroides
                ]
                clasificacion = distancias.index(min(distancias))
                self.clasificaciones[clasificacion].append(punto)

            centroides_anteriores = dict(self.centroides)

            # Paso 3: actualización. Cada centroide pasa a ser el
            # promedio de los puntos que tiene asignados
            for clasificacion in self.clasificaciones:
                self.centroides[clasificacion] = np.average(
                    self.clasificaciones[clasificacion], axis=0
                )

            # Paso 4: comprobamos si los centroides ya dejaron de moverse
            convergencia = True
            for centroide in self.centroides:
                centroide_original = centroides_anteriores[centroide]
                centroide_actual = self.centroides[centroide]
                cambio = np.sum(
                    (centroide_actual - centroide_original) / centroide_original * 100.0
                )
                if abs(cambio) > self.tolerancia:
                    convergencia = False

            if convergencia:
                print(f"El algoritmo convergió en la iteración {iteracion + 1}")
                break

        return self

    def predecir(self, punto):
        distancias = [
            np.linalg.norm(punto - self.centroides[centroide])
            for centroide in self.centroides
        ]
        return distancias.index(min(distancias))


# Un conjunto de datos pequeño, pensado para ver los grupos a simple vista:
# tres puntos agrupados cerca del origen y tres puntos agrupados más arriba
datos = np.array([
    [1, 2],
    [1.5, 1.8],
    [1, 0.6],
    [9, 11],
    [8, 2],
    [10, 2],
])

modelo_propio = KMeansDesdeCero(k=2)
modelo_propio.ajustar(datos)

print("Centroides encontrados:")
for indice, centroide in modelo_propio.centroides.items():
    print(f"  Clúster {indice}: {centroide}")

punto_nuevo = np.array([0, 2])
print(f"El punto {punto_nuevo} pertenece al clúster {modelo_propio.predecir(punto_nuevo)}")
```

Fíjate en tres detalles del código:

- Los centroides y las clasificaciones se guardan en diccionarios indexados por el número de clúster, así puedes recorrerlos y actualizarlos de forma ordenada en cada vuelta del bucle
- El criterio de convergencia compara el cambio porcentual entre el centroide anterior y el nuevo; si ese cambio es menor que la tolerancia para todos los centroides, el algoritmo se detiene aunque no haya llegado al máximo de iteraciones
- El método `predecir` reutiliza exactamente la misma lógica del paso de asignación: calcula la distancia del punto nuevo a cada centroide ya entrenado y devuelve el más cercano

Si ejecutas este código verás que el algoritmo converge en muy pocas iteraciones, porque el conjunto de datos de ejemplo tiene dos grupos muy separados entre sí. Los centroides finales quedan cerca del centro real de cada grupo de puntos, exactamente el comportamiento que esperabas del algoritmo de Lloyd.

## Resumen: lo que te llevas de este capítulo

- El aprendizaje no supervisado busca patrones en datos sin etiquetas: nadie le dice al algoritmo la respuesta correcta de antemano
- K-Means es un algoritmo de clustering plano: tú decides con antelación el número de clústeres, k, que quieres encontrar
- Antes de aplicar K-Means necesitas convertir las columnas de texto a números, por ejemplo con codificación por etiquetas
- Aplicado al Titanic sin la columna de supervivencia, K-Means agrupa a los pasajeros de una forma que coincide, en buena parte, con quién sobrevivió realmente
- El algoritmo de Lloyd, que scikit-learn usa por dentro, se reduce a repetir dos pasos: asignar cada punto al centroide más cercano y recalcular los centroides como el promedio de sus puntos asignados
- Programar K-Means desde cero confirma que el mecanismo interno es sencillo, aunque scikit-learn lo ejecuta de forma mucho más optimizada

## Ponlo en práctica

**Cambia el número de clústeres, k, en tu implementación desde cero y observa cómo cambian los grupos.** Vuelve a ejecutar `KMeansDesdeCero` con `k=3` sobre el mismo conjunto de seis puntos y compara los centroides resultantes con los que obtuviste antes con `k=2`:

```python
modelo_tres_clusteres = KMeansDesdeCero(k=3)
modelo_tres_clusteres.ajustar(datos)

print("Centroides con k=3:")
for indice, centroide in modelo_tres_clusteres.centroides.items():
    print(f"  Clúster {indice}: {centroide}")
```

**Repite el mismo experimento sobre el conjunto de datos del Titanic, probando con `n_clusters=3` y `n_clusters=4` en el bloque de scikit-learn.** Observa qué ocurre con el porcentaje de coincidencia respecto a la supervivencia real a medida que aumentas k: un número de clústeres mayor no siempre mejora la interpretación de los grupos, porque empieza a dividir un mismo patrón real en varios subgrupos artificiales.

---

**Siglas usadas en este capítulo:** KNN (K-Nearest Neighbors, k vecinos más cercanos), SVM (Support Vector Machine, máquina de vectores de soporte)
