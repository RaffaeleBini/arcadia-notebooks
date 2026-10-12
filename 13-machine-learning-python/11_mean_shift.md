# 11. Clustering con Mean Shift

## Por qué te importa esto

En el capítulo anterior aprendiste K-Means: un algoritmo de clustering que agrupa datos parecidos, pero que te obliga a decidir de antemano cuántos grupos buscas. Esa decisión no siempre es fácil. ¿Y si no sabes cuántos perfiles de cliente existen en tu empresa? ¿Y si el número real de categorías en tus datos es una incógnita?

Mean Shift resuelve justo ese problema. No le dices cuántos grupos esperas: el algoritmo los descubre él solo, observando dónde se acumulan los puntos con más densidad. Tú aportas los datos y un único parámetro, el ancho de banda (bandwidth), y Mean Shift hace el resto.

En este capítulo aplicas Mean Shift con `scikit-learn` sobre el mismo dataset del Titanic que usaste en el capítulo 10, comparas cuántos grupos encuentra frente a los tres que impusiste con K-Means, programas el algoritmo desde cero para entender su mecánica interna y descubres por qué un único radio fijo no siempre es la mejor opción.

## La idea de Mean Shift: subir la colina de densidad

Imagina una plaza llena de gente repartida en corrillos. Si sueltas una gota de agua en cualquier punto de esa plaza y la gota pudiera "sentir" hacia dónde hay más gente a su alrededor, rodaría poco a poco hacia el corrillo más numeroso cercano, luego hacia el centro exacto de ese corrillo, y ahí se quedaría quieta. Eso es exactamente lo que hace Mean Shift con cada punto de tus datos.

Cada punto empieza siendo su propio candidato a centro de grupo. El algoritmo mira a su alrededor, dentro de un radio que tú defines (el bandwidth), y calcula la media de todos los puntos que caen dentro de ese radio. Ese promedio se convierte en la nueva posición del centro, un poco desplazada hacia donde hay más densidad de puntos. Repite el proceso una y otra vez: mira alrededor, calcula la media, desplázate. Poco a poco, el centro "sube la colina de densidad" hasta llegar a la cima, el punto donde ya no hay ningún desplazamiento posible porque está rodeado de la máxima concentración de datos que puede alcanzar desde ahí.

Cuando repites esto para todos los puntos del dataset, ocurre algo interesante: muchos centros terminan convergiendo en la misma cima, como si varias gotas de agua soltadas en distintos lugares de la plaza acabaran todas en el mismo corrillo. Al final, fusionas los centros que quedan casi en el mismo sitio y lo que te queda son los grupos reales de tus datos, cada uno con su propia cima de densidad. Nadie te dijo cuántas cimas había: las encontraste dejando que el algoritmo explorara el terreno.

El bandwidth es la pieza clave de toda esta historia. Es el radio de la zona que cada punto mira a su alrededor antes de decidir hacia dónde moverse. Un bandwidth grande hace que cada punto vea una zona amplia, agrupe con facilidad y termine con pocos grupos grandes. Un bandwidth pequeño hace que cada punto solo vea a sus vecinos más próximos, sea más exigente para agrupar y termine con muchos grupos pequeños. Como verás en la sección de práctica, cambiar este único número puede cambiar por completo el resultado.

Esta capacidad de encontrar densidad sin conocer de antemano el número de grupos hace que Mean Shift aparezca en tareas muy distintas a las que trabajaste con K-Means. En segmentación de imágenes, agrupa píxeles parecidos para separar un objeto del fondo sin saber cuántos objetos hay en la foto. En seguimiento de objetos en vídeo, sigue una región de interés desplazándola hacia el pico de densidad de fotograma en fotograma. En detección de anomalías, marca como sospechoso cualquier punto que quede lejos de todas las cimas de densidad encontradas. En los tres casos, la ventaja es la misma que vas a explotar con el Titanic: nadie necesita fijar el número de grupos antes de ejecutar el algoritmo.

## Aplicar Mean Shift con scikit-learn sobre el Titanic

Vas a trabajar con el mismo archivo CSV (valores separados por comas, del inglés *comma-separated values*) del Titanic que usaste con K-Means, cargado directamente desde una URL pública con `pandas`. El objetivo es idéntico al del capítulo anterior: agrupar pasajeros según sus características y ver qué relación guardan esos grupos con la supervivencia. La diferencia es que ahora no le impones un número de grupos: dejas que Mean Shift lo decida.

Primero, carga los datos y prepáralos. Este paso es igual al del capítulo 10: rellena los huecos, convierte las columnas de texto en números y descarta las columnas que no aportan información para agrupar.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(url)

# Guarda una copia legible antes de transformar nada: la necesitarás
# para interpretar los grupos con nombres reales en vez de números
original_df = titanic.copy()

# Rellena huecos y convierte texto en números
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Fare'].fillna(titanic['Fare'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Descarta columnas que no ayudan a agrupar pasajeros
titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X = titanic.drop('Survived', axis=1)
X_escalado = StandardScaler().fit_transform(X)
```

Con los datos ya escalados, aplica Mean Shift. Fíjate en que la clase `MeanShift`, del módulo `sklearn.cluster`, no recibe ningún parámetro obligatorio con el número de grupos, a diferencia de `KMeans`. Aquí le pasas un bandwidth de partida; más adelante verás qué pasa si lo cambias.

```python
modelo = MeanShift(bandwidth=2.5)
modelo.fit(X_escalado)

etiquetas = modelo.labels_
n_grupos = len(np.unique(etiquetas))
print(f"Mean Shift encontró {n_grupos} grupos")

# Añade las etiquetas al dataframe original para poder interpretarlas
original_df['grupo'] = etiquetas

for grupo in np.unique(etiquetas):
    miembros = original_df[original_df['grupo'] == grupo]
    tasa_supervivencia = miembros['Survived'].mean()
    print(f"Grupo {grupo}: {len(miembros)} pasajeros, "
          f"supervivencia {tasa_supervivencia:.0%}")
```

Recuerda que en el capítulo 10 forzaste K-Means a encontrar tres grupos, uno por cada clase de pasajero (primera, segunda y tercera). Mean Shift, en cambio, suele encontrar también tres grupos grandes con este dataset, pero por un motivo distinto: no sabe nada de las clases del barco, simplemente detecta que la densidad de pasajeros se concentra en tres zonas del espacio de características. Un grupo aparece con una tasa de supervivencia baja, cercana al 10%, otro con una tasa intermedia, cercana al 37%, y otro con una tasa alta, cercana al 84%. Esa distribución coincide, de forma llamativa, con la de tercera, segunda y primera clase respectivamente.

Hay una diferencia importante frente a K-Means: si cambias el bandwidth o repites el ajuste, Mean Shift puede encontrar un grupo adicional muy pequeño, formado por los pasajeros que pagaron una tarifa mucho más alta que el resto, con una tasa de supervivencia cercana al 100%. K-Means nunca te habría mostrado ese matiz porque tú ya habías decidido, de antemano, que solo existían tres grupos. Mean Shift, al no tener esa restricción, es capaz de aislar un grupo extremo aunque tenga muy pocos integrantes.

**Ejecuta el código con tu propio bandwidth y compara el número de grupos que obtienes frente a los tres de K-Means.**

Elegir el valor de `bandwidth` a mano, como hiciste arriba, funciona para aprender, pero `scikit-learn` también puede estimarlo por ti a partir de la propia distribución de los datos, con la función `estimate_bandwidth` del módulo `sklearn.cluster`:

```python
from sklearn.cluster import estimate_bandwidth

bandwidth_estimado = estimate_bandwidth(X_escalado, quantile=0.2)
print(f"Bandwidth estimado: {bandwidth_estimado:.2f}")

modelo_auto = MeanShift(bandwidth=bandwidth_estimado)
modelo_auto.fit(X_escalado)
print(f"Grupos con bandwidth automático: {len(np.unique(modelo_auto.labels_))}")
```

El parámetro `quantile` controla cuánta parte del dataset entra en el cálculo: cuanto más alto, mayor tiende a salir el bandwidth estimado y menos grupos encontrarás. Este atajo es útil cuando no tienes ninguna intuición previa sobre el radio correcto, aunque para dominar de verdad cómo influye el bandwidth en el resultado no hay sustituto para probarlo tú mismo con distintos valores, como harás en la sección final del capítulo.

## Programar Mean Shift desde cero

Para entender de verdad qué hace Mean Shift por dentro, vas a programarlo tú con un dataset pequeño, sin depender de `scikit-learn`. La lógica es la misma que explicaste en la analogía de la plaza: cada punto empieza siendo un centroide, se desplaza hacia la media de sus vecinos dentro de un radio fijo, y repites el proceso hasta que ningún centroide se mueve más.

```python
import numpy as np


class MeanShiftPropio:
    def __init__(self, bandwidth=4):
        # El bandwidth es el radio de búsqueda alrededor de cada centroide
        self.bandwidth = bandwidth

    def fit(self, datos):
        # Al principio, cada punto del dataset es su propio centroide
        centroides = {i: punto for i, punto in enumerate(datos)}

        while True:
            nuevos_centroides = []

            for i in centroides:
                centroide = centroides[i]
                vecinos = []

                for punto in datos:
                    # ¿Cae este punto dentro del radio del centroide?
                    if np.linalg.norm(punto - centroide) < self.bandwidth:
                        vecinos.append(punto)

                # El nuevo centroide es la media de todos sus vecinos
                nuevo_centroide = np.average(vecinos, axis=0)
                nuevos_centroides.append(tuple(nuevo_centroide))

            # Fusiona los centroides que hayan quedado duplicados
            centroides_unicos = sorted(set(nuevos_centroides))

            centroides_anteriores = dict(centroides)
            centroides = {i: np.array(c) for i, c in enumerate(centroides_unicos)}

            # Comprueba si algún centroide se movió respecto a la vuelta anterior
            optimizado = True
            for i in centroides:
                anterior = centroides_anteriores.get(i)
                if anterior is None or not np.array_equal(centroides[i], anterior):
                    optimizado = False
                    break

            if optimizado:
                break

        self.centroides = centroides


datos = np.array([
    [1, 2], [1.5, 1.8], [1, 0.6],
    [8, 8], [9, 11], [8, 2],
    [10, 2], [9, 3],
])

modelo = MeanShiftPropio(bandwidth=4)
modelo.fit(datos)
print(modelo.centroides)
```

Repasa lo que hace cada parte. El bucle `while True` representa las rondas de desplazamiento: en cada ronda, todo centroide calcula su nuevo destino mirando quién está dentro de su radio. La línea `centroides_unicos = sorted(set(nuevos_centroides))` es la fusión de centroides: si dos centroides han llegado exactamente al mismo punto (o muy cerca, según la precisión de tus datos), el conjunto (`set`) elimina el duplicado y te deja con un solo representante de esa cima de densidad. La comprobación final, con `np.array_equal`, detiene el bucle en cuanto ningún centroide se mueve entre una ronda y la siguiente, es decir, cuando has alcanzado la convergencia.

Si ejecutas este código, verás que los ocho puntos del ejemplo, que forman dos nubes claramente separadas, terminan convergiendo en solo dos centroides finales, uno por nube. Nadie le dijo al algoritmo que había dos grupos: lo dedujo de la propia forma de los datos.

Fíjate en el coste de este planteamiento: para cada centroide, en cada ronda, recorres el dataset completo comparando distancias. Con ocho puntos no lo notas, pero con miles de filas, como el Titanic del apartado anterior, esta versión artesanal se vuelve lenta enseguida. La implementación de `MeanShift` en `scikit-learn` resuelve el mismo problema con estructuras de datos optimizadas para búsquedas de vecinos, por eso la usaste sobre el dataset grande y reservaste la versión propia para un ejemplo pequeño. El objetivo de programarla tú no es competir en velocidad, sino entender exactamente qué ocurre en cada desplazamiento.

**Cambia las coordenadas de `datos` para crear tres nubes de puntos en lugar de dos y comprueba que `MeanShiftPropio` encuentra tres centroides.**

## Ancho de banda dinámico: adaptar el radio a la densidad local

El código anterior usa un bandwidth fijo: el mismo radio para todos los puntos, en todas las zonas del espacio. Esto funciona bien cuando tus datos están repartidos de forma más o menos uniforme, pero falla en un caso muy común: cuando unas zonas del dataset están muy pobladas y otras casi vacías.

Piensa de nuevo en la plaza. Si hay un corrillo enorme y apretado en una esquina y, al otro lado, apenas tres personas dispersas, un radio único no sirve para ambos casos. Un radio pequeño, pensado para separar bien a la gente del corrillo apretado, dejará a las tres personas dispersas sin ver a nadie a su alrededor y nunca las agrupará. Un radio grande, pensado para que esas tres personas se encuentren, hará que el corrillo apretado se trague de un plumazo a corrillos vecinos que en realidad eran distintos. Un radio fijo siempre sacrifica una de las dos zonas.

La solución es un bandwidth dinámico: en lugar de un radio único, usas un radio de referencia amplio y, dentro de él, penalizas a los puntos según su distancia real al centroide. Cuanto más lejos está un punto, menos peso aporta a la media que calcula el nuevo centroide; cuanto más cerca está, más peso aporta. De este modo, un punto que está justo al lado del centroide influye mucho en hacia dónde se desplaza, mientras que uno que está en el borde del radio apenas lo mueve. El efecto práctico es que el radio "se comporta" como si fuera distinto según la densidad local: en zonas muy pobladas, el peso de los vecinos lejanos apenas cuenta y el centroide se ajusta con precisión; en zonas poco pobladas, el radio amplio permite que el centroide alcance a los pocos puntos disponibles sin perderlos.

Una forma sencilla de aplicar esta idea es dividir la distancia entre el radio en varios "escalones" y asignar un peso decreciente a cada escalón:

```python
pasos_de_peso = 100  # número de escalones de distancia dentro del radio


def calcular_peso(distancia, radio):
    escalon = int(distancia / radio)
    # Los puntos que caen fuera de los escalones definidos
    # reciben el peso mínimo, no cero
    if escalon > pasos_de_peso - 1:
        escalon = pasos_de_peso - 1
    # Al elevar al cuadrado, la caída de peso con la distancia es más pronunciada
    peso = (pasos_de_peso - escalon) ** 2
    return peso
```

Con esta función, un punto muy cercano al centroide (escalón cero) recibe el peso máximo, y uno que está justo en el límite del radio recibe un peso mucho menor, pero no nulo. Al multiplicar cada punto por su peso antes de promediarlos, el nuevo centroide queda dominado por los vecinos próximos sin ignorar del todo a los lejanos. Esta es la misma idea que aplica `scikit-learn` cuando dejas el parámetro `bandwidth` de `MeanShift` sin especificar: la librería estima un radio adecuado a partir de la densidad de tus propios datos, en lugar de obligarte a adivinarlo.

## Resumen: lo que te llevas de este capítulo

- Mean Shift descubre el número de grupos por sí solo, mientras que K-Means necesita que se lo indiques con antelación
- El bandwidth es el radio que cada punto usa para mirar a su alrededor antes de desplazarse hacia la media de sus vecinos
- El proceso repite el desplazamiento hasta que los centroides convergen y ya no se mueven
- Al fusionar centroides que han llegado casi al mismo sitio, obtienes los grupos finales sin haberlos fijado de antemano
- Sobre el Titanic, Mean Shift suele encontrar grupos con supervivencia parecida a las tres clases de pasajeros, y a veces detecta un grupo extra de tarifas muy altas que K-Means no puede aislar
- Un bandwidth fijo no siempre sirve para todo el dataset: un bandwidth dinámico adapta el peso de cada vecino según su distancia real

Antes de pasar a la práctica, vale la pena que fijes cuándo conviene cada algoritmo. Elige K-Means cuando ya sabes, por conocimiento del negocio o del problema, cuántos grupos buscas: es más rápido y más fácil de ajustar. Elige Mean Shift cuando esa cifra es precisamente lo que quieres averiguar, o cuando sospechas que el número de grupos reales no coincide con ninguna categoría obvia de tus datos, como ocurrió con el grupo de tarifas altas que apareció en el Titanic.

## Ponlo en práctica

Retoma el código de Mean Shift con `scikit-learn` sobre el Titanic y prueba distintos valores de `bandwidth` en un mismo bucle, para ver cómo cambia el número de grupos que encuentra el algoritmo:

```python
for bw in [1, 2, 2.5, 4, 6, 10]:
    modelo = MeanShift(bandwidth=bw)
    modelo.fit(X_escalado)
    n_grupos = len(np.unique(modelo.labels_))
    print(f"bandwidth={bw} -> {n_grupos} grupos")
```

**Ejecuta este bucle y anota cuántos grupos aparecen con cada valor.** Fíjate en el patrón: cuanto más pequeño es el bandwidth, más grupos tiende a encontrar el algoritmo, porque cada punto exige tener vecinos muy próximos para agruparse. Cuanto más grande es el bandwidth, menos grupos aparecen, hasta llegar al extremo de fundir todo el dataset en un único grupo. **Busca el valor de bandwidth que te dé de nuevo tres grupos y compara sus tasas de supervivencia con las que obtuviste al forzar tres grupos con K-Means en el capítulo 10.**

**Siglas usadas en este capítulo:** CSV (comma-separated values, valores separados por comas)
