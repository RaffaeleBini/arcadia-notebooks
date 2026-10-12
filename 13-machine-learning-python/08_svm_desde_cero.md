# 8. Programar un SVM desde cero

## Por qué te importa esto

En el capítulo anterior descubriste la intuición de la máquina de vectores de soporte (SVM): un algoritmo que busca la línea (o el plano) que separa dos grupos de puntos dejando el mayor margen posible a cada lado. También viste cómo entrenarla en dos líneas con `scikit-learn`, usando la clase `SVC`.

Ahora vas a abrir el capó. En este capítulo construyes tu propia clase SVM en Python, sin depender de ninguna librería de machine learning. Vas a programar tres piezas: un método `fit` que busca los mejores valores de `w` (el vector de pesos) y `b` (el sesgo) probando y descartando opciones, un método `predict` que clasifica puntos nuevos, y una visualización con Matplotlib que dibuja los puntos, el hiperplano y los márgenes.

¿Para qué sirve esto si `scikit-learn` ya lo resuelve? Porque programar el algoritmo a mano te obliga a entender qué significa realmente cada línea de la fórmula que viste en el capítulo 7. Cuando más adelante uses cualquier librería, sabrás qué está pasando por dentro y podrás diagnosticar problemas con criterio propio, no solo por ensayo y error.

Una advertencia importante: el método que vas a programar aquí busca `w` y `b` probando valores cada vez más pequeños y comprobando restricciones, sin descenso de gradiente ni solucionadores de optimización externos. Es un método sencillo, pensado para aprender, no para producción. En el capítulo 9 verás una versión más eficiente con la librería CVXOPT (optimización convexa).

## El esqueleto de la clase SVM

Imagina que estás colgando un cuadro en la pared. No calculas la posición exacta con una fórmula: lo sujetas, das un paso atrás, compruebas si está torcido, ajustas un poco, vuelves a comprobar... y repites hasta que encaja. El algoritmo que vas a programar hace exactamente eso con el vector `w`: prueba un valor, comprueba si cumple una condición sobre todos los puntos y, si no la cumple, lo ajusta y prueba otra vez.

Antes de llegar ahí, define la estructura básica de la clase. Toda clase en Python que representa un modelo suele tener esta forma: un método que la inicializa, uno que la entrena y uno que predice.

```python
import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, visualizacion=True):
        # Guarda si quieres ver el resultado en un gráfico
        self.visualizacion = visualizacion
        # Un color para cada clase: rojo para la clase 1, azul para la -1
        self.colores = {1: 'r', -1: 'b'}

        if self.visualizacion:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

        # Todavía no conocemos el vector de pesos ni el sesgo
        self.w = None
        self.b = None

    def fit(self, data):
        # Aquí irá el algoritmo que busca los mejores w y b
        pass

    def predict(self, features):
        # Devolverá la predicción según el signo de (X . W + b)
        return np.sign(np.dot(np.array(features), self.w) + self.b)

    def visualizar(self):
        # Aquí irá el código que dibuja los puntos, el hiperplano y los márgenes
        pass
```

Fíjate en los atributos `w` y `b`. Son el corazón de todo el capítulo: `w` es el vector perpendicular al hiperplano y `b` es el término que lo desplaza respecto al origen. Mientras valgan `None`, la clase no sabe clasificar nada. El trabajo del método `fit` es rellenarlos con los valores correctos.

Los datos de entrada tienen una forma concreta: un diccionario con dos claves, `-1` y `1`, una por cada clase, cuyos valores son listas de puntos. Por ejemplo:

```python
datos = {
    -1: np.array([[1, 7], [2, 8], [3, 8]]),
     1: np.array([[5, 1], [6, -1], [7, 3]])
}
```

Con este esqueleto listo, toca completar la pieza que falta: el entrenamiento.

## Entrenar el SVM: encontrar w y b paso a paso

Recuerda la restricción que debe cumplir cada punto para estar bien clasificado con margen suficiente, la que ya viste en el capítulo 7:

```
yi * (w . xi + b) >= 1
```

Aquí `xi` es un punto del dataset, `yi` es su clase (uno o menos uno) y el producto `w . xi + b` es la distancia (con signo) de ese punto al hiperplano. Si la restricción se cumple para todos los puntos, tienes un candidato válido de `w` y `b`. Entre todos los candidatos válidos, el mejor es el que tiene el `w` más pequeño en magnitud, porque un `w` pequeño corresponde a un margen grande.

La estrategia del algoritmo es simple de describir: parte de un `w` grande y ve reduciéndolo poco a poco, comprobando en cada paso si la restricción se cumple para todos los puntos. En cuanto deja de cumplirse, sabes que el valor anterior era el límite y ese es tu candidato.

Para no perder precisión ni tiempo, el algoritmo usa pasos de tamaño decreciente: empieza con pasos grandes (10% del valor máximo de las características), sigue con pasos medianos (1%) y termina con pasos pequeños (0,1%). Es la misma idea que usarías para encajar un tornillo: primero lo giras rápido y, cuando notas resistencia, giras poco a poco para no pasarte.

El sesgo `b` se explora de forma parecida, pero con pasos más grandes, porque no necesita tanta precisión como `w` y afinarlo demasiado sale caro en tiempo de cálculo.

Aquí tienes el método `fit` completo:

```python
class SVM:
    def __init__(self, visualizacion=True):
        self.visualizacion = visualizacion
        self.colores = {1: 'r', -1: 'b'}
        if self.visualizacion:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)
        self.w = None
        self.b = None

    def fit(self, data):
        self.data = data
        # Diccionario para guardar candidatos: {magnitud de w: [w, b]}
        opt_dict = {}

        # Cuatro combinaciones de signos para probar w en todas las direcciones
        transformaciones = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        # Buscamos el valor máximo y mínimo entre todas las características
        todos_los_valores = []
        for clase in self.data:
            for punto in self.data[clase]:
                for valor in punto:
                    todos_los_valores.append(valor)

        self.max_valor = max(todos_los_valores)
        self.min_valor = min(todos_los_valores)
        todos_los_valores = None  # liberamos memoria, ya no la necesitamos

        # Pasos de tamaño decreciente: grandes, medianos y pequeños
        pasos = [self.max_valor * 0.1,
                 self.max_valor * 0.01,
                 self.max_valor * 0.001]

        # b se explora con pasos más grandes: no necesita tanta precisión
        b_rango_multiplo = 5
        b_paso_multiplo = 5
        ultimo_optimo = self.max_valor * 10

        for paso in pasos:
            w = np.array([ultimo_optimo, ultimo_optimo])
            optimizado = False

            while not optimizado:
                for b in np.arange(-1 * self.max_valor * b_rango_multiplo,
                                    self.max_valor * b_rango_multiplo,
                                    paso * b_paso_multiplo):
                    for transformacion in transformaciones:
                        w_candidato = w * transformacion
                        cumple_restriccion = True

                        # Comprobamos la restricción en cada punto de cada clase
                        for clase in self.data:
                            for punto in self.data[clase]:
                                yi = clase
                                if not yi * (np.dot(w_candidato, punto) + b) >= 1:
                                    cumple_restriccion = False
                                    break
                            if not cumple_restriccion:
                                break

                        if cumple_restriccion:
                            opt_dict[np.linalg.norm(w_candidato)] = [w_candidato, b]

                if w[0] < 0:
                    # Ya no tiene sentido seguir reduciendo w en este paso
                    optimizado = True
                else:
                    w = w - paso

            # Nos quedamos con el candidato de menor magnitud: el mejor margen
            normas = sorted([n for n in opt_dict])
            eleccion = opt_dict[normas[0]]
            self.w = eleccion[0]
            self.b = eleccion[1]
            ultimo_optimo = eleccion[0][0] + paso * 2
```

Lee el bucle con calma, porque son cuatro niveles anidados:

- el bucle exterior recorre los tamaños de paso, de grande a pequeño
- para cada paso, el bucle `while` reduce `w` hasta que deja de cumplir la condición de parada
- dentro, recorres valores posibles de `b`
- y para cada `b`, pruebas las cuatro combinaciones de signos de `w`

Cada combinación que supera la comprobación de todos los puntos entra en `opt_dict` junto a su magnitud. Al terminar cada tanda de pasos, ordenas esas magnitudes y te quedas con la más pequeña: ese es tu mejor `w` y `b` hasta el momento. El resultado alimenta el siguiente tramo de pasos, cada vez más fino.

Este proceso es costoso porque revisa todos los puntos del dataset en cada combinación. Con pocos puntos, como en los ejemplos de este capítulo, tarda una fracción de segundo. Con datasets grandes, se vuelve lento enseguida, y ahí es donde métodos más avanzados como la optimización convexa del capítulo 9 marcan la diferencia.

## Predecir y visualizar el hiperplano

Con `w` y `b` ya calculados, predecir es la parte fácil: aplicas la fórmula que ya conoces del capítulo 7, `signo(w . x + b)`, y el signo del resultado te dice a qué clase pertenece el punto.

```python
    def predict(self, features):
        clasificacion = np.sign(np.dot(np.array(features), self.w) + self.b)
        if clasificacion != 0 and self.visualizacion:
            self.ax.scatter(features[0], features[1], s=200, marker='*',
                             c=self.colores[clasificacion])
        return clasificacion
```

Cuando la visualización está activa, cada punto predicho se marca en el gráfico con una estrella, del color de la clase asignada. Así comprueban de un vistazo si la predicción tiene sentido respecto a los puntos de entrenamiento.

Para dibujar el hiperplano y los márgenes necesitas una función auxiliar que, dada una coordenada `x`, calcule la `y` correspondiente sobre una de las tres líneas: el margen positivo (`w.x + b = 1`), el margen negativo (`w.x + b = -1`) o el propio hiperplano de decisión (`w.x + b = 0`). Despejando la `y` de esa ecuación obtienes la función `hiperplano`.

```python
    def visualizar(self):
        # Dibuja todos los puntos de entrenamiento, coloreados por clase
        [[self.ax.scatter(punto[0], punto[1], s=100, color=self.colores[clase])
          for punto in self.data[clase]] for clase in self.data]

        def hiperplano(x, w, b, desplazamiento):
            return (-w[0] * x - b + desplazamiento) / w[1]

        # Un poco de margen extra a los lados del gráfico
        rango_datos = (self.min_valor * 0.9, self.max_valor * 1.1)
        x_min = rango_datos[0]
        x_max = rango_datos[1]

        # Margen positivo: w.x + b = 1
        psv1 = hiperplano(x_min, self.w, self.b, 1)
        psv2 = hiperplano(x_max, self.w, self.b, 1)
        self.ax.plot([x_min, x_max], [psv1, psv2], 'k')

        # Margen negativo: w.x + b = -1
        nsv1 = hiperplano(x_min, self.w, self.b, -1)
        nsv2 = hiperplano(x_max, self.w, self.b, -1)
        self.ax.plot([x_min, x_max], [nsv1, nsv2], 'k')

        # Hiperplano de decisión: w.x + b = 0
        db1 = hiperplano(x_min, self.w, self.b, 0)
        db2 = hiperplano(x_max, self.w, self.b, 0)
        self.ax.plot([x_min, x_max], [db1, db2], 'y--')

        plt.show()
```

Ya tienes las tres piezas completas. Júntalas y prueba tu SVM con el dataset de ejemplo:

```python
datos = {
    -1: np.array([[1, 7], [2, 8], [3, 8]]),
     1: np.array([[5, 1], [6, -1], [7, 3]])
}

svm = SVM()
svm.fit(datos)

puntos_a_predecir = [[0, 1], [1, 3], [3, 4], [3, 5],
                     [5, 5], [5, 6], [6, -5], [5, 8]]

for punto in puntos_a_predecir:
    print(punto, svm.predict(punto))

svm.visualizar()
```

Al ejecutar este código obtienes dos cosas: en la consola, la clase (`1.0` o `-1.0`) asignada a cada punto nuevo; en el gráfico, los puntos de entrenamiento de cada clase, las estrellas de las predicciones y tres líneas paralelas, la del hiperplano de decisión en amarillo discontinuo y las dos de los márgenes en negro continuo. Comprueba que los puntos de la clase `-1` quedan a un lado y los de la clase `1` al otro, con las estrellas cayendo del lado que corresponde según su posición.

## Resumen: lo que te llevas de este capítulo

- La clase SVM necesita solo tres piezas: `fit` para entrenar, `predict` para clasificar y una visualización para interpretar el resultado
- Los atributos `w` y `b` empiezan vacíos y `fit` es la única función responsable de calcularlos
- El entrenamiento que programaste no usa descenso de gradiente ni librerías de optimización: prueba valores de `w` cada vez más pequeños y los descarta en cuanto algún punto incumple la restricción `yi * (w . xi + b) >= 1`
- Entre todos los candidatos válidos de `w`, el algoritmo elige siempre el de menor magnitud, porque corresponde al margen más amplio
- `predict` aplica la misma fórmula del capítulo 7, `signo(w . x + b)`, para decidir la clase de un punto nuevo
- La visualización dibuja el hiperplano de decisión y sus dos márgenes despejando la `y` de la ecuación `w.x + b = valor`, con `valor` igual a `0`, `1` o `-1`

## Ponlo en práctica

Ahora te toca comprobar que el algoritmo funciona con datos distintos a los del ejemplo.

**Copia** la clase `SVM` completa de este capítulo en un archivo nuevo. **Sustituye** el diccionario `datos` por otro conjunto pequeño de puntos de dos clases, por ejemplo:

```python
datos = {
    -1: np.array([[2, 2], [3, 1], [3, 3]]),
     1: np.array([[8, 8], [9, 6], [7, 9]])
}
```

**Entrena** el modelo con `svm.fit(datos)` y **añade** dos o tres puntos propios a la lista `puntos_a_predecir`, situados claramente cerca de una de las dos clases. **Ejecuta** el código y **observa** el gráfico: confirma que el hiperplano se sitúa entre los dos grupos y que las estrellas de tus predicciones caen del lado correcto. Si mueves un punto de entrenamiento muy cerca del otro grupo, **repite** la prueba y fíjate en cómo cambia la posición del hiperplano y de los márgenes.

**Siglas usadas en este capítulo:** SVM (Support Vector Machine, máquina de vectores de soporte)
