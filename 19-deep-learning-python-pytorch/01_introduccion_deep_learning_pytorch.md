# 01. Introducción a Deep Learning con Python y PyTorch

## Por qué importa este tema

Reconocer una cara, entender una frase o predecir si una máquina va a fallar son tareas que las personas resolvemos sin esfuerzo, pero que resultan casi imposibles de programar con reglas explícitas: hay demasiadas variantes y excepciones para anticiparlas todas a mano. El aprendizaje profundo (DL, Deep Learning) resuelve este problema de una forma distinta: en lugar de programar las reglas, entrena un modelo para que las deduzca a partir de datos.

Python y PyTorch son, hoy, una de las combinaciones más usadas para llevar esa idea a la práctica. Python aporta una sintaxis sencilla y un ecosistema enorme de librerías; PyTorch, desarrollado por el laboratorio de investigación en inteligencia artificial (IA) de Facebook (FAIR, Facebook AI Research), aporta las herramientas necesarias para construir, entrenar y evaluar redes neuronales de forma flexible. En este capítulo vas a dar tus primeros pasos con ambos: qué es el aprendizaje profundo, qué es una red neuronal, qué es un tensor y qué hace autograd por ti cuando entrenas un modelo.

## Aprendizaje profundo frente a aprendizaje automático tradicional

El aprendizaje automático (ML, Machine Learning) ya cambia la lógica de la programación tradicional: en vez de escribir reglas a mano, le das al sistema ejemplos y deja que deduzca el patrón. El aprendizaje profundo es un subcampo del ML que usa un tipo concreto de modelo, la red neuronal, y en particular redes con varias capas ocultas apiladas ("profundas"), capaces de aprender representaciones cada vez más complejas de los datos: de píxeles sueltos a bordes, de bordes a formas, de formas a objetos completos.

Piensa en la diferencia con una analogía culinaria. El ML tradicional es como seguir una receta que un cocinero experto ya perfeccionó: tú, como programador, defines a mano qué características (features) importan y en qué proporción. El aprendizaje profundo es más parecido a dejar que un aprendiz pruebe miles de combinaciones de ingredientes hasta encontrar por sí mismo la receta que mejor funciona, sin que nadie le explique de antemano qué características cuentan realmente. Esta capacidad de aprender directamente las características, en lugar de que se las den ya definidas, es lo que hace del aprendizaje profundo la técnica dominante en visión por computador, procesamiento de lenguaje natural (NLP, Natural Language Processing) y reconocimiento de voz.

## Qué es una red neuronal

Una red neuronal (NN, Neural Network) es un modelo matemático inspirado, de forma muy simplificada, en cómo se conectan las neuronas del cerebro humano. Está formada por nodos llamados neuronas, organizados en capas:

- Una **capa de entrada**, que recibe los datos ya convertidos en números (por ejemplo, el valor de cada píxel de una imagen)
- Una o varias **capas ocultas**, llamadas así porque no controlas directamente lo que ocurre en ellas
- Una **capa de salida**, que produce la predicción final

Cada neurona recibe varios valores de entrada, cada uno multiplicado por un peso, les suma un sesgo (bias) y pasa el resultado por una función de activación antes de enviarlo a la siguiente capa. La función de activación introduce no linealidad: sin ella, por muchas capas que apiles, la red entera equivaldría a una simple regresión lineal, incapaz de aprender patrones complejos. Dos funciones de activación habituales son ReLU (Rectified Linear Unit, unidad lineal rectificada), que deja pasar los valores positivos y convierte en cero los negativos, y sigmoid, que comprime cualquier valor a un rango entre cero y uno, útil para interpretar la salida como una probabilidad.

Entrenar una red neuronal consiste en ajustar sus pesos y sesgos para minimizar la diferencia entre lo que predice y lo que debería predecir. Ese ajuste se hace de forma iterativa, repitiendo un ciclo de predicción, medición del error y corrección, hasta miles de veces. Con capas de tamaño moderado, una red puede acumular fácilmente millones de parámetros: cuantos más tiene, más patrones complejos puede aprender, pero también más riesgo corre de memorizar los ejemplos de entrenamiento en lugar de generalizar a datos nuevos, un problema conocido como sobreajuste (overfitting).

## Qué es un tensor

PyTorch necesita representar absolutamente todo —un número suelto, una imagen, un lote de miles de imágenes— con una única estructura de datos sobre la que aplicar las mismas operaciones matemáticas de forma eficiente. Esa estructura es el **tensor**: un array multidimensional muy parecido a un `ndarray` de la librería NumPy, con la ventaja añadida de poder residir en la memoria de una GPU (Graphics Processing Unit) y beneficiarse así de su capacidad de cálculo en paralelo.

Puedes pensar en los tensores como una familia de contenedores de números que escala con la complejidad del dato:

- Un número aislado es un tensor de dimensión cero (un escalar)
- Una lista de números es un tensor de una dimensión (un vector)
- Una tabla de números es un tensor de dos dimensiones (una matriz)
- Una imagen en color, o un lote de varias imágenes, es un tensor de tres o más dimensiones

Da igual con qué trabajes en PyTorch —números sueltos, imágenes, texto ya convertido a valores numéricos—: siempre lo harás sobre tensores.

## Qué es autograd

Entrenar una red neuronal exige calcular, para cada peso, cuánto contribuyó al error final: es lo que se conoce como el gradiente. Hacerlo a mano, derivando cada operación de la red, sería inviable en cuanto la arquitectura crece un poco. **Autograd** es el módulo de diferenciación automática de PyTorch: registra, en un grafo, cada operación que aplicas sobre un tensor marcado con `requires_grad=True`, y cuando llamas a `.backward()` recorre ese grafo hacia atrás y calcula automáticamente el gradiente de cada tensor implicado.

Gracias a autograd, no necesitas programar la retropropagación (backpropagation) del error paso a paso: defines las operaciones hacia delante (el forward pass) y PyTorch se encarga de derivarlas hacia atrás por ti. Esta característica, junto con el hecho de que PyTorch construye el grafo de cálculo sobre la marcha —a diferencia de librerías con un grafo estático definido de antemano—, es lo que hace que depurar y experimentar con PyTorch resulte tan directo.

## Ejemplo de código comentado

### Tensores básicos

```python
import torch

# Un tensor de una dimensión (vector), a partir de una lista de Python
x = torch.tensor([5, 3])
y = torch.tensor([2, 1])

# PyTorch sobrecarga los operadores habituales: esto multiplica elemento a elemento
print(x * y)  # tensor([10, 3])

# torch.zeros crea un tensor lleno de ceros con la forma (shape) que le indiques
z = torch.zeros((2, 5))
print(z.shape)  # torch.Size([2, 5])

# view reorganiza los mismos datos en una forma distinta, sin copiarlos
z_reordenado = z.view(1, 10)
print(z_reordenado.shape)  # torch.Size([1, 10])
```

Fíjate en que `x * y` no suma ni concatena los tensores: multiplica cada posición con la posición equivalente del otro tensor (5×2=10, 3×1=3). Esta operación elemento a elemento es la base de casi todo el cálculo numérico en deep learning.

### Tu primer gradiente con autograd

```python
# requires_grad=True le dice a PyTorch que registre cada operación sobre x
x = torch.tensor([2.0, 3.0], requires_grad=True)

y = x + 2
z = y * y * 3

# backward() recorre el grafo hacia atrás y calcula el gradiente de z respecto a x
z.backward(torch.tensor([1.0, 1.0]))

print(x.grad)  # tensor([24., 30.])
```

No necesitas resolver esta derivada a mano para confiar en el resultado: lo importante es entender que `.backward()` es la instrucción que dispara todo el cálculo de gradientes, y que `x.grad` es donde PyTorch los deja disponibles para el optimizador.

### Una red neuronal mínima con nn.Module y nn.Linear

```python
import torch
import torch.nn as nn

class RedMinima(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(3, 1): una capa totalmente conectada que recibe tres
        # valores de entrada y produce un único valor de salida
        self.capa = nn.Linear(3, 1)

    def forward(self, x):
        return self.capa(x)

modelo = RedMinima()

# Un único ejemplo de entrada, con tres características
entrada = torch.tensor([[1.0, 2.0, 3.0]])

# El forward pass: PyTorch llama a forward() automáticamente al usar modelo(entrada)
salida = modelo(entrada)
print(salida)  # un tensor con un único valor, la predicción del modelo

# Como salida depende de pesos con requires_grad=True (por defecto en nn.Linear),
# puedes propagar el error hacia atrás y obtener el gradiente de cada peso
salida.backward()
print(modelo.capa.weight.grad)  # gradiente de cada uno de los tres pesos
```

Toda red neuronal en PyTorch sigue este mismo patrón: una clase que hereda de `nn.Module`, capas definidas en `__init__` (aquí, una única `nn.Linear`) y la lógica del forward pass en el método `forward`. PyTorch construye el grafo de cálculo mientras ejecuta `forward`, así que basta con llamar a `.backward()` sobre la salida para obtener los gradientes de todos los pesos implicados, exactamente igual que en el ejemplo de autograd anterior.

## Resumen de puntos clave

- El aprendizaje profundo es un subcampo del ML que usa redes neuronales con varias capas para aprender representaciones de los datos directamente, sin que tú definas a mano qué características importan
- Una red neuronal está formada por neuronas organizadas en capas (entrada, ocultas y salida); cada conexión tiene un peso, y una función de activación introduce la no linealidad necesaria para aprender patrones complejos
- Un tensor es la estructura central de PyTorch: un array multidimensional que representa cualquier dato, desde un número suelto hasta un lote de imágenes, y que puede ejecutarse en CPU (Central Processing Unit) o en GPU
- Autograd registra las operaciones sobre los tensores con `requires_grad=True` y calcula sus gradientes automáticamente al llamar a `.backward()`, evitando que tengas que derivar la retropropagación a mano
- En PyTorch, una red neuronal se define como una clase que hereda de `nn.Module`, con sus capas (por ejemplo, `nn.Linear`) declaradas en `__init__` y la lógica del forward pass en el método `forward`

## Ponlo en práctica

Abre un notebook nuevo en Google Colab e importa `torch`. Crea dos tensores de una dimensión con tres valores cada uno, uno con `torch.tensor` y valores de tu elección y otro con `torch.randn((3,))` (valores aleatorios). Multiplícalos elemento a elemento e imprime el resultado junto con su `shape`. Después, define una clase `RedMinima` igual a la del ejemplo de este capítulo, pero con `nn.Linear(3, 1)` sustituido por `nn.Linear(3, 5)`: comprueba con `print(modelo.capa.weight.shape)` cuántos pesos tiene ahora la capa y explica, en un comentario, por qué ese número cambió al cambiar el tamaño de salida.
