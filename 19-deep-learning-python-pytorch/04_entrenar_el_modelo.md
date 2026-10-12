# 04. Entrenar el modelo

## Por qué importa este tema

En el capítulo anterior construiste una red neuronal con `nn.Module`: capas `Linear`, activaciones `ReLU`, una entrada de 784 valores (los píxeles de una imagen de 28x28 de MNIST) y una salida de diez clases (los dígitos del cero al nueve). Si en este punto le pasas una imagen a esa red, obtienes una predicción... pero una predicción inútil, porque los pesos de las capas son números aleatorios. La red todavía no ha visto un solo ejemplo.

Entrenar es precisamente el proceso que convierte esos pesos aleatorios en pesos útiles. Es el capítulo donde tu red pasa de no saber nada a reconocer dígitos escritos a mano con una precisión razonable, y donde por fin entra en juego la parte de PyTorch que hace posible todo el aprendizaje profundo (Deep Learning, DL): el cálculo automático de gradientes.

## El ciclo de entrenamiento: forward, loss, backward, optimizer.step

Piensa en entrenar una red neuronal (Neural Network, NN) como ajustar la temperatura de un grifo con dos mandos, uno de agua fría y otro de caliente, sin mirar los números: abres el grifo, tocas el agua y notas cuánto te alejas de la temperatura ideal. Si está muy fría, giras el mando caliente; si está muy caliente, lo cierras un poco. Repites el gesto una y otra vez, con ajustes cada vez más finos, hasta acercarte a la temperatura que buscas.

El entrenamiento de una red sigue exactamente esa lógica, repetida miles de veces y con muchísimos más "mandos" (los pesos de la red). Cada vuelta del ciclo tiene cuatro pasos:

1. **Forward**: pasas un lote (batch) de datos por la red y obtienes una predicción. Esto ya lo viste en el capítulo anterior
2. **Loss**: comparas esa predicción con la etiqueta real y obtienes un número que mide cuánto se equivocó la red
3. **Backward**: le pides a PyTorch que calcule, para cada peso de la red, en qué dirección y con qué intensidad debería cambiar para que ese error sea un poco menor la próxima vez
4. **Optimizer.step**: aplicas esos cambios y actualizas los pesos

Vuelves a empezar con el siguiente lote de datos, y así sucesivamente. La pérdida (loss) no baja a cero de golpe, baja poco a poco, lote a lote, igual que la temperatura del grifo se acerca a la ideal con varios toques del mando en lugar de uno solo.

## Qué mide una función de pérdida

La función de pérdida (loss function) es la pieza que traduce "cuánto se equivocó la red" en un número concreto que PyTorch puede usar para corregirla. Cuanto más se aleje la predicción del valor real, mayor es la pérdida; si la predicción fuera perfecta, la pérdida sería cero. Incluso cuando la red acierta la clase, casi siempre queda algo de error, porque la predicción es una distribución de probabilidades y no una certeza absoluta.

PyTorch ofrece varias funciones de pérdida según el tipo de problema:

- Para regresión (predecir un número continuo), se usa el error cuadrático medio (Mean Squared Error, MSE), que mide el promedio de las diferencias al cuadrado entre la predicción y el valor real
- Para clasificación (predecir una categoría entre varias, como en MNIST), se usa la entropía cruzada (cross-entropy), disponible en PyTorch como `nn.CrossEntropyLoss`

Como tu red de clasificación de MNIST devuelve diez valores en bruto (uno por dígito, sin ninguna activación final), `nn.CrossEntropyLoss` es la función adecuada: internamente convierte esos valores en una distribución de probabilidades y compara esa distribución con la etiqueta correcta.

## Qué hace un optimizador

Si la función de pérdida te dice cuánto te equivocaste, el optimizador es quien decide qué hacer con esa información: ajusta los pesos de la red a partir de la pérdida y los gradientes, con el objetivo de que la pérdida baje un poco más en la siguiente vuelta del ciclo.

Dos optimizadores habituales en PyTorch, ambos disponibles en `torch.optim`, son:

- **SGD** (Stochastic Gradient Descent, descenso de gradiente estocástico): mueve cada peso en la dirección que más reduce la pérdida, con un tamaño de paso fijo
- **Adam** (Adaptive Moment Estimation, estimación adaptativa de momentos): ajusta ese tamaño de paso de forma automática para cada peso, lo que suele traducirse en una convergencia más rápida y estable con menos ajuste manual

Ese "tamaño de paso" tiene nombre propio: la tasa de aprendizaje (learning rate), casi siempre representada como una potencia negativa de diez, como 1e-3 (que significa lo mismo que 0,001). La tasa de aprendizaje es uno de los hiperparámetros más importantes que vas a ajustar en tu carrera con deep learning:

- Si es demasiado alta, el optimizador da pasos tan grandes que se pasa de largo el punto óptimo, y la pérdida puede oscilar sin converger nunca
- Si es demasiado baja, el optimizador avanza tan despacio que el entrenamiento tarda muchísimo, o se queda atascado antes de llegar a un buen resultado

No existe una tasa de aprendizaje universal: depende del problema, de los datos y del optimizador que elijas. Un valor de partida razonable con Adam suele ser 0,001; con la práctica desarrollarás intuición sobre cuándo subirlo o bajarlo.

## Qué es una época (epoch)

Una **época** es una pasada completa por todo el conjunto de datos de entrenamiento. Entrenar una sola época casi nunca es suficiente: la red apenas ha visto cada ejemplo una vez, así que necesita varias épocas para refinar sus pesos poco a poco y aprender patrones generales en lugar de memorizar ejemplos concretos. Por eso el bucle de entrenamiento tiene siempre dos niveles: un bucle exterior que recorre las épocas y un bucle interior que recorre los lotes (batches) de datos dentro de cada época.

## Ejemplo de código: entrenar la red de MNIST

Retoma la red que definiste en el capítulo anterior (una entrada de 784 valores, dos capas ocultas con activación `ReLU` y una salida de diez clases) y complétala con función de pérdida, optimizador y bucle de entrenamiento:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# La misma arquitectura de la clase anterior: tres capas Linear con ReLU entre medias
class RedMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.capa1 = nn.Linear(28 * 28, 128)
        self.capa2 = nn.Linear(128, 64)
        self.capa3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)      # aplana cada imagen 28x28 en un vector de 784 valores
        x = F.relu(self.capa1(x))
        x = F.relu(self.capa2(x))
        x = self.capa3(x)               # logits en bruto: CrossEntropyLoss se encarga del resto
        return x

red = RedMNIST()

# Función de pérdida para clasificación con varias clases
funcion_perdida = nn.CrossEntropyLoss()

# Optimizador: ajusta los pesos de la red (red.parameters()) según la tasa de aprendizaje indicada
optimizador = optim.Adam(red.parameters(), lr=0.001)

epocas = 5
for epoca in range(epocas):
    perdida_epoca = 0.0
    for imagenes, etiquetas in cargador_entrenamiento:  # cargador_entrenamiento es un DataLoader (clase 2)
        optimizador.zero_grad()               # reinicia los gradientes acumulados del paso anterior
        salida = red(imagenes)                 # forward: la red genera una predicción
        perdida = funcion_perdida(salida, etiquetas)  # loss: cuánto se equivocó la red
        perdida.backward()                     # backward: calcula el gradiente de cada peso
        optimizador.step()                     # optimizer.step: actualiza los pesos
        perdida_epoca += perdida.item()

    print(f"Época {epoca + 1}/{epocas} - pérdida media: {perdida_epoca / len(cargador_entrenamiento):.4f}")
```

Fíjate en que cada uno de los cuatro pasos del ciclo aparece como una única línea de código: `zero_grad()`, `red(imagenes)`, `perdida.backward()` y `optimizador.step()`. Todo lo demás es contabilidad para poder imprimir la pérdida media de cada época y comprobar que baja conforme avanza el entrenamiento.

`optimizador.zero_grad()` es un paso fácil de olvidar y muy fácil de explicar: PyTorch acumula gradientes por defecto en lugar de sobrescribirlos, así que si no los reinicias antes de cada `backward()`, vas sumando gradientes de lotes anteriores y el entrenamiento deja de tener sentido.

## Evaluar el modelo: precisión sobre datos que no ha visto

Una pérdida que baja es una buena señal, pero lo que de verdad te interesa es si el modelo clasifica bien imágenes nuevas, las que no usaste para entrenar. Para eso usas el conjunto de prueba (test) y el gestor de contexto `torch.no_grad()`, que le dice a PyTorch que no necesita calcular gradientes porque no vas a entrenar en este bloque, solo a medir:

```python
correctos = 0
total = 0

with torch.no_grad():
    for imagenes, etiquetas in cargador_prueba:
        salida = red(imagenes)
        predicciones = torch.argmax(salida, dim=1)   # la clase con mayor puntuación para cada imagen
        correctos += (predicciones == etiquetas).sum().item()
        total += etiquetas.size(0)

precision = correctos / total
print(f"Precisión sobre el conjunto de prueba: {precision:.1%}")
```

`torch.argmax` se queda con el índice del valor más alto de los diez que devuelve la red para cada imagen, es decir, el dígito que la red considera más probable. Comparar esa predicción con la etiqueta real, sumar los aciertos y dividir entre el total te da la precisión (accuracy): la proporción de imágenes que la red clasificó correctamente.

## Resumen de puntos clave

- El ciclo de entrenamiento tiene cuatro pasos que se repiten lote a lote: forward (predicción), loss (medir el error), backward (calcular gradientes) y optimizer.step (actualizar pesos)
- Una función de pérdida como `nn.CrossEntropyLoss` mide cuánto se aleja la predicción de la etiqueta real; en clasificación, cuanto más baja, mejor
- Un optimizador como `SGD` o `Adam` decide cómo ajustar los pesos a partir del error y los gradientes; la tasa de aprendizaje controla el tamaño de esos ajustes
- Una tasa de aprendizaje demasiado alta puede hacer que el entrenamiento no converja; una demasiado baja lo hace innecesariamente lento
- Una época es una pasada completa por todo el conjunto de entrenamiento; se necesitan varias para que la red generalice en lugar de memorizar
- `optimizador.zero_grad()` antes de cada `backward()` es imprescindible porque PyTorch acumula gradientes por defecto
- La precisión se mide sobre datos que el modelo no vio durante el entrenamiento, dentro de un bloque `torch.no_grad()`

## Ponlo en práctica

Abre un notebook nuevo en Google Colab, carga MNIST con `torchvision` como hiciste en la clase 2 y reutiliza la red de la clase 3. Entrénala durante tres épocas con `nn.CrossEntropyLoss` y `optim.SGD` (en lugar de Adam) usando una tasa de aprendizaje de 0,01, imprimiendo la pérdida media de cada época. Cuando termine, calcula la precisión sobre el conjunto de prueba y compárala con la que obtendrías usando Adam con una tasa de aprendizaje de 0,001: ¿qué optimizador llega antes a una pérdida baja con el mismo número de épocas?
