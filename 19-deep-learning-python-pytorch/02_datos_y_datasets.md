# 02. Datos y datasets en PyTorch

## Por qué importa este tema

En el capítulo anterior construiste tu primera red neuronal mínima con `nn.Module` y viste cómo PyTorch calcula gradientes de forma automática. Esa red, sin embargo, no sirve de nada si no le llegan datos con los que aprender. En Deep Learning (DL, aprendizaje profundo), el entrenamiento de un modelo depende por completo de la calidad y la organización de los datos que le proporcionas: acertar con la arquitectura de la red es solo una parte del trabajo, y muchas veces ni siquiera la más larga. Preparar, cargar e iterar sobre los datos suele ocupar la mayor parte del tiempo real de cualquier proyecto de Inteligencia Artificial (IA).

Antes de entrenar nada en las próximas clases, necesitas saber cómo PyTorch representa un conjunto de datos, de qué forma se lo sirve al modelo en porciones manejables y qué herramientas trae de fábrica para no tener que reinventar todo esto cada vez que empiezas un proyecto nuevo.

## Los datos como punto de partida

Piensa en una red neuronal como una persona que aprende un oficio a base de ver ejemplos, no de leer un manual de instrucciones. Cuanto más variados y mejor organizados estén esos ejemplos, mejor aprenderá el oficio. Si siempre le muestras los mismos tres casos, o en idéntico orden cada vez, aprenderá trucos específicos de esas situaciones en vez de principios generales que le sirvan ante datos nuevos.

Por eso, antes de escribir una sola línea de arquitectura de red, tienes que resolver tres preguntas sobre tus datos:

- ¿De dónde vienen y en qué formato están?
- ¿Cómo los divides en un conjunto para entrenar y otro para comprobar si el modelo generaliza?
- ¿Cómo se los entregas al modelo durante el entrenamiento, sin que eso se convierta en un cuello de botella?

PyTorch responde a estas tres preguntas con dos piezas que trabajan juntas: `Dataset` y `DataLoader`.

## Qué es la clase Dataset

Un `Dataset` es, ante todo, un contrato: cualquier objeto que lo cumple sabe responder a dos preguntas muy concretas. La primera es cuántos elementos tiene en total (`__len__`); la segunda es cómo se ve el ejemplo número *i* (`__getitem__`). Da igual si por debajo hay diez números en una lista, cien mil imágenes en disco o un archivo de audio: mientras el objeto sepa responder esas dos preguntas, PyTorch puede tratarlo como un dataset.

Esa simplicidad es intencionada. Puedes crear tu propio `Dataset` heredando de `torch.utils.data.Dataset` e implementando esos dos métodos con tu propia lógica de carga, o recurrir a uno de los que ya vienen listos para usar en `torchvision`, la biblioteca de PyTorch centrada en visión por computador. En clases posteriores construirás tus propios datasets personalizados; en esta clase empiezas con uno ya hecho, para centrarte en el mecanismo antes que en los detalles de tus propios datos.

## Qué aporta torchvision

`torchvision` es un módulo de PyTorch pensado sobre todo para tareas de visión por computador. Su aportación principal para esta clase es doble:

- Trae **datasets de referencia** ya empaquetados como clases `Dataset`, entre ellos MNIST (dígitos escritos a mano), CIFAR-10 (imágenes en color de diez categorías) e ImageNet (millones de imágenes con miles de categorías). Usarlos es casi "hacer trampa": te ahorra por completo el trabajo de buscar, descargar y dar formato a los datos, algo que en un proyecto real puede llevarte semanas
- Trae **transformaciones** listas para usar, en el módulo `torchvision.transforms`, que convierten los datos crudos en el formato que espera un modelo: convertir una imagen en tensor, normalizarla, recortarla o rotarla, entre otras muchas opciones

Usar estos datasets de referencia al empezar a aprender tiene sentido, pero no representa el caso habitual: en un proyecto real casi siempre trabajarás con datos propios de tu organización o tu sector, y ahí tendrás que construir tu propio `Dataset`. Volverás sobre esto en clases posteriores.

## Qué es DataLoader y por qué se entrena en lotes

Tener los datos organizados como un `Dataset` no basta: durante el entrenamiento necesitas ir sirviéndoselos al modelo de forma práctica. Aquí entra en juego `torch.utils.data.DataLoader`.

Imagina que tienes que repartir varios miles de cartas para un evento. No las metes todas en un solo sobre gigante ni las entregas de una en una: las agrupas en paquetes de tamaño manejable. Un `DataLoader` hace exactamente eso con tus datos: los agrupa en **lotes** (*batches*) de un tamaño que tú decides (`batch_size`), y te permite iterar sobre ellos con un simple bucle `for`.

Entrenar por lotes, en lugar de con todo el dataset de golpe o con un ejemplo cada vez, tiene varias ventajas:

- Un dataset grande puede no entrar entero en la memoria del ordenador; un lote sí
- El modelo actualiza sus parámetros varias veces por cada paso completo (*época*) a través del dataset, en vez de una sola vez, lo que suele acelerar el aprendizaje
- Un tamaño de lote habitual está entre ocho y 64 ejemplos, a menudo en potencias de dos como 8, 16, 32 o 64, aunque el número óptimo depende de tu problema concreto

Además, `DataLoader` acepta el parámetro `shuffle`, que mezcla el orden de los ejemplos antes de cada época. Esto es importante porque un dataset ordenado (por ejemplo, todos los "cero" antes que los "unos") puede hacer que el modelo aprenda patrones ligados a esa posición en vez de a las características reales de cada dígito. Mezclar los datos ayuda a que aprenda principios generales.

Por último, conviene que compruebes si tu dataset está **equilibrado**: es decir, que cada categoría tenga un número parecido de ejemplos. Si el 60% de tus datos pertenece a una sola categoría, el modelo puede aprender el atajo de predecir siempre esa clase para minimizar el error, en vez de aprender a distinguir de verdad entre las demás. Detectar este desequilibrio es tan sencillo como contar cuántos ejemplos hay de cada clase antes de empezar a entrenar.

## Ejemplo de código comentado

El siguiente ejemplo carga el dataset MNIST con `torchvision`, lo convierte a tensores con una transformación básica y lo envuelve en un `DataLoader` para iterar sobre un lote.

```python
import torch
from torchvision import datasets, transforms

# Transformación básica: convierte cada imagen (en formato PIL) en un tensor de PyTorch
transformacion = transforms.ToTensor()

# Dataset de entrenamiento de MNIST: descarga los datos si no existen en "./datos"
trainset = datasets.MNIST(
    root="./datos",
    train=True,
    download=True,
    transform=transformacion,
)

# Dataset de prueba (test): mismo origen, pero con la parte reservada para evaluar el modelo
testset = datasets.MNIST(
    root="./datos",
    train=False,
    download=True,
    transform=transformacion,
)

# DataLoader: sirve el dataset de entrenamiento en lotes de 10 ejemplos, mezclados en cada época
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)

# El DataLoader de test no necesita mezclar los datos: solo lo usarás para evaluar
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

# Iteramos un único lote para inspeccionarlo
for datos in trainloader:
    X, y = datos  # X: imágenes del lote: y: etiquetas (el dígito de cada imagen)
    print("Forma del lote de imágenes:", X.shape)  # torch.Size([10, 1, 28, 28])
    print("Forma del lote de etiquetas:", y.shape)  # torch.Size([10])
    break  # solo queremos ver el primer lote, no recorrer todo el dataset
```

Fíjate en la forma del lote de imágenes: `[10, 1, 28, 28]`. Esos cuatro números son, en orden, el tamaño del lote (diez imágenes), el número de canales de color (uno, porque MNIST es en escala de grises), y el alto y el ancho de cada imagen en píxeles (28 por 28). Cuando trabajes con imágenes en color, como en CIFAR-10, verás tres canales en lugar de uno.

## Resumen de puntos clave

- Los datos son la base del entrenamiento: prepararlos y organizarlos suele llevar más tiempo que definir la arquitectura de la red
- Un `Dataset` es cualquier objeto que sabe cuántos elementos tiene (`__len__`) y cómo devolver el elemento número *i* (`__getitem__`)
- `torchvision` ofrece datasets de referencia ya listos (MNIST, CIFAR-10, ImageNet) y transformaciones (`torchvision.transforms`) para dar a los datos el formato que el modelo necesita
- Un `DataLoader` sirve los datos en lotes (*batches*), permite mezclarlos con `shuffle` y evita tener que cargar el dataset entero en memoria de golpe
- Mezclar los datos (`shuffle=True`) evita que el modelo aprenda patrones ligados al orden en el que ve los ejemplos
- Un dataset desequilibrado, donde una clase tiene muchos más ejemplos que las demás, puede hacer que el modelo aprenda atajos en lugar de patrones reales

## Ponlo en práctica

Abre un notebook nuevo en Google Colab y carga el dataset CIFAR-10 con `datasets.CIFAR10` (en vez de MNIST), aplicando la misma transformación `transforms.ToTensor()`. Crea un `DataLoader` con un tamaño de lote de 16 y `shuffle=True`, itera el primer lote y compara la forma del tensor de imágenes con la que obtuviste para MNIST en este capítulo. Anota cuántos canales de color tiene cada imagen y por qué crees que es distinto al caso de los dígitos escritos a mano.
