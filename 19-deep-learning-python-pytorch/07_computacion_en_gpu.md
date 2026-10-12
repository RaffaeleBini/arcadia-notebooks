# 07. Computación en GPU con PyTorch

## Por qué importa este tema

En las clases cinco y seis construiste y entrenaste una red neuronal convolucional (Convolutional Neural Network, CNN) sobre CIFAR-10, un dataset de imágenes en color. Si has ejecutado ese entrenamiento completo, ya lo habrás notado: cada época tarda bastante más que con la red densa de MNIST de las clases anteriores. Las imágenes tienen más píxeles, los filtros convolucionales implican más operaciones y el modelo tiene más parámetros que ajustar.

Esto no es un problema de tu código: es un problema de hardware. Hasta ahora, todo lo que has entrenado ha corrido en la unidad central de procesamiento (Central Processing Unit, CPU) de la máquina, sea tu ordenador o la máquina virtual que te asigna Google Colab. La CPU es capaz de hacer ese trabajo, pero no está diseñada para hacerlo rápido. En este capítulo vas a entender por qué existe un tipo de hardware mucho mejor preparado para entrenar redes neuronales —la unidad de procesamiento gráfico (Graphics Processing Unit, GPU)— y, sobre todo, vas a aprender el patrón de código que PyTorch usa para aprovecharla: comprobar si hay una disponible, crear un `device` y mover a él tanto el modelo como los datos.

## CPU y GPU: dos formas distintas de calcular

Imagina una fábrica que empaqueta miles de cajas idénticas. Puedes montarla de dos formas muy distintas.

La primera es contratar a un puñado de operarios extremadamente cualificados, capaces de resolver cualquier imprevisto, cambiar de tarea en cualquier momento y tomar decisiones complejas sobre la marcha. Son pocos —entre dos y 64 en una CPU moderna— pero cada uno es muy potente y versátil. Así funciona una CPU: pocos núcleos, cada uno capaz de ejecutar instrucciones muy distintas entre sí, con lógica compleja y acceso rápido a una memoria (RAM, Random Access Memory o Memoria de Acceso Aleatorio) generosa.

La segunda forma es montar una cadena de montaje con miles de operarios mucho más simples, cada uno entrenado para repetir un único gesto sencillo sobre una caja tras otra, todos a la vez. Ninguno de ellos podría resolver un imprevisto complejo por su cuenta, pero si la tarea es "haz exactamente lo mismo con miles de cajas idénticas", esta cadena de montaje deja en ridículo a los pocos operarios cualificados en términos de velocidad total. Así funciona una GPU: miles de núcleos más simples, diseñados para ejecutar la misma operación sobre muchos datos a la vez, con memoria propia de alta velocidad (VRAM, Video RAM o Memoria de Vídeo).

Entrenar una red neuronal es, en el fondo, multiplicar matrices de números una y otra vez: miles de multiplicaciones y sumas idénticas repetidas sobre los píxeles de un lote (batch) de imágenes o sobre los pesos de una capa. Es exactamente el tipo de tarea que describe la cadena de montaje, no el taller de especialistas. Por eso una GPU puede entrenar el mismo modelo en una fracción del tiempo que necesitaría una CPU: no es que la GPU sea "más potente" en un sentido general, es que su arquitectura está hecha a medida para este tipo de cálculo paralelo.

## CUDA: el puente entre PyTorch y la GPU

Para que una biblioteca como PyTorch pueda enviarle instrucciones a una GPU, necesita un lenguaje común. Ese lenguaje es CUDA (Compute Unified Device Architecture), una plataforma de cómputo paralelo y una interfaz de programación de aplicaciones (Application Programming Interface, API) creada por NVIDIA que permite usar sus GPU para cálculo de propósito general, no solo para gráficos.

Cuando instalas PyTorch con soporte de GPU, el paquete incluye ya todo lo necesario para hablar con CUDA: no tienes que escribir código CUDA tú mismo. Google Colab, además, te ahorra incluso la instalación de los controladores y las herramientas de CUDA en la máquina: si activas el acelerador de hardware GPU desde el menú del entorno de ejecución, todo eso ya está preparado. Tu trabajo se reduce a pedirle a PyTorch, con unas pocas líneas de código, que use ese hardware.

## `torch.device` y `.to(device)`: decirle a PyTorch dónde calcular

PyTorch representa el "lugar" donde vive un tensor o un modelo con un objeto `torch.device`. Por defecto, cualquier tensor que creas se almacena en la memoria RAM de la CPU. Para que un cálculo se ejecute en la GPU, tienes que mover explícitamente ese tensor (o el modelo entero) a ella, usando el método `.to(device)`.

La forma estándar de crear ese `device` es comprobar antes si hay una GPU disponible con `torch.cuda.is_available()`, que devuelve `True` o `False` según haya o no una GPU compatible con CUDA en la máquina:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
```

Este patrón —una única línea con un operador condicional— es la pieza de código más importante de todo el capítulo. Con ella, tu código funciona igual de bien en tres situaciones distintas: en tu ordenador sin GPU, en un Colab con el acelerador de hardware en modo GPU, o en un servidor con varias GPU. Nunca tienes que escribir dos versiones distintas de tu código de entrenamiento: escribes una sola vez pensando en `device`, y `device` decide por ti.

Una vez tienes el `device`, mueves a él tanto el modelo como cualquier tensor de datos que vaya a interactuar con ese modelo:

```python
modelo = MiRed().to(device)
entrada = entrada.to(device)
etiquetas = etiquetas.to(device)
```

Hay una regla que no admite excepciones: todos los tensores que participan en una misma operación tienen que estar en el mismo `device`. Si intentas sumar o multiplicar un tensor que está en la CPU con otro que está en la GPU, PyTorch no hace la conversión por ti; lanza un error en tiempo de ejecución del tipo "expected device cpu but got device cuda". Esto no es una limitación torpe de PyTorch: es una decisión de diseño deliberada. Mover datos entre la memoria de la CPU y la memoria de la GPU tiene un coste real —viajan por un bus físico más lento que la memoria interna de cualquiera de los dos— así que PyTorch prefiere avisarte con un error explícito a que tu código haga transferencias de datos costosas sin que te enteres.

## Ejemplo de código comentado: entrenar la misma CNN en el device disponible

Retoma la CNN de CIFAR-10 de las clases cinco y seis y añade el patrón de `device` al bucle de entrenamiento. El código siguiente comprueba el hardware disponible, mueve el modelo una sola vez fuera del bucle, mueve cada lote de datos dentro del bucle y mide cuánto tarda una época completa:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# 1. El device es un interruptor: decide, con una sola línea, dónde se calcula todo lo demás
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# 2. La misma arquitectura convolucional de las clases 5 y 6, simplificada,
#    pensada para CIFAR-10: imágenes en color de 32x32 píxeles y diez clases
class CNNCifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = x.view(x.size(0), -1)              # aplana el mapa de activaciones
        x = F.relu(self.fc1(x))
        return self.fc2(x)                      # logits en bruto para las diez clases

# 3. Mueve el modelo al device UNA sola vez, antes de crear el optimizador
modelo = CNNCifar().to(device)
funcion_perdida = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

# 4. Bucle de entrenamiento de una época: cada lote viaja al device antes del forward
inicio = time.time()
for imagenes, etiquetas in cargador_entrenamiento:      # cargador_entrenamiento es un DataLoader (clase 2)
    imagenes = imagenes.to(device)                       # el lote de imágenes viaja al device
    etiquetas = etiquetas.to(device)                      # las etiquetas también, o CrossEntropyLoss fallará

    optimizador.zero_grad()
    salida = modelo(imagenes)
    perdida = funcion_perdida(salida, etiquetas)
    perdida.backward()
    optimizador.step()

tiempo_epoca = time.time() - inicio
print(f"Tiempo de una época en {device}: {tiempo_epoca:.2f} segundos")
```

Fíjate en dos detalles del código. Primero, el modelo se mueve al `device` una sola vez, antes de entrar en el bucle: no tiene sentido moverlo en cada iteración, porque sus parámetros ya se quedan alojados en la memoria del `device` hasta que decidas lo contrario. Segundo, cada lote de datos —tanto las imágenes como las etiquetas— se mueve al `device` dentro del bucle, porque el `DataLoader` entrega lotes nuevos en cada iteración y esos lotes nacen siempre en la CPU. Si comentas la línea de `torch.device` y forzaras `device = torch.device("cpu")`, el mismo código seguiría funcionando exactamente igual, solo que más despacio: esa es la prueba de que el patrón está bien escrito. Con una GPU activada, el bloque de código no cambia una sola línea; solo cambia el tiempo que tarda en ejecutarse.

## Buenas prácticas: minimiza las transferencias, no el hardware

Mover datos entre la CPU y la GPU no es gratis, así que conviene tener presentes algunas recomendaciones:

- Mueve el modelo al `device` una sola vez, fuera del bucle de entrenamiento, no en cada época ni en cada lote
- Mueve cada lote de datos justo antes de usarlo, nunca el dataset completo de golpe si no cabe en la memoria de la GPU: por eso trabajas por lotes en lugar de con todo el conjunto a la vez
- Comprueba con `torch.cuda.device_count()` cuántas GPU ve PyTorch si trabajas en una máquina con varias, para decidir si necesitas repartir el modelo entre ellas
- No asumas que la GPU es imprescindible en todo momento: durante el entrenamiento, con lotes grandes, la GPU marca una diferencia enorme; durante la puesta en producción de un modelo ya entrenado, si el número de consultas por minuto es bajo, la CPU puede ser perfectamente suficiente y te ahorra la complejidad de gestionar hardware especializado

## Resumen de puntos clave

- Una CPU tiene pocos núcleos muy versátiles, optimizados para tareas secuenciales complejas; una GPU tiene miles de núcleos más simples, optimizados para repetir la misma operación sobre muchos datos en paralelo, que es justo lo que necesita el entrenamiento de una red neuronal
- CUDA es la plataforma de NVIDIA que permite a PyTorch enviar instrucciones a la GPU; con PyTorch no escribes código CUDA directamente, la biblioteca ya lo gestiona por ti
- `torch.cuda.is_available()` comprueba si hay una GPU compatible disponible; el patrón `torch.device("cuda" if torch.cuda.is_available() else "cpu")` construye un `device` que funciona igual con o sin GPU
- `.to(device)` mueve tensores y modelos al dispositivo elegido; todos los tensores que participan en la misma operación deben estar en el mismo `device`, o PyTorch lanza un error
- Mueve el modelo al `device` una sola vez, fuera del bucle de entrenamiento, y cada lote de datos dentro del bucle, justo antes de usarlo
- La GPU acelera sobre todo el entrenamiento con lotes grandes; en producción, con poca carga, la CPU puede bastar

## Ponlo en práctica

Abre el notebook de teoría de esta clase en Google Colab y ejecuta la celda que crea el `device` con el entorno de ejecución en modo "Ninguno" (solo CPU): confirma que imprime `cpu`. Después, cambia el acelerador de hardware a GPU desde **Entorno de ejecución > Cambiar tipo de entorno de ejecución**, vuelve a ejecutar la misma celda y comprueba que ahora imprime `cuda`. Por último, sin mirar el código de este capítulo, escribe de memoria las tres líneas mínimas que necesitas para: comprobar si hay GPU disponible, crear el `device` correspondiente y mover un modelo cualquiera a ese `device`.
