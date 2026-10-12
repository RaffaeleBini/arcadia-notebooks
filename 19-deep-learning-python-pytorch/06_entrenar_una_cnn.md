# 06. Entrenar una CNN

## Por qué importa

En el capítulo anterior construiste tu primera red neuronal convolucional, o
CNN (Convolutional Neural Network): apilaste capas `nn.Conv2d` y de pooling
para procesar imágenes de MNIST, y comprobaste cómo cambia la forma de un
tensor al pasar por cada capa. Pero te quedaste en la puerta de entrada: nunca
llegaste a entrenar esa red, así que nunca aprendió nada.

En este capítulo cierras ese círculo. Vas a reutilizar el mismo bucle de
entrenamiento que ya conoces del capítulo 4 (pérdida, optimizador, forward,
backward) y lo vas a aplicar a la arquitectura convolucional del capítulo 5.
Además, subes de nivel el reto: en vez de MNIST, en blanco y negro, vas a
entrenar tu CNN sobre CIFAR-10, un dataset de fotografías en color con
objetos reales (aviones, coches, pájaros, gatos y otros seis tipos más). Es el
salto que separa un ejercicio de laboratorio de un problema con la
complejidad de una imagen del mundo real, y es exactamente el salto que dan
casi todos los proyectos de visión por computador cuando pasan de un
prototipo a un caso de uso serio.

## El ciclo de entrenamiento no cambia: solo cambian las manos que mueves

Imagina que aprendes a lanzar dardos con los ojos vendados. Cada intento sigue
siempre la misma secuencia de cuatro pasos: lanzas el dardo (*forward*),
alguien te dice a qué distancia has caído del centro (la pérdida o *loss*),
calculas mentalmente hacia qué lado debes corregir el brazo
(*backward*, es decir, la retropropagación del error) y ajustas tu postura
para el siguiente intento (`optimizer.step()`). Repites esta secuencia una y
otra vez, en tandas llamadas épocas, hasta que aciertas con frecuencia.

Ese ciclo de cuatro pasos es idéntico para una red densa (la que entrenaste en
el capítulo 4) y para una CNN. Lo que cambia no es el ciclo, sino qué "manos"
mueves en cada lanzamiento: en la red densa, cada lanzamiento ajusta pesos
conectados directamente a cada píxel; en la CNN, cada lanzamiento ajusta los
valores de los filtros convolucionales, que se aplican de la misma manera en
toda la imagen. Por eso puedes copiar literalmente la estructura del bucle de
entrenamiento del capítulo 4 y limitarte a cambiar el modelo que le pasas.

## De blanco y negro a color: qué cambia con CIFAR-10

Piensa en la diferencia entre leer un boceto a lápiz y mirar una fotografía a
color con gente y objetos de fondo. El boceto es fácil de interpretar porque
solo tiene un tono de gris y pocos detalles; la fotografía tiene mucha más
información y mucho más ruido visual. Ese es, en esencia, el salto de MNIST a
CIFAR-10:

- MNIST son dígitos escritos a mano, en escala de grises, centrados y sobre
  fondo limpio: un tensor de entrada con forma `(lote, 1, 28, 28)`, donde el
  1 es el único canal de color
- CIFAR-10 son fotografías en color de sesenta mil objetos reales (aviones,
  coches, pájaros, gatos, ciervos, perros, ranas, caballos, barcos y
  camiones), con fondos variados y el objeto no siempre centrado: un tensor
  de entrada con forma `(lote, 3, 32, 32)`, donde el 3 corresponde a los tres
  canales de color rojo, verde y azul, o RGB (Red-Green-Blue)

Ese cambio de 1 a 3 canales de entrada es la única modificación estructural
obligatoria en la arquitectura: la primera capa `nn.Conv2d` tiene que aceptar
3 canales de entrada en lugar de 1. El resto del patrón (varios bloques de
convolución, activación ReLU y pooling, seguidos de capas totalmente
conectadas) se mantiene exactamente igual que en el capítulo 5.

Hay un detalle práctico que merece la pena recordar del material oficial:
PyTorch no incluye una función que "aplane" automáticamente la salida de las
capas convolucionales antes de pasarla a la primera capa lineal, así que
tienes que calcular tú ese tamaño (o, como sugiere el material, pasar un
tensor de prueba por la red y observar la forma que devuelve). Al añadir
`padding=1` en cada convolución de 3x3, el ancho y el alto de la imagen solo
se reducen en las capas de pooling, lo que hace ese cálculo mucho más
predecible: partiendo de 32x32, tres bloques de pooling de tamaño 2 la dejan
en 4x4.

## Por qué una CNN suele superar a una red densa en imágenes

Supón que quisieras resolver CIFAR-10 con la misma red densa del capítulo 4,
sin ninguna capa convolucional. Antes de nada, tendrías que aplanar cada
imagen a un vector de 3.072 valores (32 × 32 píxeles × 3 canales). Si la
primera capa oculta de esa red densa tuviera, por ejemplo, 256 neuronas, solo
esa capa necesitaría aprender 3.072 × 256 ≈ 786.000 pesos, uno por cada
combinación posible de píxel de entrada y neurona de salida. Y aun así, esa
red no tendría ninguna noción de que un píxel está "al lado" de otro: para
ella, mover un gato tres píxeles a la derecha en la imagen produce un patrón
de entrada completamente distinto, que tendría que aprender por separado.

Una capa convolucional resuelve ese problema de otra manera: en vez de un peso
por cada combinación de píxel y neurona, aprende un número pequeño de filtros
(por ejemplo, 32 filtros de 3x3x3) y los reutiliza en cada posición de la
imagen. Esos mismos 32 filtros de 3x3 sobre 3 canales solo necesitan
32 × (3 × 3 × 3 + 1) ≈ 900 parámetros, y al aplicarse en cada posición de la
imagen, la red aprende a reconocer un rasgo (un borde, una textura, una
mancha de color) sin importar en qué zona de la imagen aparezca. A esa
propiedad se le llama invarianza a la traslación, y es la razón principal por
la que, entrenando el mismo número de épocas y sobre el mismo dataset de
imágenes, una CNN suele alcanzar mayor precisión que una red densa
equivalente, con muchos menos parámetros que ajustar y, por tanto, menos
riesgo de sobreajuste (*overfitting*).

## Ejemplo de código comentado

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Preparamos los datos: convertimos las imágenes a tensor y normalizamos
#    cada canal de color al rango aproximado [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 2. Definimos la arquitectura: mismo patrón que en el capítulo 5
#    (convolución + ReLU + pooling), pero con 3 canales de entrada
class ConvNetCIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 16x16 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 8x8 -> 8x8
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)  # 10 clases de CIFAR-10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = ConvNetCIFAR()

# 3. Función de pérdida y optimizador: igual que en el capítulo 4,
#    pero con CrossEntropyLoss porque son 10 clases (clasificación multiclase)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Bucle de entrenamiento: forward, loss, backward, step, repetido por épocas
epocas = 5
for epoca in range(epocas):
    model.train()
    perdida_acumulada = 0.0
    for imagenes, etiquetas in train_loader:
        optimizer.zero_grad()          # limpiamos los gradientes del paso anterior
        salidas = model(imagenes)      # forward
        loss = criterion(salidas, etiquetas)
        loss.backward()                # backward: calcula los gradientes
        optimizer.step()               # actualiza los pesos del modelo
        perdida_acumulada += loss.item()

    perdida_media = perdida_acumulada / len(train_loader)
    print(f"Época {epoca + 1}/{epocas} - pérdida media: {perdida_media:.4f}")

# 5. Evaluación final sobre el conjunto de prueba (imágenes nunca vistas)
model.eval()
correctas = 0
total = 0
with torch.no_grad():
    for imagenes, etiquetas in test_loader:
        salidas = model(imagenes)
        predicciones = torch.argmax(salidas, dim=1)
        correctas += (predicciones == etiquetas).sum().item()
        total += etiquetas.size(0)

print(f"Exactitud en el conjunto de prueba: {correctas / total:.2%}")
```

## Resumen de puntos clave

- El ciclo de entrenamiento (forward, pérdida, backward, `optimizer.step()`)
  es el mismo para una red densa y para una CNN: lo único que cambia es el
  modelo que le pasas, no la estructura del bucle
- CIFAR-10 introduce dos cambios frente a MNIST: las imágenes tienen 3 canales
  de color en lugar de uno, con forma `(lote, 3, 32, 32)`, y los objetos son
  fotografías reales con fondo y variabilidad, no dígitos limpios y centrados
- El único cambio estructural obligatorio en la arquitectura es que la
  primera capa `nn.Conv2d` debe aceptar 3 canales de entrada en vez de 1
- PyTorch no aplana automáticamente la salida de las capas convolucionales:
  tienes que calcular el tamaño de entrada de la primera capa `nn.Linear` a
  mano (o comprobarlo pasando un tensor de prueba por la red)
- Una CNN comparte los mismos filtros en toda la imagen, así que necesita
  muchos menos parámetros que una red densa equivalente y suele alcanzar
  mayor precisión sobre imágenes, gracias a la invarianza a la traslación
- `nn.CrossEntropyLoss` es la función de pérdida habitual para clasificación
  multiclase, y combina internamente softmax con la entropía cruzada

## Ponlo en práctica

Abre el notebook de teoría de esta clase, entrena la CNN de este capítulo
sobre CIFAR-10 durante cinco épocas y anota la exactitud final sobre el
conjunto de prueba. Calcula a mano (con lápiz y papel, o con una calculadora)
cuántos parámetros tendría solo la primera capa de una red densa equivalente
que recibiera las 3.072 entradas de una imagen de CIFAR-10 y las conectara a
una capa oculta de 256 neuronas. Compara esa cifra con el número de
parámetros de la primera capa `nn.Conv2d` de tu CNN (puedes obtenerlo con
`sum(p.numel() for p in model.conv1.parameters())`). Guarda ambas cifras: te
van a servir de referencia en el notebook de práctica, donde vas a completar
tú mismo el bucle de entrenamiento y el cálculo de la exactitud.
