# 05. Introducción a las redes neuronales convolucionales

## Por qué importa

En el capítulo anterior construiste y entrenaste una red densa con `nn.Linear`
para clasificar los dígitos de MNIST. Para que esa red pudiera procesarlas,
tuviste que aplanar cada imagen de 28x28 píxeles en un único vector de 784
valores. Esa decisión tiene un coste: al aplanar la imagen, rompes toda la
información sobre qué píxel está al lado de cuál. La red densa recibe una
lista de números sin ninguna noción de "arriba", "abajo" o "vecino", así que
tiene que aprender desde cero, a fuerza de ejemplos, que ciertos píxeles
suelen aparecer juntos formando un trazo o una curva.

Las redes neuronales convolucionales, o CNN (Convolutional Neural Network),
resuelven justo ese problema. En lugar de aplanar la imagen, mantienen su
estructura de cuadrícula y usan capas especializadas que sí saben aprovechar
la posición relativa de los píxeles. El resultado son modelos que aprenden
más rápido, con menos parámetros y que generalizan mejor a variaciones de la
misma imagen (un dígito ligeramente desplazado, por ejemplo). Por eso las CNN
son la base de prácticamente cualquier sistema moderno de visión por
computador, desde el reconocimiento de escritura hasta la conducción
autónoma.

En esta clase no vas a entrenar todavía la red: eso lo harás en la clase 6.
Aquí te centras en entender y construir la arquitectura, y en comprobar que
el forward pass produce las formas de tensor que esperas en cada paso.

## Qué resuelve una convolución frente a una capa densa

Piensa en cómo reconoces un dígito escrito a mano. No memorizas la imagen
píxel a píxel: te fijas en trazos locales (una curva aquí, una línea recta
allí) y combinas esas pistas para decidir si es un "3" o un "8". Una capa
densa no puede hacer eso de forma natural, porque cada una de sus neuronas
mira **todos** los píxeles de la imagen a la vez, con un peso distinto para
cada uno. Si el mismo trazo aparece un poco más arriba o más a la derecha en
otra imagen, la neurona tiene que aprender ese patrón otra vez, con pesos
nuevos, porque no reconoce que es "el mismo trazo en otro sitio".

Una capa convolucional funciona de otra manera: en vez de mirar toda la
imagen de golpe, recorre la imagen con una ventana pequeña, aplicando
**la misma operación** en cada posición. Esto tiene dos consecuencias
importantes:

- La capa detecta un patrón (por ejemplo, un borde) sin importar en qué
  parte de la imagen aparezca, porque examina la imagen entera con la misma
  ventana
- La capa necesita muchos menos parámetros que una capa densa equivalente,
  porque la misma ventana de pesos se reutiliza en todas las posiciones, en
  lugar de tener un peso distinto para cada píxel de la imagen

A esta reutilización de los mismos pesos en toda la imagen se le llama
*weight sharing* (compartición de pesos), y es la razón principal por la que
las CNN escalan mejor a imágenes grandes que las redes densas.

## Qué es un filtro (kernel)

La "ventana" que se desliza sobre la imagen se llama **filtro**, o también
**kernel**: una pequeña matriz de números, normalmente de tres por tres
posiciones. En cada posición de la imagen, el filtro multiplica cada uno de
sus valores por el píxel que tiene debajo, suma todos los resultados y ese
número se convierte en el valor de un píxel de la imagen de salida (llamada
**mapa de características**, o *feature map*). El filtro se va desplazando
por toda la imagen, generando un valor nuevo en cada posición.

Ninguno de los números del filtro se elige a mano: se inicializan al azar y
se ajustan durante el entrenamiento mediante retropropagación
(*backpropagation*), exactamente igual que los pesos de una capa `nn.Linear`.
Al terminar el entrenamiento, cada filtro termina especializado en detectar
un tipo de rasgo concreto: los filtros de las primeras capas suelen
responder a bordes, curvas o esquinas simples, y los filtros de capas más
profundas combinan esas señales para reconocer formas más complejas, como el
lazo de un "8" o el trazo cerrado de un "0".

Una capa convolucional no aprende un solo filtro, sino varios en paralelo,
cada uno buscando un rasgo distinto. En PyTorch, una capa convolucional se
define con `nn.Conv2d`:

```python
nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
```

- `in_channels`: número de canales de la imagen de entrada (uno para escala
  de grises, como MNIST; tres para una imagen en color con canales rojo,
  verde y azul, RGB)
- `out_channels`: número de filtros distintos que la capa aprende, es decir,
  cuántos mapas de características produce
- `kernel_size`: el tamaño del filtro (aquí, 3x3)
- `padding`: cuántas filas y columnas de ceros se añaden alrededor de la
  imagen antes de aplicar el filtro; con `padding=1` y un filtro de 3x3, la
  imagen de salida conserva el mismo alto y ancho que la de entrada, en
  lugar de reducirse en cada capa

## Qué es el stride

El **stride** (paso) es la distancia, en píxeles, que el filtro avanza cada
vez que se desplaza. Con `stride=1`, el filtro se mueve una posición cada
vez y examina prácticamente todas las posiciones posibles de la imagen. Con
`stride=2`, el filtro salta de dos en dos posiciones, examina la mitad de
posiciones y produce un mapa de características más pequeño.

Piensa en el stride como la longitud de zancada al recorrer la imagen: con
zancadas cortas (stride pequeño) examinas la imagen con mucho detalle pero
tardas más y generas una salida más grande; con zancadas largas (stride
grande) recorres la imagen más rápido, a costa de examinarla con menos
detalle. En las CNN sencillas, lo habitual es dejar `stride=1` en las capas
convolucionales y delegar la reducción de tamaño en las capas de pooling,
que es justo lo que vas a ver a continuación.

## Qué hace el pooling

Después de una o varias capas convolucionales, es habitual añadir una capa
de **pooling** (agrupamiento), cuyo trabajo es reducir el tamaño de los
mapas de características conservando lo esencial de lo que los filtros ya
detectaron.

La variante más común es **max pooling**: la imagen se recorre en bloques
(por ejemplo, de 2x2) y de cada bloque solo se conserva el valor más alto.
El resultado es un mapa de características con la mitad de alto y la mitad
de ancho, pero que sigue conservando las zonas donde los filtros
respondieron con más intensidad.

Piensa en el pooling como resumir un párrafo largo en una frase: pierdes
detalle, pero conservas la idea principal. Aplicado a una imagen, el pooling
aporta dos ventajas:

- Reduce la cantidad de cálculos que necesitan las capas siguientes, porque
  trabajan sobre un mapa de características más pequeño
- Hace que la red sea más robusta a pequeños desplazamientos del trazo en la
  imagen, porque ya no importa el píxel exacto donde apareció el rasgo, solo
  que apareció dentro de esa zona

En PyTorch, el pooling se añade con `nn.MaxPool2d`:

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

Aquí `kernel_size=2` define el tamaño del bloque (2x2) y `stride=2` hace que
los bloques no se solapen entre sí, de modo que el mapa de características
resultante mide exactamente la mitad de alto y de ancho que el de entrada.

## Ejemplo de código comentado

El siguiente ejemplo define una CNN sencilla para clasificar los dígitos de
MNIST (imágenes en escala de grises de 28x28 píxeles) en diez clases, una
por cada dígito del cero al nueve. La arquitectura combina dos bloques de
convolución más pooling y termina con una capa `nn.Linear`, siguiendo el
mismo patrón que describe el material oficial: extraer características con
convoluciones antes de clasificar con una capa densa.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSimple(nn.Module):
    """CNN sencilla para clasificar MNIST (1 canal, 28x28 píxeles) en 10 clases."""

    def __init__(self):
        super().__init__()
        # Primer bloque: 1 canal de entrada (escala de grises), 16 filtros
        # de 3x3, con padding=1 para no perder tamaño en los bordes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                                stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Segundo bloque: recibe los 16 mapas de características de la capa
        # anterior y aprende 32 filtros nuevos, cada uno combinando los rasgos
        # detectados por el primer bloque
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                                stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Capa final densa: clasifica a partir de los mapas de características
        # ya reducidos por las dos capas de pooling (28 -> 14 -> 7)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)   # aplanamos justo antes de la capa densa
        x = self.fc(x)
        return x


modelo = CNNSimple()
print(modelo)

# Forward pass sobre un lote de ejemplo: 4 imágenes de MNIST simuladas
lote_ejemplo = torch.randn(4, 1, 28, 28)

x = lote_ejemplo
print('Entrada:                ', x.shape)
x = F.relu(modelo.conv1(x)); print('Tras conv1 + ReLU:       ', x.shape)
x = modelo.pool1(x);         print('Tras pool1:              ', x.shape)
x = F.relu(modelo.conv2(x)); print('Tras conv2 + ReLU:       ', x.shape)
x = modelo.pool2(x);         print('Tras pool2:              ', x.shape)
x = x.view(x.size(0), -1);   print('Tras aplanar (flatten):  ', x.shape)
x = modelo.fc(x);            print('Tras fc (salida):        ', x.shape)
```

Fíjate en dos cosas del código. Primero, en PyTorch, la forma de un lote de
imágenes sigue siempre el orden `(lote, canales, alto, ancho)`: por eso el
lote de ejemplo tiene forma `(4, 1, 28, 28)`, cuatro imágenes de un canal de
28x28 píxeles, en lugar de aplanarlas como hacías con `nn.Linear`. Segundo,
el número `32 * 7 * 7` de la capa `fc` no es arbitrario: sale de seguir la
imagen paso a paso por la red (28 píxeles de entrada, sin cambios por el
padding en cada convolución, reducida a la mitad en cada pooling: 28 → 14 →
7) y multiplicar ese tamaño final por los 32 filtros del segundo bloque
convolucional. Si cambias cualquier parámetro de la arquitectura, ese
número también cambia, y tienes que recalcularlo.

## Resumen de puntos clave

- Una capa densa mira todos los píxeles de la imagen a la vez y pierde la
  información sobre qué píxel está al lado de cuál; una capa convolucional
  recorre la imagen con una ventana pequeña que reutiliza los mismos pesos
  en toda la imagen, así que detecta un patrón sin importar dónde aparezca
- El filtro (o kernel) es una pequeña matriz de números que se multiplica
  con los píxeles que tiene debajo en cada posición; sus valores no se
  eligen a mano, se aprenden durante el entrenamiento igual que los pesos
  de una capa `nn.Linear`
- El stride controla cuántos píxeles avanza el filtro en cada paso: un
  stride mayor produce una salida más pequeña y examina la imagen con menos
  detalle
- El pooling (normalmente max pooling, con `nn.MaxPool2d`) reduce el tamaño
  de los mapas de características quedándose con el valor más alto de cada
  bloque, lo que aligera los cálculos y da robustez frente a pequeños
  desplazamientos
- Una CNN sencilla en PyTorch combina bloques de `nn.Conv2d` más pooling
  para extraer características, y termina con una o varias capas
  `nn.Linear` para clasificar, definidas dentro de una clase que hereda de
  `nn.Module`
- Antes de definir la capa `nn.Linear` final necesitas calcular a mano
  cuántas dimensiones tiene el tensor una vez aplanado, siguiendo cómo
  cambia su forma en cada capa convolucional y de pooling

## Ponlo en práctica

Abre el notebook de teoría de esta clase, ejecuta la CNN de este capítulo y
observa cómo cambia la forma del tensor después de cada capa. Después,
cambia el número de filtros de `conv1` de 16 a 8 y el de `conv2` de 32 a 16,
sin tocar nada más, y vuelve a ejecutar el forward pass. ¿Qué forma tiene
ahora la entrada de la capa `fc`? Calcula el nuevo valor a mano antes de
ejecutar el código y comprueba si coincide. Vas a necesitar ese mismo
cálculo en el notebook de práctica, donde te toca añadir una capa
convolucional más a una CNN ya empezada.
