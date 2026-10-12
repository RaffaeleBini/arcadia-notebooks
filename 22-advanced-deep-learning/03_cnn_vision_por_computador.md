# Capítulo 3. Redes neuronales convolucionales: cómo ve una máquina

*Acrónimos usados en este capítulo: red neuronal convolucional (CNN), unidad lineal rectificada (ReLU), procesador gráfico (GPU), rojo-verde-azul (RGB), descenso de gradiente estocástico (SGD).*

## Por qué importa este tema

En el capítulo anterior aprendiste qué es una neurona, cómo se organiza un perceptrón multicapa y cómo el algoritmo de retropropagación ajusta los pesos de la red para minimizar el error. Todo eso funciona muy bien cuando tus datos son filas de una tabla: edad, ingresos, número de habitaciones. Pero en cuanto intentas aplicar esa misma receta a una imagen, el enfoque empieza a hacer aguas. Y las imágenes están en todas partes: son el tipo de dato detrás de algunas de las aplicaciones de inteligencia artificial (IA) con más impacto social de la última década.

Piensa en tres ejemplos muy distintos entre sí:

- Un coche autónomo que debe distinguir en milisegundos a una persona peatona de una farola
- Un sistema de diagnóstico médico que analiza una radiografía de tórax para detectar signos tempranos de neumonía
- Una aplicación de tu móvil que reconoce automáticamente qué producto agrícola fotografías en el campo

En los tres casos, la entrada es la misma: una imagen, es decir, una malla de píxeles. Y en los tres casos, el reto es el mismo: extraer patrones visuales relevantes (bordes, texturas, formas, objetos) de esa malla, de forma que la decisión final sea fiable, rápida y generalice a imágenes nunca vistas.

Este capítulo te presenta la herramienta que resolvió ese reto y que sigue siendo, más de una década después, la base de la mayoría de sistemas de visión por computador: la red neuronal convolucional (CNN). Al terminar, entenderás por qué una red totalmente conectada no es la solución adecuada para imágenes, qué hace exactamente un filtro convolucional, cómo se controla el tamaño de sus salidas con el paso (*stride*) y el relleno (*padding*), para qué sirve el *pooling*, y cómo todo esto se combina en una arquitectura que aprende, capa a capa, desde bordes simples hasta objetos completos.

## Explicación conceptual con analogías

### El problema de "aplanar" una imagen

Imagina que tienes una fotografía de nueve por nueve píxeles en blanco y negro. Para dársela a una red totalmente conectada, como la del capítulo anterior, tendrías que convertirla en un vector: coges cada fila de píxeles y las pegas una detrás de otra, formando una única lista de ochenta y uno números. Esa operación se llama aplanar (*flatten*) la imagen.

El problema es que, al aplanar, destruyes la estructura espacial. Dos píxeles que estaban pegados uno al lado del otro en la imagen original pueden acabar muy separados en el vector, mientras que dos píxeles lejanos entre sí pueden acabar cerca. La red totalmente conectada ya no sabe qué números son vecinos en el espacio real: para ella son solo una lista de valores.

Además, aparece un problema práctico. Si conectas cada píxel de entrada con cada neurona de la primera capa oculta, el número de parámetros que debe aprender la red se dispara. Una imagen en color de 32 x 32 píxeles con tres canales de color (rojo, verde y azul, o RGB) tiene 32 x 32 x 3 = 3.072 valores de entrada. Si la primera capa oculta tiene solo 1.000 neuronas, ya necesitas 3.072 x 1.000 = 3.072.000 parámetros, solo en esa capa. Escala eso a fotografías reales de varios megapíxeles y el modelo se vuelve inmanejable: demasiados parámetros, demasiada memoria, demasiado riesgo de sobreajuste.

### La plantilla que busca un patrón

Ahora piensa en cómo reconocerías un ojo dentro de una fotografía si tuvieras que hacerlo a mano. No mirarías la imagen entera de golpe: recorrerías la foto con una especie de plantilla mental de "esto es un ojo" y comprobarías, región por región, si esa región se le parece. La plantilla es siempre la misma, la deslizas sobre toda la imagen, y allí donde encaja mejor, obtienes una respuesta fuerte.

Esa es, literalmente, la idea de un filtro (o *kernel*) convolucional. Es una pequeña matriz de números, por ejemplo de tamaño 3 x 3, que se desliza sobre la imagen. En cada posición, multiplica sus valores por los píxeles que tiene debajo y suma el resultado. Si esa región de la imagen se parece al patrón codificado en el filtro (un borde vertical, una esquina, una mancha de color), la suma da un valor alto. Si no se parece, da un valor bajo. El resultado de recorrer toda la imagen con esa plantilla es un mapa de activaciones (*feature map*): una nueva imagen que indica dónde ha encontrado el patrón.

La diferencia con una plantilla tradicional de visión por computador clásica es que aquí los valores de la plantilla no los diseña una persona a mano: los aprende la red durante el entrenamiento, exactamente igual que aprende los pesos de una capa lineal.

### Dos propiedades del mundo visual que aprovecha la CNN

Este diseño no es arbitrario: explota dos propiedades que cumplen casi todas las imágenes naturales.

- **Localidad**. Los píxeles que están cerca en la imagen suelen estar correlacionados entre sí (forman parte del mismo objeto o textura), mientras que píxeles muy alejados, como la esquina superior izquierda y la inferior derecha, normalmente no tienen relación directa
- **Invariancia a la traslación**. Un patrón significativo puede aparecer en cualquier posición de la imagen. Un pájaro sigue siendo un pájaro tanto si aparece en la esquina superior como en el centro de la foto. Esta propiedad no es exclusiva de las imágenes: en el audio, un fonema puede ocurrir en cualquier instante de tiempo; en el texto, una palabra puede aparecer en cualquier posición de la página

Una capa convolucional está diseñada exactamente para explotar estas dos propiedades: conecta cada unidad de salida solo con una región local de la entrada (localidad) y reutiliza los mismos pesos en todas las posiciones de la imagen (invariancia a la traslación). A esa reutilización de pesos se la llama **compartición de pesos** (*weight sharing*), y es la clave que explicamos en detalle a continuación.

## Desarrollo teórico

### De la capa totalmente conectada a la capa localmente conectada

Para entender de dónde sale una capa convolucional, resulta útil construirla en dos pasos partiendo de lo que ya conoces.

**Paso uno: restringir la conectividad.** En una capa totalmente conectada, cada unidad de salida está conectada a todos los píxeles de entrada. En una capa localmente conectada, cada unidad de salida se conecta solo con una pequeña región de la entrada, por ejemplo una ventana de 3 x 3 píxeles. A esa región se la llama **campo receptivo** de la unidad. Como cada unidad de salida corresponde a una región distinta de la imagen, el conjunto de salidas conserva la topología espacial de la entrada: sigue teniendo forma de malla, no de vector.

Este cambio ya reduce drásticamente el número de conexiones (y por tanto de parámetros), porque cada unidad de salida solo necesita nueve pesos (más un sesgo) en lugar de uno por cada píxel de la imagen completa.

**Paso dos: compartir los pesos.** Una capa localmente conectada todavía tiene un problema: cada unidad de salida, aunque mira una región pequeña, tiene su propio conjunto de pesos, distinto del de sus vecinas. Eso significa que la red tendría que aprender por separado a detectar "un borde vertical" en la esquina superior izquierda y "un borde vertical" en el centro de la imagen, aunque sea exactamente el mismo patrón.

La solución es forzar a que todas las unidades de salida usen el mismo conjunto de pesos. Es decir, la misma plantilla de 3 x 3 se desliza por toda la imagen, aplicando siempre los mismos números. Esta operación de "plantilla que se desliza, multiplica y suma" es, formalmente, la **convolución**, y se representa con el símbolo asterisco (*).

Gracias a la compartición de pesos, la red se vuelve **equivariante a la traslación**: si desplazas el patrón de entrada, la salida se desplaza exactamente de la misma manera, en lugar de tener que aprenderse el patrón de nuevo en cada posición.

### La operación de convolución discreta

Sea $I$ la imagen de entrada (una matriz de dos dimensiones, o de tres si cuentas el eje de canales de color) y $K$ el filtro o *kernel* (una matriz pequeña, típicamente de 3 x 3 o 5 x 5). El valor de salida en la posición $(i, j)$ del mapa de activaciones se calcula como:

$$
(I * K)(i, j) = \sum_{m} \sum_{n} I(i + m,\; j + n) \cdot K(m, n)
$$

En palabras: colocas el filtro $K$ sobre la imagen $I$, centrado (o alineado) en la posición $(i, j)$, multiplicas cada peso del filtro por el píxel de la imagen que tiene debajo, y sumas todos esos productos para obtener un único número. Después, desplazas el filtro a la siguiente posición y repites. El resultado de recorrer toda la imagen es el mapa de activaciones.

Cuando la imagen tiene varios canales de entrada (por ejemplo, los tres canales RGB), el filtro también tiene un subfiltro por canal. Las contribuciones de los tres subfiltros se suman para producir un único mapa de activaciones por filtro. Además, en una capa convolucional se aprenden varios filtros a la vez, cada uno especializado en detectar un patrón distinto (un borde horizontal, una mancha roja, una textura granulada); el conjunto de mapas de activaciones que producen se apila formando los **canales de salida**. Por eso, tanto la entrada como la salida de una capa convolucional son en realidad tensores de tres dimensiones: alto, ancho y número de canales.

Tras la convolución se aplica, igual que en una capa lineal, una función de activación no lineal, casi siempre la ReLU:

$$
\text{ReLU}(x) = \max(0, x)
$$

La ReLU deja pasar sin cambios los valores positivos (activaciones fuertes, es decir, "aquí hay un patrón parecido al filtro") y anula los negativos. Es barata de calcular y evita el problema de saturación de otras funciones como la sigmoide, lo que facilita entrenar redes más profundas.

### Tamaño de la salida: kernel, stride y padding

Un detalle importante que debes controlar es cuánto se reduce (o no) el tamaño espacial de la imagen al pasar por una capa convolucional. Esto depende de tres parámetros: el tamaño del *kernel*, el paso (*stride*) y el relleno (*padding*).

**Convolución válida (*valid*).** Es la variante más simple: solo calculas la salida en las posiciones donde el filtro cabe completamente dentro de la imagen, sin salirse. Si tienes una imagen de entrada de $N \times N$ píxeles y un filtro de $F \times F$, el filtro solo tiene $N - F + 1$ posiciones posibles por cada dimensión. Por ejemplo, con una imagen de 9 x 9 y un filtro de 3 x 3, obtienes un mapa de activaciones de 7 x 7: la salida siempre es más pequeña que la entrada.

**Relleno (*padding*) y convolución "igual" (*same*).** Si apilas varias convoluciones válidas seguidas, la imagen se va encogiendo capa tras capa, lo cual no siempre es deseable. La solución es añadir una franja de ceros alrededor del borde de la imagen antes de aplicar el filtro: eso es el *padding*. Cuando añades exactamente el relleno necesario para que la salida tenga el mismo tamaño que la entrada, hablas de convolución "igual" (*same*). Para un filtro de 3 x 3, basta con una fila de ceros de grosor uno alrededor de toda la imagen. Esta variante funciona de forma natural con filtros de tamaño impar (3 x 3, 5 x 5), porque el relleno puede repartirse de forma simétrica en ambos lados.

En el extremo contrario está la convolución "completa" (*full*), que añade el máximo relleno posible: calcula una salida en cualquier posición donde el filtro toque aunque sea un solo píxel de la imagen. El resultado es una salida más grande que la entrada ($N + F - 1$), pero se usa poco en la práctica de clasificación de imágenes.

La fórmula general que combina kernel, *padding* ($P$) y *stride* ($S$) para calcular el tamaño de salida $O$ a partir de una entrada de tamaño $N$ es:

$$
O = \left\lfloor \frac{N - F + 2P}{S} \right\rfloor + 1
$$

**Paso o *stride*.** Hasta ahora hemos asumido que el filtro se desliza de uno en un píxel. El *stride* controla ese salto: con *stride* igual a dos, el filtro avanza de dos en dos píxeles, calculando solo la mitad de posiciones en cada dimensión. Esto tiene dos efectos: reduce el coste computacional (con *stride* dos, el cálculo se reduce aproximadamente en un factor de cuatro) y reduce la resolución del mapa de activaciones, lo cual resulta útil para construir una jerarquía de características cada vez más abstractas y de menor resolución espacial. Por ejemplo, una convolución válida con *stride* dos sobre una imagen de 9 x 9 y un filtro de 3 x 3 produce un mapa de salida de 4 x 4.

Existe también la **convolución dilatada**, una variante en la que no se salta posiciones de la imagen, sino valores del propio filtro: se intercalan huecos entre los pesos del *kernel*. Esto amplía el campo receptivo (la región de la imagen original que "ve" cada unidad de salida) sin aumentar el número de parámetros del filtro, algo útil cuando quieres capturar contexto más amplio sin encarecer el cálculo.

### Pooling: reducir resolución conservando lo importante

El *pooling* es otra operación clásica para reducir el tamaño espacial de los mapas de activaciones, complementaria (o alternativa) a las convoluciones con *stride*. A diferencia de la convolución, el *pooling* no tiene parámetros que aprender: aplica una función fija (máximo o media) sobre pequeñas ventanas de la imagen.

La variante más habitual es el **max pooling**: divide el mapa de activaciones en ventanas pequeñas, normalmente de 2 x 2 y sin solaparse, y se queda con el valor máximo de cada ventana:

$$
y(i, j) = \max_{(m, n) \,\in\, \text{ventana}(i, j)} x(m, n)
$$

Con una ventana de 2 x 2 y paso dos, el alto y el ancho del mapa de activaciones se reducen a la mitad, mientras el número de canales permanece igual. Por ejemplo, un mapa de 32 x 32 x 32 pasa a 16 x 16 x 32 tras un *max pooling* de 2 x 2.

El *max pooling* aporta tres beneficios prácticos:

- Reduce el coste computacional y la memoria necesaria en las capas siguientes
- Aporta un margen de invariancia a pequeñas traslaciones: si el patrón se desplaza un píxel dentro de la ventana, el valor máximo seleccionado apenas cambia
- Al reducir la resolución, disminuye también el número de parámetros en las capas totalmente conectadas que suelen ir al final de la red, lo que ayuda a controlar el sobreajuste

Existe también el **average pooling** (calcula la media en lugar del máximo, produciendo un efecto de suavizado) y el **global pooling**, que colapsa cada mapa de activaciones completo a un único valor, muy usado justo antes de la capa de clasificación final en arquitecturas modernas.

### Cómo se construye una CNN por capas

Una CNN típica no es más que una secuencia de bloques que combinan estas piezas, apilados uno tras otro:

1. **Capas convolucionales**, cada una seguida de una no linealidad (ReLU), que extraen mapas de activaciones cada vez más ricos
2. **Capas de *pooling*** (o convoluciones con *stride*), intercaladas cada cierto número de capas convolucionales, que reducen la resolución espacial
3. Tras varias rondas de convolución y *pooling*, los mapas de activaciones finales se **aplanan** en un vector
4. Ese vector alimenta una o varias **capas totalmente conectadas** (las capas lineales que ya conoces), que realizan el razonamiento de alto nivel
5. Una última capa produce las puntuaciones de salida, típicamente pasadas por una función *softmax* para obtener una distribución de probabilidad sobre las clases

Este patrón (convolución, no linealidad, *pooling*, repetido varias veces, seguido de capas totalmente conectadas) apareció ya en una de las primeras CNN con éxito práctico, LeNet-5, diseñada a finales de los años noventa para reconocer dígitos manuscritos, y sigue siendo, en esencia, el esqueleto de arquitecturas mucho más recientes.

El entrenamiento de una CNN sigue exactamente el mismo procedimiento que aprendiste con el perceptrón multicapa: defines una función de pérdida (para clasificación, casi siempre la entropía cruzada), calculas las predicciones con una pasada hacia delante, propagas el error hacia atrás con retropropagación, y actualizas los pesos, incluidos los valores de los filtros convolucionales, mediante descenso de gradiente estocástico (SGD) o alguna de sus variantes. La diferencia no está en cómo se entrena, sino en cómo está diseñada la arquitectura que se entrena.

### Por qué las CNN capturan jerarquías espaciales

Quizás la propiedad más interesante de las CNN es que, al apilar varias capas convolucionales, la red construye de forma natural una jerarquía de características cada vez más abstractas.

Cada capa convolucional actúa, en esencia, como un clasificador lineal local. Por sí sola, detecta patrones relativamente simples. Pero cuando apilas una segunda capa convolucional sobre la salida de la primera, cada unidad de esa segunda capa no mira directamente los píxeles originales, sino los mapas de activaciones ya procesados por la primera capa. Como cada unidad de la primera capa tenía un campo receptivo de, por ejemplo, 3 x 3 píxeles, y la segunda capa vuelve a mirar una ventana de 3 x 3 sobre esas salidas, el campo receptivo efectivo de la segunda capa, medido respecto a la imagen original, crece hasta 5 x 5. Si sigues apilando capas de 3 x 3, cada nueva capa amplía el campo receptivo efectivo, permitiendo a la red "ver" regiones cada vez mayores de la imagen sin necesidad de usar filtros grandes desde el principio.

Este crecimiento del campo receptivo, combinado con la reducción de resolución del *pooling*, produce la jerarquía característica de las CNN que se observa de forma empírica al visualizar qué activa a cada capa:

- Las **primeras capas** aprenden detectores muy simples y genéricos: bordes, cambios de intensidad, manchas de color
- Las **capas intermedias** combinan esos detectores simples en patrones más elaborados: texturas, esquinas, contornos característicos
- Las **capas profundas** combinan esos patrones intermedios en representaciones de partes de objetos y, finalmente, de objetos completos: una rueda, un ojo, una cara entera

Esta composición (patrones simples que se combinan en patrones cada vez más complejos) refleja precisamente la naturaleza composicional de las imágenes del mundo real: un objeto está formado por partes, y esas partes están formadas por texturas y bordes. Apilar capas convolucionales con *pooling* es, en esencia, apilar niveles de abstracción.

Conviene subrayar que las CNN, por diseño, solo son robustas frente a traslaciones (gracias a la compartición de pesos y al *pooling*). No son automáticamente invariantes a rotaciones, cambios de escala o de iluminación. Para conseguir esa robustez adicional, se recurre en la práctica a la **aumentación de datos** (*data augmentation*): durante el entrenamiento se generan versiones rotadas, escaladas o recortadas de las mismas imágenes, de forma que la red aprenda esa invariancia a partir de ejemplos, en lugar de tenerla incorporada en la arquitectura.

## Ejemplo de código comentado, en PyTorch

El siguiente ejemplo define una CNN pequeña y la entrena sobre FashionMNIST, un conjunto de imágenes en escala de grises de 28 x 28 píxeles con diez categorías de prendas de ropa, disponible directamente en `torchvision`. **Copia el código en una celda de Google Colab y ejecútalo** para ver la red entrenarse de principio a fin.

```python
# Instala/actualiza torchvision si hace falta (normalmente ya viene en Colab)
# !pip install torch torchvision --quiet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Usa GPU si Colab te la ha asignado, si no, usa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de entrenamiento: {device}")

# --- 1. Datos ---
# Normalizamos con la media y desviación típica habituales de FashionMNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

clases = [
    "camiseta", "pantalón", "jersey", "vestido", "abrigo",
    "sandalia", "camisa", "zapatilla", "bolso", "botín",
]

# --- 2. Definición de la CNN ---
class CNNSimple(nn.Module):
    def __init__(self, num_clases=10):
        super().__init__()
        # Bloque convolucional 1: 1 canal de entrada (escala de grises) -> 16 filtros
        # kernel 3x3, padding "same" (padding=1) para no reducir el tamaño aquí
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                                kernel_size=3, stride=1, padding=1)
        # Bloque convolucional 2: 16 -> 32 filtros
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                                kernel_size=3, stride=1, padding=1)
        # Max pooling 2x2: reduce el alto y el ancho a la mitad cada vez
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Tras dos pooling de 2x2 sobre una entrada de 28x28: 28 -> 14 -> 7
        # El último mapa de activaciones tiene forma (32 canales, 7, 7)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_clases)

        self.dropout = nn.Dropout(0.25)  # mitiga el sobreajuste

    def forward(self, x):
        # x llega con forma (lote, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # -> (lote, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (lote, 32, 7, 7)

        x = torch.flatten(x, start_dim=1)     # aplana todo menos el eje de lote
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                       # logits, sin softmax explícito
        return x

modelo = CNNSimple(num_clases=10).to(device)
print(modelo)

# --- 3. Función de pérdida y optimizador ---
# CrossEntropyLoss combina softmax + entropía cruzada en un solo paso
criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

# --- 4. Bucle de entrenamiento ---
def entrenar_una_epoca(modelo, loader, criterio, optimizador):
    modelo.train()
    perdida_total = 0.0
    for imagenes, etiquetas in loader:
        imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)

        optimizador.zero_grad()
        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * imagenes.size(0)
    return perdida_total / len(loader.dataset)

# --- 5. Evaluación ---
def evaluar(modelo, loader, criterio):
    modelo.eval()
    perdida_total = 0.0
    aciertos = 0
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            salidas = modelo(imagenes)
            perdida_total += criterio(salidas, etiquetas).item() * imagenes.size(0)
            predicciones = salidas.argmax(dim=1)
            aciertos += (predicciones == etiquetas).sum().item()
    perdida_media = perdida_total / len(loader.dataset)
    exactitud = aciertos / len(loader.dataset)
    return perdida_media, exactitud

# --- 6. Entrena durante unas pocas épocas ---
num_epocas = 5
for epoca in range(1, num_epocas + 1):
    perdida_train = entrenar_una_epoca(modelo, train_loader, criterio, optimizador)
    perdida_test, exactitud_test = evaluar(modelo, test_loader, criterio)
    print(f"Época {epoca}/{num_epocas} | "
          f"pérdida train: {perdida_train:.4f} | "
          f"pérdida test: {perdida_test:.4f} | "
          f"exactitud test: {exactitud_test:.2%}")

# --- 7. Prueba rápida sobre una imagen del conjunto de test ---
modelo.eval()
imagen_ejemplo, etiqueta_real = test_dataset[0]
with torch.no_grad():
    logits = modelo(imagen_ejemplo.unsqueeze(0).to(device))  # añade eje de lote
    prediccion = logits.argmax(dim=1).item()

print(f"Clase real: {clases[etiqueta_real]} | Clase predicha: {clases[prediccion]}")
```

Con cinco épocas y esta arquitectura tan pequeña, es habitual alcanzar entre 90 % y 91 % de exactitud sobre el conjunto de prueba en pocos minutos, incluso en la GPU gratuita de Colab. **Experimenta cambiando el número de filtros, añadiendo un tercer bloque convolucional o probando `AvgPool2d` en lugar de `MaxPool2d`** para comprobar en primera persona cómo afecta cada decisión al rendimiento final.

## Resumen de puntos clave

- Una red totalmente conectada no escala bien a imágenes porque destruye la estructura espacial al aplanar la entrada y necesita un número de parámetros enorme
- Las imágenes naturales cumplen dos propiedades clave que las CNN explotan: localidad (los píxeles cercanos están correlacionados) e invariancia a la traslación (un patrón puede aparecer en cualquier posición)
- Un filtro o *kernel* convolucional es una pequeña matriz de pesos aprendibles que se desliza sobre la imagen, funcionando como una plantilla que busca un patrón concreto en cada región
- La compartición de pesos (usar el mismo filtro en toda la imagen) reduce drásticamente el número de parámetros y hace que la red sea equivariante a la traslación
- La convolución discreta se calcula como $(I * K)(i, j) = \sum_{m}\sum_{n} I(i+m, j+n) \cdot K(m, n)$, y el tamaño de la salida depende del *kernel*, el *stride* y el *padding* según $O = \lfloor (N - F + 2P) / S \rfloor + 1$
- El *padding* controla si la salida se encoge (convolución válida), mantiene el tamaño (convolución "igual") o crece (convolución completa); el *stride* controla cada cuántos píxeles se desplaza el filtro
- El *pooling*, típicamente *max pooling* de 2 x 2, reduce la resolución espacial de los mapas de activaciones sin añadir parámetros, aportando eficiencia computacional y cierta invariancia a pequeñas traslaciones
- Una CNN se construye apilando bloques de convolución, no linealidad y *pooling*, para terminar en una o varias capas totalmente conectadas que producen la clasificación final
- Al apilar capas convolucionales, el campo receptivo efectivo crece, lo que permite a la red construir una jerarquía de características: bordes y texturas en las primeras capas, formas y partes de objetos en las capas intermedias, objetos completos en las capas profundas
- Las CNN solo son intrínsecamente robustas frente a traslaciones; la robustez frente a rotaciones, escalas o cambios de iluminación se consigue con aumentación de datos durante el entrenamiento

## Ponlo en práctica

**Dedica entre quince y treinta minutos a este ejercicio antes de seguir al siguiente capítulo.**

1. **Copia el código de la sección anterior en un cuaderno de Google Colab y ejecútalo tal cual**, para confirmar que reproduces una exactitud similar (en torno al 90 %) sobre el conjunto de prueba de FashionMNIST
2. **Cambia el conjunto de datos** de `datasets.FashionMNIST` a `datasets.MNIST` (dígitos manuscritos) sin modificar nada más de la arquitectura, y comprueba si la exactitud sube o baja. Reflexiona: ¿por qué crees que un mismo modelo se comporta de forma distinta en ambos conjuntos?
3. **Modifica el tamaño del *kernel*** de las dos capas convolucionales, pasándolo de `kernel_size=3` a `kernel_size=5` (ajusta también `padding=2` para mantener una convolución "igual"). Anota si cambia la exactitud final y el tiempo de entrenamiento por época
4. **Sustituye una capa `nn.MaxPool2d(2, 2)` por `nn.AvgPool2d(2, 2)`** y observa si el resultado varía
5. **Añade una impresión del número total de parámetros del modelo** con `sum(p.numel() for p in modelo.parameters())` y compáralo mentalmente con los 3.072.000 parámetros que calculamos en la sección teórica para una única capa totalmente conectada sobre una imagen pequeña de 32 x 32 píxeles en color. Esa comparación es la mejor demostración práctica de por qué la compartición de pesos importa
