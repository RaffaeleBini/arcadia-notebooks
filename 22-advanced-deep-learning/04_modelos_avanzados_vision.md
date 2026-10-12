# Redes que aprenden a base de atajos: arquitecturas avanzadas para la visión por computador

## Por qué importa este tema

Ya sabes cómo funciona una red neuronal convolucional (CNN) básica: capas de convolución que extraen bordes y texturas, capas de pooling que reducen la resolución, y una capa final que decide a qué categoría pertenece la imagen. Con eso puedes construir un clasificador razonable. Pero clasificar no es lo mismo que entender una escena.

Piensa en una fotografía de dos personas montando en bicicleta por un jardín, una de ellas en una postura poco habitual. Un clasificador entrenado sobre ImageNet, como ResNet-50, es capaz de decir "bicicleta" y "jardín". Y ahí se detiene: no sabe que hay dos bicicletas, no localiza a las personas, no percibe que una de ellas está en una orientación extraña. Aproximadamente el 80 % de la información que recibes del entorno te llega por la vista, así que un sistema que solo etiqueta imágenes está aprovechando una fracción mínima de lo que una escena visual contiene.

Este límite tiene consecuencias muy concretas. Un vehículo autónomo necesita saber exactamente dónde está cada peatón, no solo si hay peatones en la imagen. Un sistema de apoyo a personas con discapacidad visual necesita describir relaciones espaciales, no solo objetos sueltos. Un robot que debe agarrar una pieza necesita distinguirla de otras piezas iguales que están apiladas al lado. Y las arquitecturas más avanzadas que vas a conocer en este capítulo nacen precisamente para cubrir ese salto: de "qué hay en la imagen" a "dónde está, cuánto ocupa y a qué instancia concreta pertenece".

Un buen ejemplo de hasta dónde llega el aprendizaje por transferencia entre dominios lo protagonizó OpenAI con una mano robótica capaz de resolver un cubo de Rubik. El equipo entrenó el modelo íntegramente en un entorno simulado, porque entrenar directamente sobre el robot físico habría arriesgado dañar la mano con movimientos erráticos. Gracias a una técnica de aleatorización automática del dominio, que combina aumento de datos agresivo con selección de los casos más difíciles, el comportamiento aprendido en la simulación se transfirió con éxito al mundo real. Es la misma idea de fondo que vertebra este capítulo: reutilizar lo aprendido, en lugar de partir de cero cada vez que cambia la tarea o el escenario.

Verás también por qué las redes muy profundas, con decenas o cientos de capas, dejaron de funcionar bien hasta que apareció una solución sorprendentemente sencilla: dejar que la información salte capas enteras sin distorsionarse. Esa idea, la conexión residual, es la pieza que hizo posible entrenar redes de cientos de capas y que hoy sostiene buena parte de la visión por computador moderna.

## Explicación conceptual con analogías

**Las conexiones residuales como un atajo de montaña.** Imagina una ruta de senderismo con muchas etapas seguidas, cada una cuesta arriba. Si en cada etapa pierdes un poco de energía, al llegar a la etapa número 50 puede que no te quede fuerza para seguir avanzando: es exactamente lo que le pasa al gradiente cuando retrocede por una red muy profunda durante el entrenamiento, un fenómeno llamado desvanecimiento del gradiente. Ahora imagina que, junto al camino principal, existe un atajo que baja directamente desde el inicio del sendero hasta cada refugio, sin pasar por las cuestas. No sustituye al camino principal, pero garantiza que siempre tengas una vía directa para llegar. Eso es una conexión residual: una suma que lleva la entrada de un bloque directamente hasta su salida, en paralelo a las convoluciones. La red ya no tiene que aprender la transformación completa desde cero en cada bloque; le basta con aprender la diferencia, el "residuo", respecto a dejar pasar la entrada tal cual.

**El codificador-decodificador como hacer y deshacer una maleta.** Para tareas como la segmentación, donde necesitas una etiqueta por cada píxel, las redes suelen comprimir primero la imagen (el codificador reduce resolución y extrae conceptos abstractos, como quien mete la ropa doblada y compacta en la maleta) y luego la reconstruyen a su tamaño original (el decodificador la "despliega" de nuevo). El problema es que, al comprimir, se pierden arrugas y detalles finos. Las conexiones de salto entre codificador y decodificador —la misma familia de idea que el atajo residual— permiten recuperar esos detalles finos directamente desde las capas tempranas, sin tener que reconstruirlos de memoria.

**La Intersección sobre Unión (IoU) como dos alfombras superpuestas.** Si extiendes dos alfombras rectangulares sobre el suelo, una encima de la otra pero ligeramente desplazadas, la zona donde se solapan y la zona total que cubren entre las dos te dicen mucho más sobre lo bien alineadas que están que la simple distancia entre sus esquinas.

**La pérdida focal como un profesorado que reparte su atención con criterio.** En una clase con mayoría de estudiantes que ya dominan el temario y una minoría con dificultades, un profesorado eficaz no reparte su tiempo por igual: dedica menos atención a quien ya acierta y concentra el esfuerzo en quien todavía se equivoca. Esa es la lógica de la pérdida focal, pensada para que los detectores de objetos no se saturen con los miles de recuadros de fondo, fáciles de clasificar, que sobran en cualquier imagen.

## Desarrollo teórico

### Descomponer el problema de la visión

La clasificación de imágenes supervisada se puede entender como una cadena de piezas: una **entrada** (la imagen), unas **tareas** (qué se le pide al modelo), unos **priors** o supuestos incorporados a la arquitectura, un **orden** de resolución, unas **salidas** y una medida de **precisión**. Cambiar cualquiera de esas piezas abre la puerta a tareas nuevas. Puedes cambiar la tarea (de clasificar a detectar o segmentar), el tipo de entrada (de una imagen a un par de imágenes o a un vídeo) o el tipo de supervisión (de etiquetas manuales a señales que el propio dato genera). Las siguientes secciones recorren esas tres direcciones.

### El problema de la degradación y las conexiones residuales

Antes de las redes residuales, apilar más capas en una CNN no garantizaba mejores resultados. A partir de cierta profundidad, el error de entrenamiento empezaba a subir de nuevo, no por sobreajuste, sino porque la red dejaba de poder optimizarse bien: es el llamado problema de la degradación. La causa principal es el desvanecimiento del gradiente: al propagarse hacia atrás por decenas de capas, el gradiente se multiplica capa a capa y, si esos factores son pequeños, el producto tiende a cero mucho antes de llegar a las primeras capas, que dejan de aprender.

La solución que aporta ResNet (Residual Network, "red residual") es replantear lo que cada bloque debe aprender. En lugar de forzar a un conjunto de capas a aprender directamente una función deseada H(x), se les pide que aprendan el residuo F(x) = H(x) − x, y la salida real del bloque se recompone como:

y = F(x, {Wᵢ}) + x

donde x es la entrada del bloque, F(x, {Wᵢ}) es lo que calculan las convoluciones internas y la suma con x es la conexión de atajo (identity shortcut). Cuando la dimensión de x no coincide con la de F(x) —por ejemplo, porque el bloque reduce la resolución espacial o cambia el número de canales—, el atajo incorpora una proyección lineal sencilla, típicamente una convolución 1x1, antes de sumar. Con esta reformulación, si la transformación óptima para un bloque es simplemente dejar pasar la entrada sin cambios, la red solo tiene que aprender a poner F(x) cerca de cero, algo mucho más fácil que aprender una función identidad exacta con capas no lineales. Y durante la retropropagación, el gradiente dispone siempre de una vía directa a través de la suma, además de la vía que atraviesa las convoluciones, lo que mantiene señales de gradiente más fuertes en las capas iniciales.

Gracias a este mecanismo, se pudieron entrenar redes con cientos e incluso miles de capas y alcanzar precisión sin precedentes en clasificación de imágenes. La misma familia de idea —una vía directa que preserva información y gradiente— reaparece, como verás, en las conexiones de salto de U-Net y en el diseño en forma de "U" de detectores como RetinaNet.

Otra arquitectura relevante de esta etapa es Inception, también conocida como GoogLeNet. En vez de apilar convoluciones de un único tamaño, cada módulo Inception aplica en paralelo filtros de distintos tamaños sobre la misma entrada y concatena sus salidas, capturando patrones a varias escalas dentro del mismo bloque. Es un enfoque complementario al de ResNet: mientras esta ataca el problema de la profundidad, Inception ataca el problema de la escala.

### Detección de objetos: de clasificar a localizar

La detección de objetos es un problema multitarea: para cada objeto de la imagen necesitas su etiqueta de clase y su recuadro delimitador (bounding box), habitualmente parametrizado por el centro, la altura y la anchura. La clase es un problema de clasificación de siempre, pero las coordenadas del recuadro son valores continuos, así que predecirlas exige regresión, no clasificación. La pérdida más simple para ese caso es el error cuadrático medio (ECM), que promedia el cuadrado de la diferencia entre coordenadas predichas y reales:

ECM = (1/N) · Σ (yᵢ − ŷᵢ)²

El reto llega cuando una imagen contiene varios objetos: hace falta decidir qué predicción corresponde a cada objeto real. La estrategia que adoptan los detectores modernos combina dos pasos: primero discretizar el espacio de recuadros posibles y resolverlo como clasificación, y después refinar la predicción ganadora mediante regresión.

**Faster R-CNN**, un detector de dos etapas, ilustra bien este esquema. Una red troncal (por ejemplo, una ResNet) produce mapas de características a partir de la imagen. Sobre esos mapas actúa una Red de Propuesta de Regiones (RPN, Region Proposal Network), que sitúa puntos de anclaje distribuidos uniformemente por la imagen y, en cada uno, evalúa varias combinaciones de escala y proporción —habitualmente tres escalas y tres proporciones, nueve candidatos por punto—. Para cada candidato, la RPN predice una puntuación de "objectness" (probabilidad de que contenga algún objeto) y se conservan los mejores. Esas propuestas pasan después por una segunda etapa que las clasifica y ajusta sus coordenadas mediante regresión. El resultado es preciso pero relativamente lento, del orden de cinco fotogramas por segundo, en parte porque incluye operaciones no diferenciables —como elegir qué propuestas conservar— que obligan a entrenar la RPN y el resto de la red en fases separadas.

**RetinaNet**, un detector de una sola etapa, elimina esa fase intermedia: predice clases y coordenadas directamente sobre una jerarquía de mapas de características a distintas resoluciones (una Red de Pirámide de Características, FPN), lo que le permite entrenarse de extremo a extremo y funcionar más rápido, en torno a ocho fotogramas por segundo. El precio es un desequilibrio severo: la inmensa mayoría de los recuadros candidatos corresponden a fondo, fáciles de clasificar, y su pérdida acumulada domina sobre la de los pocos casos realmente informativos, generando una señal de entrenamiento pobre. Los detectores de dos etapas resuelven esto podando negativos fáciles en la primera fase; los de una sola etapa recurren a la minería de negativos difíciles (entrenar, detectar falsos positivos sobre datos no vistos, añadirlos al conjunto de entrenamiento y repetir) o, de forma más directa, a la **pérdida focal**, propuesta junto con RetinaNet. Su formulación reduce la pérdida de entropía cruzada estándar mediante un factor modulador:

FL(pₜ) = −(1 − pₜ)^γ · log(pₜ)

donde pₜ es la probabilidad que el modelo asigna a la clase correcta y γ (típicamente dos) controla cuánto se atenúa la contribución de los ejemplos ya bien clasificados. Cuando pₜ es alto (el modelo acierta con confianza), el factor (1 − pₜ)^γ se acerca a cero y esa muestra apenas pesa en la pérdida total; cuando pₜ es bajo, el factor se mantiene cercano a uno y la muestra sigue penalizando con fuerza. El entrenamiento queda así centrado en los casos difíciles, sin necesidad de la maquinaria explícita de minería de negativos.

Para **evaluar** ambos tipos de detector no se usa el ECM de entrenamiento, sino la Intersección sobre Unión:

IoU = Área de solapamiento / Área de unión

Un valor de IoU va de cero (sin solapamiento) a uno (coincidencia perfecta) y, a diferencia del ECM, valora el recuadro como un todo en vez de tratar cada coordenada por separado. Un ejemplo lo deja claro: sobre un recuadro real (50, 50, 150, 150), un recuadro predicho (48, 52, 148, 152) —casi idéntico, apenas desplazado un par de píxeles— obtiene un ECM de 4 y un IoU cercano a 1,0; un recuadro (60, 60, 160, 160), desplazado diez píxeles en cada eje, obtiene un ECM de 100 y un IoU de aproximadamente 0,669; y uno más alejado, (30, 30, 130, 130), llega a un ECM de 400 con un IoU de alrededor de 0,444. El orden de calidad que da el IoU —el primer recuadro es claramente el mejor— coincide con la intuición visual, mientras que el ECM penaliza cada coordenada por igual sin representar el solapamiento real. El IoU tiene, eso sí, una limitación: al incluir un máximo no es diferenciable, así que no puede usarse directamente como función de pérdida durante el entrenamiento, aunque existen aproximaciones diferenciables inspiradas en él, como la IoU generalizada.

### Segmentación semántica y de instancias

Cuando el recuadro delimitador resulta demasiado tosco —por ejemplo, para una persona con los brazos extendidos, cuyo recuadro incluiría mucho fondo—, conviene una representación más fina. La **segmentación semántica** asigna una etiqueta de clase a cada píxel de la imagen: es un problema de predicción densa, porque la salida debe tener la misma resolución espacial que la entrada, al contrario que en clasificación, donde la salida es un único vector.

El obstáculo es que las capas de pooling, útiles para ampliar el campo receptivo y abstraer características, reducen la resolución en el camino. Hace falta la operación inversa, el "unpooling" o desagregado. La versión más simple copia cada valor a sus vecinos inmediatos, pero produce salidas borrosas y poco definidas. Dos alternativas más elaboradas son el desagregado guiado por índices (que recuerda, del pooling original, en qué posición exacta estaba el valor máximo mediante la operación argmax, y coloca ahí el valor al reconstruir) y la deconvolución o convolución transpuesta, que invierte estructuralmente la operación de convolución para generar una reconstrucción más ordenada.

**U-Net**, diseñada originalmente para imagen médica, aplica estas ideas en una arquitectura de codificador-decodificador con forma de "U". El codificador reduce la resolución progresivamente mientras extrae características cada vez más abstractas, como haría un clasificador de imágenes; el decodificador la recupera mediante desagregado. Y, como en las redes residuales, incorpora conexiones de salto largas entre cada nivel del codificador y su nivel simétrico en el decodificador, que reinyectan detalle de alta frecuencia perdido durante la compresión y, de paso, facilitan la propagación del gradiente durante el entrenamiento. El modelo se entrena con entropía cruzada calculada píxel a píxel y promediada sobre toda la imagen. RetinaNet, pese a resolver una tarea distinta, comparte esa misma silueta en forma de "U", señal de lo transversal que es este patrón de diseño.

Cuando una imagen contiene varios objetos superpuestos de la misma clase, la segmentación semántica por sí sola resulta ambigua: no distingue una persona de otra si se solapan. La **segmentación de instancias** combina detección y segmentación para resolverlo, asignando una máscara distinta a cada instancia individual. Mask R-CNN es la referencia habitual para esta tarea.

### Transfer learning y evaluación

Entrenar una red profunda desde cero exige muchos datos y mucho cómputo. El aprendizaje por transferencia reutiliza lo que un modelo ya aprendió en una tarea o dominio de origen para acelerar el aprendizaje en una tarea o dominio de destino, bajo la premisa de que muchas características visuales —bordes, texturas, formas— son útiles en más de un contexto. Convertir un clasificador preentrenado en un detector de objetos, por ejemplo, consiste en conservar sus capas iniciales (afinándolas o congelándolas) y sustituir la capa final por otras capaces de predecir clases y coordenadas. El caso ya mencionado del cubo de Rubik resuelto por una mano robótica de OpenAI muestra la variante de transferencia entre dominios distintos: de la simulación al mundo físico.

Para comparar modelos de forma objetiva, cada tarea tiene sus propias métricas y bancos de pruebas. La clasificación usa precisión top-1 (acierta si la predicción principal coincide con la clase correcta) o top-5 (acierta si la clase correcta figura entre las cinco predicciones más probables, útil cuando hay categorías fácilmente confundibles). La detección y la segmentación usan IoU sobre bancos de pruebas como COCO o Cityscapes, del mismo modo que ImageNet sirvió de referencia para la clasificación.

### Más allá de una sola imagen

Una imagen aislada no es la única entrada posible. Un par de imágenes consecutivas permite estimar el flujo óptico: el desplazamiento de cada píxel entre ambos fotogramas, codificado como un mapa denso de valores continuos. FlowNet resuelve esta tarea con una arquitectura de codificador-decodificador semejante a U-Net, entrenada de forma supervisada minimizando la distancia euclídea entre el flujo predicho y el real (el error de punto final, EPE), sobre el conjunto sintético Flying Chairs, que superpone modelos 3D de sillas sobre fondos reales para generar automáticamente miles de pares de imágenes con verdad de referencia exacta.

Un vídeo añade además la dimensión temporal. La solución más directa es apilar los fotogramas en un volumen y sustituir las convoluciones 2D por convoluciones 3D, cuyo núcleo se desliza tanto por el espacio como por el tiempo. Redes de reconocimiento de acciones entrenadas así sobre el conjunto Kinetics —600 000 vídeos de aproximadamente diez segundos repartidos en 600 clases— alcanzan del orden de 82 % de precisión top-1. Arquitecturas como SlowFast combinan dos ramas: una de baja frecuencia de muestreo, especializada en reconocer categorías de objetos estables, y otra de alta frecuencia, especializada en captar movimiento rápido, imitando la forma en que el sistema visual humano combina distintas resoluciones temporales.

### Aprender con menos supervisión

Etiquetar manualmente cada píxel de miles de imágenes es costoso, así que resulta valioso aprender sin depender por completo de etiquetas fuertes. El aprendizaje métrico plantea el problema de otra manera: en lugar de mapear una entrada a una clase o un valor, aprende a proyectar las entradas en un espacio de representación donde las que pertenecen a la misma entidad queden cerca y las que pertenecen a entidades distintas queden lejos. La pérdida contrastiva entrena con pares y un margen m fijo:

L = y · d² + (1 − y) · max(0, m − d)²

donde d es la distancia entre las dos representaciones e y vale uno si el par pertenece a la misma entidad y cero en caso contrario. La pérdida triplete mejora esta idea usando tríos (ancla, positivo, negativo) y distancias relativas en vez de un margen absoluto:

L = max(0, d(ancla, positivo) − d(ancla, negativo) + margen)

de modo que cada clase puede formar un grupo de radio propio, sin imponer el mismo tamaño a todos los grupos. Una aplicación reciente y notable de estas ideas es el aprendizaje de representaciones a partir de aumentos de datos: se generan varias versiones transformadas de la misma imagen (recortes, cambios de color, rotaciones) y se entrena al modelo para reconocer que provienen del mismo origen. Las etiquetas se generan solas, sin intervención humana, y la calidad de las representaciones resultantes llega a igualar a la de modelos entrenados con conjuntos completamente etiquetados como ImageNet.

## Ejemplo de código comentado, en PyTorch

El siguiente bloque, pensado para ejecutarse en Google Colab, muestra primero cómo se construye a mano un bloque residual y después carga una ResNet-18 preentrenada de `torchvision` para clasificar una imagen real.

```python
# Bloque residual construido a mano ------------------------------------
import torch
import torch.nn as nn

class BloqueResidual(nn.Module):
    """Bloque residual básico: dos convoluciones 3x3 y un atajo (shortcut)
    que suma la entrada a la salida antes de la última activación."""

    def __init__(self, canales_entrada, canales_salida, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(canales_entrada, canales_salida, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(canales_salida)
        self.conv2 = nn.Conv2d(canales_salida, canales_salida, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(canales_salida)
        self.relu = nn.ReLU(inplace=True)

        # Si el bloque cambia la resolución o el número de canales,
        # el atajo necesita una proyección 1x1 para poder sumarse a F(x)
        self.atajo = nn.Sequential()
        if stride != 1 or canales_entrada != canales_salida:
            self.atajo = nn.Sequential(
                nn.Conv2d(canales_entrada, canales_salida, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(canales_salida),
            )

    def forward(self, x):
        identidad = self.atajo(x)               # y = F(x) + x
        salida = self.relu(self.bn1(self.conv1(x)))
        salida = self.bn2(self.conv2(salida))
        salida = salida + identidad              # aquí ocurre la conexión residual
        return self.relu(salida)

# Prueba rápida con un tensor aleatorio (lote=2, 3 canales, 32x32 píxeles)
entrada_prueba = torch.randn(2, 3, 32, 32)
bloque = BloqueResidual(canales_entrada=3, canales_salida=16, stride=2)
print(bloque(entrada_prueba).shape)  # torch.Size([2, 16, 16, 16])


# ResNet-18 preentrenada aplicada a una imagen real ---------------------
import urllib.request
from PIL import Image
from torchvision import models, transforms

modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
modelo.eval()  # desactiva dropout y fija las estadísticas de BatchNorm

# Descarga una imagen de ejemplo y las etiquetas de ImageNet
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "imagen_ejemplo.jpg",
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "clases_imagenet.txt",
)
with open("clases_imagenet.txt") as archivo:
    categorias = [linea.strip() for linea in archivo.readlines()]

imagen = Image.open("imagen_ejemplo.jpg").convert("RGB")
preprocesar = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # medias de ImageNet
                          std=[0.229, 0.224, 0.225]),
])
entrada = preprocesar(imagen).unsqueeze(0)  # añade la dimensión de lote

with torch.no_grad():
    salida = modelo(entrada)
probabilidades = torch.nn.functional.softmax(salida[0], dim=0)

top5_prob, top5_idx = torch.topk(probabilidades, 5)
for probabilidad, indice in zip(top5_prob, top5_idx):
    print(f"{categorias[indice]}: {probabilidad.item() * 100:.1f} %")
```

Ejecuta este código en Colab tal cual: descarga la imagen y las etiquetas por internet, así que no necesitas subir ningún archivo. El bloque residual manual te deja ver, línea a línea, dónde ocurre exactamente la suma que da nombre a la arquitectura; la segunda parte te muestra cómo aprovechar, con apenas unas líneas, una ResNet-18 ya entrenada sobre ImageNet.

## Resumen de puntos clave

- Una CNN básica clasifica bien, pero no localiza objetos ni distingue instancias: para eso hacen falta arquitecturas que redefinen tarea, entrada o tipo de supervisión
- El problema de la degradación aparece al apilar demasiadas capas: el gradiente se desvanece y la red deja de optimizarse bien, no de generalizar mal
- ResNet resuelve la degradación con conexiones residuales: cada bloque aprende y = F(x) + x, y el atajo garantiza una vía directa para el gradiente
- La detección de objetos combina clasificación (la clase) y regresión (las coordenadas del recuadro); Faster R-CNN la resuelve en dos etapas y RetinaNet en una sola, apoyada en la pérdida focal para compensar el exceso de fondo fácil
- El IoU mide la calidad de un recuadro predicho de forma más fiel que el error cuadrático medio, porque valora el solapamiento real en vez de cada coordenada por separado
- La segmentación semántica exige predicción densa; U-Net la resuelve con un codificador-decodificador y conexiones de salto que recuperan el detalle perdido al reducir resolución
- El aprendizaje por transferencia, entre tareas o entre dominios (como en el caso de la mano robótica de OpenAI entrenada en simulación), evita partir de cero en cada problema nuevo
- Un par de imágenes o un vídeo abren tareas como el flujo óptico o el reconocimiento de acciones, resueltas con codificadores-decodificadores o convoluciones 3D
- El aprendizaje métrico y los aumentos de datos permiten entrenar representaciones potentes sin depender de etiquetas manuales exhaustivas

## Ponlo en práctica

**Reserva entre 15 y 30 minutos y sigue estos pasos:**

1. **Ejecuta** el ejemplo de código completo en Google Colab y confirma que la ResNet-18 preentrenada clasifica correctamente la imagen de ejemplo
2. **Sustituye** la imagen descargada por dos o tres fotografías propias (súbelas a Colab y ajusta la ruta) y anota si el resultado top-5 acierta la categoría en cada caso
3. **Modifica** la clase `BloqueResidual`: comenta la línea `salida = salida + identidad` para que el bloque no sume el atajo, y vuelve a ejecutar la prueba con el tensor aleatorio
4. **Compara** ambas versiones del bloque a nivel conceptual: sin la suma, cada bloque tiene que reconstruir la función completa desde cero; con ella, solo tiene que aprender la diferencia respecto a dejar pasar la entrada. **Escribe** en dos o tres frases, con tus propias palabras, por qué esa diferencia facilita entrenar redes muy profundas

## Glosario de siglas empleadas en este capítulo

- CNN: red neuronal convolucional (Convolutional Neural Network)
- ECM: error cuadrático medio
- EPE: error de punto final (End-Point Error)
- FPN: red de pirámide de características (Feature Pyramid Network)
- GAN: red generativa antagónica (Generative Adversarial Network)
- IoU: Intersección sobre Unión (Intersection over Union)
- RPN: red de propuesta de regiones (Region Proposal Network)
- SGD: descenso de gradiente estocástico (Stochastic Gradient Descent)
