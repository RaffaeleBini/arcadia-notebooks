# Capítulo 2 — Fundamentos de las redes neuronales: la neurona, el perceptrón multicapa y el backpropagation

## Por qué importa este tema

En el capítulo uno viste el panorama general del aprendizaje profundo (*deep learning*, DL): qué problemas resuelve, por qué explotó a partir de 2006 y qué papel juegan la potencia de cálculo y los datos. Ahora toca abrir el motor y mirar dentro. Todo lo que construirás en este curso —redes convolucionales, redes recurrentes, mecanismos de atención, modelos generativos— se apoya en cuatro ideas que aprendes hoy: la neurona artificial, el apilamiento en capas, la función de pérdida y el algoritmo que ajusta los pesos, el backpropagation (retropropagación del error), combinado con el descenso de gradiente.

Si entiendes bien este capítulo, entiendes el "kernel" de todo el aprendizaje profundo. Si te lo saltas o lo memorizas sin comprenderlo, cada capítulo posterior te va a costar el triple, porque todos asumen que sabes qué es un peso, qué es una activación y por qué la red "aprende" al recalcular gradientes. Además, aquí escribes tu primer código real: primero backpropagation a mano, con NumPy, para ver el mecanismo sin trucos; después, la misma operación con PyTorch, el framework que usarás durante el resto del curso.

Dominar este tema también te sirve fuera del aula. Si trabajas o vas a trabajar con equipos de digitalización, innovación o datos, entender el mecanismo interno de una red neuronal te permite distinguir entre un proyecto de inteligencia artificial (IA) bien planteado y una promesa vacía envuelta en palabras de moda.

## Explicación conceptual con analogías (antes de tecnicismos)

Imagina un panel de votación con varios expertos. Cada experto observa una parte distinta de un mismo problema —por ejemplo, si una foto contiene un perro— y emite un voto con un peso de confianza: algunos expertos son más fiables (peso alto), otros aportan poco o incluso despistan (peso bajo o negativo, porque su opinión suele ir en contra). Al final, alguien suma todos los votos ponderados, añade un pequeño ajuste de calibración (el sesgo o *bias*) y decide si el resultado global supera cierto umbral. Esa es, en esencia, una neurona artificial: una suma ponderada de entradas, más un ajuste, que produce una señal de salida.

Ahora imagina que no tienes un panel, sino varios paneles en cascada. El primer panel de expertos analiza los píxeles en bruto y emite señales intermedias (por ejemplo, "aquí hay un borde", "aquí hay una zona oscura"). Esas señales alimentan un segundo panel, que las combina para detectar formas más complejas ("esto parece una oreja", "esto parece un hocico"). Un tercer panel combina esas formas para decidir la categoría final. Cada panel es una capa; el conjunto completo es un perceptrón multicapa, la arquitectura más sencilla de red neuronal profunda.

¿Cómo saben los expertos qué peso asignar a cada opinión? Al principio no lo saben: empiezan con pesos aleatorios y se equivocan mucho. Pero después de cada predicción, comparan su veredicto con la respuesta correcta y calculan cuánto se equivocaron (la pérdida o *loss*). Después viene la parte clave: recorren el panel hacia atrás, desde el resultado final hasta el primer experto, y le dicen a cada uno "tu voto contribuyó tanto a este error; ajusta tu peso un poco en esta dirección". Ese recorrido hacia atrás es el backpropagation, y el ajuste gradual, paso a paso, es el descenso de gradiente. Con miles de repeticiones, los pesos se afinan hasta que el panel completo acierta casi siempre.

Una última analogía útil: imagina que estás en la ladera de una montaña con niebla total, sin ver más allá de tus pies, y quieres llegar al punto más bajo del valle. La estrategia razonable es sentir hacia qué lado baja el terreno bajo tus pies y dar un paso en esa dirección. Repites el proceso muchas veces. Eso es exactamente el descenso de gradiente: la "pendiente" es el gradiente de la función de pérdida, y el "paso" es la tasa de aprendizaje (*learning rate*).

## Desarrollo teórico

### La neurona artificial

El modelo de neurona artificial que usamos hoy tiene su origen en el trabajo de Warren McCulloch y Walter Pitts en 1943. Su objetivo no era simular fielmente una neurona biológica, sino capturar unas pocas propiedades relevantes: una neurona real recibe señales por sus dendritas, las combina en el cuerpo celular (soma) y transmite una salida por el axón; algunas conexiones excitan la actividad y otras la inhiben. El modelo artificial traduce esto a matemáticas simples: sin estado interno en el tiempo, y con salidas de valores reales en lugar de impulsos eléctricos.

Formalmente, una neurona artificial recibe un vector de entradas $x = (x_1, x_2, \dots, x_n)$, cada una asociada a un peso $w_i$, calcula la suma ponderada, le añade un término de sesgo $b$ y aplica una función de activación $f$:

$$z = \sum_{i=1}^{n} w_i x_i + b = w \cdot x + b$$

$$y = f(z)$$

Aquí, $y$ es la salida de la neurona, $f$ es la función de activación, $w$ son los pesos, $x$ son las entradas y $b$ es el sesgo. Un peso positivo excita la entrada correspondiente (aumenta su influencia); un peso negativo la inhibe. Sin la función $f$, la operación completa sería una simple transformación afín (lineal más un desplazamiento), y encadenar transformaciones afines solo produce otra transformación afín: no ganarías nada apilando capas. La función de activación es la pieza que introduce la no linealidad necesaria para que la red pueda representar patrones complejos.

En la comunidad de aprendizaje profundo, esta operación recibe distintos nombres según el grupo de investigación que la describa: a la neurona también se le llama unidad; a los pesos, parámetros; y a la operación completa (suma ponderada más sesgo) se le llama, de forma un poco imprecisa, "capa lineal", aunque técnicamente es una transformación afín, porque incluye el término $b$. Si te suena a regresión lineal de estadística, no es casualidad: una sola neurona con activación sigmoide y pérdida de entropía cruzada es matemáticamente equivalente a la regresión logística.

### De una neurona a una capa: la operación matricial

Cuando agrupas varias neuronas que reciben las mismas entradas, cada una con su propio vector de pesos, obtienes una capa. En lugar de calcular cada neurona por separado, conviene expresar la operación con matrices, porque es mucho más eficiente computacionalmente:

$$z = Wx + b$$

Aquí $W$ es una matriz donde cada fila contiene los pesos de una neurona, $x$ es el vector de entrada y $b$ es el vector de sesgos, uno por neurona. Esta forma matricial es exactamente lo que aprovechan las unidades de procesamiento gráfico (GPU) y las unidades de procesamiento tensorial (TPU): están optimizadas para hacer multiplicaciones de matrices en paralelo, que es la operación más costosa —y más frecuente— dentro de una red neuronal.

### Funciones de activación

Repasemos las funciones de activación más habituales, todas aplicadas elemento a elemento sobre cada componente de $z$.

**Función sigmoide.** Comprime cualquier número real en el intervalo (cero, uno):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Su salida se interpreta a menudo como una probabilidad. Es diferenciable en todo punto, una propiedad importante para entrenar la red, pero tiene una limitación seria: cuando $z$ se aleja mucho de cero (muy positivo o muy negativo), la curva se aplana casi por completo y su derivada se acerca a cero. Esto se llama saturación, y provoca el problema de gradientes que se desvanecen (*vanishing gradients*): si la derivada es casi nula, el ajuste que recibe la neurona durante el entrenamiento es casi nulo también, y la red deja de aprender en esa zona.

**Función tangente hiperbólica (tanh).** Similar a la sigmoide, pero comprime los valores en el intervalo (menos uno, uno), centrada en cero. Comparte el mismo problema de saturación en los extremos.

**Unidad lineal rectificada (ReLU).** Hoy es la activación más usada en redes profundas, por su simplicidad y su buen comportamiento durante el entrenamiento:

$$\text{ReLU}(z) = \max(0, z)$$

Si la entrada es positiva, la deja pasar sin cambios; si es negativa o cero, produce cero. Su derivada es también extremadamente simple: uno para entradas positivas, cero para entradas negativas (en cero, técnicamente no es diferenciable, pero en la práctica se ignora, porque la probabilidad de caer exactamente en ese punto es insignificante). Al ser lineal a tramos, ReLU no satura del lado positivo, así que los gradientes no se desvanecen igual que con la sigmoide. Su inconveniente es el problema de las "neuronas muertas": si, por una mala inicialización o una actualización desafortunada de los pesos, todas las entradas a una neurona ReLU se vuelven negativas, esa neurona produce siempre cero y deja de aprender.

**Softmax.** Se usa casi siempre en la capa de salida cuando el problema es de clasificación con más de dos categorías. Convierte un vector de valores reales en una distribución de probabilidad (todos los valores no negativos y suman uno):

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

donde $K$ es el número de clases. Es la generalización multidimensional de la sigmoide: si aplicas softmax a un problema de dos clases, recuperas exactamente la sigmoide. Su punto débil es la inestabilidad numérica cuando los valores de $z$ son grandes, porque la exponencial puede desbordar la precisión del ordenador; en la práctica, las bibliotecas la implementan restando primero el máximo de $z$ a todos sus componentes, lo que da el mismo resultado matemático de forma numéricamente estable.

### El perceptrón multicapa: apilando capas

Un perceptrón multicapa (o red neuronal *feedforward*, porque los datos fluyen en una sola dirección, sin ciclos) apila varias capas de este tipo, una tras otra:

$$h^{(1)} = f^{(1)}\left(W^{(1)} x + b^{(1)}\right)$$
$$h^{(2)} = f^{(2)}\left(W^{(2)} h^{(1)} + b^{(2)}\right)$$
$$\dots$$
$$y = f^{(L)}\left(W^{(L)} h^{(L-1)} + b^{(L)}\right)$$

Las capas intermedias se llaman capas ocultas (*hidden layers*), porque su salida no es directamente el resultado que interesa, sino una representación intermedia. Cuantas más capas ocultas apilas, más "profunda" es la red: de ahí el nombre aprendizaje profundo.

Un ejemplo clásico que justifica la necesidad de capas ocultas es el problema XOR (or exclusivo): un conjunto de puntos en dos dimensiones donde una clase ocupa una diagonal y la otra clase, la diagonal contraria. Ninguna línea recta separa ambas clases; un modelo lineal simple —una sola capa, sin activación no lineal intermedia— no puede resolverlo, por potente que sea el ajuste de sus pesos. Basta, sin embargo, con dos neuronas en una capa oculta con activación sigmoide para resolverlo: cada neurona actúa como una línea de proyección que "dobla" el espacio de entrada, y tras esa deformación no lineal, una única línea final sí separa las clases correctamente. Este ejemplo ilustra la idea central: la capa oculta transforma el espacio de datos hasta volverlo separable, y la capa final solo necesita trazar una frontera simple sobre ese espacio ya transformado.

Este resultado se generaliza en el teorema de aproximación universal, demostrado por Cybenko en 1989 y extendido por Hornik en 1991: una red con una sola capa oculta, con suficientes neuronas y una función de activación no degenerada (no constante, acotada y continua), puede aproximar cualquier función continua con el margen de error que quieras, sobre un dominio acotado. Es un resultado importante, pero con dos matizaciones prácticas: es una prueba de existencia, no te dice cómo construir esa red ni cómo encontrar esos pesos; y el número de neuronas necesario puede crecer de forma exponencial con la dimensión de la entrada. Aquí es donde entra la profundidad: investigaciones posteriores (Montúfar, Pascanu, Cho y Bengio, 2014) muestran que el número de regiones lineales que una red con activaciones ReLU puede representar crece exponencialmente con la profundidad, pero solo de forma polinómica con la anchura. En la práctica, es mucho más eficiente añadir capas que ensanchar una única capa oculta hasta el infinito.

En visión por computador, esta profundidad suele traducirse en una jerarquía de características: las primeras capas detectan bordes y esquinas simples; las capas intermedias combinan esos bordes en formas; las capas más profundas reconocen partes de objetos (una oreja, un hocico); y la capa final combina esas partes para decidir la categoría completa (por ejemplo, "perro").

### Funciones de pérdida

Una función de pérdida (*loss function*) mide qué tan lejos está la predicción de la red del valor real. Es el termómetro que guía todo el entrenamiento: sin una pérdida bien definida, la red no tiene ninguna señal para ajustar sus pesos.

**Entropía cruzada (*cross-entropy*)**, también llamada log-verosimilitud negativa (*negative log-likelihood*) o pérdida logística. Es la más usada en problemas de clasificación. Para un problema binario, con $y \in \{0, 1\}$ la etiqueta real y $\hat{y}$ la probabilidad predicha por la red:

$$\mathcal{L}(y, \hat{y}) = -\left[y \log(\hat{y}) + (1-y)\log(1-\hat{y})\right]$$

Para $N$ muestras y, en el caso general, $K$ clases con codificación one-hot ($y_{i,k}$ vale uno si la muestra $i$ pertenece a la clase $k$, y cero en caso contrario), la fórmula se generaliza a:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})$$

Una propiedad clave de esta pérdida —y de aproximadamente el 95 % de las pérdidas usadas en aprendizaje automático— es que es aditiva respecto a las muestras: la pérdida total es la suma de las pérdidas individuales de cada muestra. Esta propiedad es la que hace viable el entrenamiento por lotes (*batches*) y el descenso de gradiente estocástico, porque puedes aproximar el gradiente total usando solo un subconjunto de los datos.

**Error cuadrático medio (MSE, *mean squared error*)**, usado sobre todo en tareas de regresión, donde la salida es un valor continuo en lugar de una categoría:

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

Un detalle técnico importante, y muy práctico: cuando combinas softmax con entropía cruzada (o sigmoide con entropía cruzada binaria), el gradiente conjunto de esa combinación se simplifica enormemente: se reduce a la diferencia entre la probabilidad predicha y la etiqueta real ($\hat{y} - y$). Esta simplificación evita divisiones por números muy pequeños, que son una fuente habitual de inestabilidad numérica si calculas cada derivada por separado. Por eso, las bibliotecas de aprendizaje profundo casi siempre ofrecen una función combinada ("entropía cruzada con logits") en lugar de encadenar softmax y entropía cruzada como dos pasos independientes.

### Descenso de gradiente

El objetivo del entrenamiento es encontrar los pesos $\theta$ (que agrupan todos los $W$ y $b$ de la red) que minimizan la pérdida $\mathcal{L}(\theta)$. El descenso de gradiente es el algoritmo estándar para hacerlo. La idea, como en la analogía de la montaña con niebla: calcula el gradiente de la pérdida respecto a los parámetros —un vector que apunta en la dirección de mayor crecimiento de la pérdida— y mueve los parámetros en la dirección contraria, porque quieres minimizar, no maximizar:

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t)$$

Aquí $\eta$ (eta) es la tasa de aprendizaje (*learning rate*), un hiperparámetro que controla el tamaño del paso. Si $\eta$ es demasiado grande, puedes saltarte el mínimo y oscilar sin converger; si es demasiado pequeño, el entrenamiento avanza con una lentitud desesperante. Bajo condiciones razonables de suavidad de la función, este método garantiza converger a un mínimo local.

En la práctica casi nunca se calcula el gradiente sobre el conjunto de datos completo en cada paso, porque sería extremadamente costoso con datasets grandes. En su lugar se usa el descenso de gradiente estocástico (SGD, *stochastic gradient descent*): en cada paso se calcula el gradiente sobre un pequeño subconjunto de datos (un *mini-batch*) en lugar de todo el conjunto. Gracias a la propiedad aditiva de la pérdida que mencionamos antes, esto sigue siendo una aproximación válida, y el algoritmo converge igualmente bajo ciertas condiciones sobre el ruido de esa aproximación.

Sobre la base del descenso de gradiente se han desarrollado variantes más sofisticadas. La más usada como punto de partida por defecto es Adam (*Adaptive Moment Estimation*), que combina dos ideas: una tasa de aprendizaje adaptativa para cada parámetro y un término de momento que suaviza las actualizaciones usando el historial de gradientes anteriores. Adam mantiene dos estimaciones móviles, del primer momento ($m_t$, similar a una media de gradientes recientes) y del segundo momento ($v_t$, similar a la varianza de esos gradientes), y las usa para escalar cada actualización de forma individual por parámetro. En la práctica, Adam con una tasa de aprendizaje de referencia de 0,001 suele ser un excelente punto de partida para casi cualquier arquitectura.

### Backpropagation: la regla de la cadena aplicada a la red completa

Ya sabes cómo se actualizan los pesos una vez tienes el gradiente. La pregunta que falta responder es: ¿cómo calculas ese gradiente de forma eficiente, cuando la red puede tener millones de parámetros repartidos en decenas de capas? La respuesta es el backpropagation (retropropagación), un algoritmo redescubierto y popularizado en 1986 por David Rumelhart, Geoffrey Hinton y Ronald Williams, que aplica sistemáticamente la regla de la cadena del cálculo diferencial.

La regla de la cadena, en su forma más simple, dice que si $y$ depende de $u$, y $u$ depende de $x$, entonces:

$$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}$$

Una red neuronal es, precisamente, una larga composición de funciones: entrada, capa lineal, activación, capa lineal, activación… hasta la pérdida final. Puedes representarla como un grafo computacional, donde cada nodo es una operación matemática simple (una suma, un producto, una exponencial) y cada arista transporta el flujo de datos entre operaciones. Backpropagation recorre este grafo en dos fases:

1. **Pase hacia adelante (*forward pass*).** Calculas la salida de la red, capa por capa, desde la entrada hasta la pérdida, guardando en memoria los valores intermedios que vas a necesitar después.
2. **Pase hacia atrás (*backward pass*).** Partiendo de la pérdida, calculas cuánto contribuye cada parámetro al error final, aplicando la regla de la cadena capa por capa, desde la salida hacia la entrada. Cada operación del grafo "sabe" calcular su propio gradiente local, y solo necesita multiplicarlo por el gradiente que le llega desde la capa siguiente.

Este segundo paso evita algo carísimo: si tuvieras que recalcular el efecto de cada peso sobre la pérdida final de forma independiente (por ejemplo, perturbando cada peso uno por uno y viendo cuánto cambia la salida), el coste crecería con el número de parámetros. Backpropagation, en cambio, calcula todos los gradientes en un único recorrido hacia atrás, con un coste comparable al del propio pase hacia adelante.

Veamos cómo se comporta cada operación típica durante el pase hacia atrás, porque esta intuición es la que necesitas para programar el ejemplo a mano de la siguiente sección:

- **Capa lineal ($z = Wx + b$).** El gradiente respecto a $x$ y respecto a $W$ es simétrico gracias a la estructura del producto punto; el gradiente respecto a $b$ es simplemente el gradiente que llega desde la capa siguiente, sin modificar, porque $b$ se suma al final.
- **ReLU.** Actúa como una compuerta: si la entrada fue positiva en el pase hacia adelante, deja pasar el gradiente sin cambios; si fue negativa o cero, lo bloquea (lo pone a cero).
- **Softmax seguido de entropía cruzada.** Como vimos, la combinación se simplifica a la diferencia entre la predicción y la etiqueta real.
- **Máximo (usado, por ejemplo, en *max pooling*).** Solo la entrada que "ganó" la competición recibe gradiente; el resto recibe cero. Esta operación no aprende a seleccionar, solo ajusta a la entrada ya seleccionada.
- **Pérdida cuadrática (MSE).** El gradiente respecto a la predicción es, de nuevo, simplemente la diferencia entre predicción y valor real.

Fíjate en un patrón: en casi todos los casos, el gradiente que "sale" de cada nodo es sencillo de calcular localmente, siempre que conserves los valores calculados durante el pase hacia adelante. Esa es la clave de la eficiencia del algoritmo, y es exactamente lo que vas a programar a mano en la siguiente sección, sobre el ejemplo más pequeño posible: una red de un único peso.

## Ejemplo de código comentado

Vas a ver el mismo ejemplo resuelto dos veces. Primero a mano, con NumPy, para que veas literalmente cada multiplicación y cada derivada del backpropagation. Después, con PyTorch, para comprobar que el framework calcula exactamente lo mismo, pero de forma automática. Ambos fragmentos son autocontenidos y funcionan directamente en Google Colab.

El modelo no puede ser más simple: una única neurona con un solo peso $w$ (sin sesgo, para no complicar el ejemplo), activación sigmoide y pérdida cuadrática, entrenada con un único par de entrada y objetivo.

$$\hat{y} = \sigma(w \cdot x), \qquad \mathcal{L} = (\hat{y} - y_{\text{objetivo}})^2$$

### Parte (a): backpropagation a mano, con NumPy

```python
import numpy as np

# Datos del ejemplo: una entrada y un objetivo fijos
x = 2.0
y_objetivo = 1.0

# Peso inicial (normalmente sería aleatorio; aquí lo fijamos para poder verificar a mano)
w = 0.3

# Tasa de aprendizaje
eta = 0.1

def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))

def derivada_sigmoide(z):
    s = sigmoide(z)
    return s * (1.0 - s)  # f'(z) = f(z) * (1 - f(z)), propiedad de la sigmoide

for paso in range(5):
    # --- Pase hacia adelante ---
    z = w * x                      # suma ponderada (aquí, un único producto)
    y_pred = sigmoide(z)           # activación
    perdida = (y_pred - y_objetivo) ** 2  # pérdida cuadrática

    # --- Pase hacia atrás (regla de la cadena, término a término) ---
    dL_dypred = 2 * (y_pred - y_objetivo)   # derivada de la pérdida respecto a y_pred
    dypred_dz = derivada_sigmoide(z)        # derivada de la sigmoide respecto a z
    dz_dw = x                               # derivada de z = w*x respecto a w

    # Regla de la cadena: dL/dw = dL/dy_pred * dy_pred/dz * dz/dw
    gradiente_w = dL_dypred * dypred_dz * dz_dw

    # --- Descenso de gradiente ---
    w = w - eta * gradiente_w

    print(f"paso {paso}: w={w:.5f}  y_pred={y_pred:.5f}  perdida={perdida:.5f}  "
          f"gradiente={gradiente_w:.5f}")
```

**¡Ejecuta esta celda en Google Colab y observa cómo la pérdida disminuye en cada paso!** Verás que `y_pred` se acerca poco a poco a `1.0` (el objetivo) y que la pérdida decrece de forma monótona: eso es el descenso de gradiente funcionando, con un backpropagation escrito a mano, sin ninguna biblioteca de diferenciación automática.

Fíjate en la línea del gradiente: es literalmente la regla de la cadena que viste en la teoría, escrita como tres multiplicaciones. Cada término corresponde a un tramo del grafo computacional: la pérdida respecto a la predicción, la predicción respecto a la suma ponderada (derivada de la sigmoide) y la suma ponderada respecto al peso. Cuando la red tiene miles de pesos y decenas de capas, el mecanismo es idéntico; solo cambia la cantidad de términos y la necesidad de organizarlos en matrices.

### Parte (b): el mismo ejemplo con PyTorch (tensores, autograd y un optimizador)

Escribir el backpropagation a mano es viable para una neurona, pero sería inviable —y propenso a errores— para una red con millones de parámetros. Aquí es donde entra PyTorch, el framework que usarás durante el resto de este curso. Su motor de diferenciación automática, llamado `autograd`, construye el grafo computacional por ti durante el pase hacia adelante y calcula todos los gradientes automáticamente cuando se lo pides, aplicando exactamente el mismo mecanismo de regla de la cadena que programaste a mano.

```python
import torch

# En Colab, PyTorch ya viene preinstalado; si te falta, descomenta la línea siguiente:
# !pip install torch

# Mismos datos que en la versión a mano
x = torch.tensor(2.0)
y_objetivo = torch.tensor(1.0)

# El peso es ahora un tensor "entrenable": requires_grad=True le dice a PyTorch
# que debe registrar todas las operaciones que involucren a w para poder derivarlas después
w = torch.tensor(0.3, requires_grad=True)

eta = 0.1
# El optimizador aplica el descenso de gradiente por nosotros: mismo mecanismo,
# menos código repetitivo
optimizador = torch.optim.SGD([w], lr=eta)

for paso in range(5):
    # --- Pase hacia adelante ---
    z = w * x
    y_pred = torch.sigmoid(z)
    perdida = (y_pred - y_objetivo) ** 2

    # --- Pase hacia atrás: autograd calcula dL/dw automáticamente ---
    optimizador.zero_grad()  # limpia gradientes acumulados de pasos anteriores
    perdida.backward()      # aquí ocurre el backpropagation completo

    print(f"paso {paso}: w={w.item():.5f}  y_pred={y_pred.item():.5f}  "
          f"perdida={perdida.item():.5f}  gradiente={w.grad.item():.5f}")

    # --- Descenso de gradiente, aplicado por el optimizador ---
    optimizador.step()
```

**¡Compara los números de ambas celdas: deben coincidir paso a paso!** El gradiente que imprime `w.grad` en la versión de PyTorch debe ser idéntico al `gradiente_w` que calculaste a mano en NumPy, porque `autograd` no hace magia: aplica la misma regla de la cadena, solo que de forma automática y generalizable a redes de cualquier tamaño.

Tres ideas clave de este segundo fragmento, que usarás sin parar en el resto del curso:

- `requires_grad=True` marca un tensor como parámetro entrenable; PyTorch construye internamente el grafo computacional cada vez que ese tensor participa en una operación
- `perdida.backward()` recorre ese grafo hacia atrás y guarda el gradiente resultante dentro de `w.grad`, exactamente el paso que programaste a mano
- El optimizador (aquí, `SGD`, descenso de gradiente estocástico) encapsula la fórmula $\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$; para probar Adam en lugar de SGD, bastaría con cambiar una línea: `torch.optim.Adam([w], lr=eta)`

A partir de aquí, cuando construyas redes con muchas capas y miles de parámetros, seguirás este mismo patrón de tres pasos —pase hacia adelante, `backward()`, `optimizador.step()`— sin necesidad de derivar nada a mano nunca más.

## Resumen de puntos clave

- Una neurona artificial calcula una suma ponderada de sus entradas, le suma un sesgo y aplica una función de activación no lineal: sin esa no linealidad, apilar capas no aporta nada nuevo
- Las funciones de activación más relevantes son la sigmoide y la tangente hiperbólica (saturan y provocan gradientes que se desvanecen), y la ReLU (evita ese problema, aunque puede generar neuronas muertas); softmax se reserva casi siempre para la capa de salida en clasificación multiclase
- Un perceptrón multicapa apila varias capas de neuronas; las capas ocultas transforman el espacio de datos hasta hacerlo separable, lo que permite resolver problemas no lineales como el XOR
- El teorema de aproximación universal garantiza que una red con una sola capa oculta puede aproximar cualquier función continua, pero no dice cómo construirla; en la práctica, aumentar la profundidad es mucho más eficiente que ensanchar una sola capa
- La función de pérdida mide el error de la predicción: entropía cruzada para clasificación, error cuadrático medio (MSE) para regresión; combinar softmax con entropía cruzada simplifica el gradiente a una simple diferencia entre predicción y etiqueta
- El descenso de gradiente actualiza los parámetros moviéndolos en la dirección contraria al gradiente de la pérdida, escalado por la tasa de aprendizaje; su variante estocástica (SGD) usa mini-lotes en lugar de todo el conjunto de datos, y Adam es hoy el optimizador de referencia
- Backpropagation aplica la regla de la cadena de forma sistemática sobre el grafo computacional de la red, en un pase hacia adelante y un pase hacia atrás, para calcular todos los gradientes con un coste comparable al de una sola predicción
- PyTorch automatiza este cálculo mediante `autograd`: marcas los parámetros con `requires_grad=True`, llamas a `.backward()` sobre la pérdida y un optimizador (`torch.optim`) aplica la actualización; es el framework que usarás durante el resto del curso

## Ponlo en práctica

Dedica entre quince y treinta minutos a esta mini-tarea en Google Colab. Reutiliza el código de la parte (b) de este capítulo como punto de partida.

1. Amplía el modelo de una sola neurona a dos pesos: sustituye `w = torch.tensor(0.3, requires_grad=True)` por dos entradas (`x1`, `x2`) y dos pesos (`w1`, `w2`), de forma que `z = w1 * x1 + w2 * x2`
2. Añade un sesgo entrenable `b` (también con `requires_grad=True`) a la suma ponderada
3. Cambia la tasa de aprendizaje `eta` a tres valores distintos (por ejemplo, 0,01, 0,5 y 5,0) y ejecuta veinte pasos de entrenamiento con cada una; observa y anota qué ocurre con la pérdida en cada caso: ¿converge con suavidad, converge más rápido, u oscila sin llegar a estabilizarse?
4. Sustituye el optimizador `SGD` por `Adam` con la misma tasa de aprendizaje inicial y compara cuántos pasos necesita cada uno para que la pérdida baje del mismo umbral, por ejemplo 0,01
5. Como reto final, cambia la activación de `torch.sigmoid` a `torch.relu` y observa qué ocurre si inicializas los pesos con valores muy negativos: ¿logra la red seguir aprendiendo, o se queda "atascada"? Relaciona lo que observes con el problema de las neuronas muertas que explicamos en la teoría

Guarda tu notebook de Colab: lo necesitarás como referencia cuando, en próximos capítulos, construyas redes con varias capas ocultas reales.
