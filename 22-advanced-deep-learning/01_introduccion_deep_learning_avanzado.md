# Capítulo 1: Introducción al deep learning avanzado

## Por qué importa este tema

Imagina un programa que aprende a jugar al ajedrez sin que nadie le explique ni una sola regla estratégica, solo jugando contra sí mismo una y otra vez. En cuatro horas de entrenamiento, ese programa ya supera al mejor motor de ajedrez del mundo de la época, un sistema llamado Stockfish que llevaba décadas de ajuste manual por parte de expertos. Ese programa se llama AlphaZero, y es uno de los hitos que vas a entender a fondo en este manual.

O imagina otro problema, mucho más antiguo y mucho más importante para la vida cotidiana: predecir la forma tridimensional que adopta una proteína a partir de la secuencia de aminoácidos que la componen. Este problema se conoce como "la paradoja de Levinthal": una proteína de solo 361 aminoácidos tiene, en teoría, unas 10¹⁷² configuraciones posibles. Si esa proteína probara mil billones de configuraciones por segundo, tardaría 10¹⁵² años en explorarlas todas, mucho más que la edad del universo. Sin embargo, en la célula, la proteína se pliega correctamente en cuestión de segundos. Durante décadas, predecir ese plegado fue uno de los grandes problemas sin resolver de la biología. El sistema AlphaFold, desarrollado por DeepMind y presentado en 2020, consiguió predecir estructuras de proteínas con una precisión cercana a la de los métodos experimentales, algo que la biología llevaba persiguiendo desde los años sesenta. Este avance está acelerando el diseño de fármacos y la comprensión de enfermedades.

Estos dos casos, uno lúdico y otro biomédico, tienen algo en común: ambos se resuelven con deep learning, o aprendizaje profundo. Y ambos son fruto de las mismas piezas técnicas que vas a estudiar a lo largo de este manual: redes neuronales profundas, redes convolucionales, mecanismos de atención, aprendizaje por refuerzo y modelos generativos.

No hace falta que trabajes en un laboratorio de investigación para que esto te importe. En DIH DATAlife trabajamos con empresas de las cadenas de valor de Agro-Mar Alimentación, Forestal-Madera, Salud-Cuidados y Biotecnología, y en todas ellas el deep learning avanzado ya se aplica de forma concreta: modelos de visión por computador que detectan enfermedades en cultivos o defectos en madera, sistemas que predicen series temporales de producción, o modelos generativos que ayudan a diseñar nuevos materiales o compuestos. Entender el mapa completo de estas técnicas, aunque sea sin dominar cada detalle matemático, te permite decidir qué herramienta encaja con qué problema y hablar con criterio con quien las implemente.

Este primer capítulo no te va a enseñar a programar una red neuronal profunda entera; eso lo harás poco a poco a lo largo del curso. Su objetivo es otro: darte el mapa completo del territorio, para que cuando llegues a los capítulos siguientes ya sepas por qué existe cada pieza y qué problema resuelve.

## Explicación conceptual con analogías

Antes de entrar en tecnicismos, construyamos una intuición sencilla.

### ¿Qué es una neurona artificial?

Piensa en un jurado de una sola persona que tiene que decidir si aprueba o no una solicitud de préstamo. Esa persona recibe varios datos: los ingresos del solicitante, su historial de pagos, su edad, sus deudas actuales. No le da a todos los datos la misma importancia: quizás el historial de pagos le pesa mucho, y la edad casi nada. Suma esas piezas de información, cada una multiplicada por su propio peso de importancia, y si el resultado supera cierto umbral, aprueba el préstamo; si no, lo rechaza.

Eso es, en esencia, una neurona artificial: recibe varias entradas, las multiplica por unos pesos que reflejan su importancia, suma todo y aplica una regla de decisión final (lo que en deep learning llamamos función de activación) para producir una salida.

### ¿Qué es una red neuronal profunda?

Ahora imagina que, en lugar de un único jurado, montamos varias filas de jurados. La primera fila recibe los datos en bruto y toma decisiones simples: "¿los ingresos son razonables?", "¿el historial es limpio?". La segunda fila no ve los datos originales, sino las decisiones de la primera fila, y combina esas decisiones para llegar a conclusiones más complejas: "¿es un perfil de riesgo bajo?". Una tercera fila combina esas conclusiones intermedias en la decisión final.

Cada fila es una capa. Cuando apilamos muchas capas, la red aprende representaciones jerárquicas: las primeras capas capturan patrones simples y las capas más profundas combinan esos patrones simples en conceptos cada vez más abstractos. A esa acumulación de capas se debe el nombre "aprendizaje profundo" (deep learning): "profundo" no habla de sofisticación filosófica, habla literalmente del número de capas apiladas.

### ¿Por qué "avanzado"?

En el capítulo 2 vas a estudiar la red neuronal más simple: la red neuronal prealimentada (feedforward), donde la información fluye en una sola dirección, de la entrada a la salida, sin ciclos. Es un buen punto de partida, pero tiene limitaciones serias frente a ciertos tipos de datos.

Piensa en tres analogías más:

- Reconocer un gato en una fotografía es distinto a leer una lista de números en una hoja de cálculo, porque una imagen tiene estructura espacial: un ojo cerca de otro ojo, una oreja cerca de la cabeza. Las redes neuronales convolucionales (CNN) están diseñadas para aprovechar esa estructura espacial, igual que quien reconoce un rostro no analiza cada píxel de forma aislada, sino que busca patrones locales (bordes, texturas) y los combina en formas más complejas.
- Entender una frase es distinto a clasificar una foto aislada, porque una frase es una secuencia donde el orden y el contexto importan: "el perro persigue al gato" no significa lo mismo que "el gato persigue al perro". Las redes neuronales recurrentes (RNN) y los mecanismos de atención están diseñados para captar esa dependencia entre elementos que aparecen en distintos momentos de una secuencia.
- Generar una imagen nueva y realista es distinto a clasificar una imagen existente, porque hay que crear algo que no existía, no solo reconocer algo que ya está ahí. Los modelos generativos, como las redes generativas antagónicas (GAN) o los autocodificadores variacionales (VAE), están diseñados precisamente para esa tarea de creación.

El deep learning avanzado es, básicamente, la familia de arquitecturas y técnicas que adaptan la idea básica de la neurona y la red de capas a cada tipo de problema: imágenes, secuencias, generación de datos nuevos, y aprendizaje con poca información etiquetada.

## Desarrollo teórico

### La neurona artificial y la red neuronal

La unidad básica de cualquier red neuronal artificial (ANN) es la neurona. Formalmente, una neurona recibe un vector de entradas x₁, x₂, ..., xₙ, cada una asociada a un peso w₁, w₂, ..., wₙ que representa su importancia relativa. La neurona calcula primero una suma ponderada:

z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

donde b es un término llamado sesgo (bias), que permite desplazar el umbral de decisión. A continuación, esa suma z pasa por una función de activación f, que introduce no linealidad y produce la salida de la neurona:

salida = f(z)

Sin esa función de activación, apilar capas no serviría de nada: una cadena de operaciones puramente lineales sigue siendo, en conjunto, una operación lineal, por muchas capas que pongas. Las funciones de activación más habituales son la sigmoide, la tangente hiperbólica y, sobre todo en redes profundas modernas, la unidad lineal rectificada (ReLU), que simplemente deja pasar los valores positivos y anula los negativos.

Una red neuronal se organiza en capas: una capa de entrada, una o varias capas ocultas y una capa de salida. Cuando el número de capas ocultas es alto, hablamos de red profunda, y de ahí el nombre de la disciplina.

Un dato histórico importante para entender por qué la profundidad importa: las primeras redes de una sola capa (los llamados perceptrones) no podían resolver problemas tan simples en apariencia como la función lógica XOR (que devuelve verdadero cuando exactamente una de las dos entradas es verdadera, y falso en cualquier otro caso). Esta limitación provocó lo que se conoce como el primer "invierno" de las redes neuronales, un periodo de escaso interés y financiación en la disciplina. Añadir capas adicionales resolvió esta limitación y abrió la puerta a las arquitecturas profundas que usamos hoy.

### Cómo aprende una red: la retropropagación

Una red neuronal no nace sabiendo nada: sus pesos empiezan con valores aleatorios. Aprender significa ajustar esos pesos para que la red produzca salidas cada vez más parecidas a las deseadas.

Ese ajuste se mide con una función de pérdida (loss function), que cuantifica la diferencia entre lo que predice la red y el resultado real esperado. El objetivo del entrenamiento es minimizar esa pérdida.

El algoritmo estándar para conseguirlo es la retropropagación (backpropagation), combinada con el descenso de gradiente. La idea, sin entrar en el detalle matemático que verás en el capítulo 5, es esta: se calcula cuánto contribuye cada peso al error final (usando la regla de la cadena del cálculo diferencial) y se actualiza cada peso en la dirección que reduce ese error. Este proceso se repite miles o millones de veces, con pequeños lotes de datos, en variantes del descenso de gradiente estocástico (SGD) como Adam o RMSprop, que adaptan automáticamente cuánto se mueve cada peso en cada paso.

### Redes convolucionales: aprovechar la estructura espacial

Las redes neuronales convolucionales (CNN), popularizadas por Yann LeCun con su arquitectura LeNet-5 para reconocer dígitos escritos a mano, introducen una idea clave: en lugar de conectar cada neurona con todos los píxeles de la imagen (lo que dispararía el número de parámetros), aplican filtros pequeños que se deslizan sobre la imagen, detectando patrones locales como bordes, texturas o formas, en distintos niveles de abstracción según la profundidad de la capa.

Formalmente, la operación de convolución en dos dimensiones, aplicada a una entrada I con un filtro (o kernel) K, se define como:

(I * K)(i, j) = ΣₘΣₙ I(i+m, j+n) · K(m, n)

En términos sencillos: el filtro K recorre la imagen I, y en cada posición (i, j) multiplica los valores del filtro por los valores de la imagen que tiene debajo, y suma el resultado. Ese único filtro, reutilizado en toda la imagen, incorpora una propiedad muy valiosa llamada invariancia a la traslación: un gato sigue siendo un gato aunque aparezca en la esquina superior izquierda de la foto o en el centro, porque el mismo filtro que detecta "orejas puntiagudas" se aplica en cualquier posición. Esta reutilización reduce drásticamente el número de parámetros frente a una red totalmente conectada y hace el entrenamiento mucho más eficiente. El capítulo 3 profundiza en esta arquitectura.

### Redes recurrentes: aprovechar el orden y el tiempo

Cuando los datos son secuenciales (texto, voz, series temporales, secuencias de ADN, vídeo), el orden importa y el contexto pasado condiciona el presente. Las redes neuronales recurrentes (RNN) están diseñadas para esto: mantienen un estado oculto que se actualiza en cada paso de la secuencia y que resume la información relevante de lo visto hasta ese momento.

El problema clásico de las RNN simples es el desvanecimiento del gradiente (vanishing gradient): cuando la secuencia es larga, la señal de error se atenúa tanto al retropropagarse hacia atrás que la red deja de aprender dependencias lejanas en el tiempo. Para solucionarlo se diseñaron arquitecturas con mecanismos de puertas que regulan qué información se conserva y cuál se olvida: las memorias a corto y largo plazo (LSTM) y las unidades recurrentes con puertas (GRU). El capítulo 6 entra en el detalle de estas arquitecturas.

### Atención: mirar solo lo que importa

Un mecanismo más reciente, y hoy central en el procesamiento del lenguaje natural (NLP), es la atención. La idea, en términos intuitivos, es que la red aprende a "poner el foco" en las partes más relevantes de la entrada para resolver una tarea concreta, en lugar de tratar toda la entrada con la misma importancia. Este mecanismo dio lugar a arquitecturas como el transformador (Transformer), base de modelos como BERT o GPT, que verás en los capítulos 7 y 8. También existen modelos con memoria externa explícita, como el ordenador neuronal diferenciable, capaces de consultar y almacenar información de forma similar a como una persona consulta un mapa para responder a una pregunta sobre una ruta.

### Modelos generativos: crear datos nuevos

Hasta ahora hemos hablado de modelos que clasifican o predicen a partir de datos existentes. Los modelos generativos persiguen un objetivo distinto: aprender la distribución de un conjunto de datos para poder generar ejemplos nuevos y plausibles.

Las redes generativas antagónicas (GAN), propuestas por Ian Goodfellow en 2014, entrenan simultáneamente dos redes que compiten entre sí: un generador, que intenta producir datos realistas (por ejemplo, imágenes), y un discriminador, que intenta distinguir los datos generados de los datos reales. Este entrenamiento se plantea como un juego de suma cero, formalizado como un problema minimax:

mín_G máx_D V(D, G) = 𝔼_{x∼p_datos(x)}[log D(x)] + 𝔼_{z∼p_z(z)}[log(1 − D(G(z)))]

Aquí, G es el generador, D es el discriminador, p_datos es la distribución de los datos reales y p_z es la distribución del espacio latente del que el generador toma sus muestras de partida. El discriminador D intenta maximizar su capacidad de distinguir lo real de lo generado, mientras el generador G intenta minimizar esa misma capacidad, es decir, engañar al discriminador. El entrenamiento continúa hasta que el generador produce datos que el discriminador ya no puede distinguir de forma fiable de los datos reales. Estudiarás esta arquitectura en profundidad en el capítulo 11.

Otra familia importante es el autocodificador variacional (VAE), un modelo de aprendizaje no supervisado que aprende variables latentes capaces de explicar los datos observados. Por ejemplo, entrenado con imágenes de sillas vistas desde distintos ángulos, un VAE puede llegar a descubrir por sí mismo factores como la rotación, el ancho o el estilo de las patas, sin que nadie le indique explícitamente esas categorías. Lo verás en el capítulo 9.

### Aprendizaje por transferencia y aprendizaje por refuerzo

El aprendizaje por transferencia (transfer learning) consiste en partir de un modelo ya entrenado en un conjunto de datos grande y ajustarlo (fine-tuning) a una tarea más específica con menos datos disponibles. Esto es especialmente valioso en dominios donde los datos etiquetados escasean, como ocurre en biología o en aplicaciones industriales muy específicas.

El aprendizaje por refuerzo (RL), por su parte, es un marco general en el que un agente interactúa con un entorno: observa su estado, ejecuta acciones y recibe una recompensa. El objetivo del agente es aprender una política que maximice la recompensa acumulada a largo plazo, no solo la recompensa inmediata. Este marco es lo bastante general como para incluir al aprendizaje supervisado y al no supervisado como casos particulares. Los casos de AlphaGo, AlphaZero y el proyecto "Capture the Flag" que mencionamos al inicio son ejemplos de deep learning combinado con aprendizaje por refuerzo: la política de decisión (qué movimiento hacer, qué acción tomar) se representa con una red neuronal profunda, entrenada mediante partidas contra sí misma, un proceso conocido como autoaprendizaje (self-play).

Un detalle instructivo del caso AlphaZero: mientras que un motor de ajedrez tradicional como Stockfish evalúa decenas de millones de posiciones antes de decidir un movimiento, AlphaZero evalúa solo decenas de miles, una reducción de mil veces, un patrón de búsqueda mucho más parecido a la intuición de un gran maestro humano, que evalúa apenas cientos de posiciones.

### Regularización: evitar el sobreajuste

Un riesgo constante al entrenar redes profundas es el sobreajuste (overfitting): que el modelo memorice los datos de entrenamiento en lugar de aprender patrones generales, y por tanto funcione mal ante datos nuevos. Las técnicas de regularización combaten este riesgo. Entre las más comunes:

- Abandono (dropout): en cada paso de entrenamiento se ignora aleatoriamente un subconjunto de neuronas, lo que obliga a la red a no depender en exceso de ninguna neurona concreta
- Decaimiento de pesos (weight decay): añade una penalización a la función de pérdida proporcional a la magnitud de los pesos, favoreciendo modelos más simples
- Aumento de datos (data augmentation): genera ejemplos adicionales de entrenamiento aplicando transformaciones como rotaciones o escalados a los datos existentes

### Aprendizaje responsable

Ningún mapa del deep learning avanzado está completo sin abordar la innovación responsable. Cuanto mayor es el poder de un sistema de inteligencia artificial (IA), mayor es también su responsabilidad asociada. Dos problemas técnicos y éticos aparecen de forma recurrente:

- Los ejemplos adversarios (adversarial examples): pequeñas modificaciones, casi imperceptibles para el ojo humano, en los datos de entrada pueden hacer que un clasificador de imágenes se equivoque de forma drástica, por ejemplo, confundiendo un ciervo con un ave
- El sesgo heredado de los datos: un modelo entrenado con datos que reflejan desigualdades históricas puede reproducir y amplificar esas mismas desigualdades en sus predicciones

Construir sistemas seguros, robustos y verificados, y anticipar sus consecuencias sociales y económicas (como el impacto en el empleo), forma parte integral de la práctica profesional del deep learning, no un añadido opcional. Este manual dedica su capítulo final a esta cuestión.

## Ejemplo de código comentado

Antes de que el capítulo 2 introduzca PyTorch, vamos a construir la pieza más pequeña de todo este edificio: una única neurona artificial, usando solo la librería NumPy. El objetivo es que veas, con tus propias manos, cómo una suma ponderada más una función de activación pueden tomar una decisión sencilla. Copia este código en una celda de Google Colab y ejecútalo tal cual.

```python
import numpy as np

# --- 1. Definimos la neurona ---
# Una neurona recibe entradas (x), las multiplica por sus pesos (w),
# suma un sesgo (b) y pasa el resultado por una función de activación.

def sigmoide(z):
    """Función de activación sigmoide: aplasta cualquier número real
    al rango (0, 1), útil para interpretar la salida como una probabilidad."""
    return 1 / (1 + np.exp(-z))

def neurona(x, w, b):
    """Calcula la salida de una única neurona artificial.
    x: vector de entradas (numpy array)
    w: vector de pesos, mismo tamaño que x
    b: sesgo (número escalar)
    """
    z = np.dot(x, w) + b   # suma ponderada: w1*x1 + w2*x2 + ... + b
    return sigmoide(z)

# --- 2. Ejemplo: ¿aprobamos un préstamo? ---
# Dos entradas normalizadas entre 0 y 1:
#   x1 = nivel de ingresos (0 = bajo, 1 = alto)
#   x2 = historial de pagos (0 = malo, 1 = excelente)
x = np.array([0.8, 0.9])

# Pesos: el historial de pagos (w2) pesa más que los ingresos (w1)
w = np.array([0.4, 1.2])
b = -0.9  # sesgo: hace más difícil aprobar por defecto

salida = neurona(x, w, b)
print(f"Salida de la neurona: {salida:.3f}")
print("Decisión:", "aprobado" if salida > 0.5 else "rechazado")

# --- 3. Probemos con un perfil de mayor riesgo ---
x_riesgo = np.array([0.2, 0.3])
salida_riesgo = neurona(x_riesgo, w, b)
print(f"\nSalida con perfil de riesgo: {salida_riesgo:.3f}")
print("Decisión:", "aprobado" if salida_riesgo > 0.5 else "rechazado")

# --- 4. Una "capa" es simplemente varias neuronas en paralelo ---
# Cada fila de la matriz W son los pesos de una neurona distinta.
def capa(x, W, b):
    """W: matriz de pesos (n_neuronas x n_entradas)
    b: vector de sesgos (uno por neurona)
    Devuelve un vector con la salida de cada neurona de la capa."""
    z = W @ x + b           # producto matricial + suma de sesgos
    return sigmoide(z)

W_capa = np.array([
    [0.4, 1.2],   # pesos de la neurona 1 (como antes)
    [1.0, -0.5],  # pesos de la neurona 2 (prioriza ingresos altos)
])
b_capa = np.array([-0.9, -0.1])

salidas_capa = capa(x, W_capa, b_capa)
print(f"\nSalidas de una capa con dos neuronas: {salidas_capa}")
```

**¡Pruébalo!** Cambia los valores de `w` y `b`, o añade una tercera entrada (por ejemplo, la edad del solicitante) y observa cómo cambia la decisión final. Este pequeño bloque, "suma ponderada más activación", es literalmente la pieza que se repite miles o millones de veces en una red profunda: lo único que cambia entre este ejemplo y AlphaFold es la cantidad de neuronas, capas y datos de entrenamiento, no la idea de fondo.

## Resumen de puntos clave

- El deep learning es un subcampo del aprendizaje automático (machine learning) que usa redes neuronales con muchas capas para aprender representaciones jerárquicas directamente a partir de datos en bruto, sin necesidad de diseñar manualmente las características (feature engineering)
- La unidad básica es la neurona artificial: suma ponderada de entradas más sesgo, seguida de una función de activación no lineal
- Las redes neuronales convolucionales (CNN) incorporan el conocimiento previo de que los datos con estructura espacial (como imágenes) se benefician de filtros locales reutilizables
- Las redes neuronales recurrentes (RNN), y sus variantes LSTM y GRU, están diseñadas para datos secuenciales donde el orden y el contexto temporal importan
- Los mecanismos de atención permiten que una red se centre en las partes más relevantes de la entrada, y son la base de arquitecturas como el transformador
- Los modelos generativos, como las GAN y los VAE, aprenden a crear datos nuevos y plausibles a partir de la distribución de un conjunto de datos
- El aprendizaje por transferencia reutiliza modelos preentrenados en tareas con pocos datos disponibles; el aprendizaje por refuerzo entrena agentes que maximizan recompensas a largo plazo mediante la interacción con un entorno
- El entrenamiento se basa en minimizar una función de pérdida mediante retropropagación y descenso de gradiente, con técnicas de regularización (dropout, decaimiento de pesos, aumento de datos) para evitar el sobreajuste
- Casos reales como AlphaGo, AlphaZero, el proyecto Capture the Flag y AlphaFold muestran que estas piezas técnicas, combinadas, resuelven problemas que antes se consideraban inabordables
- El aprendizaje responsable (robustez frente a ejemplos adversarios, gestión del sesgo, impacto social) es parte integral de la disciplina, no un añadido posterior

## Ponlo en práctica

Dedica entre quince y treinta minutos a esta mini-tarea en Google Colab, ampliando el código del apartado anterior.

1. Abre un cuaderno nuevo en Google Colab y copia el código de la sección anterior
2. Define tu propio escenario de decisión con tres entradas en lugar de dos (por ejemplo: horas de sueño, horas de estudio y nivel de estrés, para predecir si "aprobarías un examen")
3. Elige valores de pesos y sesgo que reflejen tu intuición sobre la importancia de cada factor
4. Prueba al menos tres combinaciones distintas de entradas y observa cómo cambia la salida de la neurona
5. Modifica la función `sigmoide` por una función escalón simple (`return 1 if z > 0 else 0`) y compara los resultados: ¿qué información pierdes al usar una decisión binaria en lugar de un valor continuo entre cero y uno?
6. Escribe, en un par de frases dentro de una celda de texto del propio cuaderno, qué papel jugarían los pesos si en lugar de entrenarlos a mano los ajustara automáticamente un algoritmo de retropropagación a partir de miles de ejemplos reales

Al terminar esta mini-tarea, habrás manipulado con tus propias manos la pieza fundamental que, multiplicada por millones, construye sistemas como AlphaFold. El capítulo 2 retoma esta misma idea, la neurona y la capa, para mostrarte cómo se entrena una red completa con PyTorch.
