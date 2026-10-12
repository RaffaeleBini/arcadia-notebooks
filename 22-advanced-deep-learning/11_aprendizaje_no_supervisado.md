# Capítulo 11. Aprendizaje no supervisado y aprendizaje de representaciones

*Acrónimos usados en este capítulo: inteligencia artificial (IA), aprendizaje por refuerzo (RL), red Q profunda (DQN), máquina de Boltzmann restringida (RBM), análisis de componentes principales (PCA), autocodificador variacional (VAE), cota inferior de la evidencia (ELBO), divergencia de Kullback-Leibler (KL), red generativa antagónica (GAN), representaciones de codificador bidireccional de transformadores (BERT), codificación predictiva contrastiva (CPC), error cuadrático medio (MSE), autocodificador variacional cuantizado vectorialmente (VQ-VAE), red de consulta generativa (GQN).*

## Por qué importa este tema

Hasta ahora, en este manual, casi todo lo que has entrenado ha necesitado etiquetas: una imagen y su clase, una secuencia y su traducción. Ese enfoque funciona bien, pero tiene un coste que rara vez se menciona: alguien tuvo que sentarse a etiquetar cada ejemplo. Clasificar el millón de imágenes de ImageNet costó miles de horas de trabajo humano, y en ámbitos como la medicina el etiquetado exige además personal experto: una radióloga que revisa miles de radiografías para marcar signos de neumonía no es un recurso barato.

Mientras tanto, el mundo produce una cantidad casi infinita de datos sin etiquetar: vídeo, audio, texto, registros de sensores. Si tu modelo solo aprende de lo etiquetado, desperdicias la inmensa mayoría de esa información. La pregunta que recorre este capítulo es sencilla de formular y difícil de responder: ¿puede un modelo aprender algo útil mirando solo los datos, sin que nadie le diga qué es cada cosa?

Esta es la tercera gran familia del aprendizaje automático, junto al aprendizaje supervisado y al aprendizaje por refuerzo (RL), y se distingue de las otras dos por el tipo de retroalimentación que recibe. En el aprendizaje supervisado, cada entrada llega con su etiqueta correcta. En el RL, el agente recibe una recompensa escalar, casi siempre solo al final de una tarea. En el aprendizaje no supervisado no hay señal de ese tipo: solo tienes los datos de entrada y debes decidir qué hacer con ellos, lo que plantea dos preguntas: ¿lo necesitamos realmente, si el aprendizaje supervisado de extremo a extremo ya funciona con suficientes datos y cómputo? Y si la respuesta es sí, ¿cómo sabes si tu algoritmo ha hecho un buen trabajo, sin ninguna respuesta correcta con la que comparar?

La historia reciente de la inteligencia artificial (IA) da pistas sobre la primera pregunta. Antes de 2012, el aprendizaje de representaciones era trabajo manual: ingeniería de características, métodos de núcleo y, desde 2006, el uso de máquinas de Boltzmann restringidas (RBM) para preentrenar redes profundas de forma no supervisada, técnica de Hinton y Salakhutdinov que mejoraba la convergencia de los clasificadores. Pero en 2012 AlexNet ganó el reto ImageNet por un margen enorme sin ningún preentrenamiento no supervisado, apoyándose solo en más datos, redes más profundas y mejor hardware, lo que llevó a pensar que bastaba el aprendizaje supervisado de extremo a extremo.

Diez años después, aparecieron grietas evidentes. La eficiencia de datos sigue siendo un problema: los algoritmos de RL profundo, como la red Q profunda (DQN), necesitan órdenes de magnitud más experiencia que una persona para aprender tareas sencillas. La robustez también falla: un ruido casi imperceptible sobre la imagen de un panda basta para que un clasificador de última generación la etiquete como gibón con más del 99 % de confianza. Y la generalización es débil: agentes entrenados con decenas de miles de variantes de un videojuego simple rinden mucho peor en niveles nuevos que en los de entrenamiento.

Estas limitaciones son el argumento que Yann LeCun, Geoffrey Hinton y Yoshua Bengio, los tres premios Turing, defendieron en la conferencia AAAI: antes de resolver una tarea concreta, conviene aprender a representar el mundo, tal y como parece hacer un bebé mucho antes de aprender ninguna tarea específica. Con un modelo del mundo que tenga las abstracciones adecuadas, un cambio del entorno se reduce a una o pocas variables, y te adaptas a él con muchos menos datos.

Este capítulo recorre cómo se intenta lograrlo: primero con clustering y reducción de dimensionalidad, después con los autocodificadores como ejemplo central de representación aprendida por reconstrucción, y por último con su conexión hacia el aprendizaje autosupervisado y contrastivo, la línea que hoy domina el aprendizaje de representaciones moderno, desde BERT hasta los modelos de visión entrenados sin ninguna etiqueta.

## Explicación conceptual con analogías

### Comprimir para conservar lo esencial

Imagina que debes describir una fotografía en un mensaje de texto muy corto. No puedes enviar los píxeles: tienes que resumir, algo como "una persona con chaqueta roja, de pie en un paisaje nevado, sonriendo". Con ese resumen, la otra persona se hace una idea razonablemente fiel de la foto, aunque no reconstruya los mismos píxeles.

Un autocodificador hace algo parecido, de forma automática y numérica. Consta de dos piezas:

- Un **codificador**, que comprime la imagen completa en un vector mucho más pequeño, la representación latente o *código*, igual que tú comprimiste la foto en una frase
- Un **decodificador**, que toma ese vector comprimido y reconstruye la imagen original a partir de él, igual que la otra persona se imagina la foto a partir de tu frase

Al principio del entrenamiento, el decodificador produce imágenes borrosas y poco fieles, igual que un resumen mal hecho pierde detalles importantes. Pero, al ajustar los pesos de ambas piezas para minimizar el error de reconstrucción, la red aprende qué información merece la pena conservar en ese resumen y cuál puede descartarse. El vector latente termina capturando los rasgos más informativos de los datos, sin que nadie le haya dicho explícitamente cuáles son.

### La forma de escribir treinta y siete

Piensa en tres formas de escribir el número treinta y siete: la cifra arábiga "37", los números romanos "XXXVII" y el binario "100101". Las tres representan la misma cantidad, pero cada forma facilita operaciones distintas: la arábiga muestra de un vistazo la descomposición en decenas y unidades, el binario la descomposición en potencias de dos, y los romanos casi no facilitan ninguna operación aritmética.

Esa es la idea central de "representación": la forma de representar la información es independiente de la información misma, y elegir bien esa forma simplifica el trabajo posterior. Aprender una buena representación de una imagen, un texto o el estado de un videojuego no es solo comprimir datos: es encontrar la forma que hace más sencillas las tareas posteriores (clasificar, agrupar, planificar).

### El panel de mezclas con un dial por factor

La idea de representación "desenredada" (*disentangled*), que aparece más adelante, se entiende bien con la analogía de una mesa de mezclas de sonido con un dial independiente para el volumen, otro para los graves, otro para los agudos: girar el dial de graves solo cambia los graves.

Imagina ahora un panel mal diseñado donde girar el dial de graves también sube el volumen y distorsiona los agudos: está "enredado", no puedes tocar una propiedad sin arrastrar cambios en las demás. Una representación desenredada es la versión bien diseñada de ese panel aplicada a los datos: un conjunto de variables latentes donde cada una controla, de forma independiente, un único factor generativo (posición, tamaño, color, forma), sin interferir con las demás.

## Desarrollo teórico

### Clustering y reducción de dimensionalidad: las dos ideas de entrada

Si solo tienes observaciones sin etiquetar, hay dos estrategias razonables. La primera es agrupar: decidir qué observaciones se parecen más entre sí y formar clústeres. El clustering es valioso porque permite resolver después problemas de clasificación con muchos menos datos: si aprendes algo sobre un robot dentro de un clúster de robots con ruedas, generalizas ese conocimiento al resto del clúster.

La segunda es identificar un pequeño número de ejes que expliquen la mayor parte de la variación en los datos. Si las diferencias entre un conjunto de robots se explican por su altura y su número de ruedas, representar cada robot como un punto en dos dimensiones, en lugar de como una imagen completa, resulta más interpretable y facilita la clasificación posterior.

El problema es que no existe una única forma "correcta" de agrupar o reducir dimensiones: puedes agrupar robots por tipo de pierna, por número de brazos o por altura, y las tres opciones son válidas. Esta ambigüedad explica por qué evaluar un algoritmo no supervisado es tan difícil: no hay verdad de referencia con la que comparar. En la práctica se recurre a índices de calidad de clúster, al error de reconstrucción o, de forma más pragmática, al rendimiento de un clasificador simple entrenado sobre la representación aprendida con pocos datos etiquetados: la evaluación semisupervisada, estándar de facto en la investigación actual, medida a menudo sobre ImageNet.

### Qué hace buena a una representación

La neurociencia ofrece una pista: en el flujo visual ventral del cerebro, la información sobre un objeto llega completa a la retina, pero "enredada" en el espacio de alta dimensión de las respuestas neuronales tempranas. Según la hipótesis del desenredo (*untangling*), el procesamiento visual posterior reformatea esa información hasta que las fronteras de decisión se vuelven casi lineales.

Un razonamiento similar surge en un proceso de decisión de Markov, el marco habitual del RL: la representación del estado determina la dificultad de la tarea. Al cruzar una calle, una representación que distinga "acera sin coches" de "acera con un coche acercándose" ayuda; el color exacto del coche no ayuda y puede dificultar la generalización. Pero si la tarea cambia (ahora quieres parar un taxi), ese color vuelve a ser relevante: una buena representación descarta detalles irrelevantes para la tarea actual, pero mantiene disponible la información latente necesaria si la tarea cambia.

A esto se suma la compositividad, tomada del lenguaje: el significado de una expresión compleja se obtiene combinando los significados de sus partes según ciertas reglas, lo que permite construir un número enorme de conceptos a partir de piezas y reglas finitas.

La física aporta un tercer hilo: la simetría, una transformación que deja invariante alguna propiedad relevante del sistema (el teorema de Noether conecta cada simetría continua con una cantidad conservada). Trasladado al aprendizaje de representaciones, el objetivo es que la función de representación $f$ sea **equivariante** respecto a un grupo de transformaciones $G$ de interés (traslación horizontal, vertical, cambio de color): aplicar la transformación antes o después de mapear debe dar el mismo resultado.

$$
g \cdot f(w) = f(g \cdot w) \qquad \forall\, g \in G,\ w \in W
$$

Esto contrasta con la **invariancia**, donde la representación no cambia pese a la transformación, útil cuando la posición de un objeto es irrelevante para su clase, tal y como explota la compartición de pesos de las redes convolucionales. Una representación equivariante bien construida particiona además el espacio en subespacios independientes, uno por transformación, lo que facilita la atención (por ejemplo, con una máscara binaria) y el clustering con métricas de similitud sencillas. Desenredo, equivariancia, compositividad y soporte para atención y clustering forman el criterio con el que se juzga si una técnica va en la dirección correcta.

### Autocodificadores: el ejemplo central

Un autocodificador es una red entrenada para reproducir su propia entrada, forzada a pasar por un cuello de botella que impide copiar los datos de forma trivial. Dado un vector de entrada $x$, el codificador aplica una transformación (pesos $W_e$, sesgo $b_e$, activación $f$) para obtener la representación latente $z$:

$$
z = f(W_e x + b_e)
$$

El decodificador aplica la transformación inversa (pesos $W_d$, sesgo $b_d$, activación $g$) para producir la reconstrucción $\hat{x}$:

$$
\hat{x} = g(W_d z + b_d)
$$

El entrenamiento minimiza el error de reconstrucción entre $x$ y $\hat{x}$. Para datos continuos se usa el error cuadrático medio (MSE):

$$
\mathcal{L}(x, \hat{x}) = \| x - \hat{x} \|^2
$$

y para píxeles normalizados entre cero y uno, la entropía cruzada binaria. Como $z$ tiene menos dimensiones que $x$, la red no puede aprender la función identidad: debe descubrir qué combinaciones de características bastan para reconstruir lo esencial, descartando redundancia y ruido. Existen variantes: los **autocodificadores dispersos** (*sparse*) penalizan para que solo unas pocas unidades del código estén activas a la vez; los de **eliminación de ruido** (*denoising*) reconstruyen la entrada a partir de una versión corrompida, produciendo representaciones más robustas; y los **autocodificadores variacionales (VAE)**, que tratan el espacio latente de forma probabilística.

### Autocodificadores variacionales y el objetivo ELBO

Un VAE asume que los datos $x$ se generan a partir de variables latentes $z$ muestreadas de un *prior* $p(z)$, casi siempre una gaussiana estándar, seguidas de un proceso de generación $p_\theta(x \mid z)$. El objetivo es invertir ese proceso: dado $x$, inferir la posterior $p(z \mid x)$ sobre qué latentes probablemente lo generaron.

Calcular esa posterior exacta es, en general, intratable. La solución es aproximarla con una red de inferencia (el codificador) que produce $q_\phi(z \mid x)$, y entrenar maximizando la cota inferior de la evidencia (ELBO):

$$
\log p_\theta(x) \geq \mathbb{E}_{z \sim q_\phi(z \mid x)}\big[\log p_\theta(x \mid z)\big] - D_{\mathrm{KL}}\big(q_\phi(z \mid x)\,\|\,p(z)\big)
$$

El primer término es, en esencia, el error de reconstrucción negativo. El segundo, la divergencia de Kullback-Leibler (KL) entre la posterior aproximada y el *prior*, actúa como regularizador. Por eso el *prior* funciona como parámetro de diseño: con una gaussiana de dimensiones independientes, el término KL empuja a la posterior a exhibir también esa independencia, un paso hacia representaciones desenredadas.

De aquí nace el **beta-VAE**, que multiplica el término KL por un coeficiente $\beta > 1$:

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi(z \mid x)}\big[\log p_\theta(x \mid z)\big] - \beta \, D_{\mathrm{KL}}\big(q_\phi(z \mid x)\,\|\,p(z)\big)
$$

Aumentar $\beta$ ya no da una cota exacta de la verosimilitud, pero empuja hacia representaciones más desenredadas, a costa de algo de fidelidad en la reconstrucción. Para comprobarlo se usa el **recorrido latente** (*latent traversal*): fijas todas las variables menos una, la varías de forma continua y observas cómo cambia la imagen generada. En un beta-VAE bien entrenado, cada variable suele controlar un único factor interpretable; en un VAE estándar, una sola variable mueve varios factores a la vez, la firma de una representación enredada. En el agente DARLA, esta propiedad permitió transferir políticas de RL de la simulación a un robot real con mucha menos degradación de rendimiento.

### Redes generativas antagónicas como alternativa sin reconstrucción

Una red generativa antagónica (GAN) modela la distribución de los datos sin cota de verosimilitud ni reconstrucción píxel a píxel. Enfrenta dos redes: un generador $G$, que transforma ruido $z \sim p(z)$ en una muestra sintética $G(z)$, y un discriminador $D$, que estima la probabilidad de que una muestra sea real. El objetivo es un juego minimax, donde el discriminador maximiza la expresión y el generador la minimiza, engañándolo cada vez mejor:

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}(x)}\big[\log D(x)\big] + \mathbb{E}_{z \sim p(z)}\big[\log\big(1 - D(G(z))\big)\big]
$$

El problema de una GAN estándar es que no resuelve la inferencia: no te dice la representación $z$ de un dato real $x$. Variantes como BiGAN o BigBiGAN añaden un codificador al juego adversario: el discriminador compara pares dato-código en lugar de muestras sueltas, obligando al modelo a igualar las distribuciones conjuntas sin pérdida de reconstrucción a nivel de píxel. El resultado captura información de alto nivel (qué objetos hay, de qué color es la ropa) en lugar de detalles de bajo nivel, lo que en su momento situó a estos modelos entre el estado del arte para clasificación semisupervisada.

### Aprendizaje autosupervisado: crear supervisión sin etiquetas

El aprendizaje autosupervisado diseña tareas *pretexto* que no requieren etiquetas humanas pero fuerzan al modelo a aprender características útiles:

- **Predecir la rotación** aplicada a una imagen: adivinar el ángulo exige saber qué aspecto tiene el objeto en su orientación natural
- **Colorización**: convertir fotografías en color a blanco y negro y entrenar al modelo para recuperar el color original
- **Predicción de parches**: dada una imagen dividida en fragmentos, predecir la posición relativa de uno respecto a otro
- **Ordenación temporal**: dados fotogramas de vídeo desordenados, reconstruir el orden correcto, capturando coherencia temporal
- **Predicción de tokens enmascarados**: BERT (representaciones de codificador bidireccional de transformadores) oculta palabras al azar en una frase y entrena al modelo para recuperarlas, combinado con predecir el orden correcto entre pares de frases

El hilo común: estas tareas explotan la estructura propia de cada modalidad para generar automáticamente pares de entrada y salida correcta, sin que nadie etiquete nada.

### Aprendizaje contrastivo: acercar lo parecido, alejar lo distinto

El aprendizaje contrastivo aprende representaciones sin un modelo generativo completo, usando una pérdida de clasificación construida a partir de pares: dado un dato ancla, un ejemplo positivo (una versión relacionada) debe quedar cerca de él en el espacio de representación, y los ejemplos negativos (datos no relacionados), lejos.

Un ejemplo temprano es word2vec, que aprende representaciones de palabras prediciendo qué palabras aparecen en su contexto, con ejemplos positivos (palabras cercanas) y negativos (palabras al azar). El resultado captura relaciones semánticas: entrenado por separado en inglés y en español, unos pocos pares de palabras traducidas bastan para aprender una transformación lineal que traduce el resto del vocabulario con buena fidelidad.

La codificación predictiva contrastiva (CPC) extiende la idea a secuencias: predice representaciones futuras (positivas) frente a representaciones de otros puntos (negativas), maximizando la información mutua entre contexto y futuro. En el benchmark semisupervisado de ImageNet, un modelo con representaciones CPC ajustado con un 80 % menos de etiquetas supera a uno entrenado directamente sobre píxeles.

La formalización estándar de esta familia de pérdidas es el InfoNCE, usado en SimCLR (marco simple para el aprendizaje contrastivo de representaciones visuales), que genera dos versiones aumentadas de la misma imagen (recorte, color, desenfoque) y acerca sus representaciones mientras aleja las del resto del lote. Para un par positivo $(z_i, z_j)$ dentro de un lote de $2N$ vistas:

$$
\mathcal{L}_{i,j} = -\log \frac{\exp\big(\mathrm{sim}(z_i, z_j) / \tau\big)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp\big(\mathrm{sim}(z_i, z_k) / \tau\big)}
$$

donde $\mathrm{sim}(\cdot,\cdot)$ es la similitud coseno y $\tau$ la temperatura. Con un clasificador lineal simple sobre estas representaciones, SimCLR alcanzó resultados de referencia en ImageNet, superando a arquitecturas comparables entrenadas con enfoques generativos.

### El hilo que conecta todo

Generativo (autocodificadores, VAE, GAN), autosupervisado y contrastivo no son alternativas excluyentes: son formas complementarias de aprender representaciones sin etiquetas, y a menudo se combinan. Monet usa atención junto con un beta-VAE para segmentar y desenredar objetos de una escena sin supervisión; el autocodificador variacional cuantizado vectorialmente (VQ-VAE) sustituye el espacio latente continuo por variables discretas indexadas en una tabla de códigos, logrando alta compresión con reconstrucción razonable. Y cuando la representación aprendida se combina con RL, como en DARLA o en las redes de consulta generativa (GQN), los agentes aprenden más rápido, generalizan mejor y transfieren conocimiento entre entornos: los tres puntos débiles con los que abrimos este capítulo.

## Ejemplo de código comentado, en PyTorch

El siguiente ejemplo construye un autocodificador sencillo, con capas totalmente conectadas (lineales), lo entrena sobre el conjunto MNIST de dígitos manuscritos y compara visualmente las imágenes originales con sus reconstrucciones. **Copia el código en una celda de Google Colab y ejecútalo** para verlo funcionar de principio a fin.

```python
# Instala/actualiza torchvision si hace falta (normalmente ya viene en Colab)
# !pip install torch torchvision --quiet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Usa GPU si Colab te la ha asignado, si no, usa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de entrenamiento: {device}")

# --- 1. Datos ---
# MNIST: dígitos de 28x28 en escala de grises, valores normalizados entre 0 y 1
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

DIM_ENTRADA = 28 * 28   # cada imagen se aplana a un vector de 784 valores
DIM_LATENTE = 32        # tamaño del cuello de botella: el "resumen" de cada imagen

# --- 2. Definición del autocodificador ---
class Autocodificador(nn.Module):
    def __init__(self, dim_entrada=DIM_ENTRADA, dim_latente=DIM_LATENTE):
        super().__init__()
        # Codificador: comprime progresivamente hasta el cuello de botella
        self.codificador = nn.Sequential(
            nn.Linear(dim_entrada, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dim_latente),
        )
        # Decodificador: expande de vuelta hasta el tamaño original
        # Sigmoid final porque los píxeles de MNIST están normalizados en [0, 1]
        self.decodificador = nn.Sequential(
            nn.Linear(dim_latente, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, dim_entrada),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.codificador(x)
        x_reconstruida = self.decodificador(z)
        return x_reconstruida, z

modelo = Autocodificador().to(device)
print(modelo)

# --- 3. Función de pérdida y optimizador ---
# El error de reconstrucción binario funciona bien con píxeles en [0, 1] y salida sigmoide
criterio = nn.BCELoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

# --- 4. Bucle de entrenamiento ---
def entrenar_una_epoca(modelo, loader, criterio, optimizador):
    modelo.train()
    perdida_total = 0.0
    for imagenes, _ in loader:  # las etiquetas no se usan: no hace falta supervisión
        imagenes = imagenes.view(imagenes.size(0), -1).to(device)  # aplana a vector

        optimizador.zero_grad()
        reconstruidas, _ = modelo(imagenes)
        perdida = criterio(reconstruidas, imagenes)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * imagenes.size(0)
    return perdida_total / len(loader.dataset)

num_epocas = 10
for epoca in range(1, num_epocas + 1):
    perdida_train = entrenar_una_epoca(modelo, train_loader, criterio, optimizador)
    print(f"Época {epoca}/{num_epocas} | pérdida de reconstrucción: {perdida_train:.4f}")

# --- 5. Comparación visual: originales frente a reconstrucciones ---
modelo.eval()
imagenes_prueba, _ = next(iter(test_loader))
imagenes_prueba = imagenes_prueba[:10]  # nos quedamos con diez ejemplos

with torch.no_grad():
    entrada_plana = imagenes_prueba.view(imagenes_prueba.size(0), -1).to(device)
    reconstruidas, _ = modelo(entrada_plana)
    reconstruidas = reconstruidas.view(-1, 28, 28).cpu()

fig, ejes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    ejes[0, i].imshow(imagenes_prueba[i].squeeze(), cmap="gray")
    ejes[0, i].axis("off")
    ejes[1, i].imshow(reconstruidas[i], cmap="gray")
    ejes[1, i].axis("off")

ejes[0, 0].set_title("Original", loc="left", fontsize=10)
ejes[1, 0].set_title("Reconstruida", loc="left", fontsize=10)
plt.tight_layout()
plt.show()
```

Tras diez épocas, las reconstrucciones deberían parecerse claramente a los dígitos originales, aunque algo más suaves y con bordes menos definidos: es la firma típica de una pérdida basada en reconstrucción, la misma que veías en el capítulo teórico al comparar imágenes originales y reconstruidas de un VQ-VAE. **Experimenta reduciendo `DIM_LATENTE` a dos u ocho** y observa cómo empeora la fidelidad de la reconstrucción a medida que el cuello de botella se estrecha, una comprobación directa de que el modelo se ve forzado a priorizar la información más relevante.

## Resumen de puntos clave

- El etiquetado manual de datos es caro, lento y no escala al volumen de datos sin etiquetar disponible en el mundo real; el aprendizaje no supervisado busca aprovechar esos datos sin necesidad de etiquetas
- El aprendizaje no supervisado se distingue del supervisado y del RL por el tipo de retroalimentación: no hay etiquetas explícitas ni recompensa, solo los propios datos de entrada
- Evaluar un algoritmo no supervisado es especialmente difícil porque no existe una verdad de referencia única; en la práctica se recurre a índices de calidad de clúster, al error de reconstrucción o al rendimiento de un clasificador entrenado con pocas etiquetas sobre la representación aprendida
- Una buena representación, según pistas convergentes de la neurociencia, la física y la propia IA, debería estar desenredada, soportar atención y clustering, incorporar información latente relevante para posibles tareas futuras y ser compositiva
- Un autocodificador aprende comprimiendo la entrada en un código latente (codificador) y reconstruyéndola a partir de ese código (decodificador), minimizando el error de reconstrucción; el cuello de botella lo obliga a priorizar la información más relevante
- El autocodificador variacional (VAE) trata el espacio latente de forma probabilística y se entrena maximizando la cota ELBO, que combina un término de reconstrucción y una divergencia KL que regulariza la posterior aproximada hacia el *prior* elegido
- Aumentar el peso del término KL (beta-VAE) favorece representaciones desenredadas, verificables mediante recorridos latentes, y mejora la transferencia de políticas de RL de la simulación a la realidad
- Las GAN modelan los datos sin pérdida de reconstrucción explícita mediante un juego minimax entre generador y discriminador; variantes como BiGAN o BigBiGAN añaden un codificador para poder inferir representaciones a partir de datos reales
- El aprendizaje autosupervisado crea tareas pretexto (rotación, colorización, orden temporal, predicción de tokens enmascarados como en BERT) que generan supervisión automáticamente a partir de la propia estructura de los datos
- El aprendizaje contrastivo, formalizado con la pérdida InfoNCE y popularizado por métodos como word2vec, CPC y SimCLR, acerca representaciones de pares positivos y aleja representaciones de pares negativos, sin necesidad de un modelo generativo completo

## Ponlo en práctica

**Dedica entre quince y treinta minutos a este ejercicio antes de seguir al siguiente capítulo.**

1. **Copia el código de la sección anterior en un cuaderno de Google Colab y ejecútalo tal cual**, comprobando que las reconstrucciones finales se parecen razonablemente a los dígitos originales
2. **Reduce `DIM_LATENTE` de treinta y dos a dos** y vuelve a entrenar el modelo durante las mismas diez épocas. Compara visualmente las reconstrucciones con las del experimento original: ¿qué tipo de información crees que se pierde primero cuando el cuello de botella es tan estrecho?
3. **Añade una línea de código tras el entrenamiento** que extraiga el vector latente `z` de doscientas imágenes de prueba (usando la segunda salida de `modelo(entrada_plana)`) y represéntalo en dos dimensiones con `sklearn.decomposition.PCA` o `matplotlib.pyplot.scatter`, coloreando cada punto según su etiqueta real (aunque el modelo nunca la haya visto). Observa si los dígitos iguales tienden a agruparse en el espacio latente
4. **Sustituye `nn.BCELoss()` por `nn.MSELoss()`** y vuelve a entrenar. Anota si cambia visiblemente la nitidez de las reconstrucciones
5. **Reflexiona por escrito, en dos o tres frases**, sobre qué tendrías que cambiar en este autocodificador para acercarlo a un autocodificador variacional: piensa en qué capa habría que modificar para que el codificador produzca una media y una varianza en lugar de un único vector determinista, y qué término tendrías que añadir a la función de pérdida
