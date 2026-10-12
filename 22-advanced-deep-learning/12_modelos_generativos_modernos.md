# Capítulo 12: Modelos generativos modernos de variables latentes

## Por qué importa este tema

En los capítulos anteriores conociste dos formas de generar datos nuevos con una red neuronal: el autoencoder, que aprende a comprimir y reconstruir una entrada, y la red generativa adversaria (GAN), que enfrenta un generador contra un discriminador hasta que el primero produce muestras difíciles de distinguir de las reales. Ambas ideas funcionan, pero dejan huecos importantes. El autoencoder normal no te da ningún control sobre lo que ocurre entre dos puntos de su espacio latente: no sabes qué imagen obtendrías si mueves un poco la representación interna de un dato. La GAN, por su parte, no te ofrece una forma de calcular la probabilidad de una observación ni una manera directa de codificar un dato existente en su espacio latente.

Este capítulo, correspondiente a la Clase 11 del curso ("Modelos generativos avanzados"), cierra ambos huecos con la familia de los modelos modernos de variables latentes: el autoencoder variacional (VAE, por sus siglas en inglés, *Variational Autoencoder*) y los flujos normalizadores (*normalizing flows*). Estos modelos no solo generan datos verosímiles, sino que estructuran el espacio latente como un mapa continuo y navegable, donde moverte suavemente de un punto a otro produce variaciones suaves y coherentes del dato generado.

Esa propiedad tiene consecuencias prácticas muy concretas. En diseño de fármacos y biotecnología, un espacio latente continuo permite interpolar entre dos moléculas conocidas y explorar variantes intermedias que conservan propiedades químicas deseables, en lugar de saltar a puntos arbitrarios sin sentido estructural. En diagnóstico por imagen y otras aplicaciones de salud, permite generar variaciones sintéticas de una misma condición para ampliar conjuntos de datos de entrenamiento. En creación de contenido, permite interpolar de forma creativa entre dos caras, dos estilos o dos sonidos y obtener resultados intermedios plausibles, no ruido. Y en detección de anomalías o control de calidad, el propio modelo te da una medida de cuánto se aleja un dato nuevo de lo que ha aprendido como normal. **Presta atención especial a la diferencia entre autoencoder y autoencoder variacional**: es la clave que abre todo lo demás en este capítulo.

## Explicación conceptual con analogías

### El espacio latente como mapa: direcciones fijas frente a barrios completos

Imagina que quieres representar cada foto de un rostro con una dirección postal en un mapa de ciudad. Un autoencoder normal actúa como un funcionario que, para cada foto, escribe una única dirección exacta: la calle y el número donde "vive" esa foto en el mapa. El problema es que ese funcionario coloca las direcciones sin preocuparse por el barrio: dos fotos muy parecidas podrían acabar en calles opuestas de la ciudad, y el espacio entre dos direcciones cualesquiera puede ser un descampado vacío, sin ningún significado. Si te paras en un punto del mapa donde no hay ninguna dirección registrada y le pides al decodificador que "lea" ese punto, obtienes ruido, no una cara plausible.

El autoencoder variacional cambia la regla del juego: en lugar de asignar una dirección exacta, asigna a cada foto un barrio entero, representado como una distribución de probabilidad (normalmente una campana de Gauss) centrada en un punto del mapa, con un radio de incertidumbre. Además, el entrenamiento empuja a todos esos barrios a solaparse de forma ordenada alrededor del centro del mapa, siguiendo una distribución de referencia común. El resultado es un mapa sin descampados: cualquier punto que elijas, esté o no exactamente sobre el barrio de una foto de entrenamiento, cae dentro de algún barrio o en la frontera entre varios, y el decodificador sabe interpretarlo como una cara razonable. Esta es la diferencia esencial entre ambos modelos: un autoencoder normal aprende un punto fijo por dato, mientras que uno variacional aprende una distribución de probabilidad en el espacio latente. Esa propiedad es precisamente lo que te permite caminar de un barrio a otro, mezclando dos rostros de forma gradual, o lanzar un dardo al azar sobre el mapa completo y obtener siempre una cara nueva y coherente.

### El truco de reparametrización: mover el dado fuera de la ecuación

Entrenar una red neuronal exige poder calcular gradientes y propagarlos hacia atrás por cada operación de la red. El problema es que, dentro de un autoencoder variacional, hay un paso de muestreo aleatorio: eliges un punto al azar dentro del barrio de cada foto. El muestreo aleatorio, tal cual, bloquea la propagación del gradiente, porque no existe una derivada bien definida de "un número elegido al azar" respecto a los parámetros que definen ese azar.

La solución, conocida como truco de reparametrización, es una separación de responsabilidades muy elegante. En lugar de pedirle a la red que "tire el dado" directamente, separas el proceso en dos partes: por un lado, un generador de ruido externo y fijo, que no depende de ningún parámetro entrenable, como una moneda que lanzas al aire sin que tu decisión influya en el resultado; por otro lado, una transformación determinista y diferenciable que toma ese ruido y los parámetros aprendidos (el centro del barrio y su radio de incertidumbre) y produce el punto final del espacio latente. Como esa transformación es una simple fórmula matemática diferenciable, el gradiente sí puede atravesarla sin problema, y el azar queda aislado, fuera del camino por el que circula la información de aprendizaje.

### Flujos normalizadores: origami que nunca pierde información

La fuente complementaria de este capítulo, la lección de Andriy Mnih para UCL x DeepMind, presenta otra familia de modelos de variables latentes modernos: los flujos normalizadores o modelos invertibles. Imagina una lámina de plastilina con forma de esfera perfecta, que representa una distribución simple y conocida. Un flujo normalizador la deforma mediante una secuencia de pliegues y estiramientos, cada uno reversible, hasta darle la forma irregular de tus datos reales, por ejemplo, el contorno de todos los dígitos manuscritos posibles. Como cada pliegue es invertible, en cualquier momento puedes deshacer la secuencia completa y recuperar exactamente la esfera original: no se pierde ni se crea plastilina en el proceso, solo se redistribuye su volumen. Esa conservación exacta es lo que permite calcular la probabilidad exacta de cualquier observación, algo que ni el autoencoder variacional ni la GAN pueden ofrecerte con la misma precisión.

## Desarrollo teórico

### El marco general de los modelos de variables latentes

Un modelo de variable latente define una distribución sobre las observaciones **x** introduciendo una variable latente **z**, típicamente un vector, que no se observa directamente. El modelo queda completamente especificado por dos piezas:

- La distribución previa (*prior*) de la variable latente, p(z), normalmente una gaussiana estándar
- La verosimilitud p(x|z), la red que conecta la variable latente con la observación

Juntas definen la distribución conjunta:

p(x, z) = p(x|z) · p(z)

A partir de la conjunta se derivan dos distribuciones fundamentales: la verosimilitud marginal p(x), que indica cuán probable es una observación bajo el modelo y que se maximiza durante el entrenamiento, y la distribución posterior p(z|x), que representa los valores latentes plausibles que podrían haber generado una observación concreta, es decir, la explicación del dato en términos del modelo. Generar una observación nueva es sencillo: primero muestreas z desde el prior, después muestreas x desde la verosimilitud p(x|z). El problema aparece al invertir ese proceso: calcular la posterior exige conocer la marginal,

p(z|x) = p(x, z) / p(x), con p(x) = ∫ p(x|z) p(z) dz

Esa integral, salvo en casos especiales (mezclas de gaussianas, modelos lineales-gaussianos, modelos invertibles), es intratable: implica integrar sobre un espacio de alta dimensión una función no lineal, y ni el cálculo analítico ni la integración numérica son viables. Para superar esta barrera existen dos estrategias generales, exactamente las dos familias que cubre este capítulo: diseñar modelos donde la inferencia sea tratable por construcción (los flujos normalizadores), o aceptar que el modelo es intratable y recurrir a inferencia aproximada (la inferencia variacional, base del autoencoder variacional).

### Modelos invertibles y la fórmula de cambio de variable

Un flujo normalizador genera observaciones aplicando una transformación f, invertible y diferenciable, a una muestra del prior: x = f(z). Como f es invertible, existe correspondencia uno a uno entre cada observación y su configuración latente, y la inferencia se resuelve simplemente invirtiendo la función: z = f⁻¹(x). Para entrenar por máxima verosimilitud necesitas relacionar la densidad de z con la densidad de x. Esa relación es la fórmula de cambio de variable:

log p(x) = log p(z) − log |det (∂f(z) / ∂z)|, con z = f⁻¹(x)

El determinante del Jacobiano de f actúa como factor de corrección de volumen: da cuenta de cómo la transformación estira o comprime el espacio en el entorno de cada punto, de modo que la densidad resultante siga integrando uno, igual que la densidad original. Para que el entrenamiento sea eficiente, tanto f y su inversa como el determinante de su Jacobiano —y los gradientes de ambos— deben calcularse de forma rápida.

El ejemplo más simple de modelo invertible es el análisis de componentes independientes (ICA, *Independent Component Analysis*), que usa un prior factorizado y una transformación lineal mediante una matriz cuadrada W: x = W·z. La inferencia se reduce a invertir la matriz: z = W⁻¹·x. Un caso de uso clásico es el problema del cóctel: separar las voces individuales de varios invitados a partir de las grabaciones mezcladas de varios micrófonos, algo posible gracias a la linealidad de la acústica del sonido, siempre que el prior no sea gaussiano.

Los flujos normalizadores generales se construyen encadenando transformaciones invertibles simples, ya que la composición de funciones invertibles es también invertible. Su ventaja es rara en el terreno de los modelos generativos: combinan alta capacidad expresiva con verosimilitud exacta y tratable. Su limitación principal es estructural: la dimensión del espacio latente debe coincidir exactamente con la de los datos, las variables latentes deben ser continuas, y alcanzar suficiente capacidad expresiva exige encadenar muchas capas, lo que dispara el número de parámetros y el consumo de memoria.

### Inferencia variacional: convertir la inferencia en optimización

Cuando el modelo es intractable, existen dos grandes familias de inferencia aproximada. Los métodos de Monte Carlo vía cadenas de Markov (MCMC, *Markov Chain Monte Carlo*) generan muestras de la posterior exacta dejando evolucionar una cadena de Markov; son generales y exactos en el límite de tiempo infinito, pero computacionalmente costosos y difíciles de diagnosticar. La inferencia variacional, en cambio, aproxima la posterior verdadera p(z|x) con una distribución tratable q_φ(z|x), llamada posterior variacional, parametrizada por φ. Puedes usar cualquier distribución para q, siempre que puedas muestrear de ella y calcular su log-probabilidad y el gradiente de esa log-probabilidad respecto a φ. La opción por defecto es una distribución totalmente factorizada, conocida como aproximación de campo medio.

¿Qué objetivo optimizas si la verosimilitud marginal log p(x) es intratable? La respuesta es una cota inferior. Para cualquier densidad q(z) con soporte que cubra el de p(z|x), puedes escribir:

log p(x) = log ∫ p(x, z) dz = log ∫ q(z) · [p(x, z) / q(z)] dz ≥ ∫ q(z) · log [p(x, z) / q(z)] dz

La desigualdad aplica la desigualdad de Jensen: el logaritmo de una esperanza es mayor o igual que la esperanza del logaritmo, porque el logaritmo es una función cóncava. El término de la derecha recibe el nombre de límite inferior de la evidencia (ELBO, *Evidence Lower Bound*), y es válido para cualquier q y para cualquier valor de los parámetros del modelo. Usando concretamente la posterior variacional q_φ(z|x), el ELBO se escribe:

ELBO(θ, φ; x) = E_{q_φ(z|x)} [log p_θ(x, z) − log q_φ(z|x)]

Existen otras cotas variacionales, como la cota ponderada por importancia (IWAE), que generaliza el ELBO usando varias muestras y permite ajustar la firmeza de la cota, pero el ELBO es, con diferencia, la más simple y la más utilizada.

### La divergencia de Kullback-Leibler

Para entender qué mide exactamente el ELBO necesitas la divergencia de Kullback-Leibler (KL), que cuantifica cuánto difiere una distribución q de otra distribución p:

KL(q(z) ‖ p(z)) = ∫ q(z) · log [q(z) / p(z)] dz = E_{q(z)} [log q(z) − log p(z)]

La divergencia KL cumple dos propiedades importantes: es no negativa para cualquier par de distribuciones q y p, y vale cero si y solo si q y p coinciden en casi todo punto. Debes recordar también que no es una métrica en sentido estricto, porque no es simétrica: en general, KL(q‖p) es distinta de KL(p‖q).

### Descomposición del ELBO y el hueco variacional

Reescribiendo la conjunta como p(x, z) = p(x) · p(z|x), el ELBO se descompone en dos términos:

ELBO(θ, φ; x) = log p_θ(x) − KL(q_φ(z|x) ‖ p_θ(z|x))

El primer término, la log-verosimilitud marginal, no depende de los parámetros variacionales φ. El segundo es la divergencia KL entre la posterior variacional y la posterior exacta, conocida como hueco variacional. De esta descomposición se sigue algo notable: maximizar el ELBO respecto a φ equivale exactamente a minimizar el hueco variacional, es decir, a acercar la posterior variacional a la posterior verdadera. Y esto es posible aunque ni la posterior exacta ni su divergencia KL respecto a q sean calculables por separado: el ELBO, al ser una diferencia entre dos cantidades intratables, resulta tratable. Cuando actualizas los parámetros del modelo θ para aumentar el ELBO, ese aumento puede provenir de dos fuentes: una mejora real de la log-verosimilitud marginal (justo lo que persigue el aprendizaje por máxima verosimilitud) o una reducción del hueco variacional lograda deformando el modelo para que su posterior verdadera se parezca más a la familia de q, lo que a veces sacrifica capacidad de ajuste a los datos. Por eso conviene usar la posterior variacional más expresiva que puedas permitirte computacionalmente.

Un fenómeno asociado, llamado poda variacional o colapso de la posterior en el contexto de los autoencoders variacionales, ocurre cuando el modelo deja de usar algunas dimensiones latentes: para esas dimensiones, la posterior variacional termina siendo idéntica al prior. Puede ser deseable, porque ajusta automáticamente la dimensionalidad efectiva del espacio latente, o indeseable, porque limita la capacidad del modelo de explotar toda la capacidad latente disponible.

### Inferencia amortizada y estimación de gradientes

En la inferencia variacional clásica, cada observación requiere su propio conjunto de parámetros variacionales, ajustados mediante una optimización independiente. Esto es muy ineficiente a gran escala. La inferencia amortizada resuelve el problema entrenando una red neuronal, llamada red de inferencia o codificador, que toma la observación x como entrada y produce directamente los parámetros de su posterior variacional, en lugar de recalcularlos desde cero para cada dato. Esta idea, introducida en el contexto de las máquinas de Helmholtz en los años noventa, es justo lo que popularizó el autoencoder variacional: un único conjunto de pesos, compartido entre todas las observaciones, sustituye a miles de optimizaciones independientes.

Entrenar con inferencia variacional exige calcular el gradiente del ELBO respecto a θ y respecto a φ. El gradiente respecto a los parámetros del modelo θ es sencillo, porque la distribución sobre la que se toma la esperanza, q_φ(z|x), no depende de θ: basta con mover el gradiente dentro de la esperanza, muestrear de q y promediar los gradientes del logaritmo de la conjunta. El gradiente respecto a los parámetros variacionales φ es más delicado, porque ahora es la propia distribución sobre la que integras la que depende del parámetro que quieres derivar. Existen dos grandes familias de estimadores insesgados para este caso: el estimador REINFORCE (o de razón de verosimilitud), muy general —funciona con variables discretas y continuas— pero de varianza alta; y el estimador de reparametrización (o de camino), menos general —exige variables latentes continuas y una función diferenciable— pero de varianza mucho más baja, lo que en la práctica lo convierte en la opción preferida siempre que es aplicable.

El truco de reparametrización, ya descrito con la analogía del dado, expresa una muestra de q_φ(z|x) como una transformación determinista y diferenciable g_φ de una variable de ruido ε con distribución fija, independiente de φ:

z = g_φ(ε, x), con ε ~ p(ε)

Para una gaussiana univariante, la reparametrización más habitual es:

z = μ_φ(x) + σ_φ(x) · ε, con ε ~ N(0, 1)

Esta transformación es diferenciable tanto respecto a μ como a σ, cumpliendo el requisito para propagar gradientes. Muchas distribuciones de la familia de localización-escala, como la Laplace o la Cauchy, se reparametrizan de forma análoga; otras, como la Gamma o la Dirichlet, no permiten separar el azar de los parámetros de forma explícita, aunque existen variantes de reparametrización implícita. Las variables discretas, en general, no admiten este truco porque la transformación resultante no es diferenciable. Frameworks como PyTorch implementan la reparametrización de forma automática para las distribuciones continuas más comunes.

### El autoencoder variacional

Un autoencoder variacional es, formalmente, un modelo generativo de variables latentes continuas en el que tanto la verosimilitud (decodificador) como la posterior variacional (codificador) están parametrizadas por redes neuronales. Tanto el prior como la posterior variacional suelen modelarse como gaussianas totalmente factorizadas:

- Prior: p(z) = N(z; 0, I)
- Codificador / posterior variacional: q_φ(z|x) = N(z; μ_φ(x), diag(σ_φ(x)²))
- Decodificador / verosimilitud: para datos binarios, una Bernoulli parametrizada por la red; para datos de valor real, una gaussiana cuya media y varianza diagonal calcula la red

El tipo concreto de red —capas totalmente conectadas, convolucionales, residuales— es un detalle de implementación que no altera la estructura matemática del modelo. Lo esencial es que el codificador aprende a producir, para cada x, los parámetros μ_φ(x) y σ_φ(x) de su barrio en el mapa latente, y el decodificador aprende a reconstruir x a partir de un punto muestreado de ese barrio.

El objetivo de entrenamiento, el ELBO para VAE, se descompone en dos términos tratables:

ELBO(θ, φ; x) = E_{q_φ(z|x)} [log p_θ(x|z)] − KL(q_φ(z|x) ‖ p(z))

El primer término mide cuán bien reconstruye el modelo la observación a partir de una muestra de la posterior variacional; se le conoce como error de reconstrucción negativo, y valores altos indican buena reconstrucción. El segundo término actúa como regularizador: penaliza que la posterior variacional se aleje del prior, evitando que el espacio latente codifique más información de la observación de la estrictamente necesaria, y es justo el mecanismo que mantiene el mapa latente continuo y sin descampados. Cuando tanto el prior como la posterior son gaussianas, este término tiene forma cerrada y no requiere muestreo:

KL(q_φ(z|x) ‖ p(z)) = ½ Σ_j (σ_j² + μ_j² − 1 − log σ_j²)

donde la suma recorre cada una de las dimensiones del vector latente. Entrenar un VAE consiste, por tanto, en maximizar la suma de ambos términos (o, de forma equivalente, minimizar su negativo) con descenso de gradiente, usando la reparametrización para que el error se propague tanto al codificador como al decodificador.

Desde su introducción en 2014, el VAE ha evolucionado hasta convertirse en un marco más que en un modelo único: cualquier modelo de variables latentes continuas entrenado con inferencia variacional amortizada y reparametrización cuenta como un VAE. Las extensiones más relevantes documentadas incluyen múltiples capas latentes para representaciones jerárquicas, variables latentes no gaussianas, priors y posteriores más expresivos —por ejemplo, mediante flujos normalizadores—, arquitecturas más potentes como las redes residuales, decodificadores autorregresivos que combinan las fortalezas de ambas familias de modelos, esquemas de inferencia variacional iterativa con varias actualizaciones en lugar de una sola, y técnicas de reducción de varianza para acelerar el entrenamiento.

## Ejemplo de código comentado, en PyTorch

El siguiente ejemplo implementa un autoencoder variacional pequeño y lo entrena sobre el conjunto de dígitos manuscritos MNIST, disponible directamente a través de `torchvision`. **Copia y ejecuta este código en Google Colab**, activando un entorno con GPU si está disponible, para ver reconstrucciones y muestras generadas en pocos minutos.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Datos: MNIST, imágenes de 28x28 en escala de grises, valores en [0, 1] ---
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

LATENT_DIM = 20  # dimensión del espacio latente z


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=LATENT_DIM):
        super().__init__()
        # Codificador: de la imagen aplanada a una representación oculta compartida
        self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
        # Dos cabezas separadas: una para la media, otra para el log de la varianza
        # (se predice el logaritmo de la varianza, no la varianza directamente,
        # para que la red pueda producir cualquier número real y aun así obtengamos
        # una varianza siempre positiva al exponenciar)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decodificador: del espacio latente de vuelta a los píxeles
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.encoder_hidden(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # Truco de reparametrización: z = mu + sigma * epsilon, con epsilon ~ N(0, I)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = F.relu(self.decoder_hidden(z))
        # Sigmoid porque el decodificador modela una Bernoulli por píxel (valores en [0, 1])
        return torch.sigmoid(self.decoder_out(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparameterize(mu, logvar)
        x_reconstruida = self.decode(z)
        return x_reconstruida, mu, logvar


def elbo_loss(x_reconstruida, x, mu, logvar):
    # Término de reconstrucción: log-verosimilitud de una Bernoulli, sumada sobre píxeles
    reconstruccion = F.binary_cross_entropy(
        x_reconstruida, x.view(x.size(0), -1), reduction="sum"
    )
    # KL entre q(z|x) = N(mu, sigma^2) y el prior p(z) = N(0, I), forma cerrada
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruccion + kl  # minimizar esto equivale a maximizar el ELBO


modelo = VAE().to(device)
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    modelo.train()
    perdida_total = 0.0
    for x, _ in train_loader:  # las etiquetas no se usan: el VAE es no supervisado
        x = x.to(device)
        optimizador.zero_grad()
        x_reconstruida, mu, logvar = modelo(x)
        perdida = elbo_loss(x_reconstruida, x, mu, logvar)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item()

    print(f"Época {epoch}/{EPOCHS} — pérdida media: {perdida_total / len(train_data):.2f}")

# --- Visualización: reconstrucciones frente a originales ---
modelo.eval()
with torch.no_grad():
    x_test, _ = next(iter(test_loader))
    x_test = x_test.to(device)
    x_reconstruida, _, _ = modelo(x_test)

n = 8
fig, ejes = plt.subplots(2, n, figsize=(n * 1.2, 3))
for i in range(n):
    ejes[0, i].imshow(x_test[i].cpu().squeeze(), cmap="gray")
    ejes[0, i].axis("off")
    ejes[1, i].imshow(x_reconstruida[i].cpu().view(28, 28), cmap="gray")
    ejes[1, i].axis("off")
ejes[0, 0].set_ylabel("Original")
ejes[1, 0].set_ylabel("Reconstruida")
plt.suptitle("Fila superior: dígitos originales — fila inferior: reconstrucciones del VAE")
plt.show()

# --- Muestras nuevas: dígitos generados desde puntos al azar del espacio latente ---
with torch.no_grad():
    z_aleatorio = torch.randn(n, LATENT_DIM).to(device)  # muestreo directo del prior N(0, I)
    muestras = modelo.decode(z_aleatorio).cpu()

fig, ejes = plt.subplots(1, n, figsize=(n * 1.2, 1.6))
for i in range(n):
    ejes[i].imshow(muestras[i].view(28, 28), cmap="gray")
    ejes[i].axis("off")
plt.suptitle("Dígitos generados desde puntos muestreados del espacio latente")
plt.show()
```

Con diez épocas y una arquitectura tan pequeña, no esperes dígitos perfectos, pero verás con claridad dos cosas: las reconstrucciones se parecen razonablemente a los originales, y las muestras generadas desde puntos completamente aleatorios del espacio latente producen formas reconocibles como dígitos, aunque a veces ambiguas entre dos clases, precisamente porque el espacio latente es continuo y sin descampados.

## Resumen de puntos clave

- Un autoencoder normal asigna a cada dato un punto fijo del espacio latente; un autoencoder variacional le asigna una distribución de probabilidad, lo que convierte el espacio latente en un mapa continuo apto para interpolar y generar
- Un modelo de variable latente queda definido por el prior p(z), la verosimilitud p(x|z) y su producto, la conjunta p(x, z); la inferencia consiste en calcular la posterior p(z|x), y suele ser intratable por la integral que define la marginal p(x)
- Existen dos estrategias frente a la intratabilidad: modelos tratables por construcción, como los flujos normalizadores, que usan transformaciones invertibles y la fórmula de cambio de variable para calcular verosimilitudes exactas, o inferencia aproximada mediante inferencia variacional
- La inferencia variacional aproxima la posterior verdadera con una posterior variacional q_φ(z|x) y convierte el problema de inferencia en un problema de optimización
- El límite inferior de la evidencia (ELBO) se obtiene aplicando la desigualdad de Jensen y es una cota inferior tratable de la log-verosimilitud marginal; se descompone en log p(x) menos la divergencia KL entre la posterior variacional y la posterior exacta
- La divergencia de Kullback-Leibler mide la diferencia entre dos distribuciones, es siempre no negativa, se anula solo cuando ambas distribuciones coinciden, y no es simétrica
- El truco de reparametrización separa el azar (una variable de ruido fija) de los parámetros entrenables, permitiendo propagar gradientes con varianza baja a través del muestreo; en un VAE se traduce en z = μ + σ · ε
- El objetivo de entrenamiento del VAE combina un término de reconstrucción y un término de regularización KL respecto al prior, con forma cerrada cuando ambas distribuciones son gaussianas
- Los flujos normalizadores garantizan verosimilitud exacta e inferencia sencilla, pero exigen que la dimensión latente coincida con la de los datos y suelen requerir muchas capas y parámetros
- El VAE es hoy un marco extensible: admite múltiples capas latentes, posteriores y priors más expresivos, decodificadores autorregresivos y otras mejoras arquitectónicas

## Ponlo en práctica

**Modifica el código del VAE que acabas de entrenar** y responde a estas dos preguntas con tus propios experimentos, en no más de treinta minutos:

1. Cambia `LATENT_DIM` de veinte a dos dimensiones y vuelve a entrenar el modelo. Con solo dos dimensiones latentes puedes dibujar directamente el espacio latente como un plano: pasa todo el conjunto de test por `modelo.encode(...)`, toma las medias `mu` resultantes y haz un `plt.scatter` coloreando cada punto según su etiqueta de dígito (la etiqueta que descartaste en el bucle de entrenamiento). **Observa si los dígitos similares —por ejemplo, el cuatro y el nueve— quedan cerca entre sí en el mapa**
2. Con ese mismo modelo de dos dimensiones, elige dos dígitos de test distintos, codifica ambos para obtener sus medias μ₁ y μ₂, y genera diez puntos intermedios interpolando linealmente entre ambos (`z_t = (1 - t) * mu1 + t * mu2`, con t entre cero y uno). Decodifica cada punto intermedio y muéstralos en fila con `matplotlib`. **Comprueba que la transición entre ambos dígitos es gradual, no un salto brusco**, y compara mentalmente qué ocurriría si intentaras el mismo experimento con un autoencoder normal del capítulo anterior, donde no existe ninguna garantía de que el camino entre dos puntos pase por regiones con significado.
