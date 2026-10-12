# Redes generativas antagónicas (GAN): el arte de aprender jugando

## Por qué importa este tema

Hasta ahora has visto redes que clasifican, predicen o traducen. En este capítulo cambias de perspectiva: vas a estudiar redes que crean. Una red generativa antagónica (GAN, del inglés *Generative Adversarial Network*) no responde a la pregunta "¿qué es esto?", sino a una mucho más ambiciosa: "¿cómo genero algo nuevo que parezca real?".

Desde que Ian Goodfellow y su equipo presentaron la idea en 2014, las GAN han pasado de generar dígitos borrosos en blanco y negro a producir rostros humanos fotorrealistas, escenas completas o traducciones entre estilos de imagen. Ese salto explica por qué este tema importa tanto en la práctica:

- **Generación de imágenes y arte con inteligencia artificial (IA)**. Herramientas de creación visual, diseño de producto o prototipado usan variantes de GAN para producir texturas, rostros sintéticos o ilustraciones a partir de una simple semilla de ruido
- **Aumento de datos sintéticos**. Cuando entrenas un modelo con pocos ejemplos reales (imágenes médicas, defectos de fabricación, señales poco frecuentes), una GAN bien entrenada te permite generar ejemplos adicionales realistas para reforzar el conjunto de entrenamiento
- **Traducción y edición de imagen**. Convertir un boceto en una fotografía, cambiar de estación un paisaje o transferir el estilo de un pintor a otra imagen son aplicaciones directas de esta familia de modelos
- **Deepfakes**. La misma capacidad que genera un rostro sintético convincente puede sustituir el rostro de una persona real en un vídeo sin su consentimiento

Ese último punto no es un detalle menor. La misma tecnología que permite aumentar datos médicos escasos permite también fabricar evidencia audiovisual falsa. Cuanto mejor entiendes cómo funciona el generador y qué lo hace convincente, mejor preparado o preparada estás para reconocer sus límites, detectar sus artefactos y valorar los riesgos que trae consigo. Por eso este capítulo es también la antesala del bloque dedicado a la IA responsable: no puedes razonar sobre los riesgos de una tecnología si no entiendes primero cómo funciona por dentro.

## Explicación conceptual con analogías

Imagina a un o una falsificadora de billetes y a la policía que intenta detectarla. Al principio, la falsificadora produce billetes muy toscos: papel incorrecto, tinta que se corre, proporciones erróneas. La policía los detecta sin esfuerzo. Pero la falsificadora aprende de cada rechazo: ajusta el papel, corrige el color, mejora los detalles. La policía, a su vez, también aprende: afina su criterio, se fija en marcas de agua, compara con más cuidado. Con el tiempo, ambas partes mejoran en paralelo, y los billetes falsos se vuelven cada vez más difíciles de distinguir de los auténticos.

Esa es exactamente la dinámica de una GAN. El **generador** cumple el papel de la falsificadora: parte de ruido aleatorio y produce datos —imágenes, en el caso típico— que intenta hacer pasar por reales. El **discriminador** cumple el papel de la policía: recibe tanto datos reales como datos generados y debe decidir, para cada muestra, si procede del conjunto de datos verdadero o de la fábrica del generador.

Ninguna de las dos redes ve el objetivo final de la otra de forma directa. El generador nunca observa datos reales: solo recibe la señal que le da el discriminador sobre si sus creaciones cuelan o no. El discriminador, por su parte, funciona como un **profesor** que no solo separa lo real de lo falso, sino que además proporciona un gradiente —una dirección de mejora— que el generador usa para corregir su próxima tanda de "billetes". Cuanto mejor discrimina la policía, más información recibe la falsificadora sobre en qué está fallando; cuanto mejor falsifica la falsificadora, más exigente debe volverse la policía. El entrenamiento termina, idealmente, cuando la policía ya no puede hacer mejor que lanzar una moneda al aire: en ese punto, los billetes falsos y los reales son estadísticamente indistinguibles.

## Desarrollo teórico

### Modelos generativos explícitos e implícitos

Un modelo generativo trata de aprender, a partir de una muestra de datos, la distribución subyacente que los produjo. Existen dos grandes familias. Los **modelos explícitos** aprenden directamente una distribución de probabilidad sobre los datos: puedes preguntarles por la verosimilitud de un punto nuevo y, además, muestrear de ellos. Entran en esta categoría el análisis de componentes principales probabilístico, el análisis factorial, las mezclas de gaussianas o modelos neuronales como PixelCNN, PixelRNN o WaveNet, entrenados por máxima verosimilitud.

Los **modelos implícitos**, en cambio, no modelan explícitamente la densidad de probabilidad. En su lugar, aprenden un simulador capaz de generar muestras nuevas con las mismas propiedades estadísticas que los datos originales, sin pasar en ningún momento por una fórmula de probabilidad. La GAN es el ejemplo más representativo de esta segunda familia.

### La arquitectura y el objetivo minimax

Una GAN está formada por dos redes neuronales profundas entrenadas de forma simultánea:

- El **generador** $G$ toma un vector de ruido latente $z$, muestreado de una distribución previa conocida $p(z)$ —típicamente una gaussiana multivariante— y lo transforma mediante una red determinista en una muestra generada $G(z)$. El ruido de entrada es lo que aporta variedad: a diferencia de un autocodificador variacional, el generador de una GAN no produce una distribución sobre la salida, así que toda la aleatoriedad de los datos debe entrar por $z$
- El **discriminador** $D$ recibe una muestra $x$ —real o generada— y devuelve un valor $D(x)$ interpretable como la probabilidad de que esa muestra provenga del conjunto de datos real y no del generador

El objetivo conjunto se formula como un problema minimax de suma cero:

$$\min_G \max_D \; V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

Léelo por partes. El primer término premia al discriminador cuando asigna alta probabilidad a los datos reales. El segundo lo premia cuando asigna baja probabilidad —es decir, detecta correctamente— a los datos generados. El discriminador quiere **maximizar** esa suma: quiere clasificar bien en ambos frentes. El generador, en cambio, quiere **minimizarla**: quiere que $D(G(z))$ se acerque a uno, de modo que el segundo término se haga muy negativo y el discriminador se equivoque.

En la práctica, los dos objetivos se separan en dos funciones de pérdida que se optimizan de forma alternada. El discriminador maximiza:

$$\max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

y el generador minimiza:

$$\min_G \; \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

Ese segundo término tiene un problema práctico: al inicio del entrenamiento, cuando el generador es malo, $D(G(z))$ es casi cero y el gradiente de $\log(1-D(G(z)))$ es muy plano, así que apenas empuja al generador a mejorar. Por eso, en la práctica se usa una variante **no saturante**, en la que el generador maximiza $\log D(G(z))$ en lugar de minimizar $\log(1-D(G(z)))$: el objetivo apunta en la misma dirección, pero da gradientes mucho más útiles al principio del entrenamiento.

### El bucle de entrenamiento

Entrenar una GAN significa alternar pasos de descenso (o ascenso) de gradiente estocástico entre las dos redes:

1. Muestrea un lote de datos reales y un lote de ruido; genera datos falsos con $G$
2. Actualiza $D$ con unos pocos pasos de gradiente para maximizar su capacidad de distinguir reales de generados
3. Muestrea un nuevo lote de ruido, genera datos con $G$ y actualiza $G$ para minimizar la probabilidad de que $D$ acierte
4. Repite el ciclo miles o millones de veces

No se entrena el discriminador hasta la optimalidad en cada paso: hacerlo sería computacionalmente muy costoso y, además, un discriminador perfecto puede dejar de aportar gradiente útil al generador. Por eso el equilibrio práctico consiste en dar solo unos pocos pasos de gradiente al discriminador antes de actualizar el generador.

Este proceso configura un **juego de suma cero**: lo que gana el generador lo pierde el discriminador, y viceversa. El objetivo teórico es alcanzar un **equilibrio de Nash**, un punto en el que ninguna de las dos redes puede mejorar su resultado cambiando unilateralmente su estrategia mientras la otra se mantiene fija. En ese equilibrio, el discriminador ya no logra distinguir datos reales de generados mejor que al azar.

### GAN como minimización de una divergencia

Los modelos generativos, en general, se pueden entender como la minimización de una distancia o divergencia entre la distribución real de los datos y la distribución que produce el modelo. El enfoque más habitual, la máxima verosimilitud, equivale a minimizar la divergencia de Kullback-Leibler (KL):

$$D_{KL}(p \| q) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]$$

donde $p$ es la distribución real de los datos y $q$ la distribución inducida por el modelo.

Se puede demostrar que, si el discriminador de una GAN alcanza su valor óptimo, el generador está minimizando de forma implícita la divergencia de Jensen-Shannon (JS) entre la distribución real y la generada:

$$D_{JS}(p \| q) = \tfrac{1}{2} D_{KL}(p \| m) + \tfrac{1}{2} D_{KL}(q \| m), \qquad m = \tfrac{p+q}{2}$$

Esta divergencia vale cero solo cuando ambas distribuciones son idénticas, lo que respalda intuitivamente el objetivo del entrenamiento. Sin embargo, esta conexión teórica descansa en un supuesto que rara vez se cumple: que el discriminador sea óptimo en cada paso. En la práctica, el discriminador tiene capacidad limitada, se entrena con recursos computacionales finitos y solo dispone de muestras, no de la distribución real completa.

### Por qué el entrenamiento de una GAN es tan inestable

Aquí llegas al núcleo del problema que motiva buena parte de los avances descritos más abajo. Las divergencias KL y JS comparten una debilidad: si la distribución del modelo y la distribución real no se solapan —algo muy habitual al inicio del entrenamiento, cuando el generador produce datos absurdos—, el ratio de probabilidades que aparece en su definición se dispara a infinito. El resultado es que la divergencia deja de dar una señal de aprendizaje útil: aunque el modelo se acerque progresivamente a los datos reales, la divergencia puede seguir siendo infinita o mantenerse sin cambios apreciables, así que el gradiente no informa sobre el progreso real.

Esta fragilidad teórica se traduce en varios síntomas prácticos que verás una y otra vez al entrenar GAN:

- **Inestabilidad del juego minimax**. Como las dos redes se actualizan de forma simultánea persiguiendo objetivos opuestos, el sistema puede oscilar sin converger, en vez de asentarse en el equilibrio de Nash esperado. Un discriminador que mejora demasiado rápido deja al generador sin gradiente útil; uno que mejora demasiado despacio deja pasar datos de mala calidad
- **Colapso de modo (*mode collapse*)**. El generador descubre que unas pocas salidas concretas engañan bien al discriminador y deja de explorar el resto de la distribución real. El resultado es una variedad muy limitada de muestras, aunque cada una por separado parezca convincente. Detectarlo es difícil porque la función de pérdida del generador no siempre refleja este problema de forma directa
- **Ausencia de una métrica de progreso fiable**. Al ser un juego entre dos jugadores que mejoran a la vez, el valor de la pérdida no decrece de forma monótona como en un problema de optimización clásico, así que no puedes leer directamente en la curva de pérdida si el modelo está mejorando

En la práctica, además, la GAN no realiza una minimización exacta de la divergencia JS, precisamente porque el discriminador no es una función arbitraria sino una red neuronal —o una función perteneciente a una familia acotada, como un espacio de Hilbert con núcleo reproductor—. Esa limitación, lejos de ser solo un defecto, resulta beneficiosa: al ser una aproximación suave, el discriminador no puede representar ratios infinitos, así que sigue proporcionando gradiente incluso cuando las distribuciones real y generada no se solapan todavía. Esto explica, en parte, por qué las GAN funcionan mejor en la práctica de lo que la teoría de la divergencia JS haría esperar.

### Avances que mejoran la estabilidad

La fuente describe varias líneas de trabajo, cada una atacando una causa distinta de inestabilidad.

**Redes generativas antagónicas profundas convolucionales (DCGAN)**. Sustituyen los perceptrones multicapa (MLP) originales por redes convolucionales tanto en el generador como en el discriminador, y añaden normalización por lotes (*batch normalization*), lo que suaviza notablemente el proceso de entrenamiento. Además, interpolar entre dos vectores de ruido produce imágenes intermedias semánticamente coherentes, señal de que el espacio latente aprendido tiene una estructura significativa.

**Red generativa antagónica de Wasserstein (WGAN)**. En lugar de la pérdida de entropía cruzada binaria basada en JS, usa la distancia de Wasserstein (también llamada distancia del movimiento de tierra), que no depende de un ratio de probabilidades sino de una diferencia de expectativas, por lo que sigue dando señal útil incluso sin solapamiento de soportes. Su formulación dual, restringida a funciones 1-Lipschitz —aquellas donde $|f(x_1) - f(x_2)| \le |x_1 - x_2|$—, es:

$$W(p, q) = \sup_{\|D\|_L \le 1} \; \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

El objetivo completo de la WGAN es entonces:

$$\min_G \max_{D \in \text{1-Lipschitz}} \; \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

Aquí el discriminador —llamado "crítico" en este contexto— debe mantenerse 1-Lipschitz, lo que se impone mediante recorte de pesos (*weight clipping*) o, de forma más eficaz, mediante penalización del gradiente.

**Otras divergencias**. La fuente también menciona la discrepancia media máxima (MMD, *maximum mean discrepancy*), que opera en un espacio de Hilbert con núcleo reproductor (RKHS), y las f-divergencias, entrenables mediante una cota variacional inferior similar a la usada en autocodificadores variacionales. Todas comparten una misma estructura: minimización respecto al generador, maximización respecto al discriminador.

**Redes generativas antagónicas condicionales (cGAN)**. Añaden información auxiliar $y$ —una etiqueta de clase, por ejemplo— tanto al generador como al discriminador, lo que permite controlar qué se genera en lugar de dejarlo librado al azar del ruido:

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x \mid y)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z \mid y) \mid y))]$$

**Normalización espectral (SNGAN)**. Acota a uno los valores singulares de los pesos del discriminador en cada capa, lo que impide que este resuelva su objetivo simplemente aumentando la magnitud de sus pesos y lo obliga a mejorar de verdad las señales de gradiente que transmite al generador.

**Redes de atención (AttnGAN, SAGAN)**. Incorporan mecanismos de autoatención para capturar dependencias globales entre regiones distantes de la imagen —por ejemplo, que la cabeza y la cola de un animal guarden coherencia entre sí—, algo que las capas convolucionales, limitadas a su campo receptivo local, no logran por sí solas.

**Crecimiento progresivo (PGGAN) y StyleGAN**. Las GAN de crecimiento progresivo entrenan primero a baja resolución (por ejemplo, 4x4 píxeles) y añaden capas gradualmente hasta alcanzar alta resolución, lo que estabiliza el entrenamiento en cada etapa. StyleGAN, sobre esa misma base, separa el espacio latente en un vector global —transformado mediante una red totalmente conectada en un espacio latente intermedio— y ruido espacial inyectado en cada capa, lo que ofrece control fino sobre atributos como la expresión facial o el peinado.

**BigGAN**. Escala el entrenamiento con lotes muy grandes (hasta 2048 muestras), arquitecturas más profundas y técnicas combinadas —función de pérdida bisagra, normalización espectral, autoatención, discriminador por proyección, regularización ortogonal— para generar muestras de alta fidelidad condicionadas por clase en conjuntos de datos tan diversos como ImageNet. Introduce además el "truco del truncamiento": reducir la desviación típica del ruido de entrada en el momento de la inferencia produce muestras más prototípicas y de mayor fidelidad, a costa de menor variedad.

**Técnicas contra el colapso de modo**. La fuente cita explícitamente la discriminación por minilote (*minibatch discrimination*), las GAN sin desenrollar (*unrolled GANs*) y la normalización espectral como técnicas orientadas a diversificar las muestras generadas y mitigar el colapso de modo.

### Cómo se evalúa una GAN

Al ser un modelo implícito, una GAN no permite calcular verosimilitudes, así que no puedes evaluarla como evaluarías un modelo explícito. La fuente describe dos métricas principales. La **puntuación Inception (IS)** usa un clasificador preentrenado en ImageNet para comparar, mediante divergencia KL, la distribución de etiquetas de los datos generados frente a las esperadas: premia diversidad de clases y confianza de clasificación, aunque no detecta si dentro de una misma clase las muestras son repetitivas. La **distancia de Fréchet entre Inception (FID)** compara, en cambio, la distribución de características internas del clasificador —no solo las etiquetas de salida— entre datos reales y generados, lo que captura mejor tanto la diversidad entre clases como dentro de cada clase; su variante corregida para tamaños de muestra pequeños es la distancia de Fréchet del núcleo (KID). Ninguna métrica por sí sola basta: la práctica habitual combina varias, junto con inspección de vecinos más cercanos en el espacio de características para comprobar que el modelo generaliza y no memoriza.

## Ejemplo de código comentado, en PyTorch

El siguiente ejemplo implementa una GAN mínima sobre datos sintéticos en dos dimensiones: una mezcla de ocho gaussianas dispuestas en círculo, el ejemplo de juguete clásico para observar tanto el aprendizaje como el riesgo de colapso de modo. El generador y el discriminador tienen solo un par de capas lineales. Puedes ejecutar el código directamente en Google Colab.

```python
# Instala/asegura las dependencias en Colab (normalmente ya están disponibles)
# !pip install torch matplotlib numpy -q

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Datos reales: mezcla de ocho gaussianas en círculo ---
def sample_real_data(n_samples, n_modes=8, radius=4.0, std=0.15):
    angles = 2 * np.pi * np.random.randint(0, n_modes, size=n_samples) / n_modes
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    points = centers + std * np.random.randn(n_samples, 2)
    return torch.tensor(points, dtype=torch.float32)

LATENT_DIM = 2

# --- Generador: ruido latente -> punto 2D ---
class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z):
        return self.net(z)

# --- Discriminador: punto 2D -> logit de "realidad" ---
class Discriminator(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),  # logit, sin sigmoide: se aplica dentro de la pérdida
        )

    def forward(self, x):
        return self.net(x)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

# BCEWithLogitsLoss combina sigmoide + entropía cruzada binaria de forma numéricamente estable
criterion = nn.BCEWithLogitsLoss()
opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

BATCH_SIZE = 256
N_STEPS = 6000

for step in range(1, N_STEPS + 1):
    # ---------- 1) Actualiza el discriminador ----------
    real_data = sample_real_data(BATCH_SIZE).to(device)
    z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
    fake_data = generator(z).detach()  # detach: el generador no se actualiza en este paso

    real_labels = torch.ones(BATCH_SIZE, 1, device=device)
    fake_labels = torch.zeros(BATCH_SIZE, 1, device=device)

    loss_d_real = criterion(discriminator(real_data), real_labels)
    loss_d_fake = criterion(discriminator(fake_data), fake_labels)
    loss_d = loss_d_real + loss_d_fake

    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()

    # ---------- 2) Actualiza el generador (pérdida no saturante) ----------
    z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
    fake_data = generator(z)
    # El generador "quiere" que D clasifique sus muestras como reales (etiqueta 1)
    loss_g = criterion(discriminator(fake_data), real_labels)

    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()

    if step % 1000 == 0:
        print(f"paso {step:5d} | pérdida D: {loss_d.item():.3f} | pérdida G: {loss_g.item():.3f}")

# --- Visualización: datos reales frente a muestras generadas antes y después del entrenamiento ---
with torch.no_grad():
    real_sample = sample_real_data(1000).numpy()
    z_vis = torch.randn(1000, LATENT_DIM, device=device)
    generated_sample = generator(z_vis).cpu().numpy()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(real_sample[:, 0], real_sample[:, 1], s=10, alpha=0.5, label="Datos reales")
ax.scatter(generated_sample[:, 0], generated_sample[:, 1], s=10, alpha=0.5, label="Muestras generadas")
ax.set_title("GAN mínima: datos reales frente a datos generados")
ax.legend()
ax.set_aspect("equal")
plt.show()
```

Al ejecutarlo, observa la evolución de las dos pérdidas impresas cada mil pasos: rara vez decrecen de forma monótona, justo el comportamiento inestable que describe la teoría. En el gráfico final, comprueba si el generador cubre los ocho modos de la mezcla o si, por el contrario, concentra sus muestras solo en dos o tres de ellos: ese es el colapso de modo en acción, y con esta arquitectura tan simple es habitual que aparezca en alguna de las ejecuciones.

## Resumen de puntos clave

- Una GAN es un modelo generativo implícito compuesto por un generador y un discriminador que compiten en un juego minimax de suma cero
- El objetivo clásico es $\min_G \max_D \; \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$, y en la práctica el generador se entrena con una variante no saturante para evitar gradientes planos al inicio
- Si el discriminador fuese óptimo, el generador minimizaría la divergencia de Jensen-Shannon, pero en la práctica el discriminador nunca es óptimo, así que la GAN no realiza una minimización exacta de divergencia
- La falta de solapamiento entre la distribución real y la generada hace que las divergencias KL y JS pierdan utilidad como señal de aprendizaje, una de las raíces de la inestabilidad del entrenamiento
- El colapso de modo, la inestabilidad del equilibrio del juego y la ausencia de una métrica de progreso fiable son los tres problemas centrales del entrenamiento de GAN
- La WGAN sustituye la divergencia JS por la distancia de Wasserstein, calculada sobre funciones 1-Lipschitz, lo que ofrece un gradiente más estable
- DCGAN, cGAN, SNGAN, SAGAN, PGGAN, StyleGAN, BigGAN y LOGAN son avances descritos en la fuente, cada uno atacando una causa distinta de inestabilidad o de baja calidad de muestra
- Evaluar una GAN exige combinar varias métricas —puntuación Inception, distancia de Fréchet entre Inception (FID), inspección de vecinos más cercanos—, porque ninguna captura por sí sola calidad y diversidad a la vez
- La misma capacidad generativa que produce datos sintéticos útiles habilita también los deepfakes, lo que conecta directamente este tema con la ética y la responsabilidad en el desarrollo de IA

## Ponlo en práctica

**Dedica entre quince y treinta minutos a experimentar con el código anterior.** Sigue estos pasos:

1. **Ejecuta** el ejemplo tal cual en Google Colab y observa el gráfico final
2. **Reduce** `N_STEPS` a 500 y vuelve a ejecutar: comprueba si el generador solo llega a cubrir uno o dos modos de la mezcla, un colapso de modo provocado por entrenamiento insuficiente
3. **Modifica** la arquitectura del generador para que tenga una sola capa oculta en lugar de dos, y compara la calidad final de las muestras
4. **Cambia** la pérdida del generador de la variante no saturante (`criterion(discriminator(fake_data), real_labels)`) a la variante original minimizando `criterion(discriminator(fake_data), fake_labels)` con signo invertido, y observa si el entrenamiento se vuelve más lento o inestable al principio

**Anota** en dos o tres frases qué diferencia observaste entre las variantes y por qué crees que ocurre, relacionándolo con lo explicado sobre gradientes planos y colapso de modo en el desarrollo teórico.
