# 07. Aprendizaje estructurado y adversarial

## Por qué importa este tema

Hasta ahora has entrenado modelos que tratan cada ejemplo de forma aislada: le enseñas una imagen, una fila de un CSV o una frase, y el modelo aprende a partir de sus características (features) sin más contexto. Pero en muchos problemas reales existe información adicional que el modelo no está aprovechando. Si clasificas artículos científicos, sabes qué artículo cita a cuál. Si clasificas opiniones de películas, puedes calcular qué opiniones se parecen entre sí aunque no compartan ni una palabra exacta. Y si tu modelo va a producción, alguien puede intentar engañarlo con entradas manipuladas a propósito.

Neural Structured Learning (NSL, Aprendizaje Estructurado Neuronal) es una extensión de TensorFlow (TF) diseñada para aprovechar precisamente ese tipo de información: relaciones entre ejemplos (grafos) y robustez frente a manipulaciones (aprendizaje adversarial). En este capítulo aprenderás a añadir ambas señales a un modelo de Keras que ya sabes construir, sin rediseñarlo desde cero.

## Qué aporta NSL frente al entrenamiento estándar

En el entrenamiento estándar, la función de pérdida (loss) solo mide el error del modelo en cada ejemplo por separado: predicción frente a etiqueta real. NSL añade un segundo término a esa pérdida, un término de regularización, que premia o castiga al modelo según cómo se comporta en relación con otros ejemplos.

Ese término de regularización puede venir de dos fuentes:

- **Una estructura (grafo)**: el modelo aprende a dar predicciones parecidas para ejemplos que están conectados en un grafo, ya sea porque la conexión existe en los datos (grafo natural) o porque la creas tú a partir de una medida de similitud (grafo sintetizado)
- **Una perturbación adversarial**: el modelo aprende a dar predicciones parecidas para un ejemplo y una versión de ese mismo ejemplo ligeramente manipulada para intentar confundirlo

En los dos casos, la receta es la misma y es la gran ventaja práctica de NSL: no necesitas cambiar la arquitectura del modelo. Construyes tu red con la API (Application Programming Interface, Interfaz de Programación de Aplicaciones) de Keras que ya conoces (secuencial, funcional o por subclases) y luego la envuelves con una clase del framework NSL (`GraphRegularization` o `AdversarialRegularization`). Esa clase añade el término de regularización a la pérdida de forma automática. El resultado es otro modelo de Keras que compilas, entrenas y evalúas exactamente igual que cualquier otro.

Esto importa especialmente cuando tienes pocos datos etiquetados, una situación habitual en proyectos reales, donde etiquetar cuesta tiempo y dinero. Al aprovechar relaciones entre ejemplos etiquetados y no etiquetados, NSL permite que el modelo generalice mejor incluso con una fracción pequeña de etiquetas, y a la vez lo hace más robusto ante datos ruidosos o manipulados.

## Entrenamiento con grafos naturales

Un **grafo natural** es una red de relaciones que ya existe en tus datos, sin que tengas que inventarla. Los nodos del grafo son tus ejemplos (documentos, personas usuarias, imágenes) y las aristas (edges) son relaciones reales entre ellos: citas entre artículos científicos, conexiones en una red social, interacciones entre proteínas.

Imagina que clasificas artículos científicos por tema, pero solo tienes etiqueta para una parte pequeña de ellos. Si el artículo A cita al artículo B, es razonable pensar que ambos tratan un tema parecido, aunque A no tenga etiqueta. Esa es la intuición detrás de la regularización de grafos: anima al modelo a dar predicciones parecidas para dos nodos conectados por una arista, de forma que la información fluya desde los ejemplos etiquetados hacia sus vecinos sin etiqueta.

El ejemplo de referencia de NSL para grafos naturales usa el conjunto de datos Cora, un grafo de citas entre 2.708 artículos de aprendizaje automático clasificados en siete categorías. El proceso, en pasos:

1. **Aumentar los datos con vecinos**: por cada ejemplo, añades las características de sus vecinos en el grafo a los datos de entrenamiento. NSL hace esto con la función `nsl.tools.pack_nbrs`, que combina el grafo de citas y las características de cada artículo (en Cora, una representación de qué palabras del vocabulario contiene) en un único conjunto de datos aumentado
2. **Definir un modelo base**: una red normal de Keras. En el ejemplo de Cora es un perceptrón multicapa (MLP, Multi-Layer Perceptron, una red densa de varias capas totalmente conectadas), pero puede ser cualquier arquitectura
3. **Envolver el modelo con `nsl.keras.GraphRegularization`**, pasándole una configuración (`nsl.configs.make_graph_reg_config`) que fija, entre otros, cuántos vecinos usar y el peso relativo del término de regularización de grafos frente a la pérdida principal
4. **Compilar, entrenar y evaluar** el modelo envuelto igual que cualquier modelo de Keras

Con esta técnica, el modelo de Cora mejora su precisión (accuracy) unos 2-3 puntos porcentuales frente al mismo modelo sin regularización de grafos, simplemente por aprovechar la estructura de citas que ya estaba en los datos.

## Entrenamiento con grafos sintetizados

¿Qué pasa si tus datos no traen un grafo de serie? La mayoría no lo traen: una colección de opiniones de películas, un conjunto de imágenes médicas o un histórico de ventas no vienen con "aristas" explícitas. Para esos casos, NSL permite **sintetizar** un grafo a partir de la similitud entre ejemplos.

La idea es parecida a organizar una fiesta sin lista de invitados previa: en vez de usar relaciones que ya existen, decides tú el criterio de cercanía (por ejemplo, "personas con aficiones parecidas se sientan juntas") y construyes las conexiones a partir de ese criterio. En NSL, el criterio de cercanía es una medida de similitud entre vectores.

El proceso, siguiendo el ejemplo de clasificación de sentimiento sobre el conjunto de datos IMDb (reseñas de películas de Internet Movie Database):

1. **Generar embeddings** de cada ejemplo: una representación numérica densa que captura su significado. En el ejemplo de IMDb se usa un modelo preentrenado de TensorFlow Hub (embeddings Swivel) para convertir el texto de cada reseña en un vector
2. **Construir el grafo a partir de la similitud entre embeddings**, normalmente con similitud coseno (cosine similarity): si dos ejemplos superan un umbral de similitud, NSL añade una arista entre ellos. Esto se hace con `nsl.tools.build_graph_from_config`
3. **Aumentar los datos con los vecinos del grafo sintetizado**, igual que en el caso de grafos naturales, con `nsl.tools.pack_nbrs`
4. **Definir el modelo base, envolverlo con `GraphRegularization` y entrenar**, exactamente igual que con un grafo natural

Lo importante es que, una vez construido el grafo (natural o sintetizado), el resto del proceso de regularización de grafos es idéntico. La diferencia está solo en cómo obtienes las aristas: de una relación real en el caso natural, o de una medida de similitud calculada por ti en el caso sintetizado. De hecho, en el ejemplo de IMDb esta técnica resulta especialmente útil cuando hay pocas etiquetas disponibles: con solo un 1% de los datos etiquetados, un modelo con regularización de grafos llega a superar en unos 20 puntos porcentuales de precisión al mismo modelo sin regularización.

## Regularización adversarial: aprender a resistir ataques

Hay un tercer escenario: no tienes ningún grafo, ni natural ni fácil de sintetizar, pero te preocupa que alguien manipule las entradas de tu modelo a propósito para hacerlo fallar. Aquí entra el **aprendizaje adversarial**.

Un **ejemplo adversarial** es una versión de un ejemplo original con una perturbación (un cambio) tan pequeña que un ojo humano apenas la nota, pero calculada específicamente para confundir al modelo. La perturbación no es ruido aleatorio: se calcula siguiendo la dirección que más aumenta el error del modelo (la dirección inversa del gradiente), es decir, el cambio más "eficaz" posible para provocar una predicción incorrecta.

La analogía más clara es la vacunación: expones al modelo a una versión "debilitada" del ataque (una perturbación pequeña y controlada) durante el entrenamiento, para que aprenda a reconocer y resistir esa familia de ataques cuando aparezcan versiones reales más adelante.

Aquí está la conexión con todo lo anterior de este capítulo: NSL trata cada ejemplo adversarial como un **vecino** de su ejemplo original, construyendo así una estructura sobre la marcha, ejemplo a ejemplo, incluso cuando no existe un grafo explícito. Al conectar el ejemplo original con su vecino adversarial, el modelo aprende a mantener la misma predicción para ambos, con la misma lógica de "predicciones parecidas para nodos conectados" que viste en la regularización de grafos.

TensorFlow dispone además de librerías específicas para generar ataques adversariales más elaborados, como cleverhans, que implementa métodos como el Fast Gradient Sign Method (FGSM, Método del Signo del Gradiente Rápido) o el Projected Gradient Descent (PGD, Descenso de Gradiente Proyectado), pensados para evaluar a fondo la robustez de un modelo. NSL, en cambio, integra su propio mecanismo de generación de vecinos adversariales directamente en el flujo de entrenamiento, con un enfoque más sencillo de usar para regularizar un modelo mientras lo entrenas.

En la práctica, el proceso con NSL es:

1. **Definir un modelo base** con Keras, como siempre
2. **Crear una configuración adversarial** con `nsl.configs.make_adv_reg_config`, indicando el peso del término adversarial en la pérdida (`multiplier`) y el tamaño del paso de la perturbación (`adv_step_size`)
3. **Envolver el modelo base con `nsl.keras.AdversarialRegularization`**
4. **Compilar, entrenar y evaluar** igual que siempre

El resultado, probado sobre el conjunto de datos MNIST (imágenes de dígitos manuscritos del cero al nueve), es contundente: un modelo base bien entrenado alcanza una precisión del 99% sobre datos limpios, pero esa precisión cae hasta cerca del 51% en cuanto se le presentan imágenes con perturbaciones adversariales. El mismo modelo entrenado con regularización adversarial mantiene una precisión del 95% frente a esas mismas imágenes perturbadas. La red no se ha vuelto más precisa en el caso normal, pero sí mucho más difícil de engañar.

## Ejemplo de código comentado

El siguiente ejemplo, adaptado del tutorial oficial de NSL, entrena una red neuronal convolucional (CNN, Convolutional Neural Network, una red con capas de convolución especializadas en detectar patrones visuales) sobre MNIST y añade regularización adversarial con solo unas líneas de código extra.

```python
# Instala la librería de Neural Structured Learning si aún no está disponible
!pip install --quiet neural-structured-learning

import neural_structured_learning as nsl  # convención habitual: abreviar a "nsl"
import tensorflow as tf
import tensorflow_datasets as tfds

# 1. Cargamos y normalizamos MNIST igual que en cualquier problema de clasificación de imágenes
datasets = tfds.load('mnist')
train_dataset, test_dataset = datasets['train'], datasets['test']

def normalize(features):
    features['image'] = tf.cast(features['image'], tf.float32) / 255.0
    return features

train_dataset = train_dataset.map(normalize).shuffle(10000).batch(32)
test_dataset = test_dataset.map(normalize).batch(32)

# 2. Definimos un modelo base normal, con la Functional API de Keras
def build_base_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name='image')
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

base_model = build_base_model()

# 3. Configuramos la regularización adversarial:
#    - multiplier: peso del término adversarial en la pérdida total
#    - adv_step_size: magnitud de la perturbación que se aplica a cada imagen
adv_config = nsl.configs.make_adv_reg_config(
    multiplier=0.2,
    adv_step_size=0.2,
    adv_grad_norm='infinity')

# 4. Envolvemos el modelo base con AdversarialRegularization.
#    label_keys indica qué feature del diccionario contiene la etiqueta,
#    porque el modelo adversarial necesita la etiqueta para calcular la perturbación.
adv_model = nsl.keras.AdversarialRegularization(
    base_model,
    label_keys=['label'],
    adv_config=adv_config)

# El modelo adversarial espera diccionarios {'image': ..., 'label': ...} en vez de tuplas
train_set_adv = train_dataset.map(lambda f: {'image': f['image'], 'label': f['label']})
test_set_adv = test_dataset.map(lambda f: {'image': f['image'], 'label': f['label']})

# 5. Compilamos, entrenamos y evaluamos exactamente igual que un modelo de Keras normal
adv_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
adv_model.fit(train_set_adv, epochs=5)
adv_model.evaluate(test_set_adv)
```

Fíjate en que los cambios respecto a un entrenamiento estándar son mínimos: una llamada para crear la configuración (`make_adv_reg_config`) y otra para envolver el modelo (`AdversarialRegularization`). Todo lo demás —compilar, entrenar, evaluar— es el flujo de Keras que ya conoces.

## Resumen de puntos clave

- El entrenamiento estándar evalúa cada ejemplo por separado; NSL añade un término de regularización a la pérdida para que el modelo también aprenda de las relaciones entre ejemplos
- Un grafo natural usa relaciones que ya existen en los datos (citas, conexiones sociales); un grafo sintetizado se construye artificialmente a partir de la similitud entre embeddings cuando no existe una relación explícita
- En ambos casos, el flujo es el mismo: aumentar los datos con los vecinos del grafo (`pack_nbrs`), definir un modelo base de Keras y envolverlo con `nsl.keras.GraphRegularization`
- La regularización adversarial entrena al modelo con ejemplos ligeramente perturbados a propósito (vecinos adversariales), usando `nsl.keras.AdversarialRegularization`, para que aprenda a mantener la misma predicción pese a esas perturbaciones
- En el ejemplo de MNIST, un modelo sin regularización adversarial cae del 99% al 51% de precisión frente a imágenes perturbadas, mientras que el modelo regularizado se mantiene en torno al 95%
- Ninguna de estas técnicas exige rediseñar tu red: envuelves el modelo base ya existente con una clase de NSL y sigues compilando, entrenando y evaluando como siempre

## Ponlo en práctica

Antes de tocar código, razona sobre estos tres escenarios y decide, para cada uno, si usarías un grafo natural, un grafo sintetizado o regularización adversarial:

1. Un hospital quiere clasificar historiales clínicos por especialidad. Los historiales no citan a otros historiales, pero un modelo de lenguaje puede calcular qué tan parecido es el texto de dos historiales
2. Una red social quiere predecir el interés de cada persona usuaria en un tema a partir de los intereses declarados por las personas que ya sigue
3. Una entidad bancaria quiere que su modelo de detección de fraude no falle si alguien intenta modificar ligeramente los datos de una transacción para colarse por debajo del radar

Después, abre el notebook de práctica de esta clase y comprueba tu razonamiento sobre el tercer escenario ajustando el hiperparámetro `adv_multiplier` en el ejemplo de MNIST: entrena el modelo con un valor bajo (por ejemplo, 0.05) y con uno alto (por ejemplo, 0.5), y compara cuánto cae la precisión sobre imágenes perturbadas en cada caso.
