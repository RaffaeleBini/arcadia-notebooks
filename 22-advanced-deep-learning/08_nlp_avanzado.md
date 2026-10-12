# Capítulo 8: Procesamiento del lenguaje natural con deep learning

## Por qué importa este tema

Escribes una frase en un traductor automático y, en menos de un segundo, obtienes una versión razonablemente fiel en otro idioma. Publicas una reseña de un producto y un sistema clasifica automáticamente si tu opinión es positiva o negativa. Le pides a un asistente de texto que resuma un informe de veinte páginas y te devuelve cinco líneas que capturan lo esencial. Ninguna de estas tareas parecía al alcance de una máquina hace apenas una década, y las tres pertenecen al campo del procesamiento del lenguaje natural (NLP, del inglés Natural Language Processing): la rama de la inteligencia artificial (IA) que se ocupa de que las máquinas entiendan, interpreten y generen lenguaje humano de forma útil.

El salto de calidad no es anecdótico. Entre 2010 y 2016, el número de artículos publicados en conferencias de referencia de NLP, como ACL o EMNLP, que incluían en su título las palabras "deep" o "neural" pasó de ser prácticamente cero a representar una fracción sustancial de toda la producción científica del campo. Y ese aumento de interés vino acompañado de mejoras reales: entre 2018 y 2019, el rendimiento de los mejores modelos en GLUE (General Language Understanding Evaluation), un conjunto de pruebas estándar que evalúa tareas como determinar si una frase implica a otra o clasificar un sentimiento como positivo o negativo, pasó de menos del 60 % a cerca del 85 % de acierto. Esa tendencia ascendente se ha mantenido desde entonces.

Detrás de estas cifras hay una pregunta que este capítulo responde paso a paso: ¿cómo consigue una red neuronal, que solo sabe multiplicar números y sumar vectores, procesar algo tan escurridizo como el significado de una palabra? En el capítulo anterior estudiaste las redes neuronales recurrentes (RNN) y sus variantes con memoria a corto y largo plazo (LSTM) y con unidades recurrentes con puertas (GRU), diseñadas precisamente para procesar secuencias. Ahora vas a ver cómo esas arquitecturas, junto con una pieza adicional llamada embedding, se aplican de forma concreta al lenguaje: cómo se convierte el texto en números que una red puede procesar, cómo esos números aprenden a capturar el significado y cómo se combinan en tareas como el modelado de lenguaje, la clasificación de texto o la traducción automática.

Este tema conecta directamente con problemas que ves a diario en el trabajo con empresas de los sectores de Salud-Cuidados, Agro-Mar Alimentación o administración pública: un asistente conversacional que orienta a un paciente, un sistema que analiza automáticamente comentarios de clientes sobre un producto agroalimentario, o una herramienta que clasifica y prioriza solicitudes ciudadanas. Entender cómo una red neuronal representa y procesa el lenguaje te permite evaluar con criterio qué modelo o qué proveedor encaja con cada necesidad, y anticipar dónde puede fallar el sistema. Además, este capítulo prepara el terreno para el siguiente, dedicado en profundidad al mecanismo de atención y a la arquitectura del transformador (Transformer), que hoy domina el estado del arte en NLP.

## Explicación conceptual con analogías

Imagina que, en lugar de guardar cada palabra como una etiqueta aislada sin relación con las demás, la colocas como un punto en un mapa. En ese mapa, las palabras con significados parecidos aparecen cerca unas de otras, y las que no tienen nada que ver quedan lejos: "perro" y "gato" están próximos, "perro" y "arquitectura" están en extremos opuestos. A ese mapa lo llamamos espacio de embeddings, y las coordenadas de cada palabra en ese espacio son su embedding: un vector de números reales que la red aprende durante el entrenamiento, no que nadie diseña a mano.

Este mapa no es un capricho técnico: refleja algo real sobre el lenguaje. Piensa en la palabra "cara". Puede referirse a la parte delantera de tu cabeza ("vi la expresión en su cara"), a la esfera de un reloj ("la cara del reloj marca las diez"), al acto de afrontar algo ("es hora de dar la cara") o a personas nuevas ("hay caras nuevas en la oficina"). Estos cuatro usos comparten algo (la idea de una superficie frontal que informa o representa) pero también difieren entre sí. Un sistema que tratara cada palabra como un símbolo discreto e independiente, sin relación con ningún otro, no podría capturar ni el parecido ni la diferencia entre estos sentidos. Un embedding, en cambio, puede situar todos los sentidos de "cara" en una zona cercana del mapa, pero no exactamente en el mismo punto, y dejar que el contexto de la frase decida cuál de ellos aplica en cada caso.

Esa dependencia del contexto es más fuerte de lo que parece a primera vista. El psicólogo David Rumelhart mostró en los años setenta que, ante una misma letra manuscrita ambigua (que podría leerse como "E", "V" o "W"), las personas la interpretan de forma distinta según la frase en la que aparece: en "Juan y María subieron a la colina", la leemos como una letra; en "el último evento fue el salto con pértiga", la leemos como otra. El cerebro no analiza cada símbolo de forma aislada, lo interpreta a la luz de todo lo que lo rodea. El lenguaje añade una complicación extra: estas interacciones relevantes pueden ser no locales, es decir, pueden darse entre palabras muy alejadas entre sí en la frase. En "el hombre que comió el pimiento estornudó", el verbo "estornudó" está pegado a "el pimiento", pero en realidad se refiere a "el hombre", varias palabras atrás. Además, comprendemos "el hombre que comió el pimiento estornudó" más rápido que "el gato que mordió al perro ladró", aunque ambas frases tengan la misma estructura gramatical, porque nuestro conocimiento del mundo nos dice que es habitual que alguien estornude tras comer algo picante, y muy raro que un gato ladre.

Por último, el significado de una combinación de palabras no siempre es la suma simple de sus partes. Una mascota típica (un perro o un gato) suele ser marrón, blanca o negra. Un pez es plateado o gris. Pero un "pez de mascota" (piensa en un pez de acuario) suele ser de colores vivos: naranja, verde, azul. La combinación de "mascota" y "pez" genera una propiedad que ninguno de los dos conceptos tiene por separado. Lo mismo ocurre con "planta carnívora": ni "planta" ni "carnívoro" por sí solos incluyen la idea de "comer insectos", pero la combinación sí. Un buen modelo de lenguaje no puede combinar significados con una única función fija para cualquier par de palabras: necesita aprender, a partir de la experiencia, cómo se combinan típicamente las cosas en el mundo real. Estas cuatro observaciones (las palabras tienen múltiples sentidos relacionados, la desambiguación depende del contexto, ese contexto puede ser no local, y la composición de significados depende de qué significados concretos se combinan) son la razón de fondo por la que las redes neuronales, con sus representaciones distribuidas y continuas, funcionan tan bien para el lenguaje, mucho mejor que los sistemas basados en reglas y símbolos discretos que dominaron el campo durante décadas.

## Desarrollo teórico

### De las palabras a los números: tokenización y vocabulario

Antes de que una red neuronal pueda procesar una frase, hay que convertirla en números. El primer paso se llama tokenización: dividir el texto en unidades discretas, llamadas tokens. Existen dos niveles habituales:

- Tokenización a nivel de carácter: cada letra se trata como una unidad independiente. El vocabulario es pequeño (unas decenas de símbolos), pero las secuencias resultantes son muy largas y la red debe aprender a componer palabras completas a partir de letras sueltas
- Tokenización a nivel de palabra: el texto se separa por espacios en blanco (y signos de puntuación) y cada palabra se trata como un símbolo. El vocabulario resultante es mucho mayor, pero cada token ya lleva incorporado un significado reconocible

En la práctica, muchos sistemas modernos usan un término medio, dividiendo palabras poco frecuentes en fragmentos más pequeños (subpalabras), pero para entender los fundamentos basta con quedarte con la idea de que hay que decidir un vocabulario: un conjunto finito de símbolos, típicamente construido escaneando un corpus (una colección grande de textos) y contando cuántas veces aparece cada palabra, para quedarte con las más frecuentes o con todas si los recursos lo permiten.

Una vez fijado el vocabulario, cada palabra recibe un índice numérico. La representación más directa es el vector one-hot: un vector tan largo como el tamaño del vocabulario, con todo ceros salvo un uno en la posición que corresponde a esa palabra. Si el vocabulario tiene 10 000 palabras, cada palabra es un vector de 10 000 números, con nueve mil novecientos noventa y nueve ceros y un único uno. Este vector es correcto, pero tiene un problema grave: todas las palabras están a la misma distancia unas de otras, como si "perro" y "gato" no tuvieran nada más en común que "perro" y "termodinámica". Es exactamente el problema de tratar las palabras como símbolos discretos que planteábamos en la sección anterior.

### El embedding: de un vector disperso a un vector denso

La solución es multiplicar ese vector one-hot por una matriz de pesos entrenable. Si llamamos V al tamaño del vocabulario y D a la dimensión del embedding (un valor que tú eliges, habitualmente entre unas decenas y unos pocos cientos), la primera capa de un modelo de lenguaje neuronal es una matriz de pesos E de tamaño V × D. El embedding de una palabra w, con vector one-hot xw, se obtiene así:

ew = ETxw

En la práctica esta multiplicación es solo una búsqueda en una tabla: como xw tiene un único uno, el resultado es simplemente la fila de E correspondiente al índice de esa palabra. Por eso, en PyTorch, esta operación se implementa con una capa llamada `torch.nn.Embedding`, que internamente no es más que una tabla de V filas por D columnas, donde cada fila es el vector denso de una palabra.

Lo importante es que esta matriz E no se fija a mano: se inicializa con valores aleatorios pequeños y se ajusta mediante retropropagación, igual que cualquier otro peso de la red, para minimizar la función de pérdida de la tarea que estés entrenando (predecir la siguiente palabra, clasificar un sentimiento, traducir una frase). Como consecuencia, el entrenamiento reubica las palabras en ese espacio de D dimensiones de forma que las que cumplen papeles semánticos o sintácticos parecidos terminan cerca entre sí, sin que nadie les diga explícitamente qué significa cada una.

Este fenómeno no es nuevo ni exclusivo de las arquitecturas actuales. Ya en 1991, Miikkulainen y Dyer construyeron un modelo neuronal de lenguaje con esta misma idea de representación distribuida, y a principios de los noventa Jeff Elman entrenó una red recurrente simple con el único objetivo de predecir la siguiente palabra de una secuencia; al analizar las representaciones internas que la red había aprendido, descubrió que palabras con funciones gramaticales y significados similares (sujetos, verbos, categorías semánticas) se agrupaban espontáneamente en el espacio geométrico. La red nunca recibió una etiqueta gramatical: la estructura emergió únicamente de intentar predecir qué palabra viene después.

### Word2vec: aprender embeddings directamente del contexto

Una forma popular de entrenar buenos embeddings, sin necesidad de resolver primero una tarea compleja como la traducción, es word2vec, publicado por un equipo de Google en 2013. La idea central es que el significado de una palabra queda bien reflejado por las palabras que suelen rodearla, y propone dos variantes complementarias:

- CBOW (Continuous Bag of Words): dado el contexto (las palabras vecinas), predice la palabra central que falta
- Skip-gram: dada una palabra, predice las palabras de su contexto

El objetivo del skip-gram, para un texto de T palabras w1, ..., wT y una ventana de contexto de tamaño c, es maximizar:

(1/T) · Σt=1..T Σ−c≤j≤c, j≠0 log p(wt+j | wt)

donde la probabilidad de cada palabra de contexto se calcula con una función softmax sobre todo el vocabulario:

p(wO | wI) = exp(v'wO · vwI) / Σw=1..V exp(v'w · vwI)

Aquí, vwI es el embedding de la palabra de entrada y v'wO es un segundo embedding asociado a la palabra de salida. Entrenar esta red con millones de frases hace que las palabras que aparecen en contextos parecidos ("rey" y "reina", "perro" y "gato") terminen con embeddings cercanos, e incluso permite operaciones aritméticas sorprendentes sobre el espacio vectorial, como que el vector de "rey" menos el vector de "hombre" más el vector de "mujer" se acerque al vector de "reina". Un enfoque alternativo, GloVe (Global Vectors for Word Representation), llega a resultados parecidos aprovechando estadísticas globales de co-ocurrencia de palabras en todo el corpus, en lugar de predicciones locales palabra a palabra.

Estos embeddings, tanto los de word2vec como los que aprende una red desde cero para su propia tarea, comparten una limitación importante: son estáticos. Cada palabra tiene un único vector, sin importar en qué frase aparezca, así que "banco" tendría la misma representación en "fui al banco a sacar dinero" y en "me senté en el banco del parque". Esta limitación es precisamente lo que motivará, en el capítulo siguiente, arquitecturas capaces de generar un embedding distinto para la misma palabra según el contexto concreto en el que aparece.

### Tareas típicas de NLP con deep learning

Con el texto ya convertido en una secuencia de embeddings, una red neuronal (una RNN, una LSTM, una GRU, o las arquitecturas más avanzadas que verás a continuación) puede abordar tareas muy distintas. Tres de las más habituales son:

**Modelado de lenguaje.** Consiste en estimar la probabilidad de una secuencia de palabras descomponiéndola en probabilidades condicionadas, palabra a palabra:

P(w1, ..., wT) = Πt=1..T P(wt | w1, ..., wt−1)

Entrenar un modelo para maximizar esta probabilidad sobre grandes cantidades de texto (sin necesidad de etiquetas manuales, porque la "etiqueta" de cada posición es simplemente la palabra siguiente del propio texto) le enseña a generar frases coherentes y es la base de sistemas de generación de texto y de corrección predictiva.

**Clasificación de texto.** Aquí el objetivo es asignar una etiqueta a una frase o documento completo: positivo o negativo en análisis de sentimiento, spam o no spam en filtrado de correo, la categoría temática de una noticia. Una arquitectura habitual toma los embeddings de la frase, los procesa con una capa recurrente o simplemente los combina (por ejemplo, promediándolos), y pasa el resultado por una o varias capas densas hasta obtener una probabilidad por clase. El entrenamiento minimiza la entropía cruzada entre la predicción y la etiqueta real:

L = − Σi yi · log(ŷi)

donde yi es uno si la frase pertenece a la clase i y cero en caso contrario, y ŷi es la probabilidad que predice el modelo para esa clase.

**Traducción automática.** Aquí la entrada y la salida son ambas secuencias, y de longitud distinta: una frase en español puede traducirse a una frase en inglés con más o menos palabras. Esta tarea da lugar a una familia de arquitecturas especialmente relevante para el resto del capítulo: los modelos de secuencia a secuencia.

### De secuencia a secuencia: el encoder-decoder y su límite

Un modelo de secuencia a secuencia (sequence-to-sequence, o seq2seq) resuelve el problema de mapear una secuencia de entrada a una secuencia de salida de longitud distinta combinando dos redes recurrentes, típicamente LSTM o GRU como las que estudiaste en el capítulo anterior:

- Un codificador (encoder) lee la frase de origen palabra a palabra y va actualizando su estado oculto en cada paso. Al terminar de leer toda la frase, su último estado oculto c debe resumir todo lo que necesita saberse de la frase de entrada
- Un decodificador (decoder), otra red recurrente, recibe ese vector c como punto de partida y genera la frase de destino palabra a palabra, condicionando cada palabra nueva en el vector c y en las palabras que ya ha generado:

P(yt | y<t, c)

Este esquema, propuesto a principios de la década de 2010, funcionó notablemente bien y fue durante varios años el estándar en traducción automática neuronal. Pero tiene un cuello de botella evidente: toda la información de la frase de origen, sea de cinco palabras o de cincuenta, tiene que comprimirse en un único vector c de tamaño fijo. Cuanto más larga es la frase, más difícil es que ese vector conserve los detalles relevantes, y el rendimiento del modelo se degrada de forma notable con frases largas. A esto se suma la limitación que ya conoces del capítulo anterior: incluso con puertas que mitigan el desvanecimiento del gradiente, a una red recurrente le sigue costando propagar información entre posiciones muy alejadas de la secuencia.

Esta doble limitación (el cuello de botella de un único vector de contexto y la dificultad de las redes recurrentes para relacionar posiciones lejanas) es exactamente el problema que resuelve el mecanismo de atención, que estudiarás en profundidad en el próximo capítulo. La idea, adelantada aquí solo como preludio, es dejar de forzar toda la frase de entrada en un único vector fijo y permitir que el decodificador, en cada paso, "mire hacia atrás" y pondere de forma distinta cada palabra de la entrada según su relevancia para la palabra que está generando en ese momento. Cuando ese principio se lleva al extremo, prescindiendo incluso de la recurrencia y calculando estas ponderaciones entre todas las posiciones de la secuencia en paralelo, se obtiene la arquitectura del transformador (Transformer), la base de modelos como BERT (Bidirectional Encoder Representations from Transformers) y GPT (Generative Pre-trained Transformer), que también conocerás más adelante en este manual. Por ahora, quédate con la idea central: la evolución de NLP con deep learning va de representar palabras sueltas como vectores, a combinar esos vectores con redes recurrentes para tareas de secuencia, a permitir que el modelo decida dinámicamente en qué partes de la entrada fijarse.

## Ejemplo de código comentado

Vamos a construir, de principio a fin, un clasificador de sentimiento minúsculo: recibe una frase corta en español y predice si expresa una opinión positiva o negativa. Usaremos `torch.nn.Embedding` para aprender representaciones de palabras desde cero, sobre un vocabulario hecho a mano a partir de un puñado de frases de ejemplo. Copia y ejecuta este código en una celda de Google Colab.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Datos de entrenamiento: frases minúsculas con su etiqueta ---
# Etiqueta 1 = sentimiento positivo, 0 = sentimiento negativo
frases = [
    "me encanta este producto",
    "excelente servicio y rapido",
    "una experiencia maravillosa",
    "lo recomiendo totalmente",
    "el equipo fue muy amable",
    "pesimo servicio y muy lento",
    "no lo recomiendo para nada",
    "una experiencia horrible",
    "el equipo fue muy grosero",
    "producto de mala calidad",
]
etiquetas = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# --- 2. Construimos el vocabulario a mano ---
# <PAD> rellena frases cortas para igualar longitudes dentro de un lote.
# <UNK> representa cualquier palabra no vista durante el entrenamiento.
palabras = sorted(set(palabra for frase in frases for palabra in frase.split()))
vocabulario = ["<PAD>", "<UNK>"] + palabras
indice_de = {palabra: i for i, palabra in enumerate(vocabulario)}
TAMANO_VOCAB = len(vocabulario)
print(f"Tamaño del vocabulario: {TAMANO_VOCAB} palabras")

def tokenizar(frase, longitud_max):
    """Convierte una frase en una lista de índices, rellenada con <PAD>
    hasta longitud_max."""
    indices = [indice_de.get(p, indice_de["<UNK>"]) for p in frase.split()]
    indices = indices[:longitud_max]
    indices += [indice_de["<PAD>"]] * (longitud_max - len(indices))
    return indices

LONGITUD_MAX = max(len(f.split()) for f in frases)
X = torch.tensor([tokenizar(f, LONGITUD_MAX) for f in frases], dtype=torch.long)
y = torch.tensor(etiquetas, dtype=torch.float32)

# --- 3. Definimos el modelo ---
class ClasificadorSentimiento(nn.Module):
    def __init__(self, tamano_vocab, dim_embedding):
        super().__init__()
        # padding_idx=0 evita que <PAD> reciba gradiente: no aporta significado
        self.embedding = nn.Embedding(tamano_vocab, dim_embedding, padding_idx=0)
        self.clasificador = nn.Linear(dim_embedding, 1)

    def forward(self, x):
        vectores = self.embedding(x)          # (lote, longitud, dim_embedding)
        promedio = vectores.mean(dim=1)        # promedio de la frase completa
        logit = self.clasificador(promedio)    # (lote, 1)
        return logit.squeeze(1)

DIM_EMBEDDING = 8
modelo = ClasificadorSentimiento(TAMANO_VOCAB, DIM_EMBEDDING)

# --- 4. Entrenamos ---
perdida_fn = nn.BCEWithLogitsLoss()  # combina sigmoide + entropía cruzada, más estable
optimizador = optim.Adam(modelo.parameters(), lr=0.05)

for epoca in range(200):
    optimizador.zero_grad()
    logits = modelo(X)
    perdida = perdida_fn(logits, y)
    perdida.backward()
    optimizador.step()
    if (epoca + 1) % 50 == 0:
        print(f"Época {epoca + 1}: pérdida = {perdida.item():.4f}")

# --- 5. Probamos con frases nuevas, no vistas en el entrenamiento ---
def predecir(frase):
    modelo.eval()
    with torch.no_grad():
        entrada = torch.tensor([tokenizar(frase, LONGITUD_MAX)], dtype=torch.long)
        probabilidad = torch.sigmoid(modelo(entrada)).item()
    etiqueta = "positivo" if probabilidad > 0.5 else "negativo"
    print(f"'{frase}' -> {etiqueta} (probabilidad = {probabilidad:.3f})")

predecir("me encanta el equipo")
predecir("una experiencia pesima")
predecir("servicio amable y rapido")
```

Fíjate en dos decisiones de diseño. Primero, `nn.Embedding(TAMANO_VOCAB, DIM_EMBEDDING)` crea exactamente la matriz E de V × D que viste en la teoría: una tabla con una fila por palabra del vocabulario y ocho columnas (el valor de `DIM_EMBEDDING`), inicializada al azar y ajustada por retropropagación durante el entrenamiento. Segundo, en lugar de una capa recurrente, este ejemplo promedia los embeddings de la frase (`vectores.mean(dim=1)`) antes de clasificar; es una simplificación deliberada para que el ejemplo sea legible en pocas líneas, pero pierde por completo la información del orden de las palabras. **Sustituye esa línea por una capa `nn.LSTM` o `nn.GRU`** de las que ya conoces del capítulo anterior y comprueba cómo cambia el comportamiento del modelo con frases donde el orden importa.

## Resumen de puntos clave

- El procesamiento del lenguaje natural (NLP) busca que las máquinas entiendan y generen lenguaje humano; el deep learning ha llevado el rendimiento en benchmarks como GLUE de menos del 60 % a cerca del 85 % en apenas un año
- Las palabras no son símbolos discretos e independientes: tienen sentidos múltiples y relacionados, su desambiguación depende del contexto (a veces no local), y la forma en que se combinan sus significados depende de qué palabras concretas se combinan
- La tokenización convierte el texto en una secuencia de símbolos discretos (a nivel de carácter o de palabra) a partir de un vocabulario construido sobre un corpus
- Un embedding sustituye el vector disperso one-hot por un vector denso de dimensión D, obtenido como una fila de una matriz de pesos E de tamaño V × D que la red aprende por retropropagación; palabras con funciones o significados parecidos terminan cerca en ese espacio
- Word2vec (con sus variantes CBOW y skip-gram) y GloVe son métodos clásicos para entrenar embeddings a partir de estadísticas de coocurrencia en grandes corpus, aunque producen representaciones estáticas: un único vector por palabra, sin importar el contexto
- Las tareas típicas de NLP con deep learning incluyen el modelado de lenguaje (predecir la siguiente palabra), la clasificación de texto (por ejemplo, análisis de sentimiento) y la traducción automática
- Los modelos de secuencia a secuencia (seq2seq) combinan un codificador y un decodificador recurrentes para traducir entre secuencias de longitud distinta, pero sufren un cuello de botella: toda la frase de origen debe comprimirse en un único vector de contexto de tamaño fijo
- Esa limitación, junto con la dificultad de las redes recurrentes para relacionar posiciones lejanas, motiva el mecanismo de atención y la arquitectura del transformador, que estudiarás en el próximo capítulo

## Ponlo en práctica

Dedica entre quince y treinta minutos a esta mini-tarea en Google Colab, ampliando el código de la sección anterior.

1. Abre un cuaderno nuevo en Google Colab y copia el código del ejemplo
2. Añade al menos seis frases nuevas a la lista `frases` (tres positivas y tres negativas), en español y con vocabulario distinto al del ejemplo original, y actualiza `etiquetas` en consecuencia
3. Vuelve a entrenar el modelo y comprueba si acierta con frases que no estaban en el conjunto de entrenamiento
4. Cambia `DIM_EMBEDDING` a un valor mucho menor (por ejemplo, dos) y a uno mucho mayor (por ejemplo, treinta y dos); observa cómo cambian la velocidad de entrenamiento y la pérdida final
5. **Localiza en el código la línea que promedia los embeddings de la frase** (`vectores.mean(dim=1)`) y sustitúyela por una capa `nn.LSTM` seguida de la extracción de su último estado oculto; entrena de nuevo y anota si el modelo distingue mejor frases donde el orden de las palabras cambia el significado, como "el servicio no fue amable" frente a "el servicio fue amable"
6. Escribe, en una celda de texto del propio cuaderno, dos o tres frases explicando qué información crees que pierde el modelo al promediar los embeddings en lugar de procesarlos en orden con una red recurrente

Al terminar esta mini-tarea, habrás entrenado con tus propias manos la pieza que convierte palabras en vectores y habrás comprobado en la práctica por qué el orden de las palabras importa. El capítulo siguiente retoma esta misma arquitectura encoder-decoder para mostrarte cómo el mecanismo de atención resuelve su principal limitación.
