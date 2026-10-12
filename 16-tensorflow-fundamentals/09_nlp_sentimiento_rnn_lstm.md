# 09. NLP: sentimiento, RNN, LSTM y generación de texto

## Por qué importa este tema

En el capítulo anterior aprendiste a convertir texto en números: tokenización y secuenciación. Pero convertir palabras en números es solo el primer paso. El objetivo real casi siempre es uno de estos dos: **entender** lo que dice un texto (por ejemplo, si una reseña de película es positiva o negativa) o **generar** texto nuevo (por ejemplo, una estrofa de poesía). Ambas tareas comparten una dificultad que no habías tenido que resolver hasta ahora: el orden de las palabras importa, y a veces la palabra clave para entender una frase está muy lejos de donde hace falta usarla.

En este capítulo vas a entrenar un clasificador de sentimiento real, vas a reutilizar embeddings ya entrenados por otros con TensorFlow Hub, y vas a descubrir por qué una red densa se queda corta con datos secuenciales, qué aportan las redes neuronales recurrentes (RNN, Recurrent Neural Network) y por qué, en la práctica, casi siempre se usa una variante suya llamada memoria a largo-corto plazo (LSTM, Long Short-Term Memory). Cerrarás el capítulo generando texto nuevo con un modelo LSTM entrenado con letras de canciones.

## Entrenar un clasificador de sentimiento con embeddings

Retoma el ejemplo del capítulo anterior: un conjunto de titulares de noticias etiquetados como sarcásticos o no sarcásticos (el dataset lo recopiló Rishabh Misra y está disponible públicamente). Ya sabes tokenizar los titulares, convertirlos en secuencias numéricas y rellenarlas con padding hasta una longitud común. El siguiente paso es diseñar una red que aprenda, a partir de esas secuencias y sus etiquetas, a distinguir el sarcasmo.

La arquitectura más sencilla que funciona bien para esto combina tres piezas:

```python
import tensorflow as tf

vocab_size = 10000
embedding_dim = 16
max_length = 100

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

- La capa **Embedding** convierte cada palabra (representada como un número entero) en un vector denso de `embedding_dim` valores. Al principio esos vectores son aleatorios; durante el entrenamiento, la red los ajusta para que palabras con un sentimiento parecido acaben con vectores que apuntan en direcciones parecidas
- **GlobalAveragePooling1D** promedia los vectores de todas las palabras de una frase en uno solo, de tamaño fijo, sin importar cuántas palabras tuviera la frase original
- Las dos capas **Dense** finales hacen el trabajo de clasificación propiamente dicho, terminando en una neurona con activación sigmoide que produce un valor entre cero y uno (cuanto más cercano a uno, más sarcástico)

Entrenas este modelo exactamente igual que cualquier otro con Keras: `model.fit(training_padded, training_labels, epochs=30, validation_data=(testing_padded, testing_labels))`. Con esta arquitectura tan simple es habitual llegar a una precisión (accuracy) cercana al 99% en los datos de entrenamiento y de 81% a 82% en los datos de prueba (titulares que la red nunca vio). Esa diferencia es una señal de sobreajuste (overfitting) —lo trataste en el capítulo 6— y una razón más para vigilar siempre las curvas de entrenamiento frente a las de validación.

## Qué son los embeddings y por qué capturan el sentimiento

Un embedding es una forma de representar una palabra como un punto en un espacio de varias dimensiones, de modo que palabras con significados o connotaciones parecidas queden cerca unas de otras. Imagina un plano con dos ejes: uno para "sentimiento negativo → positivo" y otro para un matiz secundario, como la intensidad. Al entrenar la red con miles de titulares etiquetados, la capa Embedding va aprendiendo qué dirección de ese espacio corresponde a "sarcástico" y cuál a "no sarcástico". Una palabra que solo aparece en titulares sarcásticos termina con un vector que apunta con fuerza hacia esa dirección; una palabra neutra queda cerca del centro.

Esto explica algo que a primera vista parece magia: si le pides al modelo que evalúe la frase "not bad, a bit meh" (nada mal, un poco soso), el resultado no depende de buscar una palabra positiva o negativa aislada, sino de sumar los vectores de todas las palabras de la frase. El resultado final es un punto en el espacio de sentimiento —por ejemplo, ligeramente positivo— aunque ninguna palabra por separado sea claramente positiva.

## Reutilizar embeddings preentrenados con TensorFlow Hub

Entrenar tus propios embeddings desde cero funciona, pero tiene un coste: necesitas muchos datos etiquetados para que la red aprenda buenas representaciones. TensorFlow Hub es un repositorio de modelos y componentes ya entrenados por Google y por la comunidad, listos para reutilizar en tus propios proyectos. Para tareas de texto, esto significa que puedes coger un embedding entrenado sobre miles de millones de palabras y usarlo directamente como primera capa de tu red, sin tener que tokenizar ni preprocesar el texto tú mismo.

El ejemplo de referencia es un clasificador de reseñas de películas del conjunto IMDB (Internet Movie Database), que contiene 50.000 reseñas ya etiquetadas como positivas o negativas:

```python
import tensorflow as tf
import tensorflow_hub as hub

# Capa de embedding preentrenada: recibe frases de texto sin tokenizar
# y devuelve un vector de 50 dimensiones por frase
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
                     epochs=10,
                     validation_data=validation_data.batch(512))
```

Fíjate en tres detalles importantes:

- La capa `hub_layer` recibe directamente cadenas de texto (`dtype=tf.string`), sin pasar por `Tokenizer` ni por `pad_sequences`: el modelo preentrenado ya incluye su propia forma de tokenizar y combinar palabras en un vector de tamaño fijo
- `trainable=True` permite que el embedding se siga ajustando (fine-tuning) durante tu entrenamiento; si lo pones en `False`, el embedding queda congelado y solo entrenas las capas que añades encima
- El resto del modelo —dos capas Dense— es idéntico en espíritu al clasificador de sarcasmo que entrenaste antes: una vez tienes un vector de tamaño fijo que representa la frase, el resto es un problema de clasificación binaria normal

Este enfoque —usar un componente preentrenado como primera capa— se llama aprendizaje por transferencia (transfer learning) y es una de las técnicas más rentables en Machine Learning: te ahorra tiempo de entrenamiento y datos, y aun con una arquitectura muy simple, este modelo alcanza una precisión cercana al 87% en el conjunto de prueba.

## Qué resuelve una red recurrente frente a una red densa

Para clasificar sentimiento, promediar los vectores de todas las palabras (como hace `GlobalAveragePooling1D`) funciona razonablemente bien porque el sentimiento general de una frase no depende demasiado del orden exacto de las palabras. Pero hay tareas donde el orden lo es todo: si quieres generar la siguiente palabra de una frase, necesitas saber qué palabras vinieron antes y en qué orden. Una red densa corriente no tiene forma de recordar esa secuencia: procesa cada entrada de forma independiente, sin memoria de lo que vino antes.

Aquí es donde entra la red neuronal recurrente (RNN). Piensa en la sucesión de Fibonacci: cada número se calcula sumando los dos anteriores, así que cada valor de la serie lleva "dentro de sí" un rastro de todos los valores previos. Una RNN aplica esa misma idea al lenguaje: cada neurona recurrente recibe la palabra actual y, además, un valor que viene de la neurona anterior (su "memoria" de lo que ya ha visto). Ese valor se va actualizando y pasando hacia adelante, palabra a palabra, de forma que la red mantiene un rastro del contexto de la frase mientras la procesa.

El problema es que ese rastro se va diluyendo con la distancia. En la frase "I lived in Ireland, so I learned how to speak ___", la palabra correcta es "Gaelic" (gaélico), pero el indicio que lo determina —"Ireland"— está muy lejos de donde hace falta usarlo. Una RNN simple tiende a dar mucho peso a las palabras cercanas y muy poco a las lejanas, así que en frases largas pierde justo la información que necesitaría para acertar.

## Qué aporta LSTM frente a una RNN simple

Este debilitamiento del contexto con la distancia tiene un nombre técnico: el problema del gradiente que desaparece (vanishing gradient problem). Durante el entrenamiento, la red ajusta sus pesos propagando el error hacia atrás en el tiempo (a través de cada paso de la secuencia); en una RNN simple, ese error se va reduciendo exponencialmente en cada paso que retrocede, así que las palabras del principio de una frase larga apenas influyen en el ajuste final. En la práctica, esto significa que una RNN simple aprende bien dependencias cortas y falla en las largas.

Las redes de memoria a largo-corto plazo (LSTM) se diseñaron específicamente para resolver este problema. La diferencia clave es que una LSTM mantiene, además del rastro habitual, un **estado de celda** (cell state) que actúa como una cinta transportadora de información: puede recorrer muchos pasos de la secuencia casi sin degradarse, llevando consigo datos relevantes de mucho más atrás en la frase. Tres "puertas" (gates) controlan qué entra, qué sale y qué se olvida de esa cinta:

- La **puerta de entrada** (input gate) decide cuánta información nueva se añade al estado de celda
- La **puerta de olvido** (forget gate) decide cuánta información antigua se descarta
- La **puerta de salida** (output gate) decide qué parte del estado de celda se usa para producir la salida de ese paso

Gracias a estas puertas, una LSTM puede aprender a "recordar" que la frase mencionó "Ireland" varias palabras atrás y usarlo para predecir correctamente "Gaelic", algo que a una RNN simple se le escapa.

Dos detalles prácticos que verás en el código:

- Una LSTM puede ser **bidireccional**: además de leer la frase de izquierda a derecha, la lee también de derecha a izquierda, y combina ambas lecturas. Esto ayuda porque, a veces, una palabra posterior aporta contexto sobre una palabra anterior. En TensorFlow se implementa envolviendo la capa LSTM en `tf.keras.layers.Bidirectional`
- Si apilas varias capas LSTM una encima de otra, todas menos la última deben devolver la secuencia completa de salidas (no solo el último valor) para que la siguiente capa tenga toda la información que necesita. Esto se controla con el parámetro `return_sequences=True`

```python
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.models import Sequential

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_sequence_len),
    Bidirectional(LSTM(64)),
    Dense(num_classes, activation='softmax')
])
```

## Generar texto nuevo con un modelo LSTM

Todo lo que has visto sobre LSTM se puede reutilizar, con un cambio de enfoque, para generar texto nuevo en lugar de clasificarlo. La idea es entrenar una red para que, dada una secuencia de palabras, prediga cuál es la palabra más probable a continuación. Una vez entrenada, puedes "sembrar" el modelo con una frase inicial y dejar que prediga palabra a palabra, añadiendo cada predicción a la frase y repitiendo el proceso: así generas texto nuevo, palabra a palabra, a partir de los patrones que aprendió.

El ejemplo de referencia usa la letra de canciones tradicionales irlandesas como corpus de entrenamiento. El proceso de preparación de los datos tiene una diferencia importante respecto al clasificador de sentimiento:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
corpus = data.lower().split("\n")          # una línea de la canción por elemento
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Por cada línea, generamos todos los n-gramas posibles:
# "en un lugar" -> ["en", "un"], ["en", "un", "lugar"]
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i + 1])

max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Todas las palabras de cada n-grama menos la última son la entrada (X);
# la última palabra es la etiqueta que el modelo debe aprender a predecir (Y)
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
```

Aquí tienes dos decisiones de diseño que conviene entender:

- Un **n-grama** es una secuencia de n palabras consecutivas. Al trocear cada línea en todos sus n-gramas posibles, conviertes una tarea de "genera la frase completa" en muchas tareas más pequeñas de "dada esta secuencia, predice la siguiente palabra", que es justo lo que necesita aprender el modelo
- Las etiquetas se codifican con **one-hot encoding**: cada palabra del vocabulario se representa como un vector binario con un uno en la posición de esa palabra y ceros en el resto. Esto permite tratar la predicción de la siguiente palabra como una clasificación con tantas clases como palabras tiene el vocabulario, entrenada con la función de pérdida `categorical_crossentropy`

El modelo en sí es una LSTM bidireccional parecida a la que ya conoces, terminada en una capa densa con tantas neuronas como palabras del vocabulario y activación softmax (para que la salida sea una distribución de probabilidad sobre la siguiente palabra):

```python
from tensorflow.keras.optimizers import Adam

model = tf.keras.Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len - 1),
    Bidirectional(LSTM(150)),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1)
```

Con esta arquitectura relativamente sencilla, es habitual alcanzar una precisión de 70% a 75% al predecir la siguiente palabra correcta; suficiente para generar texto con una coherencia local aceptable, aunque no perfecta. Para generar la poesía nueva, repites el ciclo de predicción tantas veces como palabras quieras añadir:

```python
seed_text = "I've got a bad feeling about this"
next_words = 20

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)
```

Esta misma técnica —tokenizar, generar n-gramas, entrenar una LSTM a predecir la siguiente unidad de texto y luego generar iterativamente— funciona igual si trabajas carácter a carácter en lugar de palabra a palabra; solo cambia la unidad mínima que tokenizas. Elegir palabras o caracteres es una decisión de diseño: por palabras necesitas menos pasos para generar frases largas, pero el vocabulario es mayor; por caracteres el vocabulario es mucho más pequeño, pero necesitas más pasos y más datos para que el modelo aprenda ortografía y estructura.

## Resumen de puntos clave

- Un clasificador de sentimiento sencillo combina una capa Embedding (aprende vectores de palabras), un promedio de esos vectores por frase y capas densas de clasificación
- Los embeddings capturan sentimiento y significado como direcciones en un espacio vectorial: palabras parecidas acaban con vectores parecidos
- TensorFlow Hub permite reutilizar embeddings ya entrenados por terceros como primera capa de tu modelo, sin tokenizar tú mismo el texto y con el beneficio del aprendizaje por transferencia
- Una red densa no tiene memoria de lo que ha visto antes; una RNN sí, porque cada paso recibe el resultado del paso anterior, igual que cada término de la sucesión de Fibonacci depende de los anteriores
- Las RNN simples sufren el problema del gradiente que desaparece: pierden el rastro de las palabras lejanas en frases largas
- LSTM resuelve esto con un estado de celda y tres puertas (entrada, olvido, salida) que deciden qué información conservar a largo plazo; puede ser bidireccional y se puede apilar usando `return_sequences=True`
- Generar texto nuevo es, en el fondo, un problema de clasificación: predecir la palabra (o el carácter) más probable a continuación, a partir de n-gramas de un corpus de entrenamiento y codificación one-hot de las etiquetas

## Ponlo en práctica

Coge el modelo de clasificación de sarcasmo de este capítulo (Embedding + GlobalAveragePooling1D + Dense) y, sin mirar todavía el notebook de práctica de la clase, responde por escrito a estas tres preguntas antes de tocar el código:

1. Si sustituyeras la capa `GlobalAveragePooling1D` por una capa `LSTM(64)`, ¿qué esperas que cambie en la capacidad del modelo para detectar sarcasmo? Ten en cuenta que el sarcasmo a menudo depende del orden y del contraste entre partes de la frase, no solo de las palabras sueltas
2. Si entrenases ese mismo modelo con LSTM durante muchas épocas y la precisión de entrenamiento subiera mucho más rápido que la de validación, ¿qué problema de los que ya conoces de capítulos anteriores estarías viendo?
3. Para el generador de poesía, ¿qué diferencia práctica esperarías entre sembrar el modelo con una frase muy parecida al corpus de entrenamiento (letras de canciones irlandesas) frente a sembrarlo con una frase totalmente ajena, como una noticia de tecnología?

Después, abre el notebook de práctica de la clase y comprueba tus respuestas ejecutando el código.
