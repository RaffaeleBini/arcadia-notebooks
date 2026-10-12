# 08. NLP: tokenización y secuencias

## Por qué importa este tema

Hasta ahora has entrenado modelos con números (relaciones entre X e Y) y con imágenes (píxeles, que ya son números organizados en una cuadrícula). Pero, ¿qué pasa cuando el dato de entrada es una frase como "me encanta mi perro"? Una red neuronal no entiende palabras: solo sabe multiplicar, sumar y ajustar números. Antes de que un modelo pueda aprender nada sobre un texto, necesitas convertir ese texto en números de una forma que conserve su significado.

A esta rama de la Inteligencia Artificial (IA), la que se ocupa de que las máquinas entiendan y generen lenguaje humano, se le llama Procesamiento de Lenguaje Natural (NLP, Natural Language Processing). Tareas tan distintas como detectar si una reseña es positiva o negativa, traducir un texto o generar frases nuevas tienen algo en común: todas empiezan por resolver el mismo problema de partida, convertir palabras en números. Este capítulo cubre justo ese primer paso, que vas a usar en el resto de clases de NLP del curso.

## El problema de partida: las letras no bastan

Podrías pensar que la solución más directa es codificar cada letra con un número, por ejemplo usando ASCII (American Standard Code for Information Interchange), el estándar que asigna un número a cada carácter. El problema es que dos palabras pueden compartir exactamente las mismas letras y significar cosas opuestas: "listen" (escuchar) y "silent" (silencioso) usan las mismas letras en distinto orden. Si codificas letra a letra, no hay forma de que el modelo distinga el sentido de una palabra a partir de sus caracteres sueltos.

La alternativa que funciona mejor es codificar palabras completas, no letras. Si a la palabra "me" le asignas el número uno, a "encanta" el número dos, a "mi" el número tres y a "perro" el número cuatro, la frase "me encanta mi perro" se convierte en la secuencia 1, 2, 3, 4. Si ahora tienes la frase "me encanta mi gato" y "gato" es una palabra nueva a la que asignas el número cinco, esa frase se convierte en 1, 2, 3, 5. Con solo comparar las dos secuencias numéricas, un modelo ya puede detectar que ambas frases se parecen mucho: comparten tres de sus cuatro números.

## Qué es tokenizar y cómo se construye un vocabulario

**Tokenizar** es el proceso de partir un texto en unidades más pequeñas, llamadas **tokens**, y asignar un número entero a cada una. El caso más habitual, y el que vas a usar en este curso, es tokenizar por palabras: cada palabra distinta del texto se convierte en un token. También existen otras formas de tokenizar —por caracteres, o por subpalabras (fragmentos más pequeños que una palabra completa, útiles para idiomas con muchas formas verbales o para reducir el número de tokens necesarios)— pero el enfoque por palabras es el punto de partida más claro para aprender el concepto.

Al conjunto completo de tokens distintos que conoce tu modelo se le llama **vocabulario**. Para construirlo, TensorFlow (TF) ofrece la clase `Tokenizer`, dentro del módulo de preprocesamiento de texto de Keras. Le pasas un conjunto de frases (tu corpus) y ella se encarga de:

- Recorrer todas las frases y detectar las palabras distintas
- Pasar el texto a minúsculas y quitar la puntuación automáticamente, así "perro" y "perro!" se reconocen como la misma palabra
- Asignar un número entero a cada palabra, empezando por las más frecuentes
- Guardar esa relación palabra-número en un diccionario llamado `word_index`

Fíjate en que el `Tokenizer` ya resuelve por ti un problema práctico: si en tu corpus aparece "perro!" con signo de exclamación y "perro" sin él, no vas a acabar con dos tokens distintos para la misma palabra. La limpieza de puntuación ocurre automáticamente al ajustar el tokenizador.

## Cómo manejar palabras fuera de vocabulario (OOV)

Tu modelo solo puede conocer las palabras que ha visto durante el ajuste del tokenizador. ¿Qué ocurre si, más adelante, le presentas una frase con una palabra nueva, que no estaba en el vocabulario original? Por ejemplo, entrenas tu tokenizador con frases sobre perros y gatos, y luego le pides que procese una frase que menciona un "manatí". Sin ningún mecanismo adicional, el `Tokenizer` simplemente ignoraría esa palabra, y la secuencia numérica resultante perdería una posición: la frase se acortaría y el modelo ya no sabría en qué parte de la frase faltaba información.

La solución es reservar un token especial para todas las palabras que el modelo no reconoce, llamado token **OOV** (Out Of Vocabulary, fuera de vocabulario). Al crear el `Tokenizer`, indicas qué texto quieres usar como marcador, por ejemplo `"<OOV>"`, y a partir de ese momento cualquier palabra desconocida se sustituye por ese token en lugar de desaparecer. Así la frase conserva su longitud original y su estructura, aunque se pierda el significado exacto de la palabra nueva. No es una solución perfecta —el modelo no aprende nada sobre "manatí"— pero es muchísimo mejor que perder esa posición de la secuencia.

## De frases a secuencias numéricas: por qué hace falta el padding

Con el vocabulario ya construido, el siguiente paso es convertir cada frase completa en su secuencia de números, sustituyendo cada palabra por su token. Aquí aparece un problema nuevo: las frases del mundo real tienen longitudes distintas. "Me encanta mi perro" tiene cuatro palabras; "¿Crees que mi perro es increíble?" tiene siete. Pero una red neuronal espera que todas las entradas de un mismo lote tengan la misma forma (shape), igual que ocurría con las imágenes de tamaño fijo que usaste en clases anteriores.

La solución se llama **padding** (relleno): igualar la longitud de todas las secuencias añadiendo ceros donde haga falta. TensorFlow ofrece para esto la función `pad_sequences`, que te permite decidir:

- La longitud final de todas las secuencias (`maxlen`); si no la indicas, usa la longitud de la secuencia más larga del lote
- Si el relleno se añade al principio (`padding='pre'`, el valor por defecto) o al final (`padding='post'`) de la secuencia
- Si una frase es más larga que `maxlen`, si se recorta por el principio (`truncating='pre'`) o por el final (`truncating='post'`, el valor por defecto)

Gracias al padding, un lote de frases de longitudes muy distintas se convierte en una única matriz de números con una forma homogénea, lista para entrar en una red neuronal.

## Ejemplo de código comentado

El siguiente ejemplo reproduce el flujo completo: tokenizar un corpus pequeño, manejar palabras nuevas con el token OOV y generar secuencias con padding.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Corpus de entrenamiento: cuatro frases de ejemplo
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

# Creamos el tokenizador:
# - num_words limita el vocabulario a las 100 palabras más frecuentes
# - oov_token reserva un token para las palabras que no estén en el vocabulario
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

# Ajustamos el tokenizador sobre el corpus: aquí se construye el vocabulario
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("Vocabulario:", word_index)

# Convertimos cada frase en su secuencia de números
sequences = tokenizer.texts_to_sequences(sentences)
print("Secuencias:", sequences)

# Igualamos la longitud de todas las secuencias con padding
padded = pad_sequences(sequences, maxlen=5)
print("Secuencias con padding:")
print(padded)

# Probamos con frases nuevas, con palabras que el tokenizador no conoce
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_data)
print("Secuencias de prueba (con token OOV):", test_seq)
```

Al ejecutar este código, comprueba dos detalles clave. Primero, en `word_index` la palabra "\<OOV\>" ocupa siempre el token número uno, porque es la que más falta le hace al modelo tener siempre disponible. Segundo, en las frases de prueba, palabras como "really", "loves" o "manatee" —que no estaban en el corpus de entrenamiento— se sustituyen por ese mismo token OOV en vez de desaparecer, así que la secuencia conserva su longitud y su estructura.

## Resumen de puntos clave

- Una red neuronal solo procesa números, así que cualquier texto tiene que convertirse a números antes de entrenar un modelo: a este proceso, entre otros, se dedica el Procesamiento de Lenguaje Natural (NLP)
- Codificar letra a letra no funciona porque palabras con las mismas letras pueden tener significados opuestos; codificar palabras completas sí conserva el sentido
- Tokenizar es partir el texto en unidades (tokens, normalmente palabras) y asignar un número entero a cada una; el conjunto de tokens conocidos es el vocabulario
- El token OOV evita que las palabras nuevas, no vistas durante el ajuste del tokenizador, se pierdan o descuadren la longitud de la secuencia
- El padding iguala la longitud de todas las secuencias añadiendo ceros, para que puedan procesarse juntas en un mismo lote
- `Tokenizer`, `texts_to_sequences` y `pad_sequences` son las tres piezas de TensorFlow que resuelven este flujo completo

## Ponlo en práctica

Coge estas tres frases:

- "El gato duerme en el sofá"
- "El perro juega en el jardín"
- "El gato y el perro son amigos"

A mano, en papel, construye el vocabulario que generaría un `Tokenizer`: lista cada palabra distinta (recuerda pasarla a minúsculas) y numérala empezando por la más frecuente. Después, convierte cada frase en su secuencia de números usando tu propio vocabulario. Por último, aplica padding a mano añadiendo ceros al principio de las secuencias más cortas hasta que las tres tengan la misma longitud que la más larga. Cuando termines, compara tu resultado con el que obtendrías ejecutando el código de este capítulo con esas mismas frases.
