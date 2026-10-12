[⬅ Volver al índice](00_indice.md)

# Capítulo 2 — Tu primer clasificador de sentimientos

## Por qué te importa esto

Una marca recibe miles de reseñas y comentarios cada día. Leerlos uno a uno es
inviable, pero un modelo que distinga automáticamente lo positivo de lo negativo
convierte ese ruido en una alerta útil en segundos. Este es tu primer contacto con
el procesamiento del lenguaje natural, o Natural Language Processing (NLP): la
rama de la IA que trabaja con texto en lugar de imágenes o números sueltos.

## El reto: el texto no son números

Una red neuronal solo entiende números. Antes de poder clasificar una frase como
positiva o negativa, tienes que convertirla en un vector numérico. La técnica más
sencilla se llama "bolsa de palabras" (bag-of-words): imagina un almacén con un
casillero para cada palabra distinta que existe en tu vocabulario. Para
representar una frase, marcas con un uno los casilleros de las palabras que
aparecen en ella y con un cero el resto. Pierdes el orden de las palabras, pero
conservas la información de "qué se dice", que basta para muchas tareas.

## Construye tu vocabulario

Con TensorFlow y Keras no necesitas construir el vocabulario a mano: la capa
`TextVectorization` lo hace por ti a partir de tus datos de entrenamiento.

```python
import tensorflow as tf

frases_ejemplo = [
    "me encantó esta película, es una obra maestra",
    "una pérdida de tiempo total, aburrida de principio a fin",
]

vectorizador = tf.keras.layers.TextVectorization(max_tokens=1000, output_mode="multi_hot")
vectorizador.adapt(frases_ejemplo)

print(vectorizador.get_vocabulary()[:10])
print(vectorizador(frases_ejemplo))
```

`adapt` recorre tus frases, cuenta las palabras más frecuentes, y construye el
vocabulario. `output_mode="multi_hot"` produce exactamente la bolsa de palabras que
describimos arriba: un vector con un uno por cada palabra presente en la frase.

## Entrena tu clasificador

Con las frases ya convertidas en vectores, el resto es una red densa como la que
ya conoces del capítulo anterior.

```python
modelo = tf.keras.Sequential([
    vectorizador,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

Usa una sola neurona de salida con activación sigmoide porque este es un problema
de dos clases (positivo o negativo): la salida es directamente la probabilidad de
que la reseña sea positiva.

## De diez frases a diez mil: cuantos más datos, mejor

El material original de este curso entrena primero con un corpus pequeño de reseñas
de cine (los archivos `pos.txt` y `neg.txt`, con más de cinco mil frases positivas y
cinco mil negativas) y después demuestra el mismo pipeline sobre Sentiment140, un
dataset de un millón seiscientos mil tuits etiquetados automáticamente por su
emoticono. La lección es siempre la misma: un modelo de lenguaje mejora con más
ejemplos variados, igual que una persona que ha leído más textos reconoce mejor los
matices.

```python
# Carga del corpus pequeño incluido en el material del curso
with open("pos.txt", encoding="latin-1") as f:
    positivas = [linea.strip() for linea in f if linea.strip()]
with open("neg.txt", encoding="latin-1") as f:
    negativas = [linea.strip() for linea in f if linea.strip()]

textos = positivas + negativas
etiquetas = [1] * len(positivas) + [0] * len(negativas)
```

## Resumen de puntos clave

- Un modelo de deep learning no entiende texto: primero lo conviertes en números
  con técnicas como la bolsa de palabras.
- `TextVectorization` construye el vocabulario y transforma frases en vectores por
  ti, sin escribir ese código a mano.
- Un problema de dos clases (positivo/negativo) usa una neurona de salida con
  activación sigmoide, en lugar de las diez neuronas con softmax que usaste para
  dígitos.
- Más datos de entrenamiento —de cientos a millones de ejemplos— suele traducirse
  en un modelo más preciso y más robusto ante frases nuevas.

## Ponlo en práctica

**Escribe** tres frases nuevas de tu cosecha —una claramente positiva, una
claramente negativa y una ambigua— y **pásalas** por tu modelo ya entrenado con
`modelo.predict()`. **Comprueba** si el modelo acierta en las dos primeras y qué
probabilidad asigna a la frase ambigua.

---

**Material fuente de este capítulo**: lección 2 temas 5, 6, 7 y 8 ("Processing
Data", "Preprocessing Continued", "Training and Testing on Data", "Using More
Data"); archivos `neg.txt`, `pos.txt` y el paquete `sentiment140/`.

[⬅ Capítulo anterior](01_fundamentos.md) · [Volver al índice](00_indice.md) · [Siguiente capítulo ➡](03_rnn_lstm.md)
