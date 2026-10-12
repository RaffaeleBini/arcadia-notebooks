[⬅ Volver al índice](00_indice.md)

# Capítulo 9 — Construyendo un chatbot conversacional

## Por qué te importa esto

Un chatbot bien entrenado combina dos retos que ya conoces —procesar texto
(capítulo 2) y trabajar con secuencias (capítulo 3)— con uno nuevo: aprender a
generar una respuesta completa, no solo una etiqueta. En este capítulo ves el
recorrido completo de un proyecto de este tipo, desde limpiar conversaciones
reales hasta entender los conceptos que hacen posible que un modelo "conteste"
en lugar de solo "clasificar".

## De dónde salen las conversaciones de entrenamiento

Entrenar un chatbot requiere millones de pares pregunta-respuesta reales. Una
fuente habitual son los comentarios de foros como Reddit, donde cada comentario
tiene un "padre" (el comentario al que responde) y el propio comentario funciona
como respuesta. Antes de poder usar esos datos, hace falta limpiarlos y validarlos:

```python
def es_aceptable(comentario):
    """Filtra comentarios vacíos, demasiado largos, o eliminados."""
    if len(comentario) < 1 or len(comentario) > 1000:
        return False
    if comentario in ("[deleted]", "[removed]"):
        return False
    return True

def limpiar_texto(texto):
    """Normaliza saltos de línea y comillas para un formato consistente."""
    texto = texto.replace("\n", " ").replace("\r", " ")
    texto = texto.replace('"', "'")
    return texto.strip()
```

Con millones de comentarios, este filtrado importa: un chatbot entrenado con
respuestas vacías, insultos borrados o texto mal formado aprende a imitar ese
ruido. La calidad de los datos de entrada determina, en gran medida, la calidad de
las respuestas del modelo final.

## De clasificar a generar: tokens, relleno y traducción

Los modelos anteriores producían una única etiqueta (positivo/negativo, gato/
perro). Un chatbot necesita generar una secuencia completa de palabras, palabra a
palabra. Esto se resuelve con una arquitectura tomada prestada de la traducción
automática, o Neural Machine Translation (NMT): el modelo "traduce" la pregunta de
entrada en una respuesta de salida, aunque ambas estén en el mismo idioma.

Para que el modelo trabaje con frases de longitud distinta, cada palabra se
convierte en un identificador numérico (tokenización), y las frases más cortas se
rellenan con un token especial (padding) hasta igualar la longitud de la más
larga del lote. Las arquitecturas modernas de NMT añaden además un mecanismo de
"atención": en lugar de resumir toda la frase de entrada en un único vector antes
de generar la respuesta, la atención deja que el modelo consulte, palabra a
palabra de la respuesta que va generando, qué partes concretas de la pregunta
original son más relevantes en ese momento.

```python
capa_tokenizador = tf.keras.layers.TextVectorization(
    max_tokens=20000,
    output_sequence_length=20,  # relleno/recorte a 20 tokens por frase
)
```

## Interactúa con un modelo ya entrenado

Una vez entrenado, probar un chatbot no es tan simple como llamar a
`modelo.predict()` una vez: normalmente se generan varias respuestas candidatas
—una técnica llamada "beam search"— y se elige la mejor según criterios como
evitar palabras desconocidas o penalizar respuestas demasiado repetitivas.

## Resumen de puntos clave

- Construir un chatbot empieza por limpiar y validar datos conversacionales
  reales: la calidad de entrada limita la calidad de las respuestas.
- Generar una respuesta completa —no solo una etiqueta— usa arquitecturas
  tomadas de la traducción automática (NMT), con tokenización, relleno y
  mecanismos de atención.
- Probar un chatbot entrenado suele implicar generar varias respuestas candidatas
  y elegir la mejor con criterios adicionales, no solo tomar la primera
  predicción.

## Ponlo en práctica

**Escribe** una función `limpiar_y_validar(comentarios)` que reciba una lista de
comentarios en bruto y devuelva solo los que pasan `es_aceptable` después de
pasar por `limpiar_texto`. **Pruébala** con una lista de ejemplo que incluya al
menos un comentario vacío, uno marcado como `"[deleted]"`, y uno válido.

---

**Material fuente de este capítulo**: lección 9 temas 1 a 4 ("Data Structure",
"Buffering Dataset", "Determining Insert", "Building Database"); lección 10 tema 4
("NMT Concepts and Parameters") y tema 5 ("Interacting with the Chatbot").

[⬅ Capítulo anterior](08_cnn_3d_imagen_medica.md) · [Volver al índice](00_indice.md) · [Siguiente capítulo ➡](10_tensorflowjs.md)
