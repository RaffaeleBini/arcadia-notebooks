[⬅ Volver al índice](00_indice.md)

# Capítulo 10 — Deep learning en el navegador con TensorFlow.js

## Por qué te importa esto

Todo lo que has entrenado hasta ahora corre en tu ordenador o en Colab, con
Python. Pero, ¿y si quieres que tu modelo funcione directamente en la web, sin que
el usuario instale nada ni tenga que enviar sus datos a un servidor? Para eso
existe TensorFlow.js: la misma tecnología que ya conoces, ejecutándose en
JavaScript (JS), dentro del navegador.

## Por qué ejecutar un modelo en el navegador (con una analogía)

Piensa en la diferencia entre pedirle a un chef que te cocine algo en su
restaurante (el servidor) y darle una receta ya preparada para que la sigas tú
mismo en casa (el navegador). Cocinar en casa es más lento que en un restaurante
profesional, pero no dependes de enviar tus ingredientes a otro sitio ni de esperar
a que el chef esté disponible: todo ocurre en tu propia cocina, al instante y en
privado. TensorFlow.js funciona igual: el modelo se ejecuta en el dispositivo del
usuario, sin enviar sus datos a ningún servidor, con menos latencia y más
privacidad, aunque algo más lento que un servidor dedicado con GPU.

## Entrena un modelo directamente en HTML

TensorFlow.js se puede usar sin instalar nada: basta con incluir su script en una
página HTML.

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script>
  const modelo = tf.sequential();
  modelo.add(tf.layers.dense({units: 1, inputShape: [1]}));
  modelo.compile({optimizer: "sgd", loss: "meanSquaredError"});

  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
  const ys = tf.tensor2d([2, 4, 6, 8], [4, 1]);

  modelo.fit(xs, ys, {epochs: 200}).then(() => {
    modelo.predict(tf.tensor2d([5], [1, 1])).print();
  });
</script>
```

Fíjate en el paralelismo con Keras: `tf.sequential()` es el equivalente a
`tf.keras.Sequential()`, y `tf.layers.dense` es el equivalente a
`tf.keras.layers.Dense`. Si ya sabes Keras, aprender la API de capas de
TensorFlow.js es sobre todo aprender su sintaxis en JavaScript.

## Entrena en Python, despliega en el navegador

Lo habitual no es entrenar redes complejas directamente en JavaScript —entrenar es
lento sin una GPU potente—, sino entrenar en Python con Keras, como llevas
haciendo todo el curso, y convertir después el modelo entrenado a un formato que
TensorFlow.js pueda cargar.

```bash
pip install tensorflowjs
tensorflowjs_converter --input_format=keras modelo.h5 modelo_web/
```

En el navegador, cargas ese modelo convertido y haces predicciones sin necesitar
Python en ningún momento:

```html
<script>
  async function cargarYPredicir() {
    const modelo = await tf.loadLayersModel("modelo_web/model.json");
    const prediccion = modelo.predict(tf.tensor2d([[0.5]]));
    prediccion.print();
  }
  cargarYPredicir();
</script>
```

## Un caso real: una IA que juega al Pong en el navegador

Este patrón —entrenar en Python, desplegar en el navegador— es exactamente el que
usa un proyecto clásico de este curso: una IA que aprende a jugar al Pong. El
modelo recibe la posición de la pelota y de la pala como entrada, y decide mover la
pala arriba, abajo o dejarla quieta, con el mismo enfoque de aprendizaje por
refuerzo del capítulo 6, pero desplegado como una página web jugable por
cualquiera, sin instalar nada.

## Resumen de puntos clave

- TensorFlow.js ejecuta modelos directamente en el navegador, con menos latencia
  y sin enviar datos a un servidor, a costa de algo de velocidad de cómputo.
- La API de capas de TensorFlow.js (`tf.sequential`, `tf.layers.dense`) es
  prácticamente un calco de Keras, pero en JavaScript.
- El patrón habitual es entrenar en Python con Keras y convertir el modelo
  entrenado con `tensorflowjs_converter`, en lugar de entrenar redes complejas
  directamente en el navegador.
- Un proyecto de aprendizaje por refuerzo, como el Pong con IA, puede desplegarse
  igual que cualquier otro modelo: entrenado en Python, jugado en el navegador.

## Ponlo en práctica

**Copia** el ejemplo HTML del principio de este capítulo en un archivo `.html`,
**ábrelo** en tu navegador, y **abre la consola de desarrollador** (F12) para ver
el resultado de la predicción impreso con `.print()`. **Cambia** los datos de
entrenamiento `xs`/`ys` para que representen otra relación —por ejemplo, el
triple en lugar del doble— y **comprueba** que la predicción se ajusta a la nueva
relación.

---

**Material fuente de este capítulo**: lección 10 temas 2 y 3 ("AI Pong in
TensorFlow.js", "Training Model in Python and Loading into TensorFlow.js");
`Tensorflosjs_basic.html` y carpeta `pong_game/`.

[⬅ Capítulo anterior](09_chatbot.md) · [Volver al índice](00_indice.md) · [Siguiente capítulo ➡](11_casos_de_uso.md)
