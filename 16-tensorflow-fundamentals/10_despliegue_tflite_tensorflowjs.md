# 10. Despliegue con TensorFlow Lite y TensorFlow.js

## Por qué importa este tema

Hasta ahora has entrenado y evaluado tus modelos en Google Colab, con acceso a una GPU (Graphics Processing Unit o Unidad de Procesamiento Gráfico) potente en la nube. Pero un modelo entrenado no sirve de mucho si se queda encerrado en un notebook: el objetivo final casi siempre es que alguien lo use, ya sea desde el móvil, desde una cámara conectada o desde una página web. Ahí aparece un problema práctico: un teléfono no tiene la misma memoria ni potencia de cálculo que un servidor en la nube, y una página web no puede depender de que el usuario tenga conexión estable en todo momento.

Este capítulo te enseña las dos vías que ofrece TensorFlow para resolver ese problema: TensorFlow Lite (TFLite), pensado para llevar modelos a móviles y dispositivos embebidos, y TensorFlow.js (TFJS), pensado para ejecutarlos directamente en el navegador. No vas a escribir código nativo de Android o iOS —eso queda fuera del alcance de este curso—, pero sí vas a entender qué ocurre por dentro cuando conviertes un modelo y lo preparas para funcionar fuera de un servidor.

## Edge frente a servidor: cuándo conviene ejecutar el modelo en el dispositivo

Imagina que necesitas traducir una frase en un país sin cobertura de datos. Puedes llevar contigo un diccionario de bolsillo (lo resuelves tú, al instante, esté donde esté) o llamar por teléfono a un traductor profesional (más preciso, pero dependes de que la llamada se conecte). Ejecutar un modelo **en el dispositivo** —lo que se conoce como computación en el **edge** o "borde" de la red— es la versión "diccionario de bolsillo": el modelo vive dentro del móvil, la aplicación web o el sensor, y responde sin salir de ahí. Ejecutarlo **en un servidor** es la versión "llamada telefónica": el dispositivo envía los datos a un servidor potente, este hace la predicción y devuelve el resultado.

Conviene ejecutar en el dispositivo cuando:

- Necesitas una respuesta inmediata, sin la latencia de ida y vuelta a un servidor (por ejemplo, clasificar en tiempo real lo que ve la cámara)
- La aplicación debe funcionar sin conexión a internet o con conexión inestable
- Los datos son sensibles y prefieres que no salgan del dispositivo (mejora la privacidad)
- Quieres evitar el coste de mantener un servidor de inferencia funcionando permanentemente

Conviene mantener el modelo en un servidor cuando:

- El modelo es demasiado grande o complejo para la memoria y la batería de un dispositivo modesto
- Necesitas actualizar el modelo con frecuencia y no quieres depender de que cada usuario actualice la aplicación
- El propio dispositivo no tiene capacidad de cálculo suficiente, ni siquiera con un modelo optimizado

En ambos casos hay un coste que asumir: en el edge, sacrificas algo de tamaño y precisión del modelo a cambio de velocidad y autonomía; en el servidor, sacrificas autonomía y privacidad a cambio de más potencia de cálculo disponible.

## Qué es TensorFlow Lite y cómo convertir un modelo Keras

TensorFlow Lite es una versión ligera de TensorFlow, diseñada específicamente para ejecutar modelos ya entrenados en dispositivos móviles y embebidos (smartphones, tablets, microcontroladores, dispositivos IoT). No sirve para entrenar modelos desde cero: su trabajo es tomar un modelo que ya entrenaste con Keras y convertirlo en un formato compacto y rápido de cargar, pensado para hardware con memoria y batería limitadas.

El formato de salida es un archivo con extensión `.tflite`, basado en FlatBuffers (una estructura de datos binaria optimizada para lectura rápida, similar en propósito a JSON pero mucho más eficiente). Convertir un modelo Keras entrenado a este formato es, con la Application Programming Interface (API, Interfaz de Programación de Aplicaciones) actual de TensorFlow, un proceso de pocas líneas:

```python
import tensorflow as tf

# Partimos de un modelo Keras ya entrenado
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardamos el modelo convertido en disco
with open("modelo.tflite", "wb") as f:
    f.write(tflite_model)
```

Durante la conversión puedes aplicar **cuantización (quantization)**: una optimización que reduce la precisión numérica de los pesos del modelo, por ejemplo de números decimales de 32 bits a enteros de 8 bits. Esto reduce el tamaño del archivo de forma notable y acelera la inferencia, a cambio de una pérdida de precisión normalmente pequeña. En el notebook de práctica de esta clase vas a comparar tú mismo el tamaño de un modelo con y sin esta optimización.

Una vez tienes el archivo `.tflite`, lo cargas con el **intérprete (`Interpreter`)**, el componente de TensorFlow Lite responsable de ejecutar la inferencia:

```python
interpreter = tf.lite.Interpreter(model_path="modelo.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]["index"], datos_de_entrada)
interpreter.invoke()
prediccion = interpreter.get_tensor(output_details[0]["index"])
```

Este mismo intérprete es el que, por dentro, usan las aplicaciones de Android e iOS que verás en el siguiente apartado.

## Integración conceptual en Android e iOS

No vas a escribir código nativo en este curso, pero conviene que sepas qué aspecto tiene el proceso, porque es el mismo patrón en ambas plataformas: convertir el modelo, empaquetarlo con la aplicación y cargarlo con un intérprete.

**En Android**, el flujo típico es:

1. Añades la librería de TensorFlow Lite como dependencia en el archivo `build.gradle` del proyecto
2. Copias el archivo `.tflite` (y, si es un clasificador, un archivo de etiquetas de texto) en la carpeta `assets` de la aplicación
3. Desde el código Java o Kotlin, cargas el modelo con la clase `Interpreter` y ejecutas la inferencia sobre los datos de entrada (por ejemplo, un fotograma de la cámara convertido a una estructura de bytes)
4. Si el dispositivo lo soporta, TensorFlow Lite usa automáticamente la Android Neural Networks API (NNAPI), una capa del sistema operativo pensada para acelerar operaciones de aprendizaje automático por hardware; si no está disponible, la inferencia se ejecuta en la CPU (Central Processing Unit o Unidad Central de Procesamiento)

**En iOS**, el flujo es equivalente aunque el ecosistema cambia: usas Xcode como entorno de desarrollo, añades el framework de TensorFlow Lite al proyecto, incluyes el archivo `.tflite` y el archivo de etiquetas como recursos de la aplicación, y cargas el modelo con la clase `Interpreter` desde Swift u Objective-C. Una particularidad histórica de estos ejemplos es que las aplicaciones que usan la cámara solo funcionan en un dispositivo físico, no en el simulador de Xcode.

Un ejemplo típico que aparece en ambas plataformas es el de clasificación de imágenes en tiempo real con el modelo MobileNet: la aplicación captura fotogramas de la cámara, los redimensiona y normaliza, se los pasa al intérprete, y muestra en pantalla las tres etiquetas con mayor probabilidad. Es exactamente el mismo ciclo predecir → interpretar resultado que ya conoces de Keras, solo que ejecutado dentro del propio teléfono.

## El delegado de GPU experimental

Un **delegado (delegate)**, en TensorFlow Lite, es un componente que permite al intérprete "delegar" ciertas operaciones a un hardware distinto de la CPU. El delegado de GPU experimental hace exactamente eso: aprovecha la GPU del dispositivo —diseñada para hacer muchos cálculos en paralelo, algo en lo que las operaciones matriciales de una red neuronal encajan perfectamente— usando OpenGL ES en Android y Metal en iOS.

Según las pruebas de rendimiento de TensorFlow, este delegado puede acelerar la inferencia entre dos y siete veces frente a una implementación en CPU, sobre todo en modelos complejos. Sin embargo, para modelos muy pequeños y sencillos, la mejora puede no notarse, o incluso ser contraproducente: mover los datos a la memoria de la GPU también tiene un coste, y si el modelo es tan simple que la CPU ya lo resuelve rápido, ese coste de transferencia puede no compensar. Además, no todas las operaciones de TensorFlow están soportadas por el delegado de GPU: las que no lo están, se ejecutan automáticamente en la CPU como alternativa (fallback).

Activar el delegado desde Python tiene este aspecto:

```python
delegado_gpu = tf.lite.experimental.load_delegate("libtensorflowlite_gpu_delegate.so")
interpreter = tf.lite.Interpreter(
    model_path="modelo.tflite",
    experimental_delegates=[delegado_gpu]
)
```

Ten en cuenta que esta librería del delegado es específica de cada plataforma (Android o iOS) y no está pensada para ejecutarse en un entorno de escritorio o en Colab: la ves aquí para que entiendas el concepto, no para ejecutarla en el notebook de esta clase.

## TensorFlow.js: entrenar y ejecutar modelos en el navegador

TensorFlow.js lleva la misma idea de TensorFlow Lite —ejecutar aprendizaje automático fuera de un servidor— a un entorno distinto: el navegador web y Node.js. Es una librería en JavaScript que te permite dos cosas diferentes:

1. **Ejecutar en el navegador un modelo que ya entrenaste en Python.** Conviertes tu modelo Keras al formato de TensorFlow.js con la herramienta de línea de comandos `tensorflowjs_converter`, y el resultado es un conjunto de archivos (una definición del modelo en JSON y los pesos en archivos binarios) que cualquier página web puede cargar
2. **Entrenar un modelo nuevo directamente en JavaScript**, sin pasar por Python en ningún momento, usando una API muy similar a la de Keras

TensorFlow.js aprovecha WebGL (la misma tecnología que usan los navegadores para acelerar gráficos) para ejecutar los cálculos en la GPU del dispositivo del usuario, de forma parecida a como el delegado de GPU acelera TensorFlow Lite.

Como TensorFlow.js no se ejecuta en un notebook de Python, aquí tienes un ejemplo comentado de cómo se vería en una página web. Primero, el comando de conversión que usarías en tu terminal después de entrenar el modelo en Python:

```bash
# Convierte un modelo Keras guardado en formato SavedModel o .h5
# a los archivos que TensorFlow.js necesita para cargarlo en el navegador
tensorflowjs_converter --input_format=keras \
    ruta/al/modelo.h5 \
    ruta/de/salida/modelo_tfjs
```

Y este es el código JavaScript que cargaría ese modelo convertido en una página web y haría una predicción:

```html
<!-- Cargamos la librería de TensorFlow.js desde un CDN -->
<!-- (CDN: Content Delivery Network o Red de Distribución de Contenidos) -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>

<script>
  async function clasificarEnElNavegador() {
    // Cargamos el modelo ya convertido (equivalente al .tflite, pero para web)
    const modelo = await tf.loadLayersModel("modelo_tfjs/model.json");

    // Creamos un tensor de entrada de ejemplo, igual que harías con NumPy en Python
    const entrada = tf.tensor2d([[1, 2, 3, 4]]);

    // Ejecutamos la inferencia: el equivalente en el navegador a interpreter.invoke()
    const prediccion = modelo.predict(entrada);
    prediccion.print();
  }

  clasificarEnElNavegador();
</script>
```

Si prefieres entrenar un modelo nuevo directamente en el navegador en lugar de convertir uno ya entrenado, la sintaxis es prácticamente un espejo de Keras: defines las capas con `tf.sequential()` y `tf.layers.dense()`, compilas el modelo indicando la función de pérdida y el optimizador, y lo entrenas con `model.fit()` pasando tensores de entrada y salida. Esa similitud deliberada es una de las razones por las que, si ya sabes construir modelos con Keras, aprender TensorFlow.js te va a resultar familiar.

## Resumen de puntos clave

- Ejecutar un modelo en el dispositivo (edge) reduce la latencia, funciona sin conexión y mejora la privacidad; mantenerlo en un servidor tiene sentido cuando el modelo es demasiado grande o cambia con frecuencia
- TensorFlow Lite convierte un modelo Keras ya entrenado en un archivo `.tflite` compacto, optimizado para móviles y dispositivos embebidos, mediante `tf.lite.TFLiteConverter`
- La cuantización reduce la precisión numérica de los pesos del modelo para conseguir un archivo más pequeño y una inferencia más rápida, a cambio de algo de precisión
- El intérprete (`tf.lite.Interpreter`) es el componente que carga el modelo `.tflite` y ejecuta la inferencia, tanto en Python como dentro de una aplicación Android o iOS
- El delegado de GPU experimental acelera la inferencia de TensorFlow Lite aprovechando la GPU del dispositivo, con mejoras de entre dos y siete veces en modelos complejos, aunque no siempre compensa en modelos muy sencillos
- TensorFlow.js permite ejecutar en el navegador un modelo entrenado en Python (convertido con `tensorflowjs_converter`) o entrenar uno nuevo directamente en JavaScript, con una sintaxis muy similar a la de Keras

## Ponlo en práctica

Antes de abrir el notebook de esta clase, coge papel y boli otra vez. Piensa en una aplicación de tu día a día que use inteligencia artificial (un traductor, un asistente de voz, un filtro de fotos, un corrector ortográfico) y responde a estas tres preguntas:

- ¿Crees que la predicción se calcula en el propio dispositivo o en un servidor? ¿Por qué lo crees así?
- Si tuvieras que decidir tú, ¿qué pesaría más en tu decisión: la velocidad de respuesta, el consumo de batería, la necesidad de funcionar sin conexión o la privacidad de los datos?
- ¿Qué pasaría con esa aplicación si el modelo fuera diez veces más grande? ¿Seguiría teniendo sentido ejecutarlo en el dispositivo?

Guarda tus respuestas: cuando conviertas tu propio modelo a TensorFlow Lite en el notebook de práctica y compares su tamaño con y sin cuantización, vas a comprobar en cifras reales si tu intuición era acertada.
