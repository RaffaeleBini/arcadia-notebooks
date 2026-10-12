# 11. TensorFlow en Colaboratory, GPU/TPU y migración a TF 2.0

## Por qué importa este tema

Hasta ahora has ejecutado todo el código de este curso sin preguntarte dónde corre realmente. La respuesta, casi siempre, ha sido Google Colaboratory: un entorno gratuito en la nube que te ha evitado instalar Python, TensorFlow (TF) y decenas de librerías en tu propio ordenador. En este capítulo vas a entender de verdad qué es Colab, cómo sacarle todo el partido —desde montar tu Google Drive hasta activar hardware especializado— y por qué ese hardware especializado, la GPU (Graphics Processing Unit o Unidad de Procesamiento Gráfico) y la TPU (Tensor Processing Unit o Unidad de Procesamiento Tensorial), puede reducir un entrenamiento de horas a minutos. Cerrarás el capítulo entendiendo también un problema práctico muy real: qué hacer si te encuentras código antiguo escrito para TensorFlow 1.x y necesitas adaptarlo a la sintaxis de TensorFlow 2.x que has usado en todo el curso.

## Qué es Google Colaboratory y por qué te libera de instalar nada

Google Colaboratory, conocido como Colab, es un cuaderno Jupyter alojado en la nube. Si nunca has oído hablar de Jupyter: es un formato de documento ejecutable que mezcla código, texto explicativo, imágenes y resultados en un mismo archivo, organizado en celdas. Colab es, esencialmente, un cuaderno Jupyter que vive dentro de tu Google Drive en lugar de en tu disco duro.

La diferencia clave frente a instalar Jupyter en tu propio ordenador es el **entorno de ejecución (runtime)**. Cuando abres un notebook en Colab, no estás ejecutando código en tu portátil: Google te asigna una máquina virtual remota, con Python y TensorFlow ya preinstalados, que ejecuta cada celda que lanzas. Puedes cerrar el navegador y, mientras la sesión no caduque, tu entorno sigue vivo en la nube. Esto tiene tres consecuencias prácticas que vas a usar constantemente:

- No necesitas instalar nada de forma local: abres el navegador y ya tienes Python, TensorFlow y librerías como Pandas o Matplotlib disponibles
- Puedes ejecutar comandos de terminal desde una celda de código anteponiendo un signo de exclamación (`!`), por ejemplo `!pip install tensorflow==2.15.0` para instalar una versión concreta
- El mismo notebook se puede compartir como un documento de Google Docs: por enlace, con permisos de lectura o edición, o exportado como archivo `.ipynb` para abrirlo en Jupyter, JupyterLab o GitHub

Piensa en Colab como en alquilar una cocina completamente equipada cada vez que quieres cocinar, en lugar de comprar y mantener tú mismo hornos, sartenes y utensilios en casa. No eres dueño del hardware, pero tampoco tienes que preocuparte de instalarlo, actualizarlo ni de que se estropee.

## Montar Google Drive y subir archivos

Los ejemplos y datasets pequeños que has usado en clases anteriores solían descargarse directamente desde una URL. Pero en un proyecto real necesitas trabajar con tus propios archivos: un CSV (Comma-Separated Values o Valores Separados por Comas) con tus datos, un modelo ya entrenado que quieres reutilizar, o imágenes propias. Colab te ofrece dos vías principales para eso.

La primera es **montar tu Google Drive**, es decir, conectar tu cuenta de Drive al entorno de ejecución para que el notebook pueda leer y escribir archivos allí como si fuera una carpeta más del sistema. Una vez montado, cualquier archivo de tu Drive queda accesible en una ruta como `/content/drive/MyDrive/mi_carpeta/mis_datos.csv`, y cualquier archivo que generes —un modelo entrenado, por ejemplo— puedes guardarlo ahí para que sobreviva al cierre de la sesión.

La segunda vía es **subir archivos directamente** desde tu ordenador a la sesión actual, sin pasar por Drive. Es más rápido para un archivo suelto que solo necesitas una vez, pero esos archivos desaparecen en cuanto termina la sesión de Colab, así que no sirve para guardar resultados de forma permanente.

## Cómo activar y comprobar una GPU o una TPU

Por defecto, un notebook nuevo de Colab arranca con un entorno de ejecución basado solo en CPU (Central Processing Unit o Unidad Central de Procesamiento): suficiente para explorar datos o entrenar modelos pequeños, pero lento para redes neuronales grandes. Para acelerar el entrenamiento, cambias el tipo de hardware desde el menú **Entorno de ejecución > Cambiar tipo de entorno de ejecución**, donde puedes elegir entre "Ninguno" (CPU), GPU o TPU. Este cambio reinicia la sesión, así que conviene hacerlo antes de ejecutar celdas que dependan de variables ya creadas.

¿Por qué una GPU o una TPU aceleran tanto el entrenamiento? Una CPU tiene pocos núcleos, pero muy potentes, optimizados para ejecutar tareas complejas una detrás de otra. Entrenar una red neuronal, en cambio, consiste sobre todo en multiplicar matrices de números una y otra vez: miles de operaciones idénticas y sencillas que se pueden ejecutar en paralelo perfectamente. Una GPU tiene cientos o miles de núcleos más simples, diseñados exactamente para eso: repetir la misma operación sobre muchos datos a la vez. Una TPU va un paso más allá: es un chip que Google diseñó específicamente para Machine Learning (ML), con un circuito interno optimizado para el tipo concreto de multiplicación de matrices que usan las redes neuronales, lo que la hace aún más eficiente que una GPU para modelos grandes.

La analogía más clara es la de un equipo de reparto. La CPU es un mensajero muy rápido y versátil que entrega los paquetes uno por uno. La GPU es como tener cientos de mensajeros más sencillos entregando cientos de paquetes idénticos al mismo tiempo. La TPU es ese mismo equipo de mensajeros, pero con motos diseñadas a medida solo para ese barrio y ese tipo de paquete: nada más rápido para esa tarea concreta, aunque menos flexible para cualquier otra cosa.

Para comprobar qué hardware tienes realmente disponible en tu sesión, TensorFlow te da una función clara y directa:

```python
import tensorflow as tf

# Lista los dispositivos físicos de cada tipo que ve TensorFlow en esta sesión
print("CPU:", tf.config.list_physical_devices('CPU'))
print("GPU:", tf.config.list_physical_devices('GPU'))
print("TPU:", tf.config.list_physical_devices('TPU'))
```

Si la lista de GPU o TPU aparece vacía, significa que el entorno de ejecución actual no tiene ese hardware activado (revisa el menú "Cambiar tipo de entorno de ejecución") o que Colab no tiene ninguno disponible en ese momento para tu cuenta. Con una GPU activada, verás algo como `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.

## Por qué el entrenamiento es más rápido con GPU: un ejemplo comentado

El siguiente fragmento entrena el mismo modelo sencillo dos veces —una obligando a TensorFlow a usar la CPU y otra dejándole usar la GPU si está disponible— y compara el tiempo. Es exactamente el tipo de comparación que verás repetida, con más rigor, en el notebook de práctica de esta clase.

```python
import tensorflow as tf
import time

# Datos de ejemplo: suficientes para notar la diferencia de velocidad
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0

def crear_modelo():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Forzamos la ejecución en CPU con un gestor de contexto "with tf.device(...)"
with tf.device('/CPU:0'):
    modelo_cpu = crear_modelo()
    modelo_cpu.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    inicio = time.time()
    modelo_cpu.fit(x_train, y_train, epochs=1, batch_size=128, verbose=0)
    print(f"Tiempo en CPU: {time.time() - inicio:.2f} segundos")

# Si hay GPU disponible, TensorFlow la usa automáticamente sin código adicional
if tf.config.list_physical_devices('GPU'):
    modelo_gpu = crear_modelo()
    modelo_gpu.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    inicio = time.time()
    modelo_gpu.fit(x_train, y_train, epochs=1, batch_size=128, verbose=0)
    print(f"Tiempo en GPU: {time.time() - inicio:.2f} segundos")
else:
    print("No hay GPU activada en este entorno de ejecución")
```

Fíjate en dos detalles importantes. Primero, no necesitas ningún código especial para que TensorFlow use la GPU: si está disponible y no le indicas lo contrario con `tf.device(...)`, la usa por defecto de forma automática. Segundo, el gestor de contexto `with tf.device('/CPU:0')` es la forma de forzar una operación concreta a un dispositivo concreto, algo útil sobre todo para comparaciones como esta o para depurar dónde se está ejecutando cada parte de tu modelo con `tf.debugging.set_log_device_placement(True)`.

Para trabajar con una TPU el patrón cambia un poco, porque una TPU no vive en la misma máquina que tu notebook: está en un clúster remoto al que TensorFlow se conecta mediante una **estrategia de distribución**. El ejemplo de referencia de este capítulo —adaptado del notebook original *Predict Shakespeare with Cloud TPUs and Keras*— usa este patrón:

```python
import os
import tensorflow as tf

# Conexión al clúster de TPU asignado por Colab
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu='grpc://' + os.environ['COLAB_TPU_ADDR']
)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

# La estrategia reparte el cálculo entre los núcleos de la TPU
strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
    modelo = crear_modelo()
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# A partir de aquí, fit(), evaluate() y predict() funcionan exactamente igual
# que con cualquier otro modelo de Keras, pero el cálculo corre en la TPU
modelo.fit(x_train, y_train, epochs=10, batch_size=1024)
```

La idea clave es que, salvo por la creación de la estrategia y el bloque `with strategy.scope():` donde defines y compilas el modelo, el resto del código de Keras —`fit`, `evaluate`, `predict`— es idéntico al que ya conoces. TensorFlow se encarga de repartir el entrenamiento entre los núcleos de la TPU sin que tengas que reescribir tu lógica de entrenamiento.

## Migrar código de TensorFlow 1.x a TensorFlow 2.x

Si alguna vez trabajas con un notebook o un script antiguo, es probable que te encuentres código escrito para TensorFlow 1.x, la versión que se usaba antes de 2019. La diferencia de sintaxis puede resultar desconcertante si solo has trabajado con TensorFlow 2.x, así que conviene identificar los cambios clave.

**Ejecución diferida (graph mode) frente a ejecución inmediata (eager execution)**. En TensorFlow 1.x, escribir código no ejecutaba nada de inmediato: construías un grafo de operaciones y después tenías que abrir una `tf.Session()` y llamar a `sess.run(...)` para que el grafo se evaluara y obtener un resultado. En TensorFlow 2.x, la ejecución eager (inmediata) es el comportamiento por defecto: cada operación se calcula en el momento en que la escribes, igual que en Python normal, sin sesiones de por medio.

```python
# Estilo TensorFlow 1.x: construyes el grafo y luego lo ejecutas en una sesión
import tensorflow.compat.v1 as tf1
a = tf1.constant(2)
b = tf1.constant(3)
c = tf1.add(a, b)
with tf1.Session() as sess:
    resultado = sess.run(c)
    print(resultado)  # 5

# Estilo TensorFlow 2.x: la operación se calcula al instante, sin sesión
import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)
print(c.numpy())  # 5
```

**Placeholders frente a datos directos**. TensorFlow 1.x usaba `tf.placeholder()` para reservar un hueco al que más tarde, dentro de una sesión, le pasabas los datos reales mediante un diccionario `feed_dict`. En TensorFlow 2.x no existen los placeholders: le pasas los datos directamente a tus funciones o a tu modelo de Keras, como haces con cualquier función normal de Python.

**Inicialización de variables**. En TensorFlow 1.x tenías que llamar explícitamente a `tf.global_variables_initializer()` y ejecutarlo dentro de una sesión antes de poder usar cualquier `tf.Variable`. En TensorFlow 2.x, una variable queda lista para usarse en el mismo momento en que la creas, sin ningún paso de inicialización aparte.

**Keras como API de alto nivel por defecto**. TensorFlow 1.x tenía varias formas fragmentadas de construir modelos, incluida la API de bajo nivel `tf.layers`. TensorFlow 2.x integra Keras como la forma recomendada y por defecto de definir modelos, la misma sintaxis `tf.keras.Sequential([...])` que has usado en todo el curso.

**Herramientas de compatibilidad**. Si necesitas ejecutar código antiguo sin reescribirlo entero, TensorFlow ofrece el módulo `tensorflow.compat.v1`, que expone las funciones de la versión 1.x dentro de una instalación de TensorFlow 2.x (así es como se ha escrito el ejemplo de la sesión de arriba). Para una migración más automática, TensorFlow incluye la herramienta de línea de comandos `tf_upgrade_v2`, que analiza un script en TensorFlow 1.x y genera una versión adaptada a TensorFlow 2.x, junto con un informe de los cambios aplicados:

```bash
!tf_upgrade_v2 --infile mi_script_v1.py --outfile mi_script_v2.py
```

Esta herramienta no es infalible: convierte automáticamente la mayoría de los patrones habituales —como añadir argumentos con nombre a funciones que cambiaron el orden de sus parámetros—, pero avisa en el informe de los casos que no puede resolver sola, por ejemplo cuando una función eliminó por completo un argumento (como ocurrió con `use_cudnn_on_gpu` en `tf.nn.conv2d`). En esos casos tendrás que revisar y corregir el código a mano, apoyándote en la guía oficial de migración de TensorFlow.

## Resumen de puntos clave

- Google Colaboratory es un cuaderno Jupyter en la nube: te da un entorno de ejecución remoto con Python y TensorFlow preinstalados, sin que tengas que instalar nada en tu ordenador
- Puedes montar tu Google Drive para leer y guardar archivos de forma permanente, o subir archivos sueltos directamente a la sesión si solo los necesitas de forma puntual
- Activas GPU o TPU desde "Entorno de ejecución > Cambiar tipo de entorno de ejecución"; `tf.config.list_physical_devices(...)` te confirma qué hardware ve realmente TensorFlow en tu sesión
- La GPU y la TPU aceleran el entrenamiento porque paralelizan las multiplicaciones de matrices que dominan el cálculo de una red neuronal; la TPU está además diseñada específicamente para ese tipo de operación
- Con GPU, TensorFlow usa el hardware automáticamente sin cambios de código; con TPU necesitas crear una `TPUStrategy` y definir el modelo dentro de `strategy.scope()`
- TensorFlow 2.x usa ejecución eager por defecto (sin `Session.run`), elimina los placeholders, no requiere inicializar variables a mano e integra Keras como API principal; el módulo `tensorflow.compat.v1` y la herramienta `tf_upgrade_v2` ayudan a migrar código antiguo

## Ponlo en práctica

Abre un notebook nuevo en Colab con el entorno de ejecución en modo "Ninguno" (solo CPU) y ejecuta `tf.config.list_physical_devices('GPU')`. Confirma que la lista sale vacía. Después, cambia el tipo de entorno de ejecución a GPU desde el menú, vuelve a ejecutar la misma línea y comprueba que ahora aparece un dispositivo. Por último, escribe de memoria (sin mirar el capítulo) las tres diferencias de sintaxis entre TensorFlow 1.x y 2.x que más te hayan llamado la atención, y anota un ejemplo corto de código para cada una. Ese ejercicio de memoria es la mejor forma de comprobar si de verdad interiorizaste el cambio de paradigma entre versiones.
