# 12. TensorFlow Extended: producción de modelos

## Por qué importa este tema

Hasta ahora has entrenado modelos en un notebook: cargas datos, defines una arquitectura, llamas a `fit` y revisas el resultado tú mismo, celda a celda. Ese flujo funciona de maravilla para aprender y para experimentar, pero no sobrevive al mundo real. Una empresa que quiere predecir la demanda de un producto, detectar fraude en pagos o clasificar imágenes médicas no puede depender de que alguien reejecute un notebook a mano cada vez que llegan datos nuevos. Necesita un sistema que ingiera datos, los valide, entrene, evalúe y despliegue el modelo de forma automática, repetible y auditable.

A ese salto —de "funciona en mi notebook" a "funciona en producción, de forma fiable, durante meses o años"— se le llama ingeniería de Machine Learning (ML) para producción. TensorFlow Extended (TFX) es la plataforma de código abierto que Google desarrolló para resolver justo ese problema, y es la que usa internamente para la mayoría de sus sistemas de ML en producción. Empresas como Twitter, Airbnb o PayPal también la han adoptado. Es, con diferencia, el tema más avanzado del curso: no vas a construir redes neuronales nuevas, sino a aprender cómo se organiza y automatiza todo lo que ya sabes hacer con TensorFlow (TF) para que funcione de forma sostenida fuera del notebook.

## Qué problema resuelve TFX

Cuando un modelo pasa de ser un experimento a ser parte de un producto real, aparecen exigencias que no tenías en el notebook:

- Los datos cambian con el tiempo, y esos cambios pueden ser lentos (reconocer gatos y perros apenas varía de un año a otro) o muy rápidos (predecir tendencias de mercado cambia día a día). Cuanto más rápido cambian los datos y la "verdad de referencia" (ground truth, los valores correctos que el modelo intenta predecir), más a menudo hay que reentrenar el modelo y más difícil es etiquetar ejemplos nuevos
- El código de preprocesamiento que usaste para entrenar tiene que aplicarse exactamente igual cuando el modelo ya está sirviendo predicciones reales; cualquier diferencia entre ambos entornos —lo que se conoce como sesgo entrenamiento-servicio (training/serving skew)— degrada la calidad del modelo sin que lo notes en tus métricas de entrenamiento
- Necesitas poder rastrear qué versión de los datos entrenó qué versión del modelo, comparar ejecuciones anteriores y evitar recalcular pasos costosos si nada cambió desde la última vez
- El sistema completo, no solo el modelo, tiene que cumplir los requisitos de cualquier software en producción: escalabilidad, consistencia, capacidad de prueba, seguridad y mantenibilidad

TFX no reemplaza a TensorFlow: se construye encima. Aporta un conjunto de bibliotecas y componentes reutilizables —TensorFlow Data Validation (TFDV), TensorFlow Transform (TFT), TensorFlow Model Analysis (TFMA) y TensorFlow Serving, entre otras— que cubren cada fase del ciclo de vida de un modelo: ingestión de datos, validación, transformación de features, entrenamiento, evaluación y despliegue.

## Qué es un pipeline de TFX

Piensa en un pipeline de TFX como una cadena de montaje de una fábrica. En un notebook, tú eres quien monta el coche pieza a pieza, a mano, y decides sobre la marcha si algo no encaja. En una cadena de montaje, cada estación hace una única tarea especializada, recibe piezas ya verificadas de la estación anterior y entrega piezas verificadas a la siguiente. Nadie se salta un paso ni improvisa: el proceso completo es repetible, auditable y se puede ejecutar miles de veces sin supervisión humana constante.

Un **pipeline** de TFX es exactamente eso: una secuencia de **componentes** organizados como un grafo acíclico dirigido (DAG, Directed Acyclic Graph), es decir, un diagrama de pasos conectados por flechas en el que los datos solo avanzan hacia delante y nunca vuelven a un paso anterior. Cada componente:

- Recibe uno o varios **artefactos** de entrada (datos, esquemas, modelos) producidos por componentes anteriores
- Realiza una tarea concreta y bien delimitada
- Produce un artefacto de salida, que queda registrado y disponible para los componentes siguientes

Internamente, cada componente de TFX se organiza en tres piezas: un **driver**, que recupera los artefactos de entrada y coordina la ejecución, un **executor**, que es donde ocurre el trabajo real (entrenar, transformar, validar), y un **publisher**, que registra los artefactos de salida. Como usuario del pipeline casi nunca tocas el driver ni el publisher: tu código se concentra en el executor, normalmente a través de una función Python que TFX llama por ti.

### Los componentes principales

| Componente | Qué hace |
|---|---|
| `ExampleGen` | Ingiere los datos de origen (CSV, TFRecord, bases de datos) y los divide en conjuntos de entrenamiento y evaluación |
| `StatisticsGen` | Calcula estadísticas descriptivas de cada feature (valores típicos, valores ausentes, distribuciones) |
| `SchemaGen` | Infiere un esquema (tipos de dato, rangos, categorías válidas) a partir de esas estadísticas |
| `ExampleValidator` | Compara nuevos datos contra el esquema y señala anomalías |
| `Transform` | Aplica ingeniería de features (normalización, codificación de categorías) de forma consistente entre entrenamiento y servicio |
| `Trainer` | Entrena el modelo con TensorFlow a partir de los datos (transformados o no) |
| `Evaluator` | Analiza en profundidad el rendimiento del modelo, incluso por subconjuntos de datos |
| `Pusher` | Publica el modelo entrenado hacia un destino de despliegue (TensorFlow Serving, TensorFlow Lite, TensorFlow.js, TensorFlow Hub) |

TFX también admite componentes personalizados si los estándar no cubren tu caso, y se puede ejecutar desde un notebook (como harás tú), desde la línea de comandos (TFX CLI) o integrado en un IDE.

## Tu primer pipeline: ExampleGen, Trainer y Pusher

El pipeline más simple posible con TFX combina solo tres componentes: ingiere datos, entrena un modelo y lo publica. Es exactamente el que construye el notebook `penguin_simple.ipynb` de TFX, usando el conjunto de datos Palmer Penguins (pingüinos de la Antártida) para clasificar la especie de un pingüino a partir de cuatro medidas físicas normalizadas entre 0 y 1: longitud y profundidad del pico (`culmen_length_mm`, `culmen_depth_mm`), longitud de la aleta (`flipper_length_mm`) y masa corporal (`body_mass_g`).

Primero defines el código de entrenamiento en un archivo Python independiente. TFX lo llama el **Trainer genérico**: tú escribes una función `run_fn` que recibe los argumentos de entrenamiento y guarda el modelo resultante donde TFX espera encontrarlo.

```python
# penguin_trainer.py — código que ejecutará el componente Trainer
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils
from tfx import v1 as tfx
from tfx_bsl.public import tfxio

_FEATURE_KEYS = [
    'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'
]
_LABEL_KEY = 'species'

# Como no usamos un componente de validación todavía, definimos
# manualmente el formato esperado de cada feature.
_FEATURE_SPEC = {
    **{f: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for f in _FEATURE_KEYS},
    _LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
}

def _input_fn(file_pattern, data_accessor, schema, batch_size=200):
    """Convierte los TFRecords generados por ExampleGen en un tf.data.Dataset."""
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key=_LABEL_KEY),
        schema=schema).repeat()

def _build_keras_model():
    # Una red densa sencilla: cuatro entradas, dos capas ocultas, tres salidas
    # (una por especie de pingüino).
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
    d = keras.layers.concatenate(inputs)
    for _ in range(2):
        d = keras.layers.Dense(8, activation='relu')(d)
    outputs = keras.layers.Dense(3)(d)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model

# TFX llama a esta función: es el punto de entrada del componente Trainer.
def run_fn(fn_args: tfx.components.FnArgs):
    schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)
    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema, 20)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema, 10)

    model = _build_keras_model()
    model.fit(train_dataset, steps_per_epoch=fn_args.train_steps,
              validation_data=eval_dataset, validation_steps=fn_args.eval_steps)

    # TFX espera el modelo guardado exactamente en esta ruta.
    model.save(fn_args.serving_model_dir, save_format='tf')
```

Con el código de entrenamiento listo, defines el pipeline en sí, encadenando los tres componentes:

```python
from tfx import v1 as tfx

def _create_pipeline(pipeline_name, pipeline_root, data_root,
                      module_file, serving_model_dir, metadata_path):
    # 1. Ingiere el CSV de entrada y lo divide en train/eval.
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # 2. Entrena el modelo usando el código de penguin_trainer.py.
    trainer = tfx.components.Trainer(
        module_file=module_file,
        examples=example_gen.outputs['examples'],
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=5))

    # 3. Publica el modelo entrenado en una carpeta del sistema de archivos.
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata
            .sqlite_metadata_connection_config(metadata_path),
        components=[example_gen, trainer, pusher])

# Un orquestador ejecuta el DAG. LocalDagRunner corre en local y es
# perfecto para desarrollo; en producción se usan Apache Airflow o Kubeflow.
tfx.orchestration.LocalDagRunner().run(
    _create_pipeline(
        pipeline_name="penguin-simple",
        pipeline_root="pipelines/penguin-simple",
        data_root=DATA_ROOT,
        module_file="penguin_trainer.py",
        serving_model_dir="serving_model/penguin-simple",
        metadata_path="metadata/penguin-simple/metadata.db"))
```

Fíjate en que tú nunca llamas a `fit` directamente ni gestionas archivos a mano: describes qué componentes existen y de dónde toma cada uno sus entradas (`example_gen.outputs['examples']`, `trainer.outputs['model']`), y el `LocalDagRunner` decide el orden de ejecución a partir de esas dependencias.

## Validar los datos con TensorFlow Data Validation (TFDV)

Antes de fiarte de cualquier modelo, tienes que fiarte de los datos que lo alimentan. TensorFlow Data Validation (TFDV) es la biblioteca de TFX dedicada a esto, y se apoya en tres componentes que amplían el pipeline anterior, tal como muestra `penguin_tfdv.ipynb`:

- `StatisticsGen` hace una pasada completa sobre los datos y calcula estadísticas por feature: valores mínimo y máximo, media, número de valores ausentes, histogramas
- `SchemaGen` examina esas estadísticas y genera automáticamente un **esquema**: una descripción formal de qué tipo de dato tiene cada feature, si es obligatoria y qué valores son válidos. Ese esquema queda guardado en un archivo `schema.pbtxt` que puedes revisar y ajustar a mano
- `ExampleValidator` compara los datos nuevos contra el esquema (ya sea el generado automáticamente o uno revisado por una persona) y produce un informe de anomalías: valores del tipo equivocado, categorías inesperadas, features que faltan

La idea clave es que generas el esquema una vez, lo revisas, y luego lo reutilizas como referencia para validar cada nueva remesa de datos que entra al pipeline, exactamente igual que un control de calidad revisa cada lote de piezas contra una plantilla ya aprobada.

```python
def _create_schema_pipeline(pipeline_name, pipeline_root, data_root, metadata_path):
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # NUEVO: calcula estadísticas descriptivas de cada feature.
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])

    # NUEVO: infiere un esquema a partir de esas estadísticas.
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name, pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata
            .sqlite_metadata_connection_config(metadata_path),
        components=[example_gen, statistics_gen, schema_gen])
```

Una vez tienes el esquema exportado, lo importas al pipeline de entrenamiento con un componente `Importer` y añades `ExampleValidator` justo antes del `Trainer`, de forma que cualquier anomalía quede registrada antes de gastar tiempo entrenando con datos sospechosos. Es importante saber que, por defecto, las anomalías detectadas no bloquean el pipeline: quedan documentadas para que una persona decida qué hacer.

## Transformar features con TensorFlow Transform (TFT)

Los cuatro features del ejemplo anterior ya venían normalizados entre 0 y 1. En un caso real casi nunca tienes esa suerte: recibirás medidas en unidades distintas, texto que hay que convertir en números o valores ausentes que hay que gestionar. TensorFlow Transform (TFT) es la biblioteca de TFX para este trabajo, y su idea central es sencilla pero potente: **el mismo código de transformación se ejecuta durante el entrenamiento y durante el servicio del modelo**, así evitas el sesgo entrenamiento-servicio que mencionamos antes.

El notebook `simple.ipynb` muestra el concepto en su forma más reducida, sin pipeline de por medio, solo con la biblioteca `tensorflow_transform`:

```python
import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    """Función que TFT ejecuta tanto en entrenamiento como en servicio."""
    x = inputs['x']
    y = inputs['y']
    s = inputs['s']

    # tft.mean() hace una pasada completa sobre TODO el dataset para
    # calcular la media; no es una operación por ejemplo, como sí lo son
    # las operaciones normales de TensorFlow.
    x_centered = x - tft.mean(x)
    y_normalized = tft.scale_to_0_1(y)
    # Convierte cadenas de texto en índices enteros según su frecuencia.
    s_integerized = tft.compute_and_apply_vocabulary(s)

    return {
        'x_centered': x_centered,
        'y_normalized': y_normalized,
        's_integerized': s_integerized,
    }
```

La distinción importante aquí es entre dos tipos de operaciones dentro de una `preprocessing_fn`:

- Operaciones normales de TensorFlow, que se aplican ejemplo a ejemplo y se repiten igual en entrenamiento y en servicio
- **Analizadores de TFT** (`tft.mean`, `tft.scale_to_z_score`, `tft.compute_and_apply_vocabulary`...), que necesitan una pasada completa sobre el dataset de entrenamiento para calcular una constante (una media, un vocabulario) y luego incrustan esa constante en el grafo, de forma que el servicio la reutiliza sin tener que recalcularla

Dentro de un pipeline de TFX, esto se materializa en el componente `Transform`, como muestra `penguin_tft.ipynb` sobre el conjunto de datos Palmer Penguins sin procesar (con valores en escalas distintas y algunas filas incompletas):

```python
# Dentro del módulo que usan Transform y Trainer:
def preprocessing_fn(inputs):
    outputs = {}
    for key in _FEATURE_KEYS:
        # Normaliza cada feature numérico a media 0 y varianza 1.
        outputs[key] = tft.scale_to_z_score(inputs[key])

    # Convierte la especie (texto) en un índice numérico fijo.
    table_keys = ['Adelie', 'Chinstrap', 'Gentoo']
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=table_keys, values=tf.range(len(table_keys), dtype=tf.int64),
        key_dtype=tf.string, value_dtype=tf.int64)
    table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    outputs[_LABEL_KEY] = table.lookup(inputs[_LABEL_KEY])
    return outputs
```

```python
# En la definición del pipeline:
transform = tfx.components.Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_importer.outputs['result'],
    module_file=module_file)  # aquí vive preprocessing_fn

trainer = tfx.components.Trainer(
    module_file=module_file,
    examples=example_gen.outputs['examples'],
    transform_graph=transform.outputs['transform_graph'],  # NUEVO
    train_args=tfx.proto.TrainArgs(num_steps=100),
    eval_args=tfx.proto.EvalArgs(num_steps=5))
```

El `Trainer` recibe ahora el grafo de transformación (`transform_graph`) además de los datos, y el modelo exportado incluye ese grafo como parte de su función de servicio (`serve_tf_examples_fn`). Así, cuando el modelo ya está en producción y recibe una petición nueva, aplica exactamente las mismas transformaciones que usó durante el entrenamiento, sin que nadie tenga que duplicar ese código a mano.

## Analizar el modelo con TensorFlow Model Analysis (TFMA)

Un modelo puede tener una precisión (accuracy) global aceptable y, aun así, fallar sistemáticamente para un subgrupo de datos que a tu negocio le importa mucho: por ejemplo, un solo indicador de "precisión: 92%" puede esconder que el modelo predice muy mal justo para el segmento de clientes o de productos más rentable. TensorFlow Model Analysis (TFMA) es la biblioteca que te permite ir más allá de la métrica global.

En un pipeline de TFX, esto se materializa en el componente `Evaluator`, que compara el modelo recién entrenado contra el modelo anterior "bendecido" (blessed, el último que superó la validación) y decide si el nuevo modelo es apto para producción. El notebook `penguin_tfma.ipynb` añade este componente al pipeline básico:

```python
import tensorflow_model_analysis as tfma

# Localiza el último modelo que pasó la validación, para usarlo de referencia.
model_resolver = tfx.dsl.Resolver(
    strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
    model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
    model_blessing=tfx.dsl.Channel(type=tfx.types.standard_artifacts.ModelBlessing)
).with_id('latest_blessed_model_resolver')

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='species')],
    slicing_specs=[
        tfma.SlicingSpec(),                          # el dataset completo
        tfma.SlicingSpec(feature_keys=['species']),  # cada especie por separado
    ],
    metrics_specs=[
        tfma.MetricsSpec(per_slice_thresholds={
            'sparse_categorical_accuracy': tfma.PerSliceMetricThresholds(thresholds=[
                tfma.PerSliceMetricThreshold(
                    slicing_specs=[tfma.SlicingSpec()],
                    threshold=tfma.MetricThreshold(
                        # El modelo debe superar el 60% de precisión...
                        value_threshold=tfma.GenericValueThreshold(lower_bound={'value': 0.6}),
                        # ...y no empeorar respecto al modelo anterior.
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10})))
            ]),
        })])

evaluator = tfx.components.Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)

# Pusher solo publica el modelo si el Evaluator lo "bendice".
pusher = tfx.components.Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(base_directory=serving_model_dir)))
```

Las `slicing_specs` (especificaciones de segmentación) son la pieza más importante de este ejemplo: en vez de calcular la precisión solo sobre todo el dataset, TFMA la recalcula además para cada especie de pingüino por separado. Este tipo de análisis por subconjuntos —lo que TFMA llama slicing— es precisamente lo que te permite descubrir que, aunque tu modelo funcione bien en general, falla en un segmento concreto que tu negocio no puede permitirse ignorar. Piensa en el ejemplo de una tienda de zapatos online que usa un modelo para decidir cuánto stock pedir: una caída de precisión solo en "zapatos de vestir para hombre" puede pasar inadvertida en la métrica global y, sin embargo, representar una pérdida importante si esa línea es la más rentable.

Esta idea se recoge en lo que la documentación de TFX llama el **triángulo de insights de ML**: cuando un modelo falla de cara al negocio, casi siempre es porque se ha violado una de tres suposiciones: los datos ya no son representativos, el modelo es demasiado simple para el problema actual o las condiciones de negocio han cambiado (nuevos proveedores, productos, comportamientos de la clientela). Por eso TFX no trata la evaluación como un paso único al final del proyecto, sino como un proceso continuo: cada nueva versión del modelo se compara contra la anterior, y cada caída de rendimiento se investiga empezando por los datos, con las mismas herramientas de TFDV que ya conoces.

## El metadata: la memoria del pipeline

Cada vez que se ejecuta un componente de TFX, el resultado —qué datos entraron, qué parámetros se usaron, qué artefacto se generó— se registra en un almacén llamado **ML Metadata (MLMD)**, normalmente respaldado por una base de datos relacional (SQLite en los ejemplos de este curso, MySQL en producción). Piensa en el metadata como el cuaderno de bitácora de la fábrica: no solo qué se produjo, sino con qué materiales, en qué máquina y con qué resultado, ejecución tras ejecución.

Ese registro te da varias capacidades que serían muy difíciles de replicar a mano:

- **Trazabilidad (lineage)**: puedes rastrear hacia atrás qué versión exacta de los datos entrenó un modelo concreto, o hacia delante qué modelos usaron un dataset determinado
- **Comparación entre ejecuciones**: como cada ejecución queda registrada, puedes comparar el rendimiento de la versión de hoy del modelo con la de la semana pasada
- **Reutilización de resultados (caching)**: si la entrada y el código de un componente no han cambiado desde la última ejecución, TFX puede reutilizar el resultado anterior en vez de recalcularlo, lo que ahorra tiempo especialmente en pasos costosos como la transformación de datos
- **Reproducibilidad y auditoría**: en sectores regulados, poder demostrar exactamente qué datos y qué proceso produjeron un modelo puede ser un requisito legal, no solo una buena práctica

Como estudiante no interactúas casi nunca con el metadata directamente más allá de indicar dónde guardarlo (`metadata_path` en los ejemplos de este capítulo); es TFX quien lo consulta y actualiza por ti en cada paso del pipeline.

## Procesamiento distribuido con Apache Beam

Los pipelines de este capítulo se han ejecutado en tu propia máquina, sobre un dataset de unos pocos cientos de filas. En producción, los datasets pueden tener millones de filas, y calcular estadísticas o aplicar transformaciones sobre todos esos datos en una sola máquina sería, sencillamente, demasiado lento.

Por eso varios componentes de TFX —`ExampleGen`, `StatisticsGen` y `Transform`, entre otros— no procesan los datos directamente: delegan ese trabajo en **Apache Beam**, un modelo de programación unificado para procesamiento de datos por lotes (batch) y en flujo continuo (streaming). Beam separa la lógica de "qué transformación aplicar" de "dónde se ejecuta": el mismo pipeline de Beam puede correr en tu portátil con el runner local (el que usan los ejemplos de este capítulo), o distribuirse sobre un clúster con Apache Spark, Apache Flink o Google Cloud Dataflow, sin cambiar una línea del código de transformación.

No todos los componentes necesitan Beam: el `Trainer` se apoya principalmente en las capacidades de entrenamiento distribuido de TensorFlow, y componentes como el `Pusher` solo necesitan Python puro para copiar un modelo de un sitio a otro. La ventaja para ti como desarrollador es que casi nunca tienes que pensar en esta capa: escribes tu `preprocessing_fn` una vez y TFX se encarga de ejecutarla, ya sea sobre cien filas en tu portátil o sobre cien millones en un clúster en la nube.

## Resumen de puntos clave

- TensorFlow Extended (TFX) es la plataforma de Google para llevar modelos de Machine Learning (ML) de un notebook a un sistema de producción fiable, repetible y auditable
- Un pipeline de TFX es un grafo acíclico dirigido (DAG) de componentes; cada componente toma artefactos de entrada, realiza una tarea y produce artefactos de salida
- El pipeline mínimo combina tres componentes: `ExampleGen` (ingiere datos), `Trainer` (entrena el modelo) y `Pusher` (lo publica)
- TensorFlow Data Validation (TFDV) añade `StatisticsGen`, `SchemaGen` y `ExampleValidator` para calcular estadísticas, inferir un esquema de los datos y detectar anomalías antes de entrenar
- TensorFlow Transform (TFT) añade el componente `Transform`: la función `preprocessing_fn` que escribes se ejecuta igual en entrenamiento y en servicio, evitando el sesgo entrenamiento-servicio
- TensorFlow Model Analysis (TFMA) añade el componente `Evaluator`, que analiza el rendimiento del modelo por segmentos de datos (slicing) y decide, comparando con el modelo anterior, si el nuevo modelo debe "bendecirse" para producción
- El metadata (MLMD) registra cada ejecución del pipeline, lo que permite trazabilidad, comparación entre versiones y reutilización de resultados ya calculados
- Componentes como `ExampleGen`, `StatisticsGen` y `Transform` se ejecutan sobre Apache Beam, lo que permite escalar el mismo código de un portátil a un clúster distribuido sin reescribirlo

## Ponlo en práctica

Abre el notebook de práctica de esta clase y parte del pipeline básico de tres componentes (`ExampleGen`, `Trainer`, `Pusher`) sobre el dataset Palmer Penguins. Tu tarea es añadirle un componente `StatisticsGen` y comprobar el schema inferido: ejecuta el pipeline, localiza el artefacto de estadísticas mediante el metadata del pipeline y observa qué te dice sobre cada uno de los cuatro features numéricos (mínimo, máximo, valores ausentes). Después, responde por escrito a esta pregunta: si mañana llegase un nuevo lote de datos con una especie de pingüino que no existía antes, ¿qué componente de TFX debería detectar el problema, y en qué artefacto de salida esperarías verlo reflejado?
