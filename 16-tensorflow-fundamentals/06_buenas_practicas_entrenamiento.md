# 06. Buenas prácticas de entrenamiento

## Por qué importa este tema

Ya sabes construir y entrenar modelos con Keras, la API de alto nivel de
TensorFlow. Pero conseguir que un modelo entrene no es lo mismo que conseguir
un modelo útil. Puedes entrenar una red que acierte el 99% de las veces con
los datos que ya conoce y que falle estrepitosamente en cuanto le muestras un
ejemplo nuevo. Ese modelo no sirve para nada en producción: memorizó el
examen en lugar de aprender la materia.

En este capítulo aprendes a diagnosticar ese problema —y su contrario, un
modelo que ni siquiera aprende bien los datos de entrenamiento— y a aplicar
las técnicas estándar para corregirlo. También aprendes algo igual de
práctico: cómo guardar un modelo ya entrenado para no tener que repetir horas
de entrenamiento cada vez que lo necesitas, y cómo compartirlo con otra
persona o desplegarlo en otro entorno.

Estas no son técnicas opcionales ni "avanzadas": son la diferencia entre un
ejercicio de clase y un modelo que puedes llevar a un caso real.

## Sobreajuste e infra-ajuste: dos formas de fallar

Imagina a dos personas preparando el mismo examen tipo test con un banco de
100 preguntas de práctica.

La primera se aprende de memoria la respuesta exacta de cada una de las 100
preguntas, sin entender por qué esa respuesta es la correcta. El día del
examen real, con preguntas parecidas pero no idénticas, falla casi todo:
memorizó el banco de preguntas en lugar de aprender la materia. Esto es
**sobreajuste** (overfitting): el modelo aprende tan bien los datos de
entrenamiento —incluido su ruido y sus particularidades— que pierde la
capacidad de generalizar a datos nuevos.

La segunda persona apenas ha abierto el libro: no domina ni siquiera las
preguntas de práctica. El día del examen falla tanto en las preguntas
conocidas como en las nuevas. Esto es **infra-ajuste** (underfitting): el
modelo es demasiado simple, o no se ha entrenado lo suficiente, para captar
los patrones que existen en los datos.

### Cómo detectarlos con las curvas de entrenamiento y validación

Cuando entrenas un modelo con Keras, en cada época puedes registrar dos
valores de pérdida (loss, el error que comete el modelo): la pérdida sobre
los datos de entrenamiento y la pérdida sobre un conjunto de validación que
el modelo no usa para ajustar sus pesos. Comparar esas dos curvas es la forma
más directa de diagnosticar el problema:

- **Todo va bien:** las dos curvas bajan juntas y se estabilizan a un nivel
  bajo, con una diferencia pequeña entre ellas
- **Sobreajuste:** la pérdida de entrenamiento sigue bajando, pero la de
  validación deja de bajar y empieza a subir. A partir de ese punto el modelo
  ha dejado de aprender patrones generales y ha empezado a memorizar el
  ruido de los datos de entrenamiento
- **Infra-ajuste:** las dos curvas se quedan altas y apenas bajan, incluso
  después de muchas épocas (una época es una pasada completa por todo el
  conjunto de entrenamiento). El modelo no tiene capacidad suficiente, o no
  ha entrenado lo bastante, para aprender la relación entre los datos y el
  resultado que quieres predecir

Un ejemplo real: si entrenas tres modelos con distinta capacidad (número de
capas y neuronas) sobre el mismo problema, normalmente observas que el modelo
más pequeño tarda más en bajar su pérdida de entrenamiento (le cuesta
aprender: infra-ajuste leve), mientras que el modelo más grande baja su
pérdida de entrenamiento muy rápido pero su pérdida de validación empieza a
subir enseguida (sobreajuste). El objetivo es encontrar el punto intermedio:
la capacidad mínima necesaria para aprender bien el problema, ni más ni
menos.

## Técnicas para corregir el sobreajuste

No existe una fórmula única para elegir el tamaño perfecto de un modelo:
hace falta experimentar. Pero sí existen técnicas contrastadas que reducen el
sobreajuste cuando aparece.

**Conseguir más datos de entrenamiento.** Es la solución más efectiva cuando
es viable: un modelo entrenado con datos más completos generaliza mejor de
forma natural, porque tiene menos margen para memorizar casos concretos.

**Reducir la capacidad del modelo.** Si tu red tiene muchas más neuronas y
capas de las que el problema necesita, tiene "memoria" de sobra para
memorizar el conjunto de entrenamiento entero en lugar de aprender patrones
generales. Empieza con un modelo pequeño y ve aumentando su tamaño solo si la
pérdida de validación mejora al hacerlo.

**Regularización L2 (weight decay).** Añade a la función de pérdida una
penalización proporcional al cuadrado del valor de los pesos de la red. En
la práctica, esto obliga al modelo a mantener sus pesos pequeños, lo que
limita cuánto "detalle" puede memorizar de los datos de entrenamiento. Cuanto
mayor es el parámetro de regularización (llamado lambda), más fuerte es esa
penalización. En Keras se aplica capa a capa:

```python
from tensorflow.keras import layers, regularizers

layers.Dense(512, activation="relu",
             kernel_regularizer=regularizers.l2(0.001))
```

Existe también la regularización L1, que penaliza el valor absoluto de los
pesos en lugar de su cuadrado y tiende a llevar algunos pesos exactamente a
cero (selección de características). L2 es la más habitual en redes
neuronales porque penaliza los pesos grandes sin forzar la dispersión.

**Dropout.** Es una de las técnicas de regularización más usadas en redes
neuronales. Durante el entrenamiento, la capa de dropout "apaga" al azar un
porcentaje de las neuronas de la capa anterior en cada paso (se les asigna
un valor de cero), obligando a la red a no depender en exceso de ninguna
neurona concreta. La tasa de dropout habitual está entre 20% y 50%. En la
fase de evaluación o predicción, el dropout se desactiva y se usa la red
completa:

```python
layers.Dense(512, activation="relu"),
layers.Dropout(0.5),
```

**Early stopping (parada temprana).** En lugar de fijar de antemano un
número de épocas, controlas la pérdida de validación durante el
entrenamiento y detienes el proceso en cuanto deja de mejorar durante un
número de épocas seguidas (el parámetro `patience`). Así evitas seguir
entrenando después del punto en el que el modelo empieza a sobreajustar, y
además ahorras tiempo de cómputo:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=5,
                            restore_best_weights=True)
```

**Aumento de datos (data augmentation).** Cuando conseguir más datos reales
no es posible, puedes generar variaciones artificiales de los datos que ya
tienes. En imágenes, por ejemplo, aplicar rotaciones, recortes o volteos
horizontales a las fotos de entrenamiento crea ejemplos nuevos y válidos sin
necesidad de recolectar ni una imagen más. Esto reduce el sobreajuste porque
el modelo ve más variedad y depende menos de detalles concretos de cada
imagen original.

En la práctica, ninguna de estas técnicas es excluyente: combinar dropout,
regularización L2 y early stopping suele dar mejores resultados que aplicar
una sola.

### Y si el problema es infra-ajuste

Si tu modelo ni siquiera aprende bien los datos de entrenamiento, la
solución va en la dirección contraria: aumenta la capacidad del modelo
(más capas o más neuronas), entrena durante más épocas, o revisa si las
características (features) que le das de entrada son suficientes para
resolver el problema.

## Guardar y cargar modelos

Entrenar un modelo puede llevar minutos, horas o días. Sin guardarlo,
pierdes ese trabajo en cuanto cierras el proceso: no puedes reutilizarlo, ni
compartirlo, ni seguir entrenándolo más tarde desde donde lo dejaste.
TensorFlow ofrece varias formas de guardar el trabajo, según qué necesites
conservar.

### Guardar solo los pesos: checkpoints

Un checkpoint guarda únicamente los valores aprendidos por el modelo (sus
pesos), no su arquitectura ni la configuración del optimizador. Para
recuperarlos necesitas reconstruir un modelo con exactamente la misma
arquitectura y después cargar los pesos guardados en él. Esto es útil, por
ejemplo, para retomar un entrenamiento largo que se interrumpió, o para
reutilizar los pesos aprendidos en una arquitectura ligeramente distinta.

La forma más práctica de guardar checkpoints automáticamente durante el
entrenamiento es el callback `ModelCheckpoint`:

```python
import os
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = "training/cp.ckpt"

cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                               save_weights_only=True,
                               verbose=1)

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])
```

Para reutilizar esos pesos más tarde, creas un modelo con la misma
arquitectura (sin entrenar) y le cargas los pesos guardados:

```python
model = create_model()          # misma arquitectura, pesos sin entrenar
model.load_weights(checkpoint_path)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Precisión del modelo restaurado: {acc:.2%}")
```

También puedes guardar y cargar los pesos manualmente, sin pasar por el
entrenamiento, con `model.save_weights(ruta)` y `model.load_weights(ruta)`.

### Guardar el modelo completo

Si quieres conservar la arquitectura, los pesos y el estado del optimizador
en un único archivo —para poder cargarlo y usarlo directamente sin volver a
definir el modelo en código—, usa `model.save()`. El formato recomendado en
TensorFlow 2.x es `.keras`:

```python
model.save("mi_modelo.keras")

# Más tarde, en el mismo script o en otro distinto:
modelo_cargado = tf.keras.models.load_model("mi_modelo.keras")
modelo_cargado.summary()
```

El modelo cargado con `load_model` está listo para usarse tal cual:
puedes llamar a `evaluate()` o `predict()` sin recompilarlo, porque el
archivo incluye también la configuración del optimizador y de la función de
pérdida.

TensorFlow admite otros dos formatos, que sigues encontrando en proyectos
existentes:

| Formato | Extensión | Qué guarda | Cuándo usarlo |
|---|---|---|---|
| Keras nativo | `.keras` | Arquitectura, pesos y optimizador | Formato recomendado por defecto en TF 2.x |
| SavedModel | directorio, sin extensión | Arquitectura, pesos y el grafo de ejecución completo | Cuando necesitas servir el modelo con TensorFlow Serving o usarlo en TensorFlow Lite o TensorFlow.js |
| HDF5 | `.h5` | Arquitectura, pesos y optimizador | Formato heredado (legacy), útil para compatibilidad con proyectos antiguos |

HDF5 (Hierarchical Data Format, versión 5) fue el formato estándar en las
primeras versiones de TensorFlow 2.x y sigue siendo compatible, pero para
proyectos nuevos usa `.keras`.

## Resumen de puntos clave

- El sobreajuste (overfitting) ocurre cuando la pérdida de entrenamiento
  sigue bajando mientras la de validación se estanca o sube: el modelo
  memoriza en lugar de generalizar
- El infra-ajuste (underfitting) ocurre cuando ambas pérdidas se quedan
  altas: el modelo no tiene capacidad o entrenamiento suficiente
- Comparar las curvas de entrenamiento y validación en cada época es la
  forma más directa de diagnosticar cuál de los dos problemas tienes
- Dropout, regularización L2, early stopping, aumento de datos y reducir la
  capacidad del modelo son las técnicas estándar contra el sobreajuste, y
  combinarlas suele funcionar mejor que aplicar solo una
- Los checkpoints (`ModelCheckpoint`, `save_weights` / `load_weights`)
  guardan solo los pesos y exigen reconstruir la misma arquitectura para
  reutilizarlos
- `model.save()` y `tf.keras.models.load_model()` guardan y recuperan el
  modelo completo —arquitectura, pesos y optimizador— en formato `.keras`,
  el recomendado en TensorFlow 2.x

## Ponlo en práctica

Coge cualquier modelo que hayas entrenado en un capítulo anterior de este
curso (por ejemplo, el clasificador de imágenes del capítulo 2 o el modelo
de clasificación del capítulo 5) y haz lo siguiente:

1. Entrénalo durante más épocas de las necesarias y dibuja las curvas de
   pérdida de entrenamiento y de validación. Identifica a partir de qué
   época empieza a sobreajustar
2. Añade una capa `Dropout` y, si el modelo lo permite, regularización L2 a
   alguna de sus capas densas. Vuelve a entrenar y compara las nuevas curvas
   con las del paso anterior: ¿se retrasa o desaparece el sobreajuste?
3. Guarda los pesos del mejor modelo que obtengas con un `ModelCheckpoint`
   y, en una celda o script aparte, reconstruye la arquitectura desde cero y
   carga esos pesos con `load_weights`. Comprueba que la precisión coincide
   con la del modelo original

Si quieres seguir practicando de forma guiada, abre el notebook de práctica
de esta clase: `clase04_practica_buenas_practicas_entrenamiento.ipynb`.
