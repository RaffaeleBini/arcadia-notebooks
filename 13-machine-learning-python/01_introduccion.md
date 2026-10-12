# 1. Introducción al machine learning con Python

## Por qué te importa esto

Imagina que quieres enseñarle a un ordenador a distinguir entre un correo normal y uno de spam, o a predecir el precio de una vivienda a partir de su tamaño y ubicación. No puedes escribir una regla fija para cada caso posible: hay demasiadas combinaciones y excepciones. Aquí es donde entra el aprendizaje automático (ML, del inglés Machine Learning), una rama de la inteligencia artificial (IA) que permite que un programa aprenda patrones a partir de datos, en lugar de seguir instrucciones programadas paso a paso para cada situación.

Este manual te acompaña desde cero. No necesitas saber álgebra lineal ni haber programado antes en proyectos complejos. Solo te hace falta curiosidad y ganas de practicar. A lo largo de doce capítulos vas a construir, paso a paso, tu propio criterio sobre cómo funciona el ML y cómo aplicarlo con Python, uno de los lenguajes de programación más usados en este campo gracias a su sencillez y a sus librerías especializadas.

Aprender esto te importa porque el ML ya forma parte de decisiones cotidianas: qué contenido ves en una aplicación, si un banco te concede un préstamo, cómo se diagnostican enfermedades a partir de imágenes médicas o cómo una empresa agroalimentaria detecta plagas en sus cultivos. Entender los fundamentos te da una base sólida para participar en proyectos de digitalización, evaluar herramientas de IA con criterio propio y, si lo deseas, construir tus propios modelos.

## Qué es el machine learning (con una analogía cotidiana)

En 1959, Arthur Samuel definió el ML como el campo de estudio que da a las máquinas la capacidad de aprender sin programarlas explícitamente para cada tarea. Esta idea lleva más de sesenta años desarrollándose, y hoy la tienes al alcance de tu ordenador personal.

Piensa en cómo aprende un niño o una niña a reconocer perros. Nadie le entrega una lista de reglas exactas ("cuatro patas, cola, pelo, ladra"). En vez de eso, ve muchos ejemplos de perros y de otros animales, y con el tiempo generaliza un patrón que le permite reconocer un perro que nunca antes había visto. El ML funciona de forma parecida: le muestras al modelo muchos ejemplos (datos), y el modelo ajusta sus propios parámetros internos hasta que consigue generalizar un patrón útil para predecir o clasificar casos nuevos.

Existen tres grandes tipos de aprendizaje:

- Aprendizaje supervisado: el modelo aprende a partir de ejemplos etiquetados, es decir, casos donde ya conoces la respuesta correcta. Por ejemplo, fotos de frutas ya clasificadas como "madura" o "verde"
- Aprendizaje no supervisado: el modelo busca patrones o estructuras en datos que no tienen etiquetas, como agrupar clientes con hábitos de compra similares sin decirle de antemano qué grupos existen
- Aprendizaje por refuerzo: el modelo aprende interactuando con un entorno y recibiendo una señal de recompensa, ajustando su comportamiento para maximizarla

Un malentendido habitual es pensar que el ML consiste en programar reglas muy detalladas a mano. Es justo lo contrario: la gracia del ML está en que el propio algoritmo descubre esas reglas a partir de los datos, sin que tú se las dictes una por una.

## Cómo funciona un proyecto de machine learning en Python

Casi todo proyecto de ML sigue un flujo de trabajo parecido, independientemente del algoritmo que elijas. Conocer este flujo desde el principio te ayuda a no perderte cuando lleguemos a los capítulos más técnicos.

**Paso uno: preprocesar los datos.** Los datos del mundo real casi nunca llegan limpios. Puede haber valores ausentes, texto donde esperas números o escalas muy distintas entre columnas (por ejemplo, edad en años y salario en miles de euros). En este paso limpias, transformas y normalizas los datos para que el algoritmo pueda trabajar con ellos correctamente.

**Paso dos: elegir el algoritmo.** Según el tipo de problema que tengas (predecir un número, clasificar en categorías o agrupar elementos similares), eliges un algoritmo adecuado. En este curso vas a trabajar con varios: regresión, k vecinos más cercanos (KNN, del inglés K-Nearest Neighbors), máquinas de vectores de soporte (SVM, del inglés Support Vector Machines) y algoritmos de agrupamiento.

**Paso tres: entrenar el modelo.** Divides tus datos en un conjunto de entrenamiento y otro de prueba. El modelo aprende únicamente con el conjunto de entrenamiento, igual que quien estudia practica con ejercicios antes del examen.

**Paso cuatro: evaluar el rendimiento.** Usas el conjunto de prueba, que el modelo no ha visto antes, para comprobar qué tan bien generaliza lo aprendido. Existen métricas como la exactitud, la precisión, la exhaustividad (recall) o la puntuación F1 para medir esto de forma objetiva.

**Paso cinco: predecir con datos nuevos.** Una vez que confías en el modelo, lo usas para hacer predicciones sobre datos que nunca ha visto: un cliente nuevo, una imagen nueva, una medición nueva.

Para todo este flujo, Python cuenta con librerías especializadas que te ahorran escribir código desde cero:

- **pandas**: organiza y manipula datos en tablas, de forma parecida a una hoja de cálculo pero programable
- **NumPy**: gestiona cálculos numéricos y estructuras de datos matemáticas de forma eficiente
- **scikit-learn**: ofrece implementaciones ya construidas de los algoritmos de ML más comunes, junto con herramientas para dividir datos, entrenar modelos y evaluarlos
- **Matplotlib**: genera gráficos y visualizaciones para entender tus datos y tus resultados

Antes de escribir una sola línea de código de ML, necesitas instalar estas librerías. Se instalan con pip, el gestor de paquetes de Python, desde tu terminal:

```bash
# Instala las librerías principales que usarás en todo el curso
pip install pandas numpy scikit-learn matplotlib
```

Una vez instaladas, las importas al inicio de cualquier script o cuaderno de trabajo:

```python
# Importamos las librerías principales del curso
import pandas as pd          # para cargar y organizar datos en tablas
import numpy as np           # para cálculos numéricos
from sklearn.model_selection import train_test_split  # para dividir datos
import matplotlib.pyplot as plt  # para visualizar resultados

# Comprobación rápida: si no hay errores, todo está listo
print("Librerías cargadas correctamente")
```

Este pequeño bloque de código no hace ningún cálculo de ML todavía, pero es el punto de partida de prácticamente cualquier proyecto que vas a construir en este manual. Si lo ejecutas y ves el mensaje impreso sin errores, tu entorno está listo para avanzar.

## El mapa de este curso

Este manual organiza doce capítulos en cinco bloques temáticos. Cada algoritmo que vas a estudiar se explica siguiendo tres pasos: primero la teoría (la intuición de cómo funciona), después la aplicación práctica con scikit-learn (usarlo ya construido sobre datos reales) y finalmente la implementación desde cero (programar el algoritmo tú mismo, con las matemáticas incluidas, para entender a fondo su funcionamiento interno). Este enfoque en tres pasos es la columna vertebral de todo el curso.

- **Bloque uno, introducción (capítulo uno):** este capítulo que estás leyendo, donde sitúas el ML dentro de la IA y conoces el flujo de trabajo típico con Python
- **Bloque dos, regresión (capítulos dos a cuatro):** aplicas la regresión lineal con scikit-learn, aprendes a guardar y escalar modelos e interpretar sus resultados, y por último programas la regresión lineal desde cero
- **Bloque tres, clasificación con KNN (capítulos cinco y seis):** aplicas el algoritmo de k vecinos más cercanos con scikit-learn y después lo implementas desde cero
- **Bloque cuatro, SVM (capítulos siete a nueve):** estudias los fundamentos de las máquinas de vectores de soporte, las programas desde cero y profundizas en conceptos como los núcleos (kernels) y el margen blando
- **Bloque cinco, agrupamiento y cierre (capítulos diez a doce):** trabajas con los algoritmos de agrupamiento K-Means y Mean Shift, y cierras el curso con casos de uso reales que integran todo lo aprendido

Un apunte importante: este curso no cubre redes neuronales ni aprendizaje profundo (deep learning). Esa es otra rama del ML, con su propio enfoque y complejidad, y merece un curso dedicado. Aquí te vas a centrar en algoritmos más sencillos de entender matemáticamente, que te dan una base sólida antes de dar ese salto en el futuro si lo deseas.

Vale la pena también un poco de contexto histórico. Vladimir Vapnik introdujo la SVM en 1963, pero el algoritmo no se popularizó hasta los años noventa, cuando superó a las redes neuronales en tareas de reconocimiento de caracteres manuscritos. Las SVM dominaron el campo del ML durante años, hasta el resurgimiento reciente de las redes neuronales con el deep learning. Si sientes que llegas tarde a este campo, no es así: el poder de cómputo y el acceso a herramientas que tienes hoy superan por completo lo que existía hace apenas unas décadas.

## Resumen: lo que te llevas de este capítulo

- El machine learning es una rama de la inteligencia artificial que permite a un programa aprender patrones a partir de datos, sin programar cada regla a mano
- Existen tres tipos principales de aprendizaje: supervisado, no supervisado y por refuerzo
- Un proyecto de ML sigue un flujo típico: preprocesar datos, elegir algoritmo, entrenar, evaluar y predecir
- Python es un lenguaje ideal para el ML gracias a librerías como pandas, NumPy, scikit-learn y Matplotlib
- Cada algoritmo de este curso se explica en tres pasos: teoría, aplicación con scikit-learn e implementación desde cero
- Este curso cubre regresión, KNN, SVM y agrupamiento, pero no incluye redes neuronales ni deep learning

## Ponlo en práctica

**Prepara tu entorno ahora mismo.** Antes de avanzar al capítulo dos, comprueba que tienes Python 3 instalado en tu equipo y que puedes instalar y cargar las librerías que vas a usar en todo el curso.

1. Abre tu terminal y ejecuta `python --version` para confirmar que tienes Python 3 instalado
2. Instala las librerías con el comando `pip install pandas numpy scikit-learn matplotlib`
3. Crea un archivo llamado `prueba_entorno.py` y copia el bloque de código de importación que viste en este capítulo
4. Ejecuta el archivo con `python prueba_entorno.py`
5. **Confirma** que ves el mensaje "Librerías cargadas correctamente" sin ningún error en pantalla

Si completas estos cinco pasos sin errores, tu entorno está listo para el bloque de regresión, que empieza en el próximo capítulo.

**Siglas usadas en este capítulo:** ML (Machine Learning, aprendizaje automático), IA (inteligencia artificial), KNN (K-Nearest Neighbors, k vecinos más cercanos), SVM (Support Vector Machines, máquinas de vectores de soporte)
