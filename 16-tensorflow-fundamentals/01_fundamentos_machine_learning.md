# 01. Fundamentos de Machine Learning

## Por qué importa este tema

Antes de escribir una sola línea de código con TensorFlow (TF), necesitas entender qué problema resuelve. TensorFlow es una biblioteca de código abierto desarrollada por Google que sirve para construir y entrenar modelos de Machine Learning (ML), es decir, de Aprendizaje Automático. Si no tienes claro qué significa "aprender" para una máquina, ningún ejemplo de código te va a resultar intuitivo. Este capítulo te da esa base conceptual: la diferencia entre programar y entrenar, qué es un modelo y por qué hace falta una herramienta como TensorFlow para todo esto.

## Programación tradicional frente a Machine Learning

Imagina que quieres que un ordenador reconozca si una foto muestra piedra, papel o tijera (el juego "Rock, Paper, Scissors"). Para una persona es trivial, pero para programarlo con reglas explícitas es un problema enorme: cada persona tiene manos distintas, tonos de piel diferentes y formas propias de hacer el gesto de las tijeras. Si tuvieras que describir con código, condición por condición, qué combinación exacta de píxeles representa cada gesto, necesitarías miles de líneas y aun así fallarías con casos que no previste.

Esta dificultad revela la diferencia esencial entre los dos paradigmas:

- **Programación tradicional**: tú escribes las reglas (el código) y el ordenador aplica esas reglas a los datos de entrada para producir un resultado. Reglas + datos → resultado
- **Machine Learning**: le das al ordenador los datos de entrada junto con los resultados correctos (las respuestas) y dejas que sea el propio sistema el que deduzca las reglas. Datos + resultado → reglas

En otras palabras, el ML invierte el proceso habitual de programar. En vez de anticipar tú todos los casos posibles, entrenas un modelo con muchos ejemplos —muchas fotos de piedra, papel y tijera ya etiquetadas— y dejas que encuentre los patrones que distinguen a un gesto de otro. Ese es exactamente el planteamiento que usarás en clases posteriores para construir un clasificador de imágenes real.

## Un ejemplo más sencillo que el de las manos

Reconocer gestos con la mano es un problema complejo para empezar. Por eso, antes de llegar ahí, conviene practicar con algo mucho más simple: un conjunto de números con una relación oculta entre ellos.

Fíjate en esta tabla de valores de X e Y:

| X | -1 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| Y | -3 | -1 | 1 | 3 | 5 | 7 |

Si observas el patrón, puedes deducir que la relación es Y = 2X - 1. Un programador tradicional escribiría directamente esa fórmula en código. En Machine Learning, en cambio, le entregarías al ordenador estos pares de números sin decirle la fórmula, y el sistema —mediante un proceso de prueba y ajuste— la iría deduciendo por sí mismo. Este es el ejemplo que retomarás con código real en el capítulo siguiente.

## Qué es un modelo y qué significa entrenarlo

Un **modelo** es la representación matemática que el sistema construye para capturar la relación entre los datos de entrada (features o características) y los datos de salida (etiquetas o resultados esperados). Al principio, un modelo no sabe nada: sus parámetros internos suelen inicializarse con valores aleatorios, así que sus primeras predicciones son prácticamente adivinanzas. Por ejemplo, ante la relación real Y = 2X - 1, el modelo podría empezar suponiendo algo tan alejado como Y = 5X + 5.

**Entrenar** un modelo es el proceso de ajustar esos parámetros internos, paso a paso, para que sus predicciones se acerquen cada vez más a los resultados reales. Este proceso se apoya en dos piezas clave que verás en detalle en el próximo capítulo:

- Una **función de pérdida (loss)**, que mide numéricamente cuánto se equivoca el modelo en cada intento
- Un **optimizador**, que usa esa medida de error para proponer un ajuste mejor en la siguiente vuelta

Este ciclo —predecir, medir el error, ajustar— se repite muchas veces. Cada vuelta completa sobre el conjunto de datos de entrenamiento se llama **época (epoch)**. Con el tiempo, el modelo converge hacia una fórmula que explica los datos razonablemente bien, aunque —y esto es importante— casi nunca de forma perfectamente exacta.

## Qué son los datos de entrenamiento

Los **datos de entrenamiento** son el conjunto de ejemplos que usas para que el modelo aprenda: en el caso de los números, son los pares de valores X e Y; en el caso de piedra-papel-tijera, serían miles de fotos ya etiquetadas con el gesto que representan. Cuantos más ejemplos representativos tengas, más fiable será el patrón que el modelo pueda extraer. Con solo seis pares de números, como en el ejemplo anterior, el modelo puede acercarse mucho a la fórmula real, pero no tiene forma de estar absolutamente seguro de ella.

Esto explica un comportamiento que te vas a encontrar constantemente en Machine Learning: las predicciones no suelen ser exactas, sino aproximaciones con un grado de incertidumbre. Si entrenas el modelo del ejemplo anterior y le pides que prediga Y para X = 10, es probable que obtengas algo como 18,9998 en lugar de 19 exacto. No es un error del sistema: es la consecuencia lógica de aprender a partir de una muestra limitada de datos y de trabajar con probabilidades en lugar de con certezas absolutas.

## Por qué TensorFlow es útil para todo esto

Podrías, en teoría, programar tú mismo el ciclo de predecir-medir-ajustar. Pero hacerlo a mano —calculando derivadas, gestionando matrices de parámetros, distribuyendo el cálculo entre CPU (Central Processing Unit) y GPU (Graphics Processing Unit)— es lento y propenso a errores. TensorFlow te da:

- Una estructura de datos común, el **tensor**, para representar cualquier tipo de dato numérico (números, listas, imágenes, texto convertido a números)
- Operaciones matemáticas ya optimizadas para ejecutarse en paralelo, tanto en CPU como en GPU
- Una interfaz de alto nivel, **Keras**, que te permite definir un modelo apilando capas en pocas líneas, sin escribir tú el cálculo de gradientes ni el ciclo de entrenamiento

Gracias a esto, el mismo patrón conceptual que acabas de ver con números —datos + resultado → reglas, ajustadas mediante pérdida y optimizador— se aplica igual de bien a problemas mucho más complejos: reconocer dígitos escritos a mano, clasificar imágenes médicas o analizar el sentimiento de un texto. En el capítulo siguiente vas a escribir tu primer código con TensorFlow y Keras para poner en práctica exactamente este ciclo.

## Resumen de puntos clave

- La programación tradicional parte de reglas que tú defines; el Machine Learning parte de datos y resultados, y deduce las reglas
- Un modelo es una representación matemática que relaciona entradas y salidas, y que empieza con parámetros prácticamente aleatorios
- Entrenar consiste en repetir el ciclo predecir → medir el error con una función de pérdida → ajustar con un optimizador, época tras época
- Los datos de entrenamiento son los ejemplos con los que el modelo aprende; cuantos más y más representativos, mejor generaliza el modelo
- Las predicciones de un modelo son aproximaciones probables, no certezas exactas, porque se basan en una muestra limitada de datos
- TensorFlow y Keras te dan la estructura (tensores) y las herramientas (capas, funciones de pérdida, optimizadores) para aplicar este ciclo sin programarlo desde cero

## Ponlo en práctica

Antes de abrir un notebook, coge papel y boli. Escribe esta tabla de valores:

| X | -2 | -1 | 0 | 1 | 2 |
|---|---|---|---|---|---|
| Y | -6 | -3 | 0 | 3 | 6 |

Deduce a mano la relación entre X e Y y anota la fórmula que crees que la explica. Después, imagina que eres el "optimizador": tu primera suposición es Y = X + 1. Calcula qué error (la diferencia entre tu predicción y el valor real de Y) cometerías para cada X de la tabla, y anota cómo cambiarías tu fórmula para reducir ese error en la siguiente vuelta. Ese ajuste manual, repetido muchas veces y de forma automática, es justo lo que hace TensorFlow cuando entrena un modelo.
