# 09. Casos de uso reales

## Por qué importa este tema

Has recorrido el camino completo: tensores, datos y `DataLoader`, una red
densa entrenada sobre MNIST, una red neuronal convolucional (CNN,
Convolutional Neural Network) entrenada sobre CIFAR-10, computación en GPU
(Graphics Processing Unit) y, por último, cómo analizar un modelo ya
entrenado. Todo ese aprendizaje puede quedarse en la teoría si no lo conectas
con problemas reales. Este capítulo cierra el curso mostrándote, a partir de
las propias menciones del material oficial de la certificación EITC/AI/DLPP
(European IT Certificate — Certificado Europeo de Tecnologías de la
Información — en Inteligencia Artificial, especialidad Deep Learning with
Python and PyTorch), en qué dominios se aplica cada pieza de PyTorch que ya
conoces.

No vas a aprender ninguna sintaxis nueva. Vas a ver que la clasificación de
imágenes de la clase 6, el `torch.device` de la clase 7 y la matriz de
confusión de la clase 8 no son ejercicios aislados: son exactamente las
piezas que sostienen sistemas de visión por computador, modelos de lenguaje
y despliegues en producción que ya existen fuera de este curso. El objetivo
es que dejes de pensar "¿qué técnica me falta por aprender?" y empieces a
pensar "¿qué problema quiero resolver con lo que ya sé?". Esa pregunta es
exactamente la que vas a responder en tu proyecto final.

## Visión por computador: de un dígito a un coche que conduce solo

El propio material de la certificación presenta la visión por computador
como uno de los primeros dominios de éxito del aprendizaje profundo (deep
learning), junto al procesamiento de lenguaje natural (NLP, Natural Language
Processing) y el reconocimiento de voz. La pieza que lo hace posible en
PyTorch es `torchvision`, la biblioteca que ya usaste en la clase 2 para
cargar MNIST y en la clase 6 para entrenar tu CNN sobre CIFAR-10: incluye
datasets de referencia como ImageNet, transformaciones para preprocesar
imágenes y modelos preentrenados listos para adaptar a un problema nuevo.

La arquitectura convolucional que construiste en la clase 5 no es solo un
ejercicio académico: el propio material describe la CNN como la arquitectura
que ha revolucionado la clasificación y el reconocimiento de imágenes a
color, precisamente porque cada canal RGB (Red-Green-Blue — Rojo-Verde-Azul)
se representa de forma natural como una dimensión más del tensor de entrada,
sin trucos añadidos. Esa misma técnica, escalada y combinada con datasets
mucho más grandes, es la base de sistemas de reconocimiento de objetos en
fotos, de diagnóstico por imagen médica y de la percepción visual de
vehículos con conducción autónoma.

Precisamente sobre conducción autónoma, el material insiste en un matiz que
ya trabajaste en la clase 8: entrenar un buen modelo no basta, hay que
evaluarlo en condiciones distintas a las de entrenamiento —diferente
iluminación, distinto clima, distinto tipo de carretera— para saber si de
verdad generaliza o si solo memorizó las condiciones concretas de su
conjunto de datos. Es la misma lógica del sobreajuste que viste con la
matriz de confusión de MNIST, aplicada a un contexto donde un error de
clasificación tiene consecuencias mucho más serias.

## Procesamiento de lenguaje natural: de torchtext a los grandes modelos de lenguaje

El segundo gran dominio que aparece de forma constante en el material es el
procesamiento de lenguaje natural. `torchtext`, la tercera biblioteca del
ecosistema de PyTorch junto a `torchvision` y `torchaudio`, ofrece
herramientas para cargar y preprocesar texto y construir modelos de
lenguaje, siguiendo el mismo patrón de dataset y modelo que ya conoces de la
clase 2 y la clase 3.

El material ilustra la escala que puede alcanzar este dominio con un
ejemplo concreto: GPT-3 (Generative Pre-trained Transformer 3), desarrollado
por OpenAI, es un modelo basado en la arquitectura transformer con 175.000
millones de parámetros, capaz de resolver una amplia variedad de tareas de
procesamiento de lenguaje natural con alta precisión. Es la misma idea de
"parámetros que la red aprende" que calculaste a mano en la clase 3 para tu
primera red densa, solo que multiplicada por un factor casi inimaginable. El
material también señala que PyTorch mejora de forma continua su integración
con Hugging Face Transformers, la biblioteca más usada hoy para trabajar con
modelos de lenguaje preentrenados, lo que te permite partir de un modelo ya
entrenado en lugar de entrenar uno desde cero.

## Cómputo a gran escala en GPU: por qué salir de tu portátil importa

La clase 7 te enseñó a mover tensores y modelos a una GPU con
`torch.device` y a medir la diferencia de tiempo frente a la CPU (Central
Processing Unit). El material es explícito sobre por qué esa habilidad
importa más allá del ejercicio de clase: para problemas reales, una GPU de
gama alta deja de ser un lujo y se convierte en un requisito, porque el
volumen de datos y el tamaño de los modelos superan por completo lo que una
CPU puede procesar en un tiempo razonable.

El ejemplo de GPT-3 vuelve a ser útil aquí: entrenar un modelo con 175.000
millones de parámetros exige recursos computacionales considerables, una
gestión de memoria cuidadosa —con técnicas como el paralelismo de modelos— y
tiempos de entrenamiento que solo son viables repartiendo el trabajo entre
muchas GPU o unidades de procesamiento tensorial (TPU, Tensor Processing
Unit) a la vez, un enfoque de computación de altas prestaciones (HPC,
High-Performance Computing). El material también recuerda una decisión
práctica que ya conoces: si no dispones de una GPU local, puedes alquilar
una en la nube y pagar solo por las horas que la usas, exactamente el mismo
razonamiento de coste que aplicarías en cualquier proyecto profesional.

Vale la pena recordar, además, un dato de contexto que el propio material
menciona sobre el origen de la herramienta que has usado en todo el curso:
PyTorch fue desarrollado por el laboratorio de investigación en inteligencia
artificial de Facebook (FAIR, Facebook AI Research), hoy integrado en Meta.
Es un hecho ampliamente documentado que empresas como Tesla han hablado
públicamente del uso de PyTorch en sus sistemas de visión por computador
para conducción autónoma, y que OpenAI adoptó PyTorch como su marco de
trabajo principal para el desarrollo de modelos de deep learning. No son
casos aislados de un entorno académico: son la misma librería, con la misma
sintaxis de tensores, `nn.Module` y bucle de entrenamiento que ya dominas.

## De un modelo entrenado a un producto: evaluación, interpretabilidad y despliegue

La clase 8 te enseñó a distinguir sobreajuste de infra-ajuste, a leer una
matriz de confusión y a guardar los pesos de un modelo con `torch.save`.
El material amplía esa idea con técnicas de interpretabilidad que resultan
imprescindibles cuando un modelo pasa de un notebook a una aplicación real:
visualizar qué filtros aprendió una CNN, analizar los mecanismos de atención
de un modelo de lenguaje para entender en qué palabras se fija, generar
mapas de relevancia (saliency maps) que señalan qué región de una imagen
influyó más en una predicción, o estimar la incertidumbre de un modelo con
técnicas como el Monte Carlo Dropout. Todas ellas responden a la misma
pregunta que te hiciste en la clase 8 al inspeccionar predicciones
incorrectas: no basta con saber que el modelo se equivoca, hay que entender
por qué.

Guardar los pesos de un modelo con `torch.save`, como aprendiste en la
clase 8, es solo el primer paso hacia la producción. El material describe
varias herramientas que dan el siguiente paso: `torch.quantization` reduce
el tamaño de un modelo y acelera su inferencia, algo especialmente útil para
dispositivos móviles y del internet de las cosas (IoT, Internet of Things);
TorchScript permite serializar un modelo para ejecutarlo fuera de Python, en
un entorno de producción; el formato ONNX (Open Neural Network Exchange)
facilita exportar un modelo de PyTorch para que funcione con otras
herramientas; y TorchServe es el marco de trabajo oficial de PyTorch para
servir modelos ya entrenados a través de una API (Application Programming
Interface) en producción, con monitorización incluida. Ninguna de estas
herramientas sustituye lo que ya sabes hacer: todas parten de un modelo
entrenado y evaluado exactamente como el que construiste en las clases 3, 4,
6 y 8.

## Resumen de puntos clave

- La visión por computador que trabajaste con MNIST y CIFAR-10 (clases 2, 5
  y 6) es la misma técnica que sostiene sistemas de reconocimiento de
  imágenes y de percepción en conducción autónoma, evaluados en condiciones
  distintas a las de entrenamiento, igual que aprendiste en la clase 8
- El procesamiento de lenguaje natural, con `torchtext` y modelos como GPT-3
  (175.000 millones de parámetros) o la integración con Hugging Face
  Transformers, aplica el mismo patrón de dataset, modelo y entrenamiento de
  las clases 2, 3 y 4, solo que a una escala mucho mayor
- El cómputo en GPU de la clase 7 deja de ser un detalle de rendimiento y se
  convierte en un requisito cuando el modelo o el dataset crecen; PyTorch,
  originado en el laboratorio de investigación de Facebook (FAIR), es hoy
  una herramienta de uso documentado en empresas como Tesla y OpenAI
- Evaluar un modelo (clase 8) no termina en la precisión y la matriz de
  confusión: técnicas de interpretabilidad como los mapas de relevancia, el
  análisis de atención o la estimación de incertidumbre son imprescindibles
  antes de llevar un modelo a producción
- Herramientas como `torch.quantization`, TorchScript, ONNX y TorchServe dan
  el paso final desde un modelo guardado con `torch.save` hasta un producto
  real, en un móvil, un dispositivo IoT o un servicio en la nube

## Y ahora, tu proyecto final

Has visto que cada técnica del curso tiene un dominio de aplicación real
detrás. Ahora te toca a ti elegir el tuyo. Tu proyecto final no necesita
resolver un problema tan grande como GPT-3 ni tan crítico como la conducción
autónoma: necesita demostrar que sabes recorrer el patrón completo del curso
—tensores, dataset, modelo, entrenamiento y evaluación— sobre un problema
que te interese de verdad.

### Paso 1: elige un problema y un dataset

Empieza por el problema, no por la técnica. Pregúntate qué quieres predecir
o clasificar, y por qué te importa: puede ser un problema de tu sector
profesional, algo relacionado con las cadenas de valor que trabaja DIH
DATAlife (Agro-Mar Alimentación, Forestal-Madera, Salud-Cuidados o
Biotecnología) o simplemente algo que te resulte curioso. Tienes dos
caminos para el dataset:

- Reutilizar uno de los que ya usaste en el curso (MNIST o CIFAR-10) para
  centrarte en profundizar la técnica en vez de en buscar datos
- Buscar un dataset propio en `torchvision.datasets`, en portales de datos
  abiertos o construido por ti mismo (fotos propias, un CSV que elabores a
  mano)

Si eliges datos propios, comprueba antes de nada que tienes suficientes
ejemplos por categoría y que las etiquetas son fiables: un dataset pequeño o
mal etiquetado limita mucho lo que puedes conseguir, por buena que sea la
técnica.

### Paso 2: elige la técnica del curso que vas a aplicar

Elige la técnica principal según el tipo de dato de entrada, no al revés:

- Imágenes: red neuronal convolucional (CNN), como en las clases 5 y 6
- Datos tabulares (números y categorías en filas y columnas): red densa de
  clasificación o regresión, como en las clases 3 y 4
- Un volumen de datos o un modelo grande: entrena en GPU siguiendo el
  patrón de la clase 7

Combina esa técnica principal con al menos una buena práctica de la clase 8:
evalúa el sobreajuste con la pérdida y la precisión dentro y fuera de
muestra, y calcula una matriz de confusión para saber exactamente en qué se
equivoca tu modelo.

### Paso 3: estructura el entregable

Tu entregable tiene dos piezas:

- **Un notebook** con el pipeline completo: carga y exploración del
  dataset, definición del modelo con `nn.Module`, bucle de entrenamiento,
  evaluación con métricas y matriz de confusión, y guardado del modelo
  entrenado con `torch.save`
- **Una memoria breve** (una o dos páginas) que explique en lenguaje claro
  qué problema resuelves, qué dataset usaste y sus limitaciones, qué modelo
  elegiste y por qué, qué resultados obtuviste y qué harías distinto con más
  tiempo

### Paso 4: presenta y defiende tu proyecto

Prepara una presentación breve, de cinco a diez minutos, para el resto del
grupo:

1. Plantea el problema en una frase, sin jerga técnica
2. Muestra el dataset y una o dos cifras que lo describan (número de
   ejemplos, número de categorías)
3. Explica la arquitectura del modelo con un esquema simple, no con el
   código completo
4. Muestra los resultados, incluida al menos una gráfica o la matriz de
   confusión
5. Cierra con las limitaciones del proyecto y qué pasos seguirías si
   continuaras

Los criterios que se van a valorar son: que el problema esté bien planteado
y justificado, que la técnica elegida sea coherente con el tipo de dato, que
evalúes honestamente el rendimiento del modelo —incluido el sobreajuste—,
que el notebook sea reproducible de principio a fin, y que la presentación
se entienda sin necesidad de leer el código.

Abre ahora el notebook de práctica de esta clase:
`clase09_practica_casos_de_uso_reales.ipynb`. Ahí encuentras el andamiaje
inicial de tu proyecto final, listo para que empieces a escribir la primera
celda de código.
