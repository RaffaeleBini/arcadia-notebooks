[← Anterior: [Balancea tus datos y entrena tu modelo LSTM final](11_caso_cripto_modelo_final.md) · [Índice](00_indice.md)]

# 12. Casos de uso reales de Deep Learning y próximos pasos

## Por qué te importa esto

Has construido tres proyectos completos a lo largo de este curso: un clasificador de dígitos, un clasificador de imágenes de gatos y perros, y un predictor de movimientos de precio de criptomonedas. Cada uno ilustra una familia distinta de problemas que resuelves con Deep Learning en el mundo real. Este capítulo cierra el manual repasando esos tres casos desde una perspectiva práctica: qué problema real representa cada uno, y a qué otros problemas puedes aplicar la misma arquitectura mañana.

## Caso 1: clasificación de dígitos manuscritos (MNIST)

**El problema real que representa**: automatizar la lectura de texto o símbolos escritos a mano, sin que un humano tenga que revisar cada uno. Aplicado en producción, este mismo enfoque impulsa el reconocimiento óptico de caracteres (OCR) en formularios escaneados, cheques bancarios, matrículas de vehículos o códigos postales en sobres de correo.

**Por qué importa como punto de partida**: es el problema más simple donde ves el ciclo completo (cargar datos, normalizar, construir con `Sequential`, compilar, entrenar, evaluar, predecir) sin que la complejidad de los datos te distraiga de entender ese ciclo.

**A qué otros problemas se generaliza**: cualquier clasificación de imágenes pequeñas con pocas clases bien diferenciadas, como clasificar tipos de defectos en piezas industriales fotografiadas por una cámara de control de calidad.

## Caso 2: clasificación de imágenes reales (gatos vs. perros)

**El problema real que representa**: reconocer objetos, categorías o condiciones en fotografías del mundo real, no en datasets ya perfectamente empaquetados. El pipeline de preprocesamiento que construiste (carga, escala de grises, redimensionado, balanceo) es el mismo que usarías para clasificar radiografías médicas como sanas o con anomalías, fotos de productos en un catálogo de e-commerce, o imágenes de satélite para detectar deforestación.

**Por qué importa la arquitectura CNN**: sin capas convolucionales, este problema sería computacionalmente inviable con imágenes de tamaño real. Las CNN son la base de prácticamente todo sistema de visión por computador en producción hoy, desde cámaras de seguridad con detección de objetos hasta los sistemas de conducción asistida.

**A qué otros problemas se generaliza**: detección de objetos (dónde está cada objeto en la imagen, no solo qué contiene), segmentación de imágenes (qué píxeles exactos pertenecen a cada objeto), y reconocimiento facial.

## Caso 3: predicción de series temporales (criptomonedas)

**El problema real que representa**: anticipar el comportamiento futuro de una magnitud que cambia con el tiempo, a partir de su historial reciente. El material fuente de este curso menciona explícitamente que el mismo enfoque (ventanas temporales + LSTM) se aplica a problemas muy distintos al financiero: predecir si un servidor va a sobrecalentarse a partir de sus métricas históricas, o anticipar picos de tráfico en un sitio web a partir del histórico de visitas por hora.

**Por qué importa la arquitectura RNN/LSTM**: a diferencia de los dos casos anteriores, aquí el orden de los datos es la información. Una CNN o una red densa no tienen ningún mecanismo para representar "esto vino antes que aquello"; una RNN sí.

**A qué otros problemas se generaliza**: análisis de sentimiento en texto (una frase es una secuencia de palabras), traducción automática, reconocimiento de voz, y cualquier mantenimiento predictivo basado en sensores que registran datos a lo largo del tiempo.

## El hilo común entre los tres casos

Si repasas los tres proyectos, la estructura de trabajo es siempre la misma, cambia solo el contenido de cada paso:

1. Consigue datos representativos del problema real, no solo de un caso ideal
2. Preprocesa y normaliza esos datos a un formato uniforme que la red pueda consumir
3. Elige la arquitectura según la naturaleza del dato: `Dense` para datos tabulares simples, CNN para datos con estructura espacial (imágenes), RNN/LSTM para datos con estructura temporal (secuencias)
4. Compila, entrena y usa TensorBoard para vigilar overfitting
5. Evalúa con métricas honestas, no solo con accuracy
6. Prueba con datos verdaderamente nuevos antes de confiar en el modelo

## Hacia dónde seguir después de este curso

Este manual te ha dado las bases. Estos son pasos naturales para seguir avanzando:

- **Profundiza en una familia de arquitecturas**: si te atrajo la visión por computador, explora arquitecturas CNN más avanzadas (ResNet, EfficientNet) y técnicas de aumento de datos (data augmentation)
- **Explora el procesamiento del lenguaje natural (NLP)**: los mismos conceptos de RNN/LSTM que usaste con criptomonedas son la puerta de entrada a modelos de texto, aunque hoy la mayoría de sistemas de NLP en producción usan arquitecturas de tipo transformer
- **Aprende a llevar modelos a producción**: entrenar un modelo en una notebook es solo la mitad del trabajo. Herramientas como TFX (TensorFlow Extended) te ayudan a construir pipelines completos de datos, entrenamiento y despliegue
- **Practica con datasets propios**: el ejercicio más valioso que puedes hacer ahora es tomar un problema de tu propio interés o trabajo, y aplicarle el mismo proceso de seis pasos que acabas de repasar

## Resumen de puntos clave

- MNIST representa el problema de reconocimiento de caracteres y clasificación simple, base del OCR
- Gatos vs. perros representa la clasificación de imágenes reales con CNN, base de la visión por computador en producción
- La predicción de criptomonedas representa el análisis de series temporales con RNN/LSTM, generalizable a mantenimiento predictivo, NLP y reconocimiento de voz
- El proceso de trabajo (datos, preprocesamiento, arquitectura según el tipo de dato, entrenamiento vigilado, evaluación honesta, prueba con datos nuevos) es el mismo en cualquier proyecto de Deep Learning
- Seguir aprendiendo significa, sobre todo, aplicar este proceso a un problema propio

## Ponlo en práctica

Elige un problema de tu entorno (trabajo, estudios o interés personal) que podrías resolver con alguna de las tres arquitecturas de este curso. Escribe, en un párrafo, qué datos necesitarías, qué arquitectura elegirías (Dense, CNN o RNN/LSTM) y por qué, y qué métrica usarías para saber si el modelo funciona de verdad. **Compártelo** con el resto de la clase: es la mejor forma de comprobar si de verdad interiorizaste el proceso completo de este curso.

---

[← Anterior: [Balancea tus datos y entrena tu modelo LSTM final](11_caso_cripto_modelo_final.md) · [Índice](00_indice.md)]
