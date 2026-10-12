# 13. Aplicaciones reales y proyecto final

## Por qué importa este tema

Has recorrido un camino largo: desde la primera neurona hasta pipelines completos de producción con TensorFlow Extended (TFX). Pero todo ese aprendizaje puede quedarse en la teoría si no lo conectas con problemas reales. Este capítulo cierra el curso mostrándote proyectos concretos —muchos desarrollados por estudiantes, pequeños equipos o personas autodidactas, no solo por grandes departamentos de investigación— que usan exactamente las técnicas que ya conoces: redes convolucionales (CNN, Convolutional Neural Network) para clasificar imágenes, redes recurrentes (RNN, Recurrent Neural Network) y memoria a largo-corto plazo (LSTM, Long Short-Term Memory) para series temporales, y TensorFlow Lite para llevar un modelo a un teléfono móvil sin conexión.

Ver estos casos tiene un objetivo muy concreto: que dejes de pensar "¿qué técnica me falta por aprender?" y empieces a pensar "¿qué problema quiero resolver con lo que ya sé?". Esa es exactamente la pregunta que vas a responder en tu proyecto final, el cierre práctico de todo el curso.

## Casos de uso reales de TensorFlow

Los ocho casos siguientes están tomados del material de la certificación EITC/AI/TFF (European IT Certificate — Certificado Europeo de Tecnologías de la Información — en Inteligencia Artificial, especialidad TensorFlow Fundamentals). Cada uno resuelve un problema distinto, en un sector distinto, con una combinación de técnicas que ya has visto en clases anteriores.

### Air Cognizer: predecir la calidad del aire con el móvil

En Delhi, la contaminación del aire es un problema de salud pública grave. Un grupo de estudiantes de ingeniería desarrolló Air Cognizer, una aplicación que, a partir de una foto del cielo tomada con el móvil, calcula el Índice de Calidad del Aire (AQI, Air Quality Index) de la zona. La aplicación combina tres modelos: una CNN que clasifica si la imagen contiene cielo, una LSTM que analiza patrones meteorológicos históricos como serie temporal, y un algoritmo de bosque aleatorio (RF, Random Forest) que hace la predicción final de regresión sobre el AQI a partir de todas las variables.

Lo interesante desde el punto de vista técnico es el despliegue: los tres modelos se convirtieron a TensorFlow Lite y quedaron reducidos a un binario de solo 10-20 KB, lo bastante ligero para funcionar en cualquier móvil sin conexión a internet ni consumo relevante de batería. El impacto es directo: cualquier persona en Delhi puede conocer la calidad del aire de su entorno inmediato y tomar decisiones (por ejemplo, evitar hacer ejercicio al aire libre) sin depender de estaciones de medición oficiales, que son escasas.

### Tambua: escuchar los pulmones para detectar enfermedades respiratorias

Las enfermedades respiratorias —neumonía, asma, tuberculosis pulmonar— causan más de 2,5 millones de muertes al año, muchas por diagnóstico tardío o erróneo en zonas con pocos recursos médicos, como el África subsahariana. La aplicación Tambua ataca este problema con un enfoque poco habitual: convierte el sonido de los pulmones, capturado con un estetoscopio digital, en un espectrograma (una representación visual de las frecuencias del sonido a lo largo del tiempo) y entrena una CNN para reconocer patrones asociados a cada enfermedad a partir de esa imagen.

El equipo entrenó el modelo con datos de 621 pacientes reales, en colaboración con clínicas y especialistas. Igual que en el caso anterior, el modelo se despliega con TensorFlow Lite para funcionar sin conexión: hoy 216 centros de salud, muchos de ellos clínicas rurales, usan la aplicación como apoyo al diagnóstico. El impacto no es sustituir al personal médico, sino darle una herramienta rápida y barata que reduce diagnósticos erróneos y evita pruebas innecesarias, como radiografías.

### Predecir fenómenos meteorológicos extremos con supercomputación

Cada día se generan cerca de 100 terabytes de datos climáticos procedentes de satélites, sensores y modelos numéricos. Investigadores del NERSC (National Energy Research Scientific Computing Center, en Estados Unidos) usan TensorFlow para entrenar modelos de aprendizaje profundo que detectan huracanes, tormentas severas y otros fenómenos extremos a partir de imágenes de satélite, combinando CNN para el análisis espacial de esas imágenes con RNN para capturar la evolución temporal de los patrones meteorológicos.

La particularidad de este caso es la escala: entrenar estos modelos requiere computación de altas prestaciones (HPC, High-Performance Computing) del nivel del supercomputador Summit, capaz de miles de billones de operaciones por segundo, y el equipo comprobó que TensorFlow escala bien incluso repartiendo el entrenamiento entre miles de nodos de cálculo. El impacto son sistemas de alerta temprana más precisos, que dan a las autoridades más tiempo para preparar la respuesta ante una catástrofe.

### Ayudar a paleógrafos a transcribir manuscritos medievales

Transcribir manuscritos medievales a mano es un trabajo lentísimo y muy especializado: la letra cambia de un escriba a otro, la tinta está desvaída y el idioma es arcaico. Un equipo de investigación entrenó una CNN de clasificación multiclase para reconocer caracteres manuscritos individuales, alcanzando un 95% de precisión de media por carácter. El reto no fue solo técnico sino de datos: como no existen bancos de imágenes de manuscritos medievales como los hay de gatos o perros, el equipo construyó una aplicación web propia para recoger imágenes mediante colaboración ciudadana (crowdsourcing), con estudiantes de instituto contribuyendo a etiquetar el material.

El impacto es directo sobre la investigación histórica: lo que antes exigía equipos enteros de paleógrafos trabajando durante años, ahora se acelera enormemente, y un volumen de conocimiento histórico hasta ahora prácticamente inaccesible puede consultarse en mucho menos tiempo. Es un buen ejemplo de que el aprendizaje automático (ML) no solo sirve para negocio o salud: también acelera las humanidades.

### Airbnb: poner orden en cientos de millones de fotos de alojamientos

No todos los casos de uso de TensorFlow nacen de una emergencia social: Airbnb tenía más de cinco millones de alojamientos en 81.000 ciudades y cientos de millones de fotos, muchas mal etiquetadas o repetidas (varias fotos del mismo salón y ninguna del baño). Airbnb entrenó una CNN basada en la arquitectura ResNet50 para clasificar automáticamente cada foto por tipo de estancia (dormitorio, cocina, salón, piscina...), usando su propia plataforma de ML (Bighead) para gestionar el entrenamiento distribuido en GPU y el despliegue con TensorFlow Serving.

Procesar más de quinientos millones de imágenes a mano habría tardado meses; con TensorFlow, el equipo tuvo un modelo funcional en pocos días. El impacto es puramente de producto: los huéspedes ven primero las fotos más relevantes de cada alojamiento (salón, dormitorio) en lugar de fotos poco atractivas del garaje, lo que mejora directamente la experiencia de búsqueda y reserva. Este caso demuestra que la clasificación de imágenes con CNN —la misma técnica de la clase 2 de este curso— tiene aplicación en cualquier negocio con grandes volúmenes de contenido visual.

### Detectar el gusano cogollero en los cultivos de Uganda

El gusano cogollero (fall armyworm) es una plaga que puede arruinar una cosecha entera y, sin control, agravar el hambre en regiones agrícolas de África. Nazirini Siraji, una desarrolladora de software ugandesa, aprendió TensorFlow en un Google Study Jam y, junto con su equipo, construyó una aplicación Android que usa una CNN para identificar la infestación a partir de fotos de las plantas tomadas con el móvil, sugiriendo además el tratamiento adecuado según el ciclo de vida de la plaga.

El impacto llega directamente al agricultor: detecta el problema antes de que sea demasiado tarde, sin depender de la disponibilidad de un experto agrónomo, y reduce el uso excesivo de pesticidas al indicar el tratamiento correcto en el momento correcto. Es también un buen recordatorio de que no hace falta un equipo grande ni recursos de una gran empresa para construir con TensorFlow algo con impacto real: basta con identificar bien el problema y aplicar la técnica adecuada.

### Localizar baches en las carreteras de Los Ángeles

Inspeccionar manualmente toda la red viaria de una ciudad tan extensa como Los Ángeles para localizar baches es lento y caro. Dos estudiantes universitarios, Alejandra Vasquez y Ericson Hernandez, equiparon un coche con una cámara, condujeron por distintas calles y autopistas de la ciudad para recopilar su propio conjunto de datos, y entrenaron con TensorFlow una CNN capaz de identificar baches y grietas en el firme con alta precisión a partir de esas imágenes.

El impacto es operativo: las cuadrillas de mantenimiento pueden dedicar su tiempo a repararlos en vez de a buscarlos, y la ciudad deja de depender únicamente de los avisos ciudadanos para saber dónde intervenir. Es un ejemplo claro de cómo, con una cámara y ganas de recopilar tus propios datos, puedes aplicar la clasificación de imágenes a un problema de movilidad urbana que nadie te ha "preparado" en un dataset ya existente.

### Ayudar a salvar a las abejas del mundo

Las abejas polinizan buena parte de los alimentos que consumimos, pero su población lleva años en declive por razones que no se entienden del todo. Un proyecto de conservación instaló cámaras en la entrada de las colmenas para contar cuántas abejas salen y cuántas vuelven, entrenando modelos de visión por computador con TensorFlow para analizar ese vídeo de forma automática y detectar discrepancias que apunten a un problema en el ecosistema cercano. El mismo enfoque se extiende al análisis de audio: las abejas se comunican mediante zumbidos que codifican información sobre dónde hay comida, y el aprendizaje automático puede ayudar a interpretar esos patrones sonoros.

Los datos generados se comparten con personas expertas en biodiversidad, que los usan para decidir, por ejemplo, cuándo cortar el césped de una zona o dónde plantar árboles y flores que favorezcan a los polinizadores. El impacto no es que la tecnología "salve" a las abejas por sí sola —la propia documentación del proyecto insiste en que la intervención humana sigue siendo imprescindible—, sino que le da a quienes toman decisiones sobre el territorio información que antes no tenían.

### Y hay más casos donde elegir

El material original de la certificación incluye 16 casos de uso completos, no solo estos ocho. Si quieres seguir explorando antes de definir tu proyecto final, en el mismo material encontrarás, entre otros, cómo TensorFlow ayuda al personal de Médicos Sin Fronteras a prescribir antibióticos a partir de fotos de placas de laboratorio, cómo se predicen inundaciones y grandes incendios forestales, cómo se rastrean asteroides, y cómo una aplicación de baile usa visión por computador para corregir la postura de quien la usa. Ninguna técnica de esa lista se sale de lo que ya conoces: siempre es una combinación de CNN, RNN/LSTM, modelos tabulares o despliegue con TensorFlow Lite, aplicada con criterio a un problema concreto.

## Diseña tu proyecto final: guía paso a paso

Tu proyecto final no necesita resolver un problema tan ambicioso como los anteriores. Necesita demostrar que sabes elegir una técnica adecuada, aplicarla con rigor y comunicar el resultado. Sigue estos pasos.

### Paso 1: elige un problema y un dataset

Empieza por el problema, no por la técnica. Pregúntate: ¿qué quiero predecir o clasificar, y por qué me importa? Puede ser un problema serio (salud, medioambiente, tu propio sector profesional) o simplemente algo que te resulte curioso. Tienes dos caminos para el dataset:

- Reutilizar uno de los que ya usaste en el curso (Fashion MNIST, reseñas de películas del IMDB, el CSV de pasajeros del Titanic, el dataset de pingüinos de las clases de TFX) para centrarte en profundizar la técnica en vez de en buscar datos
- Buscar un dataset propio en fuentes abiertas como Kaggle, TensorFlow Datasets, portales de datos abiertos de administraciones públicas o datos que generes tú mismo (fotos propias, un CSV que construyas a mano)

Si eliges datos propios, comprueba antes de nada que tienes suficientes ejemplos por categoría y que las etiquetas son fiables: un dataset pequeño o mal etiquetado limita mucho lo que puedes conseguir, por buena que sea la técnica.

### Paso 2: elige la técnica del curso que vas a aplicar

Elige una técnica principal en función del tipo de dato de entrada, no al revés:

- Datos tabulares (números y categorías en filas y columnas): red densa, clasificación o regresión, como en la clase 3
- Imágenes: red convolucional (CNN), como en la clase 2
- Texto: tokenización y secuenciación seguida de una RNN o LSTM, como en las clases 6 y 7
- Series temporales: RNN o LSTM, igual que en la predicción meteorológica que has visto en este capítulo

Combina esa técnica principal con al menos una buena práctica de la clase 4 (detectar sobreajuste con las curvas de entrenamiento, aplicar dropout o parada temprana) para demostrar que no solo entrenas un modelo, sino que evalúas si es fiable.

### Paso 3: estructura el entregable

Tu entregable tiene dos piezas:

- **Un notebook** con el pipeline completo: carga y exploración del dataset, preprocesado, definición del modelo, entrenamiento, evaluación con métricas y curvas de pérdida/precisión, y guardado del modelo entrenado. Si tiene sentido para tu caso, añade una celda final que convierta el modelo a TensorFlow Lite, como en la clase 8
- **Una memoria breve** (una o dos páginas) que explique en lenguaje claro: qué problema resuelves, qué dataset usaste y sus limitaciones, qué modelo elegiste y por qué, qué resultados obtuviste y qué harías distinto con más tiempo

### Paso 4: presenta y defiende tu proyecto

Prepara una presentación de cinco a diez minutos para el resto del grupo:

1. Plantea el problema en una frase, sin jerga técnica
2. Muestra el dataset y una o dos cifras que lo describan (tamaño, número de categorías)
3. Explica la arquitectura del modelo con un esquema simple, no con el código completo
4. Muestra los resultados, incluida al menos una gráfica (curva de entrenamiento, matriz de confusión, ejemplos de predicción)
5. Cierra con las limitaciones del proyecto y qué pasos seguirías si continuaras

Un plazo orientativo razonable para todo el proyecto es de una a dos semanas de trabajo autónomo tras esta clase, con una sesión final del curso dedicada a las presentaciones. Los criterios que se van a valorar son: que el problema esté bien planteado y justificado, que la técnica elegida sea coherente con el tipo de dato, que evalúes honestamente el rendimiento del modelo (incluido el sobreajuste), que el notebook sea reproducible de principio a fin, y que la presentación se entienda sin necesidad de leer el código.

## Resumen de puntos clave

- Los casos reales de TensorFlow que has visto —calidad del aire, diagnóstico respiratorio, meteorología extrema, paleografía, clasificación de fotos de alojamientos, plagas agrícolas, baches urbanos y conservación de abejas— usan las mismas técnicas del curso: CNN, RNN/LSTM, modelos tabulares y TensorFlow Lite para despliegue
- Ninguno de estos proyectos partió de recursos ilimitados: muchos los desarrollaron estudiantes o equipos pequeños con un problema claro y datos propios recopilados con esfuerzo
- El material original de la certificación incluye 16 casos de uso completos; los ocho de este capítulo son una muestra representativa por sectores
- Tu proyecto final se construye en cuatro pasos: elegir problema y dataset, elegir la técnica adecuada al tipo de dato, estructurar notebook más memoria, y presentar defendiendo tus decisiones
- Los criterios de evaluación premian el rigor (evaluar el sobreajuste, un notebook reproducible) tanto como el resultado final

## Ponlo en práctica

Antes de la próxima sesión, escribe en una sola frase el problema que quieres resolver en tu proyecto final y, en otra frase, qué dataset vas a usar (uno del curso o uno nuevo) y qué técnica de las vistas en clase encaja mejor. Compártelas con el grupo: es la primera versión de la memoria que vas a entregar, y recibir una primera reacción ahora te ahorra tiempo más adelante. Después, abre el notebook de práctica de esta clase y arranca el andamiaje del proyecto: la mejor forma de perder el miedo al proyecto final es tener ya una celda de código ejecutándose.
