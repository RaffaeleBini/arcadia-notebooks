# 12. Casos de uso reales y cierre del curso

Has llegado al final del manual. A lo largo de once capítulos has pasado de la teoría a la práctica y de la práctica al código desde cero: regresión, K vecinos más próximos (KNN, del inglés K-Nearest Neighbors), máquinas de vectores de soporte (SVM, del inglés Support Vector Machines), y clustering con K-Means y Mean Shift. En este capítulo cierras el círculo: ves dónde vive cada algoritmo fuera del aula y decides hacia dónde quieres seguir.

## Por qué te importa esto

Aprender un algoritmo en abstracto es útil, pero solo cobra sentido cuando lo conectas con un problema real. El aprendizaje automático (ML, del inglés Machine Learning) no es una colección de fórmulas: es una caja de herramientas que ya toma decisiones a diario en bancos, hospitales, coches y redes sociales. Si entiendes qué algoritmo encaja con qué tipo de pregunta, dejas de ver el ML como magia y empiezas a verlo como una herramienta más de tu trabajo, tan concreta como una hoja de cálculo o una base de datos.

Este cierre no añade algoritmos nuevos. Te ayuda a mirar atrás, reconocer el patrón detrás de cada técnica y aplicarlo mentalmente a los problemas que tú ya conoces, sean del sector que sean.

## Lo que ya sabes hacer: un repaso rápido de tu caja de herramientas

Antes de ver ejemplos, repasa qué resuelve cada algoritmo:

- **Regresión**: predice un valor numérico continuo a partir de otras variables (cuánto costará algo, cuánto va a llover, cuántas unidades se venderán)
- **KNN**: clasifica un caso nuevo mirando cuáles son sus vecinos más parecidos entre los datos ya conocidos
- **SVM**: traza la frontera que mejor separa dos o más categorías, incluso cuando los datos no son separables de forma sencilla
- **K-Means**: agrupa datos sin etiquetar en un número de grupos que decides de antemano
- **Mean Shift**: agrupa datos sin etiquetar sin que tengas que fijar cuántos grupos existen; el propio algoritmo los descubre

Con la serie de precios simulada de los capítulos dos a cuatro practicaste regresión. Con el diagnóstico de cáncer de mama de los capítulos cinco a nueve viste KNN y SVM en acción sobre un problema de clasificación real. Y con los datos de supervivencia del Titanic de los capítulos diez y once aplicaste K-Means y Mean Shift para descubrir grupos de pasajeros sin usar ninguna etiqueta. Cada dataset del manual es, en realidad, una versión simplificada de un problema que ya se resuelve así fuera del aula.

## Casos de uso reales de regresión

La regresión es probablemente la técnica de ML más extendida porque responde a una pregunta que aparece en casi cualquier negocio: ¿cuánto va a pasar?

- **Previsión de precios y demanda**: empresas de todos los sectores usan regresión para estimar el precio futuro de un producto o la cantidad que van a vender, de forma parecida a como tú estimaste una serie de precios simulada en los capítulos dos a cuatro. Esta misma lógica sirve para predecir el precio de acciones en mercados financieros
- **Previsión meteorológica**: los servicios de meteorología combinan variables como temperatura, presión y humedad para predecir cuánto lloverá o qué temperatura hará; es uno de los ejemplos clásicos de regresión aplicada a series temporales
- **Estimación de costes**: cualquier organización que necesite calcular el coste esperado de un proyecto, un tratamiento o una campaña a partir de variables conocidas puede apoyarse en un modelo de regresión entrenado con datos históricos

## Casos de uso reales de clasificación (KNN y SVM)

KNN y SVM responden a una pregunta distinta: no cuánto, sino qué categoría. Ambos algoritmos comparten aplicaciones porque los dos separan datos en grupos ya conocidos de antemano.

- **Diagnóstico médico**: el propio manual usa el diagnóstico de cáncer de mama como ejemplo central en los capítulos cinco a nueve. Es una aplicación real y muy extendida: clasificar un tumor como benigno o maligno a partir de mediciones celulares es exactamente el tipo de problema para el que se diseñaron KNN y SVM
- **Detección de correo spam**: clasificar un correo entrante como spam o no spam a partir de las palabras y patrones que contiene es una de las aplicaciones más conocidas de la clasificación automática
- **Coches autónomos**: distinguir si lo que hay delante del vehículo es un obstáculo sin importancia o un peligro real, como un bulto en la carretera frente a una persona con una manta, exige una clasificación con una precisión muy superior a la media; los sistemas de conducción autónoma dependen de decisiones de este tipo
- **Reconocimiento de escritura a mano**: las SVM se hicieron conocidas en los años noventa precisamente por superar a las redes neuronales de la época en el reconocimiento de caracteres manuscritos, un hito histórico que impulsó su adopción en la industria

## Casos de uso reales de clustering (K-Means y Mean Shift)

El clustering resuelve un problema distinto a los dos anteriores: no tienes etiquetas previas, solo datos, y quieres descubrir qué estructura esconden.

- **Segmentación de clientes**: las empresas agrupan a su clientela según hábitos de compra o comportamiento para diseñar campañas o recomendaciones distintas para cada grupo; es una extensión directa del ejercicio de recomendación de productos que ya mencionamos como aplicación típica del ML
- **Agrupación de pacientes o de explotaciones agrarias**: agrupar historiales clínicos con características parecidas, o explotaciones agrícolas con condiciones similares de suelo y producción, ayuda a diseñar protocolos o recomendaciones adaptadas a cada grupo en lugar de aplicar una única regla para todos los casos; el mismo principio que aplicaste al agrupar pasajeros del Titanic en los capítulos diez y once
- **Detección de grupos en redes sociales**: identificar comunidades de usuarios con intereses o interacciones parecidas dentro de una red social es un uso habitual del clustering, muy similar en su lógica a Mean Shift cuando no se conoce de antemano cuántas comunidades existen

## Hacia dónde seguir después de este curso

Este manual cubre cinco algoritmos clásicos, pero el aprendizaje automático es un campo mucho más amplio. Existen otras ramas, como las redes neuronales y el aprendizaje profundo (deep learning), que resuelven problemas todavía más complejos: reconocimiento de imágenes a gran escala, generación de texto o traducción automática, entre otros. Estas técnicas quedan fuera del alcance de este curso, pero son un paso natural una vez dominas los fundamentos que acabas de aprender. La potencia de cálculo actual, accesible desde servicios en la nube, hace que este siguiente paso esté hoy al alcance de cualquier persona con curiosidad y constancia.

## Resumen: lo que te llevas de este capítulo

- La regresión predice valores numéricos: precios, demanda, previsión meteorológica o estimación de costes
- KNN y SVM clasifican casos en categorías conocidas: diagnóstico médico, spam, coches autónomos o reconocimiento de escritura
- K-Means y Mean Shift descubren grupos en datos sin etiquetar: clientes, pacientes, explotaciones o comunidades
- Cada dataset que usaste en el manual (precios simulados, cáncer de mama, Titanic) es una versión de aprendizaje de un problema que ya se resuelve así en el mundo real
- Elegir el algoritmo correcto depende de la pregunta que quieres responder, no de la moda del momento
- Las redes neuronales y el aprendizaje profundo son el siguiente paso lógico, pero no el único: primero domina bien estos cinco fundamentos

## Ponlo en práctica

Piensa en un problema real de tu entorno, sea laboral, académico o personal. **Elige** uno de los cinco algoritmos de este manual y **decide** por qué encaja con ese problema. Por ejemplo, si trabajas en el sector agroalimentario, podrías usar regresión para predecir la cosecha de una campaña a partir de datos climáticos; si trabajas en salud, SVM o KNN para clasificar historiales clínicos; o si gestionas explotaciones forestales, K-Means para segmentar parcelas según sus características. **Escribe** en dos o tres frases qué datos necesitarías y qué algoritmo usarías, y **guarda** esa idea: es tu primer proyecto real de aprendizaje automático.

**Siglas usadas en este capítulo:** ML (Machine Learning, aprendizaje automático), KNN (K-Nearest Neighbors, K vecinos más próximos), SVM (Support Vector Machines, máquinas de vectores de soporte)
