# Capítulo 13: Innovación responsable e inteligencia artificial

**Acrónimos usados en este capítulo:** inteligencia artificial (IA) · redes neuronales convolucionales (CNN) · redes neuronales recurrentes (RNN) · memoria a corto y largo plazo (LSTM) · IA explicable (XAI) · explicaciones interpretables agnósticas del modelo, de alcance local (LIME) · método del signo del gradiente rápido (FGSM) · ascenso de gradiente proyectado (PGA) · satisfacibilidad módulo teorías (SMT) · lógica temporal lineal (LTL) · test de asociación de incrustaciones de palabras (WEAT) · Reglamento General de Protección de Datos (RGPD) · Unión Europea (UE) · Organización para la Cooperación y el Desarrollo Económicos (OCDE) · Organización de las Naciones Unidas (ONU) · Instituto de Ingenieros Eléctricos y Electrónicos (IEEE) · Perfilado de Gestión de Delincuentes Correccionales para Sanciones Alternativas (COMPAS) · tasa de verdaderos positivos (TPR)

## Por qué importa este tema

Ya sabes entrenar redes neuronales convolucionales (CNN) que clasifican imágenes casi con la precisión de una persona, redes recurrentes (RNN) que generan lenguaje coherente y modelos generativos que crean imágenes realistas. Ese poder técnico ya es real: AlphaFold ganó la competición CASP prediciendo estructuras de proteínas desconocidas, los modelos de lenguaje como GPT-2 y GPT-3 escriben textos gramaticalmente correctos y contextualmente coherentes, y los sistemas de aprendizaje por refuerzo superan a personas expertas en juegos tan complejos como el Go. El problema no es si la IA funciona: es qué ocurre cuando funciona sobre datos y decisiones que afectan a personas reales.

Los ejemplos sobran, y todos comparten un mismo patrón: un modelo técnicamente competente que causa un daño que nadie diseñó a propósito. El algoritmo COMPAS, usado en tribunales estadounidenses para estimar el riesgo de reincidencia, marcaba a personas afrodescendientes como de alto riesgo con una tasa de falsos positivos muy superior a la de personas blancas. En 2015, el sistema de reconocimiento de imágenes de Google Photos etiquetó fotografías de personas negras como "gorilas". Herramientas de búsqueda de empleo mostraron anuncios de puestos bien remunerados a hombres con mucha más frecuencia que a mujeres. El escándalo de Cambridge Analytica reveló que datos personales de millones de usuarios de Facebook se habían recolectado sin consentimiento real para influir en campañas políticas. Y a nivel puramente técnico, el estudio "Intriguing Properties of Neural Networks" mostró que basta una perturbación imperceptible para el ojo humano para que un clasificador confunda un panda con un gibón: trasladado a un sistema de conducción autónoma, ese mismo tipo de fallo puede confundir una señal de stop con una de ceda el paso.

Ninguno de estos sistemas se diseñó con la intención de discriminar o de fallar. El sesgo entró por la puerta trasera: datos de entrenamiento poco representativos, etiquetas que heredan prejuicios históricos, métricas de éxito que no incluyen la equidad como objetivo. Ese es exactamente el motivo por el que este capítulo cierra el bloque técnico del manual: quien diseña y entrena modelos ocupa una posición de poder, porque sus decisiones sobre datos, arquitecturas y métricas determinan qué oportunidades se abren y cuáles se cierran para otras personas. Con ese poder llega una responsabilidad que no es opcional ni externa al trabajo técnico, sino parte de él.

En DIH DATAlife lo vemos de cerca cuando acompañamos a equipos de sectores como Salud-Cuidados o Agro-Mar Alimentación en proyectos de digitalización con componentes de IA: un modelo de apoyo diagnóstico o de priorización de ayudas que funciona peor para un subgrupo de la población no es un detalle técnico menor, es un fallo de diseño con consecuencias directas sobre las personas. Este capítulo te da el vocabulario y las herramientas conceptuales para detectar ese tipo de fallo antes de que llegue a producción.

## Explicación conceptual con analogías

### La receta y el libro de normas

Piensa en entrenar un modelo con datos, sin más, como cocinar siguiendo solo lo que te enseñaron a comer de pequeño, sin ningún libro de normas de nutrición. Si de pequeño solo comiste un tipo de plato, cocinarás bien ese plato y mal casi todos los demás, sin darte ni cuenta de la limitación: para ti, tu experiencia parece la totalidad de la cocina posible. Eso es lo que el material fuente llama entrenamiento "vainilla", ordinario: el modelo hereda lo que hay en los datos, sesgado y limitado, sin ningún mecanismo que le obligue a comportarse de otra manera cuando los datos no bastan.

El aprendizaje dirigido por especificaciones añade ese libro de normas. No sustituye a la experiencia (los datos), pero impone reglas explícitas que el resultado final debe cumplir aunque los datos, por sí solos, nunca las hubieran enseñado: "sé robusto ante cambios imperceptibles en la entrada", "no discrimines por atributos irrelevantes para la tarea", "cuando no reconozcas la situación, dilo con menos confianza en vez de más". Es la diferencia entre un modelo que aprende exactamente lo que hay en los datos, con sus sesgos incluidos, y uno al que además se le exige cumplir ciertas propiedades por diseño.

### La caja negra y la consulta médica

La falta de transparencia de un modelo profundo se entiende mejor si la comparas con dos formas distintas de recibir un diagnóstico médico. En la primera, alguien te dice solo "tienes esta enfermedad", sin ninguna explicación: tienes que confiar a ciegas. En la segunda, la misma persona te explica qué síntomas, qué pruebas y qué razonamiento la llevan a esa conclusión: puedes evaluar si el razonamiento tiene sentido, pedir una prueba adicional o buscar una segunda opinión. Un modelo de aprendizaje profundo, con sus millones de parámetros interconectados, funciona por defecto como la primera consulta: produce una salida sin exponer el porqué. La IA explicable (XAI) es el conjunto de técnicas, como los mapas de saliencia o las explicaciones interpretables agnósticas del modelo de alcance local (LIME), que intenta convertir esa consulta silenciosa en la segunda, la que sí puedes cuestionar.

### El detector de mentiras que solo detecta mentiras evidentes

Una evaluación de robustez adversarial débil se parece a un detector de mentiras que solo comprueba las mentiras más obvias y torpes. Si únicamente pruebas tu modelo con ataques débiles, puedes concluir, equivocadamente, que es robusto: el problema no está en el modelo, está en que no lo has puesto a prueba lo bastante fuerte. La fuente describe estudios de 2018 en los que, al aplicar evaluaciones adversariales más rigurosas, la mayoría de las defensas publicadas hasta entonces se derrumbaron, con la excepción del entrenamiento adversarial. La lección de fondo, válida mucho más allá de la robustez adversarial: una evaluación floja da una falsa sensación de seguridad, y esa falsa sensación es en sí misma un riesgo.

## Desarrollo teórico

### Especificaciones deseables y aprendizaje dirigido por especificaciones

La pregunta que organiza la primera mitad de este capítulo es directa: ¿cómo aseguras que un algoritmo de aprendizaje automático es seguro para su despliegue? La respuesta pasa por exigirle **especificaciones deseables**, del mismo modo que cualquier producto de ingeniería pasa un control de calidad antes de salir al mercado. La fuente propone varios ejemplos concretos: un clasificador de imágenes debe ser robusto ante perturbaciones adversariales, un predictor de sistemas dinámicos debe respetar las leyes de la física, un clasificador de dígitos debe ser invariante a atributos irrelevantes como el color del trazo, un modelo que maneja datos sensibles debe cumplir privacidad diferencial, y un modelo enfrentado a datos fuera de la distribución de entrenamiento debe mostrar más incertidumbre, no más confianza.

El problema de fondo es que, cuando entrenas con datos limitados, el modelo puede aprender correlaciones espurias que mejoran la métrica de entrenamiento sin tener nada que ver con la tarea real. Si los datos están sesgados y son limitados, el modelo hereda ese sesgo y esa falta de robustez a menos que se lo impidas explícitamente. El aprendizaje dirigido por especificaciones consiste, precisamente, en incorporar esas reglas allí donde los datos no bastan.

### Robustez adversarial: formalización y entrenamiento

La especificación mejor estudiada en la fuente es la robustez adversarial. Formalmente, para una red neuronal f con parámetros θ, una entrada x y una etiqueta verdadera en formato one-hot, exiges que el índice de la probabilidad máxima en la salida coincida con el de la etiqueta verdadera para toda perturbación δ dentro de un conjunto S de cambios imperceptibles, típicamente definido como una bola de radio pequeño en la norma L-infinito:

argmax f(x + δ; θ) = argmax(etiqueta) para todo δ ∈ S

El **entrenamiento adversarial** enseña al modelo a cumplir esa especificación añadiendo, a la optimización estándar, un paso de aumento de datos: en lugar de entrenar solo con la imagen original, se entrena también con la peor perturbación posible dentro de S. Esto convierte el problema en una optimización min-max: primero se maximiza la pérdida respecto a la perturbación δ dentro de S (paso interno), y después se minimizan los parámetros respecto a esa pérdida ya maximizada (paso externo). Encontrar el máximo exacto es, en general, un problema NP-difícil, así que en la práctica se aproxima con **ascenso de gradiente proyectado** (PGA): tomas un paso de gradiente que maximiza la pérdida y, si la perturbación resultante se sale del conjunto S, la proyectas de vuelta al punto más cercano dentro de S. El **método del signo del gradiente rápido** (FGSM) es la versión más simple de esta idea para la norma L-infinito: sustituye el gradiente por su signo, y puede combinarse con optimizadores como el momento o Adam, que ya conoces de capítulos anteriores.

### Evaluar bien la robustez: fuerza de la evaluación y ofuscación de gradientes

Una evaluación adversarial es tan fiable como el esfuerzo que pones en encontrar el peor caso. La fuente identifica varias palancas que fortalecen una evaluación: aumentar el número de pasos de PGA, usar varias inicializaciones aleatorias de la perturbación (para detectar comportamientos anómalos del optimizador), probar distintos optimizadores, y complementar los ataques de caja blanca (con acceso completo a los pesos) con ataques de caja negra, que solo consultan entradas y salidas. Una evaluación débil produce una **ofuscación de gradientes**: el modelo aprende, sin que nadie se lo pida explícitamente, a construir una superficie de pérdida tan irregular que el optimizador de gradiente no encuentra el verdadero peor caso. El resultado es una falsa sensación de robustez, coherente con lo señalado en el apartado anterior: cuando se aplicaron evaluaciones más exigentes, la precisión adversarial de muchas defensas publicadas cayó a cero, y solo el entrenamiento adversarial resistió.

### Algoritmos de verificación

El entrenamiento adversarial reduce el riesgo, pero no lo elimina ni lo demuestra matemáticamente: sigue siendo una evaluación empírica sobre un conjunto finito de ataques. Los **algoritmos de verificación** buscan garantías demostrables de que ninguna perturbación dentro de S puede violar la especificación. Se dividen en dos familias. Los **algoritmos completos**, como los basados en programación entera mixta, ofrecen una prueba exhaustiva: confirman que la especificación se cumple o presentan un contraejemplo, pero su coste computacional los hace difíciles de escalar a redes profundas grandes. Los **algoritmos incompletos** son más escalables: cuando encuentran una prueba, esta es una garantía real, pero no siempre la encuentran aunque la especificación se cumpla, así que ofrecen una cota inferior sobre el grado de cumplimiento.

La técnica central de los métodos incompletos es la **propagación de cotas**: dado un conjunto de entradas posibles (por ejemplo, todas las perturbaciones dentro de S), se calculan cotas inferior y superior tras cada capa lineal y cada activación, obteniendo un conjunto de salida sobre-aproximado. Si ese conjunto sobre-aproximado queda enteramente a un lado de la frontera de decisión, puedes concluir que el conjunto real también cumple la especificación. Si el conjunto sobre-aproximado cruza la frontera, no puedes concluir nada sobre el conjunto real: la sobre-aproximación puede ser demasiado laxa. Cuanto más ajustada es la propagación de cotas, menor es la brecha entre lo que garantiza la verificación y lo que muestra la evaluación adversarial empírica, y más información útil obtienes sobre el verdadero grado de cumplimiento de la especificación.

Estas técnicas no se limitan a la robustez adversarial: la fuente menciona también la **consistencia semántica** (en un coche autónomo, confundir un gato con un perro es un error menor comparado con confundir un gato con un coche) y la **consistencia física** (un predictor de sistemas dinámicos que respete la conservación de la energía) como otras especificaciones verificables con las mismas herramientas.

### Ética de los datos de entrenamiento

La segunda mitad del capítulo se aleja de la verificación matemática y entra en el terreno ético. Todo empieza en los datos, que no son un recurso neutral sino que arrastran propiedades éticas propias. Tres preguntas resumen el punto de partida: ¿se recogieron los datos con el consentimiento de las personas representadas?, algo que casos como Cambridge Analytica demuestran que no puede darse por hecho; ¿la representación de los datos es suficientemente diversa, o favorece a unos grupos en detrimento de otros?; y ¿la etiquetación y curación de los datos contiene asociaciones prejuiciosas heredadas de sesgos humanos históricos?, como ocurrió con las primeras versiones de ImageNet, que incluían una categoría de "Personas" con etiquetas peyorativas.

Estos problemas en el flujo de datos se traducen en **sesgo algorítmico** en el mundo real: el sistema COMPAS en justicia penal, la desigualdad en anuncios de empleo bien remunerado, el peor rendimiento del reconocimiento de imágenes para minorías, y herramientas médicas que funcionan peor para personas con identidades interseccionales, con el riesgo de un acceso desigual a servicios y medicación que pueden salvar vidas.

### Poder, responsabilidad e innovación responsable

Quien diseña y desarrolla estas tecnologías ocupa una posición de poder, entendido como la capacidad de influir sobre el estado de las cosas y sobre la vida de otras personas. La investigación científica, incluida la de aprendizaje automático, no es neutral respecto a valores: es una práctica social gobernada por normas compartidas, tanto epistémicas (la necesidad de replicar hallazgos) como morales (la aceptabilidad de investigar con personas sin su consentimiento), y esas normas tienen efectos sociales profundos cuando lo que está en juego es importante y el impacto es incierto.

La **innovación responsable** se define en la fuente como un proceso transparente e iterativo mediante el cual los actores sociales y las personas que desarrollan tecnología se vuelven mutuamente receptivos a las necesidades de las otras partes, para asegurar el valor ético y social de la actividad científica. Esta responsabilidad tiene una parte intrínseca (diseñar sistemas sensibles a consideraciones éticas y sociales, limitando el riesgo de daño desde el principio) y una parte extrínseca (asegurar que la tecnología se despliega y se usa con sensatez, porque una tecnología robusta y segura todavía puede ser utilizada de forma indebida por terceros).

Al menos 84 códigos éticos distintos, propuestos por organismos como la Unión Europea (UE), la Organización para la Cooperación y el Desarrollo Económicos (OCDE), la Academia de Beijing de Inteligencia Artificial y el Future of Life Institute, coinciden en un núcleo de valores compartidos: equidad, privacidad, transparencia y no maleficencia (el principio de "no causar daño"). La Organización de las Naciones Unidas (ONU), en su Declaración de Derechos Humanos, añade el derecho colectivo de la humanidad a beneficiarse del avance científico. El problema, señala la fuente, no es la falta de buenas intenciones ni de valores compartidos: es la brecha entre esos ideales morales abstractos y los procesos concretos para aplicarlos.

### El proceso de cinco pasos y las dos pruebas finales

Para cerrar esa brecha, la fuente propone un proceso de cinco pasos para evaluar si un proyecto de investigación o desarrollo está alineado ética y socialmente:

1. **¿Tiene la tecnología usos socialmente beneficiosos?** La mayoría de las tecnologías aportan algún beneficio; entre ellos, mejorar la salud y el bienestar, ampliar la autonomía y la libertad, producir resultados más justos, fortalecer instituciones públicas como la sanidad o la educación, y abordar retos globales como el cambio climático
2. **¿Podría la tecnología causar daño, directo o indirecto?** Casi toda tecnología conlleva algún riesgo: minar la salud o la dignidad humana, restringir la libertad o la autonomía, generar tratos o resultados injustos, dañar instituciones públicas o infringir derechos humanos
3. **¿Es posible mitigar esos riesgos, o eliminarlos por completo?** Aquí entran medidas como controlar la difusión de la tecnología o del conocimiento asociado, adoptar contramedidas técnicas, ayudar al público a entender la tecnología nueva, y buscar soluciones de política pública y marcos legales
4. **Con esas medidas en marcha, ¿la acción propuesta viola alguna línea roja o restricción moral?** Estas restricciones, próximas a la ética deontológica, delimitan lo que no debe hacerse bajo ninguna circunstancia: violar protocolos de consentimiento, desarrollar armas autónomas letales, habilitar vigilancia que erosione la confianza pública, o infringir el derecho internacional y los derechos humanos
5. **Con esas medidas en marcha, ¿los beneficios superan a los riesgos?** Solo si no se cruza ninguna línea roja tiene sentido pasar a esta última pregunta, evaluando además si existen alternativas más valiosas

Incluso tras concluir que un proyecto es viable, la fuente recomienda dos pruebas finales, pensadas para contrarrestar la **cognición motivada**, es decir, la tendencia inconsciente a aceptar los argumentos que convienen a los propios intereses. La primera prueba consiste en identificar a todas las personas que podrían verse afectadas, entender quiénes son, y preguntarte si podrías explicarles con honestidad tu razonamiento y tu decisión; si no puedes, es una señal para revisar tus conclusiones. La segunda prueba consiste en proyectarte hacia el futuro y preguntarte si podrías arrepentirte de esta decisión más adelante, un ejercicio que, por ejemplo, invita a repensar prácticas con impacto ambiental a la luz del cambio climático.

### Prácticas emergentes de gobernanza

La fuente cierra señalando prácticas concretas que ya se están adoptando en la comunidad de aprendizaje automático: las **fichas de modelo** (model cards), que documentan los usos previstos y las propiedades de un sistema; las **recompensas por sesgo** (bias bounties), inspiradas en los programas de recompensas por encontrar vulnerabilidades de seguridad, pero aplicadas a detectar sesgos en modelos y conjuntos de datos; y la inclusión de una sección de impacto ético y social en el proceso de revisión de conferencias científicas como NeurIPS. A estas se suman técnicas técnicas de protección de datos como la **privacidad diferencial**, que garantiza que la inclusión o exclusión de un único dato no altera de forma perceptible la salida del modelo, y el **aprendizaje federado**, que entrena modelos sobre datos descentralizados sin necesidad de compartir los datos en bruto, reduciendo el riesgo de filtraciones. En Europa, el Reglamento General de Protección de Datos (RGPD) traduce buena parte de estos principios en obligación legal para cualquier sistema que procese datos personales.

## Ejemplo de código comentado, en Python

El código siguiente no entrena una red neuronal: **audita** un modelo de clasificación binaria ya entrenado, calculando una métrica de equidad estándar llamada **diferencia de oportunidad igual** (equal opportunity difference), que compara la tasa de verdaderos positivos (TPR) entre dos grupos demográficos. Es exactamente el tipo de auditoría que la fuente reclama cuando habla de "auditorías regulares del rendimiento del modelo en distintos grupos demográficos". **Copia y ejecuta este código en Google Colab.**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n_por_grupo = 1000

# --- Grupo A: grupo mayoritario, bien representado en el histórico ---
# La etiqueta real depende de ingresos y antigüedad laboral, con algo de ruido.
edad_a = rng.normal(40, 10, n_por_grupo)
ingresos_a = rng.normal(50, 15, n_por_grupo)
antiguedad_a = rng.normal(8, 4, n_por_grupo)
score_a = 0.04 * ingresos_a + 0.3 * antiguedad_a - 5
etiqueta_a = (score_a + rng.normal(0, 3, n_por_grupo) > 0).astype(int)

# --- Grupo B: grupo minoritario, peor representado en el histórico ---
# El proceso causal real es el mismo, pero los datos disponibles reflejan
# ingresos declarados sistemáticamente más bajos y una etiqueta con más
# ruido de medición: así se simula un sesgo de representación en el origen
# de los datos, no un sesgo del modelo en sí mismo.
edad_b = rng.normal(38, 10, n_por_grupo)
ingresos_b = rng.normal(38, 15, n_por_grupo)
antiguedad_b = rng.normal(6, 4, n_por_grupo)
score_b = 0.04 * ingresos_b + 0.3 * antiguedad_b - 5
etiqueta_b = (score_b + rng.normal(0, 5, n_por_grupo) > 0).astype(int)

df = pd.concat([
    pd.DataFrame({"edad": edad_a, "ingresos": ingresos_a, "antiguedad": antiguedad_a,
                  "grupo": "A", "etiqueta": etiqueta_a}),
    pd.DataFrame({"edad": edad_b, "ingresos": ingresos_b, "antiguedad": antiguedad_b,
                  "grupo": "B", "etiqueta": etiqueta_b}),
], ignore_index=True)

X = df[["edad", "ingresos", "antiguedad"]].values
y = df["etiqueta"].values
grupo = df["grupo"].values

X_train, X_test, y_train, y_test, grupo_train, grupo_test = train_test_split(
    X, y, grupo, test_size=0.3, random_state=42, stratify=grupo
)

# --- El "modelo ya entrenado" que vamos a auditar ---
# En un caso real, este modelo llegaría ya entrenado desde otro equipo:
# aquí lo ajustamos una sola vez, con un clasificador simple, solo para
# tener algo concreto que auditar en los pasos siguientes.
modelo = LogisticRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

resultados = pd.DataFrame({"grupo": grupo_test, "y_real": y_test, "y_pred": y_pred})


def tasa_verdaderos_positivos(subconjunto):
    """TPR = proporción de positivos reales que el modelo detecta como positivos."""
    positivos_reales = subconjunto[subconjunto["y_real"] == 1]
    if len(positivos_reales) == 0:
        return np.nan
    return (positivos_reales["y_pred"] == 1).mean()


def diferencia_oportunidad_igual(df_resultados, columna_grupo="grupo"):
    """Calcula la TPR por grupo y la diferencia entre el grupo con mejor y
    peor TPR: es la métrica de equal opportunity difference. Un valor
    cercano a cero indica que el modelo detecta a los positivos reales con
    la misma eficacia en ambos grupos; un valor alejado de cero indica que
    un grupo recibe sistemáticamente menos verdaderos positivos que el otro."""
    tprs = df_resultados.groupby(columna_grupo).apply(tasa_verdaderos_positivos)
    return tprs, tprs.max() - tprs.min()


tprs_por_grupo, diferencia = diferencia_oportunidad_igual(resultados)

print("TPR por grupo:")
print(tprs_por_grupo)
print(f"\nDiferencia de oportunidad igual: {diferencia:.3f}")

# --- Visualización ---
fig, ax = plt.subplots(figsize=(6, 5))
colores = ["#4C72B0", "#DD8452"]
barras = ax.bar(tprs_por_grupo.index, tprs_por_grupo.values, color=colores, width=0.5)

ax.set_ylim(0, 1)
ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
ax.set_title("Auditoría de equidad: TPR por grupo demográfico")
ax.axhline(y=tprs_por_grupo.max(), linestyle="--", color="gray", alpha=0.5)

for barra, valor in zip(barras, tprs_por_grupo.values):
    ax.text(barra.get_x() + barra.get_width() / 2, valor + 0.02, f"{valor:.2f}", ha="center")

ax.text(0.5, -0.12, f"Diferencia de oportunidad igual = {diferencia:.3f}",
        transform=ax.transAxes, ha="center", fontsize=10, style="italic")

plt.tight_layout()
plt.show()
```

Si ejecutas este código, verás que el grupo B obtiene una TPR notablemente más baja que el grupo A, a pesar de que el proceso causal que genera la etiqueta real es idéntico en ambos grupos: la diferencia nace del sesgo de representación que introdujimos en los datos de origen, exactamente el tipo de fallo que describe la fuente. Fíjate en que la diferencia de oportunidad igual solo mira la TPR, es decir, cómo trata el modelo a las personas que de verdad cumplen la condición positiva; existen otras métricas de equidad, como la paridad demográfica, que en cambio comparan la tasa global de predicciones positivas entre grupos sin condicionar a la etiqueta real. Ninguna métrica de equidad es universalmente correcta: **elige** la que corresponda al daño concreto que quieres evitar en tu caso de uso.

## Resumen de puntos clave

- La deuda entre poder técnico y responsabilidad es el hilo conductor del capítulo: quien diseña modelos decide qué oportunidades se abren y cuáles se cierran para otras personas
- El aprendizaje dirigido por especificaciones impone reglas explícitas (robustez, invariancia, privacidad diferencial, incertidumbre ante datos fuera de distribución) que los datos, por sí solos, no garantizan
- El entrenamiento adversarial resuelve un problema min-max: maximizar la pérdida frente a la peor perturbación imperceptible y minimizar los parámetros frente a esa pérdida, usando técnicas como PGA y FGSM
- Una evaluación adversarial débil produce ofuscación de gradientes y una falsa sensación de seguridad; fortalécela con más pasos, más inicializaciones y ataques de caja negra
- Los algoritmos de verificación, completos e incompletos, ofrecen garantías demostrables mediante propagación de cotas, a cambio de un coste computacional o de una cobertura parcial
- El sesgo algorítmico nace en los datos (consentimiento, representación, asociaciones prejudiciales) y se manifiesta en el mundo real: justicia penal, acceso al empleo, reconocimiento de imágenes y diagnóstico médico
- La innovación responsable es un proceso transparente e iterativo entre quienes desarrollan tecnología y la sociedad, con un núcleo de valores compartidos: equidad, privacidad, transparencia y no maleficencia
- El proceso de cinco pasos (beneficio social, riesgo de daño, mitigación, líneas rojas, balance final) y las dos pruebas finales (personas afectadas, arrepentimiento futuro) ofrecen un procedimiento concreto para evaluar un proyecto
- Prácticas como las fichas de modelo, las recompensas por sesgo, la privacidad diferencial y el aprendizaje federado convierten los principios éticos en procesos de gobernanza aplicables

## Ponlo en práctica

Dedica entre quince y treinta minutos a esta mini-tarea:

1. **Copia** el código de la sección de ejemplo en un cuaderno de Google Colab y ejecútalo tal cual, para confirmar que reproduces una diferencia de oportunidad igual claramente distinta de cero
2. **Modifica** la generación de datos: iguala `ingresos_b` a la misma distribución que `ingresos_a` (media 50 en vez de 38) y reduce el ruido de `etiqueta_b` al mismo nivel que el de `etiqueta_a`. Vuelve a ejecutar el código y observa cómo cambia la diferencia de oportunidad igual
3. **Añade** una segunda métrica de equidad en una nueva función: la paridad demográfica, definida como la diferencia entre la proporción de predicciones positivas (`y_pred == 1`) de cada grupo, sin condicionar a la etiqueta real
4. **Compara** ambas métricas sobre los mismos datos sesgados del paso uno: ¿coinciden en la magnitud del problema, o cuentan historias distintas?
5. **Anota** en dos o tres frases qué harías, como responsable técnico, si detectaras en producción una diferencia de oportunidad igual tan grande como la del ejemplo original: qué información recogerías antes de decidir si el modelo puede seguir en producción
