# Capítulo 2 — Cómo se construye un ordenador cuántico

## Por qué importa

En el capítulo anterior hablaste de qubits como si fueran monedas girando en el aire. Pero un qubit real es un objeto físico, y construirlo es una de las hazañas de ingeniería más difíciles de la actualidad. Entender cómo se fabrica un qubit te ayuda a comprender por qué la computación cuántica actual tiene límites (ruido, pocos qubits, tiempos de coherencia cortos) y por qué muchos de los algoritmos que vas a programar en este curso están diseñados precisamente para convivir con esas limitaciones. A esta etapa de la tecnología se la conoce como la era **NISQ** (cuántico de escala intermedia y ruidoso): tenemos ordenadores cuánticos, pero todavía imperfectos.

## Explicación conceptual

### El qubit superconductor: un circuito que se comporta como un átomo

La forma más habitual de construir un qubit hoy en día (la que usan Google o IBM) es el **qubit superconductor**, también llamado transmón. En vez de usar un átomo real, se fabrica un circuito eléctrico diminuto que, enfriado a temperaturas extremas, se comporta igual que un átomo con solo dos niveles de energía posibles: el 0 y el 1.

La pieza clave de ese circuito es la **unión Josephson**: dos trozos de metal superconductor separados por una capa aislante ultrafina. Es como una compuerta que deja pasar corriente eléctrica de una forma muy particular, sin resistencia y con un comportamiento cuántico. Gracias a esa unión, el circuito completo se comporta como si tuviera niveles de energía discretos, exactamente igual que un átomo.

### Por qué hace tanto frío dentro de un ordenador cuántico

Para que el circuito se comporte como un qubit y no como un cable normal, el metal tiene que estar en estado **superconductor** (sin resistencia eléctrica), y eso solo ocurre a temperaturas cercanas al cero absoluto. Por eso los ordenadores cuánticos de Google viven dentro de un **refrigerador de dilución**: una estructura en forma de araña metálica que enfría el chip hasta unos 15 milikelvin, más frío que el espacio exterior.

### Cómo controlas un qubit sin destruirlo

Controlar un qubit significa mandarle pulsos de microondas muy precisos para aplicarle puertas cuánticas (las que viste en el capítulo 1). El problema es que un pulso mal diseñado puede excitar el qubit a un nivel de energía que no quieres (un "estado fantasma"), estropeando el cálculo. La técnica **DRAG** (eliminación derivativa por puerta adiabática) da forma a esos pulsos de microondas para evitar justo ese error, y es una de las piezas que hace posible controlar los qubits con la precisión necesaria.

Además, toda esa electrónica de control tiene que vivir muy cerca del chip cuántico para no perder precisión por el camino, así que también se investiga **CMOS criogénico** (semiconductor complementario de óxido metálico): chips de control clásicos diseñados para funcionar a esas mismas temperaturas extremas, en vez de a temperatura ambiente.

### El enemigo común: la decoherencia

Por mucho cuidado que pongas, un qubit real siempre está expuesto a vibraciones, campos magnéticos residuales o imperfecciones del material, y con el tiempo pierde su estado cuántico: eso se llama **decoherencia**. El tiempo que un qubit aguanta en superposición antes de decoherer se llama **tiempo de coherencia**, y cuanto más corto es, menos operaciones puedes encadenar antes de que el resultado deje de ser fiable. Esta es la razón principal por la que los algoritmos de este curso (QAOA, VQE, QRL) están diseñados como circuitos cortos y poco profundos: cuanto menos dure el circuito, menos tiempo tiene la decoherencia para arruinar el resultado.

## Resumen de puntos clave

- Un qubit superconductor (transmón) es un circuito eléctrico que, enfriado casi al cero absoluto, se comporta como un átomo con dos niveles de energía
- La unión Josephson es la pieza física que hace posible ese comportamiento cuántico
- Los refrigeradores de dilución enfrían el chip a temperaturas de milikelvin, imprescindibles para mantener la superconductividad
- La técnica DRAG da forma a los pulsos de control para evitar errores; el CMOS criogénico acerca la electrónica de control al chip
- La decoherencia (pérdida del estado cuántico por el ruido del entorno) es la principal limitación de la era NISQ y explica por qué los circuitos cuánticos actuales deben ser cortos

## Ponlo en práctica

No necesitas escribir código para esta tarea. Busca el tiempo de coherencia aproximado que se suele citar para los procesadores superconductores de Google (como Sycamore) y compáralo con el tiempo que tarda en ejecutarse una sola puerta cuántica. Calcula, a partir de esos dos datos, cuántas puertas cuánticas como máximo puedes encadenar en un circuito antes de que la decoherencia empiece a dominar el resultado. Anota tu cálculo y la fuente que has usado: esta cuenta es la que explica por qué los circuitos de los próximos capítulos son tan cortos.
