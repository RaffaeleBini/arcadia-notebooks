# Advanced Reinforcement Learning for Dummies

Manual de aprendizaje por refuerzo (RL) desde cero, elaborado a partir del temario oficial de la certificación EITC/AI/ARL Advanced Reinforcement Learning. No necesitas experiencia previa en Python avanzado, álgebra lineal ni machine learning: cada capítulo construye sobre el anterior y te explica lo que necesitas en el momento en que lo necesitas.

## Cómo usar este manual

Lee los capítulos en orden: cada uno reutiliza y amplía el código y los conceptos del anterior. Empieza siempre por el [glosario](01_glosario.md) si te encuentras con una sigla que no recuerdas.

## Índice de contenidos

0. [Glosario de siglas y acrónimos](01_glosario.md)
1. [Introducción al aprendizaje por refuerzo](02_introduccion_al_rl.md)
2. [El dilema de los bandidos: exploración y explotación](03_bandidos_exploracion_explotacion.md)
3. [Procesos de Decisión de Markov: el lenguaje del RL](04_procesos_decision_markov.md)
4. [Programación dinámica: iteración de políticas y de valores](05_programacion_dinamica.md)
5. [Predicción sin modelo: Monte Carlo y Diferencias Temporales](06_prediccion_sin_modelo.md)
6. [Control sin modelo: SARSA y Q-learning](07_control_sin_modelo.md)
7. [Aproximación de funciones y Deep Q-Networks (DQN)](08_dqn_aproximacion_funciones.md)
8. [Gradientes de política y métodos Actor-Crítico](09_gradientes_politica_actor_critico.md)
9. [Planificación y modelos: cuando el agente aprende cómo funciona el mundo](10_planificacion_y_modelos.md)
10. [Rainbow y el aprendizaje distribucional: cuando varias mejoras se suman](11_temas_avanzados_deep_rl.md)
11. [Arquitecturas de agentes: de A3C a IMPALA](12_arquitecturas_agentes_drl.md)
12. [Casos de uso reales: de los juegos clásicos a AlphaStar](13_casos_de_uso_reales.md)

## Los cuatro bloques del manual

| Bloque | Capítulos | De qué trata |
|---|---|---|
| Fundamentos del RL | 1-2 | Agente, entorno, recompensa, política, y el dilema entre explorar y explotar |
| El PDM y el aprendizaje sin modelo | 3-6 | El formalismo matemático del RL y cómo aprender a partir de la experiencia, con y sin conocer el modelo del entorno |
| Deep Reinforcement Learning | 7-11 | Cómo combinar RL con redes neuronales: DQN, gradientes de política, planificación, y las mejoras que dieron lugar a los agentes modernos |
| Casos de uso reales | 12 | Cómo estas ideas se combinaron en sistemas como AlphaGo, AlphaZero y AlphaStar |

Cada capítulo sigue la misma estructura: por qué importa el tema, una explicación conceptual con analogías, un ejemplo de código funcional y comentado, un resumen de puntos clave, y una sección "Ponlo en práctica" con una mini-tarea para que experimentes por tu cuenta.

Consulta también el [TEMARIO del curso](../TEMARIO.md) para ver cómo se organiza este manual en doce clases, con sus notebooks de teoría y de práctica correspondientes.
