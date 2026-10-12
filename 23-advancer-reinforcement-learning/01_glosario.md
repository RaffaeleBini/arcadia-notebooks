# Glosario de siglas y acrónimos

Antes de entrar en materia, ten a mano esta lista. El aprendizaje por refuerzo (RL) usa muchas siglas en inglés, incluso cuando lo estudias en español, porque así aparecen en la literatura y en el código que vas a escribir. Cada capítulo explica la sigla la primera vez que la usa, pero si en algún momento pierdes el hilo, vuelve aquí.

- **A2C** — Advantage Actor-Critic. Método actor-crítico que usa la ventaja (advantage) para reducir la varianza del aprendizaje. Lo ves en el capítulo 9.
- **A3C** — Asynchronous Advantage Actor-Critic. Versión asíncrona y paralela de A2C, con varios agentes entrenando a la vez. Lo ves en el capítulo 12.
- **APM** — Acciones Por Minuto (del inglés Actions Per Minute). Medida de velocidad de juego usada en el capítulo 13 al hablar de AlphaStar.
- **C51** — Categorical 51. Algoritmo de RL distribucional que representa el retorno con 51 valores discretos en vez de un único promedio. Lo ves en el capítulo 11.
- **CNN** — Convolutional Neural Network (red neuronal convolucional). Tipo de red especializada en procesar imágenes o cuadrículas.
- **DDPG** — Deep Deterministic Policy Gradient. Método actor-crítico para acciones continuas, mencionado en el capítulo 9.
- **DP** — Dynamic Programming (programación dinámica). Familia de algoritmos que resuelven un Proceso de Decisión de Markov (PDM) cuando conoces su modelo completo. La ves en el capítulo 5.
- **DQN** — Deep Q-Network. Red neuronal que aprende valores de acción combinando Q-learning con aprendizaje profundo. Lo ves en el capítulo 8.
- **GLIE** — Greedy in the Limit with Infinite Exploration (codicioso en el límite con exploración infinita). Condición que debe cumplir una estrategia de exploración para que el control sin modelo converja a la política óptima. La ves en el capítulo 7.
- **GPI** — Generalized Policy Iteration (iteración de política generalizada). Patrón común a casi todos los algoritmos de RL: alternar evaluación y mejora de la política. Lo ves en los capítulos 5 y 7.
- **GPU** — Graphics Processing Unit (unidad de procesamiento gráfico). Hardware que acelera el entrenamiento de redes neuronales.
- **IA** — Inteligencia Artificial.
- **IMPALA** — Importance Weighted Actor-Learner Architecture. Arquitectura que escala A3C a muchos más trabajadores usando la corrección V-trace. La ves en el capítulo 12.
- **KL** — Divergencia de Kullback-Leibler. Medida de cuánto se parecen dos distribuciones de probabilidad, usada por métodos como PPO y TRPO.
- **LSTM** — Long Short-Term Memory. Tipo de red neuronal recurrente con memoria, útil cuando el agente no observa el entorno completo.
- **MC** — Monte Carlo. Familia de métodos que aprenden promediando retornos completos de episodios. La ves en el capítulo 6.
- **MCMC** — Markov Chain Monte Carlo. Técnica de muestreo mencionada de forma puntual en el material fuente.
- **MCTS** — Monte Carlo Tree Search (búsqueda en árbol de Monte Carlo). Algoritmo de planificación que explora un árbol de futuros posibles. Lo ves en los capítulos 10 y 13.
- **MDP** — Markov Decision Process. Nombre en inglés del Proceso de Decisión de Markov (PDM), la sigla que usa este manual.
- **MPC** — Model Predictive Control (control predictivo por modelo). Técnica de planificación mencionada de forma breve en el capítulo 10.
- **PDM** — Proceso de Decisión de Markov. El formalismo matemático central del RL: estados, acciones, transiciones y recompensas. Lo ves en el capítulo 4.
- **PER** — Prioritized Experience Replay (repetición de experiencias priorizada). Mejora de DQN que repasa con más frecuencia las experiencias más sorprendentes. La ves en el capítulo 11.
- **PPO** — Proximal Policy Optimization. Método de gradiente de política muy usado en la práctica, mencionado en el capítulo 9.
- **RL** — Reinforcement Learning (aprendizaje por refuerzo). El tema central de este manual.
- **SARSA** — State-Action-Reward-State-Action. Nombre del algoritmo de control on-policy que toma su nombre de la secuencia que usa para aprender. Lo ves en el capítulo 7.
- **TD** — Temporal-Difference (diferencias temporales). Familia de métodos que aprenden actualizando sus estimaciones a partir de otras estimaciones, sin esperar al final del episodio. La ves en el capítulo 6.
- **TRPO** — Trust Region Policy Optimization. Predecesor de PPO, mencionado en el capítulo 9.
- **UCB** — Upper Confidence Bound (cota superior de confianza). Estrategia de exploración que prioriza acciones con más incertidumbre. La ves en el capítulo 3.
- **UCT** — Upper Confidence bounds applied to Trees. Versión de UCB usada dentro de MCTS. La ves en el capítulo 10.

[Volver al índice](00_indice.md)
