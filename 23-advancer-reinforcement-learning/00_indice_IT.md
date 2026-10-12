# Advanced Reinforcement Learning for Dummies

Manuale di apprendimento per rinforzo (RL) da zero, elaborato a partire dal programma ufficiale della certificazione EITC/AI/ARL Advanced Reinforcement Learning. Non serve esperienza pregressa in Python avanzato, algebra lineare o machine learning: ogni capitolo si basa su quello precedente e ti spiega ciò che ti serve nel momento in cui ne hai bisogno.

## Come usare questo manuale

Leggi i capitoli in ordine: ciascuno riutilizza e amplia il codice e i concetti del precedente. Inizia sempre dal [glossario](01_glosario_IT.md) se incontri una sigla che non ricordi.

## Indice dei contenuti

0. [Glossario di sigle e acronimi](01_glosario_IT.md)
1. [Introduzione all'apprendimento per rinforzo](02_introduccion_al_rl_IT.md)
2. [Il dilemma dei bandit: esplorazione e sfruttamento](03_bandidos_exploracion_explotacion_IT.md)
3. [Processi di Decisione di Markov: il linguaggio dell'RL](04_procesos_decision_markov_IT.md)
4. [Programmazione dinamica: iterazione di politica e di valore](05_programacion_dinamica_IT.md)
5. [Predizione senza modello: Monte Carlo e Differenze Temporali](06_prediccion_sin_modelo_IT.md)
6. [Controllo senza modello: SARSA e Q-learning](07_control_sin_modelo.md)
7. [Approssimazione di funzioni e Deep Q-Networks (DQN)](08_dqn_aproximacion_funciones.md)
8. [Gradienti di politica e metodi Actor-Critic](09_gradientes_politica_actor_critico.md)
9. [Pianificazione e modelli: quando l'agente impara come funziona il mondo](10_planificacion_y_modelos.md)
10. [Rainbow e l'apprendimento distribuzionale: quando diversi miglioramenti si sommano](11_temas_avanzados_deep_rl.md)
11. [Architetture di agenti: da A3C a IMPALA](12_arquitecturas_agentes_drl.md)
12. [Casi d'uso reali: dai giochi classici ad AlphaStar](13_casos_de_uso_reales.md)

## I quattro blocchi del manuale

| Blocco | Capitoli | Di cosa tratta |
|---|---|---|
| Fondamenti dell'RL | 1-2 | Agente, ambiente, ricompensa, politica e il dilemma tra esplorare e sfruttare |
| Il PDM e l'apprendimento senza modello | 3-6 | Il formalismo matematico dell'RL e come apprendere dall'esperienza, con o senza conoscere il modello dell'ambiente |
| Deep Reinforcement Learning | 7-11 | Come combinare RL e reti neurali: DQN, gradienti di politica, pianificazione e i miglioramenti che hanno dato vita agli agenti moderni |
| Casi d'uso reali | 12 | Come queste idee sono state combinate in sistemi come AlphaGo, AlphaZero e AlphaStar |

Ogni capitolo segue la stessa struttura: perché il tema è importante, una spiegazione concettuale con analogie, un esempio di codice funzionante e commentato, un riepilogo dei punti chiave e una sezione "Mettilo in pratica" con un mini-compito per sperimentare da solo.

Consulta anche il [PROGRAMMA del corso](../TEMARIO.md) per vedere come questo manuale è organizzato in dodici lezioni, con i relativi notebook di teoria e di pratica.
