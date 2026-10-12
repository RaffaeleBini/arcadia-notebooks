# Glossario di sigle e acronimi

Prima di entrare nel vivo, tieni a portata di mano questo elenco. L'apprendimento per rinforzo (RL) usa molte sigle in inglese, anche quando lo studi in italiano, perché è così che compaiono nella letteratura e nel codice che scriverai. Ogni capitolo spiega la sigla la prima volta che la usa, ma se in qualche momento perdi il filo, torna qui.

- **A2C** — Advantage Actor-Critic. Metodo actor-critic che usa il vantaggio (advantage) per ridurre la varianza dell'apprendimento. Lo vedi nel capitolo 9.
- **A3C** — Asynchronous Advantage Actor-Critic. Versione asincrona e parallela di A2C, con più agenti che si addestrano contemporaneamente. Lo vedi nel capitolo 12.
- **APM** — Azioni Per Minuto (dall'inglese Actions Per Minute). Misura della velocità di gioco usata nel capitolo 13 quando si parla di AlphaStar.
- **C51** — Categorical 51. Algoritmo di RL distribuzionale che rappresenta il ritorno con 51 valori discreti invece che con un'unica media. Lo vedi nel capitolo 11.
- **CNN** — Convolutional Neural Network (rete neurale convoluzionale). Tipo di rete specializzata nell'elaborazione di immagini o griglie.
- **DDPG** — Deep Deterministic Policy Gradient. Metodo actor-critic per azioni continue, citato nel capitolo 9.
- **DP** — Dynamic Programming (programmazione dinamica). Famiglia di algoritmi che risolvono un Processo di Decisione di Markov (PDM) quando ne conosci il modello completo. La vedi nel capitolo 5.
- **DQN** — Deep Q-Network. Rete neurale che apprende i valori delle azioni combinando Q-learning e deep learning. La vedi nel capitolo 8.
- **GLIE** — Greedy in the Limit with Infinite Exploration (greedy nel limite con esplorazione infinita). Condizione che una strategia di esplorazione deve soddisfare affinché il controllo senza modello converga alla politica ottimale. La vedi nel capitolo 7.
- **GPI** — Generalized Policy Iteration (iterazione di politica generalizzata). Schema comune a quasi tutti gli algoritmi di RL: alternare valutazione e miglioramento della politica. Lo vedi nei capitoli 5 e 7.
- **GPU** — Graphics Processing Unit (unità di elaborazione grafica). Hardware che accelera l'addestramento delle reti neurali.
- **IA** — Intelligenza Artificiale.
- **IMPALA** — Importance Weighted Actor-Learner Architecture. Architettura che scala A3C a molti più lavoratori usando la correzione V-trace. La vedi nel capitolo 12.
- **KL** — Divergenza di Kullback-Leibler. Misura di quanto sono simili due distribuzioni di probabilità, usata da metodi come PPO e TRPO.
- **LSTM** — Long Short-Term Memory. Tipo di rete neurale ricorrente con memoria, utile quando l'agente non osserva l'ambiente nella sua interezza.
- **MC** — Monte Carlo. Famiglia di metodi che apprendono calcolando la media dei ritorni completi degli episodi. La vedi nel capitolo 6.
- **MCMC** — Markov Chain Monte Carlo. Tecnica di campionamento citata in modo puntuale nel materiale di riferimento.
- **MCTS** — Monte Carlo Tree Search (ricerca ad albero di Monte Carlo). Algoritmo di pianificazione che esplora un albero di possibili sviluppi futuri. Lo vedi nei capitoli 10 e 13.
- **MDP** — Markov Decision Process. Nome in inglese del Processo di Decisione di Markov (PDM), la sigla che usa questo manuale.
- **MPC** — Model Predictive Control (controllo predittivo basato su modello). Tecnica di pianificazione citata brevemente nel capitolo 10.
- **PDM** — Processo di Decisione di Markov. Il formalismo matematico centrale dell'RL: stati, azioni, transizioni e ricompense. Lo vedi nel capitolo 4.
- **PER** — Prioritized Experience Replay (riproduzione di esperienze prioritizzata). Miglioramento di DQN che ripassa più spesso le esperienze più sorprendenti. Lo vedi nel capitolo 11.
- **PPO** — Proximal Policy Optimization. Metodo a gradiente di politica molto usato nella pratica, citato nel capitolo 9.
- **RL** — Reinforcement Learning (apprendimento per rinforzo). Il tema centrale di questo manuale.
- **SARSA** — State-Action-Reward-State-Action. Nome dell'algoritmo di controllo on-policy che prende il nome dalla sequenza che usa per apprendere. Lo vedi nel capitolo 7.
- **TD** — Temporal-Difference (differenze temporali). Famiglia di metodi che apprendono aggiornando le proprie stime a partire da altre stime, senza aspettare la fine dell'episodio. La vedi nel capitolo 6.
- **TRPO** — Trust Region Policy Optimization. Predecessore di PPO, citato nel capitolo 9.
- **UCB** — Upper Confidence Bound (limite superiore di confidenza). Strategia di esplorazione che privilegia le azioni con più incertezza. La vedi nel capitolo 3.
- **UCT** — Upper Confidence bounds applied to Trees. Versione di UCB usata all'interno di MCTS. La vedi nel capitolo 10.

[Torna all'indice](00_indice_IT.md)
