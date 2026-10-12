[⬅ Torna all'indice](00_indice_IT.md)

# Capitolo 6 — Apprendimento per rinforzo: insegna a un'IA a giocare

## Perché è importante

Fino a ora i tuoi modelli imparavano da esempi già etichettati: "questa immagine è un gatto", "questa recensione è negativa". Ma, come addestri un'IA quando non esiste una risposta corretta predeterminata, solo la possibilità di provarci e vedere cosa succede? Questo è il terreno dell'apprendimento per rinforzo, o Reinforcement Learning (RL): la tecnica dietro i sistemi che imparano a giocare a videogiochi meglio di qualsiasi umano, a controllare robot o a ottimizzare decisioni in tempo reale.

## Come impara un agente per prova e errore (con un'analogia)

Pensa a come hai imparato ad andare in bicicletta: nessuno ti ha dato un elenco di regole esatte su quanto girare il manubrio in ogni istante. Hai provato, sei caduto, e hai aggiustato il tuo comportamento in base a ciò che funzionava (stare in piedi) e ciò che non funzionava (cadere). Con il tempo, hai sviluppato un'intuizione —una "politica"— su cosa fare in ogni situazione per massimizzare il tempo in cui rimani in equilibrio.

In RL, un "agente" interagisce con un "ambiente": in ogni fase osserva uno stato, sceglie un'azione, e riceve un premio che gli indica se quell'azione è stata buona o cattiva. L'obiettivo dell'agente è imparare una politica che massimizzi il premio cumulato nel tempo, non solo il premio immediato.

## L'ambiente CartPole

OpenAI Gym offre ambienti standardizzati per praticare RL senza dover programmare tu la simulazione. Il più classico è CartPole: un carrello su una rotaia con un'asta in equilibrio sopra, che l'agente deve mantenere verticale spostandosi a sinistra o a destra. L'ambiente termina l'episodio se l'asta si inclina troppo o se il carrello esce dai limiti; l'obiettivo è durare il massimo numero di step possibile.

```python
import gym

entorno = gym.make("CartPole-v1")
observacion, info = entorno.reset()

print("Forma dell'osservazione:", observacion.shape)
print("Azioni possibili:", entorno.action_space.n)
```

Ogni osservazione ha quattro valori: posizione e velocità del carrello, e angolo e velocità angolare dell'asta. Ci sono solo due azioni possibili: spingere a sinistra o a destra.

## Un agente che impara: Deep Q-Network

Un Deep Q-Network (DQN) usa una rete neurale per stimare, per ogni stato e ogni azione possibile, quanto premio futuro si aspetta di ottenere l'agente se sceglie quell'azione. Con questa stima, l'agente sceglie sempre l'azione con il valore atteso maggiore.

```python
import tensorflow as tf
import numpy as np

modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(24, activation="relu"),
    tf.keras.layers.Dense(24, activation="relu"),
    tf.keras.layers.Dense(2, activation="linear"),
])
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")

def elegir_accion(observacion, epsilon):
    # Con probabilità epsilon, esplora con un'azione casuale
    if np.random.random() < epsilon:
        return np.random.randint(2)
    # Se no, sfrutta la conoscenza attuale: l'azione con il valore stimato maggiore
    valores_q = modelo.predict(observacion.reshape(1, -1), verbose=0)
    return int(np.argmax(valores_q[0]))
```

Il parametro `epsilon` controlla l'equilibrio tra "esplorare" (provare azioni casuali per scoprire nuove strategie) e "sfruttare" (usare ciò che già si è imparato). All'inizio dell'addestramento conviene un `epsilon` alto —l'agente non sa ancora nulla—, e si riduce poco a poco man mano che l'agente guadagna esperienza.

## Riepilogo dei punti chiave

- Nell'apprendimento per rinforzo non ci sono risposte corrette predeterminate: l'agente impara interagendo con un ambiente e ricevendo premi.
- OpenAI Gym / Gymnasium offre ambienti standardizzati, come CartPole, per praticare senza dover programmare tu la simulazione.
- Un DQN usa una rete neurale per stimare il valore atteso di ogni azione in ogni stato.
- L'equilibrio tra esplorazione e sfruttamento, controllato da `epsilon`, è fondamentale perché l'agente scopra buone strategie senza restare intrappolato nelle prime che trova.

## Mettilo in pratica

**Modifica** la funzione `elegir_accion` in modo che `epsilon` cominci a 1.0 e si riduca moltiplicandolo per 0.99 dopo ogni episodio, fino a un minimo di 0.05. **Esegui** alcuni episodi di prova con azioni casuali e **annota** quanti step riesce a durare l'asta in piedi in media, come riferimento prima di addestrare il vero agente.

---

**Materiale sorgente di questo capitolo**: lezione 6 argomenti 1 a 4 ("Introduction", "Training Data", "Training Model", "Testing Network") sull'ambiente CartPole.

[⬅ Capitolo precedente](05_tflearn_keras_IT.md) · [Torna all'indice](00_indice_IT.md) · [Prossimo capitolo ➡](07_dogs_vs_cats_IT.md)
