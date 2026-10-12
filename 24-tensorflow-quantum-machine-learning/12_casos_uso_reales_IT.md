# Capitolo 12 — Casi d'uso reali

## Perché è importante

Hai percorso l'intero corso: dai qubit e dalle porte fino al VQE avanzato. È il momento di fare un passo indietro e rispondere alla domanda che probabilmente ti sei posto nel capitolo 1: a cosa serve tutto questo nella vita reale? Questo capitolo di chiusura raccoglie quattro applicazioni concrete di ciò che hai imparato, tutte costruite sul materiale che hai lavorato durante il corso, così da terminare con una visione chiara di dove puoi portare queste conoscenze.

## Spiegazione concettuale

### Chimica quantistica: calcolare l'energia di una molecola reale

L'esempio più citato di applicazione del VQE (capitoli 10 e 11) è il calcolo della **curva di dissociazione** di una molecola di idrogeno (H₂): l'energia del sistema in funzione della distanza tra i due atomi. Questo calcolo è esattamente lo stesso problema che hai risolto con Hamiltoniani semplici, ma ora l'Hamiltoniano si ottiene dalla fisica reale della molecola (tramite la trasformazione di Jordan-Wigner, che converte un problema di elettroni in un problema di qubit) invece di inventarlo tu. Le aziende farmaceutiche e dei materiali seguono da vicino questo tipo di calcoli perché, a lungo termine, promettono di progettare nuovi farmaci e materiali simulandone il comportamento quantistico reale, qualcosa che un computer classico riesce solo ad approssimare per molecole molto piccole.

### Ottimizzazione combinatoria: da Max-Cut ai problemi logistici

Il problema Max-Cut che hai risolto con QAOA (capitolo 9) è un modello semplificato di problemi molto più applicati: ripartire compiti tra server per ridurre la comunicazione tra di loro, raggruppare clienti per somiglianza, o progettare reti che minimizzino le interferenze. La struttura matematica (dividere un insieme di elementi collegati tra loro in gruppi che ottimizzino una certa metrica) si ripete in logistica, telecomunicazioni e progettazione di reti elettriche.

### Controllo e processo decisionale: da CartPole alla robotica

L'agente che ha imparato a bilanciare un'asta in CartPole (capitolo 8) usa lo stesso principio che serve per controllare un robot, gestire un portafoglio di investimenti o decidere quando caricare una batteria: un agente che osserva uno stato, sceglie un'azione e impara dalla ricompensa che riceve. Sostituire la rete neurale classica con un circuito quantistico è ancora un campo di ricerca attivo, ma dimostra già che l'apprendimento per rinforzo quantistico può imparare politiche di controllo competitive in problemi di dimensioni ridotte.

### Validare l'hardware: la supremazia quantistica come controllo di qualità

Infine, ricorda il capitolo 4: l'esperimento di supremazia quantistica di Sycamore non risolveva di per sé un problema utile, ma dimostrava che l'hardware quantistico si comporta come ci si aspetta. Questo stesso tipo di verifica (confrontare il comportamento reale di un chip con ciò che prevede la teoria) è il primo passo obbligatorio prima di fidarsi di una qualsiasi delle applicazioni precedenti: a poco serve un circuito di VQE o QAOA perfettamente progettato se l'hardware che lo esegue non è affidabile.

## Esempio di codice commentato (estensione opzionale)

Il seguente esempio calcola la curva di dissociazione dell'H₂ con VQE, usando la libreria `openfermion` per ottenere l'Hamiltoniano molecolare reale. Installare `openfermion` e `openfermionpyscf` in Colab richiede diversi minuti (compilano la libreria di chimica quantistica `pyscf`), quindi questo blocco resta un'**estensione opzionale**: non serve eseguirlo per completare il corso, ma te lo mostriamo commentato per farti vedere come il VQE passa da un esempio da lezione a un caso d'uso reale.

```python
# Instalación (varios minutos): !pip install openfermion openfermionpyscf
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
import openfermion as of
from openfermionpyscf import generate_molecular_hamiltonian
from scipy.sparse import linalg

def capa_ansatz(circuito, qubits, parametros):
    for i, qubit in enumerate(qubits):
        circuito += cirq.ry(parametros[3 * i]).on(qubit)
        circuito += cirq.rz(parametros[3 * i + 1]).on(qubit)
        circuito += cirq.ry(parametros[3 * i + 2]).on(qubit)
    for i in range(len(qubits) - 1):
        circuito += cirq.CNOT(qubits[i], qubits[i + 1])
    return circuito

def construir_vqe(qubits, capas, hamiltoniano):
    num_parametros = capas * 3 * len(qubits)
    parametros = sympy.symbols(f"vqe0:{num_parametros}")
    circuito = cirq.Circuit()
    for _ in range(capas):
        circuito = capa_ansatz(circuito, qubits, parametros)
    entradas = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    salida = tfq.layers.PQC(circuito, hamiltoniano, differentiator=tfq.differentiators.Adjoint())(entradas)
    return tf.keras.models.Model(inputs=entradas, outputs=salida)

distancias = [0.3, 0.6, 0.9, 1.2, 1.5, 2.0]  # distancia entre los dos átomos de hidrógeno, en angstroms
energias_reales, energias_vqe = [], []

for distancia in distancias:
    geometria = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, distancia))]
    hamiltoniano_molecular = generate_molecular_hamiltonian(geometria, "sto-3g", multiplicity=1, charge=0)
    num_qubits = of.count_qubits(hamiltoniano_molecular)
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]

    operador_jw = of.transforms.jordan_wigner(hamiltoniano_molecular)
    matriz_dispersa = of.get_sparse_operator(operador_jw)
    autovalor_real, _ = linalg.eigsh(matriz_dispersa, k=1, which="SA")
    hamiltoniano_cirq = of.transforms.qubit_operator_to_pauli_sum(operador_jw, qubits=qubits)

    modelo = construir_vqe(qubits, capas=2, hamiltoniano=hamiltoniano_cirq)
    entrada_vacia = tfq.convert_to_tensor([cirq.Circuit()])
    optimizador = tf.keras.optimizers.Adam(learning_rate=0.1)

    energia_actual, anterior = None, np.inf
    for _ in range(200):
        with tf.GradientTape() as cinta:
            energia_actual = modelo(entrada_vacia)
        gradientes = cinta.gradient(energia_actual, modelo.trainable_variables)
        optimizador.apply_gradients(zip(gradientes, modelo.trainable_variables))
        energia_actual = energia_actual.numpy()[0][0]
        if abs(energia_actual - anterior) < 1e-5:
            break
        anterior = energia_actual

    energias_vqe.append(energia_actual)
    energias_reales.append(autovalor_real[0])

plt.plot(distancias, energias_vqe, "o-", label="VQE (TensorFlow Quantum)")
plt.plot(distancias, energias_reales, "--", label="Energía exacta")
plt.xlabel("Distancia entre átomos (angstroms)")
plt.ylabel("Energía (Hartree)")
plt.title("Curva de disociación del H₂ con VQE")
plt.legend()
plt.show()
```

La curva risultante ha la forma di una "U": l'energia scende all'inizio (gli atomi si attraggono), tocca un minimo (la distanza di legame reale della molecola) e risale quando li separi troppo (il legame si rompe). Verificare che la curva del VQE si sovrapponga all'energia esatta è la validazione finale del fatto che l'algoritmo funziona su un sistema fisico reale, e non solo su esempi giocattolo.

## Riepilogo dei punti chiave

- Il VQE si applica alla chimica quantistica reale calcolando curve di dissociazione molecolare, con l'Hamiltoniano generato a partire dalla geometria reale di una molecola
- QAOA su Max-Cut modella problemi di ottimizzazione logistica, di reti e di ripartizione di risorse
- L'apprendimento per rinforzo quantistico, addestrato in ambienti di controllo come CartPole, punta verso applicazioni di robotica e di decisione sequenziale
- La verifica dell'hardware tramite la supremazia quantistica è il passo preliminare indispensabile per fidarsi di una qualsiasi delle applicazioni precedenti
- Tutte queste applicazioni partono dagli stessi elementi che hai imparato nel corso: qubit, circuiti parametrizzati, Hamiltoniani e ottimizzatori classici o quantistici

## Mettilo in pratica

Scegli una delle quattro applicazioni di questo capitolo (chimica quantistica, ottimizzazione combinatoria, apprendimento per rinforzo o validazione dell'hardware) e cerca un articolo recente (meno di due anni) che la applichi a un problema diverso dagli esempi di questo corso. Scrivi un riassunto di tre-cinque frasi che spieghi: quale problema risolve, quale algoritmo quantistico tra quelli che hai imparato usa (o una sua variante), e quale limitazione lo stesso articolo menziona sullo stato attuale della tecnologia.
