# Capitolo 5 — TensorFlow Quantum e il problema delle plateau sterili

## Perché è importante

Ora sai programmare circuiti quantistici con Cirq. Adesso ti serve il pezzo che manca: come addestrare questi circuiti con gli stessi strumenti di machine learning che useresti per addestrare una rete neurale classica (discesa del gradiente, funzioni di perdita, ottimizzatori come Adam). Questo pezzo è **TensorFlow Quantum** (TFQ), la piattaforma che collega Cirq a Keras. Ma prima di lanciarti nell'addestramento, devi conoscere un ostacolo che compare proprio quando i circuiti quantistici crescono: il problema delle **plateau sterili** (barren plateaus), che può far sì che l'addestramento non impari semplicemente nulla.

## Spiegazione concettuale

### TFQ: il ponte tra Cirq e Keras

TFQ converte un circuito Cirq in un tensore di TensorFlow (con `tfq.convert_to_tensor`) in modo che possa attraversare una rete Keras come qualsiasi altro dato. Da lì in poi, hai a disposizione alcuni layer speciali:

- `tfq.layers.PQC` (circuito quantistico parametrizzato): trasforma un circuito con simboli in un layer Keras addestrabile. Tu gli indichi quale circuito usare e quale "operatore di lettura" misurare (per esempio, `cirq.Z(qubit)`), e il layer si occupa di regolare gli angoli del circuito durante l'addestramento, proprio come un layer denso regola i suoi pesi
- `tfq.layers.Expectation` e `tfq.layers.SampledExpectation`: calcolano il valore di aspettazione di una misura su un circuito. La versione "Sampled" simula il rumore di un hardware reale, ripetendo il circuito un numero finito di volte (`repetitions`) invece di calcolare la probabilità esatta
- `tfq.differentiators.ParameterShift`: il metodo che TFQ usa per calcolare i gradienti di un circuito quantistico. A differenza di una rete classica, in un circuito quantistico non puoi "derivare" una porta quantistica con le regole di calcolo abituali; lo spostamento di parametro (parameter shift) aggira il problema valutando il circuito con il parametro leggermente spostato verso l'alto e verso il basso, e combinando i due risultati

Con questi pezzi, addestrare un circuito quantistico somiglia molto ad addestrare una rete classica: definisci un modello Keras, lo compili con una funzione di perdita e un ottimizzatore, e chiami `.fit(...)`.

### Il problema delle plateau sterili

Quando aumenti il numero di qubit o la profondità di un circuito quantistico parametrizzato inizializzato a caso, succede qualcosa di controintuitivo: il gradiente (il segnale che indica all'ottimizzatore in quale direzione muoversi) diventa, in media, praticamente zero in quasi tutto lo spazio dei parametri. Questa situazione si chiama **plateau sterile**: il paesaggio della funzione di perdita è così piatto che l'ottimizzatore non sa dove andare, come provare a scendere da una montagna nella nebbia fitta su un terreno completamente piatto.

Non è un errore nel tuo codice: è una proprietà matematica dei circuiti quantistici casuali e profondi. Più qubit aggiungi, più velocemente decade la varianza del gradiente, e questo decadimento è esponenziale. È il motivo per cui, in questo corso, lavorerai con circuiti volutamente poco profondi e con pochi qubit: meno qubit e meno strati ci sono, meno grave è il problema.

### Apprendimento a strati (layer-wise learning): un'uscita dal plateau

Una strategia per mitigare le plateau sterili è l'**apprendimento a strati**: invece di inizializzare tutto il circuito a caso e addestrare tutti i parametri insieme, prima addestri uno strato poco profondo, lo "congeli" (o lo lasci vicino ai valori addestrati) e aggiungi poco alla volta un nuovo strato. In questo modo il circuito non attraversa mai una fase di profondità elevata con parametri completamente casuali, che è proprio la combinazione che fa scattare la plateau sterile.

Anche scegliere con cura cosa misurare aiuta: nell'esempio di codice di questo capitolo verificherai che misurare la somma di più operatori Z (uno per qubit) soffre meno di plateau sterili che misurare un unico operatore che combina più qubit insieme (per esempio, Z⊗Z).

## Esempio di codice commentato

Questo esempio riproduce l'esperimento classico sulle plateau sterili: genera circuiti casuali di profondità crescente e misura come crolla la varianza del gradiente. Usiamo pochi qubit e poche ripetizioni per eseguirlo in pochi minuti, ma la tendenza è già visibile.

```python
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt

def generar_circuito_aleatorio(qubits, profundidad, parametro):
    """Un circuito con una puerta aleatoria distinta en cada qubit y en cada capa,
    entrelazado con puertas CZ. Solo el primer parámetro es simbólico (entrenable);
    el resto son ángulos aleatorios fijos, tal y como exige el experimento original."""
    circuito = cirq.Circuit()
    for qubit in qubits:
        circuito += cirq.ry(np.pi / 4.0)(qubit)

    for capa in range(profundidad):
        for i, qubit in enumerate(qubits):
            es_el_parametro_entrenable = (i == 0 and capa == 0)
            angulo = parametro if es_el_parametro_entrenable else np.random.uniform(0, 2 * np.pi)
            eleccion = np.random.uniform()
            if eleccion > 2 / 3:
                circuito += cirq.rz(angulo)(qubit)
            elif eleccion > 1 / 3:
                circuito += cirq.ry(angulo)(qubit)
            else:
                circuito += cirq.rx(angulo)(qubit)
        for origen, destino in zip(qubits, qubits[1:]):
            circuito += cirq.CZ(origen, destino)
    return circuito

def varianza_del_gradiente(num_qubits, profundidad, num_circuitos, repeticiones):
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    simbolo = sympy.symbols("parametro")
    circuitos = [generar_circuito_aleatorio(qubits, profundidad, simbolo) for _ in range(num_circuitos)]

    operador_lectura = [sum(cirq.Z(q) for q in qubits)]  # suma de Z, uno por qubit
    diferenciador = tfq.differentiators.ParameterShift()
    capa_expectativa = tfq.layers.SampledExpectation(differentiator=diferenciador)

    circuitos_tensor = tfq.convert_to_tensor(circuitos)
    valores = tf.convert_to_tensor(np.random.uniform(0, 2 * np.pi, (num_circuitos, 1)).astype(np.float32))

    with tf.GradientTape() as cinta:
        cinta.watch(valores)
        salida = capa_expectativa(circuitos_tensor, operators=operador_lectura,
                                   repetitions=repeticiones, symbol_names=[simbolo], symbol_values=valores)

    gradientes = cinta.gradient(salida, valores)
    return tf.math.reduce_std(gradientes, axis=0).numpy()[0]

# Reducido respecto al experimento original para caber en una sesión de clase
num_circuitos = 20
repeticiones = 500
tamanos_qubits = [2, 4, 6, 8]
profundidad_fija = 20

varianzas = [varianza_del_gradiente(n, profundidad_fija, num_circuitos, repeticiones) for n in tamanos_qubits]

plt.plot(tamanos_qubits, varianzas, marker="o")
plt.xlabel("Número de qubits")
plt.ylabel("Varianza del gradiente")
plt.title("La meseta estéril: la varianza cae al añadir qubits")
plt.show()
```

Eseguendolo vedrai che la varianza del gradiente cala chiaramente all'aumentare del numero di qubit, anche con questa versione ridotta dell'esperimento. Con più qubit e più ripetizioni (come nell'esperimento originale, pubblicato dallo stesso team di TensorFlow Quantum) il calo è ancora più marcato.

## Riepilogo dei punti chiave

- TFQ collega Cirq a Keras: `tfq.layers.PQC` trasforma un circuito con simboli in un layer addestrabile, e `tfq.differentiators.ParameterShift` ne calcola i gradienti
- `tfq.layers.Expectation` calcola il valore di aspettazione esatto; `tfq.layers.SampledExpectation` lo approssima simulando un numero finito di ripetizioni, come su hardware reale
- Le plateau sterili sono un crollo esponenziale della varianza del gradiente all'aumentare di qubit o profondità nei circuiti inizializzati a caso, e fanno sì che l'ottimizzatore non sappia dove andare
- L'apprendimento a strati (addestrare poco alla volta, strato dopo strato, invece di tutto il circuito insieme fin dall'inizio) è una strategia per mitigare questo problema
- Per questo, in questo corso, i circuiti che addestrerai sono volutamente piccoli e poco profondi

## Mettilo in pratica

Nell'esempio di codice, cambia l'operatore di lettura da `sum(cirq.Z(q) for q in qubits)` a un unico operatore `cirq.Z(qubits[0]) * cirq.Z(qubits[1])` (il prodotto di Z su soli due qubit). Esegui di nuovo la scansione delle dimensioni dei qubit e confronta la nuova curva di varianza con quella originale. Annota se il calo è più rapido, più lento o uguale, e collega ciò che osservi alla raccomandazione di "scegliere con cura cosa misurare" della parte concettuale.
