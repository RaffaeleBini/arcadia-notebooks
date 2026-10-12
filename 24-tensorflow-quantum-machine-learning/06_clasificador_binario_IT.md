# Capitolo 6 — La tua prima rete neurale quantistica: classificatore binario

## Perché è importante

È arrivato il momento di addestrare il tuo primo modello di machine learning quantistico dall'inizio alla fine. Costruirai un classificatore binario: un modello che impara a distinguere due categorie, l'equivalente quantistico dell'"hello world" del machine learning classico (pensa a distinguere spam da non spam, o gatti da cani, ma qui con un solo qubit). Questo esempio è volutamente semplice perché vuole insegnarti lo schema completo (dati → circuito → addestramento → previsione) che riutilizzerai in ogni capitolo rimanente del corso.

## Spiegazione concettuale

### Il problema: separare due gruppi di angoli

Genererai dati sintetici: numeri casuali vicini a 0 o vicini a 1, che userai come angolo di rotazione di un singolo qubit (ricorda la potenza frazionaria `cirq.X(qubit)**angolo` del capitolo 3). L'etichetta di ogni dato è +1 se l'angolo è vicino a 0, e -1 se è vicino a 1. Il circuito deve imparare a distinguere i due casi misurando il qubit risultante.

### Il circuito: tre rotazioni addestrabili

Il circuito di questo capitolo è il più semplice possibile: tre rotazioni (`rx`, `ry`, `rz`) su un singolo qubit, ciascuna con il proprio simbolo addestrabile. Misurando nella base X (con l'operatore `cirq.X(qubit)` come lettura), il valore di aspettazione ti dà un numero compreso tra -1 e 1: questo numero è la "previsione" del modello, confrontabile direttamente con le etichette +1/-1 dei tuoi dati.

### La funzione di perdita hinge e la sua metrica di accuratezza

Poiché le etichette sono +1 e -1 (e non 0 e 1, come in un tipico classificatore binario classico), usiamo la **perdita hinge** (`tf.keras.losses.hinge`), pensata proprio per questo formato di etichette. Dato che Keras non include di default una metrica di accuratezza compatibile con hinge, ne definiamo una personalizzata, `hinge_accuracy`, che confronta il segno della previsione con il segno dell'etichetta reale.

### Lo schema di addestramento che ripeterai per tutto il corso

1. Generi i dati e li converti in tensore con `tfq.convert_to_tensor`
2. Definisci il circuito parametrizzato e lo avvolgi in un layer `tfq.layers.PQC`
3. Compili il modello Keras con una funzione di perdita, un ottimizzatore (Adam) e una metrica
4. Addestri con `.fit(...)`, esattamente come con qualsiasi rete neurale classica

## Esempio di codice commentato

```python
import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import random
import numpy as np
import matplotlib.pyplot as plt

def generar_datos(n_entrenamiento, n_prueba):
    """Genera ángulos cercanos a 0 (etiqueta +1) o cercanos a 1 (etiqueta -1)."""
    qubit = cirq.GridQubit(0, 0)

    def lote(n):
        circuitos, etiquetas = [], []
        for _ in range(n):
            angulo = random.uniform(0, 0.1) if random.random() < 0.5 else random.uniform(0.9, 1)
            circuito = cirq.Circuit()
            circuito.append(cirq.X(qubit) ** angulo)
            circuitos.append(circuito)
            etiquetas.append(1 if angulo < 0.5 else -1)
        return tfq.convert_to_tensor(circuitos), np.array(etiquetas)

    entrenamiento, etiquetas_entrenamiento = lote(n_entrenamiento)
    prueba, etiquetas_prueba = lote(n_prueba)
    return entrenamiento, etiquetas_entrenamiento, prueba, etiquetas_prueba

def construir_circuito(qubit):
    x, y, z = sympy.symbols("rotacion_x rotacion_y rotacion_z")
    circuito = cirq.Circuit()
    circuito.append(cirq.rx(x).on(qubit))
    circuito.append(cirq.ry(y).on(qubit))
    circuito.append(cirq.rz(z).on(qubit))
    return circuito

def precision_hinge(etiqueta_real, prediccion):
    etiqueta_real = tf.squeeze(etiqueta_real) > 0.0
    prediccion = tf.squeeze(prediccion) > 0.0
    return tf.reduce_mean(tf.cast(etiqueta_real == prediccion, tf.float32))

entrenamiento, etiquetas_entrenamiento, prueba, etiquetas_prueba = generar_datos(1000, 100)

qubit = cirq.GridQubit(0, 0)
operador_lectura = [cirq.X(qubit)]
entradas = tf.keras.Input(shape=(), dtype=tf.dtypes.string)

capa_cuantica = tfq.layers.PQC(
    construir_circuito(qubit),
    operador_lectura,
    repetitions=32,
    differentiator=tfq.differentiators.ParameterShift(),
    initializer=tf.keras.initializers.Zeros,
)(entradas)

modelo = tf.keras.models.Model(inputs=entradas, outputs=capa_cuantica)
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.hinge,
    metrics=[precision_hinge],
)

historial = modelo.fit(entrenamiento, etiquetas_entrenamiento, epochs=64, batch_size=32,
                        validation_data=(prueba, etiquetas_prueba))

plt.plot(historial.history["precision_hinge"], label="Entrenamiento")
plt.plot(historial.history["val_precision_hinge"], label="Validación")
plt.legend()
plt.title("Precisión del clasificador binario cuántico")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.show()
```

Dopo le 64 epoche di addestramento, dovresti vedere l'accuratezza di validazione avvicinarsi al 100%: il circuito a un solo qubit è più che sufficiente per separare due gruppi così semplici.

## Riepilogo dei punti chiave

- Un classificatore binario quantistico trasforma ogni dato in un circuito, lo fa passare per un circuito addestrabile e misura il risultato come previsione
- Con etichette +1/-1 si usano la perdita hinge e una metrica di accuratezza su misura (`hinge_accuracy`)
- Lo schema dati → circuito parametrizzato → `PQC` → `compile` → `fit` si ripete in tutti i modelli di questo corso
- `repetitions=32` nel layer `PQC` simula che l'hardware misuri il circuito 32 volte per ogni previsione, invece di calcolare il valore esatto

## Mettilo in pratica

Cambia l'operatore di lettura da `cirq.X(qubit)` a `cirq.Z(qubit)` e riaddestra il modello da zero. Confronta la curva di accuratezza di validazione con quella dell'esempio originale. Se il modello non impara altrettanto bene, rifletti sul motivo: ricorda che i dati sono stati codificati con una rotazione sull'asse X (`cirq.X(qubit)**angolo`), quindi la base di misura più informativa non è necessariamente la stessa della base di codifica.
