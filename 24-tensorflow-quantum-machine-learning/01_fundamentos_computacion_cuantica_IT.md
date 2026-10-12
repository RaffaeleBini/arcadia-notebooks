# Capitolo 1 — Che cos'è il calcolo quantistico

## Perché è importante

Immagina di cercare l'uscita di un labirinto. Un computer classico prova un percorso, sbatte contro un muro, torna indietro e prova il successivo: uno dopo l'altro. Un computer quantistico, invece, esplora molti percorsi contemporaneamente grazie a un fenomeno chiamato sovrapposizione. Questa differenza non è un semplice trucco di velocità: apre la porta alla risoluzione di problemi di chimica, ottimizzazione e apprendimento automatico che a un computer classico richiederebbero migliaia di anni.

Google AI Quantum, il team dietro processori quantistici come Sycamore, investe da anni in questa tecnologia proprio perché promette di risolvere questo tipo di problemi. E TensorFlow Quantum, lo strumento che usi in questo corso, nasce dall'unione di quell'hardware quantistico con il machine learning che già conosci (o che conoscerai) dal mondo classico. Prima di scrivere una sola riga di codice, devi capire tre idee: il qubit, la sovrapposizione e l'entanglement (intreccio quantistico).

## Spiegazione concettuale

### Il bit classico e il qubit

Un bit classico è come un interruttore: è su 0 oppure su 1, mai su entrambi allo stesso tempo. Un qubit (bit quantistico) è come una moneta che gira in aria: mentre gira, non è né testa né croce, ma una combinazione di entrambe. Solo quando la moneta cade —quando "misuri" il qubit— si decide se il risultato è 0 o 1.

Questa combinazione si chiama **sovrapposizione**. Matematicamente, un qubit si descrive come una miscela ponderata degli stati 0 e 1, e questi pesi determinano la probabilità di ottenere ciascun risultato alla misura. Più qubit combini in sovrapposizione, più combinazioni puoi rappresentare contemporaneamente: con soli 10 qubit rappresenti già 1.024 combinazioni simultanee, un lavoro che per un computer classico richiederebbe 1.024 passi indipendenti.

### Porte quantistiche: come manipoli un qubit

Così come un computer classico usa porte logiche (AND, OR, NOT) per trasformare i bit, un computer quantistico usa **porte quantistiche** (quantum gate) per trasformare i qubit. La differenza chiave è che le porte quantistiche sono reversibili: puoi sempre annullare l'operazione. Le più comuni sono:

- **Porta Hadamard (H)**: porta un qubit in sovrapposizione perfetta (50% di probabilità di 0, 50% di probabilità di 1)
- **Porte X, Y, Z**: ruotano lo stato del qubit sui tre assi di una sfera immaginaria (la sfera di Bloch)
- **Porta CNOT**: agisce su due qubit alla volta ed è la base dell'entanglement

### Entanglement: quando due qubit smettono di essere indipendenti

Quando entangli due qubit, lo stato dell'uno resta legato allo stato dell'altro, indipendentemente dalla distanza fisica che li separa. Se misuri il primo qubit e ottieni 0, sai automaticamente quale valore uscirà dal secondo, anche se non lo hai ancora misurato. Einstein lo chiamò "azione spettrale a distanza" perché lo trovava inquietante, ma oggi è uno strumento di lavoro quotidiano nei laboratori di calcolo quantistico. L'entanglement è l'ingrediente che permette ai qubit di collaborare per risolvere un problema, invece di comportarsi come bit indipendenti.

### Misurare: il passo che cambia tutto

Finché non misuri un qubit, questo vive in sovrapposizione. Nell'istante in cui lo misuri, il qubit "collassa" su un valore concreto (0 o 1) e perde ogni informazione della sovrapposizione precedente. Questo comportamento non ha equivalenti nel calcolo classico ed è il motivo per cui programmare in quantistico richiede di pensare in modo diverso: progetti un circuito in modo che, misurando alla fine, la probabilità di ottenere la risposta corretta sia alta.

## Esempio di codice commentato

Userai Cirq, la libreria di Google per programmare circuiti quantistici (la vedrai in dettaglio nel prossimo capitolo). Ecco un primo assaggio: un qubit che mettiamo in sovrapposizione con una porta Hadamard e poi misuriamo.

```python
import cirq

# Un qubit físico situado en la posición (0, 0) de una rejilla
qubit = cirq.GridQubit(0, 0)

# Un circuito es la secuencia de operaciones que aplicas al qubit
circuito = cirq.Circuit()

# La puerta Hadamard pone el qubit en superposición: 50% 0, 50% 1
circuito.append(cirq.H(qubit))

# Medimos el qubit y guardamos el resultado con la clave "resultado"
circuito.append(cirq.measure(qubit, key="resultado"))

print(circuito)

# Simulamos el circuito 1000 veces para ver la distribución de resultados
simulador = cirq.Simulator()
resultados = simulador.run(circuito, repetitions=1000)

print(resultados.histogram(key="resultado"))
# Debería salir aproximadamente 500 veces "0" y 500 veces "1"
```

Nota un dettaglio importante: ogni volta che esegui l'intero circuito (sovrapposizione + misura), ottieni un solo 0 o un solo 1, mai entrambi insieme. Per questo ripeti l'esperimento 1.000 volte: così verifichi che, nel complesso, la moneta è uscita testa la metà delle volte e croce l'altra metà, come prevede la sovrapposizione.

## Riepilogo dei punti chiave

- Un qubit può trovarsi in sovrapposizione: una combinazione di 0 e 1 allo stesso tempo, finché non lo misuri
- Le porte quantistiche trasformano i qubit in modo reversibile; la porta Hadamard crea sovrapposizione e la porta CNOT aiuta a creare entanglement
- L'entanglement lega lo stato di più qubit tra loro, anche a distanza
- Misurare un qubit fa collassare la sua sovrapposizione su un unico valore concreto (0 o 1)
- Questi tre fenomeni (sovrapposizione, porte ed entanglement) sono la base di tutto ciò che costruirai con TensorFlow Quantum

## Mettilo in pratica

Modifica l'esempio di codice: invece di un singolo qubit con una porta Hadamard, crea un circuito con due qubit, applica una porta Hadamard al primo e poi una porta `cirq.CNOT(qubit1, qubit2)` per entangliarli. Misura entrambi i qubit ed esegui il circuito 1.000 volte. Osserva l'istogramma dei risultati: verifica che compaiano solo le combinazioni "00" e "11", mai "01" né "10". Questo è l'entanglement in azione: i due qubit coincidono sempre.
