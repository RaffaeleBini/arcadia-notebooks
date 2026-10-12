# Capitolo 3 — Il tuo primo programma quantistico con Cirq

## Perché è importante

Conosci già la teoria: qubit, sovrapposizione, entanglement e porte quantistiche. Ora è il momento di programmarli davvero. Cirq è la libreria Python che Google ha pubblicato per progettare, simulare ed eseguire circuiti quantistici, ed è la base su cui si costruisce TensorFlow Quantum. Tutto ciò che farai da questo capitolo in poi (classificatori quantistici, QAOA, VQE, apprendimento per rinforzo quantistico) si appoggia agli stessi blocchi che imparerai qui: qubit, porte e circuiti.

## Spiegazione concettuale

### I qubit vivono su una griglia

In Cirq un qubit non è solo un'idea astratta: ha una posizione. La classe `cirq.GridQubit(riga, colonna)` rappresenta un qubit fisico situato su una griglia, imitando la disposizione dei qubit su un chip reale come Sycamore. Pensa alle caselle di una scacchiera: ogni pezzo (qubit) occupa una casella precisa.

### Un circuito è una ricetta di passaggi

Un `cirq.Circuit()` è la sequenza ordinata di porte che applichi ai tuoi qubit, proprio come una ricetta di cucina è una sequenza ordinata di passaggi. Aggiungi porte con `circuito.append(...)`, e Cirq le organizza automaticamente in "momenti": un momento è un gruppo di porte che vengono eseguite insieme, su qubit diversi, per sfruttare il parallelismo dell'hardware.

### Il catalogo delle porte che userai di più

- `cirq.H(qubit)`: Hadamard, crea sovrapposizione
- `cirq.X(qubit)`, `cirq.Y(qubit)`, `cirq.Z(qubit)`: rotazioni di 180 gradi attorno a ciascun asse (gli equivalenti quantistici di "ribaltare" il qubit)
- `cirq.rx(angolo)`, `cirq.ry(angolo)`, `cirq.rz(angolo)`: rotazioni di un angolo qualsiasi attorno a ciascun asse; sono quelle che userai per "addestrare" un circuito, perché l'angolo può essere un parametro regolabile
- `cirq.CNOT(controllo, bersaglio)` e `cirq.CZ(controllo, bersaglio)`: porte a due qubit che creano entanglement
- Potenze frazionarie, come `cirq.X(qubit)**0.5`: applicano solo "una parte" della rotazione. `cirq.X(qubit)**i` equivale a una rotazione di `pi * i` radianti, un modo compatto di codificare un numero reale come angolo di rotazione (lo userai per convertire dati classici in stati quantistici)

### Parametri simbolici con sympy

Cirq si appoggia alla libreria `sympy` per definire porte con un angolo che ancora non conosci, un simbolo al posto di un numero fisso. Questo è esattamente ciò che serve perché un circuito quantistico sia "addestrabile": definisci la porta con un simbolo (per esempio, `theta`) e più avanti un ottimizzatore prova diversi valori di `theta` fino a trovare quello che risolve meglio il tuo problema. Questa idea di "parametro simbolico che si regola durante l'addestramento" è l'equivalente quantistico di un peso in una rete neurale classica.

### Simulare e misurare

Senza accesso a un computer quantistico reale, puoi simulare il circuito sul tuo computer con `cirq.Simulator()`. Il metodo `.run(circuito, repetitions=N)` esegue il circuito N volte (perché ogni singola esecuzione ti dà un solo risultato, ricorda il capitolo 1) e restituisce la statistica dei risultati.

## Esempio di codice commentato

```python
import cirq
import sympy

# Dos qubits en una rejilla de una sola fila
q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)

# Un símbolo: un ángulo que todavía no tiene valor numérico
theta = sympy.Symbol("theta")

circuito = cirq.Circuit()

# Ponemos el primer qubit en superposición
circuito.append(cirq.H(q0))

# Entrelazamos los dos qubits
circuito.append(cirq.CNOT(q0, q1))

# Rotamos el segundo qubit un ángulo "theta" sobre el eje Y (parámetro entrenable)
circuito.append(cirq.ry(theta).on(q1))

# Medimos los dos qubits
circuito.append(cirq.measure(q0, q1, key="resultado"))

print(circuito)

# Para simular, sustituimos el símbolo "theta" por un valor concreto
simulador = cirq.Simulator()
resolver = cirq.ParamResolver({"theta": 0.5})
resultado = simulador.run(circuito, resolver, repetitions=500)

print(resultado.histogram(key="resultado"))
```

Nota il `cirq.ParamResolver`: è il meccanismo che sostituisce il simbolo `theta` con un numero concreto subito prima di simulare. Questo schema (circuito con simboli + resolver che assegna i valori) è esattamente quello che TensorFlow Quantum usa internamente quando addestra un modello, solo che invece di assegnare il valore a mano, è un ottimizzatore a regolarlo automaticamente.

## Riepilogo dei punti chiave

- `cirq.GridQubit` colloca un qubit in una posizione precisa di una griglia, come su un chip reale
- `cirq.Circuit()` e `.append(...)` costruiscono la sequenza di porte del circuito
- Le porte di rotazione (`rx`, `ry`, `rz`) accettano simboli di `sympy` come angolo, il che rende il circuito "addestrabile"
- Le potenze frazionarie di una porta (`cirq.X(q)**i`) codificano un numero reale come rotazione
- `cirq.Simulator()` esegue il circuito; `cirq.ParamResolver` sostituisce i simboli con valori concreti prima di simulare

## Mettilo in pratica

Costruisci un circuito a tre qubit che: metta il primo qubit in sovrapposizione con una porta Hadamard, lo entangli con il secondo tramite un `CNOT`, entangli il secondo con il terzo tramite un altro `CNOT`, e infine misuri i tre qubit. Esegui il circuito 1.000 volte e verifica nell'istogramma che compaiano solo le combinazioni "000" e "111". Questo circuito di tre qubit entangled in catena è noto come stato GHZ, ed è uno degli esempi di entanglement multiplo più citati nel calcolo quantistico.
