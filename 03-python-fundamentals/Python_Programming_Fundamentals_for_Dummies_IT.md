# Python Programming Fundamentals for Dummies

### Manuale di guida per studiere e imparare i fondamenti di Python

Questo manuale riassume il programma del corso **EITC/CP/PPF** (European Information Technologies Certification Academy - Certified Professional - Python Programming Fundamentals), insegnato dall'European IT Certification Institute (EITCI). È stato pensato per accompagnarti durante lo studio e per servire come riferimento rapido quando pratichi con il notebook `Tic_Tac_Toe_Aprende_Python_IT.ipynb`, che crea un gioco del tris (Tic Tac Toe) passo dopo passo applicando ogni concetto.

Non è richiesta alcuna esperienza precedente. Leggi ogni lezione, prova gli esempi sul tuo computer e completa gli esercizi del notebook prima di passare alla lezione successiva.

---

## Come usare questo manuale

1. Leggi la lezione completa prima di toccare il codice
2. Apri un terminale Python o il notebook e riproduci ogni esempio
3. Annota i dubbi che sorgono nel glossario, alla fine del documento
4. Quando finisci una lezione, risolvi gli esercizi equivalenti nel notebook
5. Ripeti il processo fino a raggiungere la lezione sei

**Inizia pure e non aver paura di fare errori!** Gli errori sono il modo più rapido per imparare a programmare.

---

## Elenco di acronimi

| Acronimo | Nome completo |
|---|---|
| IDE | Integrated Development Environment (ambiente di sviluppo integrato) |
| GUI | Graphical User Interface (interfaccia utente grafica) |
| CLI | Command Line Interface (interfaccia a riga di comando) |
| POO | Programmazione orientata agli oggetti |
| PyPI | Python Package Index (indice di pacchetti Python) |
| pip | Pip Installs Packages (gestore di pacchetti Python) |
| RAM | Random Access Memory (memoria ad accesso casuale) |

---

## Indice delle lezioni

1. Introduzione a Python 3
2. Primi passi: tuple, stringhe e loop
3. Liste e il primo schema del Tic Tac Toe
4. Funzioni integrate
5. Indici e slice (fette)
6. Funzioni proprie
7. Parametri di funzione e type hints
8. Mutabilità rivisitata
9. Gestione degli errori
10. Come calcolare il vincitore in orizzontale
11. Come calcolare il vincitore in verticale
12. L'algoritmo del vincitore in diagonale
13. Iteratori e iterabili
14. Chiusura del progetto Tic Tac Toe
15. Conclusione e prossimi passi

---

## Lezione 1 · Introduzione a Python 3

Python è un linguaggio di programmazione di alto livello, creato da Guido van Rossum e pubblicato per la prima volta nel 1991. È diventato popolare per due motivi principali: è facile da leggere e serve per quasi tutto, da pagine web all'intelligenza artificiale (IA).

### Caratteristiche principali

- **Sintassi pulita**: Python usa l'indentazione (gli spazi all'inizio di ogni riga) per separare blocchi di codice, invece delle parentesi graffe `{}`. Questo costringe a scrivere codice ordinato
- **Linguaggio interpretato**: non è necessario compilare il codice. L'interprete lo esegue riga per riga, così puoi vedere il risultato all'istante
- **Tipizzazione dinamica**: non hai bisogno di dichiarare il tipo di una variabile. Python lo deduce in fase di esecuzione
- **Multiparadigma**: supporta la programmazione procedurale e la programmazione orientata agli oggetti (POO)
- **Multipiattaforma**: lo stesso codice funziona su Windows, macOS e Linux senza modifiche

### Python è un linguaggio lento?

Python è interpretato, quindi introduce un certo overhead rispetto ai linguaggi compilati come C. Tuttavia, librerie come NumPy o Pandas sono costruite su C e offrono prestazioni eccellenti per il calcolo numerico. Per la maggior parte dei progetti, la semplicità e la velocità di sviluppo compensano ampiamente questa differenza.

### Primo contatto: installazione e editor

1. Scarica Python da [python.org](https://www.python.org) (su Windows, scegli la versione a 64 bit)
2. Scegli un editor o un ambiente di sviluppo integrato (IDE): Visual Studio Code, PyCharm o lo stesso IDLE che viene fornito con Python
3. Salva i tuoi programmi con l'estensione `.py`
4. Evita di denominare i tuoi file con lo stesso nome di una libreria che stai per importare (ad esempio, non chiamare il tuo script `random.py`)

### Il tuo primo programma

```python
print("Ciao, universo")
```

La funzione `print()` mostra il testo nella console. È lo strumento più basilare per il debug (controllare cosa sta succedendo dentro il tuo codice).

### Tre pilastri per imparare a programmare

- **Sintassi**: le regole del linguaggio. Un errore di sintassi impedisce l'esecuzione del programma
- **Logica**: la capacità di scomporre un problema in passaggi piccoli e ordinati
- **Problem solving**: scegliere la struttura dati e l'algoritmo giusti per ogni situazione

---

## Lezione 2 · Primi passi: tuple, stringhe e loop

### Tuple

Una tupla è una raccolta ordinata di elementi, scritta tra parentesi `()`. È **immutabile**: una volta creata, non puoi cambiare i suoi valori.

```python
mi_tupla = (1, "mela", 3.14, True)
print(mi_tupla[0])   # 1
print(mi_tupla[1])   # mela
```

Usa le tuple per raggruppare dati che non devono cambiare, come coordinate fisse o i colori di un semaforo.

### Stringhe di testo (strings)

Una stringa è una sequenza di caratteri tra virgolette semplici `''` o doppie `""`. È anch'essa immutabile.

```python
saluto = "Ciao, mondo!"
print(saluto[0])           # C
print(saluto + " Come va?")   # concatenazione
print(saluto[0:4])         # slicing: "Ciao"
```

### Loop

I loop ripetono un blocco di istruzioni. Python offre due tipi.

**`for`**, per iterare su una sequenza nota:

```python
for lettera in "Python":
    print(lettera)
```

**`while`**, per ripetere finché si verifica una condizione:

```python
contatore = 0
while contatore < 5:
    print(contatore)
    contatore += 1
```

**Consiglio**: usa `for` quando sai quanti elementi dovrai iterare, e `while` quando la ripetizione dipenda da una condizione che cambia durante l'esecuzione (per esempio, "finché il giocatore non vince").

---

## Lezione 3 · Liste e il primo schema del Tic Tac Toe

### Liste

Una lista è una raccolta **mutabile** (modificabile) di elementi, scritta tra parentesi quadre `[]`.

```python
mi_lista = [1, 2, 3, "mela", "banana"]
mi_lista.append("pera")     # aggiunge un elemento alla fine
mi_lista.remove(2)          # elimina il valore 2
print(len(mi_lista))        # lunghezza della lista
```

Convertire una tupla in lista è semplicissimo:

```python
mi_tupla = (1, 2, 3, 4, 5)
mi_lista = list(mi_tupla)
```

### Liste di liste: rappresentare una tavola

Per rappresentare la tavola del Tic Tac Toe (una griglia 3×3), la struttura più naturale è una **lista di liste**: ogni lista interna rappresenta una riga.

```python
tavola = [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]
```

Usiamo `0` per le caselle vuote e, in seguito, `1` e `2` (o `"X"` e `"O"`) per i movimenti di ogni giocatore. Lavorare con numeri anziché caratteri facilita i confronti all'interno del codice.

### Mostrare la tavola

```python
for riga in tavola:
    for casella in riga:
        print(casella, end=" ")
    print()
```

### Vantaggi della lista di liste

- Si adatta a tavole di qualsiasi dimensione (3×3, 4×4, 5×5...)
- L'accesso a una casella è diretto: `tavola[riga][colonna]`
- Riflette visivamente la disposizione della tavola, molto utile per chi inizia
- Facilita il controllo di righe, colonne e diagonali con loop annidati

### Inizializzazione dinamica e validazione

Una versione più robusta della funzione di inizializzazione accetta la dimensione della tavola come parametro e valida i dati di input:

```python
def inizializzare_tavola(righe, colonne):
    if not isinstance(righe, int) or not isinstance(colonne, int):
        raise ValueError("Righe e colonne devono essere numeri interi")
    if righe <= 0 or colonne <= 0:
        raise ValueError("Righe e colonne devono essere positivi")
    return [[" " for _ in range(colonne)] for _ in range(righe)]
```

Questo modello —controllare i dati di input prima di usarli— evita errori difficili da tracciare successivamente.

---

## Lezione 4 · Funzioni integrate (built-in functions)

Python fornisce di serie decine di funzioni pronte all'uso, senza necessità di importare nulla.

### Funzioni matematiche

| Funzione | Cosa fa |
|---|---|
| `abs(x)` | Valore assoluto di `x` |
| `pow(x, y)` | `x` elevato a `y` |
| `max(iterable)` | Il valore massimo |
| `min(iterable)` | Il valore minimo |
| `round(x, n)` | `x` arrotondato a `n` decimali |

### Funzioni per stringhe

| Funzione | Cosa fa |
|---|---|
| `len(stringa)` | Lunghezza della stringa |
| `str(oggetto)` | Converte un oggetto in testo |
| `stringa.upper()` | Passa il testo a maiuscole |
| `stringa.lower()` | Passa il testo a minuscole |

### Altre funzioni molto utili per il progetto del Tic Tac Toe

- `range(n)`: genera una sequenza di numeri da 0 a n-1
- `enumerate(iterable)`: percorre una sequenza restituendo allo stesso tempo l'indice e il valore
- `zip(a, b)`: combina due (o più) sequenze in coppie, elemento per elemento
- `reversed(iterable)`: restituisce un iteratore che percorre la sequenza in ordine inverso
- `all(iterable)`: restituisce `True` se tutti gli elementi sono veri

```python
colori = ["rosso", "verde", "blu"]
for indice, colore in enumerate(colori):
    print(f"Il colore in posizione {indice} è {colore}")
```

---

## Lezione 5 · Indici e slice (fette)

In Python, l'indicizzazione inizia da zero. Il primo elemento di una lista o stringa è in posizione zero.

```python
numeri = [1, 2, 3, 4, 5]
numeri[0]      # 1
numeri[1]      # 2
```

### Slicing

La sintassi `[inizio:fine]` estrae una parte della sequenza. L'elemento `fine` **non** è incluso.

```python
numeri[1:4]    # [2, 3, 4]
numeri[:3]     # [1, 2, 3]  (se ometti l'inizio, si assume 0)
numeri[2:]     # [3, 4, 5]  (se ometti la fine, si assume la fine)
```

Puoi anche indicare un passo (`step`) con un secondo segno `:`.

```python
numeri[0:5:2]  # [1, 3, 5]  (salta di due in due)
```

### Indici negativi

`-1` indica l'ultimo elemento, `-2` il penultimo, e così via.

```python
numeri[-1]     # 5
numeri[-3:-1]  # [3, 4]
```

Gli indici e gli slice sono la base per percorrere righe, colonne e diagonali della tavola del Tic Tac Toe.

---

## Lezione 6 · Funzioni proprie

Una funzione è un blocco di codice con nome che esegue un compito specifico. Aiuta a dividere un problema grande in pezzi piccoli e riutilizzabili.

```python
def sommare(a, b):
    return a + b

risultato = sommare(3, 4)
print(risultato)   # 7
```

- `def` definisce la funzione
- Tra le parentesi vanno i **parametri** (i dati di input)
- `return` indica il valore di output. Se non c'è `return`, la funzione restituisce `None`

---

## Lezione 7 · Parametri di funzione e type hints

### Valori predefiniti

```python
def salutare(nome="Anonimo"):
    print(f"Ciao, {nome}!")

salutare()          # Ciao, Anonimo!
salutare("Alice")  # Ciao, Alice!
```

### Argomenti con nome (keyword arguments)

Quando una funzione ha molti parametri, puoi indicare a quale parametro corrisponde ogni valore, senza dipendere dall'ordine:

```python
def creare_giocatore(nome, simbolo="X"):
    return {"nome": nome, "simbolo": simbolo}

creare_giocatore(nome="Bea", simbolo="O")
```

Anche se Python non obbliga a dichiarare i tipi, è buona pratica documentare quale tipo di dato si aspetta ogni parametro (mediante *type hints*), per rendere il codice più facile da mantenere:

```python
def sommare(a: int, b: int) -> int:
    return a + b
```

---

## Lezione 8 · Mutabilità rivisitata

Un oggetto è **mutabile** se si può modificare dopo la creazione, **immutabile** se no.

| Mutabili | Immutabili |
|---|---|
| Liste | Stringhe |
| Dizionari | Tuple |
| Set | Numeri |

Questo dettaglio è molto importante quando due variabili puntano allo stesso oggetto:

```python
lista1 = [1, 2, 3]
lista2 = lista1        # entrambe le variabili puntano alla stessa lista
lista1.append(4)
print(lista2)           # [1, 2, 3, 4] -> è cambiata anche questa
```

Se vuoi una copia indipendente, usa `lista2 = lista1.copy()` o `lista2 = list(lista1)`.

**Perché è importante nel Tic Tac Toe**: se salvi la tavola in due variabili senza copiarla, qualunque cambiamento in una si riflette nell'altra. Tienilo presente se implementi la funzione "annulla movimento" o "salva partita".

---

## Lezione 9 · Gestione degli errori

Quando qualcosa fallisce durante l'esecuzione, Python lancia un'**eccezione**. Il blocco `try`/`except` consente di catturarla e rispondere senza che il programma si arresti.

```python
try:
    riga = int(input("Scegli una riga: "))
except ValueError:
    print("Non è un numero. Riprova")
```

### Eccezioni comuni

| Eccezione | Quando appare |
|---|---|
| `ZeroDivisionError` | Quando dividi per zero |
| `TypeError` | Quando applichi un'operazione a un tipo di dato scorretto |
| `ValueError` | Quando ricevi un valore del tipo corretto ma fuori intervallo o formato |
| `FileNotFoundError` | Quando tenti di aprire un file che non esiste |

Puoi anche definire eccezioni proprie creando una classe che eredita da `Exception`, e usare più blocchi `except` per trattare ogni errore diversamente.

**Perché è importante nel Tic Tac Toe**: quando chiedi al giocatore una riga e una colonna da tastiera, il testo inserito potrebbe non essere un numero. Il `try`/`except` evita che il gioco si chiuda di colpo di fronte a un input non valido.

---

## Lezione 10 · Come calcolare il vincitore in orizzontale

L'obiettivo è controllare se qualche riga della tavola ha le sue tre caselle uguali e occupate.

### Primo tentativo, con una bandiera di controllo

```python
def vincitore_orizzontale(tavola):
    for riga in tavola:
        senza_corrispondenza = False
        if riga[0] != "":
            for i in range(1, len(riga)):
                if riga[i] != riga[0]:
                    senza_corrispondenza = True
                    break
            if not senza_corrispondenza:
                return riga[0]
    return None
```

### Versione migliorata, con `all()`

```python
def vincitore_orizzontale(tavola):
    for riga in tavola:
        if riga[0] != "" and all(casella == riga[0] for casella in riga):
            return riga[0]
    return None
```

### Versione più concisa, con `count()`

Il metodo `count()` conta quante volte appare un valore in una lista. Se il conteggio del primo elemento coincide con la lunghezza della riga, tutte le caselle sono uguali:

```python
def vincitore_orizzontale(tavola):
    for riga in tavola:
        if riga.count("X") == len(riga):
            return "X"
        if riga.count("O") == len(riga):
            return "O"
    return None
```

**Idea chiave**: quando una prima soluzione ti sembra lunga o confusa, cerca alternative con le funzioni integrate di Python. Di solito sono più corte, più leggibili e più veloci.

---

## Lezione 11 · Come calcolare il vincitore in verticale

Controllare le colonne è equivalente a controllare le righe, ma percorrendo la tavola per indice di colonna anziché per riga.

```python
def vincitore_verticale(tavola):
    dimensione = len(tavola)
    for colonna in range(dimensione):
        valori = [tavola[riga][colonna] for riga in range(dimensione)]
        if valori[0] != "" and all(valore == valori[0] for valore in valori):
            return valori[0]
    return None
```

**Trucco liste per comprensione**: `[tavola[riga][colonna] for riga in range(dimensione)]` costruisce l'intera colonna in una sola riga, percorrendo ogni riga e prendendosi l'elemento della colonna che interessa.

Ottimizzare il codice con librerie specializzate (NumPy per il calcolo numerico, Pandas per l'analisi dei dati, o il modulo `multiprocessing` per parallelizzare i compiti) ha senso quando il progetto cresce, ma per una tavola di Tic Tac Toe l'approccio precedente è più che sufficiente.

---

## Lezione 12 · L'algoritmo del vincitore in diagonale

Ci sono due diagonali in una tavola quadrata:

- **Diagonale principale**: posizioni (0,0), (1,1), (2,2)...
- **Diagonale secondaria**: posizioni (0, n-1), (1, n-2), (2, n-3)...

```python
def vincitore_diagonale(tavola):
    dimensione = len(tavola)

    principale = [tavola[i][i] for i in range(dimensione)]
    if principale[0] != "" and all(valore == principale[0] for valore in principale):
        return principale[0]

    secondaria = [tavola[i][dimensione - 1 - i] for i in range(dimensione)]
    if secondaria[0] != "" and all(valore == secondaria[0] for valore in secondaria):
        return secondaria[0]

    return None
```

### Strumenti di Python che facilitano questo algoritmo

- **`enumerate()`**: percorre una lista restituendo indice e valore allo stesso tempo, così eviti di gestire un contatore manuale
- **`reversed()`**: restituisce un iteratore con gli elementi in ordine inverso. Se hai bisogno di indicizzare il risultato, convertilo prima in lista con `list(reversed(...))`
- **`zip()`**: combina due o più sequenze in tuple, elemento per elemento. È perfetto per percorrere, ad esempio, gli indici di riga e di colonna di una diagonale allo stesso tempo

```python
righe = range(3)
colonne_invertite = reversed(range(3))
for riga, colonna in zip(righe, colonne_invertite):
    print(riga, colonna)
```

---

## Lezione 13 · Iteratori e iterabili

- **Iterable**: qualsiasi oggetto che si può percorrere con un loop (liste, tuple, stringhe, dizionari)
- **Iteratore**: l'oggetto che effettivamente restituisce un elemento ogni volta che lo chiami con `next()`, finché non si esaurisce e lancia `StopIteration`

```python
numeri = [1, 2, 3]
iteratore = iter(numeri)
print(next(iteratore))   # 1
print(next(iteratore))   # 2
print(next(iteratore))   # 3
```

Quando scrivi `for elemento in lista:`, Python sta usando `iter()` e `next()` dietro le quinte, in modo automatico. Capire questo meccanismo aiuta a comprendere perché funzioni come `zip()`, `enumerate()` o `reversed()` restituiscono oggetti "pigri" che si convertono in liste solo quando lo chiedi esplicitamente.

---

## Lezione 14 · Chiusura del progetto Tic Tac Toe

Con tutti i concetti precedenti puoi costruire il gioco completo. I pezzi da assemblare sono:

1. **Tavola dinamica**: una lista di liste la cui dimensione può essere scelta da chi gioca
2. **Funzione `tutte_uguali(lista)`**: controlla se tutti gli elementi di una lista sono uguali e diversi da vuoto. Si riutilizza nei controlli orizzontale, verticale e diagonale, evitando di ripetere il codice
3. **Validazione di movimenti**: prima di accettare un movimento, controlla che la casella sia dentro la tavola e che sia vuota. Se non lo è, chiedi di nuovo senza perdere il turno
4. **Loop principale**: alterna i turni tra due giocatori finché c'è un vincitore o la tavola si riempie (pareggio)
5. **Messaggio di tipo di vittoria**: indica se la partita è stata vinta in orizzontale, verticale o diagonale
6. **Opzione di ripetere la partita**: al termine, chiedi se vogliono giocare di nuovo. Qualsiasi risposta diversa da "sì" o "no" deve essere gestita con un messaggio chiaro, invece di rompere il programma

### Buone pratiche nel consolidare il codice

- Evita di ripetere la stessa logica di controllo tre volte: estrai una funzione comune (`tutte_uguali`) e riutilizzala
- Dai nomi descrittivi alle variabili. Un nome come `L` o `g` complica la lettura del codice tra qualche settimana
- Aggiungi commenti solo dove il motivo di una decisione non sia evidente, non per ripetere quello che il codice già dice

---

## Lezione 15 · Conclusione e prossimi passi

Nel corso di questo manuale hai percorso i fondamenti di Python: variabili, tipi di dati, tuple, stringhe, liste, loop, funzioni, indici, gestione degli errori e iteratori. Con essi hai costruito, pezzo per pezzo, un gioco completo di Tic Tac Toe.

### Idee per continuare a fare pratica

- Converti il gioco in una classe (introduce concetti di programmazione orientata agli oggetti, POO)
- Aggiungi un'interfaccia grafica con Tkinter o Pygame
- Incorpora colori nella console con il pacchetto `colorama`
- Prova a fare in modo che il computer giochi contro di te (un primo passo verso l'intelligenza artificiale applicata ai giochi)
- Carica il tuo progetto su un repository Git per fare pratica con il controllo delle versioni

**Continua a esplorare e crea il tuo progetto!** Il modo migliore per consolidare quello che hai imparato è applicarlo a un'idea che ti motiva.

---

## Glossario rapido

- **Variabile**: nome che guarda un valore in memoria
- **Funzione**: blocco di codice riutilizzabile che riceve parametri e può restituire un risultato
- **Lista**: raccolta ordinata e mutabile di elementi
- **Tupla**: raccolta ordinata e immutabile di elementi
- **Loop**: struttura che ripete istruzioni
- **Eccezione**: errore che si verifica durante l'esecuzione del programma
- **Indice**: posizione di un elemento all'interno di una sequenza, iniziando da zero
- **Slice**: frammento estratto da una sequenza mediante `[inizio:fine:passo]`
- **Iterable**: oggetto che si può percorrere
- **Iteratore**: oggetto che fornisce gli elementi di un iterable uno alla volta

## Risorse aggiuntive

- Documentazione ufficiale: [docs.python.org](https://docs.python.org)
- Indice di pacchetti: [pypi.org](https://pypi.org)
- Comunità di domande e risposte: [stackoverflow.com](https://stackoverflow.com)

---

*Manuale elaborato a partire dal materiale didattico del programma EITC/CP/PPF Python Programming Fundamentals, dell'European IT Certification Institute (EITCI).*
