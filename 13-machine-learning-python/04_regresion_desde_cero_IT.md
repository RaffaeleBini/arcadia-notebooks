# 4. Programmare la regressione lineare da zero

## Perché questo ti riguarda

Nei due capitoli precedenti hai usato scikit-learn per addestrare una regressione lineare in un paio di righe di codice: `LinearRegression()`, `.fit()`, `.score()` e fatto. Quella libreria risolve tutto il lavoro matematico per te, e va molto bene per lavorare velocemente. Ma trasforma anche l'algoritmo in una scatola nera: gli passi dati, ricevi un modello, e non vedi cosa accade dentro.

In questo capitolo apri quella scatola. Programmerai la regressione lineare semplice con Python e NumPy puro, senza scikit-learn, per capire esattamente da dove provengono la pendenza, l'intercetta e l'R quadrato che fino a ora guardavi solo come risultati. Non è un esercizio accademico senza scopo: quando comprendi la formula, sai cosa significa davvero ogni numero che un modello restituisce, individui quando un risultato non ha senso e guadagni la fiducia necessaria per spiegare le tue decisioni a altre persone. Quella comprensione è la base di qualsiasi progetto serio di Intelligenza Artificiale (IA), inclusi quelli che combinano molte tecniche di apprendimento automatico.

Lavorerai con un unico dataset di esempio, molto piccolo, e lo porterai dall'inizio alla fine: per primo calcoli la retta a mano, poi ne misuri la qualità e, infine, verificherai che l'algoritmo si comporta come ti aspetti di fronte a dati diversi.

## La formula della migliore retta

Immagina che qualcuno ti consegni un foglio con una nuvola di punti disegnati a matita e ti chieda di tracciare, a occhio, la linea retta che meglio li attraversa. La tua mano, in modo intuitivo, cerca un equilibrio: né passare troppo vicino a alcuni punti lasciandone altri lontani, né inclinare la retta verso un lato solo perché lì ci sono più punti raggruppati. Cerchi la linea che, nel complesso, rimane più vicina a tutti i punti allo stesso tempo.

La regressione lineare automatizza esattamente quel gesto. Invece di una mano che traccia a occhio, usa una formula matematica che calcola l'unica retta che riduce al minimo la distanza totale al quadrato tra i punti reali e la retta stessa. Questo metodo si chiama minimi quadrati, e la retta risultante è conosciuta come la retta di miglior adattamento.

Ricorda l'equazione di una retta, la stessa che hai visto nella scuola secondaria:

```
y = mx + b
```

Qui, `m` è la pendenza (quanto sale o scende la retta per ogni unità che avanza `x`) e `b` è l'intercetta o il punto dove la retta incrocia l'asse verticale, quando `x` vale zero. Per tracciare la retta di miglior adattamento devi solo calcolare questi due numeri, `m` e `b`, a partire dai tuoi dati.

**La formula della pendenza (m)**, adattata dai minimi quadrati a partire dalle medie, è questa:

```
        (media(X) · media(Y)) − media(X · Y)
m  =    ─────────────────────────────────────
             media(X)² − media(X²)
```

Detto in parole semplici: moltiplichi le due medie tra loro, sottrai la media del prodotto di ogni `X` per la sua `Y` corrispondente, e dividi quel risultato per la differenza tra il quadrato della media di `X` e la media dei quadrati di `X`. Sono solo medie, moltiplicazioni, sottrazioni e una divisione, niente di algebra avanzata.

**La formula dell'intercetta (b)** è ancora più breve, e puoi calcolarla solo una volta che hai `m`:

```
b = media(Y) − m · media(X)
```

Cioè: la media di `Y`, meno la pendenza moltiplicata per la media di `X`. Con `m` e `b` in mano, hai già l'equazione completa della tua retta di miglior adattamento e puoi usarla per prevedere un valore di `y` per qualsiasi `x` che desideri.

## Programmare la pendenza e la retta in Python puro

Lavorerai con un dataset minuscolo, di solo sei punti, che ti accompagnerà durante tutto il capitolo:

```python
import numpy as np

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)
```

Nota il `dtype=np.float64`: dici a NumPy di lavorare con numeri decimali ad alta precisione, in modo che le divisioni della formula non perdano accuratezza.

Ora traduci la formula della pendenza e dell'intercetta in una singola funzione. NumPy ti permette di scrivere la formula quasi esattamente come la leggi sopra, senza cicli, perché `xs` e `ys` sono array e le operazioni si applicano a tutti i loro elementi contemporaneamente:

```python
def best_fit_slope_and_intercept(xs, ys):
    m = (
        ((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) /
        ((np.mean(xs) ** 2) - np.mean(xs ** 2))
    )
    b = np.mean(ys) - m * np.mean(xs)
    return m, b
```

Un paio di dettagli importanti della formula, perché non rimanere bloccato se la scrivi tu stesso:

- Usa l'operatore `**` (non `^`) per elevare al quadrato in Python; `^` significa un'altra cosa e dà un errore
- Posiziona le parentesi esattamente dove sono sopra: se dividi prima di sottrarre, PEMDAS (parentesi, esponenti, moltiplicazione e divisione, addizione e sottrazione) smette di valere e il risultato esce male
- `np.mean(xs) ** 2` (il quadrato della media) e `np.mean(xs ** 2)` (la media dei quadrati) sono due cose diverse: non confonderle

Ora applica la funzione al tuo dataset:

```python
m, b = best_fit_slope_and_intercept(xs, ys)
print(m, b)
# 0.42857142857142855 4.0
```

Con `m` e `b` calcolati, puoi già costruire la retta di miglior adattamento completa, punto per punto, applicando `y = mx + b` a ogni valore di `xs`:

```python
regression_line = [(m * x) + b for x in xs]
print(regression_line)
# [4.4286, 4.8571, 5.2857, 5.7143, 6.1429, 6.5714]
```

E, come qualsiasi modello addestrato, puoi usare quella stessa equazione per prevedere un valore di `y` che non è nel tuo dataset originale. Per esempio, quale valore di `y` ti aspetti se `x` valesse otto?

```python
prediccion_x8 = (m * 8) + b
print(prediccion_x8)
# 7.428571428571429
```

È tutto quello che `LinearRegression().fit()` fa dentro per una singola variabile: calcola queste due medie, risolve la formula e salva `m` e `b` per usarli poi in ogni previsione.

## Misurare quanto bene è la tua retta: l'R quadrato

Avere una retta non basta: devi sapere se quella retta rappresenta bene i tuoi dati o se è un adattamento mediocre. Per questo esiste l'R quadrato, anche detto coefficiente di determinazione, la stessa metrica che hai visto con `.score()` nel capitolo precedente. Adesso la costruirai tu.

Il punto di partenza è l'errore al quadrato. Per ogni punto del tuo dataset, l'errore è la distanza tra il valore reale di `y` e il valore che la tua retta predice per quello stesso `x`. Elevi quella distanza al quadrato per due motivi: assicuri che gli errori sopra e sotto la retta si sommino nella stessa direzione, anziché cancellarsi tra loro, e penalizzi più fortemente gli errori grandi di quelli piccoli. Sommando l'errore al quadrato di tutti i punti ottieni l'errore quadratico totale della retta.

Per sapere se quell'errore è alto o basso, hai bisogno di un punto di paragone. Il punto di paragone che usa l'R quadrato è la linea più semplice possibile: una retta orizzontale alla media di `Y`. Calcoli anche l'errore al quadrato di quella retta piatta e confronti entrambi gli errori con questa formula:

```
R² = 1 − (errore al quadrato della tua retta / errore al quadrato della media di Y)
```

Se la tua retta predice altrettanto male della semplice media di `Y`, il quoziente vale uno e l'R quadrato esce zero: il tuo modello non apporta nulla. Se la tua retta predice perfettamente, l'errore al quadrato della retta è zero, il quoziente vale zero e l'R quadrato esce uno: adattamento perfetto. Quanto più vicino a uno, tanto meglio la tua retta spiega la variazione dei dati.

Programmalo in due funzioni piccole. Prima, l'errore al quadrato, che riceve i valori reali e i valori di una linea (può essere la tua retta o la linea della media) e restituisce la somma delle differenze al quadrato:

```python
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)
```

Poi, il coefficiente di determinazione, che confronta l'errore della tua retta con l'errore della linea della media:

```python
def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [np.mean(ys_orig) for _ in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)
```

Applicalo alla retta che hai già calcolato su `xs` e `ys`:

```python
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)
# 0.5844155844155845
```

Ottieni un R quadrato di circa 0,58. Interpretato in parole: la tua retta spiega il 58% della variazione dei valori di `y`; il 42% rimanente rimane non spiegato dal modello. Non è un adattamento perfetto, ma con solo sei punti e una relazione abbastanza rumorosa, è un risultato ragionevole.

## Testare il tuo algoritmo

Hai già le due componenti dell'algoritmo: `best_fit_slope_and_intercept` e `coefficient_of_determination`. Prima di fidarti di loro, mettile alla prova con dati dove già sai quale risultato aspettarti, come faresti con un test unitario in programmazione. Per questo generi dataset sintetici dove controlli due cose: quanta variazione (rumore) hanno i punti e se esiste o no una correlazione tra `X` e `Y`.

Questa funzione crea un dataset personalizzato: `hm` è quanti punti vuoi, `variance` controlla quanto si dispersano i valori di `Y` attorno alla tendenza, `step` è quanto sale o scende `Y` in media ad ogni passo, e `correlation` può essere `'pos'` (correlazione positiva), `'neg'` (correlazione negativa) o `False` (senza correlazione):

```python
import random

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation == 'pos':
            val += step
        elif correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
```

Adesso usa questa funzione per generare tre scenari diversi e verifica, in ognuno, come risponde il tuo R quadrato:

```python
# Scenario 1: correlazione positiva chiara, poco rumore
xs1, ys1 = create_dataset(40, variance=10, step=2, correlation='pos')
m1, b1 = best_fit_slope_and_intercept(xs1, ys1)
linea1 = [(m1 * x) + b1 for x in xs1]
r2_1 = coefficient_of_determination(ys1, linea1)

# Scenario 2: correlazione positiva, ma con molto rumore
xs2, ys2 = create_dataset(40, variance=80, step=2, correlation='pos')
m2, b2 = best_fit_slope_and_intercept(xs2, ys2)
linea2 = [(m2 * x) + b2 for x in xs2]
r2_2 = coefficient_of_determination(ys2, linea2)

# Scenario 3: senza alcuna correlazione
xs3, ys3 = create_dataset(40, variance=10, step=2, correlation=False)
m3, b3 = best_fit_slope_and_intercept(xs3, ys3)
linea3 = [(m3 * x) + b3 for x in xs3]
r2_3 = coefficient_of_determination(ys3, linea3)

print("Correlazione forte, poco rumore:", r2_1)
print("Correlazione forte, molto rumore:", r2_2)
print("Senza correlazione:", r2_3)
```

Poiché `create_dataset` usa numeri casuali, i tuoi valori esatti cambieranno ogni volta che esegui il codice, ma il modello si mantiene ed è proprio quello che conferma che il tuo algoritmo funziona bene:

- Lo scenario 1 dovrebbe darti un R quadrato alto, vicino a uno: poco rumore e una tendenza chiara sono facili da adattare con una retta
- Lo scenario 2 dovrebbe darti un R quadrato più basso dell'1, anche se c'è ancora correlazione: più rumore significa più distanza tra i punti e la retta
- Lo scenario 3 dovrebbe darti un R quadrato vicino a zero: senza relazione reale tra `X` e `Y`, la tua retta non spiega quasi nulla della variazione

**Esegui il blocco completo molte volte** e osserva come cambiano i tre valori di R quadrato ad ogni esecuzione, mantenendo sempre lo stesso ordine tra loro. Se qualche volta vedi il contrario, per esempio che il dataset senza correlazione ottiene sistematicamente un R quadrato più alto di quello a correlazione forte, hai un chiaro indizio che c'è un errore nel tuo codice, non nella teoria.

## Riepilogo: cosa porti con te da questo capitolo

- La retta di miglior adattamento, `y = mx + b`, esce da due formule basate solo su medie, moltiplicazioni, sottrazioni e una divisione
- La pendenza `m` si calcola per prima; l'intercetta `b` dipende da `m` e dalle medie di `X` e `Y`
- NumPy ti permette di scrivere queste formule quasi come le leggi su carta, senza cicli manuali, perché opera su array completi
- L'R quadrato confronta l'errore al quadrato della tua retta con l'errore al quadrato della semplice media di `Y`, e va da zero (il modello non apporta nulla) a uno (adattamento perfetto)
- Generare dataset sintetici con diversa variazione e correlazione è un modo semplice di verificare che il tuo algoritmo si comporta come la teoria predice
- Adesso che conosci la scatola nera da dentro, tutto quello che scikit-learn fa con `.fit()` e `.score()` smette di essere magia e diventa aritmetica che puoi spiegare

## Mettilo in pratica

**Apri l'editor e copia il codice delle prime due sezioni di questo capitolo.** Il tuo compito è semplice e definito: cambia i valori della lista `ys` del dataset originale, per esempio a `[3, 4, 4, 6, 7, 8]`, e lascia `xs` come è.

**Esegui di nuovo tutto il flusso completo** (`best_fit_slope_and_intercept`, la costruzione di `regression_line` e `coefficient_of_determination`) e annota i tre risultati nuovi: la pendenza `m`, l'intercetta `b` e l'R quadrato.

Poi, rispondi queste tre domande con le tue parole:

- La pendenza è salita o scesa rispetto al valore originale di 0,43? Cosa ti dice questo della nuova tendenza dei tuoi dati?
- L'R quadrato è migliorato o peggiorato rispetto allo 0,58 originale? I nuovi punti sono più o meno allineati tra loro?
- Se adesso prevedi il valore di `y` per `x` uguale a otto con la tua nuova retta, il risultato ti sembra ragionevole dato il resto dei punti?

**Ripeti l'esercizio** con un terzo insieme di valori di `ys`, questa volta progettato appositamente perché la relazione con `xs` sia praticamente piatta (per esempio, `[5, 5, 5, 5, 5, 5]`), e verifica che l'R quadrato croll. Questo è lo stesso comportamento che hai già verificato con `create_dataset`, ora visto su un dataset che hai costruito tu stesso a mano.

---

**Sigle usate in questo capitolo:** IA (Intelligenza Artificiale)
