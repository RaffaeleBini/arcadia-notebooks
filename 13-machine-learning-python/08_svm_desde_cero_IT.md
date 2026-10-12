# 8. Programmare una SVM da zero

## Perché questo ti riguarda

Nel capitolo precedente hai scoperto l'intuizione della macchina a vettori di supporto (SVM): un algoritmo che cerca la linea (o il piano) che separa due gruppi di punti lasciando il maggiore margine possibile da ogni lato. Hai anche visto come addestrarla in due righe con `scikit-learn`, usando la classe `SVC`.

Adesso aprirai il cofano. In questo capitolo costruisci la tua classe SVM in Python, senza dipendere da nessuna libreria di machine learning. Programmerai tre componenti: un metodo `fit` che cerca i migliori valori di `w` (il vettore di pesi) e `b` (il bias) provando e scartando opzioni, un metodo `predict` che classifica punti nuovi, e una visualizzazione con Matplotlib che disegna i punti, l'iperpiano e i margini.

A cosa serve questo se `scikit-learn` lo risolve già? Perché programmare l'algoritmo a mano ti costringe a capire cosa significa veramente ogni riga della formula che hai visto nel capitolo 7. Quando più tardi usi qualsiasi libreria, saprai cosa sta accadendo dentro e potrai diagnosticare problemi con criterio personale, non solo per prova ed errore.

Un avvertimento importante: il metodo che programmerai qui cerca `w` e `b` provando valori sempre più piccoli e controllando restrizioni, senza discesa del gradiente né risolutori di ottimizzazione esterni. È un metodo semplice, pensato per imparare, non per produzione. Nel capitolo 9 vedrai una versione più efficiente con la libreria CVXOPT (ottimizzazione convessa).

## Lo scheletro della classe SVM

Immagina che stai appendendoalcun quadro al muro. Non calcoli la posizione esatta con una formula: lo tieni, fai un passo indietro, verifichi se è storto, regoli un po', verifica di nuovo... e ripeti finché non si adatta. L'algoritmo che programmerai fa esattamente questo con il vettore `w`: prova un valore, controlla se soddisfa una condizione su tutti i punti e, se non la soddisfa, lo regola e prova ancora.

Prima di arrivare lì, definisci la struttura di base della classe. Ogni classe in Python che rappresenta un modello di solito ha questa forma: un metodo che l'inizializza, uno che l'addestra e uno che predice.

```python
import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, visualizzazione=True):
        # Salva se vuoi vedere il risultato in un grafico
        self.visualizzazione = visualizzazione
        # Un colore per ogni classe: rosso per la classe 1, blu per il -1
        self.colori = {1: 'r', -1: 'b'}

        if self.visualizzazione:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

        # Ancora non conosciamo il vettore di pesi né il bias
        self.w = None
        self.b = None

    def fit(self, dati):
        # Qui andrà l'algoritmo che cerca i migliori w e b
        pass

    def predict(self, caratteristiche):
        # Restituirà la previsione secondo il segno di (X . W + b)
        return np.sign(np.dot(np.array(caratteristiche), self.w) + self.b)

    def visualizzare(self):
        # Qui andrà il codice che disegna i punti, l'iperpiano e i margini
        pass
```

Presta attenzione agli attributi `w` e `b`. Sono il cuore di tutto il capitolo: `w` è il vettore perpendicolare all'iperpiano e `b` è il termine che lo sposta rispetto all'origine. Finché valgono `None`, la classe non sa classificare nulla. Il lavoro del metodo `fit` è riempirli con i valori corretti.

I dati di input hanno una forma concreta: un dizionario con due chiavi, `-1` e `1`, una per ogni classe, i cui valori sono liste di punti. Per esempio:

```python
dati = {
    -1: np.array([[1, 7], [2, 8], [3, 8]]),
     1: np.array([[5, 1], [6, -1], [7, 3]])
}
```

Con questo scheletro pronto, tocca completare il pezzo che manca: l'addestramento.

## Addestrare la SVM: trovare w e b passo dopo passo

Ricorda la restrizione che deve soddisfare ogni punto per essere ben classificato con margine sufficiente, quella che hai già visto nel capitolo 7:

```
yi * (w . xi + b) >= 1
```

Qui `xi` è un punto del dataset, `yi` è la sua classe (uno o meno uno) e il prodotto `w . xi + b` è la distanza (con segno) di quel punto all'iperpiano. Se la restrizione vale per tutti i punti, hai un candidato valido di `w` e `b`. Tra tutti i candidati validi, il migliore è quello che ha il `w` più piccolo in magnitudine, perché un `w` piccolo corrisponde a un margine grande.

La strategia dell'algoritmo è semplice da descrivere: parte da un `w` grande e lo riduce poco a poco, controllando ad ogni passo se la restrizione vale per tutti i punti. Non appena smette di valere, sai che il valore precedente era il limite e quello è il tuo candidato.

Per non perdere precisione né tempo, l'algoritmo usa passi di dimensione decrescente: inizia con passi grandi (10% del valore massimo delle caratteristiche), continua con passi medi (1%) e termina con passi piccoli (0,1%). È la stessa idea che useresti per serrare un bullone: per primo lo giri veloce e, quando avverti resistenza, giri lentamente per non esagerare.

Il bias `b` si esplora in modo simile, ma con passi più grandi, perché non ha bisogno di tanta precisione come `w` e affinarlo eccessivamente costa in tempo di calcolo.

Qui hai il metodo `fit` completo:

```python
class SVM:
    def __init__(self, visualizzazione=True):
        self.visualizzazione = visualizzazione
        self.colori = {1: 'r', -1: 'b'}
        if self.visualizzazione:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)
        self.w = None
        self.b = None

    def fit(self, dati):
        self.dati = dati
        # Dizionario per salvare candidati: {magnitudine di w: [w, b]}
        opt_dict = {}

        # Quattro combinazioni di segni per provare w in tutte le direzioni
        trasformazioni = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        # Cerchiamo il valore massimo e minimo tra tutte le caratteristiche
        tutti_i_valori = []
        for classe in self.dati:
            for punto in self.dati[classe]:
                for valore in punto:
                    tutti_i_valori.append(valore)

        self.max_valore = max(tutti_i_valori)
        self.min_valore = min(tutti_i_valori)
        tutti_i_valori = None  # liberiamo memoria, non ne abbiamo più bisogno

        # Passi di dimensione decrescente: grandi, medi e piccoli
        passi = [self.max_valore * 0.1,
                 self.max_valore * 0.01,
                 self.max_valore * 0.001]

        # b si esplora con passi più grandi: non ha bisogno di tanta precisione
        b_rango_multiplo = 5
        b_passo_multiplo = 5
        ultimo_ottimo = self.max_valore * 10

        for passo in passi:
            w = np.array([ultimo_ottimo, ultimo_ottimo])
            ottimizzato = False

            while not ottimizzato:
                for b in np.arange(-1 * self.max_valore * b_rango_multiplo,
                                    self.max_valore * b_rango_multiplo,
                                    passo * b_passo_multiplo):
                    for trasformazione in trasformazioni:
                        w_candidato = w * trasformazione
                        cumple_restrizione = True

                        # Controlliamo la restrizione ad ogni punto di ogni classe
                        for classe in self.dati:
                            for punto in self.dati[classe]:
                                yi = classe
                                if not yi * (np.dot(w_candidato, punto) + b) >= 1:
                                    cumple_restrizione = False
                                    break
                            if not cumple_restrizione:
                                break

                        if cumple_restrizione:
                            opt_dict[np.linalg.norm(w_candidato)] = [w_candidato, b]

                if w[0] < 0:
                    # Non ha senso continuare a ridurre w in questo passo
                    ottimizzato = True
                else:
                    w = w - passo

            # Rimaniamo con il candidato di minore magnitudine: il miglior margine
            norme = sorted([n for n in opt_dict])
            scelta = opt_dict[norme[0]]
            self.w = scelta[0]
            self.b = scelta[1]
            ultimo_ottimo = scelta[0][0] + passo * 2
```

Leggi il ciclo con calma, perché sono quattro livelli annidati:

- il ciclo esteriore percorre le dimensioni dei passi, da grande a piccolo
- per ogni passo, il ciclo `while` riduce `w` finché smette di soddisfare la condizione di arresto
- dentro, percorri i possibili valori di `b`
- e per ogni `b`, provi le quattro combinazioni di segni di `w`

Ogni combinazione che supera il controllo di tutti i punti entra in `opt_dict` insieme alla sua magnitudine. Al termine di ogni turno di passi, ordini quelle magnitudini e rimani con la più piccola: quello è il tuo miglior `w` e `b` fino al momento. Il risultato alimenta il prossimo tratto di passi, ogni volta più fine.

Questo processo è costoso perché rivede tutti i punti del dataset ad ogni combinazione. Con pochi punti, come negli esempi di questo capitolo, tarda una frazione di secondo. Con dataset grandi, diventa lento subito, e qui è dove metodi più avanzati come l'ottimizzazione convessa del capitolo 9 fanno la differenza.

## Prevedere e visualizzare l'iperpiano

Con `w` e `b` già calcolati, prevedere è la parte facile: applichi la formula che già conosci dal capitolo 7, `segno(w . x + b)`, e il segno del risultato ti dice a quale classe appartiene il punto.

```python
    def predict(self, caratteristiche):
        classificazione = np.sign(np.dot(np.array(caratteristiche), self.w) + self.b)
        if classificazione != 0 and self.visualizzazione:
            self.ax.scatter(caratteristiche[0], caratteristiche[1], s=200, marker='*',
                             c=self.colori[classificazione])
        return classificazione
```

Quando la visualizzazione è attiva, ogni punto previsto si marca nel grafico con una stella, del colore della classe assegnata. Così verifichi a colpo d'occhio se la previsione ha senso rispetto ai punti di addestramento.

Per disegnare l'iperpiano e i margini hai bisogno di una funzione ausiliaria che, data una coordinata `x`, calcoli la `y` corrispondente su una delle tre linee: il margine positivo (`w.x + b = 1`), il margine negativo (`w.x + b = -1`) o l'iperpiano stesso di decisione (`w.x + b = 0`). Risolvendo la `y` da quell'equazione ottieni la funzione `iperpiano`.

```python
    def visualizzare(self):
        # Disegna tutti i punti di addestramento, colorati per classe
        [[self.ax.scatter(punto[0], punto[1], s=100, color=self.colori[classe])
          for punto in self.dati[classe]] for classe in self.dati]

        def iperpiano(x, w, b, spostamento):
            return (-w[0] * x - b + spostamento) / w[1]

        # Un po' di margine extra ai lati del grafico
        rango_dati = (self.min_valore * 0.9, self.max_valore * 1.1)
        x_min = rango_dati[0]
        x_max = rango_dati[1]

        # Margine positivo: w.x + b = 1
        psv1 = iperpiano(x_min, self.w, self.b, 1)
        psv2 = iperpiano(x_max, self.w, self.b, 1)
        self.ax.plot([x_min, x_max], [psv1, psv2], 'k')

        # Margine negativo: w.x + b = -1
        nsv1 = iperpiano(x_min, self.w, self.b, -1)
        nsv2 = iperpiano(x_max, self.w, self.b, -1)
        self.ax.plot([x_min, x_max], [nsv1, nsv2], 'k')

        # Iperpiano di decisione: w.x + b = 0
        db1 = iperpiano(x_min, self.w, self.b, 0)
        db2 = iperpiano(x_max, self.w, self.b, 0)
        self.ax.plot([x_min, x_max], [db1, db2], 'y--')

        plt.show()
```

Hai già i tre pezzi completi. Uniscili e prova la tua SVM con il dataset di esempio:

```python
dati = {
    -1: np.array([[1, 7], [2, 8], [3, 8]]),
     1: np.array([[5, 1], [6, -1], [7, 3]])
}

svm = SVM()
svm.fit(dati)

punti_da_prevedere = [[0, 1], [1, 3], [3, 4], [3, 5],
                     [5, 5], [5, 6], [6, -5], [5, 8]]

for punto in punti_da_prevedere:
    print(punto, svm.predict(punto))

svm.visualizzare()
```

All'esecuzione di questo codice ottieni due cose: nella console, la classe (`1.0` o `-1.0`) assegnata a ogni punto nuovo; nel grafico, i punti di addestramento di ogni classe, le stelle delle previsioni e tre linee parallele, quella dell'iperpiano di decisione in giallo tratteggiato e le due dei margini in nero continuo. Verifica che i punti della classe `-1` rimangano da un lato e quelli della classe `1` dall'altro, con le stelle cadendo dal lato che corrisponde secondo la loro posizione.

## Riepilogo: cosa porti con te da questo capitolo

- La classe SVM ha bisogno di solo tre componenti: `fit` per addestrare, `predict` per classificare e una visualizzazione per interpretare il risultato
- Gli attributi `w` e `b` iniziano vuoti e `fit` è l'unica funzione responsabile di calcolarli
- L'addestramento che hai programmato non usa discesa del gradiente né librerie di ottimizzazione: prova valori di `w` sempre più piccoli e li scarta non appena qualche punto viola la restrizione `yi * (w . xi + b) >= 1`
- Tra tutti i candidati validi di `w`, l'algoritmo sceglie sempre quello di minore magnitudine, perché corrisponde al margine più ampio
- `predict` applica la stessa formula del capitolo 7, `segno(w . x + b)`, per decidere la classe di un punto nuovo
- La visualizzazione disegna l'iperpiano di decisione e i suoi due margini risolvendo la `y` dell'equazione `w.x + b = valore`, con `valore` uguale a `0`, `1` o `-1`

## Mettilo in pratica

Adesso tocca a te verificare che l'algoritmo funziona con dati diversi da quelli dell'esempio.

**Copia** la classe `SVM` completa di questo capitolo in un file nuovo. **Sostituisci** il dizionario `dati` con un altro insieme piccolo di punti di due classi, per esempio:

```python
dati = {
    -1: np.array([[2, 2], [3, 1], [3, 3]]),
     1: np.array([[8, 8], [9, 6], [7, 9]])
}
```

**Addestra** il modello con `svm.fit(dati)` e **aggiungi** due o tre punti tuoi alla lista `punti_da_prevedere`, situati chiaramente vicino a una delle due classi. **Esegui** il codice e **osserva** il grafico: verifica che l'iperpiano si situi tra i due gruppi e che le stelle delle tue previsioni cadano dal lato corretto. Se sposti un punto di addestramento molto vicino all'altro gruppo, **ripeti** la prova e nota come cambia la posizione dell'iperpiano e dei margini.

**Sigle usate in questo capitolo:** SVM (Support Vector Machine, macchina a vettori di supporto)
