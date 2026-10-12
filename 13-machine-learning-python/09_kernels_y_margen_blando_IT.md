# 9. Kernel e margine morbido nella SVM

## Perché ti interessa

Nel capitolo sette hai scoperto l'intuizione della macchina a vettori di supporto (SVM): un algoritmo che cerca la linea, o "strada", che separa due classi lasciando il margine più grande possibile su ciascun lato. Nel capitolo otto hai programmato quell'idea da zero, con un margine "duro" che esige che tutti i punti risultino ben classificati, senza eccezioni.

Quell'impostazione ha due limiti evidenti. Primo, molti dataset reali non si possono separare con una linea retta, per quanto bene posizioni l'iperpiano. Secondo, anche quando quasi si possono separare, esigere una separazione perfetta ti lascia in balia di un solo punto rumoroso o misurato male, che può stravolgere tutto il risultato.

Questo capitolo, l'ultimo dedicato alla SVM, risolve entrambi i problemi. Prima conoscerai i kernel, lo strumento che permette di separare dati con forme complicate senza cambiare nemmeno una riga dell'algoritmo base. Poi flessibilizzerai l'esigenza di separazione perfetta con il margine morbido, controllato da un parametro chiamato `C`. Chiudi il capitolo regolando entrambi i pezzi — kernel e `C` — con la classe `SVC`, lo stesso strumento di scikit-learn che hai già usato prima.

Pensa a un problema di classificazione con due caratteristiche, per esempio la temperatura e l'umidità di un campione, e due classi mescolate in modo circolare: una categoria al centro e l'altra che forma un anello intorno. Nessuna retta separa quelle due classi, per quanto bene la posizioni, quindi la SVM del capitolo otto fallirebbe alla radice davanti a questo dataset. E anche se trovassi un problema dove le classi quasi si potessero separare con una retta, resterebbe il rischio che un solo dato anomalo — una misurazione errata, un caso raro — impedisca al margine duro di trovare qualsiasi soluzione. I kernel risolvono il primo problema; il margine morbido, il secondo. Insieme trasformano la SVM, ancora molto rigida alla fine del capitolo otto, in un algoritmo che puoi applicare con fiducia a dati reali.

## Il trucco del kernel: separare ciò che non si può separare con una linea retta

Immagina di avere un pugno di biglie rosse che formano un cerchio, circondato da un anello blu. Guardando il tavolo dall'alto, in due dimensioni, non esiste nessuna linea retta capace di separare il cerchio rosso dall'anello blu: qualunque tratto tu faccia lascia biglie dei due colori su ciascun lato. Ma se alzi quella stessa disposizione e la guardi da un'angolazione diversa, per esempio spingendo le biglie rosse del centro un po' in alto, in tre dimensioni, all'improvviso puoi far scorrere una lamina piatta tra i due gruppi. Il problema restava lo stesso: hai solo cambiato il punto di vista.

Questo è esattamente ciò che fanno i kernel nella SVM. Un kernel è una funzione che trasforma i tuoi dati, aggiungendo nuove dimensioni, fino a uno spazio dove esiste davvero una separazione lineare. La differenza rispetto ad "aggiungere colonne a mano" — per esempio, calcolare `x3 = x1 * x2` e incorporarlo come dato aggiuntivo — è che farlo esplicitamente diventa carissimo non appena il dataset cresce in caratteristiche o vuoi salire a dimensioni molto alte. Il kernel evita quel costo: calcola il risultato che otterresti in quella dimensione superiore senza mai arrivare a costruirla. Questo si chiama il "trucco del kernel".

Perché funziona questa scorciatoia? Perché, se riguardi le formule del capitolo otto, vedrai che l'algoritmo della SVM ha bisogno solo del prodotto scalare tra coppie di vettori, mai delle loro coordinate separate. Questa operazione si chiama anche prodotto interno, o inner product in inglese: non importa il nome, prende sempre due vettori e restituisce un unico numero che misura la loro somiglianza. Un kernel è precisamente una funzione che calcola quel prodotto scalare come risulterebbe nello spazio trasformato, senza viaggiare fino lì. Non devi dominare la dimostrazione matematica completa per sfruttarlo: ti basta sapere che esiste questa porta sul retro e che scikit-learn la implementa già per te.

Un esempio aiuta a fissare l'idea. Supponi di avere due caratteristiche, `x1` e `x2`, e vuoi passare a uno spazio di secondo ordine che includa `x1²`, `x2²` e `x1 * x2`. Questo è esattamente ciò che evita il kernel polinomiale. Costruire quelle tre nuove colonne per ogni dato richiede tempo di calcolo. Moltiplicare due vettori già trasformati in quel modo, invece, equivale esattamente ad applicare la formula `K(x, x') = (1 + x · x')^p` sui valori originali, senza costruire nulla esplicitamente. Il risultato è matematicamente identico, ma molto più economico da calcolare. Con due o tre caratteristiche, il risparmio di tempo è piccolo. Con centinaia o migliaia, come succede analizzando immagini, usare il kernel invece di costruire lo spazio trasformato a mano è la differenza tra un calcolo istantaneo e uno impraticabile.

I kernel più usati sono:

- **Lineare**: non trasforma nulla, è la SVM che già conosci dai capitoli sette e otto
- **Polinomiale** (`poly`): aggiunge combinazioni di potenze delle caratteristiche originali, utile quando il confine tra classi ha forma di curva
- **RBF** (Radial Basis Function o funzione a base radiale): misura la somiglianza tra punti in base alla loro distanza, ed è capace di disegnare confini molto flessibili; è il kernel predefinito in scikit-learn
- **Sigmoide**: ispirato alle reti neurali, utile in alcuni problemi di classificazione binaria

Verificalo con un dataset che, per costruzione, non ammette nessuna linea retta separatrice: due anelli concentrici, generati con `make_circles` di scikit-learn.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

# Generas un dataset con dos anillos concéntricos: imposible de
# separar con una línea recta
X, y = make_circles(n_samples=200, factor=0.4, noise=0.08, random_state=42)

# Un kernel lineal intenta trazar una recta y falla
svm_lineal = SVC(kernel='linear')
svm_lineal.fit(X, y)
print("Precisión con kernel lineal:", svm_lineal.score(X, y))

# Un kernel RBF "levanta" los datos a otra dimensión y sí los separa
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X, y)
print("Precisión con kernel RBF:", svm_rbf.score(X, y))
```

Esegui questo codice e confronta le due precisioni. Il kernel lineare resta vicino al 50%, lo stesso che azzeccare a caso tra due classi, perché nessuna retta separa un cerchio interno da un anello esterno. Il kernel RBF, invece, si avvicina al 100%, perché trova un confine curvo che circonda il cerchio interno.

Puoi visualizzare quel confine curvo direttamente:

```python
import numpy as np

# Creamos una rejilla de puntos que cubre todo el gráfico
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                      np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))
Z = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("SVM con kernel RBF sobre dos anillos concéntricos")
plt.show()
```

Eseguendolo vedrai un confine curvo, quasi circolare, che separa l'anello interno da quello esterno; qualcosa che il kernel lineare non potrebbe mai disegnare. Se vuoi verificarlo con un altro tipo di dati non lineari, prova `make_moons` invece di `make_circles` e ripeti l'esperimento con `kernel='poly'`: vedrai che trova anch'esso un confine curvo, anche se con una forma diversa da quella dell'RBF.

## Il margine morbido: cosa fare quando i dati non sono perfettamente separabili

Pensa a chi corregge esami con una regola di superamento molto rigida: chiunque resti a un solo punto dalla soglia, viene respinto, senza eccezioni. Quella regola è semplice, ma fragile: un solo esame valutato male per errore, o una brutta giornata di chi lo sostiene, stravolge tutta la classificazione tra chi passa e chi no. Chi corregge in modo più flessibile, invece, accetta che qualche caso limite entri nella categoria "superato" anche se non soddisfa la regola al cento per cento, perché sa che esigere una separazione perfetta può essere peggio che accettare poche eccezioni ragionevoli.

Il margine duro che hai programmato nel capitolo otto è come quella prima regola: esige che il vincolo `yi * (w · xi + b) >= 1` sia soddisfatto per tutti i punti, senza eccezione. Se un solo punto lo viola, l'algoritmo non trova nemmeno una soluzione. Il margine morbido introduce una variabile di tolleranza (o "slack") per ogni punto del dataset, che misura quanto gli è permesso violare il vincolo. La condizione diventa:

```
yi * (w . xi + b) >= 1 - slack_i,  con slack_i >= 0
```

Quando `slack_i` vale zero, quel punto soddisfa la condizione rigida di sempre. Più grande è `slack_i`, più gli è permesso invadere il margine o persino cadere dal lato sbagliato dell'iperpiano. L'obiettivo dell'ottimizzazione non è più solo minimizzare la grandezza di `w` (massimizzare il margine), ma anche ridurre la somma di tutte le tolleranze, pesata da un nuovo parametro: `C`.

`C` è esattamente la manopola che decide quanto rigida è quella correzione. Un valore alto di `C` penalizza molto le violazioni e fa sì che l'algoritmo si comporti quasi come un margine duro: dà priorità a classificare bene ogni punto di addestramento, anche se questo implica una strada stretta e un iperpiano molto sensibile a casi specifici, con più rischio di overfitting. Con un valore basso di `C`, l'algoritmo tollera più violazioni in cambio di un margine ampio e un confine più fluido, che generalizza meglio su dati nuovi ma può classificare male qualche punto di addestramento lungo il percorso.

Non esiste un valore di `C` universalmente corretto: dipende da quanto rumore ha il tuo dataset e da se preferisci rischiare l'overfitting o l'underfitting. Il modo abituale per deciderlo è provare diversi valori e confrontare il risultato su dati di test, cosa che farai nella prossima sezione.

Esiste inoltre un segnale indiretto molto pratico per sapere se la tua scelta di `C` (o di kernel) va nella direzione giusta: il numero di vettori di supporto che finisce per usare il modello addestrato. Se quel numero si avvicina al totale dei campioni di addestramento, è un indizio che il modello sta memorizzando casi specifici invece di imparare un pattern generale, sintomo tipico di overfitting. Come riferimento orientativo, se i vettori di supporto superano il 10% dei campioni totali, conviene sospettare overfitting e provare con un `C` minore o con un altro kernel. In scikit-learn, puoi consultare quel numero dopo l'addestramento con l'attributo `support_vectors_` del tuo modello `SVC`.

## Risolvere l'ottimizzazione con CVXOPT

Nel capitolo otto hai risolto l'ottimizzazione della SVM a mano, con un metodo semplice di prova e scarto di valori di `w`. Quel metodo funziona per imparare, ma non è quello usato nella pratica: la vera sfida di addestrare una SVM con margine morbido e kernel è un problema di programmazione quadratica (QP, dall'inglese Quadratic Programming), ed esistono librerie specializzate per risolverlo in modo molto più efficiente.

Una di queste librerie è CVXOPT (Convex Optimization, ottimizzazione convessa), pensata in Python per risolvere problemi convessi, tra cui quelli di programmazione quadratica. Non serve che tu implementi da solo la SVM con kernel completa: hai già visto l'idea di ottimizzazione nel capitolo precedente e qui basta capire che ruolo svolge CVXOPT e come si usa a un livello base.

CVXOPT non è preinstallata in Google Colab, quindi nel notebook di questa lezione la installi con un semplice comando all'inizio, prima di importarla:

```python
!pip install cvxopt
```

Il cuore di CVXOPT per questo compito è la funzione `cvxopt.solvers.qp()`, che risolve un problema con questa forma generale:

```
minimizar   (1/2) * x^T * P * x + q^T * x
sujeto a    G * x <= h
            A * x = b
```

Qui `x` è il vettore di incognite che vuoi trovare (nel caso della SVM, i moltiplicatori di Lagrange associati a ogni punto), `P` e `q` descrivono la funzione che minimizzi, e `G`, `h`, `A`, `b` definiscono i vincoli, incluso quello che limita ogni moltiplicatore secondo il parametro `C` del margine morbido. Un esempio minimo, solo per vedere la meccanica della funzione, risolve un problema quadratico semplice con due variabili:

```python
import numpy as np
from cvxopt import matrix, solvers

# Problema de ejemplo: minimizar 2x1^2 + x2^2 + x1*x2,
# sujeto a x1 >= 0, x2 >= 0, x1 + x2 >= 1
P = matrix(np.array([[4.0, 1.0], [1.0, 2.0]]))
q = matrix(np.array([0.0, 0.0]))
G = matrix(np.array([[-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]]))
h = matrix(np.array([0.0, 0.0, -1.0]))

solvers.options['show_progress'] = False
solucion = solvers.qp(P, q, G, h)
print("Solución x:", np.array(solucion['x']).flatten())
```

Eseguendolo, `solucion['x']` ti dà i valori ottimi di `x1` e `x2`. Nel caso di una SVM con kernel, `P` si costruisce a partire dalla matrice dei kernel (il risultato di applicare `K(xi, xj)` a ogni coppia di punti del dataset), e le righe di `G` e `h` incorporano sia la condizione `alpha_i >= 0` del margine duro sia il limite superiore `alpha_i <= C` introdotto dal margine morbido. Una volta risolto il problema, i moltiplicatori diversi da zero indicano quali sono i vettori di supporto, esattamente lo stesso concetto che già conosci dal capitolo sette.

Di fatto, se programmassi una tua classe SVM basata su CVXOPT, come continuazione naturale del capitolo otto, il parametro `C` apparirebbe nel metodo `__init__` di quella classe: passare `C=None` attiverebbe un margine duro identico a quello che hai già programmato, mentre assegnargli un qualsiasi numero attiverebbe il margine morbido descritto nella sezione precedente, limitando l'intervallo che può assumere ogni moltiplicatore di Lagrange nella matrice `G` del problema di programmazione quadratica. Il metodo `fit` di quella classe preparerebbe le matrici `P`, `q`, `G`, `h`, `A` e `b`, chiamerebbe `solvers.qp()` ed estrarrebbe dal risultato i moltiplicatori di Lagrange, i vettori di supporto, il bias e la funzione di proiezione necessaria per classificare nuovi punti.

Nella pratica, quando addestri una SVM con `SVC` di scikit-learn, non vedi nulla di tutto questo: la libreria risolve questo problema di programmazione quadratica internamente, con risolutori ancora più specializzati di CVXOPT, come quelli implementati dalla libreria LIBSVM su cui si appoggia scikit-learn. Conoscere questo pezzo ti serve per capire cosa succede davvero quando chiami `fit()`, anche se non avrai mai bisogno di scrivere una riga di CVXOPT in un progetto reale.

## Regolare i parametri della SVM in scikit-learn

Con la teoria di kernel e margine morbido già trattata, è il momento di metterla in pratica con i parametri di `SVC`: `kernel`, `C` e `gamma`. Ricorda il dataset del cancro al seno che hai già usato nei capitoli cinque e sette per classificare tumori come benigni o maligni.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

datos = load_breast_cancer()
X = datos.data
y = datos.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Un primo approccio, prima di automatizzare la ricerca, è un semplice ciclo che scorre vari valori di `C` e di `kernel` e annota la precisione di ogni combinazione:

```python
kernels = ['linear', 'rbf', 'poly']
valores_de_c = [0.1, 1, 10, 100]

for kernel in kernels:
    for c in valores_de_c:
        modelo = SVC(kernel=kernel, C=c)
        modelo.fit(X_train, y_train)
        precision = modelo.score(X_test, y_test)
        print(f"kernel={kernel:6s}  C={c:<6}  precisión={precision:.4f}")
```

Nota che, oltre a `kernel` e `C`, `SVC` accetta il parametro `gamma`, che ha effetto solo con i kernel `rbf`, `poly` e `sigmoid`. `gamma` controlla quanta influenza ha ogni punto di addestramento sul confine di decisione. Con un valore basso, quell'influenza arriva lontano e il confine di decisione resta fluido. Con un valore alto, invece, l'influenza si concentra vicino a ogni punto e il confine si adatta molto al dettaglio dei dati, con più rischio di overfitting. Per impostazione predefinita, scikit-learn calcola un valore ragionevole di `gamma` a partire dal numero di caratteristiche (l'opzione `'scale'`), quindi non sempre devi toccarlo a mano; conviene esplorarlo solo quando hai già fissato un kernel non lineare e vuoi affinare il risultato.

Scorrendo l'output del ciclo precedente, noterai un pattern abituale in questo dataset. Con `kernel='linear'`, la precisione cambia appena tra diversi valori di `C`, perché i dati sono già quasi separabili con una retta. Con `kernel='rbf'` o `kernel='poly'`, invece, il risultato varia in modo notevole secondo `C`, perché il confine curvo ha molto più margine per adattarsi — o fare overfitting — ai dati di addestramento. Quella differenza di comportamento è esattamente quella che hai colto nella teoria delle sezioni precedenti: quante più dimensioni e più libertà ha il confine, più conta vigilare l'equilibrio tra margine e violazioni.

Quando vuoi esplorare diverse combinazioni di parametri in modo più ordinato, senza annidare cicli a mano, `GridSearchCV` fa il lavoro per te, provando ogni combinazione con validazione incrociata e trattenendo la migliore:

```python
from sklearn.model_selection import GridSearchCV

rejilla_parametros = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.001]
}

busqueda = GridSearchCV(SVC(), rejilla_parametros, cv=5)
busqueda.fit(X_train, y_train)

print("Mejores parámetros:", busqueda.best_params_)
print("Mejor precisión en validación cruzada:", busqueda.best_score_)
print("Precisión sobre el conjunto de prueba:", busqueda.score(X_test, y_test))
```

`GridSearchCV` addestra una SVM per ogni combinazione della griglia, valutandola con cinque partizioni di validazione incrociata (`cv=5`), e salva in `best_params_` quella che ottiene la prestazione media migliore. È il modo abituale per decidere, con dati e non a occhio, quale kernel e quale valore di `C` convengono a un problema specifico.

Una volta che `GridSearchCV` ti dà la sua miglior combinazione, applica la verifica di overfitting che hai visto nella sezione sul margine morbido, ora sul modello vincitore:

```python
mejor_modelo = busqueda.best_estimator_
n_vectores_soporte = len(mejor_modelo.support_vectors_)
n_muestras = len(X_train)

print("Vectores de soporte:", n_vectores_soporte, "de", n_muestras)
print("Porcentaje:", round(100 * n_vectores_soporte / n_muestras, 1), "%")
```

Se quella percentuale schizza sopra il 10%, sai già cosa fare: torna alla griglia, aggiungi valori di `C` più bassi e verifica se la precisione sul set di test si mantiene mentre il modello diventa più semplice.

Infine, tieni presente che `SVC` è, in origine, un classificatore binario: separa solo due classi alla volta. Il dataset del cancro al seno ha esattamente due classi (benigno e maligno), quindi non lo noti. Ma se lavori con tre o più categorie, `SVC` ricorre internamente alla strategia uno-contro-uno (OvO, dall'inglese One versus One): addestra un classificatore binario per ogni coppia di classi e combina i loro voti. Puoi cambiarlo in uno-contro-il-resto (OvR, dall'inglese One versus Rest) con il parametro `decision_function_shape='ovr'` se il tuo problema ha molte categorie e preferisci addestrare meno classificatori.

## Riepilogo: cosa porti a casa da questo capitolo

- Un kernel trasforma i dati in uno spazio con più dimensioni dove esiste davvero una separazione lineare, senza bisogno di costruirle in modo esplicito; questo si chiama il trucco del kernel
- I kernel più comuni in `SVC` sono `linear`, `poly`, `rbf` (quello che usa scikit-learn per impostazione predefinita) e `sigmoid`, ciascuno adatto a forme diverse di confine
- Il margine morbido permette che alcuni punti violino quella condizione o risultino mal classificati, tramite variabili di tolleranza, invece di esigere una separazione perfetta come il margine duro del capitolo otto
- Il parametro `C` controlla questo equilibrio: più alto è, più penalizza le violazioni e più si restringe la strada di separazione; più basso è, più le tollera e più ampia la lascia
- CVXOPT è la libreria che risolve il problema di programmazione quadratica sottostante all'addestramento di una SVM con margine morbido e kernel; in Google Colab si installa separatamente con `!pip install cvxopt`
- In `SVC`, regoli kernel, `C` e `gamma` con un ciclo semplice o con `GridSearchCV`, confrontando la precisione sui dati di test per scegliere la migliore combinazione

## Mettilo in pratica

Riprendi il dataset dei due anelli concentrici della sezione sui kernel.

**Genera** il dataset con `make_moons(n_samples=200, noise=0.15, random_state=42)` invece di `make_circles`. **Addestra** due modelli `SVC` distinti su quegli stessi dati: uno con `kernel='poly'` e uno con `kernel='rbf'`, lasciando gli altri parametri predefiniti. **Calcola** la precisione di ciascuno con `score()` e **disegna** il confine di decisione di entrambi, riutilizzando il codice della griglia (`meshgrid`) di questo capitolo. **Confronta** i due confini a colpo d'occhio: osserva se uno si adatta più alla forma di "mezzaluna" del dataset rispetto all'altro, e annota quale kernel ti convince più per questo caso specifico prima di passare al prossimo tema del corso.

**Sigle usate in questo capitolo:** SVM (Support Vector Machine, macchina a vettori di supporto), RBF (Radial Basis Function, funzione a base radiale), QP (Quadratic Programming, programmazione quadratica), CVXOPT (Convex Optimization, ottimizzazione convessa)
