# 7. Fondamenti delle macchine a vettori di supporto (SVM)

## Perché questo ti riguarda

Nei capitoli cinque e sei hai scoperto la classificazione con K vicini più prossimi (KNN, dall'inglese K-Nearest Neighbors): un algoritmo intuitivo che decide la categoria di un dato nuovo guardando i suoi vicini. Funziona bene, è facile da spiegare e, precisamente per questo, di solito è il primo modello che si prova in un progetto di classificazione.

Ma KNN ha limiti. Ha bisogno di salvare tutti i dati di addestramento, diventa lento con insiemi grandi ed è sensibile al rumore quando il valore di K non si adatta bene al problema. In questo capitolo inizi un blocco nuovo dedicato a un'altra famiglia di algoritmi di classificazione: le macchine a vettori di supporto (SVM, dall'inglese Support Vector Machine).

Una SVM affronta lo stesso problema - separare i dati in categorie - da un angolo diverso: invece di votare tra i vicini, cerca il confine che meglio separa le classi e lo mantiene il più lontano possibile dai dati da ogni lato. Quella idea, apparentemente semplice, la rende uno dei dispositivi più robusti e precisi dell'intelligenza artificiale (IA) applicata, con applicazioni in diagnosi medica, riconoscimento di immagini, bioinformatica e classificazione di testi.

Questo capitolo si concentra sull'intuizione geometrica della SVM: cosa è un vettore, cosa è un iperpiano separatore e perché massimizzare il margine tra le classi è la chiave dell'algoritmo. La applicherai anche con scikit-learn sullo stesso dataset medico del capitolo cinque, così confronti direttamente i suoi risultati con quelli di KNN. Nel capitolo otto programmerai una SVM da zero, e nel capitolo nove approfondirai i kernel e il margine blando con la libreria di ottimizzazione CVXOPT. Qui, metti via il calcolatore di formule complesse: è il momento di capire il disegno prima dell'equazione completa.

Questo algoritmo si usa oggi in campi così diversi come il riconoscimento di immagini, la classificazione di testi, la bioinformatica e la finanza. Si adatta specialmente bene ai problemi con molte caratteristiche per campione - come i trenta valori numerici del dataset medico che riprenderai in questo capitolo - precisamente il terreno dove altri algoritmi iniziano a perdere precisione o velocità.

## Cos'è un vettore

Prima di parlare di SVM, devi padroneggiare il suo ingrediente di base: il vettore. Un vettore è una grandezza che ha, allo stesso tempo, lunghezza e direzione. Di solito si rappresenta come una freccia: il punto dove nasce indica l'origine, e la punta indica verso dove punta e quanto si estende.

Pensa a un'istruzione di escursionismo: "cammina tre passi a nord". Questa frase descrive un vettore completo. La direzione è "nord" e la lunghezza è "tre passi". Se invece cammini cinque passi a est, ottieni un vettore diverso, con un'altra direzione e un'altra lunghezza. Due vettori sono uguali solo quando coincidono in entrambe le cose contemporaneamente.

In matematica, un vettore si rappresenta come una lista di numeri, uno per ogni dimensione dello spazio in cui lavori. In un piano di due dimensioni, un vettore `a` con valori 3 e 4 si scrive come (3, 4): avanzi tre unità su un asse e quattro sull'altro. In una SVM, ogni dato del tuo insieme - ogni tumore, ogni email, ogni immagine - diventa un vettore: ogni caratteristica misurata occupa una dimensione, e l'insieme di quei valori definisce un punto in uno spazio con tante dimensioni quante caratteristiche hai.

Non tutti i vettori rappresentano dati. Più avanti conoscerai il vettore normale, che non descrive nessun campione concreto, ma piuttosto l'orientamento dello stesso iperpiano separatore: punta nella direzione perpendicolare al confine, come una bussola conficcata in mezzo alla strada che segna verso dove guarda. Con quel dettaglio in mente, puoi già distinguere i due ruoli che i vettori compiono dentro una SVM: alcuni rappresentano dati, e un altro definisce il confine che li separa.

### La magnitudine di un vettore

La magnitudine di un vettore è la sua lunghezza, e si calcola allo stesso modo della distanza euclidea che hai già usato con KNN: elevi al quadrato ogni componente, sommi i quadrati e estrai la radice quadrata del risultato. Per il vettore (3, 4), la magnitudine è la radice quadrata di (3² + 4²), cioè la radice quadrata di 25, che dà 5.

```python
from math import sqrt

vettore_a = (3, 4)

magnitudine = sqrt(vettore_a[0] ** 2 + vettore_a[1] ** 2)

print("Magnitudine del vettore a:", magnitudine)
# Magnitudine del vettore a: 5.0
```

Nota che questo calcolo è un'applicazione diretta del teorema di Pitagora: il vettore agisce come l'ipotenusa di un triangolo rettangolo i cui cateti sono le sue componenti.

### Il prodotto punto tra due vettori

Oltre ad avere una propria magnitudine, due vettori possono combinarsi per mezzo del prodotto punto (anche detto prodotto scalare): moltiplichi ogni componente di un vettore per la componente corrispondente dell'altro e sommi i risultati. Se il vettore `a` vale (1, 3) e il vettore `b` vale (2, 4), il prodotto punto è (1×2) + (3×4), che dà 10.

```python
vettore_a = (1, 3)
vettore_b = (2, 4)

prodotto_punto = vettore_a[0] * vettore_b[0] + vettore_a[1] * vettore_b[1]

print("Prodotto punto tra a e b:", prodotto_punto)
# Prodotto punto tra a e b: 10
```

**Salva questa operazione nella memoria**, perché il prodotto punto è il pezzo che consente a una SVM di decidere da quale lato del confine cade un dato nuovo, come vedrai nella sezione successiva.

## L'idea di SVM: separare le classi con la strada più larga possibile

Immagina una mappa con due quartieri che vuoi separare per mezzo di una strada rettilinea. Potresti tracciare quella strada in molti modi diversi, sempre che rimanga tra i due quartieri. Ma c'è un'opzione migliore di tutte le altre: la strada più larga possibile, quella che lascia il massimo spazio libero da entrambi i lati prima di toccare la prima casa di ogni quartiere.

Quella è esattamente la logica di una SVM. La linea centrale della strada è l'iperpiano separatore: il confine che divide lo spazio in due regioni, una per classe. In un piano di due dimensioni, l'iperpiano è una linea retta; in tre dimensioni, è un piano; e in dimensioni superiori - come le trenta caratteristiche del dataset medico che userai in questo capitolo - continua a chiamarsi iperpiano anche se non puoi più disegnarlo.

Le case più prossime ai bordi della strada, una da ogni lato, sono quelle che definiscono la sua larghezza. In una SVM, quei dati ricevono il nome di vettori di supporto: sono i punti di addestramento più vicini al confine, e sono gli unici che veramente importano per calcolare dove va l'iperpiano. Il resto delle case, più lontane verso l'interno di ogni quartiere, potrebbero muoversi senza alterare il tracciato della strada.

La larghezza completa della strada, misurata da marciapiede a marciapiede, è il margine. L'obiettivo di una SVM è trovare l'iperpiano che massimizza quel margine. Quanto più larga sia la strada, più spazio di sicurezza hai prima di sbagliare a classificare un dato nuovo, e meglio generalizza il modello di fronte a dati che non ha visto durante l'addestramento. Un margine stretto, in cambio, lascia il modello appiccicato ai dati di addestramento e lo rende più fragile di fronte al rumore.

### Perché ti conviene un margine ampio

Pensa di nuovo alla strada tra i due quartieri. Se la traccia molto vicino alle case di un lato, qualsiasi casa nuova che si costruisca un po' più dentro l'altro quartiere verrà male classificata, perché il confine non ha lasciato margine di manovra. Se invece la strada rimane ben centrata e larga, hai spazio sufficiente per ammettere piccole variazioni senza che il confine fallisca.

Con dati reali accade lo stesso: un margine ampio tollera meglio il rumore e i valori anomali, perché il confine non dipende da una manciata di casi estremi. Quella tolleranza è il motivo per cui una SVM addestrata con il criterio del margine massimo solitamente generalizza meglio di fronte a dati nuovi rispetto a un confine tracciato senza quel criterio, sia a mano che per mezzo di un algoritmo diverso.

### Vantaggi rispetto ad altri classificatori

Prima di passare alla parte di ottimizzazione, conviene che sappia perché questa famiglia di algoritmi gode di così buona reputazione. Tra i suoi punti forti spiccano:

- Rende bene negli spazi con molte caratteristiche, anche quando il numero di caratteristiche supera il numero di campioni
- Lo stesso criterio del margine agisce come forma di controllo contro il sovraddattamento, poiché non si lascia trascinare dal rumore dei dati di addestramento
- Ha solo bisogno di salvare i vettori di supporto per classificare dati nuovi, così è più efficiente in memoria di un algoritmo che deve ricordare tutto l'insieme di addestramento, come fa KNN

Una volta tracciata la strada, classificare un dato nuovo è semplice: guardi solo in quale marciapiede cade. Matematicamente, l'iperpiano si descrive con l'equazione:

```
w · x + b = 0
```

Qui, `x` è il vettore che rappresenta il dato, `w` è il vettore normale all'iperpiano (perpendicolare alla strada, segnando la sua orientazione) e `b` è il termine di bias, che sposta la strada verso un lato o l'altro senza cambiare il suo angolo. Per classificare un punto nuovo, calcoli il prodotto punto tra `w` e `x`, gli sommi `b`, e osservi il segno del risultato: se è positivo o zero, il punto cade da un lato della strada; se è negativo, cade dall'altro.

Vediamolo con un esempio semplice e inventato, solo per fissare l'idea, con un vettore `w` uguale a (1, 1) e un bias `b` uguale a -4:

```python
def classificare(punto, w, b):
    prodotto = punto[0] * w[0] + punto[1] * w[1] + b
    if prodotto >= 0:
        return "positiva", prodotto
    return "negativa", prodotto

w = (1, 1)
b = -4

for punto in [(3, 3), (1, 1)]:
    classe, valore = classificare(punto, w, b)
    print(f"Punto {punto} -> valore {valore} -> classe {classe}")

# Punto (3, 3) -> valore 2 -> classe positiva
# Punto (1, 1) -> valore -2 -> classe negativa
```

Il punto (3, 3) dà un risultato positivo e cade da un lato del confine; il punto (1, 1) dà un risultato negativo e cade dall'altro. Così funziona, in essenza, la previsione di qualsiasi SVM una volta addestrata: il lavoro pesante avviene prima, quando l'algoritmo trova i valori ottimali di `w` e `b`.

## Perché trovare quel iperpiano è un problema di ottimizzazione

Tracciare la strada più larga suona semplice quando la disegni a mano su una mappa con pochi punti, ma trovarla in modo esatto, con decine di caratteristiche e migliaia di dati, richiede matematica. Una SVM converte quella ricerca in un problema di ottimizzazione: trovare i valori di `w` e `b` che massimizzano il margine senza lasciare nessun dato di addestramento male classificato.

Qui appare un risultato utile: la larghezza del margine è inversamente proporzionale alla magnitudine del vettore `w`. Quanto minore sia quella magnitudine, più larga è la strada. Massimizzare il margine equivale, quindi, a minimizzare la magnitudine di `w`, sempre rispettando la condizione che ogni dato di addestramento resti correttamente classificato e fuori dal margine.

Questo tipo di problema - minimizzare una funzione soggetta a restrizioni - si conosce come programmazione quadratica, e appartiene a una categoria speciale chiamata ottimizzazione convessa. La parola "convessa" descrive la forma del terreno di ricerca: immagina una ciotola. Se lasci cadere una palla in qualsiasi punto della sua superficie, sempre rotola verso lo stesso fondo, senza rimanere bloccata in buche intermedie. Questo è proprio quello che accade con il problema della SVM: per la sua natura convessa, esiste un unico minimo globale, così qualsiasi algoritmo che "rotoli verso il fondo della ciotola" con sufficiente cura finisce per trovare la migliore soluzione possibile, senza il rischio di rimanere intrappolato in una soluzione mediocre.

Risolvere questo problema con precisione implica strumenti come i moltiplicatori di Lagrange e tecniche di ottimizzazione quadratica, che rimangono fuori dal campo di questo capitolo. Nel capitolo otto programmerai tu stesso l'algoritmo di addestramento passo dopo passo, e nel capitolo nove userai la libreria CVXOPT per risolvere il problema in modo efficiente, oltre a introdurre il margine blando - che permette alcune classificazioni errate per mezzo di un parametro chiamato `C` - e i kernel, che permettono di separare dati che non sono separabili con una linea retta. Per ora, mantieni l'idea centrale: addestrare una SVM consiste nel risolvere un problema di ottimizzazione convessa che garantisce trovare la migliore strada possibile, non una strada qualsiasi.

In pratica, calcolare la soluzione esatta con migliaia di dati sarebbe troppo lento se si risolvesse in modo diretto, così scikit-learn ricorre internamente ad algoritmi specializzati, come l'ottimizzazione minima sequenziale, che dividono il problema grande in molti problemi piccoli e li risolvono uno per uno finché convergono. Tu non hai bisogno di programmare quella parte per usare `SVC`: la libreria già la incorpora, e solo la ricostituirai a fini didattici nel capitolo otto.

## Applicare SVM con scikit-learn

Anche se la matematica completa aspetta al capitolo nove, puoi già addestrare e valutare una SVM oggi stesso, grazie alla classe `SVC` (Support Vector Classification) di scikit-learn. Ripeterai lo stesso esperimento del capitolo cinque - classificare i tumori come benigni o maligni - per confrontare direttamente il risultato di SVM con quello che hai ottenuto allora con KNN.

### Primo passo: caricare i dati e dividerli

Riusi il dataset medico già incluso in scikit-learn, con la stessa divisione tra addestramento e test che hai usato con KNN, in modo che il confronto sia equo:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dati = load_breast_cancer()
X = dati.data
y = dati.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Campioni di addestramento:", len(X_train))
print("Campioni di test:", len(X_test))
```

Usare lo stesso `random_state` del capitolo cinque garantisce che entrambi gli algoritmi vedono esattamente lo stesso ripartizione di dati, così qualsiasi differenza nella precisione finale proviene dall'algoritmo, non dalla divisione casuale.

### Secondo passo: creare, addestrare e valutare la SVM

La classe `SVC` funziona con la stessa logica di tre passaggi che già conosci: crei il classificatore, lo addestri con `fit()` e lo valuti con `score()`. Cominci con un kernel lineare, l'opzione più semplice, che cerca direttamente una strada retta senza trasformare lo spazio delle caratteristiche:

```python
from sklearn.svm import SVC

# Crei la SVM con un kernel lineare (la strada più semplice: una linea retta)
classificatore_svm = SVC(kernel="linear")

# Addestri il modello: qui accade l'ottimizzazione convessa che hai cercato
classificatore_svm.fit(X_train, y_train)

# Valuti la precisione sui dati di test
precisione_svm = classificatore_svm.score(X_test, y_test)
print("Precisione della SVM:", precisione_svm)
```

Esegui questo codice e confronta il numero che ottieni con la precisione che hai annotato nel capitolo cinque per `KNeighborsClassifier` sullo stesso dataset. In molti problemi tabulari come questo, la SVM uguaglia o persino supera il risultato di KNN, anche se il valore esatto dipende dalla ripartizione dei dati, dal kernel scelto e da altri parametri che esplorerai nel capitolo nove. L'importante adesso non è memorizzare una cifra, bensì verificare con i tuoi dati che due algoritmi diversi, addestrati sulla stessa informazione, possono giungere a decisioni simili per percorsi completamente diversi.

Tieni presente, inoltre, che la SVM condivide una particolarità con KNN: entrambi gli algoritmi calcolano distanze e prodotti punto tra vettori, così entrambi sono sensibili alla scala delle caratteristiche. Se una colonna si misura in migliaia e un'altra in unità, quella di numeri più grandi domina il risultato. Per semplicità, questo esempio addestra direttamente sui dati originali, ma in un progetto reale conviene applicare lo stesso tipo di scalatura che già conosci, per esempio con `StandardScaler`, prima di addestrare la SVM.

### Terzo passo: prevedere un caso nuovo

Proprio come con KNN, `predict()` ti permette di classificare un campione concreto:

```python
muestra_nueva = X_test[0].reshape(1, -1)

prediccion = classificatore_svm.predict(muestra_nueva)
clase_predicha = dati.target_names[prediccion[0]]

print("Classe prevista dalla SVM:", clase_predicha)
print("Classe reale:", dati.target_names[y_test[0]])
```

Puoi riusare `X_test[0]`, lo stesso campione che hai provato nel capitolo cinque, e confrontare direttamente se SVM e KNN coincidono nella loro previsione per quel caso concreto.

Se i due modelli coincidono, guadagni fiducia nella diagnosi; se differiscono, hai un segnale chiaro che quel caso concreto merita una revisione aggiuntiva prima di prendere qualsiasi decisione automatica, qualcosa di specialmente rilevante quando il risultato alimenta un processo clinico reale.

## Riepilogo: cosa porti con te da questo capitolo

- Un vettore è una grandezza con direzione e lunghezza, come l'istruzione "cammina tre passi a nord"; in SVM, ogni dato di addestramento si rappresenta come un vettore in uno spazio di caratteristiche
- La magnitudine di un vettore si calcola con la stessa formula della distanza euclidea, e il prodotto punto combina due vettori in un unico numero
- Una SVM cerca l'iperpiano - la "strada" - che separa le classi con il margine più ampio possibile; i dati più vicini ai bordi di quella strada sono i vettori di supporto
- Trovare quel iperpiano ottimale è un problema di ottimizzazione convessa: minimizzare la magnitudine del vettore `w` soggetto a classificare bene tutti i dati, un problema con un unico minimo globale
- Con scikit-learn, addestri una SVM in poche righe: `SVC`, `fit()` per addestrare, `predict()` per prevedere e `score()` per misurare la precisione, esattamente come hai fatto con `KNeighborsClassifier`
- La derivazione matematica completa dell'addestramento - moltiplicatori di Lagrange, margine blando e kernel - arriva nei capitoli otto e nove

## Mettilo in pratica

Riprendi il codice della SVM su `load_breast_cancer()` e osserva come cambia la precisione al modificare il parametro `C` di `SVC`. Non studi ancora a fondo cosa rappresenta `C` matematicamente - questo corrisponde al capitolo nove, quando lavorerai il margine blando - così per ora limitati a osservare l'effetto sul risultato.

**Scrivi un ciclo** che addestri una SVM diversa per molti valori di `C` e che stampi la precisione ottenuta in ogni caso:

```python
valori_di_c = [0.01, 0.1, 1, 10, 100]

for c in valori_di_c:
    classificatore_c = SVC(kernel="linear", C=c)
    classificatore_c.fit(X_train, y_train)
    precisione_c = classificatore_c.score(X_test, y_test)
    print(f"C = {c} -> precisione = {precisione_c:.4f}")
```

**Osserva** se la precisione cambia molto tra i valori estremi di `C` o se rimane stabile, e **annota** i tuoi risultati insieme a quelli che hai già salvato per KNN nel capitolo cinque. **Salva** anche le tue conclusioni sul kernel lineare, perché nel capitolo nove le userai come punto di partenza per capire cosa apporta esattamente il parametro `C` e cosa cambia quando introduci un kernel non lineare.

Prima di chiudere il capitolo, **rispondi** per iscritto a queste due domande, senza cercare ancora la spiegazione formale: un valore di `C` molto piccolo allarga o restringe la strada che separa le classi? E quale relazione intuitiva trovi tra quel comportamento e l'idea del margine ampio che hai lavorato nella sezione sull'iperpiano? Non hai bisogno di indovinare con precisione matematica; basta che annoti la tua intuizione attuale, perché tornerai su queste stesse domande quando arriverai al margine blando nel capitolo nove.

---

**Sigle usate in questo capitolo:** IA (Intelligenza Artificiale), KNN (k vicini più prossimi, dall'inglese K-Nearest Neighbors), SVM (Macchine a Vettori di Supporto, dall'inglese Support Vector Machine).
