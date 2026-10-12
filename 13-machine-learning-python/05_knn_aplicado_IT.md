# 5. Classificazione con K vicini più prossimi (KNN)

## Perché questo ti riguarda

Fino a ora hai lavorato con la regressione: prevedi un numero, come il prezzo di una casa o la nota di un esame. Ma molte domande del mondo reale non chiedono un numero, chiedono una categoria. Questo tumore è benigno o maligno? Questo messaggio è spam o no? Questo cliente abbandonerà il servizio o continuerà con noi?

Quel tipo di domanda appartiene alla classificazione, l'altra grande famiglia di problemi dell'intelligenza artificiale (IA) applicata. In questo capitolo cominci quel blocco con l'algoritmo più intuitivo di tutti: K vicini più prossimi (KNN, dall'inglese K-Nearest Neighbors). Capire la sua logica, la matematica che la sostiene e come applicarla in poche righe con scikit-learn su un caso reale di diagnosi medica.

Nel settore sanitario, in quello agroalimentare o in qualsiasi ambito dove classifichi campioni, prodotti o pazienti in categorie, questo algoritmo è spesso il primo modello che si prova, proprio perché è facile da spiegare a chi non programma. Quella qualità - poter giustificare una decisione automatica con una frase semplice - ha molto valore quando il progetto ha bisogno di generare fiducia in persone che non sono tecniche.

## Cos'è classificare e come funziona KNN

Classificare consiste nell'assegnare una categoria a un dato nuovo a partire dalle categorie che conosci già. Nulla di più. Se separi i vestiti in mucchi di "estate" e "inverno", stai classificando. Se una banca decide se approvare o rifiutare un prestito, sta anche classificando.

Immagina un piano con punti di due colori: alcuni marcati con più e altri con meno. Visivamente, distingui due gruppi. Adesso appare un punto nuovo, senza marchio, e vuoi decidere a quale gruppo appartiene. L'intuizione più naturale è guardare cosa c'è attorno a quel punto: se è circondato da "più", probabilmente sia un "più"; se è circondato da "meno", probabilmente sia un "meno".

Quella intuizione è esattamente KNN. Il nome lo descrive: guarda i K vicini più prossimi al punto nuovo e lascia che votino. Se la maggior parte di quei vicini appartiene a una classe, il punto nuovo si classifica in quella classe.

Pensalo con il detto popolare: "dimmi con chi vai e ti dirò chi sei". KNN applica quella stessa logica alla lettera: per sapere cosa è un dato nuovo, guarda chi sono i suoi vicini più prossimi e classificalo come la maggior parte di loro. È anche come trasferirti in un nuovo quartiere: se la maggior parte delle persone della tua strada lavora nel settore tecnologico, qualcuno che non ti conosce potrebbe supporre, con ragionevole accuratezza, che anche tu lavori in quel settore.

I passaggi dell'algoritmo sono questi:

- Salva tutti i dati già etichettati che hai disponibili (l'insieme di addestramento)
- Quando arriva un dato nuovo, calcola la sua distanza da ognuno di quei dati noti
- Seleziona gli K dati più vicini, cioè gli K vicini
- Conta quanti vicini appartengono a ogni classe
- Assegna al dato nuovo la classe con più voti

Il parametro K - il numero di vicini che consulti - lo scegli tu prima di addestrare il modello. Un K piccolo, come uno, rende il modello molto reattivo a ogni dato individuale: è flessibile, ma anche più sensibile al rumore e ai valori anomali. Un K grande leviga la decisione e la rende più stabile, ma può offuscare modelli locali importanti. Inoltre, conviene scegliere un numero dispari di vicini quando lavori con due classi, per evitare pareggi nella votazione: con K=4 potresti avere due voti contro due, e non sapresti cosa decidere; con K=3 o K=5, c'è sempre una classe maggioritaria.

Tieni presente anche queste particolarità dell'algoritmo:

- KNN non "impara" una formula durante l'addestramento, semplicemente memorizza i dati, motivo per cui è noto come algoritmo pigro (lazy learning)
- Funziona molto bene con pochi dati e poche dimensioni, ma diventa più lento man mano che il set di dati cresce, perché deve confrontare il punto nuovo con tutti i punti memorizzati
- È sensibile alla scala delle caratteristiche: se una colonna misura in migliaia e un'altra in unità, quella con numeri più grandi domina il calcolo della distanza, quindi conviene normalizzare i dati prima dell'addestramento

## La distanza euclidea: come KNN misura chi è "vicino"

Per decidere chi è vicino a chi, KNN ha bisogno di un modo di misurare la distanza tra due punti. La metrica più usuale è la distanza euclidea, che porta il nome di Euclide, il matematico greco considerato il padre della geometria. È, semplicemente, la distanza in linea retta tra due punti, la stessa che misureresti con un righello su carta.

In un piano di due dimensioni, con un punto A di coordinate (x1, y1) e un punto B di coordinate (x2, y2), la formula è questa:

```
distanza(A, B) = radice quadrata di [ (x2 - x1)^2 + (y2 - y1)^2 ]
```

È, in fondo, il teorema di Pitagora: sottrai le coordinate, elevi al quadrato ogni differenza, sommi i quadrati e estrai la radice quadrata del risultato.

Vediamo un esempio con numeri. Se il punto A è in (1, 2) e il punto B è in (4, 6):

```
distanza = radice quadrata di [ (4 - 1)^2 + (6 - 2)^2 ]
distanza = radice quadrata di [ 9 + 16 ]
distanza = radice quadrata di 25
distanza = 5
```

La stessa idea si generalizza a qualsiasi numero di dimensioni. Se i tuoi dati hanno n caratteristiche anziché due, la formula semplicemente somma più termini, uno per ogni caratteristica:

```
distanza = radice quadrata di [ (a1 - b1)^2 + (a2 - b2)^2 + ... + (an - bn)^2 ]
```

Non hai bisogno di disegnare un piano di trenta dimensioni per capirlo: la logica è la stessa di due dimensioni, solo che confronti più coordinate alla volta. Questo è proprio quello che accade nel dataset medico che userai in questo capitolo: ogni tumore è descritto con molte caratteristiche numeriche, e KNN calcola la distanza euclidea considerando tutte insieme per decidere quali tumori noti sono "vicini" al tumore nuovo.

Così apparirebbe in Python, solo come riferimento matematico, senza usarlo ancora dentro un classificatore:

```python
from math import sqrt

punto_a = (1, 2)
punto_b = (4, 6)

distanza = sqrt((punto_a[0] - punto_b[0]) ** 2 + (punto_a[1] - punto_b[1]) ** 2)

print("Distanza euclidea tra A e B:", distanza)
# Distanza euclidea tra A e B: 5.0
```

**Salva questo calcolo nella memoria**, perché nel prossimo capitolo programmerai tu stesso, da zero, sia la distanza euclidea che l'algoritmo KNN completo, senza dipendere da scikit-learn. Per ora, concentrati sull'usare la versione già costruita per capire quali risultati produce e come si comporta.

## Applicare KNN con scikit-learn

Applicherai KNN su un caso reale: classificare tumori di mammella come benigni o maligni a partire da caratteristiche numeriche misurate nel tessuto, come il raggio, la tessitura o la simmetria delle cellule. La fonte originale di questo esercizio scarica un file di dati dal repository dell'Università della California a Irvine (UCI). Qui usiamo il dataset già incluso in scikit-learn anziché scaricarlo, per non dipendere da una connessione internet: si tratta dello stesso tipo di dati di diagnosi del cancro al seno, pronto per usare con la funzione `load_breast_cancer()`.

### Primo passo: caricare ed esplorare i dati

```python
from sklearn.datasets import load_breast_cancer

datos = load_breast_cancer()

# datos.data contiene le caratteristiche (misurazioni del tumore)
# datos.target contiene l'etichetta: 0 (maligno) o 1 (benigno)
print("Forma delle caratteristiche:", datos.data.shape)
print("Nomi delle classi:", datos.target_names)
print("Prima riga di caratteristiche:", datos.data[0])
print("Etichetta della prima riga:", datos.target[0])
```

Questo dataset porta 569 campioni e 30 caratteristiche per campione, già pulite e senza valori mancanti, il che ti permette di concentrarti sull'algoritmo senza perdere tempo nel depurare i dati.

### Secondo passo: separare addestramento e test

Come nella regressione lineare, devi dividere i dati in un insieme di addestramento, che il modello usa per imparare, e un insieme di test, che riservi per verificare se il modello generalizza bene di fronte a dati che non ha visto:

```python
from sklearn.model_selection import train_test_split

X = datos.data
y = datos.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Campioni di addestramento:", len(X_train))
print("Campioni di test:", len(X_test))
```

Con `test_size=0.2` riservi il 20% dei campioni al test e il restante 80% all'addestramento. Il parametro `random_state` fissa l'aleatorietà della divisione, in modo che, se esegui di nuovo il codice, ottieni sempre lo stesso ripartizione.

### Terzo passo: creare, addestrare e valutare il classificatore

Adesso entra in gioco `KNeighborsClassifier`, la classe di scikit-learn che implementa KNN:

```python
from sklearn.neighbors import KNeighborsClassifier

# Crei il classificatore indicando quanti vicini consulterà (K = 5)
classificatore = KNeighborsClassifier(n_neighbors=5)

# Lo addestri con il metodo fit(), passandogli caratteristiche ed etichette
classificatore.fit(X_train, y_train)

# Calcoli la precisione del modello sui dati di test
precisione = classificatore.score(X_test, y_test)
print("Precisione del modello:", precisione)
```

Il metodo `fit()` salva internamente i dati di addestramento, poiché KNN non calcola nessuna formula complessa durante l'addestramento, semplicemente memorizza. Il metodo `score()` confronta le previsioni del modello su `X_test` con le etichette reali `y_test` e restituisce la percentuale di successi. Con questo dataset e K=5, è usuale ottenere una precisione superiore al 90%, un risultato che dimostra che un algoritmo così semplice può essere sorprendentemente efficace in un caso medico reale.

### Quarto passo: prevedere un caso nuovo

Se vuoi prevedere la classe di un campione particolare, usa il metodo `predict()`:

```python
# Prendi un campione qualsiasi dall'insieme di test come esempio
muestra_nueva = X_test[0].reshape(1, -1)

prediccion = classificatore.predict(muestra_nueva)
clase_predicha = datos.target_names[prediccion[0]]

print("Classe prevista:", clase_predicha)
print("Classe reale:", datos.target_names[y_test[0]])
```

Qui usi `reshape(1, -1)` perché scikit-learn si aspetta sempre una matrice di due dimensioni come input, dove ogni riga è un campione e ogni colonna una caratteristica, anche quando vuoi prevedere un solo campione alla volta.

### Come scegliere il numero di vicini K

Non esiste un valore di K universalmente corretto: dipende dal dataset e dal problema. Come riferimento generale:

- Un K piccolo rende il modello più flessibile, ma più sensibile al rumore, con rischio di sovraddattamento
- Un K grande leviga la decisione e riduce il rumore, ma può perdere sfumature, con rischio di sottoadattamento
- Con due classi, preferisci valori dispari di K per evitare pareggi nella votazione
- Con dataset piccoli, valori di K bassi (per esempio, tre o cinque) solitamente funzionano bene
- Il modo più affidabile di scegliere K è provare molti valori e confrontare la precisione risultante sull'insieme di test, come farai nella sezione di pratica

## Riepilogo: cosa porti con te da questo capitolo

- Classificare significa assegnare una categoria a un dato nuovo a partire da dati già etichettati, a differenza della regressione, che predice un numero
- KNN classifica un dato nuovo consultando gli suoi K vicini più prossimi e assegnandogli la classe maggioritaria tra loro, proprio come un quartiere influenza come ti percepiscono da fuori
- La distanza euclidea misura quanto sono vicini due punti, calcolando la radice quadrata della somma delle differenze al quadrato tra le loro coordinate, e si generalizza senza problema a decine di caratteristiche
- Con scikit-learn, applichi KNN in poche righe: `KNeighborsClassifier`, `fit()` per addestrare, `predict()` per prevedere e `score()` per misurare la precisione
- Il valore di K è una tua decisione: valori dispari evitano pareggi, e provare molti valori ti aiuta a trovare quello che meglio equilibra flessibilità e stabilità
- Nel prossimo capitolo programmerai KNN e la distanza euclidea da zero, senza scikit-learn, per capire esattamente cosa accade "dentro"

## Mettilo in pratica

Riprendi il codice del classificatore su `load_breast_cancer()` e verifica con i tuoi occhi come K influisce il risultato.

**Scrivi un ciclo** che addestri un `KNeighborsClassifier` diverso per ogni valore di K tra uno e quindici, e che stampi la precisione ottenuta in ogni caso:

```python
for k in range(1, 16):
    classificatore_k = KNeighborsClassifier(n_neighbors=k)
    classificatore_k.fit(X_train, y_train)
    precisione_k = classificatore_k.score(X_test, y_test)
    print(f"K = {k} -> precisione = {precisione_k:.4f}")
```

**Osserva** in quale valore di K si stabilizza o peggiora la precisione, e **annota** le tue conclusioni prima di continuare: c'è qualche K che si distingue chiaramente rispetto agli altri? Cosa accade con K=1 rispetto a valori più alti? Salva le tue risposte, perché ti serviranno come riferimento quando nel prossimo capitolo confronterai la tua implementazione di KNN con quella di scikit-learn.

---

**Sigle usate in questo capitolo:** IA (Intelligenza Artificiale), KNN (k vicini più prossimi, dall'inglese K-Nearest Neighbors), UCI (Università della California a Irvine).
