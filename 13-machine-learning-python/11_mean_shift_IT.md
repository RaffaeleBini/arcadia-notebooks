# 11. Clustering con Mean Shift

## Perché ti interessa

Nel capitolo precedente hai imparato K-Means: un algoritmo di clustering che raggruppa dati simili, ma che ti obbliga a decidere in anticipo quanti gruppi cerchi. Questa decisione non è sempre facile. E se non sai quanti profili di cliente esistono nella tua azienda? E se il numero reale di categorie nei tuoi dati è un'incognita?

Mean Shift risolve proprio questo problema. Non gli dici quanti gruppi aspettarti: l'algoritmo li scopre da solo, osservando dove si accumulano i punti con più densità. Tu fornisci i dati e un unico parametro, l'ampiezza di banda (bandwidth), e Mean Shift fa il resto.

In questo capitolo applichi Mean Shift con `scikit-learn` sullo stesso dataset del Titanic che hai usato nel capitolo 10, confronti quanti gruppi trova rispetto ai tre che hai imposto con K-Means, programmi l'algoritmo da zero per capire il suo meccanismo interno e scopri perché un unico raggio fisso non sempre è la scelta migliore.

## L'idea di Mean Shift: salire la collina di densità

Immagina una piazza piena di gente distribuita in capannelli. Se lasci cadere una goccia d'acqua in qualunque punto di quella piazza e la goccia potesse "sentire" dove c'è più gente intorno a sé, rotolerebbe poco a poco verso il capannello più numeroso vicino, poi verso il centro esatto di quel capannello, e lì si fermerebbe. Questo è esattamente ciò che fa Mean Shift con ogni punto dei tuoi dati.

Ogni punto inizia come proprio candidato a centro di gruppo. L'algoritmo guarda attorno a sé, dentro un raggio che tu definisci (il bandwidth), e calcola la media di tutti i punti che cadono dentro quel raggio. Quella media diventa la nuova posizione del centro, leggermente spostata verso dove c'è più densità di punti. Ripete il processo più e più volte: guarda intorno, calcola la media, spostati. Poco a poco, il centro "sale la collina di densità" finché non raggiunge la cima, il punto in cui non è più possibile alcuno spostamento perché è circondato dalla massima concentrazione di dati che può raggiungere da lì.

Quando ripeti questo per tutti i punti del dataset, succede qualcosa di interessante: molti centri finiscono per convergere sulla stessa cima, come se diverse gocce d'acqua lasciate cadere in punti diversi della piazza finissero tutte nello stesso capannello. Alla fine, fondi i centri che restano quasi nello stesso posto e quello che ti resta sono i gruppi reali dei tuoi dati, ciascuno con la propria cima di densità. Nessuno ti ha detto quante cime c'erano: le hai trovate lasciando che l'algoritmo esplorasse il terreno.

Il bandwidth è il pezzo chiave di tutta questa storia. È il raggio della zona che ogni punto osserva intorno a sé prima di decidere in che direzione muoversi. Un bandwidth grande fa sì che ogni punto veda una zona ampia, raggruppi con facilità e finisca con pochi gruppi grandi. Un bandwidth piccolo fa sì che ogni punto veda solo i vicini più prossimi, sia più esigente nel raggrupparsi e finisca con molti gruppi piccoli. Come vedrai nella sezione di pratica, cambiare questo unico numero può cambiare completamente il risultato.

Questa capacità di trovare densità senza conoscere in anticipo il numero di gruppi fa sì che Mean Shift appaia in compiti molto diversi da quelli con cui hai lavorato con K-Means. Nella segmentazione di immagini, raggruppa pixel simili per separare un oggetto dallo sfondo senza sapere quanti oggetti ci sono nella foto. Nel tracciamento di oggetti in video, segue una regione di interesse spostandola verso il picco di densità fotogramma dopo fotogramma. Nella individuazione di anomalie, segnala come sospetto qualunque punto che resti lontano da tutte le cime di densità trovate. Nei tre casi, il vantaggio è lo stesso che sfrutterai con il Titanic: nessuno deve fissare il numero di gruppi prima di eseguire l'algoritmo.

## Applicare Mean Shift con scikit-learn sul Titanic

Lavorerai con lo stesso file CSV (valori separati da virgole, dall'inglese *comma-separated values*) del Titanic che hai usato con K-Means, caricato direttamente da un URL pubblico con `pandas`. L'obiettivo è identico a quello del capitolo precedente: raggruppare i passeggeri in base alle loro caratteristiche e vedere che relazione hanno quei gruppi con la sopravvivenza. La differenza è che ora non imponi un numero di gruppi: lasci che sia Mean Shift a deciderlo.

Prima, carica i dati e preparali. Questo passo è identico a quello del capitolo 10: riempi i vuoti, converti le colonne di testo in numeri e scarta le colonne che non aggiungono informazione utile per raggruppare.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(url)

# Guarda una copia legible antes de transformar nada: la necesitarás
# para interpretar los grupos con nombres reales en vez de números
original_df = titanic.copy()

# Rellena huecos y convierte texto en números
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Fare'].fillna(titanic['Fare'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Descarta columnas que no ayudan a agrupar pasajeros
titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X = titanic.drop('Survived', axis=1)
X_escalado = StandardScaler().fit_transform(X)
```

Con i dati già scalati, applica Mean Shift. Nota che la classe `MeanShift`, del modulo `sklearn.cluster`, non riceve nessun parametro obbligatorio con il numero di gruppi, a differenza di `KMeans`. Qui le passi un bandwidth di partenza; più avanti vedrai cosa succede se lo cambi.

```python
modelo = MeanShift(bandwidth=2.5)
modelo.fit(X_escalado)

etiquetas = modelo.labels_
n_grupos = len(np.unique(etiquetas))
print(f"Mean Shift encontró {n_grupos} grupos")

# Añade las etiquetas al dataframe original para poder interpretarlas
original_df['grupo'] = etiquetas

for grupo in np.unique(etiquetas):
    miembros = original_df[original_df['grupo'] == grupo]
    tasa_supervivencia = miembros['Survived'].mean()
    print(f"Grupo {grupo}: {len(miembros)} pasajeros, "
          f"supervivencia {tasa_supervivencia:.0%}")
```

Ricorda che nel capitolo 10 hai forzato K-Means a trovare tre gruppi, uno per ogni classe di passeggero (prima, seconda e terza). Mean Shift, invece, trova solitamente anch'esso tre gruppi grandi con questo dataset, ma per un motivo diverso: non sa nulla delle classi della nave, semplicemente rileva che la densità dei passeggeri si concentra in tre zone dello spazio delle caratteristiche. Un gruppo appare con un tasso di sopravvivenza basso, vicino al 10%, un altro con un tasso intermedio, vicino al 37%, e un altro con un tasso alto, vicino all'84%. Questa distribuzione coincide, in modo notevole, con quella di terza, seconda e prima classe rispettivamente.

C'è una differenza importante rispetto a K-Means: se cambi il bandwidth o ripeti l'adattamento, Mean Shift può trovare un gruppo aggiuntivo molto piccolo, formato dai passeggeri che hanno pagato una tariffa molto più alta del resto, con un tasso di sopravvivenza vicino al 100%. K-Means non ti avrebbe mai mostrato questa nuance perché avevi già deciso, in anticipo, che esistevano solo tre gruppi. Mean Shift, non avendo quella restrizione, è capace di isolare un gruppo estremo anche se ha pochissimi membri.

**Esegui il codice con il tuo bandwidth e confronta il numero di gruppi che ottieni rispetto ai tre di K-Means.**

Scegliere il valore di `bandwidth` a mano, come hai fatto sopra, funziona per imparare, ma `scikit-learn` può anche stimarlo per te a partire dalla distribuzione stessa dei dati, con la funzione `estimate_bandwidth` del modulo `sklearn.cluster`:

```python
from sklearn.cluster import estimate_bandwidth

bandwidth_estimado = estimate_bandwidth(X_escalado, quantile=0.2)
print(f"Bandwidth estimado: {bandwidth_estimado:.2f}")

modelo_auto = MeanShift(bandwidth=bandwidth_estimado)
modelo_auto.fit(X_escalado)
print(f"Grupos con bandwidth automático: {len(np.unique(modelo_auto.labels_))}")
```

Il parametro `quantile` controlla quanta parte del dataset entra nel calcolo: più è alto, più tende a crescere il bandwidth stimato e meno gruppi troverai. Questa scorciatoia è utile quando non hai nessuna intuizione preliminare sul raggio corretto, anche se per dominare davvero come il bandwidth influenza il risultato non c'è sostituto a provarlo tu stesso con valori diversi, come farai nella sezione finale del capitolo.

## Programmare Mean Shift da zero

Per capire davvero cosa fa Mean Shift al suo interno, lo programmerai tu con un dataset piccolo, senza dipendere da `scikit-learn`. La logica è la stessa che hai spiegato nell'analogia della piazza: ogni punto inizia come un centroide, si sposta verso la media dei suoi vicini dentro un raggio fisso, e ripeti il processo finché nessun centroide si muove più.

```python
import numpy as np


class MeanShiftPropio:
    def __init__(self, bandwidth=4):
        # El bandwidth es el radio de búsqueda alrededor de cada centroide
        self.bandwidth = bandwidth

    def fit(self, datos):
        # Al principio, cada punto del dataset es su propio centroide
        centroides = {i: punto for i, punto in enumerate(datos)}

        while True:
            nuevos_centroides = []

            for i in centroides:
                centroide = centroides[i]
                vecinos = []

                for punto in datos:
                    # ¿Cae este punto dentro del radio del centroide?
                    if np.linalg.norm(punto - centroide) < self.bandwidth:
                        vecinos.append(punto)

                # El nuevo centroide es la media de todos sus vecinos
                nuevo_centroide = np.average(vecinos, axis=0)
                nuevos_centroides.append(tuple(nuevo_centroide))

            # Fusiona los centroides que hayan quedado duplicados
            centroides_unicos = sorted(set(nuevos_centroides))

            centroides_anteriores = dict(centroides)
            centroides = {i: np.array(c) for i, c in enumerate(centroides_unicos)}

            # Comprueba si algún centroide se movió respecto a la vuelta anterior
            optimizado = True
            for i in centroides:
                anterior = centroides_anteriores.get(i)
                if anterior is None or not np.array_equal(centroides[i], anterior):
                    optimizado = False
                    break

            if optimizado:
                break

        self.centroides = centroides


datos = np.array([
    [1, 2], [1.5, 1.8], [1, 0.6],
    [8, 8], [9, 11], [8, 2],
    [10, 2], [9, 3],
])

modelo = MeanShiftPropio(bandwidth=4)
modelo.fit(datos)
print(modelo.centroides)
```

Ripassa cosa fa ogni parte. Il ciclo `while True` rappresenta i round di spostamento: in ogni round, ogni centroide calcola la sua nuova destinazione guardando chi si trova entro il suo raggio. La riga `centroides_unicos = sorted(set(nuevos_centroides))` è la fusione dei centroidi: se due centroidi sono arrivati esattamente nello stesso punto (o molto vicino, secondo la precisione dei tuoi dati), l'insieme (`set`) elimina il duplicato e ti lascia con un solo rappresentante di quella cima di densità. Il controllo finale, con `np.array_equal`, arresta il ciclo non appena nessun centroide si muove da un round all'altro, cioè quando hai raggiunto la convergenza.

Se esegui questo codice, vedrai che gli otto punti dell'esempio, che formano due nuvole chiaramente separate, finiscono per convergere in soli due centroidi finali, uno per nuvola. Nessuno ha detto all'algoritmo che c'erano due gruppi: lo ha dedotto dalla forma stessa dei dati.

Nota il costo di questo approccio: per ogni centroide, a ogni round, scorri l'intero dataset confrontando distanze. Con otto punti non lo noti, ma con migliaia di righe, come il Titanic della sezione precedente, questa versione artigianale diventa lenta molto in fretta. L'implementazione di `MeanShift` in `scikit-learn` risolve lo stesso problema con strutture dati ottimizzate per ricerche di vicini, per questo l'hai usata sul dataset grande e hai riservato la versione tua per un esempio piccolo. L'obiettivo di programmarla tu non è competere in velocità, ma capire esattamente cosa succede a ogni spostamento.

**Cambia le coordinate di `datos` per creare tre nuvole di punti invece di due e verifica che `MeanShiftPropio` trovi tre centroidi.**

## Ampiezza di banda dinamica: adattare il raggio alla densità locale

Il codice precedente usa un bandwidth fisso: lo stesso raggio per tutti i punti, in tutte le zone dello spazio. Questo funziona bene quando i tuoi dati sono distribuiti più o meno uniformemente, ma fallisce in un caso molto comune: quando alcune zone del dataset sono molto popolate e altre quasi vuote.

Pensa di nuovo alla piazza. Se c'è un capannello enorme e compatto in un angolo e, dall'altra parte, appena tre persone sparse, un raggio unico non serve per entrambi i casi. Un raggio piccolo, pensato per separare bene la gente del capannello compatto, lascerà le tre persone sparse senza vedere nessuno intorno a sé e non le raggrupperà mai. Un raggio grande, pensato affinché quelle tre persone si incontrino, farà che il capannello compatto inglobi in un colpo solo capannelli vicini che in realtà erano distinti. Un raggio fisso sacrifica sempre una delle due zone.

La soluzione è un bandwidth dinamico: invece di un raggio unico, usi un raggio di riferimento ampio e, al suo interno, penalizzi i punti in base alla loro distanza reale dal centroide. Più un punto è lontano, meno peso porta alla media che calcola il nuovo centroide; più è vicino, più peso porta. In questo modo, un punto che si trova proprio accanto al centroide influenza molto la direzione in cui si sposta, mentre uno che è al margine del raggio lo muove appena. L'effetto pratico è che il raggio "si comporta" come se fosse diverso secondo la densità locale: nelle zone molto popolate, il peso dei vicini lontani conta appena e il centroide si adatta con precisione; nelle zone poco popolate, il raggio ampio permette al centroide di raggiungere i pochi punti disponibili senza perderli.

Un modo semplice per applicare questa idea è dividere la distanza entro il raggio in vari "scaglioni" e assegnare un peso decrescente a ogni scaglione:

```python
pasos_de_peso = 100  # número de escalones de distancia dentro del radio


def calcular_peso(distancia, radio):
    escalon = int(distancia / radio)
    # Los puntos que caen fuera de los escalones definidos
    # reciben el peso mínimo, no cero
    if escalon > pasos_de_peso - 1:
        escalon = pasos_de_peso - 1
    # Al elevar al cuadrado, la caída de peso con la distancia es más pronunciada
    peso = (pasos_de_peso - escalon) ** 2
    return peso
```

Con questa funzione, un punto molto vicino al centroide (scaglione zero) riceve il peso massimo, e uno che si trova proprio al limite del raggio riceve un peso molto minore, ma non nullo. Moltiplicando ogni punto per il suo peso prima di fare la media, il nuovo centroide risulta dominato dai vicini prossimi senza ignorare del tutto quelli lontani. Questa è la stessa idea che applica `scikit-learn` quando lasci il parametro `bandwidth` di `MeanShift` non specificato: la libreria stima un raggio adeguato a partire dalla densità dei tuoi stessi dati, invece di obbligarti a indovinarlo.

## Riepilogo: cosa porti a casa da questo capitolo

- Mean Shift scopre il numero di gruppi da solo, mentre K-Means ha bisogno che tu glielo indichi in anticipo
- Il bandwidth è il raggio che ogni punto usa per guardare intorno a sé prima di spostarsi verso la media dei suoi vicini
- Il processo ripete lo spostamento finché i centroidi convergono e non si muovono più
- Fondendo i centroidi che sono arrivati quasi nello stesso posto, ottieni i gruppi finali senza averli fissati in anticipo
- Sul Titanic, Mean Shift trova solitamente gruppi con sopravvivenza simile alle tre classi di passeggeri, e a volte individua un gruppo extra di tariffe molto alte che K-Means non può isolare
- Un bandwidth fisso non sempre serve per tutto il dataset: un bandwidth dinamico adatta il peso di ogni vicino in base alla sua distanza reale

Prima di passare alla pratica, vale la pena stabilire quando conviene ogni algoritmo. Scegli K-Means quando già sai, per conoscenza del business o del problema, quanti gruppi cerchi: è più rapido e più facile da regolare. Scegli Mean Shift quando quella cifra è proprio ciò che vuoi scoprire, o quando sospetti che il numero reale di gruppi non coincida con nessuna categoria ovvia dei tuoi dati, come è successo con il gruppo di tariffe alte apparso nel Titanic.

## Mettilo in pratica

Riprendi il codice di Mean Shift con `scikit-learn` sul Titanic e prova diversi valori di `bandwidth` in uno stesso ciclo, per vedere come cambia il numero di gruppi che trova l'algoritmo:

```python
for bw in [1, 2, 2.5, 4, 6, 10]:
    modelo = MeanShift(bandwidth=bw)
    modelo.fit(X_escalado)
    n_grupos = len(np.unique(modelo.labels_))
    print(f"bandwidth={bw} -> {n_grupos} grupos")
```

**Esegui questo ciclo e annota quanti gruppi appaiono con ogni valore.** Osserva il pattern: più piccolo è il bandwidth, più gruppi tende a trovare l'algoritmo, perché ogni punto esige di avere vicini molto prossimi per raggrupparsi. Più grande è il bandwidth, meno gruppi appaiono, finché arrivi all'estremo di fondere tutto il dataset in un unico gruppo. **Cerca il valore di bandwidth che ti dia di nuovo tre gruppi e confronta i loro tassi di sopravvivenza con quelli ottenuti forzando tre gruppi con K-Means nel capitolo 10.**

**Sigle usate in questo capitolo:** CSV (comma-separated values, valori separati da virgole)
