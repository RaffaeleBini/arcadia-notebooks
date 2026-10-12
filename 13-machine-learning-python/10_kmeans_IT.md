# 10. Clustering con K-Means

## Perché ti interessa

Finora, in tutti i capitoli precedenti, ogni modello imparava a partire da esempi con una risposta corretta conosciuta. Nella regressione gli davi il prezzo reale, nei k vicini più prossimi (KNN) e nella macchina a vettori di supporto (SVM) gli davi la classe reale. Questo approccio si chiama apprendimento supervisionato: qualcuno ha già etichettato i dati e l'algoritmo deve solo imparare la relazione tra le caratteristiche e quell'etichetta.

Questo capitolo apre un blocco diverso: l'apprendimento non supervisionato. Qui nessuno ti dà la risposta corretta. Consegni i dati all'algoritmo senza alcuna etichetta e gli chiedi di trovare da solo gruppi di elementi simili tra loro. Conoscerai K-Means, l'algoritmo di clustering più usato nella pratica, e lo applicherai sul dataset del Titanic per verificare qualcosa di sorprendente: senza mai dirgli chi è sopravvissuto, l'algoritmo raggruppa i passeggeri in un modo che si assomiglia parecchio alla sopravvivenza reale.

Questo tipo di tecnica ha applicazioni dirette in progetti reali di digitalizzazione: segmentare clienti senza categorie preesistenti, raggruppare aziende agricole per pattern di consumo, o individuare profili di pazienti con comportamenti simili in un dataset sanitario. Capire come funziona K-Means dall'interno ti permette di usarlo con criterio, non come una scatola magica.

## Apprendimento non supervisionato: raggruppare senza etichette

Immagina di rovesciare sul tavolo una scatola di bottoni mescolati: di colori diversi, dimensioni diverse, alcuni con quattro buchi e altri con due. Nessuno ti ha detto in anticipo quanti gruppi esistono né quale caratteristica devi usare per separarli. Eppure, in modo naturale, inizi ad accostare i bottoni simili tra loro: qui i rossi, lì i grandi, più in là quelli a due buchi. Non conoscevi le categorie prima di iniziare: le hai scoperte osservando i bottoni stessi.

Questo è esattamente ciò che fa un algoritmo di apprendimento non supervisionato. Non riceve alcuna etichetta che gli dica "questo bottone è del gruppo A" e "quest'altro è del gruppo B". Riceve solo le caratteristiche di ogni elemento (colore, dimensione, numero di buchi) e deve decidere, basandosi sulla somiglianza tra quelle caratteristiche, quali elementi vanno insieme.

Il clustering è la famiglia di tecniche che risolve questo problema. Esistono due forme principali:

- Clustering piatto (*flat*): decidi tu in anticipo quanti gruppi vuoi, per esempio due o tre, e l'algoritmo distribuisce i dati esattamente in quel numero di gruppi
- Clustering gerarchico: l'algoritmo stesso decide quanti gruppi esistono e come si relazionano tra loro, senza che tu fissi quel numero

K-Means appartiene al primo tipo: scegli tu quanti gruppi, chiamati cluster, vuoi trovare, e l'algoritmo cerca i migliori centri possibili per quella quantità. La lettera k del nome rappresenta proprio quel numero di cluster che decidi prima di iniziare.

L'intuizione centrale di K-Means è semplice: posiziona k punti centrali, chiamati centroidi, e cerca di far sì che ogni dato resti il più vicino possibile al proprio centroide assegnato. L'algoritmo sposta quei centroidi passo dopo passo finché non si adattano bene alla forma reale dei dati. Più avanti in questo capitolo vedrai il meccanismo esatto, passo per passo.

## Preparare dati con colonne di testo: passare da categorie a numeri

K-Means, come quasi tutti gli algoritmi di machine learning, capisce solo numeri. Calcola distanze tra punti, e una distanza ha senso solo tra valori numerici. Il problema è che molti dataset reali includono colonne di testo: il sesso di una persona, il porto in cui è imbarcata o il colore di un prodotto.

Lavorerai con il dataset del Titanic, che raccoglie informazioni sui passeggeri della nave: classe del biglietto, sesso, età, tariffa pagata e se sono sopravvissuti o no, tra altre colonne. Usiamo una copia pubblica stabile del dataset del Titanic invece della fonte originale, che non è più disponibile, e la carichiamo direttamente da un URL con pandas:

```python
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df.head())
print(df.dtypes)
```

Controllando `df.dtypes` vedrai che colonne come `Pclass`, `Age`, `Fare` o `Survived` sono già numeriche, ma altre, come `Sex` o `Embarked`, sono di tipo testo (`object`). Sono proprio queste colonne di testo quelle che devi convertire prima di applicare K-Means.

La tecnica più diretta si chiama codifica per etichette (*label encoding*): a ogni valore distinto di una colonna di testo assegni un numero intero univoco. Per esempio, "female" diventa 0 e "male" diventa 1. Puoi costruire tu stesso questa conversione con un dizionario, senza dipendere da nessuna libreria aggiuntiva:

```python
def convertir_columnas_texto_a_numero(df):
    df = df.copy()
    columnas = df.columns.values

    for columna in columnas:
        texto_a_numero = {}

        def convertir_valor(valor):
            return texto_a_numero[valor]

        # Solo tocamos columnas que no son ya numéricas
        if df[columna].dtype != "int64" and df[columna].dtype != "float64":
            valores_columna = df[columna].fillna(0).tolist()
            valores_unicos = set(valores_columna)

            contador = 0
            for valor_unico in valores_unicos:
                if valor_unico not in texto_a_numero:
                    texto_a_numero[valor_unico] = contador
                    contador += 1

            df[columna] = list(map(convertir_valor, df[columna].fillna(0)))

    # Las columnas que ya eran numéricas (como Age) también pueden tener
    # huecos: los rellenamos aquí para que K-Means no falle con un ValueError
    df = df.fillna(0)

    return df


# Nos quedamos con columnas relevantes y descartamos las que no aportan
# información útil para agrupar pasajeros (nombre, identificador de billete...)
df_modelo = df.drop(
    ["PassengerId", "Name", "Ticket", "Cabin"], axis=1
)
df_modelo = convertir_columnas_texto_a_numero(df_modelo)

print(df_modelo.head())
print(df_modelo.dtypes)
```

Con questo processo, `Sex` arriva a contenere 0 e 1, e `Embarked` arriva a contenere un numero intero per ogni porto di imbarco. Nota che riempiamo anche i valori mancanti con `fillna(0)` prima della conversione: la colonna `Age`, per esempio, ha dei vuoti in diversi passeggeri, e K-Means non può lavorare con valori vuoti. Riempire con zero è una soluzione semplice per questo esercizio; in un progetto reale valuteresti di riempire con la media o la mediana della colonna invece che con zero.

**Esegui `df_modelo.dtypes` dopo la conversione e verifica che tutte le colonne siano già `int64` o `float64`.** Questo è il requisito minimo per poter applicare K-Means.

## Applicare K-Means con scikit-learn sul Titanic

Con i dati già convertiti in numeri, puoi applicare `KMeans`, la classe di clustering che scikit-learn offre in `sklearn.cluster`. La domanda a cui vuoi rispondere è: se chiedi all'algoritmo di separare i passeggeri in due gruppi senza mai dirgli chi è sopravvissuto, quei due gruppi coincideranno con la sopravvivenza reale?

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Separamos la columna objetivo: la usaremos solo para comparar, nunca
# para entrenar. Dársela al algoritmo sería hacer trampa.
y = df_modelo["Survived"]
X = df_modelo.drop(["Survived"], axis=1)

# K-Means calcula distancias, así que escalamos las características para
# que ninguna domine el resultado solo por tener valores más grandes
# (la tarifa, por ejemplo, llega a superar los 500, mientras que el sexo
# solo vale 0 o 1)
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# Creamos el modelo con dos clústeres: buscamos si el algoritmo separa
# a los pasajeros en dos grupos por su cuenta
modelo = KMeans(n_clusters=2, n_init=10, random_state=42)
modelo.fit(X_escalado)

etiquetas = modelo.labels_

# Comparamos las etiquetas encontradas por el algoritmo con la
# supervivencia real. Como K-Means numera sus grupos de forma arbitraria
# (el grupo 0 no significa "no sobrevivió" por definición), probamos
# las dos correspondencias posibles y nos quedamos con la mejor
aciertos_directos = sum(etiquetas == y) / len(y)
aciertos_invertidos = sum(etiquetas != y) / len(y)
precision = max(aciertos_directos, aciertos_invertidos)

print(f"Coincidencia entre los clústeres y la supervivencia real: {precision:.2%}")
```

Se esegui questo codice vedrai che la coincidenza si aggira tra il 60 e il 70%, un risultato affatto scontato considerando che l'algoritmo non ha mai visto la colonna `Survived`. Questo confronto confirma qualcosa di importante sul clustering: anche se è una tecnica non supervisionata, in molti dataset reali i gruppi che trova per somiglianza di caratteristiche coincidono, almeno in parte, con categorie che contano davvero nel mondo reale, come la sopravvivenza. Colonne come `Sex` o `Pclass` influenzano molto questa somiglianza, perché separano in modo piuttosto netto i passeggeri in sottogruppi.

Questo risultato ricorda anche qualcosa che hai già visto con la SVM: nella pratica è comune usare il clustering come passo preliminare a un problema supervisionato. Prima raggruppi i dati senza etichette per esplorare pattern, e poi, se ti interessa, usi quei gruppi come se fossero etichette per addestrare un classificatore.

## Programmare K-Means da zero: l'algoritmo di Lloyd

La procedura che `KMeans` di scikit-learn segue al suo interno si chiama algoritmo di Lloyd, ed è molto più semplice di quanto il nome suggerisca. Consiste di quattro passi che si ripetono in ciclo:

1. Inizializzazione: scegli k punti del dataset come centroidi di partenza
2. Assegnazione: ogni punto del dataset viene assegnato al centroide più vicino
3. Aggiornamento: ogni centroide viene ricalcolato come la media di tutti i punti che gli sono assegnati
4. Ripetizione: torni ai passi 2 e 3 finché i centroidi non smettono di muoversi in modo significativo

Programmerai questo algoritmo da zero, senza usare `KMeans` di scikit-learn, su un dataset piccolo e semplice di sei punti in due dimensioni. Così vedi con totale chiarezza come si muovono i centroidi a ogni giro.

```python
import numpy as np


class KMeansDesdeCero:
    def __init__(self, k=2, tolerancia=0.001, max_iteraciones=300):
        self.k = k
        self.tolerancia = tolerancia
        self.max_iteraciones = max_iteraciones

    def ajustar(self, datos):
        # Paso 1: inicialización. Tomamos los primeros k puntos del
        # conjunto de datos como centroides de partida
        self.centroides = {}
        for i in range(self.k):
            self.centroides[i] = datos[i]

        for iteracion in range(self.max_iteraciones):
            self.clasificaciones = {i: [] for i in range(self.k)}

            # Paso 2: asignación. Cada punto se asigna al centroide
            # más cercano, medido con la distancia euclídea
            for punto in datos:
                distancias = [
                    np.linalg.norm(punto - self.centroides[centroide])
                    for centroide in self.centroides
                ]
                clasificacion = distancias.index(min(distancias))
                self.clasificaciones[clasificacion].append(punto)

            centroides_anteriores = dict(self.centroides)

            # Paso 3: actualización. Cada centroide pasa a ser el
            # promedio de los puntos que tiene asignados
            for clasificacion in self.clasificaciones:
                self.centroides[clasificacion] = np.average(
                    self.clasificaciones[clasificacion], axis=0
                )

            # Paso 4: comprobamos si los centroides ya dejaron de moverse
            convergencia = True
            for centroide in self.centroides:
                centroide_original = centroides_anteriores[centroide]
                centroide_actual = self.centroides[centroide]
                cambio = np.sum(
                    (centroide_actual - centroide_original) / centroide_original * 100.0
                )
                if abs(cambio) > self.tolerancia:
                    convergencia = False

            if convergencia:
                print(f"El algoritmo convergió en la iteración {iteracion + 1}")
                break

        return self

    def predecir(self, punto):
        distancias = [
            np.linalg.norm(punto - self.centroides[centroide])
            for centroide in self.centroides
        ]
        return distancias.index(min(distancias))


# Un conjunto de datos pequeño, pensado para ver los grupos a simple vista:
# tres puntos agrupados cerca del origen y tres puntos agrupados más arriba
datos = np.array([
    [1, 2],
    [1.5, 1.8],
    [1, 0.6],
    [9, 11],
    [8, 2],
    [10, 2],
])

modelo_propio = KMeansDesdeCero(k=2)
modelo_propio.ajustar(datos)

print("Centroides encontrados:")
for indice, centroide in modelo_propio.centroides.items():
    print(f"  Clúster {indice}: {centroide}")

punto_nuevo = np.array([0, 2])
print(f"El punto {punto_nuevo} pertenece al clúster {modelo_propio.predecir(punto_nuevo)}")
```

Nota tre dettagli del codice:

- I centroidi e le classificazioni sono salvati in dizionari indicizzati per il numero di cluster, così puoi scorrerli e aggiornarli in modo ordinato a ogni giro del ciclo
- Il criterio di convergenza confronta il cambiamento percentuale tra il centroide precedente e il nuovo; se quel cambiamento è inferiore alla tolleranza per tutti i centroidi, l'algoritmo si ferma anche se non ha raggiunto il massimo delle iterazioni
- Il metodo `predecir` riutilizza esattamente la stessa logica del passo di assegnazione: calcola la distanza del nuovo punto da ogni centroide già addestrato e restituisce il più vicino

Se esegui questo codice vedrai che l'algoritmo converge in pochissime iterazioni, perché il dataset di esempio ha due gruppi molto separati tra loro. I centroidi finali risultano vicini al centro reale di ogni gruppo di punti, esattamente il comportamento che ti aspettavi dall'algoritmo di Lloyd.

## Riepilogo: cosa porti a casa da questo capitolo

- L'apprendimento non supervisionato cerca pattern in dati senza etichette: nessuno dice all'algoritmo la risposta corretta in anticipo
- K-Means è un algoritmo di clustering piatto: decidi tu in anticipo il numero di cluster, k, che vuoi trovare
- Prima di applicare K-Means devi convertire le colonne di testo in numeri, per esempio con la codifica per etichette
- Applicato al Titanic senza la colonna della sopravvivenza, K-Means raggruppa i passeggeri in un modo che coincide, in buona parte, con chi è realmente sopravvissuto
- L'algoritmo di Lloyd, che scikit-learn usa al suo interno, si riduce a ripetere due passi: assegnare ogni punto al centroide più vicino e ricalcolare i centroidi come la media dei loro punti assegnati
- Programmare K-Means da zero confirma che il meccanismo interno è semplice, anche se scikit-learn lo esegue in modo molto più ottimizzato

## Mettilo in pratica

**Cambia il numero di cluster, k, nella tua implementazione da zero e osserva come cambiano i gruppi.** Riesegui `KMeansDesdeCero` con `k=3` sullo stesso dataset di sei punti e confronta i centroidi risultanti con quelli ottenuti prima con `k=2`:

```python
modelo_tres_clusteres = KMeansDesdeCero(k=3)
modelo_tres_clusteres.ajustar(datos)

print("Centroides con k=3:")
for indice, centroide in modelo_tres_clusteres.centroides.items():
    print(f"  Clúster {indice}: {centroide}")
```

**Ripeti lo stesso esperimento sul dataset del Titanic, provando con `n_clusters=3` e `n_clusters=4` nel blocco di scikit-learn.** Osserva cosa accade alla percentuale di coincidenza rispetto alla sopravvivenza reale man mano che aumenti k: un numero maggiore di cluster non sempre migliora l'interpretazione dei gruppi, perché inizia a dividere uno stesso pattern reale in diversi sottogruppi artificiali.

---

**Sigle usate in questo capitolo:** KNN (K-Nearest Neighbors, k vicini più prossimi), SVM (Support Vector Machine, macchina a vettori di supporto)
