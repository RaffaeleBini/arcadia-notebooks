# Capitolo 6: Ottimizzazione avanzata e regolarizzazione

## Perché questo tema è importante

Nel capitolo precedente hai imparato la discesa del gradiente: a ogni passo calcoli il gradiente della funzione di perdita e sposti i parametri nella direzione opposta, con una dimensione del passo che è segnata dal tasso di apprendimento. Quell'idea funziona, ma ha un punto debole che scopre chiunque addestri una rete neurale reale: in molti problemi, la discesa del gradiente pura converge con una lentezza esasperante, oppure oscilla senza mai stabilizzarsi nel minimo.

Questo capitolo, corrispondente alla Lezione 5 del corso ("Ottimizzazione II: tecniche avanzate e regolarizzazione"), risponde a tre domande in sequenza. Primo, perché la discesa del gradiente di base fallisce proprio nei paesaggi di perdita più comuni nel deep learning. Secondo, come il momento, una tecnica che accumula i gradienti passati, corregge gran parte di quel problema con un costo computazionale minimo. Terzo, che cosa guadagna un ottimizzatore quando usa anche informazione sulla curvatura, la cosiddetta matrice Hessiana, e perché quel guadagno ha un prezzo computazionale così alto che nelle reti neurali reali è praticabile solo tramite approssimazioni intelligenti.

Capire queste idee ti serve anche oltre la teoria. Quando in DIH DATAlife affianchiamo team tecnici di settori come Salute-Cura o Agro-Mar Alimentazione in progetti di digitalizzazione con componenti di intelligenza artificiale (IA), una delle domande più frequenti è perché un modello impiega tante ore ad addestrarsi o perché la perdita si blocca. Sapere che la causa è spesso un paesaggio di perdita mal condizionato, e sapere quale ottimizzatore corrisponde a ogni situazione, è la differenza tra perdere giorni a regolare a mano il tasso di apprendimento e scegliere in pochi minuti lo strumento giusto.

## Spiegazione concettuale con analogie

### La valle stretta: perché il gradiente puro si blocca

Immagina di camminare di notte lungo una valle molto lunga e stretta, come il letto asciutto di un fiume tra due pendii ripidi. La tua unica informazione è la pendenza che senti sotto i piedi in ogni punto in cui ti fermi: quella pendenza è il gradiente. Poiché i pendii sono molto più ripidi del letto stesso della valle, ogni passo ti spinge con forza verso un versante o verso l'altro, e solo un po' in avanti, che è la direzione in cui vuoi davvero avanzare per raggiungere l'uscita della valle.

Il risultato è uno zigzag: rimbalzi da un versante all'altro, avanzando molto poco a ogni passo verso la tua meta reale. Se accorci il passo (abbassi il tasso di apprendimento) per non uscire dalla valle, avanzi in modo sicuro ma esasperantemente lento. Se allunghi il passo per andare più veloce, esci dalla valle e oscilli ancora di più, o addirittura diverghi. È esattamente il comportamento della discesa del gradiente in una valle stretta, uno degli esempi classici per visualizzare perché il metodo di base fallisce in molti paesaggi di perdita reali.

### Il momento: la palla che rotola e accumula inerzia

Ora cambia strategia. Invece di camminare guardando solo la pendenza sotto i piedi in ogni istante, immagina di lasciare rotolare una palla pesante in quella stessa valle, spinta dalla gravità. La palla non reagisce in modo immediato a ogni irregolarità del terreno: ha inerzia, cioè conserva parte della velocità che già aveva.

Quando la palla rotola lungo le pareti della valle, le forze che la spingono verso un versante e verso l'altro si accumulano in direzioni opposte e tendono a cancellarsi: la "spinta" verso sinistra di un tratto si compensa con la "spinta" verso destra del tratto successivo. Invece, la forza che la spinge lungo il letto della valle, verso l'uscita, punta sempre nella stessa direzione a ogni passo, quindi quella componente si rafforza di continuo. Il risultato è che la palla finisce per rotolare veloce lungo la valle e rimbalza appena tra le pareti.

Questo è, con precisione, il principio del momento nell'ottimizzazione: invece di muoverti solo in base al gradiente attuale, accumuli una media dei gradienti recenti (la "velocità" della palla). Le componenti del gradiente che cambiano segno da un passo all'altro, come quelle delle pareti della valle, si annullano nella media. Le componenti che puntano sempre nella stessa direzione, come quella del fondo della valle, si rafforzano. Il momento agisce come un filtro passa-basso: attenua le oscillazioni ad alta frequenza e lascia passare la tendenza di fondo, coerente lungo molte iterazioni.

### Il secondo ordine: non solo dove scendere, ma quanto si curva il terreno

Il momento migliora molto la situazione, ma continua a usare solo il gradiente, cioè solo la pendenza locale. Immagina ora che, invece di camminare alla cieca, tu disponga di una mappa topografica completa della zona che stai attraversando: sai non solo dove scende il terreno, ma anche quanto è pronunciata la curvatura in ogni direzione. Con quella mappa puoi calcolare direttamente, con un'occhiata, il miglior punto di destinazione all'interno di quella regione, invece di fare passi piccoli e prudenti uno dopo l'altro.

Questa informazione di curvatura è esattamente ciò che fornisce la matrice Hessiana, ed è la base dei metodi del secondo ordine, come il metodo di Newton. Il prezzo di quella mappa così dettagliata è che calcolarla e "leggerla" (invertirla matematicamente) è molto più costoso che sentire semplicemente la pendenza sotto i piedi. Per questo, come vedrai più avanti, nelle reti neurali con milioni di parametri bisogna ricorrere a versioni semplificate di quella mappa.

## Sviluppo teorico

### Momento classico

La discesa del gradiente stocastica (SGD) aggiorna i parametri con la regola che già conosci:

θ_{t+1} = θ_t − η ∇_θ L(θ_t)

dove θ rappresenta i parametri del modello, η è il tasso di apprendimento e ∇_θ L(θ_t) è il gradiente della funzione di perdita L all'iterazione t.

Il momento classico introduce una variabile aggiuntiva, la velocità v, che accumula una media mobile con decadimento esponenziale dei gradienti passati:

v_t = β v_{t−1} + η ∇_θ L(θ_{t−1})
θ_t = θ_{t−1} − v_t

Il coefficiente di momento β assume valori tipici tra 0,9 e 0,99, e controlla quanto "ricorda" la velocità dei gradienti precedenti. Osserva il meccanismo: se il gradiente mantiene un segno coerente di iterazione in iterazione (come la "spinta in avanti" del fondo della valle), il suo contributo a v si somma in modo costruttivo e la velocità cresce. Se il gradiente cambia segno da un'iterazione alla successiva (come il rimbalzo tra i due versanti), i contributi si annullano in parte e la velocità in quella direzione resta piccola. L'effetto netto è una traiettoria di ottimizzazione molto più regolare, con meno oscillazioni e un avanzamento più rapido nelle direzioni di pendenza coerente.

Una variante rilevante è il gradiente accelerato di Nesterov (NAG), che invece di calcolare il gradiente nella posizione attuale, lo calcola nella posizione "anticipata" verso cui porterebbe il momento accumulato. Questa piccola correzione, uno sguardo in avanti prima di fare il passo, produce aggiornamenti un po' più informati e, nella pratica, una convergenza ancora più stabile in presenza di curvatura pronunciata o gradienti rumorosi.

### Perché il gradiente puro è così sensibile al cattivo condizionamento

La ragione matematica di fondo della valle stretta ha un nome preciso: il numero di condizionamento del problema, definito come il quoziente tra la curvatura massima e la curvatura minima del paesaggio di perdita attorno al punto attuale. Quando questo numero di condizionamento è alto (una direzione si curva molto più di un'altra, esattamente come le pareti della valle rispetto al suo letto), il numero di passi che qualsiasi metodo del primo ordine impiega per convergere cresce con quel quoziente. La discesa del gradiente, non distinguendo le direzioni, è costretta a usare un tasso di apprendimento abbastanza piccolo da non divergere nella direzione di massima curvatura, e quello stesso tasso risulta insufficiente per avanzare rapidamente nella direzione di minima curvatura. È, in sostanza, la causa radice della lentezza che hai visto nel capitolo precedente e che il momento attenua senza eliminarla del tutto.

### Metodi del secondo ordine: la matrice Hessiana

I metodi del primo ordine usano soltanto il gradiente, cioè la derivata prima della perdita. I metodi del secondo ordine incorporano anche la curvatura, rappresentata dalla matrice Hessiana, H, che raccoglie tutte le derivate parziali del secondo ordine della funzione di perdita rispetto ai parametri:

H_{ij} = ∂²L(θ) / ∂θ_i ∂θ_j

Per un modello con n parametri, H è una matrice quadrata di dimensione n × n. Ogni elemento H_ij ti dice come cambia la pendenza rispetto al parametro θ_i quando muovi il parametro θ_j: è, letteralmente, la mappa di curvatura di cui parlavamo nell'analogia.

Il metodo di Newton si ottiene approssimando la funzione di perdita, nell'intorno del punto attuale, con una funzione quadratica (una parabola in più dimensioni) tramite il suo sviluppo di Taylor del secondo ordine, e minimizzando esattamente quell'approssimazione locale. Il risultato è la seguente regola di aggiornamento:

θ_{t+1} = θ_t − η H⁻¹ ∇_θ L(θ_t)

dove H⁻¹ è la matrice inversa della Hessiana. Nota la differenza chiave rispetto alla discesa del gradiente: il gradiente non viene più usato "grezzo", ma corretto dall'inversa della curvatura. In una direzione molto curvata, H⁻¹ riduce il passo, perché una piccola variazione del parametro provoca già una grande variazione della perdita. In una direzione poco curvata, H⁻¹ amplia il passo, perché lì puoi permetterti di avanzare molto di più prima di notare l'effetto. Nell'esempio della valle stretta, il metodo di Newton "sa" in anticipo che i pendii sono molto più ripidi del letto, e regola il passo di conseguenza in un solo aggiornamento, invece di aver bisogno di centinaia di passi piccoli e zigzaganti.

È questa la ragione di fondo per cui i metodi del secondo ordine, quando sono applicabili, convergono in molti meno passi della discesa del gradiente: eliminano (o riducono drasticamente) la dipendenza dal numero di condizionamento che penalizza i metodi del primo ordine.

### Il costo computazionale: perché il secondo ordine è caro

Il metodo di Newton stesso ha un problema evidente: calcolare e, soprattutto, invertire la matrice Hessiana è costoso, e questo costo cresce molto rapidamente con il numero di parametri n:

- Memorizzare la Hessiana completa costa O(n²), perché è una matrice densa di n × n
- Invertirla con gli algoritmi standard di algebra lineare costa O(n³)

Per farti un'idea dell'ordine di grandezza: una rete neurale modesta può avere diversi milioni di parametri. Elevare quel numero al cubo dà una cifra di operazioni completamente impraticabile, anche per l'hardware più potente disponibile oggi. Confrontalo con il costo di un aggiornamento della discesa del gradiente pura, che è O(n): lineare nel numero di parametri, e quindi perfettamente scalabile a reti con milioni o miliardi di pesi. Questo divario di costo computazionale è il motivo per cui, nella pratica, quasi nessuna rete neurale profonda di dimensioni realistiche si addestra con il metodo di Newton "puro".

Esistono metodi quasi-Newton, come l'algoritmo BFGS (dalle iniziali dei suoi autori, Broyden, Fletcher, Goldfarb e Shanno), che evitano di calcolare la Hessiana esatta e costruiscono invece iterativamente un'approssimazione della sua inversa, usando solo informazione sui gradienti in passi successivi. Riducono il costo rispetto a Newton puro, ma restano onerosi per le scale di parametri abituali nel deep learning.

### Approssimazioni diagonali a blocchi e di Kronecker

Per rendere utilizzabile l'informazione di curvatura nelle reti neurali grandi, senza pagare il costo O(n³) dell'inversione completa, si ricorre ad approssimazioni che semplificano la struttura della Hessiana (o di matrici di curvatura correlate, come la matrice di informazione di Fisher o la matrice Gauss-Newton generalizzata, che in pratica si comportano meglio della Hessiana esatta per addestrare reti neurali).

**Approssimazione diagonale.** La semplificazione più drastica consiste nel tenere solo la diagonale della matrice di curvatura e azzerare tutti gli elementi al di fuori di essa. Questo riduce il costo di memorizzazione e di "inversione" a O(n), perché invertire una matrice diagonale equivale semplicemente a invertire ciascuno dei suoi elementi separatamente. Il rovescio della medaglia è che perdi ogni informazione su come si influenzano tra loro i diversi parametri, cioè proprio l'informazione che vive fuori dalla diagonale.

**Approssimazione diagonale a blocchi.** Un punto intermedio consiste nel dividere la matrice di curvatura in blocchi più piccoli lungo la diagonale (per esempio, un blocco per ogni strato della rete) e trattare ciascun blocco come se fosse indipendente dagli altri, ignorando di nuovo le interazioni tra blocchi diversi. Se dividi una Hessiana di dimensione n × n in k blocchi di dimensione (n/k) × (n/k), il costo di invertire ciascun blocco è O((n/k)³), e il costo totale di invertire i k blocchi è:

k · O((n/k)³) = O(n³ / k²)

Cioè, il costo totale si riduce di un fattore k² rispetto a invertire la matrice intera, al prezzo di perdere l'informazione di interazione tra parametri di blocchi diversi. Più blocchi usi (blocchi più piccoli), maggiore è il risparmio computazionale, ma anche maggiore la perdita di precisione nella curvatura catturata. Questa approssimazione funziona ragionevolmente bene nelle reti in cui l'interazione tra strati diversi è relativamente debole, e peggio nelle architetture in cui i parametri sono molto correlati, come spesso accade nelle reti convoluzionali.

**Approssimazione di Kronecker.** Un'idea più elaborata, nota in letteratura con il nome K-FAC (*Kronecker-Factored Approximate Curvature*, curvatura approssimata fattorizzata di Kronecker), sfrutta la struttura particolare che la matrice di curvatura assume in ciascuno strato di una rete neurale. Invece di approssimare il blocco di curvatura di uno strato come una matrice densa qualsiasi, lo approssima come un prodotto di Kronecker di due matrici molto più piccole, A e B:

H_strato ≈ A ⊗ B

Il prodotto di Kronecker ha una proprietà algebrica molto comoda: l'inversa di un prodotto di Kronecker è il prodotto di Kronecker delle inverse,

(A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹

Questo significa che, invece di invertire una matrice grande, basta invertire separatamente due matrici molto più piccole, il che riduce drasticamente il costo computazionale e rende fattibile applicare informazione di curvatura anche in reti di grandi dimensioni. Il rovescio della medaglia è che questa approssimazione funziona bene solo quando la Hessiana reale dello strato assomiglia davvero a un prodotto di Kronecker, cosa che dipende dall'architettura specifica, e che la sua implementazione è notevolmente più complessa di quella delle approssimazioni diagonali.

In sintesi, sia l'approssimazione diagonale a blocchi sia quella di Kronecker perseguono lo stesso obiettivo: conservare la parte dell'informazione di curvatura che più contribuisce alla qualità dell'aggiornamento, sacrificando la parte meno rilevante, in modo che l'uso del secondo ordine sia computazionalmente sostenibile nelle reti neurali con molti parametri.

## Esempio di codice commentato, in PyTorch

Il seguente esempio addestra lo stesso modello piccolo con tre configurazioni diverse di ottimizzatore, su un problema sintetico con curvatura mal condizionata (simile alla valle stretta dell'analogia), e confronta le loro curve di perdita. **Copia ed esegui questo codice su Google Colab** per vedere con i tuoi occhi la differenza tra SGD puro, SGD con momento e Adam.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Fijamos la semilla para que el experimento sea reproducible
torch.manual_seed(0)

# --- Datos sintéticos con curvatura mal condicionada ---
# Generamos un problema de regresión donde una característica tiene una
# escala mucho mayor que la otra: esto crea, de forma artificial, un
# paisaje de pérdida en forma de "valle estrecho", como en la analogía.
n_muestras = 500
X = torch.randn(n_muestras, 2)
X[:, 0] *= 20.0  # una dirección con mucha más varianza que la otra

pesos_reales = torch.tensor([[3.0], [-2.0]])
y = X @ pesos_reales + 0.1 * torch.randn(n_muestras, 1)


def crear_modelo():
    """Modelo lineal simple: una sola capa, sin sesgo, para poder
    comparar los optimizadores sobre el mismo punto de partida."""
    torch.manual_seed(42)  # mismos pesos iniciales en las tres pruebas
    return nn.Linear(2, 1, bias=False)


def entrenar(optimizador_fn, n_pasos=200):
    modelo = crear_modelo()
    perdida_fn = nn.MSELoss()
    optimizador = optimizador_fn(modelo.parameters())

    historial_perdida = []
    for _ in range(n_pasos):
        optimizador.zero_grad()
        pred = modelo(X)
        perdida = perdida_fn(pred, y)
        perdida.backward()
        optimizador.step()
        historial_perdida.append(perdida.item())

    return historial_perdida


# Usamos la misma tasa de aprendizaje base en los tres casos para que
# la comparación sea justa: lo que cambia es solo el optimizador.
tasa_aprendizaje = 0.001

resultados = {
    "SGD (sin momento)": entrenar(
        lambda params: torch.optim.SGD(params, lr=tasa_aprendizaje)
    ),
    "SGD con momento (0.9)": entrenar(
        lambda params: torch.optim.SGD(
            params, lr=tasa_aprendizaje, momentum=0.9
        )
    ),
    "Adam": entrenar(
        lambda params: torch.optim.Adam(params, lr=tasa_aprendizaje)
    ),
}

# --- Gráfico comparativo ---
plt.figure(figsize=(8, 5))
for nombre, perdidas in resultados.items():
    plt.plot(perdidas, label=nombre)

plt.yscale("log")  # la escala logarítmica facilita comparar velocidades de caída
plt.xlabel("Iteración")
plt.ylabel("Pérdida (MSE, escala log)")
plt.title("Comparación de optimizadores en un valle estrecho")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

Se esegui questo codice, vedrai che la curva di SGD senza momento scende in modo nettamente più lento delle altre due, proprio a causa del cattivo condizionamento che abbiamo introdotto scalando una delle caratteristiche. La curva di SGD con momento cade più rapidamente, perché smussa lo zigzag tra le direzioni. Adam di solito converge ancora più in fretta nelle prime iterazioni, perché oltre ad accumulare momento adatta il tasso di apprendimento per parametro, cosa che vedrai con più dettaglio nel paragrafo seguente.

## Ottimizzatori adattivi e regolarizzazione: buone pratiche attuali

Questo paragrafo non fa parte del contenuto centrale della fonte ufficiale di questo capitolo, che si concentra sul momento e sui metodi del secondo ordine. Lo aggiungiamo come chiusura pratica, seguendo le buone pratiche standard della disciplina, perché tu sappia che questi strumenti esistono e in quale situazione conviene ricorrervi.

**Ottimizzatori adattivi.** RMSprop e Adam, che hai già visto nelle formule della sezione teorica, regolano il tasso di apprendimento di ogni parametro individualmente, dividendo per una media mobile dell'ampiezza dei suoi gradienti recenti. In pratica, questo li rende molto più robusti della SGD pura di fronte a paesaggi di perdita con curvature molto diverse tra un parametro e l'altro, senza bisogno delle costose informazioni della Hessiana. Oggi Adam è il punto di partenza predefinito nella stragrande maggioranza dei progetti di deep learning, e RMSprop resta diffuso in compiti come l'apprendimento per rinforzo.

**Dropout.** Durante l'addestramento, il dropout disattiva casualmente una percentuale di neuroni a ogni passo (un valore abituale è tra il 20 % e il 50 %). Questo costringe la rete a non dipendere eccessivamente da neuroni specifici e migliora la sua capacità di generalizzare a dati nuovi, riducendo il sovradattamento (*overfitting*).

**Weight decay (decadimento dei pesi).** Questa tecnica aggiunge alla funzione di perdita una piccola penalità proporzionale all'ampiezza dei pesi, che spinge il modello verso soluzioni con pesi più piccoli e quindi più semplici. In PyTorch si attiva con il parametro `weight_decay` di quasi tutti gli ottimizzatori, inclusi SGD e Adam.

Nessuna di queste tecniche sostituisce ciò che hai imparato in questo capitolo su momento e curvatura: risolvono un problema diverso, quello della capacità di generalizzazione del modello, non quello della velocità di convergenza dell'ottimizzatore. Ma **combinale** con un buon ottimizzatore ogni volta che addestri un modello reale: la scelta dell'ottimizzatore e le tecniche di regolarizzazione lavorano insieme, non in modo isolato.

## Riepilogo dei punti chiave

- La discesa del gradiente pura è lenta o oscillante nei paesaggi di perdita mal condizionati, come le valli strette, perché non distingue le direzioni con curvatura diversa
- Il numero di condizionamento (quoziente tra curvatura massima e minima) determina quanti passi servono a un metodo del primo ordine per convergere
- Il momento accumula una media mobile con decadimento esponenziale dei gradienti passati, annullando le oscillazioni e rafforzando l'avanzamento nelle direzioni coerenti, come una palla che rotola e accumula inerzia
- Nesterov (NAG) migliora il momento classico calcolando il gradiente nella posizione anticipata, non in quella attuale
- I metodi del secondo ordine, come Newton, usano la matrice Hessiana (le derivate seconde della perdita) per incorporare informazione di curvatura e calcolare aggiornamenti molto più informati del solo gradiente
- Il costo della Hessiana è O(n²) in memoria e O(n³) nell'inversione, a fronte dell'O(n) della discesa del gradiente, il che rende il metodo di Newton puro impraticabile per reti con molti parametri
- Le approssimazioni diagonali a blocchi e di Kronecker (K-FAC) riducono quel costo assumendo una struttura semplificata della matrice di curvatura, al prezzo di perdere parte dell'informazione di interazione tra parametri
- Adam e RMSprop sono ottimizzatori adattivi di uso standard oggi, mentre dropout e weight decay sono tecniche di regolarizzazione abituali per migliorare la generalizzazione del modello

## Mettilo in pratica

Dedica tra quindici e trenta minuti a questa mini-attività:

1. **Copia** il codice della sezione di esempio in un notebook di Google Colab
2. **Modifica** il fattore di scala della prima caratteristica (`X[:, 0] *= 20.0`) provando i valori 2, 20 e 100, e rigenera il grafico comparativo in ciascun caso
3. **Osserva** come cambia la distanza tra le tre curve man mano che il problema è peggio condizionato: quale ottimizzatore regge meglio quando la valle diventa più stretta?
4. **Aggiungi** una quarta curva usando `torch.optim.RMSprop` e verifica se il suo comportamento assomiglia di più a quello di SGD con momento o a quello di Adam
5. **Annota** in due o tre frasi quale ottimizzatore sceglieresti come predefinito per un nuovo progetto e perché, tenendo conto di ciò che hai osservato
