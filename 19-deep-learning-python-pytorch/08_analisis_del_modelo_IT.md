# 08. Analisi del modello

## Perché questo argomento è importante

Già sai costruire una rete neurale (NN, Neural Network) con PyTorch,
addestrarla e, se hai una GPU (Graphics Processing Unit) disponibile,
accelerare quell'addestramento. Ma c'è una domanda che ancora non hai
risposto: quel modello che hai appena addestrato è davvero buono?

La perdita (loss, l'errore che commette il modello) scende durante l'
addestramento. Questo si vede bene sullo schermo, ma non ti dice quel che
davvero hai bisogno di sapere: come si comporta il modello con dati che non ha
mai visto. Un modello può avere una perdita di addestramento eccellente ed essere
inutile nella pratica, perché quel che ha imparato non è la relazione generale
tra i dati e il risultato, ma i dettagli concreti del tuo insieme di
addestramento.

In questo capitolo impari a guardare un modello addestrato con occhio critico: come
distinguere un modello che generalizza da uno che memorizza, quali metriche usare
oltre la perdita, come rilevare in cosa si sbaglia esattamente e, per
ultimo, come conservare il tuo lavoro con `torch.save` per non dover ripetere l'
addestramento ogni volta che vuoi usare il modello.

## Overfitting e underfitting: due forme di fallire

Immagina due persone che si preparano per lo stesso esame a scelta multipla con una
banca di 100 domande di pratica.

La prima si impara a memoria la risposta esatta di ognuna delle 100
domande, senza capire perché quella risposta è corretta. Il giorno dell'esame reale, con domande simili ma non identiche, fallisce quasi tutto: ha memorizzato la banca di domande invece di imparare la materia. Questo è
**overfitting**: il modello impara così bene i dati di
addestramento —include le loro particolarità e il loro rumore— che perde la
capacità di generalizzare a dati nuovi.

La seconda persona ha appena aperto il libro: non domina nemmeno
le domande di pratica. Il giorno dell'esame fallisce sia nelle domande
note che in quelle nuove. Questo è **underfitting**: il
modello è troppo semplice, o non ha avuto addestramento sufficiente, per catturare
i pattern che esistono nei dati.

### Dati "dentro il campione" e "fuori dal campione"

Per diagnosticare quale dei due problemi hai, hai bisogno di due versioni
delle stesse due metriche: la perdita e la precisione (accuracy, la
percentuale di previsioni corrette).

- **Dentro il campione (in-sample):** la perdita e la precisione calcolate
  sui dati di addestramento, gli stessi che il modello usa per
  aggiustare i suoi pesi
- **Fuori dal campione (out-of-sample):** la perdita e la precisione calcolate
  su dati che il modello non ha usato per addestrare —un insieme di
  validazione o di test—

Confrontare questi quattro valori in ogni epoca (un'epoca è una passata
completa per tutto l'insieme di addestramento) è il modo più diretto di
sapere che ti succede:

- **Tutto va bene:** la perdita dentro e fuori dal campione scendono insieme e si
  stabilizzano a un livello basso, con una piccola differenza tra loro
- **Overfitting:** la perdita dentro il campione continua a scendere, ma quella
  fuori dal campione smette di scendere e inizia a salire. Da quel punto il
  modello ha smesso di imparare pattern generali e ha iniziato a memorizzare
  il rumore dei dati di addestramento
- **Underfitting:** le due perdite rimangono alte e appena scendono, persino
  dopo molte epoche. Il modello non ha capacità sufficiente, o non ha
  avuto addestramento abbastanza, per imparare la relazione tra i dati e il
  risultato che vuoi prevedere

Per calcolare la perdita e la precisione fuori dal campione senza sprecare memoria
né tempo in qualcosa che non userai, avvolgi quel calcolo in
`torch.no_grad()`: dici a PyTorch che non hai bisogno di calcolare i gradient,
perché non addestri con quei dati.

### Perché la precisione di test importa più di quella di addestramento

È tentante concentrarsi solo sulla precisione di addestramento, perché di solito è
la più alta e "se la vede meglio". Ma quella cifra non ti dice nulla su come
si comporterà il tuo modello il giorno che riceve dati reali, diversi da quelli che
già conosce. La precisione su un insieme di test —dati che il modello non ha
visto nemmeno una volta durante l'addestramento— è l'unica cifra che si
assomiglia a quel che otterrai in produzione. Se un modello ha 99% di
precisione in addestramento e 70% in test, quel 70% è la cifra che descrive il tuo
modello, non il 99%.

## La matrice di confusione: oltre la percentuale di successo

La precisione ti dà un singolo numero, ma nasconde informazione prezioso. Un
modello con 90% di precisione su dieci categorie potrebbe stare indovinando quasi
sempre in nove di loro e sbagliando sistematicamente nella decima —e la
precisione da sola non te lo dice—.

La **matrice di confusione** risolve quel problema: è una tavola dove ogni
riga rappresenta la categoria reale di un esempio e ogni colonna rappresenta la
categoria che ha previsto il modello. Sulla diagonale principale rimangono i
successi; fuori dalla diagonale, gli errori. Se addestri un modello per
riconoscere cifre scritte a mano sull'dataset MNIST (Modified National
Institute of Standards and Technology, una banca di immagini di cifre da 0
a 9 molto usata per praticare classificazione di immagini), la matrice di
confusione ti può rivelare, per esempio, che il modello confonde spesso la cifra
quattro con la nove, ma quasi mai confonde lo zero con nessun altro numero. Quell'informazione
concreta —quali categorie si confondono tra loro— è molto più utile per decidere come
migliorare il modello che una singola percentuale globale.

## Esempio di codice commentato

Il codice seguente presume che tu abbia già un modello `rete` addestrato
(un'istanza di una classe che eredita da `nn.Module`) e un `DataLoader`
chiamato `caricatore_test` con l'insieme di test. Prima calcoliamo la
precisione e la matrice di confusione; dopo ispezionamo alcune
previsioni errate; per ultimo, conserviamo e recuperiamo il modello.

```python
import torch

NUM_CLASSI = 10  # le dieci cifre da 0 a 9

# Contiamo successi e teniamo la contabilità di in cosa si confonde il modello.
# Iniziamo con una matrice di zeri di NUM_CLASSI x NUM_CLASSI.
matrice_confusione = torch.zeros(NUM_CLASSI, NUM_CLASSI, dtype=torch.int64)
successi = 0
totale = 0

# torch.no_grad() evita di calcolare i gradient: solo stiamo valutando, non
# addestrando, così risparmiamo memoria e tempo di calcolo
rete.eval()  # disattiva strati come dropout, se il modello li ha
with torch.no_grad():
    for imagini, etichette in caricatore_test:
        salide = rete(imagini)

        # La salida ha un valore per ognuna delle dieci classi;
        # torch.argmax si prende l'indice del valore più alto, che è
        # la classe che il modello considera più probabile
        previsioni = torch.argmax(salide, dim=1)

        successi += (previsioni == etichette).sum().item()
        totale += etichette.size(0)

        # Aggiorniamo la matrice di confusione: una riga per etichetta reale,
        # una colonna per etichetta prevista
        for etichetta_reale, etichetta_prevista in zip(etichette, previsioni):
            matrice_confusione[etichetta_reale, etichetta_prevista] += 1

precisione = successi / totale
print(f"Precisione sull'insieme di test: {precisione:.2%}")
print("Matrice di confusione (righe = reale, colonne = previsto):")
print(matrice_confusione)
```

### Ispezionare previsioni errate

Sapere che il modello sbaglia l'8% delle volte è utile, ma vedere esempi concreti
di quegli errori ti aiuta a capire perché succede:

```python
# Ripetiamo la valutazione, ma questa volta conserviamo gli indici dei
# esempi dove la previsione non coincide con l'etichetta reale
indici_errati = []

rete.eval()
with torch.no_grad():
    for numero_batch, (imagini, etichette) in enumerate(caricatore_test):
        salide = rete(imagini)
        previsioni = torch.argmax(salide, dim=1)

        fallimenti = (previsioni != etichette).nonzero(as_tuple=True)[0]
        for indice in fallimenti:
            indici_errati.append({
                "batch": numero_batch,
                "posizione": indice.item(),
                "etichetta_reale": etichette[indice].item(),
                "etichetta_prevista": previsioni[indice].item(),
            })

print(f"Numero di previsioni errate: {len(indici_errati)}")
print("I tre primi errori:")
for errore in indici_errati[:3]:
    print(f"  Reale: {errore['etichetta_reale']}  →  "
          f"Previsto: {errore['etichetta_prevista']}")
```

Con quegli indici puoi recuperare le immagini originali e disegnarle con
Matplotlib per vedere, letteralmente, quali cifre confonde il tuo modello e se l'
errore è comprensibile (una sette mal scritta che sembra un uno) o
sorprendente (un'immagine perfettamente chiara classificata male).

### Conservare e caricare il modello con `torch.save` e `torch.load`

Addestrare un modello può richiedere minuti o ore. Conservarlo ti permette di
riusarlo senza ripetere quel lavoro, condividerlo con un'altra persona o
dispiegarlo in un altro ambiente. In PyTorch, il modo raccomandato è conservare solo
i pesi del modello —ciò che si chiama il suo `state_dict`, un dizionario con il
valore di ogni parametro imparato— e non l'oggetto completo:

```python
ruta_modello = "mi_modello.pt"

# Conserviamo unicamente i pesi imparati
torch.save(rete.state_dict(), ruta_modello)
```

Per recuperarlo dopo, hai bisogno di ricostruire un modello con esattamente
la stessa architettura e caricare in esso i pesi conservati:

```python
# Ricostruiamo la stessa architettura, senza addestrare
modello_restaurato = MiaReteNeurale()

# Carichiamo i pesi conservati dentro a quella architettura
modello_restaurato.load_state_dict(torch.load(ruta_modello))
modello_restaurato.eval()  # lo lasciamo pronto per prevedere, non per addestrare

# Verifichiamo che predica uguale al modello originale
with torch.no_grad():
    salida_originale = rete(imagini_di_prova)
    salida_restaurata = modello_restaurato(imagini_di_prova)
    print("Previsioni identiche?",
          torch.equal(torch.argmax(salida_originale, dim=1),
                      torch.argmax(salida_restaurata, dim=1)))
```

Questo approccio di conservare solo il `state_dict` è quello che raccomanda la
documentazione ufficiale di PyTorch, perché è più portabile: il file
conservato contiene solo numeri, non codice della classe del modello. Se più
tardi addestri su una GPU e vuoi caricare il modello su un computer che
ha solo CPU (Central Processing Unit), aggiungi l'argomento
`map_location=torch.device("cpu")` a `torch.load()` affinché i pesi si
posizionino sul dispositivo corretto.

## Riassunto dei punti chiave

- L'overfitting accade quando la perdita dentro il campione continua a scendere
  mentre quella fuori dal campione si stagna o sale: il modello memorizza in
  luogo di generalizzare
- L'underfitting accade quando entrambe le perdite rimangono alte: il modello non
  ha capacità o addestramento sufficiente
- Confrontare la perdita e la precisione dentro e fuori dal campione in ogni epoca
  è il modo più diretto di diagnosticare quale dei due problemi hai
- La precisione di test importa più che quella di addestramento, perché è l'
  unica cifra che si assomiglia a quel che il modello farà con dati reali
- La matrice di confusione mostra in quali categorie concrete sbaglia il
  modello, qualcosa che una singola percentuale di precisione non rivela
- `torch.save(modello.state_dict(), ruta)` e
  `modello.load_state_dict(torch.load(ruta))` sono il modo raccomandato di
  conservare e recuperare i pesi di un modello, sempre ricostruendo prima la
  stessa architettura

## Mettilo in pratica

Prendi il modello che hai addestrato in un capitolo anteriore di questo corso (per
esempio, la CNN del capitolo 6) e fai il seguente:

1. Valuta il modello sull'insieme di test e calcola la sua precisione e la sua
   matrice di confusione. C'è qualche categoria dove il modello fallisce molto più
   che nelle altre?
2. Cerca cinque esempi dove il modello sbaglia e ispeziona le immagini originali.
   Gli errori sono comprensibili (immagini ambigue) o ti sorprendono?
3. Conserva i pesi del modello con `torch.save`. In una cella o script
   separato, ricostruisci l'architettura da zero, carica quei pesi con
   `load_state_dict` e verifica che la precisione sull'stesso insieme di
   test coincida con quella del modello originale

Se vuoi continuare a praticare in forma guidata, apri il notebook di pratica
di questa lezione: `clase08_practica_analisis_del_modelo.ipynb`.
