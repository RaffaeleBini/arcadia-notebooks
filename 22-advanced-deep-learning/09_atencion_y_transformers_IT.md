# Capitolo 9 — Attenzione e memoria nel deep learning: il cammino verso i Transformer

## Perché questo argomento è importante

Nei capitoli precedenti hai visto le reti neurali ricorrenti (RNN) e la loro variante più robusta, le reti di memoria a breve e lungo termine (LSTM, *long short-term memory*). Entrambe le architetture elaborano una sequenza parola per parola, o passo dopo passo, e aggiornano uno stato nascosto che riassume tutto ciò che è stato visto fino a quel momento. Funziona ragionevolmente bene con frasi brevi. Il problema compare non appena la sequenza cresce: quello stato nascosto ha una dimensione fissa, e costringerlo a contenere il riassunto di una frase di quaranta parole, di un intero paragrafo o di un documento completo equivale a chiedere a una sola persona di memorizzare un libro a memoria e ripeterlo senza più guardarlo. Qualcosa si perde per strada, quasi sempre ciò che è apparso all'inizio.

Questo capitolo tratta della soluzione che il campo ha trovato per questo problema: l'attenzione (*attention*). Invece di comprimere tutto l'ingresso in un unico vettore e confidare che contenga tutto il necessario, l'attenzione permette al modello di consultare di nuovo l'intera sequenza di ingresso a ogni passo e di decidere, in modo dinamico, quali parti sono rilevanti per la previsione che sta facendo in quel momento. È la differenza tra recitare un libro a memoria e tenere il libro aperto sul tavolo mentre rispondi a domande su di esso.

L'importanza di questo tema non è solo storica. L'attenzione, nella sua forma più depurata, è il meccanismo che sostiene l'intera generazione attuale di modelli linguistici: l'architettura Transformer, pubblicata nel 2017, rinuncia del tutto alla ricorrenza e costruisce tutta la sua capacità di comprendere il contesto su un unico blocco, l'autoattenzione (*self-attention*). Modelli come GPT o BERT sono pile di questo blocco ripetute decine di volte. Capire bene l'attenzione qui è il requisito indispensabile per capire qualsiasi Transformer nei capitoli successivi: senza questo blocco costruito solidamente, il resto dell'edificio non regge.

Inoltre vedrai che l'attenzione non si limita a guardare l'ingresso di un modello: lo stesso meccanismo permette di costruire una memoria esterna indirizzabile, una sorta di blocco per appunti che la rete può leggere e scrivere in modo selettivo. Questa idea, incarnata in architetture come la macchina di Turing neurale (NTM, *Neural Turing Machine*), amplia enormemente ciò che una rete può imparare a fare, dal copiare sequenze lunghe all'eseguire semplici algoritmi.

## Spiegazione concettuale con analogie

Immagina di entrare in una biblioteca enorme alla ricerca di informazioni su "il clima in Patagonia in inverno". Non leggi tutti i libri dall'inizio alla fine: porti in testa una domanda precisa, che in questo contesto chiamiamo la query (*query*). Scorri i dorsi dei libri, ciascuno con un titolo o un'etichetta che ne riassume il contenuto; quelle etichette sono le chiavi (*keys*). Confronti la tua query con ogni chiave e decidi quanto si assomigliano: il libro intitolato "Geografia della Patagonia" si adatta molto meglio di "Storia dell'impero romano". Questo confronto produce un grado di rilevanza per ogni libro. Infine, non porti a casa il titolo, porti il contenuto del libro, cioè il suo valore (*value*), e più quel libro era rilevante secondo il tuo confronto, più peso dai al suo contenuto nel riassunto finale che costruisci.

Questo processo completo — confrontare una query con un insieme di chiavi, ottenere un grado di rilevanza per ciascuna e combinare i valori corrispondenti ponderati da quella rilevanza — è esattamente ciò che fa un meccanismo di attenzione. E funziona allo stesso modo all'interno di una rete neurale: a ogni passo, il modello genera una query ("che cosa devo sapere ora per decidere la parola successiva?"), la confronta con le chiavi di tutte le posizioni dell'ingresso, ottiene pesi di rilevanza tramite una funzione softmax e costruisce un vettore di contesto come somma pesata dei valori.

Questa idea si collega anche a qualcosa che conosci dalla tua esperienza: il cosiddetto "problema del cocktail party", quella capacità che hai di concentrarti su una sola conversazione in una festa rumorosa, filtrando automaticamente le altre voci sullo sfondo. Il tuo cervello non elabora allo stesso modo ogni suono che arriva al tuo orecchio: assegna più peso alla conversazione rilevante e scarta il resto. L'attenzione in una rete neurale persegue esattamente questo effetto, applicato a vettori invece che a onde sonore: elaborare tutta l'informazione disponibile, ma pesando fortemente solo la parte che conta per il compito in corso. È, ancora una volta, la differenza tra memorizzare un rapporto di cento pagine e rispondere a una domanda con il rapporto aperto davanti a te, potendo tornare alla pagina esatta di cui hai bisogno.

## Sviluppo teorico

### Il collo di bottiglia delle RNN in sequenza-a-sequenza

L'architettura abituale per compiti come la traduzione automatica, prima dell'arrivo dell'attenzione, era lo schema codificatore-decodificatore (*encoder-decoder*): una RNN o una LSTM, il codificatore, legge la frase di ingresso parola per parola e produce, alla fine, un unico vettore di stato nascosto che deve riassumere l'intera frase. Quel vettore, e solo quello, viene consegnato al decodificatore, un'altra RNN, che lo usa come punto di partenza per generare la frase di uscita parola per parola.

Il problema è di capacità: il vettore di stato nascosto ha una dimensione fissa, decisa in anticipo, indipendentemente dal fatto che la frase di ingresso abbia cinque parole o cinquanta. Tutta l'informazione — significato, ordine, sfumature, relazioni a lunga distanza tra parole — deve entrare in quello stesso spazio, compressa. Più la sequenza è lunga, più grave è la perdita di informazione, e nella pratica si osserva che la qualità della traduzione cala in modo marcato oltre una certa lunghezza della frase. Questo effetto è il collo di bottiglia del codificatore: tutta la comunicazione tra ingresso e uscita passa per un canale di larghezza fissa, senza possibilità di tornare a consultare direttamente le parole originali una volta compresse.

### La soluzione: lasciare che il decodificatore guardi indietro

L'attenzione risolve questo collo di bottiglia in un modo concettualmente semplice: invece che il decodificatore dipenda solo dall'ultimo stato del codificatore, gli si permette di accedere, a ogni passo di generazione, agli stati nascosti del codificatore corrispondenti a tutte le posizioni dell'ingresso, non solo all'ultima.

Formalmente, dato un insieme di vettori di ingresso $h_1, h_2, \dots, h_n$ (per esempio, lo stato nascosto del codificatore in ogni parola) e lo stato del decodificatore $s_{t-1}$ al passo precedente, il meccanismo calcola prima un punteggio di rilevanza tra lo stato del decodificatore e ciascun vettore di ingresso, tramite una funzione di punteggio $\text{score}$:

$$e_{t,i} = \text{score}(s_{t-1}, h_i)$$

Questa funzione può assumere varie forme: un semplice prodotto scalare, una trasformazione lineare seguita dal prodotto scalare (forma "generale"), oppure una piccola rete che concatena i due vettori e li fa passare per uno strato non lineare (forma additiva, la proposta originale di Bahdanau nel 2014). I punteggi vengono poi normalizzati con una funzione softmax, in modo che sommino a uno e si comportino come pesi di rilevanza:

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{n} \exp(e_{t,j})}$$

Infine, il vettore di contesto $c_t$ che riceve il decodificatore al passo $t$ è la somma di tutti i vettori di ingresso, ponderati dal loro peso di rilevanza:

$$c_t = \sum_{i=1}^{n} \alpha_{t,i} \, h_i$$

Questo vettore di contesto, diverso a ogni passo di generazione, sostituisce — o integra — lo stato fisso del codificatore. Il decodificatore non dipende più da un unico riassunto: in ogni istante ricostruisce, a partire dall'intero ingresso, il sottoinsieme di informazione più utile per quella specifica parola. Quando visualizzi questi pesi $\alpha_{t,i}$ come mappa di calore, con la sequenza di ingresso su un asse e quella di uscita sull'altro, compare spesso, nella traduzione, un andamento quasi diagonale: ogni parola di uscita presta attenzione soprattutto alla parola di ingresso che le corrisponde, con deviazioni puntuali quando l'ordine grammaticale differisce tra le due lingue, per esempio quando un verbo tedesco si sposta alla fine della frase.

### Attenzione implicita contro attenzione esplicita

Conviene distinguere due forme di attenzione. L'implicita è quella che qualsiasi rete profonda sviluppa in modo naturale durante l'addestramento: senza che nessuno la programmi, la rete finisce per rispondere con più forza a certe parti dell'ingresso che ad altre, cosa rilevabile analizzando lo jacobiano della rete, la matrice delle derivate parziali di ogni uscita rispetto a ogni ingresso. L'esplicita, invece, è quella che hai visto nella sezione precedente: un meccanismo definito in modo espresso, con pesi calcolati e visibili, che decide quali parti dei dati vengono presentate alla rete. Rispetto a quella implicita, l'attenzione esplicita offre vantaggi pratici: maggiore efficienza computazionale, migliore scalabilità con ingressi di dimensione variabile e, soprattutto, più interpretabilità, perché puoi ispezionare direttamente i pesi invece di inferire il comportamento della rete in modo indiretto.

### Attenzione basata sul contenuto e separazione chiave-valore

Una variante importante, e quella che si collega direttamente ai Transformer, è l'attenzione basata sul contenuto (*content-based attention*), chiamata anche attenzione associativa. Invece di confrontare lo stato del decodificatore con vettori di ingresso generici, il modello genera esplicitamente un vettore chiave (*key*) e lo confronta con il contenuto di ciascun elemento dei dati usando una funzione di similarità, tipicamente il prodotto scalare o la similarità del coseno. Il risultato viene normalizzato con softmax come prima.

Il miglioramento chiave arriva quando separi i dati in due ruoli distinti: una chiave, usata unicamente per calcolare il peso di attenzione, e un valore, usato unicamente per costruire l'uscita. Con i pesi $w_i$ già calcolati a partire dalle chiavi $k_i$, il vettore di lettura finale si ottiene come combinazione pesata dei valori $v_i$ corrispondenti, non delle chiavi:

$$v = \sum_{i} w_i \, v_i$$

Questa separazione — una cosa serve a decidere dove guardare, un'altra è ciò che effettivamente leggi — è precisamente l'idea della biblioteca della sezione precedente, ed è il pezzo mancante per arrivare alla formulazione moderna di query, chiave e valore.

### Formalizzazione moderna: query, chiave e valore (query, key, value)

Il Transformer generalizza e sistematizza questa idea. Data una sequenza di ingresso rappresentata come una matrice $X$, di dimensione $n \times d$ (n posizioni, ciascuna con un vettore di dimensione $d$), il modello genera tre proiezioni lineari distinte tramite tre matrici di pesi apprese $W^Q$, $W^K$ e $W^V$:

$$Q = X W^Q, \qquad K = X W^K, \qquad V = X W^V$$

$Q$ contiene le query, una per posizione; $K$ contiene le chiavi; $V$ contiene i valori. L'attenzione si calcola confrontando ogni query con tutte le chiavi tramite prodotto scalare, scalando il risultato e applicando softmax per ottenere i pesi, che infine ponderano i valori. Questa operazione è nota come attenzione scalata con prodotto scalare (*scaled dot-product attention*):

$$\text{Attenzione}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V$$

Qui $d_k$ è la dimensione delle chiavi (e, di solito, anche delle query). Lo scaling per $\sqrt{d_k}$ non è cosmetico: quando $d_k$ cresce, il prodotto scalare tende a valori molto grandi in modulo, il che spinge il softmax verso una regione di gradiente quasi nullo, lo stesso problema di saturazione che hai visto con la sigmoide nel capitolo due. Dividere per $\sqrt{d_k}$ mantiene i valori in un intervallo in cui il softmax conserva gradienti utili.

Quando query, chiave e valore provengono dalla stessa sequenza di ingresso, questo meccanismo si chiama autoattenzione (*self-attention*): ogni posizione confronta la propria query con le chiavi di tutte le altre posizioni della stessa frase, inclusa se stessa, e così il modello cattura relazioni tra parole indipendentemente da quanto distano tra loro, cosa che per una RNN diventa sempre più difficile man mano che la distanza aumenta.

### Attenzione multi-testa (multi-head attention)

Una singola attenzione scalata con prodotto scalare obbliga il modello a mescolare in un unico schema di pesi ogni tipo di relazione: sintattiche, semantiche, di prossimità, di correferenza. Il Transformer evita questo limite eseguendo più attenzioni in parallelo, ciascuna con le proprie proiezioni $W_i^Q$, $W_i^K$, $W_i^V$, addestrate in modo indipendente:

$$\text{testa}_i = \text{Attenzione}(Q W_i^Q, K W_i^K, V W_i^V)$$

$$\text{MultiTesta}(Q, K, V) = \text{Concat}(\text{testa}_1, \dots, \text{testa}_h) \, W^O$$

Le uscite delle $h$ teste vengono concatenate e riproiettate alla dimensione originale tramite un'altra matrice appresa $W^O$. Nella pratica, ogni testa tende a specializzarsi: una presta attenzione alle parole vicine, un'altra cattura la concordanza grammaticale a lunga distanza, un'altra segue i riferimenti pronominali, e questa specializzazione emerge da sola durante l'addestramento.

Vale la pena sottolineare la differenza di fondo rispetto alle architetture precedenti, basate su un controllore centrale che decideva dove guardare: nel Transformer, ogni posizione genera la propria query e la confronta con tutte le altre, senza un controllore centralizzato. L'attenzione emerge direttamente dai dati, e il modello rinuncia del tutto a ricorrenza, convoluzioni o memoria esterna; l'articolo originale che ha introdotto questa architettura nel 2017 si intitolava, in modo piuttosto letterale, "L'attenzione è tutto ciò che ti serve" (*Attention Is All You Need*).

### Memoria esterna e indirizzabile

L'attenzione serve anche a leggere e scrivere in una memoria esterna, una struttura dati separata dai pesi della rete che può crescere senza riaddestrare il modello. L'idea nasce da un limite reale delle RNN e delle LSTM: in esse, calcolo e memoria sono fusi nello stesso stato nascosto, quindi ampliare la memoria significa ingrandire quello stato e, con esso, il costo di ogni passo. Un computer convenzionale non funziona così: un piccolo processore accede a una enorme memoria RAM (memoria ad accesso casuale) senza scorrerla per intero a ogni operazione. Questa separazione tra processore e memoria è proprio ciò che le architetture di memoria esterna trasferiscono nel deep learning.

Una rete di memoria (*memory network*) tipica si organizza in quattro componenti: una mappa di caratteristiche di ingresso, che converte l'ingresso in una rappresentazione adeguata; un modulo di generalizzazione, che aggiorna la matrice di memoria con la nuova informazione; una mappa di caratteristiche di uscita, che produce una rappresentazione consultando la memoria; e un modulo di risposta, che genera l'uscita finale a partire da quella consultazione. L'interazione tra ingresso e memoria si modella, di nuovo, con l'attenzione: data una query e una matrice di memoria $M$, con righe $M_i$ (ciascuna una "riga" o slot di memoria), i pesi di attenzione determinano quanto si legge da ciascuno slot, e l'uscita è la somma pesata di quegli slot.

L'architettura di riferimento in questo ambito è la macchina di Turing neurale (NTM), presentata dal team di DeepMind nel 2014, ispirata al modello teorico di Turing. Una NTM combina un controllore — una rete neurale, ricorrente o no — con una matrice di memoria esterna a cui accede tramite "teste" di lettura e scrittura, ciascuna governata da un meccanismo di attenzione. La NTM combina due strategie di indirizzamento:

- Indirizzamento per contenuto: il controllore emette un vettore chiave e lo confronta, tramite similarità del coseno, con ciascuno slot della memoria; il risultato viene normalizzato con softmax, e un parametro aggiuntivo di nitidezza (*sharpness*) permette di concentrare il fuoco su un unico slot quando conviene, invece di distribuirlo in modo diffuso
- Indirizzamento per localizzazione: il controllore può spostare il fuoco dell'attenzione del passo precedente di una certa distanza in avanti o indietro nella memoria, tramite un piccolo nucleo di spostamento (*shift kernel*), il che permette di scorrere la memoria in modo sequenziale, come un iteratore

Combinare i due meccanismi consente comportamenti ricchi: localizzare per contenuto l'inizio di una sequenza memorizzata e poi spostarsi per localizzazione lungo quella sequenza, come farebbe un puntatore che percorre un array.

La scrittura nella NTM si ispira direttamente alle porte delle LSTM. Ogni operazione di scrittura combina un vettore di cancellazione $e_t$, con valori tra zero e uno, e un vettore di aggiunta $a_t$. L'aggiornamento di ciascuno slot di memoria $M_i$, ponderato dal peso di scrittura $w_i$ di quello slot, segue questa forma:

$$M_i \leftarrow M_i \odot (1 - w_i \, e_t) + w_i \, a_t$$

Quando il peso di cancellazione vale uno, il contenuto precedente di quello slot viene azzerato prima di sommare il vettore di aggiunta; quando vale zero, il contenuto resta intatto. È una versione differenziabile e morbida della decisione binaria "cancellare o non cancellare", esattamente nello stesso spirito delle porte di una LSTM, ma applicata a una memoria esterna molto più ampia dello stato nascosto abituale.

Questa separazione tra calcolo e memoria ha una conseguenza pratica notevole: una LSTM addestrata a copiare sequenze brevi generalizza male a sequenze molto più lunghe, perché gli elementi che può trattenere sono limitati dalla dimensione fissa del suo stato. Una NTM, invece, mantiene la fedeltà della copia anche con sequenze sensibilmente più lunghe di quelle usate nell'addestramento, perché la sua capacità di memoria non è legata alla dimensione del controllore. Questo ha permesso di addestrare questi modelli su compiti di sapore algoritmico: copiare sequenze, ripeterle un numero indicato di volte (come un ciclo "for"), o inferire probabilità di transizione in una sequenza binaria a partire dai pattern già osservati.

L'evoluzione naturale della NTM è il computer neurale differenziabile (DNC, *Differentiable Neural Computer*), pubblicato nel 2016, che aggiunge meccanismi di accesso più sofisticati e permette di lavorare con strutture a grafo, con nodi e collegamenti, invece di limitarsi alle sequenze. Un DNC addestrato può rispondere a domande sul cammino più breve tra due stazioni di una rete di trasporto, o risolvere relazioni di parentela in un albero genealogico.

## Esempio di codice commentato

Implementerai l'attenzione scalata con prodotto scalare esattamente come l'hai definita nella teoria, usando solo operazioni di base sui tensori — moltiplicazione di matrici (`matmul`, con l'operatore `@`) e una funzione softmax — senza ricorrere a strati già costruiti come `nn.MultiheadAttention`. Lavorerai su una frase breve di prova e visualizzerai i pesi di attenzione risultanti con una mappa di calore. Il frammento è autosufficiente e funziona direttamente in Google Colab.

```python
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

# Frase de ejemplo, ya tokenizada en palabras (en un modelo real usarías un tokenizador real)
tokens = ["el", "gato", "negro", "duerme", "en", "el", "sofá"]
n = len(tokens)
d = 8       # dimensión del embedding, pequeña solo para que el ejemplo sea legible
d_k = 8     # dimensión de consultas y claves

# Embeddings de partida: en un modelo entrenado vendrían de una capa de embedding aprendida;
# aquí los generamos al azar para centrarnos en el mecanismo de atención
X = torch.randn(n, d)

# Proyecciones lineales aprendibles: transforman el mismo embedding en tres papeles distintos
W_q = torch.randn(d, d_k) * 0.5
W_k = torch.randn(d, d_k) * 0.5
W_v = torch.randn(d, d_k) * 0.5

Q = X @ W_q   # consultas: qué busca cada palabra
K = X @ W_k   # claves: qué ofrece cada palabra al ser comparada
V = X @ W_v   # valores: qué contenido entrega cada palabra si es elegida

def softmax_manual(z, dim=-1):
    # Restamos el máximo antes de la exponencial por estabilidad numérica,
    # el mismo truco que viste con softmax en la capa de salida de clasificación
    z = z - z.max(dim=dim, keepdim=True).values
    exp_z = torch.exp(z)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

def atencion_escalada(Q, K, V, d_k):
    puntuaciones = Q @ K.transpose(-2, -1)            # producto punto: consulta contra cada clave
    puntuaciones = puntuaciones / (d_k ** 0.5)         # escalado por raíz de d_k
    pesos = softmax_manual(puntuaciones, dim=-1)       # cada fila suma uno: es una distribución
    salida = pesos @ V                                 # combinación ponderada de los valores
    return salida, pesos

salida, pesos_atencion = atencion_escalada(Q, K, V, d_k)

print("Forma de la salida:", salida.shape)                        # (7, 8)
print("Cada fila de pesos suma:", pesos_atencion.sum(dim=-1))      # ~1.0 en cada posición
```

**Esegui questa cella in Google Colab prima di proseguire.** Verifica che la forma di `salida` coincida con quella di `X` nel numero di posizioni, e che ogni riga di `pesos_atencion` sommi approssimativamente a uno: è il segnale che il softmax è applicato correttamente e che ogni parola distribuisce la propria attenzione come una distribuzione di probabilità sulle altre.

Ora visualizza quei pesi come mappa di calore, con la parola che chiede (query) sull'asse verticale e la parola consultata (chiave) sull'asse orizzontale:

```python
plt.figure(figsize=(6, 5))
plt.imshow(pesos_atencion.detach().numpy(), cmap="viridis")
plt.xticks(range(n), tokens, rotation=45, ha="right")
plt.yticks(range(n), tokens)
plt.xlabel("Clave (palabra observada)")
plt.ylabel("Consulta (palabra que pregunta)")
plt.title("Pesos de atención escalada por producto punto")
plt.colorbar(label="Peso de atención")
plt.tight_layout()
plt.show()
```

**Esegui questa seconda cella e osserva la mappa risultante.** Poiché embedding e proiezioni sono casuali, qui non vedrai un pattern linguistico reale — per quello servono pesi addestrati su dati reali — ma vedrai chiaramente la meccanica: ogni riga è la distribuzione di attenzione di una parola sulle altre, con toni più chiari dove il peso è maggiore. È lo stesso tipo di mappa che compare in qualsiasi strumento di visualizzazione dei Transformer, solo che qui l'hai costruita tu, da zero, con due operazioni sui tensori e una funzione softmax.

## Riepilogo dei punti chiave

- Le RNN e le LSTM soffrono di un collo di bottiglia nel comprimere un'intera sequenza in un unico stato nascosto di dimensione fissa, il che degrada le prestazioni sulle sequenze lunghe
- L'attenzione sostituisce quel riassunto unico con un vettore di contesto diverso a ogni passo, calcolato come somma pesata di tutto l'ingresso, con pesi ottenuti tramite un punteggio di rilevanza normalizzato con softmax
- L'attenzione basata sul contenuto separa i dati in chiavi, usate per decidere dove guardare, e valori, usati per costruire l'uscita; questa separazione chiave-valore è il seme diretto della formulazione moderna di query, chiave e valore
- L'attenzione scalata con prodotto scalare si calcola come $\text{softmax}(QK^{\top}/\sqrt{d_k})\,V$, dove $Q$, $K$ e $V$ sono proiezioni lineari apprese dello stesso ingresso; lo scaling per $\sqrt{d_k}$ evita che il softmax saturi con dimensioni grandi
- Quando query, chiave e valore provengono dalla stessa sequenza, il meccanismo si chiama autoattenzione, e permette di mettere in relazione direttamente qualsiasi coppia di posizioni indipendentemente dalla distanza tra loro
- L'attenzione multi-testa esegue più attenzioni in parallelo, con proiezioni indipendenti, e concatena le loro uscite: ogni testa tende a specializzarsi in un tipo diverso di relazione tra parole
- Il Transformer elimina del tutto la ricorrenza e costruisce tutta la sua capacità di modellare il contesto su blocchi di autoattenzione ripetuti, il che ha risolto in modo definitivo il collo di bottiglia e sostiene i modelli linguistici moderni
- La memoria esterna indirizzabile, come nella macchina di Turing neurale (NTM) e nella sua erede, il computer neurale differenziabile (DNC), separa calcolo e archiviazione usando l'attenzione per leggere e scrivere in modo selettivo su una matrice di memoria, combinando indirizzamento per contenuto e per localizzazione

## Mettilo in pratica

Dedica tra quindici e trenta minuti a questa mini-attività in Google Colab. Riutilizza il codice di questo capitolo come punto di partenza.

1. Sostituisci la frase di esempio con una tua di sei-dieci parole, e porta `d` e `d_k` a sedici invece di otto
2. Aggiungi una seconda testa di attenzione: crea un secondo insieme di matrici `W_q2`, `W_k2`, `W_v2`, calcola i suoi pesi di attenzione e disegna le due mappe di calore una accanto all'altra con `plt.subplots(1, 2)`; confronta se le due teste distribuiscono l'attenzione in modo diverso
3. Forza manualmente una relazione di dipendenza: moltiplica l'embedding di una parola per un valore costante alto (per esempio, `X[2] = X[2] * 5`) e riesegui l'attenzione; osserva e annota come cambiano i pesi delle altre parole verso quella posizione
4. Come sfida finale, rimuovi lo scaling per $\sqrt{d_k}$ dalla funzione `atencion_escalada` e porta `d_k` a centoventotto; confronta la mappa di calore risultante con quella della versione scalata e collega ciò che osservi alla saturazione del softmax che abbiamo spiegato nella teoria

Conserva il tuo notebook di Colab: nel prossimo capitolo costruirai un blocco Transformer completo, con attenzione multi-testa, normalizzazione e una rete feed-forward, appoggiandoti direttamente al codice di attenzione che hai appena scritto qui.

## Glossario delle sigle usate in questo capitolo

- DNC: computer neurale differenziabile (*Differentiable Neural Computer*)
- LSTM: memoria a breve e lungo termine (*long short-term memory*)
- NTM: macchina di Turing neurale (*Neural Turing Machine*)
- RAM: memoria ad accesso casuale (*random-access memory*)
- RNN: rete neurale ricorrente (*recurrent neural network*)
