# Capitolo 8: Elaborazione del linguaggio naturale con il deep learning

## Perché questo argomento è importante

Scrivi una frase in un traduttore automatico e, in meno di un secondo, ottieni una versione ragionevolmente fedele in un'altra lingua. Pubblichi la recensione di un prodotto e un sistema classifica automaticamente se la tua opinione è positiva o negativa. Chiedi a un assistente di testo di riassumere un rapporto di venti pagine e ti restituisce cinque righe che catturano l'essenziale. Nessuno di questi compiti sembrava alla portata di una macchina appena un decennio fa, e tutti e tre appartengono al campo dell'elaborazione del linguaggio naturale (NLP, dall'inglese *Natural Language Processing*): il ramo dell'intelligenza artificiale (IA) che si occupa di far sì che le macchine comprendano, interpretino e generino linguaggio umano in modo utile.

Il salto di qualità non è aneddotico. Tra il 2010 e il 2016, il numero di articoli pubblicati nelle conferenze di riferimento del NLP, come ACL o EMNLP, che includevano nel titolo le parole "deep" o "neural" è passato da essere praticamente zero a rappresentare una quota sostanziale di tutta la produzione scientifica del campo. E questo aumento di interesse è stato accompagnato da miglioramenti reali: tra il 2018 e il 2019, le prestazioni dei migliori modelli su GLUE (General Language Understanding Evaluation), un insieme di test standard che valuta compiti come stabilire se una frase implica un'altra o classificare un sentimento come positivo o negativo, sono passate da meno del 60 % a circa l'85 % di risposte corrette. Questa tendenza al rialzo si è mantenuta da allora.

Dietro queste cifre c'è una domanda a cui questo capitolo risponde passo dopo passo: come fa una rete neurale, che sa solo moltiplicare numeri e sommare vettori, a elaborare qualcosa di sfuggente come il significato di una parola? Nel capitolo precedente hai studiato le reti neurali ricorrenti (RNN) e le loro varianti con memoria a breve e lungo termine (LSTM) e con unità ricorrenti con porte (GRU), progettate proprio per elaborare sequenze. Ora vedrai come queste architetture, insieme a un elemento aggiuntivo chiamato embedding, si applicano concretamente al linguaggio: come il testo viene convertito in numeri che una rete può elaborare, come quei numeri imparano a catturare il significato e come si combinano in compiti come la modellazione del linguaggio, la classificazione di testi o la traduzione automatica.

Questo tema si collega direttamente a problemi che incontri ogni giorno nel lavoro con aziende dei settori Salute-Cura, Agro-Mare Alimentazione o pubblica amministrazione: un assistente conversazionale che orienta un paziente, un sistema che analizza automaticamente i commenti dei clienti su un prodotto agroalimentare, o uno strumento che classifica e dà priorità alle richieste dei cittadini. Capire come una rete neurale rappresenta ed elabora il linguaggio ti permette di valutare con criterio quale modello o quale fornitore si adatta a ciascuna esigenza, e di prevedere dove il sistema potrebbe sbagliare. Inoltre, questo capitolo prepara il terreno per il successivo, dedicato in profondità al meccanismo di attenzione e all'architettura del transformatore (Transformer), che oggi domina lo stato dell'arte nel NLP.

## Spiegazione concettuale con analogie

Immagina che, invece di conservare ogni parola come un'etichetta isolata, senza relazione con le altre, la collochi come un punto su una mappa. In quella mappa, le parole con significati simili compaiono vicine tra loro, e quelle che non hanno nulla a che vedere restano lontane: "cane" e "gatto" sono vicini, "cane" e "architettura" si trovano agli estremi opposti. A quella mappa diamo il nome di spazio degli embedding, e le coordinate di ogni parola in quello spazio sono il suo embedding: un vettore di numeri reali che la rete apprende durante l'addestramento, e che nessuno progetta a mano.

Questa mappa non è un capriccio tecnico: riflette qualcosa di reale sul linguaggio. Pensa alla parola "faccia". Può riferirsi alla parte anteriore della testa ("ho visto l'espressione sulla sua faccia"), al quadrante di un orologio ("la faccia dell'orologio segna le dieci"), all'atto di affrontare qualcosa ("è ora di metterci la faccia") o a persone nuove ("ci sono facce nuove in ufficio"). Questi quattro usi condividono qualcosa (l'idea di una superficie frontale che informa o rappresenta), ma differiscono anche tra loro. Un sistema che trattasse ogni parola come un simbolo discreto e indipendente, senza relazione con nessun'altra, non potrebbe cogliere né la somiglianza né la differenza tra questi sensi. Un embedding, invece, può collocare tutti i sensi di "faccia" in una zona vicina della mappa, ma non esattamente nello stesso punto, e lasciare che il contesto della frase decida quale si applica in ogni caso.

Questa dipendenza dal contesto è più forte di quanto sembri a prima vista. Lo psicologo David Rumelhart mostrò negli anni settanta che, davanti alla stessa lettera manoscritta ambigua (che potrebbe leggersi come "E", "V" o "W"), le persone la interpretano in modo diverso a seconda della frase in cui appare: in "Juan e María salirono sulla collina", la leggiamo come una lettera; in "l'ultimo evento è stato il salto con l'asta", la leggiamo come un'altra. Il cervello non analizza ogni simbolo in modo isolato, lo interpreta alla luce di tutto ciò che lo circonda. Il linguaggio aggiunge una complicazione in più: queste interazioni rilevanti possono essere non locali, cioè possono verificarsi tra parole molto distanti nella frase. In "l'uomo che ha mangiato il peperoncino ha starnutito", il verbo "ha starnutito" sta accanto a "il peperoncino", ma in realtà si riferisce a "l'uomo", parecchie parole prima. Inoltre, comprendiamo "l'uomo che ha mangiato il peperoncino ha starnutito" più in fretta di "il gatto che ha morso il cane ha abbaiato", anche se entrambe le frasi hanno la stessa struttura grammaticale, perché la nostra conoscenza del mondo ci dice che è normale starnutire dopo aver mangiato qualcosa di piccante, e molto raro che un gatto abbai.

Infine, il significato di una combinazione di parole non è sempre la semplice somma delle sue parti. Un animale domestico tipico (un cane o un gatto) è di solito marrone, bianco o nero. Un pesce è argenteo o grigio. Ma un "pesce d'acquario" (pensa a un pesce da acquario) è spesso di colori vivaci: arancione, verde, blu. La combinazione di "animale domestico" e "pesce" genera una proprietà che nessuno dei due concetti possiede separatamente. Lo stesso accade con "pianta carnivora": né "pianta" né "carnivoro" da soli includono l'idea di "mangiare insetti", ma la combinazione sì. Un buon modello di linguaggio non può combinare i significati con un'unica funzione fissa per qualsiasi coppia di parole: deve imparare, a partire dall'esperienza, come si combinano tipicamente le cose nel mondo reale. Queste quattro osservazioni (le parole hanno più sensi collegati tra loro, la disambiguazione dipende dal contesto, quel contesto può essere non locale, e la composizione dei significati dipende da quali significati concreti si combinano) sono la ragione di fondo per cui le reti neurali, con le loro rappresentazioni distribuite e continue, funzionano così bene per il linguaggio, molto meglio dei sistemi basati su regole e simboli discreti che hanno dominato il campo per decenni.

## Sviluppo teorico

### Dalle parole ai numeri: tokenizzazione e vocabolario

Prima che una rete neurale possa elaborare una frase, bisogna convertirla in numeri. Il primo passo si chiama tokenizzazione: dividere il testo in unità discrete, chiamate token. Esistono due livelli abituali:

- Tokenizzazione a livello di carattere: ogni lettera è trattata come un'unità indipendente. Il vocabolario è piccolo (qualche decina di simboli), ma le sequenze risultanti sono molto lunghe e la rete deve imparare a comporre parole intere a partire da singole lettere
- Tokenizzazione a livello di parola: il testo viene separato dagli spazi bianchi (e dai segni di punteggiatura) e ogni parola è trattata come un simbolo. Il vocabolario risultante è molto più grande, ma ogni token porta già incorporato un significato riconoscibile

Nella pratica, molti sistemi moderni usano un compromesso, dividendo le parole poco frequenti in frammenti più piccoli (subword), ma per capire i fondamenti basta tenere a mente l'idea che bisogna decidere un vocabolario: un insieme finito di simboli, tipicamente costruito scandendo un corpus (una grande collezione di testi) e contando quante volte compare ogni parola, per tenere le più frequenti o tutte, se le risorse lo consentono.

Una volta fissato il vocabolario, a ogni parola viene assegnato un indice numerico. La rappresentazione più diretta è il vettore one-hot: un vettore lungo quanto la dimensione del vocabolario, con tutti zeri tranne un uno nella posizione che corrisponde a quella parola. Se il vocabolario ha 10 000 parole, ogni parola è un vettore di 10 000 numeri, con novemilanovecentonovantanove zeri e un unico uno. Questo vettore è corretto, ma ha un problema serio: tutte le parole si trovano alla stessa distanza l'una dall'altra, come se "cane" e "gatto" non avessero niente di più in comune di "cane" e "termodinamica". È esattamente il problema di trattare le parole come simboli discreti che abbiamo posto nella sezione precedente.

### L'embedding: da un vettore sparso a un vettore denso

La soluzione è moltiplicare quel vettore one-hot per una matrice di pesi addestrabile. Se chiamiamo V la dimensione del vocabolario e D la dimensione dell'embedding (un valore che scegli tu, di solito tra qualche decina e qualche centinaio), il primo strato di un modello di linguaggio neurale è una matrice di pesi E di dimensione V × D. L'embedding di una parola w, con vettore one-hot xw, si ottiene così:

ew = Eᵀxw

Nella pratica, questa moltiplicazione è solo una ricerca in una tabella: poiché xw ha un unico uno, il risultato è semplicemente la riga di E corrispondente all'indice di quella parola. Per questo, in PyTorch, questa operazione si implementa con uno strato chiamato `torch.nn.Embedding`, che internamente non è altro che una tabella di V righe per D colonne, dove ogni riga è il vettore denso di una parola.

La cosa importante è che questa matrice E non viene fissata a mano: viene inizializzata con piccoli valori casuali e regolata tramite retropropagazione, come qualsiasi altro peso della rete, per minimizzare la funzione di perdita del compito che stai addestrando (prevedere la parola successiva, classificare un sentimento, tradurre una frase). Di conseguenza, l'addestramento ricolloca le parole in quello spazio a D dimensioni in modo che quelle che svolgono ruoli semantici o sintattici simili finiscano vicine tra loro, senza che nessuno dica esplicitamente cosa significa ciascuna.

Questo fenomeno non è né nuovo né esclusivo delle architetture attuali. Già nel 1991, Miikkulainen e Dyer costruirono un modello neurale del linguaggio con questa stessa idea di rappresentazione distribuita, e all'inizio degli anni novanta Jeff Elman addestrò una semplice rete ricorrente con l'unico obiettivo di prevedere la parola successiva di una sequenza; analizzando le rappresentazioni interne che la rete aveva appreso, scoprì che parole con funzioni grammaticali e significati simili (soggetti, verbi, categorie semantiche) si raggruppavano spontaneamente nello spazio geometrico. La rete non ricevette mai un'etichetta grammaticale: la struttura emerse semplicemente dal tentativo di prevedere quale parola viene dopo.

### Word2vec: apprendere embedding direttamente dal contesto

Un modo popolare per addestrare buoni embedding, senza dover risolvere prima un compito complesso come la traduzione, è word2vec, pubblicato da un team di Google nel 2013. L'idea centrale è che il significato di una parola è ben riflesso dalle parole che di solito la circondano, e propone due varianti complementari:

- CBOW (Continuous Bag of Words): dato il contesto (le parole vicine), prevede la parola centrale mancante
- Skip-gram: data una parola, prevede le parole del suo contesto

L'obiettivo dello skip-gram, per un testo di T parole w1, ..., wT e una finestra di contesto di dimensione c, è massimizzare:

(1/T) · Σt=1..T Σ−c≤j≤c, j≠0 log p(wt+j | wt)

dove la probabilità di ciascuna parola di contesto si calcola con una funzione softmax su tutto il vocabolario:

p(wO | wI) = exp(v'wO · vwI) / Σw=1..V exp(v'w · vwI)

Qui, vwI è l'embedding della parola di ingresso e v'wO è un secondo embedding associato alla parola di uscita. Addestrare questa rete con milioni di frasi fa sì che le parole che compaiono in contesti simili ("re" e "regina", "cane" e "gatto") finiscano con embedding vicini, e permette persino operazioni aritmetiche sorprendenti nello spazio vettoriale, come il fatto che il vettore di "re" meno il vettore di "uomo" più il vettore di "donna" si avvicini al vettore di "regina". Un approccio alternativo, GloVe (Global Vectors for Word Representation), arriva a risultati simili sfruttando statistiche globali di co-occorrenza delle parole su tutto il corpus, invece di previsioni locali parola per parola.

Questi embedding, sia quelli di word2vec sia quelli che una rete apprende da zero per il proprio compito, condividono una limitazione importante: sono statici. Ogni parola ha un unico vettore, indipendentemente dalla frase in cui compare, quindi "banco" avrebbe la stessa rappresentazione in "sono andato in banca a prelevare soldi" e in "mi sono seduto sulla panchina del parco". Questa limitazione è proprio ciò che motiverà, nel capitolo successivo, architetture capaci di generare un embedding diverso per la stessa parola a seconda del contesto concreto in cui compare.

### Compiti tipici del NLP con il deep learning

Una volta convertito il testo in una sequenza di embedding, una rete neurale (una RNN, una LSTM, una GRU, o le architetture più avanzate che vedrai più avanti) può affrontare compiti molto diversi. Tre dei più comuni sono:

**Modellazione del linguaggio.** Consiste nello stimare la probabilità di una sequenza di parole scomponendola in probabilità condizionate, parola per parola:

P(w1, ..., wT) = Πt=1..T P(wt | w1, ..., wt−1)

Addestrare un modello a massimizzare questa probabilità su grandi quantità di testo (senza bisogno di etichette manuali, perché l'"etichetta" di ogni posizione è semplicemente la parola successiva del testo stesso) gli insegna a generare frasi coerenti, ed è alla base dei sistemi di generazione di testo e della correzione predittiva.

**Classificazione di testi.** In questo caso l'obiettivo è assegnare un'etichetta a un'intera frase o documento: positivo o negativo nell'analisi del sentiment, spam o non spam nel filtraggio della posta, la categoria tematica di una notizia. Un'architettura abituale prende gli embedding della frase, li elabora con uno strato ricorrente o semplicemente li combina (per esempio, calcolandone la media), e fa passare il risultato attraverso uno o più strati densi fino a ottenere una probabilità per ogni classe. L'addestramento minimizza l'entropia incrociata tra la previsione e l'etichetta reale:

L = − Σi yi · log(ŷi)

dove yi vale uno se la frase appartiene alla classe i e zero altrimenti, e ŷi è la probabilità che il modello prevede per quella classe.

**Traduzione automatica.** Qui l'ingresso e l'uscita sono entrambi sequenze, e di lunghezza diversa: una frase in spagnolo può essere tradotta in una frase in inglese con più o meno parole. Questo compito dà origine a una famiglia di architetture particolarmente rilevante per il resto del capitolo: i modelli sequenza-a-sequenza.

### Da sequenza a sequenza: l'encoder-decoder e il suo limite

Un modello sequenza-a-sequenza (sequence-to-sequence, o seq2seq) risolve il problema di mappare una sequenza di ingresso su una sequenza di uscita di lunghezza diversa combinando due reti ricorrenti, tipicamente LSTM o GRU come quelle che hai studiato nel capitolo precedente:

- Un codificatore (encoder) legge la frase di origine parola per parola e aggiorna il proprio stato nascosto a ogni passo. Alla fine della lettura dell'intera frase, il suo ultimo stato nascosto c deve riassumere tutto ciò che serve sapere della frase di ingresso
- Un decodificatore (decoder), un'altra rete ricorrente, riceve quel vettore c come punto di partenza e genera la frase di destinazione parola per parola, condizionando ogni nuova parola al vettore c e alle parole che ha già generato:

P(yt | y<t, c)

Questo schema, proposto all'inizio degli anni 2010, ha funzionato notevolmente bene ed è stato per diversi anni lo standard nella traduzione automatica neurale. Ma ha un collo di bottiglia evidente: tutte le informazioni della frase di origine, che sia di cinque o di cinquanta parole, devono essere compresse in un unico vettore c di dimensione fissa. Più la frase è lunga, più è difficile che quel vettore conservi i dettagli rilevanti, e le prestazioni del modello si degradano in modo notevole con le frasi lunghe. A questo si aggiunge la limitazione che già conosci dal capitolo precedente: anche con le porte che attenuano la scomparsa del gradiente, a una rete ricorrente costa ancora fatica propagare informazione tra posizioni molto distanti della sequenza.

Questa doppia limitazione (il collo di bottiglia di un unico vettore di contesto e la difficoltà delle reti ricorrenti nel mettere in relazione posizioni lontane) è esattamente il problema che risolve il meccanismo di attenzione, che studierai in profondità nel prossimo capitolo. L'idea, anticipata qui solo come preludio, è smettere di forzare l'intera frase di ingresso in un unico vettore fisso e permettere al decodificatore, a ogni passo, di "guardare indietro" e pesare in modo diverso ogni parola dell'ingresso in base alla sua rilevanza per la parola che sta generando in quel momento. Quando questo principio viene portato all'estremo, rinunciando anche alla ricorrenza e calcolando questi pesi tra tutte le posizioni della sequenza in parallelo, si ottiene l'architettura del transformatore (Transformer), alla base di modelli come BERT (Bidirectional Encoder Representations from Transformers) e GPT (Generative Pre-trained Transformer), che conoscerai anche più avanti in questo manuale. Per ora, tieni a mente l'idea centrale: l'evoluzione del NLP con il deep learning va dalla rappresentazione di singole parole come vettori, alla combinazione di quei vettori con reti ricorrenti per i compiti di sequenza, fino a far sì che il modello decida dinamicamente su quali parti dell'ingresso concentrarsi.

## Esempio di codice commentato

Costruiremo, dall'inizio alla fine, un minuscolo classificatore di sentimento: riceve una breve frase in spagnolo e prevede se esprime un'opinione positiva o negativa. Useremo `torch.nn.Embedding` per apprendere rappresentazioni delle parole da zero, su un vocabolario costruito a mano a partire da una manciata di frasi di esempio. Copia ed esegui questo codice in una cella di Google Colab.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. Datos de entrenamiento: frases minúsculas con su etiqueta ---
# Etiqueta 1 = sentimiento positivo, 0 = sentimiento negativo
frases = [
    "me encanta este producto",
    "excelente servicio y rapido",
    "una experiencia maravillosa",
    "lo recomiendo totalmente",
    "el equipo fue muy amable",
    "pesimo servicio y muy lento",
    "no lo recomiendo para nada",
    "una experiencia horrible",
    "el equipo fue muy grosero",
    "producto de mala calidad",
]
etiquetas = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# --- 2. Construimos el vocabulario a mano ---
# <PAD> rellena frases cortas para igualar longitudes dentro de un lote.
# <UNK> representa cualquier palabra no vista durante el entrenamiento.
palabras = sorted(set(palabra for frase in frases for palabra in frase.split()))
vocabulario = ["<PAD>", "<UNK>"] + palabras
indice_de = {palabra: i for i, palabra in enumerate(vocabulario)}
TAMANO_VOCAB = len(vocabulario)
print(f"Tamaño del vocabulario: {TAMANO_VOCAB} palabras")

def tokenizar(frase, longitud_max):
    """Convierte una frase en una lista de índices, rellenada con <PAD>
    hasta longitud_max."""
    indices = [indice_de.get(p, indice_de["<UNK>"]) for p in frase.split()]
    indices = indices[:longitud_max]
    indices += [indice_de["<PAD>"]] * (longitud_max - len(indices))
    return indices

LONGITUD_MAX = max(len(f.split()) for f in frases)
X = torch.tensor([tokenizar(f, LONGITUD_MAX) for f in frases], dtype=torch.long)
y = torch.tensor(etiquetas, dtype=torch.float32)

# --- 3. Definimos el modelo ---
class ClasificadorSentimiento(nn.Module):
    def __init__(self, tamano_vocab, dim_embedding):
        super().__init__()
        # padding_idx=0 evita que <PAD> reciba gradiente: no aporta significado
        self.embedding = nn.Embedding(tamano_vocab, dim_embedding, padding_idx=0)
        self.clasificador = nn.Linear(dim_embedding, 1)

    def forward(self, x):
        vectores = self.embedding(x)          # (lote, longitud, dim_embedding)
        promedio = vectores.mean(dim=1)        # promedio de la frase completa
        logit = self.clasificador(promedio)    # (lote, 1)
        return logit.squeeze(1)

DIM_EMBEDDING = 8
modelo = ClasificadorSentimiento(TAMANO_VOCAB, DIM_EMBEDDING)

# --- 4. Entrenamos ---
perdida_fn = nn.BCEWithLogitsLoss()  # combina sigmoide + entropía cruzada, más estable
optimizador = optim.Adam(modelo.parameters(), lr=0.05)

for epoca in range(200):
    optimizador.zero_grad()
    logits = modelo(X)
    perdida = perdida_fn(logits, y)
    perdida.backward()
    optimizador.step()
    if (epoca + 1) % 50 == 0:
        print(f"Época {epoca + 1}: pérdida = {perdida.item():.4f}")

# --- 5. Probamos con frases nuevas, no vistas en el entrenamiento ---
def predecir(frase):
    modelo.eval()
    with torch.no_grad():
        entrada = torch.tensor([tokenizar(frase, LONGITUD_MAX)], dtype=torch.long)
        probabilidad = torch.sigmoid(modelo(entrada)).item()
    etiqueta = "positivo" if probabilidad > 0.5 else "negativo"
    print(f"'{frase}' -> {etiqueta} (probabilidad = {probabilidad:.3f})")

predecir("me encanta el equipo")
predecir("una experiencia pesima")
predecir("servicio amable y rapido")
```

Nota due scelte di progettazione. Primo, `nn.Embedding(TAMANO_VOCAB, DIM_EMBEDDING)` crea esattamente la matrice E di V × D che hai visto nella teoria: una tabella con una riga per ogni parola del vocabolario e otto colonne (il valore di `DIM_EMBEDDING`), inizializzata a caso e regolata per retropropagazione durante l'addestramento. Secondo, invece di uno strato ricorrente, questo esempio calcola la media degli embedding della frase (`vectores.mean(dim=1)`) prima di classificare; è una semplificazione deliberata per rendere l'esempio leggibile in poche righe, ma perde del tutto l'informazione sull'ordine delle parole. **Sostituisci quella riga con uno strato `nn.LSTM` o `nn.GRU`** tra quelli che già conosci dal capitolo precedente, e verifica come cambia il comportamento del modello con frasi in cui l'ordine conta.

## Riepilogo dei punti chiave

- L'elaborazione del linguaggio naturale (NLP) mira a far sì che le macchine comprendano e generino linguaggio umano; il deep learning ha portato le prestazioni su benchmark come GLUE da meno del 60 % a circa l'85 % in appena un anno
- Le parole non sono simboli discreti e indipendenti: hanno sensi multipli e collegati tra loro, la loro disambiguazione dipende dal contesto (a volte non locale), e il modo in cui si combinano i loro significati dipende da quali parole concrete si combinano
- La tokenizzazione converte il testo in una sequenza di simboli discreti (a livello di carattere o di parola) a partire da un vocabolario costruito su un corpus
- Un embedding sostituisce il vettore sparso one-hot con un vettore denso di dimensione D, ottenuto come una riga di una matrice di pesi E di dimensione V × D che la rete apprende per retropropagazione; parole con funzioni o significati simili finiscono vicine in quello spazio
- Word2vec (con le sue varianti CBOW e skip-gram) e GloVe sono metodi classici per addestrare embedding a partire da statistiche di co-occorrenza su grandi corpus, anche se producono rappresentazioni statiche: un unico vettore per parola, indipendentemente dal contesto
- I compiti tipici del NLP con il deep learning includono la modellazione del linguaggio (prevedere la parola successiva), la classificazione di testi (per esempio, l'analisi del sentiment) e la traduzione automatica
- I modelli sequenza-a-sequenza (seq2seq) combinano un codificatore e un decodificatore ricorrenti per tradurre tra sequenze di lunghezza diversa, ma soffrono di un collo di bottiglia: l'intera frase di origine deve essere compressa in un unico vettore di contesto di dimensione fissa
- Questa limitazione, insieme alla difficoltà delle reti ricorrenti nel mettere in relazione posizioni lontane, motiva il meccanismo di attenzione e l'architettura del transformatore, che studierai nel prossimo capitolo

## Mettilo in pratica

Dedica tra quindici e trenta minuti a questa mini-attività in Google Colab, ampliando il codice della sezione precedente.

1. Apri un nuovo notebook in Google Colab e copia il codice dell'esempio
2. Aggiungi almeno sei frasi nuove alla lista `frases` (tre positive e tre negative), in spagnolo e con un vocabolario diverso da quello dell'esempio originale, e aggiorna `etiquetas` di conseguenza
3. Riaddestra il modello e verifica se azzecca frasi che non erano nel set di addestramento
4. Cambia `DIM_EMBEDDING` con un valore molto più piccolo (per esempio, due) e con uno molto più grande (per esempio, trentadue); osserva come cambiano la velocità di addestramento e la perdita finale
5. **Individua nel codice la riga che calcola la media degli embedding della frase** (`vectores.mean(dim=1)`) e sostituiscila con uno strato `nn.LSTM` seguito dall'estrazione del suo ultimo stato nascosto; riaddestra e annota se il modello distingue meglio le frasi in cui l'ordine delle parole cambia il significato, come "il servizio non è stato gentile" rispetto a "il servizio è stato gentile"
6. Scrivi, in una cella di testo dello stesso notebook, due o tre frasi che spieghino quali informazioni ritieni che il modello perda nel calcolare la media degli embedding invece di elaborarli in ordine con una rete ricorrente

Al termine di questa mini-attività avrai addestrato con le tue mani il componente che trasforma le parole in vettori e avrai verificato in pratica perché l'ordine delle parole conta. Il capitolo successivo riprende questa stessa architettura encoder-decoder per mostrarti come il meccanismo di attenzione ne risolve il limite principale.
