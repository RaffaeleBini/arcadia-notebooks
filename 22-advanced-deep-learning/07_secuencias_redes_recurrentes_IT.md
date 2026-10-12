# Capitolo 7: Sequenze e reti neurali ricorrenti

**Acronimi usati in questo capitolo:** rete neurale artificiale (ANN), rete neurale convoluzionale (CNN), perceptron multistrato (MLP), rete neurale ricorrente (RNN), memoria a breve-lungo termine (LSTM, dall'inglese *Long Short-Term Memory*), unità ricorrente con porte (GRU, dall'inglese *Gated Recurrent Unit*), retropropagazione attraverso il tempo (BPTT, dall'inglese *Backpropagation Through Time*), elaborazione del linguaggio naturale (NLP), errore quadratico medio (MSE), classificazione temporale connessionista (CTC, dall'inglese *Connectionist Temporal Classification*).

## Perché questo argomento è importante (serie temporali, testo, audio, musica)

Pensa alle ultime volte in cui un sistema informatico ha dovuto capire qualcosa che riguardava te: la tastiera del tuo smartphone che suggerisce la parola successiva, un assistente vocale che trascrive quello che dici, un'applicazione musicale che consiglia la canzone successiva in base a quello che hai ascoltato prima. In tutti e tre i casi, il dato in ingresso non è una foto isolata né una tabella di numeri indipendenti: è una sequenza, una collezione di elementi in cui l'ordine conta, in cui un elemento può ripetersi e in cui la lunghezza totale non è fissata in anticipo. Una frase ha tante parole quante ne servono, un audio dura quanto dura, e una serie di vendite mensili cresce ogni mese che passa.

I capitoli precedenti di questo manuale hanno lavorato con architetture pensate per un altro tipo di dati. Il perceptron multistrato (MLP) si aspetta un vettore di dimensione fissa: se addestri un MLP con frasi di cinque parole, non saprà cosa fare con una frase di otto. E anche se forzassi quella dimensione con un riempimento artificiale, continuerebbe a non cogliere che "il gatto insegue il cane" e "il cane insegue il gatto" contengono le stesse parole ma significano cose opposte, perché un MLP di norma concatena o somma i propri ingressi senza distinguere in modo nativo le posizioni. La rete neurale convoluzionale (CNN), dal canto suo, accetta ingressi di dimensione variabile grazie ai suoi filtri scorrevoli, ma è stata progettata per catturare schemi spaziali locali, come i bordi in un'immagine; non ha un meccanismo proprio per accumulare contesto lungo una dimensione temporale estesa, elemento dopo elemento, ricordando ciò che è stato visto molti passi prima.

Questa limitazione non è un dettaglio tecnico di poco conto. Compare in qualsiasi dominio in cui i dati arrivano sotto forma di flusso: testo e linguaggio naturale (traduzione automatica, generazione di testo, analisi del sentiment), audio e voce (riconoscimento del parlato, sintesi vocale, composizione musicale), serie temporali (previsione della domanda, sensori di temperatura o di umidità, quotazioni finanziarie), video (una sequenza di fotogrammi) e persino sequenze di decisioni, come quelle che prende un agente di apprendimento per rinforzo mossa dopo mossa. Nel contesto di un hub di innovazione digitale come il nostro, questo si traduce in problemi molto concreti: sensori agricoli che generano letture ogni ora e dai quali vuoi anticipare un'infestazione, registri clinici che si accumulano visita dopo visita, oppure il parlato catturato da un dispositivo di assistenza per persone anziane. Nessuno di questi problemi si adatta bene a un modello che sa elaborare solo vettori fissi e senza memoria del passato.

Questo capitolo ti presenta la famiglia di architetture progettata specificamente per colmare questa carenza: la rete neurale ricorrente (RNN) e le sue varianti più robuste, la LSTM e la GRU. **Presta particolare attenzione al concetto di stato nascosto**, perché è il pezzo che rende possibile tutto il resto.

## Spiegazione concettuale con analogie

Immagina di leggere questa stessa frase ad alta voce, parola per parola, senza vedere il resto del paragrafo. Non elabori ogni parola in modo isolato, come se fossi appena nato in quell'istante: hai in testa un riassunto mentale di tutto ciò che hai letto fino a quel punto, e questo riassunto condiziona il modo in cui interpreti la parola nuova. Se hai appena letto "il paziente presentava febbre alta e", il tuo riassunto mentale anticipa già che la parola successiva probabilmente continuerà a descrivere sintomi, non parlerà di sport. Quando finisci di leggere una parola in più, aggiorni quel riassunto: scarti i dettagli che non sono più rilevanti, conservi ciò che conta e aggiungi l'informazione nuova. Questo riassunto mentale, che si aggiorna passo dopo passo e condiziona ogni nuova interpretazione, è esattamente ciò che in una RNN chiamiamo stato nascosto.

Un'altra analogia utile: pensa a chi prende appunti durante una lunga riunione. Non trascrive la riunione parola per parola; tiene un quaderno con un riassunto che riscrive ogni pochi minuti, incorporando il nuovo e scartando ciò che ha perso rilevanza. Alla fine della riunione, quel quaderno non contiene tutto ciò che è stato detto, ma una versione compressa e utile dell'essenziale. Una RNN fa qualcosa di simile con ogni elemento di una sequenza: mantiene un vettore di stato (il suo "quaderno") che aggiorna a ogni passo, e quel vettore è l'unica via attraverso cui l'informazione del passato arriva al presente.

Ora, questa analogia anticipa anche un problema reale. Se la riunione dura otto ore e il tuo quaderno ha solo una pagina, qualcosa dell'inizio andrà perso o dimenticato, soprattutto se riscrivi il riassunto in modo aggressivo a ogni aggiornamento. È precisamente ciò che accade alle RNN semplici con sequenze lunghe: l'informazione antica si diluisce, un fenomeno che nello sviluppo teorico chiameremo scomparsa del gradiente (vanishing gradient). Per risolverlo, più avanti conoscerai un secondo quaderno, più protetto, che cambia solo quando qualcuno decide esplicitamente che deve cambiare: questa è, in sostanza, l'idea alla base della LSTM.

## Sviluppo teorico

### Di cosa ha bisogno un buon modello di sequenze

Prima di definire l'architettura, conviene fissare quali proprietà deve avere qualsiasi funzione che riassuma una sequenza. Da quanto emerge dalla letteratura di riferimento sulla modellazione di sequenze, un buon modello deve:

- Rispettare l'ordine degli elementi, perché riordinarli può cambiare completamente il significato
- Accettare sequenze di lunghezza variabile, senza dover stabilire in anticipo quanti elementi ci saranno
- Essere differenziabile, per poter essere addestrato mediante discesa del gradiente
- Essere sensibile ai cambiamenti individuali, in modo che alterare un solo elemento possa modificare in modo apprezzabile l'output
- Preservare le dipendenze a lungo termine, catturando relazioni tra elementi lontani tra loro nella sequenza

I approcci precedenti al deep learning falliscono in qualcuno di questi punti. I modelli di N-grammi, comuni nell'elaborazione del linguaggio naturale (NLP), condizionano la probabilità di una parola unicamente alle N parole precedenti; rispettano l'ordine, ma perdono tutto il contesto oltre quella finestra fissa, e inoltre la tabella di probabilità cresce in modo esponenziale con la dimensione del vocabolario e della finestra, fino a diventare ingestibile anche con vocabolari modesti. Sommare o mediare i vettori di tutti gli elementi della sequenza ammette sì una lunghezza variabile, ma distrugge completamente l'ordine: "non mi piace, ma lo comprerei" e "mi piace, ma non lo comprerei" darebbero lo stesso vettore aggregato. Nessuno dei due approcci è differenziabile in modo naturale né scala bene. Serve un meccanismo diverso: uno che impari, con parametri regolabili tramite gradiente, a comprimere una sequenza arbitrariamente lunga in una rappresentazione utile.

### La rete neurale ricorrente

Una RNN risolve questo problema mantenendo un vettore di stato nascosto, indicato con h, che si aggiorna a ogni passo temporale combinando l'ingresso attuale con lo stato del passo precedente. Data una sequenza di ingressi x₁, x₂, ..., xₜ, l'aggiornamento dello stato nascosto al passo t è definito come:

**hₜ = tanh(Wₕ · hₜ₋₁ + Wₓ · xₜ + bₕ)**

Qui, Wₕ è la matrice dei pesi che trasforma lo stato precedente, Wₓ è la matrice dei pesi che trasforma l'ingresso attuale, bₕ è il termine di bias, e tanh è la funzione tangente iperbolica, che limita i valori dello stato tra meno uno e uno per evitare che crescano senza controllo. Lo stato iniziale h₀ viene di solito fissato a zero, anche se in compiti come la traduzione automatica può essere inizializzato con informazioni utili, per esempio un riassunto della frase di origine.

Se il compito è prevedere l'elemento successivo della sequenza, ad esempio la parola successiva in un testo, lo stato nascosto si trasforma in una distribuzione di probabilità sul vocabolario tramite un altro strato lineare seguito dalla funzione softmax:

**yₜ = softmax(Wy · hₜ + by)**

Un dettaglio cruciale: le matrici Wₕ, Wₓ e Wy sono condivise in tutti i passi temporali. La stessa "regola di aggiornamento" si applica allo stesso modo al passo tre e al passo trecento, il che permette alla rete di elaborare sequenze di qualsiasi lunghezza con un numero fisso di parametri. Quando questo processo viene disegnato passo dopo passo, estendendo la rete nel tempo, si parla di "srotolare" (unroll) la RNN: ogni passo temporale è rappresentato come una copia della stessa rete, collegata alla copia successiva attraverso lo stato nascosto.

Addestrare una RNN per compiti di previsione di sequenze (come la modellazione del linguaggio, dove l'obiettivo è massimizzare la probabilità della sequenza osservata) si formula di solito come un problema di classificazione a ogni passo temporale, usando la perdita di entropia incrociata tra la distribuzione prevista e la parola reale. La perdita totale della sequenza è la somma delle perdite di ogni passo, e l'insieme dei parametri viene regolato minimizzando quella somma su tutto l'insieme di addestramento. L'algoritmo che calcola i gradienti in questo scenario si chiama retropropagazione attraverso il tempo (BPTT): consiste nello srotolare la rete, calcolare il gradiente della perdita rispetto a ciascuna copia dei pesi condivisi e accumulare questi contributi per ottenere il gradiente finale.

### Il problema del gradiente che svanisce o esplode

Qui emerge la principale limitazione pratica della RNN semplice. Per capire perché, semplifica lo scenario: ignora gli ingressi e i bias, e supponi che la matrice Wₕ si riduca a un unico numero scalare w. La ricorrenza hₜ = w · hₜ₋₁ si può srotolare come:

**hₜ = wᵗ · h₀**

Questa espressione rivela il problema con chiarezza. Se il valore assoluto di w è maggiore di uno, wᵗ cresce in modo esponenziale con t, e lo stato (e, durante l'addestramento, il gradiente che viene retropropagato) esplode verso valori enormi: è il fenomeno noto come gradiente esplosivo. Se invece il valore assoluto di w è minore di uno, wᵗ decade esponenzialmente verso zero man mano che t cresce: è il gradiente che svanisce, o vanishing gradient. In pratica, l'aggiornamento reale di una RNN passa attraverso una funzione tanh, la cui derivata non supera mai il valore uno e di solito è piuttosto minore in gran parte del suo intervallo. Poiché il gradiente che si propaga all'indietro in BPTT è, essenzialmente, un prodotto di molti fattori di questo tipo (uno per ogni passo temporale attraversato), basta che ogni fattore sia leggermente minore di uno perché, dopo decine o centinaia di passi, il prodotto complessivo si avvicini a zero.

La conseguenza pratica è grave: quando il gradiente che arriva a un passo temporale lontano è praticamente nullo, quella parte della rete smette di ricevere un segnale di apprendimento utile, e il modello non riesce a regolare i suoi pesi per sfruttare dipendenze distanti. Considera un esempio classico: nel testo "Finalmente, Tim prevedeva di visitare la Francia nell'ultima settimana del suo viaggio. Era entusiasta di provare la gastronomia locale (...) prese il treno delle cinque del mattino verso...", prevedere con precisione la parola successiva, molto probabilmente "Parigi", richiede di ricordare il dato "Francia" menzionato decine di parole prima. Se il gradiente associato a quella dipendenza è svanito, la RNN semplice non può impararla, anche se in teoria ha la capacità sufficiente per rappresentarla.

### LSTM: memoria protetta e porte

Per risolvere questo problema è stata progettata la rete di memoria a breve-lungo termine (LSTM), pubblicata originariamente da Hochreiter e Schmidhuber. L'idea centrale è separare due flussi di informazione: uno stato di cella (cell state), indicato con c, che funge da nastro trasportatore di memoria a lungo termine con aggiornamenti per lo più additivi, e uno stato nascosto h, più simile a quello di una RNN classica, che viene calcolato a partire dallo stato di cella a ogni passo. L'accesso alla memoria a lungo termine è regolato da tre porte, ognuna implementata come uno strato con attivazione sigmoide (che produce valori tra zero e uno, interpretabili come "proporzione che lascia passare"):

**Porta di oblio**, che decide quale parte dello stato di cella precedente conservare:

fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)

**Porta di ingresso**, che decide quale informazione nuova incorporare. È composta dalla porta stessa e da un vettore di valori candidati generato con tanh:

iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)

c̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)

Con questi due elementi, lo stato di cella si aggiorna combinando ciò che si conserva del passato con ciò che si aggiunge di nuovo:

**cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ**

dove ⊙ indica il prodotto elemento per elemento. Nota che questo aggiornamento è, in gran parte, una somma: lo stato di cella precedente viene moltiplicato per la porta di oblio e vi si aggiunge il contributo nuovo, invece di essere riscritto del tutto con una trasformazione non lineare come accadeva nella RNN semplice. Questo carattere additivo è la chiave che attenua la scomparsa del gradiente: quando la porta di oblio è vicina a uno, il gradiente può fluire all'indietro attraverso molti passi temporali quasi senza attenuarsi, perché la derivata di una somma rispetto a uno dei suoi termini è semplicemente uno.

Infine, la **porta di uscita** decide quale parte dello stato di cella viene esposta come stato nascosto:

oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)

**hₜ = oₜ ⊙ tanh(cₜ)**

Con queste quattro equazioni, una LSTM può imparare, per esempio, a mantenere quasi intatta la porta di oblio mentre elabora una frase lunga in cui deve ricordare un soggetto menzionato all'inizio, e ad "aprire" la porta di ingresso solo quando compare un'informazione davvero rilevante che deve sostituire quella precedente.

### GRU: un'alternativa più leggera

L'unità ricorrente con porte (GRU), proposta da Cho e collaboratori, semplifica la LSTM fondendo lo stato di cella e lo stato nascosto in un unico vettore, e riducendo le tre porte a due: una porta di aggiornamento, che combina il ruolo delle porte di oblio e di ingresso, e una porta di reset. Le equazioni sono:

**zₜ = σ(Wz · [hₜ₋₁, xₜ] + bz)** (porta di aggiornamento)

**rₜ = σ(Wr · [hₜ₋₁, xₜ] + br)** (porta di reset)

**h̃ₜ = tanh(W · [rₜ ⊙ hₜ₋₁, xₜ] + b)** (stato candidato)

**hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ**

La porta di aggiornamento zₜ decide, in un unico passo, quanto dello stato precedente conservare e quanto dello stato candidato incorporare; quando zₜ vale zero, lo stato non cambia affatto, il che permette di propagare informazione con pochissime perdite per molti passi, proprio come faceva lo stato di cella nella LSTM. La porta di reset rₜ, invece, controlla quanta influenza ha lo stato precedente nel calcolo del candidato: se si avvicina a zero, il modello "riparte da zero" nell'elaborare l'ingresso attuale, ignorando il passato immediato.

Avendo meno parametri della LSTM, la GRU si addestra un po' più in fretta e consuma meno memoria, e in un numero considerevole di compiti raggiunge prestazioni comparabili. Non esiste una regola universale su quale scegliere; in pratica conviene provarle entrambe e confrontarle sul problema concreto, cosa che farai tu stesso nella sezione di codice.

### Applicazioni e direzione verso cui evolve il campo

RNN, LSTM e GRU condividono uno stesso schema d'uso oltre la modellazione del linguaggio: nella traduzione automatica e in altri compiti da sequenza a sequenza (seq2seq), una rete codificatrice comprime l'intera sequenza in ingresso in un vettore di contesto, e una rete decodificatrice genera la sequenza in uscita parola per parola a partire da quel vettore, un approccio che ha alimentato sistemi come il traduttore neurale di Google. Nella generazione di immagini, modelli come PixelRNN trattano un'immagine come una sequenza di pixel e prevedono ciascuno condizionato a quelli precedenti; nell'audio, architetture affini come WaveNet generano forme d'onda campione per campione. Nell'apprendimento per rinforzo, agenti come quelli di OpenAI Five o AlphaStar incorporano un nucleo LSTM per elaborare osservazioni che arrivano in modo sequenziale nel corso di una partita.

Detto questo, le LSTM e le GRU non sono l'ultima parola. La loro elaborazione è intrinsecamente sequenziale (passo dopo passo), il che limita la parallelizzazione durante l'addestramento, e anche se attenuano la scomparsa del gradiente, catturare dipendenze davvero lunghe resta costoso. Negli ultimi anni, i transformatori (Transformers), basati su meccanismi di attenzione che confrontano tutti gli elementi della sequenza tra loro in parallelo, hanno sostituito le architetture ricorrenti in buona parte dell'elaborazione del linguaggio naturale. Esplorerai questa famiglia di modelli in un capitolo successivo; per ora, tieni a mente che RNN, LSTM e GRU hanno posto le basi concettuali, come lo stato che riassume il contesto e l'addestramento per massima verosimiglianza, su cui sono stati costruiti quei modelli più recenti.

## Esempio di codice commentato, in PyTorch

Il seguente esempio addestra una piccola LSTM come modello di linguaggio a livello di carattere: dato un frammento di testo, impara a prevedere il carattere successivo e, una volta addestrata, genera testo nuovo carattere per carattere, campionando dalla distribuzione di probabilità che produce a ogni passo. Puoi copiare ed eseguire questo blocco direttamente in Google Colab.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Texto de entrenamiento: se repite para que la LSTM tenga suficientes
# ejemplos del mismo patrón y pueda converger en pocas epocas
texto = "el hub de innovacion digital ayuda a las empresas a adoptar inteligencia artificial " * 30

# Vocabulario: cada caracter distinto del texto se convierte en un indice entero
caracteres = sorted(set(texto))
char_a_idx = {c: i for i, c in enumerate(caracteres)}
idx_a_char = {i: c for i, c in enumerate(caracteres)}
tam_vocab = len(caracteres)

# Codificamos todo el texto como un tensor de indices enteros
datos = torch.tensor([char_a_idx[c] for c in texto], dtype=torch.long)

longitud_secuencia = 40

def crear_lote(datos, longitud_secuencia, tam_lote):
    # Elegimos puntos de inicio aleatorios y recortamos ventanas de longitud fija;
    # el objetivo de cada ventana es la misma ventana desplazada un caracter
    indices_inicio = torch.randint(0, len(datos) - longitud_secuencia - 1, (tam_lote,))
    entradas = torch.stack([datos[i:i + longitud_secuencia] for i in indices_inicio])
    objetivos = torch.stack([datos[i + 1:i + longitud_secuencia + 1] for i in indices_inicio])
    return entradas, objetivos

class ModeloCharLSTM(nn.Module):
    def __init__(self, tam_vocab, tam_embedding=16, tam_oculto=64, num_capas=1):
        super().__init__()
        self.embedding = nn.Embedding(tam_vocab, tam_embedding)
        self.lstm = nn.LSTM(tam_embedding, tam_oculto, num_layers=num_capas, batch_first=True)
        self.salida = nn.Linear(tam_oculto, tam_vocab)

    def forward(self, x, estado=None):
        x = self.embedding(x)
        # estado es la tupla (h, c); si se pasa None, PyTorch la inicializa en ceros
        salida_lstm, estado = self.lstm(x, estado)
        logits = self.salida(salida_lstm)
        return logits, estado

modelo = ModeloCharLSTM(tam_vocab)
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.003)

num_epocas = 300
tam_lote = 32

for epoca in range(num_epocas):
    entradas, objetivos = crear_lote(datos, longitud_secuencia, tam_lote)
    logits, _ = modelo(entradas)
    # reorganizamos (lote, tiempo, vocab) en (lote*tiempo, vocab) para tratar
    # cada posicion de cada secuencia como un ejemplo de clasificacion independiente
    perdida = F.cross_entropy(logits.reshape(-1, tam_vocab), objetivos.reshape(-1))
    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()
    if (epoca + 1) % 50 == 0:
        print(f"Epoca {epoca + 1}/{num_epocas} - perdida: {perdida.item():.4f}")

@torch.no_grad()
def generar(modelo, texto_semilla, longitud=120, temperatura=0.8):
    modelo.eval()
    entrada = torch.tensor([[char_a_idx[c] for c in texto_semilla]], dtype=torch.long)
    _, estado = modelo(entrada)  # procesamos la semilla para construir el estado inicial
    ultimo_caracter = entrada[:, -1:]
    generado = texto_semilla
    for _ in range(longitud):
        logits, estado = modelo(ultimo_caracter, estado)
        # la temperatura suaviza (>1) o afila (<1) la distribucion antes de muestrear
        probabilidades = F.softmax(logits[:, -1, :] / temperatura, dim=-1)
        idx_siguiente = torch.multinomial(probabilidades, num_samples=1)
        generado += idx_a_char[idx_siguiente.item()]
        ultimo_caracter = idx_siguiente
    modelo.train()
    return generado

print(generar(modelo, "el hub "))
```

Dopo le 300 epoche, la perdita dovrebbe scendere in modo chiaro (da un valore iniziale vicino al logaritmo della dimensione del vocabolario fino a un valore molto più basso), e il testo generato dovrebbe assomigliare sempre di più al pattern di addestramento, con parole riconoscibili anche se il corpus è minuscolo. **Esegui il blocco completo in Colab e osserva come cambia il testo generato a ogni tentativo**, cambiando il seme di testo iniziale o la temperatura di campionamento.

## Riepilogo dei punti chiave

- Una sequenza è una collezione ordinata di elementi di lunghezza variabile; testo, audio, serie temporali e video ne sono esempi comuni
- I modelli feedforward come l'MLP richiedono ingressi di dimensione fissa e non catturano dipendenze di ordine; le CNN catturano schemi spaziali locali, ma non sono state progettate per la memoria a lungo termine nel tempo
- La rete neurale ricorrente (RNN) mantiene uno stato nascosto che si aggiorna a ogni passo mediante hₜ = tanh(Wₕ · hₜ₋₁ + Wₓ · xₜ + bₕ), con gli stessi pesi condivisi in tutti i passi temporali
- L'addestramento di una RNN usa la retropropagazione attraverso il tempo (BPTT), che srotola la rete e accumula i gradienti di ogni passo
- Le RNN semplici soffrono del problema del gradiente che svanisce o esplode: in sequenze lunghe, il segnale di errore si attenua o esplode in modo esponenziale, impedendo di apprendere dipendenze lontane
- La LSTM introduce uno stato di cella con aggiornamento per lo più additivo e tre porte (oblio, ingresso, uscita) che regolano quali informazioni conservare, aggiungere o esporre
- La GRU semplifica la LSTM fondendo lo stato di cella e quello nascosto, con solo due porte (aggiornamento e reset), a costo di un po' di flessibilità ma con meno parametri
- Entrambe le architetture attenuano la scomparsa del gradiente perché permettono all'informazione di fluire con cambiamenti minimi attraverso molti passi temporali quando le porte corrispondenti lo determinano
- RNN, LSTM e GRU si usano in modellazione del linguaggio, traduzione automatica, generazione di testo e immagini, sintesi audio e apprendimento per rinforzo con osservazioni sequenziali
- I transformatori, basati sull'attenzione, hanno superato in molti compiti di NLP le architetture ricorrenti, ma ereditano concetti chiave che hai appreso in questo capitolo, come la rappresentazione del contesto e l'addestramento per massima verosimiglianza

## Mettilo in pratica

Dedica tra quindici e trenta minuti a questa mini-attività, partendo dal codice della sezione precedente:

1. **Copia il blocco di codice in un notebook di Google Colab** ed eseguilo così com'è, per verificare che si addestri senza errori e che generi testo riconoscibile
2. **Sostituisci `nn.LSTM` con `nn.GRU`** nella classe `ModeloCharLSTM` (puoi lasciare il nome della classe uguale, o rinominarla) e riaddestra con lo stesso numero di epoche; annota la perdita finale di ciascuna architettura e confrontale
3. **Cambia `tam_oculto` da 64 a 8** e riaddestra la versione LSTM; osserva quanto peggiora la perdita finale e che aspetto ha il testo generato, per verificare in pratica come la dimensione dello stato nascosto limita la capacità del modello
4. **Sostituisci il testo di addestramento** con un frammento tuo, di almeno 200 caratteri, e ripeti l'addestramento; annota quante epoche servono al modello per iniziare a produrre parole riconoscibili del tuo nuovo testo
5. Come sfida facoltativa, aggiungi un secondo strato alla LSTM (`num_layers=2`) e verifica se migliora o peggiora la perdita finale con questo corpus così piccolo; rifletti sul perché un corpus minuscolo potrebbe non trarre vantaggio da più profondità

Conserva i tuoi risultati (perdite ed esempi di testo generato) perché ti serviranno come riferimento quando, nei capitoli successivi, confronterai queste architetture ricorrenti con i meccanismi di attenzione e i transformatori.
