# Capitolo 1: Introduzione al deep learning avanzato

## Perché questo tema è importante

Immagina un programma che impara a giocare a scacchi senza che nessuno gli spieghi una sola regola strategica, giocando solo contro se stesso, partita dopo partita. In quattro ore di addestramento, quel programma supera il miglior motore di scacchi del mondo dell'epoca, un sistema chiamato Stockfish che era stato messo a punto a mano da esperti per decenni. Quel programma si chiama AlphaZero, ed è uno dei traguardi che comprenderai a fondo in questo manuale.

Oppure immagina un problema molto più antico e molto più importante per la vita quotidiana: prevedere la forma tridimensionale che assume una proteina a partire dalla sequenza di amminoacidi che la compone. Questo problema è noto come "il paradosso di Levinthal": una proteina di appena 361 amminoacidi ha, in teoria, circa 10¹⁷² configurazioni possibili. Se quella proteina provasse mille miliardi di configurazioni al secondo, impiegherebbe 10¹⁵² anni per esplorarle tutte, molto più dell'età dell'universo. Eppure, nella cellula, la proteina si ripiega correttamente in questione di secondi. Per decenni, prevedere questo ripiegamento è stato uno dei grandi problemi irrisolti della biologia. Il sistema AlphaFold, sviluppato da DeepMind e presentato nel 2020, è riuscito a prevedere strutture di proteine con una precisione vicina a quella dei metodi sperimentali, un risultato che la biologia inseguiva dagli anni sessanta. Questo progresso sta accelerando la progettazione di farmaci e la comprensione delle malattie.

Questi due casi, uno ludico e l'altro biomedico, hanno qualcosa in comune: entrambi si risolvono con il deep learning, o apprendimento profondo. E entrambi sono frutto dagli stessi componenti tecnici che studierai lungo questo manuale: reti neurali profonde, reti convoluzionali, meccanismi di attenzione, apprendimento per rinforzo e modelli generativi.

Non serve lavorare in un laboratorio di ricerca perché questo ti interessi. In DIH DATAlife collaboriamo con aziende delle catene del valore Agro-Mar Alimentazione, Forestale-Legno, Salute-Cura e Biotecnologia, e in tutte il deep learning avanzato è già applicato in modo concreto: modelli di visione artificiale che rilevano malattie nelle colture o difetti nel legno, sistemi che prevedono serie temporali di produzione, o modelli generativi che aiutano a progettare nuovi materiali o composti. Comprendere la mappa completa di queste tecniche, anche senza padroneggiare ogni dettaglio matematico, ti permette di decidere quale strumento si adatta a quale problema e di parlare con cognizione di causa con chi le implementa.

Questo primo capitolo non ti insegnerà a programmare per intero una rete neurale profonda; quello lo farai poco a poco lungo il corso. Il suo obiettivo è un altro: darti la mappa completa del territorio, così che quando arriverai ai capitoli successivi tu sappia già perché esiste ogni componente e quale problema risolve.

## Spiegazione concettuale con analogie

Prima di entrare nei tecnicismi, costruiamo un'intuizione semplice.

### Che cos'è un neurone artificiale?

Pensa a una giuria composta da una sola persona che deve decidere se approvare o meno una richiesta di prestito. Quella persona riceve diversi dati: il reddito del richiedente, la sua storia dei pagamenti, la sua età, i suoi debiti attuali. Non dà a tutti i dati la stessa importanza: magari la storia dei pagamenti pesa molto, e l'età quasi nulla. Somma questi elementi, ciascuno moltiplicato per il proprio peso di importanza, e se il risultato supera una certa soglia, approva il prestito; altrimenti lo respinge.

Questo è, in sostanza, un neurone artificiale: riceve diversi input, li moltiplica per dei pesi che riflettono la loro importanza, somma tutto e applica una regola di decisione finale (ciò che nel deep learning chiamiamo funzione di attivazione) per produrre un output.

### Che cos'è una rete neurale profonda?

Ora immagina che, invece di una sola giuria, ci siano diverse file di giurie. La prima fila riceve i dati grezzi e prende decisioni semplici: "il reddito è ragionevole?", "la storia è pulita?". La seconda fila non vede i dati originali, ma le decisioni della prima fila, e le combina per arrivare a conclusioni più complesse: "è un profilo a basso rischio?". Una terza fila combina queste conclusioni intermedie nella decisione finale.

Ogni fila è uno strato. Quando impiliamo molti strati, la rete apprende rappresentazioni gerarchiche: i primi strati catturano schemi semplici e quelli più profondi combinano questi schemi semplici in concetti sempre più astratti. È da questa sovrapposizione di strati che deriva il nome "apprendimento profondo" (deep learning): "profondo" non parla di sofisticazione filosofica, ma letteralmente del numero di strati impilati.

### Perché "avanzato"?

Nel capitolo 2 studierai la rete neurale più semplice: la rete neurale *feedforward* (prealimentata), dove l'informazione scorre in una sola direzione, dall'ingresso all'uscita, senza cicli. È un buon punto di partenza, ma presenta limitazioni serie di fronte a certi tipi di dati.

Pensa a tre analogie in più:

- Riconoscere un gatto in una fotografia è diverso dal leggere un elenco di numeri in un foglio di calcolo, perché un'immagine ha una struttura spaziale: un occhio vicino a un altro occhio, un orecchio vicino alla testa. Le reti neurali convoluzionali (CNN) sono progettate per sfruttare questa struttura spaziale, proprio come chi riconosce un volto non analizza ogni pixel in modo isolato, ma cerca schemi locali (bordi, texture) e li combina in forme più complesse.
- Capire una frase è diverso dal classificare una foto isolata, perché una frase è una sequenza in cui ordine e contesto contano: "il cane insegue il gatto" non significa la stessa cosa di "il gatto insegue il cane". Le reti neurali ricorrenti (RNN) e i meccanismi di attenzione sono progettati per cogliere la dipendenza tra elementi che compaiono in momenti diversi di una sequenza.
- Generare un'immagine nuova e realistica è diverso dal classificare un'immagine esistente, perché bisogna creare qualcosa che non esisteva, non solo riconoscere qualcosa che è già lì. I modelli generativi, come le reti generative antagoniste (GAN) o gli autocodificatori variazionali (VAE), sono progettati proprio per questo compito di creazione.

Il deep learning avanzato è, fondamentalmente, la famiglia di architetture e tecniche che adattano l'idea di base del neurone e della rete a strati a ciascun tipo di problema: immagini, sequenze, generazione di dati nuovi e apprendimento con poche informazioni etichettate.

## Sviluppo teorico

### Il neurone artificiale e la rete neurale

L'unità di base di qualsiasi rete neurale artificiale (ANN) è il neurone. Formalmente, un neurone riceve un vettore di input x₁, x₂, ..., xₙ, ciascuno associato a un peso w₁, w₂, ..., wₙ che rappresenta la sua importanza relativa. Il neurone calcola innanzitutto una somma pesata:

z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

dove b è un termine chiamato bias (sesgo o soglia), che permette di spostare la soglia di decisione. Successivamente, questa somma z passa attraverso una funzione di attivazione f, che introduce non linearità e produce l'output del neurone:

output = f(z)

Senza quella funzione di attivazione, impilare strati non servirebbe a nulla: una catena di operazioni puramente lineari resta, nel complesso, un'operazione lineare, per quanti strati tu metta. Le funzioni di attivazione più comuni sono la sigmoide, la tangente iperbolica e, soprattutto nelle reti profonde moderne, l'unità lineare rettificata (ReLU), che lascia semplicemente passare i valori positivi e azzera quelli negativi.

Una rete neurale si organizza in strati: uno strato di ingresso, uno o più strati nascosti e uno strato di uscita. Quando il numero di strati nascosti è elevato, parliamo di rete profonda, ed è da qui che deriva il nome della disciplina.

Un dato storico importante per capire perché la profondità conta: le prime reti a un solo strato (i cosiddetti percettroni) non riuscivano a risolvere problemi apparentemente semplici come la funzione logica XOR (che restituisce vero quando esattamente uno dei due ingressi è vero, e falso in tutti gli altri casi). Questo limite provocò quello che è noto come il primo "inverno" delle reti neurali, un periodo di scarso interesse e di scarsi finanziamenti per la disciplina. Aggiungere strati supplementari risolse questo limite e aprì la porta alle architetture profonde che usiamo oggi.

### Come impara una rete: la retropropagazione

Una rete neurale non nasce sapendo nulla: i suoi pesi partono da valori casuali. Imparare significa regolare quei pesi affinché la rete produca output sempre più simili a quelli desiderati.

Questo aggiustamento si misura con una funzione di perdita (*loss function*), che quantifica la differenza tra ciò che la rete predice e il risultato reale atteso. L'obiettivo dell'addestramento è minimizzare quella perdita.

L'algoritmo standard per ottenerlo è la retropropagazione (*backpropagation*), combinata con la discesa del gradiente. L'idea, senza entrare nel dettaglio matematico che vedrai nel capitolo 5, è questa: si calcola quanto ciascun peso contribuisce all'errore finale (usando la regola della catena del calcolo differenziale) e si aggiorna ogni peso nella direzione che riduce quell'errore. Questo processo si ripete migliaia o milioni di volte, con piccoli lotti di dati, in varianti della discesa del gradiente stocastica (SGD) come Adam o RMSprop, che regolano automaticamente quanto si muove ogni peso a ogni passo.

### Reti convoluzionali: sfruttare la struttura spaziale

Le reti neurali convoluzionali (CNN), rese popolari da Yann LeCun con l'architettura LeNet-5 per riconoscere cifre scritte a mano, introducono un'idea chiave: invece di collegare ogni neurone a tutti i pixel dell'immagine (cosa che farebbe esplodere il numero di parametri), applicano filtri piccoli che scorrono sull'immagine, rilevando schemi locali come bordi, texture o forme, a diversi livelli di astrazione a seconda della profondità dello strato.

Formalmente, l'operazione di convoluzione bidimensionale, applicata a un input I con un filtro (o kernel) K, si definisce come:

(I * K)(i, j) = ΣₘΣₙ I(i+m, j+n) · K(m, n)

In termini semplici: il filtro K percorre l'immagine I e, in ogni posizione (i, j), moltiplica i valori del filtro per i valori dell'immagine che ha sotto di sé, e somma il risultato. Quel singolo filtro, riutilizzato su tutta l'immagine, incorpora una proprietà molto preziosa chiamata invarianza alla traslazione: un gatto resta un gatto anche se appare nell'angolo in alto a sinistra della foto o al centro, perché lo stesso filtro che rileva "orecchie appuntite" viene applicato in qualsiasi posizione. Questo riutilizzo riduce drasticamente il numero di parametri rispetto a una rete completamente connessa e rende l'addestramento molto più efficiente. Il capitolo 3 approfondisce questa architettura.

### Reti ricorrenti: sfruttare l'ordine e il tempo

Quando i dati sono sequenziali (testo, voce, serie temporali, sequenze di DNA, video), l'ordine conta e il contesto passato condiziona il presente. Le reti neurali ricorrenti (RNN) sono progettate proprio per questo: mantengono uno stato nascosto che si aggiorna a ogni passo della sequenza e che riassume l'informazione rilevante di ciò che è stato visto fino a quel momento.

Il problema classico delle RNN semplici è la scomparsa del gradiente (*vanishing gradient*): quando la sequenza è lunga, il segnale di errore si attenua così tanto propagandosi all'indietro che la rete smette di apprendere dipendenze lontane nel tempo. Per risolverlo sono state progettate architetture con meccanismi di porte che regolano quale informazione conservare e quale dimenticare: le memorie a breve e lungo termine (LSTM) e le unità ricorrenti con porte (GRU). Il capitolo 6 entra nel dettaglio di queste architetture.

### Attenzione: guardare solo ciò che conta

Un meccanismo più recente, e oggi centrale nell'elaborazione del linguaggio naturale (NLP), è l'attenzione. L'idea, in termini intuitivi, è che la rete impara a "mettere il focus" sulle parti più rilevanti dell'input per risolvere un compito concreto, invece di trattare tutto l'input con la stessa importanza. Questo meccanismo ha dato origine ad architetture come il trasformatore (*Transformer*), alla base di modelli come BERT o GPT, che vedrai nei capitoli 7 e 8. Esistono anche modelli con memoria esterna esplicita, come la macchina di Turing neurale, capaci di consultare e memorizzare informazioni in modo simile a come una persona consulta una mappa per rispondere a una domanda su un percorso.

### Modelli generativi: creare dati nuovi

Finora abbiamo parlato di modelli che classificano o predicono a partire da dati esistenti. I modelli generativi perseguono un obiettivo diverso: apprendere la distribuzione di un insieme di dati per poter generare esempi nuovi e plausibili.

Le reti generative antagoniste (GAN), proposte da Ian Goodfellow nel 2014, addestrano simultaneamente due reti che competono tra loro: un generatore, che cerca di produrre dati realistici (per esempio immagini), e un discriminatore, che cerca di distinguere i dati generati dai dati reali. Questo addestramento si imposta come un gioco a somma zero, formalizzato come un problema minimax:

mín_G máx_D V(D, G) = 𝔼_{x∼p_datos(x)}[log D(x)] + 𝔼_{z∼p_z(z)}[log(1 − D(G(z)))]

Qui, G è il generatore, D è il discriminatore, p_datos è la distribuzione dei dati reali e p_z è la distribuzione dello spazio latente da cui il generatore prende i suoi campioni di partenza. Il discriminatore D cerca di massimizzare la propria capacità di distinguere il reale dal generato, mentre il generatore G cerca di minimizzare quella stessa capacità, ossia di ingannare il discriminatore. L'addestramento prosegue finché il generatore produce dati che il discriminatore non riesce più a distinguere in modo affidabile dai dati reali. Studierai questa architettura in profondità nel capitolo 11.

Un'altra famiglia importante è quella degli autocodificatori variazionali (VAE), un modello di apprendimento non supervisionato che apprende variabili latenti capaci di spiegare i dati osservati. Per esempio, addestrato con immagini di sedie viste da diverse angolazioni, un VAE può arrivare a scoprire da solo fattori come la rotazione, la larghezza o lo stile delle gambe, senza che nessuno gli indichi esplicitamente quelle categorie. Lo vedrai nel capitolo 9.

### Apprendimento per trasferimento e apprendimento per rinforzo

L'apprendimento per trasferimento (*transfer learning*) consiste nel partire da un modello già addestrato su un insieme di dati ampio e adattarlo (*fine-tuning*) a un compito più specifico con meno dati disponibili. Questo è particolarmente prezioso in ambiti dove i dati etichettati scarseggiano, come accade in biologia o in applicazioni industriali molto specifiche.

L'apprendimento per rinforzo (RL), dal canto suo, è un quadro generale in cui un agente interagisce con un ambiente: osserva il proprio stato, esegue azioni e riceve una ricompensa. L'obiettivo dell'agente è apprendere una politica che massimizzi la ricompensa accumulata nel lungo periodo, non solo quella immediata. Questo quadro è abbastanza generale da includere anche l'apprendimento supervisionato e quello non supervisionato come casi particolari. I casi di AlphaGo, AlphaZero e il progetto "Capture the Flag" che abbiamo menzionato all'inizio sono esempi di deep learning combinato con apprendimento per rinforzo: la politica decisionale (quale mossa fare, quale azione intraprendere) è rappresentata da una rete neurale profonda, addestrata giocando contro se stessa, un processo noto come autoapprendimento (*self-play*).

Un dettaglio istruttivo del caso AlphaZero: mentre un motore di scacchi tradizionale come Stockfish valuta decine di milioni di posizioni prima di decidere una mossa, AlphaZero ne valuta solo decine di migliaia, una riduzione di mille volte, uno schema di ricerca molto più simile all'intuizione di un grande maestro umano, che valuta appena qualche centinaio di posizioni.

### Regolarizzazione: evitare il sovradattamento

Un rischio costante nell'addestramento di reti profonde è il sovradattamento (*overfitting*): il modello memorizza i dati di addestramento invece di apprendere schemi generali, e quindi funziona male davanti a dati nuovi. Le tecniche di regolarizzazione contrastano questo rischio. Tra le più comuni:

- Abbandono (*dropout*): a ogni passo di addestramento si ignora casualmente un sottoinsieme di neuroni, il che costringe la rete a non dipendere eccessivamente da nessun neurone specifico
- Decadimento dei pesi (*weight decay*): aggiunge alla funzione di perdita una penalità proporzionale all'ampiezza dei pesi, favorendo modelli più semplici
- Aumento dei dati (*data augmentation*): genera esempi di addestramento aggiuntivi applicando trasformazioni come rotazioni o ridimensionamenti ai dati esistenti

### Apprendimento responsabile

Nessuna mappa del deep learning avanzato è completa senza affrontare l'innovazione responsabile. Più grande è il potere di un sistema di intelligenza artificiale (IA), più grande è anche la responsabilità che ne deriva. Due problemi tecnici ed etici emergono in modo ricorrente:

- Gli esempi avversari (*adversarial examples*): piccole modifiche, quasi impercettibili all'occhio umano, nei dati di ingresso possono far sbagliare in modo drastico un classificatore di immagini, per esempio scambiando un cervo per un uccello
- Il bias ereditato dai dati: un modello addestrato con dati che riflettono disuguaglianze storiche può riprodurre e amplificare quelle stesse disuguaglianze nelle sue previsioni

Costruire sistemi sicuri, robusti e verificati, e anticiparne le conseguenze sociali ed economiche (come l'impatto sull'occupazione), fa parte integrante della pratica professionale del deep learning, non è un'aggiunta facoltativa. Questo manuale dedica il suo capitolo finale a questo tema.

## Esempio di codice commentato

Prima che il capitolo 2 introduca PyTorch, costruiamo il pezzo più piccolo di tutto questo edificio: un singolo neurone artificiale, usando solo la libreria NumPy. L'obiettivo è che tu veda, con le tue mani, come una somma pesata più una funzione di attivazione possano prendere una decisione semplice. Copia questo codice in una cella di Google Colab ed eseguilo così com'è.

```python
import numpy as np

# --- 1. Definimos la neurona ---
# Una neurona recibe entradas (x), las multiplica por sus pesos (w),
# suma un sesgo (b) y pasa el resultado por una función de activación.

def sigmoide(z):
    """Función de activación sigmoide: aplasta cualquier número real
    al rango (0, 1), útil para interpretar la salida como una probabilidad."""
    return 1 / (1 + np.exp(-z))

def neurona(x, w, b):
    """Calcula la salida de una única neurona artificial.
    x: vector de entradas (numpy array)
    w: vector de pesos, mismo tamaño que x
    b: sesgo (número escalar)
    """
    z = np.dot(x, w) + b   # suma ponderada: w1*x1 + w2*x2 + ... + b
    return sigmoide(z)

# --- 2. Ejemplo: ¿aprobamos un préstamo? ---
# Dos entradas normalizadas entre 0 y 1:
#   x1 = nivel de ingresos (0 = bajo, 1 = alto)
#   x2 = historial de pagos (0 = malo, 1 = excelente)
x = np.array([0.8, 0.9])

# Pesos: el historial de pagos (w2) pesa más que los ingresos (w1)
w = np.array([0.4, 1.2])
b = -0.9  # sesgo: hace más difícil aprobar por defecto

salida = neurona(x, w, b)
print(f"Salida de la neurona: {salida:.3f}")
print("Decisión:", "aprobado" if salida > 0.5 else "rechazado")

# --- 3. Probemos con un perfil de mayor riesgo ---
x_riesgo = np.array([0.2, 0.3])
salida_riesgo = neurona(x_riesgo, w, b)
print(f"\nSalida con perfil de riesgo: {salida_riesgo:.3f}")
print("Decisión:", "aprobado" if salida_riesgo > 0.5 else "rechazado")

# --- 4. Una "capa" es simplemente varias neuronas en paralelo ---
# Cada fila de la matriz W son los pesos de una neurona distinta.
def capa(x, W, b):
    """W: matriz de pesos (n_neuronas x n_entradas)
    b: vector de sesgos (uno por neurona)
    Devuelve un vector con la salida de cada neurona de la capa."""
    z = W @ x + b           # producto matricial + suma de sesgos
    return sigmoide(z)

W_capa = np.array([
    [0.4, 1.2],   # pesos de la neurona 1 (como antes)
    [1.0, -0.5],  # pesos de la neurona 2 (prioriza ingresos altos)
])
b_capa = np.array([-0.9, -0.1])

salidas_capa = capa(x, W_capa, b_capa)
print(f"\nSalidas de una capa con dos neuronas: {salidas_capa}")
```

**Provalo!** Cambia los valores de `w` y `b`, o añade una tercera entrada (por ejemplo, la edad del solicitante) y observa cómo cambia la decisión final. Este pequeño bloque, "suma ponderada más activación", es literalmente la pieza que se repite miles o millones de veces en una red profunda: lo único que cambia entre este ejemplo y AlphaFold es la cantidad de neuronas, capas y datos de entrenamiento, no la idea de fondo.

## Riepilogo dei punti chiave

- Il deep learning è un sottocampo dell'apprendimento automatico (machine learning) che usa reti neurali con molti strati per apprendere rappresentazioni gerarchiche direttamente da dati grezzi, senza bisogno di progettare a mano le caratteristiche (*feature engineering*)
- L'unità di base è il neurone artificiale: somma pesata degli input più bias, seguita da una funzione di attivazione non lineare
- Le reti neurali convoluzionali (CNN) incorporano la conoscenza a priori che i dati con struttura spaziale (come le immagini) traggono beneficio da filtri locali riutilizzabili
- Le reti neurali ricorrenti (RNN), e le loro varianti LSTM e GRU, sono progettate per dati sequenziali in cui l'ordine e il contesto temporale contano
- I meccanismi di attenzione permettono a una rete di concentrarsi sulle parti più rilevanti dell'input, e sono alla base di architetture come il trasformatore (*Transformer*)
- I modelli generativi, come le GAN e i VAE, imparano a creare dati nuovi e plausibili a partire dalla distribuzione di un insieme di dati
- L'apprendimento per trasferimento riutilizza modelli preaddestrati per compiti con pochi dati disponibili; l'apprendimento per rinforzo addestra agenti che massimizzano ricompense nel lungo periodo attraverso l'interazione con un ambiente
- L'addestramento si basa sulla minimizzazione di una funzione di perdita tramite retropropagazione e discesa del gradiente, con tecniche di regolarizzazione (dropout, decadimento dei pesi, aumento dei dati) per evitare il sovradattamento
- Casi reali come AlphaGo, AlphaZero, il progetto Capture the Flag e AlphaFold mostrano che questi componenti tecnici, combinati, risolvono problemi che prima si ritenevano inaffrontabili
- L'apprendimento responsabile (robustezza di fronte agli esempi avversari, gestione del bias, impatto sociale) è parte integrante della disciplina, non un'aggiunta successiva

## Mettilo in pratica

Dedica tra quindici e trenta minuti a questa mini-attività in Google Colab, ampliando il codice della sezione precedente.

1. Apri un nuovo notebook in Google Colab e copia il codice della sezione precedente
2. Definisci il tuo scenario decisionale con tre input invece di due (per esempio: ore di sonno, ore di studio e livello di stress, per prevedere se "supereresti un esame")
3. Scegli valori di pesi e bias che riflettano la tua intuizione sull'importanza di ciascun fattore
4. Prova almeno tre combinazioni diverse di input e osserva come cambia l'output del neurone
5. Modifica la funzione `sigmoide` sostituendola con una semplice funzione a gradino (`return 1 if z > 0 else 0`) e confronta i risultati: quali informazioni perdi usando una decisione binaria invece di un valore continuo tra zero e uno?
6. Scrivi, in un paio di frasi all'interno di una cella di testo del notebook, quale ruolo avrebbero i pesi se, invece di impostarli a mano, li regolasse automaticamente un algoritmo di retropropagazione a partire da migliaia di esempi reali

Al termine di questa mini-attività, avrai manipolato con le tue mani il componente fondamentale che, moltiplicato per milioni, costruisce sistemi come AlphaFold. Il capitolo 2 riprende questa stessa idea, il neurone e lo strato, per mostrarti come si addestra una rete completa con PyTorch.
