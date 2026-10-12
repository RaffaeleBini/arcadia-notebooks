# Capitolo 2 — Fondamenti delle reti neurali: il neurone, il perceptron multistrato e la backpropagation

## Perché questo tema è importante

Nel capitolo uno hai visto il quadro generale del deep learning (apprendimento profondo, DL): quali problemi risolve, perché è esploso a partire dal 2006 e quale ruolo giocano la potenza di calcolo e i dati. Ora è il momento di aprire il motore e guardarci dentro. Tutto ciò che costruirai in questo corso —reti convoluzionali, reti ricorrenti, meccanismi di attenzione, modelli generativi— si appoggia su quattro idee che impari oggi: il neurone artificiale, l'impilamento in strati, la funzione di perdita e l'algoritmo che regola i pesi, la backpropagation (retropropagazione dell'errore), combinata con la discesa del gradiente.

Se capisci bene questo capitolo, capisci il "nucleo" di tutto il deep learning. Se lo salti o lo memorizzi senza comprenderlo, ogni capitolo successivo ti costerà il triplo, perché tutti presuppongono che tu sappia cos'è un peso, cos'è un'attivazione e perché la rete "impara" ricalcolando i gradienti. Inoltre, qui scrivi il tuo primo codice vero: prima la backpropagation a mano, con NumPy, per vedere il meccanismo senza trucchi; poi la stessa operazione con PyTorch, il framework che userai per il resto del corso.

Padroneggiare questo tema ti serve anche fuori dall'aula. Se lavori o lavorerai con team di digitalizzazione, innovazione o dati, comprendere il meccanismo interno di una rete neurale ti permette di distinguere un progetto di intelligenza artificiale (IA) ben impostato da una promessa vuota avvolta in parole di moda.

## Spiegazione concettuale con analogie (prima dei tecnicismi)

Immagina un panel di votazione con diversi esperti. Ogni esperto osserva una parte diversa dello stesso problema —per esempio, se una foto contiene un cane— ed esprime un voto con un peso di fiducia: alcuni esperti sono più affidabili (peso alto), altri contribuiscono poco o addirittura depistano (peso basso o negativo, perché la loro opinione va di solito in senso contrario). Alla fine, qualcuno somma tutti i voti ponderati, aggiunge un piccolo aggiustamento di calibrazione (il bias o *bias*, cioè la soglia) e decide se il risultato complessivo supera una certa soglia. Questo, in sostanza, è un neurone artificiale: una somma ponderata di input, più un aggiustamento, che produce un segnale di output.

Ora immagina di non avere un solo panel, ma diversi panel in cascata. Il primo panel di esperti analizza i pixel grezzi ed emette segnali intermedi (per esempio, "qui c'è un bordo", "qui c'è una zona scura"). Questi segnali alimentano un secondo panel, che li combina per rilevare forme più complesse ("questo sembra un orecchio", "questo sembra un muso"). Un terzo panel combina queste forme per decidere la categoria finale. Ogni panel è uno strato; l'insieme completo è un perceptron multistrato, l'architettura più semplice di rete neurale profonda.

Come fanno gli esperti a sapere quale peso assegnare a ogni opinione? All'inizio non lo sanno: partono con pesi casuali e sbagliano parecchio. Ma dopo ogni previsione confrontano il proprio verdetto con la risposta corretta e calcolano quanto hanno sbagliato (la perdita o *loss*). Poi arriva la parte chiave: percorrono il panel all'indietro, dal risultato finale fino al primo esperto, e dicono a ciascuno "il tuo voto ha contribuito tanto a questo errore; correggi un po' il tuo peso in questa direzione". Quel percorso all'indietro è la backpropagation, e l'aggiustamento graduale, passo dopo passo, è la discesa del gradiente. Con migliaia di ripetizioni, i pesi si affinano fino a quando l'intero panel indovina quasi sempre.

Un'ultima analogia utile: immagina di trovarti sul pendio di una montagna con nebbia fitta, senza vedere oltre la punta dei piedi, e di voler raggiungere il punto più basso della valle. La strategia ragionevole è percepire verso quale lato scende il terreno sotto i tuoi piedi e fare un passo in quella direzione. Ripeti il processo molte volte. Questo è esattamente la discesa del gradiente: la "pendenza" è il gradiente della funzione di perdita, e il "passo" è il tasso di apprendimento (*learning rate*).

## Sviluppo teorico

### Il neurone artificiale

Il modello di neurone artificiale che usiamo oggi ha origine nel lavoro di Warren McCulloch e Walter Pitts del 1943. Il loro obiettivo non era simulare fedelmente un neurone biologico, ma catturare alcune proprietà rilevanti: un neurone reale riceve segnali attraverso le sue dendriti, li combina nel corpo cellulare (soma) e trasmette un output attraverso l'assone; alcune connessioni eccitano l'attività e altre la inibiscono. Il modello artificiale traduce tutto ciò in matematica semplice: senza stato interno nel tempo, e con output a valori reali invece che impulsi elettrici.

Formalmente, un neurone artificiale riceve un vettore di input $x = (x_1, x_2, \dots, x_n)$, ciascuno associato a un peso $w_i$, calcola la somma pesata, vi aggiunge un termine di bias $b$ e applica una funzione di attivazione $f$:

$$z = \sum_{i=1}^{n} w_i x_i + b = w \cdot x + b$$

$$y = f(z)$$

Qui, $y$ è l'output del neurone, $f$ è la funzione di attivazione, $w$ sono i pesi, $x$ sono gli input e $b$ è il bias. Un peso positivo eccita l'input corrispondente (ne aumenta l'influenza); un peso negativo lo inibisce. Senza la funzione $f$, l'operazione completa sarebbe una semplice trasformazione affine (lineare più uno spostamento), e concatenare trasformazioni affini produce solo un'altra trasformazione affine: non guadagneresti nulla impilando strati. La funzione di attivazione è il componente che introduce la non linearità necessaria perché la rete possa rappresentare schemi complessi.

Nella comunità del deep learning, questa operazione riceve nomi diversi a seconda del gruppo di ricerca che la descrive: al neurone si dà anche il nome di unità; ai pesi, quello di parametri; e all'operazione completa (somma pesata più bias) si dà, in modo un po' impreciso, il nome di "strato lineare", anche se tecnicamente è una trasformazione affine, perché include il termine $b$. Se ti ricorda la regressione lineare della statistica, non è un caso: un singolo neurone con attivazione sigmoide e perdita di entropia incrociata è matematicamente equivalente alla regressione logistica.

### Dal neurone allo strato: l'operazione matriciale

Quando raggruppi più neuroni che ricevono gli stessi input, ciascuno con il proprio vettore di pesi, ottieni uno strato. Invece di calcolare ogni neurone separatamente, conviene esprimere l'operazione con matrici, perché è molto più efficiente dal punto di vista computazionale:

$$z = Wx + b$$

Qui $W$ è una matrice in cui ogni riga contiene i pesi di un neurone, $x$ è il vettore di input e $b$ è il vettore dei bias, uno per neurone. Questa forma matriciale è esattamente ciò che sfruttano le unità di elaborazione grafica (GPU) e le unità di elaborazione tensoriale (TPU): sono ottimizzate per eseguire moltiplicazioni di matrici in parallelo, che è l'operazione più costosa —e più frequente— all'interno di una rete neurale.

### Funzioni di attivazione

Passiamo in rassegna le funzioni di attivazione più comuni, tutte applicate elemento per elemento su ciascuna componente di $z$.

**Funzione sigmoide.** Comprime qualsiasi numero reale nell'intervallo (zero, uno):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Il suo output viene spesso interpretato come una probabilità. È derivabile in ogni punto, una proprietà importante per addestrare la rete, ma ha un limite serio: quando $z$ si allontana molto da zero (molto positivo o molto negativo), la curva si appiattisce quasi del tutto e la sua derivata tende a zero. Questo fenomeno si chiama saturazione, e provoca il problema dei gradienti che svaniscono (*vanishing gradients*): se la derivata è quasi nulla, la correzione che il neurone riceve durante l'addestramento è quasi nulla anch'essa, e la rete smette di apprendere in quella zona.

**Funzione tangente iperbolica (tanh).** Simile alla sigmoide, ma comprime i valori nell'intervallo (meno uno, uno), centrata sullo zero. Condivide lo stesso problema di saturazione agli estremi.

**Unità lineare rettificata (ReLU).** Oggi è l'attivazione più usata nelle reti profonde, per la sua semplicità e il suo buon comportamento durante l'addestramento:

$$\text{ReLU}(z) = \max(0, z)$$

Se l'input è positivo, lo lascia passare senza modifiche; se è negativo o zero, produce zero. La sua derivata è anch'essa estremamente semplice: uno per input positivi, zero per input negativi (in zero, tecnicamente non è derivabile, ma in pratica si ignora, perché la probabilità di cadere esattamente in quel punto è trascurabile). Essendo lineare a tratti, ReLU non satura dal lato positivo, quindi i gradienti non svaniscono come succede con la sigmoide. Il suo svantaggio è il problema dei "neuroni morti": se, per una cattiva inizializzazione o un aggiornamento sfortunato dei pesi, tutti gli input di un neurone ReLU diventano negativi, quel neurone produce sempre zero e smette di apprendere.

**Softmax.** Si usa quasi sempre nello strato di output quando il problema è di classificazione con più di due categorie. Converte un vettore di valori reali in una distribuzione di probabilità (tutti i valori non negativi e la somma uguale a uno):

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

dove $K$ è il numero di classi. È la generalizzazione multidimensionale della sigmoide: se applichi softmax a un problema a due classi, ottieni esattamente la sigmoide. Il suo punto debole è l'instabilità numerica quando i valori di $z$ sono grandi, perché l'esponenziale può far superare la precisione del computer; in pratica, le librerie la implementano sottraendo prima il massimo di $z$ a tutte le sue componenti, il che dà lo stesso risultato matematico in modo numericamente stabile.

### Il perceptron multistrato: impilare strati

Un perceptron multistrato (o rete neurale *feedforward*, perché i dati fluiscono in una sola direzione, senza cicli) impila diversi strati di questo tipo, uno dopo l'altro:

$$h^{(1)} = f^{(1)}\left(W^{(1)} x + b^{(1)}\right)$$
$$h^{(2)} = f^{(2)}\left(W^{(2)} h^{(1)} + b^{(2)}\right)$$
$$\dots$$
$$y = f^{(L)}\left(W^{(L)} h^{(L-1)} + b^{(L)}\right)$$

Gli strati intermedi si chiamano strati nascosti (*hidden layers*), perché il loro output non è direttamente il risultato che interessa, ma una rappresentazione intermedia. Quanti più strati nascosti impili, tanto più "profonda" è la rete: da qui il nome apprendimento profondo.

Un esempio classico che giustifica la necessità degli strati nascosti è il problema XOR (o esclusivo): un insieme di punti in due dimensioni in cui una classe occupa una diagonale e l'altra classe la diagonale opposta. Nessuna retta separa le due classi; un modello lineare semplice —un solo strato, senza attivazione non lineare intermedia— non riesce a risolverlo, per quanto sia accurata la regolazione dei suoi pesi. Bastano però due neuroni in uno strato nascosto con attivazione sigmoide per risolverlo: ciascun neurone agisce come una linea di proiezione che "piega" lo spazio di input, e dopo questa deformazione non lineare, una singola retta finale separa correttamente le classi. Questo esempio illustra l'idea centrale: lo strato nascosto trasforma lo spazio dei dati fino a renderlo separabile, e lo strato finale deve solo tracciare una frontiera semplice su quello spazio già trasformato.

Questo risultato si generalizza nel teorema di approssimazione universale, dimostrato da Cybenko nel 1989 ed esteso da Hornik nel 1991: una rete con un solo strato nascosto, con neuroni sufficienti e una funzione di attivazione non degenere (non costante, limitata e continua), può approssimare qualsiasi funzione continua con l'errore che desideri, su un dominio limitato. È un risultato importante, ma con due precisazioni pratiche: è una dimostrazione di esistenza, non ti dice come costruire quella rete né come trovare quei pesi; e il numero di neuroni necessari può crescere in modo esponenziale con la dimensione dell'input. È qui che entra in gioco la profondità: ricerche successive (Montúfar, Pascanu, Cho e Bengio, 2014) mostrano che il numero di regioni lineari che una rete con attivazioni ReLU può rappresentare cresce esponenzialmente con la profondità, ma solo in modo polinomiale con la larghezza. In pratica, è molto più efficiente aggiungere strati che allargare un unico strato nascosto fino all'infinito.

Nella visione artificiale, questa profondità si traduce di solito in una gerarchia di caratteristiche: i primi strati rilevano bordi e angoli semplici; gli strati intermedi combinano i bordi in forme; gli strati più profondi riconoscono parti di oggetti (un orecchio, un muso); e lo strato finale combina queste parti per decidere la categoria completa (per esempio, "cane").

### Funzioni di perdita

Una funzione di perdita (*loss function*) misura quanto la previsione della rete si discosti dal valore reale. È il termometro che guida tutto l'addestramento: senza una perdita ben definita, la rete non ha alcun segnale per regolare i pesi.

**Entropia incrociata (*cross-entropy*)**, chiamata anche log-verosimiglianza negativa (*negative log-likelihood*) o perdita logistica. È la più usata nei problemi di classificazione. Per un problema binario, con $y \in \{0, 1\}$ l'etichetta reale e $\hat{y}$ la probabilità prevista dalla rete:

$$\mathcal{L}(y, \hat{y}) = -\left[y \log(\hat{y}) + (1-y)\log(1-\hat{y})\right]$$

Per $N$ campioni e, nel caso generale, $K$ classi con codifica one-hot ($y_{i,k}$ vale uno se il campione $i$ appartiene alla classe $k$, e zero altrimenti), la formula si generalizza a:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})$$

Una proprietà chiave di questa perdita —e di circa il 95 % delle perdite usate nell'apprendimento automatico— è che è additiva rispetto ai campioni: la perdita totale è la somma delle perdite individuali di ciascun campione. Questa proprietà è ciò che rende praticabile l'addestramento a lotti (*batch*) e la discesa del gradiente stocastica, perché puoi approssimare il gradiente totale usando solo un sottoinsieme dei dati.

**Errore quadratico medio (MSE, *mean squared error*)**, usato soprattutto nei compiti di regressione, dove l'output è un valore continuo anziché una categoria:

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

Un dettaglio tecnico importante e molto pratico: quando combini softmax con entropia incrociata (o sigmoide con entropia incrociata binaria), il gradiente congiunto di quella combinazione si semplifica enormemente: si riduce alla differenza tra la probabilità prevista e l'etichetta reale ($\hat{y} - y$). Questa semplificazione evita divisioni per numeri molto piccoli, che sono una fonte frequente di instabilità numerica se calcoli ogni derivata separatamente. Per questo, le librerie di deep learning offrono quasi sempre una funzione combinata ("entropia incrociata con logit") invece di concatenare softmax e entropia incrociata come due passaggi indipendenti.

### Discesa del gradiente

L'obiettivo dell'addestramento è trovare i pesi $\theta$ (che raggruppano tutti i $W$ e i $b$ della rete) che minimizzano la perdita $\mathcal{L}(\theta)$. La discesa del gradiente è l'algoritmo standard per farlo. L'idea, come nell'analogia della montagna con la nebbia: calcola il gradiente della perdita rispetto ai parametri —un vettore che punta nella direzione di maggior crescita della perdita— e muovi i parametri nella direzione opposta, perché vuoi minimizzare, non massimizzare:

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t)$$

Qui $\eta$ (eta) è il tasso di apprendimento (*learning rate*), un iperparametro che controlla l'ampiezza del passo. Se $\eta$ è troppo grande, puoi saltare il minimo e oscillare senza convergere; se è troppo piccolo, l'addestramento avanza con una lentezza esasperante. In condizioni ragionevoli di regolarità della funzione, questo metodo garantisce la convergenza a un minimo locale.

In pratica, quasi mai si calcola il gradiente sull'intero insieme di dati a ogni passo, perché sarebbe estremamente costoso con dataset grandi. Si usa invece la discesa del gradiente stocastica (SGD, *stochastic gradient descent*): a ogni passo si calcola il gradiente su un piccolo sottoinsieme di dati (un *mini-batch*) anziché sull'intero insieme. Grazie alla proprietà additiva della perdita che abbiamo menzionato prima, questa resta un'approssimazione valida, e l'algoritmo converge comunque sotto certe condizioni sul rumore di quell'approssimazione.

A partire dalla discesa del gradiente si sono sviluppate varianti più sofisticate. La più usata come punto di partenza predefinito è Adam (*Adaptive Moment Estimation*), che combina due idee: un tasso di apprendimento adattivo per ogni parametro e un termine di momento che leviga gli aggiornamenti usando la storia dei gradienti precedenti. Adam mantiene due stime mobili, del primo momento ($m_t$, simile a una media dei gradienti recenti) e del secondo momento ($v_t$, simile alla varianza di quei gradienti), e le usa per scalare individualmente ogni aggiornamento per parametro. In pratica, Adam con un tasso di apprendimento di riferimento di 0,001 è di solito un ottimo punto di partenza per quasi qualsiasi architettura.

### Backpropagation: la regola della catena applicata all'intera rete

Sai già come si aggiornano i pesi una volta ottenuto il gradiente. La domanda che resta è: come calcoli quel gradiente in modo efficiente, quando la rete può avere milioni di parametri distribuiti in decine di strati? La risposta è la backpropagation (retropropagazione), un algoritmo riscoperto e reso popolare nel 1986 da David Rumelhart, Geoffrey Hinton e Ronald Williams, che applica sistematicamente la regola della catena del calcolo differenziale.

La regola della catena, nella sua forma più semplice, dice che se $y$ dipende da $u$, e $u$ dipende da $x$, allora:

$$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}$$

Una rete neurale è, precisamente, una lunga composizione di funzioni: input, strato lineare, attivazione, strato lineare, attivazione… fino alla perdita finale. Puoi rappresentarla come un grafo computazionale, dove ogni nodo è un'operazione matematica semplice (una somma, un prodotto, un'esponenziale) e ogni arco trasporta il flusso di dati tra le operazioni. La backpropagation percorre questo grafo in due fasi:

1. **Passaggio in avanti (*forward pass*).** Calcoli l'output della rete, strato per strato, dall'input fino alla perdita, memorizzando i valori intermedi che ti serviranno dopo.
2. **Passaggio all'indietro (*backward pass*).** Partendo dalla perdita, calcoli quanto contribuisce ciascun parametro all'errore finale, applicando la regola della catena strato per strato, dall'output verso l'input. Ogni operazione del grafo "sa" calcolare il proprio gradiente locale, e deve solo moltiplicarlo per il gradiente che arriva dallo strato successivo.

Questo secondo passaggio evita qualcosa di molto costoso: se dovessi ricalcolare l'effetto di ogni peso sulla perdita finale in modo indipendente (per esempio, perturbando un peso alla volta e osservando quanto cambia l'output), il costo crescerebbe con il numero di parametri. La backpropagation, invece, calcola tutti i gradienti in un unico percorso all'indietro, con un costo paragonabile a quello del passaggio in avanti.

Vediamo come si comporta ogni operazione tipica durante il passaggio all'indietro, perché questa intuizione è quella che ti serve per programmare a mano l'esempio della sezione seguente:

- **Strato lineare ($z = Wx + b$).** Il gradiente rispetto a $x$ e rispetto a $W$ è simmetrico grazie alla struttura del prodotto scalare; il gradiente rispetto a $b$ è semplicemente il gradiente che arriva dallo strato successivo, senza modifiche, perché $b$ si somma alla fine.
- **ReLU.** Agisce come una porta: se l'input era positivo nel passaggio in avanti, lascia passare il gradiente senza modifiche; se era negativo o zero, lo blocca (lo porta a zero).
- **Softmax seguito da entropia incrociata.** Come abbiamo visto, la combinazione si semplifica nella differenza tra la previsione e l'etichetta reale.
- **Massimo (usato, per esempio, nel *max pooling*).** Solo l'input che ha "vinto" la competizione riceve gradiente; il resto riceve zero. Questa operazione non impara a selezionare, si limita a correggere l'input già selezionato.
- **Perdita quadratica (MSE).** Il gradiente rispetto alla previsione è, di nuovo, semplicemente la differenza tra previsione e valore reale.

Nota uno schema: quasi in tutti i casi, il gradiente che "esce" da ogni nodo è facile da calcolare localmente, purché tu conservi i valori calcolati durante il passaggio in avanti. Questa è la chiave dell'efficienza dell'algoritmo, ed è esattamente ciò che programmerai a mano nella sezione seguente, sull'esempio più piccolo possibile: una rete con un unico peso.

## Esempio di codice commentato

Vedrai lo stesso esempio risolto due volte. Prima a mano, con NumPy, così vedi letteralmente ogni moltiplicazione e ogni derivata della backpropagation. Poi con PyTorch, per verificare che il framework calcoli esattamente lo stesso risultato, ma in modo automatico. Entrambi i frammenti sono autosufficienti e funzionano direttamente in Google Colab.

Il modello non potrebbe essere più semplice: un singolo neurone con un solo peso $w$ (senza bias, per non complicare l'esempio), attivazione sigmoide e perdita quadratica, addestrato con una sola coppia di input e obiettivo.

$$\hat{y} = \sigma(w \cdot x), \qquad \mathcal{L} = (\hat{y} - y_{\text{objetivo}})^2$$

### Parte (a): backpropagation a mano, con NumPy

```python
import numpy as np

# Datos del ejemplo: una entrada y un objetivo fijos
x = 2.0
y_objetivo = 1.0

# Peso inicial (normalmente sería aleatorio; aquí lo fijamos para poder verificar a mano)
w = 0.3

# Tasa de aprendizaje
eta = 0.1

def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))

def derivada_sigmoide(z):
    s = sigmoide(z)
    return s * (1.0 - s)  # f'(z) = f(z) * (1 - f(z)), propiedad de la sigmoide

for paso in range(5):
    # --- Pase hacia adelante ---
    z = w * x                      # suma ponderada (aquí, un único producto)
    y_pred = sigmoide(z)           # activación
    perdida = (y_pred - y_objetivo) ** 2  # pérdida cuadrática

    # --- Pase hacia atrás (regla de la cadena, término a término) ---
    dL_dypred = 2 * (y_pred - y_objetivo)   # derivada de la pérdida respecto a y_pred
    dypred_dz = derivada_sigmoide(z)        # derivada de la sigmoide respecto a z
    dz_dw = x                               # derivada de z = w*x respecto a w

    # Regla de la cadena: dL/dw = dL/dy_pred * dy_pred/dz * dz/dw
    gradiente_w = dL_dypred * dypred_dz * dz_dw

    # --- Descenso de gradiente ---
    w = w - eta * gradiente_w

    print(f"paso {paso}: w={w:.5f}  y_pred={y_pred:.5f}  perdida={perdida:.5f}  "
          f"gradiente={gradiente_w:.5f}")
```

**Esegui questa cella su Google Colab e osserva come la perdita diminuisce a ogni passo!** Vedrai che `y_pred` si avvicina poco a poco a `1.0` (l'obiettivo) e che la perdita decresce in modo monotono: è la discesa del gradiente che funziona, con una backpropagation scritta a mano, senza alcuna libreria di differenziazione automatica.

Fai caso alla riga del gradiente: è letteralmente la regola della catena vista in teoria, scritta come tre moltiplicazioni. Ogni termine corrisponde a un tratto del grafo computazionale: la perdita rispetto alla previsione, la previsione rispetto alla somma pesata (derivata della sigmoide) e la somma pesata rispetto al peso. Quando la rete ha migliaia di pesi e decine di strati, il meccanismo è identico; cambia solo la quantità di termini e la necessità di organizzarli in matrici.

### Parte (b): lo stesso esempio con PyTorch (tensori, autograd e un ottimizzatore)

Scrivere la backpropagation a mano è fattibile per un neurone, ma sarebbe impraticabile —e soggetto a errori— per una rete con milioni di parametri. Qui entra in gioco PyTorch, il framework che userai per il resto di questo corso. Il suo motore di differenziazione automatica, chiamato `autograd`, costruisce il grafo computazionale al posto tuo durante il passaggio in avanti e calcola automaticamente tutti i gradienti quando glielo chiedi, applicando esattamente lo stesso meccanismo di regola della catena che hai programmato a mano.

```python
import torch

# En Colab, PyTorch ya viene preinstalado; si te falta, descomenta la línea siguiente:
# !pip install torch

# Mismos datos que en la versión a mano
x = torch.tensor(2.0)
y_objetivo = torch.tensor(1.0)

# El peso es ahora un tensor "entrenable": requires_grad=True le dice a PyTorch
# que debe registrar todas las operaciones que involucren a w para poder derivarlas después
w = torch.tensor(0.3, requires_grad=True)

eta = 0.1
# El optimizador aplica el descenso de gradiente por nosotros: mismo mecanismo,
# menos código repetitivo
optimizador = torch.optim.SGD([w], lr=eta)

for paso in range(5):
    # --- Pase hacia adelante ---
    z = w * x
    y_pred = torch.sigmoid(z)
    perdida = (y_pred - y_objetivo) ** 2

    # --- Pase hacia atrás: autograd calcula dL/dw automáticamente ---
    optimizador.zero_grad()  # limpia gradientes acumulados de pasos anteriores
    perdida.backward()      # aquí ocurre el backpropagation completo

    print(f"paso {paso}: w={w.item():.5f}  y_pred={y_pred.item():.5f}  "
          f"perdida={perdida.item():.5f}  gradiente={w.grad.item():.5f}")

    # --- Descenso de gradiente, aplicado por el optimizador ---
    optimizador.step()
```

**Confronta i numeri delle due celle: devono coincidere passo dopo passo!** Il gradiente che stampa `w.grad` nella versione PyTorch deve essere identico al `gradiente_w` che hai calcolato a mano in NumPy, perché `autograd` non fa magie: applica la stessa regola della catena, solo in modo automatico e generalizzabile a reti di qualsiasi dimensione.

Tre idee chiave di questo secondo frammento, che userai continuamente nel resto del corso:

- `requires_grad=True` contrassegna un tensore come parametro addestrabile; PyTorch costruisce internamente il grafo computazionale ogni volta che quel tensore partecipa a un'operazione
- `perdida.backward()` percorre quel grafo all'indietro e salva il gradiente risultante dentro `w.grad`, esattamente il passaggio che hai programmato a mano
- L'ottimizzatore (qui, `SGD`, discesa del gradiente stocastica) incapsula la formula $\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$; per provare Adam invece di SGD basterebbe cambiare una riga: `torch.optim.Adam([w], lr=eta)`

D'ora in poi, quando costruirai reti con molti strati e migliaia di parametri, seguirai questo stesso schema in tre passi —passaggio in avanti, `backward()`, `optimizador.step()`— senza mai più dover derivare nulla a mano.

## Riepilogo dei punti chiave

- Un neurone artificiale calcola una somma pesata dei suoi input, vi aggiunge un bias e applica una funzione di attivazione non lineare: senza quella non linearità, impilare strati non aggiunge nulla di nuovo
- Le funzioni di attivazione più rilevanti sono la sigmoide e la tangente iperbolica (saturano e causano gradienti che svaniscono), e la ReLU (evita quel problema, anche se può generare neuroni morti); softmax si riserva quasi sempre allo strato di output nella classificazione multiclasse
- Un perceptron multistrato impila più strati di neuroni; gli strati nascosti trasformano lo spazio dei dati fino a renderlo separabile, il che permette di risolvere problemi non lineari come lo XOR
- Il teorema di approssimazione universale garantisce che una rete con un solo strato nascosto può approssimare qualsiasi funzione continua, ma non dice come costruirla; in pratica, aumentare la profondità è molto più efficiente che allargare un unico strato
- La funzione di perdita misura l'errore della previsione: entropia incrociata per la classificazione, errore quadratico medio (MSE) per la regressione; combinare softmax con entropia incrociata semplifica il gradiente in una semplice differenza tra previsione ed etichetta
- La discesa del gradiente aggiorna i parametri spostandoli nella direzione opposta al gradiente della perdita, scalato dal tasso di apprendimento; la sua variante stocastica (SGD) usa mini-batch invece di tutto l'insieme dei dati, e Adam è oggi l'ottimizzatore di riferimento
- La backpropagation applica in modo sistematico la regola della catena sul grafo computazionale della rete, in un passaggio in avanti e uno all'indietro, per calcolare tutti i gradienti con un costo paragonabile a quello di una singola previsione
- PyTorch automatizza questo calcolo tramite `autograd`: si contrassegnano i parametri con `requires_grad=True`, si chiama `.backward()` sulla perdita e un ottimizzatore (`torch.optim`) applica l'aggiornamento; è il framework che userai per il resto del corso

## Mettilo in pratica

Dedica tra quindici e trenta minuti a questa mini-attività in Google Colab. Riutilizza il codice della parte (b) di questo capitolo come punto di partenza.

1. Estendi il modello di un singolo neurone a due pesi: sostituisci `w = torch.tensor(0.3, requires_grad=True)` con due input (`x1`, `x2`) e due pesi (`w1`, `w2`), in modo che `z = w1 * x1 + w2 * x2`
2. Aggiungi un bias addestrabile `b` (anch'esso con `requires_grad=True`) alla somma pesata
3. Cambia il tasso di apprendimento `eta` in tre valori diversi (per esempio, 0,01, 0,5 e 5,0) ed esegui venti passi di addestramento con ciascuno; osserva e annota cosa succede alla perdita in ogni caso: converge con dolcezza, converge più velocemente, o oscilla senza mai stabilizzarsi?
4. Sostituisci l'ottimizzatore `SGD` con `Adam` con lo stesso tasso di apprendimento iniziale e confronta quanti passi impiega ciascuno a far scendere la perdita sotto la stessa soglia, per esempio 0,01
5. Come sfida finale, cambia l'attivazione da `torch.sigmoid` a `torch.relu` e osserva cosa succede se inizializzi i pesi con valori molto negativi: la rete riesce comunque a continuare ad apprendere, o resta "bloccata"? Collega ciò che osservi al problema dei neuroni morti che abbiamo spiegato in teoria

Salva il tuo notebook di Colab: ti servirà come riferimento quando, nei prossimi capitoli, costruirai reti con più strati nascosti reali.
