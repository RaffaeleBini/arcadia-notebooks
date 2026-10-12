# Reti che imparano a colpi di scorciatoie: architetture avanzate per la visione artificiale

## Perché questo tema è importante

Ormai sai come funziona una rete neurale convoluzionale (CNN) di base: strati di convoluzione che estraggono bordi e texture, strati di pooling che riducono la risoluzione e uno strato finale che decide a quale categoria appartiene l'immagine. Con questo puoi costruire un classificatore ragionevole. Ma classificare non è la stessa cosa che capire una scena.

Pensa a una fotografia di due persone che vanno in bicicletta in un giardino, una delle quali in una postura insolita. Un classificatore addestrato su ImageNet, come ResNet-50, è in grado di dire "bicicletta" e "giardino". E lì si ferma: non sa che ci sono due biciclette, non localizza le persone, non percepisce che una di loro è in un orientamento strano. Circa l'80 % delle informazioni che ricevi dall'ambiente ti arriva dalla vista, quindi un sistema che si limita a etichettare immagini sfrutta una frazione minima di ciò che una scena visiva contiene.

Questo limite ha conseguenze molto concrete. Un veicolo autonomo deve sapere esattamente dove si trova ogni pedone, non solo se ci sono pedoni nell'immagine. Un sistema di supporto per persone con disabilità visiva deve descrivere relazioni spaziali, non solo oggetti isolati. Un robot che deve afferrare un pezzo deve distinguerlo da altri pezzi uguali impilati accanto. E le architetture più avanzate che incontrerai in questo capitolo nascono proprio per coprire questo salto: da "che cosa c'è nell'immagine" a "dove si trova, quanto occupa e a quale istanza concreta appartiene".

Un buon esempio di quanto possa spingersi l'apprendimento per trasferimento tra domini è quello di OpenAI con una mano robotica capace di risolvere un cubo di Rubik. Il team ha addestrato il modello interamente in un ambiente simulato, perché addestrare direttamente sul robot fisico avrebbe rischiato di danneggiare la mano con movimenti erratici. Grazie a una tecnica di randomizzazione automatica del dominio, che combina un aumento dei dati aggressivo con la selezione dei casi più difficili, il comportamento appreso nella simulazione è stato trasferito con successo al mondo reale. È la stessa idea di fondo che sostiene questo capitolo: riutilizzare ciò che si è imparato, invece di ripartire da zero ogni volta che cambiano il compito o lo scenario.

Vedrai anche perché le reti molto profonde, con decine o centinaia di strati, hanno smesso di funzionare bene finché non è comparsa una soluzione sorprendentemente semplice: lasciare che l'informazione salti interi strati senza distorcersi. Quell'idea, la connessione residuale, è il componente che ha reso possibile addestrare reti di centinaia di strati e che oggi sostiene gran parte della visione artificiale moderna.

## Spiegazione concettuale con analogie

**Le connessioni residuali come una scorciatoia di montagna.** Immagina un percorso escursionistico con molte tappe consecutive, ciascuna in salita. Se a ogni tappa perdi un po' di energia, arrivato alla tappa numero 50 potrebbe non restarti forza sufficiente per continuare: è esattamente ciò che succede al gradiente quando risale una rete molto profonda durante l'addestramento, un fenomeno chiamato scomparsa del gradiente. Ora immagina che, accanto al sentiero principale, esista una scorciatoia che scende direttamente dall'inizio del percorso fino a ciascun rifugio, senza passare per le salite. Non sostituisce il sentiero principale, ma garantisce che tu abbia sempre una via diretta per arrivare. Questa è una connessione residuale: una somma che porta l'ingresso di un blocco direttamente alla sua uscita, in parallelo alle convoluzioni. La rete non deve più imparare da zero la trasformazione completa in ogni blocco; le basta imparare la differenza, il "residuo", rispetto al lasciar passare l'ingresso così com'è.

**Il codificatore-decodificatore come fare e disfare una valigia.** Per compiti come la segmentazione, in cui serve un'etichetta per ogni pixel, le reti di solito comprimono prima l'immagine (il codificatore riduce la risoluzione ed estrae concetti astratti, come chi mette i vestiti piegati e compatti in valigia) e poi la ricostruiscono alla dimensione originale (il decodificatore la "dispiega" di nuovo). Il problema è che, comprimendo, si perdono pieghe e dettagli fini. Le connessioni di salto tra codificatore e decodificatore —la stessa famiglia di idea della scorciatoia residuale— permettono di recuperare quei dettagli fini direttamente dagli strati iniziali, senza doverli ricostruire a memoria.

**L'Intersezione su Unione (IoU) come due tappeti sovrapposti.** Se stendi due tappeti rettangolari sul pavimento, uno sopra l'altro ma leggermente spostati, la zona in cui si sovrappongono e la superficie totale che coprono insieme ti dicono molto di più su quanto siano allineati che non la semplice distanza tra i loro angoli.

**La perdita focale come un corpo docente che distribuisce l'attenzione con criterio.** In una classe in cui la maggioranza degli studenti ha già padroneggiato il programma e una minoranza è in difficoltà, un corpo docente efficace non distribuisce il tempo in modo uniforme: dedica meno attenzione a chi già risponde correttamente e concentra lo sforzo su chi sbaglia ancora. È la logica della perdita focale, pensata perché i rilevatori di oggetti non vengano sopraffatti dalle migliaia di riquadri di sfondo, facili da classificare, che abbondano in qualsiasi immagine.

## Sviluppo teorico

### Scomporre il problema della visione

La classificazione supervisionata di immagini si può intendere come una catena di componenti: un **input** (l'immagine), dei **compiti** (che cosa si chiede al modello), dei **priori** o ipotesi incorporate nell'architettura, un **ordine** di risoluzione, degli **output** e una misura di **precisione**. Cambiare uno qualsiasi di questi componenti apre la porta a compiti nuovi. Puoi cambiare il compito (da classificare a rilevare o segmentare), il tipo di input (da un'immagine a una coppia di immagini o a un video) o il tipo di supervisione (da etichette manuali a segnali che il dato stesso genera). Le sezioni seguenti percorrono queste tre direzioni.

### Il problema della degradazione e le connessioni residuali

Prima delle reti residuali, impilare più strati in una CNN non garantiva risultati migliori. Oltre una certa profondità, l'errore di addestramento tornava a salire, non per sovradattamento, ma perché la rete smetteva di poter essere ottimizzata bene: è il cosiddetto problema della degradazione. La causa principale è la scomparsa del gradiente: propagandosi all'indietro attraverso decine di strati, il gradiente si moltiplica strato dopo strato e, se questi fattori sono piccoli, il prodotto tende a zero molto prima di raggiungere i primi strati, che smettono di apprendere.

La soluzione proposta da ResNet (*Residual Network*, "rete residuale") consiste nel riformulare ciò che ogni blocco deve imparare. Invece di obbligare un insieme di strati a imparare direttamente una funzione desiderata H(x), si chiede loro di imparare il residuo F(x) = H(x) − x, e l'uscita reale del blocco si ricompone come:

y = F(x, {Wᵢ}) + x

dove x è l'ingresso del blocco, F(x, {Wᵢ}) è ciò che calcolano le convoluzioni interne e la somma con x è la connessione di scorciatoia (*identity shortcut*). Quando la dimensione di x non coincide con quella di F(x) —per esempio perché il blocco riduce la risoluzione spaziale o cambia il numero di canali—, la scorciatoia incorpora una semplice proiezione lineare, tipicamente una convoluzione 1x1, prima di sommare. Con questa riformulazione, se la trasformazione ottimale per un blocco è semplicemente lasciare passare l'ingresso senza modifiche, la rete deve solo imparare a portare F(x) vicino a zero, un compito molto più facile che imparare una funzione identità esatta con strati non lineari. E durante la retropropagazione, il gradiente dispone sempre di una via diretta attraverso la somma, oltre a quella che attraversa le convoluzioni, il che mantiene segnali di gradiente più forti negli strati iniziali.

Grazie a questo meccanismo è stato possibile addestrare reti con centinaia e perfino migliaia di strati, raggiungendo una precisione senza precedenti nella classificazione di immagini. La stessa famiglia di idee —una via diretta che preserva informazione e gradiente— ricompare, come vedrai, nelle connessioni di salto di U-Net e nel disegno a forma di "U" di rilevatori come RetinaNet.

Un'altra architettura rilevante di questa fase è Inception, nota anche come GoogLeNet. Anziché impilare convoluzioni di un'unica dimensione, ogni modulo Inception applica in parallelo filtri di dimensioni diverse sullo stesso ingresso e concatena le loro uscite, catturando schemi a più scale all'interno dello stesso blocco. È un approccio complementare a quello di ResNet: mentre questa affronta il problema della profondità, Inception affronta il problema della scala.

### Rilevamento di oggetti: dal classificare al localizzare

Il rilevamento di oggetti è un problema multitask: per ogni oggetto dell'immagine servono la sua etichetta di classe e il suo riquadro di delimitazione (*bounding box*), di solito parametrizzato dal centro, dall'altezza e dalla larghezza. La classe è un problema di classificazione come un altro, ma le coordinate del riquadro sono valori continui, quindi prevederle richiede una regressione, non una classificazione. La perdita più semplice per questo caso è l'errore quadratico medio (ECM), che calcola la media del quadrato della differenza tra coordinate previste e reali:

ECM = (1/N) · Σ (yᵢ − ŷᵢ)²

La difficoltà emerge quando un'immagine contiene più oggetti: bisogna decidere quale previsione corrisponda a ciascun oggetto reale. La strategia adottata dai rilevatori moderni combina due passi: prima si discretizza lo spazio dei possibili riquadri e lo si risolve come classificazione, poi si raffina la previsione vincente tramite regressione.

**Faster R-CNN**, un rilevatore a due stadi, illustra bene questo schema. Una rete di base (per esempio una ResNet) produce mappe di caratteristiche a partire dall'immagine. Su quelle mappe agisce una Rete di Proposta di Regioni (RPN, *Region Proposal Network*), che colloca punti di ancoraggio distribuiti uniformemente sull'immagine e, in ciascuno, valuta varie combinazioni di scala e proporzione —di solito tre scale e tre proporzioni, quindi nove candidati per punto—. Per ogni candidato, la RPN prevede un punteggio di "objectness" (probabilità che contenga un oggetto qualsiasi) e si conservano i migliori. Queste proposte passano poi a un secondo stadio che le classifica e ne corregge le coordinate tramite regressione. Il risultato è preciso ma relativamente lento, nell'ordine di cinque fotogrammi al secondo, in parte perché include operazioni non derivabili —come scegliere quali proposte conservare— che obbligano ad addestrare la RPN e il resto della rete in fasi separate.

**RetinaNet**, un rilevatore a uno stadio, elimina quella fase intermedia: prevede classi e coordinate direttamente su una gerarchia di mappe di caratteristiche a risoluzioni diverse (una Rete a Piramide di Caratteristiche, FPN), il che gli permette di essere addestrato da cima a fondo e di funzionare più velocemente, intorno agli otto fotogrammi al secondo. Il prezzo è uno squilibrio severo: la stragrande maggioranza dei riquadri candidati corrisponde allo sfondo, facile da classificare, e la loro perdita accumulata domina su quella dei pochi casi davvero informativi, generando un segnale di addestramento scadente. I rilevatori a due stadi risolvono il problema potando i negativi facili nella prima fase; quelli a uno stadio ricorrono alla ricerca dei negativi difficili (addestrare, rilevare falsi positivi su dati non visti, aggiungerli al set di addestramento e ripetere) o, in modo più diretto, alla **perdita focale**, proposta insieme a RetinaNet. La sua formulazione riduce la perdita di entropia incrociata standard tramite un fattore modulatore:

FL(pₜ) = −(1 − pₜ)^γ · log(pₜ)

dove pₜ è la probabilità che il modello assegna alla classe corretta e γ (tipicamente due) controlla quanto si attenua il contributo degli esempi già ben classificati. Quando pₜ è alto (il modello indovina con sicurezza), il fattore (1 − pₜ)^γ si avvicina a zero e quel campione pesa appena nella perdita totale; quando pₜ è basso, il fattore resta vicino a uno e il campione continua a penalizzare con forza. L'addestramento si concentra così sui casi difficili, senza bisogno del meccanismo esplicito di ricerca dei negativi.

Per **valutare** entrambi i tipi di rilevatore non si usa l'ECM di addestramento, ma l'Intersezione su Unione:

IoU = Area di sovrapposizione / Area di unione

Un valore di IoU va da zero (nessuna sovrapposizione) a uno (coincidenza perfetta) e, a differenza dell'ECM, valuta il riquadro nel suo insieme invece di trattare ogni coordinata separatamente. Un esempio chiarisce il punto: rispetto a un riquadro reale (50, 50, 150, 150), un riquadro previsto (48, 52, 148, 152) —quasi identico, spostato di appena un paio di pixel— ottiene un ECM di 4 e un IoU vicino a 1,0; un riquadro (60, 60, 160, 160), spostato di dieci pixel su ciascun asse, ottiene un ECM di 100 e un IoU di circa 0,669; e uno più lontano, (30, 30, 130, 130), arriva a un ECM di 400 con un IoU di circa 0,444. L'ordine di qualità dato dall'IoU —il primo riquadro è chiaramente il migliore— coincide con l'intuizione visiva, mentre l'ECM penalizza ogni coordinata allo stesso modo senza rappresentare la sovrapposizione reale. L'IoU ha però un limite: contenendo un massimo non è derivabile, quindi non può essere usato direttamente come funzione di perdita durante l'addestramento, anche se esistono approssimazioni derivabili ispirate a lui, come la IoU generalizzata.

### Segmentazione semantica e di istanza

Quando il riquadro di delimitazione risulta troppo grossolano —per esempio per una persona con le braccia tese, il cui riquadro includerebbe molto sfondo—, conviene una rappresentazione più fine. La **segmentazione semantica** assegna un'etichetta di classe a ogni pixel dell'immagine: è un problema di predizione densa, perché l'uscita deve avere la stessa risoluzione spaziale dell'ingresso, a differenza della classificazione, in cui l'uscita è un unico vettore.

L'ostacolo è che gli strati di pooling, utili per ampliare il campo recettivo e astrarre le caratteristiche, riducono la risoluzione lungo il cammino. Serve l'operazione inversa, l'"unpooling" o disaggregazione. La versione più semplice copia ogni valore nei suoi vicini immediati, ma produce uscite sfocate e poco definite. Due alternative più elaborate sono la disaggregazione guidata dagli indici (che ricorda, dal pooling originale, in quale posizione esatta si trovava il valore massimo tramite l'operazione argmax, e lo colloca lì durante la ricostruzione) e la deconvoluzione o convoluzione trasposta, che inverte strutturalmente l'operazione di convoluzione per generare una ricostruzione più ordinata.

**U-Net**, progettata originariamente per l'imaging medico, applica queste idee in un'architettura di codificatore-decodificatore a forma di "U". Il codificatore riduce progressivamente la risoluzione mentre estrae caratteristiche sempre più astratte, come farebbe un classificatore di immagini; il decodificatore la recupera tramite disaggregazione. E, come nelle reti residuali, incorpora lunghe connessioni di salto tra ciascun livello del codificatore e il suo livello simmetrico nel decodificatore, che reiniettano il dettaglio ad alta frequenza perso durante la compressione e, di passaggio, facilitano la propagazione del gradiente durante l'addestramento. Il modello si addestra con entropia incrociata calcolata pixel per pixel e mediata su tutta l'immagine. RetinaNet, pur risolvendo un compito diverso, condivide la stessa sagoma a "U", segno di quanto sia trasversale questo schema di progettazione.

Quando un'immagine contiene più oggetti sovrapposti della stessa classe, la sola segmentazione semantica risulta ambigua: non distingue una persona dall'altra se si sovrappongono. La **segmentazione di istanza** combina rilevamento e segmentazione per risolvere il problema, assegnando una maschera distinta a ciascuna istanza individuale. Mask R-CNN è il riferimento abituale per questo compito.

### Transfer learning e valutazione

Addestrare una rete profonda da zero richiede molti dati e molto calcolo. L'apprendimento per trasferimento riutilizza ciò che un modello ha già appreso in un compito o dominio di origine per accelerare l'apprendimento in un compito o dominio di destinazione, partendo dal presupposto che molte caratteristiche visive —bordi, texture, forme— siano utili in più contesti. Trasformare un classificatore preaddestrato in un rilevatore di oggetti, per esempio, consiste nel conservare i suoi strati iniziali (affinandoli o congelandoli) e sostituire lo strato finale con altri capaci di prevedere classi e coordinate. Il caso già menzionato del cubo di Rubik risolto da una mano robotica di OpenAI mostra la variante di trasferimento tra domini diversi: dalla simulazione al mondo fisico.

Per confrontare i modelli in modo oggettivo, ogni compito ha le sue metriche e i suoi banchi di prova. La classificazione usa la precisione top-1 (è corretta se la previsione principale coincide con la classe giusta) o top-5 (è corretta se la classe giusta compare tra le cinque previsioni più probabili, utile quando ci sono categorie facilmente confondibili). Il rilevamento e la segmentazione usano l'IoU su banchi di prova come COCO o Cityscapes, così come ImageNet è servito da riferimento per la classificazione.

### Oltre una sola immagine

Un'immagine isolata non è l'unico input possibile. Una coppia di immagini consecutive permette di stimare il flusso ottico: lo spostamento di ogni pixel tra due fotogrammi, codificato come una mappa densa di valori continui. FlowNet risolve questo compito con un'architettura di codificatore-decodificatore simile a U-Net, addestrata in modo supervisionato minimizzando la distanza euclidea tra il flusso previsto e quello reale (l'errore di punto finale, EPE), sul set sintetico Flying Chairs, che sovrappone modelli 3D di sedie a sfondi reali per generare automaticamente migliaia di coppie di immagini con una verità di riferimento esatta.

Un video aggiunge inoltre la dimensione temporale. La soluzione più diretta è impilare i fotogrammi in un volume e sostituire le convoluzioni 2D con convoluzioni 3D, il cui nucleo scorre sia nello spazio sia nel tempo. Le reti di riconoscimento di azioni addestrate così sul set Kinetics —600.000 video di circa dieci secondi distribuiti in 600 classi— raggiungono un'accuratezza top-1 nell'ordine dell'82 %. Architetture come SlowFast combinano due rami: uno a bassa frequenza di campionamento, specializzato nel riconoscere categorie di oggetti stabili, e uno ad alta frequenza, specializzato nel cogliere movimenti rapidi, imitando il modo in cui il sistema visivo umano combina diverse risoluzioni temporali.

### Apprendere con meno supervisione

Etichettare manualmente ogni pixel di migliaia di immagini è costoso, quindi è prezioso apprendere senza dipendere interamente da etichette forti. L'apprendimento metrico affronta il problema in modo diverso: invece di mappare un ingresso su una classe o un valore, impara a proiettare gli ingressi in uno spazio di rappresentazione in cui quelli che appartengono alla stessa entità stanno vicini e quelli di entità diverse stanno lontani. La perdita contrastiva si addestra con coppie e un margine m fisso:

L = y · d² + (1 − y) · max(0, m − d)²

dove d è la distanza tra le due rappresentazioni e y vale uno se la coppia appartiene alla stessa entità e zero in caso contrario. La perdita tripletta migliora questa idea usando terne (ancora, positivo, negativo) e distanze relative invece di un margine assoluto:

L = max(0, d(ancora, positivo) − d(ancora, negativo) + margine)

in modo che ogni classe possa formare un gruppo di raggio proprio, senza imporre la stessa dimensione a tutti i gruppi. Un'applicazione recente e notevole di queste idee è l'apprendimento di rappresentazioni a partire dagli aumenti di dati: si generano diverse versioni trasformate della stessa immagine (ritagli, cambi di colore, rotazioni) e si addestra il modello a riconoscere che provengono dalla stessa origine. Le etichette si generano da sole, senza intervento umano, e la qualità delle rappresentazioni risultanti arriva a eguagliare quella dei modelli addestrati su insiemi completamente etichettati come ImageNet.

## Esempio di codice commentato, in PyTorch

Il blocco seguente, pensato per essere eseguito su Google Colab, mostra prima come si costruisce a mano un blocco residuale e poi carica una ResNet-18 preaddestrata di `torchvision` per classificare un'immagine reale.

```python
# Bloque residual construido a mano ------------------------------------
import torch
import torch.nn as nn

class BloqueResidual(nn.Module):
    """Bloque residual básico: dos convoluciones 3x3 y un atajo (shortcut)
    que suma la entrada a la salida antes de la última activación."""

    def __init__(self, canales_entrada, canales_salida, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(canales_entrada, canales_salida, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(canales_salida)
        self.conv2 = nn.Conv2d(canales_salida, canales_salida, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(canales_salida)
        self.relu = nn.ReLU(inplace=True)

        # Si el bloque cambia la resolución o el número de canales,
        # el atajo necesita una proyección 1x1 para poder sumarse a F(x)
        self.atajo = nn.Sequential()
        if stride != 1 or canales_entrada != canales_salida:
            self.atajo = nn.Sequential(
                nn.Conv2d(canales_entrada, canales_salida, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(canales_salida),
            )

    def forward(self, x):
        identidad = self.atajo(x)               # y = F(x) + x
        salida = self.relu(self.bn1(self.conv1(x)))
        salida = self.bn2(self.conv2(salida))
        salida = salida + identidad              # aquí ocurre la conexión residual
        return self.relu(salida)

# Prueba rápida con un tensor aleatorio (lote=2, 3 canales, 32x32 píxeles)
entrada_prueba = torch.randn(2, 3, 32, 32)
bloque = BloqueResidual(canales_entrada=3, canales_salida=16, stride=2)
print(bloque(entrada_prueba).shape)  # torch.Size([2, 16, 16, 16])


# ResNet-18 preentrenada aplicada a una imagen real ---------------------
import urllib.request
from PIL import Image
from torchvision import models, transforms

modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
modelo.eval()  # desactiva dropout y fija las estadísticas de BatchNorm

# Descarga una imagen de ejemplo y las etiquetas de ImageNet
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "imagen_ejemplo.jpg",
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "clases_imagenet.txt",
)
with open("clases_imagenet.txt") as archivo:
    categorias = [linea.strip() for linea in archivo.readlines()]

imagen = Image.open("imagen_ejemplo.jpg").convert("RGB")
preprocesar = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # medias de ImageNet
                          std=[0.229, 0.224, 0.225]),
])
entrada = preprocesar(imagen).unsqueeze(0)  # añade la dimensión de lote

with torch.no_grad():
    salida = modelo(entrada)
probabilidades = torch.nn.functional.softmax(salida[0], dim=0)

top5_prob, top5_idx = torch.topk(probabilidades, 5)
for probabilidad, indice in zip(top5_prob, top5_idx):
    print(f"{categorias[indice]}: {probabilidad.item() * 100:.1f} %")
```

Esegui questo codice su Colab così com'è: scarica l'immagine e le etichette da internet, quindi non devi caricare nessun file. Il blocco residuale manuale ti permette di vedere, riga per riga, dove avviene esattamente la somma che dà il nome all'architettura; la seconda parte ti mostra come sfruttare, con poche righe, una ResNet-18 già addestrata su ImageNet.

## Riepilogo dei punti chiave

- Una CNN di base classifica bene, ma non localizza oggetti né distingue istanze: per questo servono architetture che ridefiniscono compito, input o tipo di supervisione
- Il problema della degradazione compare quando si impilano troppi strati: il gradiente svanisce e la rete smette di ottimizzarsi bene, non di generalizzare male
- ResNet risolve la degradazione con le connessioni residuali: ogni blocco impara y = F(x) + x, e la scorciatoia garantisce una via diretta per il gradiente
- Il rilevamento di oggetti combina classificazione (la classe) e regressione (le coordinate del riquadro); Faster R-CNN lo risolve in due stadi e RetinaNet in uno solo, appoggiandosi alla perdita focale per compensare l'eccesso di sfondo facile
- L'IoU misura la qualità di un riquadro previsto in modo più fedele dell'errore quadratico medio, perché valuta la sovrapposizione reale invece di ciascuna coordinata separatamente
- La segmentazione semantica richiede una predizione densa; U-Net la risolve con un codificatore-decodificatore e connessioni di salto che recuperano il dettaglio perso riducendo la risoluzione
- L'apprendimento per trasferimento, tra compiti o tra domini (come nel caso della mano robotica di OpenAI addestrata in simulazione), evita di partire da zero a ogni nuovo problema
- Una coppia di immagini o un video aprono compiti come il flusso ottico o il riconoscimento di azioni, risolti con codificatori-decodificatori o convoluzioni 3D
- L'apprendimento metrico e gli aumenti di dati permettono di addestrare rappresentazioni potenti senza dipendere da etichette manuali esaustive

## Mettilo in pratica

**Riserva tra 15 e 30 minuti e segui questi passaggi:**

1. **Esegui** l'esempio di codice completo su Google Colab e conferma che la ResNet-18 preaddestrata classifica correttamente l'immagine di esempio
2. **Sostituisci** l'immagine scaricata con due o tre fotografie tue (caricale su Colab e adatta il percorso) e annota se il risultato top-5 indovina la categoria in ciascun caso
3. **Modifica** la classe `BloqueResidual`: commenta la riga `salida = salida + identidad` in modo che il blocco non sommi la scorciatoia, e riesegui il test con il tensore casuale
4. **Confronta** le due versioni del blocco a livello concettuale: senza la somma, ogni blocco deve ricostruire da zero la funzione completa; con la somma, deve solo imparare la differenza rispetto al lasciar passare l'ingresso. **Scrivi** in due o tre frasi, con parole tue, perché quella differenza facilita l'addestramento di reti molto profonde

## Glossario di sigle usate in questo capitolo

- CNN: rete neurale convoluzionale (*Convolutional Neural Network*)
- ECM: errore quadratico medio
- EPE: errore di punto finale (*End-Point Error*)
- FPN: rete a piramide di caratteristiche (*Feature Pyramid Network*)
- GAN: rete generativa antagonista (*Generative Adversarial Network*)
- IoU: Intersezione su Unione (*Intersection over Union*)
- RPN: rete di proposta di regioni (*Region Proposal Network*)
- SGD: discesa del gradiente stocastica (*Stochastic Gradient Descent*)
