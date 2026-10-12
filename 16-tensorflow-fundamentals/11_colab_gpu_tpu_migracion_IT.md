# 11. TensorFlow in Colaboratory, GPU/TPU e migrazione a TF 2.0

## Perché questo argomento è importante

Fino ad ora hai eseguito tutto il codice di questo corso senza chiederti dove effettivamente corre. La risposta, quasi sempre, è stata Google Colaboratory: un ambiente gratuito nel cloud che ti ha risparmiato di installare Python, TensorFlow (TF) e decine di librerie nel tuo proprio computer. In questo capitolo capirai davvero cosa è Colab, come trarne il massimo — da montare il tuo Google Drive all'attivare hardware specializzato — e perché quel hardware specializzato, la GPU (Graphics Processing Unit o Unità di Elaborazione Grafica) e la TPU (Tensor Processing Unit o Unità di Elaborazione Tensorial), può ridurre un addestramento di ore a minuti. Chiuderai il capitolo intendendo anche un problema pratico molto reale: cosa fare se trovi codice vecchio scritto per TensorFlow 1.x e hai bisogno di adattarlo alla sintassi di TensorFlow 2.x che hai usato in tutto il corso.

## Cosa è Google Colaboratory e perché ti libera di installare qualcosa

Google Colaboratory, conosciuto come Colab, è un quaderno Jupyter ospitato nel cloud. Se non hai mai sentito parlare di Jupyter: è un formato di documento eseguibile che mescola codice, testo esplicativo, immagini e risultati in uno stesso file, organizzato in celle. Colab è, essenzialmente, un quaderno Jupyter che vive dentro il tuo Google Drive invece che nel tuo disco rigido.

La differenza chiave rispetto a installare Jupyter nel tuo proprio computer è l'**ambiente di esecuzione (runtime)**. Quando apri un notebook in Colab, non stai eseguendo codice nel tuo portatile: Google ti assegna una macchina virtuale remota, con Python e TensorFlow già preinstallati, che esegue ogni cella che avvii. Puoi chiudere il browser e, finché la sessione non scade, il tuo ambiente rimane vivo nel cloud. Questo ha tre conseguenze pratiche che userai costantemente:

- Non hai bisogno di installare niente in locale: apri il browser e hai già Python, TensorFlow e librerie come Pandas o Matplotlib disponibili
- Puoi eseguire comandi da terminale da una cella di codice anteponendo un segno di esclamazione (`!`), per esempio `!pip install tensorflow==2.15.0` per installare una versione concreta
- Lo stesso notebook si può condividere come un documento di Google Docs: per link, con permessi di lettura o modifica, o esportato come file `.ipynb` per aprirlo in Jupyter, JupyterLab o GitHub

Pensa a Colab come ad affittare una cucina completamente attrezzata ogni volta che vuoi cucinare, invece di comprare e mantenere tu stesso forni, padelle e utensili in casa. Non sei proprietario dell'hardware, ma neanche devi preoccuparti di installarlo, aggiornarlo o che si rompa.

## Montare Google Drive e caricare file

Gli esempi e i dataset piccoli che hai usato nelle classi precedenti di solito si scaricavano direttamente da un URL. Ma in un progetto reale hai bisogno di lavorare con i tuoi propri file: un CSV (Comma-Separated Values o Valori Separati da Virgole) con i tuoi dati, un modello già addestrato che vuoi riutilizzare, o immagini proprie. Colab ti offre due vie principali per questo.

La prima è **montare il tuo Google Drive**, cioè connettere il tuo account di Drive all'ambiente di esecuzione affinché il notebook possa leggere e scrivere file lì come se fosse una cartella qualunque del sistema. Una volta montato, qualunque file del tuo Drive rimane accessibile in un percorso come `/content/drive/MyDrive/mia_cartella/miei_dati.csv`, e qualunque file che generi — un modello addestrato, per esempio — puoi salvarlo lì affinché sopravviva alla chiusura della sessione.

La seconda via è **caricare file direttamente** dal tuo computer alla sessione attuale, senza passare per Drive. È più veloce per un file solitario che ti serve solo una volta, ma quei file scompaiono non appena termina la sessione di Colab, così non serve per salvare i risultati in modo permanente.

## Come attivare e verificare una GPU o una TPU

Per default, un notebook nuovo di Colab si avvia con un ambiente di esecuzione basato solo su CPU (Central Processing Unit o Unità Centrale di Elaborazione): sufficiente per esplorare dati o addestrare modelli piccoli, ma lento per reti neurali grandi. Per accelerare l'addestramento, cambi il tipo di hardware dal menu **Ambiente di esecuzione > Cambia tipo di ambiente di esecuzione**, dove puoi scegliere tra "Nessuno" (CPU), GPU o TPU. Questo cambio riavvia la sessione, così conviene farlo prima di eseguire celle che dipendono da variabili già create.

Perché una GPU o una TPU accelerano così tanto l'addestramento? Una CPU ha pochi core, ma molto potenti, ottimizzati per eseguire compiti complessi uno dietro l'altro. Addestrare una rete neurale, al contrario, consiste soprattutto nel moltiplicare matrici di numeri una e un'altra volta: migliaia di operazioni identiche e semplici che si possono eseguire in parallelo perfettamente. Una GPU ha centinaia o migliaia di core più semplici, progettati esattamente per questo: ripetere la stessa operazione su molti dati contemporaneamente. Una TPU va un passo oltre: è un chip che Google ha progettato specificamente per Machine Learning (ML), con un circuito interno ottimizzato per il tipo concreto di moltiplicazione di matrici che usano le reti neurali, il che la rende ancora più efficiente di una GPU per modelli grandi.

L'analogia più chiara è quella di un team di consegna. La CPU è un corriere molto veloce e versatile che consegna i pacchi uno per uno. La GPU è come avere centinaia di corrieri più semplici che consegnano centinaia di pacchi identici contemporaneamente. La TPU è quel stesso team di corrieri, ma con moto progettate su misura solo per quel quartiere e quel tipo di pacco: niente più veloce per quel compito concreto, anche se meno flessibile per qualunque altra cosa.

Per verificare quale hardware hai veramente disponibile nella tua sessione, TensorFlow ti dà una funzione chiara e diretta:

```python
import tensorflow as tf

# Elenca i dispositivi fisici di ogni tipo che TensorFlow vede in questa sessione
print("CPU:", tf.config.list_physical_devices('CPU'))
print("GPU:", tf.config.list_physical_devices('GPU'))
print("TPU:", tf.config.list_physical_devices('TPU'))
```

Se la lista di GPU o TPU appare vuota, significa che l'ambiente di esecuzione attuale non ha quel hardware attivato (controlla il menu "Cambia tipo di ambiente di esecuzione") o che Colab non ne ha nessuno disponibile in quel momento per il tuo account. Con una GPU attivata, vedrai qualcosa come `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.

## Perché l'addestramento è più veloce con GPU: un esempio commentato

Il frammento seguente addestra lo stesso modello semplice due volte — una forzando TensorFlow a usare la CPU e l'altra lasciandogli usare la GPU se disponibile — e confronta il tempo. È esattamente il tipo di confronto che vedrai ripetuto, con più rigore, nel notebook di pratica di questa classe.

```python
import tensorflow as tf
import time

# Dati di esempio: sufficienti per notare la differenza di velocità
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0

def creare_modello():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Forziamo l'esecuzione in CPU con un gestore di contesto "with tf.device(...)"
with tf.device('/CPU:0'):
    modello_cpu = creare_modello()
    modello_cpu.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    inizio = time.time()
    modello_cpu.fit(x_train, y_train, epochs=1, batch_size=128, verbose=0)
    print(f"Tempo in CPU: {time.time() - inizio:.2f} secondi")

# Se c'è GPU disponibile, TensorFlow la usa automaticamente senza codice aggiuntivo
if tf.config.list_physical_devices('GPU'):
    modello_gpu = creare_modello()
    modello_gpu.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    inizio = time.time()
    modello_gpu.fit(x_train, y_train, epochs=1, batch_size=128, verbose=0)
    print(f"Tempo in GPU: {time.time() - inizio:.2f} secondi")
else:
    print("Non c'è GPU attivata in questo ambiente di esecuzione")
```

Nota due dettagli importanti. Primo, non hai bisogno di codice speciale affinché TensorFlow usi la GPU: se è disponibile e non le indichi diversamente con `tf.device(...)`, la usa per default automaticamente. Secondo, il gestore di contesto `with tf.device('/CPU:0')` è il modo per forzare un'operazione concreta a un dispositivo concreto, qualcosa utile soprattutto per confronti come questo o per fare debug di dove si esegue ogni parte del tuo modello con `tf.debugging.set_log_device_placement(True)`.

Per lavorare con una TPU il pattern cambia un po', perché una TPU non vive nella stessa macchina del tuo notebook: è in un cluster remoto a cui TensorFlow si connette mediante una **strategia di distribuzione**. L'esempio di riferimento di questo capitolo — adattato dal notebook originale *Predict Shakespeare with Cloud TPUs and Keras* — usa questo pattern:

```python
import os
import tensorflow as tf

# Connessione al cluster di TPU assegnato da Colab
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu='grpc://' + os.environ['COLAB_TPU_ADDR']
)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

# La strategia reparte il calcolo tra i core della TPU
strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
    modello = creare_modello()
    modello.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Da qui in poi, fit(), evaluate() e predict() funzionano esattamente come
# con qualunque altro modello di Keras, ma il calcolo corre nella TPU
modello.fit(x_train, y_train, epochs=10, batch_size=1024)
```

L'idea chiave è che, eccetto per la creazione della strategia e il blocco `with strategy.scope():` dove definisci e compili il modello, il resto del codice di Keras — `fit`, `evaluate`, `predict` — è identico a quello che conosci già. TensorFlow si incarica di reppartire l'addestramento tra i core della TPU senza che tu debba riscrivere la tua logica di addestramento.

## Migrare codice da TensorFlow 1.x a TensorFlow 2.x

Se mai lavori con un notebook o uno script vecchio, è probabile che trovi codice scritto per TensorFlow 1.x, la versione che si usava prima del 2019. La differenza di sintassi può risultare sconcertante se hai lavorato solo con TensorFlow 2.x, così vale la pena identificare i cambiamenti chiave.

**Esecuzione differita (graph mode) rispetto a esecuzione immediata (eager execution)**. In TensorFlow 1.x, scrivere codice non eseguiva niente immediatamente: costruivi un grafo di operazioni e dopo dovevi aprire una `tf.Session()` e chiamare `sess.run(...)` affinché il grafo si valutasse e ottenere un risultato. In TensorFlow 2.x, l'esecuzione eager (immediata) è il comportamento per default: ogni operazione si calcola nel momento in cui la scrivi, come in Python normale, senza sessioni di mezzo.

```python
# Stile TensorFlow 1.x: costruisci il grafo e poi lo esegui in una sessione
import tensorflow.compat.v1 as tf1
a = tf1.constant(2)
b = tf1.constant(3)
c = tf1.add(a, b)
with tf1.Session() as sess:
    risultato = sess.run(c)
    print(risultato)  # 5

# Stile TensorFlow 2.x: l'operazione si calcola al momento, senza sessione
import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)
print(c.numpy())  # 5
```

**Placeholders rispetto a dati diretti**. TensorFlow 1.x usava `tf.placeholder()` per riservare un buco a cui più tardi, dentro una sessione, gli passavi i dati reali mediante un dizionario `feed_dict`. In TensorFlow 2.x non esistono i placeholders: gli passi i dati direttamente alle tue funzioni o al tuo modello di Keras, come fai con qualunque funzione normale di Python.

**Inizializzazione di variabili**. In TensorFlow 1.x dovevi chiamare esplicitamente `tf.global_variables_initializer()` ed eseguirlo dentro una sessione prima di poter usare qualunque `tf.Variable`. In TensorFlow 2.x, una variabile è pronta per usarsi nel momento stesso in cui la crei, senza nessun passo di inizializzazione aparte.

**Keras come API di alto livello per default**. TensorFlow 1.x aveva diverse forme frammentate di costruire modelli, inclusa l'API di basso livello `tf.layers`. TensorFlow 2.x integra Keras come la forma consigliata e per default di definire modelli, la stessa sintassi `tf.keras.Sequential([...])` che hai usato in tutto il corso.

**Strumenti di compatibilità**. Se hai bisogno di eseguire codice vecchio senza riscriverlo completamente, TensorFlow offre il modulo `tensorflow.compat.v1`, che espone le funzioni della versione 1.x dentro un'installazione di TensorFlow 2.x (così è come si è scritto l'esempio della sessione qui sopra). Per una migrazione più automatica, TensorFlow include lo strumento da riga di comando `tf_upgrade_v2`, che analizza uno script in TensorFlow 1.x e genera una versione adattata a TensorFlow 2.x, insieme a un report dei cambiamenti applicati:

```bash
!tf_upgrade_v2 --infile mio_script_v1.py --outfile mio_script_v2.py
```

Questo strumento non è infallibile: converte automaticamente la maggior parte dei pattern abituali — come aggiungere argomenti con nome a funzioni che hanno cambiato l'ordine dei loro parametri — , ma avvisa nel report dei casi che non può risolvere da solo, per esempio quando una funzione ha eliminato completamente un argomento (come è successo con `use_cudnn_on_gpu` in `tf.nn.conv2d`). In quei casi dovrai rivedere e correggere il codice a mano, appoggiandoti alla guida ufficiale di migrazione di TensorFlow.

## Riassunto dei punti chiave

- Google Colaboratory è un quaderno Jupyter nel cloud: ti dà un ambiente di esecuzione remoto con Python e TensorFlow preinstallati, senza dover installare niente nel tuo computer
- Puoi montare il tuo Google Drive per leggere e salvare file in modo permanente, o caricare file solitari direttamente alla sessione se li serve solo puntualmente
- Attivi GPU o TPU da "Ambiente di esecuzione > Cambia tipo di ambiente di esecuzione"; `tf.config.list_physical_devices(...)` ti conferma quale hardware vede realmente TensorFlow nella tua sessione
- La GPU e la TPU accelerano l'addestramento perché parallelizzano le moltiplicazioni di matrici che dominano il calcolo di una rete neurale; la TPU è inoltre progettata specificamente per quel tipo di operazione
- Con GPU, TensorFlow usa l'hardware automaticamente senza cambi di codice; con TPU hai bisogno di creare una `TPUStrategy` e definire il modello dentro `strategy.scope()`
- TensorFlow 2.x usa esecuzione eager per default (senza `Session.run`), elimina i placeholders, non richiede di inizializzare variabili a mano e integra Keras come API principale; il modulo `tensorflow.compat.v1` e lo strumento `tf_upgrade_v2` aiutano a migrare codice vecchio

## Mettilo in pratica

Apri un notebook nuovo in Colab con l'ambiente di esecuzione in modo "Nessuno" (solo CPU) ed esegui `tf.config.list_physical_devices('GPU')`. Conferma che la lista esce vuota. Dopo, cambia il tipo di ambiente di esecuzione a GPU dal menu, esegui di nuovo la stessa riga e verifica che ora appare un dispositivo. Per ultimo, scrivi di memoria (senza guardare il capitolo) le tre differenze di sintassi tra TensorFlow 1.x e 2.x che più ti hanno colpito, e annota un esempio breve di codice per ognuna. Quel esercizio di memoria è il miglior modo di verificare se davvero hai interiorizzato il cambio di paradigma tra versioni.
