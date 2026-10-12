# 10. Distribuzione con TensorFlow Lite e TensorFlow.js

## Perché questo argomento è importante

Fino ad ora hai addestrato e valutato i tuoi modelli in Google Colab, con accesso a una GPU (Graphics Processing Unit o Unità di Elaborazione Grafica) potente nel cloud. Ma un modello addestrato non serve a molto se rimane rinchiuso in un notebook: l'obiettivo finale quasi sempre è che qualcuno lo usi, sia dal cellulare, sia da una fotocamera connessa o sia da una pagina web. Qui appare un problema pratico: un telefono non ha la stessa memoria né potenza di calcolo di un server nel cloud, e una pagina web non può dipendere da una connessione stabile dell'utente in ogni momento.

Questo capitolo ti insegna i due percorsi che offre TensorFlow per risolvere quel problema: TensorFlow Lite (TFLite), pensato per portare modelli a cellulari e dispositivi embebidos, e TensorFlow.js (TFJS), pensato per eseguirli direttamente nel browser. Non scriverai codice nativo di Android o iOS — quello rimane fuori dall'ambito di questo corso — ma sì capirai cosa succede dentro quando converti un modello e lo prepari per funzionare fuori da un server.

## Edge rispetto a server: quando conviene eseguire il modello nel dispositivo

Immagina che tu abbia bisogno di tradurre una frase in un paese senza copertura dati. Puoi portarti un dizionario da tasca (lo risolvi tu, al momento, ovunque tu sia) o telefonare a un traduttore professionista (più preciso, ma dipendi dalla connessione della chiamata). Eseguire un modello **nel dispositivo** — quello che si conosce come computazione al **edge** o "bordo" della rete — è la versione "dizionario da tasca": il modello vive dentro il cellulare, l'app web o il sensore, e risponde senza uscire da lì. Eseguirlo **in un server** è la versione "telefonata": il dispositivo invia i dati a un server potente, questo fa la predizione e restituisce il risultato.

Conviene eseguire nel dispositivo quando:

- Hai bisogno di una risposta immediata, senza la latenza di andata e ritorno a un server (per esempio, classificare in tempo reale quello che vede la fotocamera)
- L'app deve funzionare senza connessione a internet o con connessione instabile
- I dati sono sensibili e preferisci che non escano dal dispositivo (migliora la privacy)
- Vuoi evitare il costo di mantenere un server di inferenza funzionando permanentemente

Conviene mantenere il modello in un server quando:

- Il modello è troppo grande o complesso per la memoria e la batteria di un dispositivo modesto
- Hai bisogno di aggiornare il modello con frequenza e non vuoi dipendere da ogni utente che aggiorni l'app
- Il dispositivo stesso non ha capacità di calcolo sufficiente, neanche con un modello ottimizzato

In entrambi i casi c'è un costo da assumere: al edge, sacrifichi un po' di dimensione e precisione del modello in cambio di velocità e autonomia; al server, sacrifichi autonomia e privacy in cambio di più potenza di calcolo disponibile.

## Cosa è TensorFlow Lite e come convertire un modello Keras

TensorFlow Lite è una versione leggera di TensorFlow, progettata specificamente per eseguire modelli già addestrati in dispositivi cellulari e embebidos (smartphone, tablet, microcontrollori, dispositivi IoT). Non serve per addestrare modelli da zero: il suo lavoro è prendere un modello che hai già addestrato con Keras e convertirlo in un formato compatto e veloce da caricare, pensato per hardware con memoria e batteria limitata.

Il formato di output è un file con estensione `.tflite`, basato su FlatBuffers (una struttura dati binaria ottimizzata per la lettura veloce, simile in proposito a JSON ma molto più efficiente). Convertire un modello Keras addestrato a questo formato è, con l'Application Programming Interface (API, Interfaccia di Programmazione di Applicazioni) attuale di TensorFlow, un processo di poche righe:

```python
import tensorflow as tf

# Partiamo da un modello Keras già addestrato
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Salviamo il modello convertito in disco
with open("modelo.tflite", "wb") as f:
    f.write(tflite_model)
```

Durante la conversione puoi applicare **quantizzazione (quantization)**: un'ottimizzazione che riduce la precisione numerica dei pesi del modello, per esempio da numeri decimali di 32 bit a interi di 8 bit. Questo riduce la dimensione del file in modo notevole e accelera l'inferenza, in cambio di una perdita di precisione normalmente piccola. Nel notebook di pratica di questa classe andrai a confrontare tu stesso la dimensione di un modello con e senza questa ottimizzazione.

Una volta hai il file `.tflite`, lo carichi con l'**interprete (`Interpreter`)**, il componente di TensorFlow Lite responsabile di eseguire l'inferenza:

```python
interpreter = tf.lite.Interpreter(model_path="modelo.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]["index"], dati_di_input)
interpreter.invoke()
predizione = interpreter.get_tensor(output_details[0]["index"])
```

Questo stesso interprete è quello che, internamente, usano le app di Android e iOS che vedrai nel prossimo paragrafo.

## Integrazione concettuale in Android e iOS

Non scriverai codice nativo in questo corso, ma conviene che tu sappia che aspetto ha il processo, perché è lo stesso pattern su entrambe le piattaforme: convertire il modello, impacchettarlo con l'app e caricarlo con un interprete.

**In Android**, il flusso tipico è:

1. Aggiungi la libreria di TensorFlow Lite come dipendenza nel file `build.gradle` del progetto
2. Copi il file `.tflite` (e, se è un classificatore, un file di etichette di testo) nella cartella `assets` dell'app
3. Dal codice Java o Kotlin, carichi il modello con la classe `Interpreter` ed esegui l'inferenza sui dati di input (per esempio, un fotogramma della fotocamera convertito a una struttura di byte)
4. Se il dispositivo lo supporta, TensorFlow Lite usa automaticamente la Android Neural Networks API (NNAPI), uno strato del sistema operativo pensato per accelerare operazioni di apprendimento automatico per hardware; se non è disponibile, l'inferenza si esegue nella CPU (Central Processing Unit o Unità Centrale di Elaborazione)

**In iOS**, il flusso è equivalente anche se l'ecosistema cambia: usi Xcode come ambiente di sviluppo, aggiungi il framework di TensorFlow Lite al progetto, includi il file `.tflite` e il file di etichette come risorse dell'app, e carichi il modello con la classe `Interpreter` da Swift u Objective-C. Una particolarità storica di questi esempi è che le app che usano la fotocamera funzionano solo in un dispositivo fisico, non nel simulatore di Xcode.

Un esempio tipico che appare su entrambe le piattaforme è quello della classificazione di immagini in tempo reale con il modello MobileNet: l'app cattura fotogrammi della fotocamera, li ridimensiona e normalizza, glieli passa all'interprete, e mostra sullo schermo le tre etichette con probabilità maggiore. È esattamente lo stesso ciclo predire → interpretare risultato che già conosci di Keras, solo che eseguito dentro il proprio telefono.

## Il delegato di GPU sperimentale

Un **delegato (delegate)**, in TensorFlow Lite, è un componente che permette all'interprete di "delegare" certe operazioni a un hardware diverso dalla CPU. Il delegato di GPU sperimentale fa esattamente questo: sfrutta la GPU del dispositivo — progettata per fare molti calcoli in parallelo, qualcosa in cui le operazioni matriciali di una rete neurale si adattano perfettamente — usando OpenGL ES in Android e Metal in iOS.

Secondo i test di prestazioni di TensorFlow, questo delegato può accelerare l'inferenza da due a sette volte rispetto a un'implementazione in CPU, soprattutto in modelli complessi. Comunque, per modelli molto piccoli e semplici, il miglioramento potrebbe non notarsi, o addirittura essere controproducente: spostare i dati alla memoria della GPU ha anche un costo, e se il modello è così semplice che la CPU lo risolve già velocemente, quel costo di trasferimento potrebbe non compensare. Inoltre, non tutte le operazioni di TensorFlow sono supportate dal delegato di GPU: quelle che non lo sono, si eseguono automaticamente nella CPU come alternativa (fallback).

Attivare il delegato da Python ha questo aspetto:

```python
delegato_gpu = tf.lite.experimental.load_delegate("libtensorflowlite_gpu_delegate.so")
interpreter = tf.lite.Interpreter(
    model_path="modelo.tflite",
    experimental_delegates=[delegato_gpu]
)
```

Tieni in conto che questa libreria del delegato è specifica per ogni piattaforma (Android o iOS) e non è pensata per eseguirsi in un ambiente di scrivania o in Colab: la vedi qui affinché tu capisca il concetto, non per eseguirla nel notebook di questa classe.

## TensorFlow.js: addestrare ed eseguire modelli nel browser

TensorFlow.js porta la stessa idea di TensorFlow Lite — eseguire apprendimento automatico fuori da un server — a un ambiente diverso: il browser web e Node.js. È una libreria in JavaScript che ti permette due cose diverse:

1. **Eseguire nel browser un modello che hai già addestrato in Python.** Converti il tuo modello Keras al formato di TensorFlow.js con lo strumento da riga di comando `tensorflowjs_converter`, e il risultato è un insieme di file (una definizione del modello in JSON e i pesi in file binari) che qualunque pagina web può caricare
2. **Addestrare un modello nuovo direttamente in JavaScript**, senza passare per Python in nessun momento, usando un'API molto simile a quella di Keras

TensorFlow.js sfrutta WebGL (la stessa tecnologia che usano i browser per accelerare la grafica) per eseguire i calcoli nella GPU del dispositivo dell'utente, in modo simile a come il delegato di GPU accelera TensorFlow Lite.

Poiché TensorFlow.js non si esegue in un notebook di Python, qui hai un esempio commentato di come apparirebbe in una pagina web. Primo, il comando di conversione che useresti nel tuo terminal dopo aver addestrato il modello in Python:

```bash
# Converte un modello Keras salvato in formato SavedModel o .h5
# ai file che TensorFlow.js necessita per caricarlo nel browser
tensorflowjs_converter --input_format=keras \
    percorso/al/modelo.h5 \
    percorso/di/output/modelo_tfjs
```

E questo è il codice JavaScript che caricherrebbe quel modello convertito in una pagina web e farebbe una predizione:

```html
<!-- Carichiamo la libreria di TensorFlow.js da una CDN -->
<!-- (CDN: Content Delivery Network o Rete di Distribuzione di Contenuti) -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>

<script>
  async function classificareNelBrowser() {
    // Carichiamo il modello già convertito (equivalente al .tflite, ma per web)
    const model = await tf.loadLayersModel("modelo_tfjs/model.json");

    // Creiamo un tensor di input di esempio, come faresti con NumPy in Python
    const input = tf.tensor2d([[1, 2, 3, 4]]);

    // Eseguiamo l'inferenza: l'equivalente nel browser a interpreter.invoke()
    const predizione = model.predict(input);
    predizione.print();
  }

  classificareNelBrowser();
</script>
```

Se preferisci addestrare un modello nuovo direttamente nel browser invece di convertirne uno già addestrato, la sintassi è praticamente uno specchio di Keras: definisci gli strati con `tf.sequential()` e `tf.layers.dense()`, compili il modello indicando la funzione di perdita e l'ottimizzatore, e lo addestri con `model.fit()` passando tensori di input e output. Quella similarità deliberata è una delle ragioni per cui, se già sai costruire modelli con Keras, imparare TensorFlow.js ti risulterà familiare.

## Riassunto dei punti chiave

- Eseguire un modello nel dispositivo (edge) riduce la latenza, funziona senza connessione e migliora la privacy; mantenerlo in un server ha senso quando il modello è troppo grande o cambia con frequenza
- TensorFlow Lite converte un modello Keras già addestrato in un file `.tflite` compatto, ottimizzato per cellulari e dispositivi embebidos, mediante `tf.lite.TFLiteConverter`
- La quantizzazione riduce la precisione numerica dei pesi del modello per ottenere un file più piccolo e un'inferenza più veloce, in cambio di un po' di precisione
- L'interprete (`tf.lite.Interpreter`) è il componente che carica il modello `.tflite` ed esegue l'inferenza, sia in Python sia dentro un'app di Android o iOS
- Il delegato di GPU sperimentale accelera l'inferenza di TensorFlow Lite sfruttando la GPU del dispositivo, con miglioramenti da due a sette volte in modelli complessi, anche se non sempre compensa in modelli molto semplici
- TensorFlow.js permette di eseguire nel browser un modello addestrato in Python (convertito con `tensorflowjs_converter`) o di addestrare uno nuovo direttamente in JavaScript, con una sintassi molto simile a quella di Keras

## Mettilo in pratica

Prima di aprire il notebook di questa classe, prendi carta e penna di nuovo. Pensa in un'app del tuo giorno a giorno che usi l'intelligenza artificiale (un traduttore, un assistente vocale, un filtro di foto, un correttore ortografico) e rispondi a queste tre domande:

- Credi che la predizione si calcoli nel dispositivo stesso o in un server? Perché lo credi così?
- Se dovessi decidere tu, cosa peserebbe di più nella tua decisione: la velocità di risposta, il consumo di batteria, la necessità di funzionare senza connessione o la privacy dei dati?
- Cosa succederebbe con quella app se il modello fosse dieci volte più grande? Continuerebbe ad avere senso eseguirlo nel dispositivo?

Salva le tue risposte: quando convertirai il tuo proprio modello a TensorFlow Lite nel notebook di pratica e confronterai la sua dimensione con e senza quantizzazione, andrai a comprovare in cifre reali se la tua intuizione era corretta.
