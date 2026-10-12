# 12. TensorFlow Extended: produzione di modelli

## Perché questo argomento è importante

Fino ad ora hai addestrato modelli in un notebook: carichi dati, definisci un'architettura, chiami `fit` e rivedi il risultato tu stesso, cella per cella. Quel flusso funziona meravigliosamente per imparare e sperimentare, ma non sopravvive al mondo reale. Un'azienda che vuole predire la domanda di un prodotto, rilevare frode nei pagamenti o classificare immagini mediche non può dipendere da che qualcuno riesegua un notebook a mano ogni volta che arrivano dati nuovi. Serve un sistema che ingesti dati, li validi, addestri, valuti e distribuisca il modello in automatico, ripetibile e verificabile.

A quel salto — da "funziona nel mio notebook" a "funziona in produzione, in modo affidabile, durante mesi o anni" — si chiama ingegneria di Machine Learning (ML) per la produzione. TensorFlow Extended (TFX) è la piattaforma di codice aperto che Google ha sviluppato per risolvere esattamente quel problema, ed è quella che usa internamente per la maggior parte dei suoi sistemi di ML in produzione. Aziende come Twitter, Airbnb o PayPal l'hanno adottata anch'esse. È, di gran lunga, l'argomento più avanzato del corso: non andrai a costruire reti neurali nuove, ma a imparare come si organizza e automatizza tutto quello che già sai fare con TensorFlow (TF) affinché funzioni in modo sostenuto fuori dal notebook.

## Quale problema risolve TFX

Quando un modello passa dall'essere un esperimento a essere parte di un prodotto reale, appaiono esigenze che non avevi nel notebook:

- I dati cambiano con il tempo, e quei cambiamenti possono essere lenti (riconoscere gatti e cani varia appena di anno in anno) o molto veloci (predire tendenze di mercato cambia giorno a giorno). Quanto più velocemente cambiano i dati e la "verità di riferimento" (ground truth, i valori corretti che il modello cerca di predire), più spesso devi riaddestrare il modello e più difficile è etichettare esempi nuovi
- Il codice di preprocessing che hai usato per addestrare deve applicarsi esattamente come quando il modello già sta servendo predizioni reali; qualunque differenza tra i due ambienti — quello che si conosce come bias addestramento-servizio (training/serving skew) — degrada la qualità del modello senza che lo noti nelle tue metriche di addestramento
- Hai bisogno di potere tracciare quale versione dei dati ha addestrato quale versione del modello, confrontare esecuzioni precedenti ed evitare di ricalcolare passi costosi se niente è cambiato dall'ultima volta
- Il sistema completo, non solo il modello, deve soddisfare i requisiti di qualunque software in produzione: scalabilità, consistenza, capacità di prova, sicurezza e mantenibilità

TFX non sostituisce TensorFlow: si costruisce sopra. Apporta un insieme di librerie e componenti riutilizzabili — TensorFlow Data Validation (TFDV), TensorFlow Transform (TFT), TensorFlow Model Analysis (TFMA) e TensorFlow Serving, tra altri — che coprono ogni fase del ciclo di vita di un modello: ingestione di dati, validazione, trasformazione di features, addestramento, valutazione e distribuzione.

## Cosa è un pipeline di TFX

Pensa a un pipeline di TFX come a una catena di montaggio di una fabbrica. In un notebook, tu sei chi monta l'auto pezzo per pezzo, a mano, e decidi al volo se qualcosa non si adatta. In una catena di montaggio, ogni stazione fa un'unica attività specializzata, riceve pezzi già verificati dalla stazione precedente e consegna pezzi verificati a quella successiva. Nessuno si salta un passo né improvvisa: il processo completo è ripetibile, verificabile e si può eseguire migliaia di volte senza supervisione umana costante.

Un **pipeline** di TFX è esattamente quello: una sequenza di **componenti** organizzati come un grafo aciclico dirigito (DAG, Directed Acyclic Graph), cioè un diagramma di passi connessi da frecce dove i dati avanzano solo in avanti e non tornano mai a un passo anteriore. Ogni componente:

- Riceve uno o diversi **artefatti** di input (dati, schemi, modelli) prodotti da componenti anteriori
- Realizza un compito concreto e ben delimitato
- Produce un artefatto di output, che rimane registrato e disponibile per i componenti seguenti

Internamente, ogni componente di TFX si organizza in tre pezzi: un **driver**, che recupera gli artefatti di input e coordina l'esecuzione, un **executor**, che è dove accade il lavoro reale (addestrare, trasformare, validare), e un **publisher**, che registra gli artefatti di output. Come utente del pipeline quasi mai tocchi il driver né il publisher: il tuo codice si concentra nel executor, normalmente attraverso una funzione Python che TFX chiama per te.

### I componenti principali

| Componente | Cosa fa |
|---|---|
| `ExampleGen` | Ingesta i dati di origine (CSV, TFRecord, basi di dati) e li divide in insiemi di addestramento e valutazione |
| `StatisticsGen` | Calcola statistiche descrittive di ogni feature (valori tipici, valori assenti, distribuzioni) |
| `SchemaGen` | Inferisce uno schema (tipi di dato, range, categorie valide) da quelle statistiche |
| `ExampleValidator` | Confronta dati nuovi contro lo schema e segnala anomalie |
| `Transform` | Applica ingegneria di features (normalizzazione, codificazione di categorie) in modo coerente tra addestramento e servizio |
| `Trainer` | Addestra il modello con TensorFlow a partire dai dati (trasformati o no) |
| `Evaluator` | Analizza profondamente il rendimento del modello, anche per sottoinsiemi di dati |
| `Pusher` | Pubblica il modello addestrato verso una destinazione di distribuzione (TensorFlow Serving, TensorFlow Lite, TensorFlow.js, TensorFlow Hub) |

TFX ammette inoltre componenti personalizzati se quelli standard non coprono il tuo caso, e si può eseguire da un notebook (come farai tu), da riga di comando (TFX CLI) o integrato in un IDE.

## Il tuo primo pipeline: ExampleGen, Trainer e Pusher

Il pipeline più semplice possibile con TFX combina solo tre componenti: ingesta dati, addestra un modello e lo pubblica. È esattamente quello che costruisce il notebook `penguin_simple.ipynb` di TFX, usando il dataset Palmer Penguins (pinguini dell'Antartide) per classificare la specie di un pinguino a partire da quattro misure fisiche normalizzate tra 0 e 1: lunghezza e profondità del becco (`culmen_length_mm`, `culmen_depth_mm`), lunghezza della pinna (`flipper_length_mm`) e massa corporea (`body_mass_g`).

Per primo definisci il codice di addestramento in un file Python indipendente. TFX lo chiama il **Trainer generico**: tu scrivi una funzione `run_fn` che riceve gli argomenti di addestramento e salva il modello risultante dove TFX si aspetta di trovarlo.

```python
# penguin_trainer.py — codice che eseguirà il componente Trainer
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils
from tfx import v1 as tfx
from tfx_bsl.public import tfxio

_FEATURE_KEYS = [
    'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'
]
_LABEL_KEY = 'species'

# Poiché non usiamo ancora un componente di validazione, definiamo
# manualmente il formato atteso di ogni feature.
_FEATURE_SPEC = {
    **{f: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for f in _FEATURE_KEYS},
    _LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
}

def _input_fn(file_pattern, data_accessor, schema, batch_size=200):
    """Converte i TFRecords generati da ExampleGen in un tf.data.Dataset."""
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key=_LABEL_KEY),
        schema=schema).repeat()

def _build_keras_model():
    # Una rete densa semplice: quattro input, due strati nascosti, tre output
    # (uno per specie di pinguino).
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
    d = keras.layers.concatenate(inputs)
    for _ in range(2):
        d = keras.layers.Dense(8, activation='relu')(d)
    outputs = keras.layers.Dense(3)(d)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model

# TFX chiama questa funzione: è il punto di input del componente Trainer.
def run_fn(fn_args: tfx.components.FnArgs):
    schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)
    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema, 20)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema, 10)

    model = _build_keras_model()
    model.fit(train_dataset, steps_per_epoch=fn_args.train_steps,
              validation_data=eval_dataset, validation_steps=fn_args.eval_steps)

    # TFX si aspetta il modello salvato esattamente in questo percorso.
    model.save(fn_args.serving_model_dir, save_format='tf')
```

Con il codice di addestramento pronto, definisci il pipeline stesso, encatenando i tre componenti:

```python
from tfx import v1 as tfx

def _create_pipeline(pipeline_name, pipeline_root, data_root,
                      module_file, serving_model_dir, metadata_path):
    # 1. Ingesta il CSV di input e lo divide in train/eval.
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # 2. Addestra il modello usando il codice di penguin_trainer.py.
    trainer = tfx.components.Trainer(
        module_file=module_file,
        examples=example_gen.outputs['examples'],
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=5))

    # 3. Pubblica il modello addestrato in una cartella del sistema di file.
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata
            .sqlite_metadata_connection_config(metadata_path),
        components=[example_gen, trainer, pusher])

# Un orquestatore esegue il DAG. LocalDagRunner corre in locale ed è
# perfetto per sviluppo; in produzione si usano Apache Airflow o Kubeflow.
tfx.orchestration.LocalDagRunner().run(
    _create_pipeline(
        pipeline_name="penguin-simple",
        pipeline_root="pipelines/penguin-simple",
        data_root=DATA_ROOT,
        module_file="penguin_trainer.py",
        serving_model_dir="serving_model/penguin-simple",
        metadata_path="metadata/penguin-simple/metadata.db"))
```

Nota che tu non chiami mai `fit` direttamente né gestisci file a mano: descrivi quali componenti esistono e da dove prende i suoi input ogni uno (`example_gen.outputs['examples']`, `trainer.outputs['model']`), e il `LocalDagRunner` decide l'ordine di esecuzione a partire da quelle dipendenze.

## Validare i dati con TensorFlow Data Validation (TFDV)

Prima di fidarti di qualunque modello, hai bisogno di fidarti dei dati che lo alimentano. TensorFlow Data Validation (TFDV) è la libreria di TFX dedicata a questo, e si appoggia su tre componenti che ampliano il pipeline anteriore, come mostra `penguin_tfdv.ipynb`:

- `StatisticsGen` fa un passaggio completo sui dati e calcola statistiche per feature: valori minimo e massimo, media, numero di valori assenti, istogrammi
- `SchemaGen` esamina quelle statistiche e genera automaticamente uno **schema**: una descrizione formale di quale tipo di dato ha ogni feature, se è obbligatoria e quali valori sono validi. Quello schema rimane salvato in un file `schema.pbtxt` che puoi rivedere e aggiustare a mano
- `ExampleValidator` confronta i dati nuovi contro lo schema (sia quello generato automaticamente sia uno rivisto da una persona) e produce un report di anomalie: valori del tipo sbagliato, categorie inaspettate, features che mancano

L'idea chiave è che generi lo schema una volta, lo rivedi, e poi lo riusi come riferimento per validare ogni nuovo lotto di dati che entra al pipeline, esattamente come un controllo di qualità rivede ogni lotto di pezzi contro un modello già approvato.

```python
def _create_schema_pipeline(pipeline_name, pipeline_root, data_root, metadata_path):
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # NUOVO: calcola statistiche descrittive di ogni feature.
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])

    # NUOVO: inferisce uno schema da quelle statistiche.
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name, pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata
            .sqlite_metadata_connection_config(metadata_path),
        components=[example_gen, statistics_gen, schema_gen])
```

Una volta hai lo schema esportato, lo importi al pipeline di addestramento con un componente `Importer` e aggiungi `ExampleValidator` just prima del `Trainer`, di modo che qualunque anomalia rimanga registrata prima di spendere tempo addestrando con dati sospetti. È importante sapere che, per default, le anomalie rilevate non bloccano il pipeline: rimangono documentate affinché una persona decida cosa fare.

## Trasformare features con TensorFlow Transform (TFT)

I quattro features dell'esempio precedente venivano già normalizzati tra 0 e 1. In un caso reale quasi mai hai quella fortuna: riceverai misure in unità diverse, testo che devi convertire in numeri o valori assenti che devi gestire. TensorFlow Transform (TFT) è la libreria di TFX per questo lavoro, e la sua idea centrale è semplice ma potente: **lo stesso codice di trasformazione si esegue durante l'addestramento e durante il servizio del modello**, così eviti il bias addestramento-servizio che abbiamo menionato prima.

Nel notebook `simple.ipynb` mostra il concetto nella sua forma più ridotta, senza pipeline di mezzo, solo con la libreria `tensorflow_transform`:

```python
import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    """Funzione che TFT esegue sia in addestramento sia in servizio."""
    x = inputs['x']
    y = inputs['y']
    s = inputs['s']

    # tft.mean() fa un passaggio completo su TUTTO il dataset per
    # calcolare la media; non è un'operazione per esempio, come sì lo sono
    # le operazioni normali di TensorFlow.
    x_centered = x - tft.mean(x)
    y_normalized = tft.scale_to_0_1(y)
    # Converte stringhe di testo in indici interi secondo la loro frequenza.
    s_integerized = tft.compute_and_apply_vocabulary(s)

    return {
        'x_centered': x_centered,
        'y_normalized': y_normalized,
        's_integerized': s_integerized,
    }
```

La distinzione importante qui è tra due tipi di operazioni dentro una `preprocessing_fn`:

- Operazioni normali di TensorFlow, che si applicano esempio per esempio e si ripetono uguali in addestramento e in servizio
- **Analizzatori di TFT** (`tft.mean`, `tft.scale_to_z_score`, `tft.compute_and_apply_vocabulary`...), che necessitano un passaggio completo sul dataset di addestramento per calcolare una costante (una media, un vocabolario) e poi incrustad quella costante nel grafo, di modo che il servizio la riusa senza doverla ricalcolare

Dentro un pipeline di TFX, questo si materializza nel componente `Transform`, come mostra `penguin_tft.ipynb` sul dataset Palmer Penguins senza elaborare (con valori in scale diverse e alcune righe incomplete):

```python
# Dentro il modulo che usano Transform e Trainer:
def preprocessing_fn(inputs):
    outputs = {}
    for key in _FEATURE_KEYS:
        # Normalizza ogni feature numerico a media 0 e varianza 1.
        outputs[key] = tft.scale_to_z_score(inputs[key])

    # Converte la specie (testo) in un indice numerico fisso.
    table_keys = ['Adelie', 'Chinstrap', 'Gentoo']
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=table_keys, values=tf.range(len(table_keys), dtype=tf.int64),
        key_dtype=tf.string, value_dtype=tf.int64)
    table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    outputs[_LABEL_KEY] = table.lookup(inputs[_LABEL_KEY])
    return outputs
```

```python
# Nella definizione del pipeline:
transform = tfx.components.Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_importer.outputs['result'],
    module_file=module_file)  # qui vive preprocessing_fn

trainer = tfx.components.Trainer(
    module_file=module_file,
    examples=example_gen.outputs['examples'],
    transform_graph=transform.outputs['transform_graph'],  # NUOVO
    train_args=tfx.proto.TrainArgs(num_steps=100),
    eval_args=tfx.proto.EvalArgs(num_steps=5))
```

Il `Trainer` riceve ora il grafo di trasformazione (`transform_graph`) oltre ai dati, e il modello esportato include quel grafo come parte della sua funzione di servizio (`serve_tf_examples_fn`). Così, quando il modello è già in produzione e riceve una richiesta nuova, applica esattamente le stesse trasformazioni che ha usato durante l'addestramento, senza che nessuno debba duplicare quel codice a mano.

## Analizzare il modello con TensorFlow Model Analysis (TFMA)

Un modello può avere una precisione (accuracy) globale accettabile e, nonostante ciò, fallire sistematicamente per un sottogruppo di dati che al tuo business importa molto: per esempio, un singolo indicatore di "precisione: 92%" può nascondere che il modello predice molto male proprio per il segmento di clienti o di prodotti più redditizio. TensorFlow Model Analysis (TFMA) è la libreria che ti permette di andare oltre la metrica globale.

In un pipeline di TFX, questo si materializza nel componente `Evaluator`, che confronta il modello appena addestrato contro il modello anteriore "benedetto" (blessed, l'ultimo che ha superato la validazione) e decide se il nuovo modello è idoneo per la produzione. Il notebook `penguin_tfma.ipynb` aggiunge questo componente al pipeline base:

```python
import tensorflow_model_analysis as tfma

# Individua l'ultimo modello che ha passato la validazione, per usarlo come riferimento.
model_resolver = tfx.dsl.Resolver(
    strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
    model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
    model_blessing=tfx.dsl.Channel(type=tfx.types.standard_artifacts.ModelBlessing)
).with_id('latest_blessed_model_resolver')

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='species')],
    slicing_specs=[
        tfma.SlicingSpec(),                          # il dataset completo
        tfma.SlicingSpec(feature_keys=['species']),  # ogni specie per separato,
    ],
    metrics_specs=[
        tfma.MetricsSpec(per_slice_thresholds={
            'sparse_categorical_accuracy': tfma.PerSliceMetricThresholds(thresholds=[
                tfma.PerSliceMetricThreshold(
                    slicing_specs=[tfma.SlicingSpec()],
                    threshold=tfma.MetricThreshold(
                        # Il modello deve superare il 60% di precisione...
                        value_threshold=tfma.GenericValueThreshold(lower_bound={'value': 0.6}),
                        # ...e non peggiorare rispetto al modello anteriore.
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10})))
            ]),
        })])

evaluator = tfx.components.Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)

# Pusher pubblica il modello solo se l'Evaluator lo "benedice".
pusher = tfx.components.Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(base_directory=serving_model_dir)))
```

Le `slicing_specs` (specifiche di segmentazione) sono il pezzo più importante di questo esempio: invece di calcolare la precisione solo su tutto il dataset, TFMA la ricalcola inoltre per ogni specie di pinguino per separato. Questo tipo di analisi per sottoinsiemi — quello che TFMA chiama slicing — è proprio quello che ti permette di scoprire che, anche se il tuo modello funziona bene in generale, fallisce in un segmento concreto che il tuo business non può permettersi di ignorare. Pensa all'esempio di un negozio di scarpe online che usa un modello per decidere quanto stock ordinare: una caduta di precisione solo in "scarpe da vestire per uomo" può passare inosservata nella metrica globale e, nonostante ciò, rappresentare una perdita importante se quella linea è la più redditizia.

Questa idea si raccoglie in quello che la documentazione di TFX chiama il **triangolo di insights di ML**: quando un modello fallisce di fronte al business, quasi sempre è perché si è violato uno di tre presupposti: i dati non sono più rappresentativi, il modello è troppo semplice per il problema attuale o le condizioni di business sono cambiate (nuovi fornitori, prodotti, comportamenti della clientela). Per questo TFX non tratta la valutazione come un passo unico alla fine del progetto, ma come un processo continuo: ogni nuova versione del modello si confronta contro la precedente, e ogni caduta di rendimento si indaga iniziando dai dati, con gli stessi strumenti di TFDV che già conosci.

## Il metadata: la memoria del pipeline

Ogni volta che si esegue un componente di TFX, il risultato — quali dati sono entrati, quali parametri sono stati usati, quale artefatto è stato generato — si registra in un archivio chiamato **ML Metadata (MLMD)**, normalmente supportato da una banca dati relazionale (SQLite negli esempi di questo corso, MySQL in produzione). Pensa al metadata come al diario di bordo della fabbrica: non solo cosa è stato prodotto, ma con quali materiali, in quale macchina e con quale risultato, esecuzione per esecuzione.

Quel registro ti dà diverse capacità che sarebbero molto difficili da replicare a mano:

- **Tracciabilità (lineage)**: puoi tracciare all'indietro quale versione esatta dei dati ha addestrato un modello concreto, o in avanti quali modelli hanno usato un dataset determinato
- **Confronto tra esecuzioni**: poiché ogni esecuzione rimane registrata, puoi confrontare il rendimento della versione di oggi del modello con quella della scorsa settimana
- **Riuso di risultati (caching)**: se l'input e il codice di un componente non sono cambiati dall'ultima esecuzione, TFX può riusare il risultato precedente invece di ricalcolarlo, il che risparmi tempo specialmente in passi costosi come la trasformazione di dati
- **Riproducibilità e audit**: in settori regolati, potere dimostrare esattamente quali dati e quale processo hanno prodotto un modello può essere un requisito legale, non solo una buona pratica

Come studente non interagisci quasi mai col metadata direttamente oltre a indicare dove salvarlo (`metadata_path` negli esempi di questo capitolo); è TFX che lo consulta e aggiorna per te in ogni passo del pipeline.

## Elaborazione distribuita con Apache Beam

I pipeline di questo capitolo si sono eseguiti nella tua propria macchina, su un dataset di poche centinaia di righe. In produzione, i dataset possono avere milioni di righe, e calcolare statistiche o applicare trasformazioni su tutti quei dati in una sola macchina sarebbe, semplicemente, troppo lento.

Per questo molti componenti di TFX — `ExampleGen`, `StatisticsGen` e `Transform`, tra gli altri — non elaborano i dati direttamente: delegano quel lavoro in **Apache Beam**, un modello di programmazione unificato per elaborazione di dati per lotti (batch) e in flusso continuo (streaming). Beam separa la logica di "quale trasformazione applicare" di "dove si esegue": lo stesso pipeline di Beam può correre nel tuo portatile con il runner locale (quello che usano gli esempi di questo capitolo), o distribuirsi su un cluster con Apache Spark, Apache Flink o Google Cloud Dataflow, senza cambiare una riga del codice di trasformazione.

Non tutti i componenti hanno bisogno di Beam: il `Trainer` si appoggia principalmente sulle capacità di addestramento distribuito di TensorFlow, e componenti come il `Pusher` hanno bisogno solo di Python puro per copiare un modello da un sito all'altro. Il vantaggio per te come sviluppatore è che quasi mai devi pensare a questa layer: scrivi la tua `preprocessing_fn` una volta e TFX si incarica di eseguirla, sia su cento righe nel tuo portatile sia su cento milioni in un cluster nel cloud.

## Riassunto dei punti chiave

- TensorFlow Extended (TFX) è la piattaforma di Google per portare modelli di Machine Learning (ML) da un notebook a un sistema di produzione affidabile, ripetibile e verificabile
- Un pipeline di TFX è un grafo aciclico dirigito (DAG) di componenti; ogni componente prende artefatti di input, realizza un compito e produce artefatti di output
- Il pipeline minimo combina tre componenti: `ExampleGen` (ingesta dati), `Trainer` (addestra il modello) e `Pusher` (lo pubblica)
- TensorFlow Data Validation (TFDV) aggiunge `StatisticsGen`, `SchemaGen` e `ExampleValidator` per calcolare statistiche, inferire uno schema dei dati e rilevare anomalie prima di addestrare
- TensorFlow Transform (TFT) aggiunge il componente `Transform`: la funzione `preprocessing_fn` che scrivi si esegue uguale in addestramento e in servizio, evitando il bias addestramento-servizio
- TensorFlow Model Analysis (TFMA) aggiunge il componente `Evaluator`, che analizza il rendimento del modello per segmenti di dati (slicing) e decide, confrontando con il modello anteriore, se il nuovo modello deve "benedirsi" per la produzione
- Il metadata (MLMD) registra ogni esecuzione del pipeline, il che permette tracciabilità, confronto tra versioni e riuso di risultati già calcolati
- Componenti come `ExampleGen`, `StatisticsGen` e `Transform` si eseguono su Apache Beam, il che permette scalare lo stesso codice da un portatile a un cluster distribuito senza riscriverlo

## Mettilo in pratica

Apri il notebook di pratica di questa classe e parti dal pipeline base di tre componenti (`ExampleGen`, `Trainer`, `Pusher`) sul dataset Palmer Penguins. Il tuo compito è aggiungere un componente `StatisticsGen` e verificare lo schema inferito: esegui il pipeline, individua l'artefatto di statistiche mediante il metadata del pipeline e osserva cosa ti dice su ognuno dei quattro features numerici (minimo, massimo, valori assenti). Dopo, rispondi per scritto a questa domanda: se domani arrivasse un nuovo lotto di dati con una specie di pinguino che non esisteva prima, quale componente di TFX dovrebbe rilevare il problema, e in quale artefatto di output ti aspetteresti di vederlo riflesso?
