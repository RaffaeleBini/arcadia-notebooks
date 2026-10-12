
# Google Vision API for Dummies

### Manuale di studio per il corso EITC/AI/GVAPI (European IT Certification – Artificial Intelligence – Google Vision API)

---

## Come usare questo manuale

Hai tra le mani una guida di studio pensata per accompagnarti lungo le otto lezioni del corso di certificazione EITC/AI/GVAPI sulla Google Vision API (interfaccia di programmazione di applicazioni per la visione artificiale di Google). Ogni lezione riassume i concetti chiave, ricostruisce il codice di esempio (a volte frammentato nel materiale originale) in frammenti eseguibili e si conclude con le idee che ricorrono più frequentemente nelle domande di verifica ufficiali.

Questo manuale è la parte teorica. Per praticare, apri gli otto notebook Jupyter della cartella `notebooks/` — sono numerati nello stesso ordine delle lezioni e contengono celle di codice con esercizi guidati e le loro soluzioni espandibili.

**Iniziamo!**

---

## Elenco di acronimi

| Acronimo | Significato |
|---|---|
| API | Interfaccia di Programmazione di Applicazioni (Application Programming Interface) |
| CNN | Rete Neurale Convoluzionale (Convolutional Neural Network) |
| GCP | Google Cloud Platform |
| GCS | Archiviazione Cloud di Google (Google Cloud Storage) |
| IA | Intelligenza Artificiale |
| IAM | Gestione di Identità e Accessi (Identity and Access Management) |
| JSON | Notazione di Oggetti JavaScript (JavaScript Object Notation) |
| NLP | Elaborazione del Linguaggio Naturale (Natural Language Processing) |
| OCR | Riconoscimento Ottico dei Caratteri (Optical Character Recognition) |
| PIL | Python Imaging Library, nome storico della libreria oggi distribuita come Pillow |
| REST | Trasferimento di Stato Rappresentativo (Representational State Transfer) |
| RGB | Modello di colore Rosso, Verde e Blu (Red, Green, Blue) |
| SDK | Kit di Sviluppo Software (Software Development Kit) |
| SSD | Rilevatore Multibox a Scatto Singolo (Single Shot Multibox Detector) |
| TIFF | Formato di File Immagine Etichettato (Tagged Image File Format) |
| URI | Identificatore Uniforme di Risorsa (Uniform Resource Identifier) |
| URL | Locatore Uniforme di Risorsa (Uniform Resource Locator) |

---

## Indice

1. [Lezione 1 — Introduzione](#lezione-1--introduzione)
2. [Lezione 2 — Primi passi: configurazione e setup](#lezione-2--primi-passi-configurazione-e-setup)
3. [Lezione 3 — Comprensione del testo nei dati visuali](#lezione-3--comprensione-del-testo-nei-dati-visuali)
4. [Lezione 4 — Comprensione di immagini](#lezione-4--comprensione-di-immagini)
5. [Lezione 5 — Etichettatura di immagini](#lezione-5--etichettatura-di-immagini)
6. [Lezione 6 — Comprensione avanzata di immagini](#lezione-6--comprensione-avanzata-di-immagini)
7. [Lezione 7 — Dati visuali web](#lezione-7--dati-visuali-web)
8. [Lezione 8 — Forme e oggetti con Pillow](#lezione-8--forme-e-oggetti-con-pillow)
9. [Allegato A — Prezzi e quote](#allegato-a--prezzi-e-quote)
10. [Allegato B — Buone pratiche ed errori comuni](#allegato-b--buone-pratiche-ed-errori-comuni)
11. [Allegato C — Risorse per continuare a imparare](#allegato-c--risorse-per-continuare-a-imparare)

---

## Lezione 1 — Introduzione

### 1.1 Introduzione a Google Cloud Vision API

L'intelligenza artificiale (IA) consente alle macchine di svolgere compiti che una volta richiedevano intelligenza umana. La visione artificiale è una delle sue applicazioni più evidenti: la capacità di interpretare dati visuali. Google mette questa capacità a tua disposizione attraverso la **Google Cloud Vision API**, un servizio cloud che analizza immagini con modelli di machine learning già addestrati e ti restituisce informazioni strutturate in formato JSON.

Non hai bisogno di addestrare alcun modello personalizzato: invii un'immagine, ti autentichi con le tue credenziali e ricevi la risposta. L'API supporta diversi linguaggi di programmazione; in questo corso lavorerai con Python.

**Cosa può rilevare l'API:**

- **Etichette (labels)**: oggetti, scene e concetti presenti nell'immagine — ad esempio, "cane", "albero" o "auto"
- **Volti (faces)**: posizione di occhi, naso e bocca, oltre a emozioni come gioia, rabbia, sorpresa o tristezza, e una stima dell'intervallo di età
- **Testo (OCR)**: estrae testo stampato o manoscritto e lo converte in informazioni leggibili dalla macchina
- **Oggetti (object localization)**: localizza oggetti specifici all'interno dell'immagine, utile ad esempio per identificare prodotti su uno scaffale
- **Contenuto esplicito, monumenti e loghi**: funzionalità avanzate che vedrai nella lezione 6

Un esempio che illustra bene il potenziale dell'API: immagina un robot Raspberry Pi con fotocamera. Il robot non esegue il modello di IA nel suo hardware — invia l'immagine catturata al cloud, la Vision API l'analizza e gli restituisce il risultato in tempo reale. Con il rilevamento di volti ed emozioni, il robot può avvicinarsi a una faccia sorridente o allontanarsi davanti a un'espressione di rabbia o sorpresa.

Questo processo, chiamato **rilevamento di entità**, confronta le caratteristiche estratte dall'immagine rispetto a un database di entità note e assegna a ogni risultato un **punteggio di confidenza**, che ti permette di impostare una soglia e scartare risultati poco affidabili.

> **Approfondisci**: consulta la documentazione ufficiale su cloud.google.com/vision, i corsi del Google Cloud Learning Center e i webinar che Google organizza sull'IA.

### 1.2 Vision API rispetto a AutoML Vision API

All'interno di "Vision AI", Google offre due servizi distinti ed è importante sapere come differenziarli prima di scrivere una sola riga di codice:

| | **Vision API** | **AutoML Vision API** |
|---|---|---|
| Modello | Preaddestrato da Google | Addestri il tuo modello con le tue immagini |
| Interfaccia | Codice (API REST o client Python) | Interfaccia grafica semplice per caricare e etichettare immagini |
| Quando sceglierla | Ti basta il riconoscimento generico già addestrato da Google | Hai bisogno di riconoscere categorie molto specifiche del tuo business (ad esempio, distinguere modelli specifici di macchinari) |
| Costo | Pagamento al consumo, senza costo di addestramento | Costo aggiuntivo per l'addestramento del modello personalizzato |

Un esempio che appare nel materiale originale: se carichi una foto di un pappagallo nella Vision API, il servizio non solo riconosce "pappagallo" — può identificare la specie esatta (pappagallo cenerino africano) grazie ai dati già raccolti da Google.

**Funzionalità aggiuntive di Vision API che vedrai in dettaglio più avanti:**

- **Proprietà dell'immagine**: colori dominanti, istogramma dei colori e la loro percentuale di presenza (lezione 4)
- **Safe Search**: probabilità che un'immagine contenga contenuto per adulti, violento o inadatto (lezione 6)
- **Rilevamento web**: se l'immagine appare da qualche parte su internet, con quali pagine ed entità è correlata (lezione 7)
- **Ricerca di prodotti**: elenco di articoli visivamente simili all'immagine fornita

**Prezzi (orientativi, soggetti a modifiche nella documentazione ufficiale di Google):** disponi di mille chiamate gratuite al mese combinando tutte le categorie; da lì in poi, Google addebita per ogni mille unità aggiuntive elaborate, a seconda della categoria.

**Casi d'uso reali citati nel materiale**: Box.com e The New York Times usano Vision AI in produzione, e la stessa app Foto di Google la utilizza per organizzare automaticamente le fotografie per categorie.

Per installare la libreria client di Python avrai bisogno del pacchetto `google-cloud-vision`, che installerai con pip (il programma di installazione dei pacchetti Python) — vedrai il processo completo, passo dopo passo, nella lezione 2.

---

## Lezione 2 — Primi passi: configurazione e setup

Prima di scrivere la tua prima chiamata all'API hai bisogno di quattro cose: un progetto in Google Cloud Platform (GCP), la Vision API abilitata in quel progetto, credenziali valide e la libreria client installata nel tuo ambiente Python. Segui questi passaggi in ordine.

### Passaggio 1 — Crea un progetto in Google Cloud Platform

1. Accedi a [Google Cloud Console](https://console.cloud.google.com/) con il tuo account Google
2. Crea un nuovo progetto (o selezionane uno esistente) e annota il suo **ID progetto**
3. Conferma che il progetto ha la fatturazione (billing) configurata — il livello gratuito è sufficiente per praticare con questo corso

### Passaggio 2 — Abilita la Vision API

1. Nel menu laterale, vai a **APIs & Services > Library**
2. Cerca "Vision API" e aprila
3. Premi **Enable** per attivarla nel tuo progetto

### Passaggio 3 — Crea un account di servizio e scarica la tua chiave JSON

Un **account di servizio** (service account) è un'identità non umana che la tua applicazione usa per autenticarsi rispetto all'API, invece di farlo tu con il tuo utente personale.

1. Vai a **APIs & Services > Credentials**
2. Premi **Create Credentials > Service account**
3. Dagli un nome riconoscibile e assegnagli il ruolo **Editor** (o uno più restrittivo, come "Cloud Vision API User", se preferisci applicare il principio del privilegio minimo)
4. All'interno dell'account di servizio appena creato, vai alla scheda **Keys > Add Key > Create new key**, scegli il formato **JSON** e scarica il file
5. Salva quel file `.json` in un luogo sicuro del tuo computer — è una credenziale sensibile: non caricarlo mai in un repository pubblico

> **Salvalo bene!** Quel file JSON dà accesso al tuo progetto GCP. Trattalo con la stessa cura che useresti per una password.

### Passaggio 4 — Crea un ambiente virtuale

Un **ambiente virtuale** è una directory isolata che incapsula le dipendenze del tuo progetto, evitando conflitti di versione con altri progetti Python che hai nel tuo computer. Offre quattro vantaggi: isolamento delle dipendenze, riproducibilità tra macchine, uno spazio sicuro per sperimentare senza influire sul resto del sistema e facilità di condividere il progetto con il tuo team.

```bash
python -m venv venv_gvapi

# Su Windows
venv_gvapi\Scripts\activate

# Su macOS / Linux
source venv_gvapi/bin/activate
```

### Passaggio 5 — Installa la libreria client

Con l'ambiente virtuale attivato:

```bash
pip --version
pip install google-cloud-vision
pip list
```

`pip --version` conferma che pip funziona correttamente; `pip install google-cloud-vision` scarica la libreria client ufficiale; `pip list` ti permette di verificare che l'installazione sia stata completata.

### Passaggio 6 — Configura la variabile di ambiente delle credenziali

La libreria client cerca automaticamente le tue credenziali nella variabile di ambiente **`GOOGLE_APPLICATION_CREDENTIALS`**. Puoi definirla dal terminale o direttamente nel tuo script Python:

```python
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "percorso/alla/tua/chiave.json"
```

### Passaggio 7 — Crea il tuo primo client e verificalo

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

# Elenca i metodi disponibili del client per esplorare l'API
print(dir(client))
```

Se questa cella si esegue senza errori, il tuo ambiente è pronto! Puoi già chiamare metodi come `client.label_detection()`, `client.text_detection()`, `client.face_detection()` o `client.safe_search_detection()`, che vedremo lezione dopo lezione.

### Alternativa rapida: chiave API e gcloud CLI

Il materiale originale descrive anche un percorso più diretto, pensato per test rapidi dalla riga di comando, invece di un'applicazione Python:

1. Abilita la Vision API nella libreria API della console GCP
2. Vai a **Credentials > Create Credentials > API key**
3. Installa [Google Cloud SDK](https://cloud.google.com/sdk) nel tuo computer
4. Autentica l'SDK: `gcloud auth login`
5. Prova l'API direttamente: `gcloud ml vision detect-labels percorso/a/immagine.jpg`

Usa questa strada solo per test rapidi da riga di comando; per lo sviluppo di applicazioni con Python, l'account di servizio del flusso precedente è il percorso consigliato.

---

## Lezione 3 — Comprensione del testo nei dati visuali

### 3.1 Rilevare ed estrarre testo da un'immagine

Il metodo `text_detection` localizza regioni con testo all'interno di un'immagine e riconosce i suoi caratteri, sia stampati che manoscritti e in diverse lingue. Insieme al testo, l'API ti restituisce la lingua rilevata (`locale`), un punteggio di confidenza e le coordinate del riquadro delimitatore (`bounding_poly`) di ogni frammento.

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("immagine.jpg", "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.text_detection(image=image)
texts = response.text_annotations

# Il primo elemento contiene tutto il testo rilevato;
# i seguenti sono ogni parola/frammento separatamente
for text in texts:
    print(text.description)
```

Incapsulato in una funzione riutilizzabile:

```python
def detect_text(image_path):
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as img_file:
        content = img_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    annotations = response.text_annotations

    if annotations:
        return annotations[0].description
    return None
```

Se l'immagine è ospitata pubblicamente, invece di leggere byte da un file locale, assegna l'URL direttamente a `image.source.image_uri`:

```python
def detect_text_from_url(image_url):
    client = vision.ImageAnnotatorClient()

    image = vision.Image()
    image.source.image_uri = image_url

    response = client.text_detection(image=image)
    annotations = response.text_annotations

    if annotations:
        return annotations[0].description
    return None
```

**Pulire il testo estratto con pandas**: una volta che hai il testo in un DataFrame, puoi normarlo con espressioni regolari e con i metodi string di pandas:

```python
import pandas as pd
import re

df["text"] = df["text"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x))
df["text"] = df["text"].str.lower()
df["words"] = df["text"].apply(lambda x: x.split(" "))
```

**Applicazioni tipiche**: digitalizzazione di documenti, traduzione di cartelloni o menu, moderazione dei contenuti, estrazione di prezzi o nomi di prodotti in annunci e accessibilità (convertire il testo di un'immagine in audio).

### 3.2 Rilevare ed estrarre testo manoscritto

Il testo manoscritto è più difficile da riconoscere di quello stampato perché la scrittura varia molto da una persona all'altra. La Vision API segue quattro fasi: preelaborazione dell'immagine (ridimensionamento, riduzione del rumore, binarizzazione), rilevamento delle regioni con testo, segmentazione in unità più piccole e infine riconoscimento tramite OCR.

Per documenti (curriculum, note manoscritte, moduli), usa **`document_text_detection`** invece di `text_detection`: il risultato è organizzato gerarchicamente e ti dà un livello di confidenza a ogni livello.

```python
from google.cloud import vision

def extract_text_from_document(image_path):
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    for page in document.pages:
        for block in page.blocks:
            print("Confidenza del blocco:", block.confidence)
            for paragraph in block.paragraphs:
                print("Confidenza del paragrafo:", paragraph.confidence)
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    print(word_text)

    return document.text
```

La gerarchia della risposta è sempre la stessa: **pagine (`pages`) → blocchi (`blocks`) → paragrafi (`paragraphs`) → parole (`words`) → simboli (`symbols`)**, e ogni livello porta il suo `confidence` (un valore decimale tra zero e uno, dove uno è la certezza massima).

**A cosa serve la confidenza?** Per decidere se conviene una revisione manuale, per impostare una soglia minima in un processo automatizzato e per dare priorità a certi risultati rispetto ad altri.

**Limitazioni da tenere presenti:**

- La qualità dell'immagine di input (sfocatura, scarsa illuminazione) riduce la precisione
- I layout complessi (più colonne, tabelle, testo manoscritto mescolato con stampato) possono frammentare o omettere testo
- La calligrafia leggibile si riconosce meglio della corsiva elaborata
- Le lingue e le scritture poco comuni hanno minore precisione dell'inglese

### 3.3 Rilevare ed estrarre testo da file (PDF/TIFF)

Per elaborare file completi — non solo immagini singole — in formato PDF o TIFF, combina la Vision API con **Google Cloud Storage (GCS)**, il servizio di archiviazione cloud di Google. Il file deve risiedere in un **bucket** di GCS prima che l'API possa elaborarlo.

**Limite e prezzo**: ogni file supporta un massimo di duemila pagine; il prezzo è calcolato per unità, dove una pagina conta come una unità, con mille unità gratuite al mese e un prezzo orientativo di 0,60 $ per unità per "Document Text Detection" (soggetto alla tabella dei prezzi ufficiale vigente).

```bash
pip install --upgrade google-cloud-storage
```

**Flusso asincrono consigliato** (elaborazione in lotto via GCS):

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

gcs_source_uri = "gs://il-tuo-bucket/ingresso.pdf"
gcs_destination_uri = "gs://il-tuo-bucket/uscita/"

feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

gcs_source = vision.GcsSource(uri=gcs_source_uri)
input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")

gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=2)

async_request = vision.AsyncAnnotateFileRequest(
    features=[feature], input_config=input_config, output_config=output_config
)

operation = client.async_batch_annotate_files(requests=[async_request])
print("Elaborazione in corso...")
operation.result(timeout=180)
```

**Leggere il risultato da Google Cloud Storage:**

```python
from google.cloud import storage
import re
import json

storage_client = storage.Client()

match = re.match(r"gs://([^/]+)/(.+)", gcs_destination_uri)
bucket_name = match.group(1)
prefix = match.group(2)

bucket = storage_client.get_bucket(bucket_name)
blob_list = list(bucket.list_blobs(prefix=prefix))

output = blob_list[0]
json_string = output.download_as_string()
response_json = json.loads(json_string)

first_page_text = response_json["responses"][0]["fullTextAnnotation"]["text"]
print(first_page_text)
```

**Alternativa senza GCS**: se preferisci non usare un bucket, converti il PDF in immagini localmente con `pdf2image` e applica `text_detection` pagina per pagina:

```python
from pdf2image import convert_from_bytes

with open("file.pdf", "rb") as f:
    content = f.read()

images = convert_from_bytes(content)

for image in images:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    response = client.text_detection(image=vision.Image(content=image_bytes.getvalue()))
    for text in response.text_annotations:
        print(text.description)
```

Questa alternativa è più semplice da configurare per un notebook di pratica, anche se perde i vantaggi dell'elaborazione asincrona in lotto e dello sconto per volume di GCS.

**Applicazioni**: ricerca di clausole in contratti legali, analisi del sentiment e classificazione del testo (NLP), data mining nei fascicoli medici e generazione automatica di riassunti.

---

## Lezione 4 — Comprensione di immagini

### 4.1 Rilevamento di suggerimenti di ritaglio (crop hints)

Il metodo `crop_hints` suggerisce la regione più "saliente" (quella che concentra il contenuto visivo più rilevante) di un'immagine, come seed per ritagliarla secondo un rapporto di aspetto che tu indichi. È utile per adattare immagini grandi a diversi formati (miniature, banner, viste mobili) senza perdere il punto di interesse principale.

```python
from google.cloud import vision

def detect_crop_hints(image_path, aspect_ratios):
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    crop_hints_params = vision.CropHintsParams(aspect_ratios=aspect_ratios)
    image_context = vision.ImageContext(crop_hints_params=crop_hints_params)

    response = client.crop_hints(image=image, image_context=image_context)

    for hint in response.crop_hints_annotation.crop_hints:
        print("Riquadro delimitatore:", hint.bounding_poly)
        print("Confidenza:", hint.confidence)
        print("Frazione di importanza:", hint.importance_fraction)

    return response.crop_hints_annotation.crop_hints

detect_crop_hints("immagine.jpg", aspect_ratios=[16 / 9, 4 / 3, 1.0])
```

**Parametri che puoi regolare:**

- **`image`**: JPEG o PNG, fino a 32 megabyte
- **`max_results`**: numero massimo di suggerimenti da restituire, da uno a dieci
- **`aspect_ratios`**: elenco dei rapporti larghezza/altezza desiderati — ad esempio, `[1.0, 1.333]` per 1:1 e 4:3
- **`crop_hints_params.confidence_threshold`**: scarta suggerimenti al di sotto di una soglia di confidenza

Ogni risultato include un `bounding_poly` con i vertici del ritaglio suggerito, una `confidence` (livello di confidenza) e una `importance_fraction` (quale proporzione del suggerimento copre contenuto rilevante).

### 4.2 Rilevamento di volti

`face_detection` localizza uno o più volti in un'immagine e, per ognuno, ti dà il suo riquadro delimitatore, i suoi punti di riferimento facciali (`landmarks`: occhi, naso, bocca, orecchi) e la probabilità di quattro emozioni — gioia, tristezza, rabbia e sorpresa.

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("gruppo.jpg", "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.face_detection(image=image)
faces = response.face_annotations

likelihood_names = (
    "UNKNOWN",
    "VERY_UNLIKELY",
    "UNLIKELY",
    "POSSIBLE",
    "LIKELY",
    "VERY_LIKELY",
)

for face in faces:
    print("Gioia:", likelihood_names[face.joy_likelihood])
    print("Rabbia:", likelihood_names[face.anger_likelihood])
    print("Sorpresa:", likelihood_names[face.surprise_likelihood])
    print("Tristezza:", likelihood_names[face.sorrow_likelihood])
    print("Confidenza di rilevamento:", face.detection_confidence)
```

Ognuno di questi attributi usa la stessa **enumerazione di verosimiglianza (`Likelihood`)**, con sei livelli: `UNKNOWN`, `VERY_UNLIKELY`, `UNLIKELY`, `POSSIBLE`, `LIKELY` e `VERY_LIKELY`. La vedrai ripetuta in Safe Search (lezione 6), quindi vale la pena familiarizzarsi con essa adesso.

**Privacy**: l'API è progettata per non memorizzare né conservare le immagini che le invii, e offre opzioni per regolare il livello di sensibilità del rilevamento.

**Una limitazione importante**: la precisione dipende dal fatto che il viso sia completamente visibile. In un esempio del materiale originale, una foto di sei persone ha rilevato solo cinque volti — il sesto era parzialmente nascosto e l'API non lo ha riconosciuto. Tieni questo in considerazione se la tua applicazione dipende da un conteggio esatto di persone.

### 4.3 Rilevamento di proprietà dell'immagine

`image_properties` estrae attributi generali di un'immagine: i suoi colori dominanti (con valori RGB e un punteggio di prevalenza) e — in modo più generale — indizi sulla sua orientazione e il suo tipo (fotografia, clipart o illustrazione).

```python
from google.cloud import vision

def get_dominant_colors(image_path):
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.image_properties(image=image)
    colors = response.image_properties_annotation.dominant_colors.colors

    dominant_colors = []
    for color_info in colors:
        rgb = (color_info.color.red, color_info.color.green, color_info.color.blue)
        dominant_colors.append(
            {"rgb": rgb, "score": color_info.score, "pixel_fraction": color_info.pixel_fraction}
        )

    return dominant_colors
```

Ogni colore viene con tre valori: `color` (RGB), `score` (confidenza/prevalenza) e `pixel_fraction` (quale proporzione di pixel dell'immagine corrisponde a quel colore).

**Applicazioni**: ricerca di immagini per colore, categorizzazione (foto di prodotto rispetto a illustrazione), design grafico, analisi di tendenze di colore nella moda o nell'interior design, e correzione automatica dell'orientamento.

**Tieni presente**: i colori dominanti riflettono la prevalenza globale dei colori nell'immagine, non necessariamente l'elemento visivamente più importante della scena.

---

## Lezione 5 — Etichettatura di immagini

`label_detection` identifica oggetti, scene e persino concetti astratti (ad esempio, "felicità") presenti in un'immagine, e restituisce un elenco di etichette con il loro punteggio di confidenza.

```python
import os
from google.cloud import vision

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/percorso/alla/tua/chiave.json"

def extract_labels(image_path):
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    return [(label.description, label.score, label.topicality) for label in labels]

print(extract_labels("gatto.jpg"))
# [('cat', 0.98, 0.98), ('whiskers', 0.91, 0.91), ('small to medium-sized cats', 0.87, 0.87), ...]
```

**Campi della risposta:**

- `description`: testo dell'etichetta (ad esempio, "cat")
- `score`: punteggio di confidenza
- `topicality`: importanza dell'etichetta all'interno dell'immagine

> **Dettaglio da tenere presente**: nella versione dell'API descritta nel materiale del corso, `score` e `topicality` restituiscono lo stesso valore — un comportamento documentato che Google potrebbe correggere in versioni future. Non dimenticare di verificarlo tu stesso con le tue immagini nel notebook di pratica.

**Requisiti e limiti**: formati JPEG, PNG o GIF, con una dimensione massima di quattro megapixel.

**Errori comuni su cui prestare attenzione**: errori di autenticazione, problemi di rete, quota API esaurita, immagine corrotta o in un formato non supportato, e parametri non corretti nella chiamata. Avvolgi sempre le tue chiamate in un blocco `try/except`.

**Casi d'uso**: etichettatura automatica di prodotti nell'e-commerce, moderazione dei contenuti e organizzazione di grandi collezioni di immagini (gestione di asset digitali).

---

## Lezione 6 — Comprensione avanzata di immagini

### 6.1 Rilevamento di monumenti (landmarks)

`landmark_detection` riconosce strutture naturali o costruite dall'uomo con rilevanza culturale, storica o geografica — la Torre Eiffel, la Statua della Libertà, il Taj Mahal. Confronta le caratteristiche visive estratte dall'immagine rispetto a un database mondiale di monumenti noti.

```python
from google.cloud import vision

def detect_landmark(file_path):
    client = vision.ImageAnnotatorClient()

    with open(file_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.landmark_detection(image=image)

    for annotation in response.landmark_annotations:
        print("Monumento:", annotation.description)
        print("Ubicazione:", annotation.locations)
        print("Confidenza:", annotation.score)

    return response.landmark_annotations
```

La risposta include `description` (nome del monumento), `score` (confidenza) e `bounding_poly` (posizione del monumento all'interno dell'immagine). Funziona anche con il monumento parzialmente nascosto, da angoli inusuali o con poca luce.

### 6.2 Rilevamento di loghi

`logo_detection` identifica marchi e organizzazioni all'interno di un'immagine, anche con il logo parzialmente nascosto o distorto. Combina localizzazione di oggetti, classificazione di immagini e deep learning, e può essere integrato con il riconoscimento del testo (OCR) per leggere il testo che forma parte del logo stesso.

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("negozio.jpg", "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.logo_detection(image=image)
logos = response.logo_annotations

for logo in logos:
    print("Logo:", logo.description)
    print("Confidenza:", logo.score)
    print("Riquadro delimitatore:", logo.bounding_poly)
```

In un test descritto nel materiale originale, l'API ha riconosciuto senza problemi i loghi di Apple, Nintendo, Best Buy e Nike, ma non ha potuto identificare quelli di IBM, Circuit City e Playboy — la precisione globale si è attestata intorno al 90%. I loghi con design molto semplici (come quello di GAP) o molto intricati (come gli anelli intrecciati di Audi) sono quelli che presentano più difficoltà.

### 6.3 Rilevamento di oggetti

`object_localization` combina due compiti: **classificazione** (che cosa è ogni oggetto) e **localizzazione** (dove si trova, con il suo riquadro delimitatore). Internamente usa un modello **SSD (rilevatore multibox a scatto singolo)**, un'architettura di rete neurale convoluzionale (CNN) progettata per essere precisa ed efficiente, addestrata con milioni di immagini annotate.

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("salotto.jpg", "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.object_localization(image=image)
objects = response.localized_object_annotations

for obj in objects:
    print(f"{obj.name} (confidenza: {obj.score:.2f})")
    vertices = [(v.x, v.y) for v in obj.bounding_poly.normalized_vertices]
    print("Vertici normalizzati:", vertices)
```

Nota che `bounding_poly.normalized_vertices` restituisce coordinate **normalizzate** tra zero e uno; per disegnarle sull'immagine dovrai moltiplicarle per la larghezza e l'altezza reali (lo vedrai nella lezione 8, con Pillow).

**Organizzare i risultati con pandas:**

```python
import pandas as pd

records = []
for obj in objects:
    vertices = [(v.x, v.y) for v in obj.bounding_poly.normalized_vertices]
    records.append({"Nome": obj.name, "Confidenza": obj.score, "Vertici": vertices})

df = pd.DataFrame(records)
```

**Limitazione**: la precisione cala se l'oggetto non è completamente visibile o non è orientato verso la fotocamera — ad esempio, uno specchio visto di profilo è più difficile da riconoscere di uno visto di fronte.

### 6.4 Rilevamento di contenuto esplicito (Safe Search)

`safe_search_detection` stima la probabilità che un'immagine contenga contenuto inadatto, in cinque categorie:

1. **`adult`**: contenuto sessuale esplicito o nudità
2. **`spoof`**: immagini manipolate o pensate per ingannare
3. **`medical`**: contenuto medico o anatomico (aiuta a distinguerlo da contenuto per adulti)
4. **`violence`**: atti violenti o dannosi
5. **`racy`**: contenuto suggestivo senza arrivare a esplicito

Ogni categoria restituisce un valore della stessa enumerazione di verosimiglianza che hai visto nel rilevamento di volti: `UNKNOWN`, `VERY_UNLIKELY`, `UNLIKELY`, `POSSIBLE`, `LIKELY` o `VERY_LIKELY`.

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("immagine.jpg", "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.safe_search_detection(image=image)
safe_search = response.safe_search_annotation

likelihood_names = (
    "UNKNOWN",
    "VERY_UNLIKELY",
    "UNLIKELY",
    "POSSIBLE",
    "LIKELY",
    "VERY_LIKELY",
)

print("Adult:", likelihood_names[safe_search.adult])
print("Spoof:", likelihood_names[safe_search.spoof])
print("Medical:", likelihood_names[safe_search.medical])
print("Violence:", likelihood_names[safe_search.violence])
print("Racy:", likelihood_names[safe_search.racy])
```

**Un esempio di regola di moderazione automatica**: contrassegna un'immagine come "richiede revisione" non appena una categoria raggiunge `LIKELY` o `VERY_LIKELY`.

**Tieni presente i suoi limiti**: Safe Search può produrre falsi positivi e falsi negativi. Combinalo sempre con altre tecniche di moderazione — filtri aggiuntivi, segnalazioni della comunità utente, revisione umana — invece di fidarti di esso come unica barriera.

---

## Lezione 7 — Dati visuali web

`web_detection` cerca la tua immagine nell'indice web e nel grafo della conoscenza di Google, e ti dice se esiste da qualche parte su internet, con quali pagine è correlata e quali entità rappresenta.

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("foto.jpg", "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.web_detection(image=image)
annotations = response.web_detection

print("Etichetta stimata migliore:", [label.label for label in annotations.best_guess_labels])

for entity in annotations.web_entities:
    print("Entità:", entity.description, "- score:", entity.score)

for match in annotations.full_matching_images:
    print("Corrispondenza esatta:", match.url)

for page in annotations.pages_with_matching_images:
    print("Pagina con immagine corrispondente:", page.url)

for similar in annotations.visually_similar_images:
    print("Immagine visivamente simile:", similar.url)
```

**I sei elementi che la risposta ti restituisce:**

1. **`web_entities`**: oggetti, monumenti, loghi o altri elementi visivi correlati, ognuno con `description` e `score`
2. **`full_matching_images`**: immagini identiche o quasi identiche trovate su internet
3. **Immagini parzialmente corrispondenti**: condividono elementi visivi, ma non sono corrispondenza esatta
4. **`pages_with_matching_images`**: pagine web che contengono immagini corrispondenti
5. **`visually_similar_images`**: immagini simili sebbene non corrispondano esattamente
6. **`best_guess_labels`**: l'etichetta che meglio descrive l'immagine, secondo Google

**Casi d'uso molto concreti:**

- **Rilevare foto rubate**: confronta la tua immagine con il web intero e scopri dove è stata riutilizzata senza il tuo permesso, anche se l'hanno leggermente modificata
- **Generare etichette automatiche**: per piattaforme di foto (ad esempio, un servizio tipo Flickr), evitando l'etichettatura manuale immagine per immagine

**Una sfumatura da tenere presente**: non tutti i link che l'API ti restituisce rimangono attivi — alcuni potrebbero dare errore 404 (pagina non trovata), quindi verifica sempre prima di considerare una corrispondenza come valida.

---

## Lezione 8 — Forme e oggetti con Pillow

La Vision API ti dà numeri: riquadri delimitatori e vertici. Per *vedere* quello che ha rilevato, hai bisogno di disegnare quei numeri sull'immagine — e qui entra **Pillow** (erede della storica libreria PIL, Python Imaging Library), la libreria standard di Python per aprire, disegnare su e salvare immagini.

**Il flusso è sempre lo stesso:**

1. La Vision API rileva oggetti e ti restituisce coordinate (`bounding_poly.vertices` o `normalized_vertices`)
2. Pillow apre l'immagine e crea un oggetto di disegno (`ImageDraw.Draw`) su di essa
3. Pillow traccia rettangoli, linee o poligoni usando quelle coordinate
4. Facoltativamente, Pillow aggiunge un'etichetta di testo accanto a ogni oggetto
5. Pillow salva o mostra l'immagine risultante

### Disegnare un riquadro delimitatore semplice

```python
from PIL import Image, ImageDraw

image = Image.open("immagine.jpg")
draw = ImageDraw.Draw(image)

object_bbox = (x1, y1, x2, y2)  # angolo superiore sinistro e inferiore destro
draw.rectangle(object_bbox, outline="red", width=2)

image.save("immagine_annotata.jpg")
```

### Disegnare un poligono a partire dai vertici che l'API restituisce

Quando lavori con `object_localization` o `landmark_detection`, la Vision API ti dà un elenco di vertici (non solo due angoli), quindi conviene una funzione che unisca ogni vertice con il successivo e chiuda il contorno:

```python
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

def draw_vertices(image_source, vertices, display_text="", color="green", width=4):
    pillow_img = Image.open(BytesIO(image_source)) if isinstance(image_source, bytes) else Image.open(image_source)
    draw = ImageDraw.Draw(pillow_img)

    # Unisce ogni vertice con il successivo
    for i in range(len(vertices) - 1):
        draw.line(
            [(vertices[i][0], vertices[i][1]), (vertices[i + 1][0], vertices[i + 1][1])],
            fill=color,
            width=width,
        )

    # Chiude il poligono: unisce l'ultimo vertice con il primo
    draw.line(
        [(vertices[-1][0], vertices[-1][1]), (vertices[0][0], vertices[0][1])],
        fill=color,
        width=width,
    )

    if display_text:
        font = ImageFont.load_default()
        draw.text((vertices[0][0] + 10, vertices[0][1]), display_text, font=font, fill="white")

    return pillow_img
```

> **Consiglio pratico**: `ImageDraw` include anche un metodo `draw.polygon(...)` che disegna un poligono chiuso direttamente a partire da un elenco di punti, senza che tu debba scrivere tu il ciclo di linee — provalo nel notebook e confronta entrambi i metodi.

### Combinare Pillow con il rilevamento di oggetti reale

```python
from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision

client = vision.ImageAnnotatorClient()
image_path = "salotto.jpg"

pil_image = Image.open(image_path)
width, height = pil_image.size

with open(image_path, "rb") as image_file:
    content = image_file.read()

response = client.object_localization(image=vision.Image(content=content))
draw = ImageDraw.Draw(pil_image)
font = ImageFont.load_default()

for obj in response.localized_object_annotations:
    vertices = [(v.x * width, v.y * height) for v in obj.bounding_poly.normalized_vertices]
    draw.polygon(vertices, outline="red", width=3)
    draw.text(vertices[0], f"{obj.name} ({obj.score:.2f})", font=font, fill="red")

pil_image.save("salotto_annotato.jpg")
```

Nota il dettaglio chiave: `normalized_vertices` viene in valori tra zero e uno, quindi moltiplicali per la larghezza (`width`) e l'altezza (`height`) reali dell'immagine prima di disegnare.

**Moduli di Pillow che userai nel notebook**: `Image` (aprire, creare, mostrare e salvare), `ImageDraw` (tracciare linee, rettangoli, poligoni e testo) e `ImageFont` (caricare caratteri tipografici per le etichette di testo).

---

## Allegato A — Prezzi e quote

- Ogni categoria di analisi (etichette, volti, testo, oggetti, ecc.) offre **mille unità gratuite al mese**, combinate tra loro
- Da lì in poi, Google addebita per ogni mille unità aggiuntive elaborate, con un prezzo che varia a seconda della categoria e della regione
- Per **Document Text Detection** su file PDF/TIFF, ogni pagina conta come un'unità; il prezzo orientativo citato nel materiale è di 0,60 $ per unità
- Il numero di documenti in cui dividi un lotto non cambia il numero totale di unità fatturate — un PDF di venti pagine costa lo stesso se lo elabori in una volta che se lo dividi in due di dieci
- Consulta sempre la [pagina ufficiale dei prezzi di Cloud Vision](https://cloud.google.com/vision/pricing) prima di pianificare un uso su larga scala, poiché questi prezzi cambiano nel tempo

## Allegato B — Buone pratiche ed errori comuni

- **Non caricare mai la tua chiave JSON in un repository pubblico** — trattala come una password
- **Avvolgi sempre le chiamate all'API in `try/except`**: sono frequenti gli errori di autenticazione, di rete, di quota esaurita o di formato di immagine non supportato
- **Ricorda i limiti di dimensione**: la maggior parte dei metodi accetta JPEG o PNG fino a 32 megabyte; `label_detection` limita la dimensione a quattro megapixel
- **Non fidarti di un singolo metodo per decisioni sensibili**: ad esempio, combina Safe Search con revisione umana prima di bloccare automaticamente i contenuti
- **Distingui `text_detection` da `document_text_detection`**: il primo è più generico; il secondo, pensato per documenti, ti dà confidenza per blocco, paragrafo, parola e simbolo
- **Ricorda di normalizzare le coordinate**: metodi come `object_localization` restituiscono vertici normalizzati (tra zero e uno) — moltiplicali per la larghezza e l'altezza reali prima di disegnare con Pillow

## Allegato C — Risorse per continuare a imparare

- Documentazione ufficiale: [cloud.google.com/vision/docs](https://cloud.google.com/vision/docs)
- Riferimento per i prezzi: [cloud.google.com/vision/pricing](https://cloud.google.com/vision/pricing)
- Google Cloud Learning Center: corsi e laboratori interattivi gratuiti
- Repository ufficiale di esempi di Google Cloud su GitHub (`googleapis/python-vision`)

---

**Adesso hai tutta la base teorica!** Apri il primo notebook dalla cartella `notebooks/` e inizia a praticare con le tue immagini.
