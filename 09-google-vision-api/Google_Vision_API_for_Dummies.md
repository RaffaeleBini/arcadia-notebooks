
# Google Vision API for Dummies

### Manual de estudio para el curso EITC/AI/GVAPI (European IT Certification – Artificial Intelligence – Google Vision API)

---

## Cómo usar este manual

Tienes en tus manos una guía de estudio pensada para acompañarte a lo largo de las ocho lecciones del curso de certificación EITC/AI/GVAPI sobre la Google Vision API (interfaz de programación de aplicaciones de visión de Google). Cada lección resume los conceptos clave, reconstruye el código de ejemplo (a veces fragmentado en el material original) en fragmentos ejecutables y cierra con las ideas que más se repiten en las preguntas de repaso oficiales.

Este manual es la parte teórica. Para practicar, abre los ocho notebooks Jupyter de la carpeta `notebooks/` — están numerados en el mismo orden que las lecciones y contienen celdas de código con ejercicios guiados y sus soluciones desplegables.

**¡Empecemos!**

---

## Lista de acrónimos

| Acrónimo | Significado |
|---|---|
| API | Interfaz de Programación de Aplicaciones (Application Programming Interface) |
| CNN | Red Neuronal Convolucional (Convolutional Neural Network) |
| GCP | Google Cloud Platform |
| GCS | Almacenamiento en la Nube de Google (Google Cloud Storage) |
| IA | Inteligencia Artificial |
| IAM | Gestión de Identidades y Accesos (Identity and Access Management) |
| JSON | Notación de Objetos de JavaScript (JavaScript Object Notation) |
| NLP | Procesamiento del Lenguaje Natural (Natural Language Processing) |
| OCR | Reconocimiento Óptico de Caracteres (Optical Character Recognition) |
| PIL | Python Imaging Library, nombre histórico de la librería que hoy se distribuye como Pillow |
| REST | Transferencia de Estado Representacional (Representational State Transfer) |
| RGB | Modelo de color Rojo, Verde y Azul (Red, Green, Blue) |
| SDK | Kit de Desarrollo de Software (Software Development Kit) |
| SSD | Detector Multibox de Disparo Único (Single Shot Multibox Detector) |
| TIFF | Formato de Archivo de Imagen Etiquetado (Tagged Image File Format) |
| URI | Identificador Uniforme de Recursos (Uniform Resource Identifier) |
| URL | Localizador Uniforme de Recursos (Uniform Resource Locator) |

---

## Índice

1. [Lección 1 — Introducción](#lección-1--introducción)
2. [Lección 2 — Primeros pasos: configuración y setup](#lección-2--primeros-pasos-configuración-y-setup)
3. [Lección 3 — Comprensión del texto en datos visuales](#lección-3--comprensión-del-texto-en-datos-visuales)
4. [Lección 4 — Comprensión de imágenes](#lección-4--comprensión-de-imágenes)
5. [Lección 5 — Etiquetado de imágenes](#lección-5--etiquetado-de-imágenes)
6. [Lección 6 — Comprensión avanzada de imágenes](#lección-6--comprensión-avanzada-de-imágenes)
7. [Lección 7 — Datos visuales web](#lección-7--datos-visuales-web)
8. [Lección 8 — Formas y objetos con Pillow](#lección-8--formas-y-objetos-con-pillow)
9. [Anexo A — Precios y cuotas](#anexo-a--precios-y-cuotas)
10. [Anexo B — Buenas prácticas y errores habituales](#anexo-b--buenas-prácticas-y-errores-habituales)
11. [Anexo C — Recursos para seguir aprendiendo](#anexo-c--recursos-para-seguir-aprendiendo)

---

## Lección 1 — Introducción

### 1.1 Introducción a la Google Cloud Vision API

La inteligencia artificial (IA) permite que las máquinas ejecuten tareas que antes exigían inteligencia humana. La visión por computador es una de sus aplicaciones más visibles: la capacidad de interpretar datos visuales. Google pone esa capacidad a tu alcance a través de la **Google Cloud Vision API**, un servicio en la nube que analiza imágenes con modelos de aprendizaje automático ya entrenados y te devuelve información estructurada en formato JSON.

No necesitas entrenar ningún modelo propio: envías una imagen, te autenticas con tus credenciales y recibes la respuesta. La API admite varios lenguajes de programación; en este curso trabajarás con Python.

**Qué puede detectar la API:**

- **Etiquetas (labels)**: objetos, escenas y conceptos presentes en la imagen — por ejemplo, "perro", "árbol" o "coche"
- **Rostros (faces)**: posición de ojos, nariz y boca, además de emociones como alegría, ira, sorpresa o tristeza, y un rango de edad estimado
- **Texto (OCR)**: extrae texto impreso o manuscrito y lo convierte en información legible por máquina
- **Objetos (object localization)**: ubica objetos concretos dentro de la imagen, útil por ejemplo para identificar productos en una estantería
- **Contenido explícito, monumentos y logotipos**: funciones avanzadas que verás en la lección 6

Un ejemplo que ilustra bien el potencial de la API: imagina un robot Raspberry Pi con cámara. El robot no ejecuta el modelo de IA en su propio hardware — envía la imagen capturada a la nube, la Vision API la analiza y le devuelve el resultado en tiempo real. Con detección de rostros y emociones, el robot puede acercarse a una cara sonriente o alejarse ante una expresión de ira o sorpresa.

Este proceso, llamado **detección de entidades**, compara las características extraídas de la imagen contra una base de datos de entidades conocidas y asigna a cada resultado una **puntuación de confianza**, que te permite fijar un umbral y descartar resultados poco fiables.

> **Aprende más**: consulta la documentación oficial en cloud.google.com/vision, los cursos del Google Cloud Learning Center y los webinars que organiza Google sobre IA.

### 1.2 Vision API frente a AutoML Vision API

Dentro de "Vision AI", Google ofrece dos servicios distintos y conviene que sepas diferenciarlos antes de escribir una sola línea de código:

| | **Vision API** | **AutoML Vision API** |
|---|---|---|
| Modelo | Preentrenado por Google | Entrenas tu propio modelo con tus imágenes |
| Interfaz | Código (API REST o cliente Python) | Interfaz gráfica sencilla para subir y etiquetar imágenes |
| Cuándo elegirla | Te basta con el reconocimiento genérico ya entrenado por Google | Necesitas reconocer categorías muy específicas de tu negocio (por ejemplo, distinguir modelos concretos de una maquinaria) |
| Coste | Pago por uso, sin coste de entrenamiento | Coste adicional por el entrenamiento del modelo personalizado |

Un ejemplo que aparece en el material original: si subes una foto de un loro a la Vision API, el servicio no solo reconoce "loro" — puede identificar la especie exacta (loro gris africano) gracias a los datos ya recopilados por Google.

**Funcionalidades adicionales de la Vision API que verás en detalle más adelante:**

- **Propiedades de la imagen**: colores dominantes, histograma de color y su porcentaje de presencia (lección 4)
- **Safe Search**: probabilidad de que una imagen contenga contenido para adultos, violento o inapropiado (lección 6)
- **Detección web**: si la imagen aparece en algún lugar de internet, con qué páginas y entidades se relaciona (lección 7)
- **Búsqueda de productos**: lista de artículos visualmente similares a la imagen proporcionada

**Precios (orientativos, sujetos a cambios en la documentación oficial de Google):** dispones de mil llamadas gratuitas al mes combinando todas las categorías; a partir de ahí, Google cobra por cada mil unidades procesadas, según la categoría.

**Casos de uso reales citados en el material**: Box.com y The New York Times usan Vision AI en producción, y la propia app de fotos de Google la utiliza para organizar automáticamente las fotografías por categorías.

Para instalar la librería cliente de Python necesitarás el paquete `google-cloud-vision`, que instalarás con pip (el instalador de paquetes de Python) — verás el proceso completo, paso a paso, en la lección 2.

---

## Lección 2 — Primeros pasos: configuración y setup

Antes de escribir tu primera llamada a la API necesitas cuatro cosas: un proyecto en Google Cloud Platform (GCP), la Vision API habilitada en ese proyecto, unas credenciales válidas y la librería cliente instalada en tu entorno de Python. Sigue estos pasos en orden.

### Paso 1 — Crea un proyecto en Google Cloud Platform

1. Entra en [Google Cloud Console](https://console.cloud.google.com/) con tu cuenta de Google
2. Crea un proyecto nuevo (o selecciona uno existente) y anota su **ID de proyecto**
3. Confirma que el proyecto tiene la facturación (billing) configurada — el nivel gratuito es suficiente para practicar con este curso

### Paso 2 — Habilita la Vision API

1. En el menú lateral, ve a **APIs & Services > Library**
2. Busca "Vision API" y ábrela
3. Pulsa **Enable** para activarla en tu proyecto

### Paso 3 — Crea una cuenta de servicio y descarga tu clave JSON

Una **cuenta de servicio** (service account) es una identidad no humana que tu aplicación usa para autenticarse contra la API, en lugar de que lo hagas tú con tu usuario personal.

1. Ve a **APIs & Services > Credentials**
2. Pulsa **Create Credentials > Service account**
3. Dale un nombre reconocible y asígnale el rol **Editor** (o uno más restrictivo, como "Cloud Vision API User", si prefieres aplicar el principio de mínimo privilegio)
4. Dentro de la cuenta de servicio recién creada, ve a la pestaña **Keys > Add Key > Create new key**, elige el formato **JSON** y descarga el archivo
5. Guarda ese archivo `.json` en un lugar seguro de tu equipo — es una credencial sensible: no la subas nunca a un repositorio público

> **¡Guárdalo bien!** Ese archivo JSON da acceso a tu proyecto de GCP. Trátalo con el mismo cuidado que una contraseña.

### Paso 4 — Crea un entorno virtual

Un **entorno virtual** es un directorio aislado que encapsula las dependencias de tu proyecto, evitando conflictos de versiones con otros proyectos Python que tengas en tu equipo. Aporta cuatro ventajas: aislamiento de dependencias, reproducibilidad entre máquinas, un espacio seguro para experimentar sin afectar al resto del sistema, y facilidad para compartir el proyecto con tu equipo.

```bash
python -m venv venv_gvapi

# En Windows
venv_gvapi\Scripts\activate

# En macOS / Linux
source venv_gvapi/bin/activate
```

### Paso 5 — Instala la librería cliente

Con el entorno virtual activado:

```bash
pip --version
pip install google-cloud-vision
pip list
```

`pip --version` confirma que pip funciona correctamente; `pip install google-cloud-vision` descarga la librería cliente oficial; `pip list` te deja verificar que la instalación se completó.

### Paso 6 — Configura la variable de entorno de credenciales

La librería cliente busca automáticamente tus credenciales en la variable de entorno **`GOOGLE_APPLICATION_CREDENTIALS`**. Puedes definirla desde la terminal o directamente en tu script de Python:

```python
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ruta/a/tu/clave.json"
```

### Paso 7 — Crea tu primer cliente y compruébalo

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

# Lista los métodos disponibles del cliente para explorar la API
print(dir(client))
```

Si esta celda se ejecuta sin errores, ¡tu entorno está listo! Ya puedes llamar a métodos como `client.label_detection()`, `client.text_detection()`, `client.face_detection()` o `client.safe_search_detection()`, que iremos viendo lección a lección.

### Alternativa rápida: clave de API y gcloud CLI

El material original también describe un camino más directo, pensado para pruebas puntuales desde la línea de comandos, en lugar de una aplicación Python:

1. Habilita la Vision API en la biblioteca de API de la consola de GCP
2. Ve a **Credentials > Create Credentials > API key**
3. Instala el [Google Cloud SDK](https://cloud.google.com/sdk) en tu equipo
4. Autentica el SDK: `gcloud auth login`
5. Prueba la API directamente: `gcloud ml vision detect-labels ruta/a/imagen.jpg`

Usa esta vía solo para pruebas rápidas de línea de comandos; para el desarrollo de aplicaciones con Python, la cuenta de servicio del flujo anterior es el camino recomendado.

---

## Lección 3 — Comprensión del texto en datos visuales

### 3.1 Detectar y extraer texto de una imagen

El método `text_detection` localiza regiones con texto dentro de una imagen y reconoce sus caracteres, ya sean impresos o manuscritos y en distintos idiomas. Junto al texto, la API te devuelve el idioma detectado (`locale`), una puntuación de confianza y las coordenadas del cuadro delimitador (`bounding_poly`) de cada fragmento.

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("image.jpg", "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.text_detection(image=image)
texts = response.text_annotations

# El primer elemento contiene todo el texto detectado;
# los siguientes son cada palabra/fragmento por separado
for text in texts:
    print(text.description)
```

Encapsulado en una función reutilizable:

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

Si la imagen está alojada públicamente, en lugar de leer bytes de un archivo local, asigna la URL directamente a `image.source.image_uri`:

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

**Limpiar el texto extraído con pandas**: una vez que tienes el texto en un DataFrame, puedes normalizarlo con expresiones regulares y con los métodos de cadena de pandas:

```python
import pandas as pd
import re

df["text"] = df["text"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x))
df["text"] = df["text"].str.lower()
df["words"] = df["text"].apply(lambda x: x.split(" "))
```

**Aplicaciones típicas**: digitalización de documentos, traducción de carteles o menús, moderación de contenido, extracción de precios o nombres de producto en anuncios, y accesibilidad (convertir texto de una imagen en audio).

### 3.2 Detectar y extraer texto manuscrito

El texto manuscrito es más difícil de reconocer que el impreso porque la escritura varía mucho de una persona a otra. La Vision API sigue cuatro fases: preprocesamiento de la imagen (redimensionado, reducción de ruido, binarización), detección de las regiones con texto, segmentación en unidades más pequeñas y, por último, reconocimiento mediante OCR.

Para documentos (currículums, notas manuscritas, formularios), usa **`document_text_detection`** en lugar de `text_detection`: el resultado se organiza jerárquicamente y te da un nivel de confianza en cada nivel.

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
            print("Confianza del bloque:", block.confidence)
            for paragraph in block.paragraphs:
                print("Confianza del párrafo:", paragraph.confidence)
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    print(word_text)

    return document.text
```

La jerarquía de la respuesta es siempre la misma: **páginas (`pages`) → bloques (`blocks`) → párrafos (`paragraphs`) → palabras (`words`) → símbolos (`symbols`)**, y cada nivel trae su propio `confidence` (un valor decimal entre cero y uno, donde uno es la certeza máxima).

**¿Para qué sirve la confianza?** Para decidir si conviene una revisión manual, para fijar un umbral mínimo en un proceso automatizado y para priorizar unos resultados sobre otros.

**Limitaciones a tener en cuenta:**

- La calidad de la imagen de entrada (borrosidad, mala iluminación) reduce la precisión
- Los layouts complejos (varias columnas, tablas, texto manuscrito mezclado con impreso) pueden fragmentar u omitir texto
- La caligrafía prolija se reconoce mejor que la cursiva elaborada
- Los idiomas y escrituras poco comunes tienen menor precisión que el inglés

### 3.3 Detectar y extraer texto de archivos (PDF/TIFF)

Para procesar archivos completos —no solo imágenes sueltas— en formato PDF o TIFF, combina la Vision API con **Google Cloud Storage (GCS)**, el servicio de almacenamiento en la nube de Google. El archivo debe residir en un **bucket** de GCS antes de que la API pueda procesarlo.

**Límite y precio**: cada archivo admite un máximo de dos mil páginas; el precio se calcula por unidad, donde una página cuenta como una unidad, con mil unidades gratuitas al mes y un precio orientativo de 0,60 $ por unidad para "Document Text Detection" (sujeto a la tabla de precios oficial vigente).

```bash
pip install --upgrade google-cloud-storage
```

**Flujo asíncrono recomendado** (procesamiento por lotes vía GCS):

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

gcs_source_uri = "gs://tu-bucket/entrada.pdf"
gcs_destination_uri = "gs://tu-bucket/salida/"

feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

gcs_source = vision.GcsSource(uri=gcs_source_uri)
input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")

gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=2)

async_request = vision.AsyncAnnotateFileRequest(
    features=[feature], input_config=input_config, output_config=output_config
)

operation = client.async_batch_annotate_files(requests=[async_request])
print("Procesando...")
operation.result(timeout=180)
```

**Leer el resultado desde Google Cloud Storage:**

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

**Alternativa sin GCS**: si prefieres no usar un bucket, convierte el PDF a imágenes localmente con `pdf2image` y aplica `text_detection` página a página:

```python
from pdf2image import convert_from_bytes

with open("archivo.pdf", "rb") as f:
    content = f.read()

images = convert_from_bytes(content)

for image in images:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    response = client.text_detection(image=vision.Image(content=image_bytes.getvalue()))
    for text in response.text_annotations:
        print(text.description)
```

Esta alternativa es más sencilla de montar para un cuaderno de práctica, aunque pierde las ventajas del procesamiento asíncrono por lotes y del descuento por volumen de GCS.

**Aplicaciones**: búsqueda de cláusulas en contratos legales, análisis de sentimiento y clasificación de texto (NLP), minería de datos en historiales médicos, y generación automática de resúmenes.

---

## Lección 4 — Comprensión de imágenes

### 4.1 Detección de sugerencias de recorte (crop hints)

El método `crop_hints` sugiere la región más "saliente" (la que concentra el contenido visual más relevante) de una imagen, como semilla para recortarla según una relación de aspecto que tú indiques. Es útil para adaptar imágenes grandes a distintos formatos (miniaturas, banners, vistas móviles) sin perder el punto de interés principal.

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
        print("Cuadro delimitador:", hint.bounding_poly)
        print("Confianza:", hint.confidence)
        print("Fracción de importancia:", hint.importance_fraction)

    return response.crop_hints_annotation.crop_hints

detect_crop_hints("imagen.jpg", aspect_ratios=[16 / 9, 4 / 3, 1.0])
```

**Parámetros que puedes ajustar:**

- **`image`**: JPEG o PNG, hasta 32 megabytes
- **`max_results`**: número máximo de sugerencias a devolver, de uno a diez
- **`aspect_ratios`**: lista de relaciones ancho/alto deseadas — por ejemplo, `[1.0, 1.333]` para 1:1 y 4:3
- **`crop_hints_params.confidence_threshold`**: descarta sugerencias por debajo de un umbral de confianza

Cada resultado incluye un `bounding_poly` con los vértices del recorte sugerido, una `confidence` (nivel de confianza) y una `importance_fraction` (qué proporción de la sugerencia cubre contenido relevante).

### 4.2 Detección de rostros

`face_detection` localiza uno o varios rostros en una imagen y, por cada uno, te da su cuadro delimitador, sus puntos de referencia faciales (`landmarks`: ojos, nariz, boca, orejas) y la probabilidad de cuatro emociones — alegría, tristeza, ira y sorpresa.

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("grupo.jpg", "rb") as image_file:
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
    print("Alegría:", likelihood_names[face.joy_likelihood])
    print("Ira:", likelihood_names[face.anger_likelihood])
    print("Sorpresa:", likelihood_names[face.surprise_likelihood])
    print("Tristeza:", likelihood_names[face.sorrow_likelihood])
    print("Confianza de detección:", face.detection_confidence)
```

Cada uno de estos atributos usa la misma **enumeración de verosimilitud (`Likelihood`)**, con seis niveles: `UNKNOWN`, `VERY_UNLIKELY`, `UNLIKELY`, `POSSIBLE`, `LIKELY` y `VERY_LIKELY`. La verás repetida en Safe Search (lección 6), así que merece la pena que te familiarices con ella ahora.

**Privacidad**: la API está diseñada para no almacenar ni retener las imágenes que le envías, y ofrece opciones para ajustar el nivel de sensibilidad de la detección.

**Una limitación importante**: la precisión depende de que el rostro esté completamente visible. En un ejemplo del material original, una foto de seis personas solo detectó cinco rostros — el sexto quedó parcialmente oculto y la API no lo reconoció. Ten esto en cuenta si tu aplicación depende de un conteo exacto de personas.

### 4.3 Detección de propiedades de la imagen

`image_properties` extrae atributos generales de una imagen: sus colores dominantes (con valores RGB y una puntuación de prevalencia), y —de forma más general— pistas sobre su orientación y su tipo (fotografía, clipart o ilustración).

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

Cada color viene con tres valores: `color` (RGB), `score` (confianza/prevalencia) y `pixel_fraction` (qué proporción de píxeles de la imagen corresponde a ese color).

**Aplicaciones**: búsqueda de imágenes por color, categorización (foto de producto frente a ilustración), diseño gráfico, análisis de tendencias de color en moda o interiorismo, y corrección automática de orientación.

**Ten en cuenta**: los colores dominantes reflejan la prevalencia global de colores en la imagen, no necesariamente el elemento visualmente más importante de la escena.

---

## Lección 5 — Etiquetado de imágenes

`label_detection` identifica objetos, escenas y hasta conceptos abstractos (por ejemplo, "felicidad") presentes en una imagen, y devuelve una lista de etiquetas con su puntuación de confianza.

```python
import os
from google.cloud import vision

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/ruta/a/tu/clave.json"

def extract_labels(image_path):
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    return [(label.description, label.score, label.topicality) for label in labels]

print(extract_labels("gato.jpg"))
# [('cat', 0.98, 0.98), ('whiskers', 0.91, 0.91), ('small to medium-sized cats', 0.87, 0.87), ...]
```

**Campos de la respuesta:**

- `description`: texto de la etiqueta (por ejemplo, "cat")
- `score`: puntuación de confianza
- `topicality`: importancia de la etiqueta dentro de la imagen

> **Detalle a tener en cuenta**: en la versión de la API descrita en el material del curso, `score` y `topicality` devuelven el mismo valor — un comportamiento documentado que Google puede corregir en futuras versiones. No dejes de comprobarlo tú mismo con tus propias imágenes en el notebook de práctica.

**Requisitos y límites**: formatos JPEG, PNG o GIF, con un tamaño máximo de cuatro megapíxeles.

**Errores habituales a los que prestar atención**: fallos de autenticación, problemas de red, cuota de API agotada, imagen corrupta o en un formato no soportado, y parámetros incorrectos en la llamada. Envuelve siempre tus llamadas en un bloque `try/except`.

**Casos de uso**: etiquetado automático de productos en comercio electrónico, moderación de contenido y organización de grandes colecciones de imágenes (gestión de activos digitales).

---

## Lección 6 — Comprensión avanzada de imágenes

### 6.1 Detección de monumentos (landmarks)

`landmark_detection` reconoce estructuras naturales o construidas por el ser humano con relevancia cultural, histórica o geográfica — la Torre Eiffel, la Estatua de la Libertad, el Taj Mahal. Compara las características visuales extraídas de tu imagen contra una base de datos mundial de monumentos conocidos.

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
        print("Ubicación:", annotation.locations)
        print("Confianza:", annotation.score)

    return response.landmark_annotations
```

La respuesta incluye `description` (nombre del monumento), `score` (confianza) y `bounding_poly` (posición del monumento dentro de la imagen). Funciona incluso con el monumento parcialmente oculto, en ángulos inusuales o con poca luz.

### 6.2 Detección de logotipos

`logo_detection` identifica marcas y organizaciones dentro de una imagen, incluso con el logotipo parcialmente oculto o distorsionado. Combina localización de objetos, clasificación de imágenes y aprendizaje profundo, y puede complementarse con reconocimiento de texto (OCR) para leer el texto que forma parte del propio logo.

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("tienda.jpg", "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.logo_detection(image=image)
logos = response.logo_annotations

for logo in logos:
    print("Logo:", logo.description)
    print("Confianza:", logo.score)
    print("Cuadro delimitador:", logo.bounding_poly)
```

En una prueba descrita en el material original, la API reconoció sin problema los logotipos de Apple, Nintendo, Best Buy y Nike, pero no logró identificar los de IBM, Circuit City y Playboy — la precisión global rondó el 90%. Los logotipos con diseños muy simples (como el de GAP) o muy intrincados (como los aros entrelazados de Audi) son los que más dificultades presentan.

### 6.3 Detección de objetos

`object_localization` combina dos tareas: **clasificación** (qué es cada objeto) y **localización** (dónde está, con su cuadro delimitador). Internamente usa un modelo **SSD (detector multibox de disparo único)**, una arquitectura de red neuronal convolucional (CNN) diseñada para ser precisa y eficiente, entrenada con millones de imágenes anotadas.

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("salon.jpg", "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.object_localization(image=image)
objects = response.localized_object_annotations

for obj in objects:
    print(f"{obj.name} (confianza: {obj.score:.2f})")
    vertices = [(v.x, v.y) for v in obj.bounding_poly.normalized_vertices]
    print("Vértices normalizados:", vertices)
```

Fíjate en que `bounding_poly.normalized_vertices` devuelve coordenadas **normalizadas** entre cero y uno; para dibujarlas sobre la imagen tendrás que multiplicarlas por el ancho y el alto reales (lo verás en la lección 8, con Pillow).

**Organizar los resultados con pandas:**

```python
import pandas as pd

registros = []
for obj in objects:
    vertices = [(v.x, v.y) for v in obj.bounding_poly.normalized_vertices]
    registros.append({"Nombre": obj.name, "Confianza": obj.score, "Vértices": vertices})

df = pd.DataFrame(registros)
```

**Limitación**: la precisión baja si el objeto no está completamente visible o no está orientado hacia la cámara — por ejemplo, un espejo visto de perfil es más difícil de reconocer que uno visto de frente.

### 6.4 Detección de contenido explícito (Safe Search)

`safe_search_detection` estima la probabilidad de que una imagen contenga contenido inapropiado, en cinco categorías:

1. **`adult`**: contenido sexual explícito o desnudez
2. **`spoof`**: imágenes manipuladas o pensadas para engañar
3. **`medical`**: contenido médico o anatómico (ayuda a distinguirlo del contenido para adultos)
4. **`violence`**: actos violentos o dañinos
5. **`racy`**: contenido sugerente sin llegar a explícito

Cada categoría devuelve un valor de la misma enumeración de verosimilitud que viste en la detección de rostros: `UNKNOWN`, `VERY_UNLIKELY`, `UNLIKELY`, `POSSIBLE`, `LIKELY` o `VERY_LIKELY`.

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("imagen.jpg", "rb") as image_file:
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

**Un ejemplo de regla de moderación automática**: marca una imagen como "requiere revisión" en cuanto cualquier categoría alcance `LIKELY` o `VERY_LIKELY`.

**Ten presente sus límites**: Safe Search puede producir falsos positivos y falsos negativos. Combínalo siempre con otras técnicas de moderación —filtrado adicional, reportes de la comunidad usuaria, revisión humana— en lugar de confiar en él como única barrera.

---

## Lección 7 — Datos visuales web

`web_detection` busca tu imagen en el índice web y el grafo de conocimiento de Google, y te dice si existe en algún lugar de internet, con qué páginas se relaciona y qué entidades representa.

```python
from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open("foto.jpg", "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.web_detection(image=image)
annotations = response.web_detection

print("Mejor etiqueta estimada:", [label.label for label in annotations.best_guess_labels])

for entity in annotations.web_entities:
    print("Entidad:", entity.description, "- score:", entity.score)

for match in annotations.full_matching_images:
    print("Coincidencia exacta:", match.url)

for page in annotations.pages_with_matching_images:
    print("Página con imagen coincidente:", page.url)

for similar in annotations.visually_similar_images:
    print("Imagen visualmente similar:", similar.url)
```

**Los seis elementos que te devuelve la respuesta:**

1. **`web_entities`**: objetos, monumentos, logotipos u otros elementos visuales relacionados, cada uno con `description` y `score`
2. **`full_matching_images`**: imágenes idénticas o casi idénticas encontradas en internet
3. **Imágenes parcialmente coincidentes**: comparten elementos visuales, pero no son coincidencia exacta
4. **`pages_with_matching_images`**: páginas web que contienen imágenes coincidentes
5. **`visually_similar_images`**: imágenes similares aunque no coincidan exactamente
6. **`best_guess_labels`**: la etiqueta que mejor describe la imagen, según Google

**Casos de uso muy concretos:**

- **Detectar fotos robadas**: compara tu imagen con la web entera y descubre dónde se ha reutilizado sin tu permiso, incluso si la editaron ligeramente
- **Generar etiquetas automáticas**: para plataformas de fotos (por ejemplo, un servicio tipo Flickr), evitando el etiquetado manual imagen por imagen

**Un matiz a tener en cuenta**: no todos los enlaces que te devuelve la API siguen activos — algunos pueden dar error 404 (página no encontrada), así que verifica siempre antes de dar por buena una coincidencia.

---

## Lección 8 — Formas y objetos con Pillow

La Vision API te da números: cuadros delimitadores y vértices. Para *ver* lo que detectó, necesitas dibujar esos números sobre la imagen — y ahí entra **Pillow** (heredera de la histórica librería PIL, Python Imaging Library), la librería estándar de Python para abrir, dibujar sobre y guardar imágenes.

**El flujo siempre es el mismo:**

1. La Vision API detecta objetos y te devuelve coordenadas (`bounding_poly.vertices` o `normalized_vertices`)
2. Pillow abre la imagen y crea un objeto de dibujo (`ImageDraw.Draw`) sobre ella
3. Pillow traza rectángulos, líneas o polígonos usando esas coordenadas
4. Opcionalmente, Pillow añade una etiqueta de texto junto a cada objeto
5. Pillow guarda o muestra la imagen resultante

### Dibujar un cuadro delimitador simple

```python
from PIL import Image, ImageDraw

image = Image.open("imagen.jpg")
draw = ImageDraw.Draw(image)

object_bbox = (x1, y1, x2, y2)  # esquina superior izquierda e inferior derecha
draw.rectangle(object_bbox, outline="red", width=2)

image.save("imagen_anotada.jpg")
```

### Dibujar un polígono a partir de los vértices que devuelve la API

Cuando trabajas con `object_localization` o `landmark_detection`, la Vision API te da una lista de vértices (no solo dos esquinas), así que conviene una función que una cada vértice con el siguiente y cierre el contorno:

```python
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

def draw_vertices(image_source, vertices, display_text="", color="green", width=4):
    pillow_img = Image.open(BytesIO(image_source)) if isinstance(image_source, bytes) else Image.open(image_source)
    draw = ImageDraw.Draw(pillow_img)

    # Une cada vértice con el siguiente
    for i in range(len(vertices) - 1):
        draw.line(
            [(vertices[i][0], vertices[i][1]), (vertices[i + 1][0], vertices[i + 1][1])],
            fill=color,
            width=width,
        )

    # Cierra el polígono: une el último vértice con el primero
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

> **Consejo práctico**: `ImageDraw` también incluye un método `draw.polygon(...)` que dibuja un polígono cerrado directamente a partir de una lista de puntos, sin que tengas que escribir tú el bucle de líneas — pruébalo en el notebook y compara ambos enfoques.

### Combinar Pillow con la detección de objetos real

```python
from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision

client = vision.ImageAnnotatorClient()
image_path = "salon.jpg"

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

pil_image.save("salon_anotado.jpg")
```

Observa el detalle clave: `normalized_vertices` viene en valores entre cero y uno, así que multiplícalos por el ancho (`width`) y el alto (`height`) reales de la imagen antes de dibujar.

**Módulos de Pillow que usarás en el notebook**: `Image` (abrir, crear, mostrar y guardar), `ImageDraw` (trazar líneas, rectángulos, polígonos y texto) e `ImageFont` (cargar tipografías para las etiquetas de texto).

---

## Anexo A — Precios y cuotas

- Cada categoría de análisis (etiquetas, rostros, texto, objetos, etc.) ofrece **mil unidades gratuitas al mes**, combinadas entre sí
- A partir de ahí, Google cobra por cada mil unidades adicionales, con un precio que varía según la categoría y la región
- Para **Document Text Detection** sobre archivos PDF/TIFF, cada página cuenta como una unidad; el precio orientativo citado en el material es de 0,60 $ por unidad
- El número de documentos en que divides un lote no cambia el número total de unidades facturadas — un PDF de veinte páginas cuesta lo mismo si lo procesas de una vez que si lo divides en dos de diez
- Consulta siempre la [página oficial de precios de Cloud Vision](https://cloud.google.com/vision/pricing) antes de planificar un uso a gran escala, ya que estas cifras cambian con el tiempo

## Anexo B — Buenas prácticas y errores habituales

- **Nunca subas tu clave JSON a un repositorio público** — trátala como una contraseña
- **Envuelve siempre las llamadas a la API en `try/except`**: son frecuentes los errores de autenticación, de red, de cuota agotada o de formato de imagen no soportado
- **Recuerda los límites de tamaño**: la mayoría de los métodos aceptan JPEG o PNG hasta 32 megabytes; `label_detection` limita el tamaño a cuatro megapíxeles
- **No confíes en un único método para decisiones sensibles**: por ejemplo, combina Safe Search con revisión humana antes de bloquear contenido automáticamente
- **Distingue `text_detection` de `document_text_detection`**: el primero es más genérico; el segundo, pensado para documentos, te da confianza por bloque, párrafo, palabra y símbolo
- **Recuerda normalizar coordenadas**: los métodos como `object_localization` devuelven vértices normalizados (entre cero y uno) — multiplícalos por el ancho y el alto reales antes de dibujar con Pillow

## Anexo C — Recursos para seguir aprendiendo

- Documentación oficial: [cloud.google.com/vision/docs](https://cloud.google.com/vision/docs)
- Referencia de precios: [cloud.google.com/vision/pricing](https://cloud.google.com/vision/pricing)
- Google Cloud Learning Center: cursos y laboratorios interactivos gratuitos
- Repositorio de ejemplos oficial de Google Cloud en GitHub (`googleapis/python-vision`)

---

**¡Ya tienes toda la base teórica!** Abre ahora el primer notebook de la carpeta `notebooks/` y empieza a practicar con tus propias imágenes.
