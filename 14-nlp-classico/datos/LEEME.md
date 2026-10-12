# Datos del curso / Dati del corso

## Español

Corpus del curso NLP Clásico. Todos los textos sintéticos están **escritos originalmente para el curso** (sin marcas reales ni datos personales) y no se han copiado de ninguna otra fuente. Los notebooks buscan la carpeta `datos/` junto al curso; en Google Colab hay que subirla (panel de archivos) o dejarla en `/content/datos/`.

| Archivo | Qué es | Origen |
|---|---|---|
| `resenas_es.csv`, `resenas_it.csv` | 300 reseñas por idioma, 100 por clase (`positivo`, `negativo`, `neutro`), de dominios variados. Columnas: `id`, `texto` (texto **bruto** con ruido: mayúsculas, acentos perdidos, alargamientos, erratas, abreviaturas de chat, emojis, URLs), `texto_limpio` (versión limpia de referencia), `etiqueta`, `dominio`, `dificil` (1 = negación, ironía, juicio mixto, etc.; ≈ 15 %). | Sintético, original |
| `tickets_es.csv`, `tickets_it.csv` | 400 tickets de soporte por idioma, 100 por categoría (`facturacion`, `envio`, `cuenta`, `devoluciones`; las mismas etiquetas en ambos idiomas). Columnas: `id`, `texto` (bruto, ruido más ligero), `texto_limpio`, `categoria`. Cerca del 10-15 % de los tickets contiene palabras típicas de otra categoría. | Sintético, original |
| `gutenberg/quijote_cap1.txt` | Capítulo I de *Don Quijote de la Mancha* (Miguel de Cervantes), ≈ 1 900 palabras, en español del siglo XVII | Project Gutenberg, eBook #2000, dominio público |
| `gutenberg/promessi_sposi_cap1.txt` | Primeras ≈ 1 900 palabras del capítulo I de *I promessi sposi* (Alessandro Manzoni), en italiano del siglo XIX; conserva las marcas `_cursiva_` del original electrónico | Project Gutenberg, eBook #45334, dominio público |

Notas de calidad de los corpus sintéticos:
- Los textos brutos de las reseñas y de los tickets tienen ruido en ≈ 45 % y ≈ 28 % de los casos respectivamente; `texto_limpio` permite medir la calidad de una limpieza.
- Con TF-IDF (unigramas y bigramas) y regresión logística, la exactitud en validación cruzada de 5 particiones sobre el texto bruto es ≈ 0,78 en las reseñas y ≈ 0,88-0,89 en los tickets (valores orientativos; cambian con el preprocesamiento).
- Algunos tickets (≈ 6 %) terminan con una etiqueta de asunto ("Asunto: cobro") y varios textos incluyen veredictos breves al final; no influyen en los resultados, pero conviene tenerlo en cuenta al leer los coeficientes de un modelo.
- Los tickets en español y en italiano son adaptaciones cercanas entre sí; las reseñas son textos independientes.
- Son corpus pequeños y con un estilo controlado: los resultados no se generalizan a textos reales de cualquier origen.

Los textos de Project Gutenberg se han tomado del texto electrónico, sin el encabezado ni el pie del proyecto y con los saltos de línea reunidos en párrafos. Los textos originales son de dominio público (autores fallecidos hace más de cien años).

## Italiano

Corpus del corso NLP Classico. Tutti i testi sintetici sono **scritti originalmente per il corso** (senza marchi reali né dati personali) e non sono copiati da altre fonti. I notebook cercano la cartella `datos/` accanto al corso; in Google Colab va caricata (pannello dei file) o lasciata in `/content/datos/`.

| File | Cos'è | Origine |
|---|---|---|
| `resenas_es.csv`, `resenas_it.csv` | 300 recensioni per lingua, 100 per classe (`positivo`, `negativo`, `neutro`), di domini vari. Colonne: `id`, `texto` (testo **grezzo** con rumore: maiuscole, accenti persi, allungamenti, refusi, abbreviazioni da chat, emoji, URL), `texto_limpio` (versione pulita di riferimento), `etiqueta`, `dominio`, `dificil` (1 = negazione, ironia, giudizio misto, ecc.; ≈ 15 %). | Sintetico, originale |
| `tickets_es.csv`, `tickets_it.csv` | 400 ticket di supporto per lingua, 100 per categoria (`facturacion`, `envio`, `cuenta`, `devoluciones`; le stesse etichette in entrambe le lingue). Colonne: `id`, `texto` (grezzo, rumore più lieve), `texto_limpio`, `categoria`. Circa il 10-15 % dei ticket contiene parole tipiche di un'altra categoria. | Sintetico, originale |
| `gutenberg/quijote_cap1.txt` | Capitolo I del *Don Chisciotte della Mancia* (Miguel de Cervantes), ≈ 1 900 parole, in spagnolo del Seicento | Project Gutenberg, eBook #2000, pubblico dominio |
| `gutenberg/promessi_sposi_cap1.txt` | Prime ≈ 1 900 parole del capitolo I de *I promessi sposi* (Alessandro Manzoni), in italiano dell'Ottocento; conserva i segni `_corsivo_` del testo elettronico originale | Project Gutenberg, eBook #45334, pubblico dominio |

Note di qualità dei corpora sintetici:
- I testi grezzi di recensioni e ticket hanno rumore rispettivamente nel ≈ 45 % e nel ≈ 28 % dei casi; `texto_limpio` permette di misurare la qualità di una pulizia.
- Con TF-IDF (unigrammi e bigrammi) e regressione logistica, l'accuratezza in validazione incrociata a 5 partizioni sul testo grezzo è ≈ 0,78 per le recensioni e ≈ 0,88-0,89 per i ticket (valori orientativi; cambiano con il preprocessing).
- Alcuni ticket (≈ 6 %) terminano con un'etichetta di oggetto ("Oggetto: addebito") e vari testi contengono brevi verdetti finali; non influenzano i risultati, ma conviene tenerne conto leggendo i coefficienti di un modello.
- I ticket in spagnolo e in italiano sono adattamenti stretti l'uno dell'altro; le recensioni sono testi indipendenti.
- Sono corpora piccoli e con uno stile controllato: i risultati non si generalizzano a testi reali di qualunque provenienza.

I testi di Project Gutenberg sono stati presi dal testo elettronico, senza intestazione e piè di pagina del progetto e con gli a capo riuniti in paragrafi. I testi originali sono di pubblico dominio (autori morti da oltre cento anni).

---

## Corpus NER de noticias (clase 7) / Corpus NER di notizie (lezione 7)

### Español

| Archivo | Qué es |
|---|---|
| `noticias_ner_es.csv`, `noticias_ner_it.csv` | 67 frases de noticias sintéticas por idioma, con las entidades nombradas anotadas a mano (144 por idioma: 29 `PER`, 39 `ORG`, 57 `LOC` y 19 `MISC`; 5 frases sin ninguna entidad). Columnas: `id` (`es001`…, `it001`…), `texto` y `entidades`: un JSON con una lista de `[inicio, fin, etiqueta]`, donde `inicio` y `fin` son posiciones en **caracteres** (`texto[inicio:fin]` es la entidad, con `fin` excluido). |

Origen: **escritos originalmente para el curso**, sin copiar de otras fuentes. Las personas son inventadas; los lugares y las organizaciones son genéricos, inventados o reales, siempre en contextos neutros y sin noticias falsas ni difamatorias. La versión italiana es una **adaptación** de la española (mismas estructuras y casos difíciles, nombres y lugares adaptados), no una traducción palabra por palabra: la frase `n` de un archivo se corresponde con la frase `n` del otro.

Etiquetas (solo estas cuatro, las que producen los modelos pequeños de spaCy, para que la evaluación mida errores y no diferencias de esquema): `PER` personas; `ORG` empresas, instituciones, clubes, universidades y siglas de organizaciones; `LOC` países, ciudades, regiones, ríos, mares, montañas y lugares con nombre; `MISC` gentilicios y lenguas, eventos, premios, obras y productos con nombre.

Criterios de anotación (los mismos en las dos lenguas):
1. Se anota el nombre propio completo, sin puntuación final ni tratamiento o cargo (`la doctora` + `Elena Prado`).
2. Los sustantivos comunes de tipo en minúscula (`río`, `puerto`, `teatro`, `fiume`) no forman parte de la entidad; las partes con mayúscula sí (`Mar Cantábrico`, `Costa del Sol`, `Museo del Prado`).
3. Sin anidamiento ni solapamiento: una entidad no contiene otra.
4. Se etiqueta por el referente en la frase: un nombre de ciudad es `LOC`, y `ORG` si designa una institución que actúa (un ayuntamiento, un teatro que estrena una obra, un club).
5. Las siglas de organizaciones son `ORG` (`ONU`, `UE`, `OMS`), las de países `LOC` (`EE. UU.`, `USA`); los gentilicios y las lenguas, `MISC` aunque vayan en minúscula.
6. Las fechas, cifras, festividades y cargos no se anotan.

Casos difíciles incluidos a propósito: apellidos que coinciden con ciudades (`Carlos Toledo` / `Toledo`), siglas, gentilicios en minúscula, organizaciones con nombre de aspecto geográfico, nombres de pila sueltos, palabras en mayúscula solo por ir al principio de frase y frases sin entidades.

Verificación: cada desplazamiento coincide con el texto (la entidad no está vacía ni empieza o termina en un espacio), las etiquetas son las cuatro permitidas y no hay solapamientos.

Advertencia: es un corpus pequeño y de estilo controlado, con una sola persona anotadora: sirve para practicar la evaluación, no para medir un NER en general. Las cifras que se obtienen con él son un *ejemplo ilustrativo* y no se generalizan.

### Italiano

| File | Cos'è |
|---|---|
| `noticias_ner_es.csv`, `noticias_ner_it.csv` | 67 frasi di notizie sintetiche per lingua, con le entità nominate annotate a mano (144 per lingua: 29 `PER`, 39 `ORG`, 57 `LOC` e 19 `MISC`; 5 frasi senza alcuna entità). Colonne: `id` (`es001`…, `it001`…), `texto` ed `entidades`: un JSON con una lista di `[inizio, fine, etichetta]`, dove `inizio` e `fine` sono posizioni in **caratteri** (`texto[inicio:fin]` è l'entità, con `fine` escluso). |

Origine: **scritti originalmente per il corso**, senza copiare da altre fonti. Le persone sono inventate; i luoghi e le organizzazioni sono generici, inventati o reali, sempre in contesti neutri e senza notizie false né diffamatorie. La versione italiana è un **adattamento** di quella spagnola (stesse strutture e stessi casi difficili, nomi e luoghi adattati), non una traduzione parola per parola: la frase `n` di un file corrisponde alla frase `n` dell'altro.

Etichette (solo queste quattro, quelle che producono i modelli piccoli di spaCy, perché la valutazione misuri errori e non differenze di schema): `PER` persone; `ORG` aziende, istituzioni, club, università e sigle di organizzazioni; `LOC` paesi, città, regioni, fiumi, mari, montagne e luoghi con nome; `MISC` gentili e lingue, eventi, premi, opere e prodotti con nome.

Criteri di annotazione (gli stessi nelle due lingue):
1. Si annota il nome proprio completo, senza punteggiatura finale né titolo o carica (`la dottoressa` + `Elena Prato`).
2. I sostantivi comuni di tipo in minuscolo (`fiume`, `porto`, `teatro`, `río`) non fanno parte dell'entità; le parti con la maiuscola sì (`Mar Ligure`, `Costa Smeralda`, `Museo Egizio`).
3. Niente annidamento né sovrapposizione: un'entità non ne contiene un'altra.
4. Si etichetta per il referente nella frase: un nome di città è `LOC`, e `ORG` se designa un'istituzione che agisce (un comune, un teatro che mette in scena uno spettacolo, un club).
5. Le sigle di organizzazioni sono `ORG` (`ONU`, `UE`, `OMS`), quelle di paesi `LOC` (`EE. UU.`, `USA`); i gentili e le lingue, `MISC` anche se in minuscolo.
6. Date, cifre, festività e cariche non si annotano.

Casi difficili inclusi di proposito: cognomi che coincidono con città (`Carlo Siena` / `Siena`), sigle, gentili in minuscolo, organizzazioni con nome dall'aspetto geografico, nomi di battesimo isolati, parole in maiuscolo solo perché a inizio frase e frasi senza entità.

Verifica: ogni scostamento coincide con il testo (l'entità non è vuota né inizia o finisce con uno spazio), le etichette sono le quattro ammesse e non ci sono sovrapposizioni.

Avvertenza: è un corpus piccolo e con uno stile controllato, con una sola persona che annota: serve per esercitarsi con la valutazione, non per misurare un NER in generale. Le cifre che si ottengono sono un *esempio illustrativo* e non si generalizzano.


---

## Léxicos de polaridad (clase 6) / Lessici di polarità (lezione 6)

### Español

| Archivo | Qué es |
|---|---|
| `lexico_polaridad_es.csv`, `lexico_polaridad_it.csv` | Léxico minúsculo de polaridad para el análisis de sentimiento: 163 entradas en español (25 con −2, 47 con −1, 63 con +1, 28 con +2) y 159 en italiano (22, 45, 64, 28). Columnas: `palabra` (en minúscula, con acentos) y `polaridad` (entero: −2 muy negativa, −1 negativa, +1 positiva, +2 muy positiva). Solo contiene palabras con carga; los negadores y los intensificadores están en el código del notebook. |

Origen: **escritos originalmente para el curso**, sin copiar de ningún léxico existente. Se compilaron **de memoria, a partir de vocabulario general de opinión (valoración de productos y servicios), antes de medir nada sobre las reseñas**, y **no se han retocado después mirando los errores** (hacerlo sobre las reseñas de prueba haría las cifras optimistas). Incluyen formas flexionadas porque el curso no lematiza en esta clase. Comprobación de sospecha (clase 6): 66 de las 163 entradas en español y 89 de las 159 en italiano no aparecen en ninguna reseña del corpus.

Advertencia: son léxicos **de juguete**, con polisemias conocidas (por ejemplo `cuidado` figura con +1 en el sentido de «cuidado» = bien atendido, y aparece también como advertencia). Sirven para practicar la lógica del método, no como recurso general.

### Italiano

| File | Cos'è |
|---|---|
| `lexico_polaridad_es.csv`, `lexico_polaridad_it.csv` | Lessico minuscolo di polarità per l'analisi del sentimento: 163 voci in spagnolo (25 con −2, 47 con −1, 63 con +1, 28 con +2) e 159 in italiano (22, 45, 64, 28). Colonne: `palabra` (in minuscolo, con accenti) e `polaridad` (intero: −2 molto negativa, −1 negativa, +1 positiva, +2 molto positiva). Contiene solo parole con carica; i negatori e gli intensificatori sono nel codice del notebook. |

Origine: **scritti originalmente per il corso**, senza copiare da nessun lessico esistente. Sono stati compilati **a memoria, a partire da vocabolario generale di opinione (valutazione di prodotti e servizi), prima di misurare qualsiasi cosa sulle recensioni**, e **non sono stati ritoccati dopo guardando gli errori** (farlo sulle recensioni di test renderebbe le cifre ottimistiche). Includono forme flesse perché il corso in questa lezione non lemmatizza. Controllo di sospetto (lezione 6): 66 delle 163 voci in spagnolo e 89 delle 159 in italiano non compaiono in nessuna recensione del corpus.

Avvertenza: sono lessici **da gioco**, con polisemie note (per esempio `cuidado` compare con +1 nel senso di «ben curato» e compare anche come avvertimento). Servono per esercitare la logica del metodo, non come risorsa generale.


---

## Extractos ampliados de Gutenberg (clases 8-10) / Estratti ampliati di Gutenberg (lezioni 8-10)

### Español

| Archivo | Qué es | Origen |
|---|---|---|
| `gutenberg/quijote_cap01-14.txt` | *Don Quijote de la Mancha* (Miguel de Cervantes), primera parte, capítulos I a XIV completos (14 capítulos, 32 979 palabras), en español del siglo XVII | Project Gutenberg, eBook #2000, dominio público |
| `gutenberg/promessi_sposi_cap01-06.txt` | *I promessi sposi* (Alessandro Manzoni), capítulos I a VI completos (6 capítulos, 30 364 palabras), en italiano del siglo XIX | Project Gutenberg, eBook #45334, dominio público |

Preparación: sin el encabezado ni el pie de Project Gutenberg, con los saltos de línea reunidos en párrafos (separados por una línea en blanco) y con el título de cada capítulo como primer párrafo (`Capítulo ...` / `CAPITOLO ...`). En el texto italiano se han **quitado** las marcas `_cursiva_` del original electrónico y las líneas `[Illustrazione: ...]`. Los contenidos son genéricos y los reutilizan las clases 8 (modelos de lenguaje), 9 (topic modeling) y 10 (word embeddings). Advertencia: son lengua literaria de hace dos a cuatro siglos, no el español ni el italiano de hoy.

### Italiano

| File | Cos'è | Origine |
|---|---|---|
| `gutenberg/quijote_cap01-14.txt` | *Don Quijote de la Mancha* (Miguel de Cervantes), prima parte, capitoli da I a XIV completi (14 capitoli, 32 979 parole), in spagnolo del Seicento | Project Gutenberg, eBook #2000, pubblico dominio |
| `gutenberg/promessi_sposi_cap01-06.txt` | *I promessi sposi* (Alessandro Manzoni), capitoli da I a VI completi (6 capitoli, 30 364 parole), in italiano dell'Ottocento | Project Gutenberg, eBook #45334, pubblico dominio |

Preparazione: senza intestazione e piè di pagina di Project Gutenberg, con gli a capo riuniti in paragrafi (separati da una riga vuota) e con il titolo di ogni capitolo come primo paragrafo (`Capítulo ...` / `CAPITOLO ...`). Nel testo italiano sono stati **tolti** i segni `_corsivo_` del testo elettronico originale e le righe `[Illustrazione: ...]`. I contenuti sono generici e vengono riutilizzati dalle lezioni 8 (modelli di linguaggio), 9 (topic modeling) e 10 (word embedding). Avvertenza: è lingua letteraria di due o quattro secoli fa, non lo spagnolo né l'italiano di oggi.


---

## Corpus ampliados para embeddings (clase 10) / Corpus ampliati per gli embedding (lezione 10)

### Español

| Archivo | Qué es | Origen |
|---|---|---|
| `gutenberg/embeddings_es_obras.txt` | Cuatro obras en español unidas en un solo archivo (4,4 MB; en el orden de la tabla siguiente, separadas por una línea en blanco) | Project Gutenberg, dominio público |
| `gutenberg/embeddings_it_obras.txt` | Cuatro obras en italiano unidas en un solo archivo (2,8 MB) | Project Gutenberg, dominio público |

| Idioma | Obra y autor | eBook # | Palabras (separadas por espacios) |
|---|---|---|---|
| es | *Don Quijote de la Mancha*, Miguel de Cervantes (desde «Tasa» hasta el final) | 2000 | 386 593 |
| es | *La Regenta*, Leopoldo Alas «Clarín» (desde el prólogo) | 17073 | 306 449 |
| es | *Marianela*, Benito Pérez Galdós | 17340 | 50 531 |
| es | *Vida de Lazarillo de Tormes y de sus fortunas y adversidades* (anónimo, edición de Burgos, 1554) | 320 | 20 052 |
| it | *I promessi sposi*, Alessandro Manzoni (solo la novela: sin el ensayo introductorio de Scherillo, ni el índice ni el catálogo del editor) | 45334 | 214 989 |
| it | *Le avventure di Pinocchio*, Carlo Collodi | 52484 | 41 618 |
| it | *La vita militare: bozzetti*, Edmondo De Amicis (edición de 1869) | 45647 | 134 953 |
| it | *Novelle*, Edmondo De Amicis (5.ª ed., 1884) | 58341 | 73 195 |

Totales: **763 625 palabras** en español y **464 755** en italiano (separadas por espacios; el notebook cuenta solo las secuencias de letras: 758 107 y 473 339). Los eBook # se comprobaron leyendo el título y el autor en la cabecera del archivo antes de usarlos. Preparación: sin la cabecera y el pie de Project Gutenberg, sin la licencia ni las notas del transcriptor, con los saltos de línea reunidos en párrafos (separados por una línea en blanco), sin las marcas `_cursiva_` ni las líneas `[Illustrazione ...]`/`[Ilustración ...]` y con los guiones de diálogo `--` convertidos en `—`. No se ha corregido la ortografía ni se han modernizado las formas antiguas. Se buscó una segunda obra de Galdós (*Doña Perfecta*, eBook #15725) pero se descartó: la versión de Gutenberg es una edición escolar anglosajona con introducción, notas y vocabulario en inglés. El total por idioma queda por debajo de 5 MB. Advertencia: es lengua literaria de entre cuatro siglos y uno atrás, no el español ni el italiano de hoy, y con menos de un millón de palabras los embeddings son ruidosos.

Corpus «moderno» de la clase 10: no es un archivo nuevo, sino el texto de `resenas_*.csv` y `tickets_*.csv` (columna `texto_limpio`, 700 textos por idioma; 11 101 palabras en español y 10 757 en italiano según el corte en frases del notebook). Es **diminuto**: sirve solo para ver el contraste de registro y de cobertura de vocabulario, no para entrenar un modelo útil.

### Italiano

| File | Cos'è | Origine |
|---|---|---|
| `gutenberg/embeddings_es_obras.txt` | Quattro opere in spagnolo unite in un unico file (4,4 MB; nell'ordine della tabella seguente, separate da una riga vuota) | Project Gutenberg, pubblico dominio |
| `gutenberg/embeddings_it_obras.txt` | Quattro opere in italiano unite in un unico file (2,8 MB) | Project Gutenberg, pubblico dominio |

| Lingua | Opera e autore | eBook # | Parole (separate da spazi) |
|---|---|---|---|
| es | *Don Quijote de la Mancha*, Miguel de Cervantes (da «Tasa» fino alla fine) | 2000 | 386 593 |
| es | *La Regenta*, Leopoldo Alas «Clarín» (dal prologo) | 17073 | 306 449 |
| es | *Marianela*, Benito Pérez Galdós | 17340 | 50 531 |
| es | *Vida de Lazarillo de Tormes y de sus fortunas y adversidades* (anonimo, edizione di Burgos, 1554) | 320 | 20 052 |
| it | *I promessi sposi*, Alessandro Manzoni (solo il romanzo: senza il saggio introduttivo di Scherillo, né l'indice né il catalogo dell'editore) | 45334 | 214 989 |
| it | *Le avventure di Pinocchio*, Carlo Collodi | 52484 | 41 618 |
| it | *La vita militare: bozzetti*, Edmondo De Amicis (edizione del 1869) | 45647 | 134 953 |
| it | *Novelle*, Edmondo De Amicis (5ª ed., 1884) | 58341 | 73 195 |

Totali: **763 625 parole** in spagnolo e **464 755** in italiano (separate da spazi; il notebook conta solo le sequenze di lettere: 758 107 e 473 339). Gli eBook # sono stati verificati leggendo titolo e autore nell'intestazione del file prima di usarli. Preparazione: senza intestazione e piè di pagina di Project Gutenberg, senza la licenza né le note del trascrittore, con gli a capo riuniti in paragrafi (separati da una riga vuota), senza i segni `_corsivo_` né le righe `[Illustrazione ...]`/`[Ilustración ...]` e con i trattini di dialogo `--` convertiti in `—`. Non è stata corretta l'ortografia né sono state modernizzate le forme antiche. È stata cercata una seconda opera di Galdós (*Doña Perfecta*, eBook #15725) ma scartata: la versione di Gutenberg è un'edizione scolastica anglosassone con introduzione, note e vocabolario in inglese. Il totale per lingua resta sotto i 5 MB. Avvertenza: è lingua letteraria di tra quattro secoli e uno fa, non lo spagnolo né l'italiano di oggi, e con meno di un milione di parole gli embedding sono rumorosi.

Corpus «moderno» della lezione 10: non è un file nuovo, ma il testo di `resenas_*.csv` e `tickets_*.csv` (colonna `texto_limpio`, 700 testi per lingua; 11 101 parole in spagnolo e 10 757 in italiano secondo il taglio in frasi del notebook). È **minuscolo**: serve solo a vedere il contrasto di registro e di copertura del vocabolario, non ad addestrare un modello utile.

---

## Colección de FAQ para búsqueda y recuperación (clase 11) / Collezione di FAQ per ricerca e recupero (lezione 11)

### Español

| Archivo | Qué es |
|---|---|
| `faq_es.csv` | 40 artículos de ayuda cortos (preguntas frecuentes) de un producto de software de gestión de documentos ficticio, en 10 temas de 4 artículos cada uno (cuenta y acceso, facturación, sincronización, exportación/importación de datos, seguridad y copias de seguridad, integraciones, rendimiento y errores, aplicación móvil, trabajo en equipo, soporte técnico). Columnas: `id` (1-40), `titulo`, `texto` (2-3 frases). |
| `faq_consultas_es.csv` | 20 consultas de búsqueda con la relevancia fijada **de antemano** (antes de ejecutar ningún buscador), para evaluar con precisión@k y MRR. Columnas: `id`, `consulta`, `id_documento_relevante` (uno o varios `id` de `faq_es.csv`, separados por `;`), `tipo` (`facil`: comparte vocabulario claro con el documento; `dificil`: usa un sinónimo o una paráfrasis, pensada para que la búsqueda por palabras falle). 15 consultas fáciles y 5 difíciles. |

Origen: **escritos originalmente para este curso**, sin copiar de ninguna FAQ real ni de ningún producto existente; el nombre del producto y sus funciones son ficticios. Los 40 documentos y las 20 consultas (y su relevancia) se decidieron **antes** de ejecutar el motor de búsqueda del capítulo 11, precisamente para que la evaluación no se ajuste a los resultados. Todas las cifras del capítulo calculadas sobre esta colección (vocabulario, IDF, precisión@k, MRR) son un *ejemplo ilustrativo*: la colección es pequeña y de un solo dominio, y no se puede generalizar a buscadores reales.

### Italiano

| File | Cos'è |
|---|---|
| `faq_it.csv` | 40 brevi articoli di aiuto (domande frequenti) di un prodotto software di gestione documentale fittizio, in 10 temi da 4 articoli ciascuno (account e accesso, fatturazione, sincronizzazione, esportazione/importazione dati, sicurezza e backup, integrazioni, prestazioni ed errori, app mobile, lavoro di team, assistenza tecnica). Colonne: `id` (1-40, stessi temi dell'`id` corrispondente in `faq_es.csv`, ma testi scritti autonomamente in italiano, non una traduzione letterale), `titulo`, `texto` (2-3 frasi). |
| `faq_consultas_it.csv` | 20 query di ricerca con la rilevanza fissata **in anticipo** (prima di eseguire qualsiasi motore di ricerca), per valutare con precisione@k e MRR. Colonne: `id`, `consulta`, `id_documento_relevante` (uno o più `id` di `faq_it.csv`, separati da `;`), `tipo` (`facil`: condivide vocabolario chiaro con il documento; `dificil`: usa un sinonimo o una parafrasi, pensata perché la ricerca per parole fallisca). 15 query facili e 5 difficili. |

Origine: **scritte originalmente per questo corso**, senza copiare da nessuna FAQ reale né da alcun prodotto esistente; il nome del prodotto e le sue funzioni sono fittizi. I 40 documenti e le 20 query (e la loro rilevanza) sono stati decisi **prima** di eseguire il motore di ricerca del capitolo 11, proprio perché la valutazione non si adattasse ai risultati. Tutte le cifre del capitolo calcolate su questa collezione (vocabolario, IDF, precisione@k, MRR) sono un *esempio illustrativo*: la collezione è piccola e di un solo dominio, e non si può generalizzare a motori di ricerca reali.
