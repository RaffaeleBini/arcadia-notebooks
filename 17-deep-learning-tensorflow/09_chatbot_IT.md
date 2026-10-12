[⬅ Torna all'indice](00_indice_IT.md)

# Capitolo 9 — Costruire un chatbot conversazionale

## Perché è importante

Un chatbot ben addestrato combina due sfide che già conosci —elaborare testo (capitolo 2) e lavorare con sequenze (capitolo 3)— con una nuova: imparare a generare una risposta completa, non solo un'etichetta. In questo capitolo vedi il percorso completo di un progetto di questo tipo, da pulire conversazioni reali fino a capire i concetti che rendono possibile a un modello "rispondere" invece di solo "classificare".

## Da dove vengono le conversazioni di addestramento

Addestrare un chatbot richiede milioni di coppie domanda-risposta reali. Una fonte abituale sono i commenti di forum come Reddit, dove ogni commento ha un "genitore" (il commento a cui risponde) e il commento stesso funziona come risposta. Prima di poter usare questi dati, devi ripulirli e convalidarli:

```python
def es_aceptable(comentario):
    """Filtra commenti vuoti, troppo lunghi, o eliminati."""
    if len(comentario) < 1 or len(comentario) > 1000:
        return False
    if comentario in ("[deleted]", "[removed]"):
        return False
    return True

def limpiar_texto(texto):
    """Normalizza salti di riga e virgolette per un formato coerente."""
    texto = texto.replace("\n", " ").replace("\r", " ")
    texto = texto.replace('"', "'")
    return texto.strip()
```

Con milioni di commenti, questo filtraggio importa: un chatbot addestrato con risposte vuote, insulti eliminati o testo mal formattato impara a imitare quel rumore. La qualità dei dati di ingresso determina, in larga misura, la qualità delle risposte del modello finale.

## Da classificare a generare: token, padding e traduzione

I modelli precedenti producevano una singola etichetta (positivo/negativo, gatto/cane). Un chatbot ha bisogno di generare una sequenza completa di parole, parola dopo parola. Questo si risolve con un'architettura presa in prestito dalla traduzione automatica, o Neural Machine Translation (NMT): il modello "traduce" la domanda di ingresso in una risposta di uscita, anche se entrambe sono nello stesso idioma.

Affinché il modello lavori con frasi di lunghezza diversa, ogni parola viene convertita in un identificatore numerico (tokenizzazione), e le frasi più corte vengono riempite con un token speciale (padding) fino a uguagliare la lunghezza della più lunga nel lotto. Le architetture moderne di NMT aggiungono inoltre un meccanismo di "attenzione": invece di riassumere l'intera frase di ingresso in un singolo vettore prima di generare la risposta, l'attenzione permette al modello di consultare, parola dopo parola della risposta che sta generando, quali parti specifiche della domanda originale sono più rilevanti in quel momento.

```python
capa_tokenizador = tf.keras.layers.TextVectorization(
    max_tokens=20000,
    output_sequence_length=20,  # padding/taglio a 20 token per frase
)
```

## Interagisci con un modello già addestrato

Una volta addestrato, testare un chatbot non è semplicemente come chiamare `modelo.predict()` una volta: normalmente si generano diverse risposte candidate —una tecnica chiamata "beam search"— e si sceglie la migliore secondo criteri come evitare parole sconosciute o penalizzare risposte troppo ripetitive.

## Riepilogo dei punti chiave

- Costruire un chatbot inizia pulendo e convalidando dati conversazionali reali: la qualità di ingresso limita la qualità delle risposte.
- Generare una risposta completa —non solo un'etichetta— usa architetture prese dalla traduzione automatica (NMT), con tokenizzazione, padding e meccanismi di attenzione.
- Testare un chatbot addestrato di solito implica generare diverse risposte candidate e scegliere la migliore con criteri aggiuntivi, non solo prendere la prima previsione.

## Mettilo in pratica

**Scrivi** una funzione `limpiar_y_validar(comentarios)` che riceva un elenco di commenti grezzi e restituisca solo quelli che passano `es_aceptable` dopo aver passato per `limpiar_texto`. **Provala** con un elenco di esempio che includa almeno un commento vuoto, uno marcato come `"[deleted]"`, e uno valido.

---

**Materiale sorgente di questo capitolo**: lezione 9 argomenti 1 a 4 ("Data Structure", "Buffering Dataset", "Determining Insert", "Building Database"); lezione 10 argomento 4 ("NMT Concepts and Parameters") e argomento 5 ("Interacting with the Chatbot").

[⬅ Capitolo precedente](08_cnn_3d_imagen_medica_IT.md) · [Torna all'indice](00_indice_IT.md) · [Prossimo capitolo ➡](10_tensorflowjs_IT.md)
