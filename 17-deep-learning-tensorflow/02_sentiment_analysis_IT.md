[⬅ Torna all'indice](00_indice_IT.md)

# Capitolo 2 — Il tuo primo classificatore di sentimenti

## Perché ti importa questo

Un marchio riceve migliaia di recensioni e commenti ogni giorno. Leggerli uno a uno è non pratico, ma un modello che distingue automaticamente il positivo dal negativo converte quel rumore in un allarme utile in secondi. Questo è il tuo primo contatto con l'elaborazione del linguaggio naturale, o Natural Language Processing (NLP): il ramo dell'IA che lavora con il testo invece di immagini o numeri solitari.

## La sfida: il testo non sono numeri

Una rete neurale capisce solo numeri. Prima di poter classificare una frase come positiva o negativa, devi convertirla in un vettore numerico. La tecnica più semplice si chiama "bag of words" (borsa di parole): immagina un magazzino con uno scompartimento per ogni parola diversa che esiste nel tuo vocabolario. Per rappresentare una frase, contrassegni con un uno gli scompartimenti delle parole che appaiono in essa e con uno zero il resto. Perdi l'ordine delle parole, ma conservi l'informazione di "cosa si dice", che è sufficiente per molti compiti.

## Costruisci il tuo vocabolario

Con TensorFlow e Keras non hai bisogno di costruire il vocabolario manualmente: lo strato `TextVectorization` lo fa per te a partire dai tuoi dati di addestramento.

```python
import tensorflow as tf

frases_ejemplo = [
    "me encantó esta película, es una obra maestra",
    "una pérdida de tiempo total, aburrida de principio a fin",
]

vectorizador = tf.keras.layers.TextVectorization(max_tokens=1000, output_mode="multi_hot")
vectorizador.adapt(frases_ejemplo)

print(vectorizador.get_vocabulary()[:10])
print(vectorizador(frases_ejemplo))
```

`adapt` percorre le tue frasi, conta le parole più frequenti, e costruisce il vocabolario. `output_mode="multi_hot"` produce esattamente la bag of words che abbiamo descritto sopra: un vettore con un uno per ogni parola presente nella frase.

## Addestra il tuo classificatore

Con le frasi già convertite in vettori, il resto è una rete densa come quella che già conosci dal capitolo precedente.

```python
modelo = tf.keras.Sequential([
    vectorizador,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

Usa un singolo neurone di output con attivazione sigmoide perché questo è un problema di due classi (positivo o negativo): l'output è direttamente la probabilità che la recensione sia positiva.

## Da dieci frasi a diecimila: più dati, meglio è

Il materiale originale di questo corso addestra prima con un piccolo corpus di recensioni di film (i file `pos.txt` e `neg.txt`, con più di cinquemila frasi positive e cinquemila negative) e poi dimostra la stessa pipeline su Sentiment140, un dataset di un milione seicentomila tweet etichettati automaticamente dal loro emoticon. La lezione è sempre la stessa: un modello di linguaggio migliora con più esempi variati, proprio come una persona che ha letto più testi riconosce meglio le sfumature.

```python
# Caricamento del piccolo corpus incluso nel materiale del corso
with open("pos.txt", encoding="latin-1") as f:
    positivas = [linea.strip() for linea in f if linea.strip()]
with open("neg.txt", encoding="latin-1") as f:
    negativas = [linea.strip() for linea in f if linea.strip()]

textos = positivas + negativas
etiquetas = [1] * len(positivas) + [0] * len(negativas)
```

## Riepilogo dei punti chiave

- Un modello di deep learning non capisce il testo: prima lo converti in numeri con tecniche come la bag of words.
- `TextVectorization` costruisce il vocabolario e trasforma le frasi in vettori per te, senza scrivere quel codice manualmente.
- Un problema di due classi (positivo/negativo) usa un neurone di output con attivazione sigmoide, invece dei dieci neuroni con softmax che hai usato per le cifre.
- Più dati di addestramento — da centinaia a milioni di esempi — di solito si traduce in un modello più accurato e più robusto di fronte a frasi nuove.

## Mettilo in pratica

**Scrivi** tre frasi nuove di tua creazione — una chiaramente positiva, una chiaramente negativa e una ambigua — e **passale** attraverso il tuo modello già addestrato con `modelo.predict()`. **Controlla** se il modello indovina le prime due e che probabilità assegna alla frase ambigua.

---

**Materiale sorgente di questo capitolo**: lezione 2 temi 5, 6, 7 e 8 ("Processing Data", "Preprocessing Continued", "Training and Testing on Data", "Using More Data"); file `neg.txt`, `pos.txt` e il pacchetto `sentiment140/`.

[⬅ Capitolo precedente](01_fundamentos_IT.md) · [Torna all'indice](00_indice_IT.md) · [Capitolo successivo ➡](03_rnn_lstm_IT.md)
