# 08. NLP: tokenizzazione e sequenze

## Perché questo argomento è importante

Fino ad ora hai addestrato modelli con numeri (relazioni tra X e Y) e con immagini (pixel, che sono già numeri organizzati in una griglia). Ma cosa succede quando il dato di input è una frase come "amo il mio cane"? Una rete neurale non capisce le parole: sa solo moltiplicare, sommare e aggiustare numeri. Prima che un modello possa imparare qualcosa su un testo, hai bisogno di convertire quel testo in numeri in un modo che conservi il suo significato.

Questo ramo dell'Intelligenza Artificiale (IA), quello che si occupa che le macchine capiscano e generino linguaggio umano, si chiama Elaborazione del Linguaggio Naturale (NLP, Natural Language Processing). Compiti tanto diversi come rilevare se una recensione è positiva o negativa, tradurre un testo o generare frasi nuove hanno qualcosa in comune: tutti iniziano risolvendo lo stesso problema di partenza, convertire parole in numeri. Questo capitolo copre proprio quel primo passo, che userai nel resto delle classi di NLP del corso.

## Il problema di partenza: le lettere non bastano

Potresti pensare che la soluzione più diretta sia codificare ogni lettera con un numero, per esempio usando ASCII (American Standard Code for Information Interchange), lo standard che assegna un numero a ogni carattere. Il problema è che due parole possono condividere esattamente le stesse lettere e significare cose opposte: "listen" (ascoltare) e "silent" (silenzioso) usano le stesse lettere in ordine diverso. Se codifichi lettera per lettera, non c'è modo che il modello distingua il senso di una parola a partire dai suoi caratteri sciolti.

L'alternativa che funziona meglio è codificare parole complete, non lettere. Se alla parola "io" assegni il numero uno, ad "amo" il numero due, a "il" il numero tre e a "cane" il numero quattro, la frase "io amo il cane" si converte nella sequenza 1, 2, 3, 4. Se ora hai la frase "io amo il gatto" e "gatto" è una parola nuova a cui assegni il numero cinque, quella frase si converte in 1, 2, 3, 5. Con solo confrontare le due sequenze numeriche, un modello già può rilevare che entrambe le frasi si assomigliano molto: condividono tre dei loro quattro numeri.

## Cosa è tokenizzare e come si costruisce un vocabolario

**Tokenizzare** è il processo di dividere un testo in unità più piccole, chiamate **token**, e assegnare un numero intero a ognuna. Il caso più abituale, e quello che userai in questo corso, è tokenizzare per parole: ogni parola distinta del testo si converte in un token. Esistono anche altre forme di tokenizzazione — per caratteri, o per sottoparole (frammenti più piccoli di una parola completa, utili per lingue con molte forme verbali o per ridurre il numero di token necessari) — ma l'approccio per parole è il punto di partenza più chiaro per imparare il concetto.

L'insieme completo di token distinti che conosce il tuo modello si chiama **vocabolario**. Per costruirlo, TensorFlow (TF) offre la classe `Tokenizer`, dentro il modulo di preprocessing del testo di Keras. Le passi un insieme di frasi (il tuo corpus) e lei si incarica di:

- Percorrere tutte le frasi e rilevare le parole distinte
- Passare il testo a minuscole e togliere la punteggiatura automaticamente, così "cane" e "cane!" si riconoscono come la stessa parola
- Assegnare un numero intero a ogni parola, iniziando dalle più frequenti
- Conservare quella relazione parola-numero in un dizionario chiamato `word_index`

Nota che il `Tokenizer` già risolve per te un problema pratico: se nel tuo corpus appare "cane!" con segno di esclamazione e "cane" senza, non finirai con due token distinti per la stessa parola. La pulizia della punteggiatura avviene automaticamente quando regoli il tokenizzatore.

## Come gestire parole fuori dal vocabolario (OOV)

Il tuo modello può conoscere solo le parole che ha visto durante l'aggiustamento del tokenizzatore. Cosa succede se, più avanti, gli presenti una frase con una parola nuova, che non era nel vocabolario originale? Per esempio, addestri il tuo tokenizzatore con frasi su cani e gatti, e poi gli chiedi di elaborare una frase che menciona un "lamantino". Senza nessun meccanismo aggiuntivo, il `Tokenizer` semplicemente ignorerebbe quella parola, e la sequenza numerica risultante perderebbe una posizione: la frase si accorcerebbe e il modello non saprebbe in quale parte della frase manca l'informazione.

La soluzione è riservare un token speciale per tutte le parole che il modello non riconosce, chiamato token **OOV** (Out Of Vocabulary, fuori dal vocabolario). Quando crei il `Tokenizer`, indichi quale testo vuoi usare come marcatore, per esempio `"<OOV>"`, e da quel momento in poi qualunque parola sconosciuta si sostituisce con quel token invece di scomparire. Così la frase conserva la sua lunghezza originale e la sua struttura, anche se si perde il significato esatto della parola nuova. Non è una soluzione perfetta — il modello non impara nulla su "lamantino" — ma è molto meglio di perdere quella posizione della sequenza.

## Da frasi a sequenze numeriche: perché serve il padding

Con il vocabolario già costruito, il passo successivo è convertire ogni frase completa nella sua sequenza di numeri, sostituendo ogni parola con il suo token. Qui appare un problema nuovo: le frasi del mondo reale hanno lunghezze diverse. "Io amo il mio cane" ha quattro parole; "Credi che il mio cane sia incredibile?" ha sette. Ma una rete neurale si aspetta che tutti gli input di uno stesso lotto abbiano la stessa forma (shape), esattamente come succedeva con le immagini di dimensione fissa che hai usato nelle classi precedenti.

La soluzione si chiama **padding** (riempimento): eguagliare la lunghezza di tutte le sequenze aggiungendo zeri dove occorra. TensorFlow offre per questo la funzione `pad_sequences`, che ti permette di decidere:

- La lunghezza finale di tutte le sequenze (`maxlen`); se non la indichi, usa la lunghezza della sequenza più lunga del lotto
- Se il riempimento si aggiunge all'inizio (`padding='pre'`, il valore di default) o alla fine (`padding='post'`) della sequenza
- Se una frase è più lunga di `maxlen`, se si ritaglia dall'inizio (`truncating='pre'`) o dalla fine (`truncating='post'`, il valore di default)

Grazie al padding, un lotto di frasi di lunghezze molto diverse si converte in una matrice unica di numeri con una forma omogenea, pronta per entrare in una rete neurale.

## Esempio di codice commentato

L'esempio seguente riproduce il flusso completo: tokenizzare un corpus piccolo, gestire parole nuove con il token OOV e generare sequenze con padding.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Corpus di addestramento: quattro frasi di esempio
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

# Creiamo il tokenizzatore:
# - num_words limita il vocabolario alle 100 parole più frequenti
# - oov_token riserva un token per le parole che non stanno nel vocabolario
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

# Regola il tokenizzatore sul corpus: qui si costruisce il vocabolario
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("Vocabolario:", word_index)

# Convertiamo ogni frase nella sua sequenza di numeri
sequences = tokenizer.texts_to_sequences(sentences)
print("Sequenze:", sequences)

# Eguagliamo la lunghezza di tutte le sequenze con padding
padded = pad_sequences(sequences, maxlen=5)
print("Sequenze con padding:")
print(padded)

# Proviamo con frasi nuove, con parole che il tokenizzatore non conosce
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_data)
print("Sequenze di prova (con token OOV):", test_seq)
```

Quando esegui questo codice, verifica due dettagli chiave. Primo, in `word_index` la parola "<OOV>" occupa sempre il token numero uno, perché è quella che più serve al modello avere sempre disponibile. Secondo, nelle frasi di prova, parole come "really", "loves" o "manatee" — che non stavano nel corpus di addestramento — si sostituiscono con quel token OOV invece di scomparire, così la sequenza conserva la sua lunghezza e la sua struttura.

## Riassunto dei punti chiave

- Una rete neurale elabora solo numeri, così qualunque testo deve convertirsi a numeri prima di addestrare un modello: a questo processo, tra gli altri, si dedica l'Elaborazione del Linguaggio Naturale (NLP)
- Codificare lettera per lettera non funziona perché parole con le stesse lettere possono avere significati opposti; codificare parole complete sì conserva il senso
- Tokenizzare è dividere il testo in unità (token, normalmente parole) e assegnare un numero intero a ognuna; l'insieme di token conosciuti è il vocabolario
- Il token OOV evita che le parole nuove, non viste durante l'aggiustamento del tokenizzatore, si perdano o scardino la lunghezza della sequenza
- Il padding eguaglia la lunghezza di tutte le sequenze aggiungendo zeri, affinché possano elaborarsi insieme in uno stesso lotto
- `Tokenizer`, `texts_to_sequences` e `pad_sequences` sono i tre pezzi di TensorFlow che risolvono questo flusso completo

## Mettilo in pratica

Prendi queste tre frasi:

- "Il gatto dorme sul divano"
- "Il cane gioca nel giardino"
- "Il gatto e il cane sono amici"

A mano, su carta, costruisci il vocabolario che genererebbe un `Tokenizer`: elenca ogni parola distinta (ricorda passarla a minuscole) e numerala iniziando dalla più frequente. Dopo, converti ogni frase nella sua sequenza di numeri usando il tuo proprio vocabolario. Per ultimo, applica padding a mano aggiungendo zeri all'inizio delle sequenze più corte fino a quando tutte e tre hanno la stessa lunghezza di quella più lunga. Quando finisci, confronta il tuo risultato con quello che otterresti eseguendo il codice di questo capitolo con quelle stesse frasi.
