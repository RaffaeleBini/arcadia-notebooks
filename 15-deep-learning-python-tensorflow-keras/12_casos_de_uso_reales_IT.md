[← Precedente: [Bilancia i tuoi dati e allena il tuo modello LSTM finale](11_caso_cripto_modelo_final_IT.md) · [Indice](00_indice_IT.md)]

# 12. Casi d'uso reali del Deep Learning e prossimi passi

## Perché ti importa

Hai costruito tre progetti completi nel corso di questo corso: un classificatore di cifre, un classificatore di immagini di gatti e cani, e un predittore di movimenti di prezzo di criptovalute. Ognuno illustra una famiglia diversa di problemi che risolvi con Deep Learning nel mondo reale. Questo capitolo chiude il manuale rivedendo questi tre casi da una prospettiva pratica: quale problema reale rappresenta ogni uno, e a quali altri problemi puoi applicare la stessa architettura domani.

## Caso 1: classificazione di cifre manoscritte (MNIST)

**Il problema reale che rappresenta**: automatizzare la lettura di testo o simboli scritti a mano, senza che un umano debba rivedere ogni uno. Applicato in produzione, questo stesso approccio alimenta il riconoscimento ottico dei caratteri (OCR) in moduli scansionati, assegni bancari, targhe automobilistiche o codici postali su buste postali.

**Perché importa come punto di partenza**: è il problema più semplice dove vedi il ciclo completo (caricare dati, normalizzare, costruire con `Sequential`, compilare, allenare, valutare, prevedere) senza che la complessità dei dati ti distragga dal comprendere quel ciclo.

**A quali altri problemi si generalizza**: qualsiasi classificazione di immagini piccole con poche classi ben differenziate, come classificare i tipi di difetti nei pezzi industriali fotografati da una telecamera di controllo di qualità.

## Caso 2: classificazione di immagini reali (gatti vs. cani)

**Il problema reale che rappresenta**: riconoscere oggetti, categorie o condizioni nelle fotografie del mondo reale, non in dataset già perfettamente confezionati. La pipeline di preprocessing che hai costruito (carica, scala di grigi, ridimensionamento, bilanciamento) è la stessa che useresti per classificare radiografie mediche come sane o con anomalie, foto di prodotti in un catalogo di e-commerce, o immagini satellitari per rilevare la deforestazione.

**Perché l'architettura CNN importa**: senza livelli convoluzionali, questo problema sarebbe computazionalmente inviabile con immagini di dimensioni reali. Le CNN sono la base di praticamente ogni sistema di visione artificiale in produzione oggi, da telecamere di sicurezza con rilevamento di oggetti ai sistemi di guida assistita.

**A quali altri problemi si generalizza**: rilevamento di oggetti (dove è ogni oggetto nell'immagine, non solo cosa contiene), segmentazione di immagini (quali pixel esatti appartengono a ogni oggetto), e riconoscimento facciale.

## Caso 3: previsione di serie temporali (criptovalute)

**Il problema reale che rappresenta**: anticipare il comportamento futuro di una grandezza che cambia nel tempo, a partire dal suo storico recente. Il materiale sorgente di questo corso menziona esplicitamente che lo stesso approccio (finestre temporali + LSTM) si applica a problemi molto diversi da quello finanziario: prevedere se un server si surriscalderà a partire dalle sue metriche storiche, o anticipare picchi di traffico in un sito web a partire dallo storico di visite per ora.

**Perché l'architettura RNN/LSTM importa**: a differenza dei due casi precedenti, qui l'ordine dei dati è l'informazione. Una CNN o una rete densa non hanno nessun meccanismo per rappresentare "questo è venuto prima di quello"; una RNN sì.

**A quali altri problemi si generalizza**: analisi del sentimento nel testo (una frase è una sequenza di parole), traduzione automatica, riconoscimento vocale, e qualsiasi manutenzione predittiva basata su sensori che registrano dati nel corso del tempo.

## Il filo conduttore tra i tre casi

Se rileggi i tre progetti, la struttura di lavoro è sempre la stessa, cambia solo il contenuto di ogni passo:

1. Ottieni dati rappresentativi del problema reale, non solo di un caso ideale
2. Preprocessa e normalizza quei dati in un formato uniforme che la rete possa consumare
3. Scegli l'architettura secondo la natura del dato: `Dense` per dati tabulari semplici, CNN per dati con struttura spaziale (immagini), RNN/LSTM per dati con struttura temporale (sequenze)
4. Compila, allena e usa TensorBoard per vigilare l'overfitting
5. Valuta con metriche oneste, non solo con accuracy
6. Testa con dati veramente nuovi prima di fidarti del modello

## Verso dove proseguire dopo questo corso

Questo manuale ti ha dato le basi. Questi sono i passi naturali per continuare ad avanzare:

- **Approfondisci una famiglia di architetture**: se ti ha attratto la visione artificiale, esplora architetture CNN più avanzate (ResNet, EfficientNet) e tecniche di aumento dei dati (data augmentation)
- **Esplora l'elaborazione del linguaggio naturale (NLP)**: gli stessi concetti di RNN/LSTM che hai usato con le criptovalute sono la porta d'accesso ai modelli di testo, anche se oggi la maggior parte dei sistemi di NLP in produzione usano architetture di tipo transformer
- **Impara a portare modelli in produzione**: allenare un modello in un notebook è solo metà del lavoro. Strumenti come TFX (TensorFlow Extended) ti aiutano a costruire pipeline complete di dati, allenamento e distribuzione
- **Pratica con dataset propri**: l'esercizio più prezioso che puoi fare ora è prendere un problema di tuo interesse o del tuo lavoro, e applicarvi lo stesso processo di sei passi che hai appena ripassato

## Riepilogo dei punti chiave

- MNIST rappresenta il problema del riconoscimento di caratteri e della classificazione semplice, base dell'OCR
- Gatti vs. cani rappresenta la classificazione di immagini reali con CNN, base della visione artificiale in produzione
- La previsione di criptovalute rappresenta l'analisi di serie temporali con RNN/LSTM, generalizzabile a manutenzione predittiva, NLP e riconoscimento vocale
- Il processo di lavoro (dati, preprocessing, architettura secondo il tipo di dato, allenamento vigilato, valutazione onesta, test con dati nuovi) è lo stesso in qualsiasi progetto di Deep Learning
- Continuare ad imparare significa, soprattutto, applicare questo processo a un problema proprio

## Mettilo in pratica

Scegli un problema del tuo ambiente (lavoro, studi o interesse personale) che potresti risolvere con una delle tre architetture di questo corso. Scrivi, in un paragrafo, quali dati avresti bisogno, quale architettura sceglieresti (Dense, CNN o RNN/LSTM) e perché, e quale metrica useresti per sapere se il modello funziona davvero. **Condividilo** con il resto della classe: è il modo migliore di verificare se davvero hai interiorizzato il processo completo di questo corso.

---

[← Precedente: [Bilancia i tuoi dati e allena il tuo modello LSTM finale](11_caso_cripto_modelo_final_IT.md) · [Indice](00_indice_IT.md)]
