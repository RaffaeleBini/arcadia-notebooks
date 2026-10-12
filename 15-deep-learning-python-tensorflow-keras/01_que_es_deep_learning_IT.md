[[Indice](00_indice_IT.md) · Prossimo: [Il tuo primo modello con Keras →](02_primer_modelo_mnist_IT.md)]

# 1. Cos'è il Deep Learning? Neuroni e reti da zero

## Perché è importante per te

Ogni volta che il tuo cellulare riconosce il tuo viso, un traduttore online converte una frase al volo o una piattaforma di streaming indovina quale serie ti piacerà, c'è una rete neurale dietro. Il Deep Learning (DL, apprendimento profondo) è la tecnica che rese possibile questo salto: invece di programmare regole esplicite ("se il pixel è scuro e si trova in questa posizione, è un occhio"), costruisci un sistema che impara quelle regole da solo a partire da esempi.

Non hai bisogno di un dottorato in matematica per iniziare. Hai bisogno di capire cos'è un neurone artificiale, come si organizzano in strati e perché questo basta affinché una macchina impari pattern complessi. Questo è quello che vedi in questo capitolo.

## Il neurone artificiale, spiegato con un'analogia

Immagina una giuria di un neurone: riceve varie opinioni (ingressi), pesa quanto è importante ognuna (pesi) e emette un verdetto (uscita). Quanto più peso dà a un'opinione, più quella opinione influenza il risultato finale.

Formalmente, un neurone artificiale:

1. Riceve diversi valori di ingresso (per esempio, i pixel di un'immagine)
2. Moltiplica ogni ingresso per un peso (che indica la sua importanza)
3. Somma tutti questi prodotti e aggiunge un valore aggiuntivo chiamato bias (distorsione)
4. Passa il risultato attraverso una funzione di attivazione, che decide se il neurone "si attiva" e con quale intensità

Un singolo neurone non fa gran cosa. Il potere emerge quando colleghi migliaia di neuroni in strati.

## Strati: di ingresso, nascosti e di uscita

Una rete neurale artificiale (ANN, Artificial Neural Network) si organizza in tre tipi di strati:

- **Strato di ingresso**: riceve i dati grezzi (i pixel di una foto, le parole di una frase, il prezzo di un'azione)
- **Strati nascosti**: trasformano quei dati passo dopo passo, estraendo pattern sempre più astratti. Se ci sono due o più strati nascosti, parliamo di una rete *profonda*, da qui il nome "deep learning"
- **Strato di uscita**: fornisce il risultato finale (per esempio, la probabilità che l'immagine sia un gatto)

La differenza fondamentale rispetto al Machine Learning (ML, apprendimento automatico) tradizionale è che tu non decidi quali caratteristiche sono importanti. In un sistema classico, un ingegnere dovrebbe dire al modello "guarda le orecchie appuntite" o "guarda i baffi". In una rete profonda, i primi strati imparano a rilevare bordi e texture, gli strati intermedi combinano questi bordi in forme, e gli ultimi strati combinano quelle forme in concetti come "orecchio di gatto". La rete impara tutto questo da sola, partendo da esempi.

## Funzioni di attivazione: l'interruttore di ogni neurone

Senza una funzione di attivazione, una rete neurale sarebbe solo una calcolatrice di somme ponderate: utile, ma incapace di imparare pattern complessi e non lineari (come distinguere un gatto da un cane). Le funzioni di attivazione più comuni sono:

- **Step (Scalino)**: la più semplice, il neurone si attiva (1) o no (0), senza sfumature. Raramente si usa oggi
- **Sigmoid**: comprime qualsiasi valore in un intervallo tra zero e uno, utile quando hai bisogno di una probabilità
- **ReLU (Rectified Linear Unit)**: lascia passare i valori positivi così come sono e converte i negativi in zero. È la più usata negli strati nascosti perché è economica da calcolare e perché evita alcuni problemi di addestramento
- **Softmax**: converte un insieme di valori in una distribuzione di probabilità che somma uno, ideale per lo strato di uscita quando devi scegliere tra varie categorie (per esempio, "è uno zero, un uno, un due... o un nove")

## Un primo sguardo a Python, TensorFlow e Keras

Costruirai le tue reti con TensorFlow, una libreria di Google per il calcolo numerico e il Deep Learning, e con Keras, l'interfaccia di alto livello che rende la definizione di una rete semplice come impilare strati. Dal TensorFlow 2, Keras è incluso direttamente: non devi installarlo separatamente.

```python
# Installa TensorFlow (include già Keras) se non ce l'hai
# Su Google Colab normalmente è già preinstallato
!pip install --upgrade tensorflow

import tensorflow as tf

# Verifica la versione installata
print(tf.__version__)
```

Con questo hai già configurato l'ambiente. Nel prossimo capitolo usi questi stessi strumenti per costruire, addestrare e valutare il tuo primo modello reale.

## Riepilogo dei punti chiave

- Un neurone artificiale combina ingressi ponderati e decide la sua uscita con una funzione di attivazione
- Una rete neurale si organizza in strato di ingresso, uno o più strati nascosti e strato di uscita
- "Profonda" significa che ci sono due o più strati nascosti: qui nasce il nome "deep learning"
- Il grande vantaggio del deep learning rispetto all'ML tradizionale è che la rete estrae le caratteristiche importanti da sola
- ReLU domina negli strati nascosti, softmax domina nello strato di uscita per la classificazione multiclasse
- TensorFlow e Keras sono gli strumenti che usi in questo corso per costruire le tue reti

## Mettilo in pratica

Apri un notebook Python (sul tuo computer o su Google Colab) e installa TensorFlow con il codice di questo capitolo. Poi scrivi una funzione in Python che simuli un singolo neurone: deve ricevere un elenco di ingressi e un elenco di pesi della stessa dimensione, calcolare la somma ponderata più un bias, e restituire il risultato dell'applicazione della funzione ReLU (suggerimento: ReLU è semplicemente `max(0, x)`). Prova la tua funzione con tre ingressi diversi e verifica che non restituisca mai un numero negativo.

---

[[Indice](00_indice_IT.md) · Prossimo: [Il tuo primo modello con Keras →](02_primer_modelo_mnist_IT.md)]
