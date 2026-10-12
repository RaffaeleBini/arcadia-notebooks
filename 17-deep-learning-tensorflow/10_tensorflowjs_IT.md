[⬅ Torna all'indice](00_indice_IT.md)

# Capitolo 10 — Deep learning nel browser con TensorFlow.js

## Perché è importante

Tutto quello che hai addestrato fino ad ora funziona sul tuo computer o su Colab, con Python. Ma, e se volessi che il tuo modello funzionasse direttamente sul web, senza che l'utente installi nulla né debba inviare i suoi dati a un server? Per questo esiste TensorFlow.js: la stessa tecnologia che già conosci, eseguita in JavaScript (JS), dentro il browser.

## Perché eseguire un modello nel browser (con un'analogia)

Pensa alla differenza tra chiedere a uno chef di cucinarti qualcosa nel suo ristorante (il server) e dargli una ricetta già preparata per seguirla tu stesso a casa (il browser). Cucinare a casa è più lento che in un ristorante professionale, ma non dipendi dall'inviare i tuoi ingredienti da un'altra parte né dall'aspettare che lo chef sia disponibile: tutto succede nella tua cucina, all'istante e in privato. TensorFlow.js funziona allo stesso modo: il modello si esegue nel dispositivo dell'utente, senza inviare i suoi dati a nessun server, con meno latenza e più privacy, anche se un po' più lentamente di un server dedicato con GPU.

## Addestra un modello direttamente in HTML

TensorFlow.js può essere usato senza installare nulla: basta includere il suo script in una pagina HTML.

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script>
  const modelo = tf.sequential();
  modelo.add(tf.layers.dense({units: 1, inputShape: [1]}));
  modelo.compile({optimizer: "sgd", loss: "meanSquaredError"});

  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
  const ys = tf.tensor2d([2, 4, 6, 8], [4, 1]);

  modelo.fit(xs, ys, {epochs: 200}).then(() => {
    modelo.predict(tf.tensor2d([5], [1, 1])).print();
  });
</script>
```

Nota il parallelismo con Keras: `tf.sequential()` è l'equivalente di `tf.keras.Sequential()`, e `tf.layers.dense` è l'equivalente di `tf.keras.layers.Dense`. Se già conosci Keras, imparare l'API dei livelli di TensorFlow.js è principalmente imparare la sua sintassi in JavaScript.

## Addestra in Python, distribuisci nel browser

Di solito non si addestrare reti complesse direttamente in JavaScript —l'addestramento è lento senza una GPU potente—, ma si addestrano in Python con Keras, come hai fatto in tutto il corso, e poi si converte il modello addestrato in un formato che TensorFlow.js può caricare.

```bash
pip install tensorflowjs
tensorflowjs_converter --input_format=keras modelo.h5 modelo_web/
```

Nel browser, carichi quel modello convertito e fai previsioni senza aver bisogno di Python in nessun momento:

```html
<script>
  async function cargarYPredicir() {
    const modelo = await tf.loadLayersModel("modelo_web/model.json");
    const prediccion = modelo.predict(tf.tensor2d([[0.5]]));
    prediccion.print();
  }
  cargarYPredicir();
</script>
```

## Un caso reale: un'IA che gioca a Pong nel browser

Questo schema —addestrare in Python, distribuire nel browser— è esattamente quello usato da un progetto classico di questo corso: un'IA che impara a giocare a Pong. Il modello riceve la posizione della pallina e della racchetta come ingresso, e decide di muovere la racchetta su, giù o lasciarla ferma, con lo stesso approccio di apprendimento per rinforzo del capitolo 6, ma distribuito come una pagina web giocabile da chiunque, senza installare nulla.

## Riepilogo dei punti chiave

- TensorFlow.js esegue modelli direttamente nel browser, con meno latenza e senza inviare dati a un server, al costo di un po' di velocità di calcolo.
- L'API dei livelli di TensorFlow.js (`tf.sequential`, `tf.layers.dense`) è praticamente una copia di Keras, ma in JavaScript.
- Lo schema abituale è addestrare in Python con Keras e convertire il modello addestrato con `tensorflowjs_converter`, invece di addestrare reti complesse direttamente nel browser.
- Un progetto di apprendimento per rinforzo, come Pong con IA, può essere distribuito come qualsiasi altro modello: addestrato in Python, giocato nel browser.

## Mettilo in pratica

**Copia** l'esempio HTML dall'inizio di questo capitolo in un file `.html`, **aprilo** nel tuo browser, e **apri la console dello sviluppatore** (F12) per vedere il risultato della previsione stampato con `.print()`. **Cambia** i dati di addestramento `xs`/`ys` in modo che rappresentino un'altra relazione —per esempio, il triplo invece del doppio— e **verifica** che la previsione si adatti alla nuova relazione.

---

**Materiale sorgente di questo capitolo**: lezione 10 argomenti 2 e 3 ("AI Pong in TensorFlow.js", "Training Model in Python and Loading into TensorFlow.js"); `Tensorflosjs_basic.html` e cartella `pong_game/`.

[⬅ Capitolo precedente](09_chatbot_IT.md) · [Torna all'indice](00_indice_IT.md) · [Prossimo capitolo ➡](11_casos_de_uso_IT.md)
