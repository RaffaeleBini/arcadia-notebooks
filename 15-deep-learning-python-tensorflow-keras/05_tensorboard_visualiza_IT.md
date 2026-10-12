[← Precedente: [Reti Neurali Convoluzionali](04_redes_convolucionales_cnn_IT.md) · [Indice](00_indice_IT.md) · Prossimo: [Ottimizza i tuoi modelli →](06_optimiza_hiperparametros_IT.md)]

# 5. TensorBoard: visualizza e depura i tuoi modelli

## Perché è importante per te

Finora hai visto i risultati del tuo addestramento come numeri sciolti che Keras stampa sulla console: una riga di loss e accuracy per ogni epoch. Questo funziona con 10 epoch, ma diventa ingestibile non appena inizi a confrontare varianti di un modello o a addestrare per più tempo. TensorBoard è lo strumento di visualizzazione che viene con TensorFlow, e ti permette di vedere l'evoluzione del tuo addestramento in grafici interattivi, in tempo reale, direttamente nel tuo browser.

## Cosa puoi vedere in TensorBoard

TensorBoard organizza le informazioni in diverse schede:

- **Scalars**: grafici di loss e accuracy (di addestramento e di validazione) nel corso degli epoch. È la scheda che userai di più
- **Graphs**: una rappresentazione visiva dell'architettura del tuo modello, strato per strato
- **Histograms** e **Distributions**: come evolvono i pesi e le attivazioni di ogni strato durante l'addestramento
- **Embeddings / Projector**: proietta dati di alta dimensione in uno spazio 3D, utile per esplorare come il tuo modello raggruppa esempi simili

## Aggiungi il callback di TensorBoard al tuo addestramento

Un callback è una funzione che Keras esegue automaticamente in momenti specifici dell'addestramento (alla fine di ogni epoch, per esempio). Il callback `TensorBoard` salva i dati di addestramento in una directory di log che TensorBoard legge poi:

```python
import time
from tensorflow.keras.callbacks import TensorBoard

# Un nome univoco per modello, con timestamp, evita che due addestramento
# si sovrascrivano a vicenda nella cartella di log
nome_modello = f"cnn-gatti-cani-{int(time.time())}"
tensorboard_callback = TensorBoard(log_dir=f'logs/{nome_modello}')

model.fit(
    X, y,
    batch_size=32,
    epochs=10,
    validation_split=0.1,
    callbacks=[tensorboard_callback],
)
```

**Dai sempre un nome ai tuoi modelli con un identificatore univoco** (qui, un timestamp). Se riusi lo stesso nome in diversi addestramento, TensorBoard mescola i dati di tutti loro nello stesso grafico e perdi la capacità di confrontare.

## Lancia TensorBoard

Da un terminale, nella directory dove si trova la cartella `logs/`:

```bash
tensorboard --logdir=logs
```

Questo comando avvia un server locale. Apri `http://localhost:6006/` nel tuo browser e vedrai i grafici aggiornarsi mentre il modello si addestra. Se lavori su Google Colab, usa invece:

```python
%load_ext tensorboard
%tensorboard --logdir logs
```

## Usa la validation loss per rilevare l'overfitting

Il grafico più importante della scheda Scalars è il confronto tra `loss` (addestramento) e `val_loss` (validazione). Un modello sano mostra entrambe le curve che scendono insieme. Se `loss` continua a scendere ma `val_loss` inizia a salire, il tuo modello sta memorizzando i dati di addestramento invece di imparare pattern generali: questo è overfitting, e è il segnale più chiaro che hai bisogno di più dati, regolarizzazione (come `Dropout`), o un modello più semplice.

## Riepilogo dei punti chiave

- TensorBoard visualizza in tempo reale l'evoluzione del tuo addestramento
- Il callback `TensorBoard(log_dir=...)` è l'unica cosa che devi aggiungere a `model.fit` per attivarlo
- Dai a ogni modello un nome univoco (per esempio, con un timestamp) per non mescolare log di addestramento diversi
- La scheda Scalars, confrontando `loss` contro `val_loss`, è il tuo primo strumento per rilevare l'overfitting
- Su Colab, usa la magic `%tensorboard --logdir logs` invece del comando di terminale

## Mettilo in pratica

Addestra di nuovo il classificatore di gatti e cani del capitolo precedente, questa volta con il callback di TensorBoard attivato e un nome di modello univoco. Lancia TensorBoard e localizza il punto esatto (quale epoch) in cui `val_loss` smette di scendere. **Cattura un'immagine di quel grafico** e annota, in una cella di testo, in quale epoch inizieresti a preoccuparti per l'overfitting se questo fosse un progetto reale.

---

[← Precedente: [Reti Neurali Convoluzionali](04_redes_convolucionales_cnn_IT.md) · [Indice](00_indice_IT.md) · Prossimo: [Ottimizza i tuoi modelli →](06_optimiza_hiperparametros_IT.md)]
