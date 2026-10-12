[← Precedente: [Metti il tuo modello alla prova](07_pon_tu_modelo_a_prueba_IT.md) · [Indice](00_indice_IT.md) · Successivo: [Caso pratico: definisci il tuo obiettivo con dati di criptovalute →](09_caso_cripto_define_objetivo_IT.md)]

# 8. Reti Neurali Ricorrenti (RNN): memoria per dati sequenziali

## Perché ti importa

Una CNN è eccellente nel riconoscere pattern spaziali in un'immagine fissa, ma non ha alcun concetto di "ordine nel tempo". Cosa succede quando quello che vuoi prevedere dipende da una sequenza, come il prezzo di un'azione nel tempo, le parole di una frase o l'audio di una conversazione? Ecco dove entrano le Reti Neurali Ricorrenti (RNN, Recurrent Neural Network): reti progettate specificamente per dati dove l'ordine conta e il passato influisce sul presente. Questo capitolo apre il blocco finale del corso, dove applichi RNN a un caso reale: prevedere il movimento del prezzo di una criptovaluta.

## Cosa rende "ricorrente" una RNN

Una capa `Dense` o `Conv2D` elabora ogni input in isolamento, senza memoria di ciò che è venuto prima. Una RNN, invece, mantiene uno "stato nascosto" che si aggiorna ad ogni passo della sequenza e si passa al passo successivo. È come leggere un romanzo: al raggiungimento della pagina 50, la tua comprensione della trama non dipende solo da quella pagina, dipende da tutto quello che hai letto prima. Questo "ricordo accumulato" è esattamente ciò che lo stato nascosto di una RNN rappresenta.

## Il problema del vanishing gradient

Le RNN più semplici (talvolta chiamate "reti Elman") hanno un serio problema pratico: quando la sequenza è lunga, l'informazione dai primi passi si diluisce fino a scomparire prima di raggiungere gli ultimi passi. Questo è noto come il problema del vanishing gradient (gradiente che svanisce), e in pratica significa che una RNN semplice "dimentica" rapidamente, essendo incapace di imparare dipendenze a lungo termine.

## LSTM: memoria a breve e lungo termine

LSTM (Long Short-Term Memory) risolve questo problema con un'architettura di porte (gates) che controllano quali informazioni salvare, quale dimenticare e quale lasciare passare all'output ad ogni passo:

- **Porta di dimenticanza (forget gate)**: decide quale parte dello stato precedente scartare
- **Porta di ingresso (input gate)**: decide quale nuova informazione incorporare
- **Porta di uscita (output gate)**: decide quale parte dello stato attuale esporre come output

Grazie a queste porte, un LSTM può mantenere informazioni rilevanti durante molti passi della sequenza e scartare quello che non importa più. È l'architettura ricorrente più usata in pratica, e quella che userai nel caso delle criptovalute.

## GRU: un'alternativa più leggera

GRU (Gated Recurrent Unit) semplifica l'idea di LSTM combinando alcune delle sue porte in solo due (porta di aggiornamento e porta di reset). Con meno parametri da allenare, GRU è solitamente più veloce e talvolta funziona altrettanto bene di LSTM, anche se LSTM rimane l'opzione più robusta quando le dipendenze a lungo termine sono critiche.

## RNN unidirezionali e bidirezionali

Una RNN unidirezionale elabora la sequenza in una sola direzione (normalmente dall'inizio alla fine). Una RNN bidirezionale la elabora in entrambe le direzioni contemporaneamente e combina l'informazione: utile quando il contesto completo (passato e futuro all'interno della sequenza) è disponibile in anticipo, come nell'analizzare un testo già scritto completamente. Per prevedere il futuro dal passato (come nel caso delle criptovalute), usi la variante unidirezionale: nel momento della previsione, il "futuro" della sequenza non esiste ancora.

## Un primo esempio con la capa LSTM di Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    # 128 unidades LSTM. return_sequences=True porque apilas otra capa recurrente encima
    LSTM(128, input_shape=(timesteps, num_caracteristicas), return_sequences=True),
    Dropout(0.2),
    # Última capa recurrente: return_sequences=False (por defecto) porque ya solo necesitas el resultado final
    LSTM(128),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

`return_sequences=True` dice al livello di restituire il suo output ad ogni passo temporale, non solo l'ultimo, perché il prossimo livello LSTM ha bisogno di una sequenza completa come input.

## MNIST come sequenza: un esempio didattico

Puoi praticare RNN anche senza dati temporali "reali": trattando ogni immagine MNIST come una sequenza di 28 righe, ciascuna con 28 valori. È un esercizio artificiale, ma ti permette di confrontare direttamente le prestazioni di una RNN rispetto alla rete densa del capitolo 2 sullo stesso problema:

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train ya tiene forma (60000, 28, 28): 60.000 secuencias de 28 pasos, con 28 valores cada uno

model = Sequential([
    LSTM(128, input_shape=(28, 28), return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
```

## Riepilogo dei punti chiave

- Una RNN mantiene uno stato nascosto che si aggiorna ad ogni passo della sequenza, dando "memoria" del passato
- Il vanishing gradient fa sì che le RNN semplici dimentichino rapidamente in sequenze lunghe
- LSTM usa porte di dimenticanza, ingresso e uscita per mantenere informazioni rilevanti più a lungo
- GRU semplifica LSTM con meno porte, al costo di una certa capacità di modellamento
- `return_sequences=True` è necessario quando impili un livello ricorrente sopra un altro
- Per prevedere il futuro dal passato, usa RNN unidirezionali

## Mettilo in pratica

Riproduci l'esempio di MNIST come sequenza di questo capitolo e allena il modello LSTM per 3 epoch. Confronta il `val_accuracy` finale con quello che hai ottenuto con la rete densa del capitolo 2 con lo stesso numero di epoch. **Scrivi un'ipotesi**: perché credi che un'architettura pensata per sequenze temporali ottenga quel risultato in un problema che, in realtà, non ha una struttura temporale reale?

---

[← Precedente: [Metti il tuo modello alla prova](07_pon_tu_modelo_a_prueba_IT.md) · [Indice](00_indice_IT.md) · Successivo: [Caso pratico: definisci il tuo obiettivo con dati di criptovalute →](09_caso_cripto_define_objetivo_IT.md)]
