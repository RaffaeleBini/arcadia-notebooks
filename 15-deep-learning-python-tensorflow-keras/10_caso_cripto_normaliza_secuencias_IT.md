[← Precedente: [Caso pratico: definisci il tuo obiettivo](09_caso_cripto_define_objetivo_IT.md) · [Indice](00_indice_IT.md) · Successivo: [Bilancia i tuoi dati e allena il tuo modello LSTM finale →](11_caso_cripto_modelo_final_IT.md)]

# 10. Normalizza e costruisci sequenze temporali

## Perché ti importa

Hai già un dataset con prezzi, volumi e un'etichetta obiettivo. Ma una rete LSTM non riceve una tabella di righe indipendenti: riceve sequenze, finestre di più passi consecutivi nel tempo. E prima di costruire queste finestre, devi risolvere un problema che non esisteva nei capitoli precedenti: come separare un set di validazione onesto quando i tuoi dati hanno ordine temporale.

## Perché non puoi mescolare dati temporali a caso

Nel capitolo 3, hai mescolato (shuffle) le tue immagini di gatti e cani prima di separare allenamento e validazione, senza problemi: una foto di gatto non dipende dalla foto precedente. Con serie temporali, questo sarebbe un grave errore. Se mescoli a caso e separi dopo, è molto probabile che i dati di validazione rimangano intercalati tra i dati di allenamento nel tempo. Il tuo modello starebbe, in pratica, imparando con informazioni che includono frammenti del "futuro" rispetto ad alcuni punti di validazione, e questo gonfia artificialmente le prestazioni che vedi, senza che si sostengano con dati realmente nuovi.

La soluzione è separare per tempo, non a caso: ti mantieni con l'ultimo tratto cronologico (ad esempio, l'ultimo 5% delle date) come set di validazione, e tutto il resto come allenamento:

```python
tiempos = sorted(main_df.index.values)
umbral_validacion = tiempos[-int(0.05 * len(tiempos))]

df_validacion = main_df[main_df.index >= umbral_validacion]
df_entrenamiento = main_df[main_df.index < umbral_validacion]
```

## Normalizza ogni colonna con cambio percentuale

Le quattro criptovalute hanno scale di prezzo molto diverse (Bitcoin costa migliaia di dollari, Litecoin costa molto meno). Se dai questi valori grezzi alla rete, le colonne con numeri più grandi domineranno l'allenamento senza una vera ragione. La soluzione è normalizzare ogni colonna con il cambio percentuale rispetto al passo precedente, e poi scalarlo:

```python
from sklearn import preprocessing

def preprocesa_df(df):
    df = df.copy()
    for columna in df.columns:
        if columna != "target":
            # pct_change(): quanto è cambiato ogni valore rispetto al precedente, in proporzione
            df[columna] = df[columna].pct_change()
            df.dropna(inplace=True)
            # scale(): centra e scala i valori affinché abbiano una distribuzione comparabile
            df[columna] = preprocessing.scale(df[columna].values)

    df.dropna(inplace=True)
    return df

df_entrenamiento = preprocesa_df(df_entrenamiento)
df_validacion = preprocesa_df(df_validacion)
```

`pct_change()` converte "il prezzo è passato da 100 a 105" in "0.05" (è salito del 5%), indipendentemente dal fatto che il prezzo originale fosse 100 o 100.000. Questo mette le quattro criptovalute sullo stesso piano.

## Costruisci sequenze con una finestra scorrevole

Ora converti il DataFrame in sequenze: ogni campione di input è una finestra di `SEQ_LEN` passi consecutivi, e la sua etichetta è il `target` corrispondente all'ultimo passo di quella finestra. `collections.deque` con `maxlen` fisso è lo strumento perfetto per questo: quando aggiungi un elemento ed è già pieno, scarta automaticamente il più vecchio.

```python
from collections import deque
import numpy as np

SEQ_LEN = 60  # 60 passi di storia per ogni previsione

def crea_secuencias(df):
    datos_secuenciales = []
    ventana = deque(maxlen=SEQ_LEN)

    # Tutte le colonne tranne "target" sono caratteristiche di input
    columnas_features = [c for c in df.columns if c != "target"]

    for fila in df[columnas_features + ["target"]].values:
        ventana.append(fila[:-1])  # tutte le colonne tranne l'ultima (target)
        if len(ventana) == SEQ_LEN:
            datos_secuenciales.append([np.array(ventana), fila[-1]])

    random.shuffle(datos_secuenciales)
    return datos_secuenciales

import random

secuencias_entrenamiento = crea_secuencias(df_entrenamiento)
secuencias_validacion = crea_secuencias(df_validacion)

print(f"Secuencias de entrenamiento: {len(secuencias_entrenamiento)}")
print(f"Secuencias de validación: {len(secuencias_validacion)}")
```

Mescoli (`shuffle`) le sequenze già costruite, non le righe originali: l'ordine interno di ogni finestra di 60 passi rimane intatto, cambia solo l'ordine in cui le diverse finestre vengono presentate al modello durante l'allenamento.

## Riepilogo dei punti chiave

- Con serie temporali, separa allenamento e validazione per tempo, mai mescolando a caso prima di separare
- `pct_change()` normalizza asset con scale di prezzo molto diverse a una base comparabile
- `preprocessing.scale()` centra e scala ogni colonna già convertita in cambio percentuale
- `collections.deque(maxlen=SEQ_LEN)` costruisce finestre scorrevoli di dimensione fissa in modo efficiente
- Mescola le sequenze già costruite, mai le righe grezze prima di formare le finestre

## Mettilo in pratica

Esegui la pipeline di questo capitolo con `SEQ_LEN = 60` e verifica la forma del primo elemento di `secuencias_entrenamiento`: deve essere una tupla dove il primo elemento ha forma `(60, numero_di_colonne)` e il secondo è uno 0 o un 1. **Cambia `SEQ_LEN` a 30** e rigenerato le sequenze: quante sequenze di allenamento ottieni ora, più o meno che con 60? Ragiona perché.

---

[← Precedente: [Caso pratico: definisci il tuo obiettivo](09_caso_cripto_define_objetivo_IT.md) · [Indice](00_indice_IT.md) · Successivo: [Bilancia i tuoi dati e allena il tuo modello LSTM finale →](11_caso_cripto_modelo_final_IT.md)]
