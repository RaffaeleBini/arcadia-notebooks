# Datos del curso / Dati del corso

## Español

`churn.csv` es el conjunto de datos principal del curso, **sintético y original**, escrito para este curso (sin datos reales ni personales). Simula el abandono de clientes (*churn*) de un servicio de suscripción, y se usa como hilo conductor desde la clase 3 hasta la clase 11: cada clase lo versiona, lo entrena, lo prueba, lo sirve o lo vigila un poco más.

| Columna | Qué es |
|---|---|
| `id_cliente` | Identificador sintético del cliente (`C100000`...). |
| `antiguedad_meses` | Meses que lleva el cliente suscrito (1 a 60). |
| `plan` | `basico`, `estandar` o `premium`. |
| `contrato` | `mensual` o `anual`. |
| `uso_mensual_horas` | Horas de uso del servicio en el último mes. |
| `tickets_soporte` | Número de tickets de soporte abiertos en el último mes. |
| `metodo_pago` | `tarjeta`, `domiciliacion` o `transferencia`. |
| `factura_mensual` | Importe de la factura mensual, en euros. |
| `abandono` | Etiqueta objetivo: `1` si el cliente abandona, `0` si no. |

**Cómo se generó:** 3 000 filas, con una semilla fija (42) y una relación real (no arbitraria) entre las columnas y la etiqueta: la antigüedad y el uso mensual reducen la probabilidad de abandono; el contrato mensual, los tickets de soporte y el plan básico la aumentan; el plan premium la reduce un poco. Se añade ruido aleatorio antes de decidir la etiqueta, así que la relación no es perfecta, como en un caso real.

**Cifras de referencia** (con una `Pipeline` de `ColumnTransformer` + regresión logística, validación cruzada de 5 particiones, *ejemplo ilustrativo*, se recalculan en la clase 4):
- Proporción de abandono: **11,7 %** (dataset desbalanceado a propósito, como en un caso real).
- Exactitud de un modelo sin ajustar el desbalanceo: 0,883 — **igual que la exactitud de predecir siempre "no abandona"**: la exactitud sola engaña aquí.
- F1 del mismo modelo sin ajustar: ≈0,02 (casi no predice ningún abandono).
- AUC (que no depende del umbral): ≈0,71 — hay señal real, aunque no sea muy fuerte.
- F1 con `class_weight="balanced"` y umbral 0,5: ≈0,31.

Esto no es un error del dataset: es intencionado. Sirve para que la clase 4 (y las que la usan después) tengan un ejemplo real de por qué la exactitud sola no basta con clases desbalanceadas, sin tener que fabricar el problema con un discurso abstracto.

`churn_v2.csv` (usado en la clase 11) es una segunda versión del mismo dataset, con una distribución desplazada a propósito para poder medir un *data drift* real; se documenta aparte cuando se crea esa clase.

### Generación de `churn_v2.csv` (clase 11)

3 000 filas, semilla fija (42), reproducible. Las columnas `antiguedad_meses`, `plan`, `metodo_pago` y `factura_mensual` se remuestrean (con reemplazo, misma semilla) de la distribución empírica de `churn.csv`: son columnas de **control**, sin desplazar. Tres columnas SÍ se desplazan a propósito: `contrato` (81,3 % `mensual` frente al 61,2 % de `churn.csv`, generado con `rng.random(N) < 0.82`), `uso_mensual_horas` (remuestreada y luego multiplicada por 0,70 y desplazada -1,0 con ruido gaussiano, media resultante ≈11,5 h frente a ≈17,8 h) y `tickets_soporte` (remuestreada y luego incrementada con un `rng.poisson(1.1)`, media resultante ≈2,26 frente a ≈1,11). La etiqueta `abandono` no se copia de `churn.csv`: se genera con un modelo de referencia (una `Pipeline` `ColumnTransformer` + regresión logística `C=1,0`, sin balancear, entrenado sobre `churn.csv`) que estima `P(abandono|X)` para cada fila nueva, y luego se decide con ruido aleatorio (`rng.random(N) < probabilidad`) — la misma relación entre columnas y etiqueta que en `churn.csv`, aplicada a una distribución de entradas distinta: por construcción, esto es *data drift*, no *concept drift*. Tasa de abandono resultante: 22,5 % (frente al 11,7 % de `churn.csv`), una consecuencia esperada del desplazamiento, no un ajuste manual. Detalle completo, con las cifras de PSI y Kolmogorov-Smirnov, en `manual/11_drift_reentrenamiento.md`.

## Italiano

`churn.csv` è il dataset principale del corso, **sintetico e originale**, scritto per questo corso (senza dati reali né personali). Simula l'abbandono dei clienti (*churn*) di un servizio in abbonamento, e viene usato come filo conduttore dalla lezione 3 alla lezione 11: ogni lezione lo versiona, lo addestra, lo testa, lo serve o lo vigila un po' di più.

| Colonna | Cos'è |
|---|---|
| `id_cliente` | Identificativo sintetico del cliente (`C100000`...). |
| `antiguedad_meses` | Mesi da cui il cliente è abbonato (1-60). |
| `plan` | `basico`, `estandar` o `premium`. |
| `contrato` | `mensual` o `anual`. |
| `uso_mensual_horas` | Ore di utilizzo del servizio nell'ultimo mese. |
| `tickets_soporte` | Numero di ticket di supporto aperti nell'ultimo mese. |
| `metodo_pago` | `tarjeta`, `domiciliacion` o `transferencia`. |
| `factura_mensual` | Importo della fattura mensile, in euro. |
| `abandono` | Etichetta obiettivo: `1` se il cliente abbandona, `0` altrimenti. |

**Come è stato generato:** 3 000 righe, con un seme fisso (42) e una relazione reale (non arbitraria) tra le colonne e l'etichetta: l'anzianità e l'utilizzo mensile riducono la probabilità di abbandono; il contratto mensile, i ticket di supporto e il piano base la aumentano; il piano premium la riduce un po'. Si aggiunge rumore casuale prima di decidere l'etichetta, quindi la relazione non è perfetta, come in un caso reale.

**Cifre di riferimento** (con una `Pipeline` di `ColumnTransformer` + regressione logistica, validazione incrociata a 5 partizioni, *esempio illustrativo*, ricalcolate nella lezione 4):
- Proporzione di abbandono: **11,7 %** (dataset sbilanciato apposta, come in un caso reale).
- Accuratezza di un modello senza correggere lo sbilanciamento: 0,883 — **uguale all'accuratezza di predire sempre "non abbandona"**: l'accuratezza da sola inganna qui.
- F1 dello stesso modello senza correzioni: ≈0,02 (non predice quasi nessun abbandono).
- AUC (che non dipende dalla soglia): ≈0,71 — c'è un segnale reale, anche se non fortissimo.
- F1 con `class_weight="balanced"` e soglia 0,5: ≈0,31.

Non è un errore del dataset: è intenzionale. Serve perché la lezione 4 (e quelle che lo usano dopo) abbiano un esempio reale del perché l'accuratezza da sola non basta con classi sbilanciate, senza dover costruire il problema con un discorso astratto.

`churn_v2.csv` (usato nella lezione 11) è una seconda versione dello stesso dataset, con una distribuzione spostata apposta per poter misurare un *data drift* reale; verrà documentato a parte quando sarà creata quella lezione.

### Generazione di `churn_v2.csv` (lezione 11)

3 000 righe, seed fisso (42), riproducibile. Le colonne `antiguedad_meses`, `plan`, `metodo_pago` e `factura_mensual` vengono ricampionate (con reinserimento, stesso seed) dalla distribuzione empirica di `churn.csv`: sono colonne di **controllo**, non spostate. Tre colonne SONO spostate apposta: `contrato` (81,3 % `mensual` contro il 61,2 % di `churn.csv`, generato con `rng.random(N) < 0.82`), `uso_mensual_horas` (ricampionata e poi moltiplicata per 0,70 e spostata di -1,0 con rumore gaussiano, media risultante ≈11,5 h contro ≈17,8 h) e `tickets_soporte` (ricampionata e poi incrementata con un `rng.poisson(1.1)`, media risultante ≈2,26 contro ≈1,11). L'etichetta `abandono` non viene copiata da `churn.csv`: viene generata con un modello di riferimento (una `Pipeline` `ColumnTransformer` + regressione logistica `C=1,0`, senza bilanciare, addestrato su `churn.csv`) che stima `P(abandono|X)` per ogni nuova riga, e poi si decide con rumore casuale (`rng.random(N) < probabilità`) — la stessa relazione tra colonne ed etichetta di `churn.csv`, applicata a una distribuzione di ingresso diversa: per costruzione, questo è *data drift*, non *concept drift*. Tasso di abbandono risultante: 22,5 % (contro l'11,7 % di `churn.csv`), una conseguenza attesa dello spostamento, non un aggiustamento manuale. Dettaglio completo, con le cifre di PSI e Kolmogorov-Smirnov, in `manual/11_drift_reentrenamiento_IT.md`.
