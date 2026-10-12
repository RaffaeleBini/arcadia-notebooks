# Datos del curso / Dati del corso

## Español

Copias locales de los datasets públicos que usan los notebooks. Cada notebook carga primero el CSV local (si existe), después intenta descargar el dataset y, si no hay conexión, usa una alternativa sintética documentada. En Google Colab la carpeta `datos/` no existe: allí se usa la descarga.

| Archivo | Qué es | Origen | Usado en |
|---|---|---|---|
| `penguins.csv` | 344 pingüinos de tres especies (Palmer Penguins), con valores ausentes | Horst, Hill y Gorman (2020), datos de Gorman et al. (2014); descargado del repositorio de datos de `seaborn` | Clase 3 y demostración de la clase 14 |
| `california_housing.csv` | Muestra aleatoria de 5 000 distritos censales de California (semilla 42) | Pace y Barry (1997), en la versión de `scikit-learn` (`fetch_california_housing`) | Clases 10-12 y caso C de la clase 14 |
| `diabetes.csv` | 442 pacientes con 10 variables basales y una medida cuantitativa de la progresión de la enfermedad un año después (`target`) | Efron, Hastie, Johnstone y Tibshirani (2004), en la versión de `scikit-learn` (`load_diabetes(scaled=False)`, valores originales sin escalar) | Caso A de la clase 14 |

Los datos sintéticos del curso se generan dentro de cada notebook con semilla fija y no se guardan aquí.

## Italiano

Copie locali dei dataset pubblici usati dai notebook. Ogni notebook carica prima il CSV locale (se esiste), poi prova a scaricare il dataset e, senza connessione, usa un'alternativa sintetica documentata. In Google Colab la cartella `datos/` non esiste: lì si usa il download.

| File | Cos'è | Origine | Usato in |
|---|---|---|---|
| `penguins.csv` | 344 pinguini di tre specie (Palmer Penguins), con valori mancanti | Horst, Hill e Gorman (2020), dati di Gorman et al. (2014); scaricato dal repository dati di `seaborn` | Lezione 3 e dimostrazione della lezione 14 |
| `california_housing.csv` | Campione casuale di 5 000 distretti censuari della California (seme 42) | Pace e Barry (1997), nella versione di `scikit-learn` (`fetch_california_housing`) | Lezioni 10-12 e caso C della lezione 14 |
| `diabetes.csv` | 442 pazienti con 10 variabili di base e una misura quantitativa della progressione della malattia a un anno (`target`) | Efron, Hastie, Johnstone e Tibshirani (2004), nella versione di `scikit-learn` (`load_diabetes(scaled=False)`, valori originali non scalati) | Caso A della lezione 14 |

I dati sintetici del corso sono generati dentro ogni notebook con seme fisso e non sono salvati qui.
