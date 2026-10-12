[← Anterior: [Caso práctico: define tu objetivo](09_caso_cripto_define_objetivo.md) · [Índice](00_indice.md) · Siguiente: [Balancea tus datos y entrena tu modelo LSTM final →](11_caso_cripto_modelo_final.md)]

# 10. Normaliza y construye secuencias temporales

## Por qué te importa esto

Ya tienes un dataset con precios, volúmenes y una etiqueta objetivo. Pero una red LSTM no recibe una tabla de filas independientes: recibe secuencias, ventanas de varios pasos consecutivos de tiempo. Y antes de construir esas ventanas, tienes que resolver un problema que no existía en los capítulos anteriores: cómo separar un conjunto de validación honesto cuando tus datos tienen orden temporal.

## Por qué no puedes mezclar datos temporales al azar

En el capítulo 3, mezclaste (shuffle) tus imágenes de gatos y perros antes de separar entrenamiento y validación, sin ningún problema: una foto de gato no depende de la foto anterior. Con series temporales, eso sería un error grave. Si mezclas al azar y separas después, es muy probable que datos de validación queden intercalados entre datos de entrenamiento en el tiempo. Tu modelo estaría, en la práctica, aprendiendo con información que incluye fragmentos del "futuro" respecto a algunos puntos de validación, y eso infla artificialmente el rendimiento que ves, sin que se sostenga con datos realmente nuevos.

La solución es separar por tiempo, no al azar: te quedas con el último tramo cronológico (por ejemplo, el último 5% de las fechas) como conjunto de validación, y todo lo anterior como entrenamiento:

```python
tiempos = sorted(main_df.index.values)
umbral_validacion = tiempos[-int(0.05 * len(tiempos))]

df_validacion = main_df[main_df.index >= umbral_validacion]
df_entrenamiento = main_df[main_df.index < umbral_validacion]
```

## Normaliza cada columna con percent change

Las cuatro criptomonedas tienen escalas de precio muy distintas (Bitcoin cuesta miles de dólares, Litecoin cuesta mucho menos). Si alimentas esos valores en crudo a la red, las columnas con números más grandes dominarán el entrenamiento sin razón real. La solución es normalizar cada columna con el cambio porcentual respecto al paso anterior, y después escalarla:

```python
from sklearn import preprocessing

def preprocesa_df(df):
    df = df.copy()
    for columna in df.columns:
        if columna != "target":
            # pct_change(): cuánto cambió cada valor respecto al anterior, en proporción
            df[columna] = df[columna].pct_change()
            df.dropna(inplace=True)
            # scale(): centra y escala los valores para que tengan una distribución comparable
            df[columna] = preprocessing.scale(df[columna].values)

    df.dropna(inplace=True)
    return df

df_entrenamiento = preprocesa_df(df_entrenamiento)
df_validacion = preprocesa_df(df_validacion)
```

`pct_change()` convierte "el precio pasó de 100 a 105" en "0,05" (subió un 5%), sin importar si el precio original era 100 o 100.000. Eso pone a las cuatro criptomonedas en pie de igualdad.

## Construye secuencias con una ventana deslizante

Ahora conviertes el DataFrame en secuencias: cada muestra de entrada es una ventana de `SEQ_LEN` pasos consecutivos, y su etiqueta es el `target` correspondiente al último paso de esa ventana. `collections.deque` con `maxlen` fijo es la herramienta perfecta para esto: cuando añades un elemento y ya está lleno, descarta automáticamente el más antiguo.

```python
from collections import deque
import numpy as np

SEQ_LEN = 60  # 60 pasos de historia por cada predicción

def crea_secuencias(df):
    datos_secuenciales = []
    ventana = deque(maxlen=SEQ_LEN)

    # Todas las columnas salvo "target" son características de entrada
    columnas_features = [c for c in df.columns if c != "target"]

    for fila in df[columnas_features + ["target"]].values:
        ventana.append(fila[:-1])  # todas las columnas salvo la última (target)
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

Mezclas (`shuffle`) las secuencias ya construidas, no las filas originales: el orden interno de cada ventana de 60 pasos se mantiene intacto, solo cambia el orden en que las distintas ventanas se presentan al modelo durante el entrenamiento.

## Resumen de puntos clave

- Con series temporales, separa entrenamiento y validación por tiempo, nunca mezclando al azar antes de separar
- `pct_change()` normaliza activos con escalas de precio muy distintas a una base comparable
- `preprocessing.scale()` centra y escala cada columna ya convertida en cambio porcentual
- `collections.deque(maxlen=SEQ_LEN)` construye ventanas deslizantes de tamaño fijo de forma eficiente
- Mezcla las secuencias ya construidas, nunca las filas crudas antes de formar las ventanas

## Ponlo en práctica

Ejecuta el pipeline de este capítulo con `SEQ_LEN = 60` y comprueba la forma del primer elemento de `secuencias_entrenamiento`: debe ser una tupla donde el primer elemento tiene forma `(60, número_de_columnas)` y el segundo es un 0 o un 1. **Cambia `SEQ_LEN` a 30** y vuelve a generar las secuencias: ¿cuántas secuencias de entrenamiento obtienes ahora, más o menos que con 60? Razona por qué.

---

[← Anterior: [Caso práctico: define tu objetivo](09_caso_cripto_define_objetivo.md) · [Índice](00_indice.md) · Siguiente: [Balancea tus datos y entrena tu modelo LSTM final →](11_caso_cripto_modelo_final.md)]
