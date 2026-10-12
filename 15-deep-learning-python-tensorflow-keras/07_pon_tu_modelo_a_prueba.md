[← Anterior: [Optimiza tus modelos](06_optimiza_hiperparametros.md) · [Índice](00_indice.md) · Siguiente: [Redes Neuronales Recurrentes (RNN) →](08_redes_recurrentes_rnn.md)]

# 7. Pon tu modelo a prueba: predicciones sobre datos nuevos

## Por qué te importa esto

Un modelo que solo ha demostrado su valía sobre el conjunto de validación de su propio dataset no te dice toda la verdad. La pregunta que realmente importa es: ¿funciona con datos que nunca ha visto, tomados de un contexto distinto al de entrenamiento? Una foto que tú mismo tomes con tu móvil, no una imagen ya preparada del dataset. Esta es la prueba de fuego de cualquier modelo, y cierra el bloque de CNN y TensorBoard de este curso.

## Por qué el accuracy de validación no basta

El accuracy que ves durante el entrenamiento se calcula sobre imágenes que vienen del mismo dataset, procesadas con el mismo pipeline, capturadas probablemente en condiciones similares. Una imagen nueva y externa (out-of-sample) puede tener una iluminación distinta, un encuadre distinto, un fondo distinto. Si tu modelo solo aprendió a reconocer las particularidades del dataset de entrenamiento en vez del concepto real ("gato" o "perro"), fallará estrepitosamente con imágenes nuevas. Esto es, en esencia, una forma más exigente de detectar overfitting.

## Prepara una imagen externa igual que preparaste el dataset

La regla de oro es esta: cualquier imagen nueva debe pasar exactamente por el mismo preprocesamiento que aplicaste a tus datos de entrenamiento. Si entrenaste con imágenes en escala de grises de 50x50 píxeles, tu imagen nueva también debe llegar al modelo en escala de grises y de 50x50 píxeles:

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 50
categorias = ["Dog", "Cat"]

def preparar_imagen(ruta_archivo):
    array_imagen = cv2.imread(ruta_archivo, cv2.IMREAD_GRAYSCALE)
    array_redimensionado = cv2.resize(array_imagen, (IMG_SIZE, IMG_SIZE))
    # Misma normalización que usaste al entrenar, y la misma forma: (1, 50, 50, 1)
    return array_redimensionado.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

modelo = load_model('mi_clasificador_gatos_perros.keras')

imagen_preparada = preparar_imagen("mi_foto_de_prueba.jpg")
prediccion = modelo.predict(imagen_preparada)

# Con salida sigmoid (una neurona), redondeas: menor de 0.5 es la clase 0, mayor o igual es la clase 1
clase_predicha = categorias[int(round(prediccion[0][0]))]
print(f"El modelo dice: {clase_predicha}")
```

## Por qué la diversidad del dataset de entrenamiento importa tanto

Si tu dataset de entrenamiento solo tiene fotos de gatos blancos de raza persa, tu modelo aprenderá a reconocer eso, no el concepto general de "gato". Este problema no es exclusivo de fotos de mascotas: es la misma razón por la que un sistema de reconocimiento facial entrenado con un grupo demográfico limitado falla al aplicarse a personas fuera de ese grupo. Cuanto más diverso y representativo sea tu dataset de entrenamiento, mejor generaliza tu modelo a situaciones nuevas.

## Resumen de puntos clave

- El accuracy de validación no garantiza que tu modelo funcione con datos verdaderamente nuevos y externos
- Cualquier dato nuevo debe pasar por exactamente el mismo preprocesamiento (tamaño, escala de color, normalización) que tus datos de entrenamiento
- Probar con datos out-of-sample es una forma exigente de detectar overfitting que el conjunto de validación interno no siempre revela
- Un dataset de entrenamiento poco diverso limita la capacidad de generalización del modelo, con consecuencias que van desde errores triviales hasta sesgos serios en aplicaciones sensibles

## Ponlo en práctica

Toma tres fotos con tu propio móvil (o descarga tres imágenes de gatos o perros que no formen parte del dataset original) y pásalas por la función `preparar_imagen` de este capítulo junto con tu modelo ya entrenado del capítulo 4. **Anota el resultado de cada predicción** y, si el modelo se equivoca en alguna, revisa la imagen: ¿tiene una iluminación, un ángulo o un fondo muy distinto a las fotos del dataset de entrenamiento?

---

[← Anterior: [Optimiza tus modelos](06_optimiza_hiperparametros.md) · [Índice](00_indice.md) · Siguiente: [Redes Neuronales Recurrentes (RNN) →](08_redes_recurrentes_rnn.md)]
