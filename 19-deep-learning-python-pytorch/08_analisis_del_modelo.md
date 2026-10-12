# 08. Análisis del modelo

## Por qué importa este tema

Ya sabes construir una red neuronal (NN, Neural Network) con PyTorch,
entrenarla y, si tienes una GPU (Graphics Processing Unit) disponible,
acelerar ese entrenamiento. Pero hay una pregunta que todavía no has
respondido: ¿ese modelo que acabas de entrenar es bueno de verdad?

La pérdida (loss, el error que comete el modelo) baja durante el
entrenamiento. Eso se ve bien en la pantalla, pero no te dice lo que
realmente necesitas saber: cómo se comporta el modelo con datos que nunca ha
visto. Un modelo puede tener una pérdida de entrenamiento excelente y ser
inútil en la práctica, porque lo que ha aprendido no es la relación general
entre los datos y el resultado, sino los detalles concretos de tu conjunto de
entrenamiento.

En este capítulo aprendes a mirar un modelo entrenado con ojo crítico: cómo
distinguir un modelo que generaliza de uno que memoriza, qué métricas usar
más allá de la pérdida, cómo detectar en qué se equivoca exactamente y, por
último, cómo guardar tu trabajo con `torch.save` para no tener que repetir el
entrenamiento cada vez que quieras usar el modelo.

## Sobreajuste e infra-ajuste: dos formas de fallar

Imagina a dos personas preparando el mismo examen tipo test con un banco de
100 preguntas de práctica.

La primera se aprende de memoria la respuesta exacta de cada una de las 100
preguntas, sin entender por qué esa respuesta es la correcta. El día del
examen real, con preguntas parecidas pero no idénticas, falla casi todo:
memorizó el banco de preguntas en lugar de aprender la materia. Esto es
**sobreajuste** (overfitting): el modelo aprende tan bien los datos de
entrenamiento —incluidas sus particularidades y su ruido— que pierde la
capacidad de generalizar a datos nuevos.

La segunda persona apenas ha abierto el libro: no domina ni siquiera las
preguntas de práctica. El día del examen falla tanto en las preguntas
conocidas como en las nuevas. Esto es **infra-ajuste** (underfitting): el
modelo es demasiado simple, o no se ha entrenado lo suficiente, para captar
los patrones que existen en los datos.

### Datos "dentro de muestra" y "fuera de muestra"

Para diagnosticar cuál de los dos problemas tienes, necesitas dos versiones
de las mismas dos métricas: la pérdida y la precisión (accuracy, el
porcentaje de predicciones correctas).

- **Dentro de muestra (in-sample):** la pérdida y la precisión calculadas
  sobre los datos de entrenamiento, los mismos que el modelo usa para
  ajustar sus pesos
- **Fuera de muestra (out-of-sample):** la pérdida y la precisión calculadas
  sobre datos que el modelo no ha usado para entrenar —un conjunto de
  validación o de test—

Comparar estos cuatro valores en cada época (una época es una pasada
completa por todo el conjunto de entrenamiento) es la forma más directa de
saber qué te pasa:

- **Todo va bien:** la pérdida dentro y fuera de muestra bajan juntas y se
  estabilizan a un nivel bajo, con una diferencia pequeña entre ellas
- **Sobreajuste:** la pérdida dentro de muestra sigue bajando, pero la de
  fuera de muestra deja de bajar y empieza a subir. A partir de ese punto el
  modelo ha dejado de aprender patrones generales y ha empezado a memorizar
  el ruido de los datos de entrenamiento
- **Infra-ajuste:** las dos pérdidas se quedan altas y apenas bajan, incluso
  después de muchas épocas. El modelo no tiene capacidad suficiente, o no ha
  entrenado lo bastante, para aprender la relación entre los datos y el
  resultado que quieres predecir

Para calcular la pérdida y la precisión fuera de muestra sin gastar memoria
ni tiempo en algo que no vas a usar, envuelve ese cálculo en
`torch.no_grad()`: le dices a PyTorch que no necesitas calcular gradientes,
porque no vas a entrenar con esos datos.

### Por qué la precisión de test importa más que la de entrenamiento

Es tentador fijarte solo en la precisión de entrenamiento, porque suele ser
la más alta y la que "mejor se ve". Pero esa cifra no te dice nada sobre cómo
se comportará tu modelo el día que reciba datos reales, distintos a los que
ya conoce. La precisión sobre un conjunto de test —datos que el modelo no ha
visto ni una sola vez durante el entrenamiento— es la única cifra que se
parece a lo que vas a obtener en producción. Si un modelo tiene 99% de
precisión en entrenamiento y 70% en test, ese 70% es la cifra que describe tu
modelo, no el 99%.

## La matriz de confusión: más allá del porcentaje de acierto

La precisión te da un único número, pero esconde información valiosa. Un
modelo con 90% de precisión sobre diez categorías puede estar acertando casi
siempre en nueve de ellas y equivocándose sistemáticamente en la décima —y la
precisión sola no te lo dice—.

La **matriz de confusión** resuelve ese problema: es una tabla donde cada
fila representa la categoría real de un ejemplo y cada columna representa la
categoría que predijo el modelo. En la diagonal principal quedan los
aciertos; fuera de la diagonal, los errores. Si entrenas un modelo para
reconocer dígitos escritos a mano sobre el dataset MNIST (Modified National
Institute of Standards and Technology, un banco de imágenes de dígitos del 0
al 9 muy usado para practicar clasificación de imágenes), la matriz de
confusión te puede revelar, por ejemplo, que el modelo confunde con
frecuencia el dígito cuatro con el nueve, pero casi nunca confunde el cero
con ningún otro número. Esa información concreta —qué categorías se
confunden entre sí— es mucho más útil para decidir cómo mejorar el modelo que
un único porcentaje global.

## Ejemplo de código comentado

El código siguiente da por hecho que ya tienes un modelo `net` entrenado
(una instancia de una clase que hereda de `nn.Module`) y un `DataLoader`
llamado `test_loader` con el conjunto de test. Primero calculamos la
precisión y la matriz de confusión; después inspeccionamos algunas
predicciones incorrectas; por último, guardamos y recuperamos el modelo.

```python
import torch

NUM_CLASES = 10  # los diez dígitos del 0 al 9

# Contamos aciertos y llevamos la cuenta de en qué se confunde el modelo.
# Empezamos con una matriz de ceros de NUM_CLASES x NUM_CLASES.
matriz_confusion = torch.zeros(NUM_CLASES, NUM_CLASES, dtype=torch.int64)
aciertos = 0
total = 0

# torch.no_grad() evita calcular gradientes: solo estamos evaluando, no
# entrenando, así que ahorramos memoria y tiempo de cómputo
net.eval()  # desactiva capas como dropout, si el modelo las tiene
with torch.no_grad():
    for imagenes, etiquetas in test_loader:
        salidas = net(imagenes)

        # La salida tiene un valor por cada una de las diez clases;
        # torch.argmax se queda con el índice del valor más alto, que es
        # la clase que el modelo considera más probable
        predicciones = torch.argmax(salidas, dim=1)

        aciertos += (predicciones == etiquetas).sum().item()
        total += etiquetas.size(0)

        # Actualizamos la matriz de confusión: una fila por etiqueta real,
        # una columna por etiqueta predicha
        for etiqueta_real, etiqueta_predicha in zip(etiquetas, predicciones):
            matriz_confusion[etiqueta_real, etiqueta_predicha] += 1

precision = aciertos / total
print(f"Precisión sobre el conjunto de test: {precision:.2%}")
print("Matriz de confusión (filas = real, columnas = predicho):")
print(matriz_confusion)
```

### Inspeccionar predicciones incorrectas

Saber que el modelo se equivoca un 8% de las veces es útil, pero ver
ejemplos concretos de esos errores te ayuda a entender por qué pasa:

```python
# Repetimos la evaluación, pero esta vez guardamos los índices de los
# ejemplos donde la predicción no coincide con la etiqueta real
indices_erroneos = []

net.eval()
with torch.no_grad():
    for lote, (imagenes, etiquetas) in enumerate(test_loader):
        salidas = net(imagenes)
        predicciones = torch.argmax(salidas, dim=1)

        fallos = (predicciones != etiquetas).nonzero(as_tuple=True)[0]
        for indice in fallos:
            indices_erroneos.append({
                "lote": lote,
                "posicion": indice.item(),
                "etiqueta_real": etiquetas[indice].item(),
                "etiqueta_predicha": predicciones[indice].item(),
            })

print(f"Número de predicciones incorrectas: {len(indices_erroneos)}")
print("Los tres primeros errores:")
for error in indices_erroneos[:3]:
    print(f"  Real: {error['etiqueta_real']}  →  "
          f"Predicho: {error['etiqueta_predicha']}")
```

Con esos índices puedes recuperar las imágenes originales y dibujarlas con
Matplotlib para ver, literalmente, qué dígitos confunde tu modelo y si el
error es comprensible (un siete mal escrito que parece un uno) o
sorprendente (una imagen perfectamente clara clasificada mal).

### Guardar y cargar el modelo con `torch.save` y `torch.load`

Entrenar un modelo puede llevar minutos u horas. Guardarlo te permite
reutilizarlo sin repetir ese trabajo, compartirlo con otra persona o
desplegarlo en otro entorno. En PyTorch, la forma recomendada es guardar solo
los pesos del modelo —lo que se llama su `state_dict`, un diccionario con el
valor de cada parámetro aprendido— y no el objeto completo:

```python
ruta_modelo = "mi_modelo.pt"

# Guardamos únicamente los pesos aprendidos
torch.save(net.state_dict(), ruta_modelo)
```

Para recuperarlo más tarde, necesitas reconstruir un modelo con exactamente
la misma arquitectura y cargar en él los pesos guardados:

```python
# Reconstruimos la misma arquitectura, sin entrenar
modelo_restaurado = MiRedNeuronal()

# Cargamos los pesos guardados dentro de esa arquitectura
modelo_restaurado.load_state_dict(torch.load(ruta_modelo))
modelo_restaurado.eval()  # lo dejamos listo para predecir, no para entrenar

# Comprobamos que predice igual que el modelo original
with torch.no_grad():
    salida_original = net(imagenes_de_prueba)
    salida_restaurada = modelo_restaurado(imagenes_de_prueba)
    print("¿Predicciones idénticas?",
          torch.equal(torch.argmax(salida_original, dim=1),
                      torch.argmax(salida_restaurada, dim=1)))
```

Este enfoque de guardar solo el `state_dict` es el que recomienda la
documentación oficial de PyTorch, porque es más portable: el archivo
guardado solo contiene números, no código de la clase del modelo. Si más
adelante entrenas en una GPU y quieres cargar el modelo en un ordenador que
solo tiene CPU (Central Processing Unit), añade el argumento
`map_location=torch.device("cpu")` a `torch.load()` para que los pesos se
coloquen en el dispositivo correcto.

## Resumen de puntos clave

- El sobreajuste ocurre cuando la pérdida dentro de muestra sigue bajando
  mientras la de fuera de muestra se estanca o sube: el modelo memoriza en
  lugar de generalizar
- El infra-ajuste ocurre cuando ambas pérdidas se quedan altas: el modelo no
  tiene capacidad o entrenamiento suficiente
- Comparar la pérdida y la precisión dentro y fuera de muestra en cada época
  es la forma más directa de diagnosticar cuál de los dos problemas tienes
- La precisión de test importa más que la de entrenamiento, porque es la
  única cifra que se parece a lo que el modelo hará con datos reales
- La matriz de confusión muestra en qué categorías concretas se equivoca el
  modelo, algo que un único porcentaje de precisión no revela
- `torch.save(modelo.state_dict(), ruta)` y
  `modelo.load_state_dict(torch.load(ruta))` son la forma recomendada de
  guardar y recuperar los pesos de un modelo, siempre reconstruyendo antes la
  misma arquitectura

## Ponlo en práctica

Coge el modelo que entrenaste en un capítulo anterior de este curso (por
ejemplo, la CNN del capítulo 6) y haz lo siguiente:

1. Evalúa el modelo sobre el conjunto de test y calcula su precisión y su
   matriz de confusión. ¿Hay alguna categoría donde el modelo falle mucho más
   que en las demás?
2. Busca cinco ejemplos donde el modelo se equivoque e inspecciona las
   imágenes originales. ¿Los errores son comprensibles (imágenes ambiguas) o
   te sorprenden?
3. Guarda los pesos del modelo con `torch.save`. En una celda o script
   aparte, reconstruye la arquitectura desde cero, carga esos pesos con
   `load_state_dict` y comprueba que la precisión sobre el mismo conjunto de
   test coincide con la del modelo original

Si quieres seguir practicando de forma guiada, abre el notebook de práctica
de esta clase: `clase08_practica_analisis_del_modelo.ipynb`.
