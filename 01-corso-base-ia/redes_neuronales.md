# Introducción a las Redes Neuronales
### Material para clase introductoria (~1 hora) — Sin conocimientos previos de matemáticas avanzadas

---

## Objetivo de la clase

Al final de esta clase, los estudiantes podrán:
- Explicar qué es una neurona artificial y cómo "decide"
- Entender cómo se conectan las neuronas para formar una red
- Comprender la diferencia entre una red simple y una red compleja
- Saber qué es el Deep Learning y por qué es importante hoy en día

---

## 1. La neurona artificial

### La idea de partida: el cerebro humano

Nuestro cerebro está formado por miles de millones de células llamadas **neuronas**. Cada neurona recibe señales de otras neuronas, las "procesa" y decide si transmite o no esa señal hacia adelante.

Las redes neuronales artificiales se inspiran en esta idea, pero de forma mucho más simple: son un modelo matemático que imita, de manera muy básica, cómo una neurona toma decisiones.

### ¿Cómo "decide" una neurona artificial?

Imagina que estás decidiendo si **salir a caminar** hoy. Tu decisión depende de varios factores:

| Factor | ¿Importa mucho? | Situación hoy |
|---|---|---|
| ¿Hace buen tiempo? | Mucho | Sí |
| ¿Tienes tiempo libre? | Mucho | Sí |
| ¿Hay mucho tráfico? | Poco | Sí |

Tú no le das el mismo peso a cada factor. El **clima** y el **tiempo libre** te importan mucho más que el tráfico. Al final, sumas mentalmente la importancia de cada factor y tomas una decisión: salir o no salir.

Una neurona artificial hace exactamente esto:

1. **Recibe varias entradas** (datos: por ejemplo, "temperatura", "tiempo libre", "tráfico")
2. **Da una importancia (peso) a cada entrada** — algunas pesan más que otras
3. **Suma todo** para obtener un valor total
4. **Aplica una función de activación**: un "filtro" que decide si esa suma es suficiente para "activarse" (decir "sí, salgo") o no

> **Metáfora clave:** la función de activación es como un interruptor de luz. Si la energía acumulada supera cierto umbral, la luz se enciende (la neurona se activa); si no, queda apagada.

### En resumen
Una neurona artificial = **entradas + pesos + suma + filtro de decisión = salida**

---

## 2. Red neuronal simple

### De una neurona a un equipo

Una sola neurona puede tomar decisiones muy simples, pero el mundo real es más complicado. Por eso conectamos muchas neuronas entre sí, formando una **red neuronal**.

> **Metáfora:** piensa en una empresa pequeña. Hay personas que reciben la información del cliente (entrada), personas que la analizan y discuten internamente (capa oculta), y una persona que da la respuesta final al cliente (salida).

### Las tres partes de una red simple

1. **Capa de entrada**: recibe los datos originales (por ejemplo, características de una fruta: color, tamaño, forma)
2. **Capa oculta**: un grupo de neuronas que procesan esa información y buscan patrones
3. **Capa de salida**: da el resultado final (por ejemplo, "es una manzana" o "es una naranja")

### Ejemplo práctico: clasificar frutas

Imagina que queremos que la red distinga entre una **manzana** y una **naranja**.

- **Entradas**: color (rojo/naranja), forma (redonda/ovalada), tamaño
- **Capa oculta**: combina estas características y busca patrones ("si es rojo Y redondo, probablemente es manzana")
- **Salida**: la red da su respuesta final: "manzana" o "naranja"

### ¿Cómo "aprende" la red?

Al principio, la red no sabe nada: los pesos (la importancia de cada entrada) son aleatorios. Le mostramos muchos ejemplos de frutas ya etiquetadas ("esto es una manzana", "esto es una naranja").

Cada vez que la red se equivoca, ajusta ligeramente sus pesos para acercarse más a la respuesta correcta. Repitiendo este proceso miles de veces, la red mejora.

> **Metáfora:** es como aprender a lanzar dardos a una diana con los ojos vendados. Cada lanzamiento fallido te da información ("muy a la izquierda", "muy abajo") y vas corrigiendo poco a poco hasta acertar más seguido.

---

## 3. Red neuronal compleja

### ¿Por qué necesitamos más capas y más neuronas?

Distinguir manzanas de naranjas es un problema sencillo. Pero, ¿qué pasa si queremos que la red reconozca una **cara humana** en una foto? Esto es mucho más complejo.

Para problemas difíciles, usamos redes con **muchas capas ocultas** y **muchas neuronas en cada capa**.

### La idea de jerarquía: de lo simple a lo complejo

> **Metáfora de las piezas de Lego:** para construir una casa de Lego, primero conectas piezas pequeñas para formar paredes, luego combinas paredes para formar habitaciones, y finalmente combinas habitaciones para formar la casa completa.

Una red neuronal compleja funciona de forma parecida cuando analiza una imagen:

1. **Primeras capas**: detectan elementos muy simples (bordes, líneas, contrastes de color)
2. **Capas intermedias**: combinan esas líneas para formar formas (ojos, nariz, boca)
3. **Capas finales**: combinan esas formas para reconocer el objeto completo (una cara)

Cada capa construye sobre lo que detectó la capa anterior, igual que las piezas de Lego se combinan para formar estructuras cada vez más complejas.

### Más capas = más capacidad, pero también más dificultad

Cuantas más capas y neuronas tiene una red:
- Puede aprender patrones más complejos
- Pero también necesita **más datos** y **más tiempo** para entrenarse correctamente

---

## 4. Deep Learning (Aprendizaje Profundo)

### ¿Qué es?

**Deep Learning** (aprendizaje profundo) es simplemente el nombre que le damos a las redes neuronales que tienen **muchas capas ocultas** (no solo una o dos, sino decenas o incluso cientos).

> La palabra "profundo" (*deep*) se refiere a la **profundidad** de la red, es decir, a la cantidad de capas que tiene — no a que sea más "inteligente" en sí mismo.

### ¿Por qué se volvió tan popular?

Tres factores se combinaron en los últimos años:

1. **Muchos más datos disponibles** (fotos, textos, videos en internet)
2. **Computadoras mucho más potentes** (especialmente las tarjetas gráficas, GPUs)
3. **Mejores técnicas** para entrenar redes con muchas capas sin que "se pierdan" en el proceso

### Aplicaciones reales que probablemente ya conoces

- **Reconocimiento de imágenes**: identificar objetos, personas o escenas en fotos (por ejemplo, el desbloqueo facial del celular)
- **Reconocimiento de voz**: asistentes virtuales que entienden lo que dices
- **Traducción automática**: traductores que entienden el contexto de una frase, no solo palabra por palabra
- **Recomendaciones**: plataformas que sugieren qué ver o escuchar según tus gustos

---

## Resumen visual del recorrido

```
Neurona artificial  →  Red neuronal simple  →  Red neuronal compleja  →  Deep Learning
   (1 decisión)         (pocas capas)           (muchas capas/neuronas)    (red MUY profunda)
```

---

## Preguntas de repaso

1. ¿Qué elementos componen una neurona artificial y qué función cumple cada uno?
2. En la metáfora de "decidir si salir a caminar", ¿qué representan los "pesos" de cada factor?
3. ¿Cuáles son las tres capas básicas de una red neuronal simple y qué hace cada una?
4. ¿Cómo "aprende" una red neuronal a partir de sus errores? Explícalo con tus propias palabras.
5. Usando la metáfora de las piezas de Lego, explica por qué una red compleja necesita varias capas para reconocer una cara en una imagen.
6. ¿Qué significa exactamente la palabra "profundo" en Deep Learning?
7. Menciona dos aplicaciones reales de Deep Learning que uses en tu vida diaria.

---

*Material elaborado para una introducción conceptual a las redes neuronales, sin requerir conocimientos previos de matemáticas avanzadas.*
