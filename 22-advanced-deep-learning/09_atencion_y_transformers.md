# Capítulo 9 — Atención y memoria en el aprendizaje profundo: el camino hacia los Transformers

## Por qué importa este tema

En capítulos anteriores viste las redes neuronales recurrentes (RNN) y su variante más robusta, las redes de memoria a corto y largo plazo (LSTM, *long short-term memory*). Ambas arquitecturas procesan una secuencia palabra a palabra, o paso a paso, y van actualizando un estado oculto que resume todo lo visto hasta ese momento. Funciona razonablemente bien con frases cortas. El problema aparece en cuanto la secuencia crece: ese estado oculto tiene un tamaño fijo, y forzarlo a contener el resumen de una frase de cuarenta palabras, un párrafo entero o un documento completo es como pedirle a una sola persona que memorice un libro de viva voz y lo repita sin volver a mirarlo. Algo se pierde por el camino, casi siempre lo que apareció al principio.

Este capítulo trata sobre la solución que el campo encontró para ese problema: la atención (*attention*). En lugar de comprimir toda la entrada en un único vector y confiar en que contenga todo lo necesario, la atención permite que el modelo vuelva a consultar la secuencia de entrada completa en cada paso y decida, de forma dinámica, qué partes son relevantes para la predicción que está haciendo en ese momento. Es la diferencia entre recitar un libro de memoria y tener el libro abierto sobre la mesa mientras respondes preguntas sobre él.

La importancia de este tema no es solo histórica. La atención, en su forma más depurada, es el mecanismo que sostiene toda la generación actual de modelos de lenguaje: la arquitectura Transformer, publicada en 2017, prescinde por completo de la recurrencia y construye toda su capacidad de comprender contexto sobre un único bloque, la autoatención (*self-attention*). Modelos como GPT o BERT son pilas de este bloque repetidas decenas de veces. Entender bien la atención aquí es el requisito indispensable para entender cualquier Transformer en los capítulos siguientes: sin este bloque construido con solidez, el resto del edificio no se sostiene.

Además, verás que la atención no se limita a mirar la entrada de un modelo: el mismo mecanismo permite construir memoria externa direccionable, una especie de bloc de notas que la red puede leer y escribir de forma selectiva. Esta idea, encarnada en arquitecturas como la máquina de Turing neuronal (NTM, *Neural Turing Machine*), amplía enormemente lo que una red puede aprender a hacer, desde copiar secuencias largas hasta ejecutar algoritmos sencillos.

## Explicación conceptual con analogías

Imagina que entras en una biblioteca enorme buscando información sobre "el clima en la Patagonia en invierno". Tú no lees todos los libros de principio a fin: llevas una pregunta concreta en la cabeza, que en este contexto llamamos la consulta (*query*). Recorres los lomos de los libros, cada uno con un título o etiqueta que resume su contenido; esas etiquetas son las claves (*keys*). Comparas tu consulta con cada clave y decides cuánto se parecen: el libro titulado "Geografía de la Patagonia" encaja mucho mejor que "Historia del imperio romano". Esa comparación produce un grado de relevancia para cada libro. Finalmente, no te llevas el título a casa, te llevas el contenido del libro, es decir, su valor (*value*), y cuanto más relevante era ese libro según tu comparación, más peso le das a su contenido en el resumen final que construyes.

Ese proceso completo —comparar una consulta con un conjunto de claves, obtener un grado de relevancia para cada una y combinar los valores correspondientes ponderados por esa relevancia— es exactamente lo que hace un mecanismo de atención. Y funciona igual dentro de una red neuronal: en cada paso, el modelo genera una consulta ("¿qué necesito saber ahora para decidir la siguiente palabra?"), la compara con las claves de todas las posiciones de la entrada, obtiene pesos de relevancia mediante una función softmax y construye un vector de contexto como suma ponderada de los valores.

Esta idea también conecta con algo que ya conoces de tu propia experiencia: el llamado "problema del cóctel", esa capacidad que tienes de concentrarte en una sola conversación en una fiesta ruidosa, filtrando de forma automática las demás voces del fondo. Tu cerebro no procesa por igual cada sonido que llega a tu oído: asigna más peso a la conversación relevante y descarta el resto. La atención en una red neuronal persigue exactamente ese efecto, aplicado a vectores en lugar de a ondas sonoras: procesar toda la información disponible, pero ponderando fuertemente solo la parte que importa para la tarea en curso. Es, otra vez, la diferencia entre memorizar un informe de cien páginas y responder una pregunta con el informe abierto delante de ti, pudiendo volver a la página exacta que necesitas.

## Desarrollo teórico

### El cuello de botella de las RNN en secuencia a secuencia

La arquitectura habitual para tareas como traducción automática antes de la llegada de la atención era el esquema codificador-decodificador (*encoder-decoder*): una RNN o LSTM, el codificador, lee la frase de entrada palabra a palabra y produce, al final, un único vector de estado oculto que debe resumir toda la frase. Ese vector, y solo ese vector, se entrega al decodificador, otra RNN, que lo usa como punto de partida para generar la frase de salida palabra a palabra.

El problema es de capacidad: el vector de estado oculto tiene una dimensión fija, decidida de antemano, con independencia de que la frase de entrada tenga cinco palabras o cincuenta. Toda la información —significado, orden, matices, relaciones a larga distancia entre palabras— tiene que caber en ese mismo espacio, comprimida. Cuanto más larga es la secuencia, más grave es la pérdida de información, y en la práctica se observa que la calidad de la traducción cae de forma notable a partir de cierta longitud de frase. Este efecto es el cuello de botella del codificador: toda la comunicación entre la entrada y la salida pasa por un canal de anchura fija, sin posibilidad de volver a consultar directamente las palabras originales una vez comprimidas.

### La solución: dejar que el decodificador mire hacia atrás

La atención resuelve este cuello de botella de una forma conceptualmente sencilla: en lugar de que el decodificador dependa solo del último estado del codificador, se le permite acceder, en cada paso de generación, a los estados ocultos del codificador correspondientes a todas las posiciones de la entrada, no solo a la última.

Formalmente, dado un conjunto de vectores de entrada $h_1, h_2, \dots, h_n$ (por ejemplo, el estado oculto del codificador en cada palabra) y el estado del decodificador $s_{t-1}$ en el paso anterior, el mecanismo calcula primero una puntuación de relevancia entre el estado del decodificador y cada vector de entrada, mediante una función de puntuación $\text{score}$:

$$e_{t,i} = \text{score}(s_{t-1}, h_i)$$

Esta función puede tomar varias formas: un producto punto simple, una transformación lineal seguida de producto punto (forma "general"), o una pequeña red que concatena ambos vectores y los pasa por una capa no lineal (forma aditiva, la propuesta original de Bahdanau en 2014). Las puntuaciones se normalizan después con una función softmax, de modo que sumen uno y se comporten como pesos de relevancia:

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{n} \exp(e_{t,j})}$$

Finalmente, el vector de contexto $c_t$ que recibe el decodificador en el paso $t$ es la suma de todos los vectores de entrada, ponderados por su peso de relevancia:

$$c_t = \sum_{i=1}^{n} \alpha_{t,i} \, h_i$$

Este vector de contexto, distinto en cada paso de generación, sustituye —o complementa— al estado fijo del codificador. El decodificador ya no depende de un único resumen: en cada instante reconstruye, a partir de la entrada completa, el subconjunto de información que más le conviene para esa palabra concreta. Cuando visualizas estos pesos $\alpha_{t,i}$ como un mapa de calor, con la secuencia de entrada en un eje y la secuencia de salida en el otro, aparece con frecuencia un patrón casi diagonal en traducción: cada palabra de salida atiende sobre todo a la palabra de entrada que le corresponde, con desviaciones puntuales cuando el orden gramatical difiere entre los dos idiomas, por ejemplo cuando un verbo alemán se desplaza al final de la frase.

### Atención implícita frente a atención explícita

Conviene distinguir dos formas de atención. La implícita es la que cualquier red profunda desarrolla de forma natural durante el entrenamiento: sin que nadie la programe, la red acaba respondiendo con más fuerza a ciertas partes de la entrada que a otras, algo detectable analizando el jacobiano de la red, la matriz de derivadas parciales de cada salida respecto a cada entrada. La explícita, en cambio, es la que viste en el apartado anterior: un mecanismo definido de forma expresa, con pesos calculados y visibles, que decide qué partes de los datos se presentan a la red. Frente a la implícita, la atención explícita aporta ventajas prácticas: mayor eficiencia computacional, mejor escalabilidad ante entradas de tamaño variable y, sobre todo, más interpretabilidad, porque puedes inspeccionar los pesos directamente en lugar de inferir el comportamiento de la red de forma indirecta.

### Atención por contenido y la separación en clave-valor

Una variante importante, y la que conecta directamente con los Transformers, es la atención por contenido (*content-based attention*), también llamada atención asociativa. En lugar de comparar el estado del decodificador con vectores de entrada genéricos, el modelo genera de forma explícita un vector clave (*key*), y lo compara con el contenido de cada elemento de los datos usando una función de similitud, típicamente el producto punto o la similitud coseno. El resultado se normaliza con softmax igual que antes.

La mejora clave llega cuando separas los datos en dos papeles distintos: una clave, que se usa únicamente para calcular el peso de atención, y un valor, que se usa únicamente para construir la salida. Con pesos $w_i$ ya calculados a partir de las claves $k_i$, el vector de lectura final se obtiene como combinación ponderada de los valores $v_i$ correspondientes, no de las claves:

$$v = \sum_{i} w_i \, v_i$$

Esta separación —una cosa sirve para decidir dónde mirar, otra cosa distinta es lo que efectivamente lees— es precisamente la idea de la biblioteca del apartado anterior, y es la pieza que faltaba para llegar a la formulación moderna de consulta, clave y valor.

### Formalización moderna: consulta, clave y valor (query, key, value)

El Transformer generaliza y sistematiza esta idea. Dada una secuencia de entrada representada como una matriz $X$, de tamaño $n \times d$ (n posiciones, cada una con un vector de dimensión $d$), el modelo genera tres proyecciones lineales distintas mediante tres matrices de pesos aprendidas $W^Q$, $W^K$ y $W^V$:

$$Q = X W^Q, \qquad K = X W^K, \qquad V = X W^V$$

$Q$ contiene las consultas, una por posición; $K$ contiene las claves; $V$ contiene los valores. La atención se calcula comparando cada consulta con todas las claves mediante producto punto, escalando el resultado y aplicando softmax para obtener los pesos, que finalmente ponderan los valores. Esta operación se conoce como atención escalada por producto punto (*scaled dot-product attention*):

$$\text{Atención}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V$$

Aquí $d_k$ es la dimensión de las claves (y, habitualmente, también de las consultas). El escalado por $\sqrt{d_k}$ no es cosmético: cuando $d_k$ crece, el producto punto tiende a valores muy grandes en magnitud, lo que empuja al softmax hacia una región de gradiente casi nulo, el mismo problema de saturación que viste con la sigmoide en el capítulo dos. Dividir por $\sqrt{d_k}$ mantiene los valores en un rango donde el softmax conserva gradientes útiles.

Cuando la consulta, la clave y el valor provienen de la misma secuencia de entrada, este mecanismo se llama autoatención (*self-attention*): cada posición compara su consulta con las claves de todas las demás posiciones de la misma frase, incluida ella misma, y así el modelo captura relaciones entre palabras sin importar cuánto distan entre sí, algo que a una RNN le cuesta cada vez más conforme aumenta esa distancia.

### Atención multi-cabeza (multi-head attention)

Una sola atención escalada por producto punto obliga al modelo a mezclar en un único patrón de pesos todo tipo de relaciones: sintácticas, semánticas, de proximidad, de correferencia. El Transformer evita esa limitación ejecutando varias atenciones en paralelo, cada una con sus propias proyecciones $W_i^Q$, $W_i^K$, $W_i^V$, entrenadas de forma independiente:

$$\text{cabeza}_i = \text{Atención}(Q W_i^Q, K W_i^K, V W_i^V)$$

$$\text{MultiCabeza}(Q, K, V) = \text{Concat}(\text{cabeza}_1, \dots, \text{cabeza}_h) \, W^O$$

Las salidas de las $h$ cabezas se concatenan y se proyectan de nuevo a la dimensión original mediante otra matriz aprendida $W^O$. En la práctica, cada cabeza tiende a especializarse: alguna atiende a palabras vecinas, otra captura concordancia gramatical a larga distancia, otra sigue referencias pronominales, y esta especialización emerge sola durante el entrenamiento.

Vale la pena remarcar la diferencia de fondo con las arquitecturas anteriores, basadas en un controlador central que decidía dónde mirar: en el Transformer, cada posición genera su propia consulta y la compara con todas las demás, sin controlador centralizado. La atención emerge directamente de los datos, y el modelo prescinde por completo de recurrencia, convoluciones o memoria externa; el artículo original que introdujo esta arquitectura en 2017 se tituló, de forma bastante literal, "La atención es todo lo que necesitas" (*Attention Is All You Need*).

### Memoria externa y direccionable

La atención también sirve para leer y escribir en una memoria externa, una estructura de datos separada de los pesos de la red que puede crecer sin reentrenar el modelo. La idea nace de una limitación real de las RNN y las LSTM: en ellas, computación y memoria están fusionadas en el mismo estado oculto, así que ampliar la memoria implica agrandar ese estado y, con él, el coste de cada paso. Un ordenador convencional no funciona así: un procesador pequeño accede a una memoria RAM (memoria de acceso aleatorio) enorme sin recorrerla entera en cada operación. Esa separación entre procesador y memoria es justo lo que las arquitecturas de memoria externa trasladan al aprendizaje profundo.

Una red de memoria (*memory network*) típica se organiza en cuatro componentes: un mapa de características de entrada, que convierte la entrada en una representación adecuada; un módulo de generalización, que actualiza la matriz de memoria con la nueva información; un mapa de características de salida, que produce una representación consultando la memoria; y un módulo de respuesta, que genera la salida final a partir de esa consulta. La interacción entre la entrada y la memoria se modela, de nuevo, con atención: dada una consulta y una matriz de memoria $M$, con filas $M_i$ (cada una un "renglón" o ranura de memoria), los pesos de atención determinan cuánto se lee de cada ranura, y la salida es la suma ponderada de esas ranuras.

La arquitectura de referencia en este terreno es la máquina de Turing neuronal (NTM), presentada por el equipo de DeepMind en 2014, inspirada en el modelo teórico de Turing. Una NTM combina un controlador —una red neuronal, recurrente o no— con una matriz de memoria externa a la que accede mediante "cabezas" de lectura y escritura, cada una gobernada por un mecanismo de atención. La NTM combina dos estrategias de direccionamiento:

- Direccionamiento por contenido: el controlador emite un vector clave y lo compara, mediante similitud coseno, con cada ranura de la memoria; el resultado se normaliza con softmax, y un parámetro adicional de nitidez (*sharpness*) permite concentrar el foco en una única ranura cuando conviene, en lugar de repartirlo de forma difusa
- Direccionamiento por localización: el controlador puede desplazar el foco de atención del paso anterior una cierta distancia hacia delante o hacia atrás en la memoria, mediante un pequeño núcleo de desplazamiento (*shift kernel*), lo que permite recorrer la memoria de forma secuencial, como un iterador

Combinar ambos mecanismos permite comportamientos ricos: localizar por contenido el inicio de una secuencia almacenada y, después, desplazarse por localización a lo largo de esa secuencia, como haría un puntero recorriendo un arreglo.

La escritura en la NTM se inspira directamente en las puertas de las LSTM. Cada operación de escritura combina un vector de borrado $e_t$, con valores entre cero y uno, y un vector de adición $a_t$. La actualización de cada ranura de memoria $M_i$, ponderada por el peso de escritura $w_i$ de esa ranura, sigue esta forma:

$$M_i \leftarrow M_i \odot (1 - w_i \, e_t) + w_i \, a_t$$

Cuando el peso de borrado vale uno, el contenido previo de esa ranura se pone a cero antes de sumar el vector de adición; cuando vale cero, el contenido permanece intacto. Es una versión diferenciable y suave de la decisión binaria "borrar o no borrar", exactamente en el mismo espíritu que las puertas de una LSTM, pero aplicada a una memoria externa mucho más amplia que el estado oculto habitual.

Esta separación entre computación y memoria trae una consecuencia práctica notable: una LSTM entrenada para copiar secuencias cortas generaliza mal a secuencias mucho más largas, porque los elementos que puede retener están limitados por el tamaño fijo de su estado. Una NTM, en cambio, mantiene la fidelidad de la copia incluso con secuencias bastante más largas que las usadas en el entrenamiento, porque su capacidad de memoria no está atada al tamaño del controlador. Esto ha permitido entrenar estos modelos en tareas de sabor algorítmico: copiar secuencias, repetirlas un número indicado de veces (como un bucle "para"), o inferir probabilidades de transición en una secuencia binaria a partir de los patrones ya observados.

La evolución natural de la NTM es la computadora neuronal diferenciable (DNC, *Differentiable Neural Computer*), publicada en 2016, que añade mecanismos de acceso más sofisticados y permite trabajar con estructuras en forma de grafo, con nodos y enlaces, en vez de limitarse a secuencias. Una DNC entrenada puede responder preguntas sobre el camino más corto entre dos estaciones de una red de transporte, o resolver relaciones de parentesco en un árbol genealógico.

## Ejemplo de código comentado

Vas a implementar la atención escalada por producto punto exactamente como la definiste en la teoría, usando solo operaciones básicas de tensores —multiplicación de matrices (`matmul`, con el operador `@`) y una función softmax— sin recurrir a capas ya construidas como `nn.MultiheadAttention`. Trabajarás sobre una frase corta de juguete y visualizarás los pesos de atención resultantes con un mapa de calor. El fragmento es autocontenido y funciona directamente en Google Colab.

```python
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

# Frase de ejemplo, ya tokenizada en palabras (en un modelo real usarías un tokenizador real)
tokens = ["el", "gato", "negro", "duerme", "en", "el", "sofá"]
n = len(tokens)
d = 8       # dimensión del embedding, pequeña solo para que el ejemplo sea legible
d_k = 8     # dimensión de consultas y claves

# Embeddings de partida: en un modelo entrenado vendrían de una capa de embedding aprendida;
# aquí los generamos al azar para centrarnos en el mecanismo de atención
X = torch.randn(n, d)

# Proyecciones lineales aprendibles: transforman el mismo embedding en tres papeles distintos
W_q = torch.randn(d, d_k) * 0.5
W_k = torch.randn(d, d_k) * 0.5
W_v = torch.randn(d, d_k) * 0.5

Q = X @ W_q   # consultas: qué busca cada palabra
K = X @ W_k   # claves: qué ofrece cada palabra al ser comparada
V = X @ W_v   # valores: qué contenido entrega cada palabra si es elegida

def softmax_manual(z, dim=-1):
    # Restamos el máximo antes de la exponencial por estabilidad numérica,
    # el mismo truco que viste con softmax en la capa de salida de clasificación
    z = z - z.max(dim=dim, keepdim=True).values
    exp_z = torch.exp(z)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

def atencion_escalada(Q, K, V, d_k):
    puntuaciones = Q @ K.transpose(-2, -1)            # producto punto: consulta contra cada clave
    puntuaciones = puntuaciones / (d_k ** 0.5)         # escalado por raíz de d_k
    pesos = softmax_manual(puntuaciones, dim=-1)       # cada fila suma uno: es una distribución
    salida = pesos @ V                                 # combinación ponderada de los valores
    return salida, pesos

salida, pesos_atencion = atencion_escalada(Q, K, V, d_k)

print("Forma de la salida:", salida.shape)                        # (7, 8)
print("Cada fila de pesos suma:", pesos_atencion.sum(dim=-1))      # ~1.0 en cada posición
```

**Ejecuta esta celda en Google Colab antes de continuar.** Comprueba que la forma de `salida` coincide con la de `X` en el número de posiciones, y que cada fila de `pesos_atencion` suma aproximadamente uno: es la señal de que el softmax está bien aplicado y de que cada palabra reparte su atención como una distribución de probabilidad sobre las demás.

Ahora visualiza esos pesos como un mapa de calor, con la palabra que pregunta (consulta) en el eje vertical y la palabra consultada (clave) en el eje horizontal:

```python
plt.figure(figsize=(6, 5))
plt.imshow(pesos_atencion.detach().numpy(), cmap="viridis")
plt.xticks(range(n), tokens, rotation=45, ha="right")
plt.yticks(range(n), tokens)
plt.xlabel("Clave (palabra observada)")
plt.ylabel("Consulta (palabra que pregunta)")
plt.title("Pesos de atención escalada por producto punto")
plt.colorbar(label="Peso de atención")
plt.tight_layout()
plt.show()
```

**Ejecuta esta segunda celda y observa el mapa resultante.** Como los embeddings y las proyecciones son aleatorios, no verás aquí un patrón lingüístico real —para eso hacen falta pesos entrenados sobre datos reales—, pero sí verás con claridad la mecánica: cada fila es la distribución de atención de una palabra sobre las demás, con tonos más claros donde el peso es mayor. Es el mismo tipo de mapa que aparece en cualquier herramienta de visualización de Transformers, solo que aquí lo has construido tú, desde cero, con dos operaciones de tensores y una función softmax.

## Resumen de puntos clave

- Las RNN y las LSTM sufren un cuello de botella al comprimir una secuencia entera en un único estado oculto de tamaño fijo, lo que degrada el rendimiento en secuencias largas
- La atención sustituye ese resumen único por un vector de contexto distinto en cada paso, calculado como suma ponderada de toda la entrada, con pesos obtenidos mediante una puntuación de relevancia normalizada con softmax
- La atención por contenido separa los datos en claves, usadas para decidir dónde mirar, y valores, usados para construir la salida; esa separación clave-valor es la semilla directa de la formulación moderna de consulta, clave y valor
- La atención escalada por producto punto se calcula como $\text{softmax}(QK^{\top}/\sqrt{d_k})\,V$, donde $Q$, $K$ y $V$ son proyecciones lineales aprendidas de la misma entrada; el escalado por $\sqrt{d_k}$ evita que el softmax sature con dimensiones grandes
- Cuando la consulta, la clave y el valor provienen de la misma secuencia, el mecanismo se llama autoatención, y permite relacionar directamente cualquier par de posiciones sin importar la distancia entre ellas
- La atención multi-cabeza ejecuta varias atenciones en paralelo, con proyecciones independientes, y concatena sus salidas: cada cabeza tiende a especializarse en un tipo distinto de relación entre palabras
- El Transformer elimina por completo la recurrencia y construye toda su capacidad de modelar contexto sobre bloques de autoatención repetidos, lo que resolvió el cuello de botella de forma definitiva y sostiene a los modelos de lenguaje modernos
- La memoria externa direccionable, como en la máquina de Turing neuronal (NTM) y su sucesora, la computadora neuronal diferenciable (DNC), separa computación y almacenamiento usando atención para leer y escribir de forma selectiva sobre una matriz de memoria, combinando direccionamiento por contenido y por localización

## Ponlo en práctica

Dedica entre quince y treinta minutos a esta mini-tarea en Google Colab. Reutiliza el código de este capítulo como punto de partida.

1. Cambia la frase de ejemplo por una tuya de entre seis y diez palabras, y ajusta `d` y `d_k` a dieciséis en lugar de ocho
2. Añade una segunda cabeza de atención: crea un segundo juego de matrices `W_q2`, `W_k2`, `W_v2`, calcula sus propios pesos de atención y dibuja los dos mapas de calor uno junto al otro con `plt.subplots(1, 2)`; compara si ambas cabezas reparten la atención de forma distinta
3. Fuerza manualmente una relación de dependencia: multiplica el embedding de una palabra por un valor constante alto (por ejemplo, `X[2] = X[2] * 5`) y vuelve a ejecutar la atención; observa y anota cómo cambian los pesos de las demás palabras hacia esa posición
4. Como reto final, elimina el escalado por $\sqrt{d_k}$ de la función `atencion_escalada` y aumenta `d_k` a ciento veintiocho; compara el mapa de calor resultante con el de la versión escalada y relaciona lo que observes con la saturación del softmax que explicamos en la teoría

Guarda tu notebook de Colab: en el próximo capítulo construirás un bloque de Transformer completo, con atención multi-cabeza, normalización y una red hacia adelante, apoyándote directamente en el código de atención que acabas de escribir aquí.

## Glosario de siglas empleadas en este capítulo

- DNC: computadora neuronal diferenciable (*Differentiable Neural Computer*)
- LSTM: memoria a corto y largo plazo (*long short-term memory*)
- NTM: máquina de Turing neuronal (*Neural Turing Machine*)
- RAM: memoria de acceso aleatorio (*random-access memory*)
- RNN: red neuronal recurrente (*recurrent neural network*)
