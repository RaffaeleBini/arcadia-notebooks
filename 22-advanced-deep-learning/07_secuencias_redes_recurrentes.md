# Capítulo 7: Secuencias y redes neuronales recurrentes

**Acrónimos usados en este capítulo:** red neuronal artificial (ANN), red neuronal convolucional (CNN), perceptrón multicapa (MLP), red neuronal recurrente (RNN), memoria a corto-largo plazo (LSTM, del inglés *Long Short-Term Memory*), unidad recurrente con puertas (GRU, del inglés *Gated Recurrent Unit*), retropropagación a través del tiempo (BPTT, del inglés *Backpropagation Through Time*), procesamiento del lenguaje natural (NLP), error cuadrático medio (MSE), clasificación temporal conexionista (CTC, del inglés *Connectionist Temporal Classification*).

## Por qué importa este tema (series temporales, texto, audio, música)

Piensa en las últimas veces que un sistema informático tuvo que entender algo tuyo: el teclado de tu móvil sugiriendo la palabra siguiente, un asistente de voz transcribiendo lo que dices, una aplicación de música recomendando la siguiente canción según lo que escuchaste antes. En los tres casos, el dato de entrada no es una foto aislada ni una tabla de números independientes: es una secuencia, una colección de elementos donde el orden importa, donde un elemento puede repetirse y donde la longitud total no está fijada de antemano. Una frase tiene tantas palabras como haga falta, un audio dura lo que dure y una serie de ventas mensuales crece cada mes que pasa.

Los capítulos anteriores de este manual trabajaron con arquitecturas pensadas para otro tipo de datos. El perceptrón multicapa (MLP) espera un vector de tamaño fijo: si entrenas un MLP con frases de cinco palabras, no sabrá qué hacer con una frase de ocho. Y aunque forzaras ese tamaño con relleno artificial, seguiría sin captar que "el gato persigue al perro" y "el perro persigue al gato" contienen las mismas palabras pero significan cosas opuestas, porque un MLP normalmente concatena o suma sus entradas sin distinguir posiciones de forma nativa. La red neuronal convolucional (CNN), por su parte, sí admite entradas de tamaño variable gracias a sus filtros deslizantes, pero fue diseñada para capturar patrones espaciales locales, como bordes en una imagen; no tiene un mecanismo propio para acumular contexto a lo largo de una dimensión temporal larga, elemento a elemento, recordando lo visto muchos pasos atrás.

Esta limitación no es un detalle técnico menor. Aparece en cualquier dominio donde los datos llegan en forma de flujo: texto y lenguaje natural (traducción automática, generación de texto, análisis de sentimiento), audio y voz (reconocimiento de habla, síntesis de voz, composición musical), series temporales (previsión de demanda, sensores de temperatura o de humedad, cotizaciones financieras), vídeo (una secuencia de fotogramas) e incluso secuencias de decisiones, como las que toma un agente de aprendizaje por refuerzo jugada tras jugada. En el contexto de un hub de innovación digital como el nuestro, esto se traduce en problemas muy concretos: sensores agrícolas que generan lecturas cada hora y de los que quieres anticipar una plaga, registros clínicos que se acumulan visita tras visita, o el habla capturada por un dispositivo de asistencia a personas mayores. Ninguno de esos problemas encaja bien en un modelo que solo sabe procesar vectores fijos y sin memoria del pasado.

Este capítulo te presenta la familia de arquitecturas diseñada específicamente para resolver esa carencia: la red neuronal recurrente (RNN) y sus variantes más robustas, la LSTM y la GRU. **Presta atención especial al concepto de estado oculto**, porque es la pieza que hace posible todo lo demás.

## Explicación conceptual con analogías

Imagina que lees esta misma frase en voz alta, palabra por palabra, sin ver el resto del párrafo. No procesas cada palabra de forma aislada, como si acabaras de nacer en ese instante: llevas en la cabeza un resumen mental de todo lo que leíste hasta ese punto, y ese resumen condiciona cómo interpretas la palabra nueva. Si acabas de leer "el paciente presentaba fiebre alta y", tu resumen mental ya anticipa que la palabra siguiente probablemente continúe describiendo síntomas, no que hable de deportes. Cuando terminas de leer una palabra más, actualizas ese resumen: descartas detalles que ya no son relevantes, conservas lo importante y añades la información nueva. Ese resumen mental, que se actualiza paso a paso y condiciona cada nueva interpretación, es exactamente lo que en una RNN llamamos estado oculto.

Otra analogía útil: piensa en quien toma notas durante una reunión larga. No transcribe la reunión palabra por palabra; mantiene una libreta con un resumen que va reescribiendo cada pocos minutos, incorporando lo nuevo y descartando lo que ya perdió relevancia. Al final de la reunión, esa libreta no contiene todo lo que se dijo, sino una versión comprimida y útil de lo esencial. Una RNN hace algo parecido con cada elemento de una secuencia: mantiene un vector de estado (su "libreta") que actualiza en cada paso, y ese vector es la única vía por la que la información del pasado llega al presente.

Ahora bien, esta analogía también anticipa un problema real. Si la reunión dura ocho horas y tu libreta solo tiene una página, algo del principio se va a perder u olvidar, sobre todo si reescribes el resumen de forma agresiva en cada actualización. Esto es precisamente lo que les ocurre a las RNN simples con secuencias largas: la información antigua se diluye, un fenómeno que en el desarrollo teórico llamaremos desvanecimiento del gradiente. Para solucionarlo, más adelante vas a conocer una segunda libreta, más protegida, que solo cambia cuando alguien decide explícitamente que debe cambiar: esa es, en esencia, la idea detrás de la LSTM.

## Desarrollo teórico

### Qué necesita un buen modelo de secuencias

Antes de definir la arquitectura, conviene fijar qué propiedades debe cumplir cualquier función que resuma una secuencia. Según se desprende de la literatura de referencia sobre modelado de secuencias, un buen modelo debe:

- Respetar el orden de los elementos, porque reordenarlos puede cambiar por completo el significado
- Aceptar secuencias de longitud variable, sin necesidad de fijar de antemano cuántos elementos habrá
- Ser diferenciable, para poder entrenarse mediante descenso de gradiente
- Ser sensible a cambios individuales, de modo que alterar un solo elemento pueda modificar de forma apreciable la salida
- Preservar dependencias a largo plazo, capturando relaciones entre elementos alejados entre sí en la secuencia

Los enfoques previos al deep learning fallan en alguno de estos puntos. Los modelos de N-gramas, habituales en procesamiento del lenguaje natural (NLP), condicionan la probabilidad de una palabra únicamente a las N palabras anteriores; respetan el orden, pero pierden todo el contexto más allá de esa ventana fija, y además la tabla de probabilidades crece de forma exponencial con el tamaño del vocabulario y de la ventana, hasta volverse inmanejable incluso con vocabularios modestos. Sumar o promediar los vectores de todos los elementos de la secuencia sí admite longitud variable, pero destruye por completo el orden: "no me gusta, pero lo compraría" y "me gusta, pero no lo compraría" darían el mismo vector agregado. Ninguno de los dos enfoques es diferenciable de forma natural ni escala bien. Se necesita un mecanismo distinto: uno que aprenda, con parámetros ajustables por gradiente, a comprimir una secuencia arbitrariamente larga en una representación útil.

### La red neuronal recurrente

Una RNN resuelve este problema manteniendo un vector de estado oculto, denotado h, que se actualiza en cada paso temporal combinando la entrada actual con el estado del paso anterior. Dada una secuencia de entradas x₁, x₂, ..., xₜ, la actualización del estado oculto en el paso t se define como:

**hₜ = tanh(Wₕ · hₜ₋₁ + Wₓ · xₜ + bₕ)**

Aquí, Wₕ es la matriz de pesos que transforma el estado anterior, Wₓ es la matriz de pesos que transforma la entrada actual, bₕ es el término de sesgo, y tanh es la función tangente hiperbólica, que acota los valores del estado entre menos uno y uno para evitar que crezcan sin control. El estado inicial h₀ suele fijarse en cero, aunque en tareas como la traducción automática puede inicializarse con información útil, por ejemplo un resumen de la frase de origen.

Si la tarea es predecir el siguiente elemento de la secuencia, por ejemplo la palabra siguiente en un texto, el estado oculto se transforma en una distribución de probabilidad sobre el vocabulario mediante otra capa lineal seguida de la función softmax:

**yₜ = softmax(Wy · hₜ + by)**

Un detalle crucial: las matrices Wₕ, Wₓ y Wy se comparten en todos los pasos temporales. La misma "regla de actualización" se aplica igual en el paso tres que en el paso trescientos, lo que permite que la red procese secuencias de cualquier longitud con un número fijo de parámetros. Cuando se dibuja este proceso paso a paso, extendiendo la red a lo largo del tiempo, se habla de "desenrollar" (unroll) la RNN: cada paso temporal se representa como una copia de la misma red, conectada a la copia siguiente a través del estado oculto.

Entrenar una RNN para tareas de predicción de secuencias (como el modelado del lenguaje, donde el objetivo es maximizar la probabilidad de la secuencia observada) se plantea habitualmente como un problema de clasificación en cada paso temporal, usando la pérdida de entropía cruzada entre la distribución predicha y la palabra real. La pérdida total de la secuencia es la suma de las pérdidas de cada paso, y el conjunto de parámetros se ajusta minimizando esa suma sobre todo el conjunto de entrenamiento. El algoritmo que calcula los gradientes en este escenario se llama retropropagación a través del tiempo (BPTT): consiste en desenrollar la red, calcular el gradiente de la pérdida respecto a cada copia de los pesos compartidos y acumular esas contribuciones para obtener el gradiente final.

### El problema del gradiente que desaparece o explota

Aquí aparece la principal limitación práctica de la RNN simple. Para entender por qué, simplifica el escenario: ignora entradas y sesgos, y supón que la matriz Wₕ se reduce a un único número escalar w. La recurrencia hₜ = w · hₜ₋₁ se puede desenrollar como:

**hₜ = wᵗ · h₀**

Esta expresión revela el problema con claridad. Si el valor absoluto de w es mayor que uno, wᵗ crece de forma exponencial con t, y el estado (y, durante el entrenamiento, el gradiente que se retropropaga) explota hacia valores enormes: es el fenómeno conocido como gradiente explosivo. Si, por el contrario, el valor absoluto de w es menor que uno, wᵗ decae exponencialmente hacia cero conforme t crece: es el gradiente que se desvanece o vanishing gradient. En la práctica, la actualización real de una RNN pasa por una función tanh, cuya derivada nunca supera el valor uno y suele ser bastante menor en gran parte de su rango. Como el gradiente que se propaga hacia atrás en BPTT es, esencialmente, un producto de muchos factores de este tipo (uno por cada paso temporal atravesado), basta con que cada factor sea ligeramente menor que uno para que, tras decenas o cientos de pasos, el producto completo se aproxime a cero.

La consecuencia práctica es grave: cuando el gradiente que llega a un paso temporal lejano es prácticamente nulo, esa parte de la red deja de recibir señal de aprendizaje útil, y el modelo no consigue ajustar sus pesos para explotar dependencias distantes. Considera un ejemplo clásico: en el texto "Finalmente, Tim planeaba visitar Francia en la última semana de su viaje. Estaba encantado de probar la gastronomía local (...) tomó el tren de las cinco de la madrugada hacia...", predecir con precisión la palabra siguiente, muy probablemente "París", exige recordar el dato "Francia" mencionado docenas de palabras atrás. Si el gradiente asociado a esa dependencia se ha desvanecido, la RNN simple no puede aprenderla, aunque en teoría tenga capacidad suficiente para representarla.

### LSTM: memoria protegida y puertas

Para resolver este problema se diseñó la red de memoria a corto-largo plazo (LSTM), publicada originalmente por Hochreiter y Schmidhuber. La idea central es separar dos flujos de información: un estado de celda (cell state), denotado c, que actúa como una cinta transportadora de memoria a largo plazo con actualizaciones mayormente aditivas, y un estado oculto h, más parecido al de una RNN clásica, que se calcula a partir del estado de celda en cada paso. El acceso a la memoria a largo plazo se regula mediante tres puertas, cada una implementada como una capa con activación sigmoide (que produce valores entre cero y uno, interpretables como "proporción que deja pasar"):

**Puerta de olvido**, que decide qué parte del estado de celda anterior conservar:

fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)

**Puerta de entrada**, que decide qué información nueva incorporar. Se compone de la propia puerta y de un vector de valores candidatos generado con tanh:

iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)

c̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)

Con estas dos piezas, el estado de celda se actualiza combinando lo que se conserva del pasado con lo que se añade de nuevo:

**cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ**

donde ⊙ denota el producto elemento a elemento. Fíjate en que esta actualización es, en gran parte, una suma: el estado de celda anterior se multiplica por la puerta de olvido y se le añade la contribución nueva, en lugar de reescribirse por completo con una transformación no lineal como ocurría en la RNN simple. Ese carácter aditivo es la clave que mitiga el desvanecimiento del gradiente: cuando la puerta de olvido está próxima a uno, el gradiente puede fluir hacia atrás a través de muchos pasos temporales casi sin atenuarse, porque la derivada de una suma respecto a uno de sus términos es simplemente uno.

Por último, la **puerta de salida** decide qué parte del estado de celda se expone como estado oculto:

oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)

**hₜ = oₜ ⊙ tanh(cₜ)**

Con estas cuatro ecuaciones, una LSTM puede aprender, por ejemplo, a mantener casi intacta la puerta de olvido mientras procesa una frase larga en la que necesita recordar un sujeto mencionado al inicio, y solo "abrir" la puerta de entrada cuando aparece información realmente relevante que debe sustituir a la anterior.

### GRU: una alternativa más ligera

La unidad recurrente con puertas (GRU), propuesta por Cho y colaboradores, simplifica la LSTM fusionando el estado de celda y el estado oculto en un único vector, y reduciendo las tres puertas a dos: una puerta de actualización, que combina el papel de las puertas de olvido y entrada, y una puerta de reinicio. Las ecuaciones son:

**zₜ = σ(Wz · [hₜ₋₁, xₜ] + bz)** (puerta de actualización)

**rₜ = σ(Wr · [hₜ₋₁, xₜ] + br)** (puerta de reinicio)

**h̃ₜ = tanh(W · [rₜ ⊙ hₜ₋₁, xₜ] + b)** (estado candidato)

**hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ**

La puerta de actualización zₜ decide, en un único paso, cuánto del estado anterior conservar y cuánto del estado candidato incorporar; cuando zₜ vale cero, el estado no cambia en absoluto, lo que permite propagar información sin apenas pérdida durante muchos pasos, igual que hacía el estado de celda en la LSTM. La puerta de reinicio rₜ, por su parte, controla cuánta influencia tiene el estado anterior a la hora de calcular el candidato: si se acerca a cero, el modelo "empieza de cero" al procesar la entrada actual, ignorando el pasado inmediato.

Al tener menos parámetros que la LSTM, la GRU entrena algo más rápido y consume menos memoria, y en un número considerable de tareas alcanza un rendimiento comparable. No existe una regla universal sobre cuál elegir; en la práctica conviene probar ambas y comparar en el problema concreto, algo que harás tú mismo en la sección de código.

### Aplicaciones y hacia dónde evoluciona el campo

RNN, LSTM y GRU comparten un mismo patrón de uso más allá del modelado del lenguaje: en la traducción automática y otras tareas de secuencia a secuencia (seq2seq), una red codificadora comprime la secuencia de entrada completa en un vector de contexto, y una red decodificadora genera la secuencia de salida palabra a palabra a partir de ese vector, un enfoque que impulsó sistemas como el traductor neuronal de Google. En generación de imágenes, modelos como PixelRNN tratan una imagen como una secuencia de píxeles y predicen cada uno condicionado a los anteriores; en audio, arquitecturas emparentadas como WaveNet generan formas de onda muestra a muestra. En aprendizaje por refuerzo, agentes como los de OpenAI Five o AlphaStar incorporan un núcleo LSTM para procesar observaciones que llegan de forma secuencial a lo largo de una partida.

Con todo, las LSTM y las GRU no son la última palabra. Su procesamiento es inherentemente secuencial (paso a paso), lo que limita la paralelización durante el entrenamiento, y aunque mitigan el desvanecimiento del gradiente, capturar dependencias verdaderamente largas sigue siendo costoso. En años recientes, los transformadores (Transformers), basados en mecanismos de atención que comparan todos los elementos de la secuencia entre sí en paralelo, han desplazado a las arquitecturas recurrentes en buena parte del procesamiento del lenguaje natural. Explorarás esa familia de modelos en un capítulo posterior; por ahora, quédate con la idea de que la RNN, la LSTM y la GRU sentaron las bases conceptuales, como el estado que resume el contexto y el entrenamiento mediante maximización de probabilidad, sobre las que se construyeron esos modelos más recientes.

## Ejemplo de código comentado, en PyTorch

El siguiente ejemplo entrena una LSTM pequeña como modelo de lenguaje a nivel de carácter: dado un fragmento de texto, aprende a predecir el carácter siguiente y, una vez entrenado, genera texto nuevo carácter a carácter, muestreando de la distribución de probabilidad que produce en cada paso. Puedes copiar y ejecutar este bloque directamente en Google Colab.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Texto de entrenamiento: se repite para que la LSTM tenga suficientes
# ejemplos del mismo patrón y pueda converger en pocas epocas
texto = "el hub de innovacion digital ayuda a las empresas a adoptar inteligencia artificial " * 30

# Vocabulario: cada caracter distinto del texto se convierte en un indice entero
caracteres = sorted(set(texto))
char_a_idx = {c: i for i, c in enumerate(caracteres)}
idx_a_char = {i: c for i, c in enumerate(caracteres)}
tam_vocab = len(caracteres)

# Codificamos todo el texto como un tensor de indices enteros
datos = torch.tensor([char_a_idx[c] for c in texto], dtype=torch.long)

longitud_secuencia = 40

def crear_lote(datos, longitud_secuencia, tam_lote):
    # Elegimos puntos de inicio aleatorios y recortamos ventanas de longitud fija;
    # el objetivo de cada ventana es la misma ventana desplazada un caracter
    indices_inicio = torch.randint(0, len(datos) - longitud_secuencia - 1, (tam_lote,))
    entradas = torch.stack([datos[i:i + longitud_secuencia] for i in indices_inicio])
    objetivos = torch.stack([datos[i + 1:i + longitud_secuencia + 1] for i in indices_inicio])
    return entradas, objetivos

class ModeloCharLSTM(nn.Module):
    def __init__(self, tam_vocab, tam_embedding=16, tam_oculto=64, num_capas=1):
        super().__init__()
        self.embedding = nn.Embedding(tam_vocab, tam_embedding)
        self.lstm = nn.LSTM(tam_embedding, tam_oculto, num_layers=num_capas, batch_first=True)
        self.salida = nn.Linear(tam_oculto, tam_vocab)

    def forward(self, x, estado=None):
        x = self.embedding(x)
        # estado es la tupla (h, c); si se pasa None, PyTorch la inicializa en ceros
        salida_lstm, estado = self.lstm(x, estado)
        logits = self.salida(salida_lstm)
        return logits, estado

modelo = ModeloCharLSTM(tam_vocab)
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.003)

num_epocas = 300
tam_lote = 32

for epoca in range(num_epocas):
    entradas, objetivos = crear_lote(datos, longitud_secuencia, tam_lote)
    logits, _ = modelo(entradas)
    # reorganizamos (lote, tiempo, vocab) en (lote*tiempo, vocab) para tratar
    # cada posicion de cada secuencia como un ejemplo de clasificacion independiente
    perdida = F.cross_entropy(logits.reshape(-1, tam_vocab), objetivos.reshape(-1))
    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()
    if (epoca + 1) % 50 == 0:
        print(f"Epoca {epoca + 1}/{num_epocas} - perdida: {perdida.item():.4f}")

@torch.no_grad()
def generar(modelo, texto_semilla, longitud=120, temperatura=0.8):
    modelo.eval()
    entrada = torch.tensor([[char_a_idx[c] for c in texto_semilla]], dtype=torch.long)
    _, estado = modelo(entrada)  # procesamos la semilla para construir el estado inicial
    ultimo_caracter = entrada[:, -1:]
    generado = texto_semilla
    for _ in range(longitud):
        logits, estado = modelo(ultimo_caracter, estado)
        # la temperatura suaviza (>1) o afila (<1) la distribucion antes de muestrear
        probabilidades = F.softmax(logits[:, -1, :] / temperatura, dim=-1)
        idx_siguiente = torch.multinomial(probabilidades, num_samples=1)
        generado += idx_a_char[idx_siguiente.item()]
        ultimo_caracter = idx_siguiente
    modelo.train()
    return generado

print(generar(modelo, "el hub "))
```

Tras las 300 épocas, la pérdida debería bajar de forma clara (de un valor inicial cercano al logaritmo del tamaño del vocabulario hasta un valor mucho más bajo), y el texto generado debería parecerse cada vez más al patrón de entrenamiento, con palabras reconocibles aunque el corpus sea diminuto. **Ejecuta el bloque completo en Colab y observa cómo cambia el texto generado en cada intento**, cambiando la semilla de texto inicial o la temperatura de muestreo.

## Resumen de puntos clave

- Una secuencia es una colección ordenada de elementos de longitud variable; el texto, el audio, las series temporales y el vídeo son ejemplos habituales
- Los modelos prealimentados como el MLP exigen entradas de tamaño fijo y no capturan dependencias de orden; las CNN capturan patrones espaciales locales, pero no fueron diseñadas para memoria a largo plazo a lo largo del tiempo
- La red neuronal recurrente (RNN) mantiene un estado oculto que se actualiza en cada paso mediante hₜ = tanh(Wₕ · hₜ₋₁ + Wₓ · xₜ + bₕ), con los mismos pesos compartidos en todos los pasos temporales
- El entrenamiento de una RNN usa retropropagación a través del tiempo (BPTT), que desenrolla la red y acumula gradientes de cada paso
- Las RNN simples sufren el problema del gradiente que se desvanece o explota: en secuencias largas, la señal de error se atenúa o se dispara de forma exponencial, impidiendo aprender dependencias lejanas
- La LSTM introduce un estado de celda con actualización mayormente aditiva y tres puertas (olvido, entrada, salida) que regulan qué información conservar, añadir o exponer
- La GRU simplifica la LSTM fusionando el estado de celda y el oculto, con solo dos puertas (actualización y reinicio), a costa de algo de flexibilidad pero con menos parámetros
- Ambas arquitecturas mitigan el desvanecimiento del gradiente porque permiten que la información fluya con cambios mínimos a través de muchos pasos temporales cuando las puertas correspondientes así lo determinan
- RNN, LSTM y GRU se usan en modelado del lenguaje, traducción automática, generación de texto e imágenes, síntesis de audio y aprendizaje por refuerzo con observaciones secuenciales
- Los transformadores, basados en atención, han superado en muchas tareas de NLP a las arquitecturas recurrentes, pero heredan conceptos clave que aprendiste en este capítulo, como la representación del contexto y el entrenamiento por máxima verosimilitud

## Ponlo en práctica

Dedica entre quince y treinta minutos a esta mini-tarea, partiendo del código de la sección anterior:

1. **Copia el bloque de código en un cuaderno de Google Colab** y ejecútalo tal cual, para comprobar que entrena sin errores y que genera texto reconocible
2. **Sustituye `nn.LSTM` por `nn.GRU`** en la clase `ModeloCharLSTM` (puedes dejar el nombre de la clase igual, o renombrarla) y vuelve a entrenar con el mismo número de épocas; anota la pérdida final de cada arquitectura y compáralas
3. **Cambia `tam_oculto` de 64 a 8** y entrena de nuevo la versión LSTM; observa cuánto empeora la pérdida final y qué aspecto tiene el texto generado, para comprobar en la práctica cómo el tamaño del estado oculto limita la capacidad del modelo
4. **Sustituye el texto de entrenamiento** por otro fragmento propio, de al menos 200 caracteres, y repite el entrenamiento; fíjate en cuántas épocas necesita el modelo para empezar a producir palabras reconocibles de tu nuevo texto
5. Como reto opcional, añade una segunda capa a la LSTM (`num_layers=2`) y comprueba si mejora o empeora la pérdida final con este corpus tan pequeño; reflexiona sobre por qué un corpus diminuto puede no beneficiarse de más profundidad

Guarda tus resultados (pérdidas y ejemplos de texto generado) porque te servirán de referencia cuando, en capítulos posteriores, compares estas arquitecturas recurrentes con los mecanismos de atención y los transformadores.
