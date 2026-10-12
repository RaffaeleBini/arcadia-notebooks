[[Índice](00_indice.md) · Siguiente: [Tu primer modelo con Keras →](02_primer_modelo_mnist.md)]

# 1. ¿Qué es el Deep Learning? Neuronas y redes desde cero

## Por qué te importa esto

Cada vez que tu móvil reconoce tu cara, un traductor online convierte una frase al vuelo o una plataforma de streaming adivina qué serie te va a gustar, hay una red neuronal detrás. El Deep Learning (DL, aprendizaje profundo) es la técnica que hizo posible este salto: en vez de programar reglas explícitas ("si el pixel es oscuro y está en esta posición, es un ojo"), construyes un sistema que aprende esas reglas por sí mismo a partir de ejemplos.

No necesitas un doctorado en matemáticas para empezar. Necesitas entender qué es una neurona artificial, cómo se organizan en capas y por qué eso basta para que una máquina aprenda patrones complejos. Eso es lo que ves en este capítulo.

## La neurona artificial, explicada con una analogía

Imagina un jurado de una neurona: recibe varias opiniones (entradas), pondera cuánto le importa cada una (pesos) y emite un veredicto (salida). Cuanto más peso le da a una opinión, más influye esa opinión en el resultado final.

Formalmente, una neurona artificial:

1. Recibe varios valores de entrada (por ejemplo, los píxeles de una imagen)
2. Multiplica cada entrada por un peso (que indica su importancia)
3. Suma todos esos productos y añade un valor adicional llamado sesgo (bias)
4. Pasa ese resultado por una función de activación, que decide si la neurona "se activa" y con qué intensidad

Una sola neurona no hace gran cosa. El poder aparece cuando conectas miles de ellas en capas.

## Capas: de entrada, ocultas y de salida

Una red neuronal artificial (ANN, Artificial Neural Network) se organiza en tres tipos de capas:

- **Capa de entrada**: recibe los datos en crudo (los píxeles de una foto, las palabras de una frase, el precio de una acción)
- **Capas ocultas**: transforman esos datos paso a paso, extrayendo patrones cada vez más abstractos. Si hay dos o más capas ocultas, hablamos de una red *profunda*, de ahí el nombre "deep learning"
- **Capa de salida**: entrega el resultado final (por ejemplo, la probabilidad de que la imagen sea un gato)

La diferencia clave frente al Machine Learning (ML, aprendizaje automático) tradicional es que tú no decides qué características son importantes. En un sistema clásico, un ingeniero tendría que decirle al modelo "mira las orejas puntiagudas" o "mira los bigotes". En una red profunda, las primeras capas aprenden a detectar bordes y texturas, las capas intermedias combinan esos bordes en formas, y las capas finales combinan esas formas en conceptos como "oreja de gato". Todo eso lo aprende la red sola, a partir de ejemplos.

## Funciones de activación: el interruptor de cada neurona

Sin una función de activación, una red neuronal sería solo una calculadora de sumas ponderadas: útil, pero incapaz de aprender patrones complejos y no lineales (como distinguir un gato de un perro). Las funciones de activación más habituales son:

- **Step (escalón)**: la más simple, la neurona se activa (1) o no (0), sin matices. Apenas se usa hoy
- **Sigmoid**: aplasta cualquier valor a un rango entre cero y uno, útil cuando necesitas una probabilidad
- **ReLU (Rectified Linear Unit)**: deja pasar los valores positivos tal cual y convierte los negativos en cero. Es la más usada en capas ocultas por lo barata que es de calcular y porque evita algunos problemas de entrenamiento
- **Softmax**: convierte un conjunto de valores en una distribución de probabilidades que suma uno, ideal para la capa de salida cuando tienes que elegir entre varias categorías (por ejemplo, "es un cero, un uno, un dos... o un nueve")

## Un primer vistazo a Python, TensorFlow y Keras

Vas a construir tus redes con TensorFlow, una librería de Google para computación numérica y Deep Learning, y con Keras, la interfaz de alto nivel que hace que definir una red sea tan simple como apilar capas. Desde TensorFlow 2, Keras viene integrado directamente: no necesitas instalarlo por separado.

```python
# Instala TensorFlow (ya incluye Keras) si no lo tienes
# En Google Colab normalmente ya viene preinstalado
!pip install --upgrade tensorflow

import tensorflow as tf

# Comprueba la versión instalada
print(tf.__version__)
```

Con esto ya tienes montado el entorno. En el próximo capítulo usas estas mismas herramientas para construir, entrenar y evaluar tu primer modelo real.

## Resumen de puntos clave

- Una neurona artificial combina entradas ponderadas y decide su salida con una función de activación
- Una red neuronal se organiza en capa de entrada, una o varias capas ocultas y capa de salida
- "Profunda" significa que hay dos o más capas ocultas: ahí nace el nombre "deep learning"
- La gran ventaja del deep learning frente al ML tradicional es que la red extrae las características importantes por sí sola
- ReLU domina en capas ocultas, softmax domina en la capa de salida para clasificación multiclase
- TensorFlow y Keras son las herramientas que usas en este curso para construir tus redes

## Ponlo en práctica

Abre una notebook de Python (en tu ordenador o en Google Colab) e instala TensorFlow con el snippet de este capítulo. Después, escribe una función en Python que simule una sola neurona: debe recibir una lista de entradas y una lista de pesos del mismo tamaño, calcular la suma ponderada más un sesgo, y devolver el resultado de aplicarle la función ReLU (pista: ReLU es simplemente `max(0, x)`). Prueba tu función con tres entradas distintas y comprueba que nunca devuelve un número negativo.

---

[[Índice](00_indice.md) · Siguiente: [Tu primer modelo con Keras →](02_primer_modelo_mnist.md)]
