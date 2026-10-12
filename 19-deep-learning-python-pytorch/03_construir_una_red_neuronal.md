# 03. Construir una red neuronal con PyTorch

## Por qué importa este tema

En el capítulo anterior aprendiste a representar tus datos con `Dataset` y a servirlos en lotes con `DataLoader`. Ya tienes, por tanto, la materia prima lista. Lo que falta es el elemento que da nombre a todo el campo: la propia red neuronal, la pieza que va a recibir esos lotes de datos y aprender a transformarlos en predicciones.

Construir una red neuronal en PyTorch no es magia ni requiere álgebra avanzada: es definir una clase de Python con dos partes muy concretas. Una vez entiendes esas dos partes, puedes construir desde el clasificador más sencillo hasta arquitecturas mucho más sofisticadas, porque el patrón de fondo no cambia.

## De neuronas a arquitectura: la anatomía de una red

Piensa en una red neuronal como una cadena de producción organizada en plantas o niveles. Cada nivel es una **capa** (*layer*), y dentro de cada capa trabajan varias unidades pequeñas llamadas **neuronas**. Cada neurona hace algo muy simple: recibe varios números de entrada, calcula una suma ponderada de esos números (cada entrada multiplicada por un **peso**, más un valor de ajuste llamado **sesgo** o *bias*) y entrega un único número de salida. Ninguna neurona por separado es inteligente; lo que aporta valor es la combinación de miles de estas operaciones sencillas encadenadas entre capas.

Una red neuronal completa organiza sus capas en tres tipos:

- **Capa de entrada**: recibe los datos en crudo, por ejemplo los píxeles de una imagen ya aplanados en un vector
- **Capas ocultas**: están entre la entrada y la salida, y son las que extraen patrones cada vez más complejos a partir de los datos
- **Capa de salida**: entrega el resultado final, por ejemplo una puntuación para cada una de las clases que quieres distinguir

A esta organización completa —cuántas capas hay, cuántas neuronas tiene cada una y cómo se conectan— la llamamos **arquitectura** de la red. Igual que dos edificios pueden usar los mismos ladrillos y quedar completamente distintos según cómo los coloques, dos redes neuronales pueden usar las mismas piezas de PyTorch y comportarse de forma muy diferente según la arquitectura que elijas.

## `nn.Module`: el plano de tu red

En PyTorch, toda red neuronal se define como una clase de Python que hereda de `nn.Module`. Piensa en `nn.Module` como un plano de construcción: te obliga a organizar tu red en dos bloques bien diferenciados.

- El método `__init__` es donde **declaras** las piezas de tu red: qué capas existen y de qué tamaño son. Es como hacer el inventario de materiales antes de construir
- El método `forward` es donde **conectas** esas piezas: en qué orden pasan los datos por cada capa. Es como el plano de montaje que dice qué pieza va después de cuál

Dentro de `__init__`, la primera línea siempre es `super().__init__()`. Esta línea llama al constructor de `nn.Module`, la clase de la que heredas, y es imprescindible: sin ella, PyTorch no puede registrar correctamente las capas que definas a continuación, y tu red neuronal quedaría mal inicializada.

Cuando instancias tu clase (`modelo = MiRed()`), PyTorch asigna automáticamente valores aleatorios iniciales a los pesos y sesgos de cada capa. Esta inicialización aleatoria no es un descuido: si todas las neuronas empezaran con el mismo valor, aprenderían exactamente lo mismo entre sí y la red perdería toda su capacidad de aprender patrones variados. Empezar en aleatorio "rompe la simetría" entre neuronas y les da a cada una la oportunidad de especializarse en algo distinto durante el entrenamiento (que verás en la clase siguiente).

## `nn.Linear`: capas totalmente conectadas

La pieza más básica para construir capas es `nn.Linear`. Una capa `nn.Linear` es lo que se llama una capa **totalmente conectada** (*fully connected*) o **densa**: cada neurona de la capa recibe la salida de absolutamente todas las neuronas de la capa anterior.

```python
import torch.nn as nn

# Una capa totalmente conectada: recibe 784 valores de entrada y produce 64 de salida
capa = nn.Linear(in_features=784, out_features=64)
```

`in_features` es el número de valores que llegan a la capa (debe coincidir con la salida de la capa anterior, o con el tamaño de tus datos si es la primera capa). `out_features` es el número de neuronas que tiene esta capa, es decir, cuántos valores produce. Internamente, `nn.Linear` guarda una matriz de pesos y un vector de sesgos: son precisamente esos números los que la red va a ir ajustando durante el entrenamiento.

## Funciones de activación: por qué hacen falta

Si encadenas varias capas `nn.Linear` una detrás de otra sin nada más entre medias, matemáticamente el resultado sigue siendo equivalente a una única transformación lineal, sin importar cuántas capas apiles. Una red así jamás podría aprender patrones complejos como los que distinguen un dígito escrito a mano de otro.

Aquí es donde entran las **funciones de activación**: una operación matemática que se aplica a la salida de una capa antes de pasarla a la siguiente, e introduce no linealidad en el proceso. Piensa en una función de activación como una válvula que decide, neurona a neurona, cuánta señal deja pasar: sin ella, la información fluiría de forma demasiado rígida y predecible; con ella, la red puede aprender relaciones mucho más ricas entre los datos.

La función de activación más utilizada en las capas ocultas es la **unidad lineal rectificada** (ReLU, *Rectified Linear Unit*): deja pasar sin cambios los valores positivos y convierte en cero cualquier valor negativo. Es una operación muy barata de calcular y, aun siendo tan simple, funciona bien en la gran mayoría de arquitecturas.

Para la capa de salida de un problema de clasificación con varias categorías se usa otra función distinta: `log_softmax`. Mientras que ReLU se aplica en las capas ocultas para dar flexibilidad a la red, `log_softmax` convierte los valores finales en algo interpretable como una distribución de probabilidad, donde las diez salidas (una por cada dígito, del cero al nueve) suman en conjunto la probabilidad total repartida entre las diez clases.

## Ejemplo de código comentado

Vas a definir un clasificador para el dataset MNIST que ya cargaste en la clase anterior: recibe una imagen de dígito aplanada (784 valores, resultado de aplanar una imagen de 28x28 píxeles) y produce una puntuación para cada uno de los diez dígitos posibles (0-9).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClasificadorMNIST(nn.Module):
    def __init__(self):
        # Llamada obligatoria al constructor de nn.Module
        super().__init__()

        # Declaramos las capas: cada nn.Linear conecta por completo
        # la capa anterior con la siguiente
        self.fc1 = nn.Linear(784, 128)  # capa de entrada -> primera capa oculta
        self.fc2 = nn.Linear(128, 64)   # primera capa oculta -> segunda capa oculta
        self.fc3 = nn.Linear(64, 10)    # segunda capa oculta -> capa de salida (10 dígitos)

    def forward(self, x):
        # Aplicamos ReLU tras cada capa oculta para introducir no linealidad
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # La capa de salida no lleva ReLU: usamos log_softmax para obtener
        # una distribución de probabilidad sobre las 10 clases
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


# Instanciamos la red: aquí se inicializan pesos y sesgos al azar
modelo = ClasificadorMNIST()
print(modelo)

# Simulamos un lote de 16 imágenes MNIST ya aplanadas (16 x 784)
lote_ejemplo = torch.randn(16, 784)

# El forward pass: los datos atraviesan la red y obtenemos una predicción
salida = modelo(lote_ejemplo)
print("Forma de la salida:", salida.shape)  # torch.Size([16, 10])
```

Fíjate en el `dim=1` de `log_softmax`: el lote completo es un tensor de dos dimensiones (16 imágenes x 10 clases), y quieres que la distribución de probabilidad se calcule por fila (dimensión 1, las 10 clases de cada imagen), no mezclando las 16 imágenes del lote entre sí.

Aunque en este capítulo no vas a entrenar todavía la red (eso llega en la clase siguiente), este `forward` que acabas de ejecutar ya es la mitad del trabajo. Por debajo, PyTorch usa una herramienta llamada **autograd** para ir registrando, de forma automática, cada operación matemática que ocurre durante el `forward`. Ese registro es exactamente lo que necesitará el entrenamiento para calcular gradientes y ajustar los pesos: no tienes que programar tú esa parte, solo definir bien la arquitectura y el `forward`.

Una ventaja adicional de definir el `forward` como un método normal de Python es que puedes incluir dentro lógica condicional (`if`, bucles, lo que necesites), algo que otras bibliotecas de aprendizaje profundo no siempre permiten con la misma libertad. Por ahora te basta con encadenar capas en orden, pero conviene saber que esa flexibilidad está disponible cuando la necesites.

## Resumen de puntos clave

- Una red neuronal se organiza en capas (entrada, ocultas y salida) formadas por neuronas que combinan pesos y sesgos
- Toda red en PyTorch se define como una clase que hereda de `nn.Module`, con `__init__` para declarar las capas y `forward` para conectar el flujo de datos entre ellas
- `super().__init__()` es obligatorio al principio de `__init__`: sin él, `nn.Module` no puede registrar las capas correctamente
- `nn.Linear(in_features, out_features)` crea una capa totalmente conectada, con pesos y sesgos que se inicializan al azar
- Las funciones de activación (como ReLU) introducen no linealidad entre capas; sin ellas, apilar capas `nn.Linear` no aporta más capacidad de aprendizaje
- Para clasificación en varias clases, `log_softmax` en la capa de salida convierte las puntuaciones en una distribución de probabilidad
- El `forward` pass ya activa **autograd**, el mecanismo que PyTorch usará para calcular gradientes automáticamente durante el entrenamiento

## Ponlo en práctica

Abre un notebook nuevo en Google Colab y define tu propia clase heredando de `nn.Module`, distinta a la del ejemplo: en vez de 128 y 64 neuronas en las capas ocultas, usa 32 y 16. Recuerda mantener la entrada en 784 (los píxeles de una imagen MNIST aplanada) y la salida en 10 (los dígitos del 0 al 9). Instancia tu red, imprímela con `print()` para ver su estructura, y hazle un `forward` pass con un lote simulado de 8 imágenes usando `torch.randn(8, 784)`. Comprueba que la forma de la salida es la que esperas antes de continuar con la clase siguiente, donde por fin vas a entrenar esta arquitectura.
