# Python Programming Fundamentals for Dummies

### Manual de guía para estudiar y aprender los fundamentos de Python

Este manual resume el temario del curso **EITC/CP/PPF** (European Information Technologies Certification Academy - Certified Professional - Python Programming Fundamentals), impartido por el European IT Certification Institute (EITCI). Está pensado para acompañarte durante el estudio y para servir de referencia rápida cuando practiques con el notebook `Tic_Tac_Toe_Aprende_Python.ipynb`, que crea un juego de tres en raya (Tic Tac Toe) paso a paso aplicando cada concepto.

No hace falta experiencia previa. Lee cada lección, prueba los ejemplos en tu propio ordenador y completa los ejercicios del notebook antes de pasar a la lección siguiente.

---

## Cómo usar este manual

1. Lee la lección completa antes de tocar código
2. Abre una terminal de Python o el notebook y reproduce cada ejemplo
3. Anota las dudas que te surjan en el glosario, al final del documento
4. Cuando termines una lección, resuelve los ejercicios equivalentes en el notebook
5. Repite el proceso hasta llegar a la lección seis

**¡Empieza ya y no tengas miedo de equivocarte!** Los errores son la forma más rápida de aprender a programar.

---

## Lista de acrónimos

| Acrónimo | Nombre completo |
|---|---|
| IDE | Integrated Development Environment (entorno de desarrollo integrado) |
| GUI | Graphical User Interface (interfaz gráfica de usuario) |
| CLI | Command Line Interface (interfaz de línea de comandos) |
| POO | Programación orientada a objetos |
| PyPI | Python Package Index (índice de paquetes de Python) |
| pip | Pip Installs Packages (gestor de paquetes de Python) |
| RAM | Random Access Memory (memoria de acceso aleatorio) |

---

## Índice de lecciones

1. Introducción a Python 3
2. Primeros pasos: tuplas, cadenas de texto y bucles
3. Listas y el primer boceto del Tic Tac Toe
4. Funciones incorporadas
5. Índices y slices (cortes)
6. Funciones propias
7. Parámetros de función y tipado
8. Mutabilidad revisitada
9. Manejo de errores
10. Cómo calcular el ganador en horizontal
11. Cómo calcular el ganador en vertical
12. El algoritmo del ganador en diagonal
13. Iteradores e iterables
14. Cierre del proyecto Tic Tac Toe
15. Conclusión y siguientes pasos

---

## Lección 1 · Introducción a Python 3

Python es un lenguaje de programación de alto nivel, creado por Guido van Rossum y publicado por primera vez en 1991. Se hizo popular por dos motivos principales: es fácil de leer y sirve para casi cualquier cosa, desde páginas web hasta inteligencia artificial (IA).

### Características clave

- **Sintaxis limpia**: Python usa la indentación (los espacios al principio de cada línea) para separar bloques de código, en vez de llaves `{}`. Esto obliga a escribir código ordenado
- **Lenguaje interpretado**: no hace falta compilar el código. El intérprete lo ejecuta línea a línea, así que puedes ver el resultado al instante
- **Tipado dinámico**: no necesitas declarar el tipo de una variable. Python lo deduce en tiempo de ejecución
- **Multiparadigma**: admite programación procedural y programación orientada a objetos (POO)
- **Multiplataforma**: el mismo código funciona en Windows, macOS y Linux sin cambios

### ¿Es Python un lenguaje lento?

Python es interpretado, así que introduce cierta sobrecarga frente a lenguajes compilados como C. Aun así, librerías como NumPy o Pandas están construidas sobre C y ofrecen un rendimiento excelente para cálculo numérico. Para la mayoría de proyectos, la simplicidad y la velocidad de desarrollo compensan de sobra esa diferencia.

### Primer contacto: instalación y editor

1. Descarga Python desde [python.org](https://www.python.org) (en Windows, elige la versión de 64 bits)
2. Elige un editor o un entorno de desarrollo integrado (IDE): Visual Studio Code, PyCharm o el propio IDLE que trae Python
3. Guarda tus programas con extensión `.py`
4. Evita poner a tus archivos el mismo nombre que una librería que vayas a importar (por ejemplo, no llames `random.py` a tu script)

### Tu primer programa

```python
print("Hola, universo")
```

La función `print()` muestra texto en la consola. Es la herramienta más básica para depurar (revisar qué está pasando dentro de tu código).

### Tres pilares para aprender a programar

- **Sintaxis**: las reglas del lenguaje. Un error de sintaxis impide que el programa se ejecute
- **Lógica**: la capacidad de descomponer un problema en pasos pequeños y ordenados
- **Resolución de problemas**: elegir la estructura de datos y el algoritmo adecuados para cada situación

---

## Lección 2 · Primeros pasos: tuplas, cadenas y bucles

### Tuplas

Una tupla es una colección ordenada de elementos, escrita entre paréntesis `()`. Es **inmutable**: una vez creada, no puedes cambiar sus valores.

```python
mi_tupla = (1, "manzana", 3.14, True)
print(mi_tupla[0])   # 1
print(mi_tupla[1])   # manzana
```

Usa tuplas para agrupar datos que no deban cambiar, como coordenadas fijas o los colores de un semáforo.

### Cadenas de texto (strings)

Una cadena es una secuencia de caracteres entre comillas simples `''` o dobles `""`. También es inmutable.

```python
saludo = "Hola, mundo!"
print(saludo[0])           # H
print(saludo + " ¿Qué tal?")   # concatenación
print(saludo[0:4])         # slicing: "Hola"
```

### Bucles

Los bucles repiten un bloque de instrucciones. Python ofrece dos tipos.

**`for`**, para recorrer una secuencia conocida:

```python
for letra in "Python":
    print(letra)
```

**`while`**, para repetir mientras se cumpla una condición:

```python
contador = 0
while contador < 5:
    print(contador)
    contador += 1
```

**Consejo**: usa `for` cuando sepas por cuántos elementos vas a iterar, y `while` cuando la repetición dependa de una condición que cambia en tiempo de ejecución (por ejemplo, "hasta que el jugador gane").

---

## Lección 3 · Listas y el primer boceto del Tic Tac Toe

### Listas

Una lista es una colección **mutable** (modificable) de elementos, escrita entre corchetes `[]`.

```python
mi_lista = [1, 2, 3, "manzana", "plátano"]
mi_lista.append("pera")     # añade un elemento al final
mi_lista.remove(2)          # elimina el valor 2
print(len(mi_lista))        # longitud de la lista
```

Convertir una tupla en lista es tan sencillo como:

```python
mi_tupla = (1, 2, 3, 4, 5)
mi_lista = list(mi_tupla)
```

### Listas de listas: representar un tablero

Para representar el tablero del Tic Tac Toe (una cuadrícula de 3×3), la estructura más natural es una **lista de listas**: cada lista interior representa una fila.

```python
tablero = [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]]
```

Usamos `0` para las casillas vacías y, más adelante, `1` y `2` (o `"X"` y `"O"`) para los movimientos de cada jugador. Trabajar con números en vez de caracteres facilita las comparaciones dentro del código.

### Mostrar el tablero

```python
for fila in tablero:
    for casilla in fila:
        print(casilla, end=" ")
    print()
```

### Ventajas de la lista de listas

- Se adapta a tableros de cualquier tamaño (3×3, 4×4, 5×5...)
- El acceso a una casilla es directo: `tablero[fila][columna]`
- Refleja visualmente la disposición del tablero, algo muy útil para quien empieza
- Facilita comprobar filas, columnas y diagonales con bucles anidados

### Inicialización dinámica y validación

Una versión más robusta de la función de inicialización acepta el tamaño del tablero como parámetro y valida los datos de entrada:

```python
def inicializar_tablero(filas, columnas):
    if not isinstance(filas, int) or not isinstance(columnas, int):
        raise ValueError("Filas y columnas deben ser números enteros")
    if filas <= 0 or columnas <= 0:
        raise ValueError("Filas y columnas deben ser positivos")
    return [[" " for _ in range(columnas)] for _ in range(filas)]
```

Este patrón —comprobar los datos de entrada antes de usarlos— evita errores difíciles de rastrear más adelante.

---

## Lección 4 · Funciones incorporadas (built-in functions)

Python trae de serie decenas de funciones listas para usar, sin necesidad de importar nada.

### Funciones matemáticas

| Función | Qué hace |
|---|---|
| `abs(x)` | Valor absoluto de `x` |
| `pow(x, y)` | `x` elevado a `y` |
| `max(iterable)` | El valor máximo |
| `min(iterable)` | El valor mínimo |
| `round(x, n)` | `x` redondeado a `n` decimales |

### Funciones de cadenas

| Función | Qué hace |
|---|---|
| `len(cadena)` | Longitud de la cadena |
| `str(objeto)` | Convierte un objeto en texto |
| `cadena.upper()` | Pasa el texto a mayúsculas |
| `cadena.lower()` | Pasa el texto a minúsculas |

### Otras funciones muy útiles para el proyecto del Tic Tac Toe

- `range(n)`: genera una secuencia de números de 0 a n-1
- `enumerate(iterable)`: recorre una secuencia devolviendo a la vez el índice y el valor
- `zip(a, b)`: combina dos (o más) secuencias en pares, elemento a elemento
- `reversed(iterable)`: devuelve un iterador que recorre la secuencia en orden inverso
- `all(iterable)`: devuelve `True` si todos los elementos son verdaderos

```python
colores = ["rojo", "verde", "azul"]
for indice, color in enumerate(colores):
    print(f"El color en la posición {indice} es {color}")
```

---

## Lección 5 · Índices y slices (cortes)

En Python, la indexación empieza en cero. El primer elemento de una lista o cadena está en la posición cero.

```python
numeros = [1, 2, 3, 4, 5]
numeros[0]      # 1
numeros[1]      # 2
```

### Slicing

La sintaxis `[inicio:fin]` extrae un tramo de la secuencia. El elemento `fin` **no** se incluye.

```python
numeros[1:4]    # [2, 3, 4]
numeros[:3]     # [1, 2, 3]  (si omites el inicio, se asume 0)
numeros[2:]     # [3, 4, 5]  (si omites el fin, se asume el final)
```

También puedes indicar un paso (`step`) con un segundo signo `:`.

```python
numeros[0:5:2]  # [1, 3, 5]  (salta de dos en dos)
```

### Índices negativos

`-1` señala el último elemento, `-2` el penúltimo, y así sucesivamente.

```python
numeros[-1]     # 5
numeros[-3:-1]  # [3, 4]
```

Los índices y los slices son la base para recorrer filas, columnas y diagonales del tablero del Tic Tac Toe.

---

## Lección 6 · Funciones propias

Una función es un bloque de código con nombre que realiza una tarea concreta. Ayuda a dividir un problema grande en piezas pequeñas y reutilizables.

```python
def sumar(a, b):
    return a + b

resultado = sumar(3, 4)
print(resultado)   # 7
```

- `def` define la función
- Entre paréntesis van los **parámetros** (los datos de entrada)
- `return` indica el valor de salida. Si no hay `return`, la función devuelve `None`

---

## Lección 7 · Parámetros de función y tipado

### Valores por defecto

```python
def saludar(nombre="Anónimo"):
    print(f"Hola, {nombre}!")

saludar()          # Hola, Anónimo!
saludar("Alicia")  # Hola, Alicia!
```

### Argumentos con nombre (keyword arguments)

Cuando una función tiene muchos parámetros, puedes indicar a qué parámetro corresponde cada valor, sin depender del orden:

```python
def crear_jugador(nombre, simbolo="X"):
    return {"nombre": nombre, "simbolo": simbolo}

crear_jugador(nombre="Bea", simbolo="O")
```

Aunque Python no obliga a declarar tipos, es buena práctica documentar qué tipo de dato espera cada parámetro (mediante *type hints*), para que el código sea más fácil de mantener:

```python
def sumar(a: int, b: int) -> int:
    return a + b
```

---

## Lección 8 · Mutabilidad revisitada

Un objeto es **mutable** si se puede modificar después de crearlo, e **inmutable** si no.

| Mutables | Inmutables |
|---|---|
| Listas | Cadenas |
| Diccionarios | Tuplas |
| Conjuntos (sets) | Números |

Este detalle importa mucho cuando dos variables apuntan al mismo objeto:

```python
lista1 = [1, 2, 3]
lista2 = lista1        # ambas variables apuntan a la misma lista
lista1.append(4)
print(lista2)           # [1, 2, 3, 4] -> también cambió
```

Si quieres una copia independiente, usa `lista2 = lista1.copy()` o `lista2 = list(lista1)`.

**Por qué importa en el Tic Tac Toe**: si guardas el tablero en dos variables sin copiarlo, cualquier cambio en una se refleja en la otra. Ten esto en cuenta si implementas la función "deshacer movimiento" o "guardar partida".

---

## Lección 9 · Manejo de errores

Cuando algo falla durante la ejecución, Python lanza una **excepción**. El bloque `try`/`except` permite capturarla y responder sin que el programa se detenga.

```python
try:
    fila = int(input("Elige una fila: "))
except ValueError:
    print("Eso no es un número. Inténtalo de nuevo")
```

### Excepciones habituales

| Excepción | Cuándo aparece |
|---|---|
| `ZeroDivisionError` | Al dividir entre cero |
| `TypeError` | Al aplicar una operación a un tipo de dato incorrecto |
| `ValueError` | Al recibir un valor de tipo correcto pero fuera de rango o formato |
| `FileNotFoundError` | Al intentar abrir un archivo que no existe |

También puedes definir excepciones propias creando una clase que herede de `Exception`, y usar varios bloques `except` para tratar cada error de forma distinta.

**Por qué importa en el Tic Tac Toe**: cuando pidas al jugador una fila y una columna por teclado, el texto introducido puede no ser un número. El `try`/`except` evita que el juego se cierre de golpe ante una entrada inválida.

---

## Lección 10 · Cómo calcular el ganador en horizontal

El objetivo es comprobar si alguna fila del tablero tiene sus tres casillas iguales y ocupadas.

### Primer intento, con una bandera de control

```python
def ganador_horizontal(tablero):
    for fila in tablero:
        sin_coincidencia = False
        if fila[0] != "":
            for i in range(1, len(fila)):
                if fila[i] != fila[0]:
                    sin_coincidencia = True
                    break
            if not sin_coincidencia:
                return fila[0]
    return None
```

### Versión mejorada, con `all()`

```python
def ganador_horizontal(tablero):
    for fila in tablero:
        if fila[0] != "" and all(casilla == fila[0] for casilla in fila):
            return fila[0]
    return None
```

### Versión más concisa, con `count()`

El método `count()` cuenta cuántas veces aparece un valor en una lista. Si el conteo del primer elemento coincide con la longitud de la fila, todas las casillas son iguales:

```python
def ganador_horizontal(tablero):
    for fila in tablero:
        if fila.count("X") == len(fila):
            return "X"
        if fila.count("O") == len(fila):
            return "O"
    return None
```

**Idea clave**: cuando una primera solución te parezca larga o confusa, busca alternativas con funciones incorporadas de Python. Suelen ser más cortas, más legibles y más rápidas.

---

## Lección 11 · Cómo calcular el ganador en vertical

Comprobar las columnas es equivalente a comprobar las filas, pero recorriendo el tablero por índice de columna en vez de por fila.

```python
def ganador_vertical(tablero):
    tamano = len(tablero)
    for columna in range(tamano):
        valores = [tablero[fila][columna] for fila in range(tamano)]
        if valores[0] != "" and all(valor == valores[0] for valor in valores):
            return valores[0]
    return None
```

**Truco de listas por comprensión**: `[tablero[fila][columna] for fila in range(tamano)]` construye la columna entera en una sola línea, recorriendo cada fila y quedándose con el elemento de la columna que nos interesa.

Optimizar el código con librerías especializadas (NumPy para cálculo numérico, Pandas para análisis de datos, o el módulo `multiprocessing` para paralelizar tareas) tiene sentido cuando el proyecto crece, pero para un tablero de Tic Tac Toe el enfoque anterior es más que suficiente.

---

## Lección 12 · El algoritmo del ganador en diagonal

Hay dos diagonales en un tablero cuadrado:

- **Diagonal principal**: posiciones (0,0), (1,1), (2,2)...
- **Diagonal secundaria**: posiciones (0, n-1), (1, n-2), (2, n-3)...

```python
def ganador_diagonal(tablero):
    tamano = len(tablero)

    principal = [tablero[i][i] for i in range(tamano)]
    if principal[0] != "" and all(valor == principal[0] for valor in principal):
        return principal[0]

    secundaria = [tablero[i][tamano - 1 - i] for i in range(tamano)]
    if secundaria[0] != "" and all(valor == secundaria[0] for valor in secundaria):
        return secundaria[0]

    return None
```

### Herramientas de Python que facilitan este algoritmo

- **`enumerate()`**: recorre una lista devolviendo índice y valor a la vez, así evitas gestionar un contador manual
- **`reversed()`**: devuelve un iterador con los elementos en orden inverso. Si necesitas indexar el resultado, conviértelo antes en lista con `list(reversed(...))`
- **`zip()`**: combina dos o más secuencias en tuplas, elemento a elemento. Es perfecto para recorrer, por ejemplo, los índices de fila y de columna de una diagonal al mismo tiempo

```python
filas = range(3)
columnas_invertidas = reversed(range(3))
for fila, columna in zip(filas, columnas_invertidas):
    print(fila, columna)
```

---

## Lección 13 · Iteradores e iterables

- **Iterable**: cualquier objeto que se puede recorrer con un bucle (listas, tuplas, cadenas, diccionarios)
- **Iterador**: el objeto que realmente va devolviendo un elemento cada vez que lo llamas con `next()`, hasta que se agota y lanza `StopIteration`

```python
numeros = [1, 2, 3]
iterador = iter(numeros)
print(next(iterador))   # 1
print(next(iterador))   # 2
print(next(iterador))   # 3
```

Cuando escribes `for elemento in lista:`, Python está usando `iter()` y `next()` por debajo, de forma automática. Entender este mecanismo ayuda a comprender por qué funciones como `zip()`, `enumerate()` o `reversed()` devuelven objetos "perezosos" que solo se convierten en listas cuando lo pides de forma explícita.

---

## Lección 14 · Cierre del proyecto Tic Tac Toe

Con todos los conceptos anteriores puedes construir el juego completo. Las piezas que hay que ensamblar son:

1. **Tablero dinámico**: una lista de listas cuyo tamaño puede elegir quien juega
2. **Función `todas_iguales(lista)`**: comprueba si todos los elementos de una lista son iguales y distintos de vacío. Se reutiliza en las comprobaciones horizontal, vertical y diagonal, evitando repetir código
3. **Validación de movimientos**: antes de aceptar un movimiento, comprueba que la casilla está dentro del tablero y que está vacía. Si no lo está, pide el dato de nuevo sin perder el turno
4. **Bucle principal**: alterna turnos entre dos jugadores hasta que haya un ganador o el tablero se llene (empate)
5. **Mensaje de tipo de victoria**: indica si la partida se ganó en horizontal, vertical o diagonal
6. **Opción de repetir partida**: al finalizar, pregunta si quieren jugar de nuevo. Cualquier respuesta distinta de "sí" o "no" debe gestionarse con un mensaje claro, en vez de romper el programa

### Buenas prácticas al consolidar el código

- Evita repetir la misma lógica de comprobación tres veces: extrae una función común (`todas_iguales`) y reutilízala
- Da nombres descriptivos a las variables. Un nombre como `L` o `g` dificulta la lectura del código dentro de unas semanas
- Añade comentarios solo donde el porqué de una decisión no sea evidente, no para repetir lo que el código ya dice

---

## Lección 15 · Conclusión y siguientes pasos

A lo largo de este manual has recorrido los fundamentos de Python: variables, tipos de datos, tuplas, cadenas, listas, bucles, funciones, índices, manejo de errores e iteradores. Con ellos has construido, pieza a pieza, un juego completo de Tic Tac Toe.

### Ideas para seguir practicando

- Convierte el juego en una clase (introduce conceptos de programación orientada a objetos, POO)
- Añade una interfaz gráfica con Tkinter o Pygame
- Incorpora colores en la consola con el paquete `colorama`
- Prueba a hacer que el ordenador juegue contra ti (un primer paso hacia la inteligencia artificial aplicada a juegos)
- Sube tu proyecto a un repositorio Git para practicar control de versiones

**¡Sigue explorando y construye tu propio proyecto!** La mejor forma de consolidar lo aprendido es aplicarlo a una idea que te motive.

---

## Glosario rápido

- **Variable**: nombre que guarda un valor en memoria
- **Función**: bloque de código reutilizable que recibe parámetros y puede devolver un resultado
- **Lista**: colección ordenada y mutable de elementos
- **Tupla**: colección ordenada e inmutable de elementos
- **Bucle**: estructura que repite instrucciones
- **Excepción**: error que ocurre durante la ejecución del programa
- **Índice**: posición de un elemento dentro de una secuencia, empezando en cero
- **Slice**: fragmento extraído de una secuencia mediante `[inicio:fin:paso]`
- **Iterable**: objeto que se puede recorrer
- **Iterador**: objeto que entrega los elementos de un iterable uno a uno

## Recursos adicionales

- Documentación oficial: [docs.python.org](https://docs.python.org)
- Índice de paquetes: [pypi.org](https://pypi.org)
- Comunidad de preguntas y respuestas: [stackoverflow.com](https://stackoverflow.com)

---

*Manual elaborado a partir del material didáctico del programa EITC/CP/PPF Python Programming Fundamentals, del European IT Certification Institute (EITCI).*
