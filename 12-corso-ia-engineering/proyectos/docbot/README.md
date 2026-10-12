# DocBot — RAG Chatbot con Memoria

Asistente conversacional que responde preguntas sobre documentos indexados,
recuerda a los usuarios entre sesiones y evalúa su propia calidad.

## Instalación

```bash
cd proyectos/docbot
pip install -r requirements.txt
```

Crea un archivo `.env` en esta carpeta:
```
OPENAI_API_KEY=sk-...
```

## Arrancar la app

```bash
streamlit run app.py
```

Abre http://localhost:8501 en tu navegador.

## Uso rápido

1. Escribe tu nombre de usuario en el panel lateral
2. Pulsa **Iniciar DocBot**
3. Selecciona los artículos a indexar y pulsa **Indexar selección**
4. Escribe tu primera pregunta en el chat

## Uso desde Python

```python
from docbot import DocBot

bot = DocBot(user_id="mi_usuario")

# Indexa documentos
bot.ingestar([
    {"tipo": "wikipedia", "valor": "FastAPI",    "titulo": "FastAPI",    "categoria": "framework"},
    {"tipo": "wikipedia", "valor": "PostgreSQL", "titulo": "PostgreSQL", "categoria": "base_datos"},
])

# Conversa
r = bot.responder("What is FastAPI and when should I use it?")
print(r.respuesta)
print(f"Fuentes: {r.fuentes}")
print(f"Faithfulness: {r.faithfulness}")

# Cierra y guarda memoria
bot.cerrar_sesion()

# En la próxima sesión, el bot recordará al usuario
bot2 = DocBot(user_id="mi_usuario")
r2 = bot2.responder("Do you remember what we discussed before?")
```

## Arquitectura

```
docbot/
├── chatbot.py      # Clase principal DocBot
├── ingesta.py      # Loader → Cleaner → Chunker → Indexer
├── retrieval.py    # Embed → Retrieve → Rewrite → Generate
├── memoria.py      # Sesión persistida + memoria semántica (Chroma)
└── evaluacion.py   # Faithfulness con juez LLM
```

## Corpus de documentos

Por defecto usa artículos de Wikipedia (no necesita archivos externos).
Para indexar tus propios documentos en Markdown o texto plano:

```python
bot.ingestar([
    {"tipo": "markdown", "valor": "ruta/al/documento.md", "titulo": "Mi doc", "categoria": "interno"},
    {"tipo": "txt",      "valor": "ruta/al/texto.txt",    "titulo": "Otro",   "categoria": "interno"},
])
```

## Coste aproximado

Con `gpt-4o-mini` y un corpus de 500 chunks:
- Ingesta (500 chunks): ~$0.001
- Por pregunta (retrieval + generación): ~$0.0003
- 1.000 preguntas/día: ~$0.30/día → ~$9/mes

## Limitaciones conocidas

- No responde preguntas que requieran información fuera del corpus indexado (por diseño)
- La evaluación online de faithfulness añade ~1s de latencia por pregunta
- La memoria semántica tarda 1-2s al cerrar la sesión

---

Proyecto 1 — Plan de Formación AI Engineering (Semana 12)
