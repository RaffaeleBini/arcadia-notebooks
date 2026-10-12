# DocBot — Chatbot RAG con Memoria

Assistente conversazionale che risponde a domande su documenti indicizzati,
ricorda gli utenti tra una sessione e l'altra e valuta autonomamente la
propria qualità.

## Installazione

```bash
cd proyectos/docbot
pip install -r requirements.txt
```

Crea un file `.env` in questa cartella:
```
OPENAI_API_KEY=sk-...
```

## Avviare l'app

```bash
streamlit run app.py
```

Apri http://localhost:8501 nel tuo browser.

## Uso rapido

1. Scrivi il tuo nome utente nel pannello laterale
2. Premi **Avvia DocBot**
3. Seleziona gli articoli da indicizzare e premi **Indicizza selezione**
4. Scrivi la tua prima domanda nella chat

## Uso da Python

```python
from docbot import DocBot

bot = DocBot(user_id="mio_utente")

# Indicizza documenti
bot.ingestar([
    {"tipo": "wikipedia", "valor": "FastAPI",    "titulo": "FastAPI",    "categoria": "framework"},
    {"tipo": "wikipedia", "valor": "PostgreSQL", "titulo": "PostgreSQL", "categoria": "base_datos"},
])

# Conversa
r = bot.responder("What is FastAPI and when should I use it?")
print(r.respuesta)
print(f"Fonti: {r.fuentes}")
print(f"Faithfulness: {r.faithfulness}")

# Chiude e salva la memoria
bot.cerrar_sesion()

# Nella prossima sessione, il bot ricorderà l'utente
bot2 = DocBot(user_id="mio_utente")
r2 = bot2.responder("Do you remember what we discussed before?")
```

## Architettura

```
docbot/
├── chatbot.py      # Classe principale DocBot
├── ingesta.py      # Loader → Cleaner → Chunker → Indexer
├── retrieval.py    # Embed → Retrieve → Rewrite → Generate
├── memoria.py      # Sessione persistita + memoria semantica (Chroma)
└── evaluacion.py   # Faithfulness con giudice LLM
```

## Corpus di documenti

Per default usa articoli di Wikipedia (non serve alcun file esterno).
Per indicizzare i tuoi propri documenti in Markdown o testo semplice:

```python
bot.ingestar([
    {"tipo": "markdown", "valor": "percorso/al/documento.md", "titulo": "Mio doc", "categoria": "interno"},
    {"tipo": "txt",      "valor": "percorso/al/testo.txt",    "titulo": "Altro",   "categoria": "interno"},
])
```

## Costo approssimativo

Con `gpt-4o-mini` e un corpus di 500 chunk:
- Ingestione (500 chunk): ~$0,001
- Per domanda (retrieval + generazione): ~$0,0003
- 1.000 domande/giorno: ~$0,30/giorno → ~$9/mese

## Limitazioni note

- Non risponde a domande che richiedono informazioni fuori dal corpus indicizzato (per design)
- La valutazione online della faithfulness aggiunge ~1s di latenza per domanda
- La memoria semantica impiega 1-2s alla chiusura della sessione

---

Progetto 1 — Piano di Formazione AI Engineering (Settimana 12)
