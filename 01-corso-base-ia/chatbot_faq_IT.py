# -*- coding: utf-8 -*-
"""
Chatbot di Domande Frequenti (FAQ) per l'Assistenza Clienti.
Progetto finale del corso · Modulo 4.

Come eseguirlo:
    1) pip install -r requirements.txt
    2) streamlit run chatbot_faq_IT.py

L'app legge un PDF di FAQ, lo passa come contesto a un LLM e risponde
ai dubbi del cliente basandosi SOLO su quell'informazione.

Supporta tre fornitori: Claude (Anthropic), OpenRouter (gratis) e Ollama (locale).
"""
import os
import streamlit as st
from pypdf import PdfReader

# --------------------------------------------------------------------------
# 1. CONFIGURAZIONE DI BASE DELLA PAGINA
# --------------------------------------------------------------------------
st.set_page_config(page_title="Assistente FAQ", page_icon="💬")
RUTA_FAQ = "faq_horta_verde.pdf"   # sostituiscilo con il tuo PDF


# --------------------------------------------------------------------------
# 2. LEGGERE IL PDF DELLE FAQ (in cache per non rileggerlo a ogni interazione)
# --------------------------------------------------------------------------
@st.cache_data
def cargar_faq(ruta):
    """Legge il PDF e restituisce tutto il suo testo come stringa."""
    lector = PdfReader(ruta)
    texto = ""
    for pagina in lector.pages:
        texto += pagina.extract_text() + "\n"
    return texto


# --------------------------------------------------------------------------
# 3. CHIAMARE L'LLM (tre possibili fornitori)
# --------------------------------------------------------------------------
def responder(historial, system, proveedor, clave):
    """Invia il system prompt + la cronologia al fornitore scelto e restituisce il testo."""
    if proveedor == "Claude (Anthropic)":
        import anthropic
        cliente = anthropic.Anthropic(api_key=clave)
        r = cliente.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            system=system,
            messages=historial,
        )
        return r.content[0].text

    elif proveedor == "OpenRouter (gratis)":
        from openai import OpenAI
        cliente = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=clave)
        mensajes = [{"role": "system", "content": system}] + historial
        r = cliente.chat.completions.create(
            model="meta-llama/llama-4-maverick:free",
            messages=mensajes,
            max_tokens=500,
        )
        return r.choices[0].message.content

    else:  # Ollama (locale)
        from ollama import chat
        mensajes = [{"role": "system", "content": system}] + historial
        return chat(model="llama3.2", messages=mensajes).message.content


# --------------------------------------------------------------------------
# 4. BARRA LATERALE: scegliere il fornitore e inserire la chiave
# --------------------------------------------------------------------------
st.sidebar.header("Configurazione")
proveedor = st.sidebar.selectbox(
    "Fornitore del modello",
    ["Claude (Anthropic)", "OpenRouter (gratis)", "Ollama (locale)"],
)

def leer_secreto(nombre):
    """Legge una chiave da st.secrets senza fallire se il file dei secrets non esiste."""
    try:
        return st.secrets[nombre]
    except Exception:
        return ""

# La chiave viene cercata prima nei 'secrets' (per la pubblicazione), poi nelle
# variabili d'ambiente e, se non c'è, viene chiesta nella barra laterale
clave = ""
if proveedor != "Ollama (locale)":
    nombre_secreto = "ANTHROPIC_API_KEY" if proveedor.startswith("Claude") else "OPENROUTER_API_KEY"
    clave = leer_secreto(nombre_secreto) or os.environ.get(nombre_secreto, "")
    if not clave:
        clave = st.sidebar.text_input("Chiave API", type="password")

if st.sidebar.button("🗑️ Riavvia conversazione"):
    st.session_state.mensajes = []
    st.rerun()


# --------------------------------------------------------------------------
# 5. ISTRUZIONI DELL'ASSISTENTE (system prompt)
# --------------------------------------------------------------------------
texto_faq = cargar_faq(RUTA_FAQ)

SYSTEM = f"""Sei l'assistente virtuale dell'assistenza clienti di Horta Verde, un negozio online.
Il tuo compito è rispondere ai dubbi dei clienti in modo cordiale, chiaro e breve, in italiano.

Regole importanti:
- Rispondi ESCLUSIVAMENTE con le informazioni delle Domande Frequenti riportate più sotto.
- Se la risposta non è nelle FAQ, dillo con onestà e invita a contattare l'assistenza
  clienti (hola@hortaverde.example o 900 123 456). Non inventare dati.
- Sii vicino e professionale. Non usare tecnicismi.

=== DOMANDE FREQUENTI ===
{texto_faq}
=== FINE DELLE FAQ ==="""


# --------------------------------------------------------------------------
# 6. INTERFACCIA DI CHAT
# --------------------------------------------------------------------------
st.title("💬 Assistente dell'assistenza clienti")
st.caption("Chiedimi di ordini, spedizioni, resi, pagamenti o del tuo account.")

# Inizializzare la cronologia della conversazione
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrare i messaggi precedenti
for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Casella di input dell'utente
pregunta = st.chat_input("Scrivi la tua domanda...")
if pregunta:
    # Salvare e mostrare la domanda dell'utente
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Verificare che ci sia la chiave se il fornitore la richiede
    if proveedor != "Ollama (locale)" and not clave:
        with st.chat_message("assistant"):
            st.warning("Inserisci la tua chiave API nella barra laterale per poter rispondere.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Sto pensando..."):
                try:
                    respuesta = responder(st.session_state.mensajes, SYSTEM, proveedor, clave)
                except Exception as error:
                    respuesta = (f"Mi dispiace, si è verificato un problema tecnico "
                                 f"({type(error).__name__}). Riprova tra qualche secondo.")
            st.markdown(respuesta)
        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
