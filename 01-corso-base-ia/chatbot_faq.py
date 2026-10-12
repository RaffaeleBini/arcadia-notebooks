# -*- coding: utf-8 -*-
"""
Chatbot de Preguntas Frecuentes (FAQ) para Atención al Cliente.
Proyecto final del curso · Módulo 4.

Cómo ejecutarlo:
    1) pip install -r requirements.txt
    2) streamlit run chatbot_faq.py

La app lee un PDF de FAQ, lo entrega como contexto a un LLM y responde
las dudas del cliente basándose SOLO en esa información.

Soporta tres proveedores: Claude (Anthropic), OpenRouter (gratis) y Ollama (local).
"""
import os
import streamlit as st
from pypdf import PdfReader

# --------------------------------------------------------------------------
# 1. CONFIGURACIÓN BÁSICA DE LA PÁGINA
# --------------------------------------------------------------------------
st.set_page_config(page_title="Asistente de FAQ", page_icon="💬")
RUTA_FAQ = "faq_horta_verde.pdf"   # cambia esto por tu propio PDF


# --------------------------------------------------------------------------
# 2. LEER EL PDF DE FAQ (en caché para no releerlo en cada interacción)
# --------------------------------------------------------------------------
@st.cache_data
def cargar_faq(ruta):
    """Lee el PDF y devuelve todo su texto como una cadena."""
    lector = PdfReader(ruta)
    texto = ""
    for pagina in lector.pages:
        texto += pagina.extract_text() + "\n"
    return texto


# --------------------------------------------------------------------------
# 3. LLAMAR AL LLM (tres proveedores posibles)
# --------------------------------------------------------------------------
def responder(historial, system, proveedor, clave):
    """Envía el system prompt + el historial al proveedor elegido y devuelve el texto."""
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

    else:  # Ollama (local)
        from ollama import chat
        mensajes = [{"role": "system", "content": system}] + historial
        return chat(model="llama3.2", messages=mensajes).message.content


# --------------------------------------------------------------------------
# 4. BARRA LATERAL: elegir proveedor e introducir la clave
# --------------------------------------------------------------------------
st.sidebar.header("Configuración")
proveedor = st.sidebar.selectbox(
    "Proveedor del modelo",
    ["Claude (Anthropic)", "OpenRouter (gratis)", "Ollama (local)"],
)

def leer_secreto(nombre):
    """Lee una clave de st.secrets sin fallar si no existe el archivo de secretos."""
    try:
        return st.secrets[nombre]
    except Exception:
        return ""

# La clave se busca primero en los 'secrets' (para el despliegue), luego en las
# variables de entorno y, si no, se pide en la barra lateral
clave = ""
if proveedor != "Ollama (local)":
    nombre_secreto = "ANTHROPIC_API_KEY" if proveedor.startswith("Claude") else "OPENROUTER_API_KEY"
    clave = leer_secreto(nombre_secreto) or os.environ.get(nombre_secreto, "")
    if not clave:
        clave = st.sidebar.text_input("Clave API", type="password")

if st.sidebar.button("🗑️ Reiniciar conversación"):
    st.session_state.mensajes = []
    st.rerun()


# --------------------------------------------------------------------------
# 5. INSTRUCCIONES DEL ASISTENTE (system prompt)
# --------------------------------------------------------------------------
texto_faq = cargar_faq(RUTA_FAQ)

SYSTEM = f"""Eres el asistente virtual de atención al cliente de Horta Verde, una tienda online.
Tu tarea es responder las dudas de los clientes de forma amable, clara y breve, en español.

Reglas importantes:
- Responde ÚNICAMENTE con la información de las Preguntas Frecuentes que aparecen más abajo.
- Si la respuesta no está en las FAQ, dilo con honestidad e invita a contactar con atención
  al cliente (hola@hortaverde.example o 900 123 456). No te inventes datos.
- Sé cercano y profesional. No uses tecnicismos.

=== PREGUNTAS FRECUENTES ===
{texto_faq}
=== FIN DE LAS FAQ ==="""


# --------------------------------------------------------------------------
# 6. INTERFAZ DE CHAT
# --------------------------------------------------------------------------
st.title("💬 Asistente de atención al cliente")
st.caption("Pregúntame sobre pedidos, envíos, devoluciones, pagos o tu cuenta.")

# Inicializar el historial de la conversación
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar los mensajes anteriores
for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Caja de entrada del usuario
pregunta = st.chat_input("Escribe tu pregunta...")
if pregunta:
    # Guardar y mostrar la pregunta del usuario
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Validar que hay clave si el proveedor la necesita
    if proveedor != "Ollama (local)" and not clave:
        with st.chat_message("assistant"):
            st.warning("Introduce tu clave API en la barra lateral para poder responder.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    respuesta = responder(st.session_state.mensajes, SYSTEM, proveedor, clave)
                except Exception as error:
                    respuesta = (f"Lo siento, ha ocurrido un problema técnico "
                                 f"({type(error).__name__}). Inténtalo de nuevo en unos segundos.")
            st.markdown(respuesta)
        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
