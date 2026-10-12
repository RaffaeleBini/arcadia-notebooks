# -*- coding: utf-8 -*-
"""Prototipo v1 del chatbot de FAQ (Sesión 7).
Versión mínima: una pregunta, una respuesta. Sin historial todavía.

Ejecutar con:  streamlit run app_v1.py
"""
import os
import streamlit as st
from pypdf import PdfReader

st.title("Asistente de FAQ (prototipo)")

# 1. Leer el PDF de FAQ
lector = PdfReader("faq_horta_verde.pdf")
texto_faq = ""
for pagina in lector.pages:
    texto_faq += pagina.extract_text() + "\n"

# 2. Pedir la clave de Claude (solo para el prototipo; luego mejoraremos esto)
clave = st.text_input("Tu clave de Anthropic", type="password")

# 3. Caja para la pregunta del cliente
pregunta = st.text_input("¿En qué puedo ayudarte?")

# 4. Al pulsar el botón, preguntamos al LLM
if st.button("Preguntar") and pregunta and clave:
    import anthropic
    cliente = anthropic.Anthropic(api_key=clave)
    system = (
        "Eres el asistente de atención al cliente de Horta Verde. "
        "Responde en español, breve y amable, usando SOLO esta información:\n\n"
        + texto_faq
    )
    respuesta = cliente.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": pregunta}],
    )
    st.write(respuesta.content[0].text)
