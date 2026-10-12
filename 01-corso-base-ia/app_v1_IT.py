# -*- coding: utf-8 -*-
"""Prototipo v1 del chatbot di FAQ (Sessione 7).
Versione minima: una domanda, una risposta. Senza cronologia per ora.

Eseguire con:  streamlit run app_v1_IT.py
"""
import os
import streamlit as st
from pypdf import PdfReader

st.title("Assistente FAQ (prototipo)")

# 1. Leggere il PDF delle FAQ
lector = PdfReader("faq_horta_verde.pdf")
texto_faq = ""
for pagina in lector.pages:
    texto_faq += pagina.extract_text() + "\n"

# 2. Chiedere la chiave di Claude (solo per il prototipo; poi la miglioreremo)
clave = st.text_input("La tua chiave Anthropic", type="password")

# 3. Casella per la domanda del cliente
pregunta = st.text_input("Come posso aiutarti?")

# 4. Al clic del pulsante, interroghiamo l'LLM
if st.button("Chiedi") and pregunta and clave:
    import anthropic
    cliente = anthropic.Anthropic(api_key=clave)
    system = (
        "Sei l'assistente dell'assistenza clienti di Horta Verde. "
        "Rispondi in italiano, breve e cordiale, usando SOLO questa informazione:\n\n"
        + texto_faq
    )
    respuesta = cliente.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": pregunta}],
    )
    st.write(respuesta.content[0].text)
