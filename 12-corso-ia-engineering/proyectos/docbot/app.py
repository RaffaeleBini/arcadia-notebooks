"""
app.py — Interfaz Streamlit para DocBot.

Arrancar con:
    cd proyectos/docbot
    streamlit run app.py
"""

import streamlit as st
from docbot import DocBot, RespuestaDocBot

# ── Configuración de la página ────────────────────────────────────────────────

st.set_page_config(
    page_title="DocBot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estado de sesión ──────────────────────────────────────────────────────────

def inicializar_estado():
    if "bot" not in st.session_state:
        st.session_state.bot = None
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = "usuario"
    if "ingesta_hecha" not in st.session_state:
        st.session_state.ingesta_hecha = False


inicializar_estado()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 DocBot")
    st.caption("RAG chatbot con memoria")

    # Usuario
    user_id = st.text_input("Tu nombre de usuario", value=st.session_state.user_id, key="input_user")
    if user_id != st.session_state.user_id:
        st.session_state.user_id     = user_id
        st.session_state.bot         = None
        st.session_state.mensajes    = []
        st.session_state.ingesta_hecha = False

    st.divider()

    # Inicializar el bot
    if st.session_state.bot is None:
        evaluar = st.checkbox("Evaluar faithfulness online (más lento)", value=False)
        if st.button("🚀 Iniciar DocBot", use_container_width=True):
            with st.spinner("Iniciando..."):
                st.session_state.bot = DocBot(
                    user_id=st.session_state.user_id,
                    evaluar_online=evaluar
                )
            st.success(f"DocBot iniciado | Sesión: {st.session_state.bot.session_id[:12]}")
            st.rerun()

    if st.session_state.bot is not None:
        bot: DocBot = st.session_state.bot
        st.success(f"✓ Activo | {bot.n_docs} chunks indexados")
        st.caption(f"Sesión: `{bot.session_id[:12]}...`")

        st.divider()

        # Ingesta de documentos
        st.subheader("📥 Indexar documentos")
        with st.expander("Fuentes predefinidas (Wikipedia)", expanded=not st.session_state.ingesta_hecha):
            CORPUS_PREDEFINIDO = [
                {"tipo": "wikipedia", "valor": "FastAPI",                "titulo": "FastAPI",                "categoria": "framework"},
                {"tipo": "wikipedia", "valor": "Pydantic",               "titulo": "Pydantic",               "categoria": "libreria"},
                {"tipo": "wikipedia", "valor": "Representational state transfer", "titulo": "REST API",     "categoria": "concepto"},
                {"tipo": "wikipedia", "valor": "JSON Web Token",         "titulo": "JWT",                    "categoria": "seguridad"},
                {"tipo": "wikipedia", "valor": "Docker (software)",      "titulo": "Docker",                 "categoria": "infraestructura"},
                {"tipo": "wikipedia", "valor": "PostgreSQL",             "titulo": "PostgreSQL",             "categoria": "base_datos"},
            ]
            seleccion = st.multiselect(
                "Selecciona artículos a indexar",
                options=[f["titulo"] for f in CORPUS_PREDEFINIDO],
                default=[f["titulo"] for f in CORPUS_PREDEFINIDO[:3]]
            )

            # Fuente personalizada
            st.caption("O añade un artículo de Wikipedia:")
            custom_wiki = st.text_input("Título del artículo (en inglés)", key="custom_wiki")

            if st.button("📥 Indexar selección", use_container_width=True):
                fuentes_a_indexar = [f for f in CORPUS_PREDEFINIDO if f["titulo"] in seleccion]
                if custom_wiki:
                    fuentes_a_indexar.append({
                        "tipo": "wikipedia", "valor": custom_wiki,
                        "titulo": custom_wiki, "categoria": "custom"
                    })
                if fuentes_a_indexar:
                    with st.spinner(f"Indexando {len(fuentes_a_indexar)} fuentes..."):
                        stats = bot.ingestar(fuentes_a_indexar)
                    st.success(f"✓ {stats['chunks_totales']} chunks indexados")
                    st.session_state.ingesta_hecha = True
                    st.rerun()

        st.divider()

        # Acciones
        st.subheader("⚙️ Acciones")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Evaluar sesión", use_container_width=True):
                with st.spinner("Evaluando..."):
                    metricas = bot.evaluar_sesion(n_ultimos=5)
                st.session_state.metricas_eval = metricas

        with col2:
            if st.button("🗑️ Nueva sesión", use_container_width=True):
                bot.cerrar_sesion()
                bot.resetear()
                st.session_state.mensajes = []
                st.rerun()

        if st.button("🧹 Borrar toda mi memoria", use_container_width=True, type="secondary"):
            bot.olvidar()
            st.session_state.mensajes = []
            st.warning("Memoria borrada.")
            st.rerun()

        # Muestra métricas si hay
        if "metricas_eval" in st.session_state and st.session_state.metricas_eval:
            m = st.session_state.metricas_eval
            st.divider()
            st.subheader("📊 Métricas")
            c1, c2, c3 = st.columns(3)
            c1.metric("Tasa respuesta", f"{m.get('tasa_respuesta', 0):.0%}")
            c2.metric("Faithfulness",   f"{m.get('faithfulness_media', '—')}")
            c3.metric("Ctx relevance",  f"{m.get('context_rel_media', '—')}")
            if m.get("n_con_alucinaciones", 0) > 0:
                st.warning(f"⚠️ {m['n_con_alucinaciones']} respuestas con posibles alucinaciones")


# ── Área principal: chat ──────────────────────────────────────────────────────

st.title("💬 DocBot — Asistente de documentación")

if st.session_state.bot is None:
    st.info("👈 Inicia DocBot desde el panel lateral para empezar.")
    st.stop()

bot: DocBot = st.session_state.bot

if not st.session_state.ingesta_hecha and bot.n_docs == 0:
    st.warning("No hay documentos indexados. Usa el panel lateral para indexar fuentes.")

# Muestra el historial de mensajes
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])
        if "fuentes" in mensaje and mensaje["fuentes"]:
            st.caption(f"📌 Fuentes: {', '.join(mensaje['fuentes'])}")
        if "latencia" in mensaje:
            st.caption(f"⏱ {mensaje['latencia']:.0f}ms | {mensaje.get('n_chunks', 0)} chunks")
        if "faithfulness" in mensaje and mensaje["faithfulness"] is not None:
            color = "green" if mensaje["faithfulness"] >= 0.8 else "orange" if mensaje["faithfulness"] >= 0.6 else "red"
            st.caption(f":{color}[Faithfulness: {mensaje['faithfulness']:.2f}]")

# Input del usuario
if prompt := st.chat_input("Escribe tu pregunta sobre los documentos indexados..."):

    # Muestra el mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.mensajes.append({"rol": "user", "contenido": prompt})

    # Genera la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando en la documentación..."):
            respuesta: RespuestaDocBot = bot.responder(prompt)

        st.markdown(respuesta.respuesta)

        # Metadatos de la respuesta
        if respuesta.fuentes:
            st.caption(f"📌 Fuentes: {', '.join(respuesta.fuentes)}")
        st.caption(f"⏱ {respuesta.latencia_ms:.0f}ms | {respuesta.n_chunks} chunks usados")

        if not respuesta.respondida:
            st.info("ℹ️ No encontré información suficiente en los documentos indexados.")

        if respuesta.faithfulness is not None:
            color = "green" if respuesta.faithfulness >= 0.8 else "orange" if respuesta.faithfulness >= 0.6 else "red"
            st.caption(f":{color}[Faithfulness: {respuesta.faithfulness:.2f}]")

    # Guarda en el historial de la UI
    st.session_state.mensajes.append({
        "rol":         "assistant",
        "contenido":   respuesta.respuesta,
        "fuentes":     respuesta.fuentes,
        "latencia":    respuesta.latencia_ms,
        "n_chunks":    respuesta.n_chunks,
        "faithfulness": respuesta.faithfulness
    })
