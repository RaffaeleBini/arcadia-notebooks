"""
app.py — Interfaccia Streamlit per il Rilevatore Email IA.

Avviare con:
    cd progetti/rilevatore-email-ia
    streamlit run app.py
"""

from datetime import datetime

import streamlit as st

from rilevatore_email import (
    classifica_email,
    crea_client_gmail,
    genera_riepilogo,
    azione_automatica,
    VoceReport,
)

# ── Configurazione della pagina ───────────────────────────────────────────────

st.set_page_config(
    page_title="Rilevatore Email IA",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORE_RISCHIO = {"basso": "green", "medio": "orange", "alto": "red"}
ICONA_CATEGORIA = {"legittima": "✅", "spam": "📢", "phishing": "🎣"}


# ── Stato di sessione ──────────────────────────────────────────────────────────

def inizializza_stato():
    if "client" not in st.session_state:
        st.session_state.client = None
    if "modalita" not in st.session_state:
        st.session_state.modalita = "demo"
    if "storico" not in st.session_state:
        st.session_state.storico: list[VoceReport] = []
    if "azione_reale" not in st.session_state:
        st.session_state.azione_reale = False


inizializza_stato()


def ultima_analisi(email_id: str) -> VoceReport | None:
    for voce in reversed(st.session_state.storico):
        if voce.email_id == email_id:
            return voce
    return None


def analizza_e_registra(email: dict, client, usa_strumenti: bool) -> VoceReport:
    risultato = classifica_email(email, usa_strumenti=usa_strumenti)
    voce = VoceReport(
        email_id=email["id"],
        mittente=email.get("mittente", ""),
        oggetto=email.get("oggetto", ""),
        risultato=risultato,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    azione_automatica(voce, client, azione_reale=st.session_state.azione_reale)
    st.session_state.storico.append(voce)
    return voce


def mostra_verdetto(voce: VoceReport):
    r = voce.risultato
    if r.errore:
        st.error(f"Errore durante l'analisi: {r.errore}")
        return

    colore = COLORE_RISCHIO.get(r.livello_rischio, "gray")
    icona = ICONA_CATEGORIA.get(r.categoria, "❓")
    st.markdown(f"**Verdetto:** {icona} `{r.categoria.upper()}` — Rischio: :{colore}[**{r.livello_rischio.upper()}**]")
    st.write(r.motivazione)
    if r.indicatori:
        st.caption("Indicatori sospetti: " + ", ".join(r.indicatori))
    st.caption(f"Azione consigliata dal modello: {r.azione_consigliata}")
    if voce.azione_eseguita:
        st.caption(f"Azione automatica applicata: {voce.azione_eseguita}")
    if r.strumenti_usati:
        with st.expander(f"🔧 Strumenti richiamati dal modello ({len(r.strumenti_usati)})"):
            for s in r.strumenti_usati:
                st.json(s)
    st.caption(f"Modello: {r.modello_usato} · Latenza: {r.latenza_ms:.0f} ms")


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛡️ Rilevatore Email IA")
    st.caption("Classificazione automatica di email sospette con LLM")

    modalita_scelta = st.radio(
        "Casella email",
        ["Demo (email finte)", "Gmail reale (OAuth)"],
        index=0 if st.session_state.modalita == "demo" else 1,
    )
    nuova_modalita = "demo" if modalita_scelta.startswith("Demo") else "reale"
    if nuova_modalita != st.session_state.modalita:
        st.session_state.modalita = nuova_modalita
        st.session_state.client = None

    usa_strumenti = st.checkbox(
        "Abilita function calling (verifica URL / mittente)", value=True,
        help="Il modello può richiamare verifica_url_sospetto e verifica_reputazione_mittente prima di dare il verdetto."
    )
    st.session_state.azione_reale = st.checkbox(
        "Esegui azione reale su Gmail per rischio alto",
        value=False,
        disabled=(st.session_state.modalita == "demo"),
        help="Applica una vera etichetta Gmail invece di limitarsi a loggare l'azione. Richiede modalità reale con scope gmail.modify.",
    )

    st.divider()

    if st.session_state.client is None:
        if st.button("🔌 Connetti casella", use_container_width=True):
            with st.spinner("Connessione..."):
                try:
                    st.session_state.client = crea_client_gmail(usa_demo=(st.session_state.modalita == "demo"))
                except Exception as e:
                    st.error(f"Errore di connessione: {e}")
            st.rerun()
    else:
        st.success(f"✓ Connesso ({st.session_state.client.modalita})")
        if st.button("🗑️ Svuota storico analisi", use_container_width=True):
            st.session_state.storico = []
            st.rerun()
        if st.button("🔄 Cambia casella", use_container_width=True):
            st.session_state.client = None
            st.rerun()


# ── Area principale ────────────────────────────────────────────────────────────

st.title("🛡️ Rilevatore Email IA")
st.caption("Analizza automaticamente la casella e segnala spam/phishing, con azione automatica opzionale sulle email ad alto rischio.")

if st.session_state.client is None:
    st.info("👈 Connetti una casella email dal pannello laterale per iniziare (la modalità demo non richiede nessuna configurazione).")
    st.stop()

client = st.session_state.client

try:
    email_disponibili = client.elenca_email()
except Exception as e:
    st.error(f"Errore nel leggere la casella: {e}")
    st.stop()

if not email_disponibili:
    st.warning("Nessuna email trovata nella casella.")
    st.stop()

col_a, col_b = st.columns([3, 1])
with col_a:
    n_non_lette = sum(1 for e in email_disponibili if not e.get("letta", False))
    st.subheader(f"📥 Posta in arrivo — {len(email_disponibili)} email ({n_non_lette} non lette)")
with col_b:
    analizza_tutte = st.button("🔍 Analizza tutte", use_container_width=True, type="primary")

if analizza_tutte:
    barra = st.progress(0.0, text="Analisi in corso...")
    for i, email in enumerate(email_disponibili):
        try:
            analizza_e_registra(email, client, usa_strumenti)
        except Exception as e:
            st.error(f"Errore analizzando '{email.get('oggetto', '')}': {e}")
        barra.progress((i + 1) / len(email_disponibili), text=f"Analizzata {i + 1}/{len(email_disponibili)}")
    barra.empty()
    st.rerun()

st.divider()

# Elenco email, con analisi singola espandibile
for email in email_disponibili:
    voce = ultima_analisi(email["id"])
    badge = ""
    if voce and not voce.risultato.errore:
        colore = COLORE_RISCHIO.get(voce.risultato.livello_rischio, "gray")
        badge = f" — :{colore}[{voce.risultato.categoria.upper()}]"

    titolo = f"{'📩' if not email.get('letta') else '📧'} {email.get('oggetto', '(senza oggetto)')} — {email.get('mittente', '')}{badge}"
    with st.expander(titolo):
        st.write(email.get("corpo", ""))
        st.caption(f"Data: {email.get('data', '')}")

        if st.button("Analizza questa email", key=f"analizza_{email['id']}"):
            with st.spinner("Analisi in corso..."):
                try:
                    voce = analizza_e_registra(email, client, usa_strumenti)
                except Exception as e:
                    st.error(f"Errore: {e}")
                    voce = None

        if voce:
            mostra_verdetto(voce)


# ── Riepilogo e storico ──────────────────────────────────────────────────────

if st.session_state.storico:
    st.divider()
    st.subheader("📊 Riepilogo analisi")

    riepilogo = genera_riepilogo(st.session_state.storico)
    c1, c2, c3 = st.columns(3)
    c1.metric("Email analizzate", riepilogo["totale_analizzate"])
    c2.metric("Ad alto rischio", len(riepilogo["alto_rischio"]))
    c3.metric("Phishing rilevati", riepilogo["per_categoria"].get("phishing", 0))

    col_cat, col_rischio = st.columns(2)
    with col_cat:
        st.caption("Per categoria")
        st.write(riepilogo["per_categoria"])
    with col_rischio:
        st.caption("Per livello di rischio")
        st.write(riepilogo["per_livello_rischio"])

    with st.expander("📜 Storico completo delle analisi"):
        for voce in reversed(st.session_state.storico):
            colore = COLORE_RISCHIO.get(voce.risultato.livello_rischio, "gray")
            st.markdown(
                f"`{voce.timestamp}` — **{voce.oggetto}** ({voce.mittente}) "
                f"→ :{colore}[{voce.risultato.categoria.upper()} / {voce.risultato.livello_rischio.upper()}]"
            )
