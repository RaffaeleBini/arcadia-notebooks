"""
classificatore.py — Classificazione di email con LLM + function calling.

Il modello riceve mittente/oggetto/corpo di un'email e restituisce un
verdetto strutturato (JSON): categoria, livello di rischio, motivazione e
azione consigliata. Durante l'analisi può richiamare gli strumenti definiti
in strumenti.py (verifica_url_sospetto, verifica_reputazione_mittente) per
arricchire il ragionamento con "evidenze" prima di dare il verdetto finale.

Supporta due provider, con fallback automatico in base alle chiavi presenti
nel file .env:
  - Anthropic (Claude Sonnet 5), se ANTHROPIC_API_KEY è impostata
  - OpenAI (GPT-5), se OPENAI_API_KEY è impostata

Se sono impostate entrambe, ha priorità Anthropic. Nessuna chiave è mai
scritta nel codice: viene sempre letta da variabili d'ambiente tramite
python-dotenv.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from .strumenti import REGISTRO_STRUMENTI, SCHEMA_STRUMENTI

load_dotenv()

# verifica l'ID modello aggiornato nella doc ufficiale Anthropic
MODELLO_ANTHROPIC = "claude-sonnet-5"
# verifica l'ID modello aggiornato nella doc ufficiale OpenAI
MODELLO_OPENAI = "gpt-5"

MAX_TURNI_STRUMENTI = 4  # limite di iterazioni del ciclo di function calling

PROMPT_SISTEMA = """Sei un esperto di cybersecurity specializzato nell'analisi di email malevole.

Il tuo compito è analizzare un'email (mittente, oggetto, corpo) e determinare se è:
- "legittima": email genuina, senza indicatori di rischio
- "spam": messaggio pubblicitario indesiderato ma non necessariamente pericoloso
- "phishing": tentativo di ottenere credenziali, dati personali o denaro tramite inganno

Procedi così:
1. Analizza in dettaglio mittente, oggetto e corpo dell'email.
2. Individua i pattern sospetti dal punto di vista della sicurezza (urgenza artificiale,
   richieste di dati sensibili, link sospetti, dominio del mittente anomalo, errori
   grammaticali, promesse irrealistiche, ecc.).
3. Se nel testo compaiono URL o l'indirizzo del mittente ti sembra sospetto, usa gli
   strumenti disponibili (verifica_url_sospetto, verifica_reputazione_mittente) per
   raccogliere evidenze concrete prima di decidere.
4. Ragiona sul motivo per cui i pattern individuati sono o non sono indicativi di un
   messaggio malevolo, nel contesto complessivo del messaggio.
5. Determina la categoria finale e il livello di rischio.

Quando hai raccolto abbastanza evidenze, rispondi ESCLUSIVAMENTE con un oggetto JSON
(nessun testo prima o dopo) con questa struttura esatta:
{
  "categoria": "legittima" | "spam" | "phishing",
  "livello_rischio": "basso" | "medio" | "alto",
  "indicatori": ["elenco breve dei pattern sospetti individuati, vuoto se nessuno"],
  "motivazione": "spiegazione sintetica (2-4 frasi) del verdetto",
  "azione_consigliata": "es. 'nessuna azione', 'segnala come spam', 'metti in quarantena e non cliccare i link'"
}"""


@dataclass
class RisultatoClassificazione:
    categoria: str
    livello_rischio: str
    motivazione: str
    azione_consigliata: str
    indicatori: list[str] = field(default_factory=list)
    strumenti_usati: list[dict] = field(default_factory=list)
    modello_usato: str = ""
    latenza_ms: float = 0.0
    errore: str | None = None


# ── Selezione del provider ───────────────────────────────────────────────────

def _rileva_provider() -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "Nessuna chiave API trovata. Imposta ANTHROPIC_API_KEY o OPENAI_API_KEY "
        "nel file .env (vedi README.md)."
    )


def _crea_client(provider: str):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Conversione dello schema neutro nei formati specifici dei provider ──────

def _schema_per_openai(schema: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": s["nome"],
                "description": s["descrizione"],
                "parameters": s["parametri"],
            },
        }
        for s in schema
    ]


def _schema_per_anthropic(schema: list[dict]) -> list[dict]:
    return [
        {"name": s["nome"], "description": s["descrizione"], "input_schema": s["parametri"]}
        for s in schema
    ]


# ── Estrazione robusta del JSON dalla risposta del modello ──────────────────

def _estrai_json(testo: str) -> dict:
    testo = (testo or "").strip()
    try:
        return json.loads(testo)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", testo, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# ── Ciclo di function calling — OpenAI ───────────────────────────────────────

def _esegui_openai(client, messaggio_utente: str, usa_strumenti: bool) -> tuple[str, list[dict]]:
    messaggi = [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user", "content": messaggio_utente},
    ]
    tools = _schema_per_openai(SCHEMA_STRUMENTI) if usa_strumenti else None
    chiamate_strumenti: list[dict] = []

    for _ in range(MAX_TURNI_STRUMENTI):
        kwargs = {"model": MODELLO_OPENAI, "messages": messaggi, "temperature": 0}
        if tools:
            kwargs["tools"] = tools
        risposta = client.chat.completions.create(**kwargs)
        msg = risposta.choices[0].message

        if msg.tool_calls:
            messaggi.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                nome = tc.function.name
                argomenti = json.loads(tc.function.arguments or "{}")
                funzione = REGISTRO_STRUMENTI.get(nome)
                risultato = funzione(**argomenti) if funzione else {"errore": f"strumento sconosciuto: {nome}"}
                chiamate_strumenti.append({"strumento": nome, "argomenti": argomenti, "risultato": risultato})
                messaggi.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(risultato, ensure_ascii=False),
                })
            continue

        return msg.content or "", chiamate_strumenti

    return "", chiamate_strumenti


# ── Ciclo di function calling — Anthropic ────────────────────────────────────

def _blocco_anthropic_a_dict(blocco) -> dict:
    if blocco.type == "text":
        return {"type": "text", "text": blocco.text}
    if blocco.type == "tool_use":
        return {"type": "tool_use", "id": blocco.id, "name": blocco.name, "input": blocco.input}
    return {"type": blocco.type}


def _esegui_anthropic(client, messaggio_utente: str, usa_strumenti: bool) -> tuple[str, list[dict]]:
    messaggi = [{"role": "user", "content": messaggio_utente}]
    tools = _schema_per_anthropic(SCHEMA_STRUMENTI) if usa_strumenti else None
    chiamate_strumenti: list[dict] = []

    for _ in range(MAX_TURNI_STRUMENTI):
        kwargs = {
            "model": MODELLO_ANTHROPIC,
            "max_tokens": 1024,
            "system": PROMPT_SISTEMA,
            "messages": messaggi,
            "temperature": 0,
        }
        if tools:
            kwargs["tools"] = tools
        risposta = client.messages.create(**kwargs)

        blocchi_tool = [b for b in risposta.content if b.type == "tool_use"]
        if blocchi_tool:
            messaggi.append({
                "role": "assistant",
                "content": [_blocco_anthropic_a_dict(b) for b in risposta.content],
            })
            risultati_blocco = []
            for b in blocchi_tool:
                funzione = REGISTRO_STRUMENTI.get(b.name)
                risultato = funzione(**b.input) if funzione else {"errore": f"strumento sconosciuto: {b.name}"}
                chiamate_strumenti.append({"strumento": b.name, "argomenti": b.input, "risultato": risultato})
                risultati_blocco.append({
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": json.dumps(risultato, ensure_ascii=False),
                })
            messaggi.append({"role": "user", "content": risultati_blocco})
            continue

        testo = "".join(b.text for b in risposta.content if b.type == "text")
        return testo, chiamate_strumenti

    return "", chiamate_strumenti


# ── API pubblica ──────────────────────────────────────────────────────────────

def classifica_email(email: dict, usa_strumenti: bool = True) -> RisultatoClassificazione:
    """
    Classifica una singola email (dict con almeno mittente/oggetto/corpo).

    Esempio:
        r = classifica_email({
            "mittente": "PayPal <servizio@paypa1-security.com>",
            "oggetto": "Il tuo account è stato limitato",
            "corpo": "Verifica subito i tuoi dati: http://paypa1-security.com/verifica",
        })
        print(r.categoria, r.livello_rischio)
    """
    provider = _rileva_provider()
    client = _crea_client(provider)

    messaggio_utente = (
        f"Da: {email.get('mittente', '')}\n"
        f"Oggetto: {email.get('oggetto', '')}\n\n"
        f"{email.get('corpo', '')}"
    )

    inizio = time.perf_counter()
    try:
        if provider == "anthropic":
            testo, chiamate = _esegui_anthropic(client, messaggio_utente, usa_strumenti)
            modello_usato = MODELLO_ANTHROPIC
        else:
            testo, chiamate = _esegui_openai(client, messaggio_utente, usa_strumenti)
            modello_usato = MODELLO_OPENAI
    except Exception as e:
        latenza = (time.perf_counter() - inizio) * 1000
        return RisultatoClassificazione(
            categoria="legittima",
            livello_rischio="basso",
            motivazione="",
            azione_consigliata="",
            modello_usato=provider,
            latenza_ms=latenza,
            errore=str(e),
        )
    latenza_ms = (time.perf_counter() - inizio) * 1000

    dati = _estrai_json(testo)
    return RisultatoClassificazione(
        categoria=dati.get("categoria", "legittima"),
        livello_rischio=dati.get("livello_rischio", "basso"),
        motivazione=dati.get("motivazione", ""),
        azione_consigliata=dati.get("azione_consigliata", ""),
        indicatori=dati.get("indicatori", []),
        strumenti_usati=chiamate,
        modello_usato=modello_usato,
        latenza_ms=latenza_ms,
        errore=None if dati else "risposta del modello non interpretabile come JSON",
    )
