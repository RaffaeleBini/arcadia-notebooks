"""
strumenti.py — Funzioni richiamabili dal modello via function calling.

Sono implementazioni MOCK pensate per il corso: non chiamano nessun servizio
esterno reale, ma simulano una verifica di threat intelligence usando liste
nere locali. In un progetto reale, qui collegheresti API come VirusTotal,
Google Safe Browsing o PhishTank per una verifica in tempo reale.

Ogni funzione ha uno schema JSON gemello in SCHEMA_STRUMENTI, usato da
classificatore.py per descrivere gli strumenti al modello (sia in formato
OpenAI "tools" che in formato Anthropic "tools").
"""

from urllib.parse import urlparse

# ── Dati mock: liste nere locali "in stile VirusTotal" ──────────────────────
#
# NOTA DIDATTICA: in produzione questa lista sparirebbe. Al suo posto ci
# sarebbe una chiamata HTTP a un vero servizio di threat intelligence
# (es. VirusTotal API v3, endpoint /urls) che restituisce un verdetto
# calcolato da decine di motori antivirus e feed di reputazione.

DOMINI_URL_SOSPETTI = {
    "paypa1-security.com":        "typosquatting del dominio paypal.com (sostituzione l→1)",
    "poste-italiane-verifica.click": "dominio non ufficiale che imita Poste Italiane, TLD .click atipico per un ente pubblico",
    "banca-intesa-sicura.info":   "dominio non ufficiale che imita Intesa Sanpaolo, non risulta tra i domini registrati dalla banca",
    "login-microsoft-support.xyz": "dominio non ufficiale che imita Microsoft, TLD .xyz frequente in campagne di phishing",
    "amaz0n-verifica.net":        "typosquatting del dominio amazon (sostituzione o→0)",
    "corriere-tracking-pacco.top": "dominio generico non associato a nessun corriere reale, pattern tipico da smishing",
    "farmamiracolo-sconti.biz":   "dominio associato a campagne di spam pubblicitario aggressivo",
    "cryptoricchezza-invest.biz": "dominio associato a schemi di investimento fraudolenti (pattern 'guadagno garantito')",
}

DOMINI_MITTENTE_SOSPETTI = {
    "paypa1-security.com":         "dominio impersonifica PayPal ma non è un dominio ufficiale PayPal",
    "poste-italiane-verifica.click": "dominio impersonifica Poste Italiane ma non è un dominio ufficiale",
    "banca-intesa-sicura.info":    "dominio impersonifica Intesa Sanpaolo ma non è un dominio ufficiale",
    "login-microsoft-support.xyz": "dominio impersonifica Microsoft ma non è un dominio ufficiale",
    "amaz0n-verifica.net":         "dominio impersonifica Amazon ma non è un dominio ufficiale",
    "corriere-tracking-pacco.top": "dominio generico che si spaccia per un corriere reale",
    "mailfast-proton.com":         "dominio associato in passato a truffe del tipo 'eredità milionaria'",
    "farmamiracolo-sconti.biz":    "dominio associato a campagne di spam pubblicitario",
    "cryptoricchezza-invest.biz":  "dominio associato a schemi di investimento fraudolenti",
}

DOMINI_MITTENTE_AFFIDABILI = {
    "gmail.com", "outlook.com", "hotmail.com", "icloud.com", "protonmail.com",
    "netflix.com", "amazon.it", "amazon.com", "poste.it", "intesasanpaolo.com",
    "microsoft.com", "udemy.com",
}


def _estrai_dominio(indirizzo_o_url: str) -> str:
    """Estrae il dominio da un URL completo o da un indirizzo email 'Nome <utente@dominio>'."""
    testo = indirizzo_o_url.strip()
    if "<" in testo and ">" in testo:
        testo = testo.split("<", 1)[1].split(">", 1)[0]
    if "@" in testo and "://" not in testo:
        return testo.split("@")[-1].strip().lower()
    parsed = urlparse(testo if "://" in testo else f"http://{testo}")
    return (parsed.netloc or parsed.path).lower().split("/")[0]


# ── Strumenti richiamabili dal modello ───────────────────────────────────────

def verifica_url_sospetto(url: str) -> dict:
    """
    Verifica se un URL è presente in una lista nera locale di domini noti
    per phishing/spam.

    NOTA DIDATTICA: simulazione per il corso. In produzione collegheresti
    qui una vera API di threat intelligence (es. VirusTotal).
    """
    dominio = _estrai_dominio(url)
    sospetto = dominio in DOMINI_URL_SOSPETTI
    return {
        "url": url,
        "dominio": dominio,
        "sospetto": sospetto,
        "motivo": DOMINI_URL_SOSPETTI.get(dominio, "dominio non presente nella lista nera demo"),
        "fonte": "lista nera demo del corso (in produzione: VirusTotal / Google Safe Browsing)",
    }


def verifica_reputazione_mittente(indirizzo: str) -> dict:
    """
    Verifica la reputazione del dominio del mittente contro una lista nera
    e una lista di domini comuni/affidabili, entrambe locali e simulate.

    NOTA DIDATTICA: simulazione per il corso. In produzione collegheresti
    qui un servizio di sender reputation (es. Google Postmaster Tools,
    SpamHaus, o un motore di threat intelligence dedicato alle email).
    """
    dominio = _estrai_dominio(indirizzo)

    if dominio in DOMINI_MITTENTE_SOSPETTI:
        return {
            "indirizzo": indirizzo,
            "dominio": dominio,
            "reputazione": "sospetta",
            "motivo": DOMINI_MITTENTE_SOSPETTI[dominio],
            "fonte": "lista nera demo del corso (in produzione: servizio di sender reputation)",
        }

    if dominio in DOMINI_MITTENTE_AFFIDABILI:
        return {
            "indirizzo": indirizzo,
            "dominio": dominio,
            "reputazione": "affidabile",
            "motivo": "dominio noto e ampiamente utilizzato, nessun precedente di abuso nella lista demo",
            "fonte": "lista bianca demo del corso",
        }

    return {
        "indirizzo": indirizzo,
        "dominio": dominio,
        "reputazione": "sconosciuta",
        "motivo": "dominio non presente né nella lista nera né in quella bianca demo",
        "fonte": "lista demo del corso (in produzione: servizio di sender reputation)",
    }


# ── Registro e schema per il function calling ────────────────────────────────

REGISTRO_STRUMENTI = {
    "verifica_url_sospetto": verifica_url_sospetto,
    "verifica_reputazione_mittente": verifica_reputazione_mittente,
}

# Schema neutro (indipendente dal provider): classificatore.py lo traduce
# nel formato "tools" richiesto da OpenAI o da Anthropic.
SCHEMA_STRUMENTI = [
    {
        "nome": "verifica_url_sospetto",
        "descrizione": (
            "Verifica se un URL presente nel corpo dell'email è associato a "
            "campagne note di phishing o spam. Usalo per ogni URL sospetto "
            "trovato nel testo dell'email."
        ),
        "parametri": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "L'URL completo da verificare, es. http://esempio-login.com/verifica",
                }
            },
            "required": ["url"],
        },
    },
    {
        "nome": "verifica_reputazione_mittente",
        "descrizione": (
            "Verifica la reputazione del dominio del mittente di un'email "
            "(indirizzo o intestazione 'From'). Usalo per capire se il "
            "dominio del mittente è noto, sconosciuto o già segnalato come sospetto."
        ),
        "parametri": {
            "type": "object",
            "properties": {
                "indirizzo": {
                    "type": "string",
                    "description": "L'indirizzo email o l'intestazione From completa del mittente, es. 'Nome <utente@dominio.com>'",
                }
            },
            "required": ["indirizzo"],
        },
    },
]
