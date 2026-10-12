"""
gmail_client.py — Accesso alla casella email: client Gmail reale (OAuth) o
client demo che legge da un file JSON locale.

Entrambe le classi espongono la STESSA interfaccia:
    elenca_email(solo_non_lette=False) -> list[dict]
    leggi_email(id_email) -> dict | None
    etichetta_email(id_email, etichetta) -> bool
    modalita -> "demo" | "reale"

così il resto dell'applicazione (app.py, report.py) non deve sapere se sta
parlando con una casella vera o con dati finti.

Per collegare un vero account Gmail, vedi la sezione dedicata nel README.md
(creazione progetto Google Cloud, abilitazione Gmail API, credenziali OAuth).
"""

import base64
import json
import os
from pathlib import Path

# Scope minimi necessari: lettura per l'analisi, modifica per poter applicare
# davvero un'etichetta di quarantena alle email ad alto rischio.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

PERCORSO_DATI_DEMO = Path(__file__).resolve().parent.parent / "dati" / "email_demo.json"


class ClientGmailDemo:
    """
    Client "finto": legge/scrive le email da dati/email_demo.json.

    Non richiede nessuna credenziale: è pensato per provare l'app subito,
    senza configurare un account Google.
    """

    def __init__(self, percorso_dati: str | None = None):
        self._percorso = Path(percorso_dati) if percorso_dati else PERCORSO_DATI_DEMO
        self._email = json.loads(self._percorso.read_text(encoding="utf-8"))

    def elenca_email(self, solo_non_lette: bool = False) -> list[dict]:
        if solo_non_lette:
            return [e for e in self._email if not e.get("letta", False)]
        return list(self._email)

    def leggi_email(self, id_email: str) -> dict | None:
        return next((e for e in self._email if e["id"] == id_email), None)

    def etichetta_email(self, id_email: str, etichetta: str) -> bool:
        """Simula l'etichettatura: registra l'etichetta in memoria e nel log, nessuna scrittura reale."""
        for e in self._email:
            if e["id"] == id_email:
                e.setdefault("etichette", []).append(etichetta)
                print(f"[DEMO] Email {id_email} etichettata come '{etichetta}' (simulato, nessuna modifica reale)")
                return True
        return False

    @property
    def modalita(self) -> str:
        return "demo"


class ClientGmailReale:
    """
    Client reale basato su google-api-python-client + OAuth 2.0.

    Richiede un file di credenziali OAuth (credentials.json, scaricato da
    Google Cloud Console) e genera un token.json al primo utilizzo tramite
    il flusso di autorizzazione nel browser. Vedi README.md per la procedura
    completa di configurazione lato Google Cloud.

    I percorsi si leggono da .env:
        GMAIL_CREDENTIALS_PATH=credentials.json
        GMAIL_TOKEN_PATH=token.json
    """

    def __init__(self, percorso_credenziali: str | None = None, percorso_token: str | None = None):
        self._percorso_credenziali = percorso_credenziali or os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")
        self._percorso_token = percorso_token or os.getenv("GMAIL_TOKEN_PATH", "token.json")
        self._service = self._autentica()
        self._cache_etichette: dict[str, str] = {}

    def _autentica(self):
        # Import locali: queste librerie sono necessarie solo in modalità reale,
        # così la modalità demo funziona anche senza installarle.
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        token_path = Path(self._percorso_token)
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self._percorso_credenziali, SCOPES)
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json(), encoding="utf-8")

        return build("gmail", "v1", credentials=creds)

    def elenca_email(self, solo_non_lette: bool = False, max_risultati: int = 20) -> list[dict]:
        query = "is:unread" if solo_non_lette else ""
        risultati = self._service.users().messages().list(
            userId="me", labelIds=["INBOX"], q=query, maxResults=max_risultati
        ).execute()
        return [self.leggi_email(msg["id"]) for msg in risultati.get("messages", [])]

    def leggi_email(self, id_email: str) -> dict:
        messaggio = self._service.users().messages().get(userId="me", id=id_email, format="full").execute()
        intestazioni = {h["name"]: h["value"] for h in messaggio["payload"].get("headers", [])}
        return {
            "id": id_email,
            "mittente": intestazioni.get("From", ""),
            "oggetto": intestazioni.get("Subject", ""),
            "corpo": _estrai_corpo(messaggio["payload"]),
            "data": intestazioni.get("Date", ""),
            "letta": "UNREAD" not in messaggio.get("labelIds", []),
        }

    def etichetta_email(self, id_email: str, etichetta: str) -> bool:
        """Applica un'etichetta reale su Gmail (crea l'etichetta se non esiste)."""
        id_etichetta = self._trova_o_crea_etichetta(etichetta)
        self._service.users().messages().modify(
            userId="me", id=id_email, body={"addLabelIds": [id_etichetta]}
        ).execute()
        return True

    def _trova_o_crea_etichetta(self, nome: str) -> str:
        if nome in self._cache_etichette:
            return self._cache_etichette[nome]
        elenco = self._service.users().labels().list(userId="me").execute().get("labels", [])
        esistente = next((l for l in elenco if l["name"] == nome), None)
        if esistente:
            self._cache_etichette[nome] = esistente["id"]
            return esistente["id"]
        nuova = self._service.users().labels().create(
            userId="me", body={"name": nome, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
        ).execute()
        self._cache_etichette[nome] = nuova["id"]
        return nuova["id"]

    @property
    def modalita(self) -> str:
        return "reale"


def _estrai_corpo(payload: dict) -> str:
    """Decodifica il corpo testuale (text/plain) di un messaggio Gmail, gestendo i messaggi multipart."""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")

    for parte in payload.get("parts", []):
        if parte.get("mimeType") == "text/plain" and parte.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(parte["body"]["data"]).decode("utf-8", errors="ignore")
    for parte in payload.get("parts", []):
        if parte.get("parts"):
            trovato = _estrai_corpo(parte)
            if trovato:
                return trovato

    dati = payload.get("body", {}).get("data")
    if dati:
        return base64.urlsafe_b64decode(dati).decode("utf-8", errors="ignore")
    return ""


def crea_client_gmail(usa_demo: bool = True, **kwargs):
    """Factory: restituisce il client demo o quello reale con la stessa interfaccia."""
    if usa_demo:
        return ClientGmailDemo(**kwargs)
    return ClientGmailReale(**kwargs)
