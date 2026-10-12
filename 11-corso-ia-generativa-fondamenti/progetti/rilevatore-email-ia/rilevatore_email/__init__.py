from .classificatore import classifica_email, RisultatoClassificazione
from .gmail_client import crea_client_gmail, ClientGmailDemo, ClientGmailReale
from .report import genera_riepilogo, azione_automatica, VoceReport

__all__ = [
    "classifica_email",
    "RisultatoClassificazione",
    "crea_client_gmail",
    "ClientGmailDemo",
    "ClientGmailReale",
    "genera_riepilogo",
    "azione_automatica",
    "VoceReport",
]
