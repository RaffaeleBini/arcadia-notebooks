"""
report.py — Aggregazione dei risultati di classificazione e azione automatica
sulle email ad alto rischio.
"""

from collections import Counter
from dataclasses import dataclass

from .classificatore import RisultatoClassificazione

SOGLIA_RISCHIO_AZIONE = "alto"
ETICHETTA_QUARANTENA = "IA/Quarantena-Sospetta"


@dataclass
class VoceReport:
    """Una riga dello storico: un'email analizzata + il suo verdetto."""
    email_id: str
    mittente: str
    oggetto: str
    risultato: RisultatoClassificazione
    timestamp: str
    azione_eseguita: str | None = None


def genera_riepilogo(voci: list[VoceReport]) -> dict:
    """Calcola statistiche aggregate su un elenco di analisi già eseguite."""
    per_categoria = Counter(v.risultato.categoria for v in voci)
    per_rischio = Counter(v.risultato.livello_rischio for v in voci)
    alto_rischio = [v.email_id for v in voci if v.risultato.livello_rischio == SOGLIA_RISCHIO_AZIONE]

    return {
        "totale_analizzate": len(voci),
        "per_categoria": dict(per_categoria),
        "per_livello_rischio": dict(per_rischio),
        "alto_rischio": alto_rischio,
    }


def azione_automatica(voce: VoceReport, client_gmail, azione_reale: bool = False) -> str:
    """
    Applica l'azione automatica prevista per le email ad alto rischio.

    Di default l'azione è SIMULATA (viene solo loggata): è la scelta più
    sicura per un ambiente didattico, dove non vogliamo modificare per
    sbaglio una casella email reale. Con azione_reale=True (e un client
    Gmail reale con scope gmail.modify) viene applicata un'etichetta vera.
    """
    if voce.risultato.livello_rischio != SOGLIA_RISCHIO_AZIONE:
        azione = "nessuna azione (livello di rischio non alto)"
        voce.azione_eseguita = azione
        return azione

    if azione_reale and hasattr(client_gmail, "etichetta_email") and getattr(client_gmail, "modalita", "demo") == "reale":
        try:
            client_gmail.etichetta_email(voce.email_id, ETICHETTA_QUARANTENA)
            azione = f"email etichettata realmente su Gmail come '{ETICHETTA_QUARANTENA}'"
        except Exception as e:
            azione = f"errore durante l'azione reale su Gmail: {e}"
    else:
        azione = (
            f"[SIMULATO] sposterei l'email in quarantena / etichetta '{ETICHETTA_QUARANTENA}' "
            "(nessuna modifica reale eseguita)"
        )

    print(f"[AZIONE] {voce.email_id} ({voce.oggetto[:50]!r}): {azione}")
    voce.azione_eseguita = azione
    return azione
