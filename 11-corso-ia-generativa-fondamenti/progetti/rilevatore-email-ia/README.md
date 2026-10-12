# Rilevatore Email IA — Rilevamento automatico di email malevole

Assistente che analizza le email di una casella di posta e segnala quelle
sospette (spam o phishing), usando un LLM con output strutturato e function
calling per verificare URL e reputazione del mittente. Per le email ad alto
rischio può eseguire un'azione automatica (etichettatura, simulata di default).

## Installazione

```bash
cd progetti/rilevatore-email-ia
pip install -r requirements.txt
```

Crea un file `.env` in questa cartella con **una delle due** chiavi (basta una,
non servono entrambe — se sono presenti entrambe ha priorità Anthropic):

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

> **Nota:** nessuna chiave va mai scritta nel codice. Viene sempre letta da
> `.env` tramite `python-dotenv`. Non versionare mai il file `.env` (aggiungilo
> al tuo `.gitignore`).

## Avviare l'app

```bash
streamlit run app.py
```

Apri http://localhost:8501 nel browser.

## Uso rapido

1. Nel pannello laterale, lascia selezionata la modalità **Demo (email finte)**
   — non richiede nessuna configurazione aggiuntiva.
2. Premi **🔌 Connetti casella**.
3. Premi **🔍 Analizza tutte** per classificare l'intera posta in arrivo, oppure
   apri una singola email ed usa **Analizza questa email**.
4. Osserva il verdetto (categoria + livello di rischio con badge colorato), la
   motivazione e — se hai abilitato il function calling — gli strumenti che il
   modello ha richiamato per arricchire l'analisi.
5. In fondo alla pagina trovi il riepilogo aggregato e lo storico di tutte le
   analisi eseguite in questa sessione.

## Uso da codice Python

```python
from rilevatore_email import classifica_email, crea_client_gmail, azione_automatica, VoceReport
from datetime import datetime

# Classifica una singola email
risultato = classifica_email({
    "mittente": "PayPal <servizio@paypa1-security.com>",
    "oggetto": "Il tuo account è stato temporaneamente limitato",
    "corpo": "Verifica subito i tuoi dati: http://paypa1-security.com/verifica-account",
})

print(risultato.categoria)            # "phishing"
print(risultato.livello_rischio)      # "alto"
print(risultato.motivazione)
print(risultato.strumenti_usati)      # chiamate a verifica_url_sospetto / verifica_reputazione_mittente

# Analizza l'intera casella demo
client = crea_client_gmail(usa_demo=True)
for email in client.elenca_email():
    r = classifica_email(email)
    voce = VoceReport(
        email_id=email["id"], mittente=email["mittente"], oggetto=email["oggetto"],
        risultato=r, timestamp=datetime.now().isoformat(),
    )
    azione_automatica(voce, client, azione_reale=False)  # di default: solo log, nessuna modifica reale
```

## Architettura

```
rilevatore-email-ia/
├── app.py                          # interfaccia Streamlit
├── dati/
│   └── email_demo.json             # 15 email finte per la modalità demo (italiano)
└── rilevatore_email/
    ├── __init__.py                 # esporta l'API pubblica del package
    ├── classificatore.py           # chiamata LLM (Anthropic/OpenAI) + output JSON + ciclo di function calling
    ├── strumenti.py                # verifica_url_sospetto, verifica_reputazione_mittente + schemi JSON
    ├── gmail_client.py             # ClientGmailReale (OAuth) e ClientGmailDemo, stessa interfaccia
    └── report.py                   # VoceReport, genera_riepilogo, azione_automatica
```

Pipeline di analisi di una singola email:

1. **classificatore.py** costruisce il prompt di sistema (esperto di
   cybersecurity) e invia mittente/oggetto/corpo al modello.
2. Se il function calling è abilitato, il modello può richiamare uno o più
   strumenti da **strumenti.py** (fino a `MAX_TURNI_STRUMENTI` cicli) per
   verificare URL sospetti o la reputazione del dominio del mittente.
3. Il modello restituisce un verdetto in JSON (`categoria`, `livello_rischio`,
   `indicatori`, `motivazione`, `azione_consigliata`), interpretato in un
   oggetto `RisultatoClassificazione`.
4. **report.py** registra il risultato in una `VoceReport` e, se il rischio è
   `"alto"`, applica l'azione automatica (simulata di default).

## Gli strumenti (function calling)

`strumenti.py` implementa due funzioni richiamabili dal modello:

- `verifica_url_sospetto(url)` — controlla il dominio dell'URL contro una
  lista nera locale di domini finti tipici di campagne di phishing (es.
  typosquatting di brand noti).
- `verifica_reputazione_mittente(indirizzo)` — controlla il dominio del
  mittente contro una lista nera e una lista bianca locali.

> **Nota didattica:** sono implementazioni **mock**, pensate per il corso.
> Non chiamano nessun servizio esterno reale. In un progetto in produzione,
> al posto della lista nera locale collegheresti qui una vera API di threat
> intelligence, ad esempio **VirusTotal** (endpoint `/urls`) per gli URL o un
> servizio di sender reputation per i mittenti. La struttura del codice
> (funzione + schema JSON + registro) resta identica: cambia solo il corpo
> della funzione, che farebbe una chiamata HTTP invece di consultare un dict.

## Come collegare un vero account Gmail

Di default l'app usa `ClientGmailDemo`, che legge da `dati/email_demo.json` e
non richiede nessuna credenziale. Per collegare una casella Gmail reale:

1. Vai su [Google Cloud Console](https://console.cloud.google.com/) e crea un
   nuovo progetto (o riusane uno esistente).
2. Nel menu **API e servizi → Libreria**, cerca **Gmail API** e abilitala per
   il progetto.
3. In **API e servizi → Schermata consenso OAuth**, configura una app di tipo
   "Esterno" (in modalità test è sufficiente per uso personale) e aggiungi il
   tuo indirizzo Gmail come utente di test.
4. In **API e servizi → Credenziali**, crea credenziali di tipo **ID client
   OAuth 2.0**, applicazione desktop. Scarica il file JSON delle credenziali.
5. Salva il file scaricato in questa cartella (o dove preferisci) e aggiungi
   al tuo `.env`:
   ```
   GMAIL_CREDENTIALS_PATH=credentials.json
   GMAIL_TOKEN_PATH=token.json
   ```
6. Nel pannello laterale dell'app, seleziona **Gmail reale (OAuth)** e premi
   **Connetti casella**: al primo avvio si aprirà il browser per il consenso
   OAuth, poi verrà salvato `token.json` per le sessioni successive.

Lo scope richiesto è `gmail.readonly` (lettura) più `gmail.modify` (necessario
solo se abiliti l'azione reale di etichettatura per il rischio alto). Se
cambi gli scope dopo la prima autenticazione, cancella `token.json` e
ripeti il login.

## Costo approssimativo

Con `gpt-5` (o `claude-sonnet-5`) e function calling abilitato, esempio
illustrativo — **verifica sempre i prezzi aggiornati sul sito del provider**:

- Per email (senza chiamate a strumenti): ~$0.0005
- Per email (con 1-2 chiamate a strumenti, più token di contesto): ~$0.001
- Analisi di una casella con 100 email/giorno: ~$0.10/giorno → ~$3/mese

## Limitazioni note

- Gli strumenti `verifica_url_sospetto` e `verifica_reputazione_mittente` sono
  **mock didattici**: coprono solo i domini presenti nelle liste demo, non
  offrono protezione reale.
- L'azione automatica sul rischio alto è **simulata di default** (solo log):
  va abilitata esplicitamente, e richiede comunque una casella Gmail reale
  configurata con scope `gmail.modify`.
- Il client Gmail reale legge il corpo dell'email nella sola parte
  `text/plain`: email complesse solo in `text/html` senza parte testuale
  alternativa potrebbero risultare vuote.
- Il verdetto del modello non è infallibile: va trattato come supporto alla
  decisione, non come sostituto di un filtro antispam/antiphishing dedicato.

---

Progetto 1 — Corso Completo di IA Generativa (Modulo 12)
