# Guida di stile — Corso Completo di IA Generativa (2026)

Questo file è la guida di riferimento condivisa per tutti i materiali del corso
(dispense + notebook). Serve a mantenere coerenza terminologica e di formato tra
i 24 moduli, scritti in parallelo da autori diversi. **Va letta per intero prima
di scrivere qualunque file.**

## Lingua e tono

- Tutto il materiale è in **italiano**. Codice, nomi di variabili/funzioni e
  commenti nel codice sono in italiano quando ragionevole (es. `calcola_costo`,
  `chiamata_llm`), coerentemente con lo stile del corso di riferimento
  (11-Corso-IA-Engineering-2026, che scrive in spagnolo con lo stesso approccio).
- Tono: diretto, da formatore esperto a professionista adulto. Frasi brevi,
  niente marketing, niente entusiasmo posticcio. Spiega il "perché", non solo il
  "cosa".
- Mantieni in inglese i termini tecnici che nella pratica italiana restano in
  inglese: *prompt*, *token*, *embedding*, *fine-tuning*, *context window*,
  *temperature*, *top-p*, *few-shot*, *chain of thought*, *function calling*,
  *checkpoint*, *epoch*, *reward model*, *dataset*, *benchmark*, *fine tuning*.
- Traduci in italiano i concetti quando esiste un termine naturale, e la prima
  volta che compare in ogni dispensa indica il termine inglese tra parentesi:
  "finestra di contesto (*context window*)", "inferenza (*inference*)",
  "completamento (*completion*)", "addestramento (*training*)",
  "preaddestramento (*pretraining*)", "razionamento/allineamento
  (*alignment*)", "apprendimento per rinforzo (*reinforcement learning*)".
- Non inventare dati o benchmark numerici precisi che non siano nel materiale
  sorgente o di dominio pubblico noto. Se servono cifre di costo/prezzo
  d'esempio, etichettale sempre come "esempio illustrativo" e invita a
  verificare i prezzi aggiornati sul sito del provider.

## Modelli e provider da citare (edizione 2026)

Il materiale originale (in spagnolo, cartella `fonti_originali/`) cita modelli
del 2023-2024 (GPT-3.5/4, Claude 2/3, Gemini 1.5, LLaMA 2). Questo corso è
**"adattato e aggiornato"**: aggiorna gli esempi di modelli attuali, ma mantieni
intatti i concetti architetturali e didattici (che non cambiano).

Usa questi riferimenti come modelli "attuali" quando servono esempi:

| Provider | Modelli di riferimento 2026 | Uso tipico nell'esempio |
|---|---|---|
| Anthropic | Claude Opus 5, Claude Sonnet 5, Claude Haiku 4.5 | Sonnet 5 per bilanciamento, Haiku 4.5 per compiti veloci/economici |
| OpenAI | GPT-5, GPT-5 mini | GPT-5 mini per compiti semplici, GPT-5 per compiti complessi |
| Google | Gemini 3 Pro, Gemini 3 Flash | Flash per volumi alti, Pro per ragionamento |
| Open source / locale | Llama 3/4, Mistral, DeepSeek V3/R1 (via Ollama o LM Studio) | Esecuzione locale, privacy, costo zero |

Non inventare ID di modello precisi tipo `claude-sonnet-5-20260101`: usa nomi
descrittivi ("Claude Sonnet 5") nel testo discorsivo, e negli esempi di codice
usa gli ID reali quando li conosci con certezza, altrimenti un placeholder
plausibile con un commento `# verifica l'ID modello aggiornato nella doc ufficiale`.

Strumenti no-code: Zapier resta valido ma menziona anche alternative correnti
(n8n, Make) quando la sezione parla di automazione. Copilot/Poe nel materiale
originale erano centrali nel 2024: trattali come una voce nella panoramica
dell'ecosistema, non come argomento a sé.

## Struttura di una dispensa (`dispense/NN_slug.md`)

Segui **esattamente** questo schema, preso dal corso di riferimento
(`11-Corso-IA-Engineering-2026/dispensas/01_apis_llm_fundamentos.md`):

```markdown
# Modulo NN — Titolo della lezione

**Modulo:** NN — Nome del blocco tematico
**Stato:** [ ] Bozza  [ ] Completo  [ ] Rivisto

---

## 0. Prima di iniziare: cosa devi già sapere

(Prerequisiti espliciti, con rimando ai moduli precedenti se serve.)

---

## 1. Primo concetto
## 2. Secondo concetto
...
(sezioni numerate, con tabelle, blocchi di codice, blockquote `> **Nota:**` /
`> **Attenzione:**` / `> **Regola pratica:**` per i punti chiave, esempi
concreti con numeri quando ha senso)

---

## N. Checklist / concetti da ripassare se qualcosa non è chiaro

---

## N+1. Domande di autovalutazione

5 domande aperte, a cui lo studente risponde con parole proprie prima di
passare al notebook. Almeno una deve essere un piccolo calcolo/esercizio
numerico (es. stima di costo, stima di token).

---

*Dispensa del Corso Completo di IA Generativa — Modulo NN*
```

Lunghezza indicativa: 250-450 righe di markdown per dispensa (simile alla
dispensa di riferimento). Non riempire per riempire: se il tema del modulo è
più leggero, va bene una dispensa più corta.

## Struttura di un notebook (`notebooks/NN_slug.ipynb`)

Segui lo schema di `11-Corso-IA-Engineering-2026/notebooks/01_primeras_llamadas_api.ipynb`:

1. **Cella markdown di apertura**: titolo, modulo, tempo stimato, elenco
   puntato "Cosa costruirai", prerequisiti (`pip install ...`), variabili
   d'ambiente richieste in `.env`.
2. **Cella markdown "Cella 0 — Setup"** + cella di codice che carica `.env` e
   verifica le chiavi (stesso pattern di `verificar_clave` del riferimento, in
   italiano: `verifica_chiave`).
3. Alternanza di celle markdown (spiegazione breve, cosa osservare, eventuale
   domanda "prima di eseguire, cosa ti aspetti?") e celle di codice
   **eseguibili e realistiche**, con commenti in italiano.
4. Almeno un **esperimento guidato** con dati/parametri concreti (come
   l'esperimento sulla `temperature` nel riferimento) e una cella markdown
   subito dopo per raccogliere le osservazioni dello studente.
5. Almeno un **esercizio con `# TODO`** che lo studente deve completare, non
   solo eseguire.
6. Una sezione finale **"Reto finale" → "Sfida finale"** più corposa che
   integra i concetti del modulo in un mini-progetto eseguibile.
7. Cella finale di **autovalutazione della settimana** (stesse 4 domande del
   riferimento: cosa ha funzionato, cosa non è chiaro, cosa cambieresti, tempo
   dedicato).

### Come generare il file .ipynb (IMPORTANTE)

Non scrivere il JSON del notebook a mano cella per cella nel tool Write: è
fragile e facile da rompere (escaping, virgolette, newline). Costruisci invece
il notebook con uno script Python che generi il JSON corretto, poi eseguilo con
Bash. Usa questo pattern:

```python
import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

cells = [
    md("# Notebook NN — Titolo\n\n**Modulo:** ...\n"),
    code("import os\nfrom dotenv import load_dotenv\n..."),
    # ...
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/NN_slug.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

Dopo aver generato il file, verificalo sempre con:
`python3 -c "import json; json.load(open('notebooks/NN_slug.ipynb'))"`
per essere certo che il JSON sia valido.

## Numerazione e nomi file

- Dispense: `dispense/NN_slug_breve.md` (`NN` a due cifre, `01`-`24`).
- Notebook: `notebooks/NN_slug_breve.ipynb`, **stesso slug della dispensa
  corrispondente** (stessa numerazione, stesso nome dopo il prefisso).
- I due moduli "progetto" (12 e 23) vanno segnalati nel titolo con
  "— Progetto 1" / "— Progetto 2".

## Materiale sorgente

Il materiale originale (spagnolo) è in `fonti_originali/`:
- `Curso completo de IA generativa.docx` (convertito in `curso.txt`, estratti
  per modulo forniti separatamente a ogni autore).
- I notebook `.ipynb` originali con i casi pratici (in spagnolo/inglese) da
  cui adattare esercizi e codice.

**Non tradurre letteralmente**: riscrivi in italiano naturale, aggiorna gli
esempi di modelli/prezzi secondo la tabella sopra, e quando il materiale
originale è didatticamente debole (spiegazioni troppo generiche, esempi
deboli) migliora tu la spiegazione mantenendo il concetto tecnico corretto.

## Coerenza tra moduli

- Non ridefinire da zero concetti già coperti in moduli precedenti del corso:
  rimanda con "(vedi Modulo 0X)".
- La funzione di stima costo (`calcola_costo`) introdotta nel primo notebook
  che tratta le API va riusata (non reinventata) nei notebook successivi che
  fanno chiamate API.
- Usa sempre `python-dotenv` + variabili d'ambiente per le chiavi API, mai
  chiavi hardcoded (stesso principio di sicurezza del corso di riferimento).
