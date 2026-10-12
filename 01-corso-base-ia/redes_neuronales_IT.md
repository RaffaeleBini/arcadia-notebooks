# Introduzione alle Reti Neurali
### Materiale per una lezione introduttiva (~1 ora) — Senza conoscenze pregresse di matematica avanzata

---

## Obiettivo della lezione

Al termine di questa lezione, gli studenti sapranno:
- Spiegare cos'è un neurone artificiale e come "decide"
- Capire come si collegano i neuroni per formare una rete
- Comprendere la differenza tra una rete semplice e una rete complessa
- Sapere cos'è il Deep Learning e perché è importante oggi

---

## 1. Il neurone artificiale

### L'idea di partenza: il cervello umano

Il nostro cervello è formato da miliardi di cellule chiamate **neuroni**. Ogni neurone riceve segnali da altri neuroni, li "elabora" e decide se trasmettere o meno quel segnale in avanti.

Le reti neurali artificiali si ispirano a questa idea, ma in modo molto più semplice: sono un modello matematico che imita, in maniera molto basilare, come un neurone prende decisioni.

### Come "decide" un neurone artificiale?

Immagina di stare decidendo se **uscire a camminare** oggi. La tua decisione dipende da vari fattori:

| Fattore | Conta molto? | Situazione oggi |
|---|---|---|
| Fa bel tempo? | Molto | Sì |
| Hai tempo libero? | Molto | Sì |
| C'è molto traffico? | Poco | Sì |

Tu non dai lo stesso peso a ogni fattore. Il **clima** e il **tempo libero** ti importano molto più del traffico. Alla fine, sommi mentalmente l'importanza di ogni fattore e prendi una decisione: uscire o non uscire.

Un neurone artificiale fa esattamente questo:

1. **Riceve diversi input** (dati: ad esempio, "temperatura", "tempo libero", "traffico")
2. **Dà un'importanza (peso) a ciascun input** — alcuni pesano più di altri
3. **Somma tutto** per ottenere un valore totale
4. **Applica una funzione di attivazione**: un "filtro" che decide se quella somma è sufficiente per "attivarsi" (dire "sì, esco") oppure no

> **Metafora chiave:** la funzione di attivazione è come un interruttore della luce. Se l'energia accumulata supera una certa soglia, la luce si accende (il neurone si attiva); altrimenti resta spenta.

### In sintesi
Un neurone artificiale = **input + pesi + somma + filtro di decisione = output**

---

## 2. Rete neurale semplice

### Da un neurone a una squadra

Un singolo neurone può prendere decisioni molto semplici, ma il mondo reale è più complicato. Per questo colleghiamo molti neuroni tra loro, formando una **rete neurale**.

> **Metafora:** pensa a una piccola azienda. Ci sono persone che ricevono le informazioni del cliente (input), persone che le analizzano e discutono internamente (strato nascosto), e una persona che dà la risposta finale al cliente (output).

### Le tre parti di una rete semplice

1. **Strato di input**: riceve i dati originali (ad esempio, caratteristiche di un frutto: colore, dimensione, forma)
2. **Strato nascosto**: un gruppo di neuroni che elabora quell'informazione e cerca pattern
3. **Strato di output**: dà il risultato finale (ad esempio, "è una mela" o "è un'arancia")

### Esempio pratico: classificare i frutti

Immagina che vogliamo che la rete distingua tra una **mela** e un'**arancia**.

- **Input**: colore (rosso/arancione), forma (rotonda/ovale), dimensione
- **Strato nascosto**: combina queste caratteristiche e cerca pattern ("se è rosso E rotondo, probabilmente è una mela")
- **Output**: la rete dà la sua risposta finale: "mela" o "arancia"

### Come "impara" la rete?

All'inizio, la rete non sa nulla: i pesi (l'importanza di ogni input) sono casuali. Le mostriamo molti esempi di frutti già etichettati ("questa è una mela", "questa è un'arancia").

Ogni volta che la rete sbaglia, regola leggermente i suoi pesi per avvicinarsi di più alla risposta corretta. Ripetendo questo processo migliaia di volte, la rete migliora.

> **Metafora:** è come imparare a lanciare freccette su un bersaglio a occhi bendati. Ogni lancio fallito ti dà informazioni ("troppo a sinistra", "troppo in basso") e correggi poco a poco finché non colpisci sempre più spesso.

---

## 3. Rete neurale complessa

### Perché servono più strati e più neuroni?

Distinguere mele da arance è un problema semplice. Ma cosa succede se vogliamo che la rete riconosca un **volto umano** in una foto? Questo è molto più complesso.

Per i problemi difficili, usiamo reti con **molti strati nascosti** e **molti neuroni in ogni strato**.

### L'idea di gerarchia: dal semplice al complesso

> **Metafora dei mattoncini Lego:** per costruire una casa di Lego, prima colleghi mattoncini piccoli per formare pareti, poi combini le pareti per formare le stanze, e infine combini le stanze per formare la casa completa.

Una rete neurale complessa funziona in modo simile quando analizza un'immagine:

1. **Primi strati**: individuano elementi molto semplici (bordi, linee, contrasti di colore)
2. **Strati intermedi**: combinano quelle linee per formare forme (occhi, naso, bocca)
3. **Strati finali**: combinano quelle forme per riconoscere l'oggetto completo (un volto)

Ogni strato costruisce su quanto individuato dallo strato precedente, proprio come i mattoncini Lego si combinano per formare strutture sempre più complesse.

### Più strati = più capacità, ma anche più difficoltà

Quanti più strati e neuroni ha una rete:
- Può imparare pattern più complessi
- Ma ha anche bisogno di **più dati** e **più tempo** per addestrarsi correttamente

---

## 4. Deep Learning (Apprendimento Profondo)

### Cos'è?

Il **Deep Learning** (apprendimento profondo) è semplicemente il nome che diamo alle reti neurali che hanno **molti strati nascosti** (non solo uno o due, ma decine o addirittura centinaia).

> La parola "profondo" (*deep*) si riferisce alla **profondità** della rete, cioè al numero di strati che ha — non al fatto che sia più "intelligente" di per sé.

### Perché è diventato così popolare?

Tre fattori si sono combinati negli ultimi anni:

1. **Molti più dati disponibili** (foto, testi, video su internet)
2. **Computer molto più potenti** (in particolare le schede grafiche, le GPU)
3. **Tecniche migliori** per addestrare reti con molti strati senza che "si perdano" nel processo

### Applicazioni reali che probabilmente già conosci

- **Riconoscimento di immagini**: identificare oggetti, persone o scene nelle foto (ad esempio, lo sblocco facciale dello smartphone)
- **Riconoscimento vocale**: assistenti virtuali che capiscono ciò che dici
- **Traduzione automatica**: traduttori che capiscono il contesto di una frase, non solo parola per parola
- **Raccomandazioni**: piattaforme che suggeriscono cosa guardare o ascoltare in base ai tuoi gusti

---

## Riepilogo visivo del percorso

```
Neurone artificiale  →  Rete neurale semplice  →  Rete neurale complessa  →  Deep Learning
   (1 decisione)         (pochi strati)            (molti strati/neuroni)     (rete MOLTO profonda)
```

---

## Domande di ripasso

1. Quali elementi compongono un neurone artificiale e quale funzione svolge ciascuno?
2. Nella metafora del "decidere se uscire a camminare", cosa rappresentano i "pesi" di ogni fattore?
3. Quali sono i tre strati di base di una rete neurale semplice e cosa fa ciascuno?
4. Come "impara" una rete neurale dai propri errori? Spiegalo con parole tue.
5. Usando la metafora dei mattoncini Lego, spiega perché una rete complessa ha bisogno di più strati per riconoscere un volto in un'immagine.
6. Cosa significa esattamente la parola "profondo" nel Deep Learning?
7. Cita due applicazioni reali di Deep Learning che usi nella tua vita quotidiana.

---

*Materiale elaborato per un'introduzione concettuale alle reti neurali, senza richiedere conoscenze pregresse di matematica avanzata.*
