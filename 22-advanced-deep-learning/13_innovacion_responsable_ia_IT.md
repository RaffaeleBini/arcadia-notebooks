# Capitolo 13: Innovazione responsabile e intelligenza artificiale

**Acronimi usati in questo capitolo:** intelligenza artificiale (IA) · reti neurali convoluzionali (CNN) · reti neurali ricorrenti (RNN) · memoria a breve e lungo termine (LSTM) · IA spiegabile (XAI) · spiegazioni interpretabili indipendenti dal modello, di portata locale (LIME) · metodo del segno del gradiente veloce (FGSM) · ascesa di gradiente proiettata (PGA) · soddisfacibilità modulo teorie (SMT) · logica temporale lineare (LTL) · test di associazione di word embedding (WEAT) · Regolamento generale sulla protezione dei dati (RGPD) · Unione Europea (UE) · Organizzazione per la cooperazione e lo sviluppo economico (OCSE) · Organizzazione delle Nazioni Unite (ONU) · Istituto degli ingegneri elettrici ed elettronici (IEEE) · Profilazione per la gestione dei delinquenti correzionali per sanzioni alternative (COMPAS) · tasso di veri positivi (TPR)

## Perché questo argomento è importante

Ormai sai addestrare reti neurali convoluzionali (CNN) che classificano immagini quasi con la precisione di una persona, reti ricorrenti (RNN) che generano linguaggio coerente e modelli generativi che creano immagini realistiche. Quel potere tecnico è già reale: AlphaFold ha vinto la competizione CASP prevedendo strutture di proteine sconosciute, i modelli linguistici come GPT-2 e GPT-3 scrivono testi grammaticalmente corretti e coerenti con il contesto, e i sistemi di apprendimento per rinforzo battono giocatori esperti in giochi complessi come il Go. Il problema non è se l'IA funzioni: è che cosa succede quando funziona su dati e decisioni che incidono su persone reali.

Gli esempi abbondano, e tutti condividono lo stesso schema: un modello tecnicamente competente che causa un danno che nessuno ha progettato di proposito. L'algoritmo COMPAS, usato nei tribunali statunitensi per stimare il rischio di recidiva, segnalava come ad alto rischio le persone afrodiscendenti con un tasso di falsi positivi molto superiore a quello delle persone bianche. Nel 2015, il sistema di riconoscimento delle immagini di Google Foto ha etichettato come "gorilla" fotografie di persone nere. Gli strumenti di ricerca di lavoro hanno mostrato annunci di posti ben retribuiti agli uomini con una frequenza molto maggiore che alle donne. Lo scandalo Cambridge Analytica ha rivelato che i dati personali di milioni di utenti di Facebook erano stati raccolti senza un vero consenso per influenzare campagne politiche. E a livello puramente tecnico, lo studio "Intriguing Properties of Neural Networks" ha mostrato che basta una perturbazione impercettibile per l'occhio umano perché un classificatore confonda un panda con un gibbone: trasferito a un sistema di guida autonoma, lo stesso tipo di errore potrebbe confondere un segnale di stop con uno di dare precedenza.

Nessuno di questi sistemi è stato progettato con l'intenzione di discriminare o di sbagliare. Il pregiudizio è entrato dalla porta di servizio: dati di addestramento poco rappresentativi, etichette che ereditano pregiudizi storici, metriche di successo che non includono l'equità come obiettivo. È proprio per questo motivo che questo capitolo chiude il blocco tecnico del manuale: chi progetta e addestra modelli occupa una posizione di potere, perché le sue decisioni su dati, architetture e metriche determinano quali opportunità si aprono e quali si chiudono per altre persone. Con questo potere arriva una responsabilità che non è facoltativa né esterna al lavoro tecnico, ma ne fa parte.

In DIH DATAlife lo vediamo da vicino quando accompagniamo team di settori come Salute-Cura o Agro-Mare Alimentazione in progetti di digitalizzazione con componenti di IA: un modello di supporto alla diagnosi o di priorizzazione degli aiuti che funziona peggio per un sottogruppo della popolazione non è un dettaglio tecnico di poco conto, è un difetto di progettazione con conseguenze dirette sulle persone. Questo capitolo ti fornisce il vocabolario e gli strumenti concettuali per individuare questo tipo di difetto prima che arrivi in produzione.

## Spiegazione concettuale con analogie

### La ricetta e il libro delle norme

Pensa all'addestrare un modello con i soli dati come a cucinare seguendo solo ciò che ti hanno insegnato a mangiare da piccolo, senza alcun libro di norme nutrizionali. Se da piccolo hai mangiato solo un tipo di piatto, cucinerai bene quel piatto e male quasi tutti gli altri, senza nemmeno accorgerti del limite: per te, la tua esperienza sembra la totalità della cucina possibile. È ciò che il materiale di riferimento chiama addestramento "vaniglia", ordinario: il modello eredita ciò che c'è nei dati, con i suoi pregiudizi e i suoi limiti, senza alcun meccanismo che lo obblighi a comportarsi diversamente quando i dati non bastano.

L'apprendimento guidato da specifiche aggiunge quel libro di norme. Non sostituisce l'esperienza (i dati), ma impone regole esplicite che il risultato finale deve rispettare anche quando i dati, da soli, non le avrebbero mai insegnate: "sii robusto rispetto a cambiamenti impercettibili nell'ingresso", "non discriminare in base ad attributi irrilevanti per il compito", "quando non riconosci la situazione, dichiaralo con meno fiducia invece che con più fiducia". È la differenza tra un modello che impara esattamente ciò che c'è nei dati, pregiudizi compresi, e uno a cui si chiede in più di rispettare certe proprietà per progetto.

### La scatola nera e la visita medica

La mancanza di trasparenza di un modello profondo si capisce meglio se la confronti con due modi diversi di ricevere una diagnosi medica. Nel primo, qualcuno ti dice solo "hai questa malattia", senza alcuna spiegazione: devi fidarti a occhi chiusi. Nel secondo, la stessa persona ti spiega quali sintomi, quali esami e quale ragionamento la portano a quella conclusione: puoi valutare se il ragionamento ha senso, chiedere un esame aggiuntivo o cercare un secondo parere. Un modello di apprendimento profondo, con i suoi milioni di parametri interconnessi, funziona di default come la prima visita: produce un risultato senza esporre il perché. L'IA spiegabile (XAI) è l'insieme di tecniche, come le mappe di salienza o le spiegazioni interpretabili indipendenti dal modello di portata locale (LIME), che tenta di trasformare quella visita silenziosa nella seconda, quella che puoi mettere in discussione.

### Il rilevatore di bugie che rileva solo le bugie evidenti

Una valutazione di robustezza avversariale debole somiglia a un rilevatore di bugie che controlla solo le menzogne più ovvie e maldestre. Se testi il tuo modello solo con attacchi deboli, puoi concludere, erroneamente, che è robusto: il problema non sta nel modello, sta nel fatto che non l'hai messo alla prova abbastanza duramente. Il materiale di riferimento descrive studi del 2018 in cui, applicando valutazioni avversariali più rigorose, la maggior parte delle difese pubblicate fino a quel momento è crollata, con l'eccezione dell'addestramento avversariale. La lezione di fondo, valida ben oltre la robustezza avversariale: una valutazione debole dà una falsa sensazione di sicurezza, e quella falsa sensazione è di per sé un rischio.

## Sviluppo teorico

### Specifiche desiderabili e apprendimento guidato da specifiche

La domanda che organizza la prima metà di questo capitolo è diretta: come si assicura che un algoritmo di apprendimento automatico sia sicuro per il deployment? La risposta passa per l'imporgli **specifiche desiderabili**, allo stesso modo in cui qualsiasi prodotto dell'ingegneria supera un controllo di qualità prima di arrivare sul mercato. Il materiale propone vari esempi concreti: un classificatore di immagini deve essere robusto rispetto a perturbazioni avversariali, un predittore di sistemi dinamici deve rispettare le leggi della fisica, un classificatore di cifre deve essere invariante rispetto ad attributi irrilevanti come il colore del tratto, un modello che tratta dati sensibili deve rispettare la privacy differenziale, e un modello che si trova davanti a dati fuori dalla distribuzione di addestramento deve mostrare più incertezza, non più fiducia.

Il problema di fondo è che, quando addestri con dati limitati, il modello può imparare correlazioni spurie che migliorano la metrica di addestramento senza avere nulla a che fare con il compito reale. Se i dati sono distorti e limitati, il modello eredita quella distorsione e quella mancanza di robustezza, a meno che tu non glielo impedisca esplicitamente. L'apprendimento guidato da specifiche consiste proprio nell'incorporare queste regole dove i dati non bastano.

### Robustezza avversariale: formalizzazione e addestramento

La specifica più studiata nel materiale è la robustezza avversariale. Formalmente, per una rete neurale f con parametri θ, un ingresso x e un'etichetta vera in formato one-hot, si richiede che l'indice della probabilità massima in uscita coincida con quello dell'etichetta vera per qualsiasi perturbazione δ all'interno di un insieme S di cambiamenti impercettibili, tipicamente definito come una palla di raggio piccolo nella norma L-infinito:

argmax f(x + δ; θ) = argmax(etichetta) per ogni δ ∈ S

L'**addestramento avversariale** insegna al modello a rispettare quella specifica aggiungendo all'ottimizzazione standard un passo di aumento dei dati: invece di addestrare solo sull'immagine originale, si addestra anche sulla peggiore perturbazione possibile all'interno di S. Questo trasforma il problema in un'ottimizzazione min-max: prima si massimizza la perdita rispetto alla perturbazione δ all'interno di S (passo interno), e poi si minimizzano i parametri rispetto a quella perdita già massimizzata (passo esterno). Trovare il massimo esatto è, in generale, un problema NP-difficile, quindi in pratica lo si approssima con l'**ascesa di gradiente proiettata** (PGA): si compie un passo di gradiente che massimizza la perdita e, se la perturbazione risultante esce da S, la si proietta di nuovo nel punto più vicino all'interno di S. Il **metodo del segno del gradiente veloce** (FGSM) è la versione più semplice di questa idea per la norma L-infinito: sostituisce il gradiente con il suo segno, e può essere combinato con ottimizzatori come il momento o Adam, che conosci già dai capitoli precedenti.

### Valutare bene la robustezza: forza della valutazione e offuscamento dei gradienti

Una valutazione avversariale è affidabile quanto lo sforzo che metti nel trovare il caso peggiore. Il materiale individua diverse leve che rafforzano una valutazione: aumentare il numero di passi di PGA, usare più inizializzazioni casuali della perturbazione (per individuare comportamenti anomali dell'ottimizzatore), provare diversi ottimizzatori e completare gli attacchi white-box (con accesso completo ai pesi) con attacchi black-box, che si limitano a consultare ingressi e uscite. Una valutazione debole produce un **offuscamento dei gradienti**: il modello impara, senza che nessuno glielo chieda esplicitamente, a costruire una superficie di perdita così irregolare che l'ottimizzatore basato sul gradiente non trova il vero caso peggiore. Il risultato è una falsa sensazione di robustezza, coerente con quanto detto nel paragrafo precedente: quando sono state applicate valutazioni più esigenti, l'accuratezza avversariale di molte difese pubblicate è crollata a zero, e solo l'addestramento avversariale ha resistito.

### Algoritmi di verifica

L'addestramento avversariale riduce il rischio, ma non lo elimina né lo dimostra matematicamente: resta una valutazione empirica su un insieme finito di attacchi. Gli **algoritmi di verifica** cercano garanzie dimostrabili che nessuna perturbazione all'interno di S possa violare la specifica. Si dividono in due famiglie. Gli **algoritmi completi**, come quelli basati sulla programmazione intera mista, offrono una prova esaustiva: confermano che la specifica è soddisfatta oppure presentano un controesempio, ma il loro costo computazionale rende difficile scalarli a reti profonde e grandi. Gli **algoritmi incompleti** sono più scalabili: quando trovano una prova, questa è una garanzia reale, ma non sempre la trovano anche se la specifica è soddisfatta, quindi offrono un limite inferiore sul grado di soddisfacimento.

La tecnica centrale dei metodi incompleti è la **propagazione dei limiti**: dato un insieme di ingressi possibili (per esempio, tutte le perturbazioni all'interno di S), si calcolano limiti inferiori e superiori dopo ogni strato lineare e ogni attivazione, ottenendo un insieme di uscita sovra-approssimato. Se quell'insieme sovra-approssimato sta interamente da un lato del confine decisionale, puoi concludere che anche l'insieme reale soddisfa la specifica. Se l'insieme sovra-approssimato attraversa il confine, non puoi concludere nulla sull'insieme reale: la sovra-approssimazione può essere troppo lasca. Quanto più stretta è la propagazione dei limiti, tanto minore è il divario tra ciò che la verifica garantisce e ciò che mostra la valutazione avversariale empirica, e tanta più informazione utile ottieni sul vero grado di soddisfacimento della specifica.

Queste tecniche non si limitano alla robustezza avversariale: il materiale menziona anche la **coerenza semantica** (in un'auto autonoma, confondere un gatto con un cane è un errore minore rispetto a confondere un gatto con un'auto) e la **coerenza fisica** (un predittore di sistemi dinamici che rispetti la conservazione dell'energia) come altre specifiche verificabili con gli stessi strumenti.

### Etica dei dati di addestramento

La seconda metà del capitolo si allontana dalla verifica matematica ed entra nel terreno etico. Tutto comincia dai dati, che non sono una risorsa neutra ma trascinano con sé proprietà etiche specifiche. Tre domande riassumono il punto di partenza: i dati sono stati raccolti con il consenso delle persone rappresentate?, una cosa che casi come Cambridge Analytica dimostrano che non si può dare per scontata; la rappresentazione dei dati è sufficientemente diversificata, o favorisce alcuni gruppi a scapito di altri?; e l'etichettatura e la curatela dei dati contengono associazioni pregiudizievoli ereditate da pregiudizi umani storici?, come accadde con le prime versioni di ImageNet, che includevano una categoria "Persone" con etichette dispregiative.

Questi problemi nel flusso dei dati si traducono in **pregiudizio algoritmico** nel mondo reale: il sistema COMPAS nella giustizia penale, la disuguaglianza negli annunci di lavoro ben retribuito, le prestazioni peggiori del riconoscimento delle immagini per le minoranze, e strumenti medici che funzionano peggio per persone con identità intersezionali, con il rischio di un accesso disuguale a servizi e farmaci che possono salvare vite.

### Potere, responsabilità e innovazione responsabile

Chi progetta e sviluppa queste tecnologie occupa una posizione di potere, intesa come capacità di influenzare lo stato delle cose e la vita di altre persone. La ricerca scientifica, compresa quella sull'apprendimento automatico, non è neutrale rispetto ai valori: è una pratica sociale governata da norme condivise, sia epistemiche (la necessità di replicare i risultati) sia morali (l'accettabilità di fare ricerca con persone senza il loro consenso), e queste norme hanno effetti sociali profondi quando in gioco ci sono interessi importanti e l'impatto è incerto.

L'**innovazione responsabile** è definita nel materiale come un processo trasparente e iterativo attraverso cui gli attori sociali e le persone che sviluppano tecnologia diventano reciprocamente sensibili alle necessità delle altre parti, per garantire il valore etico e sociale dell'attività scientifica. Questa responsabilità ha una parte intrinseca (progettare sistemi sensibili alle considerazioni etiche e sociali, limitando fin dall'inizio il rischio di danno) e una parte estrinseca (assicurare che la tecnologia venga dispiegata e utilizzata con giudizio, perché una tecnologia robusta e sicura può comunque essere usata in modo improprio da terzi).

Almeno 84 codici etici diversi, proposti da organismi come l'Unione Europea (UE), l'Organizzazione per la cooperazione e lo sviluppo economico (OCSE), la Beijing Academy of Artificial Intelligence e il Future of Life Institute, convergono su un nucleo di valori condivisi: equità, privacy, trasparenza e non maleficenza (il principio del "non causare danno"). L'Organizzazione delle Nazioni Unite (ONU), nella sua Dichiarazione dei diritti umani, aggiunge il diritto collettivo dell'umanità a beneficiare del progresso scientifico. Il problema, sottolinea il materiale, non è la mancanza di buone intenzioni né di valori condivisi: è il divario tra quegli ideali morali astratti e i processi concreti per applicarli.

### Il processo in cinque passi e le due prove finali

Per colmare questo divario, il materiale propone un processo in cinque passi per valutare se un progetto di ricerca o sviluppo è allineato dal punto di vista etico e sociale:

1. **La tecnologia ha usi socialmente vantaggiosi?** La maggior parte delle tecnologie apporta qualche beneficio; tra questi, migliorare la salute e il benessere, ampliare l'autonomia e la libertà, produrre risultati più equi, rafforzare istituzioni pubbliche come la sanità o l'istruzione, e affrontare sfide globali come il cambiamento climatico
2. **La tecnologia potrebbe causare danni, diretti o indiretti?** Quasi ogni tecnologia comporta qualche rischio: minare la salute o la dignità umana, limitare la libertà o l'autonomia, generare trattamenti o risultati ingiusti, danneggiare istituzioni pubbliche o violare diritti umani
3. **È possibile mitigare quei rischi, o eliminarli del tutto?** Qui rientrano misure come controllare la diffusione della tecnologia o della conoscenza associata, adottare contromisure tecniche, aiutare il pubblico a capire la nuova tecnologia, e cercare soluzioni di politica pubblica e quadri giuridici
4. **Con quelle misure in atto, l'azione proposta viola qualche linea rossa o vincolo morale?** Questi vincoli, vicini all'etica deontologica, delimitano ciò che non deve mai essere fatto in nessuna circostanza: violare protocolli di consenso, sviluppare armi autonome letali, abilitare una sorveglianza che eroda la fiducia pubblica, o violare il diritto internazionale e i diritti umani
5. **Con quelle misure in atto, i benefici superano i rischi?** Solo se non si attraversa nessuna linea rossa ha senso passare a quest'ultima domanda, valutando anche se esistono alternative più valide

Anche dopo aver concluso che un progetto è fattibile, il materiale raccomanda due prove finali, pensate per contrastare la **cognizione motivata**, cioè la tendenza inconsapevole ad accettare gli argomenti che fanno comodo ai propri interessi. La prima prova consiste nell'identificare tutte le persone che potrebbero essere colpite, capire chi sono e chiedersi se potresti spiegare loro con onestà il tuo ragionamento e la tua decisione; se non ci riesci, è un segnale per rivedere le tue conclusioni. La seconda prova consiste nel proiettarti nel futuro e chiederti se potresti pentirti di questa decisione in seguito, un esercizio che, per esempio, invita a ripensare pratiche con impatto ambientale alla luce del cambiamento climatico.

### Pratiche emergenti di governance

Il materiale si chiude segnalando pratiche concrete già adottate nella comunità dell'apprendimento automatico: le **schede del modello** (model cards), che documentano gli usi previsti e le proprietà di un sistema; i **premi per il pregiudizio** (bias bounties), ispirati ai programmi di ricompensa per la scoperta di vulnerabilità di sicurezza, ma applicati alla ricerca di pregiudizi in modelli e dataset; e l'inclusione di una sezione di impatto etico e sociale nel processo di revisione delle conferenze scientifiche come NeurIPS. A queste si aggiungono tecniche di protezione dei dati come la **privacy differenziale**, che garantisce che l'inclusione o l'esclusione di un singolo dato non alteri in modo percepibile l'output del modello, e l'**apprendimento federato**, che addestra modelli su dati decentralizzati senza bisogno di condividere i dati grezzi, riducendo il rischio di fughe di informazioni. In Europa, il Regolamento generale sulla protezione dei dati (RGPD) traduce gran parte di questi principi in un obbligo di legge per qualsiasi sistema che tratti dati personali.

## Esempio di codice commentato, in Python

Il codice seguente non addestra una rete neurale: **audita** un modello di classificazione binaria già addestrato, calcolando una metrica di equità standard chiamata **differenza di opportunità uguale** (equal opportunity difference), che confronta il tasso di veri positivi (TPR) tra due gruppi demografici. È esattamente il tipo di verifica che il materiale richiede quando parla di "verifiche periodiche delle prestazioni del modello nei diversi gruppi demografici". **Copia ed esegui questo codice in Google Colab.**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n_por_grupo = 1000

# --- Grupo A: grupo mayoritario, bien representado en el histórico ---
# La etiqueta real depende de ingresos y antigüedad laboral, con algo de ruido.
edad_a = rng.normal(40, 10, n_por_grupo)
ingresos_a = rng.normal(50, 15, n_por_grupo)
antiguedad_a = rng.normal(8, 4, n_por_grupo)
score_a = 0.04 * ingresos_a + 0.3 * antiguedad_a - 5
etiqueta_a = (score_a + rng.normal(0, 3, n_por_grupo) > 0).astype(int)

# --- Grupo B: grupo minoritario, peor representado en el histórico ---
# El proceso causal real es el mismo, pero los datos disponibles reflejan
# ingresos declarados sistemáticamente más bajos y una etiqueta con más
# ruido de medición: así se simula un sesgo de representación en el origen
# de los datos, no un sesgo del modelo en sí mismo.
edad_b = rng.normal(38, 10, n_por_grupo)
ingresos_b = rng.normal(38, 15, n_por_grupo)
antiguedad_b = rng.normal(6, 4, n_por_grupo)
score_b = 0.04 * ingresos_b + 0.3 * antiguedad_b - 5
etiqueta_b = (score_b + rng.normal(0, 5, n_por_grupo) > 0).astype(int)

df = pd.concat([
    pd.DataFrame({"edad": edad_a, "ingresos": ingresos_a, "antiguedad": antiguedad_a,
                  "grupo": "A", "etiqueta": etiqueta_a}),
    pd.DataFrame({"edad": edad_b, "ingresos": ingresos_b, "antiguedad": antiguedad_b,
                  "grupo": "B", "etiqueta": etiqueta_b}),
], ignore_index=True)

X = df[["edad", "ingresos", "antiguedad"]].values
y = df["etiqueta"].values
grupo = df["grupo"].values

X_train, X_test, y_train, y_test, grupo_train, grupo_test = train_test_split(
    X, y, grupo, test_size=0.3, random_state=42, stratify=grupo
)

# --- El "modelo ya entrenado" que vamos a auditar ---
# En un caso real, este modelo llegaría ya entrenado desde otro equipo:
# aquí lo ajustamos una sola vez, con un clasificador simple, solo para
# tener algo concreto que auditar en los pasos siguientes.
modelo = LogisticRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

resultados = pd.DataFrame({"grupo": grupo_test, "y_real": y_test, "y_pred": y_pred})


def tasa_verdaderos_positivos(subconjunto):
    """TPR = proporción de positivos reales que el modelo detecta como positivos."""
    positivos_reales = subconjunto[subconjunto["y_real"] == 1]
    if len(positivos_reales) == 0:
        return np.nan
    return (positivos_reales["y_pred"] == 1).mean()


def diferencia_oportunidad_igual(df_resultados, columna_grupo="grupo"):
    """Calcula la TPR por grupo y la diferencia entre el grupo con mejor y
    peor TPR: es la métrica de equal opportunity difference. Un valor
    cercano a cero indica que el modelo detecta a los positivos reales con
    la misma eficacia en ambos grupos; un valor alejado de cero indica que
    un grupo recibe sistemáticamente menos verdaderos positivos que el otro."""
    tprs = df_resultados.groupby(columna_grupo).apply(tasa_verdaderos_positivos)
    return tprs, tprs.max() - tprs.min()


tprs_por_grupo, diferencia = diferencia_oportunidad_igual(resultados)

print("TPR por grupo:")
print(tprs_por_grupo)
print(f"\nDiferencia de oportunidad igual: {diferencia:.3f}")

# --- Visualización ---
fig, ax = plt.subplots(figsize=(6, 5))
colores = ["#4C72B0", "#DD8452"]
barras = ax.bar(tprs_por_grupo.index, tprs_por_grupo.values, color=colores, width=0.5)

ax.set_ylim(0, 1)
ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
ax.set_title("Auditoría de equidad: TPR por grupo demográfico")
ax.axhline(y=tprs_por_grupo.max(), linestyle="--", color="gray", alpha=0.5)

for barra, valor in zip(barras, tprs_por_grupo.values):
    ax.text(barra.get_x() + barra.get_width() / 2, valor + 0.02, f"{valor:.2f}", ha="center")

ax.text(0.5, -0.12, f"Diferencia de oportunidad igual = {diferencia:.3f}",
        transform=ax.transAxes, ha="center", fontsize=10, style="italic")

plt.tight_layout()
plt.show()
```

Se esegui questo codice, vedrai che il gruppo B ottiene una TPR sensibilmente più bassa del gruppo A, nonostante il processo causale che genera l'etichetta reale sia identico nei due gruppi: la differenza nasce dal pregiudizio di rappresentazione che abbiamo introdotto nei dati di origine, esattamente il tipo di errore descritto dal materiale. Nota che la differenza di opportunità uguale considera solo la TPR, cioè come tratta il modello le persone che soddisfano davvero la condizione positiva; esistono altre metriche di equità, come la parità demografica, che invece confrontano il tasso complessivo di previsioni positive tra gruppi senza condizionare all'etichetta reale. Nessuna metrica di equità è universalmente corretta: **scegli** quella che corrisponde al danno concreto che vuoi evitare nel tuo caso d'uso.

## Riepilogo dei punti chiave

- Il debito tra potere tecnico e responsabilità è il filo conduttore del capitolo: chi progetta modelli decide quali opportunità si aprono e quali si chiudono per altre persone
- L'apprendimento guidato da specifiche impone regole esplicite (robustezza, invarianza, privacy differenziale, incertezza davanti a dati fuori distribuzione) che i dati, da soli, non garantiscono
- L'addestramento avversariale risolve un problema min-max: massimizzare la perdita rispetto alla peggiore perturbazione impercettibile e minimizzare i parametri rispetto a quella perdita, usando tecniche come PGA e FGSM
- Una valutazione avversariale debole produce offuscamento dei gradienti e una falsa sensazione di sicurezza; rafforzala con più passi, più inizializzazioni e attacchi black-box
- Gli algoritmi di verifica, completi e incompleti, offrono garanzie dimostrabili tramite la propagazione dei limiti, in cambio di un costo computazionale o di una copertura parziale
- Il pregiudizio algoritmico nasce nei dati (consenso, rappresentazione, associazioni pregiudizievoli) e si manifesta nel mondo reale: giustizia penale, accesso al lavoro, riconoscimento delle immagini e diagnosi medica
- L'innovazione responsabile è un processo trasparente e iterativo tra chi sviluppa tecnologia e la società, con un nucleo di valori condivisi: equità, privacy, trasparenza e non maleficenza
- Il processo in cinque passi (beneficio sociale, rischio di danno, mitigazione, linee rosse, bilancio finale) e le due prove finali (persone coinvolte, rimpianto futuro) offrono una procedura concreta per valutare un progetto
- Pratiche come le schede del modello, i premi per il pregiudizio, la privacy differenziale e l'apprendimento federato trasformano i principi etici in processi di governance applicabili

## Mettilo in pratica

Dedica tra quindici e trenta minuti a questa mini-attività:

1. **Copia** il codice della sezione di esempio in un notebook di Google Colab ed eseguilo così com'è, per verificare che riproduci una differenza di opportunità uguale chiaramente diversa da zero
2. **Modifica** la generazione dei dati: porta `ingresos_b` alla stessa distribuzione di `ingresos_a` (media 50 invece di 38) e riduci il rumore di `etiqueta_b` allo stesso livello di `etiqueta_a`. Riesegui il codice e osserva come cambia la differenza di opportunità uguale
3. **Aggiungi** una seconda metrica di equità in una nuova funzione: la parità demografica, definita come la differenza tra la proporzione di previsioni positive (`y_pred == 1`) di ciascun gruppo, senza condizionare all'etichetta reale
4. **Confronta** le due metriche sugli stessi dati distorti del passo uno: coincidono nell'entità del problema, o raccontano storie diverse?
5. **Annota** in due o tre frasi che cosa faresti, come responsabile tecnico, se in produzione rilevassi una differenza di opportunità uguale grande come quella dell'esempio originale: quali informazioni raccoglieresti prima di decidere se il modello può restare in produzione
