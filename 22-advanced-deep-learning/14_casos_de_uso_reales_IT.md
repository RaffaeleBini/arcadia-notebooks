# Casi d'uso reali: il deep learning avanzato in azione

## Introduzione: dalla teoria alla pratica

Hai percorso una lunga strada. Hai iniziato dalle reti neurali più semplici, sei passato dalle reti neurali convoluzionali (CNN), hai affrontato le reti neurali ricorrenti (RNN) e le loro varianti con memoria a lungo termine (LSTM), hai padroneggiato l'ottimizzazione, esplorato l'elaborazione del linguaggio naturale (NLP), compreso l'attenzione e i Transformer, costruito reti generative avversarie (GAN) e concluso con l'apprendimento non supervisionato, i modelli generativi moderni e l'intelligenza artificiale (IA) responsabile.

Ora arriva la domanda che conta davvero: a che cosa serve tutto questo?

Questo capitolo finale non aggiunge nuova matematica né nuove architetture. Riunisce sei sistemi reali, con un nome proprio, che hanno già cambiato il loro campo, e ti mostra quali tecniche del manuale usano al loro interno. L'idea è semplice: quando avrai finito di leggere, ogni architettura che hai studiato smetterà di essere un diagramma astratto e diventerà il pezzo concreto di un prodotto che gioca a scacchi meglio di qualsiasi essere umano, traduce lingue in tempo reale, prevede la forma di una proteina o genera una voce sintetica capace di leggerti il giornale.

Vedrai anche il rovescio della medaglia. Nessuno di questi sistemi è arrivato al mondo reale senza sollevare domande scomode su pregiudizi, sicurezza o uso improprio. Per questo motivo questo capitolo si collega costantemente al tredicesimo (IA responsabile): la potenza tecnica e la responsabilità non sono capitoli separati, sono la stessa storia raccontata da due angolazioni.

## AlphaGo e AlphaZero: dominare giochi che si credevano impossibili

Nel 2016, un sistema chiamato AlphaGo ha battuto il sudcoreano Lee Sedol, giocatore professionista di nono dan, per quattro partite a una nel gioco del Go. Fino a quel momento si pensava che una macchina non avrebbe potuto dominare il Go per decenni: la scacchiera 19x19 genera un albero di possibilità così immenso che la ricerca esaustiva è impraticabile.

AlphaGo ha risolto il problema con due reti neurali che conosci già dal capitolo tre (CNN e visione artificiale): una rete di politica e una rete di valore. La rete di politica riceve la posizione sulla scacchiera e restituisce una distribuzione di probabilità sulle mosse più promettenti; la rete di valore valuta quella stessa posizione e produce un unico numero che indica quale giocatore sta vincendo. Insieme, riducono un albero di ricerca quasi infinito — una partita di Go dura tra 200 e 250 mosse — a un insieme ristretto di rami ragionevoli, qualcosa che il capitolo cinque (Ottimizzazione I) ti ha insegnato a ottenere tramite l'addestramento basato sui gradienti su grandi volumi di dati.

AlphaGo ha imparato prima imitando le partite di giocatori umani esperti. Il suo successore, AlphaZero, ha fatto un passo in più: ha imparato Go, scacchi e shogi partendo da zero, senza dati umani, giocando solo contro se stesso. In circa quattro ore di autoaddestramento, il suo punteggio Elo superava già quello di Stockfish, il programma di scacchi più forte basato su euristiche manuali. Qui vedi in azione i concetti di ottimizzazione iterativa e di regolazione dei parametri che hai studiato nei capitoli cinque e sei: ogni partita genera nuovi dati, ogni lotto di dati regola i pesi della rete e ogni regolazione migliora la partita successiva.

L'impatto va oltre la scacchiera. La stessa ricetta — due reti che si alimentano a vicenda e imparano per tentativi ed errori su larga scala — ha ispirato sistemi di ottimizzazione industriale, pianificazione logistica e progettazione di farmaci, perché lo schema "esplorare, valutare, ripetere" non è esclusivo dei giochi da tavolo.

## AlphaFold: prevedere la struttura delle proteine

Ogni proteina del corpo umano si ripiega in una forma tridimensionale specifica, e quella forma determina la sua funzione. Per decenni, scoprire sperimentalmente la struttura di una sola proteina poteva richiedere anni di lavoro di laboratorio. AlphaFold, sviluppato da DeepMind, ha cambiato questo equilibrio: ha vinto la competizione CASP (Critical Assessment of Structure Prediction) prevedendo, con una precisione senza precedenti, strutture di proteine che prima erano sconosciute.

Tecnicamente, AlphaFold combina proprio ciò che hai studiato nei capitoli tre e quattro (CNN e modelli avanzati di visione): blocchi residuali profondi che permettono di addestrare reti con moltissimi strati senza che il gradiente svanisca, un problema che il capitolo quattro ti ha spiegato in dettaglio. Su quella base incorpora meccanismi di attenzione — gli stessi che hai studiato nel capitolo nove — per mettere in relazione amminoacidi che sono lontani nella sequenza ma vicini nello spazio tridimensionale finale.

Il risultato non è un esercizio accademico. Capire la struttura di una proteina accelera la progettazione di farmaci, aiuta a comprendere le malattie a livello molecolare e apre la strada a trattamenti che prima richiedevano anni di prove ed errori in laboratorio. È l'esempio perfetto di come un'architettura addestrata a "vedere" schemi spaziali — pensata originariamente per classificare immagini — finisca per risolvere un problema biologico che non avresti nemmeno immaginato quando hai studiato per la prima volta le reti residuali.

## Traduzione automatica: dai Seq2Seq al Transformer

Per anni, tradurre un testo con una macchina ha significato concatenare regole grammaticali e dizionari statistici, con risultati goffi. Il capitolo sette (Sequenze e RNN) ti ha presentato l'alternativa: i modelli sequenza-a-sequenza (Seq2Seq), in cui un codificatore basato su RNN comprime la frase di ingresso in un vettore di contesto, e un decodificatore genera la traduzione parola per parola a partire da quel vettore.

Google ha portato questa idea in produzione con il suo sistema di traduzione automatica neurale (GNMT, Google Neural Machine Translation). GNMT ha sostituito i sistemi basati su frammenti di frase che Google usava fino ad allora e ha ridotto in modo notevole la distanza tra la qualità della traduzione automatica e quella di un traduttore umano. Lo stesso principio di codificatore-decodificatore che hai disegnato in pseudocodice nel capitolo sette è, in sostanza, il motore che ha tradotto la tua ricerca in un'altra lingua l'ultima volta che hai usato un traduttore online.

La rivoluzione successiva è arrivata con il Transformer, protagonista del capitolo nove. Sostituendo la ricorrenza con l'attenzione, il Transformer elabora tutta la frase in parallelo invece che parola per parola, catturando le dipendenze a lungo raggio con più efficacia di qualsiasi RNN. Ha stabilito nuovi riferimenti di qualità nella traduzione automatica e ha gettato le basi di sistemi di comprensione del linguaggio come BERT (Bidirectional Encoder Representations from Transformers) e di generazione di testo come GPT (Generative Pre-trained Transformer). Se ti sei mai chiesto perché il capitolo nove dedichi tanto spazio all'attenzione, ecco la risposta: quella singola idea ha sostituito un'intera famiglia di architetture in appena un paio d'anni.

## WaveNet: la voce sintetica e l'equilibrio tra beneficio e rischio

WaveNet, sviluppato da DeepMind, genera audio di altissima qualità modellando direttamente la forma d'onda, campione per campione. La sua prima versione usava RNN, ma si è evoluta verso le convoluzioni dilatate: filtri che saltano sistematicamente alcuni punti di ingresso per ampliare il campo recettivo senza far esplodere il costo computazionale. È lo stesso principio di convoluzione che hai studiato nel capitolo tre, portato in una dimensione diversa: invece di scorrere sui pixel di un'immagine, scorre su una sequenza temporale di audio, un ponte diretto con le idee sulle sequenze del capitolo sette.

Oggi questa tecnologia genera le voci che ascolti quando parli con Google Assistant. Il beneficio sociale è concreto: una voce sintetica naturale facilita l'accesso ai servizi digitali per le persone con disabilità visiva o difficoltà di lettura, cosa che il manuale di IA responsabile del capitolo tredici individua come una delle forme più chiare di beneficio tecnologico.

Ma WaveNet illustra anche l'altra faccia della medaglia. La stessa capacità che genera una voce calda e naturale può clonare la voce di una persona reale senza il suo consenso, alimentando frodi o disinformazione tramite audio falsificato. Il capitolo tredici solleva esattamente questo dilemma e propone mitigazioni tecniche concrete, come la filigrana digitale (watermarking) nell'audio sintetico, che permette di risalire alla sua origine e di sostenere la fiducia pubblica nelle registrazioni. Nessun altro caso di questo capitolo riassume meglio un'idea centrale del corso: la tecnica non è neutra, e chi la dispiega ha la responsabilità di prevedere come può essere usata contro le persone.

## GAN generative: volti che non esistono e dati che mancano

Il capitolo dieci ti ha insegnato come due reti — un generatore e un discriminatore — competano tra loro finché il generatore produce campioni indistinguibili dai dati reali. StyleGAN, sviluppato da NVIDIA, ha portato questa idea a un estremo sorprendente: genera volti umani ad alta risoluzione che non appartengono a nessuna persona reale e che risultano quasi impossibili da distinguere da una fotografia autentica. Le sue applicazioni vanno dall'intrattenimento e dalla realtà virtuale alla creazione di contenuti digitali.

Oltre al volto sintetico, le GAN risolvono un problema molto pratico in ambiti sensibili: la scarsità di dati etichettati. Nell'imaging medico, ottenere abbastanza esempi annotati di una specifica malattia è costoso e lento. Una GAN addestrata correttamente può generare immagini mediche sintetiche che ampliano il set di addestramento e migliorano le prestazioni dei modelli diagnostici, senza compromettere la privacy dei pazienti reali. Altre varianti, come SRGAN, aumentano la risoluzione di immagini satellitari o mediche preservando i dettagli fini; CycleGAN trasferisce lo stile artistico di un'immagine a un'altra; AttnGAN combina attenzione e generazione per creare immagini a partire da descrizioni testuali.

Qui si collega anche il capitolo dodici (Modelli generativi moderni): le stesse idee di spazio latente e campionamento che hai studiato con gli autoencoder variazionali (VAE) ricompaiono in queste applicazioni delle GAN, perché entrambe le famiglie di modelli condividono l'obiettivo di apprendere la distribuzione sottostante dei dati per poter generare campioni nuovi e coerenti.

## Modelli linguistici e pregiudizio: GPT-2, GPT-3 e la lezione di responsabilità

I capitoli otto (NLP avanzato) e nove (Attenzione e Transformer) ti hanno mostrato come modelli come GPT generino testo coerente parola per parola, condizionando ogni previsione su tutto ciò che è stato scritto prima. GPT-2, con 1500 milioni di parametri, ha dimostrato una capacità sorprendente di produrre testo grammaticalmente corretto e pertinente al contesto; il suo successore, GPT-3, ha ampliato quella capacità di vari ordini di grandezza.

Questo stesso potere generativo porta con sé un rischio che il capitolo tredici analizza in profondità. Uno studio intitolato "The Woman Worked as a Babysitter: On the Biases in Language Generation" ha esaminato come cambiava il testo generato da GPT-2 modificando soltanto il soggetto della frase di partenza. Cambiare "l'uomo lavorava come" in "la donna lavorava come" produceva continuazioni sistematicamente diverse e spesso cariche di pregiudizio; lo stesso accadeva confrontando "l'uomo nero lavorava come" con "l'uomo bianco lavorava come". Il modello non ha inventato questi pregiudizi: li ha appresi dai dati di addestramento, che riflettono le disuguaglianze presenti nei testi che come società consumiamo.

Un problema collegato, e non meno importante, è la fragilità di fronte a perturbazioni minime: lo studio "Intriguing Properties of Neural Networks" ha mostrato che un classificatore di immagini all'avanguardia poteva riconoscere correttamente la foto di un panda e, dopo aver aggiunto un rumore quasi impercettibile per l'occhio umano, classificarla con alta confidenza come un gibbone. Applica lo stesso errore a un sistema di guida autonoma che confonde un segnale di stop con un altro, e capirai perché il capitolo tredici insiste tanto sulla validazione robusta prima di distribuire qualsiasi modello.

La risposta dell'industria non è stata abbandonare questi modelli, ma governarli meglio: valutazioni sistematiche dei pregiudizi prima del lancio, set di addestramento più diversificati, trasparenza sui limiti noti e test di robustezza di fronte a input avversari. È esattamente il ciclo di lavoro responsabile che hai imparato nell'ultimo capitolo tecnico del corso.

## Riflessione finale del corso

Se questi sei casi dimostrano qualcosa, è che il deep learning avanzato non vive nei paper né nelle slide: vive nel traduttore che usi sul telefono, nell'assistente vocale che accende la luce del tuo salotto, nell'algoritmo che aiuta un radiologo a individuare un tumore prima e nel laboratorio che progetta un farmaco con l'aiuto di una previsione della struttura proteica che un tempo richiedeva anni.

Dimostrano anche qualcosa di più sottile: gli stessi mattoni — convoluzioni, attenzione, reti residuali, generazione avversaria, addestramento per gradiente — vengono ricombinati più e più volte per risolvere problemi che, a prima vista, non hanno nulla in comune tra loro. Padroneggiare i fondamenti che hai studiato in questo manuale non ti prepara a un singolo lavoro; ti prepara a riconoscere lo schema giusto davanti a un problema nuovo e a scegliere lo strumento adeguato dalla tua cassetta degli attrezzi.

Concludere questo corso non è la fine del percorso, è il punto di partenza. Il campo avanza ogni mese: nuove architetture, nuovi modelli, nuovi casi d'uso che oggi non esistono nemmeno. Il modo migliore per restare aggiornato non è memorizzare ogni progresso, ma mantenere la curiosità che ti ha portato fin qui. **Continua a leggere paper, continua a sperimentare con il codice e continua a chiederti non solo come funziona un modello, ma per che cosa e per chi lo stai costruendo.**

## Riepilogo dei punti chiave

- AlphaGo e AlphaZero combinano una rete di politica e una rete di valore, addestrate tramite ottimizzazione basata sui gradienti, per dominare giochi con alberi di ricerca quasi infiniti
- AlphaFold usa blocchi residuali e meccanismi di attenzione per prevedere la struttura tridimensionale delle proteine, accelerando la ricerca biomedica e la progettazione di farmaci
- I modelli Seq2Seq e, successivamente, il Transformer hanno trasformato la traduzione automatica, chiudendo gran parte della distanza tra la qualità di una macchina e quella di un traduttore umano
- WaveNet applica convoluzioni dilatate a sequenze audio per generare voce sintetica naturale, e la sua storia illustra sia il beneficio sociale sia il rischio di un uso improprio della stessa tecnologia
- Le GAN, come StyleGAN, generano immagini realistiche e dati sintetici che risolvono problemi pratici di scarsità di dati in ambiti sensibili come l'imaging medico
- I modelli linguistici come GPT-2 e GPT-3 ereditano i pregiudizi presenti nei loro dati di addestramento, il che richiede valutazione, trasparenza e test di robustezza prima di qualsiasi distribuzione
- Tutte le tecniche del manuale — convoluzioni, sequenze, attenzione, generazione avversaria, ottimizzazione — ricompaiono combinate in sistemi reali che già riguardano milioni di persone

## Mettilo in pratica

Scegli un caso d'uso reale del tuo settore — può essere agroalimentare, forestale, sanitario, biotecnologico, energetico, retail o qualsiasi altro che conosci da vicino — e rispondi a queste domande:

- Quale problema concreto risolverebbe un sistema di deep learning in quel caso?
- Che tipo di dato gestiresti: immagine, testo, serie temporale o una combinazione di più tipi?
- Quali architetture del manuale applicheresti, e perché proprio quelle e non altre? Pensa alle CNN per le immagini, alle RNN o ai Transformer per sequenze e testo, alle GAN o ai VAE se ti servono dati sintetici
- Quali rischi individueresti a partire dal capitolo tredici prima di distribuire quel sistema: pregiudizi nei dati, robustezza di fronte a input inattesi, impatto sulle persone coinvolte?
- Quale misura di mitigazione proporresti per ciascun rischio individuato?

**Scrivi la tua risposta in mezza pagina e condividila con qualcuno del tuo team o della tua classe.** Non deve essere perfetta: l'obiettivo è allenare la stessa muscolatura che hai usato in ogni capitolo di questo manuale, ma ora applicata a un problema che ti sta davvero a cuore.

## Glossario delle sigle usate in questo capitolo

- CASP: Critical Assessment of Structure Prediction
- CNN: rete neurale convoluzionale (Convolutional Neural Network)
- GAN: rete generativa avversaria (Generative Adversarial Network)
- GNMT: Google Neural Machine Translation
- GPT: Generative Pre-trained Transformer
- IA: intelligenza artificiale
- LSTM: memoria a lungo e breve termine (Long Short-Term Memory)
- BERT: Bidirectional Encoder Representations from Transformers
- NLP: elaborazione del linguaggio naturale (Natural Language Processing)
- RNN: rete neurale ricorrente (Recurrent Neural Network)
- Seq2Seq: sequenza a sequenza (Sequence-to-Sequence)
- SRGAN: Super-Resolution Generative Adversarial Network
- VAE: autoencoder variazionale (Variational Autoencoder)
