# Capitolo 2 — Come si costruisce un computer quantistico

## Perché è importante

Nel capitolo precedente hai parlato dei qubit come se fossero monete che girano in aria. Ma un qubit reale è un oggetto fisico, e costruirlo è una delle imprese di ingegneria più difficili del momento. Capire come si fabbrica un qubit ti aiuta a comprendere perché il calcolo quantistico attuale ha dei limiti (rumore, pochi qubit, tempi di coerenza brevi) e perché molti degli algoritmi che programmerai in questo corso sono progettati proprio per convivere con quei limiti. Questa fase della tecnologia è nota come era **NISQ** (quantistico a scala intermedia e rumoroso): abbiamo computer quantistici, ma ancora imperfetti.

## Spiegazione concettuale

### Il qubit superconduttore: un circuito che si comporta come un atomo

Il modo più diffuso oggi per costruire un qubit (quello usato da Google o IBM) è il **qubit superconduttore**, chiamato anche transmon. Invece di usare un atomo reale, si fabbrica un minuscolo circuito elettrico che, raffreddato a temperature estreme, si comporta esattamente come un atomo con soli due livelli di energia possibili: lo 0 e l'1.

La parte chiave di quel circuito è la **giunzione Josephson**: due strati di metallo superconduttore separati da uno strato isolante ultrasottile. È come una porta che lascia passare la corrente elettrica in modo molto particolare, senza resistenza e con un comportamento quantistico. Grazie a questa giunzione, l'intero circuito si comporta come se avesse livelli di energia discreti, esattamente come un atomo.

### Perché dentro un computer quantistico fa così freddo

Perché il circuito si comporti come un qubit e non come un normale filo, il metallo deve trovarsi nello stato **superconduttore** (senza resistenza elettrica), e questo accade solo a temperature vicine allo zero assoluto. Per questo i computer quantistici di Google stanno dentro un **refrigeratore a diluizione**: una struttura metallica a forma di ragno che raffredda il chip fino a circa 15 millikelvin, più freddo dello spazio esterno.

### Come si controlla un qubit senza distruggerlo

Controllare un qubit significa inviargli impulsi a microonde molto precisi per applicargli le porte quantistiche (quelle viste nel capitolo 1). Il problema è che un impulso progettato male può eccitare il qubit a un livello di energia non voluto (uno "stato fantasma"), rovinando il calcolo. La tecnica **DRAG** (rimozione derivativa tramite porta adiabatica) modella quegli impulsi a microonde per evitare proprio questo errore, ed è uno dei pezzi che rende possibile controllare i qubit con la precisione necessaria.

Inoltre, tutta l'elettronica di controllo deve trovarsi molto vicina al chip quantistico per non perdere precisione lungo il percorso, quindi si studia anche il **CMOS criogenico** (semiconduttore complementare a ossido metallico): chip di controllo classici progettati per funzionare alle stesse temperature estreme, invece che a temperatura ambiente.

### Il nemico comune: la decoerenza

Per quanta cura tu ci metta, un qubit reale è sempre esposto a vibrazioni, campi magnetici residui o imperfezioni del materiale, e nel tempo perde il suo stato quantistico: questo si chiama **decoerenza**. Il tempo che un qubit resiste in sovrapposizione prima di decoerere si chiama **tempo di coerenza**, e più è breve, meno operazioni puoi concatenare prima che il risultato smetta di essere affidabile. Questa è la ragione principale per cui gli algoritmi di questo corso (QAOA, VQE, QRL) sono progettati come circuiti brevi e poco profondi: più dura il circuito, meno tempo ha la decoerenza per rovinare il risultato.

## Riepilogo dei punti chiave

- Un qubit superconduttore (transmon) è un circuito elettrico che, raffreddato quasi allo zero assoluto, si comporta come un atomo con due livelli di energia
- La giunzione Josephson è il componente fisico che rende possibile questo comportamento quantistico
- I refrigeratori a diluizione raffreddano il chip a temperature di millikelvin, indispensabili per mantenere la superconduttività
- La tecnica DRAG modella gli impulsi di controllo per evitare errori; il CMOS criogenico avvicina l'elettronica di controllo al chip
- La decoerenza (perdita dello stato quantistico per il rumore dell'ambiente) è la principale limitazione dell'era NISQ e spiega perché i circuiti quantistici attuali devono essere brevi

## Mettilo in pratica

Non ti serve scrivere codice per questo compito. Cerca il tempo di coerenza approssimativo che di solito viene citato per i processori superconduttori di Google (come Sycamore) e confrontalo con il tempo necessario per eseguire una singola porta quantistica. Calcola, a partire da questi due dati, quante porte quantistiche puoi concatenare al massimo in un circuito prima che la decoerenza inizi a dominare il risultato. Annota il tuo calcolo e la fonte che hai usato: questo conto spiega perché i circuiti dei prossimi capitoli sono così brevi.
