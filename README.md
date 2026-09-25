# Graph Cuts – glatte Zonen aus verrauschten Messwerten – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-graph-cuts-demo.streamlit.app/)**

Zweite Erweiterung (Stück 14) der **Netzwerkfluss-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Fortsetzung von [projektauswahl-demo](https://github.com/sebastian-hanisch/projektauswahl-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – den **minimalen Schnitt als Modell einer binären Beschriftung mit Glattheitsstrafe** (Greig/Porteous/Seheult 1989; Boykov/Jolly 2001) – an einem wachsenden Beispiel.
Ein Liefergebiet ist in Zellen geteilt, in jeder Zelle wird eine Kennzahl gemessen (etwa eine Verspätungsrate), die in einer Zone niedrig und in der anderen hoch liegt, aber verrauscht ist. Jede Zelle bekommt eine von zwei Beschriftungen; die **Energie** setzt sich aus einem **Datenterm** (wie gut passt das Label zur Messung) und einer **Glättungsstrafe λ** für jedes Nachbarpaar mit verschiedenen Labels zusammen.
Für zwei Labels ist die beste Beschriftung ein minimaler Schnitt: Quelle und Senke sind die Zonen, jede Zelle hängt mit ihrem Datenpreis an ihnen, Nachbarn hängen mit λ aneinander. Zum Lösen kommt **Boykov–Kolmogorov** (2004) dazu, das Praxisverfahren für Gitter: zwei dauerhafte Suchbäume, die nach jedem Weg repariert statt neu gebaut werden. Vehikel: ein Raster mit bekannten wahren Zonen (Rechteck, L-Form, Scheibe, Streifen, Inseln) – so lässt sich jede Rekonstruktion **an der Wahrheit messen**.
Die Brücke zwischen Optimierung und Machine Learning: die Energie ist ein Markov Random Field, der Schnitt ihr exakter Löser (nur für zwei Labels).

**Einordnung in die Reihe (die Kanten des Graphen):** In [projektauswahl-demo](https://github.com/sebastian-hanisch/projektauswahl-demo) war die Nachbarstrafe **unendlich** (eine Voraussetzung: wer die Filiale wählt, muss das DC wählen); hier ist sie **endlich**, und der Schnitt wägt ab. Dinic und Edmonds-Karp (Kopien aus den Vorgängern) sind Vergleichsbasis für das neue Verfahren Boykov–Kolmogorov. Mehrklassen-Beschriftung (α-Expansion), nicht submodulare Energien (QPBO) und Bildbeispiele sind bewusst nicht Teil; als drittes Stück der Erweiterung E1 folgte der Gomory-Hu-Baum (gebaut: [gomory-hu-demo](https://github.com/sebastian-hanisch/gomory-hu-demo)). Bisher gebaut: die zwölf Stücke der Hauptlinie und alle drei der Erweiterung E1.
```
edmonds-karp-demo (Wurzel: Restgraph, Rückkanten, Max-Flow = Min-Cut)                  [gebaut]
  ├─ dinic-demo (viele kürzeste Wege je Phase: Niveaugraph, blockierender Fluss)        [gebaut]
  ├─ push-relabel-demo (kein Weg: Überschüsse schieben, Höhen anheben)                 [gebaut]
  └─ ssp-demo (Kosten: der billigste Weg im Restgraphen, Potenziale)                    [gebaut]
       ├─ cycle-canceling-demo, cost-scaling-demo, multicommodity-demo …                [gebaut]
            └─ … fixkosten-netzdesign-demo → benders-demo, slope-scaling-demo           [gebaut]

Erweiterung E1: der Schnitt als Modell (Kind von edmonds-karp-demo und dinic-demo)
  └─ projektauswahl-demo (Voraussetzung = unendliche Nachbarstrafe)                     [gebaut]
       └─ graph-cuts-demo (endliche Nachbarstrafe, Boykov-Kolmogorov)                   [gebaut]
            └─ gomory-hu-demo (alle Schnitte in einem Baum, Gusfield)                   [gebaut]
```

## Ergebnis (Zahlen aus den Tests)

Jede hier genannte Zahl ist in `tests/test_claims.py` belegt: Beispielraster über ihre Seeds, Verteilungen über feste Ziehungen (Seeds ab 100000, dieselben wie in den Vorgänger-Demos, 40 Ziehungen, für die Kurven 10, für die Inseln 20). Standard: Rechteck im 16 × 16-Raster, Rauschen 80 % des Zonenabstands, λ = 1, Boykov–Kolmogorov.
Rauschen, Energien und durchsuchte Kanten sind ganzzahlig und deterministisch (eigener Zufallsstrom, Rauschen als Summe von zwölf Gleichverteilten in Ganzzahl-Arithmetik, keine Gleitkomma-Zufallszahlen); Aufwand wird in durchsuchten Kanten gezählt, nie in Sekunden. Die Kopien aus den Vorgängern sind bewacht (`tests/test_copies.py`).

**Der Schnitt ist exakt – ICM nicht.** Auf 40 kleinen 4 × 4-Rastern (Rauschen 80 %, λ = 0,5) erreicht der Schnitt mit **allen drei Flussverfahren** in **40 von 40** die kleinste Energie aller 65 536 Beschriftungen; **ICM** (Iterated Conditional Modes, Besag 1986) bleibt in **17 von 40** darüber. Auf den 16 × 16-Rastern liegt ICM in **allen 40** Ziehungen über der minimalen Energie, im Mittel 5,0 %.

**Glätten hilft – im richtigen Bereich.** Falsch beschriftete Zellen im Mittel über 40 Ziehungen (Rechteck): Schwelle je Zelle **26,2 %**, Mittelwertfilter 3 × 3 **9,2 %**, ICM **7,7 %**, **Graph Cut 3,8 %**; der Graph Cut ist in 38 von 40 Ziehungen besser als Schwelle und Mittelwertfilter. Beispielraster (Seed 7): 68 / 22 / 13 / 8 von 256 Zellen falsch. L-Form 26,4 / 7,9 / 6,5 / **2,1 %**, Scheibe 26,5 / 9,4 / 9,3 / 5,8 %, große Zone mit Inseln 26,6 / 13,3 / 11,3 / 9,4 %.

**Die richtige Glättung ist ein enges Fenster.** Fehler des Graph Cut gegen λ (Rechteck, 10 Ziehungen): λ = 0 **26,2 %** (das ist die Schwelle), 0,5: 8,7 %, **1,0: 4,7 %** (bestes λ), 1,5: 11,4 %, 2,0: 22,8 %, ab 2,5 **25 %** – das ganze Raster wird **eine Zone**, der Fehler ist genau die Fläche der kleineren Zone. Der Graph Cut schlägt den Mittelwertfilter (10,4 %) nur für λ von 0,5 bis 1,4. Die minimale Energie **steigt mit λ immer** (als Minimum wachsender Funktionen): λ lässt sich nicht über die Energie wählen.

**Das beste λ sinkt mit dem Rauschen.** Bestes λ (Orakel, das die Wahrheit kennt) bei Rauschen 30 / 50 / 70 / 90 / 110 / 130 %: **2,4 / 3,0 / 1,4 / 1,0 / 0,7 / 0,5** (bei 50 % liegt es am Rand des Rasters 0 bis 3); Fehler bei bestem λ 0,2 / 1,9 / 3,7 / 5,8 / 7,9 / 12,1 %. Ein **festes λ = 1** liefert 0,9 / 3,0 / 4,3 / 5,8 / **16,5 / 22,9 %** – bei 110 und 130 % Rauschen schlechter als der Mittelwertfilter (15,8 / 19,2 %); Preset „Starkes Rauschen“ (130 %, λ = 1): 64 falsche Zellen gegen 45 beim Mittelwertfilter und 43 bei ICM. Der Datenterm ist mit $1/\sigma^2$ gewichtet: laute Messungen haben ein schwaches Datengewicht, also braucht es weniger Glättung, nicht mehr.

**Dünne Zonen leiden.** Bei zwei Zellen breiten Streifen beschriftet der Graph Cut (λ = 1) im Mittel **33,5 %** der Zellen falsch, mehr als die Schwelle je Zelle (26,8 %) und die Filter (24,5 / 23,2 %) – in **0 von 40** Ziehungen ist er besser als beide Filter; seine Energie ist trotzdem die kleinste (Preset: 215,31 gegen 233,59 bei ICM). **Die minimale Energie ist nicht die beste Rekonstruktion.**

**Kleine echte Zonen verschwinden zuerst.** Quadratische Inseln der Kantenlänge 1 bis 6 in einer 40 × 20-Karte (Rauschen 60 %, 20 Ziehungen), wiedergefunden bei λ = 0: 18 / 19 / 19 / 20 / 20 / 20; **λ = 1: 1 / 4 / 11 / 20 / 20 / 20**; λ = 1,5: 0 / 0 / 3 / 15 / 17 / 18; **λ = 2: 0 / 0 / 0 / 5 / 9 / 12**. Im Preset „Kleine Inseln“ (24 × 24, λ = 1,5) sind die 30 falschen Zellen des Graph Cut (Schwelle 161, Mittelwertfilter 39, ICM 68 von 576) genau die vier Inseln (1 + 4 + 9 + 16 Zellen).

**Boykov–Kolmogorov durchsucht am wenigsten Kanten.** Durchsuchte Kanten im Hilfsnetz (Scheibe, Rauschen 80 %, λ = 1, Mittel über 5 Ziehungen), Boykov–Kolmogorov / Dinic / Edmonds-Karp: 8 × 8: 2 912 / 7 872 / 27 788; **16 × 16: 8 446 / 39 139 / 360 773** (Dinic **4,6-mal**, Edmonds-Karp **43-mal** mehr als BK); **48 × 48: 59 433 / 327 392 / 18 778 500** (5,5-mal bzw. 316-mal). Alle drei finden dieselbe minimale Energie.

## Was nicht funktioniert hat / Vorab-Hypothesen

Vor dem Bau standen fünf Vermutungen im Plan. Gemessen:

- **„Die Genauigkeit ist über einen breiten λ-Bereich flach“ – widerlegt.** Die Kurve hat ein enges Minimum (λ 0,5 bis 1,4 schlägt den Mittelwertfilter) und kippt danach schnell: ab λ = 2,5 ist das Raster eine einzige Zone.
- **„Der Mittelwertfilter ist ähnlich gut wie der Graph Cut“ – nur bei falschem λ.** Bei passender Glättung ist der Graph Cut mehr als doppelt so genau (3,8 gegen 9,2 %); bei dünnen Zonen und bei starkem Rauschen mit festem λ = 1 verliert er gegen den Filter.
- **„Das beste λ wächst mit dem Rauschen“ – widerlegt, es sinkt** (2,4 bei 30 % bis 0,5 bei 130 %), weil der Datenterm mit $1/\sigma^2$ gewichtet ist.
- **„ICM bleibt in einem Teil der Läufe über dem Optimum“ – bestätigt, und stärker als gedacht:** in allen 40 Ziehungen des Standardrasters (Lücke 5,0 %), in 17 von 40 auf 4 × 4-Rastern.
- **„Boykov–Kolmogorov braucht weniger durchsuchte Kanten“ – bestätigt** (4,6-mal weniger als Dinic bei 16 × 16). Einschränkung: Kanten zu zählen ist keine Laufzeitmessung; bewiesen ist keine bessere Schranke, im schlimmsten Fall ist BK sogar schlechter als Dinic.
- **Negativkontrolle „Unterschiede belohnen“:** nicht submodular (negative Kapazitäten); die naive Konstruktion (auf 0 gekappt) liegt in **38 von 40** 4 × 4-Rastern über dem Minimum; im Preset (Seed 3) −7,25 gegen das Minimum −17,99 (ICM findet es dort).
- **Abweichung vom Plan:** das Hilfsnetz wird nur für Raster bis 6 × 6 gezeichnet, ohne Schritt-für-Schritt-Regler (die Wege von Boykov–Kolmogorov sind Baumreparaturen, kein Niveaugraph); das Lehrraster (4 × 4, Seed 8: Graph Cut 0 von 16 falsch, Energie 16,25 = Minimum aller Beschriftungen, ICM 16,45) zeigt Netz und Schnitt.

## Was die Demo zeigt

- **Von der Messung zu den Zonen:** Wahrheit, Messung und vier Beschriftungen (Schwelle, Mittelwertfilter, ICM, Graph Cut) nebeneinander; falsch beschriftete Zellen rot überlegt, Fehler und Energie je Verfahren; für kleine Raster das Hilfsnetz mit Schnitt.
- **Einstellungen:** wahre Zonen, Rastergröße, Rauschen, Glättung λ, Nachbarstrafe (glätten / Unterschiede belohnen), Flussverfahren, Seed.
- **Nicht nur diese Messung:** Verteilung der Fehler über 40 feste Ziehungen, ICM-Lücke, Anzahl der Ziehungen, in denen der Graph Cut beide Filter schlägt.
- **Experimente (auf Abruf):** Glättungskurve, Rauschreihe (bestes λ), Inseln, Aufwand nach Rastergröße, Gegenprobe gegen alle Beschriftungen.
- **Wo die Annahmen enden:** zwei Zonen, Potts-Strafe, eine Glättung für alles, bekannte Zonenmittelwerte und Rauschen, feste Nachbarschaft.

## Modell und Verfahren

- **Energie:** $E(x)=\sum_i D_i(x_i)+\lambda\sum_{(i,j)}[x_i\ne x_j]$ mit $D_i(l)=(y_i-\mu_l)^2/(2\sigma^2)$ (negative Log-Likelihood einer Normalverteilung, Zonenmittelwerte 0 und 1000, in Hundertsteln gerundet); 4er-Nachbarschaft.
- **Hilfsnetz:** nach Abzug von $\min(D_i(0),D_i(1))$ je Zelle eine Terminalkante ($s\to i$ mit $D_i(1)-D_i(0)$ oder $i\to t$ mit $D_i(0)-D_i(1)$), je Nachbarpaar zwei Kanten mit Kapazität λ. Schnittkapazität = Energie minus Summe der kleineren Datenpreise.
- **Submodularität:** $E(0,0)+E(1,1)\le E(0,1)+E(1,0)$; Potts erfüllt sie, Belohnen von Unterschieden nicht.
- **Boykov–Kolmogorov:** Wachstum zweier Suchbäume aus aktiven Knoten, Augmentieren längs des gefundenen Weges (gesättigte Baumkanten machen ihr Kind zur Waise), Adoption mit Zeitstempel und Entfernungsangabe; der S-Baum ist am Ende die Quellseite eines minimalen Schnitts.
- **Vergleichsverfahren:** Schwelle je Zelle (Maximum-Likelihood), Mittelwertfilter 3 × 3 mit Schwelle in der Mitte der Zonenmittelwerte, ICM (Start Schwelle, feste Zellenreihenfolge).

## Ehrliche Grenzen

- **Zwei Zonen.** Mit mehr Klassen ist die exakte Lösung im Allgemeinen NP-schwer; α-Expansion (nur erwähnt) liefert gute, nicht beweisbar beste Beschriftungen.
- **Eine Glättung für alles.** Dieselbe Strafe glättet große und löscht kleine Zonen; die Wahrheit, mit der man λ wählen könnte, kennt man in der Praxis nicht.
- **Bekannte Zonenmittelwerte und Rauschen.** Der Datenterm setzt μ und σ voraus; sie zu schätzen ist ein eigenes Thema.
- **Synthetische Daten:** ein Raster mit erzeugten Zonen, kein Kundenbezug, kein echtes Bild.

## Bewusst nicht umgesetzt

- Mehrklassen-Beschriftung ($\alpha$-Expansion), nicht submodulare Energien (QPBO), Tiefen- und Bildbeispiele.
- Schätzung von λ, μ und σ (etwa EM oder Kreuzvalidierung).
- Gomory-Hu-Baum (drittes Stück der Erweiterung E1, gebaut: [gomory-hu-demo](https://github.com/sebastian-hanisch/gomory-hu-demo)).

## Dateien

```
app.py                  Oberfläche (Streamlit)
gc_scenario.py          Raster, Zonenformen, ganzzahliges Rauschen, Zufallsgenerator, Net (Kopie)
gc_cut.py               Energie, Hilfsnetz, Schnitt, Schwelle, Mittelwertfilter, ICM, Gegenprobe
gc_bk.py                Boykov-Kolmogorov (neu)
gc_dinic.py             Dinic (Kopie aus dinic-demo)
gc_edmonds_karp.py      Edmonds-Karp (Kopie aus edmonds-karp-demo)
gc_evaluation.py        Fehlklassifikation, Verteilungen, Kurven, Inseln, Aufwand
gc_visualization.py     Plotly-Abbildungen
gc_presets.py           Permalink, Presets, Zufalls-Seed
gc_constants.py         Konstanten, Regler-Grenzen, feste Seed-Mengen, Preset-Texte
tests/                  Kern, Boykov-Kolmogorov, Auswertung, Presets, Behauptungen, Kopien, App, Regler-Zustand
```

## Lokal starten

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt
venv/Scripts/streamlit run app.py
```

## Tests ausführen

```bash
venv/Scripts/pip install -r requirements-dev.txt
venv/Scripts/python -m pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und einem eigenen Flusskern (Boykov–Kolmogorov, Dinic, Edmonds-Karp).
