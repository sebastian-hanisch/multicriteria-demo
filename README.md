# Mehrkriterien-Routing – zwei Kosten, keine beste Route – Streamlit-Demo

Achtes und **letztes Stück der Kürzeste-Wege-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", dritter Ast von [Dijkstra](../dijkstra-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – das **Label-setting für die Pareto-Front** – an einem wachsenden Beispiel.
Dijkstra vergleicht Kosten mit "kleiner". Mit **zwei** Kosten – Fahrzeit und CO₂ – gibt es diese Ordnung nicht: die schnellste Route ist selten die sauberste. Die Antwort ist eine **Pareto-Front**, und an die Stelle von "kleiner" tritt die **Dominanz**:
eine Route dominiert eine andere, wenn sie in beiden Kosten höchstens so groß und in einer kleiner ist. Das Label-setting führt an jedem Knoten **alle nicht dominierten Labels** (Zeit, CO₂) mit. Die praktische Frage "die schnellste Route mit höchstens *B* Gramm CO₂" (Resource-Constrained Shortest Path) lässt sich aus der Front ablesen.

**Einordnung in die Reihe (die Kanten des Graphen):** die Linie ist damit komplett.
```
bfs-demo (Wurzel: Kanten zählen, nicht Kosten)                                       [gebaut]
  └─ dijkstra-demo (Kosten korrekt, blind in alle Richtungen)                        [gebaut]
       ├─ bidirectional-demo → contraction-hierarchies-demo                          [gebaut]
       ├─ bellman-ford-demo ─┐                                                       [gebaut]
       │   floyd-warshall-demo ─┴→ johnson-demo (Konvergenz: Umgewichtung)           [gebaut]
       └─ multicriteria-demo (Zeit gegen CO₂, Pareto)                                [dieses Stück]
```

## Quellen

| Bestandteil | Quelle |
|---|---|
| Verfahren (Label-setting, Dominanz, Pareto-Front) | Hansen (1980) und Martins (1984); die Bücher (*Grokking Algorithms*, *Optimization Algorithms*) behandeln Pareto nur bei evolutionären Verfahren, ein Routing-Beispiel gibt es nicht zu spiegeln |
| Umsetzung, Schranken zum Ziel, A*-Ordnung (nach NAMOA*, Ulloa et al. 2020), Budget, gewichtete Summe (dichotom), Konvexhülle, Bildfolge | eigen |
| Alle Netze | **eigene Graphen und Erzeuger**: kleines Netz mit vier Front-Routen, Stadtnetz mit schnellen, schmutzigen Hauptachsen, Zufallsnetz mit Korrelation, Worst-Case-Kette |
| Zahlen | **eigene Messungen** an diesen Netzen |

Aus den Büchern stammt keine Zahl, kein Graph und kein Text. **Kein OpenStreetMap-Auszug**, also keine ODbL-Pflichten.

## Ergebnis (Zahlen aus den Tests)

| Frage | Ergebnis |
|---|---|
| Kleines Netz (8 Orte) | ✅ **vier** Front-Routen (Zeit in Minuten / CO₂ in Gramm): Autobahn 10 / 90, Umgehung 15 / 75, Landstraße 20 / 50, Nebenweg 40 / 30; die schnellste hat 200 % mehr CO₂ als die sauberste, die sauberste braucht 300 % mehr Zeit. Die Umgehung liegt über der Konvexhülle: eine gewichtete Summe findet nur **3** Routen (5 Dijkstra-Läufe). Label-setting: 16 Labels, 12 festgelegt |
| Stadtnetz (10 × 10, Hauptachsen doppelt so schmutzig) | ✅ **29** Front-Routen zwischen 130 s / 605 g und 232 s / 311 g (+94.5 % CO₂ gegen +78.5 % Zeit); nur **9** Ecken der Hülle (17 Dijkstra-Läufe), 20 Punkte nie; 1 763 Labels |
| Front gegen Größe und Gegenläufigkeit (Stadtnetz, Mittel über 5 Netze) | ⚠️ die Front wächst mit beidem: bei 6 / 10 / 20 Kreuzungen je Seite und doppelt so schmutziger Hauptachse **11.8 / 34.0 / 91.8** Punkte (20 × 20: 28.6 Punkte, wenn die Hauptachse nur schneller ist); bei 10 × 10 und Zusatz-CO₂ 0 / 0.5 / 1 **11.4 / 28.2 / 34.0** |
| Front gegen Korrelation (Zufallsnetz, 200 Knoten) | ⚠️ bei Korrelation −1 / −0.5 / 0 / 0.5 / 1 **4.4 / 2.2 / 2.0 / 1.6 / 1.0** Punkte: laufen beide Kosten gleich, gibt es keinen Kompromiss |
| Was die gewichtete Summe verpasst | ❌ im Stadtnetz **69 %** (8 × 8), **80 %** (12 × 12), **83 %** (20 × 20: 15.2 von 91.8 Punkten, 29 Dijkstra-Läufe); im gegenläufigen Zufallsnetz 41 %; bei unabhängigen Kosten **0 %** |
| Worst-Case-Kette | ❌ die Front **verdoppelt sich mit jedem Glied**: *k* = 4 / 8 / 12 → **16 / 256 / 4 096** Punkte, 61 / 1 021 / 16 381 Labels; die gewichtete Summe findet nur die 2 Extreme (bei 8 Gliedern 2 von 256, 99.2 % verpasst) |
| CO₂-Budget (Stadtnetz 12 × 12, Mittel über 5 Netze) | ✅ bei 50 % der Spanne **21 %** weniger Labels (2 890 statt 3 669), bei 10 % nur 641 (die Front im Budget hat 8 von 47.6 Punkten); je knapper das Budget, desto langsamer die schnellste Route im Budget |
| Schranken zum Ziel und A\*-Ordnung | ⚠️ ändern die Front nie (in jedem Lauf geprüft). **Allein** sparen die Schranken im Stadtnetz nichts (Dijkstra-Ordnung erreicht das Ziel spät). In **A\*-Ordnung** (Warteschlange nach Kosten plus Schranke, NAMOA\*/BOA\*) sparen sie **51 / 48 / 39 / 36 %** der Labels im 8 × 8 / 12 × 12 / 16 × 16 / 20 × 20-Stadtnetz (die Ersparnis schrumpft, weil die Front von 21 auf 92 Punkte wächst) und **89 %** im gegenläufigen Zufallsnetz (Schranken allein dort 14 %). Die Zähler enthalten die zwei Dijkstra-Läufe für die Schranken nicht |
| Korrektheit | ✅ die Front stimmt auf jedem geprüften Netz mit einer **Brute-Force-Suche über alle Routen** und mit einem naiven Label-correcting ohne Ordnung überein; kein Front-Punkt dominiert einen anderen; die Ränder sind die beiden Einzelkriterium-Optima (Dijkstra); die gewichtete Summe findet **genau** die Ecken der unabhängig berechneten Konvexhülle; jede Route existiert im Netz mit den berichteten Kosten; die Kette hat exakt 2^*k* Punkte |

Die Zähler (Labels, Dijkstra-Läufe) sind Schritte des Verfahrens und plattformfest. Laufzeiten stehen in der App nur als Messwerte (reines Python) und werden nirgends behauptet oder getestet.

## Was die Demo zeigt

1. **Mehrkriterien-Routing in Aktion** (Regler über die Label-Entnahmen + Abspielen): die Karte mit den Knoten nach der Zahl festgelegter Labels, die Front am Ziel als **Streudiagramm** (Zeit gegen CO₂; schwarze Rauten = Ecken der Hülle, orange hohle Kreise = nicht unterstützt, Konvexhülle gepunktet, Budget als Linie), die drei Routen (schnellste, sauberste, gewählte). Beim kleinen Netz zusätzlich die **Tabelle der Labels je Ort** (✓ festgelegt, ✗ dominiert).
2. **Zwei Kosten – keine beste Route:** Front-Größe, Labels (erzeugt, festgelegt, dominiert), die Routen der gewichteten Summe, der Preis der schnellsten und der saubersten Route; je nach Anzeige die Front, die Ecken der Hülle oder die **schnellste Route mit CO₂-Budget** (aus der Front gelesen, mit dem Aufwand des Label-setting mit Budget verglichen).
3. **Vergleich** (Expander); **Experimente auf Knopfdruck**: Front gegen Größe, Gegenläufigkeit und Korrelation; was die gewichtete Summe verpasst; die Worst-Case-Kette; das Budget; Schranken zum Ziel und A*-Ordnung.
4. **Wo die Annahmen enden** (Tabelle; Bezug zu NSGA-II der Populations-Linie) und **Mathematische Formulierung** (Dominanz, Korrektheit über das Optimalitätsprinzip für Vektorkosten, Konvexhülle und unterstützte Punkte, Kette, RCSP, Aufwand als Lehrbuchwert gekennzeichnet).

Bedienung: Beispielnetz per Schnellstart-Knopf laden oder in der Seitenleiste Netz und Regler wählen; Anzeige (Front / gewichtete Summe / Budget), Frontpunkt und Budget erscheinen nur, wenn die Front mehr als einen Punkt hat (sonst steht dort, dass es nur einen Kompromiss gibt). Die Adresszeile spiegelt die Konfiguration (Permalink). Höchstens 400 Knoten.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `mc_graph.py`, `mc_queues.py` | Graph in CSR-Form mit zwei Kostenarten, Warteschlange |
| `mc_algorithm.py` | Label-setting (Dominanz, Schranken, A*-Ordnung, Budget), gewichtete Summe (dichotom), Konvexhülle, Abfrage der Front, naive Referenzen |
| `mc_scenario.py` | Netze: kleines Netz, Stadtnetz mit Hauptachsen, Zufallsnetz mit Korrelation, Kette |
| `mc_evaluation.py` | Kennzahlen, Bildfolge, Experimente |
| `mc_visualization.py`, `mc_presets.py`, `mc_constants.py` | Abbildungen, Presets und Permalink, Konstanten |

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/`. Jede Zahl in Hilfetexten, Presets und Tabellen ist in `tests/test_claims.py` belegt; die Kreuzprobe läuft gegen eine Brute-Force-Suche über alle Routen und ein naives Label-correcting (Netze mit Parallelkanten, Nullkosten, unerreichbaren Zielen und identischem Start und Ziel); ein Regressionstest klickt "▶️ Abspielen" auf Netzen mit mehreren Bildern.
