"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

SPACING = 100.0                    # Meter zwischen benachbarten Kreuzungen im erzeugten Stadtnetz
JITTER = 0.25                      # Lageabweichung der Kreuzungen in Blocklängen
AXIS_EVERY = 4                     # jede vierte Zeile und Spalte ist Hauptachse
SPEED_LOCAL, SPEED_ARTERIAL = 30.0, 60.0            # km/h
GRAMS_PER_KM = 150.0               # CO2 einer Nebenstraße je km (Hauptachse: das (1 + Gegenläufigkeit)-Fache)
NOISE = 0.3                        # zufällige Streuung beider Kosten je Kante (Ampeln, Steigung, Belag)

NETS = ("small", "city", "random", "chain")
NET_LABELS = {
    "small": "🔀 Kleines Netz (vier Front-Routen)",
    "city": "🏙️ Stadtnetz mit Hauptachsen (erzeugt)",
    "random": "🕸️ Zufallsnetz mit Korrelation (erzeugt)",
    "chain": "⛓️ Worst-Case-Kette (2^k Front-Routen)",
}
SMALL_NETS = ("small",)                                    # eigener Graph mit Namen: Tabelle der Labels, feste Aufgabe
FIXED_NETS = ("small",)

SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 4, 20, 10               # n = Seite², höchstens 400 Knoten
CONFLICT_MIN, CONFLICT_MAX, DEFAULT_CONFLICT = 0.0, 1.0, 1.0   # Gegenläufigkeit: um wie viel schmutziger die schnelle Hauptachse ist
NODES_MIN, NODES_MAX, DEFAULT_NODES = 20, 200, 100
DEGREE_MIN, DEGREE_MAX, DEFAULT_DEGREE = 2.0, 6.0, 3.0
CORR_MIN, CORR_MAX, DEFAULT_CORR = -1.0, 1.0, -1.0         # Korrelation der Kosten je Kante (negativ = gegenläufig)
LINKS_MIN, LINKS_MAX, DEFAULT_LINKS = 1, 12, 6             # Glieder der Kette (2^k Front-Routen)
DEFAULT_SEED = 7
DEFAULT_NET = "small"

MODES = ("front", "weighted", "budget")
DEFAULT_MODE = "front"
MODE_LABELS = {"front": "Pareto-Front (alle Kompromisse)", "weighted": "Gewichtete Summe (Punkte der Konvexhülle)", "budget": "Schnellste Route mit CO₂-Budget"}

SWEEP_SEEDS = tuple(range(100000, 100005))

COLORS = {"fast": "#d62728", "clean": "#2ca02c", "chosen": "#1f77b4", "hull": "#111111", "unsupported": "#ff7f0e", "budget": "#9467bd", "start": "#111111", "goal": "#ff7f0e", "dominated": "#bbbbbb"}

_BASE = dict(side=DEFAULT_SIDE, conflict=DEFAULT_CONFLICT, nodes=DEFAULT_NODES, degree=DEFAULT_DEGREE, corr=DEFAULT_CORR, links=DEFAULT_LINKS, mode=DEFAULT_MODE, seed=DEFAULT_SEED)
PRESETS = {
    "🔀 Kleines Netz": {**_BASE, "net": "small"},
    "🏙️ Stadtnetz": {**_BASE, "net": "city"},
    "🕸️ Zufallsnetz": {**_BASE, "net": "random", "seed": 17},
    "⛓️ Worst-Case-Kette": {**_BASE, "net": "chain"},
}
PRESET_HELP = {
    "🔀 Kleines Netz": "Kleines Netz mit vier nicht dominierten Routen (Zeit in Minuten, CO₂ in Gramm): Autobahn 10 / 90, Umgehung 15 / 75, Landstraße 20 / 50, Nebenweg 40 / 30. Die schnellste Route hat 200 % mehr CO₂ als die sauberste, die sauberste braucht 300 % mehr Zeit. Die Umgehung liegt über der Konvexhülle: eine gewichtete Summe findet sie nie (5 Dijkstra-Läufe, 3 Routen). Label-setting erzeugt 16 Labels, 12 werden festgelegt.",
    "🏙️ Stadtnetz": "Stadtnetz (10 × 10, Hauptachsen doppelt so schmutzig): 29 Front-Routen zwischen 130 s / 605 g (schnellste) und 232 s / 311 g (sauberste) - die schnellste hat 94.5 % mehr CO₂, die sauberste braucht 78.5 % mehr Zeit. Nur 9 der 29 Punkte sind Ecken der Hülle: eine gewichtete Summe findet sie mit 17 Dijkstra-Läufen, die anderen 20 nie. Label-setting erzeugt 1 763 Labels.",
    "🕸️ Zufallsnetz": "Zufallsnetz (100 Knoten, Grad 3, gegenläufige Kosten, Seed 17): 8 Front-Routen, nur 3 davon findet die gewichtete Summe (5 Dijkstra-Läufe). Die schnellste Route (21 / 49) hat 188.2 % mehr CO₂ als die sauberste (43 / 17), die sauberste braucht 104.8 % mehr Zeit.",
    "⛓️ Worst-Case-Kette": "Kette mit 6 Gliedern: 64 = 2⁶ Front-Routen, bei jeder ist Zeit + CO₂ = 63, keine dominiert eine andere. Label-setting erzeugt 253 Labels, die gewichtete Summe findet nur die 2 Extreme (3 Dijkstra-Läufe).",
}
