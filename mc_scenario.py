"""Die Netze der Demo: kleines Netz mit vier Front-Routen (eine davon nicht unterstützt), Stadtnetz mit schnellen, schmutzigen Hauptachsen, Zufallsnetz mit einstellbarer Korrelation der beiden Kosten und die
Worst-Case-Kette mit 2^k Front-Routen. Alle Graphen sind eigene Konstruktionen, alle Kosten ganze Zahlen (Zeit in Sekunden, CO2 in Gramm)."""

import math
from dataclasses import dataclass

import numpy as np

import mc_constants as C
from mc_graph import Graph, from_arcs


@dataclass(frozen=True)
class Network:
    key: str
    graph: Graph
    title: str
    note: str
    source: int
    target: int
    geometric: bool = True         # Lage der Knoten ist eine Karte (Kanten zeichnen), sonst Knoten zufällig verteilt
    arterial: tuple = ()           # Stadtnetz: je Kante (Nummer im CSR) ob Hauptachse


# --- Kleines Netz -----------------------------------------------------------------------------------------------------------------------------

SMALL_STOPS = [("Start", 0.0, 3.0), ("Autobahn", 3.0, 6.0), ("Umgehung", 3.0, 4.2), ("Landstraße", 3.0, 2.4), ("Dorf", 3.0, 0.6), ("Nebenweg", 5.5, -0.6), ("Stadt", 5.5, 2.4), ("Ziel", 8.0, 3.0)]
# (von, nach, Zeit in Minuten, CO2 in Gramm) - ungerichtet
SMALL_LEGS = [("Start", "Autobahn", 4, 40), ("Autobahn", "Ziel", 6, 50), ("Start", "Umgehung", 7, 35), ("Umgehung", "Ziel", 8, 40), ("Start", "Landstraße", 9, 20), ("Landstraße", "Ziel", 11, 30),
              ("Start", "Dorf", 12, 10), ("Dorf", "Nebenweg", 14, 10), ("Nebenweg", "Ziel", 14, 10), ("Autobahn", "Umgehung", 3, 20), ("Landstraße", "Stadt", 5, 15), ("Stadt", "Ziel", 12, 40),
              ("Dorf", "Stadt", 6, 10), ("Landstraße", "Dorf", 4, 10)]


def small_network():
    names = [s[0] for s in SMALL_STOPS]
    idx = {n: i for i, n in enumerate(names)}
    g = from_arcs(len(names), [(idx[a], idx[b], t, c) for a, b, t, c in SMALL_LEGS], [(s[1], s[2]) for s in SMALL_STOPS], names, directed=False)
    return Network("small", g, "Kleines Netz", "", idx["Start"], idx["Ziel"], True)


# --- Stadtnetz mit Hauptachsen -------------------------------------------------------------------------------------------------------------------

def build_city(side, conflict, seed):
    """Raster mit gestörten Kreuzungen (nur Nachbarn im Raster). Jede vierte Zeile und Spalte ist Hauptachse: doppelt so schnell (60 statt 30 km/h) und je km um den Faktor (1 + conflict) schmutziger.
    Zeit in Sekunden und CO2 in Gramm, beide mit unabhängiger Streuung (Ampeln, Steigung, Belag); Start unten links, Ziel oben rechts."""
    rng = np.random.default_rng([int(seed), 404])
    n = side * side
    ij = np.stack(np.divmod(np.arange(n), side), axis=1)                      # (Zeile, Spalte)
    xy = np.stack([ij[:, 1], ij[:, 0]], axis=1) * C.SPACING + rng.uniform(-C.JITTER, C.JITTER, (n, 2)) * C.SPACING
    arcs, arterial = [], []
    for i in range(side):
        for j in range(side):
            for di, dj in ((0, 1), (1, 0)):
                i2, j2 = i + di, j + dj
                if i2 >= side or j2 >= side:
                    continue
                u, v = i * side + j, i2 * side + j2
                km = float(np.hypot(*(xy[u] - xy[v]))) / 1000.0
                art = (i % C.AXIS_EVERY == 0) if di == 0 else (j % C.AXIS_EVERY == 0)
                speed = C.SPEED_ARTERIAL if art else C.SPEED_LOCAL
                g_km = C.GRAMS_PER_KM * ((1.0 + conflict) if art else 1.0)
                t = max(1, round(km / speed * 3600.0 * (1.0 + C.NOISE * rng.random())))
                c = max(1, round(km * g_km * (1.0 + C.NOISE * rng.random())))
                arcs.append((u, v, t, c))
                arterial.append(art)
    g = from_arcs(n, arcs, xy, directed=False)
    # Kennzeichnung der Hauptachsen je Kante des CSR: über (u, v) nachschlagen
    key = {(min(u, v), max(u, v)): a for (u, v, _, _), a in zip(arcs, arterial)}
    src = g.source_of_arcs()
    flags = tuple(bool(key[(min(int(u), int(v)), max(int(u), int(v)))]) for u, v in zip(src, g.indices))
    return g, flags


def city_network(side, conflict, seed):
    g, flags = build_city(side, conflict, seed)
    note = ("Erzeugtes Stadtnetz: jede vierte Zeile und Spalte ist eine Hauptachse (orange), doppelt so schnell wie die Nebenstraßen, aber je Kilometer um den Faktor "
            f"{1 + conflict:g} schmutziger. Zeit in Sekunden, CO₂ in Gramm; Start unten links, Ziel oben rechts.")
    return Network("city", g, "Stadtnetz mit Hauptachsen", note, 0, g.n - 1, True, flags)


# --- Zufallsnetz mit Korrelation ----------------------------------------------------------------------------------------------------------------

def build_random(n, degree, corr, seed):
    """Zusammenhängender ungerichteter Zufallsgraph (jeder Knoten hängt an einem früheren, dann kommen zufällige Kanten bis zum mittleren Grad). Zeit und CO2 je Kante von 1 bis 9; `corr` mischt die
    beiden Zufallszahlen: +1 = CO2 gleich der Zeit, -1 = CO2 spiegelt die Zeit (10 - Zeit), 0 = unabhängig."""
    rng = np.random.default_rng([int(seed), 707])
    pairs = {(int(rng.integers(0, i)), i) for i in range(1, n)}
    target_m = int(round(n * degree / 2))
    while len(pairs) < target_m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u != v and (min(u, v), max(u, v)) not in pairs:
            pairs.add((min(u, v), max(u, v)))
    arcs = []
    w = abs(corr)
    for u, v in sorted(pairs):
        u1, u2 = rng.random(2)
        mixed = w * (u1 if corr >= 0 else 1.0 - u1) + (1.0 - w) * u2
        arcs.append((u, v, 1 + int(round(8 * u1)), 1 + int(round(8 * mixed))))
    xy = rng.random((n, 2)) * 1000.0
    return from_arcs(n, arcs, xy, directed=False)


def _farthest_by_hops(g, s=0):
    dist = [-1] * g.n
    dist[s] = 0
    q = [s]
    for u in q:
        for v in g.out(u):
            if dist[int(v)] < 0:
                dist[int(v)] = dist[u] + 1
                q.append(int(v))
    return int(max(range(g.n), key=lambda v: (dist[v], -v)))


def random_network(n, degree, corr, seed):
    g = build_random(int(n), float(degree), float(corr), int(seed))
    return Network("random", g, "Zufallsnetz mit Korrelation", "Erzeugter Zufallsgraph, Zeit und CO₂ je Kante von 1 bis 9; die Korrelation legt fest, wie eng beide zusammenhängen (negativ: was schnell ist, ist schmutzig). Keine Karte, nur Punkte; "
                   "das Ziel ist der vom Start am weitesten entfernte Knoten (nach Kantenzahl).", 0, _farthest_by_hops(g), False)


# --- Worst-Case-Kette ------------------------------------------------------------------------------------------------------------------------------

def chain_network(k):
    """k Glieder in Reihe, jedes mit zwei parallelen Wegen (über je einen eigenen Zwischenknoten): (2^i Sekunden, 0 g) gegen (0 s, 2^i g). Jede der 2^k Kombinationen hat Zeit + CO2 = 2^k - 1: keine dominiert eine andere."""
    arcs, xy, node = [], [(0.0, 0.0)], 0
    for i in range(k):
        u, a, b, v = node, node + 1, node + 2, node + 3
        arcs += [(u, a, 2 ** i, 0), (a, v, 0, 0), (u, b, 0, 2 ** i), (b, v, 0, 0)]
        xy += [(3 * i + 1.5, 1.0), (3 * i + 1.5, -1.0), (3 * i + 3.0, 0.0)]
        node += 3
    g = from_arcs(node + 1, arcs, xy, directed=True)
    return Network("chain", g, "Worst-Case-Kette", f"Kette aus {k} Gliedern mit je zwei Wegen: (2^i s, 0 g) gegen (0 s, 2^i g). Jede der 2^{k} Kombinationen liegt auf der Geraden Zeit + CO₂ = {2 ** k - 1}: keine dominiert eine andere.", 0, node, True)


# --- Zusammenbau -------------------------------------------------------------------------------------------------------------------------------

def make_network(net, side=C.DEFAULT_SIDE, conflict=C.DEFAULT_CONFLICT, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, corr=C.DEFAULT_CORR, links=C.DEFAULT_LINKS, seed=C.DEFAULT_SEED):
    if net == "small":
        return small_network()
    if net == "city":
        return city_network(int(side), float(conflict), int(seed))
    if net == "random":
        return random_network(int(nodes), float(degree), float(corr), int(seed))
    if net == "chain":
        return chain_network(int(links))
    raise ValueError(net)
