"""Mehrkriterien-Routing mit zwei Kosten (Zeit, CO2): Pareto-Front der Routen von s nach t.

Ein Label (Zeit, CO2) an einem Knoten ist ein Weg dorthin. Ein Label **dominiert** ein anderes, wenn es in beiden Kosten höchstens so groß und in mindestens einer kleiner ist; ein Wert wird auch verworfen, wenn ein
gleicher Wert schon da ist. Die **Pareto-Front** besteht aus allen nicht dominierten Labels am Ziel - jede von ihnen ist eine Route, die man nicht in beiden Kosten zugleich verbessern kann.

Label-setting (Hansen 1980, Martins 1984): Warteschlange lexikographisch nach (Zeit, CO2); ein entnommenes Label wird festgelegt, wenn kein früher festgelegtes Label desselben Knotens es dominiert.
Weil die Entnahme nach Zeit geordnet ist, haben alle früher festgelegten Labels eine höchstens gleiche Zeit - die Dominanzprüfung ist ein Vergleich mit dem kleinsten bisher festgelegten CO2 des Knotens.

Alles ist eigene Umsetzung auf dem CSR-Graphen aus mc_graph.py; networkx kommt nicht vor (die Referenzen für die Tests stehen unten und sind absichtlich naiv)."""

import bisect
import heapq
from dataclasses import dataclass, field

import numpy as np

from mc_graph import route_cost

INF = float("inf")
PRUNES = ("dominance", "bounds", "astar")


@dataclass
class ParetoResult:
    s: int
    target: object
    lab_node: list = field(default_factory=list)
    lab_t: list = field(default_factory=list)
    lab_c: list = field(default_factory=list)
    lab_parent: list = field(default_factory=list)          # Nummer des Vorgängerlabels (-1 = Start)
    lab_arc: list = field(default_factory=list)             # Nummer der Kante, mit der das Label erzeugt wurde
    settled: list = field(default_factory=list)             # Labelnummern in der Reihenfolge der Festlegung
    settled_by_node: dict = field(default_factory=dict)
    events: list = field(default_factory=list)              # je Entnahme (Labelnummer, "settled" | "dominated" | "bounds")
    counters: dict = field(default_factory=dict)

    def front(self, v=None):
        """Die nicht dominierten Labels am Knoten v (Standard: Ziel) als Liste (Zeit, CO2, Labelnummer), nach Zeit aufsteigend (CO2 fällt)."""
        v = self.target if v is None else v
        rows = sorted((self.lab_t[i], self.lab_c[i], i) for i in self.settled_by_node.get(v, []))
        return rows

    def arcs(self, lid):
        """Die Kanten der Route zu einem Label (vom Start aus)."""
        out = []
        while self.lab_parent[lid] >= 0:
            out.append(self.lab_arc[lid])
            lid = self.lab_parent[lid]
        return out[::-1]

    def nodes(self, lid):
        out = [self.lab_node[lid]]
        while self.lab_parent[lid] >= 0:
            lid = self.lab_parent[lid]
            out.append(self.lab_node[lid])
        return out[::-1]


def _dijkstra(g, s, w):
    """Einfaches Dijkstra mit einer Kostenart `w` (Feld je Kante): Entfernungen und die Vorgängerkante je Knoten; zählt Kantenprüfungen."""
    n = g.n
    ip, ix = g.indptr.tolist(), g.indices.tolist()
    wl = np.asarray(w, dtype=float).tolist()
    dist, parc = [INF] * n, [-1] * n
    dist[s] = 0.0
    heap = [(0.0, s)]
    done = [False] * n
    checks = 0
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for k in range(ip[u], ip[u + 1]):
            checks += 1
            v = ix[k]
            nd = d + wl[k]
            if nd < dist[v]:
                dist[v], parc[v] = nd, k
                heapq.heappush(heap, (nd, v))
    return np.array(dist), parc, checks


def target_bounds(g, target):
    """Kürzeste Zeit und kürzestes CO2 von jedem Knoten zum Ziel (Dijkstra auf dem umgedrehten Graphen): untere Schranken für jedes Label."""
    rg = g.reversed()
    return _dijkstra(rg, target, rg.weight)[0], _dijkstra(rg, target, rg.weight2)[0]


def pareto_labels(g, s, target=None, prune="dominance", budget=None):
    """Alle nicht dominierten Labels von s aus (bei gegebenem Ziel: nur die für das Ziel wichtigen, mit den Schranken aus `prune`).
    prune="dominance": nur die Dominanz der Labels desselben Knotens; "bounds": zusätzlich Schranken zum Ziel (ein Label fällt weg, wenn schon ein Ziel-Label sein bestmögliches Ende dominiert).
    prune="astar": wie "bounds", aber die Warteschlange ordnet nach (Zeit + Schranke, CO2 + Schranke) statt nach den bisherigen Kosten (NAMOA*/BOA*, Ulloa et al. 2020) - das Ziel wird früh erreicht, und die Schranken wirken.
    budget: höchstens so viel CO2 (Labels darüber fallen weg, mit Schranke zum Ziel, wenn ein Ziel gegeben ist)."""
    if prune not in PRUNES:
        raise ValueError(prune)
    n = g.n
    ip, ix = g.indptr.tolist(), g.indices.tolist()
    wt, wc = g.weight.tolist(), g.weight2.tolist()
    res = ParetoResult(int(s), None if target is None else int(target))
    res.counters = {"generated": 0, "settled": 0, "dominated": 0, "pruned_bounds": 0, "pruned_budget": 0}
    use_bounds = target is not None and (prune in ("bounds", "astar") or budget is not None)
    astar = prune == "astar" and target is not None
    lbt = lbc = None
    if use_bounds:
        lbt, lbc = target_bounds(g, int(target))
        lbt, lbc = lbt.tolist(), lbc.tolist()
    best_c = [INF] * n                         # kleinstes CO2 unter den festgelegten Labels des Knotens
    front_t, front_c = [], []                  # festgelegte Ziel-Labels (Zeit aufsteigend, CO2 fallend) für die Schranken-Prüfung

    def bounded(v, t, c):
        """Wahr, wenn das Label nicht mehr zu einer Front-Route werden kann (Ziel unerreichbar, Budget, oder ein Ziel-Label dominiert sein bestmögliches Ende)."""
        if lbt[v] == INF:
            return True
        if budget is not None and c + lbc[v] > budget:
            return "budget"
        if prune in ("bounds", "astar") and front_t:
            i = bisect.bisect_right(front_t, t + lbt[v]) - 1
            if i >= 0 and front_c[i] <= c + lbc[v]:
                return "bounds"
        return False

    def new_label(node, t, c, parent, arc):
        res.lab_node.append(node)
        res.lab_t.append(t)
        res.lab_c.append(c)
        res.lab_parent.append(parent)
        res.lab_arc.append(arc)
        return len(res.lab_node) - 1

    start = new_label(int(s), 0.0, 0.0, -1, -1)

    def key(v, t, c, lid):
        """Ordnung der Warteschlange: nach den bisherigen Kosten, bei A* nach den Kosten plus der Schranke zum Ziel (die Schranken sind konsistent, also bleibt die Dominanzprüfung je Knoten gültig)."""
        return (t + lbt[v], c + lbc[v], lid) if astar else (t, c, lid)

    heap = [key(int(s), 0.0, 0.0, start)]
    res.counters["generated"] = 1
    while heap:
        lid = heapq.heappop(heap)[2]
        v, t, c = res.lab_node[lid], res.lab_t[lid], res.lab_c[lid]
        if best_c[v] <= c:
            res.counters["dominated"] += 1
            res.events.append((lid, "dominated"))
            continue
        if use_bounds:
            why = bounded(v, t, c)
            if why:
                res.counters["pruned_budget" if why == "budget" else "pruned_bounds"] += 1
                res.events.append((lid, "bounds"))
                continue
        best_c[v] = c
        res.settled.append(lid)
        res.settled_by_node.setdefault(v, []).append(lid)
        res.counters["settled"] += 1
        res.events.append((lid, "settled"))
        if target is not None and v == target:
            front_t.append(t)
            front_c.append(c)
            continue                            # das Ziel weiter zu verlassen führt nie zu einer besseren Route zu ihm
        for k in range(ip[v], ip[v + 1]):
            w = ix[k]
            nt, nc = t + wt[k], c + wc[k]
            if best_c[w] <= nc:                 # dominiert von einem festgelegten Label des Kopfknotens (dessen Zeit ist höchstens nt)
                res.counters["dominated"] += 1
                continue
            if use_bounds:
                why = bounded(w, nt, nc)
                if why:
                    res.counters["pruned_budget" if why == "budget" else "pruned_bounds"] += 1
                    continue
            nid = new_label(w, nt, nc, lid, k)
            res.counters["generated"] += 1
            heapq.heappush(heap, key(w, nt, nc, nid))
    return res


# --- Gewichtete Summe und Konvexhülle -------------------------------------------------------------------------------------------------------------

def weighted_sum(g, s, t):
    """Die Punkte, die man mit einer gewichteten Summe a · Zeit + b · CO2 findet (dichotome Suche: zwei Extreme, dann jeweils die Normale der Verbindungsstrecke als Gewichte). Das sind die Ecken der Konvexhülle der Front.
    Ausgabe: sortierte Liste (Zeit, CO2) und die Zahl der Dijkstra-Läufe."""
    runs = 0
    big = float(g.weight2.sum() + 1.0)                              # lexikographisch: erst Zeit, dann CO2 (ganzzahlige Kosten)
    bigt = float(g.weight.sum() + 1.0)

    def solve(a, b, lex):
        nonlocal runs
        runs += 1
        w = a * g.weight + b * g.weight2 if not lex else (big * g.weight + g.weight2 if lex == "time" else bigt * g.weight2 + g.weight)
        dist, parc, _ = _dijkstra(g, s, w)
        if not np.isfinite(dist[t]):
            return None
        arcs, v = [], t
        while v != s:
            k = parc[v]
            arcs.append(k)
            v = int(np.searchsorted(g.indptr, k, side="right") - 1)
        return route_cost(g, arcs)

    first, last = solve(0, 0, "time"), solve(0, 0, "co2")
    if first is None:
        return [], runs
    pts = {first, last}

    def recurse(p, q):
        """p links (kleinere Zeit), q rechts (kleineres CO2): gibt es eine Route strikt links unter der Verbindungsstrecke?"""
        if p == q:
            return
        a, b = p[1] - q[1], q[0] - p[0]                             # Normale der Strecke pq (beide positiv)
        r = solve(a, b, None)
        if r is not None and a * r[0] + b * r[1] < a * p[0] + b * p[1]:
            pts.add(r)
            recurse(p, r)
            recurse(r, q)
    recurse(first, last)
    return sorted(pts), runs


def hull(front):
    """Die Ecken der unteren linken Konvexhülle einer Front [(Zeit, CO2), ...] (unabhängig von der dichotomen Suche berechnet); liegt ein Punkt auf einer Strecke zwischen zwei Ecken, ist er keine Ecke."""
    pts = sorted(set((float(t), float(c)) for t, c in front))
    out = []
    for p in pts:
        while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0:
            out.pop()
        out.append(p)
    return out


def rcsp(front, budget):
    """Die schnellste Route mit CO2 <= budget aus der Front [(Zeit, CO2, ...), ...]; None, wenn keine Route ins Budget passt."""
    feas = [row for row in front if row[1] <= budget]
    return min(feas, key=lambda r: (r[0], r[1])) if feas else None


# --- Referenzen für die Tests (absichtlich naiv) ---------------------------------------------------------------------------------------------------

def brute_force_front(g, s, t, limit=200000):
    """Alle einfachen Routen von s nach t durchprobieren und die nicht dominierten Kosten behalten (nur für kleine Netze)."""
    ip, ix = g.indptr.tolist(), g.indices.tolist()
    seen, costs, count = {s}, set(), [0]

    def dfs(v, tt, cc):
        count[0] += 1
        assert count[0] < limit, "Netz zu groß für die Brute-Force-Referenz"
        if v == t:
            costs.add((tt, cc))
            return
        for k in range(ip[v], ip[v + 1]):
            w = ix[k]
            if w not in seen:
                seen.add(w)
                dfs(w, tt + g.weight[k], cc + g.weight2[k])
                seen.discard(w)
    dfs(s, 0.0, 0.0)
    return nondominated(costs)


def nondominated(points):
    """Die nicht dominierten Punkte einer Menge von (Zeit, CO2), sortiert nach Zeit."""
    out, best = [], INF
    for t, c in sorted(set(points)):
        if c < best:
            out.append((float(t), float(c)))
            best = c
    return out


def label_correcting_front(g, s, t):
    """Naives Label-correcting ohne Ordnung: FIFO-Schlange, je Knoten eine Menge nicht dominierter Labels, neue Labels verdrängen alte (unabhängig vom Label-setting)."""
    from collections import deque
    ip, ix = g.indptr.tolist(), g.indices.tolist()
    sets = {v: set() for v in range(g.n)}
    sets[s].add((0.0, 0.0))
    q = deque([(s, (0.0, 0.0))])
    while q:
        v, (tt, cc) = q.popleft()
        if (tt, cc) not in sets[v]:
            continue                                                                   # inzwischen verdrängt
        for k in range(ip[v], ip[v + 1]):
            w = ix[k]
            lab = (tt + g.weight[k], cc + g.weight2[k])
            if any(a <= lab[0] and b <= lab[1] for a, b in sets[w]):
                continue
            sets[w] = {(a, b) for a, b in sets[w] if not (lab[0] <= a and lab[1] <= b)} | {lab}
            q.append((w, lab))
    return nondominated(sets[t])
