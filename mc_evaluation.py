"""Pareto-Front, gewichtete Summe und CO2-Budget: Kennzahlen für ein Netz, Bildfolge der Label-Festlegung, Experimente (Front gegen Größe, Gegenläufigkeit und Korrelation; was die gewichtete Summe verpasst;
die Worst-Case-Kette; das Budget), Urteil für die App.

Der Aufwand wird in Zählern gemessen (erzeugte, festgelegte, verdrängte Labels; Dijkstra-Läufe; plattformfest); Laufzeiten stehen nur als Messwerte in der App."""

import time
from dataclasses import dataclass

import numpy as np

import mc_algorithm as alg
import mc_constants as C
from mc_scenario import build_city, build_random, chain_network, make_network


@dataclass(frozen=True)
class Analysis:
    net: object
    res: alg.ParetoResult
    front: list                    # [(Zeit, CO2, Labelnummer)] nach Zeit aufsteigend
    hull: list                     # Ecken der Konvexhülle [(Zeit, CO2)]
    ws_runs: int                   # Dijkstra-Läufe der gewichteten Summe
    metrics: dict
    seconds: dict


def analyse(net):
    g = net.graph
    t0 = time.perf_counter()
    res = alg.pareto_labels(g, net.source, net.target)
    t_pl = time.perf_counter() - t0
    t0 = time.perf_counter()
    pts, runs = alg.weighted_sum(g, net.source, net.target)
    t_ws = time.perf_counter() - t0
    bounded = alg.pareto_labels(g, net.source, net.target, prune="bounds")
    front = res.front()
    hull = alg.hull([(t, c) for t, c, _ in front])
    hull_set = set(hull)
    fast, clean = front[0], front[-1]
    m = {"n": g.n, "m": g.m // (1 if g.directed else 2), "front": len(front), "hull": len(hull), "unsupported": sum(1 for t, c, _ in front if (t, c) not in hull_set),
         "generated": res.counters["generated"], "settled": res.counters["settled"], "dominated": res.counters["dominated"], "ws_runs": runs, "ws_points": len(pts), "ws_matches_hull": pts == hull,
         "generated_bounds": bounded.counters["generated"], "pruned_bounds": bounded.counters["pruned_bounds"],
         "fast_t": fast[0], "fast_c": fast[1], "clean_t": clean[0], "clean_c": clean[1],
         "price_fast": fast[1] / clean[1] - 1.0 if clean[1] > 0 else float("nan"), "price_clean": clean[0] / fast[0] - 1.0 if fast[0] > 0 else float("nan"),
         "events": len(res.events), "dominated_events": sum(1 for _, k in res.events if k == "dominated")} if front else {"front": 0}
    return Analysis(net, res, front, hull, runs, m, {"labels": t_pl, "ws": t_ws})


def verdict(a):
    """Code für die App: unreachable / single (nur ein Kompromiss) / front."""
    m = a.metrics
    if m["front"] == 0:
        return "unreachable"
    return "single" if m["front"] == 1 else "front"


def budget_for(a, percent):
    """Das CO2-Budget bei `percent` Prozent der Spanne zwischen der saubersten und der schnellsten Route."""
    m = a.metrics
    return m["clean_c"] + (m["fast_c"] - m["clean_c"]) * percent / 100.0


def budget_run(net, budget):
    """Label-setting mit Budget: Front (nur die im Budget) und Zähler."""
    return alg.pareto_labels(net.graph, net.source, net.target, budget=budget)


def frames(a, max_frames=60):
    """Die Bildfolge: Zahl der Entnahmen aus der Warteschlange (0 bis alle); bei vielen Entnahmen höchstens `max_frames` Zwischenstände."""
    total = len(a.res.events)
    return sorted({int(round(x)) for x in np.linspace(0, total, min(total + 1, max_frames + 1))})


def state_at(a, k):
    """Zustand nach den ersten k Entnahmen: Zahl der festgelegten Labels je Knoten, je Knoten die Liste (Zeit, CO2, Status) der entnommenen Labels, die Labels des Ziels bis dahin und das zuletzt entnommene Label."""
    res = a.res
    count = np.zeros(a.net.graph.n, dtype=int)
    popped = {}
    target_labels = []
    last = None
    for lid, status in res.events[:k]:
        v = res.lab_node[lid]
        last = lid
        popped.setdefault(v, []).append((res.lab_t[lid], res.lab_c[lid], status))
        if status == "settled":
            count[v] += 1
            if v == res.target:
                target_labels.append((res.lab_t[lid], res.lab_c[lid]))
    return count, popped, target_labels, last


# --- Experimente -------------------------------------------------------------------------------------------------------------------------------

def _mean(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def _city_row(side, conflict, seed):
    net = make_network("city", side=side, conflict=conflict, seed=seed)
    res = alg.pareto_labels(net.graph, net.source, net.target)
    front = [(t, c) for t, c, _ in res.front()]
    return {"n": net.graph.n, "front": len(front), "hull": len(alg.hull(front)), "generated": res.counters["generated"], "settled": res.counters["settled"]}


def front_vs_size(sides=(6, 8, 10, 12, 16, 20), conflicts=(0.0, 0.5, 1.0), seeds=C.SWEEP_SEEDS):
    """Stadtnetz: Größe der Front am Ziel (Start unten links, Ziel oben rechts) gegen die Netzgröße für drei Gegenläufigkeiten, dazu Ecken der Hülle und erzeugte Labels (Mittel über die Sweep-Netze)."""
    return [{"side": side, "conflict": conflict, **_mean([_city_row(side, conflict, sd) for sd in seeds])} for conflict in conflicts for side in sides]


def front_vs_corr(corrs=(-1.0, -0.5, 0.0, 0.5, 1.0), n=200, degree=3.0, seeds=C.SWEEP_SEEDS):
    """Zufallsnetz: Größe der Front gegen die Korrelation der beiden Kosten; dazu die tatsächlich gemessene Korrelation der Kantenkosten."""
    rows = []
    for corr in corrs:
        acc = []
        for sd in seeds:
            net = make_network("random", nodes=n, degree=degree, corr=corr, seed=sd)
            g = net.graph
            res = alg.pareto_labels(g, net.source, net.target)
            front = [(t, c) for t, c, _ in res.front()]
            r = float(np.corrcoef(g.weight, g.weight2)[0, 1]) if g.weight.std() > 0 and g.weight2.std() > 0 else 1.0
            acc.append({"front": len(front), "hull": len(alg.hull(front)), "r": r})
        rows.append({"corr": corr, **_mean(acc)})
    return rows


def missed_by_weighted_sum(seeds=C.SWEEP_SEEDS):
    """Was die gewichtete Summe verpasst: je Netz die Größe der Front, die Zahl ihrer Punkte, die eine gewichtete Summe finden kann (Ecken der Konvexhülle), und die Dijkstra-Läufe der dichotomen Suche (Mittel über die Sweep-Netze)."""
    specs = (("Stadtnetz 8 × 8", dict(net="city", side=8, conflict=1.0)), ("Stadtnetz 12 × 12", dict(net="city", side=12, conflict=1.0)), ("Stadtnetz 20 × 20", dict(net="city", side=20, conflict=1.0)),
             ("Zufallsnetz, gegenläufig", dict(net="random", nodes=200, degree=3.0, corr=-1.0)), ("Zufallsnetz, unabhängig", dict(net="random", nodes=200, degree=3.0, corr=0.0)))
    rows = []
    for label, kw in specs:
        acc = []
        for sd in seeds:
            net = make_network(**kw, seed=sd)
            res = alg.pareto_labels(net.graph, net.source, net.target)
            front = [(t, c) for t, c, _ in res.front()]
            pts, runs = alg.weighted_sum(net.graph, net.source, net.target)
            assert pts == alg.hull(front)
            acc.append({"front": len(front), "found": len(pts), "runs": runs})
        m = _mean(acc)
        rows.append({"label": label, **m, "missed_share": 1.0 - m["found"] / m["front"]})
    chain = make_network("chain", links=8)
    res = alg.pareto_labels(chain.graph, chain.source, chain.target)
    pts, runs = alg.weighted_sum(chain.graph, chain.source, chain.target)
    rows.append({"label": "Kette, 8 Glieder", "front": float(len(res.front())), "found": float(len(pts)), "runs": float(runs), "missed_share": 1.0 - len(pts) / len(res.front())})
    return rows


def chain_rows(ks=(2, 4, 6, 8, 10, 12)):
    """Worst-Case-Kette: Front und Labels gegen die Zahl der Glieder (die Front verdoppelt sich mit jedem Glied)."""
    rows = []
    for k in ks:
        net = chain_network(k)
        t0 = time.perf_counter()
        res = alg.pareto_labels(net.graph, net.source, net.target)
        rows.append({"k": k, "front": len(res.front()), "generated": res.counters["generated"], "settled": res.counters["settled"], "seconds": time.perf_counter() - t0})
    return rows


def budget_curve(side=12, conflict=1.0, percents=(0, 10, 25, 50, 75, 100), seeds=C.SWEEP_SEEDS):
    """Das Budget: erzeugte Labels und Zahl der Front-Routen im Budget gegen das Budget (in % der Spanne zwischen sauberster und schnellster Route), Stadtnetz (Mittel über die Sweep-Netze)."""
    rows = []
    for p in percents:
        acc = []
        for sd in seeds:
            net = make_network("city", side=side, conflict=conflict, seed=sd)
            full = alg.pareto_labels(net.graph, net.source, net.target)
            front = full.front()
            b = front[-1][1] + (front[0][1] - front[-1][1]) * p / 100.0
            capped = budget_run(net, b)
            best = alg.rcsp(front, b)
            assert [(t, c) for t, c, _ in capped.front()] == [(t, c) for t, c, _ in front if c <= b]
            acc.append({"generated": capped.counters["generated"], "full": full.counters["generated"], "front_in_budget": len(capped.front()), "front": len(front), "time": best[0]})
        rows.append({"percent": p, **_mean(acc)})
    return rows


def bounds_effect(side=20, conflict=1.0, seeds=C.SWEEP_SEEDS):
    """Schranken zum Ziel (Vorgriff auf A*): erzeugte Labels mit und ohne (Mittel über die Sweep-Netze)."""
    plain, fast = [], []
    for sd in seeds:
        net = make_network("city", side=side, conflict=conflict, seed=sd)
        plain.append(alg.pareto_labels(net.graph, net.source, net.target).counters["generated"])
        fast.append(alg.pareto_labels(net.graph, net.source, net.target, prune="bounds").counters["generated"])
    return {"plain": float(np.mean(plain)), "bounds": float(np.mean(fast))}
