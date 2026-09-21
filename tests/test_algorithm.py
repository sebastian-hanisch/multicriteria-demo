"""Pareto-Front, Label-setting, Schranken, Budget, gewichtete Summe und Konvexhülle gegen naive Referenzen (Brute-Force über alle Routen, Label-correcting ohne Ordnung)."""

import numpy as np
import pytest

import mc_algorithm as alg
from mc_graph import from_arcs, route_cost

INF = float("inf")


def _random_graph(n, m, seed, lo=1, hi=10, directed=True):
    rng = np.random.default_rng(seed)
    arcs = []
    while len(arcs) < m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u != v:
            arcs.append((u, v, float(rng.integers(lo, hi)), float(rng.integers(lo, hi))))
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=directed)


def _chain(k):
    """k Glieder, je zwei parallele Wege (über einen eigenen Zwischenknoten): (2^i, 0) gegen (0, 2^i). Alle 2^k Kombinationen liegen auf der Geraden Zeit + CO2 = 2^k - 1, keine dominiert eine andere."""
    arcs, xy = [], []
    node = 0
    for i in range(k):
        u, a, b, v = node, node + 1, node + 2, node + 3
        arcs += [(u, a, 2 ** i, 0.0), (a, v, 0.0, 0.0), (u, b, 0.0, 2 ** i), (b, v, 0.0, 0.0)]
        node += 3
    return from_arcs(node + 1, arcs, np.zeros((node + 1, 2)), directed=True), node


def _front(res):
    return [(t, c) for t, c, _ in res.front()]


CASES = [(n, m, seed) for n, m in ((6, 12), (8, 18), (9, 22), (9, 14)) for seed in range(6)]


@pytest.mark.parametrize("n,m,seed", CASES)
def test_front_equals_brute_force_and_label_correcting(n, m, seed):
    g = _random_graph(n, m, seed)
    for s, t in ((0, n - 1), (1, n - 2), (n - 1, 0)):
        ref = alg.brute_force_front(g, s, t)
        assert _front(alg.pareto_labels(g, s, t)) == ref and alg.label_correcting_front(g, s, t) == ref


@pytest.mark.parametrize("n,m,seed", CASES)
def test_front_is_free_of_dominated_points_and_every_route_is_real(n, m, seed):
    g = _random_graph(n, m, seed)
    res = alg.pareto_labels(g, 0, n - 1)
    rows = res.front()
    ts, cs = [r[0] for r in rows], [r[1] for r in rows]
    assert ts == sorted(ts) and len(set(ts)) == len(ts) and all(cs[i] > cs[i + 1] for i in range(len(cs) - 1))          # Zeit steigt, CO2 fällt: keiner dominiert einen anderen
    for t, c, lid in rows:
        arcs = res.arcs(lid)
        nodes = res.nodes(lid)
        assert route_cost(g, arcs) == (t, c) and nodes[0] == 0 and nodes[-1] == n - 1 and len(set(nodes)) == len(nodes)
        assert all(g.indices[k] == nodes[i + 1] for i, k in enumerate(arcs))


@pytest.mark.parametrize("seed", range(6))
def test_undirected_nets_and_all_nodes_fronts(seed):
    g = _random_graph(8, 12, seed, directed=False)
    res = alg.pareto_labels(g, 0)
    for v in range(1, 8):
        assert [(t, c) for t, c, _ in res.front(v)] == alg.brute_force_front(g, 0, v)


@pytest.mark.parametrize("seed", range(6))
def test_front_ends_are_the_two_single_criterion_optima(seed):
    g = _random_graph(12, 30, seed)
    res = alg.pareto_labels(g, 0, 11)
    rows = res.front()
    dt = alg._dijkstra(g, 0, g.weight)[0][11]
    dc = alg._dijkstra(g, 0, g.weight2)[0][11]
    if not rows:
        assert dt == INF and dc == INF                                                                   # unerreichbar: beide Verfahren sagen es
        return
    assert rows[0][0] == dt and rows[-1][1] == dc


@pytest.mark.parametrize("n,m,seed", CASES)
def test_bounds_and_budget_do_not_change_the_answer_and_never_generate_more(n, m, seed):
    g = _random_graph(n, m, seed)
    plain = alg.pareto_labels(g, 0, n - 1)
    fast = alg.pareto_labels(g, 0, n - 1, prune="bounds")
    assert _front(fast) == _front(plain) and fast.counters["generated"] <= plain.counters["generated"]
    front = _front(plain)
    if len(front) > 1:
        b = front[len(front) // 2][1]
        capped = alg.pareto_labels(g, 0, n - 1, budget=b)
        assert _front(capped) == [p for p in front if p[1] <= b] and capped.counters["generated"] <= plain.counters["generated"]
        both = alg.pareto_labels(g, 0, n - 1, prune="bounds", budget=b)
        assert _front(both) == _front(capped)


def test_counters_and_events_are_consistent():
    g = _random_graph(9, 22, 3)
    res = alg.pareto_labels(g, 0, 8, prune="bounds")
    c = res.counters
    assert c["settled"] == len(res.settled) == sum(1 for _, k in res.events if k == "settled")
    assert c["generated"] == len(res.lab_node) and len(res.events) <= c["generated"]
    assert sum(1 for _, k in res.events if k == "dominated") <= c["dominated"]


@pytest.mark.parametrize("n,m,seed", CASES)
def test_weighted_sum_finds_exactly_the_hull_vertices(n, m, seed):
    g = _random_graph(n, m, seed)
    front = alg.brute_force_front(g, 0, n - 1)
    pts, runs = alg.weighted_sum(g, 0, n - 1)
    assert pts == alg.hull(front) and runs >= 2 and set(pts) <= set(front)


def test_chain_has_exactly_two_to_the_k_front_points_and_the_weighted_sum_finds_two():
    for k in (1, 3, 6, 9):
        g, last = _chain(k)
        res = alg.pareto_labels(g, 0, last)
        front = _front(res)
        assert len(front) == 2 ** k and all(t + c == 2 ** k - 1 for t, c in front)
        pts, _ = alg.weighted_sum(g, 0, last)
        assert len(pts) == 2 and alg.hull(front) == pts
    g, last = _chain(4)
    assert alg.brute_force_front(g, 0, last) == _front(alg.pareto_labels(g, 0, last))


def test_hull_drops_points_on_a_segment_and_above_it():
    front = [(1, 9), (2, 7), (3, 5), (5, 3), (9, 1)]                     # (2,7) liegt über der Strecke (1,9)-(3,5)?  Strecke bei t=2: 7 -> darauf, also keine Ecke
    assert alg.hull(front) == [(1.0, 9.0), (3.0, 5.0), (5.0, 3.0), (9.0, 1.0)]
    above = [(1, 9), (2, 8), (5, 3), (9, 1)]                             # (2,8): Strecke (1,9)-(5,3) bei t=2 ist 7.5 < 8: über der Hülle
    assert alg.hull(above) == [(1.0, 9.0), (5.0, 3.0), (9.0, 1.0)]


def test_rcsp_picks_the_fastest_route_within_the_budget():
    front = [(10, 90, 0), (15, 75, 1), (20, 50, 2), (40, 30, 3)]
    assert alg.rcsp(front, 100)[:2] == (10, 90) and alg.rcsp(front, 75)[:2] == (15, 75) and alg.rcsp(front, 50)[:2] == (20, 50) and alg.rcsp(front, 30)[:2] == (40, 30)
    assert alg.rcsp(front, 29) is None and alg.rcsp([], 10) is None


def test_edge_cases_same_node_unreachable_zero_costs_parallel_arcs():
    g = from_arcs(4, [(0, 1, 1.0, 5.0), (0, 1, 3.0, 1.0), (1, 2, 0.0, 0.0), (2, 1, 0.0, 0.0)], np.zeros((4, 2)), directed=True)
    assert _front(alg.pareto_labels(g, 0, 0)) == [(0.0, 0.0)]
    assert _front(alg.pareto_labels(g, 0, 3)) == [] and alg.weighted_sum(g, 0, 3)[0] == []
    assert _front(alg.pareto_labels(g, 0, 2)) == [(1.0, 5.0), (3.0, 1.0)]                    # zwei Parallelkanten, Nullkanten und ein Nullzyklus stören nicht
    one = from_arcs(3, [(0, 1, 2.0, 0.0), (1, 2, 3.0, 0.0)], np.zeros((3, 2)), directed=True)
    assert _front(alg.pareto_labels(one, 0, 2)) == [(5.0, 0.0)]                             # ein Kostenvektor mit einer Null-Kostenart: die Front hat einen Punkt


def test_unknown_prune_is_rejected():
    with pytest.raises(ValueError):
        alg.pareto_labels(_random_graph(4, 6, 1), 0, 3, prune="astar")
