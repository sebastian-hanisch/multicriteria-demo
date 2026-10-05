"""Unabhängiges Orakel: Aufzählung aller einfachen Routen per Tiefensuche, Pareto-Menge per Definition (kein anderer Punkt ist komponentenweise kleiner oder gleich), Konvexhülle mit
exakten Brüchen (ein Punkt auf oder über einer Strecke zwischen zwei anderen ist keine Ecke). Auch mit Nullkosten, Parallelkanten, ungerichteten Netzen, Budget und allen Knoten."""

import itertools
from fractions import Fraction

import numpy as np
import pytest

import mc_algorithm as alg
from mc_graph import from_arcs


def _routes(g, s, t):
    ip, ix = g.indptr.tolist(), g.indices.tolist()
    out = []

    def dfs(v, seen, tt, cc):
        if v == t:
            out.append((float(tt), float(cc)))
            return
        for k in range(ip[v], ip[v + 1]):
            if ix[k] not in seen:
                dfs(ix[k], seen | {ix[k]}, tt + g.weight[k], cc + g.weight2[k])

    dfs(s, {s}, 0.0, 0.0)
    return out


def _pareto(points):
    pts = set(points)
    return sorted(p for p in pts if not any(q != p and q[0] <= p[0] and q[1] <= p[1] for q in pts))


def _corners(front):
    out = []
    for p in front:
        on_or_above = False
        for q, r in itertools.combinations([x for x in front if x != p], 2):
            lo, hi = sorted((q, r))
            if lo[0] < hi[0] and lo[0] <= p[0] <= hi[0]:
                lam = Fraction(p[0] - lo[0]) / Fraction(hi[0] - lo[0])
                if lo[1] + lam * (hi[1] - lo[1]) <= p[1]:
                    on_or_above = True
                    break
        if not on_or_above:
            out.append(p)
    return out


def _random_graph(rng, n, m, lo, hi, directed):
    arcs = []
    for _ in range(m):
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u != v:
            arcs.append((u, v, float(rng.integers(lo, hi)), float(rng.integers(lo, hi))))
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=directed)


@pytest.mark.parametrize("lo,hi,directed", [(0, 3, True), (1, 6, True), (1, 10, False), (0, 4, False)])
def test_front_budget_hull_and_all_node_fronts_equal_enumeration(lo, hi, directed):
    rng = np.random.default_rng(100 * lo + hi + directed)
    for _ in range(40):
        n = int(rng.integers(2, 8))
        g = _random_graph(rng, n, int(rng.integers(1, 18)), lo, hi, directed)
        s, t = (int(x) for x in rng.integers(0, n, 2))
        ref = _pareto(_routes(g, s, t))
        for prune in alg.PRUNES:
            assert [(a, b) for a, b, _ in alg.pareto_labels(g, s, t, prune=prune).front()] == ref
        if not ref:
            assert alg.weighted_sum(g, s, t)[0] == []
            continue
        budget = ref[len(ref) // 2][1]
        for prune in alg.PRUNES:
            got = [(a, b) for a, b, _ in alg.pareto_labels(g, s, t, prune=prune, budget=budget).front()]
            assert got == [p for p in ref if p[1] <= budget]
        assert alg.rcsp([(a, c, 0) for a, c in ref], budget)[:2] == min(p for p in ref if p[1] <= budget)
        assert alg.weighted_sum(g, s, t)[0] == alg.hull(ref) == _corners(ref)
        res = alg.pareto_labels(g, s)
        for v in range(n):
            assert [(a, b) for a, b, _ in res.front(v)] == _pareto(_routes(g, s, v))
