"""Gerichteter Graph in CSR-Form (Kompaktzeilen) mit ZWEI Kostenarten je Kante: Zeit (`weight`) und CO2 (`weight2`).
Nachbarn eines Knotens u stehen in indices[indptr[u]:indptr[u+1]]; Parallelkanten sind erlaubt (jede Kante hat eine eigene Nummer k = Position in indices)."""

import dataclasses
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Graph:
    n: int
    indptr: np.ndarray            # (n + 1,) int
    indices: np.ndarray           # (m,) int   Zielknoten je gerichteter Kante
    weight: np.ndarray            # (m,) float Zeit je Kante
    weight2: np.ndarray           # (m,) float CO2 je Kante
    xy: np.ndarray                # (n, 2) Lage für die Zeichnung
    names: tuple = ()             # optionale Beschriftungen
    directed: bool = False

    @property
    def m(self):
        return len(self.indices)

    def out(self, u):
        return self.indices[self.indptr[u]:self.indptr[u + 1]]

    def degree(self):
        return np.diff(self.indptr)

    def source_of_arcs(self):
        return np.repeat(np.arange(self.n), self.degree())

    def with_weights(self, w):
        """Derselbe Graph mit einer einzigen Kostenart w (für Dijkstra auf Zeit, CO2 oder einer gewichteten Summe)."""
        return dataclasses.replace(self, weight=np.asarray(w, dtype=float), weight2=np.zeros(self.m))

    def reversed(self):
        """Der Graph mit umgedrehten Kanten (für Schranken zum Ziel); Kante k des Originals ist dort Kante `perm[k]`, aber das braucht niemand: die Kosten wandern mit."""
        src = self.source_of_arcs()
        return from_arcs(self.n, [(int(v), int(u), float(w1), float(w2)) for u, v, w1, w2 in zip(src, self.indices, self.weight, self.weight2)], self.xy, self.names, True)


def from_arcs(n, arcs, xy, names=(), directed=False):
    """Baut den Graphen aus (u, v, Zeit, CO2)-Tupeln. Ungerichtet: jede Kante einmal angeben, die Rückrichtung entsteht hier. Selbstschleifen fallen weg, Parallelkanten bleiben."""
    rows = []
    for u, v, t, c in arcs:
        u, v, t, c = int(u), int(v), float(t), float(c)
        if u == v:
            continue
        rows.append((u, v, t, c))
        if not directed:
            rows.append((v, u, t, c))
    rows.sort()
    src = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    dst = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))
    w1 = np.fromiter((r[2] for r in rows), dtype=float, count=len(rows))
    w2 = np.fromiter((r[3] for r in rows), dtype=float, count=len(rows))
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return Graph(n, indptr, dst, w1, w2, np.asarray(xy, dtype=float), tuple(names), directed)


def route_cost(g, arcs):
    """Summe von Zeit und CO2 entlang einer Liste von Kantennummern."""
    return float(sum(g.weight[k] for k in arcs)), float(sum(g.weight2[k] for k in arcs))
