"""Netze (kleines Netz, Stadtnetz mit Hauptachsen, Zufallsnetz mit Korrelation, Kette), Kennzahlen, Bildfolge, Experimente."""

import networkx as nx
import numpy as np
import pytest

import mc_algorithm as alg
import mc_constants as C
import mc_evaluation as ev
import mc_scenario as sc
import mc_visualization as viz


def _connected(g):
    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    G.add_edges_from((int(u), int(v)) for u, v in zip(g.source_of_arcs(), g.indices))
    return nx.is_connected(G)


# --- Netze -----------------------------------------------------------------------------------------------------------------------------------

def test_small_network_has_the_four_front_routes_and_exactly_one_unsupported_point():
    net = sc.small_network()
    g = net.graph
    assert g.n == 8 and g.m == 2 * len(sc.SMALL_LEGS) == 28 and not g.directed and (g.names[net.source], g.names[net.target]) == ("Start", "Ziel")
    front = alg.brute_force_front(g, net.source, net.target)
    assert front == [(10.0, 90.0), (15.0, 75.0), (20.0, 50.0), (40.0, 30.0)]
    res = alg.pareto_labels(g, net.source, net.target)
    assert [[g.names[i] for i in res.nodes(lid)] for _, _, lid in res.front()] == [["Start", "Autobahn", "Ziel"], ["Start", "Umgehung", "Ziel"], ["Start", "Landstraße", "Ziel"], ["Start", "Dorf", "Nebenweg", "Ziel"]]
    assert alg.hull(front) == [(10.0, 90.0), (20.0, 50.0), (40.0, 30.0)] and [p for p in front if p not in alg.hull(front)] == [(15.0, 75.0)]


@pytest.mark.parametrize("seed", range(3))
def test_city_network_is_connected_grid_with_whole_number_costs(seed):
    net = sc.city_network(8, 1.0, seed)
    g = net.graph
    assert g.n == 64 and g.m == 2 * 2 * 8 * 7 and _connected(g) and (g.weight >= 1).all() and (g.weight2 >= 1).all()
    assert np.array_equal(g.weight, np.rint(g.weight)) and np.array_equal(g.weight2, np.rint(g.weight2)) and len(net.arterial) == g.m
    assert (net.source, net.target) == (0, 63)


def test_arterials_are_faster_and_dirtier_per_kilometre_and_the_conflict_controls_the_dirt():
    def per_km(conflict):
        net = sc.city_network(16, conflict, 3)
        g = net.graph
        src = g.source_of_arcs()
        km = np.hypot(*(g.xy[src] - g.xy[g.indices]).T) / 1000.0
        art = np.array(net.arterial)
        return (g.weight[art] / km[art]).mean(), (g.weight[~art] / km[~art]).mean(), (g.weight2[art] / km[art]).mean(), (g.weight2[~art] / km[~art]).mean()
    t_a, t_l, c_a1, c_l1 = per_km(1.0)
    _, _, c_a0, c_l0 = per_km(0.0)
    assert t_a < 0.6 * t_l                                                                        # doppelte Geschwindigkeit: halbe Zeit je km
    assert c_a1 > 1.8 * c_l1 and c_a0 == pytest.approx(c_l0, rel=0.15)                            # Faktor 2 bei Gegenläufigkeit 1, gleich bei 0
    assert sum(sc.city_network(8, 0.5, 2).arterial) > 0


def test_random_network_is_connected_and_the_correlation_shows_in_the_costs():
    r = {}
    for corr in (-1.0, 0.0, 1.0):
        g = sc.build_random(300, 3.0, corr, 4)
        assert _connected(g) and abs(g.m / g.n - 3.0) < 0.1
        r[corr] = float(np.corrcoef(g.weight, g.weight2)[0, 1])
    assert r[-1.0] < -0.95 and abs(r[0.0]) < 0.25 and r[1.0] > 0.999
    net = sc.random_network(100, 3.0, -1.0, 7)
    assert net.source == 0 and net.target != 0 and not net.geometric


@pytest.mark.parametrize("k", (1, 3, 6))
def test_chain_network_shape_and_front(k):
    net = sc.chain_network(k)
    assert net.graph.n == 3 * k + 1 and net.graph.m == 4 * k and (net.source, net.target) == (0, 3 * k)
    res = alg.pareto_labels(net.graph, net.source, net.target)
    assert len(res.front()) == 2 ** k


def test_make_network_rejects_unknown_nets():
    with pytest.raises(ValueError):
        sc.make_network("ring")


# --- Kennzahlen -------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", C.NETS)
def test_analysis_invariants_for_every_net(key):
    net = sc.make_network(key, side=6, nodes=40, links=5)
    a = ev.analyse(net)
    m = a.metrics
    assert m["front"] == len(a.front) >= 1 and m["ws_matches_hull"] and m["hull"] == len(a.hull) and m["unsupported"] == m["front"] - sum(1 for t, c, _ in a.front if (t, c) in set(a.hull))
    assert m["settled"] <= m["generated"] and m["generated_bounds"] <= m["generated"] and m["generated_astar"] <= m["generated_bounds"] and m["events"] == len(a.res.events) and m["ws_points"] == m["hull"]
    ts = [t for t, _, _ in a.front]
    assert ts == sorted(ts) and (m["fast_t"], m["clean_c"]) == (a.front[0][0], a.front[-1][1])
    assert ev.verdict(a) == ("single" if m["front"] == 1 else "front")
    if m["front"] > 1 and m["clean_c"] > 0 and m["fast_t"] > 0:
        assert m["price_fast"] == pytest.approx(m["fast_c"] / m["clean_c"] - 1) and m["price_clean"] == pytest.approx(m["clean_t"] / m["fast_t"] - 1)


def test_the_front_ends_are_the_single_criterion_optima():
    net = sc.make_network("city", side=8)
    a = ev.analyse(net)
    g = net.graph
    assert a.front[0][0] == alg._dijkstra(g, net.source, g.weight)[0][net.target] and a.front[-1][1] == alg._dijkstra(g, net.source, g.weight2)[0][net.target]


def test_budget_for_maps_percent_to_the_span_between_clean_and_fast():
    a = ev.analyse(sc.small_network())
    assert ev.budget_for(a, 0) == 30 and ev.budget_for(a, 100) == 90 and ev.budget_for(a, 50) == 60
    assert alg.rcsp(a.front, ev.budget_for(a, 50))[:2] == (20, 50)


def test_budget_run_keeps_only_routes_in_the_budget_and_generates_fewer_labels():
    net = sc.make_network("city", side=8)
    a = ev.analyse(net)
    b = ev.budget_for(a, 25)
    capped = ev.budget_run(net, b)
    assert [(t, c) for t, c, _ in capped.front()] == [(t, c) for t, c, _ in a.front if c <= b] and capped.counters["generated"] < a.metrics["generated"]


# --- Bildfolge ---------------------------------------------------------------------------------------------------------------------------------

def test_frames_and_state_follow_the_events():
    a = ev.analyse(sc.small_network())
    f = ev.frames(a)
    assert f[0] == 0 and f[-1] == len(a.res.events) == 16 and f == list(range(17))
    count0, popped0, tl0, last0 = ev.state_at(a, 0)
    assert count0.sum() == 0 and popped0 == {} and tl0 == [] and last0 is None
    count, popped, tl, last = ev.state_at(a, f[-1])
    assert count.sum() == a.metrics["settled"] and tl == [(t, c) for t, c, _ in a.front] and sum(1 for v in popped.values() for _, _, s in v if s == "dominated") == a.metrics["dominated_events"]
    big = ev.analyse(sc.make_network("city", side=12))
    fb = ev.frames(big)
    assert fb[0] == 0 and fb[-1] == len(big.res.events) and len(fb) <= 61 and fb == sorted(set(fb))


def test_table_state_marks_settled_and_dominated_labels():
    a = ev.analyse(sc.small_network())
    rows = viz.table_state(a, len(a.res.events))
    text = " ".join(r["Labels (Zeit, CO₂)"] for r in rows)
    assert [r["Ort"] for r in rows][0] == "Start" and "✓" in text and "✗" in text and text.count("✓") == a.metrics["settled"]
    assert viz.table_state(a, 0)[0]["Labels (Zeit, CO₂)"] == "–"


def test_charts_render_for_every_net_and_mode_and_frame():
    for key in C.NETS:
        a = ev.analyse(sc.make_network(key, side=6, nodes=40, links=5))
        f = ev.frames(a)
        chosen = (a.front[0][0], a.front[0][1])
        for k in (f[0], f[len(f) // 2], f[-1]):
            routes = [(a.front[0][2], "x", "#000", "solid")] if k == f[-1] else []
            viz.build_network(a, k, routes)
            for mode in C.MODES:
                viz.build_front(a, k, mode, chosen, 40.0)


# --- Experimente --------------------------------------------------------------------------------------------------------------------------------

def test_front_vs_size_and_corr_rows():
    rows = ev.front_vs_size(sides=(6, 10), conflicts=(0.0, 1.0), seeds=C.SWEEP_SEEDS[:2])
    assert [(r["conflict"], r["side"]) for r in rows] == [(0.0, 6), (0.0, 10), (1.0, 6), (1.0, 10)]
    assert rows[1]["front"] > rows[0]["front"] and rows[3]["front"] > rows[2]["front"] and all(r["hull"] <= r["front"] for r in rows)
    crows = ev.front_vs_corr(corrs=(-1.0, 1.0), n=80, seeds=C.SWEEP_SEEDS[:2])
    assert crows[0]["front"] > crows[1]["front"] == 1.0 and crows[0]["r"] < -0.9 and crows[1]["r"] > 0.99


def test_missed_by_weighted_sum_rows():
    rows = ev.missed_by_weighted_sum(seeds=C.SWEEP_SEEDS[:2])
    assert [r["label"] for r in rows][-1] == "Kette, 8 Glieder" and rows[-1]["front"] == 256 and rows[-1]["found"] == 2
    assert all(r["found"] <= r["front"] and 0 <= r["missed_share"] <= 1 for r in rows) and rows[2]["missed_share"] > rows[3]["missed_share"] > rows[4]["missed_share"] == 0


def test_chain_rows_double_with_every_link():
    rows = ev.chain_rows(ks=(2, 3, 4))
    assert [r["front"] for r in rows] == [4, 8, 16] and rows[2]["generated"] > rows[1]["generated"] > rows[0]["generated"]


def test_budget_curve_rows_and_bounds_effect():
    rows = ev.budget_curve(side=8, percents=(0, 50, 100), seeds=C.SWEEP_SEEDS[:2])
    assert rows[0]["generated"] < rows[1]["generated"] < rows[2]["generated"] <= rows[2]["full"] and rows[0]["front_in_budget"] == 1 and rows[2]["front_in_budget"] == rows[2]["front"]
    assert rows[0]["time"] > rows[2]["time"]


def test_astar_rows_have_one_row_per_net_and_the_astar_order_never_generates_more_than_the_plain_order():
    rows = ev.astar_rows(sides=(6, 8), seeds=C.SWEEP_SEEDS[:2], random_nodes=60)
    assert [r["label"] for r in rows] == ["Stadtnetz 6 × 6", "Stadtnetz 8 × 8", "Zufallsnetz 60 Knoten, gegenläufig"]
    assert all(r["astar"] <= r["bounds"] <= r["dominance"] for r in rows) and rows[1]["astar"] < 0.7 * rows[1]["dominance"]
    assert len(ev.astar_rows(sides=(6,), seeds=C.SWEEP_SEEDS[:1], random_nodes=0)) == 1
