"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-21, Toleranzen fangen Rundung ab). Labels, Fronten, Dijkstra-Läufe und ganzzahlige Kosten sind plattformfest;
Laufzeiten stehen in der App nur als Messwerte und werden hier nie geprüft."""

import numpy as np
import pytest

import mc_algorithm as alg
import mc_constants as C
import mc_evaluation as ev
import mc_scenario as sc

PRESET = {"small": "🔀 Kleines Netz", "city": "🏙️ Stadtnetz", "random": "🕸️ Zufallsnetz", "chain": "⛓️ Worst-Case-Kette"}


def _preset(key):
    p = C.PRESETS[PRESET[key]]
    net = sc.make_network(p["net"], p["side"], p["conflict"], p["nodes"], p["degree"], p["corr"], p["links"], p["seed"])
    return p, net, ev.analyse(net)


def _has(key, *needles):
    help_ = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in help_, (key, n)


# --- Preset-Hilfen -----------------------------------------------------------------------------------------------------------------------------

def test_small_preset_numbers():
    _, net, a = _preset("small")
    m = a.metrics
    assert [(t, c) for t, c, _ in a.front] == [(10.0, 90.0), (15.0, 75.0), (20.0, 50.0), (40.0, 30.0)] and (m["hull"], m["unsupported"], m["ws_runs"], m["ws_points"]) == (3, 1, 5, 3)
    assert (m["generated"], m["settled"]) == (16, 12) and m["price_fast"] == pytest.approx(2.0) and m["price_clean"] == pytest.approx(3.0)
    _has("small", "10 / 90", "15 / 75", "20 / 50", "40 / 30", "200 % mehr CO₂", "300 % mehr Zeit", "5 Dijkstra-Läufe, 3 Routen", "16 Labels, 12")


def test_city_preset_numbers():
    _, net, a = _preset("city")
    m = a.metrics
    assert (m["n"], m["front"], m["hull"], m["unsupported"], m["ws_runs"], m["generated"]) == (100, 29, 9, 20, 17, 1763)
    assert (m["fast_t"], m["fast_c"], m["clean_t"], m["clean_c"]) == (130.0, 605.0, 232.0, 311.0) and m["price_fast"] == pytest.approx(0.945, abs=0.0005) and m["price_clean"] == pytest.approx(0.785, abs=0.0005)
    _has("city", "10 × 10", "29 Front-Routen", "130 s / 605 g", "232 s / 311 g", "94.5 %", "78.5 %", "9 der 29", "17 Dijkstra-Läufen", "20 nie", "1 763 Labels")


def test_random_preset_numbers():
    p, net, a = _preset("random")
    m = a.metrics
    assert p["seed"] == 17 and (m["n"], m["front"], m["hull"], m["unsupported"], m["ws_runs"]) == (100, 8, 3, 5, 5)
    assert (m["fast_t"], m["fast_c"], m["clean_t"], m["clean_c"]) == (21.0, 49.0, 43.0, 17.0) and m["price_fast"] == pytest.approx(1.882, abs=0.001) and m["price_clean"] == pytest.approx(1.048, abs=0.001)
    _has("random", "Seed 17", "8 Front-Routen", "nur 3", "5 Dijkstra-Läufe", "(21 / 49)", "188.2 %", "(43 / 17)", "104.8 %")


def test_chain_preset_numbers():
    _, net, a = _preset("chain")
    m = a.metrics
    assert (m["front"], m["hull"], m["ws_runs"], m["generated"]) == (64, 2, 3, 253) and all(t + c == 63 for t, c, _ in a.front)
    _has("chain", "6 Gliedern", "64 = 2⁶", "Zeit + CO₂ = 63", "253 Labels", "2 Extreme", "3 Dijkstra-Läufe")


# --- Sidebar-Hilfen ----------------------------------------------------------------------------------------------------------------------------

def test_size_and_conflict_help_numbers():
    rows = {(r["conflict"], r["side"]): r for r in ev.front_vs_size()}
    assert [rows[(1.0, s)]["front"] for s in (6, 10, 20)] == pytest.approx([11.8, 34.0, 91.8], abs=0.05)                       # Hilfe zur Seitenlänge
    assert [rows[(c, 10)]["front"] for c in (0.0, 0.5, 1.0)] == pytest.approx([11.4, 28.2, 34.0], abs=0.05)                     # Hilfe zur Gegenläufigkeit
    assert all(rows[(1.0, s)]["front"] > rows[(0.0, s)]["front"] for s in (6, 8, 10, 12, 16, 20))


def test_correlation_help_numbers():
    rows = {r["corr"]: r for r in ev.front_vs_corr()}
    assert [rows[c]["front"] for c in (-1.0, -0.5, 0.0, 0.5, 1.0)] == pytest.approx([4.4, 2.2, 2.0, 1.6, 1.0], abs=0.05)
    assert rows[-1.0]["r"] == pytest.approx(-1.0, abs=0.01) and rows[1.0]["r"] == pytest.approx(1.0)


def test_chain_help_numbers():
    rows = {r["k"]: r for r in ev.chain_rows(ks=(4, 8, 12))}
    assert [(rows[k]["front"], rows[k]["generated"]) for k in (4, 8, 12)] == [(16, 61), (256, 1021), (4096, 16381)]
    assert all(rows[k]["front"] == 2 ** k for k in rows) and rows[12]["settled"] == 16381


# --- Experimente und die Tabelle "Wo die Annahmen enden" ----------------------------------------------------------------------------------------

def test_front_grows_with_size_and_conflict():
    rows = {(r["conflict"], r["side"]): r for r in ev.front_vs_size()}
    assert rows[(0.0, 20)]["front"] == pytest.approx(28.6, abs=0.05) and rows[(1.0, 20)]["front"] == pytest.approx(91.8, abs=0.05) and rows[(1.0, 20)]["hull"] == pytest.approx(15.2, abs=0.05)   # Tabelle: 92 Punkte, 15 Ecken
    sizes = [rows[(1.0, s)]["front"] for s in (6, 8, 10, 12, 16, 20)]
    assert sizes == sorted(sizes) and rows[(1.0, 20)]["generated"] > 20 * rows[(1.0, 6)]["generated"] / 20


def test_weighted_sum_misses_most_of_the_front_in_city_nets_and_nothing_in_independent_random_nets():
    rows = {r["label"]: r for r in ev.missed_by_weighted_sum()}
    assert [rows[k]["missed_share"] for k in ("Stadtnetz 8 × 8", "Stadtnetz 12 × 12", "Stadtnetz 20 × 20")] == pytest.approx([0.692, 0.803, 0.834], abs=0.002)
    assert rows["Stadtnetz 20 × 20"]["runs"] == pytest.approx(29.4, abs=0.05) and rows["Stadtnetz 20 × 20"]["found"] == pytest.approx(15.2, abs=0.05)
    assert rows["Kette, 8 Glieder"]["missed_share"] == pytest.approx(0.992, abs=0.001) and (rows["Kette, 8 Glieder"]["front"], rows["Kette, 8 Glieder"]["found"]) == (256, 2)          # Tabelle: 2 von 256
    assert rows["Zufallsnetz, unabhängig"]["missed_share"] == 0.0 and rows["Zufallsnetz, gegenläufig"]["missed_share"] == pytest.approx(0.409, abs=0.002)


def test_chain_of_twelve_links_table_numbers():
    row = ev.chain_rows(ks=(12,))[0]
    assert (row["front"], row["generated"]) == (4096, 16381)                                       # Tabelle: 4 096 Punkte, 16 381 Labels


def test_budget_claims():
    rows, eff = ev.budget_curve(), ev.bounds_effect()
    by = {r["percent"]: r for r in rows}
    assert by[50]["generated"] == pytest.approx(2890.0, abs=0.5) and by[50]["full"] == pytest.approx(3668.6, abs=0.5) and 1 - by[50]["generated"] / by[50]["full"] == pytest.approx(0.21, abs=0.005)
    assert by[10]["generated"] == pytest.approx(640.6, abs=0.5) and by[10]["front_in_budget"] == pytest.approx(8.0, abs=0.05) and by[10]["front"] == pytest.approx(47.6, abs=0.05)
    assert by[0]["front_in_budget"] == 1.0 and by[100]["front_in_budget"] == by[100]["front"]
    assert [by[p]["time"] for p in (0, 10, 25, 50, 75, 100)] == sorted([by[p]["time"] for p in (0, 10, 25, 50, 75, 100)], reverse=True)               # mehr Budget, schnellere Route
    assert eff["plain"] == pytest.approx(23889.8, abs=0.5) and eff["bounds"] == pytest.approx(23888.6, abs=0.5) and eff["bounds"] > 0.999 * eff["plain"]            # die Schranken sparen fast nichts


def test_fronts_never_depend_on_bounds_or_budget_pruning_in_the_presets():
    for key in ("small", "city", "random", "chain"):
        _, net, a = _preset(key)
        fast = alg.pareto_labels(net.graph, net.source, net.target, prune="bounds")
        assert [(t, c) for t, c, _ in fast.front()] == [(t, c) for t, c, _ in a.front], key
    _, net, a = _preset("city")
    b = ev.budget_for(a, 30)
    assert [(t, c) for t, c, _ in ev.budget_run(net, b).front()] == [(t, c) for t, c, _ in a.front if c <= b]


def test_every_front_is_free_of_dominated_points_in_every_preset_and_matches_the_naive_reference_on_the_small_net():
    for key in ("small", "city", "random", "chain"):
        _, _, a = _preset(key)
        cs = [c for _, c, _ in a.front]
        assert all(cs[i] > cs[i + 1] for i in range(len(cs) - 1)), key
    _, net, a = _preset("small")
    assert [(t, c) for t, c, _ in a.front] == alg.brute_force_front(net.graph, net.source, net.target) == alg.label_correcting_front(net.graph, net.source, net.target)
