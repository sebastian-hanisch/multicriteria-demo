"""Mehrkriterien-Routing - zwei Kosten, keine beste Route - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - das Label-setting für die Pareto-Front - und lässt stattdessen das Beispiel wachsen.
Achtes und letztes Stück der Kürzeste-Wege-Linie der "Konzepte"-Reihe, dritter Ast von Dijkstra: mit zwei Kosten (Zeit und CO2) gibt es keine beste Route mehr, sondern eine Pareto-Front.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

import mc_algorithm as alg
import mc_constants as C
import mc_evaluation as ev
from mc_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from mc_scenario import make_network
from mc_visualization import (
    build_astar,
    build_budget,
    build_chain,
    build_front,
    build_front_corr,
    build_front_size,
    build_missed,
    build_network,
    table_state,
)

st.set_page_config(page_title="Mehrkriterien-Routing – Sebastian Hanisch", layout="wide")


def _num(x):
    return f"{x:,.0f}".replace(",", ".")


def _pct(x, digits=0):
    return "–" if x is None or np.isnan(x) else f"{x:.{digits}%}"


def _rel(x, ref):
    """Relative Abweichung "+12 %" gegenüber ref; "–", wenn ref null ist (die Kette hat eine Route mit Zeit 0)."""
    return "–" if not ref else f"{x / ref - 1:+.0%}"


@st.cache_resource(show_spinner=False, max_entries=8)
def _network(params):
    return make_network(*params)


@st.cache_resource(show_spinner=False, max_entries=8)
def _analysis(params):
    return ev.analyse(_network(params))


@st.cache_data(show_spinner=False)
def _front_size():
    return ev.front_vs_size()


@st.cache_data(show_spinner=False)
def _front_corr():
    return ev.front_vs_corr()


@st.cache_data(show_spinner=False)
def _missed():
    return ev.missed_by_weighted_sum()


@st.cache_data(show_spinner=False)
def _chain():
    return ev.chain_rows()


@st.cache_data(show_spinner=False)
def _budget():
    return ev.budget_curve()


@st.cache_data(show_spinner=False)
def _astar():
    return ev.astar_rows()


st.title("🌿 Mehrkriterien-Routing – zwei Kosten, keine beste Route")
st.markdown(
    """
Dijkstra vergleicht Kosten mit "kleiner". Mit **zwei** Kosten - Fahrzeit und CO₂ - gibt es diese Ordnung nicht mehr: die schnellste Route ist selten die sauberste. Die Antwort ist keine Route, sondern eine **Pareto-Front**: alle Routen, die man nicht in beiden Kosten zugleich verbessern kann.
An die Stelle von "kleiner" tritt die **Dominanz**: eine Route dominiert eine andere, wenn sie in beiden Kosten höchstens so groß und in einer kleiner ist. Das **Label-setting** führt an jedem Knoten nicht eine Entfernung, sondern **alle nicht dominierten Labels** (Zeit, CO₂) mit.
Praktisch fragt man dann: "die schnellste Route mit höchstens *B* Gramm CO₂" - das lässt sich aus der Front ablesen.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - achtes und letztes Stück der Kürzeste-Wege-Linie der \"Konzepte\"-Reihe, dritter Ast von Dijkstra - **ein** Verfahren an einem wachsenden Beispiel. "
    "Die Schwäche des Verfahrens: die Front kann sehr groß werden (im Schlimmsten Fall exponentiell), und eine gewichtete Summe aus beiden Kosten findet nicht alle ihrer Punkte. Das Verfahren geht auf Hansen (1980) und Martins (1984) zurück; "
    "alle Netze und Zahlen dieser Demo sind eigene Graphen und Messungen."
)

with st.expander("So funktioniert Mehrkriterien-Routing", expanded=True):
    st.markdown(
        """
1. **Labels statt Entfernungen:** ein Label $(z, c)$ an einem Knoten ist ein Weg dorthin mit Zeit $z$ und CO₂ $c$. Der Start hat das Label $(0, 0)$.
2. **Dominanz:** $(z, c)$ dominiert $(z', c')$, wenn $z \\le z'$ und $c \\le c'$ (und die Labels nicht gleich sind). Ein dominiertes Label ist nie Teil einer Front-Route, es wird verworfen.
3. **Label-setting:** die Warteschlange gibt das Label mit der kleinsten Zeit (bei Gleichstand dem kleinsten CO₂) heraus. Wird es von keinem früher festgelegten Label desselben Knotens dominiert, wird es **festgelegt** und über alle Kanten zu neuen Labels erweitert. Weil die Reihenfolge nach der Zeit geht, genügt ein Vergleich mit dem kleinsten bisherigen CO₂ des Knotens.
4. **Front:** die festgelegten Labels am Ziel sind die Pareto-Front; jede ist eine Route (über die Vorgängerlabels).
5. **Gewichtete Summe:** $a \\cdot z + b \\cdot c$ mit Dijkstra minimieren findet nur **Ecken der Konvexhülle** der Front. Punkte, die über der Hülle oder auf einer Strecke zwischen zwei Ecken liegen, sind Front-Routen - aber für keine Gewichtung das eindeutige Optimum.
6. **CO₂-Budget:** die schnellste Front-Route mit $c \\le B$ (das "Resource-Constrained Shortest Path"-Problem) steht in der Front.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", C.NETS, key="net_select", format_func=lambda k: C.NET_LABELS[k],
        help="Klein und fest (vier Front-Routen), erzeugt (Stadtnetz mit Hauptachsen, Zufallsnetz mit Korrelation) oder die Worst-Case-Kette (2^k Front-Routen). Zeit in Sekunden bzw. Minuten, CO₂ in Gramm, alle Kosten ganze Zahlen; höchstens 400 Knoten.",
    )
    if net_key == "city":
        seed_widget("side_slider")
        side = st.slider("Kreuzungen je Seite", *bounds("side_slider"), key="side_slider",
                         help="Größe des Rasters: n = Seite² Knoten. Punkte der Front am Ziel (Mittel über fünf Netze, Hauptachsen doppelt so schmutzig) bei 6 / 10 / 20 Kreuzungen je Seite: 11.8 / 34.0 / 91.8.")
        st.session_state[KEPT["side_slider"]] = side
        seed_widget("conflict_slider")
        conflict = st.slider("Gegenläufigkeit (Zusatz-CO₂ der Hauptachse)", *bounds("conflict_slider"), key="conflict_slider", step=0.1,
                             help="Um wie viel schmutziger die schnelle Hauptachse je km ist (1 = doppelt so viel CO₂ wie die Nebenstraße). Punkte der Front im 10 × 10-Netz bei 0 / 0.5 / 1: 11.4 / 28.2 / 34.0 - mit 0 ist die Hauptachse nur schneller, nicht schmutziger.")
        st.session_state[KEPT["conflict_slider"]] = conflict
    else:
        side = int(st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))
        conflict = float(st.session_state.get(KEPT["conflict_slider"], C.DEFAULT_CONFLICT))
    if net_key == "random":
        seed_widget("nodes_slider")
        nodes = st.slider("Knoten", *bounds("nodes_slider"), key="nodes_slider", step=10, help="Anzahl der Knoten n.")
        st.session_state[KEPT["nodes_slider"]] = nodes
        seed_widget("degree_slider")
        degree = st.slider("Mittlerer Grad", *bounds("degree_slider"), key="degree_slider", step=0.5, help="Kanten je Knoten.")
        st.session_state[KEPT["degree_slider"]] = degree
        seed_widget("corr_slider")
        corr = st.slider("Korrelation der Kosten je Kante", *bounds("corr_slider"), key="corr_slider", step=0.25,
                         help="−1 = gegenläufig (was schnell ist, ist schmutzig), +1 = gleichläufig (beide Kosten gleich). Punkte der Front (200 Knoten, Grad 3, Mittel über fünf Netze) bei −1 / −0.5 / 0 / 0.5 / 1: 4.4 / 2.2 / 2.0 / 1.6 / 1.0.")
        st.session_state[KEPT["corr_slider"]] = corr
    else:
        nodes = int(st.session_state.get(KEPT["nodes_slider"], C.DEFAULT_NODES))
        degree = float(st.session_state.get(KEPT["degree_slider"], C.DEFAULT_DEGREE))
        corr = float(st.session_state.get(KEPT["corr_slider"], C.DEFAULT_CORR))
    if net_key == "chain":
        seed_widget("links_slider")
        links = st.slider("Glieder der Kette k", *bounds("links_slider"), key="links_slider",
                          help="Jedes Glied hat zwei Wege, (2^i s, 0 g) gegen (0 s, 2^i g). Die Front hat 2^k Punkte: bei k = 4 / 8 / 12 sind es 16 / 256 / 4 096, das Label-setting erzeugt dabei 61 / 1 021 / 16 381 Labels.")
        st.session_state[KEPT["links_slider"]] = links
    else:
        links = int(st.session_state.get(KEPT["links_slider"], C.DEFAULT_LINKS))
    if net_key in ("city", "random"):
        seed_widget("seed_input")
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für das Netz.")
    else:
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen.")

# nicht zum Netz gehörende Regler ändern das Netz nicht: sonst würden gleiche Netze unter verschiedenen Schlüsseln mehrfach berechnet
d = dict(side=C.DEFAULT_SIDE, conflict=C.DEFAULT_CONFLICT, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, corr=C.DEFAULT_CORR, links=C.DEFAULT_LINKS, seed=C.DEFAULT_SEED)
if net_key == "city":
    d.update(side=int(side), conflict=round(float(conflict), 1), seed=int(seed))
elif net_key == "random":
    d.update(nodes=int(nodes), degree=round(float(degree), 1), corr=round(float(corr), 2), seed=int(seed))
elif net_key == "chain":
    d.update(links=int(links))
params = (net_key, d["side"], d["conflict"], d["nodes"], d["degree"], d["corr"], d["links"], d["seed"])
with st.spinner("Rechne ..."):
    a = _analysis(params)
net, m, g = a.net, a.metrics, a.net.graph
small = bool(g.names)
front, res = a.front, a.res
F = len(front)
hull_ids = {(t, c): lid for t, c, lid in front}
hull_pts = a.hull

# --- Sidebar, zweiter Teil: was von der Front abhängt (nur wenn sie mehr als einen Punkt hat) ----------------------------------------------------------------------
view_key = (params,)
if st.session_state.get("mc_owner") != view_key:
    st.session_state["mc_owner"] = view_key
    st.session_state["point_slider"] = min(F // 2, max(F - 1, 0))
    st.session_state["budget_slider"] = 50
mode = "front"
with st.sidebar:
    if F > 1:
        seed_widget("mode_select")
        mode = st.selectbox("Anzeige", C.MODES, key="mode_select", format_func=lambda k: C.MODE_LABELS[k],
                            help="Pareto-Front: alle Kompromisse, eine Route wählbar. Gewichtete Summe: nur die Ecken der Konvexhülle, die eine gewichtete Summe aus Zeit und CO₂ finden kann. Budget: die schnellste Route mit höchstens B Gramm CO₂.")
        st.session_state[KEPT["mode_select"]] = mode
        if mode in ("front", "weighted"):
            top = F - 1 if mode == "front" else len(hull_pts) - 1
            if st.session_state.get("point_slider", 0) > top:
                st.session_state["point_slider"] = top // 2
            point = st.slider("Route auf der Front (0 = schnellste)" if mode == "front" else "Ecke der Hülle (0 = schnellste)", 0, top, key="point_slider",
                              help="Welche Route die Karte zeigt: 0 ist die schnellste (meist die schmutzigste), ganz rechts die sauberste.")
        else:
            budget_pct = st.slider("CO₂-Budget [% der Spanne]", 0, 100, key="budget_slider",
                                   help="0 % = das CO₂ der saubersten Route, 100 % = das der schnellsten. Gefragt ist die schnellste Route, die im Budget bleibt.")
    else:
        st.caption("Nur ein Kompromiss - schnellste und sauberste Route sind dieselbe.")
mode = mode if F > 1 else "front"
sync_query_params({"net_select": net_key, "side_slider": int(side), "conflict_slider": round(float(conflict), 1), "nodes_slider": int(nodes), "degree_slider": round(float(degree), 1),
                   "corr_slider": round(float(corr), 2), "links_slider": int(links), "mode_select": mode, "seed_input": int(seed)})

# gewählte Route
if F == 0:
    chosen, chosen_lid, budget = None, None, None
elif mode == "front":
    chosen = front[min(st.session_state.get("point_slider", 0), F - 1)]
    chosen_lid, budget = chosen[2], None
elif mode == "weighted":
    hp = hull_pts[min(st.session_state.get("point_slider", 0), len(hull_pts) - 1)]
    chosen = (hp[0], hp[1], hull_ids[hp])
    chosen_lid, budget = chosen[2], None
else:
    budget = ev.budget_for(a, st.session_state.get("budget_slider", 50))
    chosen = alg.rcsp(front, budget)
    chosen_lid = chosen[2]
last_step = len(ev.frames(a)) - 1
frame_list = ev.frames(a)
if st.session_state.get("mc_step_owner") != view_key:
    st.session_state["mc_step_owner"] = view_key
    st.session_state["mc_step"] = last_step

# --- Mehrkriterien-Routing in Aktion ---------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Mehrkriterien-Routing in Aktion")
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.slider("Entnahmen aus der Warteschlange", 0, last_step, key="mc_step",
                     help="Wie viele Labels das Label-setting schon aus der Warteschlange genommen hat (festgelegt oder als dominiert verworfen): 0 = nur der Start, ganz rechts = fertig, die Front steht. Bei vielen Labels zeigt die Ansicht etwa 60 Zwischenstände.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _render(current):
    k = frame_list[current]
    final = k >= len(res.events)
    with view_slot.container():
        routes = []
        if final and chosen is not None:
            routes = [(front[0][2], "schnellste Route", C.COLORS["fast"], "dash"), (front[-1][2], "sauberste Route", C.COLORS["clean"], "dash"), (chosen_lid, "gewählte Route", C.COLORS["chosen"], "solid")]
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(build_network(a, k, routes, height=400 if small else 460), width="stretch", key=f"net_chart_{current}")
        c2.plotly_chart(build_front(a, k, mode, (chosen[0], chosen[1]) if chosen else None, budget), width="stretch", key=f"front_chart_{current}")
        if small:
            st.markdown(f"**Labels je Ort nach {k} Entnahmen** (✓ festgelegt, ✗ dominiert)")
            st.dataframe(pd.DataFrame(table_state(a, k)), hide_index=True, width="stretch")
        if final and chosen is not None:
            unit_t, unit_c = ("min" if net.key == "small" else "s"), "g"
            extra = ""
            if F > 1:
                extra = f" - {_rel(chosen[0], front[0][0])} Zeit gegenüber der schnellsten, {_rel(chosen[1], front[-1][1])} CO₂ gegenüber der saubersten"
            st.markdown(f"**Gewählte Route:** {chosen[0]:g} {unit_t} · {chosen[1]:g} {unit_c}{extra}.")


if auto_play:
    n_frames = min(last_step + 1, 40)
    for kk in sorted({int(round(x)) for x in np.linspace(0, last_step, n_frames)}):
        _render(kk)
        time.sleep(min(0.7, 6.0 / n_frames))
    step = last_step
else:
    _render(step)
st.caption(net.note)

st.markdown("---")

# --- Zwei Kosten - keine beste Route -----------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Zwei Kosten – keine beste Route")
st.caption("**Zähler** = Schritte, die ein Verfahren ausführt: Labels, die das Label-setting erzeugt, festlegt oder als dominiert verwirft; Dijkstra-Läufe der gewichteten Summe. Sie sind plattformfest. Laufzeiten stehen nur als Messwerte im Vergleich unten.")
code = ev.verdict(a)
if code == "unreachable":
    st.warning("⚠️ Das Ziel ist vom Start aus nicht erreichbar.")
else:
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Front", _num(m["front"]), delta=f"{m['hull']} Ecken der Hülle" if F > 1 else "ein Kompromiss", delta_color="off", help="Zahl der nicht dominierten Routen am Ziel. Ecken der Konvexhülle findet eine gewichtete Summe.")
    p2.metric("Labels", _num(m["generated"]), delta=f"{_num(m['settled'])} festgelegt, {_num(m['dominated'])} dominiert", delta_color="off", help="Erzeugte Labels des Label-setting (bis alle Labels am Ziel festgelegt sind).")
    p3.metric("Gewichtete Summe", f"{m['ws_points']} Routen", delta=f"{m['ws_runs']} Dijkstra-Läufe", delta_color="off", help="Die Routen, die eine gewichtete Summe finden kann (dichotome Suche nach den Ecken der Konvexhülle).")
    if F > 1 and not np.isnan(m["price_fast"]):
        p4.metric("Preis der schnellsten Route", f"+{m['price_fast']:.0%} CO₂", delta=f"sauberste: +{m['price_clean']:.0%} Zeit", delta_color="off", help="Die schnellste Route hat so viel mehr CO₂ als die sauberste; die sauberste braucht so viel mehr Zeit als die schnellste.")
    else:
        p4.metric("Preis der schnellsten Route", "–", help="Nur ein Kompromiss oder eine Kostenart ist bei der schnellsten Route null.")
    if code == "single":
        st.info("ℹ️ In diesem Netz gibt es nur einen Kompromiss: die schnellste Route ist zugleich die sauberste. Zwei Kosten machen erst dann Arbeit, wenn sie gegeneinander laufen (Regler \"Gegenläufigkeit\" bzw. \"Korrelation\").")
    elif mode == "budget":
        bres = ev.budget_run(net, budget)
        st.success(f"✅ **Schnellste Route mit höchstens {budget:.0f} g CO₂:** {chosen[0]:g} s / {chosen[1]:g} g ({_rel(chosen[0], front[0][0])} Zeit gegenüber der schnellsten). Aus der Front abgelesen; das Label-setting mit dem Budget kommt auf dieselbe Route mit "
                   f"{_num(bres.counters['generated'])} statt {_num(m['generated'])} Labels ({1 - bres.counters['generated'] / m['generated']:.0%} weniger).")
    else:
        missed = m["front"] - m["hull"]
        st.success(f"✅ **Keine beste Route, sondern {m['front']} Kompromisse** zwischen {m['fast_t']:g} / {m['fast_c']:g} (schnellste) und {m['clean_t']:g} / {m['clean_c']:g} (sauberste). "
                   f"Eine gewichtete Summe findet nur {m['hull']} davon ({m['ws_runs']} Dijkstra-Läufe); " +
                   (f"der andere Punkt ({missed / m['front']:.0%}) ist keine Ecke der Konvexhülle - eine Route, die für keine Gewichtung das eindeutige Optimum ist." if missed == 1 else
                    f"die anderen {missed} ({missed / m['front']:.0%}) sind keine Ecken der Konvexhülle - Routen, die für keine Gewichtung das eindeutige Optimum sind."))

st.markdown("---")

# --- Vergleich -------------------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – Label-setting und gewichtete Summe im Vergleich"):
    st.table({"Verfahren": ["Label-setting (die ganze Front)", "Label-setting mit Schranken zum Ziel", "Label-setting in A*-Ordnung", "Gewichtete Summe (dichotom)"], "Zähler": [f"{_num(m['generated'])} Labels", f"{_num(m['generated_bounds'])} Labels", f"{_num(m['generated_astar'])} Labels", f"{m['ws_runs']} Dijkstra-Läufe"],
              "Laufzeit [ms]": [f"{a.seconds['labels'] * 1000:.1f}", "–", "–", f"{a.seconds['ws'] * 1000:.1f}"]})
    st.caption("Die Laufzeiten sind Messwerte dieses Laufs (reines Python, ein Lauf, schwankend). Die Schranken zum Ziel (kürzeste Zeit und kürzestes CO₂ von jedem Knoten zum Ziel, zwei Dijkstra-Läufe auf dem umgedrehten Graphen, nicht mitgezählt) verwerfen Labels, deren bestmögliches Ende schon von einem Ziel-Label dominiert wird. "
               "Die A*-Ordnung (NAMOA*, Ulloa et al. 2020) entnimmt die Labels nach Kosten plus Schranke statt nach den bisherigen Kosten. Beide ändern die Front nicht (in jedem Lauf geprüft); was sie sparen, zeigt das Experiment zu den Schranken.")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie groß wird die Front?")
if st.button("Front gegen Größe, Gegenläufigkeit und Korrelation messen (dauert einen Moment)", key="front_start"):
    st.session_state["front_on"] = True
if st.session_state.get("front_on"):
    with st.spinner("Rechne 6 Größen × 3 Gegenläufigkeiten und 5 Korrelationen × 5 Netze ..."):
        frows, crows = _front_size(), _front_corr()
    c1, c2 = st.columns(2)
    c1.markdown("**Stadtnetz: gegen Größe und Gegenläufigkeit**")
    c1.plotly_chart(build_front_size(frows), width="stretch", key="front_size_chart")
    c2.markdown("**Zufallsnetz (200 Knoten): gegen die Korrelation**")
    c2.plotly_chart(build_front_corr(crows), width="stretch", key="front_corr_chart")
    big = [r for r in frows if r["side"] == 20]
    zero, one = big[0], big[-1]
    st.caption(f"Mittel über 5 Netze; Start unten links, Ziel oben rechts (Stadtnetz) bzw. der am weitesten entfernte Knoten (Zufallsnetz). Im 20 × 20-Stadtnetz hat die Front {zero['front']:.0f} Punkte, wenn die Hauptachse nur schneller ist, und {one['front']:.0f}, wenn sie doppelt so schmutzig ist; "
               f"die Hülle bleibt bei {one['hull']:.0f} Ecken. Die Front wächst mit der Netzgröße und mit der Gegenläufigkeit. Im Zufallsnetz schrumpft sie mit der Korrelation von {crows[0]['front']:.1f} (gegenläufig) auf {crows[-1]['front']:.1f} (gleichläufig): "
               "laufen beide Kosten gleich, gibt es keinen Kompromiss.")

st.markdown("---")

st.subheader("🔬 Was die gewichtete Summe verpasst")
if st.button("Front gegen gefundene Ecken der Hülle vergleichen (dauert einen Moment)", key="missed_start"):
    st.session_state["missed_on"] = True
if st.session_state.get("missed_on"):
    with st.spinner("Rechne 5 Netztypen × 5 Netze ..."):
        mrows = _missed()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_missed(mrows), width="stretch", key="missed_chart")
    c2.table({"Netz": [r["label"] for r in mrows], "Front": [f"{r['front']:.1f}" for r in mrows], "gefunden": [f"{r['found']:.1f}" for r in mrows]})
    st1, st20, ch = mrows[0], mrows[2], mrows[-1]
    st.caption(f"Mittel über 5 Netze. Die gewichtete Summe (dichotome Suche, {st20['runs']:.0f} Dijkstra-Läufe im 20 × 20-Stadtnetz) findet nur die Ecken der Konvexhülle: im Stadtnetz verpasst sie {st1['missed_share']:.0%} (8 × 8) bis {st20['missed_share']:.0%} (20 × 20) der Front, in der Kette {ch['missed_share']:.1%} "
               f"({ch['found']:.0f} von {ch['front']:.0f}). Sind beide Kosten unabhängig, gibt es kaum etwas zu verpassen ({mrows[4]['missed_share']:.0%}). Ein Punkt, der keine Ecke der Hülle ist, ist für keine Gewichtung das eindeutige Optimum - er ist trotzdem eine Front-Route und oft ein guter Kompromiss.")

st.markdown("---")

st.subheader("🔬 Die Worst-Case-Kette")
if st.button("Front und Labels gegen die Kettenlänge messen (dauert einen Moment)", key="chain_start"):
    st.session_state["chain_on"] = True
if st.session_state.get("chain_on"):
    with st.spinner("Rechne Ketten mit 2 bis 12 Gliedern ..."):
        krows = _chain()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_chain(krows), width="stretch", key="chain_chart")
    c2.table({"Glieder k": [str(r["k"]) for r in krows], "Front (2^k)": [_num(r["front"]) for r in krows], "Labels": [_num(r["generated"]) for r in krows]})
    st.caption(f"Jedes Glied bietet zwei Wege, (2^i, 0) gegen (0, 2^i): jede der 2^k Kombinationen hat Zeit + CO₂ = 2^k − 1, keine dominiert eine andere. Bei k = {krows[-1]['k']} hat die Front {_num(krows[-1]['front'])} Punkte, das Label-setting erzeugt {_num(krows[-1]['generated'])} Labels - "
               "die Front verdoppelt sich mit jedem Glied. Das ist der Schlimmste Fall: der Aufwand hängt von der **Größe der Front** ab, nicht nur von der des Netzes.")

st.markdown("---")

st.subheader("🔬 Das CO₂-Budget")
if st.button("Budget gegen Aufwand messen (dauert einen Moment)", key="budget_start"):
    st.session_state["budget_on"] = True
if st.session_state.get("budget_on"):
    with st.spinner("Rechne 6 Budgets × 5 Netze ..."):
        brows = _budget()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_budget(brows), width="stretch", key="budget_chart")
    c2.table({"Budget [%]": [str(r["percent"]) for r in brows], "Front im Budget": [f"{r['front_in_budget']:.1f}" for r in brows], "Zeit [s]": [f"{r['time']:.0f}" for r in brows]})
    half = brows[3]
    st.caption(f"Stadtnetz 12 × 12 (Hauptachsen doppelt so schmutzig), Mittel über 5 Netze. Je knapper das Budget, desto weniger Labels bleiben übrig: bei 50 % der Spanne {half['generated']:.0f} statt {half['full']:.0f} ({1 - half['generated'] / half['full']:.0%} weniger), bei 10 % nur {brows[1]['generated']:.0f}; die Front im Budget enthält dann {brows[1]['front_in_budget']:.0f} von {brows[1]['front']:.0f} Punkten. "
               f"Die schnellste Route im Budget wird mit weniger Budget langsamer ({brows[-1]['time']:.0f} s ohne Beschränkung, {brows[0]['time']:.0f} s bei der saubersten Route).")

st.markdown("---")

st.subheader("🔬 Schranken zum Ziel und A*-Ordnung")
if st.button("Ordnungen der Warteschlange vergleichen (dauert einen Moment)", key="astar_start"):
    st.session_state["astar_on"] = True
if st.session_state.get("astar_on"):
    with st.spinner("Rechne 5 Netzgrößen × 5 Netze × 3 Verfahren ..."):
        arows = _astar()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_astar(arows), width="stretch", key="astar_chart")
    c2.table({"Netz": [r["label"].replace("Stadtnetz ", "").replace("Zufallsnetz 200 Knoten, gegenläufig", "Zufall") for r in arows], "Dijkstra": [_num(r["dominance"]) for r in arows], "A*": [_num(r["astar"]) for r in arows]})
    city = arows[:-1]
    st.caption(f"Erzeugte Labels, Mittel über 5 Netze (Stadtnetze mit doppelt so schmutzigen Hauptachsen). Schranken zum Ziel **allein** sparen in den Stadtnetzen nichts (höchstens {max(1 - r['bounds'] / r['dominance'] for r in city):.2%}): die Labels werden nach ihren bisherigen Kosten entnommen, das Ziel wird erst spät erreicht, und bis dahin ist fast alles erzeugt. "
               f"In **A\\*-Ordnung** (nach Kosten plus Schranke) kommt das Ziel früh, und die Schranken wirken: {1 - city[0]['astar'] / city[0]['dominance']:.0%} weniger Labels im 8 × 8-Netz, {1 - city[-1]['astar'] / city[-1]['dominance']:.0%} im 20 × 20-Netz - die Ersparnis **schrumpft mit der Größe**, weil die Front (im Mittel {city[0]['front']:.0f} gegen {city[-1]['front']:.0f} Punkte) mitwächst und jede Front-Route ihre Labels braucht. "
               f"Im gegenläufigen Zufallsnetz (kleine Front, {arows[-1]['front']:.1f} Punkte) sind es {1 - arows[-1]['astar'] / arows[-1]['dominance']:.0%} weniger, die Schranken allein {1 - arows[-1]['bounds'] / arows[-1]['dominance']:.0%}. Die Front ist in jedem Lauf dieselbe.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Front bleibt klein** | Sie wächst mit dem Netz und mit der Gegenläufigkeit (Stadtnetz 20 × 20: im Mittel 92 Punkte) und im Schlimmsten Fall exponentiell (Kette mit 12 Gliedern: 4 096 Punkte, 16 381 Labels). | ε-Fronten (Näherung, Literatur), CO₂-Budget |
| **Blindes Suchen genügt** | Bei einem festen Ziel spart die A*-Ordnung im Stadtnetz 36–51 % der Labels, die Schranken allein nichts; die Ersparnis schrumpft mit der Größe, die Front bleibt dieselbe. | NAMOA*, BOA* |
| **Eine gewichtete Summe genügt** | Sie findet nur die Ecken der Konvexhülle: im 20 × 20-Stadtnetz etwa 15 von 92 Punkten, in der Kette 2 von 256. | Label-setting |
| **Nur zwei additive Kosten** | Mehr Kosten heißt höhere Dimension und größere Fronten; nicht additive Größen (Zeitfenster, Batterieladung) brauchen Ressourcenverlängerungsfunktionen (Literatur, nicht gebaut). | Resource-Constrained Shortest Path |
| **Die Kosten ändern sich nicht** | Ändert sich der Verkehr, muss die Front neu berechnet werden; Vorberechnungen wie Contraction Hierarchies gibt es für mehrere Kosten nur in aufwendigen Varianten (Literatur). | (nicht gebaut) |
| **Es muss exakt sein** | Evolutionäre Verfahren wie **NSGA-II** (Populations-Linie des Portfolios) berechnen Fronten nur näherungsweise, aber ohne die Größe der exakten Front bewältigen zu müssen. | NSGA-II |
"""
)
st.caption("Damit endet die Kürzeste-Wege-Linie. Gebaut sind: Breitensuche, Dijkstra, bidirektionale Suche, Contraction Hierarchies, Bellman-Ford, Floyd-Warshall, Johnson und Mehrkriterien-Routing.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Gerichteter Graph $G=(V,E)$ mit zwei nichtnegativen Kosten $(z_e, c_e)\in\mathbb{R}_{\ge0}^2$ je Kante (Zeit, CO₂), Start $s$, Ziel $t$. Für eine Route $P$ ist $\mathbf{k}(P)=\bigl(\sum z_e,\sum c_e\bigr)$.

**Dominanz und Front.** $\mathbf{a}$ *dominiert* $\mathbf{b}$, wenn $a_1\le b_1$, $a_2\le b_2$ und $\mathbf{a}\ne\mathbf{b}$. Die **Pareto-Front** ist die Menge der nicht dominierten Kostenvektoren der Routen $s\leadsto t$; zu jedem Punkt gehört mindestens eine Route.

**Korrektheit des Label-setting.** Addiert man dieselbe Kante zu zwei Labels, bleibt die Dominanz erhalten: dominiert $\mathbf{a}$ das Label $\mathbf{b}$ an einem Knoten $v$, dominiert $\mathbf{a}+\mathbf{k}(P')$ das Label $\mathbf{b}+\mathbf{k}(P')$ für jede Fortsetzung $P'$ (**Optimalitätsprinzip für Vektorkosten**).
Ein dominiertes Label kann also nie zu einer Front-Route führen und darf verworfen werden. Die Entnahme nach (Zeit, CO₂) lexikographisch garantiert: ein entnommenes Label wird höchstens von früher entnommenen dominiert (deren Zeit ist nicht größer), also genügt der Vergleich seines CO₂ mit dem kleinsten bisherigen CO₂ des Knotens.
Terminierung: die Kosten sind nichtnegativ, jedes Label ist ein Weg; Wege mit Zyklen werden dominiert oder sind gleich (und werden als gleich verworfen).

**Gewichtete Summe.** $\min_P\, a\,z(P)+b\,c(P)$ mit $a,b>0$ ist ein Dijkstra-Problem. Sein Optimum ist immer eine **Ecke der unteren Konvexhülle** der Front. Front-Punkte, die über der Hülle oder auf einer Strecke zwischen zwei Ecken liegen ("nicht unterstützt"), sind für keine Gewichtung das eindeutige Optimum.
Die dichotome Suche nimmt die beiden Extreme (schnellste, sauberste Route) und sucht für die Normale $(c_p-c_q,\ z_q-z_p)$ der Verbindungsstrecke jeweils eine Route unterhalb der Strecke.

**Worst-Case-Kette.** Glied $i$ bietet die Wege $(2^i,0)$ und $(0,2^i)$. Wählt man für die Menge $S$ von Gliedern den ersten Weg, ist $\mathbf{k}=(\sum_{i\in S}2^i,\ \sum_{i\notin S}2^i)$ mit Summe $2^k-1$: alle $2^k$ Vektoren sind verschieden und nicht vergleichbar - die Front hat $2^k$ Punkte.

**CO₂-Budget (RCSP).** $\min z(P)$ unter $c(P)\le B$ ist das Minimum der Zeit über alle Front-Punkte mit $c\le B$: der beste zulässige Vektor ist nicht dominiert, also in der Front. Das Label-setting darf zusätzlich Labels mit $c>B$ verwerfen.

**Aufwand.** Der Aufwand hängt von der Zahl der Labels ab: höchstens $|F_v|$ nicht dominierte Labels je Knoten $v$, insgesamt $\sum_v |F_v|$ Labels, jede über die ausgehenden Kanten erweitert - bei Front-Größen in der Größenordnung der Kette exponentiell (Lehrbuchwert; gemessen: siehe Experimente).

Implementiert in `mc_graph.py` (CSR-Graph mit zwei Kostenarten), `mc_algorithm.py` (Label-setting, Schranken, Budget, gewichtete Summe, Konvexhülle und naive Referenzen), `mc_scenario.py` (Netze), `mc_evaluation.py` (Kennzahlen, Bildfolge, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
