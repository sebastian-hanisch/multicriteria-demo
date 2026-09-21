"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, alle Anzeigemodi, Randgrößen, Schritt-Zustand, ausgeblendete Regler, Abspielen, Permalink, Experimente auf Abruf, Schlüssel."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mc_constants as C
from mc_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None, timeout=600, net=None):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    if net:
        at.query_params["net"] = net
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input)}


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()


def test_default_renders_without_exception_and_states_the_front():
    at = _run()
    assert any("Mehrkriterien-Routing in Aktion" in m.value for m in at.markdown)
    assert len(at.success) == 1 and "4 Kompromisse" in at.success[0].value and not at.warning


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_one_verdict(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert len(at.success) == 1 and not at.warning and not at.error


@pytest.mark.parametrize("mode", C.MODES)
@pytest.mark.parametrize("net", C.NETS)
def test_every_mode_renders_on_every_net(mode, net):
    def setup(at):
        at.session_state["net_select"] = net
        at.session_state["mode_select"] = mode
        at.session_state["side_slider"] = 6
    at = _run(setup)
    assert len(at.success) + len(at.info) == 1 and not at.warning


def test_a_front_with_one_point_hides_the_mode_and_the_point_controls_and_says_so():
    def setup(at):
        at.session_state["net_select"] = "random"
        at.session_state["corr_slider"] = 1.0
    at = _run(setup)
    assert len(at.info) == 1 and "nur einen Kompromiss" in at.info[0].value
    assert "Anzeige" not in _labels(at) and not any(s.key == "point_slider" for s in at.sidebar.slider)


def test_budget_mode_states_the_fastest_route_within_the_budget():
    def setup(at):
        at.session_state["mode_select"] = "budget"
    at = _run(setup)
    assert len(at.success) == 1 and "Schnellste Route mit höchstens" in at.success[0].value
    at.slider(key="budget_slider").set_value(0)
    at.run()
    assert not at.exception and "höchstens 30 g" in at.success[0].value


def test_extreme_settings_render():
    def small(at):
        at.session_state["net_select"] = "city"
        at.session_state["side_slider"] = C.SIDE_MIN

    def big(at):
        at.session_state["net_select"] = "city"
        at.session_state["side_slider"] = C.SIDE_MAX
        at.session_state["conflict_slider"] = C.CONFLICT_MAX

    def flat(at):
        at.session_state["net_select"] = "city"
        at.session_state["conflict_slider"] = C.CONFLICT_MIN

    def longest_chain(at):
        at.session_state["net_select"] = "chain"
        at.session_state["links_slider"] = C.LINKS_MAX

    def shortest_chain(at):
        at.session_state["net_select"] = "chain"
        at.session_state["links_slider"] = C.LINKS_MIN
    for setup in (small, big, flat, longest_chain, shortest_chain):
        at = _run(setup)
        assert at.slider(key="mc_step").value == at.slider(key="mc_step").max


def test_hidden_controls_follow_the_net():
    small, city, rnd, chain = (_labels(_run(net=n)) for n in ("small", "city", "random", "chain"))
    assert small == {"Netz", "Anzeige", "Route auf der Front (0 = schnellste)"}                        # feste Aufgabe: kein Seed
    assert city == small | {"Kreuzungen je Seite", "Gegenläufigkeit (Zusatz-CO₂ der Hauptachse)", "Zufalls-Seed"}
    assert rnd == small | {"Knoten", "Mittlerer Grad", "Korrelation der Kosten je Kante", "Zufalls-Seed"}
    assert chain == small | {"Glieder der Kette k"}


def test_hidden_slider_values_come_back_when_the_net_is_shown_again():
    # Die erste Sicht muss das Netz mit dem Regler sein: AppTest verliert den Wert, wenn der Regler zuerst ausgeblendet war (im echten Browser bleibt er erhalten).
    at = _run(net="city")
    at.session_state["conflict_slider"] = 0.5
    at.run()
    at.session_state["net_select"] = "small"
    at.run()
    at.session_state["net_select"] = "city"
    at.run()
    assert not at.exception and at.slider(key="conflict_slider").value == 0.5


def test_the_mode_survives_a_net_with_a_single_point_front():
    def to_budget(at):
        at.session_state["mode_select"] = "budget"
    at = _run(to_budget)
    at.session_state["net_select"] = "random"
    at.session_state["corr_slider"] = 1.0
    at.run()
    at.session_state["corr_slider"] = -1.0
    at.run()
    assert not at.exception and at.selectbox(key="mode_select").value == "budget"


def test_step_slider_returns_to_the_last_step_and_the_point_to_the_middle_when_the_net_changes():
    at = _run(net="city")
    at.slider(key="mc_step").set_value(5)
    at.slider(key="point_slider").set_value(1)
    at.run()
    assert at.slider(key="mc_step").value == 5 and at.slider(key="point_slider").value == 1
    at.session_state["conflict_slider"] = 0.5
    at.run()
    assert not at.exception and at.slider(key="mc_step").value == at.slider(key="mc_step").max
    assert at.slider(key="point_slider").value == (int(at.slider(key="point_slider").max) + 1) // 2                  # F // 2 bei F Punkten


def test_every_step_of_the_small_net_renders():
    at = _run()
    for k in range(0, int(at.slider(key="mc_step").max) + 1):
        at.slider(key="mc_step").set_value(k)
        at.run()
        assert not at.exception, k


def test_play_renders_several_frames_without_duplicate_keys():
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    for setup in (lambda a: None, lambda a: a.session_state.__setitem__("net_select", "city"), lambda a: a.session_state.__setitem__("net_select", "random"), lambda a: a.session_state.__setitem__("net_select", "chain")):
        at = _run(setup)
        _play(at)
        assert not at.exception, [e.value for e in at.exception]


def test_point_and_hull_choices_change_the_shown_route():
    at = _run(net="city")
    at.slider(key="point_slider").set_value(0)
    at.run()
    first = [m.value for m in at.markdown if m.value.startswith("**Gewählte Route:**")]
    at.session_state["mode_select"] = "weighted"
    at.run()
    assert not at.exception
    at.slider(key="point_slider").set_value(at.slider(key="point_slider").max)
    at.run()
    last = [m.value for m in at.markdown if m.value.startswith("**Gewählte Route:**")]
    assert first and last and first != last


def test_permalink_parameters_select_the_net_and_are_clamped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "chain"
    at.query_params["links"] = "999"
    at.query_params["mode"] = "astar"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == "chain"
    assert at.slider(key="links_slider").value == C.LINKS_MAX and at.selectbox(key="mode_select").value == C.DEFAULT_MODE


def test_unknown_net_in_the_permalink_falls_back_to_the_default():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET


def test_experiments_run_on_demand():
    at = _run()
    assert not any("Mittel über 5 Netze" in c.value for c in at.caption)
    for key in ("front_start", "missed_start", "chain_start", "budget_start", "astar_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, (key, [e.value for e in at.exception])
    text = " ".join(c.value for c in at.caption)
    for needle in ("Die Front wächst mit der Netzgröße und mit der Gegenläufigkeit", "findet nur die Ecken der Konvexhülle", "die Front verdoppelt sich mit jedem Glied", "sparen in den Stadtnetzen nichts", "die Ersparnis **schrumpft mit der Größe**"):
        assert needle in text, needle


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    # Karte und Front stehen in der Play-Schleife: ihre Schlüssel tragen den Schritt
    assert sorted(keys) == sorted(["net_chart", "front_chart", "front_size_chart", "front_corr_chart", "missed_chart", "chain_chart", "budget_chart", "astar_chart"]), keys
    assert sum('key=f"' in c for c in calls) == 2 and all('_{current}"' in c for c in calls if 'key=f"' in c)
    viz = (ROOT / "mc_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("_base(fig") >= 5


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))
