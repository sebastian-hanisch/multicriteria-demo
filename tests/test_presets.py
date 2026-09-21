"""Presets, Permalink-Angaben und Regler-Grenzen sind untereinander stimmig."""

import pytest

import mc_constants as C
import mc_presets as P
from mc_scenario import make_network


def test_every_preset_sets_every_control_within_bounds_and_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 4
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and p["net"] in C.NETS and p["mode"] in C.MODES
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
        make_network(p["net"], p["side"], p["conflict"], p["nodes"], p["degree"], p["corr"], p["links"], p["seed"])
        assert C.PRESET_HELP[name]
    assert [p["net"] for p in C.PRESETS.values()] == list(C.NETS)


def test_setting_specs_and_kept_keys_are_consistent():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) and set(P.KEPT) <= set(P.SETTING_SPECS)
    for spec in P.SETTING_SPECS.values():
        assert spec.lo is None or spec.lo < spec.hi                                 # kein Regler mit gleichen Grenzen (Streamlit bricht ab)
    assert len({s.url_param for s in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert "mode_select" in P.KEPT                                                   # die Anzeige ist nur bei einer Front mit mehr als einem Punkt sichtbar: ihr Wert bleibt erhalten


def test_defaults_lie_inside_the_bounds_and_the_largest_net_has_at_most_400_nodes():
    for lo, hi, d in ((C.SIDE_MIN, C.SIDE_MAX, C.DEFAULT_SIDE), (C.CONFLICT_MIN, C.CONFLICT_MAX, 0.5), (C.NODES_MIN, C.NODES_MAX, C.DEFAULT_NODES), (C.DEGREE_MIN, C.DEGREE_MAX, C.DEFAULT_DEGREE),
                      (C.CORR_MIN, C.CORR_MAX, 0.0), (C.LINKS_MIN, C.LINKS_MAX, C.DEFAULT_LINKS)):
        assert lo < d < hi
    assert C.CONFLICT_MIN <= C.DEFAULT_CONFLICT <= C.CONFLICT_MAX and C.CORR_MIN <= C.DEFAULT_CORR <= C.CORR_MAX
    assert C.DEFAULT_NET == "small" and C.DEFAULT_MODE in C.MODES and C.SIDE_MAX ** 2 <= 400 and C.NODES_MAX <= 400


def test_choice_casters_reject_unknown_values():
    for key, good, bad in (("net_select", "chain", "ring"), ("mode_select", "budget", "astar")):
        cast = P.SETTING_SPECS[key].caster
        assert cast(good) == good
        with pytest.raises(ValueError):
            cast(bad)
