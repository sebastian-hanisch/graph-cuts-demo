"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, alle Flussverfahren, Randgrößen, Permalink, Experimente auf Abruf, Schlüssel und Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gc_constants as C
from gc_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

EXPECTED = {
    "🗺️ Rechteck": "Der Graph Cut beschriftet 8 von 256 Zellen falsch",
    "🔷 L-Form": "Der Graph Cut beschriftet 5 von 256 Zellen falsch",
    "🏝️ Kleine Inseln": "Der Graph Cut beschriftet 30 von 576 Zellen falsch",
    "〰️ Dünne Streifen": "beschriftet **mehr** Zellen falsch (53)",
    "🌫️ Starkes Rauschen": "beschriftet **mehr** Zellen falsch (64)",
    "🧊 Zu stark geglättet": "beschriftet **mehr** Zellen falsch (64)",
    "🚫 Nicht submodular": "nicht submodular",
    "🎓 Lehrraster": "Der Graph Cut beschriftet 0 von 16 Zellen falsch",
}


def _run(setup=None, timeout=300):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
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


def _texts(at):
    return [e.value for e in list(at.success) + list(at.warning) + list(at.info) + list(at.error)]


def _has(at, part):
    return any(part in t for t in _texts(at))


def test_default_renders_without_exception():
    at = _run()
    assert _has(at, EXPECTED["🗺️ Rechteck"]) and not at.error
    assert [m.value for m in at.metric][:4] == ["26,6 %", "8,6 %", "5,1 %", "3,1 %"]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert _has(at, EXPECTED[name]), _texts(at)


@pytest.mark.parametrize("algorithm", list(C.ALGORITHMS))
def test_every_flow_method_renders(algorithm):
    at = _run(lambda a: a.session_state.__setitem__("algorithm_radio", algorithm))
    assert _has(at, EXPECTED["🗺️ Rechteck"]) and not at.error


@pytest.mark.parametrize("shape", list(C.SHAPES))
def test_every_shape_renders(shape):
    at = _run(lambda a: a.session_state.__setitem__("shape_select", shape))
    assert not at.error and len(at.metric) >= 8


def test_extreme_sizes_render():
    for vals in ((("n_slider", C.N_MIN), ("sigma_slider", C.SIGMA_MIN), ("lam_slider", C.LAM_MIN)), (("n_slider", C.N_MAX), ("sigma_slider", C.SIGMA_MAX), ("lam_slider", C.LAM_MAX))):
        def setup(at, vals=vals):
            for key, value in vals:
                at.session_state[key] = value
        at = _run(setup)
        assert not at.error and at.metric


def test_the_help_network_appears_only_for_small_rasters():
    small = _run(lambda a: a.session_state.__setitem__("n_slider", C.HELP_MAX_N))
    big = _run(lambda a: a.session_state.__setitem__("n_slider", C.HELP_MAX_N + 1))
    assert any("Hilfsnetz dieses Rasters" in m.value for m in small.markdown) and not any("Hilfsnetz dieses Rasters" in m.value for m in big.markdown)


def test_lambda_zero_makes_the_cut_equal_the_threshold():
    at = _run(lambda a: a.session_state.__setitem__("lam_slider", 0.0))
    values = [m.value for m in at.metric][:4]
    assert values[0] == values[3]


def test_permalink_settings_are_loaded_and_clamped():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.query_params["shape"] = "disc"
    at.query_params["n"] = "99"
    at.query_params["sigma"] = "77"
    at.query_params["lam"] = "1.26"
    at.query_params["algorithm"] = "dinic"
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox(key="shape_select").value == "disc" and at.sidebar.slider(key="n_slider").value == C.N_MAX
    assert at.sidebar.slider(key="sigma_slider").value == 80 and at.sidebar.slider(key="lam_slider").value == 1.3 and at.sidebar.radio(key="algorithm_radio").value == "dinic"


def test_experiments_run_on_demand():
    at = _run()
    for key in ("curve_start", "sigma_start", "islands_start", "scaling_start", "check_start"):
        next(b for b in at.button if b.key == key).click().run()
        assert not at.exception, key
    assert any(m.label == "Schnitt = kleinste Energie" and m.value == "40 von 40" for m in at.metric)
    assert any(c.value.startswith("Bestes λ (Orakel") for c in at.caption)


def test_source_has_explicit_chart_keys_and_locked_axes():
    app = APP.read_text(encoding="utf-8")
    assert all(re.search(r"plotly_chart\(.*key=", line) for line in app.splitlines() if "st.plotly_chart(" in line)
    viz = (ROOT / "gc_visualization.py").read_text(encoding="utf-8")
    assert viz.count("return _base(fig") + viz.count("return lock_axes(fig)") >= 6 and "def lock_axes" in viz
