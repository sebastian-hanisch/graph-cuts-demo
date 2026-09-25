"""Presets: vollständig, in den Grenzen, und jedes Beispiel zeigt, was sein Hilfetext behauptet."""

import pytest

import gc_constants as C
import gc_evaluation as ev
import gc_presets as P

KEYS = set(P.PRESET_KEYS)


def _params(p):
    return ev.Params(p["shape"], p["n"], p["sigma"], p["lam"], p["mode"], p["algorithm"], p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    assert all(C.PRESET_HELP[name].strip() for name in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_values_are_inside_the_bounds_and_on_the_step_grid(name):
    p = C.PRESETS[name]
    assert p["shape"] in C.SHAPES and p["mode"] in C.MODES and p["algorithm"] in C.ALGORITHMS
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        if spec.lo is not None:
            assert spec.lo <= p[key] <= spec.hi, (name, key)
    assert (p["sigma"] - C.SIGMA_MIN) % 10 == 0 and abs(p["lam"] * 10 - round(p["lam"] * 10)) < 1e-9


def test_setting_specs_have_room_to_move():
    """Ein Regler mit lo == hi würde Streamlit abstürzen lassen."""
    assert all(spec.lo < spec.hi for spec in P.SETTING_SPECS.values() if spec.lo is not None)


def test_presets_use_seeds_outside_the_distribution_set():
    for name, p in C.PRESETS.items():
        assert p["seed"] not in C.DIST_SEEDS, name


def test_defaults_equal_the_first_preset():
    p = C.PRESETS["🗺️ Rechteck"]
    assert (p["shape"], p["n"], p["sigma"], p["lam"], p["mode"], p["algorithm"], p["seed"]) == (
        C.DEFAULT_SHAPE, C.DEFAULT_N, C.DEFAULT_SIGMA, C.DEFAULT_LAM, C.DEFAULT_MODE, C.DEFAULT_ALGORITHM, C.DEFAULT_SEED)


def test_the_presets_show_both_good_and_bad_news():
    """Gut: Rechteck, L-Form, Lehrraster (Graph Cut klar am besten). Schlecht: Streifen, starkes Rauschen, zu starke Glättung (Graph Cut schlechter als ein einfacheres Verfahren)."""
    res = {name: ev.analyse(_params(p)) for name, p in C.PRESETS.items()}
    best_other = lambda r: min(r["errors"][k] for k in ("threshold", "mean", "icm"))
    for name in ("🗺️ Rechteck", "🔷 L-Form", "🎓 Lehrraster"):
        assert res[name]["errors"]["cut"] < best_other(res[name]), name
    for name in ("🌫️ Starkes Rauschen", "🧊 Zu stark geglättet"):
        assert res[name]["errors"]["cut"] > best_other(res[name]), name
    assert res["〰️ Dünne Streifen"]["errors"]["cut"] > res["〰️ Dünne Streifen"]["errors"]["mean"]
    assert all(r["energies"]["cut"] == min(r["energies"].values()) for n, r in res.items() if n != "🚫 Nicht submodular")


def test_the_non_submodular_preset_breaks_the_cut():
    p = C.PRESETS["🚫 Nicht submodular"]
    a = ev.analyse(_params(p))
    assert not a["cut"].submodular and a["energies"]["cut"] > min(a["energies"].values())
