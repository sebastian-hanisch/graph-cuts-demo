"""Szenario (Zonen, ganzzahliges Rauschen, Zufallsstrom) und Auswertung (Kennzahlen, Verteilungen, Kurven)."""

import pytest

import gc_constants as C
import gc_cut as cu
import gc_evaluation as ev
import gc_scenario as sc

P = ev.DEFAULT_PARAMS


def test_generation_is_deterministic_and_seed_dependent():
    a, b, c = sc.make_grid("rect", 8, 8, 80, 5), sc.make_grid("rect", 8, 8, 80, 5), sc.make_grid("rect", 8, 8, 80, 6)
    assert a == b and a.obs != c.obs and a.truth == c.truth


@pytest.mark.parametrize("shape", sc.SHAPES)
def test_truth_maps_are_two_zones_of_reasonable_size(shape):
    t = sc.truth_map(shape, 16, 16)
    assert len(t) == 256 and set(t) == {0, 1} and 30 <= sum(t) <= 200


def test_zone_shapes_by_hand():
    """Rechteck 16 x 16: Zeilen und Spalten 4 bis 11, also 64 Zellen; Streifen: Breite 2 (16 // 6), also 8 Streifenpaare -> Hälfte der Zellen; L-Form: Balken 5 breit."""
    assert sum(sc.truth_map("rect", 16, 16)) == 64
    assert sum(sc.truth_map("stripes", 16, 16)) == 128
    t = sc.truth_map("ell", 16, 16)
    assert t[0] == 1 and t[5] == 0 and t[15 * 16 + 15] == 1 and t[10 * 16 + 5] == 0


def test_noise_is_integer_zero_mean_and_unit_scale():
    rng = sc.SplitMix64(42)
    xs = [sc.noise(rng) for _ in range(4000)]
    assert all(isinstance(x, int) for x in xs)
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    assert abs(mean) < 60 and 0.9e6 < var < 1.1e6          # Standardabweichung etwa 1000 (Tausendstel)


def test_observations_follow_the_truth_and_sigma():
    g = sc.make_grid("rect", 16, 16, 40, 100000)
    ones = [g.obs[i] for i in range(256) if g.truth[i]]
    zeros = [g.obs[i] for i in range(256) if not g.truth[i]]
    assert abs(sum(ones) / len(ones) - 1000) < 120 and abs(sum(zeros) / len(zeros)) < 120


def test_neighbours_are_the_four_neighbourhood():
    g = sc.make_grid("rect", 4, 3, 80, 1)
    assert len(g.neighbours()) == 3 * 3 + 4 * 2 == 17 and (0, 1) in g.neighbours() and (0, 4) in g.neighbours()


def test_analyse_covers_all_methods_and_the_cut_is_exact_in_energy():
    a = ev.analyse(P)
    assert set(a["errors"]) == set(ev.METHODS) == set(a["energies"]) and a["energies"]["cut"] == min(a["energies"].values())
    assert a["cut"].energy == a["energies"]["cut"]


@pytest.mark.parametrize("algo", list(C.ALGORITHMS))
def test_every_flow_method_gives_the_same_labeling_energy(algo):
    assert ev.analyse(P._replace(algorithm=algo))["energies"]["cut"] == ev.analyse(P)["energies"]["cut"]


def test_distribution_shapes():
    d = ev.distribution(P)
    assert d["n"] == 40 and len(d["rows"]) == 40 and all(0 <= d[k] <= 1 for k in ev.METHODS) and 0 <= d["icm_above"] <= 40
    assert all(r["e_cut"] <= r["e_icm"] for r in d["rows"])


def test_lambda_curve_shape_and_energy_monotone():
    c = ev.lambda_curve(P, lams=(0.0, 0.5, 1.0, 2.0, 3.0), seeds=C.CURVE_SEEDS[:3])
    assert len(c["cut"]) == 5 and c["energy"] == sorted(c["energy"]) and c["best_lam"] in c["lams"]


def test_island_truth_fits_the_map():
    t, pos = ev.island_truth(*C.ISLAND_GRID)
    assert len(pos) == 6 and sum(t) == sum(s * s for _, _, s in pos) == 91


def test_scaling_rows_are_ordered_by_size():
    rows = ev.scaling(sizes=(6, 8, 10), seeds=C.SCALE_SEEDS[:2])
    assert [r["nodes"] for r in rows] == sorted(r["nodes"] for r in rows) and all(r["bk"] > 0 and r["dinic"] > 0 and r["bfs"] > 0 for r in rows)
