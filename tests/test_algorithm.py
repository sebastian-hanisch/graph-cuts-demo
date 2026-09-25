"""Kern gegen Handrechnung und Gegenproben: Energie, Hilfsnetz, Schnitt gegen alle 2^n Beschriftungen, Schwelle, Mittelwertfilter, ICM, Submodularität, Monotonie in lam."""

import pytest

import gc_constants as C
import gc_cut as cu
import gc_evaluation as ev
import gc_scenario as sc


def _grid(w, h, truth, obs, sigma=500):
    return sc.Grid(w, h, tuple(truth), tuple(obs), sigma, "custom")


def test_unary_energy_by_hand():
    """Messwert 500, sigma 500: (500 - 0)^2 / (2 * 500^2) = 0,5 -> 50 Hundertstel für beide Zonen; Messwert 1000: Zone 0 kostet 2,0 (200), Zone 1 kostet 0."""
    g = _grid(2, 1, (0, 1), (500, 1000))
    d = cu.unary(g)
    assert d[0][0] == d[1][0] == 50 and d[0][1] == 200 and d[1][1] == 0


def test_energy_of_a_labeling_by_hand():
    """2 x 1, Messwerte 1000 und 0, lam = 0,7 (70): Beschriftung (1, 0) kostet 0 + 0 + 70 (die Labels sind verschieden), (1, 1) kostet 0 + 200 + 0, (0, 0) kostet 200 + 0 + 0."""
    g = _grid(2, 1, (1, 0), (1000, 0))
    assert cu.energy(g, (1, 0), 70) == 0 + 0 + 70
    assert cu.energy(g, (1, 1), 70) == 0 + 200 + 0
    assert cu.energy(g, (0, 0), 70) == 200 + 0 + 0


def test_a_single_pair_by_hand_the_cut_smooths_when_lambda_is_large():
    """Zwei Nachbarn, Messwerte 1000 und 400: allein wählt die Schwelle (1, 0). Mit lam = 0,5 (50) ist ein Unterschied teurer als der Fehler der zweiten Zelle -> beide 1."""
    g = _grid(2, 1, (1, 0), (1000, 400))
    assert cu.threshold(g) == (1, 0)
    weak = cu.graph_cut(g, 5)
    strong = cu.graph_cut(g, 500)
    assert weak.labels == (1, 0) and strong.labels in ((1, 1), (0, 0))


@pytest.mark.parametrize("seed", C.SWEEP_SEEDS)
@pytest.mark.parametrize("lam", (20, 100))
def test_cut_energy_equals_the_minimum_of_all_labelings_for_every_flow_method(seed, lam):
    g = sc.make_grid("rect", 4, 4, 80, seed)
    opt = cu.brute_force(g, lam)[0]
    for algo in cu.ALGORITHMS:
        assert cu.graph_cut(g, lam, algorithm=algo).energy == opt


@pytest.mark.parametrize("seed", C.SWEEP_SEEDS[:10])
def test_cut_energy_is_flow_plus_the_smaller_data_prices(seed):
    g = sc.make_grid("disc", 8, 8, 80, seed)
    for algo in cu.ALGORITHMS:
        s = cu.graph_cut(g, 100, algorithm=algo)
        assert s.energy == s.flow_value + s.base


def test_lambda_zero_is_the_threshold():
    for seed in C.SWEEP_SEEDS[:10]:
        g = sc.make_grid("ell", 12, 12, 80, seed)
        assert cu.energy(g, cu.graph_cut(g, 0).labels, 0) == cu.energy(g, cu.threshold(g), 0)


def test_mean_filter_by_hand():
    """Konstantes Raster: der Mittelwertfilter ändert nichts und schwellt in der Mitte der Zonenmittelwerte (500)."""
    g = _grid(3, 3, (1,) * 9, (900,) * 9)
    assert cu.mean_filter(g) == (1,) * 9
    g = _grid(3, 3, (0,) * 9, (100,) * 9)
    assert cu.mean_filter(g) == (0,) * 9
    g = _grid(3, 3, (0,) * 9, (100, 100, 100, 100, 900, 100, 100, 100, 100))
    assert cu.mean_filter(g) == (0,) * 9        # ein einzelner Ausreißer wird gemittelt


def test_icm_never_increases_the_energy_and_stops_in_a_local_minimum():
    for seed in C.SWEEP_SEEDS[:10]:
        g = sc.make_grid("rect", 10, 10, 90, seed)
        start = cu.threshold(g)
        x, sweeps = cu.icm(g, 100, start=start)
        assert cu.energy(g, x, 100) <= cu.energy(g, start, 100) and sweeps >= 1
        y, _ = cu.icm(g, 100, start=x)
        assert y == x


def test_icm_stays_above_the_optimum_in_some_runs():
    above = sum(cu.energy(g, cu.icm(g, 50)[0], 50) > cu.brute_force(g, 50)[0] for g in (sc.make_grid("rect", 4, 4, 80, s) for s in C.SWEEP_SEEDS))
    assert above > 0


def test_submodularity_check_and_the_naive_construction_under_reward():
    assert cu.is_submodular(100, cu.POTTS) and cu.is_submodular(0, cu.REWARD) and not cu.is_submodular(100, cu.REWARD)
    g = sc.make_grid("rect", 4, 4, 80, 3)
    naive = cu.graph_cut(g, 100, cu.REWARD)
    assert not naive.submodular and naive.energy > cu.brute_force(g, 100, cu.REWARD)[0]
    assert all(c >= 0 for _, _, c, _, _ in naive.net.arcs)


def test_minimal_energy_does_not_decrease_with_lambda():
    """min_x E_lam(x) ist als Minimum wachsender Funktionen nicht fallend - die Energie kann lam nicht wählen."""
    g = sc.make_grid("rect", 12, 12, 80, 100000)
    energies = [cu.graph_cut(g, lam).energy for lam in range(0, 301, 20)]
    assert energies == sorted(energies)


def test_very_large_lambda_makes_a_single_zone():
    g = sc.make_grid("rect", 12, 12, 80, 100000)
    labels = cu.graph_cut(g, 100000).labels
    assert len(set(labels)) == 1


def test_isolated_graph_cut_labels_are_binary_and_complete():
    g = sc.make_grid("disc", 9, 7, 60, 100001)
    s = cu.graph_cut(g, 100)
    assert len(s.labels) == g.n == 63 and set(s.labels) <= {0, 1}
