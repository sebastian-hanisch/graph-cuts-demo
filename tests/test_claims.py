"""Jede Zahl, die README, Hilfetexte und Beispieltexte nennen, ist hier belegt (Standardraster 16 x 16, Rauschen 80 %, lam = 1, feste Ziehungen ab Seed 100000).
Rauschen, Energien und durchsuchte Kanten sind ganzzahlig und deterministisch (eigener Zufallsstrom); Mittelwerte über Ziehungen werden mit Bändern geprüft."""

import pytest

import gc_constants as C
import gc_cut as cu
import gc_evaluation as ev
import gc_scenario as sc

P = ev.DEFAULT_PARAMS


def _analyse(name):
    p = C.PRESETS[name]
    return ev.analyse(ev.Params(p["shape"], p["n"], p["sigma"], p["lam"], p["mode"], p["algorithm"], p["seed"]))


def test_the_cut_is_exact_and_simple_methods_are_not():
    """40 4 x 4-Raster (Rauschen 80 %, lam = 0,5): der Schnitt erreicht mit allen drei Flussverfahren in 40 von 40 die kleinste Energie aller 65 536 Beschriftungen; ICM bleibt in 17 von 40 darüber;
    bei nicht submodularer Strafe (Unterschiede belohnen) liegt die naive Konstruktion in 38 von 40 über dem Minimum."""
    assert ev.small_check() == {"n": 40, "exact": 40, "icm_above": 17, "naive_above": 38}


def test_default_preset_numbers():
    """Rechteck, Seed 7: falsch beschriftet - Schwelle 68, Mittelwertfilter 22, ICM 13, Graph Cut 8 von 256; Energie Graph Cut 149,86, ICM 153,36; Boykov-Kolmogorov 225 Wege, 7 629 durchsuchte Kanten; ICM 4 Durchläufe."""
    a = _analyse("🗺️ Rechteck")
    assert a["errors"] == {"threshold": 68, "mean": 22, "icm": 13, "cut": 8}
    assert a["energies"]["cut"] == 14986 and a["energies"]["icm"] == 15336 and a["icm_sweeps"] == 4
    assert (a["cut"].n_paths, a["cut"].scanned) == (225, 7629)


def test_other_presets():
    """L-Form 61 / 14 / 12 / 5; Inseln (24 x 24, lam 1,5) 161 / 39 / 68 / 30 von 576, die 30 sind genau die vier Inseln (1 + 4 + 9 + 16); Streifen 59 / 45 / 51 / 53, Energie Graph Cut 215,31 gegen 233,59 bei ICM;
    starkes Rauschen 84 / 45 / 43 / 64; zu stark geglättet (lam 3): Schwelle 68, Mittelwertfilter 22, ICM 12, Graph Cut 64 = Fläche der Zone, alles Zone 0."""
    ell = _analyse("🔷 L-Form")
    assert ell["errors"] == {"threshold": 61, "mean": 14, "icm": 12, "cut": 5}
    isl = _analyse("🏝️ Kleine Inseln")
    assert isl["errors"] == {"threshold": 161, "mean": 39, "icm": 68, "cut": 30} and isl["grid"].n == 576 and sum(isl["grid"].truth) - sum(isl["labels"]["cut"]) == 30 == 1 + 4 + 9 + 16
    assert all(isl["labels"]["cut"][i] == isl["grid"].truth[i] for i in range(576) if isl["grid"].truth[i] == 1 and i % 24 < 12)
    stripes = _analyse("〰️ Dünne Streifen")
    assert stripes["errors"] == {"threshold": 59, "mean": 45, "icm": 51, "cut": 53} and (stripes["energies"]["cut"], stripes["energies"]["icm"]) == (21531, 23359)
    noisy = _analyse("🌫️ Starkes Rauschen")
    assert noisy["errors"] == {"threshold": 84, "mean": 45, "icm": 43, "cut": 64}
    strong = _analyse("🧊 Zu stark geglättet")
    assert strong["errors"] == {"threshold": 68, "mean": 22, "icm": 12, "cut": 64} and set(strong["labels"]["cut"]) == {0} and sum(strong["grid"].truth) == 64


def test_teaching_and_non_submodular_presets():
    """Lehrraster (4 x 4, Seed 8, Rauschen 60 %): Schwelle 3, Mittelwertfilter 5, ICM 2, Graph Cut 0 falsch; Energie 16,25 = Minimum aller 65 536 Beschriftungen, ICM bleibt bei 16,45.
    Nicht submodular (Seed 3): naive Energie -7,25 = Schwelle, Minimum -17,99 (ICM findet es)."""
    t = _analyse("🎓 Lehrraster")
    assert t["errors"] == {"threshold": 3, "mean": 5, "icm": 2, "cut": 0} and t["energies"]["cut"] == 1625 == cu.brute_force(t["grid"], t["lam"])[0] and t["energies"]["icm"] == 1645
    n = _analyse("🚫 Nicht submodular")
    assert n["energies"]["cut"] == -725 == n["energies"]["threshold"] and cu.brute_force(n["grid"], n["lam"], cu.REWARD)[0] == -1799 == n["energies"]["icm"]


@pytest.fixture(scope="module")
def dist():
    return ev.distribution(P)


def test_distribution_at_default_settings(dist):
    """40 feste Ziehungen (Rechteck 16 x 16, Rauschen 80 %, lam 1): falsch beschriftet im Mittel Schwelle 26,2 %, Mittelwertfilter 9,2 %, ICM 7,7 %, Graph Cut 3,8 %; der Graph Cut ist in 38 von 40
    besser als Schwelle und Mittelwertfilter; ICM liegt in allen 40 über der minimalen Energie, im Mittel 5,0 %."""
    assert dist["n"] == 40
    assert dist["threshold"] == pytest.approx(0.262, abs=0.005) and dist["mean"] == pytest.approx(0.092, abs=0.005) and dist["icm"] == pytest.approx(0.077, abs=0.005) and dist["cut"] == pytest.approx(0.038, abs=0.004)
    assert dist["cut_best"] == 38 and dist["icm_above"] == 40 and dist["icm_gap"] == pytest.approx(0.050, abs=0.005)


def test_noise_help_numbers():
    """Die Schwelle je Zelle beschriftet bei Rauschen 80 % etwa 26 %, bei 130 % etwa 35 % falsch (Standardraster)."""
    assert ev.distribution(P._replace(sigma=80))["threshold"] == pytest.approx(0.26, abs=0.01) and ev.distribution(P._replace(sigma=130))["threshold"] == pytest.approx(0.35, abs=0.01)


def test_other_shapes():
    """Falsch beschriftet (Schwelle / Mittelwertfilter / ICM / Graph Cut): L-Form 26,4 / 7,9 / 6,5 / 2,1 %; Scheibe 26,5 / 9,4 / 9,3 / 5,8 %; Inseln 26,6 / 13,3 / 11,3 / 9,4 %;
    dünne Streifen 26,8 / 24,5 / 23,2 / 33,5 % - dort ist der Graph Cut in 0 von 40 Ziehungen besser als beide Filter."""
    expected = {"ell": (0.264, 0.079, 0.065, 0.021), "disc": (0.265, 0.094, 0.093, 0.058), "islands": (0.266, 0.133, 0.113, 0.094), "stripes": (0.268, 0.245, 0.232, 0.335)}
    for shape, exp in expected.items():
        d = ev.distribution(P._replace(shape=shape))
        assert tuple(d[k] for k in ev.METHODS) == pytest.approx(exp, abs=0.006), shape
    assert ev.distribution(P._replace(shape="stripes"))["cut_best"] == 0


def test_lambda_curve_has_a_narrow_optimum():
    """Rechteck, 10 Ziehungen: Fehler des Graph Cut bei lam 0 26,2 % (= Schwelle), 0,5: 8,7 %, 1,0: 4,7 %, 1,5: 11,4 %, 2,0: 22,8 %, ab 2,5 25 % (ganzes Raster eine Zone); bestes lam 1,0; Mittelwertfilter 10,4 %;
    der Graph Cut schlägt den Mittelwertfilter für lam von 0,5 bis 1,4; die minimale Energie steigt mit lam."""
    c = ev.lambda_curve(P)
    at = dict(zip(c["lams"], c["cut"]))
    assert at[0.0] == pytest.approx(0.262, abs=0.005) and at[0.5] == pytest.approx(0.087, abs=0.005) and at[1.0] == pytest.approx(0.047, abs=0.004)
    assert at[1.5] == pytest.approx(0.114, abs=0.006) and at[2.0] == pytest.approx(0.228, abs=0.01) and at[2.5] == pytest.approx(0.25, abs=0.005) and at[3.0] == pytest.approx(0.25, abs=0.005)
    assert c["best_lam"] == 1.0 and c["mean"] == pytest.approx(0.1035, abs=0.005) and c["threshold"] == pytest.approx(0.262, abs=0.005) and c["window"] == (0.5, 1.4)
    assert c["energy"] == sorted(c["energy"])


def test_the_best_lambda_falls_with_noise():
    """Bestes lam je Rauschen (Orakel, 10 Ziehungen): 30 %: 2,4; 50 %: 3,0 (am Rand des Rasters); 70 %: 1,4; 90 %: 1,0; 110 %: 0,7; 130 %: 0,5. Fehler bei bestem lam 0,2 / 1,9 / 3,7 / 5,8 / 7,9 / 12,1 %;
    bei lam = 1 sind es 0,9 / 3,0 / 4,3 / 5,8 / 16,5 / 22,9 % gegen den Mittelwertfilter 2,0 / 4,4 / 8,6 / 12,4 / 15,8 / 19,2 %."""
    rows = {r["sigma"]: r for r in ev.sigma_series(P)}
    assert [rows[s]["best_lam"] for s in (30, 50, 70, 90, 110, 130)] == [2.4, 3.0, 1.4, 1.0, 0.7, 0.5]
    assert [round(100 * rows[s]["best_err"], 1) for s in (30, 50, 70, 90, 110, 130)] == pytest.approx([0.2, 1.9, 3.7, 5.8, 7.9, 12.1], abs=0.1)
    assert [round(100 * rows[s]["err_lam1"], 1) for s in (30, 50, 70, 90, 110, 130)] == pytest.approx([0.9, 3.0, 4.3, 5.8, 16.5, 22.9], abs=0.1)
    assert [round(100 * rows[s]["mean"], 1) for s in (30, 50, 70, 90, 110, 130)] == pytest.approx([2.0, 4.4, 8.6, 12.4, 15.8, 19.2], abs=0.1)
    assert rows[110]["err_lam1"] > rows[110]["mean"] and rows[130]["err_lam1"] > rows[130]["mean"] and rows[90]["err_lam1"] < rows[90]["mean"]


def test_small_islands_disappear_first():
    """40 x 20-Karte, Rauschen 60 %, 20 Ziehungen, Inseln der Kantenlänge 1 bis 6. lam 0: 18 / 19 / 19 / 20 / 20 / 20 wiedergefunden; lam 1: 1 / 4 / 11 / 20 / 20 / 20; lam 1,5: 0 / 0 / 3 / 15 / 17 / 18; lam 2: 0 / 0 / 0 / 5 / 9 / 12."""
    r = ev.islands(60)
    row = lambda lam: [r["found"][lam][s] for s in r["sizes"]]
    assert r["n"] == 20 and row(0.0) == [18, 19, 19, 20, 20, 20] and row(1.0) == [1, 4, 11, 20, 20, 20] and row(1.5) == [0, 0, 3, 15, 17, 18] and row(2.0) == [0, 0, 0, 5, 9, 12]
    assert row(0.6) == [8, 18, 17, 20, 20, 20]


def test_boykov_kolmogorov_searches_least():
    """Durchsuchte Kanten (Mittel über 5 Ziehungen, Scheibe, Rauschen 80 %, lam 1): 8 x 8: BK 2 912, Dinic 7 872, Edmonds-Karp 27 788; 16 x 16: 8 446 / 39 139 / 360 773 (Dinic 4,6-mal, Edmonds-Karp 43-mal mehr als BK);
    48 x 48: 59 433 / 327 392 / 18 778 500 (5,5-mal bzw. 316-mal mehr)."""
    rows = {r["n"]: r for r in ev.scaling()}
    assert (rows[8]["bk"], rows[8]["dinic"], rows[8]["bfs"]) == pytest.approx((2912, 7872, 27788), rel=0.02)
    assert (rows[16]["bk"], rows[16]["dinic"], rows[16]["bfs"]) == pytest.approx((8446, 39139, 360773), rel=0.02)
    assert rows[16]["dinic"] / rows[16]["bk"] == pytest.approx(4.6, abs=0.2) and rows[16]["bfs"] / rows[16]["bk"] == pytest.approx(43, abs=2)
    assert (rows[48]["bk"], rows[48]["dinic"], rows[48]["bfs"]) == pytest.approx((59433, 327392, 18778500), rel=0.02)
    assert rows[48]["dinic"] / rows[48]["bk"] == pytest.approx(5.5, abs=0.3) and rows[48]["bfs"] / rows[48]["bk"] == pytest.approx(316, abs=15)
    assert all(r["bk"] < r["dinic"] < r["bfs"] for r in rows.values()) and rows[8]["nodes"] == 66 and rows[48]["nodes"] == 2306
