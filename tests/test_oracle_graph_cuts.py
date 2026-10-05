"""Orakel-Regressionstest: Graph Cut gegen eigene Energie (exakte Bruchrechnung), Aufzählung aller 2^n Beschriftungen und networkx.

Beliebige kleine Raster (auch 1 x n, Gleichstände bei Messwert 500, Messwerte außerhalb der Zonen), Glättung 0 bis 4,0."""

import random
from fractions import Fraction
from itertools import product

import pytest

nx = pytest.importorskip("networkx")

import gc_cut as cu  # noqa: E402
import gc_scenario as sc  # noqa: E402


def _d(y, label, sigma):
    """Datenpreis in Hundertsteln, kaufmännisch gerundet, per Bruchrechnung statt Ganzzahl-Trick."""
    return int(Fraction((y - sc.MU[label]) ** 2 * 100, 2 * sigma * sigma) + Fraction(1, 2))


def _pairs(g):
    return [(y * g.w + x, y * g.w + x + step) for y in range(g.h) for x in range(g.w) for step, ok in ((1, x + 1 < g.w), (g.w, y + 1 < g.h)) if ok]


def _energy(g, x, lam, sign):
    return sum(_d(g.obs[i], x[i], g.sigma) for i in range(g.n)) + sign * lam * sum(1 for a, b in _pairs(g) if x[a] != x[b])


def _grid(rng):
    w, h = rng.randint(1, 4), rng.randint(1, 3)
    obs = tuple(rng.choice([0, 500, 1000, rng.randint(-1500, 2500)]) for _ in range(w * h))
    return sc.Grid(w, h, (0,) * (w * h), obs, rng.choice([100, 200, 500, 800, 1500]), "custom")


def _nx_cut_value(g, lam):
    G = nx.DiGraph()
    G.add_nodes_from(["S", "T"])
    base = 0
    for i in range(g.n):
        d0, d1 = _d(g.obs[i], 0, g.sigma), _d(g.obs[i], 1, g.sigma)
        base += min(d0, d1)
        if d1 > d0:
            G.add_edge("S", i, capacity=d1 - d0)
        elif d0 > d1:
            G.add_edge(i, "T", capacity=d0 - d1)
    if lam:
        for a, b in _pairs(g):
            G.add_edge(a, b, capacity=lam)
            G.add_edge(b, a, capacity=lam)
    return nx.maximum_flow_value(G, "S", "T") + base


def test_graph_cut_energy_is_the_minimum_of_all_labelings():
    rng = random.Random(11)
    for _ in range(40):
        g = _grid(rng)
        lam = rng.choice([0, 1, 50, 100, 150, 300, rng.randint(0, 400)])
        best = min(_energy(g, x, lam, 1) for x in product((0, 1), repeat=g.n))
        assert cu.unary(g)[0] == tuple(_d(y, 0, g.sigma) for y in g.obs)
        for algo in cu.ALGORITHMS:
            s = cu.graph_cut(g, lam, algorithm=algo)
            assert s.energy == _energy(g, s.labels, lam, 1) == best == _nx_cut_value(g, lam) == s.flow_value + s.base


def test_naive_cut_under_reward_never_beats_the_true_optimum_and_energy_function_agrees():
    rng = random.Random(12)
    for _ in range(30):
        g = _grid(rng)
        lam = rng.choice([1, 50, 100, 300])
        best = min(_energy(g, x, lam, -1) for x in product((0, 1), repeat=g.n))
        assert cu.brute_force(g, lam, cu.REWARD)[0] == best
        s = cu.graph_cut(g, lam, cu.REWARD)
        assert s.energy == _energy(g, s.labels, lam, -1) >= best


def test_threshold_and_mean_filter_against_exact_arithmetic():
    rng = random.Random(13)
    for _ in range(40):
        g = _grid(rng)
        assert cu.threshold(g) == tuple(1 if _d(o, 1, g.sigma) < _d(o, 0, g.sigma) else 0 for o in g.obs)
        want = []
        for y in range(g.h):
            for x in range(g.w):
                vals = [g.obs[yy * g.w + xx] for yy in range(max(0, y - 1), min(g.h, y + 2)) for xx in range(max(0, x - 1), min(g.w, x + 2))]
                want.append(1 if Fraction(sum(vals), len(vals)) > 500 else 0)
        assert cu.mean_filter(g) == tuple(want)
