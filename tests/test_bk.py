"""Boykov-Kolmogorov gegen Dinic, Edmonds-Karp und Handrechnung: gleicher Flusswert, zulässiger Fluss, Erhaltung, Schnitt = Fluss, Baum = erreichbare Menge."""

import pytest

import gc_bk as bk
import gc_dinic as dn
import gc_edmonds_karp as ek
import gc_scenario as sc


def _net(n, arcs):
    names = tuple(str(i) for i in range(n))
    return sc.Net(names, names, tuple((0, 0) for _ in names), tuple((u, v, c, 0, 0) for u, v, c in arcs), 0, 1, False)


def _random(n, m, seed, maxcap=20):
    rng = sc.SplitMix64(seed)
    arcs = []
    for _ in range(m):
        u, v = rng.below(n), rng.below(n)
        if u != v:
            arcs.append((u, v, 1 + rng.below(maxcap)))
    return _net(n, arcs)


def _grid(w, h, seed):
    """Gitternetz mit Terminalkanten wie bei einem Graph Cut, aber zufälligen Kapazitäten."""
    rng = sc.SplitMix64(seed)
    arcs = []
    for i in range(w * h):
        if rng.below(2):
            arcs.append((0, i + 2, 1 + rng.below(30)))
        else:
            arcs.append((i + 2, 1, 1 + rng.below(30)))
    for y in range(h):
        for x in range(w):
            i = y * w + x
            for j in ((i + 1) if x + 1 < w else None, (i + w) if y + 1 < h else None):
                if j is not None:
                    c = 1 + rng.below(20)
                    arcs += [(i + 2, j + 2, c), (j + 2, i + 2, c)]
    return _net(w * h + 2, arcs)


def _check(net):
    a = bk.bk_maxflow(net)
    d = dn.dinic(net, keep_flows=False)
    e = ek.max_flow(net, keep_flows=False)
    assert a.value == d.value == e.value
    caps = [c for _, _, c, _, _ in net.arcs]
    assert all(0 <= f <= c for f, c in zip(a.flow, caps))
    bal = [0] * net.n
    for f, (u, v, _, _, _) in zip(a.flow, net.arcs):
        bal[u] -= f
        bal[v] += f
    assert all(bal[i] == 0 for i in range(net.n) if i not in (net.s, net.t)) and -bal[net.s] == a.value
    assert sum(c for (u, v, c, _, _) in net.arcs if a.reach[u] and not a.reach[v]) == a.value
    return a, d


def test_tiny_networks_by_hand():
    """S -> A (3), A -> T (2), S -> T (1): Fluss 3; zwei parallele Wege mit Engpass; kein Weg."""
    assert bk.bk_maxflow(_net(3, [(0, 2, 3), (2, 1, 2), (0, 1, 1)])).value == 3
    two = _net(4, [(0, 2, 5), (0, 3, 4), (2, 1, 3), (3, 1, 6), (2, 3, 2)])
    assert bk.bk_maxflow(two).value == 9        # Schnitt {S}: 5 + 4
    assert bk.bk_maxflow(_net(3, [(0, 2, 5)])).value == 0


@pytest.mark.parametrize("seed", range(120))
def test_random_networks_match_dinic_and_edmonds_karp(seed):
    n = 4 + seed % 20
    _check(_random(n, 3 * n, seed))


def _reachable(net, flow):
    """Von S im Restgraphen (Vorwärtskante mit Rest > 0, Rückkante mit Fluss > 0) erreichbare Knoten - unabhängig vom Verfahren."""
    seen, stack = {net.s}, [net.s]
    while stack:
        x = stack.pop()
        for (u, v, c, _, _), f in zip(net.arcs, flow):
            if u == x and f < c and v not in seen:
                seen.add(v)
                stack.append(v)
            if v == x and f > 0 and u not in seen:
                seen.add(u)
                stack.append(u)
    return seen


@pytest.mark.parametrize("seed", range(30))
def test_grid_networks_match_and_the_tree_is_the_reachable_set(seed):
    net = _grid(5 + seed % 4, 4 + seed % 3, seed)
    a, _ = _check(net)
    assert {v for v in range(net.n) if a.reach[v]} == _reachable(net, a.flow)


def test_bk_counts_less_search_than_dinic_and_edmonds_karp_on_a_grid():
    net = _grid(16, 16, 5)
    a = bk.bk_maxflow(net)
    d = dn.dinic(net, keep_flows=False)
    e = ek.max_flow(net, keep_flows=False)
    assert a.scanned_total < d.scanned_total < e.scanned_total and a.scanned_total == a.grow_scanned + a.adopt_scanned
