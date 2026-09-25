"""Binäre Beschriftung mit Glattheitsstrafe als minimaler Schnitt (Greig, Porteous, Seheult 1989; Boykov und Jolly 2001), dazu einfachere Verfahren zum Vergleich.

Energie einer Beschriftung x (0/1 je Zelle): E(x) = Summe D_i(x_i) + lam * (Zahl der Nachbarpaare mit verschiedener Beschriftung). D_i(l) ist der Preis, Zelle i mit l zu beschriften
(hier die negative Log-Likelihood einer Normalverteilung, ganzzahlig): D_i(l) = (y_i - mu_l)^2 / (2 sigma^2), in Hundertsteln gerundet. Die Nachbarstrafe ist der "Potts"-Term.

Hilfsnetz: Quelle S, Senke T, je Zelle ein Knoten. Nach Abzug des kleineren Einzelpreises min(D_i(0), D_i(1)) bleibt je Zelle nur eine Terminalkante: Quelle -> i mit D_i(1) - D_i(0), wenn Label 1
teurer ist, sonst i -> Senke mit D_i(0) - D_i(1). Jedes Nachbarpaar bekommt zwei Kanten der Kapazität lam. Ein Schnitt trennt die Zellen in Quellseite (Label 0) und Senkenseite (Label 1) und kostet genau
E(x) minus die Summe der kleineren Einzelpreise: **minimaler Schnitt = beste Beschriftung**. Das geht nur, wenn die Nachbarstrafe **submodular** ist: E(0,0) + E(1,1) <= E(0,1) + E(1,0), das heißt
keine negative Kantenkapazität.
"""

from dataclasses import dataclass

import gc_bk
import gc_dinic
import gc_edmonds_karp as ek
import gc_scenario as sc

K_SRC, K_SNK, K_NBR = range(3)
ALGORITHMS = ("bk", "dinic", "bfs")
POTTS, REWARD = "potts", "reward"


def unary(grid):
    """D[l][i] in Hundertsteln: (y - mu_l)^2 / (2 sigma^2), ganzzahlig gerundet."""
    den = 2 * grid.sigma * grid.sigma
    return tuple(tuple(((y - sc.MU[l]) ** 2 * 100 + den // 2) // den for y in grid.obs) for l in (0, 1))


def pair_cost(lam, mode):
    """Kosten (E00, E01, E10, E11) eines Nachbarpaars; lam in Hundertsteln."""
    return (0, lam, lam, 0) if mode == POTTS else (0, -lam, -lam, 0)


def is_submodular(lam, mode=POTTS):
    e00, e01, e10, e11 = pair_cost(lam, mode)
    return e00 + e11 <= e01 + e10


def energy(grid, labels, lam, mode=POTTS):
    d = unary(grid)
    e = sum(d[labels[i]][i] for i in range(grid.n))
    pc = pair_cost(lam, mode)
    for i, j in grid.neighbours():
        e += pc[2 * labels[i] + labels[j]]
    return e


def errors(labels, truth):
    return sum(1 for a, b in zip(labels, truth) if a != b)


def build_net(grid, lam, mode=POTTS):
    """Hilfsnetz. Bei nicht submodularer Strafe werden negative Kapazitäten auf 0 gekappt (die naive Konstruktion: der Schnitt ist dann NICHT mehr die beste Beschriftung).
    Rückgabe (Netz, Summe der kleineren Einzelpreise)."""
    d = unary(grid)
    n = grid.n
    names = ("Quelle S", "Senke T") + tuple(f"Zelle ({i % grid.w}, {i // grid.w})" for i in range(n))
    labels = ("S", "T") + tuple("" for _ in range(n))
    pos = ((-1, grid.h // 2), (grid.w, grid.h // 2)) + tuple((i % grid.w, grid.h - 1 - i // grid.w) for i in range(n))
    arcs = []
    base = 0
    for i in range(n):
        d0, d1 = d[0][i], d[1][i]
        base += min(d0, d1)
        if d1 > d0:
            arcs.append((0, i + 2, d1 - d0, 0, K_SRC))
        elif d0 > d1:
            arcs.append((i + 2, 1, d0 - d1, 0, K_SNK))
    cap = max(0, lam if mode == POTTS else -lam)
    if cap:
        for i, j in grid.neighbours():
            arcs.append((i + 2, j + 2, cap, 0, K_NBR))
            arcs.append((j + 2, i + 2, cap, 0, K_NBR))
    return sc.Net(names, labels, pos, tuple(arcs), 0, 1, False), base


@dataclass(frozen=True)
class CutSolution:
    labels: tuple          # 0/1 je Zelle (Quellseite = 0)
    energy: int
    flow_value: int
    scanned: int
    n_paths: int
    net: object
    base: int
    algorithm: str
    submodular: bool


def graph_cut(grid, lam, mode=POTTS, algorithm="bk"):
    """Beschriftung per minimalem Schnitt. Nicht submodular (mode == REWARD, lam > 0): naive Konstruktion mit auf 0 gekappten Kapazitäten - keine Optimalitätsgarantie."""
    net, base = build_net(grid, lam, mode)
    if algorithm == "bk":
        res = gc_bk.bk_maxflow(net)
        reach, scanned, paths = res.reach, res.scanned_total, res.n_paths
    elif algorithm == "dinic":
        res = gc_dinic.dinic(net, keep_flows=False)
        reach, scanned, paths = res.reach, res.scanned_total, res.n_paths
    else:
        res = ek.max_flow(net, rule="bfs", keep_flows=False)
        reach, scanned, paths = res.reach, res.scanned_total, len(res.rounds)
    labels = tuple(0 if reach[i + 2] else 1 for i in range(grid.n))
    return CutSolution(labels, energy(grid, labels, lam, mode), res.value, scanned, paths, net, base, algorithm, is_submodular(lam, mode))


# --- einfachere Verfahren ----------------------------------------------------------------------------------------------------------

def threshold(grid):
    """Je Zelle allein: das billigere Label (Maximum-Likelihood); bei Gleichstand 0."""
    d = unary(grid)
    return tuple(1 if d[1][i] < d[0][i] else 0 for i in range(grid.n))


def mean_filter(grid, radius=1):
    """Mittelwert der Messung im Fenster (2 radius + 1)^2 (am Rand kleiner), dann Schwelle in der Mitte der Zonenmittelwerte."""
    out = []
    for y in range(grid.h):
        for x in range(grid.w):
            total = count = 0
            for yy in range(max(0, y - radius), min(grid.h, y + radius + 1)):
                for xx in range(max(0, x - radius), min(grid.w, x + radius + 1)):
                    total += grid.obs[yy * grid.w + xx]
                    count += 1
            out.append(1 if 2 * total > (sc.MU[0] + sc.MU[1]) * count else 0)
    return tuple(out)


def icm(grid, lam, mode=POTTS, start=None, max_sweeps=100):
    """Iterated Conditional Modes (Besag 1986): Zelle für Zelle das Label mit dem kleinsten lokalen Preis, bis sich nichts mehr ändert. Rückgabe (Beschriftung, Durchläufe)."""
    d = unary(grid)
    pc = pair_cost(lam, mode)
    nbrs = [[] for _ in range(grid.n)]
    for i, j in grid.neighbours():
        nbrs[i].append(j)
        nbrs[j].append(i)
    x = list(start if start is not None else threshold(grid))
    sweeps = 0
    while sweeps < max_sweeps:
        sweeps += 1
        changed = False
        for i in range(grid.n):
            best, best_cost = x[i], None
            for l in (0, 1):
                cost = d[l][i] + sum(pc[2 * l + x[j]] for j in nbrs[i])
                if best_cost is None or cost < best_cost:
                    best, best_cost = l, cost
            if best != x[i]:
                x[i] = best
                changed = True
        if not changed:
            break
    return tuple(x), sweeps


def brute_force(grid, lam, mode=POTTS):
    """Beste Beschriftung durch Aufzählen aller 2^n (nur für n <= 20; vektorisiert). Rückgabe (kleinste Energie, eine Beschriftung mit dieser Energie)."""
    import numpy as np
    assert grid.n <= 20
    d = np.array(unary(grid), dtype=np.int64)
    pc = pair_cost(lam, mode)
    codes = np.arange(1 << grid.n, dtype=np.int64)
    bits = (codes[:, None] >> np.arange(grid.n, dtype=np.int64)) & 1                      # Beschriftung k: Bit i = Label der Zelle i
    e = np.where(bits == 1, d[1], d[0]).sum(axis=1)
    for i, j in grid.neighbours():
        e = e + np.array(pc, dtype=np.int64)[2 * bits[:, i] + bits[:, j]]
    k = int(np.argmin(e))
    return int(e[k]), tuple(int(v) for v in bits[k])
