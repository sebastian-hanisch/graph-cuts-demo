"""Auswertung: Fehlklassifikation gegen die bekannte Wahrheit, Energie, Verteilung über feste Rauschziehungen, Glättungskurve, Inseln, Rauschreihe, Aufwand nach Größe.
Alle Zufallsziehungen kommen aus festen Seeds (gc_constants), unabhängig vom Nutzer-Seed. Die Glättung `lam` steht in Einheiten (1,0 = 100 Hundertstel)."""

import statistics
from collections import namedtuple

import gc_constants as C
import gc_cut as cu
import gc_scenario as sc

Params = namedtuple("Params", "shape n sigma lam mode algorithm seed")
DEFAULT_PARAMS = Params(C.DEFAULT_SHAPE, C.DEFAULT_N, C.DEFAULT_SIGMA, C.DEFAULT_LAM, C.DEFAULT_MODE, C.DEFAULT_ALGORITHM, C.DEFAULT_SEED)

METHODS = ("threshold", "mean", "icm", "cut")
METHOD_LABELS = {"threshold": "Schwelle je Zelle", "mean": "Mittelwertfilter 3 × 3", "icm": "ICM (lokal)", "cut": "Graph Cut (minimaler Schnitt)"}


def lam_int(lam):
    return round(lam * 100)


def grid_of(params):
    return sc.make_grid(params.shape, params.n, params.n, params.sigma, params.seed)


def analyse(params):
    """Alle Verfahren auf der Ziehung der Einstellung."""
    g = grid_of(params)
    lam = lam_int(params.lam)
    cut = cu.graph_cut(g, lam, params.mode, params.algorithm)
    icm_labels, sweeps = cu.icm(g, lam, params.mode)
    labels = {"threshold": cu.threshold(g), "mean": cu.mean_filter(g), "icm": icm_labels, "cut": cut.labels}
    errs = {k: cu.errors(v, g.truth) for k, v in labels.items()}
    energies = {k: cu.energy(g, v, lam, params.mode) for k, v in labels.items()}
    return dict(grid=g, labels=labels, errors=errs, energies=energies, cut=cut, icm_sweeps=sweeps, lam=lam)


def distribution(params, seeds=C.SWEEP_SEEDS):
    """Fehlklassifikation (Anteil der Zellen) je Verfahren über feste Rauschziehungen; dazu wie oft ICM über der minimalen Energie bleibt."""
    lam = lam_int(params.lam)
    rows = []
    for seed in seeds:
        g = sc.make_grid(params.shape, params.n, params.n, params.sigma, seed)
        cut = cu.graph_cut(g, lam, params.mode)
        icm_labels, _ = cu.icm(g, lam, params.mode)
        e_icm = cu.energy(g, icm_labels, lam, params.mode)
        rows.append(dict(seed=seed, threshold=cu.errors(cu.threshold(g), g.truth) / g.n, mean=cu.errors(cu.mean_filter(g), g.truth) / g.n,
                         icm=cu.errors(icm_labels, g.truth) / g.n, cut=cu.errors(cut.labels, g.truth) / g.n, e_cut=cut.energy, e_icm=e_icm,
                         icm_above=e_icm > cut.energy, icm_gap=(e_icm - cut.energy) / cut.energy if cut.energy else 0.0,
                         cut_best=cu.errors(cut.labels, g.truth) < min(cu.errors(cu.threshold(g), g.truth), cu.errors(cu.mean_filter(g), g.truth))))
    out = {k: statistics.fmean(r[k] for r in rows) for k in METHODS}
    out.update(n=len(rows), icm_above=sum(1 for r in rows if r["icm_above"]), icm_gap=statistics.fmean(r["icm_gap"] for r in rows),
               cut_best=sum(1 for r in rows if r["cut_best"]), rows=rows)
    return out


def lambda_curve(params, lams=C.LAMBDAS, seeds=C.CURVE_SEEDS):
    """Mittlere Fehlklassifikation des Graph Cut je Glättung (über feste Ziehungen), dazu Schwelle und Mittelwertfilter als Bezug und die Energie des Optimums je Glättung."""
    grids = [sc.make_grid(params.shape, params.n, params.n, params.sigma, s) for s in seeds]
    cut_err, energy = [], []
    for lam in lams:
        sols = [cu.graph_cut(g, lam_int(lam)) for g in grids]
        cut_err.append(statistics.fmean(cu.errors(s.labels, g.truth) / g.n for s, g in zip(sols, grids)))
        energy.append(statistics.fmean(s.energy for s in sols))
    thr = statistics.fmean(cu.errors(cu.threshold(g), g.truth) / g.n for g in grids)
    mean = statistics.fmean(cu.errors(cu.mean_filter(g), g.truth) / g.n for g in grids)
    best = min(range(len(lams)), key=lambda k: cut_err[k])
    better = [lams[k] for k in range(len(lams)) if cut_err[k] < mean]
    return dict(lams=tuple(lams), cut=cut_err, energy=energy, threshold=thr, mean=mean, best_lam=lams[best], best_err=cut_err[best],
                window=(better[0], better[-1]) if better else None, n=len(grids))


def sigma_series(params, sigmas=C.SIGMAS, lams=C.LAMBDAS, seeds=C.CURVE_SEEDS):
    """Bestes lam (Orakel: kleinste mittlere Fehlklassifikation über die Ziehungen) je Rauschstärke, dazu Fehler bei lam = 1 und beim Mittelwertfilter."""
    rows = []
    for sigma in sigmas:
        c = lambda_curve(params._replace(sigma=sigma), lams, seeds)
        at_one = c["cut"][c["lams"].index(1.0)] if 1.0 in c["lams"] else None
        rows.append(dict(sigma=sigma, best_lam=c["best_lam"], best_err=c["best_err"], err_lam1=at_one, mean=c["mean"], threshold=c["threshold"], window=c["window"]))
    return rows


def island_truth(w, h, sizes=C.ISLAND_SIZES):
    """Wahrheit mit quadratischen Inseln der Kantenlängen `sizes` in einer 0-Zone. Rückgabe (Wahrheit, [(x0, y0, Kante)])."""
    t = [0] * (w * h)
    pos, x0 = [], 2
    for k, s in enumerate(sizes):
        y0 = 3 + (k % 2) * 9
        for y in range(y0, y0 + s):
            for x in range(x0, x0 + s):
                t[y * w + x] = 1
        pos.append((x0, y0, s))
        x0 += s + 3
    assert x0 <= w + 3 and all(y0 + s <= h for _, y0, s in pos)
    return tuple(t), pos


def islands(sigma, lams=C.ISLAND_LAMS, seeds=C.ISLAND_SEEDS):
    """Wie viele Ziehungen jede Insel wiederfinden (mindestens die Hälfte ihrer Zellen mit Label 1), je Glättung und Inselgröße."""
    w, h = C.ISLAND_GRID
    truth, pos = island_truth(w, h)
    grids = [sc.observe(truth, w, h, sigma, s) for s in seeds]
    found = {}
    for lam in lams:
        counts = {s: 0 for _, _, s in pos}
        for g in grids:
            lab = cu.graph_cut(g, lam_int(lam)).labels
            for x0, y0, s in pos:
                cells = [lab[y * w + x] for y in range(y0, y0 + s) for x in range(x0, x0 + s)]
                counts[s] += 2 * sum(cells) >= len(cells)
        found[lam] = counts
    return dict(found=found, n=len(grids), sizes=tuple(s for _, _, s in pos), sigma=sigma)


def scaling(sizes=C.SCALE_SIZES, seeds=C.SCALE_SEEDS, sigma=C.DEFAULT_SIGMA, lam=C.DEFAULT_LAM):
    """Durchsuchte Kanten von Boykov-Kolmogorov, Dinic und Edmonds-Karp im Hilfsnetz je Rastergröße (Mittel über feste Ziehungen); alle drei finden dieselbe minimale Energie."""
    rows = []
    for n in sizes:
        acc = {a: [] for a in cu.ALGORITHMS}
        paths = {a: [] for a in cu.ALGORITHMS}
        for seed in seeds:
            g = sc.make_grid("disc", n, n, sigma, seed)
            energies = set()
            for a in cu.ALGORITHMS:
                s = cu.graph_cut(g, lam_int(lam), algorithm=a)
                acc[a].append(s.scanned)
                paths[a].append(s.n_paths)
                energies.add(s.energy)
            assert len(energies) == 1
        net = cu.build_net(g, lam_int(lam))[0]
        rows.append(dict(n=n, nodes=net.n, arcs=net.m, **{a: statistics.fmean(acc[a]) for a in cu.ALGORITHMS}, **{"paths_" + a: statistics.fmean(paths[a]) for a in cu.ALGORITHMS}))
    return rows


def small_check(seeds=C.SWEEP_SEEDS, n=4, sigma=80, lam=0.5):
    """4 × 4 Raster: der Schnitt (alle drei Flussverfahren) gegen alle 65 536 Beschriftungen; ICM und die naive Konstruktion bei nicht submodularer Strafe zum Vergleich."""
    exact = icm_above = naive_above = 0
    for seed in seeds:
        g = sc.make_grid("rect", n, n, sigma, seed)
        l = lam_int(lam)
        opt = cu.brute_force(g, l)[0]
        exact += all(cu.graph_cut(g, l, algorithm=a).energy == opt for a in cu.ALGORITHMS)
        icm_labels, _ = cu.icm(g, l)
        icm_above += cu.energy(g, icm_labels, l) > opt
        naive_above += cu.graph_cut(g, l, cu.REWARD).energy > cu.brute_force(g, l, cu.REWARD)[0]
    return dict(n=len(seeds), exact=exact, icm_above=icm_above, naive_above=naive_above)
