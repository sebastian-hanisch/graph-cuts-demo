"""Konstanten, Regler-Grenzen, Presets und feste Seed-Mengen der Demo "Graph Cuts: glatte Zonen aus verrauschten Messwerten"."""

# --- Regler ---------------------------------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 4, 32, 16              # Kantenlänge des quadratischen Rasters (Zellen)
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 20, 150, 80  # Rauschen in Prozent des Zonenabstands, Schritt 10
LAM_MIN, LAM_MAX, DEFAULT_LAM = 0.0, 3.0, 1.0      # Glättung, Schritt 0,1
DEFAULT_SEED = 7
SEED_MAX = 2_000_000_000

SHAPES = {
    "rect": "Rechteck",
    "ell": "L-Form",
    "disc": "Scheibe",
    "stripes": "Streifen (dünne Zonen)",
    "islands": "Große Zone mit kleinen Inseln",
}
DEFAULT_SHAPE = "rect"
MODES = {"potts": "Glätten (Potts, submodular)", "reward": "Unterschiede belohnen (nicht submodular)"}
DEFAULT_MODE = "potts"
ALGORITHMS = {"bk": "Boykov–Kolmogorov", "dinic": "Dinic", "bfs": "Edmonds-Karp"}
DEFAULT_ALGORITHM = "bk"
HELP_MAX_N = 6            # bis zu dieser Kantenlänge wird das Hilfsnetz gezeichnet

# --- feste Seed-Mengen (dieselben wie in den Flussdemos; unabhängig vom Nutzer-Seed) ---------------------------------------------
DIST_SEEDS = tuple(range(100000, 100100))
SWEEP_SEEDS = DIST_SEEDS[:40]
CURVE_SEEDS = DIST_SEEDS[:10]
ISLAND_SEEDS = DIST_SEEDS[:20]
LAMBDAS = tuple(k / 10 for k in range(0, 31))          # Glättungswerte der Kurven
SIGMAS = (30, 50, 70, 90, 110, 130)
SCALE_SIZES = (8, 12, 16, 24, 32, 48)                  # Kantenlängen für den Aufwand
SCALE_SEEDS = DIST_SEEDS[:5]
ISLAND_SIZES = (1, 2, 3, 4, 5, 6)
ISLAND_GRID = (40, 20)
ISLAND_LAMS = (0.0, 0.3, 0.6, 1.0, 1.5, 2.0)

COLORS = {"zone0": "#dbe9f6", "zone1": "#1f77b4", "wrong": "#d62728", "right": "#2ca02c", "faint": "rgba(150,150,150,0.45)", "node": "#111111", "path": "#ff7f0e"}

# --- Presets -----------------------------------------------------------------------------------------------------------------
_BASE = dict(shape=DEFAULT_SHAPE, n=DEFAULT_N, sigma=DEFAULT_SIGMA, lam=DEFAULT_LAM, mode=DEFAULT_MODE, algorithm=DEFAULT_ALGORITHM, seed=DEFAULT_SEED)
PRESETS = {
    "🗺️ Rechteck": {**_BASE},
    "🔷 L-Form": {**_BASE, "shape": "ell"},
    "🏝️ Kleine Inseln": {**_BASE, "shape": "islands", "n": 24, "lam": 1.5},
    "〰️ Dünne Streifen": {**_BASE, "shape": "stripes"},
    "🌫️ Starkes Rauschen": {**_BASE, "sigma": 130},
    "🧊 Zu stark geglättet": {**_BASE, "lam": 3.0},
    "🚫 Nicht submodular": {**_BASE, "n": 4, "mode": "reward", "lam": 1.0, "seed": 3},
    "🎓 Lehrraster": {**_BASE, "n": 4, "sigma": 60, "seed": 8},
}
# Jede Zahl in diesen Texten ist in tests/test_claims.py belegt (Zufallsraster über die Seeds der Presets)
PRESET_HELP = {
    "🗺️ Rechteck": "Rechteck im 16 × 16-Raster, Rauschen 80 %, λ = 1: die Schwelle je Zelle beschriftet 68 von 256 Zellen falsch, der Mittelwertfilter 22, ICM 13, der Graph Cut 8. Sein Fluss hat 225 Wege und 7 629 durchsuchte Kanten (Boykov–Kolmogorov).",
    "🔷 L-Form": "L-förmige Zone: Schwelle 61 falsche Zellen, Mittelwertfilter 14, ICM 12, Graph Cut 5.",
    "🏝️ Kleine Inseln": "Große Zone mit vier kleinen Inseln (Kantenlänge 1 bis 4) im 24 × 24-Raster, λ = 1,5: der Graph Cut beschriftet nur 30 von 576 Zellen falsch (Mittelwertfilter 39, ICM 68, Schwelle 161) - und diese 30 sind genau die vier Inseln (1 + 4 + 9 + 16 Zellen): die Glättung löscht echte kleine Zonen.",
    "〰️ Dünne Streifen": "Zwei Zellen breite Streifen: hier hilft die Glättung nicht. Der Graph Cut beschriftet 53 von 256 Zellen falsch, der Mittelwertfilter 45, ICM 51, die Schwelle 59; die Energie des Graph Cut ist trotzdem die kleinste (215,31 gegen 233,59 bei ICM).",
    "🌫️ Starkes Rauschen": "Rauschen 130 % des Zonenabstands bei λ = 1: der Graph Cut beschriftet 64 von 256 Zellen falsch, mehr als der Mittelwertfilter (45) und ICM (43); die Schwelle 84. Ein festes λ passt nicht zu jedem Rauschen.",
    "🧊 Zu stark geglättet": "λ = 3 im Standardraster: das ganze Raster wird eine Zone, der Graph Cut beschriftet genau die 64 Zellen der kleineren Zone falsch (25 %) - schlechter als der Mittelwertfilter (22) und ICM (12).",
    "🚫 Nicht submodular": "4 × 4-Raster, Unterschiede werden belohnt: die Kantenkapazitäten wären negativ, der Schnitt (auf 0 gekappt) hat die Energie -7,25 - dieselbe wie die Schwelle -, das Minimum aller 65 536 Beschriftungen ist -17,99 (ICM findet es hier).",
    "🎓 Lehrraster": "4 × 4-Raster mit dem Hilfsnetz: der Graph Cut beschriftet 0 von 16 Zellen falsch (Schwelle 3, Mittelwertfilter 5, ICM 2) und hat die Energie 16,25 = das Minimum aller 65 536 Beschriftungen; ICM bleibt bei 16,45 hängen.",
}
