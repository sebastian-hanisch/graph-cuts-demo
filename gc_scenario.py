"""Szenario: ein Liefergebiets-Raster mit zwei wahren Zonen und verrauschten Messwerten je Zelle.

Ein Gebiet ist in Zellen geteilt; in jeder Zelle wird eine Kennzahl gemessen (etwa die Verspätungsrate der Lieferungen), die in einer Zone niedrig und in der anderen hoch liegt, aber verrauscht ist.
Gesucht sind die Zonen: jede Zelle bekommt eine von zwei Beschriftungen. Die wahren Zonen sind bekannt (Rechteck, L-Form, Scheibe, Streifen, Inseln), so lässt sich jede Rekonstruktion an der Wahrheit messen.

Alles ist ganzzahlig und läuft über einen eigenen Zufallsgenerator (SplitMix64 auf Python-Ints): Das Rauschen ist die Summe von zwölf Gleichverteilten (Irwin-Hall, näherungsweise normalverteilt) in reiner
Ganzzahl-Arithmetik, so dass keine Gleitkomma-Zufallszahlen und keine über Plattformen verschiedenen Rundungen in die Zahlen des Textes eingehen.
Messwerte stehen in Tausendsteln (Zone 0: Mittelwert 0, Zone 1: Mittelwert 1000).
"""

from dataclasses import dataclass

_MASK = (1 << 64) - 1
MU = (0, 1000)                 # Mittelwert der Messung je Zone (Tausendstel)
SHAPES = ("rect", "ell", "disc", "stripes", "islands")


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n):
        """Ganzzahl in 0..n-1 (die Modulo-Verzerrung bei n <= 1e6 liegt unter 1e-13)."""
        return self.next() % n


@dataclass(frozen=True)
class Net:
    names: tuple      # Anzeigename je Knoten (Hover)
    labels: tuple     # Kurzbeschriftung je Knoten (Karte)
    pos: tuple        # ((x, y), ...) je Knoten
    arcs: tuple       # ((u, v, Kapazität, Kosten je Einheit, Art), ...)
    s: int
    t: int
    logistic: bool    # True: Werke/DCs/Filialen; False: Lehrnetz mit frei benannten Knoten

    @property
    def n(self):
        return len(self.names)

    @property
    def m(self):
        return len(self.arcs)

    def total_capacity_out_of_s(self):
        return sum(c for u, _, c, _, _ in self.arcs if u == self.s)


@dataclass(frozen=True)
class Grid:
    w: int
    h: int
    truth: tuple      # Wahrheit je Zelle (0/1), zeilenweise (Zelle i = y * w + x)
    obs: tuple        # Messwert je Zelle in Tausendsteln
    sigma: int        # Standardabweichung des Rauschens in Tausendsteln (Zonenabstand = 1000)
    shape: str

    @property
    def n(self):
        return self.w * self.h

    def cell(self, i):
        return i % self.w, i // self.w

    def neighbours(self):
        """Paare (i, j) mit i < j, 4-Nachbarschaft."""
        out = []
        for y in range(self.h):
            for x in range(self.w):
                i = y * self.w + x
                if x + 1 < self.w:
                    out.append((i, i + 1))
                if y + 1 < self.h:
                    out.append((i, i + self.w))
        return tuple(out)


def truth_map(shape, w, h):
    """Wahre Zonen: Zone 1 ist die 'hohe' Zone. Alle Formen sind fest und hängen nur von der Rastergröße ab."""
    t = [[0] * w for _ in range(h)]
    if shape == "rect":                    # ein Rechteck in der Mitte (halbe Kantenlängen)
        for y in range(h // 4, h - h // 4):
            for x in range(w // 4, w - w // 4):
                t[y][x] = 1
    elif shape == "ell":                   # L-Form: linke Spalte und untere Zeile, je ein Drittel breit
        bar = max(1, min(w, h) // 3)
        for y in range(h):
            for x in range(w):
                if x < bar or y >= h - bar:
                    t[y][x] = 1
    elif shape == "disc":                  # Scheibe in der Mitte
        cx, cy, r = (w - 1) / 2, (h - 1) / 2, min(w, h) * 0.32
        for y in range(h):
            for x in range(w):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    t[y][x] = 1
    elif shape == "stripes":               # senkrechte Streifen, abwechselnd (Breite ein Sechstel, mindestens 2)
        bar = max(2, w // 6)
        for y in range(h):
            for x in range(w):
                t[y][x] = (x // bar) % 2
    elif shape == "islands":                # große Zone links plus kleine Inseln mit Kantenlänge 1 bis 4
        for y in range(h):
            for x in range(w // 2):
                t[y][x] = 1
        for k, size in enumerate((1, 2, 3, 4)):
            x0, y0 = w // 2 + 1 + (k % 2) * (w // 4), 1 + (k // 2) * (h // 2)
            for y in range(y0, min(h, y0 + size)):
                for x in range(x0, min(w, x0 + size)):
                    t[y][x] = 1
    else:
        raise ValueError(shape)
    return tuple(v for row in t for v in row)


def noise(rng):
    """Näherungsweise normalverteilt mit Standardabweichung 1000 (Tausendstel): Summe von zwölf Gleichverteilten minus 6 in Ganzzahlen (Irwin-Hall, Varianz 1)."""
    return sum(rng.below(1001) for _ in range(12)) - 6000


def make_grid(shape, w, h, sigma, seed):
    """Raster mit Messung. `sigma`: Rauschen in Prozent des Zonenabstands (Standardabweichung, 100 = so groß wie der Abstand)."""
    truth = truth_map(shape, w, h)
    rng = SplitMix64(seed)
    obs = tuple(MU[truth[i]] + noise(rng) * sigma // 100 for i in range(w * h))
    return Grid(w, h, truth, obs, sigma * 10, shape)


def observe(truth, w, h, sigma, seed, shape="custom"):
    """Messung zu einer vorgegebenen Wahrheit (für Lehrraster und Inseln)."""
    rng = SplitMix64(seed)
    obs = tuple(MU[truth[i]] + noise(rng) * sigma // 100 for i in range(w * h))
    return Grid(w, h, tuple(truth), obs, sigma * 10, shape)
