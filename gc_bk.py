"""Boykov-Kolmogorov: maximaler Fluss mit zwei wachsenden Suchbäumen (Boykov und Kolmogorov 2004), das Praxisverfahren für Graph Cuts auf Gittern.

Statt für jeden Weg neu zu suchen (Edmonds-Karp) oder Phase für Phase einen Niveaugraph zu bauen (Dinic), hält BK **zwei Suchbäume** dauerhaft: einen von der Quelle S und einen von der Senke T.
Drei Schritte wiederholen sich:

1. **Wachstum:** aktive Knoten der beiden Bäume suchen freie Nachbarn und nehmen sie in ihren Baum auf; treffen sich ein S-Baum-Knoten und ein T-Baum-Knoten über eine Kante mit Restkapazität, ist ein Verbesserungsweg gefunden.
2. **Augmentieren:** der Weg (über die Baumkanten von S bis T) wird um seinen Engpass aufgefüllt; jede dabei gesättigte Baumkante macht ihren Kindknoten zum **Verwaisten**.
3. **Adoption:** ein Verwaister sucht im eigenen Baum einen neuen Elternknoten, der noch mit der Wurzel verbunden ist (Ursprungsprüfung, beschleunigt durch Zeitstempel und Entfernungsangabe nach dem Papier); findet er keinen, wird er frei, und seine Kinder werden ebenfalls verwaist.

Der Baum wird also nach einem Weg **repariert statt neu gebaut**. Bewiesen ist keine bessere Laufzeitschranke als bei anderen Verfahren (schlimmstenfalls sogar schlechter), aber auf Gittergraphen mit kurzen Wegen ist BK in der Praxis oft das schnellste Verfahren. Am Ende bilden die Knoten des S-Baums die Quellseite eines minimalen Schnitts.

Restkanten wie in den anderen Fluss-Demos: Kante 2i ist die Vorwärtskante der Netzkante i (Rest = Kapazität - Fluss), Kante 2i+1 die Rückkante (Rest = Fluss).
Aufwand wird in durchsuchten Kanten gezählt (jede in Wachstum und Adoption angesehene Restkante, dazu jeder Schritt einer Ursprungsprüfung), nie in Sekunden.
"""

from collections import deque
from dataclasses import dataclass

FREE, TREE_S, TREE_T = 0, 1, 2
ROOT, ORPHAN = -1, -2
INF = 1 << 60


@dataclass(frozen=True)
class Result:
    value: int
    reach: tuple          # bool je Knoten: Knoten des S-Baums am Ende (= von S im Schluss-Restgraphen erreichbar)
    n_paths: int          # aufgefüllte Verbesserungswege
    scanned_total: int    # durchsuchte Kanten insgesamt
    grow_scanned: int
    adopt_scanned: int
    orphans: int          # insgesamt verwaiste Knoten
    flow: tuple           # Fluss je Netzkante


def _adjacency(net):
    adj = [[] for _ in range(net.n)]
    head = [0] * (2 * net.m)
    for i, (u, v, _, _, _) in enumerate(net.arcs):
        adj[u].append(2 * i)
        adj[v].append(2 * i + 1)
        head[2 * i], head[2 * i + 1] = v, u
    return adj, head


def bk_maxflow(net):
    adj, head = _adjacency(net)
    n, s, t = net.n, net.s, net.t
    res = [0] * (2 * net.m)
    for i, (_, _, cap, _, _) in enumerate(net.arcs):
        res[2 * i] = cap
    tree = [FREE] * n
    parent = [None] * n                  # Kante vom Knoten zu seinem Elternknoten (ROOT bei S und T, ORPHAN bei Verwaisten)
    ts = [0] * n
    dist = [0] * n
    tree[s], tree[t] = TREE_S, TREE_T
    parent[s] = parent[t] = ROOT
    active = deque([s, t])
    in_active = [False] * n
    in_active[s] = in_active[t] = True
    time = 1
    ts[s] = ts[t] = time
    dist[s] = dist[t] = 1
    value = paths = orphans_total = 0
    grow_scanned = adopt_scanned = 0

    def tree_cap(a, node_tree):
        """Restkapazität der Kante a (Kante vom Knoten i zu seinem Nachbarn j) in Baumrichtung: S-Baum: j -> i, T-Baum: i -> j."""
        return res[a ^ 1] if node_tree == TREE_S else res[a]

    while True:
        # --- Wachstum --------------------------------------------------------------------------------------------------------------
        found = None
        while active:
            p = active[0]
            if tree[p] == FREE:
                active.popleft()
                in_active[p] = False
                continue
            for a in adj[p]:
                grow_scanned += 1
                q = head[a]
                if tree[p] == TREE_S:
                    if res[a] <= 0:
                        continue
                    if tree[q] == FREE:
                        tree[q], parent[q], ts[q], dist[q] = TREE_S, a ^ 1, ts[p], dist[p] + 1
                        active.append(q)
                        in_active[q] = True
                    elif tree[q] == TREE_T:
                        found = a                          # Kante p -> q verbindet S-Baum und T-Baum
                        break
                else:
                    if res[a ^ 1] <= 0:
                        continue
                    if tree[q] == FREE:
                        tree[q], parent[q], ts[q], dist[q] = TREE_T, a ^ 1, ts[p], dist[p] + 1
                        active.append(q)
                        in_active[q] = True
                    elif tree[q] == TREE_S:
                        found = a ^ 1                      # Kante q -> p
                        break
            if found is not None:
                break
            active.popleft()
            in_active[p] = False
        if found is None:
            break
        # --- Augmentieren ------------------------------------------------------------------------------------------------------------
        mid = found
        s_arcs, t_arcs = [], []                             # Flusskanten (in Flussrichtung) im S- bzw. T-Teil des Weges
        x = head[mid ^ 1]                                   # Knoten im S-Baum
        while parent[x] != ROOT:
            a = parent[x]
            s_arcs.append((a ^ 1, x))
            x = head[a]
        x = head[mid]                                       # Knoten im T-Baum
        while parent[x] != ROOT:
            a = parent[x]
            t_arcs.append((a, x))
            x = head[a]
        bottleneck = min([res[mid]] + [res[a] for a, _ in s_arcs] + [res[a] for a, _ in t_arcs])
        res[mid] -= bottleneck
        res[mid ^ 1] += bottleneck
        orphan_queue = deque()
        for a, x in s_arcs + t_arcs:
            res[a] -= bottleneck
            res[a ^ 1] += bottleneck
            if res[a] == 0:
                parent[x] = ORPHAN
                orphan_queue.append(x)
                orphans_total += 1
        value += bottleneck
        paths += 1
        time += 1
        # --- Adoption -------------------------------------------------------------------------------------------------------------------
        while orphan_queue:
            i = orphan_queue.popleft()
            if parent[i] != ORPHAN:
                continue
            best, best_d = None, INF
            for a in adj[i]:
                adopt_scanned += 1
                j = head[a]
                if tree[j] != tree[i] or tree_cap(a, tree[i]) <= 0 or parent[j] == ORPHAN:
                    continue
                d = 0                                        # Entfernung von j zur Wurzel, wenn j noch mit ihr verbunden ist
                y = j
                while True:
                    adopt_scanned += 1
                    if ts[y] == time:
                        d += dist[y]
                        break
                    pa = parent[y]
                    d += 1
                    if pa == ROOT:
                        ts[y], dist[y] = time, 1
                        break
                    if pa == ORPHAN:
                        d = INF
                        break
                    y = head[pa]
                if d < INF:
                    if d < best_d:
                        best, best_d = a, d
                    y = j
                    while ts[y] != time:                     # Zeitstempel und Entfernung entlang des Weges vermerken
                        ts[y], dist[y] = time, d
                        d -= 1
                        y = head[parent[y]]
            if best is not None:
                parent[i] = best
                ts[i], dist[i] = time, best_d + 1
                continue
            for a in adj[i]:                                 # kein neuer Elternknoten: i wird frei, Nachbarn und Kinder reagieren
                adopt_scanned += 1
                j = head[a]
                if tree[j] != tree[i]:
                    continue
                if tree_cap(a, tree[i]) > 0 and not in_active[j]:
                    active.append(j)
                    in_active[j] = True
                pj = parent[j]
                if pj is not None and pj >= 0 and head[pj] == i:
                    parent[j] = ORPHAN
                    orphan_queue.append(j)
                    orphans_total += 1
            tree[i] = FREE
            parent[i] = None
    flow = tuple(net.arcs[i][2] - res[2 * i] for i in range(net.m))
    return Result(value, tuple(x == TREE_S for x in tree), paths, grow_scanned + adopt_scanned, grow_scanned, adopt_scanned, orphans_total, flow)
