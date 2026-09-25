"""Plotly-Abbildungen: Karten (Wahrheit, Messung, Beschriftungen mit Fehlern), Hilfsnetz mit Fluss und Schnitt, Kurven (Glättung, Rauschen, Inseln), Verteilung, Aufwand.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Karten haben gleichen Maßstab (scaleanchor) mit automatischem Bereich; unnötige feste Bereiche werden
vermieden (ein vorgegebener Bereich wird beim ersten Zeichnen in schmaler Breite eingefroren)."""

import plotly.graph_objects as go

import gc_constants as C
import gc_cut as cu


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, w, h, height):
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1, constrain="domain")
    fig.update_yaxes(visible=False, autorange="reversed", constrain="domain")
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=4, b=4), plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return lock_axes(fig)


def _grid_values(values, w, h):
    return [[values[y * w + x] for x in range(w)] for y in range(h)]


def build_labels(labels, w, h, truth=None, height=300):
    """Beschriftung als zweifarbige Karte (hell = Zone 0, dunkel = Zone 1); falsch beschriftete Zellen (gegen `truth`) sind rot überlegt."""
    fig = go.Figure(go.Heatmap(z=_grid_values(labels, w, h), colorscale=[[0, C.COLORS["zone0"]], [1, C.COLORS["zone1"]]], zmin=0, zmax=1, showscale=False, xgap=1 if w <= 24 else 0, ygap=1 if h <= 24 else 0,
                               hovertemplate="Zelle (%{x}, %{y}): Zone %{z}<extra></extra>"))
    if truth is not None:
        wrong = [1 if labels[i] != truth[i] else None for i in range(w * h)]
        if any(wrong):
            fig.add_trace(go.Heatmap(z=_grid_values(wrong, w, h), colorscale=[[0, "rgba(214,39,40,0.75)"], [1, "rgba(214,39,40,0.75)"]], zmin=0, zmax=1, showscale=False, hoverinfo="skip",
                                     xgap=1 if w <= 24 else 0, ygap=1 if h <= 24 else 0))
    return _map_layout(fig, w, h, height)


def build_obs(obs, w, h, height=300):
    """Messwerte je Zelle (Tausendstel; Zone 0 hat den Mittelwert 0, Zone 1 den Mittelwert 1000)."""
    fig = go.Figure(go.Heatmap(z=_grid_values([o / 1000 for o in obs], w, h), colorscale="Blues", zmin=-1, zmax=2, showscale=False, xgap=1 if w <= 24 else 0, ygap=1 if h <= 24 else 0,
                               hovertemplate="Zelle (%{x}, %{y}): Messwert %{z:.2f}<extra></extra>"))
    return _map_layout(fig, w, h, height)


def build_network(grid, sol, height=460):
    """Hilfsnetz eines kleinen Rasters: links die Quelle S, rechts die Senke T, dazwischen die Zellen. Terminalkanten (grün: Quelle → Zelle, rot: Zelle → Senke) mit der Kapazität, graue Nachbarkanten
    mit der Glättung; der Schnitt (rot, dick) trennt die Quellseite (Label 0, grün) von der Senkenseite (Label 1, grau)."""
    net = sol.net
    fig = go.Figure()
    reach = {i + 2: sol.labels[i] == 0 for i in range(grid.n)}
    reach[net.s], reach[net.t] = True, False
    cut = [(u, v) for (u, v, _, _, _) in net.arcs if reach[u] and not reach[v]]
    cut_set = set(cut)
    label_pts = []
    for (u, v, cap, _, kind) in net.arcs:
        (x0, y0), (x1, y1) = net.pos[u], net.pos[v]
        if kind == cu.K_NBR and (u > v):
            continue                                              # Nachbarkanten je Paar nur einmal zeichnen (Kapazität in beide Richtungen gleich)
        color = {cu.K_SRC: "rgba(44,160,44,0.55)", cu.K_SNK: "rgba(214,39,40,0.5)", cu.K_NBR: "rgba(120,120,120,0.6)"}[kind]
        is_cut = (u, v) in cut_set or (kind == cu.K_NBR and ((v, u) in cut_set or (u, v) in cut_set))
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=C.COLORS["wrong"] if is_cut else color, width=4 if is_cut else 1.4), hoverinfo="skip", showlegend=False))
        if grid.w <= 4 and (is_cut or kind == cu.K_NBR):
            t = 0.5 if kind == cu.K_NBR else 0.6
            label_pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, f"{cap / 100:g}", is_cut))
    for x, y, text, is_cut in label_pts:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(size=10, color="#111"), bgcolor="rgba(255,255,255,0.85)", borderpad=1)
    idx = list(range(net.n))
    colors = [C.COLORS["right"] if reach[v] else "#8c8c8c" for v in idx]
    fig.add_trace(go.Scatter(x=[net.pos[v][0] for v in idx], y=[net.pos[v][1] for v in idx], mode="markers+text", text=["S", "T"] + ["" for _ in range(grid.n)], textposition="top center", hovertext=list(net.names), hoverinfo="text",
                             marker=dict(symbol=["square" if v in (net.s, net.t) else "circle" for v in idx], size=[14 if v in (net.s, net.t) else 12 for v in idx], color=colors, line=dict(width=1.5, color="#333")), showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return _base(fig, height)


def build_dist(dist, height=320):
    """Mittlere Fehlklassifikation je Verfahren über die festen Ziehungen."""
    import gc_evaluation as ev
    keys = list(ev.METHODS)
    fig = go.Figure(go.Bar(x=[ev.METHOD_LABELS[k] for k in keys], y=[dist[k] for k in keys], marker_color=["#8c8c8c", "#1f77b4", "#9467bd", C.COLORS["wrong"]],
                           text=[f"{dist[k]:.1%}" for k in keys], textposition="outside"))
    fig.update_yaxes(title="falsch beschriftete Zellen", tickformat=".0%", range=[0, max(dist[k] for k in keys) * 1.25 + 0.01])
    return _base(fig, height)


def build_lambda_curve(c, current=None, height=340):
    """Fehlklassifikation des Graph Cut gegen die Glättung, mit Schwelle und Mittelwertfilter als Bezug; gepunktet die minimale Energie (steigt mit λ, taugt nicht zur Wahl)."""
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=list(c["lams"]), y=c["cut"], mode="lines+markers", name="Graph Cut", line=dict(color=C.COLORS["wrong"])), secondary_y=False)
    fig.add_hline(y=c["mean"], line=dict(color="#1f77b4", dash="dash"), annotation_text="Mittelwertfilter", annotation_position="top right")
    fig.add_hline(y=c["threshold"], line=dict(color="#8c8c8c", dash="dash"), annotation_text="Schwelle je Zelle", annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=list(c["lams"]), y=[e / 100 for e in c["energy"]], mode="lines", name="minimale Energie (rechte Achse)", line=dict(color="#555", dash="dot")), secondary_y=True)
    if current is not None:
        fig.add_vline(x=current, line=dict(color="#111", dash="dash"), annotation_text="Ihre Einstellung", annotation_position="top left")
    fig.update_xaxes(title="Glättung λ")
    fig.update_yaxes(title="falsch beschriftete Zellen", tickformat=".0%", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title="Energie", secondary_y=True, showgrid=False, rangemode="tozero")
    return _base(fig, height)


def build_sigma_series(rows, height=320):
    """Fehlklassifikation gegen das Rauschen: Schwelle, Mittelwertfilter, Graph Cut mit lam = 1 und Graph Cut mit dem besten lam (Orakel)."""
    fig = go.Figure()
    s = [r["sigma"] for r in rows]
    for key, name, color, dash in (("threshold", "Schwelle je Zelle", "#8c8c8c", "solid"), ("mean", "Mittelwertfilter", "#1f77b4", "solid"), ("err_lam1", "Graph Cut, λ = 1", "#ff7f0e", "dot"), ("best_err", "Graph Cut, bestes lam", C.COLORS["wrong"], "solid")):
        fig.add_trace(go.Scatter(x=s, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, dash=dash),
                                 text=[f"λ {r['best_lam']:g}" for r in rows] if key == "best_err" else None, hovertemplate="%{y:.1%}<extra>" + name + "</extra>"))
    fig.update_xaxes(title="Rauschen (Prozent des Zonenabstands)")
    fig.update_yaxes(title="falsch beschriftete Zellen", tickformat=".0%", rangemode="tozero")
    return _base(fig, height)


def build_islands(res, height=340):
    """Anteil der Ziehungen, in denen eine Insel wiedergefunden wird, gegen ihre Kantenlänge, eine Linie je Glättung."""
    fig = go.Figure()
    for k, (lam, counts) in enumerate(res["found"].items()):
        fig.add_trace(go.Scatter(x=list(res["sizes"]), y=[counts[s] / res["n"] for s in res["sizes"]], mode="lines+markers", name=f"λ = {lam:g}"))
    fig.update_xaxes(title="Kantenlänge der Insel (Zellen)", dtick=1)
    fig.update_yaxes(title="wiedergefunden in Ziehungen", tickformat=".0%", range=[-0.02, 1.05])
    return _base(fig, height)


def build_scaling(rows, height=340):
    """Durchsuchte Kanten gegen die Knotenzahl des Hilfsnetzes (doppelt logarithmisch): Boykov-Kolmogorov, Dinic, Edmonds-Karp."""
    fig = go.Figure()
    n = [r["nodes"] for r in rows]
    for key, label, color, dash in (("bk", "Boykov–Kolmogorov", C.COLORS["wrong"], "solid"), ("dinic", "Dinic", "#2ca02c", "solid"), ("bfs", "Edmonds-Karp", "#1f77b4", "dot")):
        fig.add_trace(go.Scatter(x=n, y=[r[key] for r in rows], mode="lines+markers", name=label, line=dict(color=color, dash=dash)))
    fig.add_trace(go.Scatter(x=n, y=[r["arcs"] for r in rows], mode="lines", name="Kanten des Hilfsnetzes", line=dict(color="#555", dash="dashdot")))
    fig.update_xaxes(title="Knoten des Hilfsnetzes", type="log")
    fig.update_yaxes(title="durchsuchte Kanten", type="log")
    return _base(fig, height)
