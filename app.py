"""Graph Cuts - glatte Zonen aus verrauschten Messwerten - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - den minimalen Schnitt als Modell einer binären Beschriftung mit Glattheitsstrafe - und lässt stattdessen das Beispiel wachsen.
Zweite Erweiterung (Stück 14) der Netzwerkfluss-Linie der "Konzepte"-Reihe: Fortsetzung der Demo "Projektauswahl", dort war die Nachbarstrafe unendlich (Voraussetzung), hier ist sie endlich. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import gc_constants as C
import gc_cut as cu
import gc_evaluation as ev
from gc_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from gc_visualization import (
    build_dist,
    build_islands,
    build_labels,
    build_lambda_curve,
    build_network,
    build_obs,
    build_scaling,
    build_sigma_series,
)

st.set_page_config(page_title="Graph Cuts – Sebastian Hanisch", layout="wide")


def _f(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def _int(x):
    return f"{int(round(x)):,}".replace(",", " ")


def _pct(x, digits=1):
    return f"{100 * x:.{digits}f} %".replace(".", ",")


@st.cache_resource(show_spinner=False, max_entries=32)
def _analysis(params):
    return ev.analyse(ev.Params(*params))


@st.cache_resource(show_spinner=False, max_entries=16)
def _distribution(params):
    return ev.distribution(ev.Params(*params))


st.title("🗺️ Graph Cuts – glatte Zonen aus verrauschten Messwerten")
st.markdown(
    """
Ein Liefergebiet ist in Zellen geteilt, und in jeder Zelle wird eine Kennzahl gemessen (etwa die Verspätungsrate), die in einer **Zone** niedrig und in der anderen hoch liegt - aber **verrauscht**. Wer jede Zelle für sich beurteilt, bekommt ein zerfleddertes Bild; wer die Nachbarn mitschauen lässt, bekommt glatte Zonen.
Beides lässt sich als **Energie** schreiben: ein **Datenterm** (wie gut passt das Label zur Messung dieser Zelle) plus eine **Glättungsstrafe** für jedes Nachbarpaar mit verschiedenen Labels. Für zwei Labels ist die beste Beschriftung ein **minimaler Schnitt** (Greig, Porteous, Seheult 1989; Boykov und Jolly 2001): Quelle und Senke sind die beiden Zonen, jede Zelle hängt mit ihrem Datenpreis an ihnen, Nachbarn hängen mit der Glättung aneinander.
Diese Demo zeigt den Bau des Netzes, vergleicht den Schnitt mit einfacheren Verfahren gegen die **bekannte Wahrheit** - und wo die Glättung zu viel oder zu wenig tut.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - zweite Erweiterung der Netzwerkfluss-Linie der \"Konzepte\"-Reihe, Fortsetzung der Demo \"Projektauswahl\" - **ein** Verfahren an einem wachsenden Beispiel. "
    "Dort war die Nachbarstrafe **unendlich** (eine Voraussetzung), hier ist sie endlich. Zum Rechnen kommt **Boykov–Kolmogorov** dazu, das Praxisverfahren für Gitter. Machine Learning und Optimierung treffen sich hier: die Energie ist ein Markov Random Field, der Schnitt ihr exakter Löser."
)

with st.expander("So funktioniert der Graph Cut", expanded=True):
    st.markdown(
        r"""
1. **Energie:** eine Beschriftung $x\in\{0,1\}^{\text{Zellen}}$ kostet $E(x)=\sum_i D_i(x_i)+\lambda\sum_{(i,j)}[x_i\ne x_j]$. Der Datenterm $D_i(l)=(y_i-\mu_l)^2/(2\sigma^2)$ ist der Preis, Zelle $i$ mit Zone $l$ zu erklären; $\lambda$ bestraft jedes Nachbarpaar (4er-Nachbarschaft) mit verschiedenen Labels.
2. **Hilfsnetz:** Quelle $s$ (Zone 0), Senke $t$ (Zone 1), je Zelle ein Knoten. Nach Abzug des kleineren Datenpreises bleibt je Zelle **eine** Terminalkante: $s\to i$ mit $D_i(1)-D_i(0)$, wenn Zone 1 teurer ist, sonst $i\to t$ mit $D_i(0)-D_i(1)$. Jedes Nachbarpaar bekommt zwei Kanten mit Kapazität $\lambda$.
3. **Schnitt:** ein Schnitt trennt die Zellen in Quellseite (Zone 0) und Senkenseite (Zone 1) und kostet genau $E(x)$ (bis auf die Summe der kleineren Datenpreise): **minimaler Schnitt = beste Beschriftung**. Zellen mit eindeutiger Messung zahlen am Terminal, Zellen mit unklarer Messung folgen ihren Nachbarn.
4. **Submodular:** das klappt nur, wenn ungleiche Nachbarn nicht *belohnt* werden: $E(0,0)+E(1,1)\le E(0,1)+E(1,0)$, also keine negative Kantenkapazität. Wer Unterschiede belohnt, bekommt kein Schnittproblem mehr.
5. **Lösen:** ein Fluss von $s$ nach $t$. **Boykov–Kolmogorov** hält zwei Suchbäume (von $s$ und von $t$) dauerhaft, füllt Wege zwischen ihnen auf und repariert danach den Baum, statt neu zu suchen; Dinic und Edmonds-Karp sind zum Vergleich wählbar.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
names = list(C.PRESETS.keys())
for row in range(0, len(names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, names[row:row + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    shape = st.selectbox("Wahre Zonen", list(C.SHAPES), key="shape_select", format_func=lambda k: C.SHAPES[k],
                         help="Die Form der wahren Zone 1 im Raster. Rechteck, L-Form und Scheibe sind große, glatte Zonen; Streifen sind dünn (zwei Zellen breit), die Inseln haben Kantenlängen von 1 bis 4.")
    n = st.slider("Raster [Zellen je Kante]", *bounds("n_slider"), key="n_slider", help="Quadratisches Raster; bis 6 × 6 Zellen wird das Hilfsnetz gezeichnet, bei 4 × 4 lassen sich alle 65 536 Beschriftungen durchprobieren.")
    sigma = st.slider("Rauschen [% des Zonenabstands]", *bounds("sigma_slider"), key="sigma_slider", step=10,
                      help="Standardabweichung der Messung, in Prozent des Abstands der beiden Zonenmittelwerte. Bei 80 % liegt die Schwelle je Zelle im Standardraster bei etwa 26 % falscher Zellen, bei 130 % bei etwa 35 %.")
    lam = st.slider("Glättung λ", *bounds("lam_slider"), key="lam_slider", step=0.1, format="%.1f",
                    help="Strafe für jedes Nachbarpaar mit verschiedenen Labels, in Einheiten des Datenpreises. 0 = keine Glättung (Schwelle je Zelle). Zu viel Glättung löscht die kleinere Zone ganz.")
    mode = st.radio("Nachbarstrafe", list(C.MODES), key="mode_radio", format_func=lambda k: C.MODES[k],
                    help="Glätten bestraft ungleiche Nachbarn (submodular: der Schnitt ist exakt). Unterschiede belohnen ist die Negativkontrolle: negative Kantenkapazitäten, der Schnitt ist nicht mehr die beste Beschriftung.")
    algorithm = st.radio("Flussverfahren", list(C.ALGORITHMS), key="algorithm_radio", format_func=lambda k: C.ALGORITHMS[k],
                         help="Alle drei finden dieselbe minimale Energie; sie unterscheiden sich in den durchsuchten Kanten. Auf einem 16 × 16-Raster durchsucht Boykov–Kolmogorov etwa 4,6-mal weniger Kanten als Dinic und etwa 43-mal weniger als Edmonds-Karp.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Messung ziehen", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed (neues Rauschen bei gleicher Wahrheit). Die Verteilungen über 40 feste Ziehungen weiter unten ändern sich dabei nicht.")

sync_query_params({"shape_select": shape, "n_slider": int(n), "sigma_slider": int(sigma), "lam_slider": round(float(lam), 1), "mode_radio": mode, "algorithm_radio": algorithm, "seed_input": int(seed)})

params = (shape, int(n), int(sigma), round(float(lam), 1), mode, algorithm, int(seed))
with st.spinner("Rechne..."):
    a = _analysis(params)
g, labels, errs, energies, cut = a["grid"], a["labels"], a["errors"], a["energies"], a["cut"]
w = h = int(n)

# --- Karten -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Von der Messung zu den Zonen")
row1 = st.columns(2)
row2 = st.columns(2)
row3 = st.columns(2)
row1[0].markdown("**Wahrheit** (Zone 1 dunkel)")
row1[0].plotly_chart(build_labels(g.truth, w, h), width="stretch", key="map_truth")
row1[1].markdown("**Messung** (verrauscht)")
row1[1].plotly_chart(build_obs(g.obs, w, h), width="stretch", key="map_obs")
maps = [("threshold", row2[0]), ("mean", row2[1]), ("icm", row3[0]), ("cut", row3[1])]
for key, col in maps:
    col.markdown(f"**{ev.METHOD_LABELS[key]}** – {errs[key]} von {g.n} falsch")
    col.plotly_chart(build_labels(labels[key], w, h, truth=g.truth), width="stretch", key=f"map_{key}")
st.caption("Hell = Zone 0, dunkel = Zone 1; rot überlegte Zellen sind gegen die Wahrheit falsch beschriftet. "
           "Die Schwelle beurteilt jede Zelle allein; der Mittelwertfilter mittelt über 3 × 3 Zellen; ICM (Iterated Conditional Modes) verbessert von der Schwelle aus Zelle für Zelle die Energie; der Graph Cut findet die minimale Energie.")

m1, m2, m3, m4 = st.columns(4)
for col, key in zip((m1, m2, m3, m4), ev.METHODS):
    col.metric(ev.METHOD_LABELS[key], f"{_pct(errs[key] / g.n)}", delta=f"Energie {energies[key] / 100:g}", delta_color="off", help="Anteil falsch beschrifteter Zellen gegen die Wahrheit; darunter die Energie der Beschriftung (Datenterm + Glättung, in Einheiten).")

best_other = min(errs["threshold"], errs["mean"], errs["icm"])
if mode == cu.REWARD and params[3] > 0:
    st.warning(f"⚠️ Unterschiede zu belohnen ist **nicht submodular**: die Kantenkapazitäten wären negativ, der Schnitt ist hier mit auf 0 gekappten Kapazitäten gebaut und liefert die Energie {energies['cut'] / 100:g}, ICM {energies['icm'] / 100:g}. "
               + ("Das exakte Minimum aller 65 536 Beschriftungen: " + f"{cu.brute_force(g, a['lam'], mode)[0] / 100:g}." if g.n <= 16 else "Bei diesem Raster ist die Antwort des Schnitts nicht mehr optimal."))
elif errs["cut"] < best_other:
    st.success(f"✅ Der Graph Cut beschriftet {errs['cut']} von {g.n} Zellen falsch ({_pct(errs['cut'] / g.n)}), das beste einfachere Verfahren {best_other} ({_pct(best_other / g.n)}); die Schwelle allein {errs['threshold']}.")
elif errs["cut"] == best_other:
    st.info(f"ℹ️ Der Graph Cut ist hier nicht besser als das beste einfachere Verfahren (beide {errs['cut']} falsche Zellen). Mit anderer Glättung oder anderem Rauschen kann sich das ändern.")
else:
    st.warning(f"⚠️ Der Graph Cut beschriftet **mehr** Zellen falsch ({errs['cut']}) als das beste einfachere Verfahren ({best_other}) - obwohl seine Energie die kleinste ist ({energies['cut'] / 100:g} gegen {energies['icm'] / 100:g} bei ICM). "
               "Die minimale Energie ist nicht die beste Rekonstruktion: die Glättung passt hier nicht zur Wahrheit (zu stark, zu schwach oder dünne Zonen).")
if params[3] > 0 and mode == cu.POTTS and sum(labels["cut"]) in (0, g.n):
    st.info("ℹ️ Die Glättung ist so stark, dass das ganze Raster **eine** Zone wird: die kleinere Zone wird weggeglättet, der Fehler ist dann genau ihre Fläche.")

st.table({"Verfahren": [ev.METHOD_LABELS[k] for k in ev.METHODS], "falsch beschriftet": [f"{errs[k]} von {g.n}" for k in ev.METHODS], "Energie": [f"{energies[k] / 100:g}" for k in ev.METHODS]})
st.caption(f"Flussverfahren {C.ALGORITHMS[algorithm]}: {_int(cut.n_paths)} aufgefüllte Wege, {_int(cut.scanned)} durchsuchte Kanten im Hilfsnetz ({_int(cut.net.n)} Knoten, {_int(cut.net.m)} Kanten); ICM braucht {a['icm_sweeps']} Durchläufe.")

if int(n) <= C.HELP_MAX_N:
    st.markdown("**Das Hilfsnetz dieses Rasters**")
    st.plotly_chart(build_network(g, cut), width="stretch", key="help_net")
    st.caption("Links die Quelle S (Zone 0), rechts die Senke T (Zone 1). Grün: Terminalkante Quelle → Zelle (Zone 1 wäre teurer), rot: Zelle → Senke (Zone 0 wäre teurer), grau: Nachbarkanten mit der Glättung λ. "
               "Der Schnitt (rote, dicke Kanten) trennt grüne Zellen (Quellseite = Zone 0) von grauen (Senkenseite = Zone 1); seine Kapazität plus die Summe der kleineren Datenpreise ist die minimale Energie.")
else:
    st.caption(f"Das Hilfsnetz wird bis {C.HELP_MAX_N} × {C.HELP_MAX_N} Zellen gezeichnet; für ein größeres Raster ist es zu dicht.")

st.markdown("---")

# --- Verteilung -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Nicht nur diese eine Messung")
st.markdown(f"**Vierzig feste Rauschziehungen** mit denselben Einstellungen (Zonen {C.SHAPES[shape]}, Raster {n} × {n}, Rauschen {sigma} %, Glättung {params[3]:g}), getrennt vom Seed oben.")
dist = _distribution(params)
d1, d2, d3, d4 = st.columns(4)
d1.metric("Schwelle je Zelle", _pct(dist["threshold"]), help="Mittlerer Anteil falscher Zellen über die Ziehungen.")
d2.metric("Mittelwertfilter", _pct(dist["mean"]))
d3.metric("ICM", _pct(dist["icm"]), delta=f"über dem Optimum in {dist['icm_above']} von {dist['n']}", delta_color="off", help="ICM bleibt in einem lokalen Minimum der Energie hängen: seine Energie liegt über der minimalen (mittlere Lücke siehe unten).")
d4.metric("Graph Cut", _pct(dist["cut"]), delta=f"bester in {dist['cut_best']} von {dist['n']}", delta_color="off", help="Ziehungen, in denen der Graph Cut weniger Zellen falsch beschriftet als Schwelle und Mittelwertfilter.")
st.plotly_chart(build_dist(dist), width="stretch", key="dist_chart")
st.caption(f"Mittel über {dist['n']} feste Ziehungen. Die Energie von ICM liegt im Mittel {_pct(dist['icm_gap'])} über der minimalen; der Graph Cut ist exakt (auf 4 × 4 Rastern gegen alle Beschriftungen geprüft, siehe Gegenprobe unten).")

st.markdown("---")

# --- Experimente -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie viel Glättung ist richtig?")
st.caption("Die Fehlklassifikation des Graph Cut gegen die Glättung λ, gemittelt über 10 feste Ziehungen. Die Kurve hat ein enges Minimum: mit λ = 0 ist es die Schwelle, mit zu großem λ kollabiert das Raster zu einer Zone. Die minimale Energie (gepunktet) steigt mit λ immer - **λ lässt sich nicht über die Energie wählen**.")
if st.button("Kurve durchrechnen (dauert wenige Sekunden)", key="curve_start"):
    st.session_state["curve_on"] = True
if st.session_state.get("curve_on"):
    with st.spinner("Rechne 31 Glättungswerte × 10 Ziehungen..."):
        curve = ev.lambda_curve(ev.Params(*params))
    st.plotly_chart(build_lambda_curve(curve, current=params[3]), width="stretch", key="curve_chart")
    win = curve["window"]
    st.caption(f"Bestes λ (Orakel, das die Wahrheit kennt): {curve['best_lam']:g} mit {_pct(curve['best_err'])} falschen Zellen; Mittelwertfilter {_pct(curve['mean'])}, Schwelle {_pct(curve['threshold'])}. "
               + (f"Der Graph Cut schlägt den Mittelwertfilter für λ von {win[0]:g} bis {win[1]:g}." if win else "Der Graph Cut schlägt den Mittelwertfilter für kein λ."))

st.subheader("🔬 Wie hängt das beste λ vom Rauschen ab?")
st.caption("Für jede Rauschstärke das λ mit der kleinsten mittleren Fehlklassifikation (Orakel), gegen den Mittelwertfilter und gegen das feste λ = 1. Das beste λ sinkt mit dem Rauschen - der Datenterm ist mit $1/\\sigma^2$ gewichtet, laute Messungen haben also ein schwaches Datengewicht - und ein festes λ = 1 ist bei starkem Rauschen schlechter als der Mittelwertfilter.")
if st.button("Rauschreihe durchrechnen (dauert einige Sekunden)", key="sigma_start"):
    st.session_state["sigma_on"] = True
if st.session_state.get("sigma_on"):
    with st.spinner("Rechne 6 Rauschstärken × 31 Glättungswerte × 10 Ziehungen..."):
        rows = ev.sigma_series(ev.Params(*params))
    st.plotly_chart(build_sigma_series(rows), width="stretch", key="sigma_chart")
    st.table({"Rauschen [%]": [r["sigma"] for r in rows], "bestes λ": [_f(r["best_lam"]) for r in rows], "falsch bei bestem λ": [_pct(r["best_err"]) for r in rows], "falsch bei λ = 1": [_pct(r["err_lam1"]) for r in rows], "Mittelwertfilter": [_pct(r["mean"]) for r in rows]})
    st.caption("Wahre Zonen und Raster wie in der Seitenleiste (Standard: Rechteck, 16 × 16), 10 feste Ziehungen je Rauschstärke; λ-Raster 0 bis 3 in Schritten von 0,1 (im Standardraster liegt das beste λ bei 50 % Rauschen am oberen Rand, es könnte höher liegen).")

st.subheader("🔬 Welche kleinen Zonen verschwinden?")
st.caption("Quadratische Inseln der Kantenlänge 1 bis 6 in einer 40 × 20-Karte (Rauschen 60 %). Eine Insel gilt als wiedergefunden, wenn mindestens die Hälfte ihrer Zellen die richtige Zone bekommt. Eine Glättung, die glatte Zonen liefert, löscht auch **echte** kleine Zonen - kleine zuerst.")
if st.button("Inseln durchrechnen (dauert wenige Sekunden)", key="islands_start"):
    st.session_state["islands_on"] = True
if st.session_state.get("islands_on"):
    with st.spinner("Rechne 6 Glättungswerte × 20 Ziehungen..."):
        isl = ev.islands(60)
    st.plotly_chart(build_islands(isl), width="stretch", key="islands_chart")
    st.table({"Glättung λ": [f"{lm:g}" for lm in isl["found"]], **{f"Kante {s}": [f"{c[s]} von {isl['n']}" for c in isl["found"].values()] for s in isl["sizes"][:4]}})
    st.caption("Je größer λ, desto größer die Mindestfläche: bei λ = 1 werden Inseln ab Kantenlänge 4 (16 Zellen) zuverlässig gefunden, bei λ = 2 ist selbst die 6 × 6-Insel nur noch in 12 von 20 Ziehungen da. Die ersten vier Kantenlängen stehen in der Tabelle, alle sechs im Diagramm.")

st.subheader("🔬 Ist Boykov–Kolmogorov wirklich schneller?")
st.caption("Durchsuchte Kanten der drei Flussverfahren im Hilfsnetz von 8 × 8 bis 48 × 48 Zellen (Rauschen 80 %, λ = 1, Mittel über 5 feste Ziehungen). Alle drei finden dieselbe minimale Energie.")
if st.button("Raster durchrechnen (dauert einige Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne 6 Rastergrößen × 5 Ziehungen × 3 Verfahren..."):
        sc_rows = ev.scaling()
    st.plotly_chart(build_scaling(sc_rows), width="stretch", key="scaling_chart")
    st.table({"Raster": [f"{r['n']} × {r['n']}" for r in sc_rows], "Knoten": [_int(r["nodes"]) for r in sc_rows], "Boykov–Kolmogorov": [_int(r["bk"]) for r in sc_rows], "Dinic": [_int(r["dinic"]) for r in sc_rows],
              "Edmonds-Karp": [_int(r["bfs"]) for r in sc_rows], "Dinic ÷ BK": [_f(r["dinic"] / r["bk"], 1) for r in sc_rows], "Edmonds-Karp ÷ BK": [_f(r["bfs"] / r["bk"], 0) for r in sc_rows]})
    st.caption("Auf diesen Gitternetzen durchsucht Boykov–Kolmogorov am wenigsten Kanten, und der Abstand wächst mit dem Raster. Zählt man Kanten statt Sekunden, bleibt offen, wie viel davon in einer Maschinenimplementierung ankommt; bewiesen ist keine bessere Laufzeitschranke, im schlimmsten Fall ist BK sogar schlechter als Dinic.")

st.subheader("🔬 Stimmt die Beschriftung wirklich?")
st.caption("Gegenprobe auf 4 × 4 Rastern (Rauschen 80 %, λ = 0,5): alle 65 536 Beschriftungen durchprobieren und mit dem Schnitt (drei Flussverfahren) vergleichen; dazu ICM und die naive Konstruktion bei nicht submodularer Strafe.")
if st.button("40 kleine Raster prüfen", key="check_start"):
    st.session_state["check_on"] = True
if st.session_state.get("check_on"):
    chk = ev.small_check()
    c1, c2, c3 = st.columns(3)
    c1.metric("Schnitt = kleinste Energie", f"{chk['exact']} von {chk['n']}", help="Alle drei Flussverfahren erreichen die minimale Energie aller 65 536 Beschriftungen.")
    c2.metric("ICM über dem Minimum", f"{chk['icm_above']} von {chk['n']}", help="ICM bleibt in einem lokalen Minimum hängen.")
    c3.metric("Naiver Schnitt bei Belohnung über dem Minimum", f"{chk['naive_above']} von {chk['n']}", help="Bei nicht submodularer Strafe (Unterschiede belohnen) liefert der Schnitt mit gekappten Kapazitäten nicht das Minimum.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist - und wer ansetzt |
|---|---|
| **Zwei Zonen** | Mit mehr als zwei Zonen ist die exakte Lösung im Allgemeinen NP-schwer; **α-Expansion** löst Folgen von Zwei-Zonen-Schnitten und liefert eine gute, nicht beweisbar beste Beschriftung (nur erwähnt, nicht gebaut). |
| **Ungleiche Nachbarn kosten** | Belohnt die Energie Unterschiede (nicht submodular), gibt es negative Kapazitäten; der Schnitt ist dann nicht mehr exakt (Preset „Nicht submodular“). **QPBO** und ähnliche Verfahren setzen dort an (nicht gebaut). |
| **Eine Glättung für alles** | Dieselbe Strafe glättet große und löscht kleine Zonen, verwischt dünne Streifen und passt nicht zu jedem Rauschen (Kurven oben). Die Wahrheit, mit der man λ wählen könnte, kennt man in der Praxis nicht. |
| **Bekannte Zonenmittelwerte und Rauschen** | Der Datenterm setzt μ und σ voraus; hier sind sie bekannt. Sie zu schätzen (etwa per EM-Verfahren) ist ein eigenes Thema. |
| **Vier Nachbarn** | Die Nachbarschaft ist fest; größere Nachbarschaften machen das Hilfsnetz dichter, ändern aber das Prinzip nicht. |
"""
)
st.caption("Die Netzwerkfluss-Linie ist als Ganzes geplant: die zwölf Stücke der Hauptlinie (gebaut), dazu die Erweiterung E1: **Projektauswahl** (gebaut) und **Graph Cuts** (dieses Stück); optional folgt der Gomory-Hu-Baum.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Energie.** Zellen $V$, Nachbarpaare $\mathcal{N}$ (4er-Nachbarschaft), Beschriftung $x\in\{0,1\}^V$:
$$E(x)=\sum_{i\in V}D_i(x_i)+\lambda\sum_{(i,j)\in\mathcal N}[x_i\neq x_j],\qquad D_i(l)=\frac{(y_i-\mu_l)^2}{2\sigma^2}.$$
Alle Größen sind ganzzahlig (Messwerte in Tausendsteln, Energie in Hundertsteln), damit jede Zahl auf allen Plattformen dieselbe ist.

**Schnitt.** Setze $m_i=\min(D_i(0),D_i(1))$. Kanten $(s,i)$ mit Kapazität $D_i(1)-D_i(0)$, wenn positiv, sonst $(i,t)$ mit $D_i(0)-D_i(1)$; Kanten $(i,j)$ und $(j,i)$ mit Kapazität $\lambda$ für jedes Nachbarpaar. Für einen Schnitt $(S\cup\{s\},\bar S\cup\{t\})$ mit $x_i=0$ für $i\in S$ gilt
$$c(S)=E(x)-\sum_i m_i,$$
denn (s,i) wird durchschnitten, wenn $x_i=1$, und (i,t), wenn $x_i=0$; eine Nachbarkante wird genau dann durchschnitten, wenn die Labels verschieden sind. Der minimale Schnitt liefert also die minimale Energie.

**Submodularität.** Eine paarweise Energie $E_{ij}$ lässt sich nur dann als nichtnegative Kante bauen, wenn $E_{ij}(0,0)+E_{ij}(1,1)\le E_{ij}(0,1)+E_{ij}(1,0)$ (Kolmogorov und Zabih 2004). Die Potts-Strafe $\lambda[x_i\ne x_j]$ erfüllt das mit $0\le 2\lambda$; eine belohnende Strafe $-\lambda[x_i\ne x_j]$ nicht.

**Monotonie in $\lambda$.** $\lambda\mapsto\min_x E_\lambda(x)$ ist als Minimum wachsender Funktionen nicht fallend: die minimale Energie kann deshalb nicht zur Wahl von $\lambda$ dienen.

**Boykov–Kolmogorov.** Zwei Suchbäume $T_S,T_T$ mit Wurzeln $s,t$; *Wachstum* aus aktiven Knoten über Kanten mit Restkapazität, bis ein Knoten von $T_S$ einen von $T_T$ berührt; *Augmentieren* längs des Weges (jede gesättigte Baumkante macht ihr Kind zur Waise); *Adoption*: eine Waise sucht einen Elternknoten im eigenen Baum, dessen Weg zur Wurzel intakt ist (geprüft mit Zeitstempel und Entfernung), sonst wird sie frei. Am Ende bilden die Knoten von $T_S$ die Quellseite eines minimalen Schnitts.

**ICM** (Besag 1986): wiederhole für jede Zelle $x_i\leftarrow\arg\min_l\big(D_i(l)+\lambda\sum_{j\sim i}[l\ne x_j]\big)$, bis sich nichts mehr ändert; endet in einem lokalen Minimum.

Implementiert in `gc_scenario.py` (Raster, Zonenformen, ganzzahliges Rauschen, eigener Zufallsgenerator), `gc_cut.py` (Energie, Hilfsnetz, Schnitt, Schwelle, Mittelwertfilter, ICM, Gegenprobe), `gc_bk.py` (Boykov–Kolmogorov), `gc_dinic.py` und `gc_edmonds_karp.py` (Kopien der Vorgänger-Demos), `gc_evaluation.py` (Kennzahlen, Verteilungen, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
