"""Plotly-Abbildungen: Karte mit Label-Festlegungen und Routen, Pareto-Front als Streudiagramm (Zeit gegen CO2) mit Konvexhülle und Budget, Label-Tabelle, Experimente. Achsen sind gesperrt (fixedrange),
damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import mc_constants as C
from mc_evaluation import state_at


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.14), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _segments(g, mask=None):
    """Kanten als eine Linienspur (None trennt die Segmente); bei ungerichteten Netzen Hin- und Rückrichtung nur einmal."""
    src = g.source_of_arcs()
    dst = g.indices
    keep = np.ones(len(src), dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    if not g.directed:
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        first = np.zeros(len(src), dtype=bool)
        _, idx = np.unique(lo * g.n + hi, return_index=True)
        first[idx] = True
        keep &= first
    u, v = src[keep], dst[keep]
    x = np.full(3 * len(u), None, dtype=object)
    y = np.full(3 * len(u), None, dtype=object)
    x[0::3], x[1::3] = g.xy[u, 0], g.xy[v, 0]
    y[0::3], y[1::3] = g.xy[u, 1], g.xy[v, 1]
    return x, y


def _fmt(x):
    return f"{x:g}"


def route_points(a, lid):
    """Koordinaten der Route zu einem Label."""
    return a.net.graph.xy[a.res.nodes(lid)]


def build_network(a, k, routes=(), height=520):
    """Die Karte nach `k` Entnahmen aus der Warteschlange: Knoten nach der Zahl der festgelegten Labels gefärbt; `routes` = [(Label, Name, Farbe, Stil)] werden gezeichnet; bei kleinen Netzen mit Namen, Label-Zahl und Kosten an den Kanten."""
    net, g = a.net, a.net.graph
    small = bool(g.names)
    count, popped, _, last = state_at(a, k)
    fig = go.Figure()
    if net.geometric:
        ex, ey = _segments(g)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.45)", width=1), hoverinfo="skip", showlegend=False))
        if net.arterial and any(net.arterial):
            ax_, ay_ = _segments(g, net.arterial)
            fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines", line=dict(color="rgba(230,140,0,0.75)", width=3), name="Hauptachse (schnell, schmutzig)", hoverinfo="skip"))
    if small:
        src = g.source_of_arcs()
        seen = set()
        for u, v, t, c in zip(src.tolist(), g.indices.tolist(), g.weight.tolist(), g.weight2.tolist()):
            if (v, u) in seen:
                continue
            seen.add((u, v))
            mid = (g.xy[u] + g.xy[v]) / 2
            fig.add_annotation(x=mid[0], y=mid[1], text=f"{_fmt(t)} min · {_fmt(c)} g", showarrow=False, font=dict(size=10, color="#555"), bgcolor="rgba(255,255,255,0.85)")
    n = g.n
    size = 16 if small else (3 if n > 1500 else (5 if n > 300 else 8))
    reached = count > 0
    if small:
        text = [f"{g.names[v]}<br>{count[v]} Label{'s' if count[v] != 1 else ''}" if reached[v] else g.names[v] for v in range(n)]
        vmax = max(int(count.max()), 1)
        fig.add_trace(go.Scatter(x=g.xy[:, 0], y=g.xy[:, 1], mode="markers+text", showlegend=False, text=text, textposition="top center", hoverinfo="skip",
                                 marker=dict(size=size, color=np.where(reached, count, np.nan) if reached.any() else "white", colorscale="Viridis", cmin=0, cmax=vmax, showscale=False, line=dict(color="gray", width=1.5))))
    else:
        rest = np.where(~reached)[0]
        if len(rest):
            fig.add_trace(go.Scatter(x=g.xy[rest, 0], y=g.xy[rest, 1], mode="markers", showlegend=False, hoverinfo="skip", marker=dict(size=size, color="rgba(200,200,200,0.9)")))
        got = np.where(reached)[0]
        if len(got):
            fig.add_trace(go.Scatter(x=g.xy[got, 0], y=g.xy[got, 1], mode="markers", showlegend=False, customdata=count[got], hovertemplate="%{customdata} festgelegte Labels<extra></extra>",
                                     marker=dict(size=size, color=count[got], colorscale="Viridis", cmin=1, cmax=max(int(count.max()), 2), colorbar=dict(title="Labels<br>je Knoten", thickness=12, len=0.6))))
    if not routes and last is not None and k < len(a.res.events):
        pts = route_points(a, last)
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color="rgba(30,30,30,0.7)", width=3, dash="dot"), name="Route des zuletzt entnommenen Labels", hoverinfo="skip"))
    for lid, name, color, dash in routes:
        pts = route_points(a, lid)
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=color, width=6 if dash == "solid" else 3, dash=dash), name=name, hoverinfo="skip"))
    for node, name, color, symbol in ((net.source, "Start", C.COLORS["start"], "diamond"), (net.target, "Ziel", C.COLORS["goal"], "star")):
        lab = g.names[node] if small else f"{node}"
        fig.add_trace(go.Scatter(x=[g.xy[node, 0]], y=[g.xy[node, 1]], mode="markers", name=f"{name}: {lab}", hoverinfo="skip", marker=dict(size=15, color=color, symbol=symbol, line=dict(color="white", width=1.5))))
    fig.update_xaxes(visible=False)
    if not small and net.key != "chain":
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    if small:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        pad = 0.16 * (hi - lo)
        fig.update_xaxes(range=[lo[0] - pad[0], hi[0] + 1.6 * pad[0]])
        fig.update_yaxes(range=[lo[1] - pad[1], hi[1] + pad[1]])
    return _base(fig, height)


def build_front(a, k, mode="front", chosen=None, budget=None, height=420):
    """Zeit gegen CO2 am Ziel. Vor dem Ende: die bisher am Ziel festgelegten Labels. Danach die Front: Ecken der Konvexhülle (schwarze Rauten: die gewichtete Summe findet sie), nicht unterstützte Punkte (orange, hohl: nur das
    Label-setting findet sie); im Modus "Gewichtete Summe" sind die nicht unterstützten Punkte abgeblendet, im Modus "Budget" ist die Zone über dem Budget schraffiert. `chosen` = (Zeit, CO2) des gewählten Punkts."""
    m = a.metrics
    final = k >= len(a.res.events)
    fig = go.Figure()
    unit_t, unit_c = ("min" if a.net.key == "small" else ("s" if a.net.key in ("city", "chain") else "Einh.")), ("g" if a.net.key in ("small", "city", "chain") else "Einh.")
    if not final:
        _, _, tl, _ = state_at(a, k)
        if tl:
            fig.add_trace(go.Scatter(x=[p[0] for p in tl], y=[p[1] for p in tl], mode="markers", name="bisher am Ziel festgelegt", marker=dict(size=9, color=C.COLORS["chosen"], opacity=0.75), hovertemplate="%{x:g} · %{y:g}<extra></extra>"))
        fig.update_layout(annotations=[dict(text="Die Front entsteht: Labels am Ziel, nach Zeit geordnet", x=0.5, y=1.02, xref="paper", yref="paper", showarrow=False, font=dict(size=11, color="#666"))])
    else:
        hull = set(a.hull)
        pts = [(t, c) for t, c, _ in a.front]
        sup = [p for p in pts if p in hull]
        uns = [p for p in pts if p not in hull]
        if a.hull and len(a.hull) > 1:
            fig.add_trace(go.Scatter(x=[p[0] for p in a.hull], y=[p[1] for p in a.hull], mode="lines", name="Konvexhülle", line=dict(color="rgba(0,0,0,0.35)", dash="dot", width=2), hoverinfo="skip"))
        if uns:
            fig.add_trace(go.Scatter(x=[p[0] for p in uns], y=[p[1] for p in uns], mode="markers", name="nicht unterstützt (nur Label-setting findet sie)",
                                     marker=dict(size=10, symbol="circle-open", color=C.COLORS["unsupported"], line=dict(width=2), opacity=0.35 if mode == "weighted" else 1.0), hovertemplate="%{x:g} · %{y:g}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[p[0] for p in sup], y=[p[1] for p in sup], mode="markers", name="Ecke der Konvexhülle (gewichtete Summe)", marker=dict(size=12, symbol="diamond", color=C.COLORS["hull"]), hovertemplate="%{x:g} · %{y:g}<extra></extra>"))
        if mode == "budget" and budget is not None:
            fig.add_hline(y=budget, line=dict(color=C.COLORS["budget"], dash="dash"), annotation_text=f"Budget {budget:.0f} g", annotation_position="top right")
            top = max(p[1] for p in pts)
            fig.add_hrect(y0=budget, y1=top * 1.08, fillcolor="rgba(148,103,189,0.10)", line_width=0)
        if chosen is not None:
            fig.add_trace(go.Scatter(x=[chosen[0]], y=[chosen[1]], mode="markers", name="gewählte Route", marker=dict(size=20, symbol="circle-open", color=C.COLORS["chosen"], line=dict(width=4)), hoverinfo="skip"))
    fig.update_layout(xaxis=dict(title=f"Zeit [{unit_t}]", rangemode="tozero"), yaxis=dict(title=f"CO₂ [{unit_c}]", rangemode="tozero"))
    return _base(fig, height)


def table_state(a, k):
    """Kleines Netz: je Ort die bis Schritt k entnommenen Labels (Zeit, CO2), festgelegte mit ✓, dominierte mit ✗."""
    g = a.net.graph
    _, popped, _, _ = state_at(a, k)
    rows = []
    for v in range(g.n):
        labs = popped.get(v, [])
        text = "  ".join(f"({_fmt(t)}, {_fmt(c)}) {'✓' if s == 'settled' else '✗'}" for t, c, s in labs) if labs else "–"
        rows.append({"Ort": g.names[v], "Labels (Zeit, CO₂)": text})
    return rows


# --- Experimente -------------------------------------------------------------------------------------------------------------------------------

def build_front_size(rows, height=340):
    fig = go.Figure()
    for conflict, color in ((0.0, "#1f77b4"), (0.5, "#ff7f0e"), (1.0, "#d62728")):
        sel = [r for r in rows if r["conflict"] == conflict]
        fig.add_trace(go.Scatter(x=[r["n"] for r in sel], y=[r["front"] for r in sel], mode="lines+markers", name=f"Front, Faktor {1 + conflict:g}", line=dict(color=color)))
    sel = [r for r in rows if r["conflict"] == 1.0]
    fig.add_trace(go.Scatter(x=[r["n"] for r in sel], y=[r["hull"] for r in sel], mode="lines+markers", name="davon Ecken der Hülle (Faktor 2)", line=dict(color="#111111", dash="dash")))
    fig.update_layout(xaxis=dict(title="Knoten im Netz", type="log"), yaxis=dict(title="Punkte am Ziel", type="log"))
    return _base(fig, height)


def build_front_corr(rows, height=340):
    fig = go.Figure()
    x = [f"{r['corr']:g}" for r in rows]
    fig.add_trace(go.Bar(x=x, y=[r["front"] for r in rows], name="Front", marker_color="#d62728"))
    fig.add_trace(go.Bar(x=x, y=[r["hull"] for r in rows], name="Ecken der Hülle", marker_color="#111111"))
    fig.update_layout(barmode="group", xaxis_title="Korrelation der Kosten je Kante (−1 gegenläufig … +1 gleichläufig)", yaxis_title="Punkte am Ziel")
    return _base(fig, height)


def build_missed(rows, height=340):
    fig = go.Figure()
    x = [r["label"] for r in rows]
    fig.add_trace(go.Bar(x=x, y=[r["front"] for r in rows], name="Punkte der Front", marker_color="#d62728"))
    fig.add_trace(go.Bar(x=x, y=[r["found"] for r in rows], name="davon von der gewichteten Summe gefunden", marker_color="#111111"))
    fig.update_layout(barmode="group", yaxis=dict(title="Punkte am Ziel", type="log"))
    return _base(fig, height)


def build_chain(rows, height=320):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["k"] for r in rows], y=[r["front"] for r in rows], mode="lines+markers", name="Punkte der Front (2^k)", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=[r["k"] for r in rows], y=[r["generated"] for r in rows], mode="lines+markers", name="erzeugte Labels", line=dict(color="#1f77b4", dash="dash")))
    fig.update_layout(xaxis_title="Glieder der Kette k", yaxis=dict(title="Zähler", type="log"))
    return _base(fig, height)


def build_budget(rows, height=320):
    fig = go.Figure()
    x = [f"{r['percent']}" for r in rows]
    fig.add_trace(go.Bar(x=x, y=[r["generated"] for r in rows], name="erzeugte Labels mit Budget", marker_color="#9467bd"))
    fig.add_trace(go.Scatter(x=x, y=[r["full"] for r in rows], mode="lines", name="ohne Budget", line=dict(color="#111111", dash="dash")))
    fig.update_layout(xaxis_title="Budget in % der Spanne zwischen sauberster (0) und schnellster Route (100)", yaxis_title="erzeugte Labels")
    return _base(fig, height)
